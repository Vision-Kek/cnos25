from typing import Dict, Any
import torch
import numpy as np
import torchvision
from torchvision.ops.boxes import batched_nms, box_area
import logging
import time
from PIL import Image
import os.path as osp
import json
import pandas as pd
import glob
from functools import partial
import multiprocessing
from tqdm import tqdm

from src.utils.inout import save_json, load_json, save_json_bop23, save_npz
from src.utils.bbox_utils import xyxy_to_xywh, xywh_to_xyxy, force_binary_mask

lmo_object_ids = np.array(
    [
        1,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
    ]
)  # object ID of occlusionLINEMOD is different


def mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order="F")):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle

 #https://github.com/facebookresearch/sam2/blob/main/sam2/utils/amg.py
def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


class BatchedData:
    """
    A structure for storing data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, batch_size, data=None, **kwargs) -> None:
        self.batch_size = batch_size
        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self):
        assert self.batch_size is not None, "batch_size is not defined"
        return np.ceil(len(self.data) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size : (idx + 1) * self.batch_size]

    def cat(self, data, dim=0):
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data):
        self.data.append(data)

    def stack(self, dim=0):
        self.data = torch.stack(self.data, dim=dim)


class Detections:
    """
    A structure for storing detections.
    """

    def __init__(self, data) -> None:
        if isinstance(data, str):
            data = self.load_from_file(data)
        for key, value in data.items():
            setattr(self, key, value)
        self.keys = list(data.keys())
        if "boxes" in self.keys:
            if isinstance(self.boxes, np.ndarray):
                self.to_torch()
            self.boxes = self.boxes.long()

    def remove_very_small_detections(self, config):
        img_area = self.masks.shape[1] * self.masks.shape[2]
        box_areas = box_area(self.boxes) / img_area
        mask_areas = self.masks.sum(dim=(1, 2)) / img_area
        keep_idxs = torch.logical_and(
            box_areas > config.min_box_size**2, mask_areas > config.min_mask_size
        )
        # logging.info(f"Removing {len(keep_idxs) - keep_idxs.sum()} detections")
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idxs])

    def retain_top_n_confident_detections(self, n):
        for k, v in vars(self).items():
            setattr(self, k, v[:n])

    def apply_nms_per_object_id(self, nms_thresh=0.5):
        keep_idxs = BatchedData(None)
        all_indexes = torch.arange(len(self.object_ids), device=self.boxes.device)
        for object_id in torch.unique(self.object_ids):
            idx = self.object_ids == object_id
            idx_object_id = all_indexes[idx]
            keep_idx = torchvision.ops.nms(
                self.boxes[idx].float(), self.scores[idx].float(), nms_thresh
            )
            keep_idxs.cat(idx_object_id[keep_idx])
        keep_idxs = keep_idxs.data
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idxs])

    def apply_nms(self, nms_thresh=0.5):
        keep_idx = torchvision.ops.nms(
            self.boxes.float(), self.scores.float(), nms_thresh
        )
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idx])

    def add_attribute(self, key, value):
        setattr(self, key, value)
        self.keys.append(key)

    def __len__(self):
        return len(self.boxes)

    def check_size(self):
        mask_size = len(self.masks)
        box_size = len(self.boxes)
        score_size = len(self.scores)
        object_id_size = len(self.object_ids)
        assert (
            mask_size == box_size == score_size == object_id_size
        ), f"Size mismatch {mask_size} {box_size} {score_size} {object_id_size}"

    def to_numpy(self):
        for key in self.keys:
            setattr(self, key, getattr(self, key).cpu().numpy())

    def to_torch(self):
        for key in self.keys:
            a = getattr(self, key)
            if isinstance(a, torch.Tensor): continue
            setattr(self, key, torch.from_numpy(getattr(self, key)))

    def save_to_file(
        self, scene_id, frame_id, runtime, file_path, dataset_name, return_results=False, save_mask=True, save_score_distribution=False
    ):
        """
        scene_id, image_id, category_id, bbox, time
        """
        boxes = xyxy_to_xywh(self.boxes)
        results = {
            "scene_id": scene_id,
            "image_id": frame_id,
            "category_id": self.object_ids + 1
            if dataset_name != "lmo"
            else lmo_object_ids[self.object_ids],
            "score": self.scores,
            "bbox": boxes,
            "time": runtime,
        }
        if save_mask:
            results["segmentation"] = self.masks
        if save_score_distribution:
            assert hasattr(self, "score_distribution"), "score_distribution is not defined"
            results["score_distribution"] = self.score_distribution
        save_npz(file_path, results)
        if return_results:
            return results

    def filter(self, idxs):
        for key in self.keys:
            setattr(self, key, getattr(self, key)[idxs])

    def clone(self):
        """
        Clone the current object
        """
        return Detections(self.__dict__.copy())

def load_from_file(file_path):
    data = np.load(file_path)
    boxes = xywh_to_xyxy(np.array(data["bbox"]))
    output = {
        "object_ids": data["category_id"] - 1,
        "boxes": boxes,
        "scores": data["score"],
    }
    if "segmentation" in data.keys():
        output["masks"] = data["segmentation"]
    if "score_distribution" in data.keys():
        output["score_distribution"] = data["score_distribution"]
    logging.info(f"Loaded {file_path}")
    return Detections(output)

def load_framewise_detections_from_json(json_file_path):
    json_df = pd.DataFrame(load_json(json_file_path))
    # as in load_from_file
    frame_detections = []
    frame_info = []
    frame_groups = json_df.groupby(['scene_id', 'image_id']).groups
    # iterate over frames
    for (scene_id, image_id), df_row_idcs in frame_groups.items():

        boxes, scores, object_ids, times = [], [], [], []
        # iterate over all detections (box, score, id, mask) for this scene+frame
        for row in df_row_idcs:
            bbox, score, category_id, runtime= json_df.iloc[row][['bbox', 'score', 'category_id', 'time']]
            boxes.append(xywh_to_xyxy(np.array(bbox)))
            scores.append(score)
            object_ids.append(category_id - 1)  # 0 to n
            times.append(runtime)
            # segmentation is in rle, would need to rle_2_boolmask

        frame_pred = {
            "object_ids": np.stack(object_ids),
            "boxes": np.stack(boxes),
            "scores": np.stack(scores),
            "runtimes": np.stack(times),
        }

        frame_detections.append(Detections(frame_pred)) # create Detections object here
        frame_info.append({'scene_id': scene_id, 'image_id': image_id})
    return frame_detections, frame_info

def convert_npz_to_json(idx, list_npz_paths, save_segmentation_results=True):
    npz_path = list_npz_paths[idx]
    detections = np.load(npz_path)
    results = []
    results_with_score_distribution = []
    for idx_det in range(len(detections["bbox"])):
        result = {
            "scene_id": int(detections["scene_id"]),
            "image_id": int(detections["image_id"]),
            "category_id": int(detections["category_id"][idx_det]),
            "bbox": detections["bbox"][idx_det].tolist(),
            "score": float(detections["score"][idx_det]),
            "time": float(detections["time"]),
        }
        if save_segmentation_results:
            if "segmentation" in detections.keys():
                result["segmentation"] = mask_to_rle(
                    force_binary_mask(detections["segmentation"][idx_det])
                )
        results.append(result)

        if "score_distribution" in detections.keys():
            result_with_score_distribution = result.copy()
            result_with_score_distribution["score_distribution"] = detections["score_distribution"][idx_det].tolist()
            results_with_score_distribution.append(result_with_score_distribution)
    return results, results_with_score_distribution


def convert_npz_to_json_loop(experiment_result_dir, outfile_name, npzs_dir='predictions',
                             no_score_distr=False, masks=True):
    # can use self.all_gather to gather results from all processes
    # but it is simpler just load the results from files so no file is missing
    result_paths = sorted(
        #glob.glob(f"{experiment_result_dir}/*.npz") +
        glob.glob(f"{experiment_result_dir}/{npzs_dir}/*.npz")
    )
    result_paths = sorted(
        [path for path in result_paths if "runtime" not in path]
    )

    logging.info(f"Found {len(result_paths)} npz files in {experiment_result_dir}.")
    num_workers = 10
    logging.info(f"Converting npz to json requires {num_workers} workers ...")
    pool = multiprocessing.Pool(processes=num_workers)
    convert_npz_to_json_with_idx = partial(
        convert_npz_to_json,
        list_npz_paths=result_paths,
        save_segmentation_results=masks
    )
    detections = list(
        tqdm(
            pool.imap_unordered(
                convert_npz_to_json_with_idx, range(len(result_paths))
            ),
            total=len(result_paths),
            desc="Converting npz to json",
        )
    )
    formatted_detections = []
    #formatted_detections_with_score_distribution = []
    for detection in tqdm(detections, desc="Loading results ..."):
        formatted_detections.extend(detection[0])
        #formatted_detections_with_score_distribution.extend(detection[1])

    detections_path = osp.join(experiment_result_dir, f"{outfile_name}.json")
    save_json_bop23(detections_path, formatted_detections)
    logging.info(f"Saved predictions (BOP format) to {detections_path}")

    # if not no_score_distr:
    #     detections_path = osp.join(results_dir, experiment_name, f"{experiment_name}_with_score_distribution.json")
    #     save_json_bop23(detections_path, formatted_detections_with_score_distribution)
    #     logging.info(f"Saved predictions (BOP format + score distribution) to {detections_path} ")
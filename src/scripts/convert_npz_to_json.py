# Do the convert_npz_to_json_loop in a script that can run in a background
# Standalone script, no dependency to a project
import os.path as osp
import glob
import numpy as np
import logging
from tqdm import tqdm
from functools import partial
import multiprocessing
import json


def save_json_bop23(path, info):
    # save to json without sorting keys or changing format
    with open(path, "w") as f:
        json.dump(info, f)

def force_binary_mask(mask, threshold=0.):
    mask = np.where(mask > threshold, 1, 0)
    return mask

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


def convert_npz_to_json_loop(results_dir, experiment_name, no_score_distr=False):
    # can use self.all_gather to gather results from all processes
    # but it is simpler just load the results from files so no file is missing
    experiment_base_dir = osp.join(results_dir, experiment_name)
    result_paths = sorted(
        glob.glob(f"{experiment_base_dir}/*.npz") + glob.glob(f"{experiment_base_dir}/**/*.npz")
    )
    result_paths = sorted(
        [path for path in result_paths if "runtime" not in path]
    )

    print(f"Found {len(result_paths)} npz files in {experiment_base_dir}.")
    num_workers = 10
    logging.info(f"Converting npz to json requires {num_workers} workers ...")
    pool = multiprocessing.Pool(processes=num_workers)
    convert_npz_to_json_with_idx = partial(
        convert_npz_to_json,
        list_npz_paths=result_paths,
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
    formatted_detections_with_score_distribution = []
    for detection in tqdm(detections, desc="Loading results ..."):
        formatted_detections.extend(detection[0])
        formatted_detections_with_score_distribution.extend(detection[1])

    detections_path = osp.join(results_dir, experiment_name, f"{experiment_name}.json")
    save_json_bop23(detections_path, formatted_detections)
    logging.info(f"Saved predictions (BOP format) to {detections_path}")

    if not no_score_distr:
        detections_path = osp.join(results_dir, experiment_name, f"{experiment_name}_with_score_distribution.json")
        save_json_bop23(detections_path, formatted_detections_with_score_distribution)
        logging.info(f"Saved predictions (BOP format + score distribution) to {detections_path} ")

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="base result dir")
    parser.add_argument("--experiment_name", required=True, help="subdir under result dir")
    parser.add_argument('--no_score_distr', type=bool, default=True, help='whether to save score distribution')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(f"args: {args}")
    convert_npz_to_json_loop(args.results_dir, args.experiment_name, args.no_score_distr)

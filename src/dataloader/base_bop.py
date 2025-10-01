import logging, os
import os.path as osp
from tqdm import tqdm
import time
import numpy as np
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import os.path as osp
import pandas as pd
from src.utils.inout import load_json, save_json, casting_format_to_save_json
import torch
from src.utils.bbox_utils import CropResizePad

class BaseBOP(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        dataset_name=None,
        **kwargs,
    ):
        """
        Read a dataset in the BOP format.
        See https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md
        """
        self.root_dir = root_dir
        self.split = split

    def load_list_scene(self, split=None):
        if isinstance(split, str):
            if split is not None:
                split_folder = osp.join(self.root_dir, split)
            self.list_scenes = sorted(
                [
                    osp.join(split_folder, scene)
                    for scene in os.listdir(split_folder)
                    if os.path.isdir(osp.join(split_folder, scene))
                    and scene != "models"
                ]
            )
        elif isinstance(split, list):
            self.list_scenes = []
            for scene in split:
                if not isinstance(scene, str):
                    scene = f"{scene:06d}"
                if os.path.isdir(osp.join(self.root_dir, scene)):
                    self.list_scenes.append(osp.join(self.root_dir, scene))
            self.list_scenes = sorted(self.list_scenes)
        else:
            raise NotImplementedError
        logging.info(f"Found {len(self.list_scenes)} scenes")

    def load_scene(self, path, use_visible_mask=True):
        # Load rgb and mask images
        rgb_paths = sorted(Path(path).glob("rgb/*.[pj][pn][g]"))
        if use_visible_mask:
            mask_paths = sorted(Path(path).glob("mask_visib/*.[pj][pn][g]"))
        else:
            mask_paths = sorted(Path(path).glob("mask/*.[pj][pn][g]"))
        # load poses
        scene_gt = load_json(osp.join(path, "scene_gt.json"))
        scene_gt_info = load_json(osp.join(path, "scene_gt_info.json"))
        scene_camera = load_json(osp.join(path, "scene_camera.json"))
        return {
            "rgb_paths": rgb_paths,
            "mask_paths": mask_paths,
            "scene_gt": scene_gt,
            "scene_gt_info": scene_gt_info,
            "scene_camera": scene_camera,
        }

    def load_metaData(self, reset_metaData, mode="query", split="test", level=2):
        start_time = time.time()
        if mode == "query":
            metaData = {
                "scene_id": [],
                "frame_id": [],
                "rgb_path": [],
                "depth_path": [],
                "intrinsic": [],
            }
            logging.info(f"Loading metaData for split {split}")
            metaData_path = osp.join(self.root_dir, f"{split}_metaData.json")
            if reset_metaData:
                for scene_path in tqdm(self.list_scenes, desc="Loading metaData"):
                    scene_id = scene_path.split("/")[-1]
                    # HOT3D test split contains different modalities and sensors depending on the scene.
                    if self.dataset_name in ["hot3d"]:
                        eval_modality = self.dp_split["eval_modality"](int(scene_id))
                        eval_sensor = self.dp_split["eval_sensor"](int(scene_id))
                        
                    for im_id in self.target_images_per_scene[int(scene_id)]:
                        if self.dataset_name in ["hot3d"]:
                            rgb_path = self.dp_split[
                                f"{eval_modality}_{eval_sensor}_tpath"
                            ].format(scene_id=int(scene_id), im_id=im_id)
                            assert osp.exists(rgb_path), f'Hot3d {rgb_path=} does not exist. If you use non-bop-format dataset (no gray1/gray2 folders), use BaseBOPHOT3D instead.'
                        else:
                            rgb_path = self.dp_split["rgb_tpath"].format(
                                scene_id=int(scene_id), im_id=im_id
                            )
                        metaData["scene_id"].append(scene_id)
                        metaData["frame_id"].append(im_id)
                        metaData["rgb_path"].append(rgb_path)
                        metaData["depth_path"].append(None)
                        metaData["intrinsic"].append(None)
                        
                # casting format of metaData
                metaData = casting_format_to_save_json(metaData)
                save_json(metaData_path, metaData)
            else:
                metaData = load_json(metaData_path)
        elif mode == "template":
            list_obj_ids, list_idx_template = [], []
            for obj_id in self.obj_ids:
                for idx_template in range(len(self.templates_poses)):
                    list_obj_ids.append(obj_id)
                    list_idx_template.append(idx_template)
            metaData = {
                "obj_id": list_obj_ids,
                "idx_template": list_idx_template,
            }

        self.metaData = pd.DataFrame.from_dict(metaData, orient="index")
        self.metaData = self.metaData.transpose()
        # # shuffle data
        self.metaData = self.metaData.sample(frac=1, random_state=2021).reset_index(
            drop=True
        )
        finish_time = time.time()
        logging.info(
            f"Finish loading metaData of size {len(self.metaData)} in {finish_time - start_time:.2f} seconds"
        )
        return

    def __len__(self):
        return len(self.metaData)

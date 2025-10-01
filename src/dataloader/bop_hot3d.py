from dataclasses import dataclass
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
import torch
from enum import auto, Enum
from typing import Any
import tarfile
import imageio
import pandas as pd

from src.dataloader.base_bop import BaseBOP
from src.utils.inout import load_json


def load_image(
    tar: Any, frame_key: str, stream_key: str, dtype: Any = np.uint8
) -> np.ndarray:
    """Loads an image from the specified frame and stream of a clip.

    Args:
        tar: File handler of an open tar file with clip data.
        frame_key: Key of the frame from which to load the image.
        stream_key: Key of the stream from which to load the image.
        dtype: Desired type of the loaded image.
    Returns:
        Numpy array with the loaded image.
    """

    file = tar.extractfile(f"{frame_key}.image_{stream_key}.jpg")
    return imageio.imread(file).astype(dtype)

@dataclass
class Target(object):
    im_id: int
    scene_id: int
    device_type: str
    tar_filepath: str
    stream_id: str
    clip_name: str

    def to_json(self):
        return {
            "frame_id": self.im_id, # frame_id: consistent naming with metadata of other bop datasets
            "scene_id": self.scene_id,
            "device_type": self.device_type,
            "tar_filepath": self.tar_filepath,
            "stream_id": self.stream_id,
            "clip_name": self.clip_name,
        }

# The query (test) dataloader
class BaseBOPHOT3D(BaseBOP):
    def __init__(
        self,
        root_dir,
        **kwargs,
    ):
        self._root_dir = root_dir
        print(f"root_dir: {root_dir}")

        clip_definitions_filepath = f"{root_dir}/clip_definitions.json"
        targets_filepath = f"{root_dir}/test_targets_bop24.json"
                
        clip_definitions = load_json(clip_definitions_filepath)
        print(f"len clip_definitions: {len(clip_definitions)}")

        target_payload = load_json(targets_filepath)
        n_aria = n_quest3 = 0
        targets = []
        for tgt in target_payload:
            im_id = int(tgt["im_id"])
            scene_id=int(tgt["scene_id"])
            device_type = str(clip_definitions[str(scene_id)]["device"])
            clip_name = f"clip-{scene_id:06d}"

            tar_folder = None
            stream_id = None
            if device_type == "Aria":
                tar_folder = f"{root_dir}/test_aria"
                stream_id = "214-1"
                n_aria += 1
            elif device_type == "Quest3":
                tar_folder = f"{root_dir}/test_quest3"
                stream_id = "1201-1"
                n_quest3 += 1

            tar_filepath = f"{tar_folder}/{clip_name}.tar"
            target_obj = Target(                
                im_id=im_id,
                scene_id=scene_id,
                device_type=device_type,
                tar_filepath=tar_filepath,
                stream_id=stream_id,
                clip_name=clip_name,
            )
            targets += [target_obj]

        self._targets = targets[0:]
        self.metaData = pd.DataFrame([target.to_json() for target in targets])
        print(f"len targets: {len(self._targets)}, {n_aria=}, {n_quest3=}.")
        print('Using 214-1 rgb img for Aria targets and 1201-1 gray img for Quest3 targets')
        
        # self._rgb_transform = T.Compose(
        #     [
        #         T.ToTensor(),
        #         T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     ]
        # )

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, idx):
        st = self._targets[idx]
        tar = tarfile.open(st.tar_filepath, mode="r")
        frame_id = st.im_id
        scene_id = st.scene_id
        stream_id = st.stream_id

        frame_key = f"{frame_id:06d}"
        image_np: np.ndarray = load_image(tar, frame_key, stream_id)
        image = Image.fromarray(np.uint8(image_np))
        # image = self._rgb_transform(image.convert("RGB")) # jonas: return PIL not tensor
        return dict(
            image=image,
            scene_id=scene_id,
            frame_id=frame_id,
        )

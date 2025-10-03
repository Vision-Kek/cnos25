from dataclasses import dataclass
from enum import auto, Enum
from typing import Any
import logging, os
import os.path as osp
from pathlib import Path
import tarfile
import imageio
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.bbox_utils import CropResizePad
from src.dataloader.base_bop import BaseBOP
from src.utils.inout import load_json


try:
    from bop_toolkit_lib import dataset_params
    from bop_toolkit_lib import inout as bop_inout
except ImportError:
    raise ImportError(
        "Please install bop_toolkit_lib: pip install git+https://github.com/thodan/bop_toolkit.git"
    )


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


# The template (reference) dataloader
class BOPHOT3DTemplate(Dataset):
    def __init__(
            self,
            template_dir,
            obj_ids,
            image_size,
            num_imgs_per_obj=50,
            **kwargs,
    ):
        self.template_dir = template_dir
        self.model_free_onboarding = True
        self.dataset_name = template_dir.split("/")[-2]
        if obj_ids is None:
            obj_ids = [
                int(obj_id[4:10])  # pattern: obj_0000xxx
                for obj_id in os.listdir(template_dir)
                if osp.isdir(osp.join(template_dir, obj_id)) and obj_id.startswith('obj_')
            ]
            obj_ids = sorted(np.unique(obj_ids).tolist())
            logging.info(f"Found {obj_ids} objects in {self.template_dir}")

        self.num_imgs_per_obj = num_imgs_per_obj  # to avoid memory issue
        self.obj_ids = obj_ids
        self.image_size = image_size

        # for HOT3D, we have black objects so we use gray background
        logging.info("Use gray background for HOT3D")
        self.proposal_processor = CropResizePad(
            self.image_size,
            pad_value=0.5,  # gray background
        )

    def load_template_poses(self, level_templates, pose_distribution):
        raise NotImplementedError('Model-based not implemented. Go back to https://github.com/nv-nguyen/cnos.')

    def __getitem__modelfree__(self, idx):
        templates_cropped, masks_cropped, boxes, image_paths = [], [], [], []
        # Currently using Aria device for sampling ref images onboarding_static -> object_ref_aria_static
        static_onboarding = True if "onboarding_static" in self.template_dir else False
        if static_onboarding:
            # HOT3D names the two videos with _1 and _2 instead of _up and _down
            obj_dirs = [
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}_1",
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}_2",
            ]
            num_selected_imgs = self.num_imgs_per_obj // 2  # 100 for 2 videos
        else:
            obj_dirs = [
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}",
            ]
            num_selected_imgs = self.num_imgs_per_obj

        for obj_dir in obj_dirs:
            if not osp.exists(obj_dir):
                continue
            obj_dir = Path(obj_dir)
            obj_rgbs = sorted(Path(obj_dir).glob("*214-1.[pj][pn][g]"))

            # use 1201-2 and not 1201-1 stream because in -1 stream some boxes are negative
            obj_rgbs += sorted(Path(obj_dir).glob("*1201-2.[pj][pn][g]"))

            obj_masks = [None for _ in obj_rgbs]
            assert len(obj_rgbs) == len(obj_masks), f"rgb and mask mismatch in {obj_dir}"

            # If HOT3D + dynamic onboarding, we have the bbox for only the first image.
            # therefore, we select the first image only.
            if not static_onboarding:
                selected_idx = [0, 0, 0, 0, 0]  # required aggregation top k
            else:
                # select random samples
                selected_idx = list(np.random.choice(len(obj_rgbs), num_selected_imgs, replace=False))
            for idx_img in tqdm(selected_idx):
                image = Image.open(obj_rgbs[idx_img])
                # support _1201-x gray stream, too
                # previously only rgb images were used for templates, but Quest3 test images are gray
                img_suffix = str(obj_rgbs[idx_img]).split('.image_')[-1]  # either 214-1.jpg or 1201-x.jpg
                json_path = str(obj_rgbs[idx_img]).replace(f"image_{img_suffix}", "objects.json")
                info = bop_inout.load_json(json_path)
                obj_id = [k for k in info.keys()][0]
                try:
                    # box is derived from .objects.json
                    bbox = np.int32(info[obj_id][0]["boxes_amodal"][img_suffix.split('.')[0]])
                except KeyError as e:
                    logging.warning(f"No such key {img_suffix.split('.')[0]}, "
                                    f"available {info[obj_id][0]['boxes_amodal'].keys()}. "
                                    f"Skipping and sampling anotehr one.")
                    selected_idx.append(np.random.choice(len(obj_rgbs), 1).item())
                    continue
                mask = np.ones((image.size[1], image.size[0])) * 255

                mask = torch.from_numpy(np.array(mask) / 255).float()
                image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
                image_paths.append(obj_rgbs[idx_img])

                template = image.permute(2, 0, 1)
                mask = mask.unsqueeze(-1).permute(2, 0, 1)
                box = torch.tensor(bbox)
                if box.min() < 0: logging.warning(f'{box.min()=} < 0')

                template_cropped = self.proposal_processor(images=template.unsqueeze(0), boxes=box.unsqueeze(0))[0]

                # sometimes the cropping is off by 1px (e.g. yields a 223 instead of 224 crop) -> catch this case
                target_size = [self.image_size, self.image_size]
                if template_cropped.shape[-2:] != torch.Size(target_size):
                    logging.warning(f'Shape mismatch after cropping template IMAGE: {template_cropped.shape[-2:]}, '
                                    f'interpolating')
                    template_cropped = torch.nn.functional.interpolate(
                        template_cropped.unsqueeze(0),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    )[0]

                templates_cropped.append(T.ToPILImage()(template_cropped))  # return PIL image, not tensor

                mask_cropped = self.proposal_processor(images=mask.unsqueeze(0), boxes=box.unsqueeze(0))[0]

                # sometimes the cropping is off by 1px (e.g. yields a 223 instead of 224 crop) -> catch this case
                if mask_cropped.shape[-2:] != torch.Size(target_size):
                    logging.warning(f'Shape mismatch after cropping template MASK: {mask_cropped.shape[-2:]}, '
                                    f'interpolating')
                    mask_cropped = torch.nn.functional.interpolate(
                        mask_cropped.unsqueeze(0),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    )[0]

                masks_cropped.append(mask_cropped)

        masks_cropped = torch.stack(masks_cropped, dim=0)
        return {
            "templates": templates_cropped,  # PIL image List
            "template_masks": masks_cropped[:, 0, :, :],  # tensors
            "image_paths": image_paths
        }

    def __getitem__modelbased__(self, idx):
        raise NotImplementedError('Model-based not implemented. Go back to https://github.com/nv-nguyen/cnos.')

    def __len__(self):
        return len(self.obj_ids)

    def __getitem__(self, idx):
        if self.model_free_onboarding:
            return self.__getitem__modelfree__(idx)
        else:
            return self.__getitem__modelbased__(idx)


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
            "frame_id": self.im_id,  # frame_id: consistent naming with metadata of other bop datasets
            "scene_id": self.scene_id,
            "device_type": self.device_type,
            "tar_filepath": self.tar_filepath,
            "stream_id": self.stream_id,
            "clip_name": self.clip_name,
        }


# The query (test) dataloader
class BaseBOPHOT3D(BaseBOP):
    def __init__(self, root_dir, split, **kwargs):
        super().__init__(root_dir, split, **kwargs)

        # clip_definitions contain the mapping from scene id to device (i.e. Quest3/Aria)
        clip_definitions_filepath = f"{root_dir}/clip_definitions.json"
        clip_definitions = load_json(clip_definitions_filepath)
        logging.info(f"HOT3D {len(clip_definitions)=}")

        targets_filepath = f"{root_dir}/test_targets_bop24.json"
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
        logging.info(f"len targets: {len(self._targets)}, {n_aria=}, {n_quest3=}.")
        logging.info(f'Using 214-1 rgb img for Aria targets and 1201-1 gray img for Quest3 targets')

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
        return dict(
            image=image, # PIL.Image
            scene_id=scene_id,
            frame_id=frame_id,
        )
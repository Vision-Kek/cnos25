from dataclasses import dataclass
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
import json
import fnmatch

from src.utils.bbox_utils import CropResizePad
from src.dataloader.base_bop import BaseBOP
from src.utils.inout import load_json


def find_in_tar(tar_path, pattern):
    """Find files in a tar archive that match a given pattern.

    Args:
        tar_path (str): Path to the tar file to search
        pattern (str): File pattern to match using fnmatch/glob syntax

    Returns:
        list: List of filenames within the tar that match the pattern and are regular files
    """
    with tarfile.open(tar_path, 'r') as tar:
        all_names = tar.getnames()
        matches = [name for name in all_names
                  if fnmatch.fnmatch(name, pattern) and tar.getmember(name).isfile()]
    return matches

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

def ensure_target_size(tensor, target_size):
    """Ensure a tensor meets target size requirements, interpolating if necessary.

    Args:
        tensor: Input tensor to resize
        target_size: Target size as (height, width) tuple

    Returns:
        Tensor resized to target size if original size doesn't match,
        otherwise returns original tensor unchanged.
    """
    if tensor.shape[-2:] == torch.Size(target_size):
        return tensor  # ok
    else:
        logging.warning(f'Shape mismatch after cropping template: {tensor.shape}, interpolating to {target_size}')
        return torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        )[0]


class BOPHOT3DTemplate(Dataset):
    """Dataset class for loading template/reference images from BOP HOT3D dataset.

     Used for model-free onboarding where template images are cropped and processed
     for object detection/pose estimation.

     Args:
         template_dir (str): Directory containing template tar files
         obj_ids (list, optional): List of object IDs to include. If None, auto-detects from directory
         image_size (tuple): Target image size for processing
         num_imgs_per_obj (int): Number of images to sample per object
         **kwargs: Additional arguments
     """
    def __init__(
            self,
            template_dir,
            obj_ids,
            image_size,
            num_imgs_per_obj=50,
            **kwargs,
    ):
        self.logger = logging.getLogger(__name__)
        self.template_dir = template_dir
        self.model_free_onboarding = True
        self.dataset_name = template_dir.split("/")[-2]
        if obj_ids is None:
            obj_ids = [
                int(obj_id[4:10])  # pattern: obj_0000xxx
                for obj_id in os.listdir(template_dir)
                if obj_id.startswith('obj_') and obj_id.endswith('.tar')
            ]
            obj_ids = sorted(np.unique(obj_ids).tolist())
            self.logger.info(f"Found {obj_ids} objects in {self.template_dir}")
            if len(obj_ids) == 0 and len(os.listdir(template_dir)) > 0:
                self.logger.warning(f'no .tars found in {template_dir}, make sure you use the latest Hot3D version from HF')

        self.num_imgs_per_obj = num_imgs_per_obj  # to avoid memory issue
        self.obj_ids = obj_ids
        self.image_size = image_size

        # for HOT3D, we have black objects so we use gray background
        self.logger.info("Use gray background for HOT3D")
        self.proposal_processor = CropResizePad(
            self.image_size,
            pad_value=0.5,  # gray background
        )

    def load_template_poses(self, level_templates, pose_distribution):
        raise NotImplementedError('Model-based not implemented. Go back to https://github.com/nv-nguyen/cnos.')

    def __getitem__modelfree__(self, idx):
        """Get template data for model-free onboarding.

        Currently using Aria device for sampling ref images. Uses 214-1 stream for rgbs and 1201-2 stream for grays.

        Args:
            idx (int): Index of the object to retrieve

        Returns:
            dict: Contains:
                - templates (list): List of PIL images of cropped templates
                - template_masks (Tensor): Binary masks for the templates
                - image_paths (list): Paths to the source images
        """
        templates_cropped, masks_cropped, boxes, image_paths = [], [], [], []
        # onboarding_static -> object_ref_aria_static
        static_onboarding = True if "onboarding_static" in self.template_dir else False
        if static_onboarding:
            # In a previous HOT3D version, HOT3D named the two videos with _1 and _2 instead of _up and _down
            # Since 30fe9674782f32e1e5edba98476b6ff4300132c5 10/25, it is named _up, _down, too
            obj_tars = [
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}_up.tar",
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}_down.tar",
            ]
            num_selected_imgs = self.num_imgs_per_obj // len(obj_tars)  # e.g. 100 / 2 = 50
        else:
            obj_tars = [
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}.tar",
            ]
            num_selected_imgs = self.num_imgs_per_obj
        for obj_tar in obj_tars:
            if not osp.exists(obj_tar):
                continue

            obj_rgbs = find_in_tar(obj_tar, "*image_214-1.[pj][pn][g]")
            # support _1201-x gray stream, too
            # previously only rgb images were used for templates, but Quest3 test images are gray
            # use 1201-2 and not 1201-1 stream because in -1 stream some boxes are negative
            obj_grays = find_in_tar(obj_tar, "*image_1201-2.[pj][pn][g]")
            obj_imgfiles = obj_rgbs + obj_grays
            # masks available since 10/2025 https://groups.google.com/g/bop-benchmark/c/K0CcuRM2CQ8
            obj_maskfiles = find_in_tar(obj_tar, "*mask_214-1.png")
            obj_maskfiles += [None for _ in obj_grays]  # but available only for rgb (214-1), not for gray (1201-2)
            assert len(obj_imgfiles) == len(obj_maskfiles), (f"rgb and mask mismatch in {obj_tar}, "
                                                             f"{len(obj_imgfiles)},{len(obj_maskfiles)}")
            # If HOT3D + dynamic onboarding, we have the bbox for only the first image.
            # therefore, we select the first image only.
            if not static_onboarding:
                selected_idx = [0, 0, 0, 0, 0]  # required aggregation top k
            else:
                # select random samples
                selected_idx = list(np.random.choice(len(obj_imgfiles), num_selected_imgs, replace=False))

            for idx_img in tqdm(selected_idx):
                filename = obj_imgfiles[idx_img]
                img_suffix = str(filename).split('.image_')[-1]  # either 214-1.jpg or 1201-x.jpg
                with (tarfile.open(obj_tar, 'r') as tar):
                    # extract the image from the tar
                    imgfile = tar.extractfile(obj_imgfiles[idx_img])
                    image_np = imageio.imread(imgfile).astype(np.uint8)

                    # extract mask if available
                    mask_path = str(obj_imgfiles[idx_img]).replace(
                        f"image_{img_suffix}", f"mask_{img_suffix}").replace('jpg','png')
                    try:
                        maskfile = tar.extractfile(mask_path)
                        mask = Image.open(maskfile).copy()
                    except KeyError:
                        mask = None

                    # extract the info json that corresponds to the current image
                    json_path = str(obj_imgfiles[idx_img]).replace(f"image_{img_suffix}", "objects.json")
                    jsonfile = tar.extractfile(json_path)
                    info = json.load(jsonfile)

                image_np = np.array(Image.fromarray(image_np).convert('RGB'))
                # white mask if mask is not available
                mask_np = np.array(mask) / 255 if mask else np.ones((image_np.shape[0], image_np.shape[1]))

                template = torch.tensor(image_np / 255).permute(2, 0, 1)
                mask = torch.tensor(mask_np).unsqueeze(-1).permute(2, 0, 1)
                template = torch.where(mask > 0, template, self.proposal_processor.pad_value) # APPLY MASK

                obj_id = [k for k in info.keys()][0]
                try:
                    # box is derived from .objects.json
                    bbox = np.int32(info[obj_id][0]["boxes_amodal"][img_suffix.split('.')[0]])
                except KeyError as e:
                    self.logger.warning(f"No such key {img_suffix.split('.')[0]}, "
                                    f"available {info[obj_id][0]['boxes_amodal'].keys()}. "
                                    f"Skipping and sampling another one.")
                    selected_idx.append(np.random.choice(len(obj_imgfiles), 1).item())
                    continue

                box = torch.tensor(bbox)
                if box.min() < 0: self.logger.warning(f'{box.min()=} < 0')

                template_cropped = self.proposal_processor(images=template.unsqueeze(0), boxes=box.unsqueeze(0))[0]
                # sometimes the cropping is off by 1px (e.g. yields a 223 instead of 224 crop) -> catch this case
                template_cropped = ensure_target_size(template_cropped, target_size=[self.image_size, self.image_size])
                templates_cropped.append(T.ToPILImage()(template_cropped))  # return PIL image, not tensor

                mask_cropped = self.proposal_processor(images=mask.unsqueeze(0), boxes=box.unsqueeze(0))[0]
                # sometimes the cropping is off by 1px (e.g. yields a 223 instead of 224 crop) -> catch this case
                mask_cropped = ensure_target_size(mask_cropped, target_size=[self.image_size, self.image_size])
                masks_cropped.append(mask_cropped)

                image_paths.append(obj_imgfiles[idx_img])

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
            "frame_id": self.im_id,
            "scene_id": self.scene_id,
            "device_type": self.device_type,
            "tar_filepath": self.tar_filepath,
            "stream_id": self.stream_id,
            "clip_name": self.clip_name,
        }


class BOPHOT3DTest(BaseBOP):
    """Base dataset class for BOP HOT3D query/test images.

    Handles loading of test images from both Aria and Quest3 devices.

    Args:
        root_dir (str): Root directory of the dataset
        split (str): Dataset split ('test', 'val')
        **kwargs: Additional arguments
    """
    def __init__(self, root_dir, split, **kwargs):
        super().__init__(root_dir, split, **kwargs)

        # clip_definitions contain the mapping from scene id to device (i.e. Quest3/Aria)
        clip_definitions_filepath = f"{root_dir}/clip_definitions.json"
        clip_definitions = load_json(clip_definitions_filepath)
        self.logger.info(f"HOT3D {len(clip_definitions)=}")

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
        self.logger.info(f"len targets: {len(self._targets)}, {n_aria=}, {n_quest3=}.")
        self.logger.info(f'Using 214-1 rgb img for Aria targets and 1201-1 gray img for Quest3 targets')

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, idx):
        """Get a test image and its metadata.

       Args:
           idx (int): Index of the test target to retrieve

       Returns:
           dict: Contains:
               - image (PIL.Image): The test image
               - scene_id (int): Scene identifier
               - frame_id (int): Frame identifier
       """
        st = self._targets[idx]
        tar = tarfile.open(st.tar_filepath, mode="r")
        frame_id = st.im_id
        scene_id = st.scene_id
        stream_id = st.stream_id

        frame_key = f"{frame_id:06d}"
        image_np: np.ndarray = load_image(tar, frame_key, stream_id)
        image = Image.fromarray(np.uint8(image_np))
        return dict(
            image=image,
            scene_id=scene_id,
            frame_id=frame_id,
        )
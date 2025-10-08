import logging
from tqdm import tqdm
import os
import os.path as osp
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from src.utils.bbox_utils import CropResizePad
from src.dataloader.base_bop import BaseBOP


try:
    from bop_toolkit_lib import dataset_params
    from bop_toolkit_lib import inout as bop_inout
except ImportError:
    raise ImportError(
        "Please install bop_toolkit_lib: pip install git+https://github.com/thodan/bop_toolkit.git"
    )


# The template (reference) dataloader
class BOPTemplate(Dataset):
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
                if osp.isdir(osp.join(template_dir, obj_id)) and obj_id.startswith('obj_')
            ]
            obj_ids = sorted(np.unique(obj_ids).tolist())
            self.logger.info(f"Found {obj_ids} objects in {self.template_dir}")

        if "hot3d" in template_dir:
            raise ValueError("For loading HOT3D onboarding data, please use class BOPHOT3DTemplate.")

        self.num_imgs_per_obj = num_imgs_per_obj  # to avoid memory issue
        self.obj_ids = obj_ids
        self.image_size = image_size

        self.proposal_processor = CropResizePad(
            self.image_size,
            pad_value=0  # black background
        )

    def load_template_poses(self, level_templates, pose_distribution):
        raise NotImplementedError('Model-based not implemented. Go back to https://github.com/nv-nguyen/cnos.')

    def __getitem__modelfree__(self, idx):
        templates_cropped, masks_cropped, boxes, image_paths = [], [], [], []
        static_onboarding = True if "onboarding_static" in self.template_dir else False
        if static_onboarding:
            obj_dirs = [
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}_up",
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}_down",
            ]
            num_selected_imgs = self.num_imgs_per_obj // 2  # 100 for 2 videos

            # Objects 34-40 of HANDAL have only one "up" video as these objects are symmetric
            num_video = 0
            for obj_dir in obj_dirs:
                if osp.exists(obj_dir):
                    num_video += 1
            assert (
                num_video > 0
            ), f"No video found for object {self.obj_ids[idx]} in {self.template_dir}"
            if num_video == 1:
                num_selected_imgs = self.num_imgs_per_obj
        else:
            obj_dirs = [
                f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}",
            ]
            num_selected_imgs = self.num_imgs_per_obj
        for obj_dir in obj_dirs:
            if not osp.exists(obj_dir):
                continue
            obj_dir = Path(obj_dir)

            # list all rgb
            obj_rgbs = sorted(Path(obj_dir).glob("rgb/*.[pj][pn][g]"))
            # list all masks
            obj_masks = sorted(Path(obj_dir).glob("mask_visib/*.[pj][pn][g]"))

            assert len(obj_rgbs) == len(
                obj_masks
            ), f"rgb and mask mismatch in {obj_dir}"

            # select random samples
            selected_idx = list(np.random.choice(
                len(obj_rgbs), num_selected_imgs, replace=False
            ))

            for idx_img in tqdm(selected_idx):
                image = Image.open(obj_rgbs[idx_img])

                mask = Image.open(obj_masks[idx_img])
                image = np.asarray(image) * np.expand_dims(np.asarray(mask) > 0, -1) # apply mask
                image = Image.fromarray(image)
                bbox = mask.getbbox() # box is derived from non-zero mask

                mask = torch.from_numpy(np.array(mask) / 255).float()
                image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
                image_paths.append(obj_rgbs[idx_img])

                template = image.permute(2, 0, 1)
                mask = mask.unsqueeze(-1).permute(2, 0, 1)
                box = torch.tensor(bbox)
                if box.min() < 0:  self.logger.warning(f'{box.min()=} < 0')

                template_cropped = self.proposal_processor(images=template.unsqueeze(0), boxes=box.unsqueeze(0))[0]
                templates_cropped.append(T.ToPILImage()(template_cropped)) # return PIL image, not tensor

                mask_cropped = self.proposal_processor(images=mask.unsqueeze(0), boxes=box.unsqueeze(0))[0]
                masks_cropped.append(mask_cropped)

        masks_cropped = torch.stack(masks_cropped,dim=0)
        return {
            "templates": templates_cropped, # PIL image List
            "template_masks": masks_cropped[:, 0, :, :], # tensors
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


# The query (test) dataloader
class BOPTest(BaseBOP):
    def __init__(self, root_dir, split, **kwargs):
        super().__init__(root_dir, split, **kwargs)

        self.dataset_name = kwargs.get("dataset_name", None)
        # dp_split from bop_toolkit_lib/dataset_params is required to read keys["rgb_tpath,eval_modality","eval_sensor"]
        self.dp_split = dataset_params.get_split_params(
           Path(self.root_dir).parent,
           kwargs.get("dataset_name", None),
           split=split,
        )

        self.target_images_per_scene = {}
        self.load_required_test_images_from_target_file(split)
        self.load_list_scene(split=split)
        self.load_metaData(reset_metaData=True, split=split)
        # shuffle metadata
        self.metaData = self.metaData.sample(frac=1, random_state=2021).reset_index()

    def load_required_test_images_from_target_file(self, split='test') -> None:
        # List all the files in the target directory.
        dataset_dir = Path(self.root_dir)

        # If multiple files are found, use the bop_version to select the correct one.
        target_files = list(dataset_dir.glob(f"{split}_targets_bop*.json"))
        if len(target_files) > 1:
            bop_version = "bop19"
            if self.dataset_name in ["hot3d", "hopev2", "handal"]:
                bop_version = "bop24"
            target_files = [f for f in target_files if bop_version in str(f)]
        assert (
            len(target_files) == 1
        ), f"Expected one target file, found {len(target_files)}"
        print(f"Loading target file: {target_files[0]}")
        targets = bop_inout.load_json(str(target_files[0]))
        for item in targets:
            scene_id, im_id = int(item["scene_id"]), int(item["im_id"])
            if scene_id not in self.target_images_per_scene:
                self.target_images_per_scene[scene_id] = []
            self.target_images_per_scene[scene_id].append(im_id)

    def __getitem__(self, idx):
        rgb_path = self.metaData.iloc[idx]["rgb_path"]
        scene_id = self.metaData.iloc[idx]["scene_id"]
        frame_id = self.metaData.iloc[idx]["frame_id"]

        return dict(
            image=Image.open(rgb_path),
            scene_id=scene_id,
            frame_id=frame_id,
        )

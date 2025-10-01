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
    from bop_toolkit_lib import dataset_params, inout
except ImportError:
    raise ImportError(
        "Please install bop_toolkit_lib: pip install git+https://github.com/thodan/bop_toolkit.git"
    )

# TODO Too many if dataset_name=="hot3d". <ake a own class for Hot3D Template too.
# The template (reference) dataloader
class BOPTemplate(Dataset):
    def __init__(
        self,
        template_dir,
        obj_ids,
        image_size,
        per_modality,
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

        # for HOT3D, we have black objects so we use gray background
        if "hot3d" in template_dir:
            self.use_gray_background = True
            logging.info("Use gray background for HOT3D")
        else:
            self.use_gray_background = False
        self.num_imgs_per_obj = num_imgs_per_obj  # to avoid memory issue
        self.per_modality = per_modality # whether to extract templates modality-wise
        self.obj_ids = obj_ids
        self.image_size = image_size

        self.proposal_processor = CropResizePad(
            self.image_size,
            pad_value=0.5 if self.use_gray_background else 0,
        )

    def __len__(self):
        return len(self.obj_ids)

    def load_template_poses(self, level_templates, pose_distribution):
        raise NotImplementedError('Model-based not implemented. Go back to https://github.com/nv-nguyen/cnos.')

    def __getitem__modelfree__(self, idx):
        templates_cropped, masks_cropped, boxes, image_paths = [], [], [], []
        static_onboarding = True if "onboarding_static" in self.template_dir else False
        if static_onboarding:
            # HOT3D names the two videos with _1 and _2 instead of _up and _down
            if self.dataset_name == "hot3d":
                obj_dirs = [
                    f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}_1",
                    f"{self.template_dir}/obj_{self.obj_ids[idx]:06d}_2",
                ]
            else:
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
            if self.dataset_name == "hot3d":
                obj_rgbs = sorted(Path(obj_dir).glob("*214-1.[pj][pn][g]"))
                if self.per_modality:
                    # use 1201-2 and not 1201-1 stream because in -1 stream some boxes are negative
                    obj_rgbs += sorted(Path(obj_dir).glob("*1201-2.[pj][pn][g]"))
                obj_masks = [None for _ in obj_rgbs]
            else:
                # list all rgb
                obj_rgbs = sorted(Path(obj_dir).glob("rgb/*.[pj][pn][g]"))
                # list all masks
                obj_masks = sorted(Path(obj_dir).glob("mask_visib/*.[pj][pn][g]"))
            assert len(obj_rgbs) == len(
                obj_masks
            ), f"rgb and mask mismatch in {obj_dir}"
            
            # If HOT3D + dynamic onboarding, we have the bbox for only the first image.
            # therefore, we select the first image only.
            if self.dataset_name == "hot3d" and not static_onboarding:
                selected_idx = [0, 0, 0, 0, 0] # required aggregation top k
            else:
                # random selection here
                selected_idx = list(np.random.choice(
                    len(obj_rgbs), num_selected_imgs, replace=False
                ))
            for idx_img in tqdm(selected_idx):
                image = Image.open(obj_rgbs[idx_img])
                if self.dataset_name == "hot3d":
                    # support _1202-x Quest3, too # but maybe the Aria rgb images are enough for the templates, and Quest3 is only used for inference.
                    img_suffix = str(obj_rgbs[idx_img]).split('.image_')[-1] # either 214-1.jpg or 1201
                    json_path = str(obj_rgbs[idx_img]).replace(
                        f"image_{img_suffix}", "objects.json"
                    )
                    info = inout.load_json(json_path)
                    obj_id = [k for k in info.keys()][0]
                    try:
                        bbox = np.int32(info[obj_id][0]["boxes_amodal"][img_suffix.split('.')[0]]) # box is derived from .objects.json
                    except KeyError as e:
                        logging.warning(f"No such key {img_suffix.split('.')[0]}, available {info[obj_id][0]['boxes_amodal'].keys()}")
                        logging.warning(f'Sampling another one')
                        selected_idx.append(np.random.choice(len(obj_rgbs), 1).item())
                        continue
                    mask = np.ones((image.size[1], image.size[0])) * 255
                else:
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
                if box.min() < 0: logging.warning(f'{box.min()=} < 0')

                template_cropped = self.proposal_processor(images=template.unsqueeze(0), boxes=box.unsqueeze(0))[0]

                # sometimes the cropping yields a 223 instead of 224 crop -> catch this case
                target_size = [self.image_size, self.image_size]
                if template_cropped.shape[-2:] != torch.Size(target_size):
                    logging.warning(f'Shape mismatch after cropping template IMAGE: {template_cropped.shape[-2:]}, interpolating')
                    template_cropped = torch.nn.functional.interpolate(
                        template_cropped,
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    )

                templates_cropped.append(T.ToPILImage()(template_cropped)) # return PIL image, not tensor

                mask_cropped = self.proposal_processor(images=mask.unsqueeze(0), boxes=box.unsqueeze(0))

                # sometimes the cropping yields a 223 instead of 224 crop -> catch this case
                if mask_cropped.shape[-2:] != torch.Size(target_size):
                    logging.warning(f'Shape mismatch after cropping template MASK: {mask_cropped.shape[-2:]}, interpolating')
                    mask_cropped = torch.nn.functional.interpolate(
                        mask_cropped,
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    )

                masks_cropped.append(mask_cropped)

        #templates_cropped = torch.cat(templates_cropped,dim=0)
        masks_cropped = torch.cat(masks_cropped,dim=0)
        return {
            "templates": templates_cropped, # PIL images
            "template_masks": masks_cropped[:, 0, :, :], # tensors
            "image_paths": image_paths
        }


    def __getitem__modelbased__(self, idx):
        raise NotImplementedError('Model-based not implemented. Go back to https://github.com/nv-nguyen/cnos.')

    def __getitem__(self, idx):
        if self.model_free_onboarding:
            return self.__getitem__modelfree__(idx)
        else:
            return self.__getitem__modelbased__(idx)


# The query (test) dataloader
class BaseBOPTest(BaseBOP):
    def __init__(
        self,
        root_dir,
        split,
        **kwargs,
    ):
        self.root_dir = root_dir
        self.split = split
        # dp_split is only required for hot3d dataset.
        self.dataset_name = kwargs.get("dataset_name", None)
        self.dp_split = dataset_params.get_split_params(
           Path(self.root_dir).parent,
           kwargs.get("dataset_name", None),
           split=split,
        )
        # HOT3D test split contains all test images, not only the ones required for evaluation.
        # to speed up the inference, it is faster to only load the images required for evaluation.
        self.load_required_test_images_from_target_file(split)
        self.load_list_scene(split=split)
        self.load_metaData(reset_metaData=True, split=split)
        # shuffle metadata
        self.metaData = self.metaData.sample(frac=1, random_state=2021).reset_index()
        # self.rgb_transform = T.Compose(
        #     [
        #         T.ToTensor(),
        #         T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     ]
        # )

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
        targets = inout.load_json(str(target_files[0]))
        self.target_images_per_scene = {}
        for item in targets:
            scene_id, im_id = int(item["scene_id"]), int(item["im_id"])
            if scene_id not in self.target_images_per_scene:
                self.target_images_per_scene[scene_id] = []
            self.target_images_per_scene[scene_id].append(im_id)

    def __getitem__(self, idx):
        rgb_path = self.metaData.iloc[idx]["rgb_path"]
        scene_id = self.metaData.iloc[idx]["scene_id"]
        frame_id = self.metaData.iloc[idx]["frame_id"]
        image = Image.open(rgb_path)
        #image = self.rgb_transform(image.convert("RGB"))
        return dict(
            image=image,
            scene_id=scene_id,
            frame_id=frame_id,
        )

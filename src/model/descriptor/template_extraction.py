import hydra
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch
import PIL
from PIL import Image
import logging
import os
import os.path as osp
from collections import defaultdict
from tqdm import tqdm


def pil_center_pad(image, target_size):
    """
    Pad an image to center it in the target size.

    Args:
        image: PIL Image object
        target_size: tuple (width, height) for output size

    Returns:
        Padded PIL Image
    """
    # Create a new image with the target size and background color
    padded_image = Image.new('RGB', target_size, (0, 0, 0))  # black background

    # Calculate position to paste the original image (centered)
    x = (target_size[0] - image.width) // 2
    y = (target_size[1] - image.height) // 2

    # Paste the original image onto the padded image
    padded_image.paste(image, (x, y))
    return padded_image


class TemplateEmbExtraction:
    def __init__(self, ref_dataloader, dataset_name, obj_names, out_dir=None):
        self.obj_names = obj_names
        self.dataset_name = dataset_name
        self.ref_dataset = ref_dataloader

        self.ref_images = {} # The cropped templates extracted
        self.ref_masks = {} # The corresponding masks
        self.ref_sample_descriptors = defaultdict(list) # The resulting template embeddings; key e.g. bb_patch_objname
        self.ref_img_pths = {} # key: objname, val: the image path that was sampled by ref_dataset

        if out_dir is None: self.out_dir = osp.join(self.ref_dataset.template_dir, 'descriptors')
        if not osp.exists(self.out_dir):
            logging.info(f'Creating descriptor output dir {self.out_dir}.')
            os.makedirs(self.out_dir, exist_ok=True)

    def set_ref_images(self, cancel_after=None):
        # Calc ref samples: Retrieve images
        logging.info(f'Setting reference images for {self.obj_names}')
        for i, (object_class, ref_sample) in enumerate(zip(self.obj_names, self.ref_dataset)):
            # cache_template_images: store image name indices, too, which makes it easier to post-verify features
            self.ref_img_pths[object_class] = ref_sample["image_paths"]
            self.ref_images[object_class] = ref_sample["templates"]#[imgproc.torch_to_pil(tensor_im) for tensor_im in ref_sample["templates"]]
            self.ref_masks[object_class] = ref_sample["template_masks"].cpu()
            logging.info(f'done {i}/{len(self.obj_names)}')
            if cancel_after and (i+1) >= cancel_after: break

    def save(self, out_file, overwrite=False):
        out_pth = osp.join(self.out_dir, out_file)
        if osp.exists(out_pth) and not overwrite: raise FileExistsError()
        ref_sample_embeds = {k: v.cpu() for k, v in tqdm(self.ref_sample_descriptors.items())}
        data_to_save = {
            'template_embeds': ref_sample_embeds,
            'template_imgs': self.ref_img_pths,
            'template_masks': self.ref_masks,
        }
        # save to file so you can reload them later without recomputing
        logging.info(f'Saving to {out_pth}.')
        torch.save(data_to_save, out_pth)

    def check_saved(self, out_file, dino_with_bb_anb_patch_feats=True):
        out_pth = osp.join(self.out_dir, out_file)
        logging.info(f'opening {out_pth} of size {osp.getsize(out_pth)/1e6:.1f}M')
        x = torch.load(out_pth, map_location=torch.device('cpu'), weights_only=False)
        logging.info('entries in saved file: ', {k:len(v) for k,v in x.items()})
        assert 'template_embeds' in x.keys()
        if dino_with_bb_anb_patch_feats:
            embs_dict = x['template_embeds']
            n_objects = len(embs_dict) / 2
            sample = next(iter(embs_dict.values()))
            logging.info(sample.shape)
            k, p, d = sample.view(sample.shape[0], -1, sample.shape[-1]).shape
            logging.info(f"{n_objects} objects with {k} templates each, dict size "
                  f"{n_objects * (2 * k * p * d + 2 * k * d) / 1e6:.1f}M * {sample.dtype}")
            logging.info('keys | length | shape')
            for k, v in embs_dict.items():
                logging.info(f"{k, len(v), v[0].shape}")

    def delete(self, out_file):
        out_pth = osp.join(self.out_dir, out_file)
        logging.info(f'Removing {out_pth}.')
        os.remove(out_pth)


class DinoTemplateExtraction(TemplateEmbExtraction):
    def __init__(self, descriptor_model, ref_dataloader, obj_names, dataset_name='hopev2', n_templates=100):
        print(ref_dataloader)
        super().__init__(ref_dataloader, dataset_name, obj_names)

        self.descriptor_model = descriptor_model.model
        self.ref_dataset.num_imgs_per_obj = n_templates

    def calc_ref_embs(self):
        # Iterate over object classes (n)
        for class_name, template_imgs in tqdm(self.ref_images.items(), desc="Computing template descriptors ..."):
            assert isinstance(template_imgs[0], PIL.Image.Image), f'got {type(template_imgs[0])}'

            template_img_tensors = torch.stack([self.descriptor_model.rgb_normalize(im) for im in template_imgs]) # processor
            template_img_tensors = template_img_tensors.to(self.descriptor_model.device)

            res = self.descriptor_model.chunked_fwd(template_img_tensors)

            self.ref_sample_descriptors['x_norm_clstoken_' + class_name] = res['x_norm_clstoken']
            self.ref_sample_descriptors['x_norm_patchtokens_' + class_name] = res['x_norm_patchtokens']
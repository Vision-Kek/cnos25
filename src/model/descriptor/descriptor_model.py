from collections import defaultdict
import torch
import torchvision.transforms as T
import logging
import numpy as np

from src.utils.bbox_utils import CropResizePad


class DescriptorModel:
    def __init__(
        self,
        model,
        image_size,
        chunk_size
    ):
        logging.info(
            f"Initializing DescriptorModel..."
        )
        # not sure why torch.hub.load does only move transformer blocks, but not PatchEmbed and LayerNorm
        self.device = next(model.parameters()).device
        for module in model.modules():
            module.to(self.device)
        self.model = model
        self.model.use_patch_tokens = False  # don't concat class+patch to 2048d

        self.proposal_size = image_size
        self.chunk_size = chunk_size
        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        # use for global feature
        self.rgb_proposal_processor = CropResizePad(self.proposal_size)
        logging.info(
            f"Init DescriptorModel with {self.proposal_size=} done."
        )

    def process_rgb_proposals(self, image_pil, masks, boxes):
        """
        1. Normalize image with DINOv2 transform
        2. Mask and crop each proposals
        3. Resize each proposals to predefined longest image size
        """
        num_proposals = len(boxes)
        rgb = self.rgb_normalize(image_pil).to(boxes.device).float()
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1) # n-plicate image for each proposal
        masked_rgbs = rgbs if masks.sum()==0 else rgbs * masks.unsqueeze(1) # masks.sum==0 means masks are not used
        processed_masked_rgbs = self.rgb_proposal_processor(masked_rgbs, boxes)  # [N, 3, target_size, target_size]
        return processed_masked_rgbs

    @torch.no_grad()
    def chunked_fwd(self, image_tensor):
        n_chunks = ((len(image_tensor) - 1) // self.chunk_size) + 1
        logging.debug(f'Forwarding in {n_chunks} chunks.')
        chunks = [image_tensor[offset:offset + self.chunk_size] for offset in self.chunk_size * torch.arange(n_chunks)]

        res = defaultdict(list)
        for chunk in chunks:
            for k,v in self.model.forward_features(chunk).items():
                res[k].append(v)
        res = {k: torch.cat(v) for k,v in res.items() if isinstance(v[0],torch.Tensor)}
        return res

import hydra
import torch
import logging
import numpy as np
import os
import os.path as osp

from src.utils.detection_utils import Detections
from src.model import matching


class CNOS25:
    def __init__(self, proposal, descriptor, matching_metric, device):
        logging.info('Initializing CNOS25')
        self.device = device
        # Init proposal
        self.proposal = proposal
        # Init descriptor
        self.descriptor = descriptor
        self.feat_type = descriptor.cfg.token_type
        self.reference_descriptors = self.load_reference_descriptors()  # the templates
        # Init matching
        self.matching_strategy = matching_metric
        logging.info('Initializing CNOS25 Done.')

    def load_reference_descriptors(self):
        logging.info(f'Available descriptors {os.listdir(self.descriptor.cfg.cache_dir)}'
                     f'selected {self.descriptor.cfg.cache_file}.')
        descriptor_pth = osp.join(self.descriptor.cfg.cache_dir, self.descriptor.cfg.cache_file)
        logging.info(f'Loading descriptors from {descriptor_pth}...')

        dino_descriptors = torch.load(descriptor_pth, map_location=torch.device('cpu'), weights_only=False)
        if 'template_embeds' in dino_descriptors: # support this as well as flat dict
            logging.info('found template_embeds key')
            dino_descriptors = dino_descriptors['template_embeds']
        # select the feature type (templates) shape:(n,k,d)
        dino_descriptors = torch.stack([v for k, v in dino_descriptors.items() if k.startswith(self.feat_type)])
        logging.info(f'Loaded {self.feat_type}, {dino_descriptors.shape}.')
        if len(dino_descriptors) == 0:
            logging.info(f'No keys starting with {self.feat_type}, available {dino_descriptors.keys()}')
        elif dino_descriptors.shape[-1] == 2048:
            logging.info(f'Found {dino_descriptors.shape}, likely because of dino `torch.cat(features, dim=-1)`.  Using only [:1024]')
            dino_descriptors = dino_descriptors[...,:1024]

        return dino_descriptors.to(self.device)

    def proposal_fwd_pass(self, pil_image):
        proposals = self.proposal.model.predict_proposals(pil_image)
        proposals = Detections({'masks': proposals.mask, 'boxes': proposals.xyxy})
        # keep only max top-n
        proposals.retain_top_n_confident_detections(self.proposal.model.cfg.keep_n_top_confident_proposals)
        return proposals

    def dino_fwd_pass(self, pil_image, proposals):
        if len(proposals) == 0:
            logging.info('Empty proposals. Skipping.')
            return torch.zeros(0,self.reference_descriptors.shape[-1], device=self.device)

        proposal_img_tensors = self.descriptor.model.process_rgb_proposals(pil_image, proposals.masks, proposals.boxes)

        assert self.feat_type in ['x_norm_clstoken','etc...']
        proposal_descriptors = self.descriptor.model.chunked_fwd(proposal_img_tensors)[self.feat_type]
        return proposal_descriptors

    def classify(self, proposal_descriptors):
        if len(proposal_descriptors) == 0:
            return torch.tensor([]), torch.tensor([])

        score_per_proposal, assigned_idx_object = matching.classify_proposals_through_matching(
            object_proposal_class_embeddings=proposal_descriptors,
            ref_samples=self.reference_descriptors,
            matching_strategy=self.matching_strategy)
        return score_per_proposal, assigned_idx_object
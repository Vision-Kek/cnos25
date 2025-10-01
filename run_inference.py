import logging
import os.path as osp
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


def inference_loop(cfg, model, query_dataloader):
    kekbopeval = instantiate(cfg.bopeval)

    for i,sample in enumerate(tqdm(query_dataloader)):
        sample = sample[0] # bs=1
        if cfg.dataset_name == 'hot3d': sample['image'] = sample['image'].convert('RGB')
        start = time.time()
        # Proposal stage
        proposals = model.proposal_fwd_pass(sample['image'])

        # Classification stage
        proposal_descriptors = model.dino_fwd_pass(sample['image'], proposals)
        score_per_proposal, assigned_idx_object = model.classify(proposal_descriptors)

        torch.cuda.synchronize() # for accurate runtime measure
        runtime = time.time() - start
        kekbopeval.save_detections_to_npzfile(sample=sample,
                                              boxes=proposals.boxes[:cfg.n_max_save].cpu(),
                                              class_idcs=assigned_idx_object[:cfg.n_max_save].cpu(),
                                              scores=score_per_proposal[:cfg.n_max_save].cpu(),
                                              runtime=runtime,
                                              box_fmt='xyxy',
                                              masks=proposals.masks.cpu() if cfg.save_masks else None)

    kekbopeval.generate_json_from_npzs()
    if cfg.split != 'test': kekbopeval.measure_AP()

    if cfg.postprocess is not None:
        logging.info('Postprocessing...')
        hydra.utils.instantiate(cfg.postprocess,
                                result_json=osp.join(kekbopeval.result_dir, f'{kekbopeval.bopstyle_experiment_name}.json'))
        if cfg.split != 'test': kekbopeval.measure_AP(resultfile_name=f'nms-{kekbopeval.bopstyle_experiment_name}')
        logging.info('Postprocessing Done.')

@hydra.main(version_base=None, config_path="configs", config_name="run_inference")
def run_inference(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False) # allows adding new keys
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_path = hydra_cfg.runtime.output_dir
    logging.info(f"The outputs of hydra will be stored in: {output_path}")

    if cfg.dataset_name == 'hot3d':
        from src.dataloader import bop_hot3d
        query_dataset = bop_hot3d.BaseBOPHOT3D(root_dir=osp.join(cfg.data.root_dir, 'datasets', 'hot3d'))
    else:
        query_dataloader_config = cfg.data.query_dataloader.copy()
        logging.info("Initializing query dataloader...")
        logging.info(f"{cfg.data.root_dir=}")
        query_dataloader_config.dataset_name = cfg.dataset_name
        query_dataloader_config.split = cfg.split
        query_dataloader_config.root_dir += f"{cfg.dataset_name}"
        query_dataset = instantiate(query_dataloader_config)
    logging.info("Initializing query dataloader Done.")

    logging.info("Initializing model...")
    model = instantiate(cfg.model)
    logging.info("Initializing model Done.")

    no_collate = lambda batch: batch # don't convert to tensor/device
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=1,  # only support a single image for now
        num_workers=cfg.machine.num_workers,
        collate_fn=no_collate,
        shuffle=False,
    )
    if 'static' in cfg.onboarding:
        logging.info("Using static onboarding images")
    else:
        raise NotImplementedError

    logging.info(f"---" * 20)
    inference_loop(cfg, model, query_dataloader)
    logging.info(f"---" * 20)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_inference()

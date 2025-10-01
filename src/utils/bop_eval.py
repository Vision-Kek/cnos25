import json
import logging
import os
import os.path as osp
import torchvision
import torch
import hydra

from src.utils.detection_utils import Detections

# experiment_name is both the name of the folder where the predictions are stored as well as the {experiment_name}.json prediction file as well as used by the bop toolkit to infer dataset and split
class BopEval:
    def __init__(self, experiment_name, dataset_base, targets_to_be_evaluated,
                 dataset_name, split, bop_toolkit_dir, result_dir=None, verbose=1):
        if result_dir is None:
            try:
                hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
                result_dir = hydra_cfg.runtime.output_dir
                logging.info(f'BopEval: result_dir is not set, defaulting to hydra output_dir {result_dir}')
            except ValueError as e:
                result_dir = f'results/{experiment_name}'
                logging.warning(e, f'result_dir not set and could not get HydraConfig. Falling back to {result_dir}')
        self.verbose = verbose
        self.result_dir = result_dir
        self.dataset_base = dataset_base
        self.dataset_name = dataset_name

        self.boptoolkit_script_dir = osp.join(bop_toolkit_dir, 'scripts')

        if split not in experiment_name:
            logging.warning(f'Warn: {split} not in {experiment_name}, bop eval script infers split from name_prediction_file')
        if self.dataset_name not in experiment_name:
            logging.warning(f'Warn: {self.dataset_name} not in {experiment_name}, bop eval script infers dataset from name_prediction_file')
        if experiment_name.count('_') > 1 or experiment_name.count('-') > 1:
            logging.warning("don't use _/- in your name_exp, it'll conflict with bop toolkit eval script.")
        self.bopstyle_experiment_name = experiment_name

        if isinstance(targets_to_be_evaluated, list):
            # Only evaluate selected samples (=targets for evaluation script)
            logging.info(f'Setting evaluation targets to {targets_to_be_evaluated}.')
            self.targets_file='/tmp/tmptargets.json'
            json.dump(targets_to_be_evaluated, open(self.targets_file,'w'))
        elif isinstance(targets_to_be_evaluated, str):
            logging.info(f'Setting evaluation targets to {targets_to_be_evaluated}.')
            self.targets_file = targets_to_be_evaluated
        else: raise ValueError(f'Pass either path or list of dicts as targets_to_be_evaluated, got {type(targets_to_be_evaluated)}')

        dataset_dir=osp.join(dataset_base,dataset_name)
        assert osp.exists(dataset_dir), f'{dataset_dir} not found'

    # Calls the eval script to measure AP
    def measure_AP(self, resultfile_name=None, dryrun=False):
        if resultfile_name is None: resultfile_name = self.bopstyle_experiment_name
        cmd = f'BOP_PATH={self.dataset_base} python '\
              + osp.join(self.boptoolkit_script_dir, 'eval_bop22_coco.py')\
              + (f' --results_path={self.result_dir} --result_filenames={resultfile_name}.json '
                 f'--targets_filename={self.targets_file} --ann_type="bbox" '
                 f'--eval_path={self.result_dir}')
        logging.info(f'Executing {cmd}')
        if dryrun: return
        os.system(cmd)
        score_out_file_pth = osp.join(self.result_dir, resultfile_name, 'scores_bop22_coco_bbox.json')
        score_out_file_target_pth = osp.join(self.result_dir, f'{resultfile_name}_scores_bop22_coco_bbox.json')
        logging.info(f'eval_bop22_coco.py wrote to {score_out_file_pth}, moving to {score_out_file_target_pth}.')
        os.system(f'mv {score_out_file_pth} {score_out_file_target_pth}')
        os.system(f'rmdir {osp.join(self.result_dir, resultfile_name)}')

    def pad_to_xyxy(self, pil_image, boxes):
        # multiply all coords w. longer side cuz of padding
        l = longer_edge = max(pil_image.size)
        # output of owll is in cxcywh but results expected in xyxy
        return torchvision.ops.box_convert(boxes, 'cxcywh', 'xyxy') * torch.tensor([[l, l, l, l]], device=boxes.device)

    # boxes: either
    # * cxcywh, in a canvas(coordinates) of padded to square [l,l] where l=max(h,w), then pass `in_fmt='vit_out'`
    # * xyxy, in canvas(coordinates) of sample['image']
    def save_detections_to_npzfile(self, sample, boxes, class_idcs, scores, runtime=0, box_fmt='vit_out', masks=None):
        import time
        start = time.time()
        if box_fmt=='vit_out': boxes = self.pad_to_xyxy(sample['image'], boxes)
        elif box_fmt=='xyxy': pass
        else: raise ValueError(f'invalid box format {box_fmt}')
        detection_dict = {
            'boxes': boxes,
            'object_ids': class_idcs,
            'scores': scores
        }
        if masks is not None:
            detection_dict['masks'] = masks

        detections = Detections(detection_dict)

        out_dir = osp.join(self.result_dir, 'predictions')
        os.makedirs(out_dir, exist_ok=True)
        out_file = osp.join(out_dir, f"scene{sample['scene_id']}_frame{sample['frame_id']}")
        detections.save_to_file(
            scene_id=int(sample['scene_id']),
            frame_id=int(sample['frame_id']),
            runtime=runtime,
            file_path=out_file,
            dataset_name=self.dataset_name,
            save_mask=masks is not None
        )
        if self.verbose > 0: logging.info('Saved detections to', out_file, f'in {(time.time() - start):.3f}s')

    def generate_json_from_npzs(self, save_masks=True):
        from src.utils.detection_utils import convert_npz_to_json_loop
        convert_npz_to_json_loop(experiment_result_dir=self.result_dir,
                                 outfile_name=self.bopstyle_experiment_name,
                                 no_score_distr=True, masks=save_masks)
import hydra
import logging
import os
import os.path as osp
import numpy as np
from tqdm import tqdm

import src.utils.detection_utils
from src.utils.bop_eval import BopEval


# need the load_framewise_detections_from_json conversion function, currently can't save mask
def from_json(result_json, target_dir, nms_method, nms_thresh):
    frames_detections, frames_info = src.utils.detection_utils.load_framewise_detections_from_json(result_json)
    # print(len(frame_detections), len(frame_info))

    for frame_det, frame_info in tqdm(zip(frames_detections, frames_info)):
        getattr(frame_det, nms_method)(nms_thresh) # invoke nms here

        scene_id = f"{frame_info['scene_id']:06d}"
        frame_id = frame_info['image_id']
        out_pth = osp.join(target_dir, f"scene{scene_id}_frame{frame_id}")
        frame_det.save_to_file(scene_id, frame_id, runtime=frame_det.runtimes[0], file_path=out_pth, dataset_name='doesnt matter',
                               save_mask=False)
    logging.info(f'Finished applying nms. npzs written to {target_dir}')

def from_npzs(source_npzs, target_dir, nms_method, nms_thresh):
    for npzpth in tqdm(source_npzs):
        detections = src.utils.detection_utils.load_from_file(npzpth)
        scene_id, frame_id = npzpth.split('/')[-1].split('.')[0].split('_')
        scene_id = scene_id.replace('scene', '')  # just leave number
        frame_id = frame_id.replace('frame', '')  # just leave number
        print(np.load(npzpth)['segmentation'])

        getattr(detections, nms_method)(nms_thresh) # invoke nms here

        out_pth = osp.join(target_dir, f"scene{scene_id}_frame{frame_id}")
        detections.save_to_file(scene_id, frame_id, runtime=0, file_path=out_pth, dataset_name='doesnt matter')

# result_json: in
# out_dir: out
# npz_out_dir: temporary dir if remove_intermediate_npzs=True
def nms_result_json(result_json, out_dir=None,
                nms_method='apply_nms_per_object_id', nms_thresh=0.25,
                npz_out_dirname='predictions_nms', remove_intermediate_npzs=True):
    result_dir, result_name = osp.dirname(result_json), osp.basename(result_json).split('.json')[0]

    if out_dir is None:
        logging.info(f'postprocessing: out_dir defaulting to {result_dir=}')
        out_dir = result_dir

    npz_out_dir = osp.join(out_dir, npz_out_dirname)
    os.makedirs(npz_out_dir, exist_ok=True)

    from_json(result_json, npz_out_dir, nms_method, nms_thresh) # main call here

    # merge generated npzs back to json
    src.utils.detection_utils.convert_npz_to_json_loop(experiment_result_dir=out_dir,
                                                       outfile_name=f'nms-{result_name}',
                                                       npzs_dir=npz_out_dirname,
                                                       no_score_distr=True, masks=False)
    if remove_intermediate_npzs:
        cmd = f'rm {npz_out_dir} -r'
        logging.info(f'Removing npzs: {cmd}')
        os.system(cmd)

    return out_dir

@hydra.main(version_base=None, config_path="../../configs", config_name="run_nms")
def main(cfg):
    if cfg.out_dir == 'hydra':
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        out_dir = hydra_cfg.runtime.output_dir
    else: out_dir = cfg.out_dir

    out_dir = hydra.utils.instantiate(cfg.postprocess, result_json=cfg.result_json, out_dir=out_dir)
    # run the eval script on the generated postprocessed json
    if cfg.measure_AP:
        experiment_name = osp.basename(cfg.result_json).split('.json')[0]
        # You could also read split/dataset_name like:
        # split = experiment_name.split('-')[-1]
        # dataset_name = experiment_name.split('_')[-1].split('-')[0]
        kekbopeval = BopEval(experiment_name=experiment_name,
                             dataset_base=cfg.data.query_dataloader.root_dir,
                             targets_to_be_evaluated='val_targets_bop24.json',
                             dataset_name=cfg.dataset_name, split='val',
                             bop_toolkit_dir=cfg.bop_toolkit_dir,
                             result_dir=out_dir, verbose=0)
        kekbopeval.measure_AP(resultfile_name=f'nms-{experiment_name}')

if __name__ == "__main__":
    main()
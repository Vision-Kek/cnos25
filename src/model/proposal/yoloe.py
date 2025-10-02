import os
from types import SimpleNamespace
from PIL import Image
import logging
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLOE
from ultralytics.utils.ops import scale_masks

from src.utils.bbox_utils import boxes_rot90


class YoloE:
    def __init__(self, ultralyticsmodel, text_prompts, cfg):
        self.model = ultralyticsmodel
        self.model.to(cfg.device)
        self.conf_thresh = cfg.conf_thresh
        self.verbose = cfg.verbose
        self.cfg = cfg
        if text_prompts is not None: self.set_texts(text_prompts)

    def set_texts(self, names):
        logging.info(f'YoloE.set_texts {names}')
        names = list(names)
        self.model.set_classes(names, self.model.get_text_pe(names))

    # pil_image: list of PIL.Image or single PIL.Image
    def fwd(self, pil_images, as_sv_detection=False, verbose=False, **kwargs):
        results = self.model.predict(pil_images, conf=self.conf_thresh, verbose=self.verbose, **kwargs)

        if as_sv_detection:
            return sv.Detections.from_ultralytics(results[0])

        detections = []
        # results: list of ultralytics.engine.results.Results
        for res in results:
            # handle empty, otherwise AttributeError: 'NoneType' object has no attribute 'data'
            if res.masks is None:
                logging.warning('Warn: YoloE results: No detections.')
                mask_backscaled = torch.tensor([])
            else:
                # scale mask back to original image size
                mask_backscaled = scale_masks(masks=res.masks.data.expand(1,-1,-1,-1), shape=res.masks.orig_shape)[0]
            detections.append(SimpleNamespace(xyxy=res.boxes.xyxy,
                                              mask=mask_backscaled,
                                              cls=res.boxes.cls,
                                              conf=res.boxes.conf))
        if len(detections) == 1: return detections[0]
        else: return detections

    @staticmethod
    def create_rotation_batch(pil_image, angles):
        # PIL: angle + -> counter-clockwise ; - -> clockwise
        return [pil_image.rotate(phi, expand=True) for phi in angles]

    @staticmethod
    # param k: the rotation that was applied on the input image, sign same as in torch.rot90
    def back_rot90(yolo_res, img_size_in, k):
        for i in range(abs(k)):  # 0 times if k=0, 1 times if k=1 (90deg), 2 times if k=2 (180deg)
            yolo_res.xyxy = boxes_rot90(yolo_res.xyxy, img_size_in[::1 - 2 * i], k=-np.sign(k))  # -np.sign: back-rot
            yolo_res.mask = torch.rot90(yolo_res.mask, dims=(-2, -1), k=-np.sign(k))  # -np.sign: back-rot
        return yolo_res

    def preprocess(self, pil_image):
        angles = list(self.cfg.rotate_input_images)
        if len(angles) == 0:
            input_batch = pil_image
        else:
            input_batch = self.create_rotation_batch(pil_image, angles)
        return input_batch

    def postprocess(self, yolo_res):
        angles = list(self.cfg.rotate_input_images)
        if len(angles) == 0:
            return yolo_res
        elif len(angles) == 1:
            selected_frame_idx = 0
        else:
            assert len(yolo_res) == len(angles), len(yolo_res)
            # retain the detections on the frame that have the max overall confidences
            cumm_confs = torch.stack([
                frame_dets.conf[:self.cfg.keep_n_top_confident_proposals].sum() for frame_dets in yolo_res
            ])
            assert len(cumm_confs) == len(angles), len(cumm_confs)
            selected_frame_idx = cumm_confs.argmax()
            assert selected_frame_idx in range(len(angles)), selected_frame_idx
            yolo_res = yolo_res[selected_frame_idx]

        # need to back-rotate predicted boxes and masks to the original img orientation
        input_img_size = yolo_res.mask.shape[-1], yolo_res.mask.shape[-2] # (n,h,w) -> (x,y)
        yolo_res = self.back_rot90(yolo_res, input_img_size, k=int(angles[selected_frame_idx] / 90))
        return yolo_res

    def predict_proposals(self, pil_image):
        input_batch = self.preprocess(pil_image)
        res = self.fwd(input_batch)
        proposals = self.postprocess(res)
        return proposals

# adapted from https://github.com/THU-MIG/yoloe/blob/main/predict_text_prompt.py
def visualize(pil_image, detections, top_k=-1, show_class_label=True):
    if 0 < top_k < len(detections):
        selected = torch.tensor(detections.confidence).topk(top_k).indices
        detections = detections[selected]

    resolution_wh = pil_image.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    labels = [
        f"{class_name+' ' if show_class_label else ''}{confidence:.2f}"
        for class_name, confidence in zip(detections["class_name"], detections.confidence)
    ]

    annotated_image = pil_image.copy()
    annotated_image = sv.MaskAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        opacity=0.4
    ).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.BoxAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        thickness=thickness
    ).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_scale=text_scale,
        smart_position=True
    ).annotate(scene=annotated_image, detections=detections, labels=labels)

    return annotated_image
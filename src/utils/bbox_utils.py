import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
import matplotlib
import matplotlib.pyplot as plt
import logging
from copy import deepcopy
from typing import Tuple

from src.utils.segment_anything_transforms import ResizeLongestSide
from src.utils import imgproc


class CustomResizeLongestSide(ResizeLongestSide):
    def __init__(self, target_length: int, dividable_size: int) -> None:
        ResizeLongestSide.__init__(
            self,
            target_length=target_length,
        )
        self.dividable_size = dividable_size

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int, dividable_size: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        (newh, neww) = make_bbox_dividable((newh, neww), dividable_size)
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(
            image.shape[0],
            image.shape[1],
            self.target_length,
            dividable_size=self.dividable_size,
        )
        return np.array(resize(to_pil_image(image), target_size))

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(
            image.shape[2],
            image.shape[3],
            self.target_length,
            dividable_size=self.dividable_size,
        )
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length, self.dividable_size
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)


class CropResizePad:
    def __init__(self, target_size, pad_value=0.0):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.target_ratio = self.target_size[1] / self.target_size[0]
        self.target_h, self.target_w = target_size
        self.target_max = max(self.target_h, self.target_w)
        self.pad_value = pad_value

    def __call__(self, images, boxes):
        box_sizes = boxes[:, 2:] - boxes[:, :2]
        scale_factor = self.target_max / torch.max(box_sizes, dim=-1)[0]
        processed_images = []
        for image, box, scale in zip(images, boxes, scale_factor):
            # crop and scale
            image = image[:, box[1] : box[3], box[0] : box[2]]
            image = F.interpolate(image.unsqueeze(0), scale_factor=scale.item())[0]
            # pad and resize
            original_h, original_w = image.shape[1:]
            original_ratio = original_w / original_h

            # check if the original and final aspect ratios are the same within a margin
            if self.target_ratio != original_ratio:
                padding_top = max((self.target_h - original_h) // 2, 0)
                padding_bottom = self.target_h - original_h - padding_top
                padding_left = max((self.target_w - original_w) // 2, 0)
                padding_right = self.target_w - original_w - padding_left
                image = F.pad(
                    image, (padding_left, padding_right, padding_top, padding_bottom), value=self.pad_value
                )
            assert image.shape[1] == image.shape[2], logging.info(
                f"image {image.shape} is not square after padding"
            )
            image = F.interpolate(
                image.unsqueeze(0), scale_factor=self.target_h / image.shape[1]
            )[0]
            processed_images.append(image)
        return torch.stack(processed_images)


arr_boxcrop = lambda arr,box: arr[box[1]:box[3],box[0]:box[2]]

def coords_to_0_1(boxes, imgsize, tensorlib=torch):
    h,w = imgsize[:2]
    return tensorlib.stack([boxes[...,0]/w, boxes[...,1]/h, boxes[...,2]/w, boxes[...,3]/h],-1)

def coords_0_1_to_img_size(boxes, imgsize, tensorlib=torch):
    h,w = imgsize[:2]
    return tensorlib.stack([boxes[...,0]*w, boxes[...,1]*h, boxes[...,2]*w, boxes[...,3]*h],-1)

def coords_resize(boxes, old_imgsize, new_imgsize, tensorlib=np):
    c_01 = coords_to_0_1(boxes,old_imgsize,tensorlib)
    return coords_0_1_to_img_size(c_01,new_imgsize,tensorlib)


def clamp_bboxes(bboxes, img_width, img_height):
    """
    Clamp to image dims.
    """
    # Create min and max tensors for clamping
    min_vals = torch.tensor([0, 0, 0, 0], device=bboxes.device)
    max_vals = torch.tensor([img_width, img_height, img_width, img_height],
                            device=bboxes.device)

    return torch.clamp(bboxes, min=min_vals, max=max_vals)

def draw_bboxes_torchvision_style(image_tensor, boxes, labels, show=True):
    drawn = torchvision.utils.draw_bounding_boxes(
        image_tensor,  # image tensor [3,H,W] with values 0–255
        boxes=boxes,
        labels=[str(l) for l in labels],
        colors="red",
        width=2
    )
    if show:
        plt.imshow(imgproc.torch_to_pil(drawn))
        plt.axis("off")
        plt.show()
    else:
        fig=plt.figure(figsize=(4,4))
        plt.imshow(imgproc.torch_to_pil(drawn))
        plt.axis("off")
        canvas = matplotlib.backends.backend_agg.FigureCanvas(fig)
        canvas.draw()
        img_array = np.asarray(canvas.get_renderer().buffer_rgba())[:,:,:3] #rgba to rgb
        plt.close()
        return img_array


def xyxy_to_xywh(bbox):
    if len(bbox.shape) == 1:
        """Convert [x1, y1, x2, y2] box format to [x, y, w, h] format."""
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
    elif len(bbox.shape) == 2:
        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    else:
        raise ValueError("bbox must be a numpy array of shape (4,) or (N, 4)")


def xywh_to_xyxy(bbox):
    """Convert [x, y, w, h] box format to [x1, y1, x2, y2] format."""
    if len(bbox.shape) == 1:
        x, y, w, h = bbox
        return [x, y, x + w - 1, y + h - 1]
    elif len(bbox.shape) == 2:
        x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        return np.stack([x, y, x + w, y + h], axis=1)
    else:
        raise ValueError("bbox must be a numpy array of shape (4,) or (N, 4)")

# For 90° clockwise rotation:
# x' = y, y' = width - x
def calc_counterclockwise_rotated_coords(x1, y1, x2, y2, image_width):
    # Rotate coordinates
    new_x1 = y1
    new_y1 = image_width - x2
    new_x2 = y2
    new_y2 = image_width - x1
    return new_x1, new_y1, new_x2, new_y2

def calc_clockwise_rotated_coords(x1, y1, x2, y2, image_height):
    # Rotate coordinates
    new_x1 = image_height - y2
    new_y1 = x1
    new_x2 = image_height - y1
    new_y2 = x2
    return new_x1, new_y1, new_x2, new_y2

    """
    Rotate bounding boxes 90° clockwise or -90 counterclockwise
    k: as in np.rot90; Number of times the array is rotated by 90 degrees counterclockwise
    boxes: [[x1, y1, x2, y2], ...] in original image coordinates
    image_height, image_width: dimensions of original image
    """
def boxes_rot90(boxes, pil_image_size, k):
    image_width, image_height = pil_image_size

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    if k > 0:  rotated_coords = calc_counterclockwise_rotated_coords(x1, y1, x2, y2, image_width)
    else:  rotated_coords = calc_clockwise_rotated_coords(x1, y1, x2, y2, image_height)

    new_boxes = torch.stack(rotated_coords, dim=1)
    # Ensure coordinates are properly ordered (x1 < x2, y1 < y2)
    sorted_boxes_x,_ = torch.sort(new_boxes[:, [0, 2]], dim=1)
    sorted_boxes_y,_ = torch.sort(new_boxes[:, [1, 3]], dim=1)
    assert torch.all(sorted_boxes_x == new_boxes[:, [0, 2]]), (sorted_boxes_x, new_boxes[:, [0, 2]])
    assert torch.all(sorted_boxes_y == new_boxes[:, [1, 3]])
    return new_boxes

def get_bbox_size(bbox):
    return [bbox[2] - bbox[0], bbox[3] - bbox[1]]


def make_bbox_dividable(bbox_size, dividable_size, ceil=True):
    if ceil:
        new_size = np.ceil(np.array(bbox_size) / dividable_size) * dividable_size
    else:
        new_size = np.floor(np.array(bbox_size) / dividable_size) * dividable_size
    return new_size


def make_bbox_square(old_bbox):
    size_to_fit = np.max([old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1]])
    new_bbox = np.array(old_bbox)
    old_bbox_size = [old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1]]
    # Add padding into y axis
    displacement = int((size_to_fit - old_bbox_size[1]) / 2)
    new_bbox[1] = old_bbox[1] - displacement
    new_bbox[3] = old_bbox[3] + displacement
    # Add padding into x axis
    displacement = int((size_to_fit - old_bbox_size[0]) / 2)
    new_bbox[0] = old_bbox[0] - displacement
    new_bbox[2] = old_bbox[2] + displacement
    return new_bbox


def crop_image(image, bbox, format="xyxy"):
    if format == "xyxy":
        image_cropped = image[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
    elif format == "xywh":
        image_cropped = image[
            bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :
        ]
    return image_cropped


def force_binary_mask(mask, threshold=0.):
    mask = np.where(mask > threshold, 1, 0)
    return mask
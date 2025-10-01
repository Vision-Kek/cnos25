import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import torch.nn.functional as F


norm = lambda t: (t - t.min())/(t.max() - t.min())
denorm = lambda t,min_,max_: t*(max_-min_) + min_

percentilerange = lambda t,perc:  t.min() + perc*(t.max()-t.min())
midrange = lambda t: percentilerange(t, .5)

downsample_mask = lambda mask, H, W: F.interpolate(mask.unsqueeze(1), size=(H, W), mode='bilinear',
                                                       align_corners=False).squeeze(1)

T_torch_to_pil = transforms.ToPILImage()
torch_to_pil = lambda img_tensor: T_torch_to_pil(norm(img_tensor))
upscale_torch = lambda img_tensor,hw: transforms.Resize((hw[0],hw[1]))(img_tensor)


def ensure_pil(img):
    if "pil" in str(type(img)).lower():
        return img
    else:
        return transforms.ToPILImage()(img)

def ensure_tensor(img):
    if "tensor" in str(type(img)).lower():
        return img
    else:
        return transforms.ToTensor()(img)

def pilImageRow(*imgs, maxwidth=800, bordercolor=0x000000): # from your beloved seg utils notebook
    imgs = [ensure_pil(im) for im in imgs]
    dst = Image.new('RGB', (sum(im.width for im in imgs), imgs[0].height))
    for i, im in enumerate(imgs):
        loc = [x0, y0, x1, y1] = [i*im.width, 0, (i+1)*im.width, im.height]
        dst.paste(im, (x0, y0))
        ImageDraw.Draw(dst).rectangle(loc, width=2, outline=bordercolor)
    factor_to_big = dst.width / maxwidth
    dst = dst.resize((int(dst.width/factor_to_big),int(dst.height/factor_to_big)))
    return dst


def resize_pil_keep_aspect_ratio(image, target_width=None, target_height=None):
    """
    Resize an image while maintaining its aspect ratio.

    Args:
    - image (PIL.Image): The image to resize.
    - target_width (int, optional): The desired width. If provided, the height will be inferred.
    - target_height (int, optional): The desired height. If provided, the width will be inferred.

    Returns:
    - PIL.Image: The resized image.
    """
    if not target_width and not target_height:
        raise ValueError("You must specify either target_width or target_height.")

    original_width, original_height = image.size

    if target_width:
        # Calculate the height that maintains the aspect ratio
        new_height = int((target_width / original_width) * original_height)
        new_size = (target_width, new_height)
    elif target_height:
        # Calculate the width that maintains the aspect ratio
        new_width = int((target_height / original_height) * original_width)
        new_size = (new_width, target_height)

    return image.resize(new_size)

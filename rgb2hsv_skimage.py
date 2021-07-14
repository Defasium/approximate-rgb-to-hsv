import skimage.color as sc
import numpy as np
from rgb2hsv_cv2 import contrast

def rgb2hsv_skimage(batch):
    hsv_batch = np.empty_like(batch, dtype=np.float32)
    for i, im in enumerate(batch):
        hsv_batch[i] = sc.rgb2hsv(im)
    return hsv_batch

def hsv2rgb_skimage(batch):
    rgb_batch = np.empty_like(batch, dtype=np.float32)
    for i, im in enumerate(batch):
        rgb_batch[i] = sc.hsv2rgb(im)
    return rgb_batch

def augment_skimage(batch, h=0.5, s=1.2, v=0.9, c=1.1):
    rgb_batch = np.empty_like(batch, dtype=np.uint8)
    for i, im in enumerate(batch):
        tmp = sc.rgb2hsv(im)
        tmp[:, :, 0] *= h
        tmp[:, :, 1] *= s
        tmp[:, :, 2] *= v
        tmp = np.uint8(np.clip(tmp, 0, 255))
        rgb_batch[i] = contrast(sc.hsv2rgb(tmp), c)

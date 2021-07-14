import matplotlib.colors as mc
import numpy as np
from rgb2hsv_cv2 import contrast

def rgb2hsv_matplotlib(batch):
    hsv_batch = np.empty_like(batch, dtype=np.float32)
    for i, im in enumerate(batch):
        hsv_batch[i] = mc.rgb_to_hsv(im)
    return hsv_batch

def hsv2rgb_matplotlib(batch):
    rgb_batch = np.empty_like(batch, dtype=np.float32)
    for i, im in enumerate(batch):
        rgb_batch[i] = mc.hsv_to_rgb(im)
    return rgb_batch

def augment_matplotlib(batch, h=0.5, s=1.2, v=0.9, c=1.1):
    rgb_batch = np.empty_like(batch, dtype=np.uint8)
    for i, im in enumerate(batch):
        tmp = mc.rgb_to_hsv(im)
        tmp[:, :, 0] *= h
        tmp[:, :, 1] *= s
        tmp[:, :, 2] *= v
        tmp = np.uint8(np.clip(tmp, 0, 255))
        rgb_batch[i] = contrast(mc.hsv_to_rgb(tmp), c)

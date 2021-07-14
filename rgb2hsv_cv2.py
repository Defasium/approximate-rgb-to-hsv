
import cv2
import numpy as np
import numba
from numba import jit
from rgb2hsv import U8_0, U8_128, U8_255

def rgb2hsv_cv2(rgb):
    hsv_batch = np.empty_like(rgb, dtype=np.uint8)
    for i, im in enumerate(rgb):
        hsv_batch[i] = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    return hsv_batch


def hsv2rgb_cv2(hsv):
    rgb_batch = np.empty_like(hsv, dtype=np.uint8)
    for i, im in enumerate(hsv):
        rgb_batch[i] = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)
    return rgb_batch


@jit(nopython=True, fastmath=True, error_model='numpy', parallel=True)
def contrast(img, c):
    output = np.empty(img.shape, dtype=np.uint8)
    ba = img.shape[0]
    wi = img.shape[1]
    he = img.shape[2]
    for i in numba.prange(ba):
        for j in numba.prange(wi):
            for k in numba.prange(he):
                output[i][j][k] = max(min(U8_128+(img[i][j][k]-U8_128)*c, U8_255), U8_0)
    return output

def augment_cv2(batch, h=0.5, s=1.2, v=0.9, c=1.1):
    rgb_batch = np.empty_like(batch, dtype=np.uint8)
    for i, im in enumerate(batch):
        tmp = np.float32(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
        tmp[:, :, 0] *= h
        tmp[:, :, 1] *= s
        tmp[:, :, 2] *= v
        rgb_batch[i] = contrast(cv2.cvtColor(tmp, cv2.COLOR_HSV2RGB), c)
    return rgb_batch

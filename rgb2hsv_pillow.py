
from PIL import ImageEnhance
from PIL import Image
import numpy as np
from rgb2hsv_cv2 import contrast

enhancer1 = ImageEnhance.Sharpness
enhancer2 = ImageEnhance.Color
enhancer3 = ImageEnhance.Contrast
enhancer4 = ImageEnhance.Brightness

def enhance(img, f1, f2, f3, f4):
    for enc, f in zip([enhancer1, enhancer2, enhancer4],
                      [f1, f2, f4]):
        img = enc(img).enhance(f1)
    return img

def augment_pillow(batch):
    rgb_batch = np.empty_like(batch, dtype=np.uint8)
    for i, im in enumerate(batch):
        rgb_batch[i] = np.array(enhance(Image.fromarray(im), 2, 0, 2, 0.8))

def augment_pillow_2(batch, h=0.5, s=1.2, v=0.9, c=1.1):
    rgb_batch = np.empty_like(batch, dtype=np.uint8)
    for i, im in enumerate(batch):
        tmp = np.float32(Image.fromarray(im).convert('HSV')).copy()
        tmp[:, :, 0] *= h
        tmp[:, :, 1] *= s
        tmp[:, :, 2] *= v
        tmp = np.uint8(np.clip(tmp, 0, 255))
        rgb_batch[i] = contrast(np.float32(Image.fromarray(tmp, mode='HSV').convert('RGB')), c)

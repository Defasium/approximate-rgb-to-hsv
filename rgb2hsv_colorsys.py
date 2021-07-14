import numpy as np
from colorsys import rgb_to_hsv, hsv_to_rgb
from rgb2hsv_cv2 import contrast

rgb_to_hsv_vect = np.vectorize(rgb_to_hsv)
hsv_to_rgb_vect = np.vectorize(hsv_to_rgb)

def augment_colorsys(batch, h=0.5, s=1.2, v=0.9, c=1.1):
    rgb_batch = np.empty_like(batch, dtype=np.uint8)
    for i, im in enumerate(batch):
        tmp = im.reshape(-1, 3)
        hc, sc, vc = rgb_to_hsv_vect(tmp[:, 0], tmp[:, 1], tmp[:, 2])
        hc *= h
        sc *= s
        vc = vc*v
        rc, gc, bc = hsv_to_rgb_vect(hc, sc, vc)
        rgb_batch[i] = contrast(np.vstack([rc, gc, bc]).T.reshape(im.shape), c)
    return rgb_batch

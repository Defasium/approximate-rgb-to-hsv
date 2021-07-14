
import numpy as np
import numba
from numba import jit

# HSV: Hue, Saturation, Value
# H: position in the spectrum
# S: color saturation ("purity")
# V: color brightness
# https://github.com/python/cpython/blob/main/Lib/colorsys.py

HALF_BYTE_U8 = np.uint8(128)
BYTE_U8 = np.uint8(255)
ZERO_U8 = np.uint8(0)
SIX_FP32 = np.float32(6)
ZERO_FP32 = np.float32(0.)
ONE_FP32 = np.float32(1.)
ZERO_I16 = np.int16(0)
ONE_I16 = np.int16(1)
ONE_SIXTH_FP32 = np.float32(1/6)
ONE_THIRD_FP32 = np.float32(1/3)
TWO_THIRDS_FP32 = np.float32(2/3)
U8_128 = np.int32(128)
U8_0 = np.int32(0)
U8_255 = np.int32(255)
INVERSE_U8 = np.zeros((256,), dtype=np.float64)
for ii in range(1, INVERSE_U8.size):
    INVERSE_U8[ii] = ONE_SIXTH_FP32 * 1/ii


@jit(nopython=True, fastmath=True, cache=False,
     error_model='numpy', parallel=False)
def rgb_to_hsv(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return ZERO_FP32, ZERO_FP32, maxc
    s = np.int16(maxc-minc) / (maxc)
    inv_delta = INVERSE_U8[maxc-minc]
    if r == maxc:
        h = np.int16(g-b) * inv_delta
    else:
        if g == maxc:
            h = ONE_THIRD_FP32 + np.int16(b-r)*inv_delta
        else:
            h = TWO_THIRDS_FP32 + np.int16(r-g)*inv_delta
    return h, s, v

@jit(nopython=True, fastmath=True, error_model='numpy')
def p(v, s):
    return v*(ONE_FP32 - s)

@jit(nopython=True, fastmath=True, error_model='numpy')
def q(v, s, f):
    return v*(ONE_FP32 - s*f)

@jit(nopython=True, fastmath=True, error_model='numpy')
def t(v, s, f):
    return v*(ONE_FP32 - s*(ONE_FP32-f))

@jit(nopython=True, fastmath=True, cache=True, error_model='numpy', parallel=False)
def hsv_to_rgb(h, s, v):
    if not s:
        return v, v, v
    i = np.int8(h * SIX_FP32) # takes into account negative hues
    f = np.float32((h * SIX_FP32) - i)
    i = i%6
    if i == 0:
        return v, t(v, s, f), p(v, s)
    if i == 1:
        return q(v, s, f), v, p(v, s)
    if i == 2:
        return p(v, s), v, t(v, s, f)
    if i == 3:
        return p(v, s), q(v, s, f), v
    if i == 4:
        return t(v, s, f), p(v, s), v
    return v, p(v, s), q(v, s, f)
    # Cannot get here


@jit(nopython=True, fastmath=True, error_model='numpy', parallel=True)
def rgb2hsv_numba(batch):
    output = np.empty_like(batch, dtype=np.uint8)
    bshape, wshape, hshape, _ = batch.shape
    factor = 255.
    for u in numba.prange(bshape):
        for i in numba.prange(wshape):
            for j in numba.prange(hshape):
                r, g, b = batch[u, i, j, 0], batch[u, i, j, 1], batch[u, i, j, 2]
                h, s, v = rgb_to_hsv(r, g, b)
                output[u, i, j, 0] = h * factor
                output[u, i, j, 1] = s * factor
                output[u, i, j, 2] = v
    return output


@jit(nopython=True, fastmath=True, error_model='numpy', parallel=True)
def hsv2rgb_numba(batch):
    output = np.empty_like(batch, dtype=np.uint8)
    bshape, wshape, hshape, _ = batch.shape
    factor = np.float32(1/255.)
    for u in numba.prange(bshape):
        for i in numba.prange(wshape):
            for j in numba.prange(hshape):
                arr = hsv_to_rgb(batch[u, i, j, 0] * factor,
                                 batch[u, i, j, 1] * factor,
                                 batch[u, i, j, 2])
                for c in numba.prange(3):
                    output[u, i, j, c] = arr[c]
    return output


def augment_numba(batch, colormap=None, hsvmap=None,
                  hues=None, sats=None, brights=None, contrasts=None):
    if colormap is not None:
        return augment_numba_cache(batch, colormap, hsvmap,
                                   hues, sats, brights, contrasts)
    return augment_numba_direct(batch, hues, sats, brights, contrasts)


@jit(nopython=True, fastmath=True, nogil=True, error_model='numpy', parallel=True)
def augment_numba_direct(batch, hues=None, sats=None, brights=None, contrasts=None):
    output = np.empty_like(batch, dtype=np.uint8)
    bshape, wshape, hshape, _ = batch.shape
    h_factor = 1.
    s_factor = 1.
    b_factor = 1.
    c_factor = 1.
    for u in numba.prange(bshape):
        if hues is not None:
            h_factor = hues[u]
        if sats is not None:
            s_factor = sats[u]
        if brights is not None:
            b_factor = brights[u]
        if contrasts is not None:
            c_factor = contrasts[u]
        for i in numba.prange(wshape):
            for j in numba.prange(hshape):
                r, g, b = batch[u, i, j, 0], batch[u, i, j, 1], batch[u, i, j, 2]
                h, s, v = rgb_to_hsv(r, g, b)
                arr = hsv_to_rgb(h * h_factor, s * s_factor, v * b_factor)
                for k in numba.prange(3):
                    output[u, i, j, k] = min(max((U8_128 + (arr[k] - U8_128) * c_factor),
                                                 U8_0), U8_255)
    return output


@jit(nopython=True, fastmath=True, nogil=False, error_model='numpy', parallel=True)
def augment_numba_cache(batch, colormap, hsvmap, qf=0,
                        hues=None, sats=None, brights=None, contrasts=None):
    output = np.empty(batch.shape, dtype=np.uint8)
    bshape, wshape, hshape, _ = batch.shape
    h_factor = 1.
    s_factor = 1.
    b_factor = 1.
    c_factor = 1.
    if qf > 0:
        for u in numba.prange(bshape):
            if hues is not None:
                h_factor = hues[u]
            if sats is not None:
                s_factor = sats[u]
            if brights is not None:
                b_factor = brights[u]
            if contrasts is not None:
                c_factor = contrasts[u]
            for i in numba.prange(wshape):
                for j in numba.prange(hshape):
                    rr, gg, bb = batch[u, i, j, 0] >> qf, \
                                 batch[u, i, j, 1] >> qf, \
                                 batch[u, i, j, 2] >> qf
                    hh, ss, vv = colormap[rr, gg, bb, 0], \
                                 colormap[rr, gg, bb, 1], \
                                 colormap[rr, gg, bb, 2]
                    hh = np.uint8(hh * h_factor) >> qf
                    ss = np.uint16(max(U8_0,
                                       min(ss * s_factor, U8_255))) >> qf
                    vv = np.uint16(max(U8_0,
                                       min(vv * b_factor, U8_255))) >> qf
                    for k in numba.prange(3):
                        output[u, i, j, k] = min(max((U8_128 +
                                                      (hsvmap[hh, ss, vv, k] -
                                                       U8_128) * c_factor), U8_0),
                                                 U8_255)
    else:
        for u in numba.prange(bshape):
            if hues is not None:
                h_factor = hues[u]
            if sats is not None:
                s_factor = sats[u]
            if brights is not None:
                b_factor = brights[u]
            if contrasts is not None:
                c_factor = contrasts[u]
            for i in numba.prange(wshape):
                for j in numba.prange(hshape):
                    rr, gg, bb = batch[u, i, j, 0], batch[u, i, j, 1], batch[u, i, j, 2]
                    hh, ss, vv = colormap[rr, gg, bb, 0], \
                                 colormap[rr, gg, bb, 1], \
                                 colormap[rr, gg, bb, 2]
                    hh = np.uint8(hh * h_factor)
                    ss = np.uint16(max(U8_0,
                                       min(ss * s_factor, U8_255)))
                    vv = np.uint16(max(U8_0,
                                       min(vv * b_factor, U8_255)))
                    for k in numba.prange(3):
                        output[u, i, j, k] = min(max((U8_128 +
                                                      (hsvmap[hh, ss, vv, k] -
                                                       U8_128) * c_factor), U8_0),
                                                 U8_255)
    return output


@jit(nopython=True, fastmath=True, error_model='numpy', parallel=True)
def get_all_hsv_codes(rr, gg, bb, return_orig=False, qf=1):
    output = np.empty((rr // qf,
                       gg // qf,
                       bb // qf, 3), dtype=np.uint8)
    output2 = np.empty((rr, gg, bb, 3), dtype=np.uint8)
    for r in numba.prange(output.shape[0]):
        for g in numba.prange(output.shape[1]):
            for b in numba.prange(output.shape[1]):
                output2[r, g, b, 0] = r
                output2[r, g, b, 1] = g
                output2[r, g, b, 2] = b
                rr = r // qf * qf
                gg = g // qf * qf
                bb = b // qf * qf
                h, s, v = rgb_to_hsv(rr, gg, bb)
                rr //= qf
                gg //= qf
                bb //= qf
                output[rr, gg, bb, 0] = h * 255
                output[rr, gg, bb, 1] = s * 255
                output[rr, gg, bb, 2] = v
    if return_orig:
        return output2
    return output


@jit(nopython=True, fastmath=True, error_model='numpy', parallel=True, boundscheck=False)
def rgb2hsv_cache(batch, colormap, qf=0):
    output = np.empty_like(batch, dtype=np.uint8)
    ba = batch.shape[0]
    wi = batch.shape[1]
    he = batch.shape[2]
    if qf:
        for i in numba.prange(ba):
            for j in numba.prange(wi):
                for k in numba.prange(he):
                    rr = batch[i, j, k, 0] >> qf
                    gg = batch[i, j, k, 1] >> qf
                    bb = batch[i, j, k, 2] >> qf
                    for c in numba.prange(3):
                        output[i, j, k, c] = colormap[rr, gg, bb, c]
    else:
        for i in numba.prange(ba):
            for j in numba.prange(wi):
                for k in numba.prange(he):
                    rr = batch[i, j, k, 0]
                    gg = batch[i, j, k, 1]
                    bb = batch[i, j, k, 2]
                    for c in numba.prange(3):
                        output[i, j, k, c] = colormap[rr, gg, bb, c]
    return output



@jit(nopython=True, fastmath=True, error_model='numpy', parallel=True,
     locals=dict())
def rgb_to_hsv_numpy(r: np.ndarray, g: np.ndarray, b: np.ndarray):
    maxc = np.maximum(r, np.maximum(g, b))
    minc = np.minimum(r, np.minimum(g, b))
    v = maxc
    mask0 = minc == maxc
    s = (maxc - minc) / maxc
    inv_delta = ONE_SIXTH_FP32 / (maxc - minc) #1/6 = 0.1(6)
    rc = r * inv_delta
    gc = g * inv_delta
    bc = b * inv_delta
    mask1 = ONE_FP32 * (r == maxc)
    mask2 = (ONE_FP32 - mask1) * (g == maxc)
    mask3 = ONE_FP32 - mask2
    h = (mask3 * (TWO_THIRDS_FP32 + rc-gc) + \
         mask1 * (gc - bc) + \
         mask2 * (ONE_THIRD_FP32 + bc - rc))
    h[np.where(mask0)] = 0
    s[np.where(mask0)] = 0
    return h, s, v

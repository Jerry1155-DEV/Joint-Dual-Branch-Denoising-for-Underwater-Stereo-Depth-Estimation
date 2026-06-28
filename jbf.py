import cv2
import numpy as np


def _jbf_to_float01(x):
    y = x.astype(np.float64)

    if y.size > 0 and y.max() > 1.0:
        y = y / 255.0

    return y


def _jbf_as_3d(x):
    if x.ndim == 2:
        return x[:, :, None]

    return x


def _jbf_gray(x):
    if x.ndim == 2:
        return x[:, :, None]

    if x.shape[2] == 1:
        return x

    return (0.114 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.299 * x[:, :, 2])[:, :, None]


def _jbf_range_pair(center, neighbor):
    center_3d = _jbf_as_3d(center)
    neighbor_3d = _jbf_as_3d(neighbor)

    if center_3d.shape[2] == neighbor_3d.shape[2]:
        return center_3d, neighbor_3d

    return _jbf_gray(center), _jbf_gray(neighbor)


def warp_right_to_left(right, disparity):
    right_f = _jbf_to_float01(right)

    if disparity is None:
        return right_f

    d = disparity.astype(np.float32)

    if d.ndim == 3:
        d = d[:, :, 0]

    h, w = right_f.shape[:2]

    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )

    map_x = x - d
    map_y = y

    return cv2.remap(
        right_f.astype(np.float32),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    ).astype(np.float64)


def joint_bilateral_filter_jbf(
    src,
    guide=None,
    disparity=None,
    r=5,
    sigma_s=3.0,
    sigma_r=0.08,
    eps=1e-8
):
    if not isinstance(src, np.ndarray) or src.ndim not in (2, 3):
        raise ValueError("src must be a grayscale or BGR image")

    src_f = _jbf_to_float01(src)

    if guide is None:
        neighbor_f = src_f
    else:
        if guide.shape[:2] != src.shape[:2]:
            raise ValueError("src and guide must have the same height and width")

        neighbor_f = warp_right_to_left(guide, disparity) if disparity is not None else _jbf_to_float01(guide)

    r = int(r)

    if r <= 0:
        return src.copy()

    sigma_s = float(sigma_s)
    sigma_r = float(sigma_r)

    if sigma_r > 1.0:
        sigma_r = sigma_r / 255.0

    src_value = _jbf_as_3d(src_f)
    center_range, neighbor_range = _jbf_range_pair(src_f, neighbor_f)

    h, w = src.shape[:2]

    yy, xx = np.meshgrid(
        np.arange(-r, r + 1),
        np.arange(-r, r + 1),
        indexing="ij"
    )

    spatial = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_s ** 2))

    src_pad = np.pad(src_value, ((r, r), (r, r), (0, 0)), mode="reflect")
    guide_pad = np.pad(neighbor_range, ((r, r), (r, r), (0, 0)), mode="reflect")

    acc = np.zeros_like(src_value, dtype=np.float64)
    norm = np.zeros((h, w, 1), dtype=np.float64)

    for yi, dy in enumerate(range(-r, r + 1)):
        for xi, dx in enumerate(range(-r, r + 1)):
            src_q = src_pad[r + dy:r + dy + h, r + dx:r + dx + w, :]
            guide_q = guide_pad[r + dy:r + dy + h, r + dx:r + dx + w, :]

            diff = center_range - guide_q

            if diff.shape[2] == 1:
                dist2 = diff[:, :, 0] ** 2
            else:
                dist2 = np.mean(diff ** 2, axis=2)

            weight = spatial[yi, xi] * np.exp(-dist2 / (2.0 * sigma_r ** 2 + eps))

            acc += src_q * weight[:, :, None]
            norm += weight[:, :, None]

    out = acc / (norm + eps)

    if src.ndim == 2:
        out = out[:, :, 0]

    if np.issubdtype(src.dtype, np.integer):
        return np.clip(out * 255.0 + 0.5, 0, 255).astype(src.dtype)

    return np.clip(out, 0.0, 1.0).astype(src.dtype)

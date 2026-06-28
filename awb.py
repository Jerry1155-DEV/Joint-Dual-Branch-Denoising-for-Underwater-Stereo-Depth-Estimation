import numpy as np


def _awb_srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _awb_cct_mccamy(rgb, eps=1e-8):
    linear = _awb_srgb_to_linear(np.clip(rgb, 0.0, 1.0))
    r = linear[:, :, 0]
    g = linear[:, :, 1]
    b = linear[:, :, 2]

    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    denom = X + Y + Z + eps
    x = X / denom
    y = Y / denom

    n = (x - 0.3320) / (y - 0.1858 + eps)
    cct = -449.0 * n ** 3 + 3525.0 * n ** 2 - 6823.3 * n + 5520.33

    valid = np.isfinite(cct) & (cct > 0)

    if np.any(valid):
        fill = np.median(cct[valid])
        cct = np.where(valid, cct, fill)
    else:
        cct = np.full(cct.shape, 6500.0, dtype=np.float64)

    return np.clip(cct, 1000.0, 40000.0)


def adaptive_white_balance_awb(
    image,
    n_bins=5,
    luminance_quantiles=(0.10, 0.95),
    min_pixels=64,
    gain_clip=(0.4, 3.0),
    eps=1e-6
):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image with shape HxWx3")

    bgr = image.astype(np.float64)

    if bgr.max() > 1.0:
        bgr = bgr / 255.0

    rgb = bgr[:, :, ::-1]

    h, w, _ = rgb.shape

    r_ch = rgb[:, :, 0]
    g_ch = rgb[:, :, 1]
    b_ch = rgb[:, :, 2]

    luminance = 0.2126 * r_ch + 0.7152 * g_ch + 0.0722 * b_ch

    q0, q1 = luminance_quantiles

    if q0 > 1.0 or q1 > 1.0:
        q0, q1 = q0 / 100.0, q1 / 100.0

    low, high = np.quantile(luminance, [q0, q1])

    channel_max = rgb.max(axis=2)
    channel_min = rgb.min(axis=2)

    confident = (
        (luminance >= low)
        & (luminance <= high)
        & (channel_max < 0.98)
        & (channel_min > 0.01)
    )

    if np.count_nonzero(confident) < min_pixels:
        confident = np.ones((h, w), dtype=bool)

    cct = _awb_cct_mccamy(rgb)
    cct_feature = np.log(cct + eps)
    values = cct_feature[confident]

    if values.size == 0:
        return image.copy()

    n_bins = max(1, int(n_bins))

    if values.max() - values.min() < eps:
        bin_id = np.zeros((h, w), dtype=np.int32)
    else:
        edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
        bin_id = np.digitize(cct_feature, edges[1:-1], right=False).astype(np.int32)

    mean_global = rgb[confident].mean(axis=0)

    global_gain = np.array(
        [
            mean_global[1] / (mean_global[0] + eps),
            1.0,
            mean_global[1] / (mean_global[2] + eps)
        ],
        dtype=np.float64
    )

    global_gain = np.clip(global_gain, gain_clip[0], gain_clip[1])

    gains = np.tile(global_gain.reshape(1, 3), (n_bins, 1))

    for k in range(n_bins):
        mask = confident & (bin_id == k)

        if np.count_nonzero(mask) >= min_pixels:
            mean_k = rgb[mask].mean(axis=0)

            gain_k = np.array(
                [
                    mean_k[1] / (mean_k[0] + eps),
                    1.0,
                    mean_k[1] / (mean_k[2] + eps)
                ],
                dtype=np.float64
            )

            gains[k] = np.clip(gain_k, gain_clip[0], gain_clip[1])

    out_rgb = rgb * gains[bin_id]
    out_bgr = np.clip(out_rgb[:, :, ::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)

    return out_bgr

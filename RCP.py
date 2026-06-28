import cv2
import numpy as np


def red_inverse_channel_prior_rcp(
    image,
    radius=7,
    top_percent=0.001,
    t_min=0.10,
    scattering=(1.0, 1.0, 1.0),
    omega=1.0,
    ratio_clip=(0.05, 5.0),
    eps=1e-6
):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image with shape HxWx3")

    bgr = image.astype(np.float64)

    if bgr.max() > 1.0:
        bgr = bgr / 255.0

    rgb = bgr[:, :, ::-1]

    h, w, _ = rgb.shape

    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    score = R - np.maximum(G, B)

    count = max(1, int(round(h * w * top_percent)))
    flat_score = score.reshape(-1)
    idx = np.argpartition(flat_score, -count)[-count:]

    B_inf = rgb.reshape(-1, 3)[idx].mean(axis=0)
    B_inf = np.clip(B_inf, eps, 1.0 - eps)

    kernel_size = 2 * int(radius) + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    min_inv_R = cv2.erode(1.0 - R, kernel, borderType=cv2.BORDER_REFLECT_101)

    t_R = 1.0 - omega * min_inv_R / (1.0 - B_inf[0] + eps)
    t_R = np.clip(t_R, eps, 1.0)

    b = np.asarray(scattering, dtype=np.float64)
    b = np.clip(b, eps, None)

    ratio_G = (b[1] * B_inf[0]) / (b[0] * B_inf[1] + eps)
    ratio_B = (b[2] * B_inf[0]) / (b[0] * B_inf[2] + eps)

    ratio_G = float(np.clip(ratio_G, ratio_clip[0], ratio_clip[1]))
    ratio_B = float(np.clip(ratio_B, ratio_clip[0], ratio_clip[1]))

    t_G = np.power(t_R, ratio_G)
    t_B = np.power(t_R, ratio_B)

    transmission = np.stack([t_R, t_G, t_B], axis=2)
    transmission = np.maximum(transmission, t_min)

    J = (rgb - B_inf.reshape(1, 1, 3)) / transmission + B_inf.reshape(1, 1, 3)

    out_bgr = np.clip(J[:, :, ::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)

    return out_bgr

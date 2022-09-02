import numpy as np


def rgb_to_packed_colors(
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
) -> np.ndarray:
    r, g, b = r.astype(np.uint32), g.astype(np.uint32), b.astype(np.uint32)
    packed_colors = r << 16 | g << 8 | b
    return packed_colors


def packed_colors_to_rgb(packed_colors):
    r = np.array((packed_colors & 0xff0000) >> 16, dtype=np.uint8)
    g = np.array((packed_colors & 0xff00) >> 8, dtype=np.uint8)
    b = np.array((packed_colors & 0xff), dtype=np.uint8)
    return r, g, b

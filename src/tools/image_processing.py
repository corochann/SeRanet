from __future__ import print_function
import numpy as np


def nearest_neighbor_2x(pre_scaled_x):
    # print('pre_scaled_x.shape', pre_scaled_x.shape)
    scaled_x = np.ndarray((pre_scaled_x.shape[0],
                         pre_scaled_x.shape[1],
                         pre_scaled_x.shape[2] * 2,
                         pre_scaled_x.shape[3] * 2),
                         )
    # dtype=train_set_x.dtype)
    scaled_x[:, :, ::2, ::2] = pre_scaled_x[:] / 255.
    scaled_x[:, :, 1::2, ::2] = pre_scaled_x[:] / 255.
    scaled_x[:, :, ::2, 1::2] = pre_scaled_x[:] / 255.
    scaled_x[:, :, 1::2, 1::2] = pre_scaled_x[:] / 255.
    return scaled_x.astype(np.float32)


def image_padding(x, pad):
    # PADDING
    # print('image padding ', image_padding, ' pixels...')
    new_img = np.ndarray((x.shape[0], x.shape[1],
                          x.shape[2] + 2 * pad,
                          x.shape[3] + 2 * pad))
    for i in np.arange(x.shape[0]):
        for j in np.arange(x.shape[1]):
            new_img[i, j, :, :] = np.pad(x[i, j, :, :], pad, "edge")
    return new_img.astype(np.float32)


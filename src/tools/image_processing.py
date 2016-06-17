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



# PREPROCESSING
def preprocess(pre_scaled_x, image_padding=0):
    """
    preprocessing image consists of 3 parts
    1. Scaling: scale original input image to twice scale, using nearest neighbor method.
    2. Normalization: normalize each pixel's value from 0~255 to 0~1
    3. Padding: pad edge using np.pad.
                because image size will reduce during convolution in neural network
    :param pre_scaled_x:  original input image (numpy.array)
    :param image_padding: value of pixel to be padded
    :return:
    """
    print('pre_scaled_x.shape', pre_scaled_x.shape)
    scaled_x = np.empty((pre_scaled_x.shape[0],
                         pre_scaled_x.shape[1],
                         pre_scaled_x.shape[2] * 2,
                         pre_scaled_x.shape[3] * 2),
                        )
    # dtype=train_set_x.dtype)
    scaled_x[:, :, ::2, ::2] = pre_scaled_x[:] / 255.
    scaled_x[:, :, 1::2, ::2] = pre_scaled_x[:] / 255.
    scaled_x[:, :, ::2, 1::2] = pre_scaled_x[:] / 255.
    scaled_x[:, :, 1::2, 1::2] = pre_scaled_x[:] / 255.

    # print('pre_scaled_x = ', pre_scaled_x)
    # print('scaled_x = ', scaled_x)

    # PADDING
    if image_padding > 0:
        # print('image padding ', image_padding, ' pixels...')
        new_img = np.empty((scaled_x.shape[0], scaled_x.shape[1],
                            scaled_x.shape[2] + 2 * image_padding,
                            scaled_x.shape[3] + 2 * image_padding), dtype=scaled_x.dtype)
        for i in np.arange(scaled_x.shape[0]):
            for j in np.arange(scaled_x.shape[1]):
                new_img[i, j, :, :] = np.pad(scaled_x[i, j, :, :], image_padding, "edge")
        scaled_x = new_img

    return scaled_x.astype(np.float32)

def preprocess_scale_padding(pre_scaled_x, image_padding=0):
    """
    preprocessing image consists of 3 parts
    1. Scaling: scale original input image to twice scale, using nearest neighbor method.
    2. Normalization: normalize each pixel's value from 0~255 to 0~1
    3. Padding: pad edge using np.pad.
                because image size will reduce during convolution in neural network
    :param pre_scaled_x:  original input image (numpy.array)
    :param image_padding: value of pixel to be padded
    :return:
    """
    print('pre_scaled_x.shape', pre_scaled_x.shape)
    scaled_x = np.empty((pre_scaled_x.shape[0],
                         pre_scaled_x.shape[1],
                         pre_scaled_x.shape[2] * 2,
                         pre_scaled_x.shape[3] * 2),
                        )
    # dtype=train_set_x.dtype)
    scaled_x[:, :, ::2, ::2] = pre_scaled_x[:] / 255.
    scaled_x[:, :, 1::2, ::2] = pre_scaled_x[:] / 255.
    scaled_x[:, :, ::2, 1::2] = pre_scaled_x[:] / 255.
    scaled_x[:, :, 1::2, 1::2] = pre_scaled_x[:] / 255.

    # print('pre_scaled_x = ', pre_scaled_x)
    # print('scaled_x = ', scaled_x)

    # PADDING
    if image_padding > 0:
        # print('image padding ', image_padding, ' pixels...')
        new_img = np.empty((scaled_x.shape[0], scaled_x.shape[1],
                            scaled_x.shape[2] + 2 * image_padding,
                            scaled_x.shape[3] + 2 * image_padding), dtype=scaled_x.dtype)
        for i in np.arange(scaled_x.shape[0]):
            for j in np.arange(scaled_x.shape[1]):
                new_img[i, j, :, :] = np.pad(scaled_x[i, j, :, :], image_padding, "edge")
        scaled_x = new_img

    return scaled_x.astype(np.float32)


# PREPROCESSING
def preprocess_without_scale(pre_scaled_x, image_padding=0):
    """
    preprocessing image consists of 2 parts
    1. Normalization: normalize each pixel's value from 0~255 to 0~1
    2. Padding: pad edge using np.pad.
                because image size will reduce during convolution in neural network
    :param pre_scaled_x:  original input image (numpy.array)
    :param image_padding: value of pixel to be padded
    :return:
    """
    print('pre_scaled_x.shape', pre_scaled_x.shape)
    scaled_x = np.empty(pre_scaled_x.shape)
    # dtype=train_set_x.dtype)
    scaled_x[:] = pre_scaled_x[:] / 255.
    # print('pre_scaled_x = ', pre_scaled_x)
    # print('scaled_x = ', scaled_x)

    # PADDING
    if image_padding > 0:
        # print('image padding ', image_padding, ' pixels...')
        new_img = np.empty((scaled_x.shape[0], scaled_x.shape[1],
                            scaled_x.shape[2] + 2 * image_padding,
                            scaled_x.shape[3] + 2 * image_padding), dtype=scaled_x.dtype)
        for i in np.arange(scaled_x.shape[0]):
            for j in np.arange(scaled_x.shape[1]):
                new_img[i, j, :, :] = np.pad(scaled_x[i, j, :, :], image_padding, "edge")
        scaled_x = new_img

    return scaled_x.astype(np.float32)


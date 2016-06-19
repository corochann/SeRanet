import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import os

import src.tools.image_processing as image_processing

""" Configuration """
file_path = os.path.dirname(os.path.realpath(__file__))
arch_folder = os.path.join(file_path, '../../data/arch/basic_cnn_small')

total_padding = 10


class basic_cnn_small(Chain):
    """
    Basic CNN model.
    The network consists of Convolutional layer and leaky relu is used for activation
    """
    def __init__(self, inout_ch):
        super(basic_cnn_small, self).__init__(
            conv1=L.Convolution2D(in_channels=inout_ch, out_channels=16, ksize=3, stride=1),
            conv2=L.Convolution2D(16, 16, 3, stride=1),
            conv3=L.Convolution2D(16, 32, 3, stride=1),
            conv4=L.Convolution2D(32, 32, 3, stride=1),
            conv5=L.Convolution2D(32, inout_ch, 3, stride=1),
        )
        self.train = True

    def __call__(self, x, t=None):
        self.clear()

        h = F.leaky_relu(self.conv1(x), slope=0.1)
        h = F.leaky_relu(self.conv2(h), slope=0.1)
        h = F.leaky_relu(self.conv3(h), slope=0.1)
        h = F.leaky_relu(self.conv4(h), slope=0.1)
        h = F.clipped_relu(self.conv5(h), z=1.0)
        if self.train:
            self.loss = F.mean_squared_error(h, t)
            return self.loss
        else:
            return h

    def preprocess_x(self, x_data):
        """
        model specific preprocessing
        :param x_data:
        :return:
        """
        scaled_x = image_processing.nearest_neighbor_2x(x_data)
        return image_processing.image_padding(scaled_x, total_padding // 2)

    def clear(self):
        self.loss = None
        # self.accuracy = None

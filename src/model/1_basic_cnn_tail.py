
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class basic_cnn_tail(Chain):
    """
    Basic CNN model.
    The network consists of Convolutional layer and leaky relu is used for activation
    It is same structure with waifu2x
    """
    def __init__(self, inout_ch):
        super(basic_cnn_tail, self).__init__(
            conv1=L.Convolution2D(in_channels=inout_ch, out_channels=32, ksize=3, stride=1),
            conv2=L.Convolution2D(32, 32, 3, stride=1),
            conv3=L.Convolution2D(32, 64, 3, stride=1),
            conv4=L.Convolution2D(64, 64, 3, stride=1),
            conv5=L.Convolution2D(64, 128, 3, stride=1),
            conv6=L.Convolution2D(128, 128, 3, stride=1),
            conv7=L.Convolution2D(128, inout_ch, 3, stride=1),
        )
        self.train = True

    def __call__(self, x_data):
        x = Variable(x_data)  # x_data.astype(np.float32)

        h = F.leaky_relu(self.conv1(x), slope=0.1)
        h = F.leaky_relu(self.conv2(h), slope=0.1)
        h = F.leaky_relu(self.conv3(h), slope=0.1)
        h = F.leaky_relu(self.conv4(h), slope=0.1)
        h = F.leaky_relu(self.conv5(h), slope=0.1)
        h = F.leaky_relu(self.conv6(h), slope=0.1)
        h = F.clipped_relu(self.conv7(h), z=1.0)
        return h

    def clear(self):
        self.loss = None
        self.accuracy = None

    def forward(self, x_data, t_data):
        self.clear()
        x = chainer.Variable(x_data)  # x_data.astype(np.float32)
        t = chainer.Variable(t_data)  # [Note]: x_data, t_data must be np.float32 type

        h = F.leaky_relu(self.conv1(x), slope=0.1)
        h = F.leaky_relu(self.conv2(h), slope=0.1)
        h = F.leaky_relu(self.conv3(h), slope=0.1)
        h = F.leaky_relu(self.conv4(h), slope=0.1)
        h = F.leaky_relu(self.conv5(h), slope=0.1)
        #h = F.leaky_relu(self.conv52(h), slope=0.1)
        #h = F.leaky_relu(self.conv53(h), slope=0.1)
        h = F.leaky_relu(self.conv6(h), slope=0.1)
        h = F.clipped_relu(self.conv7(h), z=1.0)

        #self.loss = F.huber_loss(h, t, delta= 1 / 256.)
        self.loss = F.mean_squared_error(h, t)
        # self.accuracy = F.accuracy(h, t)  # type inconpatible
        if self.train:
            return self.loss
        else:
            return h.data
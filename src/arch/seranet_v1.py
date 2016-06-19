import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import os

import src.functions as CF
import src.tools.image_processing as image_processing
from convolution_rbm import ConvolutionRBM

""" Configuration """
file_path = os.path.dirname(os.path.realpath(__file__))
arch_folder = os.path.join(file_path, '../../data/arch/seranet_v1')

total_padding = 18

class seranet_v1(Chain):
    """
    SeRanet
    """
    def __init__(self, inout_ch):
        super(seranet_v1, self).__init__(
            conv1=L.Convolution2D(in_channels=inout_ch, out_channels=64, ksize=5, stride=1),
            conv2=L.Convolution2D(64, 64, 5),
            conv3=L.Convolution2D(64, 128, 5),
            #conv4=L.Convolution2D(64, 64, 3),
            #conv5=L.Convolution2D(64, 128, 3),
            seranet_v1_crbm=seranet_v1_crbm(inout_ch),
            #crbm1=ConvolutionRBM(inout_ch, 64, 5),
            #crbm2=ConvolutionRBM(64, 64, 5),
            #crbm3=ConvolutionRBM(64, 128, 5),
            #crbm4=ConvolutionRBM(64, 64, 3),
            #crbm5=ConvolutionRBM(64, 128, 3),
            # fusion
            convlu6=L.Convolution2D(256, 512, 1),
            convru6=L.Convolution2D(256, 512, 1),
            convld6=L.Convolution2D(256, 512, 1),
            convrd6=L.Convolution2D(256, 512, 1),
            convlu7=L.Convolution2D(512, 256, 1),
            convru7=L.Convolution2D(512, 256, 1),
            convld7=L.Convolution2D(512, 256, 1),
            convrd7=L.Convolution2D(512, 256, 1),
            convlu8=L.Convolution2D(256, 128, 3),
            convru8=L.Convolution2D(256, 128, 3),
            convld8=L.Convolution2D(256, 128, 3),
            convrd8=L.Convolution2D(256, 128, 3),
            # splice
            conv9=L.Convolution2D(128, 128, 3),
            conv10=L.Convolution2D(128, 128, 3),
            conv11=L.Convolution2D(128, 128, 3),
            conv12=L.Convolution2D(128, inout_ch, 3),
        )
        # Initialization: lu, ld, lu, rd layers start from same initialW value
        self.convru6.W.data[...] = self.convlu6.W.data[...]
        self.convld6.W.data[...] = self.convlu6.W.data[...]
        self.convrd6.W.data[...] = self.convlu6.W.data[...]
        self.convru7.W.data[...] = self.convlu7.W.data[...]
        self.convld7.W.data[...] = self.convlu7.W.data[...]
        self.convrd7.W.data[...] = self.convlu7.W.data[...]
        self.convru8.W.data[...] = self.convlu8.W.data[...]
        self.convld8.W.data[...] = self.convlu8.W.data[...]
        self.convrd8.W.data[...] = self.convlu8.W.data[...]
        self.train = True

    def __call__(self, x, t=None):
        self.clear()
        h1 = F.leaky_relu(self.conv1(x), slope=0.1)
        h1 = F.leaky_relu(self.conv2(h1), slope=0.1)
        h1 = F.leaky_relu(self.conv3(h1), slope=0.1)
        #h1 = F.leaky_relu(self.conv4(h1), slope=0.1)
        #h1 = F.leaky_relu(self.conv5(h1), slope=0.1)

        h2 = self.seranet_v1_crbm(x)
        #h2 = self.crbm1(x)
        #h2 = self.crbm2(h2)
        #h2 = self.crbm3(h2)
        #h2 = self.crbm4(h2)
        #h2 = self.crbm5(h2)
        # Fusion
        h12 = F.concat((h1, h2), axis=1)

        lu = F.leaky_relu(self.convlu6(h12), slope=0.1)
        lu = F.leaky_relu(self.convlu7(lu), slope=0.1)
        lu = F.leaky_relu(self.convlu8(lu), slope=0.1)
        ru = F.leaky_relu(self.convru6(h12), slope=0.1)
        ru = F.leaky_relu(self.convru7(ru), slope=0.1)
        ru = F.leaky_relu(self.convru8(ru), slope=0.1)
        ld = F.leaky_relu(self.convld6(h12), slope=0.1)
        ld = F.leaky_relu(self.convld7(ld), slope=0.1)
        ld = F.leaky_relu(self.convld8(ld), slope=0.1)
        rd = F.leaky_relu(self.convrd6(h12), slope=0.1)
        rd = F.leaky_relu(self.convrd7(rd), slope=0.1)
        rd = F.leaky_relu(self.convrd8(rd), slope=0.1)

        # Splice
        h = CF.splice(lu, ru, ld, rd)

        h = F.leaky_relu(self.conv9(h), slope=0.1)
        h = F.leaky_relu(self.conv10(h), slope=0.1)
        h = F.leaky_relu(self.conv11(h), slope=0.1)
        h = F.clipped_relu(self.conv12(h), z=1.0)
        if self.train:
            self.loss = F.mean_squared_error(h, t)
            return self.loss
        else:
            return h

    @staticmethod
    def preprocess_x(x_data):
        """
        model specific preprocessing
        :param x_data:
        :return:
        """
        return image_processing.image_padding(x_data, total_padding // 2)

    def clear(self):
        """
        Release memory before calculation
        (self.loss contains graph path info for back prop, which consumes memory.)
        :return:
        """
        self.loss = None
        # self.accuracy = None


class seranet_v1_crbm(Chain):
    """
    Sub-network of seranet, used for pre-training
    """
    def __init__(self, inout_ch, pretrain_level=0, k=1, pcd_flag=0, lambda_w=0., p=0.1, lambda_s=0.):
        """

        :param inout_ch:
        :param pretrain_level: 0 indicates not pretraining stage
        :param k:
        :param pcd_flag:
        :param lambda_w:
        :param p:
        :param lambda_s:
        """
        super(seranet_v1_crbm, self).__init__(
            crbm1=ConvolutionRBM(inout_ch, 64, 5, wscale=0.01),
            crbm2=ConvolutionRBM(64, 64, 5, wscale=0.01),
            crbm3=ConvolutionRBM(64, 128, 5, wscale=0.01),
        )
        self.pretrain_level = pretrain_level
        if self.pretrain_level == 0:
            'do nothing'
        elif self.pretrain_level == 1:
            self.crbm1.set_rbm_training_parameter(k=k, pcd_flag=pcd_flag, lambda_w=lambda_w, p=p, lambda_s=lambda_s,
                                                  rbm_train_debug=True)
        elif self.pretrain_level == 2:
            self.crbm2.set_rbm_training_parameter(k=k, pcd_flag=pcd_flag, lambda_w=lambda_w, p=p, lambda_s=lambda_s,
                                                  rbm_train_debug=True)
        elif self.pretrain_level == 3:
            self.crbm3.set_rbm_training_parameter(k=k, pcd_flag=pcd_flag, lambda_w=lambda_w, p=p, lambda_s=lambda_s,
                                                  rbm_train_debug=True)
        else:
            raise ValueError('pretrain_level ', pretrain_level, ' is out of range')

    def clear(self):
        self.loss = None
        # self.accuracy = None

    def __call__(self, x):
        self.clear()
        if self.pretrain_level == 0:
            h = self.crbm1(x)
            h = self.crbm2(h)
            h = self.crbm3(h)
            return h
        elif self.pretrain_level == 1:
            h = self.crbm1(x)
            self.loss = h
            return h
        elif self.pretrain_level == 2:
            h = self.crbm1(x)
            h.unchain_backward()
            h = self.crbm2(h)
            self.loss = h
            return h
        elif self.pretrain_level == 3:
            h = self.crbm1(x)
            h = self.crbm2(h)
            h.unchain_backward()
            h = self.crbm3(h)
            self.loss = h
            return h
        else:
            raise ValueError('pretrain level is out of range')

    def init_persistent_params(self, x):
        if self.pretrain_level == 0:
            print '[Error] It is not pretraining phase'
        elif self.pretrain_level == 1:
            print 'debug'
            self.crbm1.init_persistent_params(x)
        elif self.pretrain_level == 2:
            h = self.crbm1(x)
            h.unchain_backward()
            self.crbm2.init_persistent_params(h)
        elif self.pretrain_level == 3:
            h = self.crbm1(x)
            h = self.crbm2(h)
            h.unchain_backward()
            self.crbm3.init_persistent_params(h)
        else:
            raise ValueError('pretrain level is out of range')


    @staticmethod
    def preprocess_x(x_data):
        """
        model specific preprocessing
        :param x_data:
        :return:
        """
        return image_processing.image_padding(x_data, total_padding // 2)


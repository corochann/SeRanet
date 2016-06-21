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
arch_folder = os.path.join(file_path, '../../data/arch/seranet_split')

total_padding = 18


class seranet_split(Chain):
    """
    SeRanet split
    From the beggining until the end, 4 pixel position, lu, ru, ld and rd network are independent
    """
    def __init__(self, inout_ch):
        super(seranet_split, self).__init__(
            convlu1=L.Convolution2D(in_channels=inout_ch, out_channels=32, ksize=3, stride=1),
            convru1=L.Convolution2D(in_channels=inout_ch, out_channels=32, ksize=3, stride=1),
            convld1=L.Convolution2D(in_channels=inout_ch, out_channels=32, ksize=3, stride=1),
            convrd1=L.Convolution2D(in_channels=inout_ch, out_channels=32, ksize=3, stride=1),
            convlu2=L.Convolution2D(32, 64, 3),
            convru2=L.Convolution2D(32, 64, 3),
            convld2=L.Convolution2D(32, 64, 3),
            convrd2=L.Convolution2D(32, 64, 3),
            convlu3=L.Convolution2D(64, 64, 3),
            convru3=L.Convolution2D(64, 64, 3),
            convld3=L.Convolution2D(64, 64, 3),
            convrd3=L.Convolution2D(64, 64, 3),
            convlu4=L.Convolution2D(64, 96, 3),
            convru4=L.Convolution2D(64, 96, 3),
            convld4=L.Convolution2D(64, 96, 3),
            convrd4=L.Convolution2D(64, 96, 3),
            convlu5=L.Convolution2D(96, 96, 3),
            convru5=L.Convolution2D(96, 96, 3),
            convld5=L.Convolution2D(96, 96, 3),
            convrd5=L.Convolution2D(96, 96, 3),
            crbm1=ConvolutionRBM(inout_ch, 32, 3),
            crbm2=ConvolutionRBM(32, 64, 3),
            crbm3=ConvolutionRBM(64, 64, 3),
            crbm4=ConvolutionRBM(64, 96, 3),
            crbm5=ConvolutionRBM(96, 96, 3),
            convlu6=L.Convolution2D(192, 512, 1),
            convru6=L.Convolution2D(192, 512, 1),
            convld6=L.Convolution2D(192, 512, 1),
            convrd6=L.Convolution2D(192, 512, 1),
            convlu7=L.Convolution2D(512, 192, 1),
            convru7=L.Convolution2D(512, 192, 1),
            convld7=L.Convolution2D(512, 192, 1),
            convrd7=L.Convolution2D(512, 192, 1),
            convlu8=L.Convolution2D(192, 160, 3),
            convru8=L.Convolution2D(192, 160, 3),
            convld8=L.Convolution2D(192, 160, 3),
            convrd8=L.Convolution2D(192, 160, 3),
            convlu9=L.Convolution2D(160, 128, 3),
            convru9=L.Convolution2D(160, 128, 3),
            convld9=L.Convolution2D(160, 128, 3),
            convrd9=L.Convolution2D(160, 128, 3),
            convlu10=L.Convolution2D(128, 64, 3),
            convru10=L.Convolution2D(128, 64, 3),
            convld10=L.Convolution2D(128, 64, 3),
            convrd10=L.Convolution2D(128, 64, 3),
            convlu11=L.Convolution2D(64, inout_ch, 3),
            convru11=L.Convolution2D(64, inout_ch, 3),
            convld11=L.Convolution2D(64, inout_ch, 3),
            convrd11=L.Convolution2D(64, inout_ch, 3),
        )
        # Initialization: lu, ld, lu, rd layers start from same initialW value
        self.convru1.W.data[...] = self.convlu1.W.data[...]
        self.convld1.W.data[...] = self.convlu1.W.data[...]
        self.convrd1.W.data[...] = self.convlu1.W.data[...]
        self.convru2.W.data[...] = self.convlu2.W.data[...]
        self.convld2.W.data[...] = self.convlu2.W.data[...]
        self.convrd2.W.data[...] = self.convlu2.W.data[...]
        self.convru3.W.data[...] = self.convlu3.W.data[...]
        self.convld3.W.data[...] = self.convlu3.W.data[...]
        self.convrd3.W.data[...] = self.convlu3.W.data[...]
        self.convru4.W.data[...] = self.convlu4.W.data[...]
        self.convld4.W.data[...] = self.convlu4.W.data[...]
        self.convrd4.W.data[...] = self.convlu4.W.data[...]
        self.convru5.W.data[...] = self.convlu5.W.data[...]
        self.convld5.W.data[...] = self.convlu5.W.data[...]
        self.convrd5.W.data[...] = self.convlu5.W.data[...]
        self.convru6.W.data[...] = self.convlu6.W.data[...]
        self.convld6.W.data[...] = self.convlu6.W.data[...]
        self.convrd6.W.data[...] = self.convlu6.W.data[...]
        self.convru7.W.data[...] = self.convlu7.W.data[...]
        self.convld7.W.data[...] = self.convlu7.W.data[...]
        self.convrd7.W.data[...] = self.convlu7.W.data[...]
        self.convru8.W.data[...] = self.convlu8.W.data[...]
        self.convld8.W.data[...] = self.convlu8.W.data[...]
        self.convrd8.W.data[...] = self.convlu8.W.data[...]
        self.convru9.W.data[...] = self.convlu9.W.data[...]
        self.convld9.W.data[...] = self.convlu9.W.data[...]
        self.convrd9.W.data[...] = self.convlu9.W.data[...]
        self.convru10.W.data[...] = self.convlu10.W.data[...]
        self.convld10.W.data[...] = self.convlu10.W.data[...]
        self.convrd10.W.data[...] = self.convlu10.W.data[...]
        self.convru11.W.data[...] = self.convlu11.W.data[...]
        self.convld11.W.data[...] = self.convlu11.W.data[...]
        self.convrd11.W.data[...] = self.convlu11.W.data[...]
        self.train = True

    def __call__(self, x, t=None):
        self.clear()

        lu = F.leaky_relu(self.convlu1(x), slope=0.1)
        lu = F.leaky_relu(self.convlu2(lu), slope=0.1)
        lu = F.leaky_relu(self.convlu3(lu), slope=0.1)
        lu = F.leaky_relu(self.convlu4(lu), slope=0.1)
        lu = F.leaky_relu(self.convlu5(lu), slope=0.1)

        ru = F.leaky_relu(self.convru1(x), slope=0.1)
        ru = F.leaky_relu(self.convru2(ru), slope=0.1)
        ru = F.leaky_relu(self.convru3(ru), slope=0.1)
        ru = F.leaky_relu(self.convru4(ru), slope=0.1)
        ru = F.leaky_relu(self.convru5(ru), slope=0.1)

        ld = F.leaky_relu(self.convld1(x), slope=0.1)
        ld = F.leaky_relu(self.convld2(ld), slope=0.1)
        ld = F.leaky_relu(self.convld3(ld), slope=0.1)
        ld = F.leaky_relu(self.convld4(ld), slope=0.1)
        ld = F.leaky_relu(self.convld5(ld), slope=0.1)

        rd = F.leaky_relu(self.convrd1(x), slope=0.1)
        rd = F.leaky_relu(self.convrd2(rd), slope=0.1)
        rd = F.leaky_relu(self.convrd3(rd), slope=0.1)
        rd = F.leaky_relu(self.convrd4(rd), slope=0.1)
        rd = F.leaky_relu(self.convrd5(rd), slope=0.1)

        cr = self.crbm1(x)
        cr = self.crbm2(cr)
        cr = self.crbm3(cr)
        cr = self.crbm4(cr)
        cr = self.crbm5(cr)

        # JOIN CR

        lucr = F.concat((lu, cr), axis=1)
        rucr = F.concat((ru, cr), axis=1)
        ldcr = F.concat((ld, cr), axis=1)
        rdcr = F.concat((rd, cr), axis=1)

        lucr = F.leaky_relu(self.convlu6(lucr), slope=0.1)
        lucr = F.leaky_relu(self.convlu7(lucr), slope=0.1)
        lucr = F.leaky_relu(self.convlu8(lucr), slope=0.1)
        lucr = F.leaky_relu(self.convlu9(lucr), slope=0.1)
        lucr = F.leaky_relu(self.convlu10(lucr), slope=0.1)
        lucr = F.clipped_relu(self.convlu11(lucr), z=1.0)

        rucr = F.leaky_relu(self.convru6(rucr), slope=0.1)
        rucr = F.leaky_relu(self.convru7(rucr), slope=0.1)
        rucr = F.leaky_relu(self.convru8(rucr), slope=0.1)
        rucr = F.leaky_relu(self.convru9(rucr), slope=0.1)
        rucr = F.leaky_relu(self.convru10(rucr), slope=0.1)
        rucr = F.clipped_relu(self.convru11(rucr), z=1.0)

        ldcr = F.leaky_relu(self.convld6(ldcr), slope=0.1)
        ldcr = F.leaky_relu(self.convld7(ldcr), slope=0.1)
        ldcr = F.leaky_relu(self.convld8(ldcr), slope=0.1)
        ldcr = F.leaky_relu(self.convld9(ldcr), slope=0.1)
        ldcr = F.leaky_relu(self.convld10(ldcr), slope=0.1)
        ldcr = F.clipped_relu(self.convld11(ldcr), z=1.0)

        rdcr = F.leaky_relu(self.convrd6(rdcr), slope=0.1)
        rdcr = F.leaky_relu(self.convrd7(rdcr), slope=0.1)
        rdcr = F.leaky_relu(self.convrd8(rdcr), slope=0.1)
        rdcr = F.leaky_relu(self.convrd9(rdcr), slope=0.1)
        rdcr = F.leaky_relu(self.convrd10(rdcr), slope=0.1)
        rdcr = F.clipped_relu(self.convrd11(rdcr), z=1.0)

        h = CF.splice(lucr, rucr, ldcr, rdcr)

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
        self.loss = None
        # self.accuracy = None


class seranet_crbm(Chain):
    """
    Sub-network of seranet, used for pre-training
    """
    def __init__(self, inout_ch, pretrain_level=1):
        super(seranet_crbm, self).__init__(
            crbm1=ConvolutionRBM(inout_ch, 32, 3),
            crbm2=ConvolutionRBM(32, 64, 3),
            crbm3=ConvolutionRBM(64, 64, 3),
            crbm4=ConvolutionRBM(64, 128, 3),
            crbm5=ConvolutionRBM(128, 128, 3),
        )
        self.pretrain_level = pretrain_level
        if self.pretrain_level == 1:
            self.crbm1.rbm_train = True
        elif self.pretrain_level == 2:
            self.crbm2.rbm_train = True
        elif self.pretrain_level == 3:
            self.crbm3.rbm_train = True
        elif self.pretrain_level == 4:
            self.crbm4.rbm_train = True
        elif self.pretrain_level == 5:
            self.crbm5.rbm_train = True

    def clear(self):
        self.loss = None
        # self.accuracy = None

    def __call__(self, x):
        h = self.crbm1(x)
        if self.pretrain_level == 1:
            return h
        h = self.crbm2(h)
        if self.pretrain_level == 2:
            return h
        h = self.crbm3(h)
        if self.pretrain_level == 3:
            return h
        h = self.crbm4(h)
        if self.pretrain_level == 4:
            return h
        h = self.crbm5(h)
        return h


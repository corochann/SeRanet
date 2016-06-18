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
training_process_folder_yonly = os.path.join(file_path, '../../data/training_process_seranet_yonly')
training_process_folder_rgb = os.path.join(file_path, '../../data/training_process_seranet_rgb')

total_padding = 18


class seranet(Chain):
    """
    SeRanet
    """
    def __init__(self, inout_ch):
        super(seranet, self).__init__(
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
            convlu4=L.Convolution2D(64, 128, 3),
            convru4=L.Convolution2D(64, 128, 3),
            convld4=L.Convolution2D(64, 128, 3),
            convrd4=L.Convolution2D(64, 128, 3),
            convlu5=L.Convolution2D(128, 128, 3),
            convru5=L.Convolution2D(128, 128, 3),
            convld5=L.Convolution2D(128, 128, 3),
            convrd5=L.Convolution2D(128, 128, 3),
            crbm1=ConvolutionRBM(inout_ch, 32, 3),
            crbm2=ConvolutionRBM(32, 64, 3),
            crbm3=ConvolutionRBM(64, 64, 3),
            crbm4=ConvolutionRBM(64, 128, 3),
            crbm5=ConvolutionRBM(128, 128, 3),
            convlu6=L.Convolution2D(256, 256, 3),
            convru6=L.Convolution2D(256, 256, 3),
            convld6=L.Convolution2D(256, 256, 3),
            convrd6=L.Convolution2D(256, 256, 3),
            convlu7=L.Convolution2D(256, 256, 3),
            convru7=L.Convolution2D(256, 256, 3),
            convld7=L.Convolution2D(256, 256, 3),
            convrd7=L.Convolution2D(256, 256, 3),
            convlu8=L.Convolution2D(256, 128, 3),
            convru8=L.Convolution2D(256, 128, 3),
            convld8=L.Convolution2D(256, 128, 3),
            convrd8=L.Convolution2D(256, 128, 3),
            convlu9=L.Convolution2D(128, inout_ch, 3),
            convru9=L.Convolution2D(128, inout_ch, 3),
            convld9=L.Convolution2D(128, inout_ch, 3),
            convrd9=L.Convolution2D(128, inout_ch, 3),
        )
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

        lucr = CF.fusion(lu, cr)
        rucr = CF.fusion(ru, cr)
        ldcr = CF.fusion(ld, cr)
        rdcr = CF.fusion(rd, cr)

        lucr = F.leaky_relu(self.convlu6(lucr), slope=0.1)
        lucr = F.leaky_relu(self.convlu7(lucr), slope=0.1)
        lucr = F.leaky_relu(self.convlu8(lucr), slope=0.1)
        lucr = F.clipped_relu(self.convlu9(lucr), z=1.0)

        rucr = F.leaky_relu(self.convru6(rucr), slope=0.1)
        rucr = F.leaky_relu(self.convru7(rucr), slope=0.1)
        rucr = F.leaky_relu(self.convru8(rucr), slope=0.1)
        rucr = F.clipped_relu(self.convru9(rucr), z=1.0)

        ldcr = F.leaky_relu(self.convld6(ldcr), slope=0.1)
        ldcr = F.leaky_relu(self.convld7(ldcr), slope=0.1)
        ldcr = F.leaky_relu(self.convld8(ldcr), slope=0.1)
        ldcr = F.clipped_relu(self.convld9(ldcr), z=1.0)

        rdcr = F.leaky_relu(self.convrd6(rdcr), slope=0.1)
        rdcr = F.leaky_relu(self.convrd7(rdcr), slope=0.1)
        rdcr = F.leaky_relu(self.convrd8(rdcr), slope=0.1)
        rdcr = F.clipped_relu(self.convrd9(rdcr), z=1.0)

        h = CF.splice(lucr, rucr, ldcr, rdcr)

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
        return image_processing.image_padding(x_data, total_padding // 2)

    def clear(self):
        self.loss = None
        # self.accuracy = None

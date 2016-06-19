"""
Convolutional Restricted Boltzmann Machine using chainer
"""

from __future__ import print_function

import os, sys
import argparse
import timeit

import numpy as np
try:
    import PIL.Image as Image
except ImportError:
    import Image

import chainer
from chainer import computational_graph
from chainer import cuda, Variable, optimizers, serializers, Chain

import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O

import pickle
import gzip

import src.functions as CF


class ConvolutionRBM(chainer.Chain):
    """
    Convolutional Restricted Boltzmann Machine
    it supports sparse regularization and weight-decay during training
    in_channels: # of input channel (feature map)
    out_channels: # of output channel (feature map)
    ksize: filter size of convolution
    stride: stride of convolution
    conv.W: Matrix(out_channels, in_channels, filter height=ksize, filter width=ksize)
            - parameter               <- pre registered by L.Convolution2D
    conv.b: (out_channels, ) - hbias  <- pre registered by L.Convolution2D
    conv.a: (in_channels, ) - vbias   <- added by add_param
    real: 1 = visible unit is real value, 0 = visible unit is binary 0 or 1.
    lambda_w: scalar - weight decay coefficient (L2 regularization)
    p:        scalar - sparsity activation rate
    lambda_s: scalar - sparsity coefficient()
    gpu:      scalar - use gpu or cpu. -1: cpu, natural number: gpu
    """
    def __init__(self, in_channels, out_channels, ksize, stride=1, real=0, wscale=1.0):
        super(ConvolutionRBM, self).__init__(
            conv=L.Convolution2D(in_channels, out_channels, ksize, stride=stride, wscale=wscale),
        )

#        if gpu >= 0:
#            cuda.check_cuda_available()
#            xp = cuda.cupy # if gpu >= 0 else np
        self.conv.add_param("a", in_channels)  # dtype=xp.float32
        self.conv.a.data.fill(0.)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.real = real

        self.rbm_train = False  # default value is false

    def __call__(self, x):
        if self.rbm_train == True:
            """
            :param x:      Variable (batch_size, in_channels, image_height, image_width) - input data (training data)
            :param x_prev: Variable (batch_size, in_channels, image_height, image_width)
                                - persistent chain, hold persistent state for visible units of model.
                                  (used only for persistent CD.)
            :param q_prev: xp (batch_size, image_height_out, image_width_out)
            :return:
            """
            batch_size = x.data.shape[0]
            if self.pcd_flag == 0:
                "do nothing, use x"
                # Usual CD, only x is used for constructing vh.
                # h_prev_data = x
                v_input = x
            else:
                # Persistent CD, only x_prev_data is used for constructing vh, except for the 1st trial
                # x_prev_data.unchain_backward()
                v_input = self.x_prev
            # t1 = timeit.default_timer()
            ph_mean, ph_sample, vh_mean, vh_sample = self.contrastive_divergence(v_input, self.k)
            # t2 = timeit.default_timer()
            if self.pcd_flag == 1:
                self.x_prev.data[:] = vh_sample.data[:]  # update x_prev_data for next usage
            # v = Variable(x)
            # vh = Variable(vh_sample.astype(xp.float32))  # vh_sample is not variable
            # vh_sample.unchain_backward()
            vh_mean.unchain_backward()
            # ph_mean.unchain_backward()

            '''
            http://deeplearning.net/tutorial/rbm.html eq.(5)
            '''
            positive_phase = self.free_energy(x)
            negative_phase = self.free_energy(vh_mean)
            if self.rbm_train_debug:
                if self.count % 1000 < 5:
                    print('[CRBM debug] free energy: data (positive phase) = ', positive_phase.data,
                          ', model (negative_phase) = ', negative_phase.data)
                self.count += 1
            self.loss = (positive_phase - negative_phase) / batch_size
            # t3 = timeit.default_timer()
            # print('vh_sample = ', vh_sample.data, ', shape ', vh_sample.data.shape)
            # print('sum vh_sample = ', F.sum(vh_sample).data, ', shape ', vh_sample.data.shape)
            """ Sparsity """
            # (out_channel, image_height_out, image_width_out) - average activation rate
            lambda_q = 0.9

            q = lambda_q * self.q_prev + (1 - lambda_q) * F.sum(ph_mean, axis=0) / batch_size
            self.q_prev[:] = q.data[:]
            # self.loss += self.lambda_s * F.sum((q - self.p) * (q - self.p))  # Sparsity squared penalty
            self.loss += -self.lambda_s * F.sum(self.p * F.log(q) + (1 - self.p) * F.log(1 - q))  # Sparsity log penalty

            self.loss += self.lambda_w * 0.5 * F.sum(self.conv.W * self.conv.W)  # Weight decay   L2 regularization
            # self.loss += self.lambda_w * F.sum(F.leaky_relu(self.conv.W, slope=-1.))  # Weight decay  L1 regularization
            # t4 = timeit.default_timer()
            # print('CRBM call: ', t2-t1, ' ', t3-t2, ' ', t4-t3)  # CRBM call:  0.00819110870361   0.00195288658142   0.000527143478394

            # print('self.free_energy x = ', self.free_energy(x).data,
            #      'self.free_energy vh_sample = ', self.free_energy(vh_sample).data,
            # 'self.loss before = ', self.loss.data,
            #      'self.loss after sparse = ', self.loss.data)
            # print('self.conv.a', self.conv.a.data)
            # print('self.conv.b', self.conv.b.data)
            return self.loss
        else:
            return F.sigmoid(self.conv(x))

    def set_rbm_training_parameter(self, k=1, pcd_flag=0, lambda_w=0., p=0.1, lambda_s=0., std=np.asarray([1.0]),
                                   rbm_train_debug=False):
        """
        This function need to be called for rbm_training
        After that, it is also necessary to call init_persistent_params
        :param k:
        :param pcd_flag:
        :param lambda_w:
        :param p:
        :param lambda_s:
        :param std:
        :param rbm_train_debug:
        :return:
        """
        self.rbm_train = True
        self.rbm_train_debug = rbm_train_debug
        self.count = 0  # count for debug print
        self.k = k
        self.pcd_flag = pcd_flag
        self.lambda_w = lambda_w
        self.p = p
        self.lambda_s = lambda_s
        self.std = std  # only used for real (Gaussian-Bernoulli RBM)
        #self.std_ch = xp.reshape(self.std, (1, in_channels, 1, 1))
        #self.gpu = gpu
        self.x_prev = None
        self.q_prev = None

    def init_persistent_params(self, x):
        """
        x_prev: Variable (batch_size, in_channels, image_height, image_width)
                                - persistent chain, hold persistent state for visible units of model.
                                  (used only for persistent CD.)
        q_prev: xp (batch_size, image_height_out, image_width_out)
        :param x:
        :return:
        """
        if self.pcd_flag == 1:
            self.x_prev = x
        h_mean = self.propup(x)

        # print('h_mean shape', h_mean.data.shape)
        xp = cuda.get_array_module(h_mean.data)
        prev_q = xp.asarray(xp.zeros((h_mean.data.shape[1:])), dtype=xp.float32)
        self.q_prev = prev_q

    def forward(self, v_data, v_prev_data=None, q_prev=None):
        """

        :param v_data:      Variable (batch_size, in_channels, image_height, image_width) - input data (training data)
        :param v_prev_data: Variable (batch_size, in_channels, image_height, image_width)
                            - persistent chain, hold persistent state for visible units of model.
                              (used only for persistent CD.)
        :param q_prev: xp (batch_size, image_height_out, image_width_out)
        :return:
        """
        batch_size = v_data.data.shape[0]
        if self.pcd_flag == 0:
            "do nothing, use v_data"
            # Usual CD, only v_data is used for constructing vh.
            # h_prev_data = v_data
            v_input = v_data
        else:
            # Persistent CD, only v_prev_data is used for constructing vh, except for the 1st trial
            #v_prev_data.unchain_backward()
            v_input = v_prev_data
        # t1 = timeit.default_timer()
        ph_mean, ph_sample, vh_mean, vh_sample = self.contrastive_divergence(v_input, self.k)
        # t2 = timeit.default_timer()
        v_prev_data.data[:] = vh_sample.data[:]  # update v_prev_data for next usage
        # v = Variable(v_data)
        # vh = Variable(vh_sample.astype(xp.float32))  # vh_sample is not variable
        #vh_sample.unchain_backward()
        vh_mean.unchain_backward()
        #ph_mean.unchain_backward()

        '''
        http://deeplearning.net/tutorial/rbm.html eq.(5)
        '''
        self.loss = (self.free_energy(v_data) - self.free_energy(vh_mean)) / batch_size
        # t3 = timeit.default_timer()
        #print('vh_sample = ', vh_sample.data, ', shape ', vh_sample.data.shape)
        #print('sum vh_sample = ', F.sum(vh_sample).data, ', shape ', vh_sample.data.shape)
        """ Sparsity """
        # (out_channel, image_height_out, image_width_out) - average activation rate
        lambda_q = 0.9
        q = lambda_q * q_prev + (1 - lambda_q) * F.sum(ph_mean, axis=0) / batch_size
        q_prev[:] = q.data[:]
        #self.loss += self.lambda_s * F.sum((q - self.p) * (q - self.p))  # Sparsity squared penalty
        self.loss += -self.lambda_s * F.sum(self.p * F.log(q) + (1 - self.p) * F.log(1 - q))  # Sparsity log penalty

        self.loss += self.lambda_w * 0.5 * F.sum(self.conv.W * self.conv.W)  # Weight decay   L2 regularization
        #self.loss += self.lambda_w * F.sum(F.leaky_relu(self.conv.W, slope=-1.))  # Weight decay  L1 regularization
        # t4 = timeit.default_timer()
        # print('CRBM call: ', t2-t1, ' ', t3-t2, ' ', t4-t3)  # CRBM call:  0.00819110870361   0.00195288658142   0.000527143478394

        #print('self.free_energy v_data = ', self.free_energy(v_data).data,
        #      'self.free_energy vh_sample = ', self.free_energy(vh_sample).data,
              #'self.loss before = ', self.loss.data,
        #      'self.loss after sparse = ', self.loss.data)
        #print('self.conv.a', self.conv.a.data)
        #print('self.conv.b', self.conv.b.data)
        return self.loss

    def free_energy(self, v):
        """
        :param Variable (batch_size, in_channels, image_height, image_width) - input data (training data)
        :return: scalar
        """
        batch_size = v.data.shape[0]
        in_channels = self.in_channels
        real = self.real
        if real == 0:
            '''
            visible layer is 0, 1 (bit)
            vbias_term = 1 * SUM(a(i) * v(i))
            '''
            v_sum = F.sum(v, axis=(2, 3))  # sum over image_height & image_width
            # Originally, it should return sum for each batch.
            # but it returns scalar, which is sum over batches, since sum is used at the end anyway.
            vbias_term = F.sum(F.matmul(v_sum, self.conv.a))
            wx_b = self.conv(v)

        else:
            '''
            visible layer takes real value
            vbias_term = 0.5 * SUM((v(i)-a(i)) * (v(i) - a(i)))
            '''
            #TODO: check
            #m = Variable(xp.ones((batch_size, 1), dtype=xp.float32))
            n = F.reshape(self.conv.a, (1, in_channels, 1, 1))
            xp = cuda.get_array_module(n.data)
            std_ch = xp.reshape(self.std, (1, in_channels, 1, 1))

            #v_ = v - F.matmul(m, n)
            v_ = (v - F.broadcast_to(n, v.data.shape)) / std_ch
            vbias_term = F.sum(0.5 * v_ * v_)
            wx_b = self.conv(v / std_ch)


        hidden_term = F.sum(F.log(1 + F.exp(wx_b)))
        # print('vbias = ', vbias_term.data, ', hidden = ', hidden_term.data, 'F.exp(wx_b) = ', F.exp(wx_b).data)
        return - vbias_term - hidden_term

    def propup(self, vis):
        """
        This function propagates the visible units activation upwards to the hidden units
        Eq.(7)
        :param vis: Variable Matrix(batch_size, in_channels, image_height, image_width)
                    - given v_sample
        :return: Variable Matrix(batch_size, out_channels, image_height_out, image_width_out)
                 - probability for each hidden units to be h_i=1
        """
        # conv.W: Matrix(out_channels, in_channels, filter height=ksize, filter width=ksize)
        # conv.b: Vec   (out_channels, )
        if self.real == 0:
            pre_sigmoid_activation = self.conv(vis)
        else:
            pre_sigmoid_activation = self.conv(vis / self.std_ch)
        # F.matmul(vis, self.conv.W, transb=True) + F.broadcast_to(self.conv.b, (vis.data.shape[0], self.n_hidden))
        return F.sigmoid(pre_sigmoid_activation)

    def propdown(self, hid):
        """ This function propagates the hidden units activation downwords to the visible units
        :param hid: Variable Matrix(batch_size, out_channels, image_height_out, image_width_out)  - given h_sample
        :return: Variable Matrix(batch_size, in_channels, image_height, image_width) - probability for each visible units to be v_j = 1
        """
        batch_size = hid.data.shape[0]
        if self.real == 0:
            W_flipped = F.swapaxes(CF.flip(self.conv.W, axes=(2, 3)), axis1=0, axis2=1)
            pre_sigmoid_activation = F.convolution_2d(hid, W_flipped, self.conv.a, pad=self.ksize-1)
                # F.matmul(hid, self.l.W) + F.broadcast_to(self.l.a, (batch_size, self.n_visible))
            v_mean = F.sigmoid(pre_sigmoid_activation)
            #print('W info ', self.conv.W.data.shape, 'W_flipped info ', W_flipped.data.shape)
            #print('W info ', self.conv.W.data[3, 0, 2, 3], 'W_flipped info ', W_flipped.data[0, 3, 8, 7])
            #print('W info ', self.conv.W.data[3, 0, 8, 7], 'W_flipped info ', W_flipped.data[0, 3, 2, 3])
            #print('W info ', self.conv.W.data[19, 0, 4, 0], 'W_flipped info ', W_flipped.data[0, 19, 6, 10])
            #print('pre_sigmoidactivation', F.sum(pre_sigmoid_activation).data)
            #print('v_mean', v_mean.data.shape)
            #print('v_mean sum', F.sum(v_mean).data)
            #print('hid', hid.data.shape)

        else:
            # TODO: check
            W_flipped = F.swapaxes(CF.flip(self.conv.W, axes=(2, 3)), axis1=0, axis2=1)
            v_mean = F.convolution_2d(hid, W_flipped, self.conv.a, pad=self.ksize-1)
        return v_mean

    def sample_h_given_v(self, v0_sample):
        """ get a sample of the hiddens by gibbs sampling
        :param v0_sample: Variable, see vis above
        :return:
        h1_mean:   Variable Matrix(batch_size, out_channels, image_height_out, image_width_out)
        h1_sample: Variable Matrix(batch_size, out_channels, image_height_out, image_width_out)
                   - actual sample for hidden units, populated by 0 or 1.
        """
        h1_mean = self.propup(v0_sample)
        xp = cuda.get_array_module(h1_mean.data)
        if xp == cuda.cupy:
            h1_sample = cuda.cupy.random.random_sample(size=h1_mean.data.shape)
            h1_sample[:] = h1_sample[:] < h1_mean.data[:]
        else:  # xp == np
            h1_sample = np.random.binomial(size=h1_mean.data.shape, n=1, p=h1_mean.data)
        return h1_mean, Variable(h1_sample.astype(xp.float32))

    def sample_v_given_h(self, h0_sample):
        """ get a sample of the visible units by gibbs sampling
        :param h0_sample: see hid above
        :return:
        v1_mean: Variable Matrix(batch_size, in_channels, image_height, image_width)
        v1_sample: Variable Matrix(batch_size, in_channels, image_height, image_width)
                   - actual sample for visible units, populated by 0 or 1.
                     if real value is used, v1_sample[i] is gaussian distribution over mean=v1_mean[i] & variance 1
        """
        v1_mean = self.propdown(h0_sample)
        xp = cuda.get_array_module(v1_mean.data)
        if self.real == 0:
            if xp == cuda.cupy:
                v1_sample = cuda.cupy.random.random_sample(size=v1_mean.data.shape)
                # print('before sample ', v1_sample, 'v1_mean.data', v1_mean.data)
                v1_sample[:] = v1_sample[:] < v1_mean.data[:]
                # print('after sample ', v1_sample, 'v1_mean.data', v1_mean.data)
            else:  # xp == np
                v1_sample = np.random.binomial(size=v1_mean.data.shape, n=1, p=v1_mean.data)
        else:
            # TODO: impl real
            batch_size = h0_sample.data.shape[0]
            # TODO: check proper variance implementation
            #v1_sample = v1_mean.data + xp.random.randn(batch_size, self.n_visible) / 255.  # reduced variance by 255
            #v1_sample = v1_mean.data + xp.random.randn(batch_size, self.in_channels, v1_mean.data.shape[2], v1_mean.data.shape[3]) * self.std  # reduced variance by 255
            v1_sample = v1_mean.data
        return v1_mean, Variable(v1_sample.astype(xp.float32))

    def gibbs_hvh(self, h0_sample):
        """ 1 step of Gibbs sampling, starting from the hidden state
        :param h0_sample: Variable
        :return: Variable
        """
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return v1_mean, v1_sample, h1_mean, h1_sample

    def gibbs_vhv(self, v0_sample):
        """ 1 step of Gibbs sampling, starting from the visible state
        :param v0_sample: Variable Matrix(batch_size, in_channels, image_height, image_width)
                          - given v_sample
        :return: Variable
        """
        # t1 = timeit.default_timer()
        h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        # t2 = timeit.default_timer()
        v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        # t3 = timeit.default_timer()
        # print('gibbsvhv time: sample_h_given_v ', t2 - t1, ' sec')  # 0.000492095947266  sec
        # print('gibbsvhv time: sample_v_given_h ', t3 - t2, ' sec')  # 0.000334024429321  sec
        return h1_mean, h1_sample, v1_mean, v1_sample

    def contrastive_divergence(self, v0_sample, k=1):
        """
        CD-K, PCD-k
        :param v0_sample: Variable Matrix(batch_size, in_channels, image_height, image_width)
        :param k: #of iteration for CD
        :return:
        """
        vh_sample = v0_sample
        for step in xrange(k):
            ph_mean, ph_sample, vh_mean, vh_sample = self.gibbs_vhv(vh_sample)
        return ph_mean, ph_sample, vh_mean, vh_sample

    def reconstruct(self, v):
        """

        :param v: Variable Matrix(batch_size, in_channels, image_height, image_width)
        :return: reconstructed_v, Variable Matrix(batch_size, in_channels, image_height, image_width)
        """
        batch_size = v.data.shape[0]
        xp = cuda.get_array_module(v.data)
        if self.real == 0:
            h = F.sigmoid(self.conv(v))
        else:
            std_ch = xp.reshape(self.std, (1, self.in_channels, 1, 1))
            h = F.sigmoid(self.conv(v / std_ch))
        # F.sigmoid(F.matmul(v, self.l.W, transb=True) + F.broadcast_to(self.l.b, (batch_size, self.n_hidden)))
        W_flipped = F.swapaxes(CF.flip(self.conv.W, axes=(2, 3)), axis1=0, axis2=1)
        reconstructed_v = F.sigmoid(F.convolution_2d(h, W_flipped, self.conv.a, pad=self.ksize-1))
            # = F.sigmoid(F.matmul(h, self.l.W) + F.broadcast_to(self.l.a, (batch_size, self.n_visible)))
        return reconstructed_v

    @staticmethod
    def sigmoid(x):
        xp = cuda.get_array_module(x.data)
        return 1. / (1 + xp.exp(-x))


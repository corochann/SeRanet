from __future__ import print_function

import sys
import cv2

import os
import numpy as np
import cPickle as pickle
import timeit
import time
from argparse import ArgumentParser

try:
    import PIL.Image as Image
except ImportError:
    import Image

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from tools.prepare_data import load_data
from tools.utils import tile_raster_images

if __name__ == '__main__':

    """ Pre setup """

    # Get params (Arguments)
    parser = ArgumentParser(description='SeRanet ConvolutionalRBM pre-training')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--arch', '-a', default='seranet_v1',
                        help='model selection (seranet_v1)')
    parser.add_argument('--level', '-l', type=int, default=1, help='Pretraining level')
    parser.add_argument('--batchsize', '-B', type=int, default=20, help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', default=1000, type=int, help='Number of max epochs to learn')
    parser.add_argument('--color', '-c', default='rgb', help='training scheme for input/output color: (yonly, rgb)')
    parser.add_argument('--pcd', '-p', default=1, type=int, help='pcd_flag')
    parser.add_argument('--kcd', '-k', default=1, type=int, help='cd-k')
    #parser.add_argument('--real', '-r', default=0, type=int,
    #                    help='0: use binary unit (Bernoulli), 1: use real unit (Gaussian-Bernoulli)')

    lambda_w = 1.0   # weight decay
    p = 0.05         # sparsity rate
    lambda_s = 10.0   # sparsity

    args = parser.parse_args()

    n_epoch = args.epoch           # #of training epoch
    batch_size = args.batchsize    # size of minibatch
    visualize_test_img_number = 5  # #of images to visualize for checking training performance
    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >= 0 else np

    if args.color == 'yonly':
        inout_ch = 1
    elif args.color == 'rgb':
        inout_ch = 3
    else:
        raise ValueError('Invalid color training scheme')

    # Prepare model
    print('prepare model')
    if args.arch == 'seranet':
        import arch.seranet as model_arch
        model = model_arch.seranet(inout_ch=inout_ch)
    elif args.arch == 'seranet_v1':
        import arch.seranet_v1 as model_arch
        model = model_arch.seranet_v1_crbm(inout_ch=inout_ch,
                                           pretrain_level=args.level,
                                           k=args.kcd,
                                           pcd_flag=args.pcd,
                                           lambda_w=lambda_w,
                                           p=p,
                                           lambda_s=lambda_s)
    else:
        raise ValueError('Invalid architecture name')

    # Directory/File setting for training log
    arch_folder = model_arch.arch_folder
    # Directory/File setting for training log
    training_process_folder = os.path.join(arch_folder, args.color, 'pretraining_crbm_level' + str(args.level))
    if args.level > 1:
        pretrained_model_path = os.path.join(arch_folder, args.color, 'pretraining_crbm_level' + str(args.level - 1),
                                         'my.model')
        serializers.load_npz(pretrained_model_path, model)

    if not os.path.exists(training_process_folder):
        os.makedirs(training_process_folder)
    os.chdir(training_process_folder)
    train_log_file_name = 'train.log'
    train_log_file = open(os.path.join(training_process_folder, train_log_file_name), 'w')
    #total_image_padding = 14 #24 #18 #14

    """ Load data """
    print('loading data')

    datasets = load_data(mode=args.color)

    np_train_dataset, np_valid_dataset, np_test_dataset = datasets
    np_train_set_x, np_train_set_y = np_train_dataset
    np_valid_set_x, np_valid_set_y = np_valid_dataset
    np_test_set_x, np_test_set_y = np_test_dataset

    n_train = np_train_set_x.shape[0]
    n_valid = np_valid_set_x.shape[0]
    n_test = np_test_set_x.shape[0]

    """ Preprocess """
    #print('preprocess')
    start_time = timeit.default_timer()

    def normalize_image(np_array):
        np_array /= 255.
        np_array.astype(np.float32)

    normalize_image(np_train_set_x)
    normalize_image(np_valid_set_x)
    normalize_image(np_test_set_x)
    normalize_image(np_train_set_y)
    normalize_image(np_valid_set_y)
    normalize_image(np_test_set_y)

    end_time = timeit.default_timer()
    print('preprocess time %i sec' % (end_time - start_time))
    print('preprocess time %i sec' % (end_time - start_time), file=train_log_file)

    """ Setup GPU """
    """ Model, optimizer setup """
    print('setup model')
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # optimizer = optimizers.Adam(alpha=0.001)
    optimizer = optimizers.AdaDelta()
    # optimizer = optimizers.MomentumSGD(lr=0.0001, momentum=0.5)  # 0.0001 -> value easily explodes
    optimizer.setup(model)

    """
    TRAINING
    Early stop method is used for training to avoid overfitting,
    Reference: https://github.com/lisa-lab/DeepLearningTutorials
    """
    print('training')

    patience = 30000
    patience_increase = 2
    improvement_threshold = 0.995  # 0.997

    validation_frequency = min(n_train, patience // 2) * 2

    best_validation_loss = np.inf
    iteration = 0
    best_iter = 0
    test_score = 0.
    done_looping = False

    plotting_time = 0.

    x_batch = model.preprocess_x(np_train_set_x[0: 0 + batch_size])
    x = Variable(xp.asarray(x_batch, dtype=xp.float32))
    model.init_persistent_params(x)

    for epoch in xrange(1, n_epoch + 1):
        print('epoch: %d' % epoch)
        start_time = timeit.default_timer()
        perm = np.random.permutation(n_train)
        sum_loss = 0

        for i in xrange(0, n_train, batch_size):
            # start_iter_time = timeit.default_timer()
            iteration += 1
            if iteration % 1000 == 0:
                print('training @ iter ', iteration)
            x_batch = np_train_set_x[perm[i: i + batch_size]].copy()
            #x_batch = xp.asarray(train_scaled_x[perm[i: i + batch_size]])
            #y_batch = xp.asarray(np_train_set_y[perm[i: i + batch_size]])
            x_batch = model.preprocess_x(x_batch)
            # print('x_batch', x_batch.shape, x_batch.dtype)

            x = Variable(xp.asarray(x_batch, dtype=xp.float32))
            optimizer.update(model, x)
            sum_loss += float(model.loss.data) * batch_size
            # end_iter_time = timeit.default_timer()
            # print("iter took: %f sec" % (end_iter_time - start_iter_time))  # GPU -> iter took: 0.138625 sec

        print("train mean loss: %f" % (sum_loss / n_train))
        print("train mean loss: %f" % (sum_loss / n_train), file=train_log_file)

        # Validation
        sum_loss = 0
        for i in xrange(0, n_valid, batch_size):
            x_batch = np_valid_set_x[i: i + batch_size]
            x_batch = model.preprocess_x(x_batch)

            x = Variable(xp.asarray(x_batch, dtype=xp.float32))
            sum_loss += float(model(x).data) * batch_size

        this_validation_loss = (sum_loss / n_valid)
        print("valid mean loss: %f" % this_validation_loss)
        print("valid mean loss: %f" % this_validation_loss, file=train_log_file)
        if this_validation_loss < best_validation_loss:
            if this_validation_loss < best_validation_loss * improvement_threshold:
                patience = max(patience, iteration * patience_increase)
                print('update patience -> ', patience, ' iteration')

            best_validation_loss = this_validation_loss
            best_iter = iteration

            sum_loss = 0
            for i in xrange(0, n_test, batch_size):
                x_batch = np_test_set_x[i: i + batch_size]

                x_batch = model.preprocess_x(x_batch)

                x = Variable(xp.asarray(x_batch, dtype=xp.float32))

                sum_loss += float(model(x).data) * batch_size
            test_score = (sum_loss / n_test)
            print('  epoch %i, test cost of best model %f' %
                  (epoch, test_score))
            print('  epoch %i, test cost of best model %f' %
                  (epoch, test_score), file=train_log_file)

            # Save best model
            print('saving model')
            serializers.save_npz('my.model', model)
            serializers.save_npz('my.state', optimizer)

        if patience <= iteration:
            done_looping = True
            print('done_looping')
            break

        end_time = timeit.default_timer()
        print('epoch %i took %i sec' % (epoch, end_time - start_time))
        print('epoch %i took %i sec' % (epoch, end_time - start_time), file=train_log_file)

        # Construct image from the weight matrix
        n_chains = 20
        n_samples = 10

        weight = model.get_target_crbm().conv.W.data[:, 0:1, ...]
        ksize = model.get_target_crbm().ksize

        if args.gpu >= 0:
            weight = cuda.to_cpu(weight)
        if epoch < 10 or epoch % 10 == 0:
            print(' ... plotting RBM weight')
            image = Image.fromarray(
                tile_raster_images(
                    # X=rbm.W.get_value(borrow=True).T,
                    X=weight,
                    img_shape=(ksize, ksize),
                    tile_shape=(10, 20),
                    tile_spacing=(1, 1)
                )
            )
            image.save('filters_at_epoch%i.png' % epoch)

        if args.level == 1 and (epoch < 10 or epoch % 10 == 0):
            """ SAMPLING FROM the RBM """
            print(' ... plotting RBM reconstruct data')
            image_size = 116
            image_data = np.zeros(
                # ((image_size + 1) * n_samples + 1, (image_size + 1) * n_chains - 1),
                ((image_size + 1) * n_chains + 1, (image_size + 1) * n_samples - 1),
                dtype='uint8'
            )
            reconstruct_x = Variable(xp.asarray(np_train_set_x[0: 0 + n_samples], dtype=xp.float32))
            for idx in xrange(n_chains):

                image_piece = reconstruct_x.data[:, 0:1, ...]
                if args.gpu >= 0:
                    image_piece = cuda.to_cpu(image_piece)
                image_data[(image_size + 1) * idx:(image_size + 1) * idx + image_size, :] = tile_raster_images(
                    X=image_piece,
                    img_shape=(image_size, image_size),
                    tile_shape=(1, n_samples),
                    tile_spacing=(1, 1)
                )

                # h1_mean, h1_sample, v1_mean, reconstruct_x = rbm.gibbs_vhv(reconstruct_x)
                reconstruct_x = model.get_target_crbm().reconstruct(reconstruct_x)
            image = Image.fromarray(image_data)
            image.save('samples_reconstruct_epoch%i.png' % epoch)

            image_data = np.zeros(
                # ((image_size + 1) * n_samples + 1, (image_size + 1) * n_chains - 1),
                ((image_size + 1) * n_chains + 1, (image_size + 1) * n_samples - 1),
                dtype='uint8'
            )

            reconstruct_x = Variable(xp.asarray(np_train_set_x[0: 0 + n_samples], dtype=xp.float32))
            for idx in xrange(n_chains):
                image_piece = reconstruct_x.data[:, 0:1, ...]
                if args.gpu >= 0:
                    image_piece = cuda.to_cpu(image_piece)
                image_data[(image_size + 1) * idx:(image_size + 1) * idx + image_size, :] = tile_raster_images(
                    X=image_piece,
                    img_shape=(image_size, image_size),
                    tile_shape=(1, n_samples),
                    tile_spacing=(1, 1)
                )
                h1_mean, h1_sample, v1_mean, reconstruct_x = model.get_target_crbm().gibbs_vhv(reconstruct_x)

            image = Image.fromarray(image_data)
            image.save('samples_gibbs_vhv_epoch%i.png' % epoch)

    end_time = timeit.default_timer()
    pretraining_time = (end_time - start_time) - plotting_time

    print('Training took %i min %i sec' %
          (pretraining_time / 60., pretraining_time % 60))

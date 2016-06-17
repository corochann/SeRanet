from __future__ import print_function

import sys
import cv2

import os
import numpy as np
import cPickle as pickle
import timeit
import time
from argparse import ArgumentParser

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from tools.prepare_data import load_data
from tools.image_processing import preprocess

if __name__ == '__main__':

    """ Pre setup """

    # Get params (Arguments)
    parser = ArgumentParser(description='SRCNN chainer')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--arch', '-a', default='basic_cnn_tail', help='model selection (basic_cnn_tail, basic_cnn_middle, ...)')
    parser.add_argument('--batchsize', '-B', type=int, default=32, help='Learning minibatch size')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250, help='Validation minibatch size')
    parser.add_argument('--epoch', '-E', default=1000, type=int, help='Number of epochs to learn')
    parser.add_argument('--color', '-c', default='rgb', help='training scheme for input/output color: \'yonly\' or \'rgb\' ')

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
    if args.arch == 'basic_cnn_tail':
        import arch.basic_cnn_tail as model_arch
        model = model_arch.basic_cnn_tail(inout_ch=inout_ch)
    else:
        raise ValueError('Invalid architecture name')

    # Directory/File setting for training log
    if args.color == 'yonly':
        training_process_folder = model_arch.training_process_folder_yonly
    elif args.color == 'rgb':
        training_process_folder = model_arch.training_process_folder_rgb

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
    #train_scaled_x = preprocess(np_train_set_x, total_image_padding // 2)
    #valid_scaled_x = preprocess(np_valid_set_x, total_image_padding // 2)
    #test_scaled_x = preprocess(np_test_set_x, total_image_padding // 2)

    def normalize_image(np_array):
        np_array /= 255.
        np_array.astype(np.float32)

    normalize_image(np_train_set_x)
    normalize_image(np_valid_set_x)
    normalize_image(np_test_set_x)
    normalize_image(np_train_set_y)
    normalize_image(np_valid_set_y)
    normalize_image(np_test_set_y)
    #np_train_set_y /= 255.  # normalize
    #np_valid_set_y /= 255.
    #np_test_set_y /= 255.
    #np_train_set_y = np_train_set_y.astype(np.float32)
    #np_valid_set_y = np_valid_set_y.astype(np.float32)
    #np_test_set_y = np_test_set_y.astype(np.float32)

    end_time = timeit.default_timer()
    print('preprocess time %i sec' % (end_time - start_time))
    print('preprocess time %i sec' % (end_time - start_time), file=train_log_file)

    """ SHOW Test images (0~visualize_test_img_number) """
    for i in xrange(visualize_test_img_number):
        cv2.imwrite(os.path.join(training_process_folder, 'photo' + str(i) + '_xinput.jpg'),
                    np_test_set_x[i].transpose(1, 2, 0) * 255.)
        cv2.imwrite(os.path.join(training_process_folder, 'photo' + str(i) + '_original.jpg'),
                    np_test_set_y[i].transpose(1, 2, 0) * 255.)

    """ Setup GPU """
    """ Model, optimizer setup """
    print('setup model')
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # optimizer = optimizers.Adam(alpha=0.0001)
    # optimizer = optimizers.AdaDelta()
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)


    """ Training """
    print('training')

    patience = 30000
    patience_increase = 2
    improvement_threshold = 0.997  # 0.995

    validation_frequency = min(n_train, patience // 2) * 2

    best_validation_loss = np.inf
    iteration = 0
    best_iter = 0
    test_score = 0.
    done_looping = False

    for epoch in xrange(1, n_epoch + 1):
        print('epoch: %d' % epoch)
        start_time = timeit.default_timer()
        perm = np.random.permutation(n_train)
        sum_loss = 0

        for i in xrange(0, n_train, batch_size):
            # start_iter_time = timeit.default_timer()
            iteration += 1
            if iteration % 100 == 0:
                print('training @ iter ', iteration)
            x_batch = np_train_set_x[perm[i: i + batch_size]].copy()
            y_batch = np_train_set_y[perm[i: i + batch_size]].copy()
            #x_batch = xp.asarray(train_scaled_x[perm[i: i + batch_size]])
            #y_batch = xp.asarray(np_train_set_y[perm[i: i + batch_size]])
            x_batch = model.preprocess_x(x_batch)
            print('x_batch', x_batch.shape, x_batch.dtype)

            x = Variable(xp.asarray(x_batch))
            t = Variable(xp.asarray(y_batch))

            optimizer.update(model, x, t)
            sum_loss += float(model.loss.data) * len(y_batch)

            #optimizer.zero_grads()
            #loss = model.forward(x, t)
            #loss.backward()
            #optimizer.update()
            #sum_loss += float(loss.data) * len(y_batch)
            # end_iter_time = timeit.default_timer()
            # print("iter took: %f sec" % (end_iter_time - start_iter_time))  # GPU -> iter took: 0.138625 sec
            time.sleep(10.)

        print("train mean loss: %f" % (sum_loss / n_train))
        print("train mean loss: %f" % (sum_loss / n_train), file=train_log_file)

        # Validation
        sum_loss = 0
        for i in xrange(0, n_valid, batch_size):
            #x_batch = xp.asarray(valid_scaled_x[i:i + batch_size])
            #y_batch = xp.asarray(np_valid_set_y[i:i + batch_size])
            x_batch = np_valid_set_x[i: i + batch_size]
            y_batch = np_valid_set_y[i: i + batch_size]

            x_batch = model.preprocess_x(x_batch)

            x = Variable(xp.asarray(x_batch, dtype=xp.float32))
            t = Variable(xp.asarray(y_batch, dtype=xp.float32))

            loss = model.forward(x, t)
            sum_loss += float(loss.data) * len(y_batch)

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
                #x_batch = xp.asarray(test_scaled_x[i:i + batch_size])
                #y_batch = xp.asarray(np_test_set_y[i:i + batch_size])
                x_batch = np_test_set_x[i: i + batch_size]
                y_batch = np_test_set_y[i: i + batch_size]

                x_batch = model.preprocess_x(x_batch)

                x = Variable(xp.asarray(x_batch, dtype=xp.float32))
                t = Variable(xp.asarray(y_batch, dtype=xp.float32))

                loss = model.forward(x, t)
                sum_loss += float(loss.data) * len(y_batch)
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

        # Check test imagles
        if epoch // 10 == 0 or epoch % 10 == 0:
            #model.train = False
            #x_batch = xp.asarray(test_scaled_x[0:5])
            #y_batch = xp.asarray(np_test_set_y[0:5])
            x_batch = np_test_set_x[0:5]
            #output = model.forward(x_batch, y_batch)
            output = model(x_batch)

            if (args.gpu >= 0):
                output = cuda.cupy.asnumpy(output)

            #print('output_img0: ', output[0].transpose(1, 2, 0) * 255.)
            for photo_id in xrange(visualize_test_img_number):
                cv2.imwrite(os.path.join(training_process_folder,
                                         'photo' + str(photo_id) + '_epoch' + str(epoch) + '.jpg'),
                            output[photo_id].transpose(1, 2, 0) * 255.)
            #model.train = True

        end_time = timeit.default_timer()
        print('epoch %i took %i sec' % (epoch, end_time - start_time))
        print('epoch %i took %i sec' % (epoch, end_time - start_time), file=train_log_file)

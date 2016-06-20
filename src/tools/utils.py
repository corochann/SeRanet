""" utility functions
"""

import numpy
import os
import six.moves.cPickle as pickle
import gzip

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """ Transform an array with image
    This function is for visualizing datasets whose rows are images,
    also columns of matrices for transforming those rows.
    :param X: 2D ndarray or tuple of 4 channels - 2D array in which every row is a flattened image
    :param img_shape: tuple (height, width)     - the original shape of each image
    :param tile_shape: tuple (rows, cols)       - #of images to tile
    :param tile_spacing:
    :param scale_rows_to_unit_interval: if the values ned to be scaled before being plotted to [0, 1 or not]
    :param output_pixel_vals: if output should be pixel values (i.e. int8) or floats
    :return: a 2D array with same dtype as X - array suitable for viewing as an image
    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # out_shape - output image shape
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        #
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)

        return out_array
    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array



def load_data(dataset):
    ''' loads the MNIST dataset

    :param dataset:
    :return:
    '''

    # 1.Load data

    data_dir, data_file = os.path.split(dataset)
    # 1-1. Check dataset is in the data directory
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    # 1-2. Download if not in data directory
    if(not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # 2. load the dataset
    with gzip.open(dataset, 'rb') as f:
        # train_set, valid_set, test_set: tuple(input, target)
        # input:  Matrix(sample_number, n_in) - each sample data in each row
        # target: vector(sample_number, )     - correct label
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        """ load dataset into shared variables
        :param data_xy: tuple(input, target) - train_set, valid_set, test_set
        :param borrow:
        :return:
        """
        data_x, data_y = data_xy
        data_x = numpy.asarray(data_x, dtype=numpy.float32)
        data_y = numpy.asarray(data_x, dtype=numpy.float32)
        #shared_x = theano.shared(numpy.asarray(data_x, dtype=numpy.float32), borrow=borrow)
        #shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        # y is int, but when storing data on the GPU it must be 'floats' format
        # So store shared_variable as 'float' but we return y by casting it into int

        #return shared_x, T.cast(shared_y, 'int32')
        return data_x, data_y

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

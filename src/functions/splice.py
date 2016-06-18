from chainer import function, cuda
from chainer.utils import type_check


class Splice(function.Function):
    """Fusion numpy/cupy element"""

    def __init__(self):
        'do nothing'

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 4,
        )
        lu_type, ru_type, ld_type, rd_type = in_types
        type_check.expect(
            lu_type.ndim == 4,
            lu_type.shape == ru_type.shape,
            lu_type.shape == ld_type.shape,
            lu_type.shape == rd_type.shape,
        )

    @property
    def label(self):
        return 'Splice'

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        lu, ru, ld, rd = inputs
        output = xp.empty((
            lu.shape[0],
            lu.shape[1],
            lu.shape[2] * 2,
            lu.shape[3] * 2,
        ), dtype=xp.float32)
        output[:, :, 0::2, 0::2] = lu[:]
        output[:, :, 0::2, 1::2] = ru[:]
        output[:, :, 1::2, 0::2] = ld[:]
        output[:, :, 1::2, 1::2] = rd[:]
        return output,

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        gy = grad_outputs[0]
        glu = gy[:, :, 0::2, 0::2]
        gru = gy[:, :, 0::2, 1::2]
        gld = gy[:, :, 1::2, 0::2]
        grd = gy[:, :, 1::2, 1::2]
        return glu, gru, gld, grd,


def splice(lu, ru, ld, rd):
    """
    splice each numpy/cupy element to form a one numpy/cupy array
    it is used to construct twice size of the input
    :param lu:
    :param ru:
    :param ld:
    :param rd:
    :return:
    """
    return Splice()(lu, ru, ld, rd)

from chainer import function
from chainer.utils import type_check


class Flip(function.Function):
    """Flip (order in reverse) element"""

    def __init__(self, axes=None):
        self.axes = axes

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1,)

    @property
    def label(self):
        return 'Flip'

    def forward(self, inputs):
        x = inputs[0]

        slice_obj = []
        for i in xrange(x.ndim):
            if (self.axes is None) or (i in self.axes):
                slice_obj.append(slice(None, None, -1))
            else:
                slice_obj.append(slice(None, None, None))
        y = x[tuple(slice_obj)]
        return y,

    def backward(self, inputs, grad_outputs):
        gy = grad_outputs[0]
        slice_obj = []
        for i in xrange(gy.ndim):
            if (self.axes is None) or (i in self.axes):
                slice_obj.append(slice(None, None, -1))
            else:
                slice_obj.append(slice(None, None, None))
        gx = gy[tuple(slice_obj)]
        return gx,


def flip(x, axes=None):
    """Flip (order in reverse) the element of an input variable without copy

    :param x: (~chainer.Variable) Input variable
    :param axes: (tuple of ints) By default, flip all axes,
    otherwise flip only specified axes
    :return: (~chainer.Variable) Variable whose element is flipped
    """
    return Flip(axes)(x)

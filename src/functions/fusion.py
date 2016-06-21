"""
This function is DEPRECATED.
use chainer.function.concat instead!
"""
from chainer import function, cuda
from chainer.utils import type_check


class Fusion(function.Function):
    """Fusion numpy/cupy element"""

    def __init__(self, in_channel_list=None):
        self.in_channel_list = in_channel_list

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 2,
        )
        h1_type, h2_type = in_types
        type_check.expect(

            h1_type.shape[2] == h2_type.shape[2],
            h1_type.shape[3] == h2_type.shape[3]
        )

    @property
    def label(self):
        return 'Fusion'

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        h1, h2 = inputs

        return xp.concatenate((h1, h2), axis=1),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        gy = grad_outputs[0]

        gh1, gh2 = xp.split(gy, [self.in_channel_list[0], ], axis=1)

        return gh1, gh2


def fusion(h1, h2):
    """

    :param h1:
    :param h2:
    :return:
    """
    """Flip (order in reverse) the element of an input variable without copy

    :param x: (~chainer.Variable) Input variable
    :param axes: (tuple of ints) By default, flip all axes,
    otherwise flip only specified axes
    :return: (~chainer.Variable) Variable whose element is flipped
    """
    in_channel_h1 = h1.data.shape[1]
    in_channel_h2 = h2.data.shape[1]
    return Fusion((in_channel_h1, in_channel_h2))(h1, h2)

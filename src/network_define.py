""" DPRNN training network wrapper. """

import mindspore.nn as nn

class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        net (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """
    def __init__(self, net, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._net = net
        self._loss = loss_fn

    def construct(self, padded_mixture, mixture_lengths, padded_source):
        estimate_source = self._net(padded_mixture)
        loss, _, estimate_source, _ = self._loss(padded_source, estimate_source, mixture_lengths)
        return loss

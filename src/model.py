import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer
from mindspore import context, Tensor, set_seed

class GlobalLayerNorm(nn.Cell):
    '''
        Calculate Global Layer Normalization
        dim: (int or list or torch.Size) –
            input shape from an expected input of size
        eps: a value added to the denominator for numerical stability.
    '''
    def __init__(self, dim, shape, eps=1e-8):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.mean = ops.ReduceMean(keep_dims=True)
        self.sqrt = ops.Sqrt()
        self.pow = ops.Pow()
        if shape == 3:
            self.weight = Tensor.from_numpy(np.ones((self.dim, 1), np.float32))
            self.bias = Tensor.from_numpy(np.zeros((self.dim, 1), np.float32))
        if shape == 4:
            self.weight = Tensor.from_numpy(np.ones((self.dim, 1, 1), np.float32))
            self.bias = Tensor.from_numpy(np.zeros((self.dim, 1, 1), np.float32))

    def construct(self, x):
        # x = N x C x K x S or N x C x L
        # gln: mean,var N x 1 x 1
        if x.ndim == 4:
            mean = self.mean(x, (1, 2, 3))
            var = self.pow(x - mean, 2)
            var = self.mean(var, (1, 2, 3))
            x = self.weight * (x - mean) / self.sqrt(var + self.eps) + self.bias
        if x.ndim == 3:
            mean = self.mean(x, (1, 2))
            var = self.pow(x - mean, 2)
            var = self.mean(var, (1, 2))
            x = self.weight * (x - mean) / self.sqrt(var + self.eps) + self.bias
        return x

def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape)
    return nn.GroupNorm(1, dim, eps=1e-8)

class Encoder(nn.Cell):
    '''
        DPRNN Encoder part
        kernel_size: the length of filters
        out_channels: the number of filters
    '''

    def __init__(self, kernel_size=2, out_channels=64):
        super(Encoder, self).__init__()
        self.expand_dims = ops.ExpandDims()
        self.conv1d = nn.Conv1d(1, out_channels, kernel_size=kernel_size,
                                stride=kernel_size // 2, group=1, weight_init="HeUniform", pad_mode="pad")
        self.relu = ops.ReLU()

    def construct(self, x):
        """
            Input:
                x: [B, T], B is batch size, T is times
            Returns:
                x: [B, C, T_out]
                T_out is the number of time steps
        """
        # B x T -> B x 1 x T
        expandDims = self.expand_dims(x, 1)
        # B x 1 x T -> B x C x T_out
        conv = self.conv1d(expandDims)
        relu = self.relu(conv)
        return relu

class Decoder(nn.Cell):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input.
        It is also known as a fractionally-strided convolution
        or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, in_channels=256, kernel_size=2):
        super(Decoder, self).__init__()
        self.conv1dTranspose = nn.Conv1dTranspose(in_channels, 1, kernel_size=kernel_size,
                                                  stride=kernel_size // 2, weight_init="HeUniform", pad_mode="pad")
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze()
        self.squeeze1 = ops.Squeeze(axis=1)

    def construct(self, x):
        """
        x: [B, N, L]
        """
        convTrans = self.conv1dTranspose(x if x.ndim == 3 else self.expand_dims(x, 1))

        if self.squeeze(convTrans).ndim == 1:
            out = self.squeeze1(convTrans)
        else:
            out = self.squeeze(convTrans)
        return out


class Dual_RNN_Block(nn.Cell):
    '''
        Implementation of the intra-RNN and the inter-RNN
        input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs
                        of each LSTM layer except the last layer,
                        with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, out_channels,
                 hidden_channels, rnn_type='LSTM', norm='ln',
                 dropout=0, bidirectional=False, num_spks=2):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.intra_rnn = nn.LSTM(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn = nn.LSTM(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        # Norm
        self.intra_norm = select_norm(norm, out_channels, 4)
        self.inter_norm = select_norm(norm, out_channels, 4)
        # Linear
        self.intra_linear = nn.Dense(
            hidden_channels * 2 if bidirectional else hidden_channels, out_channels, weight_init="HeUniform")
        self.inter_linear = nn.Dense(
            hidden_channels * 2 if bidirectional else hidden_channels, out_channels, weight_init="HeUniform")

    def construct(self, x):
        '''
            x: [B, N, K, S]
            out: [Spks, B, N, K, S]
        '''
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_tran = x.transpose((0, 3, 2, 1)).view(B*S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_tran)
        # [BS, K, N]
        intra_linear = self.intra_linear(intra_rnn.view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        intra_view = intra_linear.view(B, S, K, N)
        # [B, N, K, S]
        intra_trans = intra_view.transpose((0, 3, 2, 1))
        intra_final = self.intra_norm(intra_trans)

        # [B, N, K, S]
        intra_res = intra_final + x

        # inter RNN
        # [BK, S, N]
        inter_tran = intra_res.transpose((0, 2, 3, 1)).view(B*K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_tran)
        # [BK, S, N]
        inter_linear = self.inter_linear(inter_rnn.view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        inter_view = inter_linear.view(B, K, S, N)
        # [B, N, K, S]
        inter_trans = inter_view.transpose((0, 3, 1, 2))
        inter_final = self.inter_norm(inter_trans)
        # [B, N, K, S]
        out = inter_final + intra_res

        return out


class Dual_Path_RNN(nn.Cell):
    '''
        Implementation of the Dual-Path-RNN model
        input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            bn_channels (int): Number of channels after the bottleneck.
            Defaults to 128.
            hidden_channels (int): Number of neurons in the RNNs cell state.
            Defaults to 128.
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs
                        of each LSTM layer except the last layer,
                        with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    '''

    def __init__(self, in_channels, out_channels, hidden_channels, bn_channels,
                 rnn_type='LSTM', norm='ln', dropout=0,
                 bidirectional=False, num_layers=4, K=200, num_spks=2):
        super(Dual_Path_RNN, self).__init__()
        self.K = K
        self.selectNorm = norm
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.expand_dims = ops.ExpandDims()
        self.norm = select_norm(norm, in_channels, 3)
        self.norm1 = select_norm(norm, in_channels, 4)
        self.conv1d = nn.Conv1d(in_channels, bn_channels, 1, weight_init="HeUniform")

        self.dual_rnn = nn.CellList([])
        for _ in range(num_layers):
            self.dual_rnn.append(Dual_RNN_Block(bn_channels, hidden_channels,
                                                rnn_type=rnn_type, norm=norm, dropout=dropout,
                                                bidirectional=bidirectional))
        self.zeros = ops.Zeros()
        self.concat_op2 = ops.Concat(2)
        self.concat_op3 = ops.Concat(3)
        self.conv2d = nn.Conv2d(
            bn_channels, bn_channels*num_spks, kernel_size=1, has_bias=True, weight_init="HeUniform")
        self.end_conv1x1 = nn.Conv1d(bn_channels, out_channels, 1, weight_init="HeUniform")
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.SequentialCell(nn.Conv1d(bn_channels, bn_channels, 1,
                                                  has_bias=True, weight_init="XavierUniform"),
                                        nn.Tanh())
        self.output_gate = nn.SequentialCell(nn.Conv1d(bn_channels, bn_channels, 1,
                                                       has_bias=True, weight_init="XavierUniform"),
                                             nn.Sigmoid())

    def construct(self, x):
        '''
            x: [B, N, L]

        '''
        # [B, N, L]
        if self.selectNorm == 'ln':
            x = self.expand_dims(x, 0).transpose((0, 2, 1, 3))
            x = self.norm1(x)
            x = x.transpose((0, 2, 1, 3)).squeeze(axis=0)
        else:
            x = self.norm(x)
        # [B, N, L]
        conv1d = self.conv1d(x)
        # [B, N, K, S]
        segment, gap = self._Segmentation(conv1d, self.K)
        # [B, N*spks, K, S]
        for i in range(self.num_layers):
            segment = self.dual_rnn[i](segment)
        prelu = self.prelu(segment)
        conv2d = self.conv2d(prelu)
        # [B*spks, N, K, S]
        B, _, K, S = conv2d.shape
        view = conv2d.view(B*self.num_spks, -1, K, S)
        # [B*spks, N, L]
        overAdd = self._over_add(view, gap)
        gate = self.output(overAdd) * self.output_gate(overAdd)
        # [spks*B, N, L]
        endconv1d = self.end_conv1x1(gate)
        # [B*spks, N, L] -> [B, spks, N, L]
        _, N, L = endconv1d.shape
        viewFinal = endconv1d.view(B, self.num_spks, N, L)
        activate = self.activation(viewFinal)
        # [spks, B, N, L]
        out = activate.transpose(1, 0, 2, 3)

        return out

    def _padding(self, data, K):
        '''
            padding the audio times
            K: chunks of length
            P: hop size
            input: [B, N, L]
        '''
        B, N, L = data.shape
        P = K // 2
        gap = K - (P + L % K) % K

        if gap > 0:
            pad = self.zeros((B, N, gap), mindspore.float32)
            data = self.concat_op2([data, pad])

        _pad = self.zeros((B, N, P), mindspore.float32)
        data = self.concat_op2([_pad, data, _pad])

        return data, gap

    def _Segmentation(self, data, K):
        '''
            the segmentation stage splits
            K: chunks of length
            P: hop size
            input: [B, N, L]
            output: [B, N, K, S]
        '''
        B, N, _ = data.shape
        P = K // 2
        data, gap = self._padding(data, K)
        # [B, N, K, S]
        input1 = data[:, :, :-P].view(B, N, -1, K)
        input2 = data[:, :, P:].view(B, N, -1, K)

        output = self.concat_op3([input1, input2]).view(
            B, N, -1, K).transpose((0, 1, 3, 2))

        return output, gap

    def _over_add(self, data, gap):
        '''
            Merge sequence
            input: [B, N, K, S]
            gap: padding length
            output: [B, N, L]
        '''
        B, N, K, _ = data.shape
        P = K // 2
        # [B, N, S, K]
        inputTrans = data.transpose((0, 1, 3, 2)).view(B, N, -1, K * 2)

        input1 = inputTrans[:, :, :, :K].view(B, N, -1)[:, :, P:]
        input2 = inputTrans[:, :, :, K:].view(B, N, -1)[:, :, :-P]
        output = input1 + input2
        # [B, N, L]
        if gap > 0:
            output = output[:, :, :-gap]

        return output

class Dual_RNN_model(nn.Cell):
    '''
        model of Dual Path RNN
        input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            hidden_channels: The hidden size of RNN
            kernel_size: Encoder and Decoder Kernel size
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs
                        of each LSTM layer except the last layer,
                        with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: True
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    '''
    def __init__(self, in_channels, out_channels, hidden_channels, bn_channels,
                 kernel_size=2, rnn_type='LSTM', norm='ln', dropout=0,
                 bidirectional=True, num_layers=6, K=200, num_spks=2):
        super(Dual_RNN_model, self).__init__()
        self.encoder = Encoder(kernel_size=kernel_size, out_channels=in_channels)
        self.separation = Dual_Path_RNN(in_channels, out_channels, hidden_channels, bn_channels,
                                        rnn_type=rnn_type, norm=norm, dropout=dropout,
                                        bidirectional=bidirectional, num_layers=num_layers, K=K, num_spks=num_spks)
        self.decoder = Decoder(in_channels=in_channels, kernel_size=kernel_size)
        self.num_spks = num_spks
        self.print = ops.Print()
        self.stack = ops.Stack()

        for p in self.get_parameters():
            if p.ndim > 1:
                mindspore.common.initializer.HeNormal(p)

    def construct(self, x):
        """ forward x: [B, L] """
        # [B, N, L]
        e = self.encoder(x)
        # [spks, B, N, L]
        s = self.separation(e)
        # [B, N, L] -> [B, L]
        audio = []
        for i in range(self.num_spks):
            audio.append(self.decoder(s[i] * e))
        stack = self.stack(audio)
        result = stack.transpose((1, 0, 2))
        return result

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)
    set_seed(42)
    rnn = Dual_RNN_model(64, 64, 128, 128, bidirectional=True, norm='gln', num_layers=6, dropout=0.0)
    ones = ops.Ones()
    one = ones((1, 32000), mindspore.float32)
    rnn_out = rnn(one)
    print(rnn_out)

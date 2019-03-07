# model.py ---
#
# Filename: model.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Thu Jan 24 17:28:40 2019 (-0800)
# Version:
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), Visual Computing Group @ University of Victoria.

# Code:

import numpy as np
import torch
from torch import nn


class MyConv2d(nn.Module):
    """Custom Convolution class """

    def __init__(self, inchannel, outchannel, ksize=3, stride=1, padding=0):
        """Initialization of the custom conv2d module

        Note that his module behaves similarly to the original Conv2d
        module. We will implement the convolution layer ourselves for the 3x3
        case only just to make sure we understand convolutions.

        """

        # Run initialization for super class
        super(MyConv2d, self).__init__()

        # Assertions to simplify our case. We will only implement for the
        # default configuration defined above and nothing else.
        assert(ksize == 3)
        assert(stride == 1)
        assert(padding == 0)

        # Our custom convolution kernel. We'll initialize it using Kaiming He's
        # initialization with normal distribution
        self.weight = nn.Parameter(
            torch.randn((outchannel, inchannel, ksize, ksize)),
            requires_grad=True)

        self.bias = nn.Parameter(
            torch.randn((outchannel,)),
            requires_grad=True)

    def forward(self, x):
        """Forward pass for the 3x3 cross correlation.

        Note that we apply cross correlation, not the convolution, inorder to
        adhere to the pytorch implementations. If you really want to do
        convolutions, you'll need to flip the kernels.

        Implemented purely with PyTorch operations so that we don't have to
        implement the backward pass. Note that we implement the 3x3 case only,
        and do not need to consider cases other than the 3x3 conv with stride 1
        and no padding.

        You are **NOT ALLOWED** to use the convolution operations implemented
        in PyTorch. Instead, use `torch.permute`, `torch.reshape`, and
        `torch.matmul`.

        Also assume the shape of `x` is in the `NCHW` form.

        """

        assert(len(x.shape) == 4)

        # TODO (50 points): Implement according to the above comment.

        TODO


        return x_out


class MyNetwork(nn.Module):
    """Network class """

    def __init__(self, config, input_shp):
        """Initialization of the model.

        Parameters
        ----------

        config:
            Configuration object that holds the command line arguments.

        input_shp: tuple or list
            Shape of each input data sample.

        """

        # Run initialization for super class
        super(MyNetwork, self).__init__()

        # Store configuration
        self.config = config

        # Placeholder for layers
        self.layers = {}
        indim = input_shp[0]

        # Retrieve Conv, Act, Pool functions from configurations. We'll use
        # these for our code below.
        if config.conv2d == "torch":
            self.Conv2d = nn.Conv2d
        elif config.conv2d == "custom":
            self.Conv2d = MyConv2d
        self.Activation = getattr(nn, config.activation)
        self.Pool2d = getattr(nn, config.pool2d)
        self.Linear = nn.Linear

        # TODO (25 Points): Create Convolutional Layers. We will have
        # `config.num_conv_outer` number of `levels`, where each level is
        # composes of `config.num_conv_inner` number of blocks. Each block will
        # be made of a convolution layer with kernel size of `config.ksize` and
        # `config.nchannel_base*2^`level` number of channels, followed by relu
        # activation. Each outer loop will end with a 2x2 max pooling
        # operation. For example, a possible configuration could be, denoting
        # covolution layer as C, activation as A, max pooling as P, then,
        # CACAP-CACAP.
        #
        # To later flatten the outputs, we will keep track of the current
        # ``feature map'' shape as `cur_h` and `cur_w`
        cur_h, cur_w = input_shp[-2:]

        # Channels in the input layer
        chan_in = input_shp[0]

        # Levels
        for _i in range(config.num_conv_outer):
            # Blocks
            for _j in range(config.num_conv_inner):
                chan_out = config.nchannel_base * 2**_i
                setattr(self, "conv_conv2d_{}_{}".format(_i, _j),
                        nn.Conv2d(chan_in, chan_out, config.ksize))
                setattr(self, "conv_relu_{}_{}".format(_i, _j),
                        nn.ReLU())
                chan_in = chan_out

            setattr(self, "conv_pool_{}".format(_i),
                    nn.MaxPool2d(2))

        # Fully connected layers and output layers
        outdim = config.num_unit  # outdim was unset?
        indim = outdim
        # indim = outdim * cur_h * cur_w  # this seems wrong
        for _i in range(config.num_hidden):
            # outdim = config.num_unit  # outdim is invariant ?
            setattr(self, "fc_linear_{}".format(
                _i), self.Linear(indim, outdim))
            setattr(self, "fc_relu_{}".format(_i), self.Activation())
            # indim = outdim  # indim is invariant ?
        self.output = nn.Linear(indim, config.num_class)

        print(self)

    def forward(self, x):
        """Forward pass for the model

        Parameters
        ----------

        x: torch.Tensor
            Input data for the model to be applied. Note that this data is
            typically in the shape of BCHW or BC, where B is the number of
            elements in the batch, and C is the number of dimension of our
            feature. H, W is when we use raw images. In the current assignment,
            it wil l be of shape BC.

        Returns
        -------

        x: torch.Tensor

            We will reuse the variable name, because often the case it's more
            convenient to do so. We will first normalize the input, and then
            feed it to our linear layer by simply calling the layer as a
            function with normalized x as argument.

        """

        # Roughly make input to be within -1 to 1 range
        x = (x - 128.) / 128.

        # TODO (5 points): Apply the convolutional layers

        # Levels
        for _i in range(self.config.num_conv_outer):
            # Blocks
            for _j in range(self.config.num_conv_inner):
                m = getattr(self, "conv_conv2d_{}_{}".format(_i, _j))
                r = getattr(self, "conv_relu_{}_{}".format(_i, _j))
                x = r(m(x))  # convolution and activation
            p = getattr(self, "conv_pool_{}".format(_i))
            x = p(x)  # pool

        # Flatten
        x = x.view(x.size(0), -1)

        # Apply the fully connected layers
        for _i in range(self.config.num_hidden):
            x = getattr(self, "fc_linear_{}".format(_i))(x)
            x = getattr(self, "fc_relu_{}".format(_i))(x)
        x = self.output(x)

        return x


#
# model.py ends here

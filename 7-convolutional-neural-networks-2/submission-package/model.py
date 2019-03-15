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


class ConvBlock(nn.Module):
    """Convolution block """

    def __init__(self, indim, outdim, ksize=3, stride=1, activation=nn.ReLU):
        """Initialization of the custom conv2d module

        For simplicity, we will only implement the case where
        `indim`==`outdim`. We will also compute our padding ourselves and not
        change the input-output shapes. This will simplify many things. We also
        consider only the case when `stride` == 1.

        """

        # Run initialization for super class
        super(ConvBlock, self).__init__()

        # Check ksize, stride requirements
        assert (ksize % 2) == 1
        assert stride == 1
        assert indim == outdim

        # Store proper activation function depending on configuration
        self.activ = activation

        # TODO (10 points): Compute padding according to `ksize`. Make sure
        # that this will not cause image width and height to change.
        padding = ((outdim - 1) * stride - indim + ksize) - 1

        # tests
        print(indim, outdim, ksize, stride, padding)
        assert padding == padding//1, "non-integer padding"
        # assert outdim == (indim - ksize + 2 * padding) / stride + 1, "output size does not match"

        # TODO (20 points): Create ResNet Block.
        #
        # We will follow the architecture in slide 76 of lecture 21, but with
        # our `_conv` function as our conv ``block''. We'll also use
        # nn.Sequential() and its `add_module' function. Note that the 64 and
        # 256 in that slide are just examples, and you should instead use indim
        # and outdim.
        #
        # Also note that we are creating these layers with support for
        # different `ksize`, `stride`, `padding`, unlike previous assignment.
        self.layers = nn.Sequential()
        self.layers.add_module("conv_1", self._conv(outdim, outdim, 1, 1, 0))
        self.layers.add_module("conv_2", self._conv(outdim, outdim, ksize, stride, padding))
        self.layers.add_module("conv_3", self._conv(outdim, outdim, 1, 1, 0))

    def _conv(self, indim, outdim, ksize, stride, padding):
        """Function to make conv layers easier to create.

        Returns a nn.Sequential object which has bn-conv-activation.

        """

        # TODO (5 points): Implement this function
        layer = nn.Sequential(nn.BatchNorm2d(indim),
                              nn.Conv2d(indim, outdim, (ksize, ksize), (stride, stride), (padding, padding)),
                              self.activ())

        return layer

    def forward(self, x):
        """Forward pass our block.

        Note that we are implementing a resnet here. Thus, one path should go
        through our `layers`, whereas the other path should go through
        intact. They should then get added together (see again slide 76 of
        lecture 21). We will not use any activation after they are added.

        """

        assert(len(x.shape) == 4)

        # TODO (5 points): Implement according to the above comment.
        x1 = x.clone().detach()
        x2 = self.layers(x)
        x3 = x1 + x2

        return x3


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
            self.Conv2d = ConvBlock
        self.Activation = getattr(nn, config.activation)
        self.Pool2d = getattr(nn, config.pool2d)
        self.Linear = nn.Linear

        # Resnet Blocks, similar to slide 73 of lecture 21. However, for
        # simplicity, we'll make is slightly different. Note that we used
        # nn.Sequential this time.
        self.convs = nn.Sequential()
        cur_h, cur_w = input_shp[-2:]
        for _i in range(config.num_conv_outer):
            #
            # NOTE THE NEW LAYER ON THESE LINES!
            #
            # We have a dedicated 1x1 layer to get more channels. Note also
            # that this is a pure linear convolution layer.
            outdim = config.nchannel_base * 2 ** _i
            self.convs.add_module(
                "conv_{}_base".format(_i), nn.Conv2d(indim, outdim, 1, 1, 0))
            indim = outdim
            for _j in range(config.num_conv_inner):
                # We now use our selected convolution layer. Note that our
                # resnet implementation will have a different call style to
                # vanilla conv2d of torch, so we'll just do an ugly if-else
                # here.
                if config.conv2d == "torch":
                    self.convs.add_module(
                        "conv_{}_{}".format(_i, _j),
                        self.Conv2d(indim, outdim, config.ksize, 1, 1))
                    self.convs.add_module(
                        "act_{}_{}".format(_i, _j),
                        self.Activation())
                    cur_h = cur_h - (config.ksize - 1)
                    cur_w = cur_w - (config.ksize - 1)
                elif config.conv2d == "custom":
                    self.convs.add_module(
                        "conv_{}_{}".format(_i, _j),
                        self.Conv2d(indim, outdim, config.ksize, 1, self.Activation))
            self.convs.add_module(
                "conv_{}_pool".format(_i), self.Pool2d(2, 2))
            cur_h = cur_h // 2
            cur_w = cur_w // 2

        # Final output layer. We'll assume that conv layer outputs are global
        # average pooled
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

        # Apply conv layers
        x = self.convs(x)

        # Global average pooling
        x = x.mean(-1).mean(-1)

        # Output layer
        x = self.output(x)

        return x


#
# model.py ends here

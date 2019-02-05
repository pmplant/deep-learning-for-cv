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


class MyNetwork(nn.Module):
    """Network class """

    def __init__(self, config, input_shp, mean=None, std=None):
        """Initialization of the model.

        Parameters
        ----------

        config:
            Configuration object that holds the command line arguments.

        input_shp: tuple or list
            Shape of each input data sample.

        mean: np.array
            Mean value to be used for data normalization. We will store this in
            a torch.Parameter

        std: np.array
            Std value to be used for data normalization. We will store this in
            a torch.Parameter
        """

        # Run initialization for super class
        super(MyNetwork, self).__init__()

        # Store configuration
        self.config = config


        # TODO (5 points): Create torch.Tensor for holding mean, std. We will
        # apply these later, and if nothing is given, we would want to do
        # nothing to our data, i.e. mean should be set to zeros, std should be
        # set to ones. We also want all tensors to be `float32`
        self.mean = torch.zeros(len(input_shp), dtype=torch.float32)
        self.std = torch.ones(len(input_shp), dtype=torch.float32)

        print(len(input_shp))
        print(self.mean.size())

        # TODO (5 points): Wrap the created Tensors as parameters, so that we
        # can easily save and load (we can later get a list of everything
        # inside the morel by doing model.parameters(). Also make sure we mark
        # that these will not be updated by the optimizer by saying that
        # gradient computation should not be performed
        self.mean = nn.Parameter(self.mean, requires_grad=False)
        self.std = nn.Parameter(self.std, requires_grad=False)

        print(self.mean.size())
        print(torch.from_numpy(mean.astype(np.float32)).size())

        # If mean, std is provided, update values accordingly
        if mean is not None and std is not None:
            #
            # ALTHOUGH THIS IS NOT PART OF YOUR IMPLEMENTATION, SEE BELOW.
            #
            # Note here that we use [:] so that actually assign the values, not
            # simply change the reference.
            self.mean[:] = torch.from_numpy(mean.astype(np.float32))
            self.std[:] = torch.from_numpy(std.astype(np.float32))

        # TODO (5 points): Create a linear layer (torch.nn.Linear) and assign
        # it to an attribute of the model. We'll use this layer at the forward
        # pass. Note that the output dimension of this layer should be
        # `config.num_class`. This layer does what we implemented earlier on
        # the previous assignments.
        self.fc = nn.Linear(input_shp[0], config.num_class)

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

        # TODO (10 points): Implement the forward pass
        x_norm = (x - self.mean) / (torch.max(x) - torch.min(x)).item()
        x = self.fc(x_norm)

        return x


#
# model.py ends here

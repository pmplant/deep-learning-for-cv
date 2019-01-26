# regularizor.py ---
#
# Filename: regularizor.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Sun Jan 14 20:45:46 2018 (-0800)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
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


def l2_loss(W):
    """ Loss term related to the weight """

    loss = np.sum(W**2)

    return loss


def l2_grad(W):
    """ Gradient for the l2 regularizor """

    return 2.0 * W


#
# regularizor.py ends here

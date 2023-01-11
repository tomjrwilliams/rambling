
import math

import numpy
import pandas

import torch

from . import graphs


def constant(v, *args, **kwargs):
    # eg. args might be q
    return v
#!/usr/bin/env python3
"""
Miscellaneous utility functions
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None

import inspect
from collections import OrderedDict

# from skimage.io import imsave
import torch
# import orjson as json

import numpy as np
from PIL import Image
# import scipy.ndimage
# import matplotlib.pyplot as plt


# import glob
# import random



def open_image_as_nparray(img_path, dtype):
    return np.array(Image.open(img_path), dtype=dtype)


def EMSG(errtype="ERROR"):
    if errtype in ["E", "ERR", "ERROR"]:
        errtype = "[ERROR] "
    elif errtype in ["W", "WARN", "WARNING"]:
        errtype = "[WARN] "
    elif errtype in ["I", "INFO", "INFORMATION"]:
        errtype = "[INFO] "

    previous_frame = inspect.currentframe().f_back
    _, _, fname, _, _ = inspect.getframeinfo(previous_frame)

    return errtype + " in function " + fname + " :"


def clean_distributed_state_dict(distributed_state_dict):
    new_state_dict = OrderedDict()
    for k, v in distributed_state_dict.items():
        name = k.replace("module.", "")  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)
    rt /= world_size
    return rt


def main():
    """ Dummy main module for Library """
    pass

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()

# encoding=utf-8
"""
universal.py
@time: 2022/7/1 13:27
@description: 
"""
import base64
from io import BytesIO
import numpy as np
from PIL import Image


def img_to_array(im):
    """convert Image to numpy array, and unify the channels to 3"""
    im_in = np.asarray(im)
    if len(im_in.shape) == 2:
        im_in = im_in[:, :, None]
    if im_in.shape[2] == 1:
        im_in = np.repeat(im, 3, axis=2)
    elif im_in.shape[2] == 4:
        im_in = im_in[:, :, 0:3]
    return im_in


def b64_to_img(b64_str):
    """convert b64 str to Image object"""
    return Image.open(BytesIO(base64.b64decode(b64_str)))

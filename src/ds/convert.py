# encoding=utf-8
"""
convert.py
@time: 2022/7/1 00:16
@description: 
"""
from collections import defaultdict

import numpy as np
from PIL import Image

from universal import img_to_array, b64_to_img, img_resize


def combine(im_in_str, im_bg_mat, bg_color):
    """

    :param bg_color: int
    :param im_in_str: b64_str
    :param im_bg_mat: numpy array
    :return:
    """

    color_mapping = {0: (255, 255, 255), 1: (255, 0, 0), 2: (60, 140, 220)}

    # convert b64 to image to array
    im_in = b64_to_img(im_in_str)
    im_in = img_to_array(im_in)

    # segmentation
    im_bg_mat = np.repeat(np.asarray(im_bg_mat)[:, :, None], 3, axis=2) / 255
    foreground = im_in * im_bg_mat  # + np.full(im_in.shape, 255) * (1 - im_bg_mat)

    # fill pure color
    foreground += np.full(im_in.shape, color_mapping[bg_color]) * (1 - im_bg_mat)

    # to image
    im_out = Image.fromarray(np.uint8(foreground))
    # resize
    im_out = img_resize(im_out)

    return im_out


if __name__ == '__main__':
    """test"""
    import requests
    import base64

    from src.ds.inference import segment

    url_portrait = 'https://www.whitehouse.gov/wp-content/uploads/2021/01/45_donald_trump.jpg'
    im_in_string = base64.b64encode(requests.get(url_portrait).content).decode()
    im_bg_array = segment(im_in_string)

    im_combined = combine(im_in_string, im_bg_array, 0)
    im_combined.show('combined')

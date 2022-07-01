# encoding=utf-8
"""
convert.py
@time: 2022/7/1 00:16
@description: 
"""
import numpy as np
from PIL import Image

from src.utils.universal import img_to_array, b64_to_img


def combine(im_in_str, im_bg_mat):
    """

    :param im_in_str: b64_str
    :param im_bg_mat: numpy array
    :return:
    """
    # convert b64 to image
    im_in = b64_to_img(im_in_str)

    # calculate display resolution
    w, h = im_in.width, im_in.height
    # rw, rh = 800, int(h * 800 / (3 * w))

    im_in = img_to_array(im_in)

    # obtain predicted foreground

    im_bg_mat = np.repeat(np.asarray(im_bg_mat)[:, :, None], 3, axis=2) / 255
    foreground = im_in * im_bg_mat + np.full(im_in.shape, 255) * (1 - im_bg_mat)

    # # combine image, foreground, and alpha into one line
    # combined = np.concatenate((im_in, foreground, im_bg_mat * 255), axis=1)
    # combined = Image.fromarray(np.uint8(combined)).resize((rw, rh))

    im_out = Image.fromarray(np.uint8(foreground))

    return im_out


if __name__ == '__main__':
    """test"""
    import requests
    import base64

    from src.utils.inference import segment

    url_portrait = 'https://www.whitehouse.gov/wp-content/uploads/2021/01/45_donald_trump.jpg'
    im_in_string = base64.b64encode(requests.get(url_portrait).content).decode()
    im_bg_array = segment(im_in_string)

    im_combined = combine(im_in_string, im_bg_array)
    im_combined.show('combined')


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


def img_b64(im: Image):
    buffer = BytesIO()
    im.save(buffer, 'png')
    return base64.b64encode(buffer.getvalue()).decode()


def img_resize(im_in: Image):
    """resize image, limit the max dim no more than 1000px"""
    maximum = 800  # TODO write in yaml
    w, h = im_in.size
    if w > maximum or h > maximum:
        if w > h:
            im_out = im_in.resize((maximum, int(maximum * h / w)), Image.ANTIALIAS)  # high quality
        else:
            im_out = im_in.resize((int(maximum * w / h), maximum), Image.ANTIALIAS)
        return im_out
    else:
        return im_in



if __name__ == '__main__':
    print(img_b64(Image.new('RGB', (200, 200))))

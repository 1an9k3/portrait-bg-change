# encoding=utf-8
"""
inference.py
@time: 2022/6/28 00:12
@description: 
"""
import base64
import os
from io import BytesIO

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.modnet import MODNet
from universal import img_to_array

# TODO store in yaml

CKPT_PATH = os.path.join(os.path.dirname(__file__), '', 'models/ckpt')
CKPT_FILENAME = 'modnet_photographic_portrait_matting.ckpt'


def segment(input_img: str):
    """call the model"""
    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(os.path.join(CKPT_PATH, CKPT_FILENAME))
    else:
        weights = torch.load(os.path.join(CKPT_PATH, CKPT_FILENAME), map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()

    # inference
    # load  # TODO be more efficient
    im = Image.open(BytesIO(base64.b64decode(input_img)))
    # unify image channels to 3
    im = img_to_array(im)

    # convert image to PyTorch tensor
    im = Image.fromarray(im)
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()

    matte = (matte * 255).astype('uint8')

    return matte


if __name__ == '__main__':
    """test"""
    import requests

    url_portrait = 'https://www.whitehouse.gov/wp-content/uploads/2021/01/45_donald_trump.jpg'
    im_in_str = base64.b64encode(requests.get(url_portrait).content).decode()
    im_out_array = segment(im_in_str)
    im_out = Image.fromarray(im_out_array, mode='L')  # bg
    print(im_out.format, im_out.size)
    im_out.show('out')

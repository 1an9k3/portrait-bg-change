# encoding=utf-8
"""
demo_api.py
@time: 2022/7/2 16:40
@description: 
"""

from io import BytesIO
import base64
import requests
from PIL import Image

if __name__ == '__main__':
    url_portrait = 'https://www.whitehouse.gov/wp-content/uploads/2021/01/45_donald_trump.jpg'
    with requests.post('http://127.0.0.1:8000/inference',
                       json={'rid': 'test',
                             'image': base64.b64encode(requests.get(url_portrait).content).decode()}) as res:
        # print(res.text)
        assert res.status_code == 200
        im_out = Image.open(BytesIO(base64.b64decode(res.json()['image'])))
        im_out.show('out')

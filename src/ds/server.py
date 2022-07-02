# encoding=utf-8
"""
server.py
@time: 2022/6/26 23:27
@description: 
"""

from loguru import logger
from mosec import Server, Worker

from inference import segment
from pydantic_models import SegmentationRequest
from pydantic import ValidationError

from convert import combine
from universal import img_b64


class Segmentation(Worker):
    """call the machine learning model"""

    def forward(self, data: dict) -> dict:
        """json in, json out"""
        try:
            req = SegmentationRequest(**data)
            logger.info(f"===Req: {req.rid} Start===")
            out_img_array = segment(req.image)
            color = 0 if not req.color else req.color
            logger.info(f"====Bg Color: {color}====")
            im_out = combine(req.image, out_img_array, color)
            b64_out = img_b64(im_out)
            logger.info(f"===Req: {req.rid} Success===")
            return {'rid': req.rid, 'image': b64_out}
        except ValidationError as e:
            logger.error(f"===ValidationError: {e}===")
            return {'error_code': 400}


if __name__ == '__main__':
    server = Server()
    server.append_worker(Segmentation)
    server.run()

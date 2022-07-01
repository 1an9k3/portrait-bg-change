# encoding=utf-8
"""
model_server.py
@time: 2022/6/26 23:27
@description: 
"""

from loguru import logger

from mosec import Server, Worker

from src.utils.inference import segment
from src.models.pydantic_models import SegmentationRequest
from pydantic import ValidationError


class Segmentation(Worker):
    """call the machine learning model"""

    def forward(self, data: dict) -> dict:
        """json in, json out"""
        try:
            req = SegmentationRequest(**data)
            out_img_b64 = segment(req.image)
            return {'rid': req.rid, 'image': out_img_b64}
        except ValidationError as e:
            logger.error(f'===Error Occurred When Validating Request: {e}===')
            return {'err_code': 400}
        finally:
            logger.error(f"===Unknown Error===")
            return {'err_code': 500}


if __name__ == '__main__':
    server = Server()
    server.append_worker(Segmentation)
    server.run()

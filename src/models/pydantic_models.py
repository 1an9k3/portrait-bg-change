# encoding=utf-8
"""
pydantic_models.py
@time: 2022/6/27 13:57
@description: 
"""
from pydantic import BaseModel, constr


class SegmentationRequest(BaseModel):
    """it contains information to do segmentation"""
    rid: str
    image: constr(regex=r'^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{4}|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)$', strict=True)

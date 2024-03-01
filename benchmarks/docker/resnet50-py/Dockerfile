# This is an example for resnet50
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

# kernel(payload: str) takes Base64 string as input
COPY ./kernel.py /app/

# RUN pip install -r requirements.txt

# ENV CUDA_VISIBLE_DEVICES=0
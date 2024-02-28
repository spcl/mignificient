# This is an example for resnet50
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

COPY ./run.py /app/

# RUN pip install -r requirements.txt

# ENV CUDA_VISIBLE_DEVICES=0

ENV INPUTS="/bench/dog.jpg"

CMD /opt/conda/bin/python3 run.py ${INPUTS}
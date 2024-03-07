FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

# run.py is program entry
COPY ./run.py ./kernel.py ./requirements.txt /app/

RUN pip3 install -r /app/requirements.txt

CMD python3 /app/run.py

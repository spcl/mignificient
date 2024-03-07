FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

# test1.py is program entry
COPY ./test1.py ./bert.py ./utils.py ./model ./requirements.txt /app/

RUN pip3 install -r /app/requirements.txt

CMD python3 /app/test1.py

CUDA_VISIBLE_DEVICES=0 \
sarus run -e INPUTS=/bench/dog2.jpg --mount=type=bind,source=/users/pzhou/projects/mignificient/benchmarks/docker/resnet50-py/input,destination=/bench zhoupengyu1998/resnet50-py-test:v1.0 
# sarus run --mount=type=bind,source=/users/pzhou/projects/mignificient/benchmarks/docker/resnet50-py/input,destination=/bench zhoupengyu1998/resnet50-py-test:v1.0 python3 /app/run.py /bench/dog.jpg
# sarus run --mount=type=bind,source=/users/pzhou/projects/mignificient/benchmarks/docker/resnet50-py,destination=/bench pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime python3 /bench/run.py

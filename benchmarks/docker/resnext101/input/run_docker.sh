CUDA_VISIBLE_DEVICES=0 \
sarus run --mount=type=bind,source=/users/pzhou/projects/mignificient/benchmarks/docker/resnet50-py/input,destination=/bench zhoupengyu1998/resnet50-py-test:v1.0 python3 /app/main.py /bench/dog.jpg
CUDA_VISIBLE_DEVICES=0 \
sarus run --mount=type=bind,source=/users/pzhou/projects/mignificient/benchmarks/docker/resnext101/input,destination=/bench zhoupengyu1998/vgg19-test:v1.0 python3 /app/main.py /bench/dog.jpg
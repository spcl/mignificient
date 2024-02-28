use sarus to pull the image from my repo:

sarus pull zhoupengyu1998/resnet50-py-test:v1.0

to start the container with customized input image, change INPUTS=/bench/<IMG_NAME>.jpg accordingly in inputs/run_docker.sh and run:
sh run_docker.sh
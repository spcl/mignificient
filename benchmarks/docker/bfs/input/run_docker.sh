CUDA_VISIBLE_DEVICES=0 \
sarus run -e INPUTS=/bench/graph1MW_6.txt --mount=type=bind,source=/users/pzhou/projects/mignificient/benchmarks/docker/bfs/input,destination=/bench zhoupengyu1998/bfs-test:v1.0 

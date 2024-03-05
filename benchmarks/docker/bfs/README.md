The original BFS CUDA code was obtained from Pawan Harish and P. J. Narayanan at IIIT, 
who have given us permission to include it as part of Rodinia under Rodinia's license.

Input Preprocessing: ```main.cu```
Kernel: ```bfs.cu```
Cuda Kernel Functions: ```kernel.cu, kernel2.cu```

to test the kernel function invocation:
```./bfs ./input/graph1MW_6.txt```
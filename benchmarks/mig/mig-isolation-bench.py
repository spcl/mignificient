#!/usr/bin/env python

import os
import subprocess

def cmd(command):
    o = os.popen(command)
    return o.read().rstrip()

def cmd_prt(command):
    os.system(command)

device = 0
home = os.environ['HOME']
out_dir = f'/users/pzhou/projects/gpuless/benchmark-results/mig-isolation/mig-results'
date = cmd('date --iso-8601=seconds')
out_file = f'{out_dir}/mig-isolation-bench-{date}.out'
n = 1 << 27
# program = f'./cuda-stream/stream -n {n}'
# program = "/users/pzhou/lib/conda/bin/python3 /users/pzhou/projects/gpuless/src/benchmarks/resnet50-py/run-mig.py"
# program = "/users/pzhou/projects/gpuless/src/benchmarks/bfs/bfs-mig /users/pzhou/projects/gpuless/src/benchmarks/bfs/graph1MW_6.txt"
program = "/users/pzhou/lib/conda/bin/python3 /users/pzhou/projects/gpuless/src/benchmarks/BERT-SQuAD/test1.py"
out = []

filter_id1 = ' | head -n1 | sed -rn "s/.*instance ID[ ]+([0-9]*).*/\\1/p"'
filter_ids = ' | grep "created GPU instance" | sed -rn "s/.*instance ID[ ]+([0-9]*).*/\\1/p"'

def get_gpuid(dev):
    cmd = f'nvidia-smi -L | grep "GPU {dev}:" | sed -rn "s/.*GPU-([a-f0-9-]*).*/\\1/p"'
    p = subprocess.Popen(['bash', '-c', cmd], stdout=subprocess.PIPE)
    output, error = p.communicate()
    return output.decode('ascii').rstrip()

def create_partitions(partitions):
    i = cmd(f'sudo nvidia-smi mig -i {device} -cgi {partitions} -C {filter_ids}')
    ids = i.split('\n')
    for k, v in enumerate(ids):
        ids[k] = v.rstrip()
    return ids

def cmd_async(command):
    return subprocess.Popen(['bash', '-c', command], stdout=subprocess.PIPE)

def read_process(p):
    output, error = p.communicate()
    return output.decode('ascii')

def read_stream_output(ident, stream_out):
    lines = stream_out.split('\n')
    ls = []
    for l in lines[1:]: # mig specific
        # s = l.split(',')
        # if len(s) != 2:
        #     continue
        ls.append(f'{ident},{l}\n')
    return ls

# bandwidth no MIG (full GPU)
r = cmd(f'CUDA_VISIBLE_DEVICES={device} {program}')
out += read_stream_output('no partition', r)

gpu_uuid = get_gpuid(device)
print(f'GPU {device}: {gpu_uuid}')

# enable MIG, clear config
cmd(f'sudo nvidia-smi -i {device} -mig 1')
cmd(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 1g 5gb, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 19 -C {filter_id1}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
out += read_stream_output('1g.5gb isolated', r)
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 1g 10gb, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 15 -C {filter_id1}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
out += read_stream_output('1g.10gb isolated', r)
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 2g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 14 -C {filter_id1}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
out += read_stream_output('2g isolated', r)
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 3g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 9 -C {filter_id1}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
out += read_stream_output('3g isolated', r)
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 4g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 5 -C {filter_id1}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
out += read_stream_output('4g isolated', r)
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# bandwidth 7g, in isolation
i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 0 -C {filter_id1}')
r = cmd(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
out += read_stream_output('7g isolated', r)
cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# #
# # using all partitions at the same time
# #

# # bandwidth 1g+2g+4g
# ids = create_partitions('19,14,5')
# p1 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[0]}/0 {program}')
# p2 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[1]}/0 {program}')
# p4 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[2]}/0 {program}')
# out += read_stream_output('1g+2g+4g (1g)', read_process(p1))
# out += read_stream_output('1g+2g+4g (2g)', read_process(p2))
# out += read_stream_output('1g+2g+4g (4g)', read_process(p4))
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# # bandwidth 3g+3g
# ids = create_partitions('9,9')
# p3_1 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[0]}/0 {program}')
# p3_2 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{ids[1]}/0 {program}')
# out += read_stream_output('3g+3g (3g)', read_process(p3_1))
# out += read_stream_output('3g+3g (3g)', read_process(p3_2))
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# #
# # bandwidth for compute instances
# #

# # bandwidth 1c
# i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 0 {filter_id1}')
# cmd_prt(f'sudo nvidia-smi mig -i {device} -gi {i} -cci 0')
# p1 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
# out += read_stream_output('1c isolated', read_process(p1))
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# # bandwidth 3c
# i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 0 {filter_id1}')
# cmd_prt(f'sudo nvidia-smi mig -i {device} -gi {i} -cci 2')
# p1 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
# out += read_stream_output('3c isolated', read_process(p1))
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# # bandwidth 1c+3c+3c
# i = cmd(f'sudo nvidia-smi mig -i {device} -cgi 0 {filter_id1}')
# cmd_prt(f'sudo nvidia-smi mig -i {device} -gi {i} -cci 0,2,2')
# p1 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/0 {program}')
# p2 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/1 {program}')
# p3 = cmd_async(f'CUDA_VISIBLE_DEVICES=MIG-GPU-{gpu_uuid}/{i}/2 {program}')
# out += read_stream_output('1c+3c+3c (1c)', read_process(p1))
# out += read_stream_output('1c+3c+3c (3c)', read_process(p2))
# out += read_stream_output('1c+3c+3c (3c)', read_process(p3))
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dci')
# cmd_prt(f'sudo nvidia-smi mig -i {device} -dgi')

# disable MIG
cmd_prt(f'sudo nvidia-smi -i {device} -mig 0')

# write output
f = open(out_file, 'w')
for line in out:
    f.write(line)
f.close()

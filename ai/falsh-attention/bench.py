import math
import sys
import torch
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from torch.utils.benchmark import Timer

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash.cu','flash2.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()


def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def minimal_attn_forward(q, k, v):
    return minimal_attn.forward(q, k, v)
def minimal_attn_forward2(q, k, v):
    return minimal_attn.forward2(q, k, v)

print('=== benchmarking minimal flash attention ===')
minimal_timer = Timer(
    stmt="minimal_attn_forward(q, k, v)",
    globals={"minimal_attn_forward": minimal_attn_forward, "q": q.half(), "k": k.half(), "v": v.half()},
    label="Minimal Flash Attention"
)
minimal_result = minimal_timer.timeit(20)
print(minimal_result)

print('=== benchmarking minimal flash attention v2 ===')
minimal2_timer = Timer(
    stmt="minimal_attn_forward2(q, k, v)",
    globals={"minimal_attn_forward2": minimal_attn_forward2, "q": q.half(), "k": k.half(), "v": v.half()},
    label="Minimal Flash Attention v2"
)
minimal2_result = minimal2_timer.timeit(20)
print(minimal2_result)

# 计算效率提升
speedup = minimal_result.median / minimal2_result.median
percent = (speedup - 1) * 100
print(f"minimal_attn_forward2 比 minimal_attn_forward 提升效率:({percent:.2f}%)")

# Compare results
def print_red(text):
    print(f"\033[91m{text}\033[0m", file=sys.stderr)
def check(name,standard,out,atol=1e-2):
    if not torch.allclose(standard, out, rtol=0, atol=atol):
        print_red(name+"attn values sanity check: FAILED")

# Sanity check for correctness
check("attn vs forward", manual_attn(q, k, v), minimal_attn_forward(q, k, v).float())
check("attn vs forward2", manual_attn(q, k, v), minimal_attn_forward2(q, k, v).float())
check("forward vs forward2", minimal_attn_forward(q, k, v), minimal_attn_forward2(q, k, v))

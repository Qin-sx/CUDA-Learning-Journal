## RTX4060 性能
1. autotune带来的性能提升最大
2. causal=True 时，区分off-band/on-band的性能提升较大

batch 8 head_num 8 n_ctx 1024 head_dim 128

Summary (ms)
| causal | kernel | time_ms | perf_of_flash |
| --- | --- | --- | --- |
| False | sdpa | 2.329 | 97.0% |
| False | flash-attn | 2.260 | 100.0% |
| False | v1 (基线：每个 tile 都做完整 causal mask) | 3.237 | 69.8% |
| False | v2 (限制 causal 范围到当前行块) | 3.244 | 69.7% |
| False | v3 (划分 off-band/on-band，避免 off-band 做 mask) | 3.245 | 69.6% |
| False | v4 (exp2 softmax 提升吞吐) | 2.844 | 79.4% |
| False | v5 (autotune tile 尺寸与 launch 参数) | 2.840 | 79.6% |
| False | v6 (autotune + 提前剪枝无效配置) | 2.403 | 94.0% |
| False | v7 (block pointer) | 2.427 | 93.1% |
| False | v8 (对齐提示 + multiple_of) | 2.418 | 93.4% |
| True | sdpa | 1.382 | 98.3% |
| True | flash-attn | 1.359 | 100.0% |
| True | v1 (基线：每个 tile 都做完整 causal mask) | 3.231 | 42.1% |
| True | v2 (限制 causal 范围到当前行块) | 2.014 | 67.5% |
| True | v3 (划分 off-band/on-band，避免 off-band 做 mask) | 1.937 | 70.2% |
| True | v4 (exp2 softmax 提升吞吐) | 1.774 | 76.6% |
| True | v5 (autotune tile 尺寸与 launch 参数) | 1.724 | 78.8% |
| True | v6 (autotune + 提前剪枝无效配置) | 1.480 | 91.8% |
| True | v7 (block pointer) | 1.494 | 91.0% |
| True | v8 (对齐提示 + multiple_of) | 1.492 | 91.1% |

[triton官网实现链接](https://github.com/triton-lang/triton/blob/2ccb09ed14dad4a47a9489b46cbc72b1101d8cee/python/tutorials/06-fused-attention.py#L4)

## H800 性能

比flash-attn==2.8.3更快可能是因为没有考虑反向传播？

batch 16 head_num 8

### head_dim 128

| n_ctx | causal | kernel | time_ms | tflops | perf_of_flash |
| --- | --- | --- | --- | --- | --- |
| 1024 | False | flash-attn | 0.285 | 120.717 | 100.0% |
| 1024 | False | v8 | 0.277 | 124.041 | 102.8% |
| 1024 | True | flash-attn | 0.197 | 87.186 | 100.0% |
| 1024 | True | v8 | 0.190 | 90.410 | 103.7% |
| 2048 | False | flash-attn | 1.100 | 124.896 | 100.0% |
| 2048 | False | v8 | 1.063 | 129.242 | 103.5% |
| 2048 | True | flash-attn | 0.667 | 103.074 | 100.0% |
| 2048 | True | v8 | 0.621 | 110.717 | 107.4% |
| 4096 | False | flash-attn | 4.542 | 121.033 | 100.0% |
| 4096 | False | v8 | 4.036 | 136.219 | 112.5% |
| 4096 | True | flash-attn | 2.445 | 112.472 | 100.0% |
| 4096 | True | v8 | 2.234 | 123.064 | 109.4% |
| 8192 | False | flash-attn | 17.146 | 128.251 | 100.0% |
| 8192 | False | v8 | 16.240 | 135.410 | 105.6% |
| 8192 | True | flash-attn | 9.228 | 119.166 | 100.0% |
| 8192 | True | v8 | 8.639 | 127.287 | 106.8% |
| 16384 | False | flash-attn | 70.573 | 124.638 | 100.0% |
| 16384 | False | v8 | 63.237 | 139.098 | 111.6% |
| 16384 | True | flash-attn | 35.578 | 123.624 | 100.0% |
| 16384 | True | v8 | 31.344 | 140.322 | 113.5% |

### head_dim 64

| n_ctx | causal | kernel | time_ms | tflops | perf_of_flash |
| --- | --- | --- | --- | --- | --- |
| 1024 | False | flash-attn | 0.169 | 101.550 | 100.0% |
| 1024 | False | v8 | 0.152 | 112.746 | 111.0% |
| 1024 | True | flash-attn | 0.125 | 68.703 | 100.0% |
| 1024 | True | v8 | 0.115 | 74.653 | 108.7% |
| 2048 | False | flash-attn | 0.668 | 102.934 | 100.0% |
| 2048 | False | v8 | 0.589 | 116.583 | 113.3% |
| 2048 | True | flash-attn | 0.397 | 86.553 | 100.0% |
| 2048 | True | v8 | 0.356 | 96.599 | 111.6% |
| 4096 | False | flash-attn | 2.642 | 104.055 | 100.0% |
| 4096 | False | v8 | 2.382 | 115.421 | 110.9% |
| 4096 | True | flash-attn | 1.454 | 94.550 | 100.0% |
| 4096 | True | v8 | 1.327 | 103.614 | 109.6% |
| 8192 | False | flash-attn | 10.381 | 105.921 | 100.0% |
| 8192 | False | v8 | 9.383 | 117.184 | 110.6% |
| 8192 | True | flash-attn | 5.545 | 99.153 | 100.0% |
| 8192 | True | v8 | 4.834 | 113.749 | 114.7% |
| 16384 | False | flash-attn | 41.428 | 106.161 | 100.0% |
| 16384 | False | v8 | 37.908 | 116.018 | 109.3% |
| 16384 | True | flash-attn | 21.382 | 102.851 | 100.0% |
| 16384 | True | v8 | 19.647 | 111.934 | 108.8% |
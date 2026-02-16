## 性能提升
1. autotune带来的性能提升最大
2. causal=True 时，区分off-band/on-band的性能提升较大

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
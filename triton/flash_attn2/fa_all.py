import math
import time

import torch
import triton

from fa1 import _flash_fwd_kernel_v1
from fa2 import _flash_fwd_kernel_v2
from fa3 import _flash_fwd_kernel_v3
from fa4 import _flash_fwd_kernel_v4
from fa5 import _flash_fwd_kernel_v5
from fa6 import _flash_fwd_kernel_v6
from fa7 import _flash_fwd_kernel_v7
from fa8 import _flash_fwd_kernel_v8


KERNELS = [
	{
		"name": "v1",
		"kernel": _flash_fwd_kernel_v1,
		"autotune": False,
		"block_m": 128,
		"block_n": 64,
		"num_warps": 4,
		"num_stages": 2,
		"notes": "基线：每个 tile 都做完整 causal mask",
	},
	{
		"name": "v2",
		"kernel": _flash_fwd_kernel_v2,
		"autotune": False,
		"block_m": 128,
		"block_n": 64,
		"num_warps": 4,
		"num_stages": 2,
		"notes": "限制 causal 范围到当前行块",
	},
	{
		"name": "v3",
		"kernel": _flash_fwd_kernel_v3,
		"autotune": False,
		"block_m": 128,
		"block_n": 64,
		"num_warps": 4,
		"num_stages": 2,
		"notes": "划分 off-band/on-band，避免 off-band 做 mask",
	},
	{
		"name": "v4",
		"kernel": _flash_fwd_kernel_v4,
		"autotune": False,
		"block_m": 128,
		"block_n": 64,
		"num_warps": 4,
		"num_stages": 2,
		"notes": "exp2 softmax 提升吞吐",
	},
	{
		"name": "v5",
		"kernel": _flash_fwd_kernel_v5,
		"autotune": True,
		"notes": "autotune tile 尺寸与 launch 参数",
	},
	{
		"name": "v6",
		"kernel": _flash_fwd_kernel_v6,
		"autotune": True,
		"notes": "autotune + 提前剪枝无效配置",
	},
	{
		"name": "v7",
		"kernel": _flash_fwd_kernel_v7,
		"autotune": True,
		"notes": "block pointer",
	},
	{
		"name": "v8",
		"kernel": _flash_fwd_kernel_v8,
		"autotune": True,
		"notes": "对齐提示 + multiple_of",
	},
]


def _run_kernel(meta, q, k, v, sm_scale, causal):
	z, h, n_ctx, head_dim = q.shape
	o = torch.empty_like(q)
	l = torch.empty((z, h, n_ctx), device=q.device, dtype=torch.float32)

	if meta.get("autotune", False):
		grid = lambda META: (triton.cdiv(n_ctx, META["BLOCK_M"]), z * h)
		meta["kernel"][grid](
			q, k, v, sm_scale, l, o,
			q.stride(0), q.stride(1), q.stride(2), q.stride(3),
			k.stride(0), k.stride(1), k.stride(2), k.stride(3),
			v.stride(0), v.stride(1), v.stride(2), v.stride(3),
			o.stride(0), o.stride(1), o.stride(2), o.stride(3),
			Z=z, H=h, N_CTX=n_ctx,
			HEAD_DIM=head_dim,
			CAUSAL=causal,
		)
	else:
		block_m = meta["block_m"]
		block_n = meta["block_n"]
		grid = (triton.cdiv(n_ctx, block_m), z * h)
		meta["kernel"][grid](
			q, k, v, sm_scale, l, o,
			q.stride(0), q.stride(1), q.stride(2), q.stride(3),
			k.stride(0), k.stride(1), k.stride(2), k.stride(3),
			v.stride(0), v.stride(1), v.stride(2), v.stride(3),
			o.stride(0), o.stride(1), o.stride(2), o.stride(3),
			Z=z, H=h, N_CTX=n_ctx,
			BLOCK_M=block_m,
			BLOCK_N=block_n,
			HEAD_DIM=head_dim,
			CAUSAL=causal,
			num_warps=meta["num_warps"],
			num_stages=meta["num_stages"],
		)

	return o


def _benchmark_kernel(meta, q, k, v, sm_scale, causal, warmup, iters):
	for _ in range(warmup):
		_run_kernel(meta, q, k, v, sm_scale, causal)
	torch.cuda.synchronize()

	t0 = time.perf_counter()
	for _ in range(iters):
		_run_kernel(meta, q, k, v, sm_scale, causal)
	torch.cuda.synchronize()

	return (time.perf_counter() - t0) / iters


def _benchmark_sdpa(q, k, v, sm_scale, causal, warmup, iters):
	for _ in range(warmup):
		with torch.backends.cuda.sdp_kernel(
			enable_flash=True, enable_mem_efficient=False, enable_math=False
		):
			torch.nn.functional.scaled_dot_product_attention(
				q, k, v, is_causal=causal, scale=sm_scale
			)
	torch.cuda.synchronize()

	t0 = time.perf_counter()
	for _ in range(iters):
		with torch.backends.cuda.sdp_kernel(
			enable_flash=True, enable_mem_efficient=False, enable_math=False
		):
			torch.nn.functional.scaled_dot_product_attention(
				q, k, v, is_causal=causal, scale=sm_scale
			)
	torch.cuda.synchronize()
	return (time.perf_counter() - t0) / iters


def _benchmark_flash_attn(q, k, v, sm_scale, causal, warmup, iters):
	try:
		from flash_attn.flash_attn_interface import flash_attn_func
	except Exception:
		return None

	q_fa = q.transpose(1, 2).contiguous()
	k_fa = k.transpose(1, 2).contiguous()
	v_fa = v.transpose(1, 2).contiguous()

	for _ in range(warmup):
		flash_attn_func(q_fa, k_fa, v_fa, causal=causal, softmax_scale=sm_scale)
	torch.cuda.synchronize()

	t0 = time.perf_counter()
	for _ in range(iters):
		flash_attn_func(q_fa, k_fa, v_fa, causal=causal, softmax_scale=sm_scale)
	torch.cuda.synchronize()
	return (time.perf_counter() - t0) / iters


def _print_ms(label, seconds):
	print(f"{label}: {seconds * 1e3:.3f} ms")


def _render_markdown_table(rows):
	lines = [
		"| causal | kernel | time_ms | perf_of_flash |",
		"| --- | --- | --- | --- |",
	]
	for row in rows:
		pct_text = row["perf_of_flash"]
		lines.append(
			f"| {row['causal']} | {row['kernel']} | {row['time_ms']:.3f} | {pct_text} |"
		)
	return "\n".join(lines)


def main():
	device = "cuda"
	z, h, n_ctx, head_dim = 8, 8, 1024, 128
	q = torch.randn((z, h, n_ctx, head_dim), device=device, dtype=torch.float16)
	k = torch.randn((z, h, n_ctx, head_dim), device=device, dtype=torch.float16)
	v = torch.randn((z, h, n_ctx, head_dim), device=device, dtype=torch.float16)
	sm_scale = 1.0 / math.sqrt(head_dim)

	for causal in [False, True]:
		ref = torch.nn.functional.scaled_dot_product_attention(
			q, k, v, is_causal=causal, scale=sm_scale
		)

		print(f"\n=== Accuracy (causal={causal}) ===")
		for meta in KERNELS:
			out = _run_kernel(meta, q, k, v, sm_scale, causal)
			try:
				torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
				print(f"{meta['name']}: OK | {meta['notes']}")
			except AssertionError:
				max_diff = (out - ref).abs().max().item()
				print(f"{meta['name']}: FAIL (max diff {max_diff:.4e}) | {meta['notes']}")

	warmup = 50
	iters = 2000
	results = []
	flash_times = {}

	for causal in [False, True]:
		print(f"\n=== Benchmark (causal={causal}) ===")
		sdpa_time = _benchmark_sdpa(q, k, v, sm_scale, causal, warmup, iters)
		_print_ms("sdpa", sdpa_time)
		results.append({"causal": causal, "kernel": "sdpa", "time_ms": sdpa_time * 1e3})

		flash_time = _benchmark_flash_attn(q, k, v, sm_scale, causal, warmup, iters)
		if flash_time is None:
			print("flash-attn: not installed")
		else:
			_print_ms("flash-attn", flash_time)
			flash_times[causal] = flash_time
			results.append({
				"causal": causal,
				"kernel": "flash-attn",
				"time_ms": flash_time * 1e3,
			})

		for meta in KERNELS:
			t_kernel = _benchmark_kernel(meta, q, k, v, sm_scale, causal, warmup, iters)
			_print_ms(f"{meta['name']} ({meta['notes']})", t_kernel)
			results.append({
				"causal": causal,
				"kernel": f"{meta['name']} ({meta['notes']})",
				"time_ms": t_kernel * 1e3,
			})

	for row in results:
		flash_time = flash_times.get(row["causal"])
		if flash_time is None:
			row["perf_of_flash"] = "n/a"
		else:
			pct = (flash_time * 1e3 / row["time_ms"]) * 100.0
			row["perf_of_flash"] = f"{pct:.1f}%"

	md = _render_markdown_table(results)
	print("\n=== Summary (ms) ===")
	print(md)
	output_path = "benchmark_results.md"
	with open(output_path, "w", encoding="utf-8") as f:
		f.write(md + "\n")
	print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
	main()

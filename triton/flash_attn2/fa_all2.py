
import math
from triton.testing import do_bench

import torch
import triton

from fa8 import _flash_fwd_kernel_v8


KERNELS = [
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

	def fn():
		_run_kernel(meta, q, k, v, sm_scale, causal)

	return do_bench(fn, rep=iters)





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

	def fn():
		flash_attn_func(q_fa, k_fa, v_fa, causal=causal, softmax_scale=sm_scale)

	return do_bench(fn, rep=iters)


def _print_ms(label, ms):
	print(f"{label}: {ms:.3f} ms")


def _compute_flops(z, h, n_ctx, head_dim, causal: bool):
	# FLOPs for attention: multiply-adds counted as 2 FLOPs.
	# For full attention: 2 * head_dim * n_ctx * n_ctx per head.
	# For causal (triangular) attention: 2 * head_dim * n_ctx * (n_ctx+1)/2 = head_dim * n_ctx * (n_ctx+1)
	if causal:
		ops_per_head = head_dim * n_ctx * (n_ctx + 1)
	else:
		ops_per_head = 2 * head_dim * n_ctx * n_ctx
	return ops_per_head * z * h


def _render_markdown_table(rows):
	lines = [
		"| n_ctx | causal | kernel | time_ms | tflops | perf_of_flash |",
		"| --- | --- | --- | --- | --- | --- |",
	]
	for row in rows:
		pct_text = row["perf_of_flash"]
		tflops_text = f"{row.get('tflops', 0):.3f}"
		lines.append(
			f"| {row.get('n_ctx', '')} | {row['causal']} | {row['kernel']} | {row['time_ms']:.3f} | {tflops_text} | {pct_text} |"
		)
	return "\n".join(lines)


def main():
	device = "cuda"
	z, h, head_dim = 16, 8, 64
	n_ctx_list = [1024, 2048, 4096, 8192, 16384]

	warmup = 50
	iters = 500
	results = []
	flash_times = {}

	for n_ctx in n_ctx_list:
		print(f"\n--- Running for n_ctx={n_ctx} ---")
		q = torch.randn((z, h, n_ctx, head_dim), device=device, dtype=torch.float16)
		k = torch.randn((z, h, n_ctx, head_dim), device=device, dtype=torch.float16)
		v = torch.randn((z, h, n_ctx, head_dim), device=device, dtype=torch.float16)
		sm_scale = 1.0 / math.sqrt(head_dim)

		for causal in [False, True]:
			# Use flash-attn as the reference for accuracy if available.
			try:
				from flash_attn.flash_attn_interface import flash_attn_func
			except Exception:
				flash_attn_func = None

			if flash_attn_func is not None:
				q_fa = q.transpose(1, 2).contiguous()
				k_fa = k.transpose(1, 2).contiguous()
				v_fa = v.transpose(1, 2).contiguous()
				ref = flash_attn_func(q_fa, k_fa, v_fa, causal=causal, softmax_scale=sm_scale)
			else:
				ref = None
				print(f"\nflash-attn not installed; skipping accuracy checks (causal={causal}, n_ctx={n_ctx})")

			print(f"\n=== Accuracy (n_ctx={n_ctx}, causal={causal}) ===")
			for meta in KERNELS:
				out = _run_kernel(meta, q, k, v, sm_scale, causal)
				if ref is None:
					print(f"{meta['name']}: (no reference) | {meta['notes']}")
					continue
				try:
					torch.testing.assert_close(out, ref.transpose(1, 2), atol=1e-2, rtol=1e-2)
					print(f"{meta['name']}: OK | {meta['notes']}")
				except AssertionError:
					max_diff = (out - ref.transpose(1, 2)).abs().max().item()
					print(f"{meta['name']}: FAIL (max diff {max_diff:.4e}) | {meta['notes']}")

		for causal in [False, True]:
			print(f"\n=== Benchmark (n_ctx={n_ctx}, causal={causal}) ===")

			flash_time = _benchmark_flash_attn(q, k, v, sm_scale, causal, warmup, iters)
			if flash_time is None:
				print("flash-attn: not installed")
			else:
				_print_ms("flash-attn", flash_time)
				# record flash-attn time and compute TFLOPS
				flash_times[(causal, n_ctx)] = flash_time
				flops = _compute_flops(z, h, n_ctx, head_dim, causal)
				tflops = flops / (flash_time / 1000.0) / 1e12 if flash_time > 0 else 0.0
				results.append({
					"n_ctx": n_ctx,
					"causal": causal,
					"kernel": "flash-attn",
					"time_ms": flash_time,
					"flops": flops,
					"tflops": tflops,
				})

			for meta in KERNELS:
				t_kernel = _benchmark_kernel(meta, q, k, v, sm_scale, causal, warmup, iters)
				_print_ms(f"{meta['name']} ({meta['notes']})", t_kernel)
				# compute flops and tflops
				flops = _compute_flops(z, h, n_ctx, head_dim, causal)
				tflops = flops / (t_kernel / 1000.0) / 1e12 if t_kernel > 0 else 0.0
				results.append({
					"n_ctx": n_ctx,
					"causal": causal,
					"kernel": f"{meta['name']} ({meta['notes']})",
					"time_ms": t_kernel,
					"flops": flops,
					"tflops": tflops,
				})

	for row in results:
		flash_time = flash_times.get((row["causal"], row.get("n_ctx")))
		if flash_time is None:
			row["perf_of_flash"] = "n/a"
		else:
			pct = (flash_time / row["time_ms"]) * 100.0
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

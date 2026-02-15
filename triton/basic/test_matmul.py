import torch
import triton
import triton.language as tl
import os
import inspect
import ast
import sys

# --- 辅助函数 ---
def print_stage(title):
    print(f"\n\n{'=' * 30} {title} {'=' * 30}")

def save_to_file(filename, content):
    try:
        abs_path = os.path.abspath(filename)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[保存成功] -> 已将完整内容保存到: {abs_path}")
    except Exception as e:
        print(f"[保存失败] -> 无法写入文件 {filename}。错误: {e}")

def print_snippet(content, lines=25):
    lines_list = content.splitlines()
    if len(lines_list) > lines:
        print('\n'.join(lines_list[:lines]))
        print(f"\n... (控制台预览被截断，总共 {len(lines_list)} 行) ...")
    else:
        print(content)

# =========================================================
# 1. Triton matmul kernel
# =========================================================
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator = tl.dot(a, b, accumulator)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    tl.store(c_block_ptr, accumulator.to(tl.float16), boundary_check=(0, 1))

# =========================================================
# 阶段 1: Python 静态分析
# =========================================================
print_stage("阶段 1: Python 静态分析")

base_filename = "matmul_kernel"

try:
    print("\n------- 1.1: Python 源代码 (Source Code) -------")
    kernel_source = inspect.getsource(matmul_kernel.fn)
    print(kernel_source)
    save_to_file(f"{base_filename}.py", kernel_source)

    print("\n------- 1.2: Python 抽象语法树 (AST) -------")
    kernel_ast = ast.parse(kernel_source)
    if sys.version_info >= (3, 9):
        ast_dump_str = ast.dump(kernel_ast, indent=4)
    else:
        ast_dump_str = ast.dump(kernel_ast)
    print_snippet(ast_dump_str, lines=20)
    save_to_file(f"{base_filename}.ast.txt", ast_dump_str)
except (IOError, TypeError) as e:
    print(f"\n[错误] 无法获取 Python 源代码或 AST: {e}")

# =========================================================
# 阶段 2: 触发 JIT 编译
# =========================================================
print_stage("阶段 2: 触发 JIT 编译")

try:
    M, N, K = 128, 128, 128
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
except Exception as e:
    print(f"无法初始化 CUDA tensor: {e}")
    exit()

BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 32

grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

print("正在运行 matmul 内核以触发 JIT 编译...")
matmul_kernel[grid](
    a, b, c,
    M, N, K,
    a.stride(0), a.stride(1),
    b.stride(0), b.stride(1),
    c.stride(0), c.stride(1),
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
)
torch.cuda.synchronize()
print("内核运行完成。所有 IR 现已生成并缓存。")

# =========================================================
# 阶段 3: 提取、打印并保存所有编译产物
# =========================================================
print_stage("阶段 3: 查看所有编译产物")

try:
    device_id = torch.cuda.current_device()
    device_cache = matmul_kernel.device_caches[device_id][0]
    compiled_kernel_object = list(device_cache.values())[0]

    print("\n------- 3.1: Triton IR (TTIR) -------")
    ttir = compiled_kernel_object.asm["ttir"]
    print_snippet(ttir)
    save_to_file(f"{base_filename}.ttir", ttir)

    print("\n------- 3.2: Triton-GPU IR (TTGIR) -------")
    ttgir = compiled_kernel_object.asm["ttgir"]
    print_snippet(ttgir)
    save_to_file(f"{base_filename}.ttgir", ttgir)

    print("\n------- 3.3: LLVM IR (LLIR) -------")
    llir = compiled_kernel_object.asm["llir"]
    print_snippet(llir)
    save_to_file(f"{base_filename}.llir", llir)

    print("\n------- 3.4: PTX (NVIDIA 汇编) -------")
    ptx = compiled_kernel_object.asm["ptx"]
    print_snippet(ptx)
    save_to_file(f"{base_filename}.ptx", ptx)

    print("\n[全部完成] 所有中间产物均已保存到文件中。")

except (KeyError, IndexError, TypeError) as e:
    print(f"\n[错误] 提取编译产物失败: {e}")
    print("JIT 缓存内容:", matmul_kernel.cache)

# =========================================================
# 阶段 4: 结果验证 (与 torch.matmul 对比)
# =========================================================
print_stage("阶段 4: 结果验证")

with torch.no_grad():
    ref = torch.matmul(a, b)
    # 允许一定误差（FP16 运算）
    max_diff = (c - ref).abs().max().item()
    mean_diff = (c - ref).abs().mean().item()

print(f"最大绝对误差: {max_diff:.6e}")
print(f"平均绝对误差: {mean_diff:.6e}")

# 可选阈值检查
tol = 1e-2
if max_diff < tol:
    print(f"[验证通过] max_diff < {tol}")
else:
    print(f"[验证失败] max_diff >= {tol}")
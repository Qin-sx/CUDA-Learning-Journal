import torch
import triton
import triton.language as tl
import os
import inspect  # 用于获取 Python 源代码
import ast      # 用于将源代码解析为 AST
import sys

# --- 辅助函数 ---

def print_stage(title):
    """打印一个统一的阶段分隔符"""
    print(f"\n\n{'=' * 30} {title} {'=' * 30}")

def save_to_file(filename, content):
    """将完整内容保存到文件，并打印确认消息"""
    try:
        abs_path = os.path.abspath(filename)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[保存成功] -> 已将完整内容保存到: {abs_path}")
    except Exception as e:
        print(f"[保存失败] -> 无法写入文件 {filename}。错误: {e}")

def print_snippet(content, lines=25):
    """在控制台打印代码片段，如果太长则截断"""
    lines_list = content.splitlines()
    if len(lines_list) > lines:
        print('\n'.join(lines_list[:lines]))
        print(f"\n... (控制台预览被截断，总共 {len(lines_list)} 行) ...")
    else:
        print(content)


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,  # 必须能被4整除
):
    pid = tl.program_id(axis=0)
    # 每个block处理BLOCK_SIZE * 4个元素
    block_start = pid * BLOCK_SIZE * 4

    offsets = block_start + tl.arange(0, BLOCK_SIZE * 4)
    mask = offsets < n_elements

    # 保证指针对齐(起始地址要对齐16字节，offsets自带步进4，内存不溢出)
    tl.multiple_of(x_ptr, 16)
    tl.multiple_of(y_ptr, 16)
    tl.multiple_of(output_ptr, 16)
    tl.multiple_of(offsets, 4)  # 步进4保证每次4对齐
    tl.max_contiguous(offsets, 4)

    # 加载+写回 (Triton会自动合并load/store为vec4形式)
    x_vec = tl.load(x_ptr + offsets, mask=mask)
    y_vec = tl.load(y_ptr + offsets, mask=mask)
    out_vec = x_vec + y_vec
    tl.store(output_ptr + offsets, out_vec, mask=mask)


# =================================================================
# 阶段 1: Python 静态分析 (Triton 编译器的第一步)
# =================================================================
print_stage("阶段 1: Python 静态分析")

# 定义基础文件名
base_filename = "add_kernel"

try:
    # --- 1.1: Python 源代码 ---
    print("\n------- 1.1: Python 源代码 (Source Code) -------")
    kernel_source = inspect.getsource(add_kernel.fn)
    
    # 打印 (源码很短，完整打印)
    print(kernel_source)
    
    # 保存
    source_filename = f"{base_filename}.py"
    save_to_file(source_filename, kernel_source)
    
    # 解释
    print("\n[解释]: Triton 编译器首先会读取你编写的 Python 源代码字符串。")

    # --- 1.2: Python 抽象语法树 (AST) ---
    print("\n------- 1.2: Python 抽象语法树 (AST) -------")
    kernel_ast = ast.parse(kernel_source)
    
    if sys.version_info >= (3, 9):
        ast_dump_str = ast.dump(kernel_ast, indent=4)
    else:
        ast_dump_str = ast.dump(kernel_ast)

    # 打印片段
    print_snippet(ast_dump_str, lines=20)

    # 保存
    ast_filename = f"{base_filename}.ast.txt"
    save_to_file(ast_filename, ast_dump_str)
    
    # 解释
    print("\n[解释]: Python 的 'ast' 模块将源代码转换为这种树状结构。")
    print("Triton 的前端会遍历这个 AST，将其转换为它自己的第一个 IR (TTIR)。")

except (IOError, TypeError) as e:
    print(f"\n[错误] 无法获取 Python 源代码或 AST: {e}")
    print("这在某些环境（如打包的执行文件）中可能失败。跳过阶段 1。")

# =================================================================
# 阶段 2: 触发 JIT 编译
# =================================================================
print_stage("阶段 2: 触发 JIT 编译")

# 准备数据
try:
    size = 1024
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    output = torch.empty(size, device='cuda', dtype=torch.float32)
except Exception as e:
    print(f"无法初始化 CUDA tensor: {e}")
    print("请确保您有可用的 NVIDIA GPU 并已正确安装 PyTorch。")
    exit()

# 定义 Grid
grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE'] * 4),)

# 运行内核以触发 JIT
print("正在运行内核以触发 JIT 编译...")
add_kernel[grid](x, y, output, size, BLOCK_SIZE=128)
torch.cuda.synchronize()
print("内核运行完成。所有 IR 现已生成并缓存。")

# =================================================================
# 阶段 3: 提取、打印并保存所有编译产物
# =================================================================
print_stage("阶段 3: 查看所有编译产物")

try:
    # 1. 从 JIT 缓存中获取已编译的内核对象
    device_id = torch.cuda.current_device()
    device_cache = add_kernel.device_caches[device_id][0]
    print(f"device_cache 类型: {type(device_cache)}")
    print(f"device_cache 内容: {device_cache}")
    compiled_kernel_object = list(device_cache.values())[0]
    
    # --- 3.1: Triton IR (TTIR) ---
    print("\n------- 3.1: Triton IR (TTIR) -------")
    print(f"(从 .asm['ttir'] 提取)\n")
    ttir = compiled_kernel_object.asm['ttir']
    print_snippet(ttir) # 打印片段
    ttir_filename = f"{base_filename}.ttir"
    save_to_file(ttir_filename, ttir) # 保存完整文件
    print("\n[解释]: 这是第一个 Triton IR (Triton Dialect)。")
    print("它还是硬件**无关**的，非常高级。")
    print("注意看 `tt.load`、`tt.addf`、`tt.store` 和 `tensor<128xf32>` 这样的高级操作和类型。")

    # --- 3.2: Triton-GPU IR (TTGIR) ---
    print("\n------- 3.2: Triton-GPU IR (TTGIR) -------")
    print(f"(从 .asm['ttgir'] 提取)\n")
    ttgir = compiled_kernel_object.asm['ttgir']
    print_snippet(ttgir) # 打印片段
    ttgir_filename = f"{base_filename}.ttgir"
    save_to_file(ttgir_filename, ttgir) # 保存完整文件
    print("\n[解释]: 这是硬件**相关**的 IR (Triton-GPU Dialect)。")
    print("这是最关键的优化阶段。注意 `ttg.Dialect` 和 `#triton_gpu.blocked` 这样的数据布局 (layout) 声明。")
    print("它描述了数据如何映射到线程块和共享内存。")

    # --- 3.3: LLVM IR (LLIR) ---
    print("\n------- 3.3: LLVM IR (LLIR) -------")
    print(f"(从 .asm['llir'] 提取)\n")
    llir = compiled_kernel_object.asm['llir']
    print_snippet(llir) # 打印片段
    llir_filename = f"{base_filename}.llir"
    save_to_file(llir_filename, llir) # 保存完整文件
    print("\n[解释]: 降低到 LLVM IR。代码变得非常冗长，但更接近底层。")
    print("注意看 `!llvm.ptr<f32, 1>` (全局内存指针), `!llvm.ptr<f32, 3>` (共享内存指针), 和 `llvm.getelementptr` (地址计算)。")

    # --- 3.4: PTX (NVIDIA 汇编) ---
    print("\n------- 3.4: PTX (NVIDIA 汇编) -------")
    print(f"(从 .asm['ptx'] 提取)\n")
    ptx = compiled_kernel_object.asm['ptx']
    print_snippet(ptx) # 打印片段
    ptx_filename = f"{base_filename}.ptx"
    save_to_file(ptx_filename, ptx) # 保存完整文件
    print("\n[解释]: 最终生成的 PTX 汇编代码。")
    print("Triton 将此代码交给 NVIDIA 驱动中的 `ptxas` 编译器，由它编译成最终的 SASS 机器码。")
    print("注意 `.visible .entry add_kernel` (内核入口) 和 `ld.global`, `st.global` (向量化加载/存储) 指令。")
    
    print("\n[全部完成] 所有中间产物均已保存到文件中。")

except (KeyError, IndexError, TypeError) as e:
    print(f"\n[错误] 提取编译产物失败: {e}")
    print("JIT 缓存内容:", add_kernel.cache)


# --- 执行 PyTorch 原生版本 (作为 Benchmark) ---
output_torch = x + y

# --- 正确性检查 ---
if torch.allclose(output, output_torch, rtol=1e-5, atol=1e-6):
    print("[正确性检查] 通过：Triton 输出与 PyTorch 输出一致。")
else:
    max_diff = (output - output_torch).abs().max().item()
    print(f"[正确性检查] 失败：最大误差 = {max_diff}")
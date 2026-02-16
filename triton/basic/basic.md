## Block-level programming

在 Triton 中，**不直接指定**线程数，这是它与 CUDA C++ 最显著的区别之一。

在 CUDA 中你需要写 `<<<grid, block>>>`，但在 Triton 中，你关注的是 **`BLOCK_SIZE`（每个块处理多少数据）**，而具体的线程数是由另一个参数 **`num_warps`** 决定的。

以下是具体的逻辑：

### 1. 核心计算公式

一个 Block 中的总线程数公式为：

线程数=num\_warps×32线程数=num\_warps×32


*(注：32 是 NVIDIA GPU 的硬件常量，表示一个 Warp 里的线程数)*



### 2. 默认情况是多少？

如果你没有显式指定，Triton 默认通常是 `num_warps=4`。

* **计算：** 4×32=1284×32=128 个线程。

* **你的例子：**

  * `BLOCK_SIZE = 1024`（每个 Block 负责 1024 个元素）

  * `num_warps = 4`（每个 Block 启动 128 个线程）

  * **平均每个线程处理：** 1024÷128=81024÷128=8 个元素。

### 3. 在哪里看到或修改它？

在启动 Kernel 的地方，你可以显式指定：

```c++
add_kernel[grid](
    x, y, output, 
    n_elements, 
    BLOCK_SIZE=1024,
    num_warps=8  # 这里可以修改，8 * 32 = 256 个线程
)
```

### 4. Triton 如何隐式控制每个线程？

Triton 不直接暴露线程索引，而是让你用 **向量化的索引表达式** 来描述“每个程序实例要处理的元素”。
具体方式是：

- `tl.program_id(axis=0)` 决定当前 program instance（逻辑上的 block）。
- `tl.arange(0, BLOCK_SIZE)` 生成一段连续的向量索引。
- 这些索引组合成 `offsets`，Triton 再根据 `num_warps` 和硬件限制，**自动映射到线程和 SIMD 指令**。

所以“每个线程做什么”并不是你手动写的，而是由 Triton 在编译阶段把向量化表达式拆解到线程层面来完成。


## 编译过程

参考：https://blog.csdn.net/wwlsm_zql/article/details/154389004

Triton 的编译链大致分为以下阶段：

### 1. Python 前端

- 读取 `@triton.jit` 内核的 Python 源码。
- 解析为 AST，并生成 Triton 的高层 IR（TTIR）。

### 2. Triton-GPU IR（TTGIR）

- TTIR 降到与 GPU 架构相关的 IR。
- 在这里进行关键优化与布局决策（block/thread 映射、共享内存布局等）。

### 3. LLVM IR（LLIR）

- TTGIR 进一步降到 LLVM IR。
- 进入更底层的优化与地址计算。

### 4. PTX 与 SASS

- LLVM 生成 PTX。
- PTX 交给 NVIDIA 驱动中的 `ptxas` 编译为 SASS（最终机器码）。

### 5. 缓存与复用

- 编译结果按设备与内核参数缓存。
- 后续调用命中缓存则直接复用。

### 自动生成的指令

在vector_add中，自动生成向量化读取指令。（如果用for循环无法生成）
在matmul中，自动生成共享内存指令。

无法生成向量化读取指令的版本
```python
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,  # 每个“片段”的大小
):
    # 现在每个 pid (Program ID) 负责处理 4 个 BLOCK_SIZE 的数据
    pid = tl.program_id(axis=0)
    
    # 计算当前 Program 处理的总起始位置
    # 因为每个 Program 处理 4 倍的数据量，所以步长要乘以 4
    block_start = pid * BLOCK_SIZE * 4

    # 向量化提示：如果起始地址对齐，LLVM 更容易生成 ld.global.v4
    tl.multiple_of(block_start, 16)

    # 让每个线程处理 4 个连续元素（单线程向量化）
    base = tl.arange(0, BLOCK_SIZE) * 4
    tl.max_contiguous(base, 4)

    # 使用循环显式处理 4 个连续元素
    for i in range(4):
        curr_offsets = block_start + base + i
        mask = curr_offsets < n_elements

        x = tl.load(x_ptr + curr_offsets, mask=mask)
        y = tl.load(y_ptr + curr_offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + curr_offsets, output, mask=mask)
```

生成的向量化读取指令
```
	// begin inline asm
	mov.u32 %r1, 0x0;
	mov.u32 %r2, 0x0;
	mov.u32 %r3, 0x0;
	mov.u32 %r4, 0x0;
	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
	// end inline asm
	.loc	1 59 28                         // test_vec_add.py:59:28
	add.s64 	%rd2, %rd5, %rd7;
	.loc	1 59 20                         // test_vec_add.py:59:20
	// begin inline asm
	mov.u32 %r5, 0x0;
	mov.u32 %r6, 0x0;
	mov.u32 %r7, 0x0;
	mov.u32 %r8, 0x0;
	@%p1 ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ];
	// end inline asm
```

生成的共享内存指令
```
  // begin inline asm
	cp.async.cg.shared.global [ %r127 + 0 ], [ %rd29 + 0 ], 0x10, %r128;
	// end inline asm
```

## triton语法

### `make_block_ptr`&#x20;

`tl.make_block_ptr` 是 Triton 中一种现代化的、更高级的指针管理机制。它的作用是：**定义一个具有结构的“多维数据块指针”，将内存中的一维扁平地址抽象为具有形状和步长的二维（或多维）逻辑块。**

你可以把它理解为给 GPU 指令提供了一张\*\*“带导航的精密地图”\*\*。

#### 1. 核心参数拆解

在你的代码 triton\_matmul.py 中：

```python
a_block_ptr = tl.make_block_ptr(
    base=a_ptr,                      # 1. 基础地址：显存中数据的起点
    shape=(M, K),                    # 2. 完整形状：整个矩阵的大小
    strides=(stride_am, stride_ak),  # 3. 步长：行与行、列与列之间隔了多少数
    offsets=(pid_m * BLOCK_SIZE_M, 0), # 4. 起点坐标：当前这个块从矩阵的哪个位置开始取
    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), # 5. 块形状：你要取出的“补丁”有多大
    order=(1, 0)                     # 6. 内存顺序：数据在内存中是按行存 (1,0) 还是按列存 (0,1)
)
```

#### 2. 为什么要用它？（与旧方式的区别）

在老版本的 Triton 或 CUDA C++ 中，你需要手动计算每个线程的地址：
`ptr = base + i * stride_0 + j * stride_1`。

使用 `tl.make_block_ptr` 的优势在于：

* **自动处理复杂的步长**：你只需要定义一次矩阵的结构（Shape, Stride），剩下的地址换算全部交给 Triton。

* **硬件指令优化**：它能帮助 Triton 编译器生成针对 Tensor Core 优化的加载指令（如 NVIDIA 的 `cp.async` 或 `ldmatrix`），从而极大提升读取效率。

* **边界保护更简单**：当你使用 `tl.load(ptr, boundary_check=(0, 1))` 时，Triton 会根据你在 `make_block_ptr` 中定义的 `shape` 自动判断是否越界并补零（Padding），而不需要你手写复杂的 `mask` 逻辑。

* **配合 `tl.advance`**：如前所述，它让“滑动窗口”的操作变得非常直观。



### `advance`&#x20;

在 Triton 的现代 **Block Pointer（块指针）** 模式中，`tl.advance` 的作用是：**按照指定的偏移量，平滑地“滑动”这个块指针到下一个位置。**

你可以把它想象成在桌面上移动一个“取景框”。

#### 1. 核心作用：指针偏移的自动化

在矩阵乘法 C=A×B*C*=*A*×*B* 的 K*K* 维循环中，我们需要不断读取 A*A* 矩阵的“下一列块”和 B*B* 矩阵的“下一行块”。

* **对于 `a_block_ptr`**：它指向的是 A*A* 矩阵（形状为 M×K*M*×*K*）。为了进行点积，我们需要在 K*K* 轴上移动。

  * `tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))`：意思是“在第 0 维（M）不动，在第 1 维（K）向右移动 `BLOCK_SIZE_K` 个距离”。

* **对于 `b_block_ptr`**：它指向的是 B*B* 矩阵（形状为 K×N*K*×*N*）。

  * `tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))`：意思是“在第 0 维（K）向下移动 `BLOCK_SIZE_K` 个距离，在第 1 维（N）不动”。

#### 2. 为什么要用 `tl.advance` 而不是手动加法？

在使用 `tl.make_block_ptr` 创建指针后，该指针不仅包含了基础地址，还包含了矩阵的**形状、步长和边界信息**。

* **自动处理 Stride**：你不需要手动去乘以 `stride_ak` 或 `stride_bk`。当你告诉它移动 `BLOCK_SIZE_K` 个元素时，`tl.advance` 会自动查看你初始化时设置的 `strides` 参数，换算出准确的字节偏移。

* **保持属性一致性**：它会返回一个新的块指针，这个新指针完美继承了老指针的 `shape`、`strides` 和 `block_shape`。这使得你可以在循环中简单地复用这一行代码，而不需要重新定义复杂的指针逻辑。

* **配合边界检查 (`boundary_check`)**：当指针移动到矩阵边缘时，`tl.advance` 配合 `tl.load` 中的 `boundary_check` 参数，可以非常安全地处理那些“剩下的、不够一个完整 Block”的数据，防止非法内存访问。

#### 3. 直观演示

假设 `BLOCK_SIZE_K = 32`：

* **第 1 次循环**：读取 A\[0:M,0:32] 和 B\[0:32,0:N]。

* **执行 `tl.advance`**：

  * `a_block_ptr` 移动到下标 32。

  * `b_block_ptr` 移动到下标 32。

* **第 2 次循环**：读取 A\[0:M,32:64] 和 B\[32:64,0:N]。

**总结：**
`tl.advance` 就是**高效、安全地移动“数据采集窗口”**。它让你告别了麻烦的底层地址计算，只需关注逻辑上的维度移动。

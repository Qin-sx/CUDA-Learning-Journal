import triton
import triton.language as tl

# Triton kernel: Flash Attention fwd v2
# 优化点：causal 只计算当前行块内的范围。
@triton.jit
def _flash_fwd_kernel_v2(
    Q, K, V, sm_scale,
    L,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # -- 获取当前程序的 ID --
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # -- 计算当前 Batch/Head 的偏移 --
    off_z = off_hz // H
    off_h = off_hz % H

    # -- 计算当前 Batch/Head 在内存中的起始指针 --
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    # -- 初始化当前程序处理的行/列索引 --
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # -- 初始化指针 --
    q_ptrs = Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    k_ptrs = K + k_offset + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk)
    v_ptrs = V + v_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
    o_ptrs = Out + o_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)

    # -- 初始化在线 softmax 的状态变量 --
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # -- 加载 Q 块 --
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # -- 根据 causal 控制 K/V 遍历范围 --
    if CAUSAL:
        start_n = 0
        end_n = (start_m + 1) * BLOCK_M
    else:
        start_n = 0
        end_n = N_CTX

    for cur_n in range(start_n, end_n, BLOCK_N):
        # 加载 K/V
        k = tl.load(k_ptrs + cur_n * stride_kn, mask=(cur_n + offs_n)[None, :] < N_CTX, other=0.0)
        v = tl.load(v_ptrs + cur_n * stride_vn, mask=(cur_n + offs_n)[:, None] < N_CTX, other=0.0)

        # QK^T
        qk = tl.dot(q, k) * sm_scale

        # causal mask
        if CAUSAL:
            offs_n_curr = cur_n + offs_n
            mask = offs_m[:, None] >= offs_n_curr[None, :]
            qk = tl.where(mask, qk, -1.0e9)

        # 在线 softmax
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        m_next = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_next)
        beta = tl.exp(m_ij - m_next)

        acc = acc * alpha[:, None]
        acc = tl.dot((p * beta[:, None]).to(v.dtype), v, acc)

        l_i = l_i * alpha + l_ij * beta
        m_i = m_next

    # 写回输出
    acc = acc / l_i[:, None]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
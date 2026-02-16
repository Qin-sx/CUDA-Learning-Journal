import triton
import triton.language as tl

# Triton kernel: Flash Attention fwd v7
# 优化点：使用 block pointer。
_configs = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
    for bm in [32, 64, 128]
    for bn in [32, 64, 128]
    for s in [2, 3]
    for w in [2, 4, 8]
]


def _keep(conf):
    block_m = conf.kwargs["BLOCK_M"]
    block_n = conf.kwargs["BLOCK_N"]
    return not (block_m * block_n <= 32 * 32 and conf.num_warps >= 8)


def _prune_invalid_configs(configs, named_args, **kwargs):
    n_ctx = kwargs["N_CTX"]
    head_dim = kwargs["HEAD_DIM"]
    causal = kwargs["CAUSAL"]

    pruned = []
    for conf in configs:
        block_m = conf.kwargs["BLOCK_M"]
        block_n = conf.kwargs["BLOCK_N"]
        if block_m > n_ctx:
            continue
        if block_n > head_dim:
            continue
        if causal and block_m < block_n:
            continue
        pruned.append(conf)
    return pruned


@triton.autotune(
    configs=list(filter(_keep, _configs)),
    key=["N_CTX", "HEAD_DIM", "CAUSAL"],
    prune_configs_by={"early_config_prune": _prune_invalid_configs},
)
@triton.jit
def _flash_fwd_kernel_v7(
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

    # -- 初始化当前程序处理的行索引 --
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # -- 使用 block pointer 管理指针 --
    q_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    k_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    v_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    o_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # -- 初始化在线 softmax 的状态变量 --
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # -- 加载 Q 块 --
    q = tl.load(q_ptr, boundary_check=(0, 1), padding_option="zero")

    # -- off-band / on-band 分阶段计算 --
    if not CAUSAL:
        start_n = 0
        end_n = N_CTX
        for cur_n in range(start_n, end_n, BLOCK_N):
            # 加载 K/V
            k = tl.load(tl.advance(k_ptr, (0, cur_n)), boundary_check=(0, 1), padding_option="zero")
            v = tl.load(tl.advance(v_ptr, (cur_n, 0)), boundary_check=(0, 1), padding_option="zero")

            # QK^T
            qk = tl.dot(q, k) * sm_scale
            qk = qk * 1.44269504
            # 在线 softmax
            m_ij = tl.max(qk, 1)
            p = tl.math.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            m_next = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp2(m_i - m_next)
            beta = tl.math.exp2(m_ij - m_next)

            acc = acc * alpha[:, None]
            acc = tl.dot((p * beta[:, None]).to(v.dtype), v, acc)

            l_i = l_i * alpha + l_ij * beta
            m_i = m_next
    else:
        start_n = 0
        end_n = start_m * BLOCK_M
        for cur_n in range(start_n, end_n, BLOCK_N):
            # off-band：不需要 mask
            k = tl.load(tl.advance(k_ptr, (0, cur_n)), boundary_check=(0, 1), padding_option="zero")
            v = tl.load(tl.advance(v_ptr, (cur_n, 0)), boundary_check=(0, 1), padding_option="zero")

            # QK^T
            qk = tl.dot(q, k) * sm_scale
            qk = qk * 1.44269504
            # 在线 softmax
            m_ij = tl.max(qk, 1)
            p = tl.math.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            m_next = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp2(m_i - m_next)
            beta = tl.math.exp2(m_ij - m_next)

            acc = acc * alpha[:, None]
            acc = tl.dot((p * beta[:, None]).to(v.dtype), v, acc)

            l_i = l_i * alpha + l_ij * beta
            m_i = m_next

        start_n = start_m * BLOCK_M
        end_n = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
        for cur_n in range(start_n, end_n, BLOCK_N):
            # on-band：需要 causal mask
            k = tl.load(tl.advance(k_ptr, (0, cur_n)), boundary_check=(0, 1), padding_option="zero")
            v = tl.load(tl.advance(v_ptr, (cur_n, 0)), boundary_check=(0, 1), padding_option="zero")

            # QK^T
            qk = tl.dot(q, k) * sm_scale
            qk = qk * 1.44269504
            offs_n = cur_n + tl.arange(0, BLOCK_N)
            mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(mask, qk, -1.0e9)

            # 在线 softmax
            m_ij = tl.max(qk, 1)
            p = tl.math.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            m_next = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp2(m_i - m_next)
            beta = tl.math.exp2(m_ij - m_next)

            acc = acc * alpha[:, None]
            acc = tl.dot((p * beta[:, None]).to(v.dtype), v, acc)

            l_i = l_i * alpha + l_ij * beta
            m_i = m_next

    # 写回输出
    acc = acc / l_i[:, None]
    tl.store(o_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))
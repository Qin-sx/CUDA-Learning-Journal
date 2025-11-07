##

### CE

CE 是指 Copy Engine，和SM是不同的形式。使用方式是cudaMemcpyAsync等。

https://github.com/NVIDIA/cuda-samples/issues/31

### LSU 

LSU 是指 Load-Store Unit。是SM上的单元。

### TMA

TMA 是指 Tensor Memory Access。也是在SM上实现。

具体可以参考 https://zhuanlan.zhihu.com/p/709750258

总结：之前没有考虑从global memory大量搬运数据到shared memory。类似昇腾的`copy_gm_to_ubuf`成块大量搬运数据。


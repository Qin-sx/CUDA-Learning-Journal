https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/

## 单节点内host和device的地址统一为一块连续的大虚拟地址

>A CUDA-aware MPI implementation must handle buffers differently depending on whether it resides in host or device memory. An MPI implementation could offer different APIs for host and device buffers, or it could add an additional argument indicating where the passed buffer lives. Fortunately, neither of these approaches is necessary because of the Unified Virtual Addressing (UVA) feature introduced in CUDA 4.0 (on Compute Capability 2.0 and later GPUs). With UVA the host memory and the memory of all GPUs in a system (a single node) are combined into one large (virtual) address space.

## 分页内存会导致无法使用DMA和RDMA

DMA（Direct Memory Access）：DMA是一种允许硬件设备直接访问系统内存的技术，无需CPU逐字节处理数据。它通过DMA控制器管理数据传输，从而减轻CPU的负担，提高数据传输效率。
PCIe（Peripheral Component Interconnect Express）：PCIe是一种高速串行计算机扩展总线标准，广泛用于连接高性能设备，如GPU、SSD等。它支持高带宽和低延迟的数据传输。

>Host memory allocated with malloc is usually pageable, that is, the memory pages associated with the memory can be moved around by the kernel, for example to the swap partition on the hard drive. Memory paging has an impact on copying data by DMA and RDMA. DMA and RDMA transfers work independently of the CPU and thus also independently of the OS kernel, so memory pages must not be moved by the kernel while they are being copied. Inhibiting the movement of memory pages is called memory “pinning”. So memory that cannot be moved by the kernel and thus can be used in DMA and RDMA transfers is called pinned memory. As a side note, pinned memory can also be used to speed up host-to-device and device-to-host transfer in general. 


## cuda-aware传输过程

### 1. 有GPUDirect RDMA

应该可以直接GPU传输到GPU

>If GPUDirect RDMA is available the buffer can be directly moved to the network without touching the host memory at all. So the data is directly moved from the buffer in the device memory of MPI Rank 0 to the device memory of MPI Rank 1 with a PCI-E DMA → RDMA → PCI-E DMA sequence as indicated in the picture below by the red arrow with a green outlineline.

### 2. 没有GPUDirect RDMA

需要CPU的网络架构，但是不会用到CPU的内存，所以流水线可以重叠，部分传输过程可以被掩盖

>If no variant of GPUDirect is available, for example if the network adapter does not support GPUDirect, the situation is a little bit more complicated. The buffer needs to be first moved to the pinned CUDA driver buffer and from there to the pinned buffer of the network fabric in the host memory of MPI Rank 0. After that it can be sent over the network. On the receiving MPI Rank 1 these steps need to be carried out in reverse.

>Although this involves multiple memory transfers, the execution time for many of them can be hidden by executing the PCI-E DMA transfers, the host memory copies and the network transfers in a pipelined fashion as shown below.

### 3. 没有CUDA-aware 

需要传输回GPU的内存，流水线一部分无法重叠，所以部分传输过程无法被掩盖

>In contrast, if a non-CUDA-aware MPI implementation is used, the programmer has to take care of staging the data through host memory, by executing the following sequence of calls.

>MPI Rank 0 will execute a cudaMemcpy from device to host followed by MPI_Send.

```c++
cudaMemcpy(s_buf_h,s_buf_d,size,cudaMemcpyDeviceToHost);
MPI_Send(s_buf_h,size,MPI_CHAR,1,100,MPI_COMM_WORLD);
```

>MPI Rank 1 will execute MPI_Recv followed by a cudaMemcpy from device to host.

```c++
MPI_Recv(r_buf_h,size,MPI_CHAR,0,100,MPI_COMM_WORLD,&stat);
cudaMemcpy(r_buf_d,r_buf_h,size,cudaMemcpyHostToDevice);
```

>This will not only introduce an additional memory copy within each node’s host memory, but it will also stall the pipeline after the first cudaMemcpy and after the MPI_Recv on MPI rank 1, so the execution time will be much longer as visualized in the diagram below.
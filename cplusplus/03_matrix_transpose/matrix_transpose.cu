#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#define TYPE int
#define N 128
#define M 128
#define BLOCK_SIZE 32
#define NUM_PER_THREAD 8

// 1 2 3 4
// 5 6 7 8

// 1 5
// 2 6
// 3 7
// 4 8
__global__ void warm_up()
{
    int indexX = threadIdx.x + blockIdx.x * blockDim.x;
    int indexY = threadIdx.y + blockIdx.y * blockDim.y;
    if (indexX < N && indexY < M)
    {
        float a = 0.0f;
        float b = 1.0f;
        float c = a + b;
    }
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
__global__ void matrix_transpose_naive(T *input, T *output)
{

    int indexX = threadIdx.x + blockIdx.x * blockDim.x;
    int indexY = threadIdx.y + blockIdx.y * blockDim.y;

    if (indexX < N && indexY < M)
    {
        int index = indexY * N + indexX;
        int transposedIndex = indexX * M + indexY;

        // this has discoalesced global memory store
        output[transposedIndex] = input[index];

        // this has discoalesced global memore load
        // output[index] = input[transposedIndex];
        // printf("%d, %d, %d, %d, %d \n", indexX, indexY, index, transposedIndex, input[index]);
    }
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
__global__ void matrix_transpose_shared_uncoalesed(T *input, T *output)
{

    __shared__ T sharedMemory[BLOCK_SIZE][BLOCK_SIZE];

    // global index
    int indexX = threadIdx.x + blockIdx.x * blockDim.x;
    int indexY = threadIdx.y + blockIdx.y * blockDim.y;
    if (indexX < N && indexY < M)
    {
        int index = indexY * N + indexX;
        int transposedIndex = indexX * M + indexY;

        // local index
        int localIndexX = threadIdx.x;
        int localIndexY = threadIdx.y;

        // reading from global memory in coalesed manner and performing tanspose in shared memory
        sharedMemory[localIndexX][localIndexY] = input[index];

        __syncthreads();

        // writing into global memory in coalesed fashion via transposed data in shared memory
        output[transposedIndex] = sharedMemory[localIndexX][localIndexY];
    }
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
__global__ void matrix_transpose_shared_uncoalesed_no_conflict(T *input, T *output)
{

    __shared__ T sharedMemory[BLOCK_SIZE][BLOCK_SIZE + 1];

    // global index
    int indexX = threadIdx.x + blockIdx.x * blockDim.x;
    int indexY = threadIdx.y + blockIdx.y * blockDim.y;
    if (indexX < N && indexY < M)
    {
        int index = indexY * N + indexX;
        int transposedIndex = indexX * M + indexY;

        // local index
        int localIndexX = threadIdx.x;
        int localIndexY = threadIdx.y;

        // reading from global memory in coalesed manner and performing tanspose in shared memory
        sharedMemory[localIndexX][localIndexY] = input[index];

        __syncthreads();

        // writing into global memory in coalesed fashion via transposed data in shared memory
        output[transposedIndex] = sharedMemory[localIndexX][localIndexY];
    }
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
__global__ void matrix_transpose_shared_coalesed(T *input, T *output)
{

    __shared__ T sharedMemory[BLOCK_SIZE][BLOCK_SIZE];

    // global index
    int indexX = threadIdx.x + blockIdx.x * blockDim.x;
    int indexY = threadIdx.y + blockIdx.y * blockDim.y;
    // local index
    int localIndexX = threadIdx.x;
    int localIndexY = threadIdx.y;

    if (indexX < N && indexY < M)
    {
        int index = indexY * N + indexX;

        // reading from global memory in coalesed manner and performing tanspose in shared memory
        sharedMemory[localIndexX][localIndexY] = input[index];
    }
    __syncthreads();

    // transposed global memory index
    int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
    int tindexY = threadIdx.y + blockIdx.x * blockDim.y;
    if (tindexX < M && tindexY < N)
    {
        int transposedIndex = tindexY * M + tindexX;
        // writing into global memory in coalesed fashion via transposed data in shared memory
        output[transposedIndex] = sharedMemory[localIndexY][localIndexX];
    }
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
__global__ void matrix_transpose_shared_coalesed_no_conflict(T *input, T *output)
{

    __shared__ T sharedMemory[BLOCK_SIZE][BLOCK_SIZE + 1];

    // global index
    int indexX = threadIdx.x + blockIdx.x * blockDim.x;
    int indexY = threadIdx.y + blockIdx.y * blockDim.y;
    // local index
    int localIndexX = threadIdx.x;
    int localIndexY = threadIdx.y;

    if (indexX < N && indexY < M)
    {
        int index = indexY * N + indexX;

        // reading from global memory in coalesed manner and performing tanspose in shared memory
        sharedMemory[localIndexX][localIndexY] = input[index];
    }
    __syncthreads();

    // transposed global memory index
    int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
    int tindexY = threadIdx.y + blockIdx.x * blockDim.y;
    if (tindexX < M && tindexY < N)
    {
        int transposedIndex = tindexY * M + tindexX;
        // writing into global memory in coalesed fashion via transposed data in shared memory
        output[transposedIndex] = sharedMemory[localIndexY][localIndexX];
    }
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
__global__ void matrix_transpose_multi_elem(T *input, T *output)
{

    __shared__ T sharedMemory[BLOCK_SIZE][BLOCK_SIZE + 1];

    // global index
    int indexX = threadIdx.x + blockIdx.x * blockDim.x;
    int indexY = threadIdx.y + blockIdx.y * BLOCK_SIZE;
    // local index
    int localIndexX = threadIdx.x;
    int localIndexY = threadIdx.y;

    int row_stride = BLOCK_SIZE / NUM_PER_THREAD;

    if (indexX < N)
    {
        #pragma unroll
        for (int yOffset = 0; yOffset < BLOCK_SIZE; yOffset += row_stride)
        {
            int indexYNew = indexY + yOffset;
            if (indexYNew < M)
            {
                int index = indexYNew * N + indexX;

                // reading from global memory in coalesed manner and performing tanspose in shared memory
                sharedMemory[localIndexX][localIndexY + yOffset] = input[index];
                // printf("localIndexX = %d, localIndexY + yOffset = %d, localIndexY = %d, yOffset = %d, index = %d, indexYNew = %d, indexX = %d\n",
                //        localIndexX, localIndexY + yOffset, localIndexY, yOffset, index, indexYNew, indexX);
                // if(index == 1024 || index == 992)
                // {
                //   printf("index = %d, input[index] = %d, indexX = %d, indexYNew = %d, localIndexX = %d, localIndexY = %d, localIndexY + yOffset= %d, yOffset = %d, blockIdx.x = %d, blockIdx.y = %d\n",
                //       index, input[index], indexX, indexYNew, localIndexX, localIndexY, localIndexY + yOffset, yOffset, blockIdx.x, blockIdx.y);
                // }
            }

        }
    }


    __syncthreads();

    // transposed global memory index
    int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
    int tindexY = threadIdx.y + blockIdx.x * BLOCK_SIZE;
    if (tindexX < M)
    {
        #pragma unroll
        for (int yOffset = 0; yOffset < BLOCK_SIZE; yOffset += row_stride)
        {
            int tindexYNew = tindexY + yOffset;
            if(tindexYNew < N)
            {
                int transposedIndex = tindexYNew * M + tindexX;
                // writing into global memory in coalesed fashion via transposed data in shared memory
                output[transposedIndex] = sharedMemory[localIndexY + yOffset][localIndexX];
                // printf("localIndexY + yOffset = %d, localIndexX = %d, localIndexY = %d, yOffset = %d, transposedIndex = %d, tindexYNew = %d, tindexX = %d\n",
                //        localIndexY + yOffset, localIndexX,  localIndexY, yOffset, transposedIndex, tindexYNew, tindexX);
            }

        }
    }
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
__global__ void matrix_transpose_multi_elem2(T *input, T *output)
{

    __shared__ T sharedMemory[BLOCK_SIZE][BLOCK_SIZE];

    // global index
    int indexX = threadIdx.x + blockIdx.x * blockDim.x;
    int indexY = threadIdx.y + blockIdx.y * BLOCK_SIZE;
    // local index
    int localIndexX = threadIdx.x;
    int localIndexY = threadIdx.y;

    int row_stride = BLOCK_SIZE / NUM_PER_THREAD;

    if (indexX < N)
    {
        #pragma unroll
        for (int yOffset = 0; yOffset < BLOCK_SIZE; yOffset += row_stride)
        {
            int indexYNew = indexY + yOffset;
            if (indexYNew < M)
            {
                int index = indexYNew * N + indexX;

                // reading from global memory in coalesed manner and performing tanspose in shared memory
                sharedMemory[localIndexX][((localIndexY + yOffset) + localIndexX) % BLOCK_SIZE] = input[index];
                // printf("localIndexX = %d, localIndexY + yOffset = %d, ((localIndexY + yOffset) + localIndexX) mod BLOCK_SIZE = %d, localIndexY = %d, yOffset = %d, index = %d, indexYNew = %d, indexX = %d\n",
                //        localIndexX, localIndexY + yOffset, ((localIndexY + yOffset) + localIndexX) % BLOCK_SIZE, localIndexY, yOffset, index, indexYNew, indexX);
                // if(index == 1024 || index == 992)
                // {
                //   printf("index = %d, input[index] = %d, indexX = %d, indexYNew = %d, localIndexX = %d, localIndexY = %d, localIndexY + yOffset= %d, yOffset = %d, blockIdx.x = %d, blockIdx.y = %d\n",
                //       index, input[index], indexX, indexYNew, localIndexX, localIndexY, localIndexY + yOffset, yOffset, blockIdx.x, blockIdx.y);
                // }
            }

        }
    }


    __syncthreads();

    // transposed global memory index
    int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
    int tindexY = threadIdx.y + blockIdx.x * BLOCK_SIZE;
    if (tindexX < M)
    {
        #pragma unroll
        for (int yOffset = 0; yOffset < BLOCK_SIZE; yOffset += row_stride)
        {
            int tindexYNew = tindexY + yOffset;
            if(tindexYNew < N)
            {
                int transposedIndex = tindexYNew * M + tindexX;
                // writing into global memory in coalesed fashion via transposed data in shared memory
                output[transposedIndex] = sharedMemory[localIndexY + yOffset][(localIndexX + (localIndexY + yOffset)) % BLOCK_SIZE];
                // printf("localIndexY + yOffset = %d, localIndexX = %d, (localIndexX + (localIndexY + yOffset)) mod BLOCK_SIZE = %d, localIndexY = %d, yOffset = %d, transposedIndex = %d, tindexYNew = %d, tindexX = %d\n",
                //        localIndexY + yOffset, localIndexX, (localIndexX + (localIndexY + yOffset)) % BLOCK_SIZE, localIndexY, yOffset, transposedIndex, tindexYNew, tindexX);
            }
        }
    }
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
__global__ void matrix_transpose_multi_elem3(T *input, T *output)
{

    __shared__ T sharedMemory[BLOCK_SIZE][BLOCK_SIZE];

    // global index
    int indexX = threadIdx.x + blockIdx.x * blockDim.x;
    int indexY = threadIdx.y + blockIdx.y * BLOCK_SIZE;
    // local index
    int localIndexX = threadIdx.x;
    int localIndexY = threadIdx.y;

    int row_stride = BLOCK_SIZE / NUM_PER_THREAD;

    if (indexX < N)
    {
        #pragma unroll
        for (int yOffset = 0; yOffset < BLOCK_SIZE; yOffset += row_stride)
        {
            int indexYNew = indexY + yOffset;
            if (indexYNew < M)
            {
                int index = indexYNew * N + indexX;

                // reading from global memory in coalesed manner and performing tanspose in shared memory
                sharedMemory[localIndexX][(localIndexY + yOffset) ^ localIndexX] = input[index];
            }

        }
    }


    __syncthreads();

    // transposed global memory index
    int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
    int tindexY = threadIdx.y + blockIdx.x * BLOCK_SIZE;
    if (tindexX < M)
    {
        #pragma unroll
        for (int yOffset = 0; yOffset < BLOCK_SIZE; yOffset += row_stride)
        {
            int tindexYNew = tindexY + yOffset;
            if(tindexYNew < N)
            {
                int transposedIndex = tindexYNew * M + tindexX;
                // writing into global memory in coalesed fashion via transposed data in shared memory
                output[transposedIndex] = sharedMemory[localIndexY + yOffset][localIndexX ^ (localIndexY + yOffset)];
            }
        }
    }
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
void print_output(T *a, T *b)
{
    for (int i = 0; i < N * M; ++i)
    {
        if (i % N == 0)
        {
            std::cout << std::endl;
        }
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < N * M; ++i)
    {
        if (i % M == 0)
        {
            std::cout << std::endl;
        }
        std::cout << b[i] << " ";
    }
}

int main()
{
    // Allocate space for host copies of a, b
    thrust::host_vector<TYPE> a(N * M);
    thrust::host_vector<TYPE> b(N * M);

    // Allocate space for device copies of a, b
    thrust::device_vector<TYPE> d_a(N * M, 0);
    thrust::device_vector<TYPE> d_b(N * M, 0);
    thrust::sequence(d_a.begin(), d_a.end(), 0, 1);

    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 no_of_blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    warm_up<<<no_of_blocks, threads_per_block>>>();
    matrix_transpose_naive<<<no_of_blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()));
    matrix_transpose_shared_uncoalesed<<<no_of_blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()));
    matrix_transpose_shared_uncoalesed_no_conflict<<<no_of_blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()));
    matrix_transpose_shared_coalesed<<<no_of_blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()));
    matrix_transpose_shared_coalesed_no_conflict<<<no_of_blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()));

    dim3 threads_per_block_multielem(BLOCK_SIZE, int(BLOCK_SIZE / NUM_PER_THREAD), 1);
    // // std::cout <<"no_of_blocks"<<no_of_blocks.x <<" "<<no_of_blocks.y<<" "<<no_of_blocks.z<< std::endl;
    matrix_transpose_multi_elem<<<no_of_blocks, threads_per_block_multielem>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()));

    matrix_transpose_multi_elem2<<<no_of_blocks, threads_per_block_multielem>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()));
    matrix_transpose_multi_elem3<<<no_of_blocks, threads_per_block_multielem>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()));

    cudaDeviceSynchronize();

    thrust::copy(d_a.begin(), d_a.end(), a.begin());
    thrust::copy(d_b.begin(), d_b.end(), b.begin());

    // print_output(a.data(), b.data());

    return 0;
}
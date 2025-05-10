#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <random>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>


using namespace nvcuda;

#define TYPE half
#define TYPE_MATC float
#define TYPEV half2
// #define M 4
// #define K 3
// #define N 2

#define M 512
#define K 512
#define N 512

#define BLOCK_SIZE 32

#define BLOCK_SIZE_M BLOCK_SIZE
#define BLOCK_SIZE_N BLOCK_SIZE
#define NUM_PER_THREAD_M 4
#define NUM_PER_THREAD_N 1
#define DIVIDE_M (BLOCK_SIZE_M / NUM_PER_THREAD_M)
#define DIVIDE_N (BLOCK_SIZE_N / NUM_PER_THREAD_N)

#define VECTOR_N 4

#define SUBK BLOCK_SIZE
// #define SUBK 32

#define FLOAT4(pointer) (reinterpret_cast<TYPEV*>(&(pointer))[0])
#define HALF2(pointer) (reinterpret_cast<TYPEV*>(&(pointer))[0])

// M 4 K 3
//  1  2  3
//  4  5  6
//  7  8  9
// 10 11 12

// K 3 N 2
// 1 4
// 2 5
// 3 6

// M 4 N 2
// 1*1 + 2*2 + 3*3 = 14         1*4 + 2*5 + 3*6 = 32
// 4*1 + 5*2 + 6*3 = 32         4*4 + 5*5 + 6*6 = 77
// 7*1 + 8*2 + 9*3 = 50         7*4 + 8*5 + 9*6 = 122
// 10*1 + 11*2 + 12*3 = 68      10*4 + 11*5 + 12*6 = 167
//  x  x
// y
// y
// y
// y

__global__ void
warm_up()
{
    int indexX = threadIdx.x + blockIdx.x * blockDim.x;
    int indexY = threadIdx.y + blockIdx.y * blockDim.y;
    if (indexX < M && indexY < N)
    {
        float a = 0.0f;
        float b = 1.0f;
        float c = a + b;
    }
}

template <typename T, typename U>
__global__ void matrix_multiplication0(const T *a, const T *b, U *c)
{
    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    int idY = blockIdx.y * blockDim.y + threadIdx.y;
    if (idY < M && idX < N)
    {
        T cTmp = 0;
        for (int i = 0; i < K; ++i)
        {
            cTmp += a[idY * K + i] * b[i * N + idX];
        }
        c[idY * N + idX] = cTmp;
    }
}

template <typename T, typename U>
__global__ void matrix_multiplication1(const T *a, const T *b, U *c)
{
    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    int idY = blockIdx.y * blockDim.y + threadIdx.y;

    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    __shared__ T as[BLOCK_SIZE_M][SUBK];
    __shared__ T bs[SUBK][BLOCK_SIZE_N];

    {
        U cTmp = 0;

        // read a and b into shared memory by SUBK
        // SUBK * BLOCK_SIZE_M 1, SUBK * BLOCK_SIZE_M 2, SUBK * BLOCK_SIZE_M 3...
        // BLOCK_SIZE_N * SUBK 1, BLOCK_SIZE_N * SUBK 2, BLOCK_SIZE_N * SUBK 3...
        for (int i = 0; i < K; i += SUBK)
        {
            int idXTmp = i + tidX;
            if (idY < M && idXTmp < K && tidX < SUBK)
                as[tidY][tidX] = a[idY * K + idXTmp]; // tidX < SUBK, tidY < BLOCK_SIZE_M
            int idYTmp = i + tidY;
            if (idYTmp < K && idX < N && tidY < SUBK)
                bs[tidY][tidX] = b[idYTmp * N + idX]; // tidX < BLOCK_SIZE_N, tidY < SUBK
            __syncthreads();
            for (int j = 0; j < SUBK; ++j)
            {
                // 0  0  1  1
                // 0  0  1  1
                // 2  2  3  3
                // 2  2  3  3
                // i0    i1
                // j0 j1 j0 j1
                if (i + j < K)
                    cTmp += __half2float(as[tidY][j] * bs[j][tidX]); // why no bank conflict? as[0,1...][tidX] = 0 yes, as[0][tidX] = 0 no.  different position in same bank will cause bank conflict
                // if(idX == 0 && idY == 0)
                // {
                //   printf("i: %d, j: %d, idX: %d, idY: %d, as: %d, bs: %d, cTmp: %d\n", i, j, idX, idY, as[tidY * BLOCK_SIZE_M + j], bs[j * BLOCK_SIZE_N + tidX], cTmp);
                // }
                // if(idX == 8 && idY == 0)
                // {
                //   printf("i: %d, j: %d, idX: %d, idY: %d, as: %d, bs: %d, cTmp: %d\n", i, j, idX, idY, as[tidY * BLOCK_SIZE_M + j], bs[j * BLOCK_SIZE_N + tidX], cTmp);
                // }
            }
            __syncthreads();
        }

        if (idY < M && idX < N)
            c[idY * N + idX] = cTmp;
    }
}

template <typename T, typename U>
__global__ void matrix_multiplication2(const T *a, const T *b, U *c)
{
    int idX = blockIdx.x * (blockDim.x * NUM_PER_THREAD_N) + threadIdx.x;
    int idY = blockIdx.y * (blockDim.y * NUM_PER_THREAD_M) + threadIdx.y;

    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    __shared__ T as[BLOCK_SIZE_M][SUBK];
    __shared__ T bs[SUBK][BLOCK_SIZE_N];

    {
        // T cTmp = 0;
        U cTmp[NUM_PER_THREAD_M][NUM_PER_THREAD_N] = {0};

        for (int i = 0; i < K; i += SUBK)
        {
#pragma unroll
            for (int m = 0; m < BLOCK_SIZE_M; m += DIVIDE_M)
            {
#pragma unroll
                for (int n = 0; n < BLOCK_SIZE_N; n += DIVIDE_N)
                {
                    int idXTmp = i + tidX + n;
                    int idYTmp = idY + m;
                    if (idYTmp < M && idXTmp < K && (tidX + n) < SUBK)
                        as[tidY + m][tidX + n] = a[idYTmp * K + idXTmp];
                    // printf("i: %d, tidX: %d, tidY: %d, m: %d, (tidY + m) * BLOCK_SIZE_M + (tidX + n): %d, as: %d\n", i, tidX, tidY, m, (tidY + m) * BLOCK_SIZE_M + (tidX + n), as[(tidY + m) * BLOCK_SIZE_M + (tidX + n)]);
                }
                // as[(tidY + m) * BLOCK_SIZE_M + tidX] = a[(idY + m) * K + i + tidX];
                // printf("i: %d, tidX: %d, tidY: %d, m: %d, (tidY + m) * BLOCK_SIZE_M + tidX: %d, as: %d\n", i, tidX, tidY, m, (tidY + m) * BLOCK_SIZE_M + tidX, as[(tidY + m) * BLOCK_SIZE_M + tidX]);
            }

#pragma unroll
            for (int n = 0; n < BLOCK_SIZE_N; n += DIVIDE_N)
            {
#pragma unroll
                for (int m = 0; m < BLOCK_SIZE_M; m += DIVIDE_M)
                {
                    int idXTmp = idX + n;
                    int idYTmp = i + tidY + m;
                    if (idYTmp < K && idXTmp < N && tidY + m < SUBK)
                        bs[tidY + m][tidX + n] = b[idYTmp * N + idXTmp];

                    // printf("i: %d, tidX: %d, tidY: %d, m: %d, n: %d, (tidY + m) * BLOCK_SIZE_N + (tidX + n): %d, bs: %d\n", i, tidX, tidY, m, n, (tidY + m) * BLOCK_SIZE_N + (tidX + n), bs[(tidY + m) * BLOCK_SIZE_N + (tidX + n)]);
                }
                // bs[tidY * BLOCK_SIZE_N + (tidX + n)] = b[(i + tidY) * N + idX + n];
                // printf("i: %d, tidX: %d, tidY: %d, n: %d, tidY * BLOCK_SIZE_N + (tidX + n): %d, bs: %d\n", i, tidX, tidY, n, tidY * BLOCK_SIZE_N + (tidX + n), bs[tidY * BLOCK_SIZE_N + (tidX + n)]);
            }

            __syncthreads();
            for (int j = 0; j < SUBK; ++j)
            {
#pragma unroll
                for (int m = 0; m < NUM_PER_THREAD_M; ++m)
                {
#pragma unroll
                    for (int n = 0; n < NUM_PER_THREAD_N; ++n)
                    {
                        if (i + j < K)
                        {
                            // if ((idY + m * DIVIDE_M) < M && (idX + n * DIVIDE_N) < N && i + j < K)
                            cTmp[m][n] += __half2float(as[(tidY + m * DIVIDE_M)][j] * bs[j][(tidX + n * DIVIDE_N)]);
                            // if (idX == 0 && idY == 0)
                            // {
                            //     printf("i: %d, j: %d, m: %d, n: %d, idX: %d, idY: %d, as: %d, bs: %d, cTmp: %d\n", i, j, m, n, idX, idY, as[(tidY + m * DIVIDE_M)][j], bs[j][(tidX + n * DIVIDE_N)], cTmp[m][n]);
                            // }
                        }
                    }
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int m = 0; m < NUM_PER_THREAD_M; ++m)
        {
#pragma unroll
            for (int n = 0; n < NUM_PER_THREAD_N; ++n)
            {
                if ((idY + m * DIVIDE_M) < M && (idX + n * DIVIDE_N) < N)
                    c[(idY + m * DIVIDE_M) * N + idX + n * DIVIDE_N] = cTmp[m][n];
            }
        }

        // c[idY * N + idX] = cTmp;
    }
}

template <typename T, typename U>
__global__ void matrix_multiplication3(const T *a, const T *b, U *c)
{
    int idX = blockIdx.x * (blockDim.x * NUM_PER_THREAD_N) + threadIdx.x;
    int idY = blockIdx.y * (blockDim.y * NUM_PER_THREAD_M) + threadIdx.y;

    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    __shared__ T as[BLOCK_SIZE_M][SUBK];
    __shared__ T bs[SUBK][BLOCK_SIZE_N];

    {
        // T cTmp = 0;
        U cTmp[NUM_PER_THREAD_M][NUM_PER_THREAD_N] = {0};

        for (int i = 0; i < K; i += SUBK)
        {
#pragma unroll
            for (int m = 0; m < BLOCK_SIZE_M; m += DIVIDE_M)
            {
#pragma unroll
                for (int n = 0; n < BLOCK_SIZE_N; n += DIVIDE_N)
                {
                    int idXTmp = i + tidX + n;
                    int idYTmp = idY + m;
                    if (idYTmp < M && idXTmp < K && (tidX + n) < SUBK)
                        as[tidY + m][tidX + n] = a[idYTmp * K + idXTmp];
                }
            }

#pragma unroll
            for (int n = 0; n < BLOCK_SIZE_N; n += DIVIDE_N)
            {
#pragma unroll
                for (int m = 0; m < BLOCK_SIZE_M; m += DIVIDE_M)
                {
                    int idXTmp = idX + n;
                    int idYTmp = i + tidY + m;
                    if (idYTmp < K && idXTmp < N && tidY + m < SUBK)
                        bs[tidY + m][tidX + n] = b[idYTmp * N + idXTmp];
                }
            }

            __syncthreads();
            for (int j = 0; j < SUBK; ++j)
            {
                T ar;
                T br[NUM_PER_THREAD_N];
                for (int n = 0; n < NUM_PER_THREAD_N; ++n)
                {
                  br[n] = bs[j][(tidX + n * DIVIDE_N)];
                }

#pragma unroll
                for (int m = 0; m < NUM_PER_THREAD_M; ++m)
                {
                    ar = as[(tidY + m * DIVIDE_M)][j];
#pragma unroll
                    for (int n = 0; n < NUM_PER_THREAD_N; ++n) // NUM_PER_THREAD_N = 1 for bank conflict
                    {
                        // br = bs[j][(tidX + n * DIVIDE_N)];
                        if (i + j < K)
                        {
                            cTmp[m][n] += __half2float(ar * br[n]);
                        }
                    }
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int m = 0; m < NUM_PER_THREAD_M; ++m)
        {
#pragma unroll
            for (int n = 0; n < NUM_PER_THREAD_N; ++n)
            {
                if ((idY + m * DIVIDE_M) < M && (idX + n * DIVIDE_N) < N)
                    c[(idY + m * DIVIDE_M) * N + idX + n * DIVIDE_N] = cTmp[m][n];
            }
        }
    }
}

template <typename T, typename U>
__global__ void matrix_multiplication4(const T *a, const T *b, U *c)
{
    int idX = blockIdx.x * (blockDim.x * NUM_PER_THREAD_N) + threadIdx.x;
    int idY = blockIdx.y * (blockDim.y * NUM_PER_THREAD_M) + threadIdx.y;

    int tidX = threadIdx.x;
    int tidY = threadIdx.y;

    __shared__ T as[2][BLOCK_SIZE_M][SUBK];
    __shared__ T bs[2][SUBK][BLOCK_SIZE_N];

    {
        // T cTmp = 0;
        U cTmp[NUM_PER_THREAD_M][NUM_PER_THREAD_N] = {0};

#pragma unroll
        for (int m = 0; m < BLOCK_SIZE_M; m += DIVIDE_M)
        {
#pragma unroll
            for (int n = 0; n < BLOCK_SIZE_N; n += DIVIDE_N)
            {
                int idXTmp = tidX + n;
                int idYTmp = idY + m;
                if (idYTmp < M && idXTmp < K && tidX + n < SUBK)
                    as[0][tidY + m][tidX + n] = a[idYTmp * K + idXTmp];
            }
        }

#pragma unroll
        for (int n = 0; n < BLOCK_SIZE_N; n += DIVIDE_N)
        {
#pragma unroll
            for (int m = 0; m < BLOCK_SIZE_M; m += DIVIDE_M)
            {
                int idXTmp = idX + n;
                int idYTmp = tidY + m;
                if (idYTmp < K && idXTmp < N && tidY + m < SUBK)
                    bs[0][tidY + m][tidX + n] = b[idYTmp * N + idXTmp];
            }
        }

        int iBuffer = 0;

        for (int i = 0; i < K;)
        {
            __syncthreads();
            for (int j = 0; j < SUBK; ++j)
            {
                T ar;
                T br;
#pragma unroll
                for (int m = 0; m < NUM_PER_THREAD_M; ++m)
                {
                    ar = as[iBuffer][(tidY + m * DIVIDE_M)][j];
#pragma unroll
                    for (int n = 0; n < NUM_PER_THREAD_N; ++n)
                    {
                        br = bs[iBuffer][j][(tidX + n * DIVIDE_N)];
                        if (i + j < K)
                        {
                            cTmp[m][n] += __half2float(ar * br);
                        }
                    }
                }
            }

            i += SUBK;
            iBuffer ^= 1;
            if (i < K)
            {
#pragma unroll
                for (int m = 0; m < BLOCK_SIZE_M; m += DIVIDE_M)
                {
#pragma unroll
                    for (int n = 0; n < BLOCK_SIZE_N; n += DIVIDE_N)
                    {
                        int idXTmp = i + tidX + n;
                        int idYTmp = idY + m;
                        if (idYTmp < M && idXTmp < K && (tidX + n) < SUBK)
                            as[iBuffer][tidY + m][tidX + n] = a[idYTmp * K + idXTmp];
                    }
                }

#pragma unroll
                for (int n = 0; n < BLOCK_SIZE_N; n += DIVIDE_N)
                {
#pragma unroll
                    for (int m = 0; m < BLOCK_SIZE_M; m += DIVIDE_M)
                    {
                        int idXTmp = idX + n;
                        int idYTmp = i + tidY + m;
                        if (idYTmp < K && idXTmp < N && (tidY + m < SUBK))
                            bs[iBuffer][tidY + m][tidX + n] = b[idYTmp * N + idXTmp];
                    }
                }
            }
        }

#pragma unroll
        for (int m = 0; m < NUM_PER_THREAD_M; ++m)
        {
#pragma unroll
            for (int n = 0; n < NUM_PER_THREAD_N; ++n)
            {
                if ((idY + m * DIVIDE_M) < M && (idX + n * DIVIDE_N) < N)
                    c[(idY + m * DIVIDE_M) * N + idX + n * DIVIDE_N] = cTmp[m][n];
            }
        }
    }
}

template <typename T, typename U>
__global__ void matrix_multiplication5(T *a, T *b, U *c)
{
    int idX = blockIdx.x * (blockDim.x * NUM_PER_THREAD_N * VECTOR_N) + threadIdx.x * VECTOR_N;
    int idY = blockIdx.y * (blockDim.y * NUM_PER_THREAD_M) + threadIdx.y;

    int tidX = threadIdx.x * VECTOR_N;
    int tidY = threadIdx.y;

    __shared__ T as[BLOCK_SIZE_M][SUBK];
    __shared__ T bs[SUBK][BLOCK_SIZE_N];

    {
        // T cTmp = 0;
        U cTmp[NUM_PER_THREAD_M][NUM_PER_THREAD_N][VECTOR_N] = {0};

        for (int i = 0; i < K; i += SUBK)
        {
#pragma unroll
            for (int m = 0; m < BLOCK_SIZE_M; m += DIVIDE_M)
            {
#pragma unroll
                for (int n = 0; n < BLOCK_SIZE_N; n += DIVIDE_N)
                {
                    int idXTmp = i + tidX + n;
                    int idYTmp = idY + m;
                    if (idYTmp < M && idXTmp < K && (tidX + n) < SUBK)
                    {
                        // printf("i: %d, tidX: %d, tidY: %d, idX: %d, idY: %d, tidY + m: %d, tidX * VECTOR_N + n: %d, idYTmp * K: %d, idXTmp * VECTOR_N: %d, idYTmp * K + idXTmp * VECTOR_N: %d\n",
                        //  i, tidX, tidY, idX, idY, tidY + m, tidX * VECTOR_N + n, idYTmp * K, idXTmp * VECTOR_N, idYTmp * K + idXTmp * VECTOR_N);
                        
                        if constexpr (std::is_same<T, half>::value)
                        {
                            HALF2(as[tidY + m][tidX + n]) = HALF2(a[idYTmp * K + idXTmp]);
                            HALF2(as[tidY + m][tidX + n + 2]) = HALF2(a[idYTmp * K + idXTmp + 2]);
                        }
                        else if constexpr (std::is_same<T, float>::value)
                        {
                            FLOAT4(as[tidY + m][tidX + n]) = FLOAT4(a[idYTmp * K + idXTmp]);
                        }
                        
                    }
                        // as[tidY + m][tidX + n] = a[idYTmp * K + idXTmp];
                }
            }

#pragma unroll
            for (int n = 0; n < BLOCK_SIZE_N; n += DIVIDE_N)
            {
#pragma unroll
                for (int m = 0; m < BLOCK_SIZE_M; m += DIVIDE_M)
                {
                    int idXTmp = idX + n;
                    int idYTmp = i + tidY + m;
                    if (idYTmp < K && idXTmp < N && tidY + m < SUBK && (tidX) % 4 == 0)
                    {
                        if constexpr (std::is_same<T, half>::value)
                        {
                            HALF2(bs[tidY + m][tidX + n]) = HALF2(b[idYTmp * N + idXTmp]);
                            HALF2(bs[tidY + m][tidX + n + 2]) = HALF2(b[idYTmp * N + idXTmp + 2]);
                        }
                        else if constexpr (std::is_same<T, float>::value)
                        {
                            FLOAT4(bs[tidY + m][tidX + n]) = FLOAT4(b[idYTmp * N + idXTmp]);
                        }
                    }
                        // bs[tidY + m][tidX + n] = b[idYTmp * N + idXTmp];
                }
            }

            __syncthreads();
            for (int j = 0; j < SUBK; ++j)
            {
                // T ar;
                // T br;
#pragma unroll
                for (int m = 0; m < NUM_PER_THREAD_M; ++m)
                {
                    T ar = as[(tidY + m * DIVIDE_M)][j];
#pragma unroll
                    for (int n = 0; n < NUM_PER_THREAD_N; ++n)
                    {
                        T br0 = bs[j][(tidX + 0 + n * DIVIDE_N)];
                        T br1 = bs[j][(tidX + 1 + n * DIVIDE_N)];
                        T br2 = bs[j][(tidX + 2 + n * DIVIDE_N)];
                        T br3 = bs[j][(tidX + 3 + n * DIVIDE_N)];
                        if (i + j < K)
                        {
                            cTmp[m][n][0] += __half2float(ar * br0);
                            cTmp[m][n][1] += __half2float(ar * br1);
                            cTmp[m][n][2] += __half2float(ar * br2);
                            cTmp[m][n][3] += __half2float(ar * br3);
                        }
                    }
                }
            }
            // printf("33i: %d, tidX: %d, tidY: %d, idX: %d, idY: %d\n", i, tidX, tidY, idX, idY);
            __syncthreads();
        }

#pragma unroll
        for (int m = 0; m < NUM_PER_THREAD_M; ++m)
        {
#pragma unroll
            for (int n = 0; n < NUM_PER_THREAD_N; ++n)
            {
                if ((idY + m * DIVIDE_M) < M && (idX + n * DIVIDE_N) < N)
                {
                    c[(idY + m * DIVIDE_M) * N + idX + 0 + n * DIVIDE_N] = cTmp[m][n][0];
                    c[(idY + m * DIVIDE_M) * N + idX + 1 + n * DIVIDE_N] = cTmp[m][n][1];
                    c[(idY + m * DIVIDE_M) * N + idX + 2 + n * DIVIDE_N] = cTmp[m][n][2];
                    c[(idY + m * DIVIDE_M) * N + idX + 3 + n * DIVIDE_N] = cTmp[m][n][3];
                }
                    // c[(idY + m * DIVIDE_M) * N + idX + n * DIVIDE_N] = cTmp[m][n];
            }
        }
    }
}


template <typename T, typename U>
void print_output(T *a, T *b, U *c)
{
    for (int i = 0; i < M * K; ++i)
    {
        if (i % K == 0)
        {
            std::cout << std::endl;
        }
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < K * N; ++i)
    {
        if (i % N == 0)
        {
            std::cout << std::endl;
        }
        std::cout << b[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < M * N; ++i)
    {
        if (i % N == 0)
        {
            std::cout << std::endl;
        }
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T, typename U>
void matrix_multiplication_cpu(const T *a, const T *b, U *c)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0;
            for (int k = 0; k < K; ++k)
            {
                sum +=  __half2float(a[i * K + k] * b[k * N + j]);
            }
            c[i * N + j] = __float2half(sum);
        }
    }
}

template <typename T, typename U>
void matrix_multiplication_cublas(const T *a, const T *b, U *c)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0;
    const float beta = 0.0;

    // cuBLAS expects column-major order, so we need to transpose the matrices
    cublasStatus_t status = cublasGemmEx(handle,
                                         CUBLAS_OP_N, CUBLAS_OP_N,
                                         N, M, K,
                                         &alpha,
                                         b, CUDA_R_16F, N,
                                         a, CUDA_R_16F, K,
                                         &beta,
                                         c, CUDA_R_32F, N,
                                         CUDA_R_32F,
                                         CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS GEMM failed" << std::endl;
    }

    cublasDestroy(handle);
}


// reference: https://github.com/xlite-dev/hgemm-tensorcores-mma/blob/main/kernels/hgemm/wmma/hgemm_wmma.cu
template <typename T, typename U, const int WMMA_M=16, const int WMMA_N=16, const int WMMA_K=16>
__global__ void matrix_multiplication_wmma(const T *a, const T *b, U *c)
{
    const int NUM_K_TILES = (K + WMMA_K - 1) / WMMA_K;

    const int load_gmem_a_m = blockIdx.y * WMMA_M;
    const int load_gmem_b_n = blockIdx.x * WMMA_N;

    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

    #pragma unroll
    for (int k = 0; k < NUM_K_TILES; ++k)
    {
        const int load_gmem_a_k = k * WMMA_K;
        const int load_gmem_b_k = k * WMMA_K;

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_M, T, wmma::row_major> A_frag;
        wmma::load_matrix_sync(A_frag, a + load_gmem_a_m * K + load_gmem_a_k, K);

        wmma::fragment<wmma::matrix_b, WMMA_K, WMMA_N, WMMA_N, T, wmma::row_major> B_frag;
        wmma::load_matrix_sync(B_frag, b + load_gmem_b_k * N + load_gmem_b_n, N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

        __syncthreads();
    }

    // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, U> C_out_frag;
    // for (int i = 0; i < C_out_frag.num_elements; i++) {
    //     C_out_frag.x[i] = __float2half(C_frag.x[i]);
    // }
    wmma::store_matrix_sync(c + load_gmem_a_m * N + load_gmem_b_n, C_frag, N, 
        wmma::mem_row_major);
}


int main()
{
    // Allocate space for host copies of a, b
    thrust::host_vector<TYPE> a(M * K);
    thrust::host_vector<TYPE> b(K * N);
    thrust::host_vector<TYPE_MATC> c(M * N);
    thrust::host_vector<TYPE_MATC> c_cpu(M * N);

    // Randomly initialize a and b
    std::random_device rd;
    std::mt19937 gen(rd());
    // std::uniform_int_distribution<TYPE> dis(0, 9);
    std::uniform_real_distribution<float> dis(0, 1.0);

    for (int i = 0; i < M * K; ++i)
    {
        a[i] = __float2half(dis(gen));
        // a[i] = __float2half(1.0);

        // a[i] = 1;
        // a[i] = i;
    }


    for (int i = 0; i < K * N; ++i)
    {
        b[i] = __float2half(dis(gen));
        // b[i] = __float2half(1.0);

        // b[i] = 1;
        // b[i] = i;
    }

    // Allocate space for device copies of a, b
    thrust::device_vector<TYPE> d_a = a;
    thrust::device_vector<TYPE> d_b = b;
    thrust::device_vector<TYPE_MATC> d_c(M * N, 0);

    dim3 threads_per_block(BLOCK_SIZE_N, BLOCK_SIZE_M, 1); // x y z
    dim3 no_of_blocks((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, 1);

    warm_up<<<no_of_blocks, threads_per_block>>>();
    matrix_multiplication0<<<no_of_blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()), thrust::raw_pointer_cast(d_c.data()));
    matrix_multiplication1<<<no_of_blocks, threads_per_block>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()), thrust::raw_pointer_cast(d_c.data()));

    dim3 threads_per_block_multi(BLOCK_SIZE_N / NUM_PER_THREAD_N, BLOCK_SIZE_M / NUM_PER_THREAD_M, 1); // x y z
    dim3 no_of_blocks_multi((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M, 1);
    matrix_multiplication2<<<no_of_blocks_multi, threads_per_block_multi>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()), thrust::raw_pointer_cast(d_c.data()));

    matrix_multiplication3<<<no_of_blocks_multi, threads_per_block_multi>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()), thrust::raw_pointer_cast(d_c.data()));
    matrix_multiplication4<<<no_of_blocks_multi, threads_per_block_multi>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()), thrust::raw_pointer_cast(d_c.data()));

    dim3 threads_per_block_multi2(BLOCK_SIZE_N / NUM_PER_THREAD_N / VECTOR_N, BLOCK_SIZE_M / NUM_PER_THREAD_M, 1); // x y z
    matrix_multiplication5<<<no_of_blocks_multi, threads_per_block_multi2>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()), thrust::raw_pointer_cast(d_c.data()));

    // Perform CUTLASS matrix multiplication
    matrix_multiplication_cublas(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()), thrust::raw_pointer_cast(d_c.data()));

    dim3 no_of_blocks_wmma((N + 16 - 1) / 16, (M + 16 - 1) / 16); 
    dim3 threads_per_block_mma(32, 1);
    matrix_multiplication_wmma<<<no_of_blocks_wmma, threads_per_block_mma>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()), thrust::raw_pointer_cast(d_c.data()));

    thrust::copy(d_c.begin(), d_c.end(), c.begin());

    // Perform CPU matrix multiplication
    matrix_multiplication_cpu(a.data(), b.data(), c_cpu.data());

    // print_output(a.data(), b.data(), c.data());
    // print_output(a.data(), b.data(), c_cpu.data());

    // Verify the results
    bool match = true;
    for (int i = 0; i < M * N; ++i)
    {
        if (fabs(c[i]-c_cpu[i]) / fabs(c_cpu[i]) > 0.001)
        {
            match = false;
            std::cout <<i<< " error "<<fabs(c[i]-c_cpu[i])<<" " << c[i] << " " <<c_cpu[i]<<std::endl;
            break;
        }
    }

    if (match)
        std::cout << "Results match!" << std::endl;
    else
        std::cout << "Results do not match!" << std::endl;

    return 0;
}

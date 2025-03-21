#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id == cudaSuccess) {
        std::cout << "Detected " << deviceCount << " CUDA Capable GPU(s)\n";
    } else {
        std::cerr << "cudaGetDeviceCount returned " << cudaGetErrorString(error_id)
                  << " (" << error_id << ")\n";
        return -1;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        error_id = cudaGetDeviceProperties(&prop, i);

        if (error_id == cudaSuccess) {
            std::cout << "Device " << i << ":\n";
            std::cout << "  Name: " << prop.name << "\n";
            std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
            std::cout << "  Total Global Memory: " << prop.totalGlobalMem << " bytes (" << prop.totalGlobalMem/(1024.0*1024.0*1024.0) << "GB)\n";
            std::cout << "  Clock Rate: " << prop.clockRate << " kHz\n";
            std::cout << "  Memory Clock Rate: " << prop.memoryClockRate << " kHz\n";
            std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
            std::cout << "  L2 Cache Size: " << prop.l2CacheSize << " bytes (" << prop.l2CacheSize/(1024.0*1024.0) << "MB)\n";
            std::cout << "  Multiprocessor Count: " << prop.multiProcessorCount << "\n";
            std::cout << "  Max Threads Per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
            std::cout << "  Max Threads Per Block: " << prop.maxThreadsPerBlock << "\n";
            std::cout << "  Max Blocks Per Multiprocessor: " << prop.maxBlocksPerMultiProcessor << "\n";
            std::cout << "  Max Grid Size (x): " << prop.maxGridSize[0] << "\n";
            std::cout << "  Max Grid Size (y): " << prop.maxGridSize[1] << "\n";
            std::cout << "  Max Grid Size (z): " << prop.maxGridSize[2] << "\n";
            std::cout << "  Shared Memory Per SM (Streaming Multiprocessor): " 
                      << prop.sharedMemPerMultiprocessor 
                      << " bytes (" 
                      << prop.sharedMemPerMultiprocessor/(1024.0) 
                      << " KB)\n";
            std::cout << "  Shared Memory Per Block: " << prop.sharedMemPerBlock 
                      << " bytes (" << prop.sharedMemPerBlock/(1024.0) << " KB)\n";
        } else {
            std::cerr << "cudaGetDeviceProperties returned " << cudaGetErrorString(error_id)
                      << " (" << error_id << ")\n";
            return -1;
        }
    }

    return 0;
}
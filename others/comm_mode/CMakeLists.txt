# 设置要求的最低CMake版本
cmake_minimum_required(VERSION 3.18)

# 定义项目名称，并启用C++和CUDA语言支持
project(CommMode LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# 添加可执行文件目标
# CMake会自动使用nvcc来编译.cu文件
add_executable(${PROJECT_NAME} main.cu)

# (可选，但强烈推荐) 为目标GPU指定计算能力(Compute Capability)
# 这可以优化性能并减少编译时间。请根据你的GPU型号进行修改。
#   Turing (RTX 20 series): 75
#   Ampere (RTX 30 series): 86
#   Hopper (H100): 90
#   Ada Lovelace (RTX 40 series): 89
# set_target_properties(lsu_app PROPERTIES
#     CUDA_ARCHITECTURES "75;86;89"
# )
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "90"
)

# 打印生成的可执行文件名
message(STATUS "Executable will be named: ${PROJECT_NAME}")
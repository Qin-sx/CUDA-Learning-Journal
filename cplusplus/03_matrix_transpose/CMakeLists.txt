cmake_minimum_required(VERSION 3.8)
project(matrix_transpose)

# 启用 CUDA 支持
enable_language(CUDA)

# 获取当前目录下所有的 .cu 文件
file(GLOB SOURCES "*.cu")

# 设置可执行文件的输出目录为 bin 文件夹
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 设置编译选项（如 -lineinfo）
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# 如果需要调试信息，可以添加以下选项
# target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g>)

# 如果需要优化，可以添加以下选项
# target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
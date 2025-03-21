cmake_minimum_required(VERSION 3.18)

# Set project name and enable CUDA language
project(GPU_Info LANGUAGES CXX CUDA)

# Enable C++11 (or higher if needed)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures for RTX 4060 (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Find CUDA package
find_package(CUDA REQUIRED)

# Add executable
add_executable(${PROJECT_NAME} GPU_Info.cu)

# Set CUDA specific properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
)

# If you need to specify include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})

# Link against CUDA libraries if needed
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})

# Set the output directory for the executable to a "bin" folder
# This will place the generated .exe file inside the "bin" directory under the build directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
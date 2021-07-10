cmake_minimum_required(VERSION 3.19)
project(PTXSynthesistCore)

set(CMAKE_CXX_STANDARD 20)

add_library(${PROJECT_NAME}
        include/ptx_generator.h
        src/ptx_generator.cpp)

find_package(CUDA REQUIRED)
find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVRTC_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)

target_include_directories(${PROJECT_NAME}
        PUBLIC include
        PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES} ${CUDA_NVRTC_LIB})
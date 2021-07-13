cmake_minimum_required(VERSION 3.19)
project(PTXSynthesistCore)

set(CMAKE_CXX_STANDARD 20)

include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)

CPMAddPackage("gh:pybind/pybind11@2.6.2")

add_library(${PROJECT_NAME}
        include/ptx_generator.h
        src/ptx_generator.cpp

        include/ptx_executor.h
        src/ptx_executor.cpp

        include/ptx_interpreter.h
        src/ptx_interpreter.cpp

        include/ptx_code.h include/kernel_param.h src/kernel_param.cpp include/string_processing.h src/string_processing.cpp)

find_package(CUDA REQUIRED)
find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVRTC_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)
find_library(CUDA_LIB libcuda cuda HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)

target_include_directories(${PROJECT_NAME}
        PUBLIC include
        PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${CUDA_LIBRARIES}
        ${CUDA_NVRTC_LIB}
        ${CUDA_LIB}
        pybind11::embed)
#include "ptx_executor.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include <pybind11/embed.h>

namespace py = pybind11;

void throw_on_error(CUresult status)
{
    if (status != CUDA_SUCCESS)
    {
        throw std::runtime_error("Error!");
    }
}

struct PTXExecutorImpl
{
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
};

PTXExecutor::PTXExecutor()
    : impl(new PTXExecutorImpl())
{
  try {
    throw_on_error(cuInit(0));

    int device_count = 0;
    throw_on_error(cuDeviceGetCount(&device_count));

    if (device_count == 0)
    {
      throw std::runtime_error("No GPU available");
    }

    throw_on_error(cuDeviceGet(&impl->device, 0));
    throw_on_error(cuCtxCreate(&impl->context, 0, impl->device));
  }
  catch(...) {
    impl.reset();
  }
}

PTXExecutor::~PTXExecutor()
{
    if (impl)
    {
      throw_on_error(cuCtxDestroy (impl->context));
    }
}

std::vector<float> PTXExecutor::execute(
        int iterations,
        void **kernel_args,
        unsigned int threads_in_block,
        unsigned int blocks_in_grid,
        const char *code)
{
    py::scoped_interpreter guard{};

    py::exec(R"(
          kwargs = dict(name="World", number=42)
          message = "Hello, {name}! The answer is {number}".format(**kwargs)
          print(message)
      )");

    std::vector<float> measurements(iterations, 0.0f);

    if (!impl)
    {
      measurements.resize(1);
      measurements[0] = 0.0f;

      return measurements;
    }

    // TODO Test
    int n = threads_in_block * blocks_in_grid;

    int *x, *y, *result;
    cudaMalloc(&x, sizeof(int) * n);
    cudaMalloc(&y, sizeof(int) * n);
    cudaMalloc(&result, sizeof(int) * n);

    kernel_args[0] = &n;
    kernel_args[1] = &x;
    kernel_args[2] = &y;
    kernel_args[3] = &result;

    CUevent begin, end;
    throw_on_error(cuEventCreate(&begin, 0));
    throw_on_error(cuEventCreate(&end, 0));

    throw_on_error(cuModuleLoadDataEx(&impl->module, code, 0, nullptr, nullptr));
    throw_on_error(cuModuleGetFunction(&impl->kernel, impl->module, "kernel"));


    for (int iteration = 0; iteration < iterations; iteration++)
    {
        throw_on_error(cuEventRecord(begin, 0));

        // void *kernel_args[] = { &a, &d_x, &d_y, &d_out, &n };
        throw_on_error(cuLaunchKernel(impl->kernel,
                // gridDim
                                      blocks_in_grid,
                                      1,
                                      1,

                // blockDim
                                      threads_in_block,
                                      1,
                                      1,

                // Shmem
                                      0,

                // Stream
                                      0,

                // Params
                                      kernel_args,
                                      nullptr));

        throw_on_error(cuEventRecord(end, 0));
        throw_on_error(cuEventSynchronize(end));

        float ms {};
        cuEventElapsedTime(&ms, begin, end);

        measurements[iteration] = ms;
    }

    throw_on_error(cuModuleUnload(impl->module));

    throw_on_error(cuEventDestroy(end));
    throw_on_error(cuEventDestroy(begin));

    // TODO Test
    cudaFree(x);
    cudaFree(y);
    cudaFree(result);

    return measurements;
}

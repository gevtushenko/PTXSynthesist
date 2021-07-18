#include "ptx_executor.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <iostream>
#include <map>
#include <set>

#include <pybind11/embed.h>

namespace py = pybind11;

float median(int begin, int end, const std::vector<float> &sorted_list)
{
  int count = end - begin;

  if (count % 2)
  {
    return sorted_list.at(count / 2 + begin);
  }
  else
  {
    float right = sorted_list.at(count / 2 + begin);
    float left = sorted_list.at(count / 2 - 1 + begin);
    return (right + left) / 2.0;
  }
}

Measurement::Measurement(const std::string &name, std::vector<float> &&elapsed_times)
  : name(name)
  , elapsed_times(elapsed_times)
{
  std::vector<float> sorted_times(elapsed_times);
  std::sort(sorted_times.begin(), sorted_times.end());

  min_time = sorted_times.front();
  median_time = median(0, sorted_times.size(), sorted_times);
  max_time = sorted_times.back();
}

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

bool is_list(const py::handle &obj)
{
  return py::isinstance<py::list>(obj);
}

bool is_integer(const py::handle &obj)
{
  return py::isinstance<py::int_>(obj);
}

class IndependentScalars
{
private:
  py::dict locals;
  std::map<std::string, int> independent_scalar_params;

public:
  explicit IndependentScalars(const std::vector<KernelParameter> &params)
  {
    // TODO parse AST + build a graph

    /*
     *  import ast
     *  code = "temperature*x"
     *  st = ast.parse(code)
     *  for node in ast.walk(st):
     *  if type(node) is ast.Name:
     *    print(node.id)
     */

    for (unsigned int step = 0; step < params.size(); step++)
    {
      for (const KernelParameter &param : params)
      {
        if (independent_scalar_params.count(param.name()) > 0)
          continue;

        locals["__param__"] = param.initializer();

        try
        {
          py::exec(R"(
          __result__ = eval(__param__)
          )", py::globals(), locals);

          py::handle result = locals["__result__"];

          if (is_integer(result))
          {
            independent_scalar_params[param.name()] = result.cast<int>();
            locals[param.name()] = result;
          }
        }
        catch (...)
        {
          // Suppose that the only reason for this exception could be dependent variables
        }
      }

      if (independent_scalar_params.size() == params.size())
      {
        break;
      }
    }
  }

  [[nodiscard]] const py::dict &get_locals() const
  {
    return locals;
  }

  [[nodiscard]] const std::map<std::string, int> &get_values() const
  {
    return independent_scalar_params;
  }
};

class IndependentLists
{
private:
  py::dict locals;
  std::map<std::string, py::list> lists;
  const std::vector<KernelParameter> &params;

  template <typename ActionType>
  void process(
    std::map<std::string, py::list>::iterator it,
    py::dict &workspace,
    ActionType action)
  {
    if (it == lists.end())
    {
      for (const KernelParameter &param: params)
      {
        if (lists.contains(param.name()))
        {
          continue;
        }

        workspace["__param__"] = param.initializer();

        try
        {
          py::exec(R"(
            __result__ = eval(__param__)
          )", py::globals(), workspace);

          workspace[param.name()] = workspace["__result__"];
        }
        catch(...)
        {
          throw std::runtime_error("Unsupported case");
        }
      }

      action(workspace);
    }
    else
    {
      const std::string &param_name = it->first;
      const py::list &list = it->second;

      std::map<std::string, py::list>::iterator next_it = ++it;

      for (const py::handle &val: list)
      {
        workspace[param_name.c_str()] = val;
        process(next_it, workspace, action);
      }
    }
  }

public:
  explicit IndependentLists(
    const IndependentScalars &scalars,
    const std::vector<KernelParameter> &params)
    : locals(scalars.get_locals())
    , params(params)
  {
    // TODO parse AST + build a graph

    for (const KernelParameter &param : params)
    {
      if (scalars.get_values().count(param.name()) > 0)
      {
        continue;
      }

      locals["__param__"] = param.initializer();

      try
      {
        py::exec(R"(
        __result__ = eval(__param__)
        )", py::globals(), locals);

        py::handle result = locals["__result__"];

        if (is_list(result))
        {
          lists[param.name()] = result.cast<py::list>();
        }
      }
      catch (...)
      {
      }
    }
  }

  template <typename ActionType>
  void process(ActionType action)
  {
    py::dict workspace = locals;

    process(lists.begin(), workspace, action);
  }
};

std::string param_values_to_name(const std::map<std::string, int> &kernel_params_values)
{
  if (kernel_params_values.empty())
    return "";

  std::string result;

  for (auto [name, val]: kernel_params_values)
  {
    result += name + "=" + std::to_string(val) + ",";
  }

  result.pop_back();
  return result;
}

std::vector<Measurement> PTXExecutor::execute(
  const std::vector<KernelParameter> &params,
  const char *code)
{
  py::scoped_interpreter guard{};
  using namespace py::literals;

  std::vector<Measurement> measurements;

  // iterations, block size, grid size
  const unsigned int predefined_params_number = 3;
  const unsigned int params_number = params.size() - predefined_params_number;

  std::map<std::string, int> kernel_params_values;
  std::unique_ptr<void*[]> kernel_params(new void*[params_number]);

  auto locals = py::dict();

  IndependentScalars scalars(params);
  IndependentLists lists(scalars, params);

  throw_on_error(cuModuleLoadDataEx(&impl->module, code, 0, nullptr, nullptr));
  throw_on_error(cuModuleGetFunction(&impl->kernel, impl->module, "kernel"));

  CUevent begin, end;
  throw_on_error(cuEventCreate(&begin, 0));
  throw_on_error(cuEventCreate(&end, 0));

  lists.process([&](const py::dict &values) {
    std::vector<void*> arrays_memory(params_number);

    int array_id = 0;

    for (int param_id = 0; param_id < params_number; param_id++)
    {
      const KernelParameter &param = params[param_id];

      if (!values.contains(param.name()))
      {
        throw std::runtime_error("Something's gone wrong");
      }

      if (param.is_pointer())
      {
        const int array_size = values[param.name()].cast<int>();
        cudaMalloc(&arrays_memory[array_id], array_size);
        kernel_params[param_id] = &arrays_memory[array_id++];
      }
      else
      {
        std::cout << param.name() << ": " << values[param.name()].cast<int>() << std::endl;
        kernel_params_values[param.name()] = values[param.name()].cast<int>();
        kernel_params[param_id] = &kernel_params_values[param.name()];
      }
    }

    const unsigned int iterations = values["iterations"].cast<int>();
    const unsigned int threads_in_block = values["threads_in_block"].cast<int>();
    const unsigned int blocks_in_grid = values["blocks_in_grid"].cast<int>();

    std::vector<float> elapsed_times(iterations);

    for (unsigned int i = 0; i < iterations; i++)
    {
      throw_on_error(cuEventRecord(begin, 0));

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
                                    kernel_params.get(),
                                    nullptr));

      throw_on_error(cuEventRecord(end, 0));
      throw_on_error(cuEventSynchronize(end));

      cuEventElapsedTime(&elapsed_times[i], begin, end);
    }

    for (int i = 0; i < array_id; i++)
    {
      cudaFree(arrays_memory[i]);
    }

    measurements.emplace_back(param_values_to_name(kernel_params_values),
                              std::move(elapsed_times));
  });

  throw_on_error(cuModuleUnload(impl->module));

  throw_on_error(cuEventDestroy(end));
  throw_on_error(cuEventDestroy(begin));

  return measurements;
}

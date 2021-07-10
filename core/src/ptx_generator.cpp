#include "ptx_generator.h"

#include <stdexcept>

#include <nvrtc.h>

void throw_on_error(nvrtcResult result)
{
  if (result != NVRTC_SUCCESS)
  {
    throw std::runtime_error("Error!");
  }
}

PTXGenerator::PTXGenerator()
{

}

PTXGenerator::~PTXGenerator()
{

}

std::optional<PTXCode> PTXGenerator::gen(const char *cuda, const std::vector<const char *> &options)
{
  try {
    nvrtcProgram program;
    throw_on_error(nvrtcCreateProgram(&program, cuda, "main.cu", 0, nullptr, nullptr));

    const int options_num = static_cast<int>(options.size());

    // const char *options[] = { "--gpu-architecture=compute_86", "-lineinfo" };
    throw_on_error(nvrtcCompileProgram(program, options_num, options.data()));

    size_t ptx_size {};
    throw_on_error(nvrtcGetPTXSize(program, &ptx_size));

    std::string ptx_code;
    ptx_code.resize(ptx_size);
    throw_on_error(nvrtcGetPTX(program, ptx_code.data()));
    throw_on_error(nvrtcDestroyProgram(&program));

    return PTXCode(std::move(ptx_code));
  }
  catch (...) {
  }

  return {};
}

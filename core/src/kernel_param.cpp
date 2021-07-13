#include "kernel_param.h"
#include "string_processing.h"

#include <iostream>

KernelParameter::KernelParameter(const std::string &cuda_code_sample)
{
  if (0)
    std::cout << cuda_code_sample << "\n";

  parse_name(cuda_code_sample);
  parse_initializer(cuda_code_sample);
}

void KernelParameter::parse_name(const std::string &cuda_code_sample)
{
  auto it = cuda_code_sample.find("/*");

  if (it == std::string::npos)
  {
    throw std::runtime_error("Can't find initializer in: " + cuda_code_sample);
  }

  std::string_view left_part(cuda_code_sample.data(), it);
  left_part = trim(left_part);

  if (left_part.find('*') != std::string::npos)
  {
    param_is_pointer = true;
  }

  std::size_t name_pos = find_name_pos(left_part);
  param_name = left_part.substr(name_pos, left_part.size() - name_pos);

  if (0)
    std::cout << "'" << param_name << "'" << "\n";
}

void KernelParameter::parse_initializer(const std::string &cuda_code_sample)
{
  auto left_a = cuda_code_sample.find("/*");

  if (left_a == std::string::npos)
  {
    throw std::runtime_error("Can't find initializer in: " + cuda_code_sample);
  }

  auto right_a = cuda_code_sample.find_last_of("*/");

  left_a += 2;
  right_a -= 1;

  std::string_view right_part(cuda_code_sample.data() + left_a, right_a - left_a);
  param_initializer = trim(right_part);
}

bool KernelParameter::is_pointer() const
{
  return param_is_pointer;
}

const char *KernelParameter::name() const
{
  return param_name.c_str();
}

const char *KernelParameter::initializer() const
{
  return param_initializer.c_str();
}

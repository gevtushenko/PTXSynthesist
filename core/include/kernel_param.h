#ifndef PTXSYNTHESIST_KERNEL_PARAM_H
#define PTXSYNTHESIST_KERNEL_PARAM_H

#include <string>

class KernelParameter
{
  bool param_is_pointer {};
  std::string param_name;
  std::string param_initializer;

  void parse_name(const std::string &cuda_code_sample);
  void parse_initializer(const std::string &cuda_code_sample);

public:
  explicit KernelParameter(const std::string &cuda_code_sample);

  bool is_pointer() const;

  const char *name() const;
  const char *initializer() const;
};

#endif // PTXSYNTHESIST_KERNEL_PARAM_H

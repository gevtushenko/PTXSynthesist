#include "ptx_interpreter.h"
#include "string_processing.h"

#include <vector>
#include <iostream>
#include <string_view>

std::string_view extract_kernel_name(std::string_view ptx)
{
  const std::string entry_kw = ".entry ";
  auto entry_pos = ptx.find(entry_kw);

  if (entry_pos == std::string::npos)
  {
    throw std::runtime_error("Can't parse entry");
  }

  auto kernel_name_pos = entry_pos + entry_kw.size();
  auto brace_pos = ptx.find('(', kernel_name_pos);
  return ptx.substr(kernel_name_pos, brace_pos - kernel_name_pos);
}

std::vector<std::string_view> split(const std::string_view str, const char delim = ',')
{
  std::vector<std::string_view> result;

  int index_comma_to_left_of_column = 0;
  int index_comma_to_right_of_column = -1;

  for (int i = 0; i < static_cast<int>(str.size()); i++)
  {
    if (str[i] == delim)
    {
      index_comma_to_left_of_column = index_comma_to_right_of_column;
      index_comma_to_right_of_column = i;
      int index = index_comma_to_left_of_column + 1;
      int length = index_comma_to_right_of_column - index;

      std::string_view column(str.data() + index, length);
      result.push_back(trim(column));
    }
  }

  const std::string_view final_column(str.data() + index_comma_to_right_of_column + 1, str.size() - index_comma_to_right_of_column - 1);
  result.push_back(trim(final_column));
  return result;
}

struct KernelParam
{
  std::string_view type;
  std::string_view name;

  explicit KernelParam(std::string_view param)
  {
    std::vector<std::string_view> columns = split(param, ' ');

    if (columns.size() != 3 || columns[0] != ".param")
    {
      throw std::runtime_error("Can't parse kernel params");
    }

    type = columns[1];
    name = columns[2];
  }
};

std::vector<KernelParam> extract_kernel_params(std::string_view ptx)
{
  const std::string entry_kw = ".entry ";
  auto entry_pos = ptx.find(entry_kw);

  if (entry_pos == std::string::npos)
  {
    throw std::runtime_error("Can't parse entry");
  }

  auto kernel_name_pos = entry_pos + entry_kw.size();
  auto begin_brace_pos = ptx.find('(', kernel_name_pos) + 1;
  auto end_brace_pos = ptx.find(')', kernel_name_pos);

  auto params_view = ptx.substr(begin_brace_pos, end_brace_pos - begin_brace_pos);
  auto param_raw_view = split(params_view);

  std::vector<KernelParam> result;

  for (auto &param: param_raw_view)
  {
    result.push_back(KernelParam(param));
  }

  return result;
}

std::string_view extract_kernel_body(std::string_view ptx)
{
  const std::string entry_kw = ".entry ";
  auto entry_pos = ptx.find(entry_kw);

  if (entry_pos == std::string::npos)
  {
    throw std::runtime_error("Can't parse entry");
  }

  auto begin_brace_pos = ptx.find('{', entry_pos) + 1;
  auto end_brace_pos = ptx.find_last_of('}') - 1;

  auto result = ptx.substr(begin_brace_pos, end_brace_pos - begin_brace_pos);

  return result;
}

void PTXInterpreter::interpret(const std::string &ptx)
{
  std::cout << extract_kernel_name(ptx) << std::endl;
  std::cout << std::endl;

  for (auto param: extract_kernel_params(ptx))
    std::cout << param.type << ": " << param.name << std::endl;
  std::cout << std::endl;

  std::cout << extract_kernel_body(ptx) << std::endl;
}

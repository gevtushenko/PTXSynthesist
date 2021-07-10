//
// Created by evtus on 7/10/2021.
//

#ifndef PTXSYNTHESIST_PTX_GENERATOR_H
#define PTXSYNTHESIST_PTX_GENERATOR_H

#include <optional>
#include <string>
#include <vector>

class PTXCode
{
  std::string code;

public:
  explicit PTXCode(std::string &&code)
      : code(std::move(code))
  { }

  [[nodiscard]] const char* get_ptx() const
  {
    return code.c_str();
  }
};

class PTXGenerator
{
public:
  PTXGenerator();
  ~PTXGenerator();

  [[nodiscard]] std::optional<PTXCode> gen(const char *cuda, const std::vector<const char *> &options);
};

#endif //PTXSYNTHESIST_PTX_GENERATOR_H

//
// Created by evtus on 7/10/2021.
//

#ifndef PTXSYNTHESIST_PTX_GENERATOR_H
#define PTXSYNTHESIST_PTX_GENERATOR_H

#include <optional>
#include <string>
#include <vector>

#include "ptx_code.h"

class PTXGenerator
{
public:
  PTXGenerator();
  ~PTXGenerator();

  [[nodiscard]] std::optional<PTXCode> gen(const char *cuda, const std::vector<const char *> &options);
};

#endif //PTXSYNTHESIST_PTX_GENERATOR_H

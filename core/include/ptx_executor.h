//
// Created by gevtushenko on 7/11/21.
//

#ifndef PTXSYNTHESIST_PTX_EXECUTOR_H
#define PTXSYNTHESIST_PTX_EXECUTOR_H

#include <vector>
#include <memory>

#include "kernel_param.h"

struct PTXExecutorImpl;

class PTXExecutor
{
public:
    PTXExecutor();
    ~PTXExecutor();

    std::vector<float> execute(
      const std::vector<KernelParameter> &params,
      const char *code);

private:
    std::unique_ptr<PTXExecutorImpl> impl;
};

#endif //PTXSYNTHESIST_PTX_EXECUTOR_H

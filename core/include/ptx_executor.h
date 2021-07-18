//
// Created by gevtushenko on 7/11/21.
//

#ifndef PTXSYNTHESIST_PTX_EXECUTOR_H
#define PTXSYNTHESIST_PTX_EXECUTOR_H

#include <vector>
#include <memory>

#include "kernel_param.h"

struct PTXExecutorImpl;

class Measurement
{
  float min_time {};
  float median_time {};
  float max_time {};

  std::vector<float> elapsed_times;

public:
  Measurement(std::vector<float> &&elapsed_times);

  [[nodiscard]] float get_min() const { return min_time; }
  [[nodiscard]] float get_median() const { return median_time; }
  [[nodiscard]] float get_max() const { return max_time; }
  [[nodiscard]] const std::vector<float>& get_elapsed_times() { return elapsed_times; }
};

class PTXExecutor
{
public:
    PTXExecutor();
    ~PTXExecutor();

    std::vector<Measurement> execute(
      const std::vector<KernelParameter> &params,
      const char *code);

private:
    std::unique_ptr<PTXExecutorImpl> impl;
};

#endif //PTXSYNTHESIST_PTX_EXECUTOR_H

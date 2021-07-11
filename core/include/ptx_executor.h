//
// Created by gevtushenko on 7/11/21.
//

#ifndef PTXSYNTHESIST_PTX_EXECUTOR_H
#define PTXSYNTHESIST_PTX_EXECUTOR_H

#include <vector>
#include <memory>

struct PTXExecutorImpl;

class PTXExecutor
{
public:
    PTXExecutor();
    ~PTXExecutor();

    std::vector<float> execute(
            int iterations,
            void **kernel_args,
            unsigned int threads_in_block,
            unsigned int blocks_in_grid,
            const char *code);

private:
    std::unique_ptr<PTXExecutorImpl> impl;
};

#endif //PTXSYNTHESIST_PTX_EXECUTOR_H

#ifndef PTXSYNTHESIST_PTX_INTERPRETER_H
#define PTXSYNTHESIST_PTX_INTERPRETER_H

#include <string>

class PTXInterpreter
{
public:
  void interpret(const std::string &ptx);
};

#endif //PTXSYNTHESIST_PTX_INTERPRETER_H

//
// Created by gevtushenko on 7/11/21.
//

#ifndef PTXSYNTHESIST_PTX_CODE_H
#define PTXSYNTHESIST_PTX_CODE_H

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

#endif //PTXSYNTHESIST_PTX_CODE_H

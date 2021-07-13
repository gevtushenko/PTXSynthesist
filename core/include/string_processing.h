//
// Created by gevtushenko on 7/13/21.
//

#ifndef PTXSYNTHESIST_STRING_PROCESSING_H
#define PTXSYNTHESIST_STRING_PROCESSING_H

#include <string_view>

std::string_view trim(std::string_view str);
std::size_t find_name_pos(std::string_view s);

#endif // PTXSYNTHESIST_STRING_PROCESSING_H

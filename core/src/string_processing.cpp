#include "string_processing.h"

#include <string>

std::string_view trim(std::string_view str)
{
  str.remove_prefix(std::min(str.find_first_not_of("\t\n "), str.size()));

  auto last_space = str.find_last_not_of("\t\n ");

  if (last_space != std::string::npos)
  {
    last_space++;
    str.remove_suffix(str.size() - last_space);
  }

  return str;
}

std::size_t find_name_pos(std::string_view s)
{
  for (long int i = s.size() - 1; i >= 0; i--)
  {
    char c = s[i];

    if (std::isalpha(c) || std::isdigit(c) || c == '_')
    {
      continue;
    }

    if (i == s.size() - 1)
    {
      break;
    }

    return i + 1;
  }

  return std::string::npos;
}

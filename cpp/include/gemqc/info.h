#pragma once
/// \file
/// Build/configuration information of gemQC and its libraries.

#include <map>
#include <string>

namespace gemqc {

/// libint version and compiled-in limits (angular momentum, derivative orders)
std::map<std::string, int> libint_info();

}  // namespace gemqc

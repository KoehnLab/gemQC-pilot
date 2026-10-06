#pragma once
/// \file
/// Periodic table lookup: element symbol <-> nuclear charge Z.

#include <string>

namespace gemqc {

inline constexpr int max_element_z = 118;

/// Z for a symbol (case-insensitive, e.g. "O", "cl"); throws if unknown
int element_z(const std::string& symbol);

/// symbol for Z (1..118); throws if out of range
std::string element_symbol(int z);

}  // namespace gemqc

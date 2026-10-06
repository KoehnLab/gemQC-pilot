#include "gemqc/elements.h"

#include <array>
#include <cctype>
#include <stdexcept>

namespace gemqc {

namespace {
// index = Z (index 0 unused)
const std::array<const char*, max_element_z + 1> symbols = {
    "X",
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

// normalize capitalization: "cL" -> "Cl"
std::string canonical(const std::string& s) {
  std::string r;
  for (size_t i = 0; i < s.size(); ++i)
    r += i == 0 ? std::toupper(static_cast<unsigned char>(s[i]))
                : std::tolower(static_cast<unsigned char>(s[i]));
  return r;
}
}  // namespace

int element_z(const std::string& symbol) {
  const auto sym = canonical(symbol);
  for (int z = 1; z <= max_element_z; ++z)
    if (sym == symbols[z]) return z;
  throw std::invalid_argument("unknown element symbol '" + symbol + "'");
}

std::string element_symbol(int z) {
  if (z < 1 || z > max_element_z)
    throw std::out_of_range("nuclear charge out of range: " + std::to_string(z));
  return symbols[z];
}

}  // namespace gemqc

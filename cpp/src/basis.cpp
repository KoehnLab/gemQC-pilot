#include "gemqc/basis.h"

#include <libint2/config.h>

#include <algorithm>
#include <stdexcept>

#include "gemqc/elements.h"

namespace gemqc {

namespace {
AtomBasis per_atom(const Molecule& mol, const ElementBasis& basis, const std::string& name) {
  AtomBasis out;
  for (size_t ia = 0; ia < mol.natoms(); ++ia) {
    const auto it = basis.find(mol.atom(ia).z);
    if (it == basis.end())
      throw std::invalid_argument("basis '" + name + "' has no functions for " +
                                  element_symbol(mol.atom(ia).z));
    out.push_back(it->second);
  }
  return out;
}
}  // namespace

BasisSet::BasisSet(const Molecule& mol, const ElementBasis& basis, std::string name)
    : BasisSet(mol, per_atom(mol, basis, name), name) {}

BasisSet::BasisSet(const Molecule& mol, const AtomBasis& basis, std::string name)
    : name_(std::move(name)) {
  if (basis.size() != mol.natoms())
    throw std::invalid_argument("basis '" + name_ + "': shells for " +
                                std::to_string(basis.size()) + " atoms, molecule has " +
                                std::to_string(mol.natoms()));
  for (size_t ia = 0; ia < mol.natoms(); ++ia) {
    const auto& atom = mol.atom(ia);
    for (const auto& sd : basis[ia]) {
      if (sd.l > LIBINT_MAX_AM)
        throw std::invalid_argument(
            "angular momentum l=" + std::to_string(sd.l) +
            " exceeds libint limit " + std::to_string(LIBINT_MAX_AM));
      if (sd.exponents.size() != sd.coefficients.size() || sd.exponents.empty())
        throw std::invalid_argument("inconsistent shell definition");
      // libint normalizes the contraction (embeds primitive norms)
      shells_.emplace_back(
          libint2::svector<double>(sd.exponents.begin(), sd.exponents.end()),
          libint2::svector<libint2::Shell::Contraction>{
              {sd.l, sd.pure,
               libint2::svector<double>(sd.coefficients.begin(),
                                        sd.coefficients.end())}},
          atom.r);
      offsets_.push_back(nbf_);
      shell_atom_.push_back(ia);
      nbf_ += shells_.back().size();
    }
  }
}

void BasisSet::set_centers(const Molecule& mol) {
  for (size_t i = 0; i < shells_.size(); ++i) {
    if (shell_atom_[i] >= mol.natoms())
      throw std::invalid_argument("molecule does not match basis set");
    shells_[i].move(mol.atom(shell_atom_[i]).r);
  }
}

int BasisSet::max_l() const {
  int l = 0;
  for (const auto& s : shells_) l = std::max(l, s.contr[0].l);
  return l;
}

size_t BasisSet::max_nprim() const {
  size_t n = 0;
  for (const auto& s : shells_) n = std::max(n, s.nprim());
  return n;
}

std::vector<size_t> BasisSet::bf_to_atom() const {
  std::vector<size_t> map;
  map.reserve(nbf_);
  for (size_t i = 0; i < shells_.size(); ++i)
    map.insert(map.end(), shells_[i].size(), shell_atom_[i]);
  return map;
}

}  // namespace gemqc

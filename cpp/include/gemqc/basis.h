#pragma once
/// \file
/// Atom-centred Gaussian basis set, stored as libint2 shells.

#include <libint2/shell.h>

#include <map>
#include <string>
#include <vector>

#include "gemqc/molecule.h"

namespace gemqc {

/// Segmented contracted shell as read from a basis library
/// (coefficients refer to normalized primitives, as in BSE/most libraries)
struct ShellDef {
  int l;
  bool pure;  ///< spherical (true) or Cartesian (false)
  std::vector<double> exponents;
  std::vector<double> coefficients;
};

/// shell definitions per element (key: nuclear charge Z)
using ElementBasis = std::map<int, std::vector<ShellDef>>;
/// shell definitions per atom (index: atom of the molecule)
using AtomBasis = std::vector<std::vector<ShellDef>>;

class BasisSet {
 public:
  BasisSet(const Molecule& mol, const ElementBasis& basis,
           std::string name = "");
  /// shells per atom (e.g. different basis sets for atoms of one element)
  BasisSet(const Molecule& mol, const AtomBasis& basis, std::string name = "");

  /// move shells to the current atom positions of mol (same atom list!)
  void set_centers(const Molecule& mol);

  const std::string& name() const { return name_; }
  const std::vector<libint2::Shell>& shells() const { return shells_; }
  const libint2::Shell& shell(size_t i) const { return shells_.at(i); }
  size_t nshells() const { return shells_.size(); }
  size_t nbf() const { return nbf_; }
  int max_l() const;
  size_t max_nprim() const;
  const std::vector<size_t>& shell_offsets() const { return offsets_; }  ///< first bf of each shell
  const std::vector<size_t>& shell_to_atom() const { return shell_atom_; }
  std::vector<size_t> bf_to_atom() const;

 private:
  std::string name_;
  std::vector<libint2::Shell> shells_;
  std::vector<size_t> offsets_;
  std::vector<size_t> shell_atom_;
  size_t nbf_ = 0;
};

}  // namespace gemqc

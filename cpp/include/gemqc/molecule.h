#pragma once
/// \file
/// Molecular structure: atoms (nuclear charge + position), total charge, spin.
/// Coordinates are always stored in bohr.

#include <Eigen/Core>
#include <array>
#include <string>
#include <vector>

namespace gemqc {

using Coordinates = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

struct Atom {
  int z;                    ///< nuclear charge
  std::array<double, 3> r;  ///< position (bohr)
};

class Molecule {
 public:
  Molecule() = default;
  explicit Molecule(std::string name) : name_(std::move(name)) {}

  // --- construction / modification ---
  void add_atom(int z, const std::array<double, 3>& r);
  void set_position(size_t i, const std::array<double, 3>& r);
  void set_coordinates(const Coordinates& xyz);  ///< N x 3, bohr
  void set_charge(int q) { charge_ = q; }
  void set_multiplicity(int m);
  void set_name(const std::string& n) { name_ = n; }

  // --- queries ---
  const std::string& name() const { return name_; }
  size_t natoms() const { return atoms_.size(); }
  const Atom& atom(size_t i) const { return atoms_.at(i); }
  const std::vector<Atom>& atoms() const { return atoms_; }
  std::vector<int> atomic_numbers() const;
  Coordinates coordinates() const;  ///< N x 3, bohr
  int charge() const { return charge_; }
  int multiplicity() const { return multiplicity_; }
  int n_electrons() const;
  double nuclear_repulsion() const;  ///< E_nuc in hartree

  /// throws std::invalid_argument if charge/multiplicity are inconsistent
  void validate() const;

 private:
  std::string name_;
  std::vector<Atom> atoms_;
  int charge_ = 0;
  int multiplicity_ = 1;  // 2S+1
};

}  // namespace gemqc

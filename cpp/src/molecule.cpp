#include "gemqc/molecule.h"

#include <cmath>
#include <stdexcept>

#include "gemqc/elements.h"

namespace gemqc {

void Molecule::add_atom(int z, const std::array<double, 3>& r) {
  if (z < 1 || z > max_element_z)
    throw std::invalid_argument("invalid nuclear charge " + std::to_string(z));
  atoms_.push_back({z, r});
}

void Molecule::set_position(size_t i, const std::array<double, 3>& r) {
  atoms_.at(i).r = r;
}

void Molecule::set_coordinates(const Coordinates& xyz) {
  if (static_cast<size_t>(xyz.rows()) != atoms_.size())
    throw std::invalid_argument("coordinate array must have natoms rows");
  for (size_t i = 0; i < atoms_.size(); ++i)
    atoms_[i].r = {xyz(i, 0), xyz(i, 1), xyz(i, 2)};
}

void Molecule::set_multiplicity(int m) {
  if (m < 1) throw std::invalid_argument("multiplicity must be >= 1");
  multiplicity_ = m;
}

std::vector<int> Molecule::atomic_numbers() const {
  std::vector<int> z;
  for (const auto& a : atoms_) z.push_back(a.z);
  return z;
}

Coordinates Molecule::coordinates() const {
  Coordinates xyz(atoms_.size(), 3);
  for (size_t i = 0; i < atoms_.size(); ++i)
    for (int k = 0; k < 3; ++k) xyz(i, k) = atoms_[i].r[k];
  return xyz;
}

int Molecule::n_electrons() const {
  int n = -charge_;
  for (const auto& a : atoms_) n += a.z;
  return n;
}

double Molecule::nuclear_repulsion() const {
  double e = 0.0;
  for (size_t i = 0; i < atoms_.size(); ++i)
    for (size_t j = 0; j < i; ++j) {
      const auto& a = atoms_[i].r;
      const auto& b = atoms_[j].r;
      const double r = std::sqrt((a[0] - b[0]) * (a[0] - b[0]) +
                                 (a[1] - b[1]) * (a[1] - b[1]) +
                                 (a[2] - b[2]) * (a[2] - b[2]));
      if (r < 1e-8)
        throw std::runtime_error("atoms " + std::to_string(j) + " and " +
                                 std::to_string(i) + " coincide");
      e += atoms_[i].z * atoms_[j].z / r;
    }
  return e;
}

void Molecule::validate() const {
  const int ne = n_electrons();
  if (ne < 0) throw std::invalid_argument("negative number of electrons");
  // number of unpaired electrons (m-1) must have parity of n_electrons
  if ((multiplicity_ - 1) % 2 != ne % 2 || multiplicity_ - 1 > ne)
    throw std::invalid_argument(
        "multiplicity " + std::to_string(multiplicity_) +
        " impossible for " + std::to_string(ne) + " electrons");
}

}  // namespace gemqc

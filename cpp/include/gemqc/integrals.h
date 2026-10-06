#pragma once
/// \file
/// One-electron integrals over a BasisSet (libint2 engines).
/// Function ordering within a shell follows libint (spherical: m = -l..l).

#include <Eigen/Core>
#include <array>

#include "gemqc/basis.h"
#include "gemqc/molecule.h"

namespace gemqc {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// initialize libint (idempotent; called automatically by integral routines)
void init_libint();

Matrix overlap(const BasisSet& basis);                                   ///< S
Matrix kinetic(const BasisSet& basis);                                   ///< T
Matrix nuclear_attraction(const BasisSet& basis, const Molecule& mol);  ///< V (<0)
/// <mu| r_k - O_k |nu>, k = x,y,z (electronic position, no charge sign)
std::array<Matrix, 3> dipole(const BasisSet& basis,
                             const std::array<double, 3>& origin = {0, 0, 0});
/// <mu| (r_a - O_a)(r_b - O_b) |nu> for ab = xx, xy, xz, yy, yz, zz (second moments)
std::array<Matrix, 6> second_moments(const BasisSet& basis,
                                     const std::array<double, 3>& origin = {0, 0, 0});

/// mixed-basis versions: rectangular matrices <mu in b1| op |nu in b2>
Matrix overlap(const BasisSet& b1, const BasisSet& b2);
Matrix kinetic(const BasisSet& b1, const BasisSet& b2);
Matrix nuclear_attraction(const BasisSet& b1, const BasisSet& b2,
                          const Molecule& mol);
std::array<Matrix, 3> dipole(const BasisSet& b1, const BasisSet& b2,
                             const std::array<double, 3>& origin = {0, 0, 0});

}  // namespace gemqc

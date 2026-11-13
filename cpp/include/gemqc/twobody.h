#pragma once
/// \file
/// Generic two-electron integrals over an operator g(r12) (Coulomb, Slater-type geminal,
/// Yukawa, contracted Gaussian geminals) for 2-, 3- and 4-centre classes with mixed basis
/// sets (orbital, CABS, fitting).
/// Building blocks of the F12 integrals; chemists' notation (12|g|34).

#include <libint2/engine.h>

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "gemqc/basis.h"
#include "gemqc/integrals.h"

namespace gemqc {

/// two-electron operator scale * g(r12)
struct TwoBodyOperator {
  enum class Kind {
    coulomb,        ///< 1/r12
    stg,            ///< exp(-zeta r12)
    stg_x_coulomb,  ///< exp(-zeta r12)/r12
    cgtg,           ///< g(r12) = sum_i c_i exp(-a_i r12^2)
    cgtg_x_coulomb, ///< g(r12)/r12
    delcgtg2,       ///< (grad_1 g)^2 = sum_ij 4 a_i a_j c_i c_j r12^2 exp(-(a_i+a_j) r12^2)
    /// (grad_1 g) . (grad_1 g2) = sum_ij 4 a_i b_j c_i d_j r12^2 exp(-(a_i+b_j) r12^2)
    /// (needs a libint with Operator::delcgtga_delcgtgb, see has_delcgtgab)
    delcgtga_delcgtgb,
  };
  Kind kind = Kind::coulomb;
  double zeta = 0.0;   ///< Slater exponent (stg, stg_x_coulomb)
  double scale = 1.0;  ///< factor applied to all integrals
  std::vector<std::pair<double, double>> gtg;   ///< (a_i, c_i) of g (cgtg kinds)
  std::vector<std::pair<double, double>> gtg2;  ///< (b_j, d_j) of g2 (delcgtga_delcgtgb)

  /// kind from its name: "coulomb", "stg", "stg_x_coulomb", "cgtg", "cgtg_x_coulomb",
  /// "delcgtg2" or "delcgtga_delcgtgb"
  static Kind kind_from(const std::string& name);
  std::string name() const;
  /// true if libint provides Operator::delcgtga_delcgtgb (compile time)
  static bool has_delcgtgab();
};

/// libint engine of an operator for shells up to max_l, max_nprim; the braket must be
/// given here if max_l exceeds the 4-centre limit (fitting functions in 2-/3-centre
/// integrals), else it may be set later
libint2::Engine two_body_engine(const TwoBodyOperator& op, size_t max_nprim, int max_l,
                                int deriv = 0,
                                libint2::BraKet braket = libint2::BraKet::invalid);

/// (s1 s2|g|s3 s4) into e.results() with precomputed shell-pair data (may be null),
/// dispatched on the operator kind (unscaled)
void compute_quartet(libint2::Engine& e, const TwoBodyOperator& op, const libint2::Shell& s1,
                     const libint2::Shell& s2, const libint2::Shell& s3,
                     const libint2::Shell& s4, const libint2::ShellPair* p12 = nullptr,
                     const libint2::ShellPair* p34 = nullptr);

/// Pair integrals K^ij(nu, si) = sum_{mu, la in bo} (mu nu|g|la si) C_{mu i} C_{la j} for
/// nu in b1, si in b2 and the given orbital pairs (i, j) (columns of Co over bo): one pass
/// over the quartets (bo b1|bo b2) with Schwarz screening; parallel over the shells of b1
/// (each thread owns its result rows). The occupied orbitals need only the orbital basis
/// bo, so the cost is n_o^2 n_1 n_2 instead of N'^4 for the RI basis; for b1 = b2 (same
/// object) the bra-ket symmetry halves the quartets.
std::vector<Matrix> pair_integrals(const TwoBodyOperator& op, const BasisSet& bo,
                                   const BasisSet& b1, const BasisSet& b2, const Matrix& Co,
                                   const std::vector<std::pair<int, int>>& pairs,
                                   double threshold = 1e-12);

/// MO integrals (i j|g|k l) = sum (mu nu|g|la si) C1_mu,i C2_nu,j C3_la,k C4_si,l with
/// mu, nu, la, si in b1, b2, b3, b4: (n1*n2, n3*n4), row i*n2 + j, column k*n4 + l. One pass
/// over the quartets with Schwarz screening, parallel over the bra shell pairs; the
/// permutational symmetry of equal (basis, coefficients) is used (bra, ket, bra-ket), and
/// only the shells and columns with nonzero coefficients enter (masked orbitals).
/// blocks (optional): only the column blocks (i, j, k, l) listed are needed (column block
/// (column / block_size) % nblocks); the quartets of shells whose columns lie in other
/// blocks are skipped (also through the symmetry images), so the other blocks of the
/// result are incomplete.
Matrix mo_integrals(const TwoBodyOperator& op, const BasisSet& b1, const Matrix& C1,
                    const BasisSet& b2, const Matrix& C2, const BasisSet& b3, const Matrix& C3,
                    const BasisSet& b4, const Matrix& C4, double threshold = 1e-12,
                    const std::vector<std::array<int, 4>>& blocks = {},
                    Eigen::Index block_size = 0, int nblocks = 0);

/// (P|g|Q) over two fitting (or any) basis sets: (n1, n2)
Matrix two_body_2c(const TwoBodyOperator& op, const BasisSet& b1, const BasisSet& b2);

/// (P|g|mu nu): (naux, n1*n2), column mu*n2 + nu
Matrix two_body_3c(const TwoBodyOperator& op, const BasisSet& aux, const BasisSet& b1,
                   const BasisSet& b2);

/// (P|g|k q) = sum_{mu nu} (P|g|mu nu) C1_{mu k} C2_{nu q} over one basis (MO 3-index
/// integrals, no AO 3-index array): (naux, n1*n2), column k*n2 + q
Matrix two_body_3c_mo(const TwoBodyOperator& op, const BasisSet& aux, const BasisSet& basis,
                      const Matrix& C1, const Matrix& C2);

/// (mu nu|g|la si), row-major n1*n2*n3*n4 (small cases only; checked against the memory limit)
std::vector<double> two_body_4c(const TwoBodyOperator& op, const BasisSet& b1,
                                const BasisSet& b2, const BasisSet& b3, const BasisSet& b4);

}  // namespace gemqc

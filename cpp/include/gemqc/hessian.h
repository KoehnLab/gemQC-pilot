#pragma once
/// \file
/// Orbital Hessian products of closed-shell RHF in the space of occupied-virtual
/// rotations kappa(a, i), computed directly in the AO basis (one Fock build for
/// a batch of trial vectors of any type and irrep):
///   singlet (A+B) v = de v + C^T (2 J[X+X^T] - K[X+X^T]) C    (RHF -> RHF, real)
///   triplet (A+B) v = de v - C^T K[X+X^T] C                   (RHF -> UHF, real)
///   complex (A-B) v = de v - C^T K[X-X^T] C                   (imaginary rotations)
/// with X = C K(v) C^T and de v = F_vv K - K F_oo. The energy Hessian of real
/// singlet rotations is 4(A+B), the gradient 4 F_ai.

#include <utility>
#include <vector>

#include "gemqc/fock.h"

namespace gemqc {

using Pairs = std::vector<std::pair<Eigen::Index, Eigen::Index>>;  ///< (virtual a, occupied i)

enum class HessianType { singlet = 0, triplet = 1, complex = 2 };

/// rotations (a, i) with irrep(a) x irrep(i) = irrep (characters empty: C1)
Pairs rotation_pairs(const std::vector<bool>& occ, const std::vector<int>& irreps,
                     const std::vector<std::vector<double>>& characters, int irrep);

/// nmo x nmo matrix with v on the (a, i) positions; and back
Matrix pairs_to_matrix(const Pairs& pairs, const Eigen::VectorXd& v, Eigen::Index nmo);
Eigen::VectorXd matrix_to_pairs(const Pairs& pairs, const Matrix& M);

struct HessianRequest {
  HessianType type;
  int irrep;             ///< irrep of the rotations (skeleton Fock build)
  const Pairs* pairs;
  Eigen::VectorXd v;
};

/// (A+B) v or (A-B) v for all requests; C: orbitals (nbf x nmo), Fmo: MO Fock
/// matrix; one Fock build (split into batches if memory is short)
std::vector<Eigen::VectorXd> hessian_products(JKBuilder& fb, const Matrix& C, const Matrix& Fmo,
                                              const std::vector<HessianRequest>& requests);

/// UHF (per spin s, v = [v_alpha; v_beta], X_s = C_s K(v_s) C_s^T):
///   real    (A+B) v: de_s v_s + C_s^T (J[Xs_a + Xs_b] - K[Xs_s]) C_s, Xs = X + X^T
///   complex (A-B) v: de_s v_s - C_s^T K[X_s - X_s^T] C_s
/// The energy Hessian of real rotations is 2(A+B), the gradient 2 F^s_ai.
struct UhfHessianRequest {
  bool complex;
  int irrep;
  const Pairs* pairs[2];  ///< alpha, beta rotations
  Eigen::VectorXd v;
};
std::vector<Eigen::VectorXd> uhf_hessian_products(JKBuilder& fb, const Matrix* C,
                                                  const Matrix* Fmo,
                                                  const std::vector<UhfHessianRequest>& requests);

/// energy Hessian of real, totally symmetric rotations times v: RHF (one spin
/// set) 4(A+B) v, UHF (two, v = [v_alpha; v_beta]) 2(A+B) v
Eigen::VectorXd energy_hessian_product(JKBuilder& fb, const std::vector<Matrix>& C,
                                       const std::vector<Matrix>& Fmo,
                                       const std::vector<Pairs>& pairs, const Eigen::VectorXd& v);

}  // namespace gemqc

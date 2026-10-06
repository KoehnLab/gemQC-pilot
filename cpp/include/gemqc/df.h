#pragma once
/// \file
/// Density-fitted J/K builds (robust Coulomb metric):
///   (mu nu|lambda sigma) ~ sum_PQ (mu nu|P) [V^-1]_PQ (Q|lambda sigma),  V_PQ = (P|Q).
/// V^-1 = M M^T from the eigenvectors of V (eigenvalues below a cutoff dropped).
///   J[D]: gamma_P = (P|D), c = V^-1 gamma, J = sum_P (mu nu|P) c_P
///   K[D]: D = sum_k s_k a_k b_k^T (eigen-decomposition or SVD),
///         X^Q_mu k = sum_P M_PQ (P|mu nu) a_nu k, K = sum_Q,k s_k X^Q_k (Y^Q_k)^T
/// The raw 3-index integrals (P|mu nu) are kept in blocks (fitting shell, orbital shell
/// pair): blocks ranked by their Schwarz estimate sqrt((P|P)) sqrt((mu nu|mu nu)); the
/// largest are stored as long as they fit into memory_fraction of the memory limit
/// (all without a limit), the others are recomputed in every build (hybrid storage);
/// blocks below the threshold are skipped.
/// Storage "fitted": instead B^Q_mu nu = sum_P M_PQ (P|mu nu) is stored (lower triangle
/// in shell-pair blocks, pairs with any unscreened block), so that J, K and the MO
/// quantities need no metric step; the raw integrals are then only computed directly
/// (gradient). "auto": fitted if B fits into memory_fraction of the memory limit.

#include <Eigen/Core>
#include <string>
#include <utility>
#include <vector>

#include "gemqc/fock.h"

namespace gemqc {

class DFBuilder : public JKBuilder {
 public:
  /// basis: orbital basis; aux: fitting basis (spherical functions); threshold:
  /// screening of 3-index blocks; memory_fraction: share of the memory limit for
  /// stored 3-index integrals (0: none, fully direct); metric_cutoff: relative cutoff
  /// of V's eigenvalues; storage: "raw" (hybrid raw integrals), "fitted" (B) or "auto"
  DFBuilder(const BasisSet& basis, const BasisSet& aux, double threshold = 1e-12,
            double memory_fraction = 0.5, double metric_cutoff = 1e-10,
            const std::string& storage = "raw");

  JK compute(const std::vector<Matrix>& densities, bool symmetric = true,
             bool with_k = true, const std::vector<int>& irreps = {}) override;
  JK pair_jk(const Matrix& Ca, std::vector<std::pair<int, int>> pairs = {}) const override;
  size_t memory_estimate(size_t ndens, bool with_k, int nthreads) const override;
  bool exact() const override { return false; }
  std::string describe() const override;
  double screened_percent() const override;

  /// MO 3-index quantities B^Q_pu = sum_P M_PQ (P|pu) with p from C (nbf x np) and u from
  /// Ca (nbf x m); row-major (nfit, np*m), column p*m + u. (pu|qv) = sum_Q B^Q_pu B^Q_qv.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mo_three_index(
      const Matrix& C, const Matrix& Ca) const;
  /// Coulomb matrices of fitted densities: J_d[mu, nu] = sum_PQ (mu nu|P) M_PQ b[Q, d],
  /// b: (nfit, nd) coefficients in the metric-vector basis (e.g. columns of B^Q_uv)
  std::vector<Matrix> fitted_coulomb(const Eigen::MatrixXd& b) const;

  /// M with V^-1 = M M^T (naux x nfit): fitted quantities X^Q to the raw basis, M X
  const Eigen::MatrixXd& metric_factor() const { return M_; }
  size_t naux() const { return aux_.nbf(); }           ///< fitting functions
  size_t nfit() const { return size_t(M_.cols()); }    ///< kept metric eigenvectors
  size_t blocks_stored() const { return n_stored_; }   ///< 3-index blocks in memory
  size_t blocks_direct() const { return blocks_.size() - n_stored_; }  ///< recomputed
  size_t blocks_screened() const { return n_screened_; }
  size_t stored_bytes() const { return (store_.size() + B_.size()) * sizeof(double); }
  bool fitted() const { return fitted_; }  ///< B stored (storage "fitted")

 private:
  struct Block {
    unsigned P, s1, s2;  // fitting shell, orbital shells s1 >= s2
    long offset;         // into store_, -1: computed on the fly
  };
  BasisSet basis_, aux_;
  double threshold_;
  Eigen::MatrixXd M_;                 // V^-1 = M M^T (naux x nfit)
  std::vector<Block> blocks_;
  std::vector<std::vector<size_t>> by_aux_;   // block indices per fitting shell
  std::vector<std::vector<size_t>> by_pair_;  // block indices per orbital shell pair
  std::vector<double> store_;
  size_t n_stored_ = 0, n_screened_ = 0;
  size_t max_nprim_ = 0;
  int max_l_ = 0;
  bool fitted_ = false;
  std::vector<std::pair<unsigned, unsigned>> bpairs_;  // shell pairs s1 >= s2 in B
  std::vector<size_t> bpair_off_;  // column offset of a pair's n1 x n2 block in a row of B
  size_t bcols_ = 0;               // columns of B
  std::vector<double> B_;          // B[Q, pair block (x, z)], row-major (nfit x bcols)

  /// B = M^T (P|mu nu) for the pairs with any unscreened block, in chunks of pairs
  void build_fitted();

  /// (a) Y[P, mu k] = sum_nu (P|mu nu) A[nu, k] for all fitting functions P (row-major,
  /// column mu*k_b + k); per P a dense symmetric (P|mu nu) times A
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> half_transform(
      const Eigen::MatrixXd& A) const;
  /// X[Q, mu k] = sum_P M_PQ Y[P, mu k] (from B or from the raw integrals)
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> fitted_transform(
      const Eigen::MatrixXd& A) const;
  /// sum over blocks: gamma[P, d] = sum_mu nu (P|mu nu) D_d[mu, nu]
  Eigen::MatrixXd contract_densities(const std::vector<Matrix>& D) const;
  /// J_d[mu, nu] = sum_P (mu nu|P) c[P, d]
  std::vector<Matrix> expand(const Eigen::MatrixXd& c) const;
};

}  // namespace gemqc

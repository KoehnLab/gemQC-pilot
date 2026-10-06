#pragma once
/// \file
/// Direct J/K build from 4-index electron repulsion integrals (no density fitting).
///   J[D]_pq = sum_rs (pq|rs) D_rs,   K[D]_pq = sum_rs (pr|qs) D_rs
/// Integrals are computed on the fly per shell quartet with 8-fold permutational
/// symmetry, Schwarz screening combined with density-block norms, and OpenMP.

#include <libint2/shell.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gemqc/basis.h"
#include "gemqc/twobody.h"
#include "gemqc/integrals.h"

namespace gemqc {

struct JK {
  std::vector<Matrix> J, K;  ///< one per density (K empty if not requested)
};

/// Common interface of J/K builders: exact 4-index integrals (FockBuilder) or
/// density fitting (DFBuilder). The SCF, Hessian and CASSCF code use only this.
class JKBuilder {
 public:
  virtual ~JKBuilder() = default;
  /// J (and K) for a set of densities (see FockBuilder::compute)
  virtual JK compute(const std::vector<Matrix>& densities, bool symmetric = true,
                     bool with_k = true, const std::vector<int>& irreps = {}) = 0;
  /// J and K of the orbital-pair densities c_u c_v^T (see FockBuilder::pair_jk)
  virtual JK pair_jk(const Matrix& Ca, std::vector<std::pair<int, int>> pairs = {}) const = 0;
  /// point-group symmetry: skeleton build for the exact builder, ignored otherwise
  virtual void set_symmetry(const std::vector<std::vector<int>>& perm,
                            const std::vector<std::vector<double>>& sign,
                            const std::vector<std::vector<double>>& characters = {}) {}
  virtual int symmetry_order() const { return 1; }
  /// memory for one compute() call (bytes)
  virtual size_t memory_estimate(size_t ndens, bool with_k, int nthreads) const = 0;
  /// exact integrals: incremental builds F[D] = F[D_old] + G[dD] save work
  virtual bool exact() const { return true; }
  /// one line on the builder for the output
  virtual std::string describe() const = 0;
  /// share of the work skipped by screening in the last compute() (percent)
  virtual double screened_percent() const { return 0.0; }
};

class FockBuilder : public JKBuilder {
 public:
  /// threshold: neglect quartets with estimated contribution below it;
  /// op: two-electron operator (default Coulomb; others e.g. for F12 pair integrals,
  /// Schwarz factors of the operator itself)
  explicit FockBuilder(const BasisSet& basis, double threshold = 1e-12,
                       const TwoBodyOperator& op = TwoBodyOperator{});

  /// J (and K) for a set of densities; symmetric=false handles general
  /// (non-symmetric) densities, e.g. for response/orbital-rotation methods
  /// (exactly symmetric or antisymmetric ones are detected and cheaper);
  /// irreps: irrep of each density for the skeleton build (empty: all totally
  /// symmetric)
  JK compute(const std::vector<Matrix>& densities, bool symmetric = true,
             bool with_k = true, const std::vector<int>& irreps = {}) override;

  /// J and K of the orbital-pair densities c_u c_v^T (u <= v, columns of Ca):
  /// J_uv(mu, nu) = (mu nu|uv), K_uv(mu, lambda) = (mu u|lambda v), in the order
  /// (0,0), (0,1), ..., (1,1), ... as compute() with these densities would give.
  /// One integral pass over unique bra shell pairs (ket pairs also unique); the first
  /// quarter transformation is done as the integrals come in (cost ~ N^4 m instead of
  /// N^4 m^2, per thread memory ~ N m per bra function pair; threads limited by the
  /// memory limit);
  /// K rows shared between threads are accumulated per thread and added under a
  /// per-shell lock; no point-group symmetry.
  /// pairs: (u, v) to compute, in this order (empty: all u <= v)
  JK pair_jk(const Matrix& Ca, std::vector<std::pair<int, int>> pairs = {}) const override;

  /// Point-group symmetry (skeleton Fock build): operations in the AO basis,
  /// R_g phi_mu = sign[g][mu] phi_perm[g][mu] (first operation: identity),
  /// characters[k][g] of irrep k (optional; needed for densities of irrep k > 0,
  /// i.e. R_g D R_g^T = characters[k][g] D). Only symmetry-unique shell quartets
  /// are computed; the results are then symmetrized with the characters.
  void set_symmetry(const std::vector<std::vector<int>>& perm,
                    const std::vector<std::vector<double>>& sign,
                    const std::vector<std::vector<double>>& characters = {}) override;
  int symmetry_order() const override { return int(ops_perm_.size()); }

  double threshold() const { return threshold_; }
  const TwoBodyOperator& op() const { return op_; }
  const Matrix& schwarz() const { return schwarz_; }  ///< shell-pair sqrt(max|(ab|ab)|)
  size_t significant_pairs() const;
  /// statistics of the last compute() call
  size_t quartets_computed() const { return n_computed_; }
  size_t quartets_screened() const { return n_screened_; }
  int threads_used() const { return threads_used_; }
  /// memory for accumulators of one compute() call (bytes)
  size_t memory_estimate(size_t ndens, bool with_k, int nthreads) const override;
  std::string describe() const override;
  double screened_percent() const override;

 private:
  BasisSet basis_;
  double threshold_;
  TwoBodyOperator op_;
  libint2::Engine engine(double precision) const;  ///< engine of op_
  /// (s1 s2|s3 s4) with precomputed shell-pair data, dispatched on op_ (unscaled)
  void quartet(libint2::Engine& e, const libint2::Shell& s1, const libint2::Shell& s2,
               const libint2::Shell& s3, const libint2::Shell& s4,
               const libint2::ShellPair* p12, const libint2::ShellPair* p34) const;
  Matrix schwarz_;
  // significant shell pairs: for each s1 the list of s2 <= s1 (ascending),
  // with precomputed libint primitive-pair data
  std::vector<std::vector<size_t>> pairs_;
  std::vector<std::vector<std::shared_ptr<libint2::ShellPair>>> pair_data_;
  size_t n_computed_ = 0, n_screened_ = 0;
  // symmetry: AO operations, shell images, bra pairs that are orbit maxima
  std::vector<std::vector<int>> ops_perm_;
  std::vector<std::vector<double>> ops_sign_;
  std::vector<std::vector<size_t>> shell_map_;
  std::vector<std::vector<bool>> pair_is_max_;
  std::vector<std::vector<double>> characters_;
  Matrix symmetrize(const Matrix& M, int irrep) const;
  int threads_used_ = 1;
};

/// Full ERI tensor (pq|rs) in chemists' notation, row-major n^4 (small cases only)
std::vector<double> eri_tensor(const BasisSet& basis);

}  // namespace gemqc

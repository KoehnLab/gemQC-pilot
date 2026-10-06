#include "gemqc/fock.h"

#include <libint2.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#ifdef GEMQC_HAVE_OPENMP
#include <omp.h>
#endif

#include "gemqc/resources.h"

namespace gemqc {

using libint2::BraKet;
using libint2::Operator;

namespace {

libint2::Engine coulomb_engine(const BasisSet& b, double precision) {
  init_libint();
  libint2::Engine e(Operator::coulomb, b.max_nprim(), b.max_l(), 0);
  e.set_precision(precision);
  return e;
}

// max |D| within each shell block, maximized over all densities
Matrix shell_block_norms(const BasisSet& b, const std::vector<Matrix>& ds) {
  const auto ns = b.nshells();
  const auto& off = b.shell_offsets();
  Matrix n = Matrix::Zero(ns, ns);
  for (size_t s1 = 0; s1 < ns; ++s1)
    for (size_t s2 = 0; s2 < ns; ++s2) {
      const auto n1 = b.shell(s1).size(), n2 = b.shell(s2).size();
      for (const auto& d : ds)
        n(s1, s2) = std::max(n(s1, s2),
                             d.block(off[s1], off[s2], n1, n2).cwiseAbs().maxCoeff());
    }
  return n;
}

}  // namespace

libint2::Engine FockBuilder::engine(double precision) const {
  auto e = two_body_engine(op_, basis_.max_nprim(), basis_.max_l());
  e.set_precision(precision);
  return e;
}

void FockBuilder::quartet(libint2::Engine& e, const libint2::Shell& s1, const libint2::Shell& s2,
                          const libint2::Shell& s3, const libint2::Shell& s4,
                          const libint2::ShellPair* p12, const libint2::ShellPair* p34) const {
  compute_quartet(e, op_, s1, s2, s3, s4, p12, p34);
}

FockBuilder::FockBuilder(const BasisSet& basis, double threshold, const TwoBodyOperator& op)
    : basis_(basis), threshold_(threshold), op_(op) {
  const auto ns = basis_.nshells();
  const auto& sh = basis_.shells();

  // Schwarz factors sqrt(max |(ab|ab)|); computed without libint's primitive
  // pre-screening, whose estimate can drop (ab|ab) of distant pairs entirely
  // (e.g. tight s / d on atoms 3 A apart) and would zero their bound
  auto engine = this->engine(0.0);
  const auto& buf = engine.results();
  schwarz_ = Matrix::Zero(ns, ns);
  for (size_t s1 = 0; s1 < ns; ++s1)
    for (size_t s2 = 0; s2 <= s1; ++s2) {
      quartet(engine, sh[s1], sh[s2], sh[s1], sh[s2], nullptr, nullptr);
      double m = 0.0;
      if (buf[0]) {
        const size_t n12 = sh[s1].size() * sh[s2].size();
        for (size_t i = 0; i < n12 * n12; ++i) m = std::max(m, std::abs(buf[0][i]));
      }
      schwarz_(s1, s2) = schwarz_(s2, s1) = std::sqrt(m * std::abs(op_.scale));
    }

  // significant pairs: could contribute above threshold with any partner
  const double smax = schwarz_.maxCoeff();
  const double ln_prec = std::log(std::numeric_limits<double>::epsilon());
  pairs_.resize(ns);
  pair_data_.resize(ns);
  for (size_t s1 = 0; s1 < ns; ++s1)
    for (size_t s2 = 0; s2 <= s1; ++s2)
      if (schwarz_(s1, s2) * smax >= threshold_) {
        pairs_[s1].push_back(s2);
        pair_data_[s1].push_back(
            std::make_shared<libint2::ShellPair>(sh[s1], sh[s2], ln_prec));
      }
}

namespace {
using Quartet = std::array<size_t, 4>;

// canonical form under the 8-fold permutational symmetry of (ab|cd)
Quartet canonical(size_t a, size_t b, size_t c, size_t d) {
  if (a < b) std::swap(a, b);
  if (c < d) std::swap(c, d);
  if (a < c || (a == c && b < d)) std::swap(a, c), std::swap(b, d);
  return {a, b, c, d};
}
}  // namespace

JK FockBuilder::pair_jk(const Matrix& Ca, std::vector<std::pair<int, int>> pairs) const {
  const size_t n = basis_.nbf(), m = size_t(Ca.cols());
  if (size_t(Ca.rows()) != n) throw std::invalid_argument("pair_jk: Ca has wrong row count");
  if (pairs.empty())
    for (int u = 0; u < int(m); ++u)
      for (int w = u; w < int(m); ++w) pairs.emplace_back(u, w);
  for (const auto& [u, w] : pairs)
    if (u < 0 || w < 0 || u >= int(m) || w >= int(m))
      throw std::invalid_argument("pair_jk: orbital index out of range");
  const size_t np = pairs.size();
  JK out;
  out.J.assign(np, Matrix::Zero(n, n));
  out.K.assign(np, Matrix::Zero(n, n));
  using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const RowMat C = Ca;
  const auto& sh = basis_.shells();
  const auto& off = basis_.shell_offsets();
  const size_t ns = sh.size();
  const double smax = schwarz_.maxCoeff();
  const double eng_prec = std::numeric_limits<double>::epsilon();
  // per thread: (mu nu|lambda v) of one bra shell pair and K rows of two shells
  size_t maxf = 0;
  for (const auto& s : sh) maxf = std::max(maxf, s.size());
  const size_t per_thread = (maxf * maxf * n * m + 2 * np * maxf * n) * sizeof(double);
  const size_t result = 2 * np * n * n * sizeof(double);
  int nthreads = resources::threads();
  while (nthreads > 1 && !resources::fits(result + nthreads * per_thread)) --nthreads;
  if (!resources::fits(result + per_thread))
    throw std::runtime_error("pair_jk needs " + resources::format_bytes(result + per_thread) +
                             ", exceeds memory limit");
  std::vector<std::mutex> row_lock(ns);  // K rows of a shell, written by several threads
  // K_p(rows of a shell block) += contribution, under the shell's lock
  auto flush = [&](size_t s, const std::vector<Eigen::MatrixXd>& Kb) {
    std::lock_guard<std::mutex> guard(row_lock[s]);
    for (size_t p = 0; p < np; ++p) out.K[p].middleRows(off[s], Kb[p].rows()) += Kb[p];
  };
#pragma omp parallel num_threads(nthreads)
  {
    auto engine = this->engine(eng_prec);
    const auto& buf = engine.results();
    std::vector<RowMat> Bp;                     // (mu nu|lambda v), functions of a bra pair
    std::vector<Eigen::MatrixXd> K1(np), K2(np);  // K rows of shells s1 and s2
    Eigen::MatrixXd Jm(m, m);
#pragma omp for schedule(dynamic, 1)
    for (size_t s1 = 0; s1 < ns; ++s1) {
      const size_t f1 = off[s1], n1 = sh[s1].size();
      for (auto& k : K1) k.setZero(n1, n);
      for (size_t i12 = 0; i12 < pairs_[s1].size(); ++i12) {  // bra pairs s2 <= s1
        const size_t s2 = pairs_[s1][i12];
        const size_t f2 = off[s2], n2 = sh[s2].size();
        const auto* sp12 = pair_data_[s1][i12].get();
        if (Bp.size() < n1 * n2) Bp.resize(n1 * n2);
        for (size_t k = 0; k < n1 * n2; ++k) Bp[k].setZero(n, m);
        for (size_t s3 = 0; s3 < ns; ++s3) {
          const size_t f3 = off[s3], n3 = sh[s3].size();
          for (size_t i34 = 0; i34 < pairs_[s3].size(); ++i34) {
            const size_t s4 = pairs_[s3][i34];
            if (schwarz_(s1, s2) * schwarz_(s3, s4) < threshold_) continue;
            const size_t f4 = off[s4], n4 = sh[s4].size();
            quartet(engine, sh[s1], sh[s2], sh[s3], sh[s4], sp12, pair_data_[s3][i34].get());
            if (buf[0] == nullptr) continue;
            // first quarter transformation as the integrals come in: sum over sigma
            const double* v = buf[0];
            const bool both = s3 != s4;  // (lambda sigma) and (sigma lambda)
            for (size_t a = 0; a < n1 * n2; ++a)
              for (size_t c = 0; c < n3; ++c)
                for (size_t d = 0; d < n4; ++d, ++v) {
                  Bp[a].row(f3 + c) += *v * C.row(f4 + d);
                  if (both) Bp[a].row(f4 + d) += *v * C.row(f3 + c);
                }
          }
        }
        const bool offdiag = s2 != s1;
        if (offdiag)
          for (auto& k : K2) k.setZero(n2, n);
        for (size_t a = 0; a < n1; ++a)
          for (size_t b = 0; b < n2; ++b) {
            const size_t mu = f1 + a, nu = f2 + b;
            const RowMat& B = Bp[a * n2 + b];  // (mu nu|lambda v)
            Jm.noalias() = C.transpose() * B;  // (mu nu|u v)
            for (size_t p = 0; p < np; ++p) {
                const int u = pairs[p].first, w = pairs[p].second;
                out.J[p](mu, nu) = out.J[p](nu, mu) = Jm(u, w);
                K1[p].row(a) += C(nu, u) * B.col(w).transpose();  // (mu u|lambda w)
                if (offdiag) K2[p].row(b) += C(mu, u) * B.col(w).transpose();  // (nu u|..)
              }
          }
        if (offdiag) flush(s2, K2);
      }
      flush(s1, K1);
    }
  }
  if (op_.scale != 1.0)
    for (size_t p = 0; p < np; ++p) out.J[p] *= op_.scale, out.K[p] *= op_.scale;
  return out;
}

void FockBuilder::set_symmetry(const std::vector<std::vector<int>>& perm,
                               const std::vector<std::vector<double>>& sign,
                               const std::vector<std::vector<double>>& characters) {
  const size_t n = basis_.nbf(), ns = basis_.nshells();
  if (perm.size() != sign.size() || perm.empty())
    throw std::invalid_argument("symmetry: perm and sign lists differ");
  for (const auto& chi : characters)
    if (chi.size() != perm.size())
      throw std::invalid_argument("symmetry: character table has wrong dimension");
  characters_ = characters;
  const auto& off = basis_.shell_offsets();
  std::vector<size_t> shell_of(n);
  for (size_t s = 0; s < ns; ++s)
    for (size_t f = 0; f < basis_.shell(s).size(); ++f) shell_of[off[s] + f] = s;
  ops_perm_ = perm;
  ops_sign_ = sign;
  shell_map_.assign(perm.size(), std::vector<size_t>(ns));
  for (size_t g = 0; g < perm.size(); ++g) {
    if (perm[g].size() != n || sign[g].size() != n)
      throw std::invalid_argument("symmetry: operation has wrong dimension");
    for (size_t s = 0; s < ns; ++s) shell_map_[g][s] = shell_of[perm[g][off[s]]];
  }
  // bra pair (s1 >= s2) can only start a representative quartet if no
  // operation maps it onto a larger pair
  pair_is_max_.assign(ns, {});
  for (size_t s1 = 0; s1 < ns; ++s1)
    for (size_t s2 : pairs_[s1]) {
      bool is_max = true;
      for (const auto& m : shell_map_) {
        size_t a = m[s1], b = m[s2];
        if (a < b) std::swap(a, b);
        if (a > s1 || (a == s1 && b > s2)) { is_max = false; break; }
      }
      pair_is_max_[s1].push_back(is_max);
    }
}

// (1/h) sum_g chi_k(g) R_g M R_g^T: skeleton -> full matrix for a density of
// irrep k (the image quartets contribute R_g J_q[R_g^T D R_g] R_g^T)
Matrix FockBuilder::symmetrize(const Matrix& M, int irrep) const {
  const auto n = M.rows();
  Matrix out = Matrix::Zero(n, n);
  for (size_t g = 0; g < ops_perm_.size(); ++g) {
    const double chi = irrep > 0 ? characters_[irrep][g] : 1.0;
    const auto& p = ops_perm_[g];
    const auto& sg = ops_sign_[g];
    for (Eigen::Index i = 0; i < n; ++i)
      for (Eigen::Index j = 0; j < n; ++j) out(p[i], p[j]) += chi * sg[i] * sg[j] * M(i, j);
  }
  return out / double(ops_perm_.size());
}

size_t FockBuilder::significant_pairs() const {
  size_t n = 0;
  for (const auto& p : pairs_) n += p.size();
  return n;
}

std::string FockBuilder::describe() const {
  const size_t ns = basis_.nshells();
  return "exact 4-index integrals" +
         (op_.kind == TwoBodyOperator::Kind::coulomb ? std::string() : " (" + op_.name() + ")") +
         ", significant shell pairs " +
         std::to_string(significant_pairs()) + " of " + std::to_string(ns * (ns + 1) / 2);
}

double FockBuilder::screened_percent() const {
  const size_t nq = n_computed_ + n_screened_;
  return nq ? 100.0 * double(n_screened_) / double(nq) : 0.0;
}

size_t FockBuilder::memory_estimate(size_t ndens, bool with_k, int nthreads) const {
  const size_t n = basis_.nbf();
  return size_t(nthreads + 1) * ndens * (with_k ? 2 : 1) * n * n * sizeof(double);
}

JK FockBuilder::compute(const std::vector<Matrix>& densities, bool symmetric,
                        bool with_k, const std::vector<int>& irreps) {
  const size_t n = basis_.nbf();
  const size_t nd = densities.size();
  for (const auto& d : densities)
    if (size_t(d.rows()) != n || size_t(d.cols()) != n)
      throw std::invalid_argument("density matrix has wrong dimension");
  if (!irreps.empty() && irreps.size() != nd)
    throw std::invalid_argument("one irrep per density required");
  for (int k : irreps)  // irreps matter only for the skeleton build
    if (ops_perm_.size() > 1 && (k < 0 || (k > 0 && size_t(k) >= characters_.size())))
      throw std::invalid_argument("density irrep " + std::to_string(k) +
                                  " not in the character table");

  // K per density: symmetric (+1) or antisymmetric (-1) densities need only 4
  // of the 8 permutational terms (then (anti)symmetrized), general ones (0) all
  std::vector<int> kind(nd, 1);
  if (!symmetric)
    for (size_t id = 0; id < nd; ++id) {
      const auto& d = densities[id];
      kind[id] = d == d.transpose() ? 1 : (d == -d.transpose() ? -1 : 0);
    }
  // J needs only the symmetric part of D; K of (anti)symmetric D from dk
  std::vector<Matrix> dsym(densities);
  for (size_t id = 0; id < nd; ++id)
    if (kind[id] != 1) dsym[id] = 0.5 * (dsym[id] + dsym[id].transpose()).eval();
  const auto& dk = densities;

  // threads limited by memory for private accumulators
  int nthreads = resources::threads();
  while (nthreads > 1 && !resources::fits(memory_estimate(nd, with_k, nthreads)))
    --nthreads;
  if (!resources::fits(memory_estimate(nd, with_k, 1)))
    throw std::runtime_error("Fock build needs " +
                             resources::format_bytes(memory_estimate(nd, with_k, 1)) +
                             ", exceeds memory limit");
  threads_used_ = nthreads;

  const auto dnorm = shell_block_norms(basis_, dsym);
  const auto knorm = symmetric ? dnorm : shell_block_norms(basis_, dk);
  // full primitive precision: libint's coarse primitive estimate is not a
  // strict bound, so do not loosen it with the screening threshold
  const double eng_prec = std::numeric_limits<double>::epsilon();
  const auto& sh = basis_.shells();
  const auto& off = basis_.shell_offsets();
  const size_t ns = sh.size();
  const bool use_sym = ops_perm_.size() > 1;

  std::vector<std::vector<Matrix>> jacc(nthreads), kacc(nthreads);
  size_t n_comp = 0, n_scr = 0;

#pragma omp parallel num_threads(nthreads) reduction(+ : n_comp, n_scr)
  {
#ifdef GEMQC_HAVE_OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    auto& J = jacc[tid];
    auto& K = kacc[tid];
    J.assign(nd, Matrix::Zero(n, n));
    if (with_k) K.assign(nd, Matrix::Zero(n, n));
    auto engine = this->engine(eng_prec);
    const auto& buf = engine.results();

#pragma omp for schedule(dynamic, 1)
    for (size_t s1 = 0; s1 < ns; ++s1) {
      const size_t f1 = off[s1], n1 = sh[s1].size();
      for (size_t i12 = 0; i12 < pairs_[s1].size(); ++i12) {
        if (use_sym && !pair_is_max_[s1][i12]) continue;  // not an orbit maximum
        const size_t s2 = pairs_[s1][i12];
        const size_t f2 = off[s2], n2 = sh[s2].size();
        const auto* sp12 = pair_data_[s1][i12].get();
        for (size_t s3 = 0; s3 <= s1; ++s3) {
          const size_t f3 = off[s3], n3 = sh[s3].size();
          const size_t s4_max = (s1 == s3) ? s2 : s3;
          for (size_t i34 = 0; i34 < pairs_[s3].size(); ++i34) {
            const size_t s4 = pairs_[s3][i34];
            if (s4 > s4_max) break;
            // screening: Schwarz bound x largest relevant density element
            double dmax = std::max(dnorm(s1, s2), dnorm(s3, s4));
            if (with_k)
              dmax = std::max({dmax, knorm(s1, s3), knorm(s2, s4), knorm(s1, s4),
                               knorm(s2, s3), knorm(s3, s1), knorm(s4, s2),
                               knorm(s4, s1), knorm(s3, s2)});
            if (schwarz_(s1, s2) * schwarz_(s3, s4) * dmax < threshold_) {
              ++n_scr;
              continue;
            }
            // symmetry: compute only the orbit maximum, weighted by the orbit size
            double wsym = 1.0;
            if (use_sym) {
              const Quartet q{s1, s2, s3, s4};
              Quartet images[8];
              int nimg = 0;
              bool rep = true;
              for (const auto& m : shell_map_) {
                const Quartet k = canonical(m[s1], m[s2], m[s3], m[s4]);
                if (k > q) { rep = false; break; }
                if (std::find(images, images + nimg, k) == images + nimg) images[nimg++] = k;
              }
              if (!rep) continue;
              wsym = nimg;
            }
            const size_t f4 = off[s4], n4 = sh[s4].size();
            quartet(engine, sh[s1], sh[s2], sh[s3], sh[s4], sp12, pair_data_[s3][i34].get());
            ++n_comp;
            if (buf[0] == nullptr) continue;

            // degeneracy of the quartet under the 8 permutations
            const double deg = wsym * (s1 == s2 ? 1.0 : 2.0) * (s3 == s4 ? 1.0 : 2.0) *
                               (s1 == s3 ? (s2 == s4 ? 1.0 : 2.0) : 2.0);
            const double* v = buf[0];
            for (size_t a = 0, q = 0; a < n1; ++a) {
              const size_t p1 = f1 + a;
              for (size_t b = 0; b < n2; ++b) {
                const size_t p2 = f2 + b;
                for (size_t c = 0; c < n3; ++c) {
                  const size_t p3 = f3 + c;
                  for (size_t d = 0; d < n4; ++d, ++q) {
                    const size_t p4 = f4 + d;
                    const double val = v[q] * deg;
                    for (size_t id = 0; id < nd; ++id) {
                      const auto& D = dsym[id];
                      if (kind[id] != -1) {
                        J[id](p1, p2) += D(p3, p4) * val;
                        J[id](p3, p4) += D(p1, p2) * val;
                      }
                      if (!with_k) continue;
                      auto& Kd = K[id];
                      if (kind[id] != 0) {  // 4 terms, (anti)symmetrized below
                        const auto& G = dk[id];
                        Kd(p1, p3) += G(p2, p4) * val;
                        Kd(p2, p4) += G(p1, p3) * val;
                        Kd(p1, p4) += G(p2, p3) * val;
                        Kd(p2, p3) += G(p1, p4) * val;
                      } else {  // all 8 permutations explicitly
                        const auto& G = dk[id];
                        const double w = 0.125 * val;
                        Kd(p1, p3) += G(p2, p4) * w;
                        Kd(p3, p1) += G(p4, p2) * w;
                        Kd(p1, p4) += G(p2, p3) * w;
                        Kd(p4, p1) += G(p3, p2) * w;
                        Kd(p2, p3) += G(p1, p4) * w;
                        Kd(p3, p2) += G(p4, p1) * w;
                        Kd(p2, p4) += G(p1, p3) * w;
                        Kd(p4, p2) += G(p3, p1) * w;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  n_computed_ = n_comp;
  n_screened_ = n_scr;

  // reduce thread contributions and apply symmetrization/scaling
  JK res;
  for (size_t id = 0; id < nd; ++id) {
    Matrix J = Matrix::Zero(n, n), K = Matrix::Zero(n, n);
    for (int t = 0; t < nthreads; ++t) {
      J += jacc[t][id];
      if (with_k) K += kacc[t][id];
    }
    // .eval(): avoid Eigen aliasing of M = f(M, M^T)
    J = (0.25 * (J + J.transpose())).eval();
    if (with_k && kind[id] != 0) K = (0.125 * (K + double(kind[id]) * K.transpose())).eval();
    if (use_sym) {  // skeleton -> full matrices
      const int k = irreps.empty() ? 0 : irreps[id];
      J = symmetrize(J, k);
      if (with_k) K = symmetrize(K, k);
    }
    res.J.push_back(op_.scale * J);
    if (with_k) res.K.push_back(op_.scale * K);
  }
  return res;
}

std::vector<double> eri_tensor(const BasisSet& basis) {
  const size_t n = basis.nbf();
  const size_t bytes = n * n * n * n * sizeof(double);
  if (!resources::fits(bytes))
    throw std::runtime_error("ERI tensor needs " + resources::format_bytes(bytes) +
                             ", exceeds memory limit");
  std::vector<double> eri(n * n * n * n, 0.0);
  const auto& sh = basis.shells();
  const auto& off = basis.shell_offsets();
  auto engine = coulomb_engine(basis, std::numeric_limits<double>::epsilon());
  const auto& buf = engine.results();
  for (size_t s1 = 0; s1 < sh.size(); ++s1)
    for (size_t s2 = 0; s2 < sh.size(); ++s2)
      for (size_t s3 = 0; s3 < sh.size(); ++s3)
        for (size_t s4 = 0; s4 < sh.size(); ++s4) {
          engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(sh[s1], sh[s2], sh[s3], sh[s4]);
          if (buf[0] == nullptr) continue;
          size_t q = 0;
          for (size_t a = 0; a < sh[s1].size(); ++a)
            for (size_t b = 0; b < sh[s2].size(); ++b)
              for (size_t c = 0; c < sh[s3].size(); ++c)
                for (size_t d = 0; d < sh[s4].size(); ++d, ++q) {
                  const size_t p = off[s1] + a, r = off[s2] + b, s = off[s3] + c,
                               t = off[s4] + d;
                  eri[((p * n + r) * n + s) * n + t] = buf[0][q];
                }
        }
  return eri;
}

}  // namespace gemqc

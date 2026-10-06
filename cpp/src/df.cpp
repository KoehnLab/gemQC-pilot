#include "gemqc/df.h"

#include <libint2.hpp>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

#include <libint2/config.h>

#ifdef GEMQC_HAVE_OPENMP
#include <omp.h>
#endif

#include "gemqc/resources.h"

namespace gemqc {

using libint2::BraKet;
using libint2::Operator;
using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace {

int thread_id() {
#ifdef GEMQC_HAVE_OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

libint2::Engine engine(BraKet bk, size_t nprim, int l, double precision) {
  init_libint();
  libint2::Engine e(Operator::coulomb, nprim, l, 0, precision, {}, bk);  // l up to the
  e.set(bk);                                                             // 2-/3-centre limit
  return e;
}

/// D = sum_k s_k a_k b_k^T with small terms dropped (a == b for symmetric D)
void factorize(const Matrix& D, bool symmetric, Eigen::MatrixXd& a, Eigen::MatrixXd& b,
               Eigen::VectorXd& s) {
  const double tiny = 1e-14;
  if (symmetric) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (D + D.transpose()));
    const auto& w = es.eigenvalues();
    const double wmax = w.cwiseAbs().maxCoeff();
    std::vector<int> keep;
    for (int k = 0; k < int(w.size()); ++k)
      if (std::abs(w(k)) > tiny * std::max(wmax, 1e-300)) keep.push_back(k);
    a.resize(D.rows(), long(keep.size()));
    s.resize(long(keep.size()));
    for (size_t k = 0; k < keep.size(); ++k) {
      a.col(long(k)) = es.eigenvectors().col(keep[k]);
      s(long(k)) = w(keep[k]);
    }
    b = a;
    return;
  }
  Eigen::BDCSVD<Eigen::MatrixXd> svd(Eigen::MatrixXd(D), Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto& w = svd.singularValues();
  long r = 0;
  while (r < w.size() && w(r) > tiny * std::max(w(0), 1e-300)) ++r;
  a = svd.matrixU().leftCols(r);
  b = svd.matrixV().leftCols(r);
  s = w.head(r);
}

}  // namespace

DFBuilder::DFBuilder(const BasisSet& basis, const BasisSet& aux, double threshold,
                     double memory_fraction, double metric_cutoff, const std::string& storage)
    : basis_(basis), aux_(aux), threshold_(threshold) {
  if (storage != "raw" && storage != "fitted" && storage != "auto")
    throw std::invalid_argument("density fitting: storage must be raw, fitted or auto");
  for (const auto& sh : aux_.shells())
    if (sh.contr[0].l > 1 && !sh.contr[0].pure)
      throw std::invalid_argument("density fitting: the fitting basis must be spherical");
  max_nprim_ = std::max(basis_.max_nprim(), aux_.max_nprim());
  max_l_ = std::max(basis_.max_l(), aux_.max_l());
  const auto& sh = basis_.shells();
  const auto& ash = aux_.shells();
  const auto& aoff = aux_.shell_offsets();
  const size_t ns = sh.size(), na = ash.size(), naux = aux_.nbf();
  const int nt = resources::threads();

  // metric V = (P|Q) and V^-1 = M M^T
  Eigen::MatrixXd V(naux, naux);
  {
    auto e2 = engine(BraKet::xs_xs, max_nprim_, max_l_, std::numeric_limits<double>::epsilon());
    const auto& buf = e2.results();
    for (size_t p = 0; p < na; ++p)
      for (size_t q = 0; q <= p; ++q) {
        e2.compute2<Operator::coulomb, BraKet::xs_xs, 0>(ash[p], libint2::Shell::unit(), ash[q],
                                                         libint2::Shell::unit());
        const size_t np_ = ash[p].size(), nq = ash[q].size();
        for (size_t i = 0; i < np_; ++i)
          for (size_t j = 0; j < nq; ++j) {
            const double v = buf[0] ? buf[0][i * nq + j] : 0.0;
            V(aoff[p] + i, aoff[q] + j) = V(aoff[q] + j, aoff[p] + i) = v;
          }
      }
  }
  Eigen::VectorXd aux_diag(na);  // sqrt(max (P|P)) per fitting shell
  for (size_t p = 0; p < na; ++p)
    aux_diag(p) = std::sqrt(V.diagonal().segment(aoff[p], ash[p].size()).cwiseAbs().maxCoeff());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(V);
  const auto& w = es.eigenvalues();
  std::vector<long> keep;
  for (long k = 0; k < w.size(); ++k)
    if (w(k) > metric_cutoff * w.maxCoeff()) keep.push_back(k);
  M_.resize(naux, long(keep.size()));
  for (size_t k = 0; k < keep.size(); ++k)
    M_.col(long(k)) = es.eigenvectors().col(keep[k]) / std::sqrt(w(keep[k]));

  // Schwarz factors of the orbital shell pairs, sqrt(max |(ab|ab)|)
  Eigen::MatrixXd schwarz = Eigen::MatrixXd::Zero(ns, ns);
  {
    init_libint();
    libint2::Engine e4(Operator::coulomb, basis_.max_nprim(), basis_.max_l(), 0);
    e4.set_precision(0.0);
    const auto& buf = e4.results();
    for (size_t s1 = 0; s1 < ns; ++s1)
      for (size_t s2 = 0; s2 <= s1; ++s2) {
        e4.compute2<Operator::coulomb, BraKet::xx_xx, 0>(sh[s1], sh[s2], sh[s1], sh[s2]);
        double m = 0.0;
        if (buf[0]) {
          const size_t n12 = sh[s1].size() * sh[s2].size();
          for (size_t i = 0; i < n12 * n12; ++i) m = std::max(m, std::abs(buf[0][i]));
        }
        schwarz(s1, s2) = schwarz(s2, s1) = std::sqrt(m);
      }
  }

  // blocks above the threshold, ranked by their estimate; store the largest
  std::vector<std::pair<double, Block>> cand;
  for (size_t s1 = 0; s1 < ns; ++s1)
    for (size_t s2 = 0; s2 <= s1; ++s2)
      for (size_t p = 0; p < na; ++p) {
        const double est = schwarz(s1, s2) * aux_diag(p);
        if (est < threshold_) {
          ++n_screened_;
          continue;
        }
        cand.push_back({est, Block{unsigned(p), unsigned(s1), unsigned(s2), -1}});
      }
  std::stable_sort(cand.begin(), cand.end(),
                   [](const auto& x, const auto& y) { return x.first > y.first; });
  const size_t limit = resources::memory();
  // memory_fraction 0: nothing stored (fully direct); no limit: everything stored
  double budget = memory_fraction <= 0 ? 0.0 : (limit ? memory_fraction * double(limit) : 1e300);
  // fitted storage: B for the shell pairs with any unscreened block; raw integrals direct
  std::vector<char> pair_used(ns * (ns + 1) / 2, 0);
  for (const auto& c : cand) pair_used[size_t(c.second.s1) * (c.second.s1 + 1) / 2 + c.second.s2] = 1;
  for (size_t s1 = 0; s1 < ns; ++s1)
    for (size_t s2 = 0; s2 <= s1; ++s2)
      if (pair_used[s1 * (s1 + 1) / 2 + s2]) {
        bpairs_.emplace_back(unsigned(s1), unsigned(s2));
        bpair_off_.push_back(bcols_);
        bcols_ += sh[s1].size() * sh[s2].size();
      }
  const double bbytes = double(nfit()) * double(bcols_) * sizeof(double);
  if (storage == "fitted" && limit && bbytes > double(limit))
    throw std::runtime_error("density fitting: fitted 3-index integrals need " +
                             resources::format_bytes(size_t(bbytes)) + ", exceed memory limit");
  fitted_ = storage == "fitted" || (storage == "auto" && bbytes <= budget);
  if (fitted_) budget = 0.0;
  size_t used = 0;
  for (auto& [est, b] : cand) {
    const size_t n = ash[b.P].size() * sh[b.s1].size() * sh[b.s2].size();
    if (double((used + n) * sizeof(double)) <= budget) {
      b.offset = 0;  // stored
      used += n;
      ++n_stored_;
    }
    blocks_.push_back(b);
  }
  // store ordered by fitting shell, then shell pair (contiguous per fitting shell)
  std::sort(blocks_.begin(), blocks_.end(), [](const Block& x, const Block& y) {
    return std::tie(x.P, x.s1, x.s2) < std::tie(y.P, y.s1, y.s2);
  });
  used = 0;
  for (auto& b : blocks_)
    if (b.offset >= 0) {
      b.offset = long(used);
      used += ash[b.P].size() * sh[b.s1].size() * sh[b.s2].size();
    }
  store_.assign(used, 0.0);
  by_aux_.assign(na, {});
  by_pair_.assign(ns * (ns + 1) / 2, {});
  for (size_t i = 0; i < blocks_.size(); ++i) {
    const auto& b = blocks_[i];
    by_aux_[b.P].push_back(i);
    by_pair_[size_t(b.s1) * (b.s1 + 1) / 2 + b.s2].push_back(i);
  }
#pragma omp parallel num_threads(nt)
  {
    auto e3 = engine(BraKet::xs_xx, max_nprim_, max_l_, std::numeric_limits<double>::epsilon());
    const auto& buf = e3.results();
#pragma omp for schedule(dynamic, 16)
    for (long i = 0; i < long(blocks_.size()); ++i) {
      const auto& b = blocks_[size_t(i)];
      if (b.offset < 0) continue;
      e3.compute2<Operator::coulomb, BraKet::xs_xx, 0>(ash[b.P], libint2::Shell::unit(),
                                                       sh[b.s1], sh[b.s2]);
      const size_t n = ash[b.P].size() * sh[b.s1].size() * sh[b.s2].size();
      if (buf[0]) std::copy(buf[0], buf[0] + n, store_.begin() + b.offset);
    }
  }
  if (fitted_) build_fitted();
}

void DFBuilder::build_fitted() {
  const auto& sh = basis_.shells();
  const auto& ash = aux_.shells();
  const auto& aoff = aux_.shell_offsets();
  const long naux = long(aux_.nbf()), nf = long(nfit()), npair = long(bpairs_.size());
  B_.assign(size_t(nf) * bcols_, 0.0);
  // chunks of pairs: raw R[P, pair block] (about 64 MB), B[:, chunk] = M^T R
  const size_t chunk_cols = std::max<size_t>(1, (size_t(1) << 23) / size_t(naux));
  using Stride = Eigen::OuterStride<>;
  RowMat R;
  for (long i0 = 0; i0 < npair;) {
    long i1 = i0 + 1;
    while (i1 < npair && bpair_off_[size_t(i1)] + sh[bpairs_[size_t(i1)].first].size() *
                                                   sh[bpairs_[size_t(i1)].second].size() -
                                  bpair_off_[size_t(i0)] <= chunk_cols)
      ++i1;
    const size_t c0 = bpair_off_[size_t(i0)];
    const size_t c1 = i1 < npair ? bpair_off_[size_t(i1)] : bcols_;
    R.setZero(naux, long(c1 - c0));
#pragma omp parallel num_threads(resources::threads())
    {
      auto e3 = engine(BraKet::xs_xx, max_nprim_, max_l_, std::numeric_limits<double>::epsilon());
      const auto& buf = e3.results();
#pragma omp for schedule(dynamic, 1)
      for (long ip = i0; ip < i1; ++ip) {  // columns of pair ip: this thread only
        const auto [s1, s2] = bpairs_[size_t(ip)];
        const size_t n12 = sh[s1].size() * sh[s2].size(), col = bpair_off_[size_t(ip)] - c0;
        for (size_t i : by_pair_[size_t(s1) * (s1 + 1) / 2 + s2]) {
          const auto& b = blocks_[i];
          e3.compute2<Operator::coulomb, BraKet::xs_xx, 0>(ash[b.P], libint2::Shell::unit(),
                                                           sh[b.s1], sh[b.s2]);
          if (!buf[0]) continue;
          for (size_t q = 0; q < ash[b.P].size(); ++q)
            std::copy(buf[0] + q * n12, buf[0] + (q + 1) * n12, &R(long(aoff[b.P] + q), long(col)));
        }
      }
    }
    Eigen::Map<RowMat, 0, Stride>(B_.data() + c0, nf, long(c1 - c0), Stride(long(bcols_))).noalias() =
        M_.transpose() * R;
    i0 = i1;
  }
}

namespace {
/// data of a 3-index block: stored, or computed with the thread's engine (nullptr: zero)
struct BlockSource {
  libint2::Engine e3;
  const double* get(const std::vector<double>& store, long offset, const libint2::Shell& P,
                    const libint2::Shell& a, const libint2::Shell& b) {
    if (offset >= 0) return store.data() + offset;
    e3.compute2<Operator::coulomb, BraKet::xs_xx, 0>(P, libint2::Shell::unit(), a, b);
    return e3.results()[0];
  }
};
}  // namespace

RowMat DFBuilder::half_transform(const Eigen::MatrixXd& A) const {
  const auto& sh = basis_.shells();
  const auto& off = basis_.shell_offsets();
  const auto& ash = aux_.shells();
  const auto& aoff = aux_.shell_offsets();
  const long n = long(basis_.nbf()), kb = A.cols();
  // per fitting function q of shell p: lower triangle of S[mu, nu] = (q|mu nu) (n x n,
  // cache resident) from the blocks s1 >= s2, then Y[q, mu k] = S A (symmetric product);
  // blocks recomputed directly are kept for all q of the shell
  RowMat Y(long(aux_.nbf()), n * kb);  // every row is written below
#pragma omp parallel num_threads(resources::threads())
  {
    BlockSource src{engine(BraKet::xs_xx, max_nprim_, max_l_,
                           std::numeric_limits<double>::epsilon())};
    RowMat S(n, n);                   // lower triangle: (q|mu nu), mu >= nu
    std::vector<const double*> data;  // block data of shell p (nullptr: zero)
    std::vector<double> direct;       // directly computed blocks of shell p
#pragma omp for schedule(dynamic, 1)
    for (long p = 0; p < long(ash.size()); ++p) {  // rows of fitting shell p: this thread only
      const auto& blk = by_aux_[size_t(p)];
      const size_t nP = ash[size_t(p)].size();
      size_t nd = 0;
      for (size_t i : blk)
        if (blocks_[i].offset < 0) nd += nP * sh[blocks_[i].s1].size() * sh[blocks_[i].s2].size();
      direct.resize(nd);
      data.clear();
      nd = 0;
      for (size_t i : blk) {
        const auto& b = blocks_[i];
        const double* v = src.get(store_, b.offset, ash[b.P], sh[b.s1], sh[b.s2]);
        if (v && b.offset < 0) {  // engine buffer: copy before the next computation
          const size_t m = nP * sh[b.s1].size() * sh[b.s2].size();
          std::copy(v, v + m, direct.begin() + long(nd));
          v = direct.data() + nd;
          nd += m;
        }
        data.push_back(v);
      }
      S.triangularView<Eigen::Lower>().setZero();  // same blocks for all q of the shell
      for (size_t q = 0; q < nP; ++q) {
        for (size_t j = 0; j < blk.size(); ++j) {
          if (!data[j]) continue;
          const auto& b = blocks_[blk[j]];
          const size_t n1 = sh[b.s1].size(), n2 = sh[b.s2].size();
          const double* v = data[j] + q * n1 * n2;
          double* s = S.data() + off[b.s1] * size_t(n) + off[b.s2];  // S[f1 + x, f2 + z]
          for (size_t x = 0; x < n1; ++x, s += n, v += n2)
            for (size_t z = 0; z < n2; ++z) s[z] = v[z];
        }
        Eigen::Map<RowMat>(Y.row(long(aoff[size_t(p)] + q)).data(), n, kb).noalias() =
            S.selfadjointView<Eigen::Lower>() * A;
      }
    }
  }
  return Y;
}

RowMat DFBuilder::fitted_transform(const Eigen::MatrixXd& A) const {
  if (!fitted_) return M_.transpose() * half_transform(A);
  const auto& sh = basis_.shells();
  const auto& off = basis_.shell_offsets();
  const long n = long(basis_.nbf()), kb = A.cols(), nf = long(nfit());
  // per Q: lower triangle of B^Q[mu, nu] from the pair blocks (same pairs for all Q, so
  // the rest stays zero), X[Q, mu k] = B^Q A (symmetric product)
  RowMat X(nf, n * kb);
#pragma omp parallel num_threads(resources::threads())
  {
    RowMat S = RowMat::Zero(n, n);
#pragma omp for schedule(dynamic, 4)
    for (long q = 0; q < nf; ++q) {
      const double* v = B_.data() + size_t(q) * bcols_;
      for (const auto& [s1, s2] : bpairs_) {
        const size_t n1 = sh[s1].size(), n2 = sh[s2].size();
        double* s = S.data() + off[s1] * size_t(n) + off[s2];  // S[f1 + x, f2 + z]
        for (size_t x = 0; x < n1; ++x, s += n, v += n2)
          for (size_t z = 0; z < n2; ++z) s[z] = v[z];
      }
      Eigen::Map<RowMat>(X.row(q).data(), n, kb).noalias() = S.selfadjointView<Eigen::Lower>() * A;
    }
  }
  return X;
}

std::vector<Matrix> DFBuilder::fitted_coulomb(const Eigen::MatrixXd& b) const {
  if (!fitted_) return expand(M_ * b);
  // J_d = sum_Q B^Q b[Q, d] on the pair blocks
  const auto& sh = basis_.shells();
  const auto& off = basis_.shell_offsets();
  const long n = long(basis_.nbf());
  const Eigen::MatrixXd j = Eigen::Map<const RowMat>(B_.data(), long(nfit()), long(bcols_)).transpose() * b;
  std::vector<Matrix> J(size_t(b.cols()), Matrix::Zero(n, n));
  for (size_t i = 0; i < bpairs_.size(); ++i) {
    const auto [s1, s2] = bpairs_[i];
    const long n1 = long(sh[s1].size()), n2 = long(sh[s2].size()), f1 = long(off[s1]), f2 = long(off[s2]);
    for (long d = 0; d < b.cols(); ++d)
      for (long x = 0, c = long(bpair_off_[i]); x < n1; ++x)
        for (long z = 0; z < n2; ++z, ++c) J[size_t(d)](f1 + x, f2 + z) = J[size_t(d)](f2 + z, f1 + x) = j(c, d);
  }
  return J;
}

Eigen::MatrixXd DFBuilder::contract_densities(const std::vector<Matrix>& D) const {
  const auto& sh = basis_.shells();
  const auto& off = basis_.shell_offsets();
  const auto& ash = aux_.shells();
  const auto& aoff = aux_.shell_offsets();
  const size_t nd = D.size();
  Eigen::MatrixXd g = Eigen::MatrixXd::Zero(long(aux_.nbf()), long(nd));
#pragma omp parallel num_threads(resources::threads())
  {
    BlockSource src{engine(BraKet::xs_xx, max_nprim_, max_l_,
                           std::numeric_limits<double>::epsilon())};
#pragma omp for schedule(dynamic, 1)
    for (long p = 0; p < long(ash.size()); ++p)
      for (size_t i : by_aux_[size_t(p)]) {
        const auto& b = blocks_[i];
        const double* v = src.get(store_, b.offset, ash[b.P], sh[b.s1], sh[b.s2]);
        if (!v) continue;
        const size_t nP = ash[b.P].size(), n1 = sh[b.s1].size(), n2 = sh[b.s2].size();
        const size_t f1 = off[b.s1], f2 = off[b.s2];
        for (size_t q = 0; q < nP; ++q)
          for (size_t x = 0; x < n1; ++x)
            for (size_t z = 0; z < n2; ++z, ++v)
              for (size_t d = 0; d < nd; ++d) {
                const double dd = b.s1 != b.s2 ? D[d](f1 + x, f2 + z) + D[d](f2 + z, f1 + x)
                                               : D[d](f1 + x, f2 + z);
                g(long(aoff[b.P] + q), long(d)) += *v * dd;
              }
      }
  }
  return g;
}

std::vector<Matrix> DFBuilder::expand(const Eigen::MatrixXd& c) const {
  const auto& sh = basis_.shells();
  const auto& off = basis_.shell_offsets();
  const auto& ash = aux_.shells();
  const auto& aoff = aux_.shell_offsets();
  const size_t n = basis_.nbf(), nd = size_t(c.cols()), ns = sh.size();
  std::vector<Matrix> J(nd, Matrix::Zero(n, n));
  std::vector<std::pair<size_t, size_t>> pairs;
  for (size_t s1 = 0; s1 < ns; ++s1)
    for (size_t s2 = 0; s2 <= s1; ++s2) pairs.emplace_back(s1, s2);
#pragma omp parallel num_threads(resources::threads())
  {
    BlockSource src{engine(BraKet::xs_xx, max_nprim_, max_l_,
                           std::numeric_limits<double>::epsilon())};
    std::vector<double> acc;
#pragma omp for schedule(dynamic, 4)
    for (long ip = 0; ip < long(pairs.size()); ++ip) {  // J(mu, nu) of a pair: this thread only
      const auto [s1, s2] = pairs[size_t(ip)];
      const size_t n1 = sh[s1].size(), n2 = sh[s2].size(), f1 = off[s1], f2 = off[s2];
      acc.assign(nd * n1 * n2, 0.0);
      for (size_t i : by_pair_[size_t(ip)]) {
        const auto& b = blocks_[i];
        const double* v = src.get(store_, b.offset, ash[b.P], sh[b.s1], sh[b.s2]);
        if (!v) continue;
        const size_t nP = ash[b.P].size();
        for (size_t q = 0; q < nP; ++q)
          for (size_t xz = 0; xz < n1 * n2; ++xz, ++v)
            for (size_t d = 0; d < nd; ++d) acc[d * n1 * n2 + xz] += *v * c(long(aoff[b.P] + q), long(d));
      }
      for (size_t d = 0; d < nd; ++d)
        for (size_t x = 0; x < n1; ++x)
          for (size_t z = 0; z < n2; ++z)
            J[d](f1 + x, f2 + z) = J[d](f2 + z, f1 + x) = acc[d * n1 * n2 + x * n2 + z];
    }
  }
  return J;
}

JK DFBuilder::compute(const std::vector<Matrix>& densities, bool symmetric, bool with_k,
                      const std::vector<int>&) {
  const long n = long(basis_.nbf());
  for (const auto& D : densities)
    if (D.rows() != n || D.cols() != n) throw std::invalid_argument("DF: density has wrong size");
  JK out;
  if (fitted_) {  // J: b = B d (d: D on the pair blocks, D + D^T off the diagonal), J = B^T b
    Eigen::MatrixXd d(long(bcols_), long(densities.size()));
    for (size_t i = 0; i < bpairs_.size(); ++i) {
      const auto [s1, s2] = bpairs_[i];
      const long n1 = long(basis_.shells()[s1].size()), n2 = long(basis_.shells()[s2].size());
      const long f1 = long(basis_.shell_offsets()[s1]), f2 = long(basis_.shell_offsets()[s2]);
      for (size_t k = 0; k < densities.size(); ++k)
        for (long x = 0, c = long(bpair_off_[i]); x < n1; ++x)
          for (long z = 0; z < n2; ++z, ++c)
            d(c, long(k)) = densities[k](f1 + x, f2 + z) + (s1 != s2 ? densities[k](f2 + z, f1 + x) : 0.0);
    }
    const Eigen::Map<const RowMat> B(B_.data(), long(nfit()), long(bcols_));
    out.J = fitted_coulomb(B * d);
  } else {  // J: gamma = (P|D), c = V^-1 gamma
    const Eigen::MatrixXd g = contract_densities(densities);
    out.J = expand(M_ * (M_.transpose() * g));
  }
  if (!with_k) return out;
  // K: per density, in balanced batches of factor vectors (memory: X = M^T Y and, from the
  // raw integrals, Y, naux * n * batch each, twice for non-symmetric D)
  const size_t limit = resources::memory();
  const size_t free_bytes =
      limit ? (limit > stored_bytes() ? limit - stored_bytes() : 0) : size_t(1) << 30;
  for (const auto& D : densities) {
    Eigen::MatrixXd a, b;
    Eigen::VectorXd s;
    const bool sym = symmetric || (D - D.transpose()).cwiseAbs().maxCoeff() <= 1e-14 * D.cwiseAbs().maxCoeff();
    factorize(D, sym, a, b, s);
    const long per_vec = (sym ? 1 : 2) * (fitted_ ? 1 : 2) * long(naux()) * n * long(sizeof(double));
    const long kbfit = std::max(1L, long(free_bytes / 4) / std::max(per_vec, 1L));
    const long nbatch = (a.cols() + kbfit - 1) / kbfit;
    const long kbmax = nbatch > 0 ? (a.cols() + nbatch - 1) / nbatch : 1;
    const int nt = resources::threads();
    std::vector<Eigen::MatrixXd> acc(size_t(nt), Eigen::MatrixXd::Zero(n, n));
    for (long k0 = 0; k0 < a.cols(); k0 += kbmax) {
      const long kb = std::min(kbmax, a.cols() - k0);
      const RowMat Xa = fitted_transform(a.middleCols(k0, kb));
      const RowMat Xb = sym ? RowMat() : fitted_transform(b.middleCols(k0, kb));
      const Eigen::VectorXd sk = s.segment(k0, kb);
      // K += sum_Q Xa_Q diag(s) Xb_Q^T over groups of g Q's gathered into panels W (n x g*kb,
      // column k*g + j for Q j); symmetric: W scaled by sqrt|s_k|, rank updates (lower
      // triangle) of the parts with s < 0 (first, ascending eigenvalues) and s > 0
      const long nneg = sym ? long((sk.array() < 0).count()) : 0;
      const long g = std::max(1L, 512 / kb), ngroup = (Xa.rows() + g - 1) / g;
      const Eigen::VectorXd wa = sym ? Eigen::VectorXd(sk.cwiseAbs().cwiseSqrt()) : Eigen::VectorXd::Ones(kb);
#pragma omp parallel num_threads(nt)
      {
        Eigen::MatrixXd Wa, Wb;
#pragma omp for schedule(dynamic, 1)
        for (long ig = 0; ig < ngroup; ++ig) {
          const long q0 = ig * g, gq = std::min(g, Xa.rows() - q0);
          Wa.resize(n, gq * kb);
          if (!sym) Wb.resize(n, gq * kb);
          for (long j = 0; j < gq; ++j) {
            const Eigen::Map<const RowMat> A(Xa.row(q0 + j).data(), n, kb);
            for (long k = 0; k < kb; ++k) Wa.col(k * gq + j) = wa(k) * A.col(k);
            if (sym) continue;
            const Eigen::Map<const RowMat> B(Xb.row(q0 + j).data(), n, kb);
            for (long k = 0; k < kb; ++k) Wb.col(k * gq + j) = sk(k) * B.col(k);
          }
          auto& K = acc[size_t(thread_id())];
          if (!sym) {
            K.noalias() += Wa * Wb.transpose();
            continue;
          }
          if (nneg > 0) K.selfadjointView<Eigen::Lower>().rankUpdate(Wa.leftCols(nneg * gq), -1.0);
          if (nneg < kb) K.selfadjointView<Eigen::Lower>().rankUpdate(Wa.rightCols((kb - nneg) * gq), 1.0);
        }
      }
    }
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n, n);
    for (const auto& x : acc) K += x;
    if (sym) K = Eigen::MatrixXd(K.selfadjointView<Eigen::Lower>());
    out.K.push_back(Matrix(K));
  }
  return out;
}

JK DFBuilder::pair_jk(const Matrix& Ca, std::vector<std::pair<int, int>> pairs) const {
  const long n = long(basis_.nbf()), m = Ca.cols();
  if (Ca.rows() != n) throw std::invalid_argument("DF pair_jk: Ca has wrong row count");
  if (pairs.empty())
    for (int u = 0; u < int(m); ++u)
      for (int w = u; w < int(m); ++w) pairs.emplace_back(u, w);
  const long np = long(pairs.size());
  const size_t need = (2 * naux() * size_t(n) * size_t(m) + 2 * size_t(np) * size_t(n * n)) * sizeof(double);
  if (!resources::fits(stored_bytes() + need))
    throw std::runtime_error("DF pair_jk needs " + resources::format_bytes(need) +
                             ", exceeds memory limit");
  // X[Q, mu u] = sum_P M_PQ (P|mu nu) C[nu, u]; b^uw_Q = sum_mu C[mu, w] X[Q, mu u]
  const RowMat X = fitted_transform(Eigen::MatrixXd(Ca));
  Eigen::MatrixXd g(X.rows(), np);
  const Eigen::MatrixXd Ct = Ca.transpose();
#pragma omp parallel for num_threads(resources::threads()) schedule(static)
  for (long q = 0; q < X.rows(); ++q) {
    const Eigen::MatrixXd t = Ct * Eigen::Map<const RowMat>(X.row(q).data(), n, m);  // (w, u)
    for (long p = 0; p < np; ++p) g(q, p) = t(pairs[size_t(p)].second, pairs[size_t(p)].first);
  }
  JK out;
  out.J = fitted_coulomb(g);
  // K_uv = sum_Q X^Q_u (X^Q_v)^T, per orbital u: (n x nfit)
  std::vector<Eigen::MatrixXd> Xu(size_t(m), Eigen::MatrixXd(n, X.rows()));
  for (long q = 0; q < X.rows(); ++q)
    for (long mu = 0; mu < n; ++mu)
      for (long u = 0; u < m; ++u) Xu[size_t(u)](mu, q) = X(q, mu * m + u);
  out.K.resize(size_t(np));
  for (long p = 0; p < np; ++p) {
    const auto [u, w] = pairs[size_t(p)];
    out.K[size_t(p)] = Matrix(Xu[size_t(u)] * Xu[size_t(w)].transpose());
  }
  return out;
}

RowMat DFBuilder::mo_three_index(const Matrix& C, const Matrix& Ca) const {
  const long n = long(basis_.nbf()), np = C.cols(), m = Ca.cols();
  if (C.rows() != n || Ca.rows() != n)
    throw std::invalid_argument("DF mo_three_index: orbitals have wrong row count");
  const size_t need = ((fitted_ ? 0 : naux()) + nfit()) * size_t(n * m) * sizeof(double) +
                      nfit() * size_t(np * m) * sizeof(double);
  if (!resources::fits(stored_bytes() + need))
    throw std::runtime_error("DF 3-index MO integrals need " + resources::format_bytes(need) +
                             ", exceed memory limit");
  // X[Q, mu u] = sum_P M_PQ (P|mu nu) Ca[nu, u], then B[Q, p u] = C[mu, p] X[Q, mu u]
  const RowMat X = fitted_transform(Eigen::MatrixXd(Ca));
  RowMat Bmo(X.rows(), np * m);
  const Eigen::MatrixXd Ct = C.transpose();
#pragma omp parallel for num_threads(resources::threads()) schedule(static)
  for (long q = 0; q < X.rows(); ++q) {
    const Eigen::Map<const RowMat> x(X.row(q).data(), n, m);
    Eigen::Map<RowMat>(Bmo.row(q).data(), np, m).noalias() = Ct * x;
  }
  return Bmo;
}

size_t DFBuilder::memory_estimate(size_t ndens, bool with_k, int nthreads) const {
  const size_t n2 = basis_.nbf() * basis_.nbf();
  size_t b = stored_bytes() + ndens * 2 * n2 * sizeof(double);
  // K: per thread an n x n accumulator and slab, two panels (n x 512) and the directly
  // computed blocks of a fitting shell (at most nP n^2 / 2)
  size_t nP = 0;
  for (const auto& s : aux_.shells()) nP = std::max(nP, s.size());
  const size_t per_thread = 2 * n2 + 1024 * basis_.nbf() + (blocks_direct() && !fitted_ ? nP * n2 / 2 : 0);
  if (with_k)
    b += size_t(nthreads) * per_thread * sizeof(double) + 4 * naux() * basis_.nbf() * sizeof(double);
  return b;
}

std::string DFBuilder::describe() const {
  const size_t total = blocks_.size();
  const double pct = total ? 100.0 * double(n_stored_) / double(total) : 100.0;
  char line[256];
  if (fitted_) {
    std::snprintf(line, sizeof(line),
                  "density fitting: %zu fitting functions (%zu metric vectors kept), fitted "
                  "3-index integrals stored (%s)",
                  naux(), nfit(), resources::format_bytes(stored_bytes()).c_str());
    return line;
  }
  std::snprintf(line, sizeof(line),
                "density fitting: %zu fitting functions (%zu metric vectors kept), 3-index "
                "blocks %.0f%% stored (%s), rest direct",
                naux(), nfit(), pct, resources::format_bytes(stored_bytes()).c_str());
  return line;
}

double DFBuilder::screened_percent() const {
  const size_t total = blocks_.size() + n_screened_;
  return total ? 100.0 * double(n_screened_) / double(total) : 0.0;
}

}  // namespace gemqc

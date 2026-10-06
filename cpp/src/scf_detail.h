#pragma once
// Internal helpers shared by the SCF drivers (RHF/UHF engine in scf.cpp, ROHF
// in rohf.cpp): DIIS, symmetry-blocked orthogonalization and diagonalization,
// occupation, densities, L-BFGS, canonicalization.

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemqc/log.h"
#include "gemqc/scf.h"

namespace gemqc::detail {

using Clock = std::chrono::steady_clock;
inline double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Pulay DIIS on (orthonormal-basis) Fock matrices with commutator errors
class Diis {
 public:
  explicit Diis(int maxsize) : max_(maxsize) {}
  int size() const { return int(f_.size()); }

  Matrix extrapolate(const Matrix& f, const Matrix& e) {
    f_.push_back(f);
    e_.push_back(e);
    if (size() > max_) f_.pop_front(), e_.pop_front();
    while (size() > 1) {
      const int m = size();
      Matrix b = Matrix::Zero(m + 1, m + 1);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j <= i; ++j)
          b(i, j) = b(j, i) = e_[i].cwiseProduct(e_[j]).sum();
      const double scale = b.diagonal().head(m).maxCoeff();
      b.topLeftCorner(m, m) /= scale;
      b.row(m).head(m).setConstant(-1.0);
      b.col(m).head(m).setConstant(-1.0);
      Vector rhs = Vector::Zero(m + 1);
      rhs(m) = -1.0;
      Eigen::ColPivHouseholderQR<Matrix> qr(b);
      if (qr.rank() < m + 1) {  // ill-conditioned: drop oldest vector
        f_.pop_front(), e_.pop_front();
        continue;
      }
      const Vector c = qr.solve(rhs);
      Matrix fx = Matrix::Zero(f.rows(), f.cols());
      for (int i = 0; i < m; ++i) fx += c(i) * f_[i];
      return fx;
    }
    return f;
  }

 private:
  int max_;
  std::deque<Matrix> f_, e_;
};

// orthonormal, symmetry-blocked orbital space: columns of X, grouped by irrep
struct Blocks {
  Matrix X;
  std::vector<Eigen::Index> start, size;
};

// canonical orthogonalization per irrep (drops eigenvalues of S below lindep)
inline Blocks orthogonalizer(const Matrix& S, const std::vector<Matrix>& so, double lindep) {
  Blocks b;
  std::vector<Matrix> parts;
  Eigen::Index nmo = 0;
  for (const auto& U : so) {
    b.start.push_back(nmo);
    if (U.cols() == 0) {  // irrep without basis functions (Eigen fails on 0x0)
      parts.emplace_back(S.rows(), 0);
      b.size.push_back(0);
      continue;
    }
    Eigen::SelfAdjointEigenSolver<Matrix> es(U.transpose() * S * U);
    const Vector& s = es.eigenvalues();
    Eigen::Index first = 0;
    while (first < s.size() && s(first) < lindep) ++first;
    const Eigen::Index nk = s.size() - first;
    parts.push_back(U * es.eigenvectors().rightCols(nk) *
                    s.tail(nk).cwiseSqrt().cwiseInverse().asDiagonal());
    b.size.push_back(nk);
    nmo += nk;
  }
  b.X = Matrix(S.rows(), nmo);
  for (size_t k = 0; k < parts.size(); ++k) b.X.middleCols(b.start[k], b.size[k]) = parts[k];
  return b;
}

// eigenpairs of the block-diagonal Fo, sorted by energy; irrep of each orbital
inline void diagonalize(const Matrix& Fo, const Blocks& b, Matrix& Co, Vector& eps,
                 std::vector<int>& irrep) {
  struct Entry { double e; int k; Eigen::Index col; };
  std::vector<Entry> all;
  std::vector<Matrix> vecs;
  for (size_t k = 0; k < b.size.size(); ++k) {
    if (b.size[k] == 0) { vecs.emplace_back(); continue; }
    Eigen::SelfAdjointEigenSolver<Matrix> es(
        Fo.block(b.start[k], b.start[k], b.size[k], b.size[k]));
    vecs.push_back(es.eigenvectors());
    for (Eigen::Index i = 0; i < b.size[k]; ++i) all.push_back({es.eigenvalues()(i), int(k), i});
  }
  std::stable_sort(all.begin(), all.end(), [](auto& a, auto& c) { return a.e < c.e; });
  const auto nmo = Fo.rows();
  Co = Matrix::Zero(nmo, nmo);
  eps.resize(nmo);
  irrep.resize(nmo);
  for (Eigen::Index j = 0; j < nmo; ++j) {
    const auto& a = all[j];
    Co.col(j).segment(b.start[a.k], b.size[a.k]) = vecs[a.k].col(a.col);
    eps(j) = a.e;
    irrep[j] = a.k;
  }
}

// occupied orbitals: lowest per irrep for fixed counts, else aufbau
inline std::vector<bool> occupy(const std::vector<int>& irrep, size_t nocc,
                         const std::vector<int>& per_irrep, size_t nirrep) {
  std::vector<bool> occ(irrep.size(), false);
  if (per_irrep.empty()) {
    for (size_t j = 0; j < nocc && j < irrep.size(); ++j) occ[j] = true;
    return occ;
  }
  std::vector<int> left(per_irrep);
  for (size_t j = 0; j < irrep.size(); ++j)
    if (left[irrep[j]] > 0) occ[j] = true, --left[irrep[j]];
  for (size_t k = 0; k < nirrep; ++k)
    if (left[k] > 0) throw std::runtime_error("not enough orbitals in irrep for occupation");
  return occ;
}

inline std::vector<int> count_per_irrep(const std::vector<int>& irrep, const std::vector<bool>& occ,
                                 size_t nirrep) {
  std::vector<int> n(nirrep, 0);
  for (size_t j = 0; j < irrep.size(); ++j) n[irrep[j]] += occ[j];
  return n;
}

// density of occupied orbitals with occupation f (RHF 2, UHF 1)
inline Matrix density(const Matrix& C, const std::vector<bool>& occ, double f) {
  Matrix D = Matrix::Zero(C.rows(), C.rows());
  for (size_t j = 0; j < occ.size(); ++j)
    if (occ[j]) D.noalias() += f * C.col(j) * C.col(j).transpose();
  return D;
}

// --- quasi-Newton (orbital rotations) ---
// kappa: nmo x nmo matrix, nonzero only for (virtual a, occupied i) of equal irrep

inline std::vector<Eigen::Index> indices(const std::vector<bool>& occ, bool want) {
  std::vector<Eigen::Index> idx;
  for (size_t j = 0; j < occ.size(); ++j)
    if (occ[j] == want) idx.push_back(Eigen::Index(j));
  return idx;
}

// L-BFGS with diagonal preconditioner on masked (virtual, occupied) matrices
class Lbfgs {
 public:
  explicit Lbfgs(int m) : m_(m) {}
  int size() const { return int(s_.size()); }
  void clear() { s_.clear(), y_.clear(); }
  void push(const Matrix& s, const Matrix& y) {
    if (s.cwiseProduct(y).sum() <= 1e-14) return;  // curvature condition
    s_.push_back(s), y_.push_back(y);
    if (size() > m_) s_.pop_front(), y_.pop_front();
  }
  // -H^{-1} g (two-loop recursion), h0: diagonal Hessian estimate
  Matrix step(const Matrix& g, const Matrix& h0, const Matrix& mask) const {
    Matrix q = g;
    std::vector<double> alpha(s_.size()), rho(s_.size());
    for (int i = size() - 1; i >= 0; --i) {
      rho[i] = 1.0 / y_[i].cwiseProduct(s_[i]).sum();
      alpha[i] = rho[i] * s_[i].cwiseProduct(q).sum();
      q -= alpha[i] * y_[i];
    }
    Matrix r = q.cwiseQuotient(h0).cwiseProduct(mask);
    for (int i = 0; i < size(); ++i) {
      const double beta = rho[i] * y_[i].cwiseProduct(r).sum();
      r += (alpha[i] - beta) * s_[i];
    }
    return -r;
  }

 private:
  int m_;
  std::deque<Matrix> s_, y_;
};

inline void canonicalize(const Matrix& F, Matrix& C, Vector& eps, std::vector<int>& irrep,
                  std::vector<bool>& occ, size_t nirrep) {
  const Matrix Fmo = C.transpose() * F * C;
  struct Entry { double e; int k; bool o; Vector c; };
  std::vector<Entry> all;
  for (size_t k = 0; k < nirrep; ++k)
    for (bool o : {true, false}) {
      std::vector<Eigen::Index> idx;
      for (size_t j = 0; j < irrep.size(); ++j)
        if (irrep[j] == int(k) && occ[j] == o) idx.push_back(Eigen::Index(j));
      if (idx.empty()) continue;
      const Eigen::Index m = idx.size();
      Matrix sub(m, m), Cs(C.rows(), m);
      for (Eigen::Index a = 0; a < m; ++a) {
        Cs.col(a) = C.col(idx[a]);
        for (Eigen::Index b = 0; b < m; ++b) sub(a, b) = Fmo(idx[a], idx[b]);
      }
      Eigen::SelfAdjointEigenSolver<Matrix> es(sub);
      const Matrix Cn = Cs * es.eigenvectors();
      for (Eigen::Index a = 0; a < m; ++a)
        all.push_back({es.eigenvalues()(a), int(k), o, Cn.col(a)});
    }
  std::stable_sort(all.begin(), all.end(), [](auto& a, auto& b) { return a.e < b.e; });
  for (size_t j = 0; j < all.size(); ++j) {
    C.col(j) = all[j].c;
    eps(j) = all[j].e;
    irrep[j] = all[j].k;
    occ[j] = all[j].o;
  }
}

inline std::string occupation_string(const std::vector<int>& n) {
  std::string s;
  for (int k : n) s += log::format(" %3d", k);
  return s;
}

// symmetry-adapted blocks of the SCF (C1: one block of all AOs) and irrep names
inline std::vector<Matrix> symmetry_blocks(const ScfSymmetry& sym, size_t n,
                                           std::vector<std::string>& names) {
  std::vector<Matrix> so = sym.so;
  names = sym.irreps;
  if (so.empty()) {
    so = {Matrix::Identity(n, n)};
    names = {"A"};
  }
  Eigen::Index nso = 0;
  for (const auto& U : so) {
    if (size_t(U.rows()) != n) throw std::invalid_argument("symmetry functions: wrong dimension");
    nso += U.cols();
  }
  if (size_t(nso) != n || names.size() != so.size())
    throw std::invalid_argument("symmetry functions do not span the basis");
  return so;
}

}  // namespace gemqc::detail

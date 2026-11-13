#include "gemqc/twobody.h"

#include <libint2.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <limits>
#include <stdexcept>

#include "gemqc/resources.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gemqc {

using libint2::BraKet;
using libint2::Operator;

TwoBodyOperator::Kind TwoBodyOperator::kind_from(const std::string& name) {
  if (name == "coulomb") return Kind::coulomb;
  if (name == "stg") return Kind::stg;
  if (name == "stg_x_coulomb") return Kind::stg_x_coulomb;
  if (name == "cgtg") return Kind::cgtg;
  if (name == "cgtg_x_coulomb") return Kind::cgtg_x_coulomb;
  if (name == "delcgtg2") return Kind::delcgtg2;
  if (name == "delcgtga_delcgtgb") return Kind::delcgtga_delcgtgb;
  throw std::invalid_argument("unknown two-body operator '" + name +
                              "' (coulomb, stg, stg_x_coulomb, cgtg, cgtg_x_coulomb, delcgtg2, "
                              "delcgtga_delcgtgb)");
}

std::string TwoBodyOperator::name() const {
  switch (kind) {
    case Kind::coulomb: return "coulomb";
    case Kind::stg: return "stg";
    case Kind::stg_x_coulomb: return "stg_x_coulomb";
    case Kind::cgtg: return "cgtg";
    case Kind::cgtg_x_coulomb: return "cgtg_x_coulomb";
    case Kind::delcgtg2: return "delcgtg2";
    case Kind::delcgtga_delcgtgb: return "delcgtga_delcgtgb";
  }
  return "";
}

bool TwoBodyOperator::has_delcgtgab() {
#ifdef GEMQC_HAVE_DELCGTGAB
  return true;
#else
  return false;
#endif
}

namespace {

/// check a Gaussian expansion (non-empty, exponents > 0)
void check_gtg(const std::vector<std::pair<double, double>>& gtg) {
  if (gtg.empty()) throw std::invalid_argument("Gaussian geminal without terms");
  for (const auto& [a, c] : gtg)
    if (!(a > 0.0)) throw std::invalid_argument("Gaussian geminal exponent must be > 0");
}

}  // namespace

libint2::Engine two_body_engine(const TwoBodyOperator& op, size_t max_nprim, int max_l,
                                int deriv, BraKet braket) {
  init_libint();
  const double eps = std::numeric_limits<double>::epsilon();
  using K = TwoBodyOperator::Kind;
  if (op.kind == K::coulomb)
    return libint2::Engine(Operator::coulomb, max_nprim, max_l, deriv, eps,
                           {}, braket);
  if (op.kind == K::delcgtga_delcgtgb) {
#ifdef GEMQC_HAVE_DELCGTGAB
    check_gtg(op.gtg);
    check_gtg(op.gtg2);
    return libint2::Engine(Operator::delcgtga_delcgtgb, max_nprim, max_l, deriv, eps,
                           std::array<libint2::ContractedGaussianGeminal, 2>{op.gtg, op.gtg2},
                           braket);
#else
    throw std::runtime_error("operator delcgtga_delcgtgb needs a libint with this kernel "
                             "(custom build, docs/developer/libint_build.md)");
#endif
  }
  if (op.kind == K::cgtg || op.kind == K::cgtg_x_coulomb || op.kind == K::delcgtg2) {
    check_gtg(op.gtg);
    const auto oper = op.kind == K::cgtg             ? Operator::cgtg
                      : op.kind == K::cgtg_x_coulomb ? Operator::cgtg_x_coulomb
                                                     : Operator::delcgtg2;
    return libint2::Engine(oper, max_nprim, max_l, deriv, eps,
                           libint2::ContractedGaussianGeminal(op.gtg), braket);
  }
  // libint's Slater/Yukawa core integrals fail (crash) for tiny exponents
  if (!(op.zeta >= 1e-3))
    throw std::invalid_argument("Slater-type operator needs zeta >= 1e-3");
  const auto oper = op.kind == K::stg ? Operator::stg : Operator::stg_x_coulomb;
  return libint2::Engine(oper, max_nprim, max_l, deriv, eps, op.zeta, braket);
}

void compute_quartet(libint2::Engine& e, const TwoBodyOperator& op, const libint2::Shell& s1,
                     const libint2::Shell& s2, const libint2::Shell& s3,
                     const libint2::Shell& s4, const libint2::ShellPair* p12,
                     const libint2::ShellPair* p34) {
  using K = TwoBodyOperator::Kind;
  switch (op.kind) {
    case K::coulomb:
      e.compute2<Operator::coulomb, BraKet::xx_xx, 0>(s1, s2, s3, s4, p12, p34);
      break;
    case K::stg:
      e.compute2<Operator::stg, BraKet::xx_xx, 0>(s1, s2, s3, s4, p12, p34);
      break;
    case K::stg_x_coulomb:
      e.compute2<Operator::stg_x_coulomb, BraKet::xx_xx, 0>(s1, s2, s3, s4, p12, p34);
      break;
    case K::cgtg:
      e.compute2<Operator::cgtg, BraKet::xx_xx, 0>(s1, s2, s3, s4, p12, p34);
      break;
    case K::cgtg_x_coulomb:
      e.compute2<Operator::cgtg_x_coulomb, BraKet::xx_xx, 0>(s1, s2, s3, s4, p12, p34);
      break;
    case K::delcgtg2:
      e.compute2<Operator::delcgtg2, BraKet::xx_xx, 0>(s1, s2, s3, s4, p12, p34);
      break;
    case K::delcgtga_delcgtgb:
#ifdef GEMQC_HAVE_DELCGTGAB
      e.compute2<Operator::delcgtga_delcgtgb, BraKet::xx_xx, 0>(s1, s2, s3, s4, p12, p34);
#endif
      break;
  }
}

namespace {

int threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

int thread_id() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

// one engine per thread for shells of all given bases, with the braket type set
std::vector<libint2::Engine> engines(const TwoBodyOperator& op,
                                     const std::vector<const BasisSet*>& bases, BraKet bk) {
  size_t nprim = 0;
  int l = 0;
  for (const auto* b : bases) {
    nprim = std::max(nprim, b->max_nprim());
    l = std::max(l, b->max_l());
  }
  std::vector<libint2::Engine> e(threads(), two_body_engine(op, nprim, l, 0, bk));
  for (auto& x : e) x.set(bk);
  return e;
}

}  // namespace

Matrix two_body_2c(const TwoBodyOperator& op, const BasisSet& b1, const BasisSet& b2) {
  auto eng = engines(op, {&b1, &b2}, BraKet::xs_xs);
  const auto& sh1 = b1.shells();
  const auto& sh2 = b2.shells();
  const auto& o1 = b1.shell_offsets();
  const auto& o2 = b2.shell_offsets();
  Matrix V = Matrix::Zero(b1.nbf(), b2.nbf());
  const auto unit = libint2::Shell::unit();
#pragma omp parallel for schedule(dynamic)
  for (size_t p = 0; p < sh1.size(); ++p) {
    auto& e = eng[thread_id()];
    const auto& buf = e.results();
    for (size_t q = 0; q < sh2.size(); ++q) {
      e.compute(sh1[p], unit, sh2[q], unit);
      if (buf[0] == nullptr) continue;
      Eigen::Map<const Matrix> blk(buf[0], sh1[p].size(), sh2[q].size());
      V.block(o1[p], o2[q], sh1[p].size(), sh2[q].size()) = op.scale * blk;
    }
  }
  return V;
}

Matrix two_body_3c(const TwoBodyOperator& op, const BasisSet& aux, const BasisSet& b1,
                   const BasisSet& b2) {
  const size_t n1 = b1.nbf(), n2 = b2.nbf();
  const size_t bytes = aux.nbf() * n1 * n2 * sizeof(double);
  if (!resources::fits(bytes))
    throw std::runtime_error("3-index integrals need " + resources::format_bytes(bytes) +
                             ", exceeds memory limit");
  auto eng = engines(op, {&aux, &b1, &b2}, BraKet::xs_xx);
  const auto& sha = aux.shells();
  const auto& sh1 = b1.shells();
  const auto& sh2 = b2.shells();
  const auto& oa = aux.shell_offsets();
  const auto& o1 = b1.shell_offsets();
  const auto& o2 = b2.shell_offsets();
  Matrix V = Matrix::Zero(aux.nbf(), n1 * n2);
  const auto unit = libint2::Shell::unit();
#pragma omp parallel for schedule(dynamic)
  for (size_t p = 0; p < sha.size(); ++p) {
    auto& e = eng[thread_id()];
    const auto& buf = e.results();
    for (size_t a = 0; a < sh1.size(); ++a)
      for (size_t b = 0; b < sh2.size(); ++b) {
        e.compute(sha[p], unit, sh1[a], sh2[b]);
        if (buf[0] == nullptr) continue;
        size_t q = 0;
        for (size_t i = 0; i < sha[p].size(); ++i)
          for (size_t j = 0; j < sh1[a].size(); ++j)
            for (size_t k = 0; k < sh2[b].size(); ++k, ++q)
              V(oa[p] + i, (o1[a] + j) * n2 + o2[b] + k) = op.scale * buf[0][q];
      }
  }
  return V;
}

namespace {

/// Shells of `basis` on which some of the given columns of C (all if empty) are nonzero
std::vector<char> nonzero_shells(const BasisSet& basis, const Matrix& C,
                                 const std::vector<char>& cols = {}) {
  const auto& sh = basis.shells();
  const auto& off = basis.shell_offsets();
  std::vector<char> used(sh.size(), 0);
  for (size_t a = 0; a < sh.size(); ++a)
    for (size_t x = 0; x < sh[a].size() && !used[a]; ++x)
      for (Eigen::Index k = 0; k < C.cols(); ++k)
        if ((cols.empty() || cols[k]) && C(off[a] + x, k) != 0.0) {
          used[a] = 1;
          break;
        }
  return used;
}

using Range = std::pair<Eigen::Index, Eigen::Index>;

/// Column range [lo, hi) of the nonzero entries of C in the rows of each shell of `basis`
/// (lo = hi for zero rows): masked orbitals are nonzero only in the columns of their group
std::vector<Range> column_ranges(const BasisSet& basis, const Matrix& C) {
  const auto& sh = basis.shells();
  const auto& off = basis.shell_offsets();
  std::vector<Range> r(sh.size(), {0, 0});
  for (size_t a = 0; a < sh.size(); ++a) {
    Eigen::Index lo = C.cols(), hi = 0;
    for (size_t x = 0; x < sh[a].size(); ++x)
      for (Eigen::Index k = 0; k < C.cols(); ++k)
        if (C(off[a] + x, k) != 0.0) lo = std::min(lo, k), hi = std::max(hi, k + 1);
    if (lo < hi) r[a] = {lo, hi};
  }
  return r;
}

/// Number of functions in the shells marked in `used`
size_t nfunctions(const BasisSet& basis, const std::vector<char>& used) {
  size_t n = 0;
  for (size_t a = 0; a < used.size(); ++a)
    if (used[a]) n += basis.shells()[a].size();
  return n;
}

}  // namespace

Matrix two_body_3c_mo(const TwoBodyOperator& op, const BasisSet& aux, const BasisSet& basis,
                      const Matrix& C1, const Matrix& C2) {
  const size_t nbf = basis.nbf(), n1 = C1.cols(), n2 = C2.cols();
  if (size_t(C1.rows()) != nbf || size_t(C2.rows()) != nbf)
    throw std::invalid_argument("two_body_3c_mo: orbital matrices do not match the basis");
  auto eng = engines(op, {&aux, &basis}, BraKet::xs_xx);
  const auto& sha = aux.shells();
  const auto& sh = basis.shells();
  const auto& oa = aux.shell_offsets();
  const auto& ob = basis.shell_offsets();
  Matrix R = Matrix::Zero(aux.nbf(), n1 * n2);
  const auto unit = libint2::Shell::unit();
  const Eigen::MatrixXd C1t = C1.transpose();  // (n1, nbf)
  // nonzero columns of C1 per shell: pairs (a b) with C1 = 0 on both shells give nothing
  // (masked orbitals, orbital-basis orbitals in the RI basis)
  const auto r1 = column_ranges(basis, C1);
#pragma omp parallel for schedule(dynamic)
  for (size_t p = 0; p < sha.size(); ++p) {
    auto& e = eng[thread_id()];
    const auto& buf = e.results();
    const size_t np = sha[p].size();
    // H[i](nu, k) = sum_mu (P_i|mu nu) C1_{mu k}: first index transformed per aux function
    std::vector<Eigen::MatrixXd> H(np, Eigen::MatrixXd::Zero(nbf, n1));
    for (size_t a = 0; a < sh.size(); ++a)
      for (size_t b = 0; b <= a; ++b) {
        const auto [la, ha] = r1[a];
        const auto [lb, hb] = r1[b];
        if (la == ha && lb == hb) continue;
        e.compute(sha[p], unit, sh[a], sh[b]);
        if (buf[0] == nullptr) continue;
        const size_t na = sh[a].size(), nb = sh[b].size();
        for (size_t i = 0; i < np; ++i) {
          Eigen::Map<const Matrix> blk(buf[0] + i * na * nb, na, nb);  // (mu in a, nu in b)
          H[i].block(ob[b], la, nb, ha - la).noalias() +=
              blk.transpose() * C1.block(ob[a], la, na, ha - la);
          if (a != b)
            H[i].block(ob[a], lb, na, hb - lb).noalias() +=
                blk * C1.block(ob[b], lb, nb, hb - lb);
        }
      }
    for (size_t i = 0; i < np; ++i) {
      const Eigen::MatrixXd T = op.scale * (H[i].transpose() * C2);  // (n1, n2)
      for (size_t k = 0; k < n1; ++k) R.row(oa[p] + i).segment(k * n2, n2) = T.row(k);
    }
  }
  return R;
}

std::vector<Matrix> pair_integrals(const TwoBodyOperator& op, const BasisSet& bo,
                                   const BasisSet& b1, const BasisSet& b2, const Matrix& Co,
                                   const std::vector<std::pair<int, int>>& pairs,
                                   double threshold) {
  const size_t no = bo.nbf(), n1 = b1.nbf(), n2 = b2.nbf(), m = Co.cols(), np = pairs.size();
  if (size_t(Co.rows()) != no) throw std::invalid_argument("pair_integrals: Co does not match bo");
  for (const auto& [i, j] : pairs)
    if (i < 0 || j < 0 || i >= int(m) || j >= int(m))
      throw std::invalid_argument("pair_integrals: orbital index out of range");
  const size_t bytes = (&b1 == &b2 ? 2 : 1) * np * n1 * n2 * sizeof(double);
  if (!resources::fits(bytes))
    throw std::runtime_error("pair integrals need " + resources::format_bytes(bytes) +
                             ", exceeds memory limit");
  const auto &sho = bo.shells(), &sh1 = b1.shells(), &sh2 = b2.shells();
  const auto &oo = bo.shell_offsets(), &o1 = b1.shell_offsets(), &o2 = b2.shell_offsets();
  size_t nprim = std::max({bo.max_nprim(), b1.max_nprim(), b2.max_nprim()});
  int lmax = std::max({bo.max_l(), b1.max_l(), b2.max_l()});
  const double ln_prec = std::log(std::numeric_limits<double>::epsilon());
  // shell-pair data and Schwarz factors sqrt(max |(a b|a b)|) of (bo, b) shell pairs
  auto pair_data = [&](const BasisSet& b, std::vector<std::shared_ptr<libint2::ShellPair>>& sp,
                       Matrix& q) {
    const auto& sb = b.shells();
    sp.resize(sho.size() * sb.size());
    q = Matrix::Zero(sho.size(), sb.size());
    auto e = two_body_engine(op, nprim, lmax);
    e.set_precision(0.0);
    const auto& buf = e.results();
    for (size_t a = 0; a < sho.size(); ++a)
      for (size_t c = 0; c < sb.size(); ++c) {
        sp[a * sb.size() + c] = std::make_shared<libint2::ShellPair>(sho[a], sb[c], ln_prec);
        compute_quartet(e, op, sho[a], sb[c], sho[a], sb[c]);
        double mx = 0.0;
        if (buf[0]) {
          const size_t n = sho[a].size() * sb[c].size();
          for (size_t k = 0; k < n * n; ++k) mx = std::max(mx, std::abs(buf[0][k]));
        }
        q(a, c) = std::sqrt(mx * std::abs(op.scale));
      }
  };
  std::vector<std::shared_ptr<libint2::ShellPair>> sp1, sp2;
  Matrix q1, q2;
  pair_data(b1, sp1, q1);
  if (&b2 == &b1)
    sp2 = sp1, q2 = q1;
  else
    pair_data(b2, sp2, q2);
  const double qmax2 = q2.size() ? q2.maxCoeff() : 0.0;
  std::vector<Matrix> R(np, Matrix::Zero(n1, n2));
  const Eigen::MatrixXd C = Co;
  const double eps = std::numeric_limits<double>::epsilon();
  std::vector<libint2::Engine> eng(threads(), two_body_engine(op, nprim, lmax));
  for (auto& e : eng) e.set_precision(eps);
  // shells of bo with nonzero coefficients of the first (i) / second (j) orbitals of the
  // pairs (masked orbitals, F12XG); mu runs over ui, la over uj, the other shells give nothing
  std::vector<char> coli(m, 0), colj(m, 0);
  for (const auto& [i, j] : pairs) coli[i] = 1, colj[j] = 1;
  const auto ui = nonzero_shells(bo, C, coli), uj = nonzero_shells(bo, C, colj);
  const auto rc = column_ranges(bo, Co);
  std::vector<char> uu(sho.size());
  for (size_t a = 0; a < sho.size(); ++a) uu[a] = ui[a] || uj[a];
  // b1 = b2: (mu nu|la si) = (la si|mu nu); only kets (c d) <= bra (a b) are computed, the
  // partner (c d|a b) contributes K^ij(si, nu) += C_la,i C_mu,j v, collected as
  // L^ij(nu, si) = sum_mu C_mu,j H[(mu nu)](i, si) from the kets (c d) < (a b), added as L^T.
  // mu and la then run over the union of ui and uj: used only if cheaper than ui x uj
  const double nu_ = nfunctions(bo, uu);
  const bool sym = &b1 == &b2 && nu_ * nu_ <= 2.0 * nfunctions(bo, ui) * nfunctions(bo, uj);
  const auto& umu = sym ? uu : ui;
  const auto& ula = sym ? uu : uj;
  std::vector<Matrix> L(sym ? np : 0, Matrix::Zero(n1, n2));
#pragma omp parallel for schedule(dynamic)
  for (size_t b = 0; b < sh1.size(); ++b) {  // nu shell: rows of the result owned here
    auto& e = eng[thread_id()];
    const auto& buf = e.results();
    const size_t nb = sh1[b].size();
    std::vector<Eigen::MatrixXd> Kb(np, Eigen::MatrixXd::Zero(nb, n2));
    std::vector<Eigen::MatrixXd> Lb(sym ? np : 0, Eigen::MatrixXd::Zero(nb, n2));
    for (size_t a = 0; a < sho.size(); ++a) {  // mu shell (orbital basis)
      if (!umu[a] || q1(a, b) * qmax2 < threshold) continue;
      const size_t na = sho[a].size(), nab = na * nb;
      // H[(mu nu)](j, si) = sum_la (mu nu|la si) C_{la j}
      std::vector<Eigen::MatrixXd> H(nab, Eigen::MatrixXd::Zero(m, n2));
      Eigen::MatrixXd W, T;  // W(la, k*nd + si): the quartet block reordered; T = C^T W
      auto add = [&](size_t c, size_t d) {
        if (q1(a, b) * q2(c, d) < threshold) return;
        compute_quartet(e, op, sho[a], sh1[b], sho[c], sh2[d], sp1[a * sh1.size() + b].get(),
                        sp2[c * sh2.size() + d].get());
        if (buf[0] == nullptr) return;
        const size_t nc = sho[c].size(), nd = sh2[d].size();
        W.resize(nc, nab * nd);
        const double* v = buf[0];
        for (size_t k = 0; k < nab; ++k)  // buffer order (k, la, si)
          for (size_t l = 0; l < nc; ++l)
            for (size_t t = 0; t < nd; ++t) W(l, k * nd + t) = *v++;
        // one product per quartet, over the nonzero columns of C on shell c
        const auto [lo, hi] = rc[c];
        T.noalias() = C.block(oo[c], lo, nc, hi - lo).transpose() * W;
        for (size_t k = 0; k < nab; ++k)
          H[k].block(lo, o2[d], hi - lo, nd) += T.middleCols(k * nd, nd);
      };
      for (size_t c = 0; c < sho.size(); ++c) {
        if (!ula[c]) continue;
        for (size_t d = 0; d < sh2.size(); ++d)
          if (!sym || c < a || (c == a && d < b)) add(c, d);
      }
      if (sym) {  // partner contributions of the kets (c d) < (a b), then the diagonal ket
        for (size_t x = 0; x < na; ++x)
          for (size_t y = 0; y < nb; ++y)
            for (size_t p = 0; p < np; ++p) {
              const auto [i, j] = pairs[p];
              Lb[p].row(y) += C(oo[a] + x, j) * H[x * nb + y].row(i);
            }
        add(a, b);
      }
      for (size_t x = 0; x < na; ++x)
        for (size_t y = 0; y < nb; ++y) {
          const auto& Hk = H[x * nb + y];
          for (size_t p = 0; p < np; ++p) {
            const auto [i, j] = pairs[p];
            Kb[p].row(y) += C(oo[a] + x, i) * Hk.row(j);
          }
        }
    }
    for (size_t p = 0; p < np; ++p) {
      R[p].middleRows(o1[b], nb) = op.scale * Kb[p];
      if (sym) L[p].middleRows(o1[b], nb) = op.scale * Lb[p];
    }
  }
  for (size_t p = 0; p < L.size(); ++p) R[p] += L[p].transpose();
  return R;
}

namespace {

/// Shell pairs (a, b) of (b1, b2) with their libint pair data and Schwarz factors
/// sqrt(max |(a b|a b)|)
struct ShellPairs {
  std::vector<std::pair<size_t, size_t>> ab;
  std::vector<std::shared_ptr<libint2::ShellPair>> data;
  std::vector<double> q;
};

ShellPairs shell_pairs(const TwoBodyOperator& op, const BasisSet& b1, const BasisSet& b2,
                       std::vector<std::pair<size_t, size_t>> ab, size_t nprim, int lmax) {
  const auto &s1 = b1.shells(), &s2 = b2.shells();
  const double ln_prec = std::log(std::numeric_limits<double>::epsilon());
  ShellPairs sp{std::move(ab), {}, {}};
  sp.data.resize(sp.ab.size());
  sp.q.resize(sp.ab.size());
  std::vector<libint2::Engine> eng(threads(), two_body_engine(op, nprim, lmax));
  for (auto& e : eng) e.set_precision(0.0);
#pragma omp parallel for schedule(dynamic)
  for (size_t t = 0; t < sp.ab.size(); ++t) {
    auto& e = eng[thread_id()];
    const auto [a, b] = sp.ab[t];
    sp.data[t] = std::make_shared<libint2::ShellPair>(s1[a], s2[b], ln_prec);
    compute_quartet(e, op, s1[a], s2[b], s1[a], s2[b]);
    double mx = 0.0;
    if (e.results()[0]) {
      const size_t n = s1[a].size() * s2[b].size();
      for (size_t k = 0; k < n * n; ++k) mx = std::max(mx, std::abs(e.results()[0][k]));
    }
    sp.q[t] = std::sqrt(mx * std::abs(op.scale));
  }
  return sp;
}

}  // namespace

Matrix mo_integrals(const TwoBodyOperator& op, const BasisSet& b1, const Matrix& C1,
                    const BasisSet& b2, const Matrix& C2, const BasisSet& b3, const Matrix& C3,
                    const BasisSet& b4, const Matrix& C4, double threshold,
                    const std::vector<std::array<int, 4>>& blocks, Eigen::Index block_size,
                    int nblocks) {
  if (size_t(C1.rows()) != b1.nbf() || size_t(C2.rows()) != b2.nbf() ||
      size_t(C3.rows()) != b3.nbf() || size_t(C4.rows()) != b4.nbf())
    throw std::invalid_argument("mo_integrals: orbital matrices do not match the bases");
  const Eigen::Index n1 = C1.cols(), n2 = C2.cols(), n3 = C3.cols(), n4 = C4.cols();
  const Eigen::Index n12 = n1 * n2, n34 = n3 * n4;
  const auto same = [](const BasisSet& x, const Matrix& X, const BasisSet& y, const Matrix& Y) {
    return &x == &y && X.cols() == Y.cols() && X == Y;
  };
  const bool s12 = same(b1, C1, b2, C2), s34 = same(b3, C3, b4, C4);
  const bool s1234 = same(b1, C1, b3, C3) && same(b2, C2, b4, C4);
  const size_t bytes = (threads() + 1) * size_t(n12 * n34) * sizeof(double);
  if (!resources::fits(bytes))
    throw std::runtime_error("MO integrals need " + resources::format_bytes(bytes) +
                             ", exceeds memory limit");
  const auto &sh1 = b1.shells(), &sh2 = b2.shells(), &sh3 = b3.shells(), &sh4 = b4.shells();
  const auto &o1 = b1.shell_offsets(), &o2 = b2.shell_offsets(), &o3 = b3.shell_offsets(),
             &o4 = b4.shell_offsets();
  const auto r1 = column_ranges(b1, C1), r2 = column_ranges(b2, C2),
             r3 = column_ranges(b3, C3), r4 = column_ranges(b4, C4);
  const auto used = [](const Range& r) { return r.first < r.second; };
  const size_t nprim =
      std::max({b1.max_nprim(), b2.max_nprim(), b3.max_nprim(), b4.max_nprim()});
  const int lmax = std::max({b1.max_l(), b2.max_l(), b3.max_l(), b4.max_l()});
  // shell pairs with nonzero coefficients on both shells; b <= a for a symmetric pair
  auto pairs = [&](const BasisSet& x, const BasisSet& y, const std::vector<Range>& rx,
                   const std::vector<Range>& ry, bool sym) {
    std::vector<std::pair<size_t, size_t>> ab;
    for (size_t a = 0; a < x.shells().size(); ++a)
      for (size_t b = 0; b < (sym ? a + 1 : y.shells().size()); ++b)
        if (used(rx[a]) && used(ry[b])) ab.emplace_back(a, b);
    return shell_pairs(op, x, y, std::move(ab), nprim, lmax);
  };
  // column blocks: block of a shell (column block (lo / block_size) % nblocks of its
  // nonzero columns; nblocks = any if they span several blocks), the allowed block tuples
  // (index nblocks = any), and the test of a quartet: some symmetry image allowed
  const bool filter = !blocks.empty() && block_size > 0 && nblocks > 0;
  const int nx = nblocks + 1;
  auto shell_blocks = [&](const std::vector<Range>& r) {
    std::vector<int> g(r.size(), nblocks);
    if (filter)
      for (size_t a = 0; a < r.size(); ++a) {
        const auto [lo, hi] = r[a];
        if (lo < hi && lo / block_size == (hi - 1) / block_size)
          g[a] = int((lo / block_size) % nblocks);
      }
    return g;
  };
  const auto g1 = shell_blocks(r1), g2 = shell_blocks(r2), g3 = shell_blocks(r3),
             g4 = shell_blocks(r4);
  std::vector<char> tab(filter ? nx * nx * nx * nx : 0, 0);
  for (int x = 0; x < int(tab.size()); ++x) {
    const int y[4] = {x / (nx * nx * nx), x / (nx * nx) % nx, x / nx % nx, x % nx};
    for (const auto& t : blocks) {
      bool m = true;
      for (int i = 0; i < 4; ++i) m = m && (y[i] == nblocks || y[i] == t[i]);
      if (m) tab[x] = 1;
    }
  }
  auto needed = [&](size_t a, size_t b, size_t c, size_t d) {
    if (!filter) return true;
    auto at = [&](int p, int q, int r, int t) { return tab[((p * nx + q) * nx + r) * nx + t]; };
    for (int i = 0; i < (s12 ? 2 : 1); ++i)
      for (int j = 0; j < (s34 ? 2 : 1); ++j) {
        const int p = i ? g1[b] : g1[a], q = i ? g2[a] : g2[b];
        const int r = j ? g3[d] : g3[c], t = j ? g4[c] : g4[d];
        if (at(p, q, r, t) || (s1234 && at(r, t, p, q))) return true;
      }
    return false;
  };
  const ShellPairs bra = pairs(b1, b2, r1, r2, s12);
  const ShellPairs ket = s1234 ? bra : pairs(b3, b4, r3, r4, s34);
  const double qmax = ket.q.empty() ? 0.0 : *std::max_element(ket.q.begin(), ket.q.end());
  const double eps = std::numeric_limits<double>::epsilon();
  std::vector<libint2::Engine> eng(threads(), two_body_engine(op, nprim, lmax));
  for (auto& e : eng) e.set_precision(eps);
  const Eigen::Index nf3 = b3.nbf();
  // Per bra shell pair (a b), k = (mu nu) in a x b (index x nb + y):
  //   I1(la, k n4 + N) = sum_si (mu nu|la si) C4_si,N  (both orderings of a symmetric ket)
  //   I2 = C3^T I1,  Z(k, L n4 + N) = I2(L, k n4 + N)
  //   Y(K n2 + M, L n4 + N) += sum_{mu nu} C1_mu,K C2_nu,M Z(k, .)  (both orderings if s12)
  // over the column ranges of the coefficients of each shell. s1234: only kets <= bra
  // (weight 1/2 on the diagonal), Y + Y^T at the end.
  Matrix Y = Matrix::Zero(n12, n34);
#pragma omp parallel
  {
    auto& e = eng[thread_id()];
    const auto& buf = e.results();
    Matrix Yl = Matrix::Zero(n12, n34);
    Eigen::MatrixXd I1, I2, P;
    Matrix Vt, Z, A, R;
    // Yl rows K n2 + M += sum_x C1_(p x),K sum_y C2_(q y),M Z(k(x, y)) for x in shell p of
    // b1, y in shell q of b2; k = x nb + y (p = a, q = b) or y nb + x (swapped, p = b, q = a)
    auto contract = [&](size_t p, size_t q, Eigen::Index nb, bool swap) {
      const Eigen::Index np = sh1[p].size(), nq = sh2[q].size();
      const auto [lo1, hi1] = r1[p];
      const auto [lo2, hi2] = r2[q];
      const Eigen::Index w2 = hi2 - lo2;
      A.resize(np, w2 * n34);  // A(x, (M - lo2) n34 + LN)
      for (Eigen::Index x = 0; x < np; ++x) {
        Eigen::Map<const Matrix, 0, Eigen::OuterStride<>> Zx(
            Z.data() + (swap ? x : x * nb) * n34, nq, n34,
            Eigen::OuterStride<>((swap ? nb : 1) * n34));
        Eigen::Map<Matrix>(A.row(x).data(), w2, n34).noalias() =
            C2.block(o2[q], lo2, nq, w2).transpose() * Zx;
      }
      R.noalias() = C1.block(o1[p], lo1, np, hi1 - lo1).transpose() * A;
      for (Eigen::Index K = lo1; K < hi1; ++K)
        Eigen::Map<Eigen::RowVectorXd>(Yl.row(K * n2 + lo2).data(), w2 * n34) +=
            R.row(K - lo1);
    };
#pragma omp for schedule(dynamic)
    for (size_t s = 0; s < bra.ab.size(); ++s) {
      const auto [a, b] = bra.ab[s];
      if (bra.q[s] * qmax < threshold) continue;
      const Eigen::Index na = sh1[a].size(), nb = sh2[b].size(), nab = na * nb;
      I1.setZero(nf3, nab * n4);
      bool any = false;
      const size_t nket = s1234 ? s + 1 : ket.ab.size();
      for (size_t t = 0; t < nket; ++t) {
        if (bra.q[s] * ket.q[t] < threshold) continue;
        const auto [c, d] = ket.ab[t];
        if (!needed(a, b, c, d)) continue;
        compute_quartet(e, op, sh1[a], sh2[b], sh3[c], sh4[d], bra.data[s].get(),
                        ket.data[t].get());
        if (buf[0] == nullptr) continue;
        any = true;
        const double w = s1234 && t == s ? 0.5 : 1.0;
        const Eigen::Index nc = sh3[c].size(), nd = sh4[d].size();
        Eigen::Map<const Matrix> V(buf[0], nab * nc, nd);  // rows (k, la), columns si
        {
          const auto [lo, hi] = r4[d];
          P.noalias() = w * V * C4.block(o4[d], lo, nd, hi - lo);
          for (Eigen::Index k = 0; k < nab; ++k)
            I1.block(o3[c], k * n4 + lo, nc, hi - lo) += P.middleRows(k * nc, nc);
        }
        if (s34 && c != d) {  // (mu nu|si la): la contracted, si free
          Vt.resize(nab * nd, nc);
          for (Eigen::Index k = 0; k < nab; ++k)
            Vt.middleRows(k * nd, nd) = V.middleRows(k * nc, nc).transpose();
          const auto [lo, hi] = r4[c];
          P.noalias() = w * Vt * C4.block(o4[c], lo, nc, hi - lo);
          for (Eigen::Index k = 0; k < nab; ++k)
            I1.block(o3[d], k * n4 + lo, nd, hi - lo) += P.middleRows(k * nd, nd);
        }
      }
      if (!any) continue;
      I2.noalias() = C3.transpose() * I1;  // (n3, nab n4)
      Z.resize(nab, n34);
      for (Eigen::Index k = 0; k < nab; ++k)
        for (Eigen::Index L = 0; L < n3; ++L)
          Z.row(k).segment(L * n4, n4) = I2.row(L).segment(k * n4, n4);
      contract(a, b, nb, false);
      if (s12 && a != b) contract(b, a, nb, true);
    }
#pragma omp critical
    Y += Yl;
  }
  Y *= op.scale;
  if (s1234) Y += Matrix(Y.transpose());
  return Y;
}

std::vector<double> two_body_4c(const TwoBodyOperator& op, const BasisSet& b1,
                                const BasisSet& b2, const BasisSet& b3, const BasisSet& b4) {
  const size_t n1 = b1.nbf(), n2 = b2.nbf(), n3 = b3.nbf(), n4 = b4.nbf();
  const size_t bytes = n1 * n2 * n3 * n4 * sizeof(double);
  if (!resources::fits(bytes))
    throw std::runtime_error("4-index integrals need " + resources::format_bytes(bytes) +
                             ", exceeds memory limit");
  std::vector<double> V(n1 * n2 * n3 * n4, 0.0);
  auto eng = engines(op, {&b1, &b2, &b3, &b4}, BraKet::xx_xx);
  const auto &s1 = b1.shells(), &s2 = b2.shells(), &s3 = b3.shells(), &s4 = b4.shells();
  const auto &o1 = b1.shell_offsets(), &o2 = b2.shell_offsets(), &o3 = b3.shell_offsets(),
             &o4 = b4.shell_offsets();
#pragma omp parallel for schedule(dynamic)
  for (size_t a = 0; a < s1.size(); ++a) {
    auto& e = eng[thread_id()];
    const auto& buf = e.results();
    for (size_t b = 0; b < s2.size(); ++b)
      for (size_t c = 0; c < s3.size(); ++c)
        for (size_t d = 0; d < s4.size(); ++d) {
          e.compute(s1[a], s2[b], s3[c], s4[d]);
          if (buf[0] == nullptr) continue;
          size_t q = 0;
          for (size_t i = 0; i < s1[a].size(); ++i)
            for (size_t j = 0; j < s2[b].size(); ++j)
              for (size_t k = 0; k < s3[c].size(); ++k)
                for (size_t l = 0; l < s4[d].size(); ++l, ++q) {
                  const size_t p = o1[a] + i, r = o2[b] + j, s = o3[c] + k, t = o4[d] + l;
                  V[((p * n2 + r) * n3 + s) * n4 + t] = op.scale * buf[0][q];
                }
        }
  }
  return V;
}

}  // namespace gemqc

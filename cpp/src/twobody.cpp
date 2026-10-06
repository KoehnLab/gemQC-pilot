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
                                int deriv) {
  init_libint();
  const double eps = std::numeric_limits<double>::epsilon();
  using K = TwoBodyOperator::Kind;
  if (op.kind == K::coulomb)
    return libint2::Engine(Operator::coulomb, max_nprim, max_l, deriv, eps);
  if (op.kind == K::delcgtga_delcgtgb) {
#ifdef GEMQC_HAVE_DELCGTGAB
    check_gtg(op.gtg);
    check_gtg(op.gtg2);
    return libint2::Engine(Operator::delcgtga_delcgtgb, max_nprim, max_l, deriv, eps,
                           std::array<libint2::ContractedGaussianGeminal, 2>{op.gtg, op.gtg2});
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
                           libint2::ContractedGaussianGeminal(op.gtg));
  }
  // libint's Slater/Yukawa core integrals fail (crash) for tiny exponents
  if (!(op.zeta >= 1e-3))
    throw std::invalid_argument("Slater-type operator needs zeta >= 1e-3");
  const auto oper = op.kind == K::stg ? Operator::stg : Operator::stg_x_coulomb;
  return libint2::Engine(oper, max_nprim, max_l, deriv, eps, op.zeta);
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
  std::vector<libint2::Engine> e(threads(), two_body_engine(op, nprim, l));
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
#pragma omp parallel for schedule(dynamic)
  for (size_t p = 0; p < sha.size(); ++p) {
    auto& e = eng[thread_id()];
    const auto& buf = e.results();
    const size_t np = sha[p].size();
    // H[i](nu, k) = sum_mu (P_i|mu nu) C1_{mu k}: first index transformed per aux function
    std::vector<Eigen::MatrixXd> H(np, Eigen::MatrixXd::Zero(nbf, n1));
    for (size_t a = 0; a < sh.size(); ++a)
      for (size_t b = 0; b <= a; ++b) {
        e.compute(sha[p], unit, sh[a], sh[b]);
        if (buf[0] == nullptr) continue;
        const size_t na = sh[a].size(), nb = sh[b].size();
        for (size_t i = 0; i < np; ++i) {
          Eigen::Map<const Matrix> blk(buf[0] + i * na * nb, na, nb);  // (mu in a, nu in b)
          H[i].middleRows(ob[b], nb).noalias() +=
              blk.transpose() * C1.middleRows(ob[a], na);
          if (a != b)
            H[i].middleRows(ob[a], na).noalias() += blk * C1.middleRows(ob[b], nb);
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
  // b1 = b2: (mu nu|la si) = (la si|mu nu); only kets (c d) <= bra (a b) are computed, the
  // partner (c d|a b) contributes K^ij(si, nu) += C_la,i C_mu,j v, collected as
  // L^ij(nu, si) = sum_mu C_mu,j H[(mu nu)](i, si) from the kets (c d) < (a b), added as L^T
  const bool sym = &b1 == &b2;
  std::vector<Matrix> L(sym ? np : 0, Matrix::Zero(n1, n2));
#pragma omp parallel for schedule(dynamic)
  for (size_t b = 0; b < sh1.size(); ++b) {  // nu shell: rows of the result owned here
    auto& e = eng[thread_id()];
    const auto& buf = e.results();
    const size_t nb = sh1[b].size();
    std::vector<Eigen::MatrixXd> Kb(np, Eigen::MatrixXd::Zero(nb, n2));
    std::vector<Eigen::MatrixXd> Lb(sym ? np : 0, Eigen::MatrixXd::Zero(nb, n2));
    for (size_t a = 0; a < sho.size(); ++a) {  // mu shell (orbital basis)
      if (q1(a, b) * qmax2 < threshold) continue;
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
        T.noalias() = C.middleRows(oo[c], nc).transpose() * W;  // one product per quartet
        for (size_t k = 0; k < nab; ++k) H[k].middleCols(o2[d], nd) += T.middleCols(k * nd, nd);
      };
      for (size_t c = 0; c < sho.size(); ++c)
        for (size_t d = 0; d < sh2.size(); ++d)
          if (!sym || c < a || (c == a && d < b)) add(c, d);
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

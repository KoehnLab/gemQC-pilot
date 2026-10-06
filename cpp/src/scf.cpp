#include "gemqc/scf.h"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <stdexcept>

#include "gemqc/fock.h"
#include "gemqc/hessian.h"
#include "gemqc/log.h"
#include "gemqc/resources.h"
#include "scf_detail.h"

namespace gemqc {

using namespace detail;

namespace {

// --- second-order (augmented Hessian) step ---
// Lowest eigenpair (mu, [1, x]) of [[0, g^T], [g, H]] in a subspace grown by
// preconditioned residuals: x solves (H - mu) x = -g with mu below the lowest
// eigenvalue of H, i.e. a descent step also with negative curvature. The step
// is scaled down to the trust radius; predicted: g^T x + x^T H x / 2.
struct AhStep {
  Vector x;
  double predicted = 0.0;
  int micro = 0;
};

AhStep ah_step(const Vector& g, const Vector& hdiag, const std::function<Vector(const Vector&)>& hv,
               double tol, int max_micro, double radius) {
  const Eigen::Index n = g.size();
  Matrix V(n, 0), HV(n, 0);
  Vector t = g.normalized(), y, x;
  AhStep res;
  for (int it = 1; it <= max_micro; ++it) {
    V.conservativeResize(n, it), HV.conservativeResize(n, it);
    V.col(it - 1) = t, HV.col(it - 1) = hv(t);
    res.micro = it;
    // augmented Hessian in the subspace
    Matrix Hs = V.transpose() * HV;
    Hs = (0.5 * (Hs + Hs.transpose())).eval();
    Matrix M = Matrix::Zero(it + 1, it + 1);
    M.block(1, 1, it, it) = Hs;
    M.block(1, 0, it, 1) = V.transpose() * g;
    M.block(0, 1, 1, it) = M.block(1, 0, it, 1).transpose();
    Eigen::SelfAdjointEigenSolver<Matrix> es(M);
    const double mu = es.eigenvalues()(0);
    const Vector c = es.eigenvectors().col(0);
    if (std::abs(c(0)) < 1e-8) break;  // degenerate: keep the previous step
    y = c.tail(it) / c(0);
    x = V * y;
    const Vector r = g + HV * y - mu * x;  // residual of (H - mu) x = -g
    if (r.norm() < tol) break;
    t = Vector(n);
    for (Eigen::Index p = 0; p < n; ++p) t(p) = -r(p) / std::max(hdiag(p) - mu, 1e-2);
    for (int pass = 0; pass < 2; ++pass) t -= V * (V.transpose() * t);
    if (t.norm() < 1e-12) break;
    t.normalize();
  }
  if (x.size() == 0) x = -g.cwiseQuotient(hdiag), y = V.transpose() * x;
  if (x.norm() > radius) {
    y *= radius / x.norm();
    x *= radius / x.norm();
  }
  const Matrix Hs = V.transpose() * HV;
  res.predicted = g.dot(x) + 0.5 * y.dot(Hs * y);
  res.x = x;
  return res;
}

// canonical orbitals: diagonalize the MO Fock matrix within occupied and
// virtual orbitals of each irrep; sort by energy
}  // namespace

// C <- C exp(K), K = [[0, -k^T], [k, 0]] in (occupied, virtual) blocks; exact via
// the SVD k = U s W^T: exp_oo = 1 + W(cos s - 1)W^T, exp_vv = 1 + U(cos s - 1)U^T,
// exp_vo = U sin s W^T, exp_ov = -W sin s U^T
Matrix rotate_orbitals(const Matrix& C, const Matrix& kappa, const std::vector<bool>& occ) {
  const auto o = indices(occ, true), v = indices(occ, false);
  const Eigen::Index no = o.size(), nv = v.size();
  if (no == 0 || nv == 0) return C;
  Matrix k(nv, no), Co(C.rows(), no), Cv(C.rows(), nv);
  for (Eigen::Index a = 0; a < nv; ++a)
    for (Eigen::Index i = 0; i < no; ++i) k(a, i) = kappa(v[a], o[i]);
  for (Eigen::Index i = 0; i < no; ++i) Co.col(i) = C.col(o[i]);
  for (Eigen::Index a = 0; a < nv; ++a) Cv.col(a) = C.col(v[a]);
  Eigen::BDCSVD<Matrix> svd(k, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Matrix& U = svd.matrixU();
  const Matrix& W = svd.matrixV();
  const Vector& sv = svd.singularValues();
  const Vector cm1 = sv.array().cos() - 1.0, sn = sv.array().sin();
  const Matrix Eoo = Matrix::Identity(no, no) + W * cm1.asDiagonal() * W.transpose();
  const Matrix Evv = Matrix::Identity(nv, nv) + U * cm1.asDiagonal() * U.transpose();
  const Matrix Evo = U * sn.asDiagonal() * W.transpose();
  const Matrix Co_new = Co * Eoo + Cv * Evo;
  const Matrix Cv_new = -Co * Evo.transpose() + Cv * Evv;
  Matrix Cn = C;
  for (Eigen::Index i = 0; i < no; ++i) Cn.col(o[i]) = Co_new.col(i);
  for (Eigen::Index a = 0; a < nv; ++a) Cn.col(v[a]) = Cv_new.col(a);
  return Cn;
}

size_t Wavefunction::n_occ() const {
  size_t n = 0;
  for (Eigen::Index i = 0; i < occ.size(); ++i) n += occ(i) > 0.0;
  return n;
}

namespace {

// SCF state of one spin set (RHF: one set, occupation 2; UHF: alpha and beta)
struct Spin {
  size_t nocc = 0;
  Matrix C, D, D_built, G, F;
  Vector eps;
  std::vector<int> irreps, occupation, fixed_occ;
  std::vector<bool> occupied;
};

// spin matrices stacked vertically (DIIS, L-BFGS work on all spins at once)
Matrix stack(const std::vector<Matrix>& m) {
  Matrix out(m[0].rows() * Eigen::Index(m.size()), m[0].cols());
  for (size_t s = 0; s < m.size(); ++s) out.middleRows(s * m[0].rows(), m[0].rows()) = m[s];
  return out;
}

// occupation per irrep: " 3 1 1 0" (RHF) or " 3 1 1 0 |  2 1 1 0" (UHF alpha | beta)
std::string occupation_text(const std::vector<Spin>& sp, bool fixed = false) {
  std::string t;
  for (size_t s = 0; s < sp.size(); ++s)
    t += (s ? "  |" : "") + occupation_string(fixed ? sp[s].fixed_occ : sp[s].occupation);
  return t;
}

// SCF for nspin = 1 (RHF) or 2 (UHF); guess: densities per spin (empty: core)
Wavefunction run_scf(const Molecule& mol, const BasisSet& basis, const ScfOptions& opt,
                     std::vector<Matrix> guess, const ScfSymmetry& sym,
                     const std::optional<Wavefunction>& start, int nspin,
                     std::shared_ptr<JKBuilder> jk) {
  using log::format;
  const auto t_start = Clock::now();
  mol.validate();
  const bool unrestricted = nspin == 2;
  const double f = unrestricted ? 1.0 : 2.0;  // electrons per occupied orbital
  const int nel = mol.n_electrons(), mult = mol.multiplicity();
  std::vector<Spin> sp(nspin);
  if (!unrestricted) {
    if (mult != 1 || nel % 2) throw std::invalid_argument("RHF requires a closed-shell singlet");
    sp[0].nocc = nel / 2;
  } else {
    if (mult < 1 || (nel + mult - 1) % 2 || mult - 1 > nel)
      throw std::invalid_argument(
          format("UHF: multiplicity %d is not possible with %d electrons", mult, nel));
    sp[0].nocc = (nel + mult - 1) / 2;
    sp[1].nocc = (nel - mult + 1) / 2;
  }
  const size_t n = basis.nbf();
  const char* spin_name[2] = {"alpha", "beta"};

  // symmetry blocks (C1: one block of all AOs)
  std::vector<std::string> irrep_names;
  const std::vector<Matrix> so = symmetry_blocks(sym, n, irrep_names);
  const size_t nirrep = so.size();
  if (!unrestricted && !sym.occupation_beta.empty())
    throw std::invalid_argument("beta occupation only for UHF");
  if (!sym.occupation_open.empty())
    throw std::invalid_argument("open-shell occupation only for ROHF");
  const std::vector<int>* occ_in[2] = {&sym.occupation, &sym.occupation_beta};
  for (int s = 0; s < nspin; ++s) {
    const auto& o = *occ_in[s];
    if (o.empty()) continue;
    if (o.size() != nirrep) throw std::invalid_argument("occupation needs one number per irrep");
    int tot = 0;
    for (int k : o) tot += k;
    if (tot != int(sp[s].nocc))
      throw std::invalid_argument(
          unrestricted ? format("%s occupation sums to %d, but %zu %s electrons", spin_name[s], tot,
                                sp[s].nocc, spin_name[s])
                       : format("occupation sums to %d, but %zu closed shells", tot, sp[s].nocc));
    sp[s].fixed_occ = o;  // may also be frozen during the iterations
  }
  bool occ_given = true;  // occupation of all spins fixed by input
  for (int s = 0; s < nspin; ++s) occ_given = occ_given && !occ_in[s]->empty();

  // one-electron part and orthogonalization X (canonical per irrep)
  const Matrix S = overlap(basis);
  const Matrix h = kinetic(basis) + nuclear_attraction(basis, mol);
  const Blocks blocks = orthogonalizer(S, so, opt.lindep);
  const Matrix& X = blocks.X;
  const Eigen::Index nmo = X.cols();
  for (const auto& x : sp)
    if (size_t(nmo) < x.nocc) throw std::runtime_error("too few orbitals for occupation");

  // DIIS subspace limited by memory (F and error vector per entry)
  int diis_size = opt.diis_size;
  const size_t mat = size_t(nspin) * nmo * nmo * sizeof(double);
  const size_t base = size_t(6 + 6 * nspin) * n * n * sizeof(double);
  while (diis_size > 2 && !resources::fits(base + 2 * diis_size * mat)) --diis_size;

  if (!jk) jk = std::make_shared<FockBuilder>(basis, opt.screening);
  JKBuilder& fb = *jk;
  if (sym.ao_perm.size() > 1) fb.set_symmetry(sym.ao_perm, sym.ao_sign, sym.characters);
  const bool incremental = opt.incremental && fb.exact();
  const size_t fock_mem = fb.memory_estimate(nspin, true, resources::threads());
  const std::string occ_label =
      unrestricted ? "alpha | beta orbitals per irrep:" : "closed shells per irrep:";

  log::line();
  if (unrestricted) {
    log::line("Unrestricted Hartree-Fock (direct SCF)");
    log::line(format("  basis functions %zu, orbitals %ld, alpha electrons %zu, beta electrons %zu",
                     n, long(nmo), sp[0].nocc, sp[1].nocc));
  } else {
    log::line("Restricted closed-shell Hartree-Fock (direct SCF)");
    log::line(format("  basis functions %zu, orbitals %ld, doubly occupied %zu",
                     n, long(nmo), sp[0].nocc));
  }
  log::line("  " + fb.describe());
  if (nmo < Eigen::Index(n))
    log::line(format("  %ld linearly dependent functions removed (S eigenvalue < %.1e)",
                     long(n - nmo), opt.lindep));
  if (nirrep > 1) {
    std::string names, sizes;
    for (size_t k = 0; k < nirrep; ++k)
      names += format(" %4s", irrep_names[k].c_str()), sizes += format(" %4ld", long(blocks.size[k]));
    log::line(format("  symmetry %s, irreps          %s", sym.group.c_str(), names.c_str()));
    log::line(format("  orbitals per irrep            %s", sizes.c_str()));
    if (fb.symmetry_order() > 1)
      log::line(format("  skeleton Fock build: symmetry-unique integrals only (group order %d)",
                       fb.symmetry_order()));
    for (int s = 0; s < nspin; ++s)
      if (!occ_in[s]->empty())
        log::line(format("  %s per irrep (fixed)%*s%s",
                         unrestricted ? (s ? "beta orbitals " : "alpha orbitals") : "closed shells",
                         unrestricted ? 1 : 0, "", occupation_string(*occ_in[s]).c_str()));
    if (!occ_given && sym.freeze_occupation > 0)
      log::line(format("  occupation per irrep by aufbau, frozen after iteration %d",
                       sym.freeze_occupation));
  }
  if (start) log::line("  start from given orbitals (occupation per irrep kept)");
  log::line(format("  convergence: rms(grad) < %.1e, |dE| < %.1e, max. %d iterations",
                   opt.convergence, opt.energy_convergence, opt.max_iterations));
  const bool use_qn = opt.solver == "qn", use_newton = opt.solver == "newton";
  const bool use_so = use_qn || use_newton;  // orbital-rotation solvers after DIIS
  if (!use_so && opt.solver != "diis")
    throw std::invalid_argument("unknown SCF solver '" + opt.solver + "' (diis, qn, newton)");
  std::string so_text;
  if (use_qn)
    so_text = format("quasi-Newton (L-BFGS, history %d, max. step %.2f)", opt.qn_history,
                     opt.qn_max_step);
  else if (use_newton)
    so_text = format("Newton (augmented Hessian, trust radius %.2f, max. %d micro-iterations)",
                     opt.trust_radius, opt.newton_micro);
  if (use_so && start)
    log::line("  " + so_text + " from iteration 1");
  else if (use_so) {
    log::line("  " + so_text);
    log::line(format("  DIIS before: switch from iteration %d with occupation per irrep stable for %d "
                     "iterations", std::max(2, opt.qn_start), opt.qn_stable));
  }
  if (use_so)
    log::line(format("  screening %.1e, %s Fock build", opt.screening,
                     incremental ? "incremental" : "full"));
  else
    log::line(format("  DIIS size %d%s, level shift %.3f, screening %.1e, %s Fock build",
                     diis_size, diis_size < opt.diis_size ? " (memory limit)" : "",
                     opt.level_shift, opt.screening,
                     incremental ? "incremental" : "full"));
  if (opt.energy_convergence < 1e3 * opt.screening)
    log::line(format("  WARNING: energy convergence %.1e is close to the integral screening "
                     "noise (screening %.1e)", opt.energy_convergence, opt.screening));
  log::line(format("  threads %d, memory estimate %s%s", resources::threads(),
                   resources::format_bytes(base + 2 * diis_size * mat + fock_mem).c_str(),
                   resources::memory()
                       ? (", limit " + resources::format_bytes(resources::memory())).c_str()
                       : ""));
  log::line();
  if (use_qn)
    log::line("  hist: DIIS vectors, or L-BFGS pairs (marked q); rej: step rejected (energy rose), "
              "step halved from the last accepted orbitals");
  if (use_newton)
    log::line("  hist: DIIS vectors, or Newton micro-iterations (marked n); rej: step rejected "
              "(energy rose), step halved");
  log::line(" iter        energy / Eh           dE / Eh    rms(grad)    max(grad)  hist"
            "    max|dD|  Fock  screened   t(Fock)/s   t(iter)/s");
  log::flush();

  Wavefunction wf{mol, basis};
  wf.method = unrestricted ? "UHF" : "RHF";

  // initial densities
  if (start) {  // given orbitals and occupation
    if (start->unrestricted() != unrestricted)
      throw std::invalid_argument("start orbitals belong to a different method");
    for (int s = 0; s < nspin; ++s) {
      const Matrix& C = s ? start->C_b : start->C;
      const Vector& occ = s ? start->occ_b : start->occ;
      const auto& irr = s ? start->irreps_b : start->irreps;
      if (C.rows() != Eigen::Index(n) || C.cols() != nmo || occ.size() != nmo ||
          irr.size() != size_t(nmo))
        throw std::invalid_argument("start orbitals have wrong dimension");
      auto& x = sp[s];
      x.C = C;
      x.irreps = irr;
      x.occupied.clear();
      for (Eigen::Index j = 0; j < nmo; ++j) x.occupied.push_back(occ(j) > 0.0);
      x.occupation = count_per_irrep(x.irreps, x.occupied, nirrep);
      x.fixed_occ = x.occupation;
      x.D = density(x.C, x.occupied, f);
    }
  } else if (!guess.empty()) {
    for (const auto& g : guess)
      if (g.rows() != Eigen::Index(n) || g.cols() != Eigen::Index(n))
        throw std::invalid_argument("guess density has wrong dimension");
    if (guess.size() == size_t(nspin)) {
      for (int s = 0; s < nspin; ++s) sp[s].D = guess[s];
    } else if (guess.size() == 1) {  // total density: split by electrons per spin
      for (int s = 0; s < nspin; ++s) sp[s].D = guess[0] * (double(sp[s].nocc) * f / nel);
    } else {
      throw std::invalid_argument("guess: one density (total) or one per spin");
    }
  } else {  // core Hamiltonian guess
    Matrix Co;
    Vector e;
    std::vector<int> irr;
    diagonalize(X.transpose() * h * X, blocks, Co, e, irr);
    for (auto& x : sp) x.D = density(X * Co, occupy(irr, x.nocc, x.fixed_occ, nirrep), f);
  }

  wf.e_nuc = mol.nuclear_repulsion();
  wf.group = nirrep > 1 ? sym.group : "C1";
  wf.irrep_names = irrep_names;
  for (auto& x : sp) x.G = x.D_built = Matrix::Zero(n, n);
  std::vector<std::vector<int>> occ_prev(nspin);
  Diis diis(diis_size);
  double e_old = 0.0, t_fock_total = 0.0, rms_old = 1.0;
  bool force_full = false;
  // QN/Newton state: accepted point (orbitals, energy, gradient) and last step,
  // matrices of all spins stacked
  const int so_first = std::max(2, opt.qn_start);
  const double so_tol = std::max(1e-9, 1e3 * opt.screening);  // energy noise
  Lbfgs lbfgs(opt.qn_history);
  std::vector<Matrix> C_acc(nspin);
  Matrix g_acc, step;
  double e_acc = 0.0;
  bool so_used = false;
  double radius = opt.trust_radius, predicted = 0.0;  // Newton trust region
  int micro = 0;
  int occ_stable = 0;  // DIIS iterations with unchanged occupation per irrep
  auto rotate_all = [&](const std::vector<Matrix>& C0, const Matrix& k) {
    for (int s = 0; s < nspin; ++s)
      sp[s].C = rotate_orbitals(C0[s], k.middleRows(s * nmo, nmo), sp[s].occupied);
  };

  for (int iter = 1; iter <= opt.max_iterations; ++iter) {
    const auto t_iter = Clock::now();

    // two-electron parts, full or incremental (accumulated screening errors:
    // periodic rebuilds and full builds near convergence);
    // RHF G = J - K/2, UHF G_s = J[D_a + D_b] - K[D_s]
    const bool full = !incremental || iter == 1 || force_full ||
                      (iter - 1) % opt.rebuild_every == 0 || rms_old < opt.incremental_until;
    std::vector<Matrix> dD;
    for (const auto& x : sp) dD.push_back(full ? x.D : Matrix(x.D - x.D_built));
    const auto t_f = Clock::now();
    auto jk = fb.compute(dD);
    const double t_fock = seconds_since(t_f);
    t_fock_total += t_fock;
    const double screened = fb.screened_percent();
    const Matrix J = unrestricted ? Matrix(jk.J[0] + jk.J[1]) : jk.J[0];
    wf.e_one = wf.e_two = 0.0;
    for (int s = 0; s < nspin; ++s) {
      auto& x = sp[s];
      const Matrix dG = J - (unrestricted ? 1.0 : 0.5) * jk.K[s];
      x.G = full ? dG : Matrix(x.G + dG);
      x.D_built = x.D;
      x.F = h + x.G;
      wf.e_one += x.D.cwiseProduct(h).sum();
      wf.e_two += 0.5 * x.D.cwiseProduct(x.G).sum();
    }
    const double energy = wf.e_one + wf.e_two + wf.e_nuc;
    const double de = iter > 1 ? energy - e_old : 0.0;
    e_old = energy;

    // orbital gradient: commutators (2/f)(F D S - S D F) in the orthonormal
    // basis (same scale for RHF and UHF)
    std::vector<Matrix> err;
    double sq = 0.0, gmax = 0.0;
    for (const auto& x : sp) {
      const Matrix FDS = x.F * x.D * S;
      err.push_back((2.0 / f) * X.transpose() * (FDS - FDS.transpose()) * X);
      sq += err.back().squaredNorm();
      gmax = std::max(gmax, err.back().cwiseAbs().maxCoeff());
    }
    const double rms = std::sqrt(sq / double(nspin * nmo * nmo));
    wf.energy = energy;
    wf.gradient_rms = rms;
    wf.iterations = iter;

    // convergence is only accepted with a full (exact) Fock build
    const bool criteria =
        rms < opt.convergence && (iter > 1 && std::abs(de) < opt.energy_convergence);
    const bool converged = criteria && full;
    force_full = criteria && !full;
    rms_old = rms;

    std::string note;
    double ddmax = 0.0;
    const bool so_now = use_so && (start.has_value() ||
                                   (iter >= so_first && (so_used || occ_stable >= opt.qn_stable ||
                                                         occ_given)));
    if (so_now) {
      // QN / Newton: occupation is fixed from here on
      if (!so_used) {
        for (auto& x : sp) x.fixed_occ = x.occupation;
        so_used = true;
      }
      if (converged) {
        // done: F belongs to the current orbitals; canonicalized below
      } else if (step.size() && energy > e_acc + so_tol && step.cwiseAbs().maxCoeff() > 1e-6) {
        // energy rose: back to the accepted orbitals with half the step
        step *= 0.5;
        predicted = 0.0;
        radius = 0.5 * step.norm();
        rotate_all(C_acc, step);
        note = "  rej";
      } else {
        // gradient 2f F_ai and diagonal Hessian estimate 2f (F_aa - F_ii) of
        // the symmetry-allowed rotations, per spin
        std::vector<Matrix> Fmo, masks, h0s, gs;
        std::vector<Pairs> pairs(nspin);
        for (int s = 0; s < nspin; ++s) {
          const auto& x = sp[s];
          Fmo.push_back(x.C.transpose() * x.F * x.C);
          Matrix mask = Matrix::Zero(nmo, nmo), h0 = Matrix::Ones(nmo, nmo);
          for (Eigen::Index a = 0; a < nmo; ++a)
            for (Eigen::Index i = 0; i < nmo; ++i)
              if (!x.occupied[a] && x.occupied[i] && x.irreps[a] == x.irreps[i]) {
                mask(a, i) = 1.0;
                h0(a, i) = std::max(2.0 * f * (Fmo[s](a, a) - Fmo[s](i, i)), 0.1 * f);
                pairs[s].push_back({a, i});
              }
          gs.push_back(2.0 * f * Fmo[s].cwiseProduct(mask));
          masks.push_back(mask);
          h0s.push_back(h0);
        }
        const Matrix g = stack(gs), h0 = stack(h0s), mask = stack(masks);
        if (use_newton) {
          // trust radius from the ratio of actual and predicted energy change
          if (predicted < 0.0) {
            const double ratio = (energy - e_acc) / predicted;
            if (ratio < 0.25) radius = std::max(0.5 * step.norm(), 1e-3);
            else if (ratio > 0.75 && step.norm() > 0.9 * radius)
              radius = std::min(2.0 * radius, 2.0 * opt.trust_radius);
          }
          // augmented Hessian step with the exact energy Hessian (direct
          // products); micro-iterations converged to |r| < min(0.1, |g|) |g|
          // (superlinear, quadratic near convergence)
          std::vector<Vector> gv, hv0;
          Eigen::Index np = 0;
          for (int s = 0; s < nspin; ++s) {
            gv.push_back(matrix_to_pairs(pairs[s], gs[s]));
            hv0.push_back(matrix_to_pairs(pairs[s], h0s[s]));
            np += gv.back().size();
          }
          Vector gvec(np), hd(np);
          for (int s = 0, o = 0; s < nspin; o += int(gv[s].size()), ++s)
            gvec.segment(o, gv[s].size()) = gv[s], hd.segment(o, gv[s].size()) = hv0[s];
          std::vector<Matrix> Cs;
          for (const auto& x : sp) Cs.push_back(x.C);
          auto hv = [&](const Vector& v) { return energy_hessian_product(fb, Cs, Fmo, pairs, v); };
          const double gn = gvec.norm();
          const auto ah = ah_step(gvec, hd, hv, std::max(std::min(0.1, gn) * gn, 1e-12),
                                  opt.newton_micro, radius);
          std::vector<Matrix> st;
          for (int s = 0, o = 0; s < nspin; o += int(pairs[s].size()), ++s)
            st.push_back(pairs_to_matrix(pairs[s], ah.x.segment(o, pairs[s].size()), nmo));
          step = stack(st);
          predicted = ah.predicted;
          micro = ah.micro;
        } else {
          if (step.size()) lbfgs.push(step, g - g_acc);
          step = lbfgs.step(g, h0, mask);
          const double smax = step.cwiseAbs().maxCoeff();
          if (smax > opt.qn_max_step) step *= opt.qn_max_step / smax;
        }
        for (int s = 0; s < nspin; ++s) C_acc[s] = sp[s].C;
        g_acc = g, e_acc = energy;
        rotate_all(C_acc, step);
      }
      if (!converged)
        for (auto& x : sp) {
          const Matrix D_new = density(x.C, x.occupied, f);
          ddmax = std::max(ddmax, (D_new - x.D).cwiseAbs().maxCoeff());
          x.D = D_new;
        }
    } else {
      // new orbitals from (extrapolated, shifted) Fock matrices
      std::vector<Matrix> Fo;
      for (const auto& x : sp) Fo.push_back(X.transpose() * x.F * X);
      if (diis_size > 1 && iter >= opt.diis_start && !converged) {
        const Matrix fx = diis.extrapolate(stack(Fo), stack(err));
        for (int s = 0; s < nspin; ++s) Fo[s] = fx.middleRows(s * nmo, nmo);
      }
      bool stable = true;
      for (int s = 0; s < nspin; ++s) {
        auto& x = sp[s];
        if (opt.level_shift > 0.0 && !converged) {
          const Matrix Po = X.transpose() * S * x.D * S * X / f;  // occupied projector
          Fo[s] += opt.level_shift * (Matrix::Identity(nmo, nmo) - Po);
        }
        Matrix Co;
        diagonalize(Fo[s], blocks, Co, x.eps, x.irreps);
        x.C = X * Co;
        x.occupied = occupy(x.irreps, x.nocc, x.fixed_occ, nirrep);
        x.occupation = count_per_irrep(x.irreps, x.occupied, nirrep);
        if (sym.freeze_occupation > 0 && iter == sym.freeze_occupation && x.fixed_occ.empty())
          x.fixed_occ = x.occupation;
        stable = stable && x.occupation == occ_prev[s];
        const Matrix D_new = density(x.C, x.occupied, f);
        ddmax = std::max(ddmax, (D_new - x.D).cwiseAbs().maxCoeff());
        x.D = D_new;
      }
      occ_stable = stable ? occ_stable + 1 : 1;
    }

    const int hist = so_now ? (use_newton ? micro : lbfgs.size()) : diis.size();
    log::line(format("%5d  %20.12f  %14.6e  %11.3e  %11.3e  %3d%s  %10.3e  %4s  %7.1f%%  %10.2f  %10.2f%s",
                     iter, energy, de, rms, gmax, hist, so_now ? (use_newton ? "n" : "q") : " ",
                     ddmax, full ? "full" : "incr", screened, t_fock, seconds_since(t_iter),
                     note.c_str()));
    micro = 0;
    bool changed = false;
    for (int s = 0; s < nspin; ++s) changed = changed || sp[s].occupation != occ_prev[s];
    if (nirrep > 1 && changed) log::line("       " + occ_label + occupation_text(sp));
    log::flush();
    for (int s = 0; s < nspin; ++s) occ_prev[s] = sp[s].occupation;
    if (converged) {
      wf.converged = true;
      break;
    }
  }

  // canonical orbitals (after QN/Newton), occupation numbers, HOMO/LUMO
  double homo[2] = {-1e300, -1e300}, lumo[2] = {1e300, 1e300};
  std::vector<Vector> occs;
  for (int s = 0; s < nspin; ++s) {
    auto& x = sp[s];
    if (so_used) {
      if (x.eps.size() != nmo) x.eps.resize(nmo);
      canonicalize(x.F, x.C, x.eps, x.irreps, x.occupied, nirrep);
    }
    occs.push_back(Vector::Zero(nmo));
    for (Eigen::Index j = 0; j < nmo; ++j) {
      if (x.occupied[j]) occs[s](j) = f, homo[s] = std::max(homo[s], x.eps(j));
      else lumo[s] = std::min(lumo[s], x.eps(j));
    }
  }
  wf.C = sp[0].C, wf.eps = sp[0].eps, wf.occ = occs[0];
  wf.irreps = sp[0].irreps, wf.occupation = sp[0].occupation;
  wf.D = sp[0].D;
  if (unrestricted) {
    wf.C_b = sp[1].C, wf.eps_b = sp[1].eps, wf.occ_b = occs[1];
    wf.irreps_b = sp[1].irreps, wf.occupation_b = sp[1].occupation;
    wf.D = sp[0].D + sp[1].D;
    wf.D_spin = sp[0].D - sp[1].D;
    // <S^2> = Sz(Sz+1) + N_beta - sum_ij |<i_alpha|j_beta>|^2
    const double sz = 0.5 * (double(sp[0].nocc) - double(sp[1].nocc));
    Matrix Ca(n, sp[0].nocc), Cb(n, sp[1].nocc);
    for (size_t j = 0, a = 0, b = 0; j < size_t(nmo); ++j) {
      if (sp[0].occupied[j]) Ca.col(a++) = sp[0].C.col(j);
      if (sp[1].occupied[j]) Cb.col(b++) = sp[1].C.col(j);
    }
    wf.s2_exact = sz * (sz + 1.0);
    wf.s2 = wf.s2_exact + double(sp[1].nocc) - (Ca.transpose() * S * Cb).squaredNorm();
    // natural orbitals: eigenvectors of the total density (S metric) per irrep
    struct Entry { double occ; int k; Vector c; };
    std::vector<Entry> all;
    for (size_t k = 0; k < nirrep; ++k) {
      if (blocks.size[k] == 0) continue;
      const Matrix Xk = X.middleCols(blocks.start[k], blocks.size[k]);
      Eigen::SelfAdjointEigenSolver<Matrix> es(Xk.transpose() * S * wf.D * S * Xk);
      const Matrix Cn = Xk * es.eigenvectors();
      for (Eigen::Index a = 0; a < blocks.size[k]; ++a)
        all.push_back({es.eigenvalues()(a), int(k), Cn.col(a)});
    }
    std::stable_sort(all.begin(), all.end(), [](auto& a, auto& b) { return a.occ > b.occ; });
    wf.C_nat = Matrix(n, nmo), wf.occ_nat = Vector(nmo), wf.irreps_nat.resize(nmo);
    for (size_t j = 0; j < all.size(); ++j)
      wf.C_nat.col(j) = all[j].c, wf.occ_nat(j) = all[j].occ, wf.irreps_nat[j] = all[j].k;
  }

  // summary
  log::line();
  if (wf.converged)
    log::line(format("  SCF converged in %d iterations", wf.iterations));
  else
    log::line(format("  WARNING: SCF not converged after %d iterations", wf.iterations));
  log::line(format("  one-electron energy      %20.12f Eh", wf.e_one));
  log::line(format("  two-electron energy      %20.12f Eh", wf.e_two));
  log::line(format("  nuclear repulsion        %20.12f Eh", wf.e_nuc));
  log::line(format("  total energy             %20.12f Eh", wf.energy));
  if (unrestricted)
    log::line(format("  <S^2>                    %20.12f     (S(S+1) = %.6f)", wf.s2, wf.s2_exact));
  for (int s = 0; s < nspin; ++s) {
    const std::string lab = unrestricted ? format("%-5s  ", spin_name[s]) : "";
    const bool has_lumo = sp[s].nocc < size_t(nmo), has_homo = sp[s].nocc > 0;
    log::line(format("  %sHOMO %14s%s", lab.c_str(),
                     has_homo ? format("%.6f Eh", homo[s]).c_str() : "-",
                     has_lumo ? format("   LUMO %14.6f Eh   gap %10.6f Eh", lumo[s],
                                       lumo[s] - homo[s]).c_str()
                              : ""));
    if (has_homo && lumo[s] < homo[s])
      log::line(format("  NOTE: %soccupation is not aufbau (LUMO below HOMO)",
                       unrestricted ? (std::string(spin_name[s]) + " ").c_str() : ""));
  }
  log::line(format("  time: Fock builds %.2f s, total %.2f s", t_fock_total,
                   seconds_since(t_start)));
  return wf;
}

}  // namespace

Wavefunction rhf(const Molecule& mol, const BasisSet& basis, const ScfOptions& opt,
                 const std::optional<Matrix>& guess, const ScfSymmetry& sym,
                 const std::optional<Wavefunction>& start, std::shared_ptr<JKBuilder> jk) {
  std::vector<Matrix> g;
  if (guess) g.push_back(*guess);
  return run_scf(mol, basis, opt, g, sym, start, 1, std::move(jk));
}

Wavefunction uhf(const Molecule& mol, const BasisSet& basis, const ScfOptions& opt,
                 const std::vector<Matrix>& guess, const ScfSymmetry& sym,
                 const std::optional<Wavefunction>& start, std::shared_ptr<JKBuilder> jk) {
  return run_scf(mol, basis, opt, guess, sym, start, 2, std::move(jk));
}

}  // namespace gemqc

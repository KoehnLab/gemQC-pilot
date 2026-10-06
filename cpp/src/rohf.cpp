// High-spin ROHF with a general open-shell energy expression (Roothaan,
// McWeeny): orbital shells k with occupation fraction f_k (2 f_k electrons per
// orbital; virtual: f = 0) and coupling coefficients a_kl, b_kl,
//   E = sum_k 2 f_k tr(D_k h) + sum_kl f_k f_l tr(D_k (2 a_kl J_l - b_kl K_l)),
// D_k = sum_{i in k} c_i c_i^T. Shell Fock operators
//   F_k = f_k (h + sum_l f_l (2 a_kl J_l - b_kl K_l)),
// gradient of a rotation mixing q (shell k) with p (shell l): 4 (F_k - F_l)_pq.
// DIIS uses the effective Fock matrix with diagonal blocks F_T = h + J[P] - K[P]/2
// (P total density) and off-diagonal blocks (F_k - F_l)/(f_k - f_l), for
// high-spin ROHF those of Guest and Saunders (closed-open F_beta, open-virtual
// F_alpha, closed-virtual (F_alpha + F_beta)/2); its eigenvalues keep the aufbau
// order. The final canonical orbitals diagonalize F_k / f_k within each shell
// (virtual: operator of the highest occupied shell), as in Molpro: closed
// (F_alpha + F_beta)/2, open and virtual F_alpha.

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>

#include "gemqc/fock.h"
#include "gemqc/log.h"
#include "gemqc/resources.h"
#include "gemqc/scf.h"
#include "scf_detail.h"

namespace gemqc {

using namespace detail;

namespace {

// open-shell energy expression: occupied shells with fractions and couplings
struct ShellModel {
  std::vector<double> f;
  Matrix a, b;
  std::vector<std::string> names;
};

// start orbitals with their shells (0 closed, 1 open/active, 2 virtual)
struct StartOrbitals {
  Matrix C;
  std::vector<int> irreps, shell;
};

// high-spin ROHF: closed shells (f = 1) and parallel-spin open shells (f = 1/2)
ShellModel high_spin() {
  ShellModel m;
  m.f = {1.0, 0.5};
  m.a = Matrix::Ones(2, 2);
  m.b = Matrix::Ones(2, 2);
  m.b(1, 1) = 2.0;
  m.names = {"closed", "open"};
  return m;
}

// exp(K) of an antisymmetric matrix (scaling and squaring of the Taylor series)
Matrix expm(const Matrix& K) {
  const Eigen::Index n = K.rows();
  const double norm = K.cwiseAbs().colwise().sum().maxCoeff();
  int squarings = 0;
  while (norm / std::ldexp(1.0, squarings) > 0.25) ++squarings;
  const Matrix A = K / std::ldexp(1.0, squarings);
  Matrix E = Matrix::Identity(n, n), T = Matrix::Identity(n, n);
  for (int j = 1; j <= 12; ++j) {
    T = (T * A / double(j)).eval();
    E += T;
  }
  for (int i = 0; i < squarings; ++i) E = (E * E).eval();
  return E;
}

// shell of each orbital (0 closed, 1 open, 2 virtual): the lowest orbitals
// (energy order) or the lowest ones per irrep for fixed counts
std::vector<int> assign_shells(const std::vector<int>& irrep, size_t nclosed, size_t nopen,
                               const std::vector<int>& fixed_closed,
                               const std::vector<int>& fixed_open, size_t nirrep) {
  std::vector<int> shell(irrep.size(), 2);
  const std::vector<int>* fixed[2] = {&fixed_closed, &fixed_open};
  const size_t count[2] = {nclosed, nopen};
  for (int k = 0; k < 2; ++k) {
    if (fixed[k]->empty()) {
      size_t left = count[k];
      for (size_t j = 0; j < irrep.size() && left > 0; ++j)
        if (shell[j] == 2) shell[j] = k, --left;
      if (left > 0) throw std::runtime_error("too few orbitals for occupation");
    } else {
      std::vector<int> left(*fixed[k]);
      for (size_t j = 0; j < irrep.size(); ++j)
        if (shell[j] == 2 && left[irrep[j]] > 0) shell[j] = k, --left[irrep[j]];
      for (size_t g = 0; g < nirrep; ++g)
        if (left[g] > 0) throw std::runtime_error("not enough orbitals in irrep for occupation");
    }
  }
  return shell;
}

std::vector<int> shell_counts(const std::vector<int>& irrep, const std::vector<int>& shell,
                              int k, size_t nirrep) {
  std::vector<int> n(nirrep, 0);
  for (size_t j = 0; j < irrep.size(); ++j) n[irrep[j]] += shell[j] == k;
  return n;
}

// SCF for a closed and an open shell (ROHF)
Wavefunction open_shell_scf(const Molecule& mol, const BasisSet& basis, const ScfOptions& opt,
                            const ShellModel& model, size_t nclosed, size_t nopen,
                            const std::string& method, const std::vector<Matrix>& guess,
                            const ScfSymmetry& sym, const StartOrbitals* start,
                            std::shared_ptr<JKBuilder> jk) {
  using log::format;
  const auto t_start = Clock::now();
  const int nshell = int(model.f.size());  // occupied shells; virtual = nshell
  const size_t n = basis.nbf();

  std::vector<std::string> irrep_names;
  const std::vector<Matrix> so = symmetry_blocks(sym, n, irrep_names);
  const size_t nirrep = so.size();
  if (!sym.occupation_beta.empty()) throw std::invalid_argument("beta occupation only for UHF");
  std::vector<int> fixed[2] = {sym.occupation, sym.occupation_open};
  const size_t count[2] = {nclosed, nopen};
  for (int k = 0; k < 2; ++k) {
    if (fixed[k].empty()) continue;
    if (fixed[k].size() != nirrep) throw std::invalid_argument("occupation needs one number per irrep");
    int tot = 0;
    for (int x : fixed[k]) tot += x;
    if (tot != int(count[k]))
      throw std::invalid_argument(format("%s occupation sums to %d, but %zu %s shells",
                                         model.names[k].c_str(), tot, count[k],
                                         model.names[k].c_str()));
  }
  const bool occ_given = !fixed[0].empty() && !fixed[1].empty();

  const Matrix S = overlap(basis);
  const Matrix h = kinetic(basis) + nuclear_attraction(basis, mol);
  const Blocks blocks = orthogonalizer(S, so, opt.lindep);
  const Matrix& X = blocks.X;
  const Eigen::Index nmo = X.cols();
  if (size_t(nmo) < nclosed + nopen) throw std::runtime_error("too few orbitals for occupation");

  const bool use_qn = opt.solver == "qn";
  if (opt.solver == "newton")
    throw std::invalid_argument("the Newton solver is not available for " + method + " (diis, qn)");
  if (!use_qn && opt.solver != "diis")
    throw std::invalid_argument("unknown SCF solver '" + opt.solver + "' (diis, qn)");
  int diis_size = opt.diis_size;
  const size_t mat = size_t(nmo) * nmo * sizeof(double);
  const size_t base = size_t(10 + 4 * nshell) * n * n * sizeof(double);
  while (diis_size > 2 && !resources::fits(base + 2 * diis_size * mat)) --diis_size;
  if (!jk) jk = std::make_shared<FockBuilder>(basis, opt.screening);
  JKBuilder& fb = *jk;
  if (sym.ao_perm.size() > 1) fb.set_symmetry(sym.ao_perm, sym.ao_sign, sym.characters);
  const bool incremental = opt.incremental && fb.exact();
  const size_t fock_mem = fb.memory_estimate(nshell, true, resources::threads());

  // header
  log::line();
  log::line("Restricted open-shell Hartree-Fock (high spin, direct SCF)");
  log::line(format("  basis functions %zu, orbitals %ld, closed shells %zu, %s orbitals %zu",
                   n, long(nmo), nclosed, model.names[1].c_str(), nopen));
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
    for (int k = 0; k < 2; ++k)
      if (!fixed[k].empty())
        log::line(format("  %-6s orbitals per irrep (fixed) %s", model.names[k].c_str(),
                         occupation_string(fixed[k]).c_str()));
    if (!occ_given && sym.freeze_occupation > 0)
      log::line(format("  occupation per irrep by aufbau, frozen after iteration %d",
                       sym.freeze_occupation));
  }
  if (start) log::line("  start from given orbitals (occupation per irrep kept)");
  log::line(format("  convergence: rms(grad) < %.1e, |dE| < %.1e, max. %d iterations",
                   opt.convergence, opt.energy_convergence, opt.max_iterations));
  if (use_qn) {
    log::line(format("  quasi-Newton (L-BFGS, history %d, max. step %.2f)%s", opt.qn_history,
                     opt.qn_max_step, start ? " from iteration 1" : ""));
    if (!start)
      log::line(format("  DIIS before: switch from iteration %d with occupation per irrep stable "
                       "for %d iterations", std::max(2, opt.qn_start), opt.qn_stable));
    log::line(format("  screening %.1e, %s Fock build", opt.screening,
                     incremental ? "incremental" : "full"));
  } else {
    log::line(format("  DIIS size %d%s (effective Fock matrix), level shift %.3f, screening %.1e, "
                     "%s Fock build", diis_size, diis_size < opt.diis_size ? " (memory limit)" : "",
                     opt.level_shift, opt.screening, incremental ? "incremental" : "full"));
  }
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
  log::line(" iter        energy / Eh           dE / Eh    rms(grad)    max(grad)  hist"
            "    max|dD|  Fock  screened   t(Fock)/s   t(iter)/s");
  log::flush();

  auto f_of = [&](int k) { return k < nshell ? model.f[k] : 0.0; };

  // initial orbitals: given, from the Fock matrix of guess densities, or core
  Matrix C;
  Vector eps;
  std::vector<int> irreps, shell;
  if (start) {
    if (start->C.rows() != Eigen::Index(n) || start->C.cols() != nmo ||
        start->shell.size() != size_t(nmo) || start->irreps.size() != size_t(nmo))
      throw std::invalid_argument("start orbitals have wrong dimension");
    C = start->C, irreps = start->irreps, shell = start->shell;
    for (int k = 0; k < 2; ++k) {
      size_t cnt = 0;
      for (int x : shell) cnt += x == k;
      if (cnt != (k ? nopen : nclosed))
        throw std::invalid_argument(format("start orbitals: %zu %s orbitals, expected %zu", cnt,
                                           model.names[k].c_str(), k ? nopen : nclosed));
    }
    for (int g : irreps)
      if (g < 0 || size_t(g) >= nirrep) throw std::invalid_argument("start orbitals: invalid irrep");
    fixed[0] = shell_counts(irreps, shell, 0, nirrep);
    fixed[1] = shell_counts(irreps, shell, 1, nirrep);
  } else {
    Matrix F0 = X.transpose() * h * X;  // orthonormal basis
    for (const auto& g : guess)
      if (g.rows() != Eigen::Index(n) || g.cols() != Eigen::Index(n))
        throw std::invalid_argument("guess density has wrong dimension");
    if (guess.size() == 1) {  // total density: its (RHF-like) Fock matrix
      auto jk = fb.compute({guess[0]});
      F0 += X.transpose() * (jk.J[0] - 0.5 * jk.K[0]) * X;
    } else if (guess.size() == size_t(nshell)) {
      // shell densities: effective Fock matrix from shell projectors R_k
      auto jk = fb.compute(guess);
      std::vector<Matrix> Fk(nshell + 1, Matrix::Zero(nmo, nmo)), R(nshell + 1);
      Matrix FT = h;
      R[nshell] = Matrix::Identity(nmo, nmo);
      for (int k = 0; k < nshell; ++k) {
        Matrix G = Matrix::Zero(n, n);
        for (int l = 0; l < nshell; ++l)
          G += model.f[l] * (2.0 * model.a(k, l) * jk.J[l] - model.b(k, l) * jk.K[l]);
        Fk[k] = X.transpose() * (model.f[k] * (h + G)) * X;
        FT += model.f[k] * (2.0 * jk.J[k] - jk.K[k]);
        R[k] = X.transpose() * S * guess[k] * S * X;
        R[nshell] -= R[k];
      }
      FT = (X.transpose() * FT * X).eval();
      F0 = Matrix::Zero(nmo, nmo);
      for (int k = 0; k <= nshell; ++k) {
        F0 += R[k] * FT * R[k];
        for (int l = k + 1; l <= nshell; ++l) {
          const Matrix Wkl = (Fk[k] - Fk[l]) / (f_of(k) - f_of(l));
          F0 += R[k] * Wkl * R[l] + R[l] * Wkl.transpose() * R[k];
        }
      }
    } else if (!guess.empty()) {
      throw std::invalid_argument("guess: total density or one density per shell (closed, open)");
    }
    Matrix Co;
    diagonalize(F0, blocks, Co, eps, irreps);
    C = X * Co;
    shell = assign_shells(irreps, nclosed, nopen, fixed[0], fixed[1], nirrep);
  }

  Wavefunction wf{mol, basis};
  wf.method = method;
  wf.e_nuc = mol.nuclear_repulsion();
  wf.group = nirrep > 1 ? sym.group : "C1";
  wf.irrep_names = irrep_names;
  auto shell_density = [&](const Matrix& Cm, int k) {
    Matrix D = Matrix::Zero(n, n);
    for (Eigen::Index j = 0; j < nmo; ++j)
      if (shell[j] == k) D.noalias() += Cm.col(j) * Cm.col(j).transpose();
    return D;
  };
  std::vector<Matrix> D(nshell), D_built(nshell, Matrix::Zero(n, n)), J(nshell), K(nshell);
  for (int k = 0; k < nshell; ++k) D[k] = shell_density(C, k);

  Diis diis(diis_size);
  Lbfgs lbfgs(opt.qn_history);
  Matrix C_acc, g_acc, step;
  std::vector<Matrix> Fsh(nshell);
  std::vector<int> occ_prev[2];
  double e_old = 0.0, e_acc = 0.0, t_fock_total = 0.0, rms_old = 1.0;
  bool force_full = false, qn_used = false;
  int occ_stable = 0;
  const int qn_first = std::max(2, opt.qn_start);
  const double qn_tol = std::max(1e-9, 1e3 * opt.screening);

  for (int iter = 1; iter <= opt.max_iterations; ++iter) {
    const auto t_iter = Clock::now();
    // J and K of the shell densities (full or incremental)
    const bool full = !incremental || iter == 1 || force_full ||
                      (iter - 1) % opt.rebuild_every == 0 || rms_old < opt.incremental_until;
    std::vector<Matrix> dD;
    for (int k = 0; k < nshell; ++k) dD.push_back(full ? D[k] : Matrix(D[k] - D_built[k]));
    const auto t_f = Clock::now();
    auto jk = fb.compute(dD);
    const double t_fock = seconds_since(t_f);
    t_fock_total += t_fock;
    const double screened = fb.screened_percent();
    for (int k = 0; k < nshell; ++k) {
      J[k] = full ? jk.J[k] : Matrix(J[k] + jk.J[k]);
      K[k] = full ? jk.K[k] : Matrix(K[k] + jk.K[k]);
      D_built[k] = D[k];
    }

    // shell Fock operators, energy, total-density Fock matrix F_T
    Matrix FT = h;
    double e = wf.e_nuc;
    wf.e_one = 0.0;
    for (int k = 0; k < nshell; ++k) {
      Matrix G = Matrix::Zero(n, n);
      for (int l = 0; l < nshell; ++l)
        G += model.f[l] * (2.0 * model.a(k, l) * J[l] - model.b(k, l) * K[l]);
      Fsh[k] = model.f[k] * (h + G);
      FT += model.f[k] * (2.0 * J[k] - K[k]);
      wf.e_one += 2.0 * model.f[k] * D[k].cwiseProduct(h).sum();
      e += D[k].cwiseProduct(model.f[k] * h + Fsh[k]).sum();
    }
    const double energy = e;
    wf.e_two = energy - wf.e_one - wf.e_nuc;
    const double de = iter > 1 ? energy - e_old : 0.0;
    e_old = energy;

    // MO quantities: effective Fock matrix W and gradient (p in a higher shell
    // than q, same irrep): g_pq = 4 (F_k - F_l)_pq
    std::vector<Matrix> Fmo;
    for (int k = 0; k < nshell; ++k) Fmo.push_back(C.transpose() * Fsh[k] * C);
    Matrix W = C.transpose() * FT * C, err_mo = Matrix::Zero(nmo, nmo),
           g = Matrix::Zero(nmo, nmo), mask = Matrix::Zero(nmo, nmo), h0 = Matrix::Ones(nmo, nmo);
    for (Eigen::Index p = 0; p < nmo; ++p)
      for (Eigen::Index q = 0; q < nmo; ++q) {
        const int l = shell[p], k = shell[q];
        if (l == k) continue;
        if (irreps[p] != irreps[q]) { W(p, q) = 0.0; continue; }
        const int lo = std::min(k, l), hi = std::max(k, l);
        auto fd = [&](Eigen::Index r, Eigen::Index s) {  // (F_lo - F_hi)_rs
          return (lo < nshell ? Fmo[lo](r, s) : 0.0) - (hi < nshell ? Fmo[hi](r, s) : 0.0);
        };
        const double df = f_of(lo) - f_of(hi);
        W(p, q) = fd(p, q) / (df != 0.0 ? df : 1.0);
        if (l > k) {  // rotation parameter kappa_pq: q in shell k mixes with p
          mask(p, q) = 1.0;
          g(p, q) = 4.0 * fd(p, q);
          err_mo(p, q) = 2.0 * W(p, q), err_mo(q, p) = -2.0 * W(p, q);
          h0(p, q) = std::max(4.0 * (fd(p, p) - fd(q, q)), 0.2);
        }
      }
    const Matrix Co = X.transpose() * S * C;  // orbitals in the orthonormal basis
    const Matrix err = Co * err_mo * Co.transpose();
    const double rms = std::sqrt(err.squaredNorm() / double(nmo * nmo));
    const double gmax = err.cwiseAbs().maxCoeff();
    wf.energy = energy;
    wf.gradient_rms = rms;
    wf.iterations = iter;
    const bool criteria =
        rms < opt.convergence && (iter > 1 && std::abs(de) < opt.energy_convergence);
    const bool converged = criteria && full;
    force_full = criteria && !full;
    rms_old = rms;

    std::string note;
    double ddmax = 0.0;
    const bool qn_now = use_qn && (start != nullptr ||
                                   (iter >= qn_first && (qn_used || occ_stable >= opt.qn_stable ||
                                                         occ_given)));
    if (qn_now) {
      if (!qn_used) {
        fixed[0] = shell_counts(irreps, shell, 0, nirrep);
        fixed[1] = shell_counts(irreps, shell, 1, nirrep);
        qn_used = true;
      }
      if (converged) {
      } else if (step.size() && energy > e_acc + qn_tol && step.cwiseAbs().maxCoeff() > 1e-6) {
        step *= 0.5;
        C = C_acc * expm(step - step.transpose());
        note = "  rej";
      } else {
        if (step.size()) lbfgs.push(step, g - g_acc);
        step = lbfgs.step(g, h0, mask);
        const double smax = step.cwiseAbs().maxCoeff();
        if (smax > opt.qn_max_step) step *= opt.qn_max_step / smax;
        C_acc = C, g_acc = g, e_acc = energy;
        C = C * expm(step - step.transpose());  // C_q += kappa_pq C_p
      }
    } else {
      // new orbitals from the (extrapolated, shifted) effective Fock matrix
      Matrix Fo = Co * W * Co.transpose();
      if (diis_size > 1 && iter >= opt.diis_start && !converged) Fo = diis.extrapolate(Fo, err);
      if (opt.level_shift > 0.0 && !converged) {  // shift open shells by half, virtuals fully
        Vector shift(nmo);
        for (Eigen::Index j = 0; j < nmo; ++j) shift(j) = shell[j] == 1 ? 0.5 : (shell[j] == 2);
        Fo += opt.level_shift * Co * shift.asDiagonal() * Co.transpose();
      }
      Matrix Cn;
      diagonalize(Fo, blocks, Cn, eps, irreps);
      C = X * Cn;
      shell = assign_shells(irreps, nclosed, nopen, fixed[0], fixed[1], nirrep);
      bool stable = true;
      for (int k = 0; k < 2; ++k) {
        const auto cnt = shell_counts(irreps, shell, k, nirrep);
        stable = stable && cnt == occ_prev[k];
        if (sym.freeze_occupation > 0 && iter == sym.freeze_occupation && fixed[k].empty())
          fixed[k] = cnt;
      }
      occ_stable = stable ? occ_stable + 1 : 1;
    }
    if (!converged)
      for (int k = 0; k < nshell; ++k) {
        const Matrix Dn = shell_density(C, k);
        ddmax = std::max(ddmax, (Dn - D[k]).cwiseAbs().maxCoeff());
        D[k] = Dn;
      }

    log::line(format("%5d  %20.12f  %14.6e  %11.3e  %11.3e  %3d%s  %10.3e  %4s  %7.1f%%  %10.2f  %10.2f%s",
                     iter, energy, de, rms, gmax, qn_now ? lbfgs.size() : diis.size(),
                     qn_now ? "q" : " ", ddmax, full ? "full" : "incr", screened, t_fock,
                     seconds_since(t_iter), note.c_str()));
    bool changed = false;
    std::vector<int> cnt[2];
    for (int k = 0; k < 2; ++k) {
      cnt[k] = shell_counts(irreps, shell, k, nirrep);
      changed = changed || cnt[k] != occ_prev[k];
    }
    if (nirrep > 1 && changed)
      log::line("       closed | " + model.names[1] + " orbitals per irrep:" +
                occupation_string(cnt[0]) + "  |" + occupation_string(cnt[1]));
    log::flush();
    occ_prev[0] = cnt[0], occ_prev[1] = cnt[1];
    if (converged) {
      wf.converged = true;
      break;
    }
  }

  // canonical orbitals: F_k / f_k diagonalized within each shell and irrep
  // (virtual: operator of the highest occupied shell); order: closed, open
  // (active), virtual, each by energy
  {
    std::vector<Matrix> Fcan;
    for (int k = 0; k < nshell; ++k) Fcan.push_back(C.transpose() * Fsh[k] * C / model.f[k]);
    struct Entry { double e; int k, s; Vector c; };
    std::vector<Entry> all;
    for (size_t g = 0; g < nirrep; ++g)
      for (int s = 0; s <= nshell; ++s) {
        std::vector<Eigen::Index> idx;
        for (Eigen::Index j = 0; j < nmo; ++j)
          if (irreps[j] == int(g) && shell[j] == s) idx.push_back(j);
        if (idx.empty()) continue;
        const Eigen::Index m = idx.size();
        Matrix sub(m, m), Cs(n, m);
        for (Eigen::Index a = 0; a < m; ++a) {
          Cs.col(a) = C.col(idx[a]);
          for (Eigen::Index b = 0; b < m; ++b)
            sub(a, b) = Fcan[std::min(s, nshell - 1)](idx[a], idx[b]);
        }
        Eigen::SelfAdjointEigenSolver<Matrix> es(sub);
        const Matrix Cn = Cs * es.eigenvectors();
        for (Eigen::Index a = 0; a < m; ++a) all.push_back({es.eigenvalues()(a), int(g), s, Cn.col(a)});
      }
    std::stable_sort(all.begin(), all.end(),
                     [](auto& a, auto& b) { return a.s != b.s ? a.s < b.s : a.e < b.e; });
    eps.resize(nmo);
    for (size_t j = 0; j < all.size(); ++j)
      C.col(j) = all[j].c, eps(j) = all[j].e, irreps[j] = all[j].k, shell[j] = all[j].s;
  }

  wf.C = C, wf.eps = eps, wf.irreps = irreps;
  wf.occ = Vector::Zero(nmo);
  double e_closed = -1e300, e_open_lo = 1e300, e_open_hi = -1e300, e_virt = 1e300;
  for (Eigen::Index j = 0; j < nmo; ++j) {
    wf.occ(j) = 2.0 * f_of(shell[j]);
    if (shell[j] == 0) e_closed = std::max(e_closed, eps(j));
    else if (shell[j] == 1) e_open_lo = std::min(e_open_lo, eps(j)), e_open_hi = std::max(e_open_hi, eps(j));
    else e_virt = std::min(e_virt, eps(j));
  }
  wf.occupation = shell_counts(irreps, shell, 0, nirrep);
  wf.occupation_open = shell_counts(irreps, shell, 1, nirrep);
  const Matrix Dc = shell_density(C, 0), Do = shell_density(C, 1);
  wf.D = 2.0 * Dc + 2.0 * model.f[1] * Do;
  wf.D_spin = Do;  // high spin: spin density = open shells, <S^2> exact
  const double sz = 0.5 * double(nopen);
  wf.s2 = wf.s2_exact = sz * (sz + 1.0);

  log::line();
  if (wf.converged)
    log::line(format("  SCF converged in %d iterations", wf.iterations));
  else
    log::line(format("  WARNING: SCF not converged after %d iterations", wf.iterations));
  log::line(format("  one-electron energy      %20.12f Eh", wf.e_one));
  log::line(format("  two-electron energy      %20.12f Eh", wf.e_two));
  log::line(format("  nuclear repulsion        %20.12f Eh", wf.e_nuc));
  log::line(format("  total energy             %20.12f Eh", wf.energy));
  log::line(format("  <S^2> = S(S+1)           %20.12f", wf.s2));
  log::line(format("  highest closed %s   %s orbitals %.6f .. %.6f Eh%s",
                   nclosed ? format("%.6f Eh", e_closed).c_str() : "-", model.names[1].c_str(),
                   e_open_lo, e_open_hi,
                   size_t(nmo) > nclosed + nopen ? format("   lowest virtual %.6f Eh", e_virt).c_str()
                                                 : ""));
  log::line("  (orbital energies: closed shells from (F_alpha + F_beta)/2, open and virtual from "
              "F_alpha)");
  log::line(format("  time: Fock builds %.2f s, total %.2f s", t_fock_total,
                   seconds_since(t_start)));
  return wf;
}

}  // namespace

Wavefunction rohf(const Molecule& mol, const BasisSet& basis, const ScfOptions& opt,
                  const std::vector<Matrix>& guess, const ScfSymmetry& sym,
                  const std::optional<Wavefunction>& start, std::shared_ptr<JKBuilder> jk) {
  mol.validate();
  const int nel = mol.n_electrons(), mult = mol.multiplicity();
  if (mult < 2 || (nel - mult + 1) % 2 || mult - 1 > nel)
    throw std::invalid_argument(log::format(
        "ROHF: needs an open-shell state (multiplicity %d, %d electrons)", mult, nel));
  const size_t nopen = mult - 1, nclosed = (nel - nopen) / 2;
  StartOrbitals st;
  if (start) {
    st.C = start->C, st.irreps = start->irreps;
    for (Eigen::Index j = 0; j < start->occ.size(); ++j)
      st.shell.push_back(start->occ(j) > 1.5 ? 0 : (start->occ(j) > 0.5 ? 1 : 2));
  }
  return open_shell_scf(mol, basis, opt, high_spin(), nclosed, nopen, "ROHF", guess, sym,
                        start ? &st : nullptr, std::move(jk));
}

}  // namespace gemqc

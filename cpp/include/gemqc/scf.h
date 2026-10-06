#pragma once
/// \file
/// Hartree-Fock SCF: restricted closed-shell (RHF) and unrestricted (UHF);
/// direct Fock build, DIIS / quasi-Newton / Newton solvers.

#include <Eigen/Core>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gemqc/basis.h"
#include "gemqc/fock.h"
#include "gemqc/integrals.h"
#include "gemqc/molecule.h"

namespace gemqc {

using Vector = Eigen::VectorXd;

struct ScfOptions {
  double convergence = 1e-6;         ///< RMS orbital gradient (DIIS error)
  double energy_convergence = 1e-8;  ///< |dE| in Eh
  int max_iterations = 100;
  int diis_size = 8;                 ///< max. DIIS subspace (0: no DIIS)
  int diis_start = 1;                ///< first iteration with extrapolation
  double level_shift = 0.0;          ///< virtual level shift in Eh
  double screening = 1e-12;          ///< integral screening threshold
  bool incremental = true;           ///< incremental Fock build (F[D] = F[D_old] + G[dD])
  int rebuild_every = 10;            ///< full Fock rebuild interval (incremental mode)
  double incremental_until = 1e-5;   ///< full builds once rms(grad) is below this
  double lindep = 1e-7;              ///< drop overlap eigenvalues below (linear dep.)
  /// solver: "diis" (Roothaan-Hall + DIIS), "qn" (quasi-Newton, L-BFGS on
  /// occupied-virtual orbital rotations) or "newton" (augmented Hessian with
  /// exact Hessian products, trust region); occupation fixed when QN/Newton start
  std::string solver = "diis";
  int qn_start = 2;                  ///< earliest QN/Newton iteration (earlier: DIIS)
  int qn_stable = 3;                 ///< QN/Newton once the aufbau occupation per irrep
                                     ///< was unchanged for this many DIIS iterations
  int qn_history = 10;               ///< L-BFGS history length
  double qn_max_step = 0.5;          ///< max. rotation angle per step (rad)
  double trust_radius = 0.5;         ///< Newton: initial trust radius |kappa| (max. 2x)
  int newton_micro = 20;             ///< Newton: max. micro-iterations per step
};

/// Symmetry blocking of the SCF in the computational point group (D2h subgroup):
/// symmetry-adapted functions per irrep, occupied orbitals per irrep
struct ScfSymmetry {
  std::string group = "C1";
  std::vector<std::string> irreps;  ///< names in Molpro order
  std::vector<Matrix> so;           ///< per irrep: AO -> symmetry-adapted functions (nbf x n_k)
  std::vector<int> occupation;      ///< occupied orbitals per irrep: RHF closed shells,
                                    ///< UHF alpha (empty: aufbau)
  std::vector<int> occupation_beta; ///< UHF: beta orbitals per irrep (empty: aufbau)
  std::vector<int> occupation_open; ///< ROHF: open-shell orbitals per irrep (empty: aufbau)
  int freeze_occupation = 0;        ///< aufbau occupation fixed after this iteration (0: never)
  /// AO operations for the skeleton Fock build (empty: no symmetry in Fock build)
  std::vector<std::vector<int>> ao_perm;
  std::vector<std::vector<double>> ao_sign;
  std::vector<std::vector<double>> characters;  ///< [irrep][operation]
};

/// SCF result; orbitals are columns of C (nbf x nmo). UHF: C, eps, occ, irreps,
/// occupation belong to the alpha spin, the *_b fields to beta. ROHF: occupation
/// counts closed shells, occupation_open the open shells
/// per irrep; the orbitals are ordered closed, open, virtual, each by
/// energy.
struct Wavefunction {
  Molecule molecule;
  BasisSet basis;
  std::string method = "RHF";   ///< RHF, UHF or ROHF
  Matrix C;
  Vector eps;  ///< orbital energies
  Vector occ;  ///< occupation numbers (RHF 2 or 0, UHF 1 or 0, ROHF 2, 1 or 0)
  Matrix D;    ///< total density
  double energy = 0.0, e_nuc = 0.0, e_one = 0.0, e_two = 0.0;
  double gradient_rms = 0.0;
  bool converged = false;
  int iterations = 0;
  std::string group = "C1";
  std::vector<std::string> irrep_names;
  std::vector<int> irreps;      ///< irrep (0-based) of each orbital
  std::vector<int> occupation;  ///< occupied orbitals per irrep (ROHF: closed shells)
  std::vector<int> occupation_open;  ///< ROHF: open shells per irrep
  /// UHF: beta orbitals, spin density (alpha - beta; also ROHF), <S^2> and its
  /// exact value S(S+1), natural orbitals of the total density (occupation
  /// descending)
  Matrix C_b;
  Vector eps_b, occ_b;
  std::vector<int> irreps_b, occupation_b;
  Matrix D_spin;
  double s2 = 0.0, s2_exact = 0.0;
  Matrix C_nat;
  Vector occ_nat;
  std::vector<int> irreps_nat;
  bool unrestricted() const { return method == "UHF"; }
  size_t n_occ() const;
};

/// jk (all SCF functions): J/K builder, e.g. density fitting (default: exact
/// 4-index integrals with screening); incremental builds only for exact ones.

/// Closed-shell RHF; guess: initial total density (core Hamiltonian guess if empty);
/// sym: symmetry blocking (default: none, i.e. C1); start: initial orbitals and
/// occupation (e.g. displaced along an instability; overrides guess, occupation
/// per irrep is kept, QN/Newton start at once)
Wavefunction rhf(const Molecule& mol, const BasisSet& basis, const ScfOptions& opt,
                 const std::optional<Matrix>& guess = std::nullopt,
                 const ScfSymmetry& sym = ScfSymmetry(),
                 const std::optional<Wavefunction>& start = std::nullopt,
                 std::shared_ptr<JKBuilder> jk = nullptr);

/// UHF for any multiplicity (alpha: (N + M - 1) / 2 electrons); guess: one
/// matrix (total density, split in proportion to the electrons per spin) or two
/// (alpha, beta densities); empty: core guess. sym, start as for rhf.
Wavefunction uhf(const Molecule& mol, const BasisSet& basis, const ScfOptions& opt,
                 const std::vector<Matrix>& guess = {},
                 const ScfSymmetry& sym = ScfSymmetry(),
                 const std::optional<Wavefunction>& start = std::nullopt,
                 std::shared_ptr<JKBuilder> jk = nullptr);

/// High-spin ROHF (closed shells + M - 1 open shells, general open-shell energy
/// expression with shells and coupling coefficients); guess: total density
/// (orbitals from its Fock matrix), or closed and open shell densities (unit
/// occupation; effective Fock matrix), or empty (core guess); solvers diis and
/// qn; sym: occupation = closed, occupation_open = open shells per irrep; start
/// as for rhf
Wavefunction rohf(const Molecule& mol, const BasisSet& basis, const ScfOptions& opt,
                  const std::vector<Matrix>& guess = {},
                  const ScfSymmetry& sym = ScfSymmetry(),
                  const std::optional<Wavefunction>& start = std::nullopt,
                  std::shared_ptr<JKBuilder> jk = nullptr);

/// C exp(K) with K = [[0, -k^T], [k, 0]] for kappa(a, i) (virtual a, occupied i;
/// nmo x nmo, other elements ignored); exact exponential
Matrix rotate_orbitals(const Matrix& C, const Matrix& kappa, const std::vector<bool>& occ);

}  // namespace gemqc

// Python bindings: module gemqc._core
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>

#include "gemqc/basis.h"
#include "gemqc/df.h"
#include "gemqc/elements.h"
#include "gemqc/fock.h"
#include "gemqc/grid.h"
#include "gemqc/info.h"
#include "gemqc/integrals.h"
#include "gemqc/log.h"
#include "gemqc/molecule.h"
#include "gemqc/resources.h"
#include "gemqc/scf.h"
#include "gemqc/twobody.h"
#include "gemqc/units.h"

namespace py = pybind11;
using namespace gemqc;

PYBIND11_MODULE(_core, m) {
  m.doc() = "gemQC C++ backend";

  // --- info, units, elements ---
  m.def("libint_info", &libint_info,
        "libint version and compiled-in limits (max AM, derivative orders)");
  auto mu = m.def_submodule("units", "physical constants (CODATA 2018)");
  mu.attr("bohr_in_angstrom") = units::bohr_in_angstrom;
  mu.attr("angstrom_in_bohr") = units::angstrom_in_bohr;
  mu.attr("hartree_in_ev") = units::hartree_in_ev;
  mu.attr("hartree_in_kjmol") = units::hartree_in_kjmol;
  mu.attr("au_dipole_in_debye") = units::au_dipole_in_debye;
  mu.attr("hartree_in_wavenumber") = units::hartree_in_wavenumber;
  mu.attr("fine_structure") = units::fine_structure;
  mu.attr("electron_g_factor") = units::electron_g_factor;
  m.def("element_z", &element_z, py::arg("symbol"));
  m.def("element_symbol", &element_symbol, py::arg("z"));

  // --- Molecule ---
  py::class_<Molecule>(m, "Molecule", "Molecular structure (coordinates in bohr)")
      .def(py::init<>())
      .def(py::init<std::string>(), py::arg("name"))
      .def("add_atom", &Molecule::add_atom, py::arg("z"), py::arg("r"),
           "append atom with nuclear charge z at position r (bohr)")
      .def("set_position", &Molecule::set_position, py::arg("i"), py::arg("r"))
      .def_property("name", &Molecule::name, &Molecule::set_name)
      .def_property("charge", &Molecule::charge, &Molecule::set_charge)
      .def_property("multiplicity", &Molecule::multiplicity,
                    &Molecule::set_multiplicity)
      .def_property("coordinates", &Molecule::coordinates,
                    &Molecule::set_coordinates, "N x 3 array (bohr), copy")
      .def_property_readonly("natoms", &Molecule::natoms)
      .def_property_readonly("atomic_numbers", &Molecule::atomic_numbers)
      .def_property_readonly("symbols",
                             [](const Molecule& mol) {
                               std::vector<std::string> s;
                               for (const auto& a : mol.atoms())
                                 s.push_back(element_symbol(a.z));
                               return s;
                             })
      .def_property_readonly("n_electrons", &Molecule::n_electrons)
      .def("nuclear_repulsion", &Molecule::nuclear_repulsion)
      .def("validate", &Molecule::validate)
      .def("__len__", &Molecule::natoms)
      .def("__repr__", [](const Molecule& mol) {
        return "<Molecule '" + mol.name() + "' natoms=" +
               std::to_string(mol.natoms()) + ">";
      });

  // --- basis set ---
  py::class_<ShellDef>(m, "ShellDef", "contracted shell of a basis library")
      .def(py::init<int, bool, std::vector<double>, std::vector<double>>(),
           py::arg("l"), py::arg("pure"), py::arg("exponents"),
           py::arg("coefficients"))
      .def_readwrite("l", &ShellDef::l)
      .def_readwrite("pure", &ShellDef::pure)
      .def_readwrite("exponents", &ShellDef::exponents)
      .def_readwrite("coefficients", &ShellDef::coefficients);

  py::class_<BasisSet>(m, "BasisSet", "atom-centred basis set (libint shells)")
      .def(py::init<const Molecule&, const ElementBasis&, std::string, bool>(),
           py::arg("molecule"), py::arg("element_basis"), py::arg("name") = "",
           py::arg("auxiliary") = false,
           "auxiliary: fitting basis (2-/3-centre integrals only; higher l allowed)")
      .def(py::init<const Molecule&, const AtomBasis&, std::string, bool>(),
           py::arg("molecule"), py::arg("atom_basis"), py::arg("name") = "",
           py::arg("auxiliary") = false,
           "shells per atom (list over the atoms of the molecule)")
      .def("set_centers", &BasisSet::set_centers, py::arg("molecule"))
      .def_property_readonly("name", &BasisSet::name)
      .def_property_readonly("nshells", &BasisSet::nshells)
      .def_property_readonly("nbf", &BasisSet::nbf)
      .def_property_readonly("max_l", &BasisSet::max_l)
      .def_property_readonly("max_nprim", &BasisSet::max_nprim)
      .def_property_readonly("shell_offsets", &BasisSet::shell_offsets)
      .def_property_readonly("shell_to_atom", &BasisSet::shell_to_atom)
      .def_property_readonly("bf_to_atom", &BasisSet::bf_to_atom)
      .def(
          "shell",
          [](const BasisSet& b, size_t i) {
            const auto& s = b.shell(i);
            py::dict d;
            d["l"] = s.contr[0].l;
            d["pure"] = s.contr[0].pure;
            d["center"] = s.O;
            d["atom"] = b.shell_to_atom()[i];
            d["exponents"] = std::vector<double>(s.alpha.begin(), s.alpha.end());
            // coefficients with primitive normalization embedded (libint convention)
            d["coefficients"] = std::vector<double>(s.contr[0].coeff.begin(),
                                                    s.contr[0].coeff.end());
            return d;
          },
          py::arg("i"), "shell data as dict")
      .def("__len__", &BasisSet::nshells)
      .def("__repr__", [](const BasisSet& b) {
        return "<BasisSet '" + b.name() + "' nshells=" +
               std::to_string(b.nshells()) + " nbf=" + std::to_string(b.nbf()) + ">";
      });

  // --- one-electron integrals (basis2 given: rectangular <basis|op|basis2>) ---
  // second basis defaults to the first (None -> square, symmetric matrix)
  auto second = [](const BasisSet& b1, const BasisSet* b2) -> const BasisSet& {
    return b2 ? *b2 : b1;
  };
  m.def(
      "overlap",
      [second](const BasisSet& b1, const BasisSet* b2) {
        return overlap(b1, second(b1, b2));
      },
      py::arg("basis"), py::arg("basis2") = nullptr, "overlap matrix S");
  m.def(
      "kinetic",
      [second](const BasisSet& b1, const BasisSet* b2) {
        return kinetic(b1, second(b1, b2));
      },
      py::arg("basis"), py::arg("basis2") = nullptr, "kinetic energy matrix T");
  m.def(
      "nuclear_attraction",
      [second](const BasisSet& b1, const Molecule& mol, const BasisSet* b2) {
        return nuclear_attraction(b1, second(b1, b2), mol);
      },
      py::arg("basis"), py::arg("molecule"), py::arg("basis2") = nullptr,
      "nuclear attraction matrix V");
  m.def(
      "dipole",
      [second](const BasisSet& b1, const std::array<double, 3>& origin,
               const BasisSet* b2) { return dipole(b1, second(b1, b2), origin); },
      py::arg("basis"), py::arg("origin") = std::array<double, 3>{0, 0, 0},
      py::arg("basis2") = nullptr,
      "[X, Y, Z] matrices of <mu|r_k - O_k|nu> (no electron charge sign)");
  m.def("second_moments", &second_moments, py::arg("basis"),
        py::arg("origin") = std::array<double, 3>{0, 0, 0},
        "[XX, XY, XZ, YY, YZ, ZZ] matrices of <mu|(r_a - O_a)(r_b - O_b)|nu>");
  m.def("basis_values", &basis_values, py::arg("basis"), py::arg("points"),
        "values of the basis functions at points (npoints, 3) in bohr: (npoints, nbf)");

  m.def(
      "eri",
      [](const BasisSet& b) {
        const auto n = static_cast<py::ssize_t>(b.nbf());
        auto v = new std::vector<double>(eri_tensor(b));
        py::capsule owner(v, [](void* p) { delete static_cast<std::vector<double>*>(p); });
        return py::array_t<double>({n, n, n, n}, v->data(), owner);
      },
      py::arg("basis"), "full ERI tensor (pq|rs), chemists' notation (small cases)");

  // --- generic two-electron operators (F12 building blocks) ---
  py::class_<TwoBodyOperator>(m, "TwoBodyOperator",
                              "scale * g(r12): coulomb 1/r, stg exp(-zeta r), "
                              "stg_x_coulomb exp(-zeta r)/r; with gtg = [(a_i, c_i)] of\n"
                              "g = sum_i c_i exp(-a_i r^2): cgtg g, cgtg_x_coulomb g/r, "
                              "delcgtg2 (grad_1 g)^2; with gtg2 of g2: delcgtga_delcgtgb "
                              "(grad_1 g).(grad_1 g2)")
      .def(py::init([](const std::string& kind, double zeta, double scale,
                       const std::vector<std::pair<double, double>>& gtg,
                       const std::vector<std::pair<double, double>>& gtg2) {
             return TwoBodyOperator{TwoBodyOperator::kind_from(kind), zeta, scale, gtg, gtg2};
           }),
           py::arg("kind") = "coulomb", py::arg("zeta") = 0.0, py::arg("scale") = 1.0,
           py::arg("gtg") = std::vector<std::pair<double, double>>{},
           py::arg("gtg2") = std::vector<std::pair<double, double>>{})
      .def_property_readonly("kind", &TwoBodyOperator::name)
      .def_readonly("zeta", &TwoBodyOperator::zeta)
      .def_readonly("scale", &TwoBodyOperator::scale)
      .def_readonly("gtg", &TwoBodyOperator::gtg)
      .def_readonly("gtg2", &TwoBodyOperator::gtg2)
      .def_static("has_delcgtgab", &TwoBodyOperator::has_delcgtgab,
                  "true if libint provides the kernel delcgtga_delcgtgb")
      .def("__repr__", [](const TwoBodyOperator& o) {
        return "TwoBodyOperator(" + o.name() + ", zeta=" + std::to_string(o.zeta) +
               ", scale=" + std::to_string(o.scale) + ")";
      });
  m.def(
      "two_body_2c",
      [second](const TwoBodyOperator& op, const BasisSet& b1, const BasisSet* b2) {
        py::gil_scoped_release release;
        return two_body_2c(op, b1, second(b1, b2));
      },
      py::arg("operator"), py::arg("basis"), py::arg("basis2") = nullptr,
      "(P|g|Q) (n1, n2)");
  m.def(
      "two_body_3c",
      [second](const TwoBodyOperator& op, const BasisSet& aux, const BasisSet& b1,
               const BasisSet* b2) {
        py::gil_scoped_release release;
        return Eigen::MatrixXd(two_body_3c(op, aux, b1, second(b1, b2)));
      },
      py::arg("operator"), py::arg("aux"), py::arg("basis"), py::arg("basis2") = nullptr,
      "(P|g|mu nu) (naux, n1*n2), column mu*n2 + nu");
  m.def(
      "two_body_3c_mo",
      [](const TwoBodyOperator& op, const BasisSet& aux, const BasisSet& basis,
         const Matrix& C1, const Matrix& C2) {
        py::gil_scoped_release release;
        return Eigen::MatrixXd(two_body_3c_mo(op, aux, basis, C1, C2));
      },
      py::arg("operator"), py::arg("aux"), py::arg("basis"), py::arg("C1"), py::arg("C2"),
      "(P|g|k q) = sum (P|g|mu nu) C1_mu,k C2_nu,q (naux, n1*n2), column k*n2 + q");
  m.def(
      "pair_integrals",
      [](const TwoBodyOperator& op, const BasisSet& bo, const BasisSet& b1, const BasisSet& b2,
         const Matrix& Co, const std::vector<std::pair<int, int>>& pairs, double threshold) {
        py::gil_scoped_release release;
        return pair_integrals(op, bo, b1, b2, Co, pairs, threshold);
      },
      py::arg("operator"), py::arg("bo"), py::arg("b1"), py::arg("b2"), py::arg("Co"),
      py::arg("pairs"), py::arg("threshold") = 1e-12,
      "K^ij(nu, si) = sum_{mu,la in bo} (mu nu|g|la si) C_mu,i C_la,j, nu in b1, si in b2,\n"
      "for the orbital pairs (i, j) (columns of Co over bo): list of (n1, n2) matrices");
  m.def(
      "two_body_4c",
      [](const TwoBodyOperator& op, const BasisSet& b1, const BasisSet* b2,
         const BasisSet* b3, const BasisSet* b4) {
        const BasisSet& c2 = b2 ? *b2 : b1;
        const BasisSet& c3 = b3 ? *b3 : b1;
        const BasisSet& c4 = b4 ? *b4 : c3;
        std::vector<double>* v;
        {
          py::gil_scoped_release release;
          v = new std::vector<double>(two_body_4c(op, b1, c2, c3, c4));
        }
        py::capsule owner(v, [](void* p) { delete static_cast<std::vector<double>*>(p); });
        return py::array_t<double>({py::ssize_t(b1.nbf()), py::ssize_t(c2.nbf()),
                                    py::ssize_t(c3.nbf()), py::ssize_t(c4.nbf())},
                                   v->data(), owner);
      },
      py::arg("operator"), py::arg("basis"), py::arg("basis2") = nullptr,
      py::arg("basis3") = nullptr, py::arg("basis4") = nullptr,
      "(mu nu|g|la si) (n1, n2, n3, n4), chemists' notation (small cases); basis2 and basis3\n"
      "default to basis, basis4 to basis3");

  // --- output and resources ---
  m.def(
      "set_output",
      [](py::object func) {
        if (func.is_none())
          log::set_sink({});
        else if (py::hasattr(func, "flush"))
          log::set_sink([func](const std::string& s) { func(s); },
                        [func] { func.attr("flush")(); });
        else
          log::set_sink([func](const std::string& s) { func(s); });
      },
      py::arg("func"), "route C++ output lines to func(str) (func.flush() if present); None: stdout");
  m.def("set_output_enabled", &log::set_enabled, py::arg("on"));
  // drop Python callback before interpreter shutdown
  py::module_::import("atexit").attr("register")(
      py::cpp_function([] { log::set_sink({}); }));
  m.def("set_memory", &resources::set_memory, py::arg("bytes"), "memory limit (0: none)");
  m.def("memory", &resources::memory);
  m.def("set_threads", &resources::set_threads, py::arg("n"), "threads (0: default)");
  m.def("threads", &resources::threads);
  m.def("format_bytes", &resources::format_bytes, py::arg("bytes"));

  // --- J/K builds: exact (FockBuilder) or density fitting (DFBuilder) ---
  py::class_<JKBuilder, std::shared_ptr<JKBuilder>>(m, "JKBuilder",
                                                    "J/K builder interface (exact or DF)")
      .def(
          "compute",
          [](JKBuilder& fb, const std::vector<Matrix>& ds, bool symmetric, bool with_k,
             const std::vector<int>& irreps) {
            JK jk;
            {
              py::gil_scoped_release release;
              jk = fb.compute(ds, symmetric, with_k, irreps);
            }
            return py::make_tuple(jk.J, jk.K);
          },
          py::arg("densities"), py::arg("symmetric") = true, py::arg("with_k") = true,
          py::arg("irreps") = std::vector<int>{},
          "(J list, K list) for a list of densities (irreps: of each density, skeleton build)")
      .def(
          "pair_jk",
          [](const JKBuilder& fb, const Matrix& Ca, std::vector<std::pair<int, int>> pairs) {
            JK jk;
            {
              py::gil_scoped_release release;
              jk = fb.pair_jk(Ca, pairs);
            }
            return py::make_tuple(jk.J, jk.K);
          },
          py::arg("Ca"), py::arg("pairs") = std::vector<std::pair<int, int>>{},
          "(J list, K list) of the pair densities c_u c_v^T for pairs (u, v) (default: all\n"
          "u <= v)")
      .def("set_symmetry", &JKBuilder::set_symmetry, py::arg("perm"), py::arg("sign"),
           py::arg("characters") = std::vector<std::vector<double>>{},
           "skeleton build with AO operations (perm, sign); characters[irrep][op] "
           "for densities that are not totally symmetric (exact builder only)")
      .def_property_readonly("symmetry_order", &JKBuilder::symmetry_order)
      .def_property_readonly("exact", &JKBuilder::exact)
      .def("describe", &JKBuilder::describe);
  py::class_<FockBuilder, JKBuilder, std::shared_ptr<FockBuilder>>(
      m, "FockBuilder", "direct J/K build (4-index ERIs)")
      .def(py::init<const BasisSet&, double, const TwoBodyOperator&>(), py::arg("basis"),
           py::arg("threshold") = 1e-12, py::arg("operator") = TwoBodyOperator{},
           "operator: two-electron operator of the integrals (default Coulomb)")
      .def_property_readonly("threshold", &FockBuilder::threshold)
      .def_property_readonly("schwarz", &FockBuilder::schwarz)
      .def_property_readonly("significant_pairs", &FockBuilder::significant_pairs)
      .def_property_readonly("quartets_computed", &FockBuilder::quartets_computed)
      .def_property_readonly("quartets_screened", &FockBuilder::quartets_screened)
      .def_property_readonly("threads_used", &FockBuilder::threads_used);
  py::class_<DFBuilder, JKBuilder, std::shared_ptr<DFBuilder>>(
      m, "DFBuilder", "density-fitted J/K build (Coulomb metric, hybrid 3-index storage)")
      .def(py::init([](const BasisSet& basis, const BasisSet& aux, double threshold,
                       double memory_fraction, double metric_cutoff, const std::string& storage) {
             py::gil_scoped_release release;
             return std::make_shared<DFBuilder>(basis, aux, threshold, memory_fraction,
                                                metric_cutoff, storage);
           }),
           py::arg("basis"), py::arg("aux"), py::arg("threshold") = 1e-12,
           py::arg("memory_fraction") = 0.5, py::arg("metric_cutoff") = 1e-10,
           py::arg("storage") = "raw",
           "storage: raw (hybrid raw 3-index integrals), fitted (B = M^T (P|mu nu) stored) or "
           "auto (fitted if B fits into memory_fraction of the limit)")
      .def(
          "mo_three_index",
          [](const DFBuilder& df, const Matrix& C, const Matrix& Ca) {
            py::gil_scoped_release release;
            return Eigen::MatrixXd(df.mo_three_index(C, Ca));
          },
          py::arg("C"), py::arg("Ca"),
          "B^Q_pu = sum_P M_PQ (P|pu), shape (nfit, np*m), column p*m + u:\n"
          "(pu|qv) = sum_Q B^Q_pu B^Q_qv")
      .def(
          "fitted_coulomb",
          [](const DFBuilder& df, const Eigen::MatrixXd& b) {
            py::gil_scoped_release release;
            return df.fitted_coulomb(b);
          },
          py::arg("b"), "AO Coulomb matrices of fitted densities b (nfit, nd), e.g. B^Q_uv")
      .def("metric_factor", [](const DFBuilder& d) { return Eigen::MatrixXd(d.metric_factor()); },
           "M (naux, nfit) with V^-1 = M M^T")
      .def_property_readonly("naux", &DFBuilder::naux)
      .def_property_readonly("nfit", &DFBuilder::nfit)
      .def_property_readonly("blocks_stored", &DFBuilder::blocks_stored)
      .def_property_readonly("blocks_direct", &DFBuilder::blocks_direct)
      .def_property_readonly("blocks_screened", &DFBuilder::blocks_screened)
      .def_property_readonly("stored_bytes", &DFBuilder::stored_bytes)
      .def_property_readonly("fitted", &DFBuilder::fitted);

  // --- SCF ---
  py::class_<ScfOptions>(m, "ScfOptions")
      .def(py::init<>())
      .def_readwrite("convergence", &ScfOptions::convergence)
      .def_readwrite("energy_convergence", &ScfOptions::energy_convergence)
      .def_readwrite("max_iterations", &ScfOptions::max_iterations)
      .def_readwrite("diis_size", &ScfOptions::diis_size)
      .def_readwrite("diis_start", &ScfOptions::diis_start)
      .def_readwrite("level_shift", &ScfOptions::level_shift)
      .def_readwrite("screening", &ScfOptions::screening)
      .def_readwrite("incremental", &ScfOptions::incremental)
      .def_readwrite("rebuild_every", &ScfOptions::rebuild_every)
      .def_readwrite("incremental_until", &ScfOptions::incremental_until)
      .def_readwrite("lindep", &ScfOptions::lindep)
      .def_readwrite("solver", &ScfOptions::solver)
      .def_readwrite("qn_start", &ScfOptions::qn_start)
      .def_readwrite("qn_history", &ScfOptions::qn_history)
      .def_readwrite("qn_stable", &ScfOptions::qn_stable)
      .def_readwrite("qn_max_step", &ScfOptions::qn_max_step)
      .def_readwrite("trust_radius", &ScfOptions::trust_radius)
      .def_readwrite("newton_micro", &ScfOptions::newton_micro);

  py::class_<ScfSymmetry>(m, "ScfSymmetry", "symmetry blocking of the SCF")
      .def(py::init<>())
      .def_readwrite("group", &ScfSymmetry::group)
      .def_readwrite("irreps", &ScfSymmetry::irreps)
      .def_readwrite("so", &ScfSymmetry::so)
      .def_readwrite("occupation", &ScfSymmetry::occupation)
      .def_readwrite("occupation_beta", &ScfSymmetry::occupation_beta)
      .def_readwrite("occupation_open", &ScfSymmetry::occupation_open)
      .def_readwrite("freeze_occupation", &ScfSymmetry::freeze_occupation)
      .def_readwrite("ao_perm", &ScfSymmetry::ao_perm)
      .def_readwrite("ao_sign", &ScfSymmetry::ao_sign)
      .def_readwrite("characters", &ScfSymmetry::characters);

  py::class_<Wavefunction>(m, "Wavefunction", "SCF result (orbitals in columns of C)")
      .def_readonly("molecule", &Wavefunction::molecule)
      .def_readonly("basis", &Wavefunction::basis)
      .def_readonly("C", &Wavefunction::C)
      .def_readonly("eps", &Wavefunction::eps)
      .def_readonly("occ", &Wavefunction::occ)
      .def_readonly("D", &Wavefunction::D)
      .def_readonly("energy", &Wavefunction::energy)
      .def_readonly("e_nuc", &Wavefunction::e_nuc)
      .def_readonly("e_one", &Wavefunction::e_one)
      .def_readonly("e_two", &Wavefunction::e_two)
      .def_readonly("gradient_rms", &Wavefunction::gradient_rms)
      .def_readonly("converged", &Wavefunction::converged)
      .def_readonly("iterations", &Wavefunction::iterations)
      .def_readonly("group", &Wavefunction::group)
      .def_readonly("irrep_names", &Wavefunction::irrep_names)
      .def_readonly("irreps", &Wavefunction::irreps)
      .def_readonly("occupation", &Wavefunction::occupation)
      .def_readonly("method", &Wavefunction::method)
      .def_readonly("C_b", &Wavefunction::C_b)
      .def_readonly("eps_b", &Wavefunction::eps_b)
      .def_readonly("occ_b", &Wavefunction::occ_b)
      .def_readonly("irreps_b", &Wavefunction::irreps_b)
      .def_readonly("occupation_b", &Wavefunction::occupation_b)
      .def_readonly("occupation_open", &Wavefunction::occupation_open)
      .def_readonly("D_spin", &Wavefunction::D_spin)
      .def_readonly("s2", &Wavefunction::s2)
      .def_readonly("s2_exact", &Wavefunction::s2_exact)
      .def_readonly("C_nat", &Wavefunction::C_nat)
      .def_readonly("occ_nat", &Wavefunction::occ_nat)
      .def_readonly("irreps_nat", &Wavefunction::irreps_nat)
      .def_property_readonly("unrestricted", &Wavefunction::unrestricted)
      .def_property_readonly("n_occ", &Wavefunction::n_occ)
      .def("__repr__", [](const Wavefunction& w) {
        return "<Wavefunction " + w.method + " E=" + std::to_string(w.energy) +
               (w.converged ? " converged>" : " NOT converged>");
      });

  m.def("rhf", &rhf, py::arg("molecule"), py::arg("basis"),
        py::arg("options") = ScfOptions(), py::arg("guess") = std::nullopt,
        py::arg("symmetry") = ScfSymmetry(), py::arg("start") = std::nullopt, py::arg("jk") = std::shared_ptr<JKBuilder>(),
        "closed-shell RHF; guess: initial total density (default: core guess);\n"
        "symmetry: blocking by irreps (default: C1); start: initial orbitals (Wavefunction);\n"
        "jk: J/K builder (default: exact integrals; e.g. a DFBuilder)");

  m.def("uhf", &uhf, py::arg("molecule"), py::arg("basis"),
        py::arg("options") = ScfOptions(), py::arg("guess") = std::vector<Matrix>{},
        py::arg("symmetry") = ScfSymmetry(), py::arg("start") = std::nullopt, py::arg("jk") = std::shared_ptr<JKBuilder>(),
        "UHF; guess: [total density] or [alpha, beta densities] (default: core guess);\n"
        "symmetry: blocking by irreps (default: C1); start: initial orbitals (Wavefunction);\n"
        "jk: J/K builder (default: exact integrals; e.g. a DFBuilder)");

  m.def("rohf", &rohf, py::arg("molecule"), py::arg("basis"),
        py::arg("options") = ScfOptions(), py::arg("guess") = std::vector<Matrix>{},
        py::arg("symmetry") = ScfSymmetry(), py::arg("start") = std::nullopt, py::arg("jk") = std::shared_ptr<JKBuilder>(),
        "high-spin ROHF; guess: [total density] or [closed, open shell densities] (default:\n"
        "core guess); symmetry: blocking by irreps, occupation (closed) and occupation_open\n"
        "per irrep; start: initial orbitals");
}

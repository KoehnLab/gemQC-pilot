"""Tasks mp2-f12 and df-mp2-f12: MP2-F12 with fixed amplitudes (SP ansatz, ansatz 3),
approximations 3C and 3*C (Molpro), optionally NOZ and HY1 (theory:
docs/theory.md).

Closed-shell Hylleraas functional with the geminal amplitudes T^ij_ij = 3/8, T^ij_ji = 1/8
(1/2 singlet, 1/4 triplet) and T~ = 2T - T(swap)::

    E = sum_ij [2 T~ V + T~ (B - (e_i + e_j) X) T] + coupling (3C)

evaluated pair by pair (f12.PairIntermediates). Singlet and triplet parts use the
symmetric (factor 1) and antisymmetric (factor 3) amplitude components. 3C couples the
conventional amplitudes, T^ab_ij = -(K + C c)/D, and includes the F C term in B; 3*C
omits both.
"""

import time

import numpy as np

from .basislib import build_basis
from .cabs import build_ri_space, cabs_singles
from .correlation import (PARTS, has_correlated_core, orbital_spaces, pair_partition,
                          print_spaces, space_results)
from .f12 import ExactF12Integrals, F12Integrals, PairIntermediates
from .geminal import Geminal
from .options import InputError
from .restart import save_orbitals


def fixed_amplitude_energies(inter, eps_occ, eps_vir, pair_energies=False):
    """MP2 and MP2-F12 (3*C, 3C) energies with singlet and triplet parts.

    Args:
        inter: f12.PairIntermediates.
        eps_occ: Orbital energies of the active occupied orbitals.
        eps_vir: Orbital energies of the active virtual orbitals.
        pair_energies: Also return the pair energies.

    Returns:
        dict name -> (singlet, triplet) for 'mp2', '3*C', '3C' (F12 corrections, the
        3C coupling included in '3C'); with pair_energies 'pairs' {name: (no, no, 2)
        array}: singlet and triplet energy of the pair ij (with ji) for i >= j.
    """
    no = len(eps_occ)
    res = {k: np.zeros(2) for k in ("mp2", "3*C", "3C")}
    pairs = {k: np.zeros((no, no, 2)) for k in res}
    for i in range(no):
        for j in range(i + 1):
            d = inter.pair(i, j)
            w = 1.0 if i == j else 2.0  # (i, j) and (j, i)
            eij = eps_occ[i] + eps_occ[j]
            if i == j:
                ts, ta = np.array([0.5]), np.zeros(1)
            else:
                ts, ta = np.array([0.25, 0.25]), np.array([0.125, -0.125])
            V, X = d["V"], d["X"]
            for name, B in (("3*C", d["B3sC"]), ("3C", d["B3C"])):
                W = B - eij * X
                e = np.array([2 * ts @ V + ts @ W @ ts, 3 * (2 * ta @ V + ta @ W @ ta)])
                res[name] += w * e
                pairs[name][i, j] += w * e
            # conventional amplitudes, uncoupled (MP2) and coupled to the geminals (3C)
            Dab = eij - eps_vir[:, None] - eps_vir[None, :]
            K, C = d["K"], d["C"]
            Cc = 0.5 * C if i == j else 0.375 * C + 0.125 * C.T
            for name, Kx in (("mp2", K), ("3C", K + Cc)):
                T = Kx / Dab
                e = np.array([np.sum(0.5 * (T + T.T) * Kx), 3 * np.sum(0.5 * (T - T.T) * Kx)])
                res[name] += w * e
                pairs[name][i, j] += w * e
    # the 3C entry holds MP2 + coupling + F12: keep the correction only
    res["3C"] -= res["mp2"]
    pairs["3C"] -= pairs["mp2"]
    out = {k: tuple(v) for k, v in res.items()}
    if pair_energies:
        out["pairs"] = pairs
    return out


def ri_space(system, spaces, options, df, out):
    """RISpace of the task: CABS basis by the chain of driver.aux_basis_name, Fock matrices
    with the JK fitting basis of the system (df) or exact integrals.

    Returns:
        (RISpace, CABS basis name, how chosen).
    """
    from .driver import aux_basis_name
    name, how = aux_basis_name(system, "cabs", options, system.cabs_used)
    jk = None
    if df:
        jk = (system.df.basis if system.df is not None and not options.get("jk_auxbasis")
              else aux_basis_name(system, "jk", options, None)[0])
    a = spaces.alpha
    try:
        ri = build_ri_space(system.molecule, system.basis.name, a.C, a.nfrozen + a.nocc, name,
                            options["cabs_threshold"], jk, options["threshold"])
    except (ValueError, FileNotFoundError) as e:
        raise InputError(f"system '{system.name}': CABS: {e}") from None
    system.cabs_used = name
    return ri, name, how


def print_ri_space(ri, how, out):
    """Basis sets and CABS construction."""
    out.value("CABS basis", f"{ri.name} ({how}), {len(ri.aux_idx)} functions")
    out.value("RI basis", f"{ri.basis.nbf} functions (orbital basis + CABS)")
    out.value("CABS orbitals", f"{ri.ncabs} ({ri.dropped} dropped; smallest eigenvalue "
              f"{ri.eig_kept:.2e}, of the auxiliary overlap {ri.eig_aux:.2e})")
    out.value("Fock matrices", ("density fitting with " + ri.jk_auxbasis) if ri.jk_auxbasis
              else "exact integrals")
    out.value("CABS setup time", f"{ri.time:.2f} s")


def task_mp2_f12(system, options, ctx):
    """MP2-F12 (or DF-MP2-F12) with fixed amplitudes on top of the last SCF (RHF).

    Args:
        system: System (with an RHF wavefunction).
        options: Task options (all keys of its table in gemqc.options).
        ctx: RunContext.

    Returns:
        dict of results (scalars and vectors go to the JSON file).
    """
    out = ctx.out
    wf = system.wavefunction
    df = bool(options.get("df"))
    method = ("DF-" if df else "") + "MP2-F12"
    if wf is None:
        raise InputError(f"mp2-f12: needs an SCF task for system '{system.name}' before")
    out.section(method)
    spaces = orbital_spaces(wf, options["frozen"], options["deleted"], options["semicanonical"],
                            jk=system.scf_jk)
    if spaces.reference != "RHF":
        raise InputError("mp2-f12: closed-shell (RHF) references only so far")
    try:
        geminal = Geminal.make(options["gamma"], options["geminal"],
                               options["geminal_exponents"], options["geminal_coefficients"])
    except ValueError as e:
        raise InputError(f"mp2-f12: {e}") from None
    approx = _approximation(options)
    print_spaces(spaces, list(wf.irrep_names), out)
    out.value("geminal", geminal.describe())
    if not geminal.exact:
        out.value("  exponents", " ".join(f"{a:.5f}" for a in geminal.exponents))
        out.value("  coefficients", " ".join(f"{c:.5f}" for c in geminal.coefficients))
    out.value("approximation", approx + " (fixed amplitudes 1/2, 1/4; ansatz 3)")
    ri, cabs_name, how = ri_space(system, spaces, options, df, out)
    print_ri_space(ri, how, out)
    out.flush()

    a = spaces.alpha
    e_singles = cabs_singles_energy(ri, a, options, out)

    fit = None
    if df:
        fit = fitting_basis(system, options, out)
        ints = F12Integrals(ri, a.nfrozen, a.nocc, fit, geminal, options["integrals"])
        out.value("3-index integrals", f"{ints.store.describe()}, {ints.time:.2f} s")
    else:
        ints = ExactF12Integrals(ri, a.nfrozen, a.nocc, geminal, options["threshold"],
                                 options["integrals"])
        out.value("pair integrals", f"exact, {ints.store.describe()}, {ints.time:.2f} s")
    out.flush()
    res = pair_energies(ints, ri, a, options, out)
    data = report(system, options, ctx, method, spaces, res, e_singles, "mp2_f12",
                  {"cabs_auxbasis": cabs_name, "frozen": a.nfrozen, "gamma": geminal.gamma})
    return {"method": method, "reference": spaces.reference, **data,
            "gamma": geminal.gamma, "cabs_auxbasis": cabs_name, "ncabs": ri.ncabs,
            **({"density_fitting": {"c_auxbasis": fit.name, "naux": fit.nbf}} if df else {})}


def fitting_basis(system, options, out):
    """Correlation fitting basis of the F12 integrals (chain of driver.aux_basis_name,
    printed)."""
    from .driver import aux_basis_name
    fit_name, fit_how = aux_basis_name(system, "c", options, system.c_used or (
        system.df_c.basis if system.df_c is not None else None))
    system.c_used = fit_name
    try:
        fit = build_basis(system.molecule, fit_name, auxiliary=True)
    except (ValueError, FileNotFoundError) as e:
        raise InputError(f"system '{system.name}': correlation fitting basis: {e}") from None
    out.value("correlation fitting basis", f"{fit_name} ({fit_how}), {fit.nbf} functions")
    return fit


def cabs_singles_energy(ri, a, options, out):
    """CABS singles correction (printed) if options['cabs_singles'], else 0."""
    if not options["cabs_singles"]:
        return 0.0
    core = options["all_electron_cabs"] or a.nfrozen == 0
    occ = None if core else np.arange(a.nfrozen, a.nfrozen + a.nocc)
    e_singles = cabs_singles(ri, occ)[0]
    out()
    out.value("CABS singles", e_singles, "Eh", "18.12f")
    out.value("  occupied orbitals", "all" if core else "correlated (frozen core excluded)")
    out.flush()
    return e_singles


def pair_energies(ints, ri, a, options, out):
    """Pair intermediates and fixed-amplitude energies (fixed_amplitude_energies); closes
    the integrals."""
    t0 = time.time()
    try:
        inter = PairIntermediates(ints, ri, noz=options["noz"], hy1=options["hy1"],
                                  nvir_act=a.nvirt)
        res = fixed_amplitude_energies(inter, a.eps[a.occ], a.eps[a.virt], True)
    finally:
        ints.close()
    out.value("pair energies", f"{time.time() - t0:.2f} s")
    return res


def report(system, options, ctx, method, spaces, res, e_singles, group, attrs):
    """Energy table, totals, parts by core/valence pairs and pair energies (printed), HDF5
    group `group` (option save).

    Args:
        spaces: Orbital spaces (correlation.orbital_spaces).
        res: Result of fixed_amplitude_energies (with the pair energies).
        e_singles: CABS singles energy.
        group: HDF5 group name of the task.
        attrs: Further HDF5 attributes (e.g. frozen, gamma, cabs_auxbasis).

    Returns:
        dict of the energies, the approximation, the pair energies and the orbital spaces
        (JSON entries).
    """
    out = ctx.out
    a = spaces.alpha
    e_scf = system.wavefunction.energy
    tag = _tag(options)
    e_mp2 = sum(res["mp2"])
    out()
    out(f"  {'':34s} {'singlet':>18s} {'triplet':>18s} {'total':>18s}")
    rows = [("MP2", res["mp2"])] + [(f"F12 correction {n}{tag}", res[n]) for n in ("3*C", "3C")]
    for label, (s_, t_) in rows:
        out(f"  {label:34s} {s_:18.12f} {t_:18.12f} {s_ + t_:18.12f}")
    out()
    sel = _approximation(options).upper()
    e_f12 = sum(res[sel])
    out.value("SCF energy", e_scf, "Eh", "18.12f")
    out.value("CABS singles", e_singles, "Eh", "18.12f")
    out.value("reference energy", e_scf + e_singles, "Eh", "18.12f")
    out.value("MP2 correlation energy", e_mp2, "Eh", "18.12f")
    out.value("F12 correction", e_f12, "Eh", "18.12f")
    out.value("  approximation", sel + tag)
    out.value("MP2-F12 correlation energy", e_mp2 + e_f12, "Eh", "18.12f")
    out.value("MP2-F12 total energy", e_scf + e_singles + e_mp2 + e_f12, "Eh", "18.12f")
    core = has_correlated_core(spaces)
    if core:  # MP2 and F12 parts by pairs
        mask = a.core_mask()
        parts = {k: pair_partition(res["pairs"][k].sum(axis=-1), mask, mask)
                 for k in ("mp2", sel)}
        out()
        out(f"  Correlation energy by pairs / Eh (core: inner and outer core; F12: {sel})")
        out(f"  {'':24s}" + "".join(f"{h:>18s}" for h in ("MP2", "F12", "MP2-F12")))
        for n, name in enumerate(PARTS):
            m, f = parts["mp2"][n], parts[sel][n]
            out(f"    {name:22s}" + "".join(f"{v:18.12f}" for v in (m, f, m + f)))
    rows = _f12_pair_rows(res["pairs"], sel, a.occ_labels())
    if options["pair_energies"]:
        _print_f12_pairs(_f12_pair_rows(res["pairs"], sel, a.occ_labels(core)), sel, out, core)

    data = {"energy": e_scf + e_singles + e_mp2 + e_f12, "e_scf": e_scf,
            "e_reference": e_scf + e_singles, "e_cabs_singles": e_singles,
            "mp2_correlation_energy": e_mp2, "e_mp2_singlet": res["mp2"][0],
            "e_mp2_triplet": res["mp2"][1], "e_f12": e_f12,
            "e_f12_3c": sum(res["3C"]), "e_f12_3sc": sum(res["3*C"]),
            "e_f12_singlet": res[sel][0], "e_f12_triplet": res[sel][1],
            "correlation_energy": e_mp2 + e_f12}
    if core:
        for k, tag_ in (("mp2", "mp2"), (sel, "f12")):
            data.update({f"e_{tag_}_{p.replace('-', '_')}": float(v)
                         for p, v in zip(PARTS, parts[k])})
    from .mp2 import pair_json
    pairs_json = pair_json(rows, ("mp2_singlet", "mp2_triplet", "f12_singlet", "f12_triplet"))
    if options["save"]:
        h5 = {"e_f12": np.array(e_f12)}
        h5.update({f"pair_energies_{k.replace('*', 's').lower()}": v
                   for k, v in res["pairs"].items()})
        save_orbitals(ctx.h5_path, system.name, group, system.molecule, system.basis.name,
                      h5, {**data, "method": method, "approximation": sel + tag, **attrs})
        out(f"  results written to {ctx.h5_path.name} [{system.name}/{group}]")
    return {**data, "approximation": sel + tag, "pair_energies": pairs_json,
            **space_results(spaces, len(system.wavefunction.irrep_names))}


def _approximation(options):
    return options["approximation"].lower()


def _tag(options):
    flags = [f for f in ("noz", "hy1") if options[f]]
    return "(" + ",".join(["FIX"] + [f.upper() for f in flags]) + ")"


def _f12_pair_rows(pairs, sel, labels):
    """Rows (i, j, MP2 singlet, triplet, F12 singlet, triplet) for i >= j in energy order,
    with the orbital labels n.k of the active occupied orbitals; triplets None for i = j."""
    rows = []
    for i in range(pairs["mp2"].shape[0]):
        for j in range(i + 1):
            (ms, mt), (fs, ft) = pairs["mp2"][i, j], pairs[sel][i, j]
            rows.append((labels[i], labels[j], ms, mt if i != j else None, fs,
                         ft if i != j else None))
    return rows


def _print_f12_pairs(rows, sel, out, core=False):
    """MP2, F12 and MP2-F12 pair energies, singlet and triplet (triplet for i > j)."""
    from .mp2 import _fmt
    w, d = 16, 12
    out()
    note = "; *: core orbitals" if core else ""
    out(f"  Pair energies / Eh (i>=j; triplet pairs for i>j; F12: {sel}{note}):")
    out(f"  {'':13s}" + "".join(f" {h:^{2 * w + 1}s}" for h in ("MP2", "F12", "MP2-F12")).rstrip())
    out(f"  {'i':>6s} {'j':>6s}" + f" {'singlet':>{w}s} {'triplet':>{w}s}" * 3)
    for i, j, ms, mt, fs, ft in rows:
        tot = (ms + fs, None if mt is None else mt + ft)
        vals = (ms, mt, fs, ft) + tot
        out(f"  {i:>6s} {j:>6s}" + "".join(f" {_fmt(v, w, d)}" for v in vals).rstrip())

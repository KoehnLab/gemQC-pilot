"""Task mp2: second-order Moller-Plesset energy (spin-orbital equations).

Closed shells (RHF), with D_ij^ab = e_i + e_j - e_a - e_b and T_ij^ab = (ia|jb)/D_ij^ab::

    E_OS = sum_ijab T_ij^ab (ia|jb),  E_SS = sum_ijab T_ij^ab [(ia|jb) - (ib|ja)],
    e_ij = sum_ab T_ij^ab [2 (ia|jb) - (ib|ja)]  (pair energies).

Open shells (UHF orbitals, or semicanonical ROHF orbitals of correlation.orbital_spaces)::

    E_SS = 1/4 sum_sigma sum_ijab [(ia|jb) - (ib|ja)]^2 / D  (same spin),
    E_OS = sum_{i a alpha, j b beta} (ia|jb)^2 / D  (opposite spin),
    E_singles = sum_sigma sum_ia f_ia^2 / (e_i - e_a)  (ROHF: not Brillouin).

SCS-MP2 (Grimme): E = 6/5 E_OS + 1/3 E_SS (+ singles).
"""

import numpy as np

from . import _core
from .correlation import (PARTS, has_correlated_core, orbital_spaces, pair_partition,
                          print_spaces, space_results)
from .options import InputError
from .restart import save_orbitals
from .transform import MOIntegrals

SCS_OS, SCS_SS = 6.0 / 5.0, 1.0 / 3.0


def _denominator(eo1, ev1, eo2, ev2, i0, i1):
    """D[i, a, j, b] = e_i + e_j - e_a - e_b for rows i0..i1."""
    return (eo1[i0:i1, None, None, None] - ev1[None, :, None, None]
            + eo2[None, None, :, None] - ev2[None, None, None, :])


def mp2_energy(spaces, ints, pair_energies=False):
    """MP2 correlation energy from the integrals of transform.MOIntegrals.

    Args:
        spaces: correlation.OrbitalSpaces.
        ints: MOIntegrals with the class 'ovov'.
        pair_energies: Also return the pair energies.

    Returns:
        dict: os, ss, singles, corr (= os + ss + singles), scs (SCS-MP2 correlation);
        with pair_energies: pairs {case: matrix}: closed shells 'singlet' and 'triplet'
        (lower triangle i >= j, pair ij and ji together; triplet zero for i = j), open
        shells per spin case 'aa', 'bb', 'ab' (e_ij; e_ij + e_ji is the pair ij).
    """
    a, b = spaces.alpha, spaces.beta
    eps = {"a": (a.eps[a.occ], a.eps[a.virt]), "b": (b.eps[b.occ], b.eps[b.virt])}
    res = {"os": 0.0, "ss": 0.0, "singles": 0.0}
    pairs = {}
    if ints.closed:
        eo, ev = eps["a"]
        n = len(eo)
        coul, exch = np.zeros((n, n)), np.zeros((n, n))
        for i0, i1, X in ints.blocks("aa"):
            T = X / _denominator(eo, ev, eo, ev, i0, i1)
            Xs = X.transpose(0, 3, 2, 1)  # (ib|ja)
            coul[i0:i1] = np.einsum("iajb,iajb->ij", T, X)
            exch[i0:i1] = np.einsum("iajb,iajb->ij", T, Xs)
        res["os"] += coul.sum()
        res["ss"] += coul.sum() - exch.sum()
        # pair ij and ji: singlet (coul + exch), triplet 3 (coul - exch); i = j: singlet only
        low = np.tril(np.ones((n, n)), -1)
        pairs["singlet"] = low * (coul + exch) + np.diag(np.diag(coul))
        pairs["triplet"] = low * 3.0 * (coul - exch)
    else:
        for case in ints.cases:
            eo1, ev1 = eps[case[0]]
            eo2, ev2 = eps[case[1]]
            e = np.zeros((len(eo1), len(eo2)))
            for i0, i1, X in ints.blocks(case):
                D = _denominator(eo1, ev1, eo2, ev2, i0, i1)
                if case == "ab":
                    e[i0:i1] = np.einsum("iajb,iajb->ij", X, X / D)
                else:
                    A = X - X.transpose(0, 3, 2, 1)
                    e[i0:i1] = 0.25 * np.einsum("iajb,iajb->ij", A, A / D)
            res["os" if case == "ab" else "ss"] += e.sum()
            pairs[case] = e
        if spaces.reference == "ROHF":
            for s in (a, b):
                f = s.F[s.occ, s.virt]
                d = s.eps[s.occ][:, None] - s.eps[s.virt][None, :]
                res["singles"] += float(np.sum(f * f / d))
    res["corr"] = res["os"] + res["ss"] + res["singles"]
    res["scs"] = SCS_OS * res["os"] + SCS_SS * res["ss"] + res["singles"]
    if pair_energies:
        res["pairs"] = pairs
    return res


def rc_builder(system, options, out):
    """DFBuilder in the correlation fitting basis of a system (built once; rebuilt
    if the task option c_auxbasis asks for another one)."""
    from .basislib import build_basis
    from .driver import aux_basis_name
    if system.df_c is not None and options.get("c_auxbasis") \
            and options["c_auxbasis"] != system.df_c.basis:
        system.df_c = None
    if system.df_c is not None and system.c_used and system.c_used != system.df_c.basis:
        system.df_c = None  # another basis was chosen by a later task (as before)
    if system.df_c is None:
        try:
            name, how = aux_basis_name(system, "c", options, system.c_used)
            aux = build_basis(system.molecule, name, auxiliary=True)
        except (ValueError, FileNotFoundError) as e:
            raise InputError(f"system '{system.name}': density fitting (correlation): {e}") from None
        from types import SimpleNamespace
        try:
            builder = _core.DFBuilder(system.basis, aux, storage=options.get("jk_storage") or "auto")
        except RuntimeError as e:
            raise InputError(f"system '{system.name}': {e} (use jk_storage: raw)") from None
        system.df_c = SimpleNamespace(builder=builder, basis=name, how=how, naux=aux.nbf)
        system.c_used = name
        out(f"  correlation fitting basis {name} ({how}), {aux.nbf} functions")
    else:
        out(f"  correlation fitting basis {system.df_c.basis} (as before)")
    return system.df_c.builder


def task_mp2(system, options, ctx):
    """MP2 (or DF-MP2) energy on top of the last SCF of the system.

    Args:
        system: System (with an RHF, ROHF or UHF wavefunction).
        options: Task options (all keys of its table in gemqc.options).
        ctx: RunContext.

    Returns:
        dict of results (scalars and vectors go to the JSON file).
    """
    out = ctx.out
    wf = system.wavefunction
    if wf is None:
        raise InputError(f"mp2: needs an SCF task for system '{system.name}' before")
    df = bool(options.get("df"))
    out.section(("DF-" if df else "") + "MP2")
    spaces = orbital_spaces(wf, options["frozen"], options["deleted"], options["semicanonical"],
                            jk=system.scf_jk)  # spin Fock matrices as in the SCF (ROHF)
    print_spaces(spaces, list(wf.irrep_names), out)
    if df:
        builder = rc_builder(system, options, out)
    else:
        builder = _core.FockBuilder(system.basis, options["threshold"])
    out.flush()
    ints = MOIntegrals(spaces, builder, df, options["integrals"])
    try:
        res = mp2_energy(spaces, ints, pair_energies=True)
    finally:
        out.value("integrals", ints.describe())
        ints.close()
    e_scf = wf.energy
    out()
    if spaces.reference != "RHF":
        if spaces.reference == "UHF":
            out.value("<S^2> of the reference", f"{wf.s2:.6f}")
        out.value("singles energy", res["singles"], "Eh", "18.12f")
    out.value("opposite-spin energy", res["os"], "Eh", "18.12f")
    out.value("same-spin energy", res["ss"], "Eh", "18.12f")
    out.value("MP2 correlation energy", res["corr"], "Eh", "18.12f")
    out.value("SCS-MP2 correlation energy", res["scs"], "Eh", "18.12f")
    out.value("SCF energy", e_scf, "Eh", "18.12f")
    out.value("SCS-MP2 total energy", e_scf + res["scs"], "Eh", "18.12f")
    out.value("MP2 total energy", e_scf + res["corr"], "Eh", "18.12f")
    core = has_correlated_core(spaces)
    parts = _partition(res["pairs"], spaces) if core else None
    if core:
        out()
        out("  Correlation energy by pairs (core: inner and outer core)")
        for name, v in zip(PARTS, parts):
            out.value(f"  {name}", v, "Eh", "18.12f")
        if spaces.reference == "ROHF":
            out.value("  singles", res["singles"], "Eh", "18.12f")
    rows = pair_rows(res["pairs"], spaces)
    if options["pair_energies"]:
        _print_pairs(spaces, pair_rows(res["pairs"], spaces, core), out, core)
    data = {"energy": e_scf + res["corr"], "correlation_energy": res["corr"],
            "e_os": res["os"], "e_ss": res["ss"], "e_singles": res["singles"],
            "scs_energy": e_scf + res["scs"]}
    if core:
        data.update({f"e_{p.replace('-', '_')}": float(v) for p, v in zip(PARTS, parts)})
    if spaces.reference == "RHF":
        data_pairs = {"pair_energies": pair_json(rows, ("singlet", "triplet"))}
    else:
        data_pairs = {"pair_energies": {_CASES[c]: pair_json(r, ("e",)) for c, r in rows.items()}}
    if options["save"]:
        pdata = {f"pair_energies_{k}": v for k, v in res.get("pairs", {}).items()}
        save_orbitals(ctx.h5_path, system.name, "mp2", system.molecule, system.basis.name,
                      pdata or {"correlation_energy": np.array(res["corr"])},
                      {**data, "reference": spaces.reference, "method": ("DF-" if df else "") + "MP2",
                       "e_scf": e_scf, "frozen": spaces.alpha.nfrozen,
                       "deleted": spaces.alpha.ndeleted})
        out(f"  results written to {ctx.h5_path.name} [{system.name}/mp2]")
    from .driver import _df_c_results
    return {"method": ("DF-" if df else "") + "MP2", "reference": spaces.reference,
            "e_scf": e_scf, **data, **data_pairs, **_df_c_results(system, df),
            **space_results(spaces, len(wf.irrep_names))}


def pair_rows(pairs, spaces, flag_core=False):
    """Pair energies as rows (i, j, values...) with the orbital labels n.k of the correlated
    occupied orbitals (i after j in energy order; flag_core: core orbitals marked '*'):
    closed shells (i, j, singlet, triplet) (triplet None for i = j); open shells
    {case: [(i, j, e)]} (same spin: e_ij + e_ji)."""
    la, lb = spaces.alpha.occ_labels(flag_core), spaces.beta.occ_labels(flag_core)
    if "singlet" in pairs:
        S, T = pairs["singlet"], pairs["triplet"]
        return [(la[i], la[j], S[i, j], T[i, j] if i != j else None)
                for i in range(len(S)) for j in range(i + 1)]
    lab = {"aa": (la, la), "bb": (lb, lb), "ab": (la, lb)}
    return {case: [(lab[case][0][i], lab[case][1][j],
                    e[i, j] + (e[j, i] if case != "ab" else 0.0))
                   for i in range(e.shape[0]) for j in range(e.shape[1])
                   if case == "ab" or i > j] for case, e in pairs.items()}


def pair_json(rows, names):
    """JSON form of pair rows: {'i': [...], 'j': [...], name: [...]} (None -> 0.0)."""
    cols = list(zip(*rows)) if rows else [()] * (2 + len(names))
    out = {"i": list(cols[0]), "j": list(cols[1])}
    for k, name in enumerate(names):
        out[name] = [0.0 if v is None else float(v) for v in cols[2 + k]]
    return out


def _fmt(v, width=18, digits=12):
    return " " * width if v is None else f"{v:{width}.{digits}f}"


_CASES = {"aa": "alpha-alpha", "bb": "beta-beta", "ab": "alpha-beta"}


def _partition(pairs, spaces):
    """Core, core-valence and valence parts of the MP2 pair energies (without singles)."""
    a, b = spaces.alpha.core_mask(), spaces.beta.core_mask()
    if "singlet" in pairs:
        return pair_partition(pairs["singlet"] + pairs["triplet"], a, a)
    masks = {"aa": (a, a), "bb": (b, b), "ab": (a, b)}
    return sum(pair_partition(e, *masks[case]) for case, e in pairs.items())


def _print_pairs(spaces, rows, out, core=False):
    """Pair energies (orbital labels n.k of the correlated occupied orbitals, core
    orbitals marked '*' if correlated)."""
    out()
    note = "; *: core orbitals" if core else ""
    if spaces.reference == "RHF":
        out(f"  Pair energies / Eh (i>=j; triplet pairs for i>j{note}):")
        out(f"  {'i':>6s} {'j':>6s} {'singlet':>18s} {'triplet':>18s}")
        for i, j, s, t in rows:
            out(f"  {i:>6s} {j:>6s} {_fmt(s)} {_fmt(t)}".rstrip())
        return
    for case, r in rows.items():
        out(f"  Pair energies {_CASES[case]} / Eh" + (f" ({note[2:]})" if core else "") + ":")
        out.table(["i", "j", "e_ij"], r, [">6s", ">6s", "18.12f"])

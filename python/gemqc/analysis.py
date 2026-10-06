"""Orbital analysis: per-orbital Mulliken weights (orbital character).

Each orbital is normalized, sum(m,n) C(m,i) S(m,n) C(n,i) = 1; the terms
C(m,i) (SC)(m,i) are grouped by atom and angular momentum of the AO m."""

import numpy as np

from .output import _LNAMES

WIDTH = 132  # maximum output line length


def ao_groups(mol, basis, sym=None, merge=True):
    """Group of each AO by atom and angular momentum.

    Args:
        mol: Molecule.
        basis: BasisSet.
        sym: Symmetry object (for merging), or None.
        merge: Merge symmetry-equivalent atoms: 'O p' (one class of this element) or
            'H(2) s' (the H atoms equivalent to atom 2); else per atom, '1O p'.

    Returns:
        (group index of each AO (nbf,), group names).
    """
    natoms, syms = mol.natoms, mol.symbols
    cls = list(range(natoms))  # class of each atom: lowest equivalent atom
    if merge and sym is not None:
        for a in range(natoms):
            cls[a] = int(min(sym.atom_map[:, a]))
    reps = sorted(set(cls))
    per_element = {}
    for r in reps:
        per_element[syms[r]] = per_element.get(syms[r], 0) + 1
    def atom_name(r):
        if not merge or sym is None:
            return f"{r + 1}{syms[r]}"
        return syms[r] if per_element[syms[r]] == 1 else f"{syms[r]}({r + 1})"
    shells = [basis.shell(s) for s in range(basis.nshells)]
    keys = sorted({(cls[sh["atom"]], sh["l"]) for sh in shells})  # atom class, then l
    index = {k: g for g, k in enumerate(keys)}
    group = np.empty(basis.nbf, dtype=int)
    for s, sh in enumerate(shells):
        n = 2 * sh["l"] + 1 if sh["pure"] else (sh["l"] + 1) * (sh["l"] + 2) // 2
        o = basis.shell_offsets[s]
        group[o:o + n] = index[(cls[sh["atom"]], sh["l"])]
    return group, [f"{atom_name(a)} {_LNAMES[l]}" for a, l in keys]


def mulliken_weights(C, S, group, ngroups):
    """Per-orbital Mulliken weights of AO groups.

    Args:
        C: Orbitals (nbf, norb), S-normalized.
        S: Overlap matrix.
        group: Group index of each AO.
        ngroups: Number of groups.

    Returns:
        Weights (ngroups, norb); each column sums to 1.
    """
    W = np.asarray(C) * (np.asarray(S) @ np.asarray(C))
    G = np.zeros((ngroups, len(group)))
    G[group, np.arange(len(group))] = 1.0
    return G @ W


def character_items(w, names, threshold=0.6):
    """Largest weights (by absolute value) until their sum reaches threshold.

    Returns:
        Strings like '0.712 O p'; equal weights in group order.
    """
    items, acc = [], 0.0
    order = sorted(range(len(w)), key=lambda g: (-round(abs(w[g]), 3), g))  # ties: atom order
    for g in order:
        if acc >= threshold or abs(w[g]) < 5e-4:
            break
        items.append(f"{w[g]:.3f} {names[g]}")
        acc += w[g]
    return items


def wrap_items(head, items, width=WIDTH, sep="  "):
    """Lines of head followed by items, at most width characters.

    Continuation lines start in the column where the items start in the first line.
    """
    lines, line, col = [], head, len(head)
    first = True
    for it in items:
        add = it if first else sep + it
        if not first and len(line) + len(add) > width:
            lines.append(line)
            line, add = " " * col, it
        line += add
        first = False
    lines.append(line)
    return lines


class Character:
    """Orbital character for printouts (groups of a system and options).

    Args:
        system: System.
        threshold: Accumulated weight up to which weights are listed (0: off).
        merge: Merge symmetry-equivalent atoms.
    """

    def __init__(self, system, threshold=0.6, merge=True):
        from . import _core
        self.threshold = float(threshold)
        self.merge = merge
        self.group, self.names = ao_groups(system.molecule, system.basis, system.symmetry, merge)
        self.S = _core.overlap(system.basis)

    @property
    def enabled(self):
        """True unless the threshold is 0."""
        return self.threshold > 0

    def heading(self):
        """Explanation line(s) for the output."""
        how = "symmetry-equivalent atoms merged" if self.merge else "per atom"
        text = (f"character: Mulliken weights per atom and l ({how}), "
                f"largest until their sum reaches {self.threshold}")
        if any("(" in n for n in self.names):
            text += "\n  (X(n): the atoms X equivalent to atom n)"
        return text

    def items(self, C):
        """Item lists (character_items) of the orbitals in the columns of C."""
        W = mulliken_weights(C, self.S, self.group, len(self.names))
        return [character_items(W[:, i], self.names, self.threshold) for i in range(W.shape[1])]

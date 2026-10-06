"""Formatted screen output (can be switched off)."""

import sys

import numpy as np

_LNAMES = "spdfghiklmnoqrtuvwxyz"
_WIDTH = 72

# startup logo: a pair of cut gems (geminal) beside a figlet-style title
_GEM = [
    r"    *        *",
    r"   /\      /\ ",
    r"  /  \    /  \ ",
    r" /____\--/____\ ",
    r" \    /  \    /",
    r"  \  /    \  /",
    r"   \/      \/",
]
_TITLE = [
    r"                ___   ___         _ _     _   ",
    r" __ _ ___ _ __ / _ \ / __|__ _ __(_) |___| |_ ",
    r"/ _` / -_) '  \ (_) | (_|___| '_ \ | / _ \  _|",
    r"\__, \___|_|_|_\__\_\\___|  | .__/_|_\___/\__|",
    r"|___/                       |_|               ",
]


def logo(version, gap=3):
    """Startup logo lines: gem with the title beside it, version on the last row."""
    width = max(len(g) for g in _GEM) + gap
    right = [""] + _TITLE + [f"v{version}"]
    right += [""] * (len(_GEM) - len(right))
    return [(g.ljust(width) + r).rstrip() for g, r in zip(_GEM, right)]


class Output:
    """Printer for human-readable output; silent if enabled=False.

    Args:
        enabled: Print (else all methods do nothing).
        stream: Output stream (default stdout).
    """

    def __init__(self, enabled=True, stream=None):
        self.enabled = enabled
        self.stream = stream or sys.stdout

    def __call__(self, text=""):
        if self.enabled:
            print(text, file=self.stream)

    def flush(self):
        """Push buffered output out (progress lines, e.g. per SCF iteration)."""
        if self.enabled:
            self.stream.flush()

    def banner(self, text):
        """Title centred between rules."""
        self("=" * _WIDTH)
        self(text.center(_WIDTH).rstrip())
        self("=" * _WIDTH)

    def logo(self, version):
        """Startup logo between rules."""
        self("=" * _WIDTH)
        for line in logo(version):
            self("  " + line)
        self("\n     ** gemQC - quantum chemistry with geminals **\n")
        self(" conceived and directed by A. Koehn, University of Stuttgart, 2026")
        self(" coded using an AI agent-based approach\n")
        self("=" * _WIDTH)

    def section(self, title):
        """Blank line, title and underline."""
        self()
        self(title)
        self("-" * len(title))

    def value(self, label, value, unit="", fmt=""):
        """Line 'label  value unit' with the value formatted by fmt."""
        v = format(value, fmt) if fmt else str(value)
        self(f"  {label:<28s} {v} {unit}".rstrip())

    def table(self, headers, rows, fmts):
        """Aligned table; fmts are format specs incl. width, e.g. '>4s', '14.8f'."""
        widths = [max([len(h)] + [len(format(r[i], f)) for r in rows])
                  for i, (h, f) in enumerate(zip(headers, fmts))]
        self("  " + "  ".join(f"{h:>{w}s}" for h, w in zip(headers, widths)))
        for r in rows:
            self("  " + "  ".join(f"{format(v, f):>{w}s}" for v, f, w in zip(r, fmts, widths)))

    def matrix(self, m, labels=None, ncol=6, fmt="12.6f"):
        """Matrix in column blocks with optional basis function labels."""
        m = np.asarray(m)
        labels = labels or [str(i + 1) for i in range(m.shape[0])]
        w = int(fmt.split(".")[0])
        lw = max(len(s) for s in labels)
        for c0 in range(0, m.shape[1], ncol):
            cols = range(c0, min(c0 + ncol, m.shape[1]))
            self(" " * (lw + 2) + "".join(f"{labels[c]:>{w}s}" for c in cols))
            for r in range(m.shape[0]):
                self(f"  {labels[r]:<{lw}s}" + "".join(format(m[r, c], fmt) for c in cols))
            self()


def _component_labels(l, pure):
    """Labels of the functions of a shell in libint ordering."""
    if l == 0:
        return ["s"]
    if pure:
        if l == 1:
            return ["py", "pz", "px"]  # m = -1, 0, +1
        return [f"{_LNAMES[l]}{m:+d}" if m else f"{_LNAMES[l]}0" for m in range(-l, l + 1)]
    labs = []
    for i in range(l, -1, -1):  # x^i y^j z^k, libint standard (CCA) order
        for j in range(l - i, -1, -1):
            labs.append("x" * i + "y" * j + "z" * (l - i - j) or "s")
    return labs


def bf_labels(mol, basis):
    """Label of each basis function, e.g. '1O d+2' (atom number, element, component)."""
    syms = mol.symbols
    labels = []
    for i in range(basis.nshells):
        sh = basis.shell(i)
        a = sh["atom"]
        labels += [f"{a + 1}{syms[a]} {c}" for c in _component_labels(sh["l"], sh["pure"])]
    return labels

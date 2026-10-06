"""RHF, UHF and ROHF energies vs. Molpro 2025.5 (closed-shell RHF: gthresh
energy=1d-8, orbital=1d-8, nosym; UHF and ROHF: default thresholds, CH2 in C1,
NH2 in C2v).

Butane detects the lost-Schwarz-pair bug (distant tight-s/d pairs, fixed):
the buggy build was 1.6e-8 Eh too low.
"""

from pathlib import Path

import pytest

import gemqc
from gemqc import _core

DATA = Path(__file__).parent / "data"

REFERENCES = [  # (xyz, basis, Molpro energy / Eh)
    ("h2o.xyz", "def2-SVP", -75.960841436478),
    ("h2o.xyz", "def2-TZVPP", -76.062277305532),
    ("butane.xyz", "def2-SVP", -157.184381601459),
]


@pytest.mark.parametrize("xyz, basis, e_ref", REFERENCES)
def test_rhf_vs_molpro(xyz, basis, e_ref):
    _core.set_output_enabled(False)
    try:
        mol = gemqc.read_xyz(DATA / xyz)
        bs = gemqc.build_basis(mol, basis)
        opt = gemqc.ScfOptions()
        opt.convergence, opt.energy_convergence = 1e-8, 1e-10
        wf = gemqc.rhf(mol, bs, opt, gemqc.sad_guess(mol, bs))
    finally:
        _core.set_output_enabled(True)
    assert wf.converged
    assert wf.energy == pytest.approx(e_ref, abs=1e-9)


UHF_REFERENCES = [  # (xyz, multiplicity, Molpro energy / Eh, <S^2>)
    ("ch2.xyz", 3, -38.893808306337, 2.01732165),
    ("nh2.xyz", 2, -55.515719320942, 0.75857639),
]


@pytest.mark.parametrize("xyz, mult, e_ref, s2_ref", UHF_REFERENCES)
def test_uhf_vs_molpro(xyz, mult, e_ref, s2_ref):
    _core.set_output_enabled(False)
    try:
        mol = gemqc.read_xyz(DATA / xyz)
        mol.multiplicity = mult
        bs = gemqc.build_basis(mol, "def2-SVP")
        opt = gemqc.ScfOptions()
        opt.convergence, opt.energy_convergence = 1e-8, 1e-10
        wf = gemqc.uhf(mol, bs, opt, [gemqc.sad_guess(mol, bs)])
    finally:
        _core.set_output_enabled(True)
    assert wf.converged
    assert wf.energy == pytest.approx(e_ref, abs=1e-9)
    assert wf.s2 == pytest.approx(s2_ref, abs=1e-7)


ROHF_REFERENCES = [  # (xyz, multiplicity, Molpro energy / Eh, orbital energies:
    ("ch2.xyz", 3, -38.888300592063,  # closed, open, virtual, each by energy)
     [-11.250961, -0.833265, -0.595294, -0.465494, -0.404360, 0.198980, 0.253922]),
    ("nh2.xyz", 2, -55.511421521766,
     [-15.581042, -1.073532, -0.615217, -0.478463, -0.501448, 0.173605, 0.252552]),
]


@pytest.mark.parametrize("xyz, mult, e_ref, eps_ref", ROHF_REFERENCES)
def test_rohf_vs_molpro(xyz, mult, e_ref, eps_ref):
    """Energy and canonical orbital energies (closed (Fa + Fb)/2, open and
    virtual Fa, as in Molpro)."""
    _core.set_output_enabled(False)
    try:
        mol = gemqc.read_xyz(DATA / xyz)
        mol.multiplicity = mult
        bs = gemqc.build_basis(mol, "def2-SVP")
        opt = gemqc.ScfOptions()
        opt.convergence, opt.energy_convergence = 1e-8, 1e-10
        wf = gemqc.rohf(mol, bs, opt, [gemqc.sad_guess(mol, bs)])
    finally:
        _core.set_output_enabled(True)
    assert wf.converged and wf.method == "ROHF"
    assert wf.energy == pytest.approx(e_ref, abs=1e-9)
    assert wf.eps[:7] == pytest.approx(eps_ref, abs=2e-6)

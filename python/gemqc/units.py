"""Unit handling at the input/output boundary (internal data: atomic units)."""

from ._core import units as _u

BOHR_IN_ANGSTROM = _u.bohr_in_angstrom
HARTREE_IN_EV = _u.hartree_in_ev
HARTREE_IN_KJMOL = _u.hartree_in_kjmol
AU_DIPOLE_IN_DEBYE = _u.au_dipole_in_debye
HARTREE_IN_WAVENUMBER = _u.hartree_in_wavenumber
FINE_STRUCTURE = _u.fine_structure
ELECTRON_G_FACTOR = _u.electron_g_factor

# length units: factor converting value in unit -> bohr
_LENGTH = {
    "bohr": 1.0,
    "au": 1.0,
    "angstrom": 1.0 / BOHR_IN_ANGSTROM,
    "ang": 1.0 / BOHR_IN_ANGSTROM,
    "a": 1.0 / BOHR_IN_ANGSTROM,
}


def length_factor(unit: str) -> float:
    """Factor converting lengths in `unit` to bohr.

    Raises:
        ValueError: Unknown unit.
    """
    try:
        return _LENGTH[unit.strip().lower()]
    except KeyError:
        raise ValueError(f"unknown length unit '{unit}' "
                         f"(known: {', '.join(sorted(_LENGTH))})") from None

#pragma once
/// \file
/// Physical constants and unit conversion factors (CODATA 2018).
/// All internal quantities in gemQC are in atomic units.

namespace gemqc::units {

inline constexpr double bohr_in_angstrom = 0.529177210903;  ///< a0 in Angstrom
inline constexpr double angstrom_in_bohr = 1.0 / bohr_in_angstrom;
inline constexpr double hartree_in_ev = 27.211386245988;
inline constexpr double hartree_in_kjmol = 2625.4996394799;
inline constexpr double au_dipole_in_debye = 2.541746473;  ///< e*a0 in Debye
inline constexpr double hartree_in_wavenumber = 219474.6313632;  ///< Eh in cm^-1
inline constexpr double fine_structure = 7.2973525693e-3;        ///< alpha = 1/c (au)
inline constexpr double electron_g_factor = 2.00231930436256;    ///< |g_e|

}  // namespace gemqc::units

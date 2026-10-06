# Basis sets and initial guess

## Basis sets

Basis sets are JSON files from [basissetexchange.org](https://www.basissetexchange.org),
looked up by name (case-insensitive) in `$GEMQC_DIR/lib/basis`, then in `lib/basis`
of the repository. Spherical and Cartesian shells are supported; the function order
within a shell follows libint (spherical: $m = -l, \dots, l$). Basis sets with
effective core potentials (def2 for Rb and heavier) are not supported yet and are
rejected.

Core-valence basis sets have no functions for H and He: for these, any p(w)CVXZ set
(all types, e.g. aug-cc-pwCVTZ, aug-cc-pwCVTZ-C) falls back to pVXZ (aug-cc-pVTZ,
aug-cc-pVTZ-C) if the file lacks them; the output marks such elements, e.g.
`H (6s3p2d) -> [4s3p2d]  (from aug-cc-pVTZ)`. An auxiliary basis of a p(w)CVXZ set
without a table entry or file is the default of pVXZ (e.g. aug-cc-pwCVTZ: JK fitting
aug-cc-pVTZ-JK; correlation fitting aug-cc-pwCVTZ-C from its file). If
the fitting basis of a p(w)CVXZ set lacks H or He, these atoms take the fitting basis of
pVXZ (cc-pCVTZ-F12: correlation fitting `aug-cc-pwCVTZ-C;O=cc-pCVTZ-F12-C` for water).

Different basis sets for elements or single atoms are given as a mapping, see
[Systems](../input/index.md#basis-sets-per-element-and-atom); the shells are then placed
per atom (`BasisSet` from a list of shell definitions per atom), and the basis is
named by its canonical name, e.g. `cc-pVTZ;O=aug-cc-pVTZ;4=cc-pVQZ`.

## Auxiliary basis sets

Correlated and density-fitted tasks use auxiliary basis sets of three kinds, chosen per
task (option, else the one used before for the system, else the system key, else the
default table `lib/basis/auxbasis.yaml`); with several orbital basis sets the default
is taken per atom from its orbital basis, and a mapping may name only some atoms:

- `jk_auxbasis`: JK fitting (Coulomb and exchange) of the `df-` tasks, files `<basis>-JK`
  (e.g. cc-pVTZ-JK; cc-pVDZ uses cc-pVTZ-JK; def2 sets use def2-universal-JK;
  cc-p(C)VXZ-F12 use cc-pV(X+1)Z-JK);
- `c_auxbasis`: correlation (RI, MP2FIT) fitting, files `<basis>-C` (cc-pVXZ-F12 use
  aug-cc-pwCVXZ-C, cc-pCVXZ-F12 their own files);
- `cabs_auxbasis`: complementary auxiliary basis of F12 methods, files `<basis>-CABS`
  (aug-cc-pVXZ-CABS = OptRI, cc-p(C)VXZ-F12-CABS).

The defaults leave gaps for some elements: cc-pVXZ-JK lack He, Li, Be, Na, Mg,
aug-cc-pwCVXZ-C lack Li, Be, Na, Mg, and cc-p(C)VTZ-F12-CABS and cc-p(C)VQZ-F12-CABS
lack He; give the fitting basis explicitly there (e.g. `jk_auxbasis: def2-universal-JK`).

aug-cc-pVXZ-JK (X = T, Q, 5) are not on the Basis Set Exchange. They are generated
from cc-pVXZ-JK by `lib/basis/augment_jk.py` (one diffuse function per element and
angular momentum: the even-tempered extension of the two smallest exponents, or a/2.5
for a single exponent) and equal Molpro's AVXZ/JKFIT; see corresponding file headers. The
cc-pVXZ-JK sets of Ne, Ar and Kr are missing on the Basis Set Exchange; they are taken
from Molpro's library (extrapolated sets, noted in the element references).

## SAD guess

The superposition of atomic densities uses neutral atoms in their ground
configuration (aufbau with the usual exceptions, e.g. Cr, Cu), spherically averaged
over partially filled shells, in the minimal ANO-R0 basis (whose contractions are the
atomic natural orbitals 1s, 2s, 2p, ...). The molecular density is projected onto the
actual basis, $D = S^{-1} S_{TA} D_A S_{AT} S^{-1}$; the output reports how many
electrons the projection keeps.

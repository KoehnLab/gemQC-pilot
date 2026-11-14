# Basis set library

The basis set data files in this directory (`*.json`) are not covered by the
gemQC-pilot license (`LICENSE`, BSD 3-Clause). They are included for convenience
only and remain subject to the terms of their sources:

- Data obtained from the [Basis Set Exchange](https://www.basissetexchange.org)
  (BSE JSON format; the source of each set is given in its `revision_description`).
- The `aug-cc-pVXZ-JK` sets (X = T, Q, 5) are derived from the BSE cc-pVXZ-JKFIT sets
  by `augment_jk.py` (one diffuse function per angular momentum).

When using these basis sets, please cite the Basis Set Exchange

- B. P. Pritchard, D. Altarawy, B. Didier, T. D. Gibson, T. L. Windus, J. Chem. Inf.
  Model. **59**, 4814 (2019), [doi:10.1021/acs.jcim.9b00725](https://doi.org/10.1021/acs.jcim.9b00725)

and the original papers of the basis sets, as given by the Basis Set Exchange.

The other files in this directory (`augment_jk.py`, `auxbasis.yaml`, this README)
are part of gemQC-pilot and covered by its license.

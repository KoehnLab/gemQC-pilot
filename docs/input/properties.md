# Properties (option `properties`)

After `scf`, `rhf`, `rohf` and `uhf` the dipole and quadrupole moments are printed by
default (`properties: false` switches this off). The block starts with the origin of the
moments:

```
  Properties   (origin of the moments: centre of mass, 0.000000 0.000000 0.618924 bohr)
```

- **Origin**: the centre of mass (standard atomic weights), in the computational frame.
  Dipole moments of neutral molecules and quadrupole moments of molecules without
  dipole do not depend on it.
- **Dipole moment** μ = −Σ D_μν ⟨μ|r|ν⟩ + Σ_A Z_A R_A (au; the norm also in Debye),
  electronic, nuclear and total.
- **Quadrupole moment**: traceless (Buckingham),
  Θ_ab = ½ Σ_i q_i (3 r_ia r_ib − r_i² δ_ab), in au, components xx, yy, zz, xy, xz, yz.

## Densities

Densities are stored in the HDF5 file (`save: true`, see [Output](../output.md)):

| task | datasets |
|---|---|
| scf (RHF, ROHF), uhf | `D` (total), `D_spin` (α − β; UHF and ROHF) |

The moments are stored as attributes (`dipole`, `quadrupole`, `properties_origin`).
The JSON results contain `dipole`, `dipole_debye`, `quadrupole` and `properties_origin`.

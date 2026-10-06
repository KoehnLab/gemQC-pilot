# Quick start

## A first calculation

`water.yaml`, next to an XYZ file `h2o.xyz` (coordinates in Å):

```yaml
systems:
  water:
    coordinates: h2o.xyz
    basis: def2-SVP
tasks:
  - system: water
    task: scf
    orbitals: long
```

```bash
gemqc water.yaml                # or: python -m gemqc water.yaml
gemqc -q water.yaml             # no screen output
gemqc -m 4GB -t 8 water.yaml    # memory limit and threads (override the input)
gemqc -a runs/w1 water.yaml     # results runs/w1.json, restart data runs/w1.h5
```

The output goes to the screen (redirect it to a file); the main results are
written to `water.json`, restart data (molecule, basis, orbitals, densities) to
`water.h5`, both next to the input, or to the path and base name given with `-a`
(`--archive`; also used for cube files and for restart data read by later runs with
the same `-a`). Errors are printed on both stdout and stderr.

## More tasks

Tasks run in the given order; a task without `system` uses the previous one.
Later tasks use the results of earlier ones (in this run, or from the restart file):

```yaml
systems:
  ch2: {coordinates: ch2.xyz, basis: def2-SVP, multiplicity: 3}
tasks:
  - {system: ch2, task: uhf}
  - {task: mp2}
```

See [the input reference](input/index.md) for all tasks and options.

## Python

```python
import gemqc

mol = gemqc.read_xyz("h2o.xyz")            # coordinates stored in bohr
basis = gemqc.build_basis(mol, "def2-SVP")
opt = gemqc.ScfOptions()
opt.convergence = 1e-8
wf = gemqc.rhf(mol, basis, opt, gemqc.sad_guess(mol, basis))
print(wf.energy, wf.eps)                   # numpy arrays

results = gemqc.run("water.yaml", quiet=True)   # run an input: [(task, results)]
```

See [the Python API](api.md) for more.

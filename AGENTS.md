## Main target

- read TARGET.md

## Coding policy

- keep a clear human-readable code structure
- provide compact comments in the code
- double precision numerics is standard
- developer guide (architecture, conventions, recipes for tasks and C++ functions,
  testing): docs/developer/index.md
- task options are defined only in python/gemqc/options.py (check, defaults, docs)
- docstrings: Python Google style; C++ brief `///` comments on the public API in headers
- keep the documentation in docs/ up to date with the code

## Commit policy

- always ask user to review changes before a commit
- use a compact commit message
- never push unless explicitly prompted

## Test runs

- For test runs the user will provide examples in subfolders called sandbox*


## Off-bounds

- Keep strictly to this folder and its subfolders, no reading, writing, globbing outside this folder
- If global searches are required, ask the user and give a reason


## Scope of this project

- Set up a quantum chemistry code that will be used for exploring new methods and algorithms
- Main idea: C++ backend for efficient computations, python frontend for driving computations and pilot coding

## Input

- The input to the program is provided in YAML format
- Molecular structures can be read in from alternative files (start with XYZ, may consider other formats later)
- Basis sets: JSON format as provided by basissetexchange.org (def2-SVP is already in lib/basis/)

## Output

- Main data of current project is stored in JSON file(s)
- Nicely formatted output on screen (can be switched off)
- Huge intermediates or restart data on HDF5
- Avoid having too many files

## Units

- All computations are done in atomic units 
- Provide other unit systems for input output, but all internal data is in atomic units


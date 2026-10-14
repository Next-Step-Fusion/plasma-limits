Hi! This repository is an implementation of the tokamak pedestal density model outlined in https://doi.org/10.1088/1741-4326/ad4b3e paired with various EPED Neural Networks (EPEDNN). 

This code was created by John Anthony Labbate (john.a.labbate@columbia.edu) and Andrew Oak Nelson. Enjoy!


Requirements:
- Python 3.10+

Dependencies that need to be installed:
- numpy
- scipy
- matplotlib
- jupyter / ipykernel (to run the example notebooks)
- Firedrake 2026.4.1
- OpenFusionToolkit
- omfit-classes
- Uncertainties
- juliapkg
- juliacall
- tensorflow>=2.18.1 (only needed for the `epednn_mit` MIT-trained EPEDNN models, e.g. `epednn_model='EPED_SPARC'`)
- radas

Vendored dependencies (git submodules under `dependencies/`, no separate install step needed):
- `dependencies/EPEDNN.jl` - Julia EPEDNN package, automatically registered into the Julia environment at runtime via `juliapkg.add(..., dev=True)`
- `dependencies/epednn_mit` - Python EPEDNN package, imported directly via `sys.path` insertion in `solver.py` (not `pip install`ed, so its `requires-python>=3.12` constraint does not apply)
EPEDNNs should automatically work with no additional installation setup necessary once the above dependencies are in place.

Prebuilt C/Fortran libraries (already vendored under `dependencies/`, used only as build-time dependencies when compiling OpenFUSIONToolkit from source):
- OpenBLAS 0.3.30
- UMFPACK 6.3.5
- arpack-ng 3.9.1
- fox 4.1.2
- hdf5 1.14.6
- metis 5.1.0

To pull in the git submodules after cloning:
```
git submodule update --init --recursive src/pedestal_prediction/dependencies
```
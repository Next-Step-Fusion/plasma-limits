# plasma-limits

Python library for computing tokamak plasma density limits for advanced scenario development.

## Installation

Requires Python ≥ 3.11. The package is `plasma_limits`, with the subpackages
`plasma_limits.density_limit` and `plasma_limits.pedestal_prediction`.

### Development (this repository, with [uv](https://docs.astral.sh/uv/))

```bash
git submodule update --init --recursive dependencies/ESCAPE
uv sync
```

`uv sync` installs plasma-limits and ESCAPE (from `dependencies/ESCAPE`) as editable packages.

### In another Python environment

```bash
pip install /path/to/plasma-limits            # or: pip install -e /path/to/plasma-limits
```

This is enough for `plasma_limits.density_limit`. The pedestal prediction also needs ESCAPE.
Install it first, as an editable install from an ESCAPE checkout (ESCAPE finds its Julia
EPEDNN model relative to its source files):

```bash
pip install -e "/path/to/plasma-limits/dependencies/ESCAPE[eped1]"
```

ESCAPE is not a dependency of the package: the PyPI package `escape` is an unrelated project.

## Usage

### Density limit

The density limit is based on dimensionless collisionality and pressure [1, 2].

See [examples/density_limit_example.py](examples/density_limit_example.py) for a full example
that creates `PlasmaState` inside time loop and calculates Greenwald fraction and DL26 instability
metric over time.

DL26 limit is based on Andrew D. Maris's PhD thesis "Prediction and control of the tokamak density limit", 2026.

### Pedestal prediction

`plasma_limits.pedestal_prediction.ESCAPE_interface` is an in-memory interface to ESCAPE
(EPED Saarelma-Connor Adapted Pedestal Evolution). ESCAPE couples the Saarelma-Connor pedestal
density model [3] with EPED neural networks (EPEDNN). It physically targets ELMy H-mode plasmas.
ESCAPE was created by John Anthony Labbate and Andrew Oak Nelson.

The interface reads and writes no files: you give all inputs as arrays, and you get the results
as objects. The only files that are read are the EPED model weights and the atomic-rate tables
that come with ESCAPE. Install ESCAPE as described in [Installation](#installation).

| Name | Purpose |
|---|---|
| `PedestalPlasmaState` | All physical inputs of one run: equilibrium, kinetic profiles, species, heating power |
| `SCConfig`, `FreeParams` | Saarelma-Connor solver settings and free parameters |
| `EPEDConfig`, `load_eped_model` | Select and load the EPED model (`EPED1` or `EPED_SPARC`) |
| `run_escape` | Run the ESCAPE loop for one plasma; returns an `ESCAPEResult` |
| `ESCAPEResult`, `IterationRecord` | Convergence flag and one record per iteration (profiles, pedestal height and width, neutrals) |
| `ESCAPERunError` | A failed run; it keeps the records made before the failure |

```python
from plasma_limits.pedestal_prediction.ESCAPE_interface import (
    PedestalPlasmaState, SCConfig, load_eped_model, run_escape)

state = PedestalPlasmaState(R=..., Z=..., psi=..., ...)  # see the docstring for fields and units
eped = load_eped_model()                                 # load once
result = run_escape(state, eped, SCConfig(), max_iter=5)
print(result.converged, result.final.pedestal_height, result.final.pedestal_width)
```

Notes:
- The default model is 1D with the scipy implementation. The 3D model and
  `implementation='firedrake'` need Firedrake; see the ESCAPE documentation for its installation.
- Load the EPED model once with `load_eped_model`, and give the same `EPEDModel` to each run of a
  scan or a time loop. Each call of `load_eped_model` loads the model again.
- EPED1 runs in Julia, which is not safe across `fork`. For parallel runs, start the processes
  with `spawn` (`multiprocessing.get_context("spawn")`).
- With the 3D model, `implementation='firedrake'` and `SCConfig(verbose=True)`, ESCAPE saves one
  diagnostic PNG per iteration to the working directory. With the default `verbose=False`, nothing
  is written.
- ESCAPE prints its progress to stdout on each iteration.

Examples (each saves a plot of the profiles to `outputs/` in the working directory):
- [examples/ESCAPE_ARC_example.py](examples/ESCAPE_ARC_example.py): ARC, from the g-file and the
  profile files of ESCAPE's ARC example. Reads the g-file with `freeqdsk`.
- [examples/ESCAPE_CENTAUR_example.py](examples/ESCAPE_CENTAUR_example.py): CENTAUR, from an
  NSFsim history file (`examples/CENTAUR/centaur.h5`). CENTAUR has negative triangularity, for
  which ESCAPE gives unphysical results. This example only shows the data flow.

### References

[1] Maris, A. D. et al. Correlation of the L-mode density limit with edge collisionality. Nucl. Fusion (2024) doi:10.1088/1741-4326/ad90f0.

[2] Maris, A. D. et al. Real-time avoidance of the L-mode and H-mode density limit via machine-learned stability metrics. Nucl. Fusion 66, 056002 (2026).

[3] Saarelma-Connor pedestal density model: https://doi.org/10.1088/1741-4326/ad4b3e

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
See [examples/density_limit_example.py](examples/density_limit_example.py) for a full example
that creates `PlasmaState` inside time loop and calculates Greenwald fraction and DL26 instability
metric over time.

DL26 limit is based on Andrew D. Maris's PhD thesis "Prediction and control of the tokamak density limit", 2026.

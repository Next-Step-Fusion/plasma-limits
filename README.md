# plasma-limits

Python library for computing tokamak plasma density limits for advanced scenario development.

## Installation

Requires Python ≥ 3.11. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv pip install -e .
```

## Usage

### Density limit
See [examples/density_limit_example.py](examples/density_limit_example.py) for a full example
that creates `PlasmaState` inside time loop and calculates Greenwald fraction and DL26 instability
metric over time.

DL26 limit is based on Andrew D. Maris's PhD thesis "Prediction and control of the tokamak density limit", 2026.

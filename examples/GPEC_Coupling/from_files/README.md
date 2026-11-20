# DIII-D beta scans from prepared equilibria

Run the same GPEC analysis on prepared TokaMaker and NSFsim equilibrium sets. Neither equilibrium
producer, OpenFUSIONToolkit/OFT, nor a Python–Julia bridge is required during this analysis.
The example calculates a conditional **n=1 no-wall ideal-stability threshold**. It does not yet
validate the 2/1 tearing workflow, a resistive-wall mode, or an all-mode device beta limit.

## Install

Requirements: Git, Bash, `shasum`, Julia 1.11 or 1.12, and Python 3.11 or newer.
The local numerical reference uses Julia 1.12.7 on macOS; the CI workflow tests Julia 1.11 and
1.12 on Linux. Check the PR's CI results before treating those platforms as validated.

While this change is a stacked PR, clone its **source branch**, as shown in GitHub's checkout
instructions. The parent `mhd_stability` branch will not contain this example until it is merged.
For a branch containing this example:

```bash
git clone --recurse-submodules --branch <branch-containing-this-example> <repository-url>
cd plasma-limits
```

For an existing clone, first check out the branch containing this example, then initialize GPEC:

```bash
git submodule update --init --recursive
cd examples/GPEC_Coupling/from_files
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-plotting.txt
bash setup.sh
```

The submodule `external/GPEC` is pinned to `f3dac441acc95b88a32461c7e3ed9a1a3a131d09`
(an upstream develop commit reporting `v2.0.1-33-gf3dac441`). `setup.sh` binds this example's
Julia environment to that checkout, installs dependencies and precompiles packages. The first
setup requires network access and can take several minutes. This is the Julia implementation:
**no separate build of legacy Fortran GPEC is required**. Some Julia dependencies download native
library artifacts. The runner verifies GPEC's loaded path, exact commit and tracked-file cleanliness.

Do not use `git submodule update --remote` for a reproducible run: it can move away from the tested
pin. A deliberate upgrade must update the gitlink, `GPEC_COMMIT`, environment and numerical references.

## Run

With the Python environment activated, from this directory:

```bash
bash run_example.sh outputs/example
```

Use a new output directory each time. This runs all 22 prepared equilibria and automatically
recomputes the two threshold brackets with three alternative settings (34 calculations total).
It then applies the comparison gate and saves a plot. A previous warm-cache macOS run took about
8 minutes; first-use compilation and other platforms can take longer. Allow several GB of output
space for native `gpec.h5` files. `outputs/` is ignored by Git.

For the calculation and summary without plotting:

```bash
julia --project=environment -t 2 scripts/run_scans.jl outputs/example both
python3 scripts/summarize.py outputs/example --strict
```

Replace `both` with `tokamaker` or `nsfsim` to run one dataset. An optional third argument selects
a single preset (`baseline`, `fine`, `core`, `riccati`) for all points, without automatic bracket
checks; such exploratory runs do not pass `--strict` until all required checks are present.

## Inputs and constraints

| Dataset | Equilibria | Source and scan constraints |
|---|---:|---|
| TokaMaker | 8 | DIII-D shot 147131 at 2300 ms. Target the original total current and contour, retain the baseline `jphi-linterp` current-profile prescription, scale thermal pressure/density at fixed temperature. These are numerical targets, not exact-preservation guarantees. |
| NSFsim | 14 | Existing generic DIII-D `d3d_basic` scenario. Fix total current and boundary mode, increase prescribed deuterium/tritium densities at fixed temperature profiles, allow the current-profile shape to evolve. |

These are **different equilibrium families**, not a matched-shot comparison. Their beta limits
need not agree. Adding an NSFsim reconstruction of Adriana's shot is subsequent work.
The manifests under `inputs/` record each multiplier, file path, producer beta and equilibrium
SHA-256. The runner checks the hashes before use. Original generation metadata is retained
alongside each dataset; paths in that provenance describe the original generation environment.
`kinetics.csv` files are retained for future tearing analysis and are unused by this ideal scan.

Each pressure point is a separately solved equilibrium. Scaling pressure in a single gEQDSK file
without re-solving force balance is not a substitute for generating a new equilibrium.
The two generators used different radial profile coordinates; see their manifests/provenance.

## Outputs and numerical checks

- `results.toml`: all case results, GPEC beta_N, minimum free-boundary energy, internal crossing
  count, response-matrix asymmetry, actual domain cutoffs, input/config hashes and warnings.
- `summary.json`: the first screened positive-to-negative delta-W bracket, its linear zero,
  alternative estimates, scan completeness, failures and solver warnings.
- `beta_limits.png`: full scans and close-ups of the threshold brackets.
- `<dataset>_<point>_<preset>/`: `gpec.toml`, input copy, `solver.log`, `result.toml` and `gpec.h5`.
- `runtime.toml`: GPEC source/commit, Julia version, threads and selected datasets.

Both datasets use **GPEC's normalized beta beta_N**, not ordinary beta in percent. Delta W uses
GPEC's eigenmode-energy normalization, not raw joules or a growth rate. A negative minimum energy
indicates an ideal instability in the modeled domain. A positive free-boundary energy is not
accepted as proof of stability when the internal-crossing check fails.

The threshold extractor does not skip failed points before a bracket. The strict summary requires
complete, unique baseline coverage, no solver failures, and passing bracket checks. Internally
unstable points above the first threshold remain visible; they do not replace the earlier threshold.
For example, NSFsim's last two baseline points have positive free-boundary energy but internal
instability, so they must not be described as re-stabilization.

The presets are:

| Preset | Changes from baseline |
|---|---|
| Baseline | Forward integration; automatic radial mesh with `psi_accuracy=0.001`, `mtheta=514`, sidebands 8, `psilow=1e-4`, `qlow=1.02`, ODE tolerance `1e-10`, singular crossing distance `1e-6`. |
| Fine | `psi_accuracy=0.00025`, `mtheta=1028`, sidebands 12. |
| Core | Fine settings with `psilow=1e-5`. |
| Riccati | Fine settings with Riccati integration and singular crossing distance `1e-4`. This changes both formulation and crossing distance. |

The demonstration gate requires finite energies, final plasma/vacuum matrix asymmetry below
`1e-3`, imaginary energy divided by the sum of plasma/vacuum energy magnitudes below `1e-3`, and
zero internal crossings at bracket endpoints. The four interpolated threshold estimates must
span at most 1% of the baseline estimate. This does not bound interpolation error or establish
convergence of every parameter separately. Solver warnings are reported, not automatically erased
or treated as resolved by this gate.

## Reference and limitations

The included [reference summary](reference/summary.json) and [plot](reference/beta_limits.png)
come from the pinned macOS run, before packaging into this repository:

| Dataset | Baseline solved beta_N bracket | Linear zero | Spread across settings |
|---|---|---:|---:|
| TokaMaker | 2.312921–2.492243 | 2.325320 | 0.438% |
| NSFsim | 2.240710–2.296981 | 2.255849 | 0.189% |

Report approximately **2.33** and **2.26**, with the brackets and scope. Interpolation is not an
additional equilibrium solve. The older notebook result used different settings/GPEC and a
producer beta convention; this is not a bit-for-bit replay of that older result.

The model excludes the axis-side region below its actual start and truncates the outer domain
using `rational_offset_fraction=0.2`. Those excluded regions are not certified stable. TokaMaker's
last two baseline points activate a larger low-q core cutoff, after its first threshold. Other
mode numbers, conducting-wall models, and convergence of the input gEQDSK grid are not validated
here (TokaMaker exports are 257x257, NSFsim 65x65).

Some TokaMaker internal W-inverse non-Hermiticity warnings persist, particularly with Riccati,
even though the final-response metrics and threshold comparison pass. Its g-files also trigger
an FPOL precision warning relevant to Delta-prime: investigate/regenerate that export before
claiming converged tearing indices. The NSFsim bracket checks in the reference had no numerical
warnings, only the expected physical instability warning at negative-energy endpoints.

## Tests

```bash
python3 -m unittest discover -s scripts -p 'test_*.py' -v
bash run_example.sh outputs/test
python3 scripts/check_reference.py outputs/test
```

The reference check requires the same bracketing point IDs and a baseline threshold within 1%
of the recorded value, in addition to the numerical-setting comparison gate. CI runs these commands
from a recursive checkout. This example does not change the GPEC physics kernel; the upstream
GPEC regression harness remains the appropriate additional check for a kernel upgrade.

"""In-memory interface to the ESCAPE pedestal model (git submodule ``ESCAPE/``).

ESCAPE couples the Saarelma-Connor pedestal density model with EPEDNN. Its
own entry point, ``ESCAPE_solve``, reads and writes files. This module wraps
the same model without any file I/O: the caller builds all inputs in memory,
and results are returned in memory. The only files read from disk are the
EPED model weights and the atomic-rate tables bundled with ESCAPE.

Public API (added step by step):
    PedestalPlasmaState  -- every physical input (equilibrium, profiles, species)
    FreeParams, SCConfig -- Saarelma-Connor free parameters and solver settings
    EPEDConfig           -- which EPED model to use
    load_eped_model      -- load an EPED model once per process
    run_escape           -- run the ESCAPE loop, returning an ESCAPEResult
    IterationRecord, ESCAPEResult, ESCAPERunError

ESCAPE is imported lazily, on first use, so importing this module needs only
the standard library and numpy.

The module works from a source checkout or an editable install only:
``pedestal_prediction`` is a namespace package, and ESCAPE is located through
the submodule directory ``dependencies/ESCAPE`` at the repository root.
"""
from pathlib import Path

__all__ = []

ESCAPE_ROOT = Path(__file__).resolve().parents[2] / "dependencies" / "ESCAPE"

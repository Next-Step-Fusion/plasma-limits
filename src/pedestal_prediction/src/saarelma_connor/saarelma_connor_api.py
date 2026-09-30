"""Public entry point: the Saarelma-Connor model class.

The implementation is split across three modules by topic --

    src.saarelma_connor.saarelma_connor_base
        SaarelmaConnorBase -- equilibrium loading, geometry, atomic rates,
        the EPED-NN interface, the shared __init__, and the unified
        `solve()` dispatcher
    src.saarelma_connor.solver_3d
        ThreeDSolverMixin  -- the coupled three-equation ("3D")
        non-dimensional solver
    src.saarelma_connor.solver_1d
        OneDSolverMixin    -- the original single-equation ("1D")
        Saarelma-Connor model

-- but they are one class, assembled here.

Because there is one class and one constructor, the physics model is chosen
per solve rather than per instantiation::

    from src.saarelma_connor.saarelma_connor_api import saarelma_connor

    model = saarelma_connor(P_tot_e=5e6, mhd_fp=..., kprof_fp=..., ...)
    res_3d = model.solve(model='3D', x_res=40)
    res_1d = model.solve(model='1D', x_res=200)

Both calls return the same dictionary schema (see
:meth:`src.saarelma_connor.saarelma_connor_base.SaarelmaConnorBase._build_result_dict`), so downstream
code parses one structure regardless of which solver ran.
"""

from src.saarelma_connor.saarelma_connor_base import SaarelmaConnorBase
from src.saarelma_connor.solver_3d import ThreeDSolverMixin
from src.saarelma_connor.solver_1d import OneDSolverMixin


class saarelma_connor(OneDSolverMixin, ThreeDSolverMixin, SaarelmaConnorBase):
    """Equilibrium setup plus both the 1D and coupled 3-equation solvers.

    Built once through the shared ``__init__`` of
    :class:`~src.saarelma_connor.saarelma_connor_base.SaarelmaConnorBase`; the physics model is selected
    per call via :meth:`~src.saarelma_connor.saarelma_connor_base.SaarelmaConnorBase.solve`.

    The mixin order is arbitrary -- the three classes share no attribute
    names, so the MRO never has to choose between them.
    """


__all__ = ['saarelma_connor']

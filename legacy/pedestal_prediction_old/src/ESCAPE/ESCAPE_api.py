"""Public entry point: the ESCAPE class (Saarelma-Connor + EPEDNN).
"""

from src.saarelma_connor.saarelma_connor_api import saarelma_connor
from src.epednn.epednn_call import epednn_class
from src.ESCAPE.ESCAPE_state import ESCAPE_state


class ESCAPE(ESCAPE_state, saarelma_connor, epednn_class):
    """
    Built once through the shared ``__init__`` of
    :class:`~src.ESCAPE_state`;

    The mixin order is arbitrary -- the three classes share no attribute
    names, so the MRO never has to choose between them.
    """

__all__ = ['ESCAPE']
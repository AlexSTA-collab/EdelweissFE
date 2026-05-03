# Lightweight Python adapter that wraps the compiled/extension
# `MarmotInterfaceElementWrapper` and provides small missing helpers
# (acceptLastState, resetToLastValidState) expected by the FEModel.

from typing import Any

try:
    # prefer compiled/extension implementation if available
    from edelweissfe.elements.marmotelement.interface_element import (
        MarmotInterfaceElementWrapper as _CWrapper,
    )
except Exception:
    # fall back to importing the module (source) if installation layout differs
    from edelweissfe.elements.marmotelement import interface_element as _cmod

    _CWrapper = getattr(_cmod, "MarmotInterfaceElementWrapper")


class MarmotInterfaceElementWrapper:
    """Adapter that instantiates the underlying compiled Cython wrapper and
    forwards most attribute access to it. Adds small Python implementations
    for lifecycle helpers that some driver code expects (acceptLastState,
    resetToLastValidState).

    The goal is minimal, non-invasive bridging: the underlying C++ element is
    constructed by the compiled wrapper, all heavy work is delegated there.
    """

    def __init__(self, elementType: str, elNumber: int):
        self._c = _CWrapper(elementType, elNumber)
        # ensure compiled wrapper has a persistent missing-state buffer dict
        try:
            if not hasattr(self._c, "_missingStateBuffers"):
                self._c._missingStateBuffers = {}
        except Exception:
            # best-effort: if the compiled wrapper does not allow attribute
            # assignment here, ignore and rely on later behavior.
            pass

    def __getattr__(self, name: str) -> Any:
        # forward lookups to the compiled wrapper
        return getattr(self._c, name)

    # lifecycle helpers expected by FEModel drivers
    def acceptLastState(self):
        """Accept the computed state by copying temp -> main state array."""
        try:
            # compiled wrapper exposes memoryviews / numpy arrays as attributes
            self._c._stateVars[:] = self._c._stateVarsTemp
        except Exception:
            # best-effort fallback: if attributes not present, ignore
            pass

    def resetToLastValidState(self):
        """Restore last accepted state by copying main -> temp state array."""
        try:
            self._c._stateVarsTemp[:] = self._c._stateVars
        except Exception:
            pass

    # allow direct attribute access for convenience
    @property
    def marmotElement(self):
        return getattr(self._c, "marmotElement", None)

    def __repr__(self):
        return f"<MarmotInterfaceElementWrapper adapter for {self._c!r}>"

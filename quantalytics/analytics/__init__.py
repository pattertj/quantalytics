"""Combined analytics helpers replacing the old ``metrics`` / ``stats`` packages."""

from . import core as _core
from . import performance as _performance

__all__ = _core.__all__ + _performance.__all__

for _module in (_core, _performance):
    for _export in _module.__all__:
        globals()[_export] = getattr(_module, _export)

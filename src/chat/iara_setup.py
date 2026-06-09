"""Namespace shim: registers the local iara package under the construct_cost_ai path.

The iara framework was authored inside a monorepo where the package root is
`construct_cost_ai.infra.ai.frameworks.iara.src`.  Here we have only the iara
subtree, so we inject the namespace chain into sys.modules and point the leaf
package's __path__ at the real source directory.  Python's import machinery
then resolves any `from construct_cost_ai.infra...` import correctly.
"""

import sys
import types
from pathlib import Path

# Absolute path to  src/utils/frameworks/iara/src/
_IARA_SRC: Path = (
    Path(__file__).parent.parent / "utils" / "frameworks" / "iara" / "src"
).resolve()

_NAMESPACE_CHAIN = [
    "construct_cost_ai",
    "construct_cost_ai.infra",
    "construct_cost_ai.infra.ai",
    "construct_cost_ai.infra.ai.frameworks",
    "construct_cost_ai.infra.ai.frameworks.iara",
]

_LEAF = "construct_cost_ai.infra.ai.frameworks.iara.src"


def setup() -> None:
    """Inject the construct_cost_ai namespace into sys.modules (idempotent)."""
    if _LEAF in sys.modules:
        return

    for name in _NAMESPACE_CHAIN:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__package__ = name
            mod.__path__ = []  # namespace package — no physical directory
            mod.__spec__ = None
            sys.modules[name] = mod

    # The leaf package must have __path__ pointing at the real src/ directory
    # so that Python finds sub-packages (agents, config, models, utils) there.
    leaf = types.ModuleType(_LEAF)
    leaf.__path__ = [str(_IARA_SRC)]
    leaf.__package__ = _LEAF
    leaf.__file__ = str(_IARA_SRC / "__init__.py")
    leaf.__spec__ = None
    sys.modules[_LEAF] = leaf

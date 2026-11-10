"""How the coverage probes write CrossGL.

The CrossGL IR has no compatibility guarantee. Every CrossGL probe is generated from this package and the expression
syntax :class:`~tools.coverage.dialect.Dialect` shares with GLSL, so a pull request that changes how CrossGL writes a
compute kernel, a storage buffer, a type or a built-in function updates this package too, and no probe text is pinned
anywhere else. :mod:`.dialect` writes the programs and :mod:`.templates` holds the template of each construct.
"""

from .dialect import DIALECT, ES_DIALECT
from .templates import CONSTRUCTS

__all__ = ["CONSTRUCTS", "DIALECT", "ES_DIALECT"]

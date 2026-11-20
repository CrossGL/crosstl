"""Lookups in rename tables, which hold only the names a translation step renamed.

An importer or generator renames a declaration whose spelling the output cannot use (a keyword, a reserved prefix,
a clash with another declaration) and records the new spelling in a table. The table holds only the names it
renamed, so a name it does not hold was not renamed and keeps its spelling. :func:`renamed` reads such a table. It
is not for a table that translates every name of one language into another; a name missing from that kind of table
has no translation and must be refused.
"""

from __future__ import annotations

__all__ = ["renamed"]


def renamed(renames, name):
    """``name`` as the output spells it: the spelling ``renames`` gives it, or ``name`` when it was not renamed."""
    if name in renames:
        return renames[name]
    return name

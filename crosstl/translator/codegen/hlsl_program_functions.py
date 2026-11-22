"""One whole-program function walk for the DirectX generator's analysis phase.

``collect_functions(root)`` walks every object reachable from ``root`` and keeps the nodes that carry both a
``body`` and ``parameters``: functions, stage-local functions and lambdas. A render attempt's analysis phase runs
from the point where the generator takes its function list until it starts emitting text. In that phase the
generator calls ``collect_functions`` on the same program about 30 times: once directly, then once inside each
sampler, texture, resource-size and stage-dependency analysis. A large Metal library has about 150,000 reachable
objects, so each call is a full walk of that size.

The analysis phase reads the program and does not rewrite it. The generator already hands the list it took at the
start of the phase to its overload, parameter and resource-pointer analyses, and ``hlsl_function_body_scans``
relies on the same rule for function bodies. :class:`HlslProgramFunctions` keeps that list for the phase, and
``collect_functions`` answers from it when it is asked about the same root object.

The kept list is used only while the containers that hold top-level and stage-local functions are unchanged. For
the root and for each stage, every list or tuple attribute must be the same object with the same length as when the
list was taken, and the stages must be the same objects. A pass that appends, removes or replaces a function or a
stage therefore makes the generator walk again. Every answer is a new list, because callers extend the list they
receive. The generator drops the kept list when emission starts and when the attempt ends, including by an error.
"""

from __future__ import annotations

from typing import Any

__all__ = ["HlslProgramFunctions", "program_container_shape"]

_SEQUENCES = (list, tuple)


def program_container_shape(root: Any) -> tuple:
    """Identity and length of every list or tuple attribute of ``root`` and of each of its stages."""
    holders = [root]
    stages = getattr(root, "stages", None)
    if isinstance(stages, dict):
        # StageMap.values() yields every stage declared for a key, not only the first.
        holders.extend(stages.values())
    shape = []
    for holder in holders:
        attributes = getattr(holder, "__dict__", None)
        if attributes is None:
            shape.append((id(holder), None, None, None))
            continue
        for name, value in attributes.items():
            if isinstance(value, _SEQUENCES):
                shape.append((id(holder), name, id(value), len(value)))
    return tuple(shape)


class HlslProgramFunctions:
    """The function list of one program, kept for one analysis phase."""

    __slots__ = ("root", "functions", "shape")

    def __init__(self, root: Any, functions: list) -> None:
        self.root = root
        self.functions = tuple(functions)
        self.shape = program_container_shape(root)

    def lookup(self, root: Any) -> list | None:
        """A new copy of the kept list for ``root``, or ``None`` when the generator must walk."""
        if root is not self.root or program_container_shape(root) != self.shape:
            return None
        return list(self.functions)

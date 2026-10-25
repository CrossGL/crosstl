"""Zero values of GLSL arrays.

The GLSL generator zero-initializes the globals that C-family sources leave zeroed. A GLSL array constructor takes
one argument per element (GLSL 4.60 section 5.4.4, "Array Constructors"), so the zero value of ``T[N]`` repeats the
zero value of ``T`` ``N`` times. An array whose size the translator cannot evaluate has no such spelling and is refused
with :class:`OpenGLZeroValueError`.
"""

from __future__ import annotations

import re

__all__ = ["OpenGLZeroValueError", "zero_array_value"]

_ARRAY_SPELLING = re.compile(r"(.+)\[(\d+)\]")


class OpenGLZeroValueError(ValueError):
    """Raised when an array's zero value has no GLSL constructor."""

    project_diagnostic_code = "project.translate.opengl-zero-value-unsupported"
    missing_capabilities = ("opengl.zero-value-array-constructor",)

    def __init__(self, type_name):
        self.type_name = type_name
        super().__init__(
            f"OpenGL cannot zero-initialize the array type {type_name}: its size is not a constant the "
            "translator can evaluate, and a GLSL array constructor takes one argument per element"
        )


def zero_array_value(codegen, array_type, mapped_type):
    """``T[N](z, ..., z)`` for ``array_type`` (an ``ArrayType`` node or ``None``) spelled ``mapped_type``."""
    if array_type is not None:
        count = codegen.positive_int_value(getattr(array_type, "size", None))
        if count is None:
            raise OpenGLZeroValueError(mapped_type)
        element = codegen.zero_value_expression(array_type.element_type)
        return f"{mapped_type}({', '.join([element] * count)})"
    spelled = _ARRAY_SPELLING.fullmatch(mapped_type)
    if spelled is None or int(spelled.group(2)) <= 0:
        return None
    element = codegen.zero_value_expression(spelled.group(1))
    return f"{mapped_type}({', '.join([element] * int(spelled.group(2)))})"

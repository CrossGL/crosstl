"""HLSL declarations of specialization constants that have no default.

A CrossGL specialization constant without a default (an MSL ``[[function_constant(N)]]`` declared without a value,
or ``const T name @constant_id(N);``) must be given its value when the pipeline is created. Direct3D has no
specialization step: HLSL source is compiled once per variant, so the host supplies the value as a preprocessor
macro (``dxc -D``, the ``pDefines`` of ``D3DCompile``), the way an OpenGL driver that compiles GLSL source receives
it. The macro has the same name as on the GLSL source path (:func:`glsl_specialization_macro_name`), so one host
definition serves both. Without the macro the source stops at an ``#error`` that names it: a missing value is a
compile error, never a silently chosen default.

.. code-block:: hlsl

    #if !defined(CROSSGL_SPECIALIZE_forward)
    #error "specialization constant forward (id 1) has no default: define CROSSGL_SPECIALIZE_forward"
    #endif
    static const bool forward = (bool)(CROSSGL_SPECIALIZE_forward);

The declaration is a ``static const`` with a constant initializer, so it stays a constant expression: it can size an
array or a ``groupshared`` allocation, as the specialized value does in SPIR-V.

Host reflection (``crosstl.project.host_reflection``) recognises the guard through
:data:`HLSL_SPECIALIZATION_MACRO_RE` and reports the constant as required.
"""

from __future__ import annotations

import re

from .glsl_specialization_macros import glsl_specialization_macro_name

__all__ = [
    "HLSL_SPECIALIZATION_MACRO_RE",
    "hlsl_specialization_macro_name",
    "render_hlsl_required_specialization_constant",
]

HLSL_SPECIALIZATION_MACRO_RE = re.compile(
    r"^[ \t]*#[ \t]*if[ \t]+!defined\((?P<macro>CROSSGL_SPECIALIZE_\w+)\)[ \t]*\n"
    r'[ \t]*#[ \t]*error[ \t]+"specialization constant \w+ \(id (?P<id>\d+)\) has no default[^\n]*\n'
    r"[ \t]*#[ \t]*endif[ \t]*\n"
    r"[ \t]*static[ \t]+const[ \t]+(?P<type>[A-Za-z_]\w*)[ \t]+(?P<name>[A-Za-z_]\w*)[ \t]*=[ \t]*"
    r"\((?P=type)\)\((?P=macro)\);",
    re.MULTILINE,
)


def hlsl_specialization_macro_name(name: str) -> str:
    """The macro through which the host supplies the specialization constant ``name`` to HLSL."""
    return glsl_specialization_macro_name(name)


def render_hlsl_required_specialization_constant(
    name: str, constant_id: int, mapped_type: str
) -> str:
    """Declare the default-less specialization constant ``name`` from its host macro, guarded by ``#error``.

    ``mapped_type`` is the HLSL scalar type, one of the types the emitter accepts for specialization constants.
    """
    macro = hlsl_specialization_macro_name(name)
    return (
        f"/* CrossGL DirectX specialization constant id {constant_id} has no default; "
        f"the host defines {macro}. */\n"
        f"#if !defined({macro})\n"
        f'#error "specialization constant {name} (id {constant_id}) has no default: '
        f'define {macro}"\n'
        "#endif\n"
        f"static const {mapped_type} {name} = ({mapped_type})({macro});\n"
    )

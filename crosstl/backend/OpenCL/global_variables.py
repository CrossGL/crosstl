"""OpenCL C variables in the global address space that kernels can write.

A program-scope variable in the global address space, such as ``global int counter = 3;``, is one object that every
work-item of every kernel in the program reads and writes, and it keeps its value from one kernel launch to the
next. OpenCL C 2.0 and later also place an unqualified program-scope variable, and a ``static`` variable declared
inside a function, in the global address space (OpenCL C 3.0, "Address Space Qualifiers"; OpenCL C 1.2 accepts
only ``__constant`` at program scope).

A program-scope variable declared ``global`` or ``__global`` imports as a CrossGL ``device`` variable,
``var<device> counter: i32 = 3;``: one initialized object in device memory. CUDA and HIP output declare it
``__device__``, and every other target refuses it (:mod:`crosstl.translator.semantics.device_variables`).

The importer still refuses the other writable ones: a program-scope variable without an address-space qualifier
(``int counter = 3;``, ``static int counter = 3;``), an ``extern`` one and a function-scope ``static`` one. A
module-scope ``var`` is one copy per invocation, and a storage buffer is memory the host binds,
while the OpenCL host API cannot bind a buffer to a program-scope variable, so writing either would change what the
kernels compute. Read-only ones, ``const`` or in the constant address space, keep their import: a per-invocation
copy of a value nobody writes reads the same.
"""

from __future__ import annotations

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "OpenCLGlobalVariableUnsupportedError",
    "is_read_only_global_variable",
    "refuse_writable_global_variable",
]

_CONSTANT_SPACE = frozenset({"__constant", "__constant__", "constant"})
# Program-scope declarations in these spaces are invalid OpenCL C; another part of the importer decides them.
_OTHER_SPACES = frozenset(
    {
        "__local",
        "__local__",
        "local",
        "__shared__",
        "__private",
        "__private__",
        "private",
    }
)
_READ_ONLY = frozenset({"const", "constexpr"})


class OpenCLGlobalVariableUnsupportedError(TranslationDiagnostic):
    """A writable OpenCL C variable in the global address space, one object shared by every work-item."""

    code = CODE_PREFIX + "opencl-global-variable-unsupported"


def _is_read_only(qualifiers: set[str], vtype: str) -> bool:
    if "*" in vtype:
        # `const float *p` points at constant data, but p itself can be reassigned; `float *const p` cannot.
        return bool(_READ_ONLY & set(vtype.rsplit("*", 1)[1].split()))
    return bool(_READ_ONLY & (qualifiers | set(vtype.replace("[", " ").split())))


def is_read_only_global_variable(node) -> bool:
    """Whether a program-scope declaration cannot be written after initialization."""
    qualifiers = set(getattr(node, "qualifiers", None) or [])
    vtype = str(getattr(node, "vtype", "") or "")
    return bool(qualifiers & _CONSTANT_SPACE) or _is_read_only(qualifiers, vtype)


def refuse_writable_global_variable(node, *, program_scope: bool) -> None:
    """Raise :class:`OpenCLGlobalVariableUnsupportedError` when ``node`` declares a writable global-space variable.

    ``program_scope`` says whether ``node`` is declared outside every function. Inside a function only a ``static``
    variable is in the global address space; a ``__global float *p`` local is a private pointer into it.
    """
    qualifiers = set(getattr(node, "qualifiers", None) or [])
    if qualifiers & (_CONSTANT_SPACE | _OTHER_SPACES):
        return
    if not program_scope and "static" not in qualifiers:
        return
    if is_read_only_global_variable(node):
        return
    where = "program-scope" if program_scope else "function-scope static"
    raise OpenCLGlobalVariableUnsupportedError(
        f"OpenCL C {where} variable '{node.name}' is in the global address space: it is one object that "
        "every work-item writes and that keeps its value between kernel launches, and CrossGL has no such "
        "variable (a module-scope var is one copy per invocation, and the OpenCL host cannot bind a storage "
        "buffer to it)"
    )

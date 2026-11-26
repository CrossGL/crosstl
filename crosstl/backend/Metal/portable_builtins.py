"""CrossGL for the Metal library functions that CrossGL spells differently or computes from other operations.

Metal Shading Language Specification 4.1:

* ``dfdx`` and ``dfdy`` (section 6.8, fragment functions) are CrossGL's ``dFdx`` and ``dFdy``, and ``rint``
  (section 6.5), which rounds to the nearest integral value with ties to even, is ``roundEven``. These are renamed by
  :func:`renamed_metal_function`.
* ``max3(x, y, z)`` and ``fmax3`` return ``max(x, max(y, z))`` and ``fmax(x, fmax(y, z))``, ``min3`` and ``fmin3``
  the same with ``min`` and ``fmin`` (sections 6.4 and 6.5), which is what they import as.
* ``median3`` and ``fmedian3`` return the middle value of ``x``, ``y`` and ``z``. That reads ``x`` and ``y`` twice,
  ``max(min(x, y), min(max(x, y), z))``, so each call imports as a call of a CrossGL helper of that formula, one per
  argument type, which evaluates every argument once. A call whose argument type is unknown is refused.
* ``simd_is_helper_thread`` and ``quad_is_helper_thread`` (sections 6.9.2 and 6.9.3) are true in a helper thread,
  which is CrossGL's ``gl_HelperInvocation``. Only a fragment function has helper threads, which compute derivatives
  for a quad, so the call is ``false`` in any other stage function and in a helper of a program without a fragment
  function (on an Apple M2 Max a kernel's ``simd_is_helper_thread()`` is false in every thread).
* ``simd_shuffle_rotate_down(data, delta)`` reads lane ``(lane + delta) % width``, which is CrossGL's
  ``WaveRotate(data, delta)``; ``simd_shuffle_rotate_up`` reads lane ``(lane - delta) % width``, which is
  ``WaveRotate(data, 0u - delta)`` because the SIMD-group width is a power of two that divides 2^32. ``builtin_tables``
  maps both, and ``simd_is_first``, to their CrossGL operations.

A call that binds a user function of the same name is left alone.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "METAL_RENAMED_FUNCTIONS",
    "generate_portable_builtin_support_code",
    "render_portable_builtin_call",
    "renamed_metal_function",
]

METAL_RENAMED_FUNCTIONS = {"dfdx": "dFdx", "dfdy": "dFdy", "rint": "roundEven"}
# The trinary functions written as two nested calls: Metal name -> CrossGL operation.
_NESTED = {"max3": "max", "fmax3": "fmax", "min3": "min", "fmin3": "fmin"}
_MEDIANS = frozenset({"median3", "fmedian3"})
_HELPER_THREADS = frozenset({"simd_is_helper_thread", "quad_is_helper_thread"})


def _library_name(converter: Any, name: Any) -> str | None:
    """The unqualified name of a call that names the Metal library function, or None when it names a user one.

    The ``fast`` and ``precise`` namespaces only change the accuracy of inexact functions; every function this module
    imports is exact, so their variants import alike.
    """
    text = str(name).lstrip(":")
    qualified = text.startswith("metal::")
    if qualified:
        text = text[len("metal::") :]
    for namespace in ("fast::", "precise::"):
        if text.startswith(namespace):
            text, qualified = text[len(namespace) :], True
    if "::" in text or (not qualified and text in converter.user_function_names):
        return None
    return text


def renamed_metal_function(converter: Any, name: Any) -> str | None:
    """The CrossGL name of a Metal library function CrossGL spells differently, or None."""
    library_name = _library_name(converter, name)
    return None if library_name is None else METAL_RENAMED_FUNCTIONS.get(library_name)


def _median_helper(converter: Any, expr: Any) -> str:
    types = {converter.expression_mapped_type(argument) for argument in expr.args}
    if len(types) != 1 or None in types:
        return converter.refuse_import(
            f"Metal {expr.name} needs one argument type for its CrossGL helper; the import resolved "
            f"{', '.join(sorted(str(type_name) for type_name in types))}",
            "median3",
        )
    (type_name,) = types
    required = converter.__dict__.setdefault("required_metal_median_types", set())
    required.add(type_name)
    return converter.metal_precise_math_unique_helper_name(
        ("median3", type_name), f"__crossgl_metal_median3_{type_name}"
    )


def _helper_thread(converter: Any) -> str:
    qualifier = getattr(getattr(converter, "current_function", None), "qualifier", None)
    if qualifier == "fragment":
        return "gl_HelperInvocation"
    if qualifier is None and getattr(
        converter, "metal_program_has_fragment_entry", False
    ):
        return "gl_HelperInvocation"
    return "false"


def render_portable_builtin_call(
    converter: Any,
    source_name: Any,
    function_name: str,
    expr: Any,
    arguments: list[str],
) -> str | None:
    """The CrossGL call for a Metal function this module imports, or None when the call is no such function.

    ``function_name`` is the CrossGL name the call resolved to and ``arguments`` its generated arguments.
    """
    name = _library_name(converter, source_name)
    if name is None:
        return None
    if name in _NESTED and len(arguments) == 3:
        operation = _NESTED[name]
        return (
            f"{operation}({arguments[0]}, {operation}({arguments[1]}, {arguments[2]}))"
        )
    if name in _MEDIANS and len(arguments) == 3:
        return f"{_median_helper(converter, expr)}({', '.join(arguments)})"
    if name in _HELPER_THREADS and not arguments:
        return _helper_thread(converter)
    if (
        name == "simd_shuffle_rotate_up"
        and function_name == "WaveRotate"
        and len(arguments) == 2
    ):
        return f"WaveRotate({arguments[0]}, 0u - {arguments[1]})"
    return None


def generate_portable_builtin_support_code(converter: Any, indent: int = 0) -> str:
    """The CrossGL helpers :func:`render_portable_builtin_call` called, in a fixed order."""
    code = ""
    for type_name in sorted(
        converter.__dict__.get("required_metal_median_types") or ()
    ):
        helper = converter.metal_precise_math_unique_helper_name(
            ("median3", type_name), f"__crossgl_metal_median3_{type_name}"
        )
        code += (
            f"@metal_static\n{type_name} {helper}({type_name} x, {type_name} y, {type_name} z) {{\n"
            "    return max(min(x, y), min(max(x, y), z));\n"
            "}\n"
        )
    pad = "    " * indent
    return "".join(pad + line + "\n" for line in code.strip().splitlines())

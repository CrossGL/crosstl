"""WGSL lowering of the NaN-avoiding minimum and maximum ``fmin``, ``fmax``, ``fmin3`` and ``fmax3``.

CrossGL keeps C's, CUDA's and MSL's ``fmin`` and ``fmax`` (C17 section 7.12.12, MSL 3.2 section 6.5): when one
operand is a NaN they return the other, as SPIR-V's ``NMin`` and ``NMax`` do, and they return a NaN only when both
operands are NaNs. MSL's ``fmin3(x, y, z)`` and ``fmax3(x, y, z)`` are ``fmin(x, fmin(y, z))`` and
``fmax(x, fmax(y, z))``. WGSL declares none of these functions (WGSL section 17.5, "Numeric Built-in Functions"),
and lets an implementation assume that no NaN occurs during execution (WGSL section 15.7, "Floating Point
Evaluation"), so neither its ``min`` and ``max`` nor a comparison of an operand with itself can be relied on to
return the operand that is not a NaN. The call is therefore written to a helper per operand type that tests the
binary32 bits of each operand, which ``bitcast`` exposes unchanged (WGSL section 17.2), as the ``isnan`` lowering in
:mod:`.wgsl_float_classification` does::

    fn crosstl_fmax_f32(x: f32, y: f32) -> f32 {
        let x_nan = ((bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u);
        let y_nan = ((bitcast<u32>(y) & 0x7fffffffu) > 0x7f800000u);
        return select(select(max(x, y), x, y_nan), y, x_nan);
    }

A vector helper tests every component with ``vecN<u32>`` bits and ``vecN<bool>`` selects. A scalar operand next to a
vector one is splatted, and an integer scalar operand is converted to ``f32``, as C converts the operands of
``fmaxf``. The three-operand functions nest two helper calls, which WGSL evaluates left to right (WGSL section 14.1),
so each operand is read once and in source order. WGSL output computes ``half`` values as ``f32``, so every float
operand is binary32 here; an operand of another type, or one whose type is unknown, is refused with
:class:`WGSLNanAwareMinMaxError` rather than reinterpreted.
"""

from __future__ import annotations

from ..ast import IdentifierNode

__all__ = [
    "WGSLNanAwareMinMaxError",
    "is_nan_aware_min_max_call",
    "wgsl_nan_aware_min_max_call",
    "wgsl_nan_aware_min_max_helpers",
]

# Each function, with the two-operand function it applies and its operand count.
_FUNCTIONS = {
    "fmin": ("fmin", 2),
    "fmax": ("fmax", 2),
    "fmin3": ("fmin", 3),
    "fmax3": ("fmax", 3),
}
# The WGSL built-in each helper calls once neither operand is a NaN.
_ORDERED = {"fmin": "min", "fmax": "max"}


class WGSLNanAwareMinMaxError(ValueError):
    """Raised when an ``fmin``/``fmax`` call has no exact WGSL helper: an operand is not a binary32 float."""

    project_diagnostic_code = "project.translate.wgsl-nan-aware-min-max-unsupported"
    missing_capabilities = ("wgsl.nan-aware-min-max-lowering",)

    def __init__(self, message, *, operation, operand_types=()):
        super().__init__(message)
        self.operation = operation
        self.operand_types = tuple(operand_types)


def is_nan_aware_min_max_call(codegen, call):
    """Whether ``call`` is the built-in ``fmin``, ``fmax``, ``fmin3`` or ``fmax3``."""
    function = call.function
    return (
        isinstance(function, IdentifierNode)
        and function.name in _FUNCTIONS
        and not codegen.is_user_function_name(function.name)
    )


def wgsl_nan_aware_min_max_call(codegen, call):
    """The helper call that computes ``call``, registering the helper on ``codegen``."""
    name = call.function.name
    operation, arity = _FUNCTIONS[name]
    operands = list(call.arguments)
    if len(operands) != arity:
        raise WGSLNanAwareMinMaxError(
            f"WGSL target supports {name}() with exactly {arity} arguments; got "
            f"{len(operands)}",
            operation=name,
        )
    types = [codegen.expression_type(operand) for operand in operands]
    target_type, width = _result_type(codegen, types)
    if target_type is None:
        described = ", ".join(_describe_type(codegen, item) for item in types)
        raise WGSLNanAwareMinMaxError(
            f"WGSL target cannot lower {name}() on {described}: WGSL has no {name} "
            "built-in, and the NaN test it is lowered to needs f32 or vecN<f32> "
            "operands",
            operation=name,
            operand_types=[_describe_type(codegen, item) for item in types],
        )
    value_type = "f32" if width == 1 else f"vec{width}<f32>"
    suffix = "f32" if width == 1 else f"vec{width}_f32"
    helper = codegen.safe_wgsl_identifier(f"crosstl_{operation}_{suffix}")
    codegen._nan_aware_min_max_helpers.setdefault(helper, (operation, value_type))
    rendered = []
    for operand, operand_type in zip(operands, types):
        if width > 1 and codegen.integer_scalar_type(operand_type) is not None:
            # vecN<f32>(e) splats only an f32 e, so the integer is converted first.
            converted = f"f32({codegen.generate_expression(operand)})"
            rendered.append(f"{value_type}({converted})")
        else:
            rendered.append(
                codegen.generate_expression_for_target(operand, target_type)
            )
    value = f"{helper}({rendered[0]}, {rendered[1]})"
    for operand in rendered[2:]:
        value = f"{helper}({value}, {operand})"
    return value


def wgsl_nan_aware_min_max_helpers(codegen):
    """The helper functions that the generated calls use, in the order of their first call."""
    helpers = codegen._nan_aware_min_max_helpers
    return "\n\n".join(_helper(helper, *helpers[helper]) for helper in helpers)


def _helper(helper, operation, value_type):
    bits = "u32" if value_type == "f32" else value_type.replace("f32", "u32")

    def nan_test(operand):
        if bits == "u32":
            return f"((bitcast<u32>({operand}) & 0x7fffffffu) > 0x7f800000u)"
        return (
            f"((bitcast<{bits}>({operand}) & {bits}(0x7fffffffu)) > "
            f"{bits}(0x7f800000u))"
        )

    return (
        f"fn {helper}(x: {value_type}, y: {value_type}) -> {value_type} {{\n"
        f"    let x_nan = {nan_test('x')};\n"
        f"    let y_nan = {nan_test('y')};\n"
        f"    return select(select({_ORDERED[operation]}(x, y), x, y_nan), y, x_nan);\n"
        "}"
    )


def _result_type(codegen, types):
    """The operand type every operand is written as, and its width; ``(None, 0)`` when there is none.

    Every operand must be an ``f32`` or ``vecN<f32>`` value, or an integer scalar that converts to ``f32``. The
    vector operands must have one width, and at least one operand must be floating-point.
    """
    target, width, floating = None, 1, False
    for operand_type in types:
        if operand_type is None:
            return None, 0
        scalar = codegen.scalar_type_name(operand_type)
        if scalar == "f32":
            floating = True
            if target is None:
                target = operand_type
            continue
        if scalar in {"i32", "u32"}:
            continue
        shape = codegen.vector_shape(operand_type)
        if shape is None or shape[0] != "f32":
            return None, 0
        if width not in {1, shape[1]}:
            return None, 0
        floating, target, width = True, operand_type, shape[1]
    if not floating:
        return None, 0
    return target, width


def _describe_type(codegen, operand_type):
    if operand_type is None:
        return "an operand of unknown type"
    try:
        return f"a {codegen.type_name_string(operand_type)} operand"
    except ValueError:
        return f"a {operand_type} operand"

"""CUDA and HIP spelling of CrossGL ``powi``, a floating-point base raised to an integer power.

CrossGL spells HIP's integer-power functions under one name, ``powi(x, n)``, for both precisions, as HIP's input
reads ``powif`` (HIP Math API, single precision: ``float powif(float base, int iexp)``) and ``powi`` (double
precision: ``double powi(double base, int iexp)``). Both compute ``base`` to the power ``iexp`` with the C rules for
an integral exponent (C17 F.10.4.4): a negative base keeps its sign for an odd exponent and zero to a negative power
is an infinity.

- HIP output calls ``powif`` for a ``float`` base and ``powi`` for a ``double`` base.
- CUDA declares no integer-power function, and nvcc rejects ``powi``. CUDA output calls the double-precision ``pow``
  with the exponent converted to ``double``, which every ``int`` converts to exactly, and converts a ``float``
  result back. ``pow`` follows the same C rules for an integral exponent, and its double-precision result rounds to
  the ``float`` value ``powif`` computes.

A base of any other type, or a vector base, has no CUDA or HIP function and is refused.
"""

from __future__ import annotations

__all__ = ["INTEGER_POWER_FUNCTIONS", "integer_power_call"]

# ``powif`` stays a name of the set too, so a CrossGL program that spells HIP's single-precision function reaches it.
INTEGER_POWER_FUNCTIONS = frozenset({"powi", "powif"})

_INTEGER_COMPONENTS = frozenset(
    {"int", "uint", "int64", "uint64", "int16", "uint16", "int8", "uint8"}
)


def integer_power_call(codegen, func_name, raw_args, args, refusal):
    """Return the CUDA or HIP call of CrossGL ``powi(x, n)``, or ``None`` when ``func_name`` is not ``powi``.

    ``codegen`` is the CUDA or HIP generator, ``raw_args`` the argument nodes, ``args`` their generated code and
    ``refusal`` the generator's typed refusal (:class:`~.cuda_refusals.CudaLoweringRefused` or
    :class:`~.hip_refusals.HipLoweringRefused`).
    """
    if (
        func_name not in INTEGER_POWER_FUNCTIONS
        or len(args) != 2
        or len(raw_args or []) != 2
        or codegen.is_user_defined_function(func_name)
    ):
        return None
    base_type = _scalar_type(codegen, raw_args[0])
    exponent_type = _scalar_type(codegen, raw_args[1])
    if func_name == "powif" and base_type == "double":
        base_type = "float"
    if base_type not in {"float", "double"} or (
        exponent_type is not None and exponent_type not in _INTEGER_COMPONENTS
    ):
        described = base_type or "an unresolved type"
        raise refusal(
            [
                f"integer power: {func_name} of a {described} base and a {exponent_type or 'unresolved'} "
                f"exponent: {refusal.label} raises only a float or double scalar to an int power"
            ]
        )
    base, exponent = args
    if exponent_type not in {None, "int"}:
        exponent = f"(int)({exponent})"
    if refusal.target_name == "hip":
        name = "powif" if base_type == "float" else "powi"
        if base_type == "float" and _scalar_type(codegen, raw_args[0]) == "double":
            base = f"(float)({base})"
        return f"{name}({base}, {exponent})"
    power = f"pow((double)({base}), (double)({exponent}))"
    return f"(float){power}" if base_type == "float" else power


def _scalar_type(codegen, node):
    """The CrossGL scalar component type of ``node``, or ``None`` for a vector, matrix or an unresolved type."""
    result_type = codegen.expression_result_type(node)
    if result_type is None:
        return None
    if (
        codegen.vector_type_info(result_type) is not None
        or codegen.native_vector_component_type(result_type) is not None
    ):
        return "vector"
    return codegen.scalar_component_type(result_type)

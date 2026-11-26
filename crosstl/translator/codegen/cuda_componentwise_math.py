"""CUDA and HIP output: C math library functions applied to vectors, one component at a time.

The CUDA Math API declares the C99 math functions for ``float`` (``acoshf``, ``fmodf``, ...) and ``double``
(``acosh``, ``fmod``, ...), and HIP's math library declares the same names (CUDA Math API Reference, "Single
Precision Mathematical Functions" and "Double Precision Mathematical Functions"; HIP "Math API"). Neither declares
them for the vector types ``float3``, ``double2`` and so on. Metal, GLSL, HLSL, SPIR-V and Slang define these
functions componentwise on vectors (MSL 3.2, 6.5 "Math Functions"; GLSL 4.60, 8.1-8.3; GLSL.std.450), so a CrossGL
program imported from them calls ``acosh``, ``fmod``, ``copysign`` or ``ilogb`` on a vector, which CUDA and HIP
reject with "no instance of overloaded function". The generators build a ``__device__`` helper that applies the
scalar function to each component and rebuilds the vector, the way they already lower ``sin`` or ``atan2``.

A scalar operand of a two-operand function next to a vector one is applied to every component, as GLSL's
``genType`` functions with a ``float`` parameter do. ``half`` components are computed in ``float`` and narrowed
again, as the generators' other half math helpers do. ``ilogb`` returns ``int`` components.
"""

from ..ast import FunctionCallNode

# One-operand functions whose float and double spellings are ``<name>f`` and ``<name>``.
UNARY_FUNCTIONS = frozenset(
    {
        "acosh", "asinh", "atanh", "cbrt", "erf", "erfc", "erfcinv", "erfinv", "expm1", "fabs", "lgamma",
        "log10", "log1p", "logb", "nearbyint", "rint", "tgamma",
    }
)  # fmt: skip
# One-operand functions with an int result.
INTEGER_RESULT_FUNCTIONS = frozenset({"ilogb"})
# Two-operand functions of two values of one type.
BINARY_FUNCTIONS = frozenset(
    {"copysign", "fdim", "fmax", "fmin", "fmod", "hypot", "nextafter", "remainder"}
)

COMPONENTWISE_FUNCTIONS = UNARY_FUNCTIONS | INTEGER_RESULT_FUNCTIONS | BINARY_FUNCTIONS

_FLOATING_COMPONENTS = {"float", "double", "half"}


def _arity(func_name):
    return 2 if func_name in BINARY_FUNCTIONS else 1


def _call_name(func_name, raw_args):
    if not isinstance(func_name, str) or func_name not in COMPONENTWISE_FUNCTIONS:
        return None
    if len(raw_args or []) != _arity(func_name):
        return None
    return func_name


def _operand_shapes(generator, raw_args):
    """The vector info of the vector operand and, per operand, whether it is a vector; ``None`` if none is."""
    vector_info = None
    shapes = []
    for raw_arg in raw_args:
        info = generator.math_vector_type_info(
            generator.expression_result_type(raw_arg)
        )
        shapes.append(info is not None)
        if info is None:
            continue
        if vector_info is not None and info["type"] != vector_info["type"]:
            return None, None
        vector_info = info
    if vector_info is None or vector_info["component_type"] not in _FLOATING_COMPONENTS:
        return None, None
    return vector_info, shapes


def componentwise_math_result_type(generator, func_name, raw_args):
    """The result type of a componentwise math call on a vector, or ``None`` when it is not one."""
    func_name = _call_name(func_name, raw_args)
    if func_name is None or generator.is_user_defined_function(func_name):
        return None
    vector_info, _ = _operand_shapes(generator, raw_args)
    if vector_info is None:
        return None
    if func_name in INTEGER_RESULT_FUNCTIONS:
        return generator.vector_type_for_components(
            "int", len(vector_info["components"])
        )
    return vector_info["type"]


def generate_componentwise_math_call(generator, func_name, raw_args, args):
    """``func_name(args)`` through a per-component helper when an operand is a vector, else ``None``."""
    func_name = _call_name(func_name, raw_args)
    if func_name is None or len(args) != len(raw_args):
        return None
    vector_info, shapes = _operand_shapes(generator, raw_args)
    if vector_info is None:
        return None
    helper_name = _require_helper(generator, func_name, vector_info, shapes)
    return f"{helper_name}({', '.join(args)})"


def _scalar_spelling(func_name, component_type):
    return func_name if component_type == "double" else f"{func_name}f"


def _parameter_type(generator, vector_info):
    component_type = vector_info["component_type"]
    if component_type == "half":
        return "__half"
    return generator.map_type(component_type)


def _component_call(func_name, component_type, operands, narrow):
    if component_type == "half":
        operands = [f"__half2float({operand})" for operand in operands]
        call = f"{func_name}f({', '.join(operands)})"
        if narrow and func_name not in INTEGER_RESULT_FUNCTIONS:
            return f"__float2half({call})"
        return call
    return f"{_scalar_spelling(func_name, component_type)}({', '.join(operands)})"


def _require_helper(generator, func_name, vector_info, shapes):
    vector_type = vector_info["type"]
    shape_suffix = (
        "" if all(shapes) else "_" + "".join("v" if s else "s" for s in shapes)
    )
    helper_name = generator.sanitize_helper_name(
        f"cgl_{vector_type}_{func_name}{shape_suffix}"
    )
    if helper_name in generator.helper_functions:
        return helper_name

    component_type = vector_info["component_type"]
    scalar_type = _parameter_type(generator, vector_info)
    names = ("value",) if len(shapes) == 1 else ("lhs", "rhs")
    parameters = ", ".join(
        f"{vector_type if is_vector else scalar_type} {name}"
        for name, is_vector in zip(names, shapes)
    )
    if func_name in INTEGER_RESULT_FUNCTIONS:
        result_type = generator.vector_type_for_components(
            "int", len(vector_info["components"])
        )
        result_info = generator.vector_type_info(result_type)
        constructor = result_info["constructor"]
        result_type = result_info["type"]
        narrow = False
    else:
        result_type = vector_type
        constructor = vector_info["constructor"]
        narrow = vector_info.get("narrow_lanes", True)
    components = [
        _component_call(
            func_name,
            component_type,
            [
                f"{name}.{component}" if is_vector else name
                for name, is_vector in zip(names, shapes)
            ],
            narrow,
        )
        for component in vector_info["components"]
    ]
    generator.helper_functions[helper_name] = (
        f"__device__ inline {result_type} {helper_name}({parameters})\n"
        "{\n"
        f"    return {constructor}({', '.join(components)});\n"
        "}"
    )
    return helper_name


def is_componentwise_math_call(node):
    """Whether ``node`` calls one of the componentwise math functions this module lowers."""
    if not isinstance(node, FunctionCallNode):
        return False
    function_expr = getattr(node, "function", getattr(node, "name", None))
    func_name = getattr(function_expr, "name", function_expr)
    return isinstance(func_name, str) and func_name in COMPONENTWISE_FUNCTIONS

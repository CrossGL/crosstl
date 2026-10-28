"""Translate unchanged ``unary.metal``, ``binary.metal`` and ``ternary.metal`` vectors on use.

MLX instantiates ``v_<op><in><out>`` (``unary_v``), ``vv_<op><type>``
(``binary_vv``), ``sv_<op><type>`` and ``vs_<op><type>`` (``binary_sv`` and
``binary_vs``, whose first or second operand is one scalar) and
``v_Select<type>`` (``ternary_v``) for every real dtype an operation supports. The prebuilt families cover the float32, 32-bit integer,
64-bit integer and Boolean sets the host was first proven on; every other real
instantiation, such as the 16-bit floats' transcendentals, the narrow
integers' arithmetic or a float16 selection, is translated once from the
pinned source and cached by ``lazy_kernel_cache``.

The tables below mirror the pinned instantiation macros exactly, including
the explicit ``complex64`` instantiations (``complex64_t`` storage is a packed
float32 pair on every target) and the ``ToFP8`` and ``FromFP8`` conversions
``fast::ConvertFP8`` launches. The per-operation float profiles are the ones the complete
binary and unary kernel proofs pin to reproduce Metal, including the atan2
profile of the complex logarithms and inverse sines and cosines.
"""

from demos.integrations.mlx.portable_host.element_storage import (
    COMPLEX,
    DTYPES,
    FLOATING,
)
from demos.integrations.mlx.portable_host.lazy_kernel_cache import (
    KernelRecipe,
    LazyKernelCache,
)
from demos.integrations.mlx.portable_host.packages import (
    BINARY_SOURCE,
    SELECTION_ENTRIES,
    SELECTION_SOURCE,
    UNARY_SOURCE,
)

CACHE = "elementwise"
# Ternary vectors share the unary prefix, so their kind names the template.
UNARY, BINARY, TERNARY = "v_", "vv_", "ternary_v"
# binary.cpp's ScalarVector and VectorScalar kernels: one operand is a[0] or b[0].
SCALAR_VECTOR, VECTOR_SCALAR = "sv_", "vs_"
BINARY_KERNELS = (BINARY, SCALAR_VECTOR, VECTOR_SCALAR)
FLOATS = ("float16", "float32", "bfloat16")
INTEGERS = tuple(dtype for dtype in DTYPES if dtype.startswith(("int", "uint")))
# instantiate_unary_types / instantiate_unary_float / instantiate_unary_int
UNARY_TYPED = ("Abs", "Ceil", "Floor", "Negative", "Sign", "Square")
UNARY_FLOAT = (
    "ArcCos",
    "ArcCosh",
    "ArcSin",
    "ArcSinh",
    "ArcTan",
    "ArcTanh",
    "Cos",
    "Cosh",
    "Exp",
    "Expm1",
    "Log",
    "Log2",
    "Log10",
    "Log1p",
    "Sigmoid",
    "Erf",
    "ErfInv",
    "Sin",
    "Sinh",
    "Sqrt",
    "Rsqrt",
    "Tan",
    "Tanh",
    "Round",
)
# The explicit instantiate_unary_base_same(op, complex64, complex64_t) list.
UNARY_COMPLEX = (
    "Abs",
    "ArcCos",
    "ArcSin",
    "ArcTan",
    "Conjugate",
    "Cos",
    "Cosh",
    "Exp",
    "Log",
    "Log1p",
    "Log2",
    "Log10",
    "Negative",
    "Sign",
    "Sin",
    "Sinh",
    "Square",
    "Sqrt",
    "Rsqrt",
    "Tan",
    "Tanh",
    "Round",
)
UNARY_TYPES = {
    **{operation: DTYPES for operation in UNARY_TYPED},
    **{operation: FLOATS for operation in UNARY_FLOAT},
    "BitwiseInvert": INTEGERS,
    "LogicalNot": ("bool_",),
}
for _operation in UNARY_COMPLEX:
    UNARY_TYPES[_operation] = (*UNARY_TYPES.get(_operation, ()), COMPLEX)
# instantiate_unary_base(Real|Imag, complex64, float32, complex64_t, float)
UNARY_PARTS = ("Real", "Imag")
# instantiate_unary_all(ToFP8, <float>, uint8, ...) and
# instantiate_unary_all(FromFP8, uint8, <float>, ...): E4M3 bytes.
FP8_CONVERSIONS = tuple(
    (operation, source, destination)
    for dtype in FLOATS
    for operation, source, destination in (
        ("ToFP8", dtype, "uint8"),
        ("FromFP8", "uint8", dtype),
    )
)
# instantiate_binary_types / _types_bool / _float / _integer and the explicit
# NaNEqual, logical and Boolean bitwise instantiations.
BINARY_TYPED = (
    "Add",
    "Divide",
    "Maximum",
    "Minimum",
    "Multiply",
    "Subtract",
    "Power",
    "Remainder",
)
COMPARISONS = ("Equal", "Greater", "GreaterEqual", "Less", "LessEqual", "NotEqual")
BINARY_TYPES = {
    **{operation: (*DTYPES, COMPLEX) for operation in (*BINARY_TYPED, *COMPARISONS)},
    "LogAddExp": (*FLOATS, COMPLEX),
    "ArcTan2": FLOATS,
    "NaNEqual": (*FLOATS, COMPLEX),
    "LogicalOr": ("bool_",),
    "LogicalAnd": ("bool_",),
    **{
        operation: (*INTEGERS, "bool_")
        for operation in ("BitwiseAnd", "BitwiseOr", "BitwiseXor")
    },
    "LeftShift": INTEGERS,
    "RightShift": INTEGERS,
}
BOOLEAN_RESULTS = frozenset((*COMPARISONS, "NaNEqual"))
# instantiate_ternary_types(Select); the selection family installs the
# float32, 32-bit integer and Boolean vectors.
TERNARY_TYPES = tuple(
    dtype for dtype in (*DTYPES, COMPLEX) if f"v_Select{dtype}" not in SELECTION_ENTRIES
)
ENTRIES = {
    **{
        f"{UNARY}{operation}{dtype}{dtype}": (UNARY, operation, dtype, dtype)
        for operation, dtypes in UNARY_TYPES.items()
        for dtype in dtypes
    },
    **{
        f"{UNARY}{operation}{COMPLEX}float32": (UNARY, operation, COMPLEX, "float32")
        for operation in UNARY_PARTS
    },
    **{
        f"{UNARY}{operation}{source}{destination}": (
            UNARY,
            operation,
            source,
            destination,
        )
        for operation, source, destination in FP8_CONVERSIONS
    },
    **{
        f"{kernel}{operation}{dtype}": (
            kernel,
            operation,
            dtype,
            "bool_" if operation in BOOLEAN_RESULTS else dtype,
        )
        for kernel in BINARY_KERNELS
        for operation, dtypes in BINARY_TYPES.items()
        for dtype in dtypes
    },
    **{
        f"{UNARY}Select{dtype}": (TERNARY, "Select", dtype, dtype)
        for dtype in TERNARY_TYPES
    },
}
PROFILES = (
    ("binary32_fma_profile", "rne-flush"),
    ("binary32_sqrt_profile", "flush-subnormals"),
    ("binary32_rsqrt_profile", "flush-subnormals"),
)
BINARY32 = frozenset({"float32", "bfloat16"})
# Complex operations whose polar forms the unary proofs pin with atan2.
COMPLEX_ATAN2 = frozenset({"Log", "Log2", "Log10", "ArcSin", "ArcCos"})


def signature(entry):
    """``(kernel, operation, input dtype, output dtype)`` of an entry."""
    try:
        return ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown elementwise entry") from error


def operand_types(entry):
    """The element type of every source parameter of ``entry``, its output last."""
    kernel, _, source, destination = signature(entry)
    if kernel == UNARY:
        return {"in": source, "out": destination}
    if kernel in BINARY_KERNELS:
        return {"a": source, "b": source, "c": destination}
    return {"a": "bool_", "b": source, "c": source, "d": destination}


def operation_profiles(kernel, operation, dtype):
    """The pinned Metal float profiles one operation and input type need."""
    profiles = []
    if kernel == TERNARY:
        return ()
    if kernel == UNARY:
        if operation == "Log" and (dtype in FLOATING or dtype == COMPLEX):
            profiles.append(("binary32_log_profile", "flush-subnormals"))
        if dtype == COMPLEX and operation in COMPLEX_ATAN2:
            profiles.append(("binary32_atan2_profile", "flush-subnormals"))
        return tuple(profiles)
    if operation == "ArcTan2":
        profiles.append(("binary32_atan2_profile", "flush-subnormals"))
    if operation == "Remainder" and dtype == "float16":
        profiles.append(("binary16_remainder_profile", "binary32-quotient"))
    if dtype in BINARY32:
        if operation in {"Remainder", "Minimum", "Maximum"}:
            profiles.append(("binary32_comparison_profile", "flush-subnormals"))
        if operation == "Remainder":
            profiles.append(
                ("binary32_remainder_profile", "flush-arithmetic-subnormals")
            )
        if operation in {"Remainder", "Add", "Subtract"}:
            profiles.append(("binary32_additive_profile", "rne-flush"))
        if operation == "Divide":
            profiles.append(("binary32_division_profile", "rne-flush"))
        if operation == "Multiply":
            profiles.append(("binary32_multiplication_profile", "rne-flush"))
        if operation == "Power":
            profiles.append(("binary32_power_operand_profile", "flush-subnormals"))
    if operation == "Power" and dtype == "float32":
        profiles.append(("binary32_power_accuracy_profile", "portable-finite"))
    return tuple(profiles)


def options(target, entry):
    """The project source options of ``entry``; every target shares them."""
    kernel, operation, dtype, _ = signature(entry)
    return "".join(
        f'{option} = "{value}"\n'
        for option, value in (*PROFILES, *operation_profiles(kernel, operation, dtype))
    )


def _kernel_signature(*kernels):
    def checked(entry):
        result = signature(entry)
        if result[0] not in kernels:
            raise ValueError("Unknown elementwise entry")
        return result

    return checked


SOURCES = {
    UNARY: UNARY_SOURCE,
    **dict.fromkeys(BINARY_KERNELS, BINARY_SOURCE),
    TERNARY: SELECTION_SOURCE,
}


def _source(entry):
    return SOURCES[signature(entry)[0]]


UNARY_RECIPE = KernelRecipe(
    "Unary", UNARY_SOURCE, _kernel_signature(UNARY), options, __file__
)
BINARY_RECIPE = KernelRecipe(
    "Binary", BINARY_SOURCE, _kernel_signature(*BINARY_KERNELS), options, __file__
)
TERNARY_RECIPE = KernelRecipe(
    "Ternary", SELECTION_SOURCE, _kernel_signature(TERNARY), options, __file__
)


class ElementwisePackageCache:
    """One target's lazy vector packages, one cache per source."""

    def __init__(self, root, directory, target):
        self.target = target
        self.caches = {
            recipe.source: LazyKernelCache(root, directory, target, recipe)
            for recipe in (UNARY_RECIPE, BINARY_RECIPE, TERNARY_RECIPE)
        }

    def get(self, entry):
        """The ``(descriptor, package directory)`` of ``entry``."""
        return self.caches[_source(entry)].get(entry)

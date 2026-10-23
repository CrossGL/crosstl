"""Translate unchanged ``unary.metal`` and ``binary.metal`` vector entries on use.

MLX instantiates ``v_<op><in><out>`` (``unary_v``) and ``vv_<op><type>``
(``binary_vv``) for every real dtype an operation supports. The prebuilt
families cover the float32, 32-bit integer, 64-bit integer and Boolean sets
the host was first proven on; every other real instantiation, such as the
16-bit floats' transcendentals or the narrow integers' arithmetic, is
translated once from the pinned source and cached by ``lazy_kernel_cache``.

The tables below mirror the pinned instantiation macros exactly, including
the explicit ``complex64`` instantiations (``complex64_t`` storage is a packed
float32 pair on every target) and without the FP8 conversions (``ConvertFP8``
has no host port). The per-operation float profiles are the ones the complete
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
from demos.integrations.mlx.portable_host.packages import BINARY_SOURCE, UNARY_SOURCE

CACHE = "elementwise"
UNARY, BINARY = "v_", "vv_"
ENTRY_PREFIXES = (UNARY, BINARY)
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
        f"{BINARY}{operation}{dtype}": (
            BINARY,
            operation,
            dtype,
            "bool_" if operation in BOOLEAN_RESULTS else dtype,
        )
        for operation, dtypes in BINARY_TYPES.items()
        for dtype in dtypes
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


def operation_profiles(kernel, operation, dtype):
    """The pinned Metal float profiles one operation and input type need."""
    profiles = []
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


def _kernel_signature(kernel):
    def checked(entry):
        result = signature(entry)
        if result[0] != kernel:
            raise ValueError("Unknown elementwise entry")
        return result

    return checked


def _source(entry):
    return UNARY_SOURCE if signature(entry)[0] == UNARY else BINARY_SOURCE


UNARY_RECIPE = KernelRecipe(
    "Unary", UNARY_SOURCE, _kernel_signature(UNARY), options, __file__
)
BINARY_RECIPE = KernelRecipe(
    "Binary", BINARY_SOURCE, _kernel_signature(BINARY), options, __file__
)


class ElementwisePackageCache:
    """One target's lazy ``v_`` and ``vv_`` packages, one cache per source."""

    def __init__(self, root, directory, target):
        self.target = target
        self.caches = {
            UNARY_SOURCE: LazyKernelCache(root, directory, target, UNARY_RECIPE),
            BINARY_SOURCE: LazyKernelCache(root, directory, target, BINARY_RECIPE),
        }

    def get(self, entry):
        """The ``(descriptor, package directory)`` of ``entry``."""
        return self.caches[_source(entry)].get(entry)

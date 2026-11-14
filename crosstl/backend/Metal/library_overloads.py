"""The non-template overloads of the Metal standard library functions that take scalar and vector arguments.

A user template named like a library function, such as ``template <typename T, enable_if_t<...>> T max(T, T)``,
competes with the library's overloads wherever lookup finds both: at global scope beside ``using namespace metal;``.
The library declares ``max``, ``isnan`` and their relatives as non-template functions, one per scalar and vector type
(``METAL_FUNC float max(float x, float y)``). When the library has an overload whose parameter types are the
argument types, it is as good a match as the template's deduced specialization, and C++ prefers the non-template
function ([over.match.best]): ``max(x, 1.0f)`` on a ``float x`` calls ``metal::max`` even though the user template
is enabled for ``float``. Without such an overload, the template's exact match beats every library overload that
needs a conversion, as ``isnan(int)`` beside an integral-only ``isnan`` template does.

``_OVERLOAD_GROUPS`` lists the declarations of ``metal_common``, ``metal_geometric``, ``metal_integer``,
``metal_math`` and ``metal_relational`` in the Metal toolchain (Metal compiler 32023, MSL 3.2) that belong to
namespace ``metal`` itself, not to ``metal::fast`` or ``metal::precise``. Each group is a parameter pattern, the
types its ``T`` stands for (element types times vector widths, width 1 being the scalar) and the functions that
declare it. Overloads behind feature macros, such as the ``double`` ones, are included, so the table may list
overloads a target lacks; ``thread T &`` parameters are kept as written.
"""

from functools import lru_cache
from typing import Dict, FrozenSet, Optional, Sequence, Tuple

from crosstl.translator.refusals import MetalSourceRefusal

__all__ = [
    "LIBRARY_OVERLOAD_NAMES",
    "library_overload_precedence",
    "library_parameter_lists",
    "library_type_spelling",
]

_OVERLOAD_GROUPS: Tuple[Tuple[str, str, object], ...] = (
    (
        "T",
        "bfloat double float half x 1 2 3 4 8 16",
        "isfinite isinf isnan isnormal signbit",
    ),
    ("T", "bool x 1 2 3 4 8 16", "all any"),
    (
        "T",
        "char double float half int long short uchar uint ulong ushort x 1 2 3 4 8 16",
        "abs",
    ),
    (
        "T",
        "char int long short uchar uint ulong ushort x 1 2 3 4 8 16",
        "clz ctz popcount reverse_bits",
    ),
    (
        "T",
        "double float half x 1 2 3 4 8 16",
        (
            "acos acosh asin asinh atan atanh ceil cos cosh cospi exp exp10 exp2 fabs "
            "floor fract ilogb log log10 log2 rint round rsqrt saturate sign sin sinh "
            "sinpi sqrt tan tanh tanpi trunc"
        ),
    ),
    ("T", "double float half x 2 3 4 8 16", "length length_squared normalize"),
    ("T", "uchar uint ushort x 2", "interleave"),
    ("T", "uint ulong ushort x 1", "deinterleave"),
    ("T,T", "bfloat double float half x 1 2 3 4 8 16", "isordered isunordered"),
    (
        "T,T",
        "char double float half int long short uchar uint ulong ushort x 1 2 3 4 8 16",
        "max min",
    ),
    (
        "T,T",
        "char int long short uchar uint ulong ushort x 1 2 3 4 8 16",
        "absdiff addsat hadd mulhi rhadd rotate subsat",
    ),
    (
        "T,T",
        "double float half x 1 2 3 4 8 16",
        "atan2 copysign divide fdim fmax fmin fmod nextafter pow powr step",
    ),
    ("T,T", "double float half x 2 3 4 8 16", "distance distance_squared dot reflect"),
    ("T,T", "double float half x 3", "cross"),
    ("T,T", "int uint x 1 2 3 4 8 16", "mul24"),
    ("T,T", "uchar uint ushort x 1", "interleave"),
    ("T,T,T", "bool x 1 2 3 4 8 16", "select"),
    (
        "T,T,T",
        "char double float half int long short uchar uint ulong ushort x 1 2 3 4 8 16",
        "clamp max3 median3 min3",
    ),
    (
        "T,T,T",
        "char int long short uchar uint ulong ushort x 1 2 3 4 8 16",
        "madhi madsat",
    ),
    (
        "T,T,T",
        "double float half x 1 2 3 4 8 16",
        "fma fmax3 fmedian3 fmin3 mix smoothstep",
    ),
    ("T,T,T", "double float half x 2 3 4 8 16", "faceforward"),
    ("T,T,T", "int uint x 1 2 3 4 8 16", "mad24"),
    ("T,T,T", "uint x 1", "extract_bits"),
    ("T,T,T,T", "uint x 1", "insert_bits"),
    (
        "T,T,bool",
        "bfloat char double float half int long short uchar uint ulong ushort x 1",
        "select",
    ),
    (
        "T,T,bool16",
        "bfloat char double float half int long short uchar uint ulong ushort x 16",
        "select",
    ),
    (
        "T,T,bool2",
        "bfloat char double float half int long short uchar uint ulong ushort x 2",
        "select",
    ),
    (
        "T,T,bool3",
        "bfloat char double float half int long short uchar uint ulong ushort x 3",
        "select",
    ),
    (
        "T,T,bool4",
        "bfloat char double float half int long short uchar uint ulong ushort x 4",
        "select",
    ),
    (
        "T,T,bool8",
        "bfloat char double float half int long short uchar uint ulong ushort x 8",
        "select",
    ),
    ("T,T,double", "double x 2 3 4 8 16", "refract"),
    ("T,T,float", "float x 2 3 4 8 16", "refract"),
    ("T,T,half", "half x 2 3 4 8 16", "refract"),
    (
        "T,T,uint,uint",
        "char int long short uchar ulong ushort x 1 2 3 4 8 16; uint x 2 3 4 8 16",
        "insert_bits",
    ),
    ("T,int", "double float half x 1", "ldexp"),
    ("T,int16", "double float half x 16", "ldexp"),
    ("T,int2", "double float half x 2", "ldexp"),
    ("T,int3", "double float half x 3", "ldexp"),
    ("T,int4", "double float half x 4", "ldexp"),
    ("T,int8", "double float half x 8", "ldexp"),
    ("T,thread double &", "double x 1", "modf sincos"),
    ("T,thread double16 &", "double x 16", "modf sincos"),
    ("T,thread double2 &", "double x 2", "modf sincos"),
    ("T,thread double3 &", "double x 3", "modf sincos"),
    ("T,thread double4 &", "double x 4", "modf sincos"),
    ("T,thread double8 &", "double x 8", "modf sincos"),
    ("T,thread float &", "float x 1", "modf sincos"),
    ("T,thread float16 &", "float x 16", "modf sincos"),
    ("T,thread float2 &", "float x 2", "modf sincos"),
    ("T,thread float3 &", "float x 3", "modf sincos"),
    ("T,thread float4 &", "float x 4", "modf sincos"),
    ("T,thread float8 &", "float x 8", "modf sincos"),
    ("T,thread half &", "half x 1", "modf sincos"),
    ("T,thread half16 &", "half x 16", "modf sincos"),
    ("T,thread half2 &", "half x 2", "modf sincos"),
    ("T,thread half3 &", "half x 3", "modf sincos"),
    ("T,thread half4 &", "half x 4", "modf sincos"),
    ("T,thread half8 &", "half x 8", "modf sincos"),
    ("T,thread int &", "double float half x 1", "frexp"),
    ("T,thread int16 &", "double float half x 16", "frexp"),
    ("T,thread int2 &", "double float half x 2", "frexp"),
    ("T,thread int3 &", "double float half x 3", "frexp"),
    ("T,thread int4 &", "double float half x 4", "frexp"),
    ("T,thread int8 &", "double float half x 8", "frexp"),
    (
        "T,uint,uint",
        "char int long short uchar ulong ushort x 1 2 3 4 8 16; uint x 2 3 4 8 16",
        "extract_bits",
    ),
)

# Spellings the importer may infer for an element type, and the library's spelling of the same type.
_ELEMENT_SPELLINGS = {
    "int8_t": "char",
    "uint8_t": "uchar",
    "int16_t": "short",
    "uint16_t": "ushort",
    "int32_t": "int",
    "uint32_t": "uint",
    "int64_t": "long",
    "uint64_t": "ulong",
    "size_t": "ulong",
    "float16_t": "half",
    "bfloat16_t": "bfloat",
}


# The element types the library's overloads are spelled with.
_LIBRARY_ELEMENTS = frozenset(
    (
        "bool",
        "char",
        "uchar",
        "short",
        "ushort",
        "int",
        "uint",
        "long",
        "ulong",
        "half",
        "bfloat",
        "float",
        "double",
    )
)


def library_type_spelling(element: str, width: int) -> Optional[str]:
    """The library's spelling of a scalar (``width`` 1) or vector of ``element``.

    ``None`` when ``element`` is not a scalar type the library spells.
    """
    if element in _ELEMENT_SPELLINGS:
        element = _ELEMENT_SPELLINGS[element]
    elif element not in _LIBRARY_ELEMENTS:
        return None
    return element if width == 1 else f"{element}{width}"


def _group_types(types: str) -> Tuple[str, ...]:
    spellings = []
    for family in types.split(";"):
        elements, widths = family.split(" x ")
        for element in elements.split():
            for width in widths.split():
                spelling = library_type_spelling(element, int(width))
                if spelling is None:
                    raise MetalSourceRefusal(
                        f"unknown library element type {element!r}"
                    )
                spellings.append(spelling)
    return tuple(spellings)


@lru_cache(maxsize=1)
def _overloads() -> Dict[str, FrozenSet[Tuple[str, ...]]]:
    overloads: Dict[str, set] = {}
    for pattern, types, names in _OVERLOAD_GROUPS:
        parameters = pattern.split(",")
        for spelling in _group_types(types):
            signature = tuple(spelling if part == "T" else part for part in parameters)
            for name in str(names).split():
                overloads.setdefault(name, set()).add(signature)
    return {name: frozenset(signatures) for name, signatures in overloads.items()}


LIBRARY_OVERLOAD_NAMES: FrozenSet[str] = frozenset(
    name for _pattern, _types, names in _OVERLOAD_GROUPS for name in str(names).split()
)


def library_parameter_lists(name: str) -> FrozenSet[Tuple[str, ...]]:
    """The parameter lists of the library's non-template overloads of ``name``."""
    return _overloads().get(name, frozenset())


def _value_type(parameter: str) -> Tuple[str, bool]:
    """A library parameter's value type, and whether it is a reference."""
    if parameter.endswith("&"):
        return parameter[:-1].replace("thread", "", 1).strip(), True
    return parameter, False


def library_overload_precedence(
    name: str,
    argument_types: Sequence[Optional[str]],
    template_parameter_types: Sequence[Optional[str]],
) -> str:
    """Which function overload resolution picks between an enabled template and the library's overloads of ``name``.

    ``argument_types`` and ``template_parameter_types`` use the library's spellings (``library_type_spelling``),
    a template parameter that is a non-const ``thread`` reference ending in ``&``; ``None`` marks an argument or
    parameter that is not a scalar or vector. Returns ``"library"`` when the library
    has an overload exactly as good as the template's exact match, ``"template"`` when the template wins, and
    ``"unproven"`` when the outcome depends on conversions, references or declarations the table does not model.
    """
    lists = [
        parameters
        for parameters in library_parameter_lists(name)
        if len(parameters) == len(argument_types)
    ]
    if not lists:
        return "template"
    if None in argument_types:
        return "unproven"
    for parameters in lists:
        values = [_value_type(parameter) for parameter in parameters]
        if [value for value, _reference in values] == list(argument_types):
            if any(reference for _value, reference in values):
                return "unproven"
            # Each argument initializes this overload's by-value parameter of its own type with the identity
            # conversion, which no conversion sequence of the template betters ([over.ics.rank]/3.2; the
            # reference-binding rules apply only when both sequences bind references). So the template ties on
            # every argument, is worse on one, or is not viable; a tie goes to the non-template function
            # ([over.match.best]/2.4). This holds whatever the template's parameter types are.
            return "library"
    template_values = [
        _value_type(parameter) if parameter is not None else (None, False)
        for parameter in template_parameter_types[: len(argument_types)]
    ]
    if [value for value, _reference in template_values] != list(argument_types):
        return "unproven"
    if any(reference for _value, reference in template_values):
        # Viable only if the argument is an lvalue, which the caller does not know.
        return "unproven"
    return "template"

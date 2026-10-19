"""OpenCL C conversion built-ins (OpenCL C 6.4.3): ``convert_<type>[<n>][_sat][_<rounding>](x)``.

Each conversion is imported into CrossGL so that its OpenCL meaning survives:

- a conversion to ``int`` or ``uint`` that saturates or names a rounding mode becomes a generic hub intrinsic
  (``__cgl_convert_sat_int_rte(x)``, ``__cgl_convert_uint_rtp(x)``, ...; see
  ``crosstl/translator/semantics/conversions.py``), which the hub resolves by operand type and every target
  discharges through the ``convert_sat`` rule family;
- a saturated conversion to ``char``, ``uchar``, ``short`` or ``ushort`` saturates to ``int`` first and then clamps
  to the narrow range, which is exact; a narrow conversion with only a rounding mode rounds through ``int``;
- every other conversion to an integer type, and a conversion to a floating-point type in the default
  round-to-nearest-even mode, is a CrossGL constructor cast.

A float operand of a non-saturated conversion with a rounding mode also saturates: OpenCL C 6.4.3.3 makes its
out-of-range result implementation-defined, and saturation is a defined choice, whereas a target's own cast may be
undefined. Constructs CrossGL cannot carry raise :class:`OpenCLConversionUnsupportedError`: ``_sat`` on a
floating-point destination (not valid OpenCL C), directed rounding to a floating-point type, saturation or directed
rounding to a 64-bit integer, and vectors of 8 or 16 lanes.
"""

from __future__ import annotations

import re

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

CONVERSION_RE = re.compile(
    r"convert_(?P<type>u?char|u?short|u?int|u?long|float|double|half)"
    r"(?P<lanes>2|3|4|8|16)?(?P<sat>_sat)?(?:_(?P<mode>rte|rtz|rtp|rtn))?"
)
_FLOATING = frozenset({"float", "double", "half"})
# CrossGL's own vector spellings, which every emitter constructs; narrow vectors keep the importer's spelling.
_VECTOR_PREFIXES = {
    "int": "ivec",
    "uint": "uvec",
    "float": "vec",
    "double": "dvec",
    "half": "f16vec",
}
_NARROW_RANGES = {
    "char": (-128, 127),
    "uchar": (0, 255),
    "short": (-32768, 32767),
    "ushort": (0, 65535),
}


class OpenCLConversionUnsupportedError(TranslationDiagnostic):
    """An OpenCL C conversion built-in that CrossGL cannot carry exactly."""

    code = CODE_PREFIX + "opencl-conversion-unsupported"


def is_conversion_name(name) -> bool:
    return isinstance(name, str) and CONVERSION_RE.fullmatch(name) is not None


def format_conversion(name, argument, crossgl_type):
    """The CrossGL text of ``name(argument)``, with ``argument`` already rendered; ``None`` if not a conversion.

    ``crossgl_type`` maps an OpenCL type spelling (``uchar4``) to its CrossGL spelling.
    """
    match = CONVERSION_RE.fullmatch(name) if isinstance(name, str) else None
    if match is None:
        return None
    base, lanes = match["type"], int(match["lanes"] or 1)
    saturate, mode = match["sat"] is not None, match["mode"]
    if lanes > 4:
        raise OpenCLConversionUnsupportedError(
            f"{name} converts {lanes}-lane vectors, which CrossGL vectors cannot hold"
        )
    destination = _type_name(base, lanes, crossgl_type)
    if base in _FLOATING:
        if saturate:
            raise OpenCLConversionUnsupportedError(
                f"{name} saturates to a floating-point type; OpenCL C 6.4.3.3 allows _sat only for integer "
                "destinations"
            )
        if mode not in (None, "rte"):
            raise OpenCLConversionUnsupportedError(
                f"{name} rounds to a floating-point type in mode _{mode}; CrossGL conversions to floating-point "
                "types round to nearest even"
            )
        return f"{destination}({argument})"
    directed = mode not in (None, "rtz")
    if not saturate and not directed:
        return f"{destination}({argument})"
    if base in ("long", "ulong"):
        raise OpenCLConversionUnsupportedError(
            f"{name} saturates or rounds to a 64-bit integer, which the convert_sat rule family does not cover"
        )
    suffix = f"{'_sat' if saturate else ''}{'_' + mode if mode else ''}"
    if base in ("int", "uint"):
        return f"__cgl_convert_{base}{suffix}({argument})"
    widened = f"__cgl_convert_int{suffix}({argument})"
    if not saturate:
        # Narrowing an int is modular, as OpenCL C 6.4.3.3 requires through C99 6.3.1.3 for unsigned types.
        return f"{destination}({widened})"
    low, high = _NARROW_RANGES[base]
    if lanes == 1:
        return f"{destination}(clamp({widened}, {low}, {high}))"
    bound = _type_name("int", lanes, crossgl_type)
    return f"{destination}(clamp({widened}, {bound}({low}), {bound}({high})))"


def _type_name(base, lanes, crossgl_type):
    if lanes > 1 and base in _VECTOR_PREFIXES:
        return f"{_VECTOR_PREFIXES[base]}{lanes}"
    return crossgl_type(base if lanes == 1 else f"{base}{lanes}")

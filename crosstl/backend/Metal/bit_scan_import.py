"""MSL ``clz`` and ``ctz`` imported as CrossGL bit scans.

MSL ``clz(x)`` counts the leading zero bits of ``x`` and ``ctz(x)`` its trailing zero bits, per component; for a
zero component both return the component's bit width (MSL 3.2 section 6.13, "Integer functions"). CrossGL has the
GLSL scans ``findMSB`` and ``findLSB`` (GLSL 4.60 section 8.8), which return the bit index or -1 for zero, and
``findMSB`` of a negative signed value finds the highest bit that differs from the sign bit. So, for a ``W``-bit
component, the import scans the component's unsigned bit pattern ``u``:

* ``clz(x) = T((W - 1) - findMSB(u))``, which is ``W`` for zero;
* ``ctz(x) = T(min(U(findLSB(u)), W))``, where ``findLSB`` of zero converts to the all-ones unsigned value.

8- and 16-bit components are widened to 32 bits and masked to their own width. 64-bit components, which
``findMSB`` does not take, and arguments whose integer type is unknown are refused rather than guessed.
"""

import re

SCANS = frozenset({"clz", "ctz"})

_UNSIGNED_VECTOR = {1: "uint", 2: "uvec2", 3: "uvec3", 4: "uvec4"}


class MetalBitScanError(ValueError):
    """An MSL ``clz``/``ctz`` call whose argument has no 32-bit-or-narrower integer component type."""

    project_diagnostic_code = "project.translate.metal-bit-scan-unsupported"
    missing_capabilities = ("metal.bit-scan",)

    def __init__(self, function, argument_type):
        self.function = function
        self.argument_type = argument_type
        self.reason = (
            f"MSL {function}() of a {argument_type or 'value of unknown type'} has no CrossGL bit-scan lowering: "
            "findMSB and findLSB take 32-bit integers, so only integer components of 8, 16 or 32 bits are imported"
        )
        super().__init__(self.reason)


def scan_function(name, user_function_names):
    """``clz``/``ctz`` when ``name`` calls the MSL function (``metal::`` qualified or unqualified), else ``None``."""
    text = str(name).lstrip(":")
    qualified = text.startswith("metal::")
    unscoped = text[len("metal::") :] if qualified else text
    if unscoped not in SCANS:
        return None
    if not qualified and unscoped in user_function_names:
        return None
    return unscoped


def lower_scan(function, argument, metal_type, mapped_type, vector_parts, scalar_info):
    """The CrossGL expression for ``function(argument)``.

    ``metal_type`` is the argument's MSL type and ``mapped_type`` its CrossGL type; ``vector_parts`` is
    ``(element type, width)`` for a vector and ``scalar_info(type)`` gives ``(family, signed, bits)``.
    """
    element, width = vector_parts(metal_type) or (metal_type, 1)
    info = scalar_info(element) if element else None
    if (
        info is None
        or info[0] != "integer"
        or info[2] not in (8, 16, 32)
        or width not in _UNSIGNED_VECTOR
        or not mapped_type
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(mapped_type))
    ):
        raise MetalBitScanError(function, metal_type)
    _family, signed, bits = info
    unsigned = _UNSIGNED_VECTOR[width]
    pattern = (
        argument
        if (bits == 32 and not signed and width == 1)
        else f"{unsigned}({argument})"
    )
    if bits < 32:
        mask = (
            f"{unsigned}({hex((1 << bits) - 1)}u)"
            if width > 1
            else f"{hex((1 << bits) - 1)}u"
        )
        pattern = f"({pattern} & {mask})"
    if function == "clz":
        count = f"{bits - 1} - findMSB({pattern})"
    else:
        limit = f"{unsigned}({bits}u)" if width > 1 else f"{bits}u"
        count = f"min({unsigned}(findLSB({pattern})), {limit})"
    return f"{mapped_type}({count})"

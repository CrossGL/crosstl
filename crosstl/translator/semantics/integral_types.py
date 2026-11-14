"""Integer types in the constant expressions the translator evaluates: widths, conversions and type aliases.

The constant evaluators of :mod:`.binding_constants`, :mod:`.constant_extents` and :mod:`.static_assertions` read
a program's integer ``const`` declarations and conversions such as ``uint8_t(x)`` and ``(ushort)x``. Two rules of the
source languages decide the values they compute:

* A conversion to an integer type, explicit or by initializing a constant of that type, yields the value congruent
  to the source value modulo ``2**N``, where ``N`` is the width of the type, that the type can represent (C++20
  [conv.integral]/3, which MSL and HLSL follow; GLSL constructors keep the bit pattern of a 32-bit integer). ``uint8_t(300)`` is ``44`` and ``ushort(65537)`` is ``1``. The evaluators used to keep the source
  value, so ``static_assert(uint8_t(300) == 44)`` failed and ``[[buffer(uint8_t(258))]]`` bound slot 258 instead of
  2. :func:`convert_to_integer_type` performs the conversion.
* A type alias names the type it aliases (C++17 [dcl.typedef]/1). The MSL, HLSL and GLSL generators keep the
  program-scope aliases of CrossGL, so a constant declared ``const t C = 2;`` after ``typedef uint t;`` still names
  ``t`` when the evaluators run, and they refused it as "not an integer constant". :class:`ProgramTypeAliases`
  resolves those names to the type at the end of the alias chain.
"""

from __future__ import annotations

from .. import ast as A

__all__ = [
    "INTEGER_REPRESENTATIONS",
    "ProgramTypeAliases",
    "convert_to_integer_type",
]

# Width in bits and signedness of every integer type spelling the evaluators accept. MSL's scalar types make ``char``
# a signed 8-bit integer and ``long``, ``ulong`` and ``size_t`` 64-bit ones.
INTEGER_REPRESENTATIONS = {
    "char": (8, True),
    "int8_t": (8, True),
    "int8": (8, True),
    "uchar": (8, False),
    "uint8_t": (8, False),
    "uint8": (8, False),
    "short": (16, True),
    "int16_t": (16, True),
    "int16": (16, True),
    "ushort": (16, False),
    "uint16_t": (16, False),
    "uint16": (16, False),
    "int": (32, True),
    "int32_t": (32, True),
    "uint": (32, False),
    "uint32_t": (32, False),
    "long": (64, True),
    "int64_t": (64, True),
    "int64": (64, True),
    "ulong": (64, False),
    "uint64_t": (64, False),
    "uint64": (64, False),
    "size_t": (64, False),
}


def convert_to_integer_type(value: int, type_name: str) -> int:
    """``value`` converted to the integer type ``type_name``: the representable value congruent modulo ``2**N``."""
    bits, signed = INTEGER_REPRESENTATIONS[type_name]
    modulus = 1 << bits
    value %= modulus
    if signed and value >= modulus >> 1:
        value -= modulus
    return value


def _type_name(type_value):
    name = getattr(type_value, "name", None)
    if isinstance(name, str):
        return name
    return type_value if isinstance(type_value, str) else None


class ProgramTypeAliases:
    """The program-scope type aliases of a CrossGL program, each resolved to the type at the end of its chain.

    An alias that carries a qualifier or an attribute (``typedef const uint cu;``) is not recorded: the evaluators
    then treat its name as an unknown type, as before.
    """

    def __init__(self, program):
        self.aliased = {}
        for node in getattr(program, "global_variables", None) or []:
            if not (
                isinstance(node, A.VariableNode)
                and getattr(node, "is_type_alias", False)
            ):
                continue
            if getattr(node, "qualifiers", None) or getattr(node, "attributes", None):
                continue
            name = getattr(node, "name", None)
            aliased = getattr(node, "var_type", None)
            if isinstance(name, str) and aliased is not None:
                self.aliased[name] = aliased

    @staticmethod
    def _alias_name(type_value):
        """The name ``type_value`` spells when it could name an alias: a plain name without generic arguments."""
        if isinstance(type_value, str):
            return type_value
        if isinstance(type_value, A.NamedType) and not type_value.generic_args:
            return type_value.name
        if isinstance(type_value, A.PrimitiveType):
            return type_value.name
        return None

    def resolve_type(self, type_value):
        """The type ``type_value`` denotes: itself unless it names a program-scope alias."""
        seen = set()
        name = self._alias_name(type_value)
        while name in self.aliased and name not in seen:
            seen.add(name)
            type_value = self.aliased[name]
            name = self._alias_name(type_value)
        return type_value

    def resolve(self, type_name):
        """The name of the type ``type_name`` denotes, or ``None`` when that type has no plain name."""
        return _type_name(self.resolve_type(type_name))

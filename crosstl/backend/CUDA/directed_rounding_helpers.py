"""CrossGL functions that compute CUDA's directed-rounding intrinsics exactly.

``__fadd_rd``, ``__dmul_ru``, ``__fsqrt_rz``, ``__fmaf_rd``, ``__ll2float_ru``, ``__double2float_rz`` and their siblings
(CUDA Math API, "Single Precision Intrinsics", "Double Precision Intrinsics" and "Type Casting Intrinsics"; HIP
declares the conversions under the same names) return the IEEE 754 result rounded down (``_rd``), up (``_ru``) or
toward zero (``_rz``). CrossGL, like the shading languages it targets, has only the round-to-nearest operations, so
the importers write each call as a call of a CrossGL function built here.

The functions use only integer arithmetic and bit casts. The operands are decoded from their bits, the exact result
is formed as an integer significand (a double-width product, a long-division quotient or a digit-by-digit square
root, with a sticky bit for any nonzero remainder), and :func:`round` packs it into the format in the requested
direction, with the IEEE 754 rules for overflow, subnormal results, signed zeros, infinities and NaN. No step relies
on the accuracy of a target's floating-point division or square root, on fused multiply-add, on contraction
settings or on whether the target flushes subnormal operands of its own arithmetic. NaN results carry CUDA's
canonical encodings (``0x7fffffff`` and ``0xfff8000000000000``).

``binary32`` works on 32-bit limbs, so the single-precision functions need no 64-bit integers, except the
conversions from 64-bit integers and from ``double``; ``binary64`` works on 64-bit limbs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from string import Template

__all__ = ["DIRECTED_ROUNDING_MODES", "DirectedRoundingLibrary"]

# The mode argument of the internal functions: rounding down, up and toward zero.
DIRECTED_ROUNDING_MODES = {"rd": 0, "ru": 1, "rz": 2}


@dataclass(frozen=True)
class _Format:
    name: str
    float_type: str
    limb: str
    width: int
    precision: int
    exponent_bits: int
    to_bits: str
    from_bits: str
    nan: int

    @property
    def bias(self):
        return (1 << (self.exponent_bits - 1)) - 1

    def literal(self, value):
        if self.width == 32:
            return f"{value:#x}u" if value > 9 else f"{value}u"
        high, low = value >> 32, value & 0xFFFFFFFF
        if not high:
            return f"u64({low:#x}u)" if low > 9 else f"u64({low}u)"
        if not low:
            return f"(u64({high:#x}u) << 32u)"
        return f"((u64({high:#x}u) << 32u) | u64({low:#x}u))"

    def substitutions(self):
        w, p = self.width, self.precision
        infinity = ((1 << self.exponent_bits) - 1) << (p - 1)
        sqrt_shift = p + 4 - (p & 1)
        values = {
            "T": self.float_type,
            "U": self.limb,
            "W": str(w),
            "W2": str(2 * w),
            "P": str(p),
            "P2": str(p + 2),
            "Pm1u": f"{p - 1}u",
            "Hu": f"{w // 2}u",
            "Wm2u": f"{w - 2}u",
            "zero": self.literal(0),
            "one": self.literal(1),
            "sign": self.literal(1 << (w - 1)),
            "abs": self.literal((1 << (w - 1)) - 1),
            "inf": self.literal(infinity),
            "max": self.literal(infinity - 1),
            "nan": self.literal(self.nan),
            "hidden": self.literal(1 << (p - 1)),
            "mantissa": self.literal((1 << (p - 1)) - 1),
            "halfmask": self.literal((1 << (w // 2)) - 1),
            "onebits": self.literal(self.bias << (p - 1)),
            "to_bits": self.to_bits,
            "from_bits": self.from_bits,
            "emin": str(1 - self.bias),
            "emax": str(self.bias),
            "biasm1": str(self.bias - 1),
            "subexp": str(1 - self.bias - (p - 1)),
            "biaspm1": str(self.bias + p - 1),
            "align": str(w - 1 - p),
            "pairexp": str(2 * w - 1 - p),
            "producttop": str(2 * p - 1 - w),
            "productshift": str(2 * w - 1 - 2 * p),
            "sqrtshift": str(sqrt_shift),
            "sqrtsplit": str(w - sqrt_shift),
        }
        return values


_F32 = _Format(
    "f32", "f32", "u32", 32, 24, 8, "floatBitsToUint", "uintBitsToFloat", 0x7FFFFFFF
)
_F64 = _Format(
    "f64", "f64", "u64", 64, 53, 11, "as_type<u64>", "as_type<f64>", 0xFFF8000000000000
)
_FORMATS = {"f32": _F32, "f64": _F64}

# Generic helpers: key -> (dependencies in the same format, CrossGL template). ``$msb`` names findMSB for 32-bit
# limbs and the ``msb`` helper for 64-bit ones.
_MSB64 = """i32 $self(u64 x) {
    u32 high = u32(x >> 32u);
    if (high != 0u) {
        return 32 + findMSB(high);
    }
    return findMSB(u32(x & u64(0xffffffffu)));
}"""

_UNPACK = """void $self($U magnitude, out $U m, out i32 e) {
    $U field = magnitude >> ${Pm1u};
    $U significand = magnitude & $mantissa;
    if (field == $zero) {
        i32 lz = $P - 1 - $msb(significand);
        m = significand << u32(lz);
        e = $subexp - lz;
    } else {
        m = significand | $hidden;
        e = i32(field) - $biaspm1;
    }
}"""

# Rounds (-1)^neg * (high:low) * 2^e. Adding the kept significand to the exponent field carries a rounded-up
# subnormal into the smallest normal binade and a rounded-up largest significand into the next binade; above the
# largest finite binade that carry yields the infinity encoding, which is right because it only happens when the
# direction rounds away from zero.
_ROUND = """$T $self(bool neg, i32 e, $U high, $U low, i32 mode) {
    $U hi = high;
    $U lo = low;
    i32 lead = $msb(lo);
    if (hi != $zero) {
        lead = $W + $msb(hi);
    }
    i32 shift = $W2 - 1 - lead;
    if (shift >= $W) {
        hi = lo << u32(shift - $W);
        lo = $zero;
    } else if (shift > 0) {
        hi = (hi << u32(shift)) | (lo >> u32($W - shift));
        lo = lo << u32(shift);
    }
    i32 exponent = e + lead;
    bool away = (mode == 0 && neg) || (mode == 1 && !neg);
    $U sign = neg ? $sign : $zero;
    $U overflow = sign | (away ? $inf : $max);
    if (exponent > $emax) {
        return $from_bits(overflow);
    }
    i32 keep = $P;
    if (exponent < $emin) {
        keep = $P - ($emin - exponent);
    }
    $U kept = $zero;
    bool inexact = true;
    if (keep > 0) {
        kept = hi >> u32($W - keep);
        inexact = (hi << u32(keep)) != $zero || lo != $zero;
    }
    if (inexact && away) {
        kept = kept + $one;
    }
    i32 field = exponent + $biasm1;
    if (exponent < $emin) {
        field = $emin + $biasm1;
    }
    $U bits = ($U(field) << ${Pm1u}) + kept;
    return $from_bits(sign | bits);
}"""

_JAM = """void $self($U hi, $U lo, i32 d, out $U rhi, out $U rlo) {
    rhi = $zero;
    if (d == 0) {
        rhi = hi;
        rlo = lo;
    } else if (d < $W) {
        rhi = hi >> u32(d);
        rlo = (lo >> u32(d)) | (hi << u32($W - d));
        if ((lo << u32($W - d)) != $zero) {
            rlo = rlo | $one;
        }
    } else if (d < $W2) {
        rlo = hi >> u32(d - $W);
        $U lost = lo;
        if (d > $W) {
            lost = lost | (hi << u32($W2 - d));
        }
        if (lost != $zero) {
            rlo = rlo | $one;
        }
    } else {
        rlo = $zero;
        if ((hi | lo) != $zero) {
            rlo = $one;
        }
    }
}"""

_SUM = """$T $self(bool na, i32 ea, $U ha, $U la, bool nb, i32 eb, $U hb, $U lb, i32 mode) {
    bool swap = eb > ea || (eb == ea && (hb > ha || (hb == ha && lb > la)));
    bool neg = na;
    i32 e = ea;
    i32 d = ea - eb;
    $U hi = ha;
    $U lo = la;
    $U smaller_hi = hb;
    $U smaller_lo = lb;
    if (swap) {
        neg = nb;
        e = eb;
        d = eb - ea;
        hi = hb;
        lo = lb;
        smaller_hi = ha;
        smaller_lo = la;
    }
    $U sh = $zero;
    $U sl = $zero;
    $jam(smaller_hi, smaller_lo, d, sh, sl);
    if (na == nb) {
        $U sum_lo = lo + sl;
        $U carry = $zero;
        if (sum_lo < lo) {
            carry = $one;
        }
        return $round(neg, e, hi + sh + carry, sum_lo, mode);
    }
    $U borrow = $zero;
    if (lo < sl) {
        borrow = $one;
    }
    $U difference_lo = lo - sl;
    $U difference_hi = hi - sh - borrow;
    if (difference_hi == $zero && difference_lo == $zero) {
        if (mode == 0) {
            return $from_bits($sign);
        }
        return $from_bits($zero);
    }
    return $round(neg, e, difference_hi, difference_lo, mode);
}"""

_MULWIDE = """void $self($U a, $U b, out $U hi, out $U lo) {
    $U a0 = a & $halfmask;
    $U a1 = a >> ${Hu};
    $U b0 = b & $halfmask;
    $U b1 = b >> ${Hu};
    $U p00 = a0 * b0;
    $U middle = a0 * b1 + (p00 >> ${Hu});
    $U middle2 = a1 * b0 + (middle & $halfmask);
    hi = a1 * b1 + (middle >> ${Hu}) + (middle2 >> ${Hu});
    lo = (middle2 << ${Hu}) | (p00 & $halfmask);
}"""

_ADD = """$T $self($U ab, $U bb, i32 mode) {
    $U aa = ab & $abs;
    $U ba = bb & $abs;
    bool na = ab != aa;
    bool nb = bb != ba;
    if (aa > $inf || ba > $inf) {
        return $from_bits($nan);
    }
    if (aa == $inf) {
        if (ba == $inf && na != nb) {
            return $from_bits($nan);
        }
        return $from_bits(ab);
    }
    if (ba == $inf) {
        return $from_bits(bb);
    }
    if (ba == $zero) {
        if (aa == $zero) {
            bool zero_neg = na && nb;
            if (mode == 0) {
                zero_neg = na || nb;
            }
            return $from_bits(zero_neg ? $sign : $zero);
        }
        return $from_bits(ab);
    }
    if (aa == $zero) {
        return $from_bits(bb);
    }
    $U ma = $zero;
    i32 ea = 0;
    $unpack(aa, ma, ea);
    $U mb = $zero;
    i32 eb = 0;
    $unpack(ba, mb, eb);
    return $sum(na, ea - $pairexp, ma << ${align}u, $zero, nb, eb - $pairexp, mb << ${align}u, $zero, mode);
}"""

_MUL = """$T $self($U ab, $U bb, i32 mode) {
    $U aa = ab & $abs;
    $U ba = bb & $abs;
    $U sign = (ab ^ bb) & $sign;
    if (aa > $inf || ba > $inf) {
        return $from_bits($nan);
    }
    if (aa == $inf || ba == $inf) {
        if (aa == $zero || ba == $zero) {
            return $from_bits($nan);
        }
        return $from_bits(sign | $inf);
    }
    if (aa == $zero || ba == $zero) {
        return $from_bits(sign);
    }
    $U ma = $zero;
    i32 ea = 0;
    $unpack(aa, ma, ea);
    $U mb = $zero;
    i32 eb = 0;
    $unpack(ba, mb, eb);
    $U hi = $zero;
    $U lo = $zero;
    $mulwide(ma, mb, hi, lo);
    return $round(sign != $zero, ea + eb, hi, lo, mode);
}"""

_FMA = """$T $self($U ab, $U bb, $U cb, i32 mode) {
    $U aa = ab & $abs;
    $U ba = bb & $abs;
    $U ca = cb & $abs;
    $U product_sign = (ab ^ bb) & $sign;
    bool np = product_sign != $zero;
    bool nc = cb != ca;
    if (aa > $inf || ba > $inf || ca > $inf) {
        return $from_bits($nan);
    }
    if (aa == $inf || ba == $inf) {
        if (aa == $zero || ba == $zero || (ca == $inf && np != nc)) {
            return $from_bits($nan);
        }
        return $from_bits(product_sign | $inf);
    }
    if (ca == $inf) {
        return $from_bits(cb);
    }
    if (aa == $zero || ba == $zero) {
        if (ca == $zero) {
            bool zero_neg = np && nc;
            if (mode == 0) {
                zero_neg = np || nc;
            }
            return $from_bits(zero_neg ? $sign : $zero);
        }
        return $from_bits(cb);
    }
    $U ma = $zero;
    i32 ea = 0;
    $unpack(aa, ma, ea);
    $U mb = $zero;
    i32 eb = 0;
    $unpack(ba, mb, eb);
    $U ph = $zero;
    $U pl = $zero;
    $mulwide(ma, mb, ph, pl);
    if (ca == $zero) {
        return $round(np, ea + eb, ph, pl, mode);
    }
    i32 s = $productshift + 1;
    if ((ph >> ${producttop}u) != $zero) {
        s = $productshift;
    }
    $U hi = (ph << u32(s)) | (pl >> u32($W - s));
    $U lo = pl << u32(s);
    $U mc = $zero;
    i32 ec = 0;
    $unpack(ca, mc, ec);
    return $sum(np, ea + eb - s, hi, lo, nc, ec - $pairexp, mc << ${align}u, $zero, mode);
}"""

_DIV = """$T $self($U ab, $U bb, i32 mode) {
    $U aa = ab & $abs;
    $U ba = bb & $abs;
    $U sign = (ab ^ bb) & $sign;
    if (aa > $inf || ba > $inf || (aa == $inf && ba == $inf) || (aa == $zero && ba == $zero)) {
        return $from_bits($nan);
    }
    if (aa == $inf || ba == $zero) {
        return $from_bits(sign | $inf);
    }
    if (ba == $inf || aa == $zero) {
        return $from_bits(sign);
    }
    $U ma = $zero;
    i32 ea = 0;
    $unpack(aa, ma, ea);
    $U mb = $zero;
    i32 eb = 0;
    $unpack(ba, mb, eb);
    $U remainder = ma;
    i32 e = ea - eb;
    if (remainder < mb) {
        remainder = remainder << 1u;
        e = e - 1;
    }
    $U quotient = $zero;
    for (i32 i = 0; i < $P2; i++) {
        quotient = quotient << 1u;
        if (remainder >= mb) {
            remainder = remainder - mb;
            quotient = quotient | $one;
        }
        remainder = remainder << 1u;
    }
    $U significand = quotient << 1u;
    if (remainder != $zero) {
        significand = significand | $one;
    }
    return $round(sign != $zero, e - $P2, $zero, significand, mode);
}"""

_SQRT = """$T $self($U xb, i32 mode) {
    $U xa = xb & $abs;
    if (xa == $zero || xb == $inf) {
        return $from_bits(xb);
    }
    if (xa > $inf || xb != xa) {
        return $from_bits($nan);
    }
    $U m = $zero;
    i32 e = 0;
    $unpack(xa, m, e);
    if ((e & 1) != 0) {
        m = m << 1u;
        e = e - 1;
    }
    $U radicand_hi = m >> ${sqrtsplit}u;
    $U radicand_lo = m << ${sqrtshift}u;
    $U remainder = $zero;
    $U root = $zero;
    for (i32 i = 0; i < $W; i++) {
        remainder = (remainder << 2u) | (radicand_hi >> ${Wm2u});
        radicand_hi = (radicand_hi << 2u) | (radicand_lo >> ${Wm2u});
        radicand_lo = radicand_lo << 2u;
        $U trial = (root << 2u) | $one;
        root = root << 1u;
        if (remainder >= trial) {
            remainder = remainder - trial;
            root = root | $one;
        }
    }
    $U significand = root << 1u;
    if (remainder != $zero) {
        significand = significand | $one;
    }
    return $round(false, (e - $sqrtshift) / 2 - 1, $zero, significand, mode);
}"""

# Rounds the integer (-1)^neg * magnitude.
_FROM_U64 = {
    "f32": (
        """f32 $self(bool neg, u64 magnitude, i32 mode) {
    if (magnitude == u64(0u)) {
        return uintBitsToFloat(0u);
    }
    return $round(neg, 0, u32(magnitude >> 32u), u32(magnitude & u64(0xffffffffu)), mode);
}"""
    ),
    "f64": (
        """f64 $self(bool neg, u64 magnitude, i32 mode) {
    if (magnitude == u64(0u)) {
        return as_type<f64>(u64(0u));
    }
    return $round(neg, 0, u64(0u), magnitude, mode);
}"""
    ),
}

_FROM_U32 = """f32 $self(bool neg, u32 magnitude, i32 mode) {
    if (magnitude == 0u) {
        return uintBitsToFloat(0u);
    }
    return $round(neg, 0, 0u, magnitude, mode);
}"""

_NARROW = """f32 $self(f64 x, i32 mode) {
    u64 xb = as_type<u64>(x);
    u64 xa = xb & ((u64(0x7fffffffu) << 32u) | u64(0xffffffffu));
    bool neg = xb != xa;
    if (xa > (u64(0x7ff00000u) << 32u)) {
        return uintBitsToFloat(0x7fffffffu);
    }
    if (xa == (u64(0x7ff00000u) << 32u)) {
        return uintBitsToFloat(neg ? 0xff800000u : 0x7f800000u);
    }
    if (xa == u64(0u)) {
        return uintBitsToFloat(neg ? 0x80000000u : 0u);
    }
    u64 m = u64(0u);
    i32 e = 0;
    $unpack(xa, m, e);
    u64 s = m << 11u;
    return $round(neg, e - 11, u32(s >> 32u), u32(s & u64(0xffffffffu)), mode);
}"""

_GENERIC = (
    ("unpack", ("msb",), _UNPACK),
    ("round", ("msb",), _ROUND),
    ("jam", (), _JAM),
    ("sum", ("jam", "round"), _SUM),
    ("mulwide", (), _MULWIDE),
    ("add", ("unpack", "sum"), _ADD),
    ("mul", ("unpack", "mulwide", "round"), _MUL),
    ("fma", ("unpack", "mulwide", "round", "sum"), _FMA),
    ("div", ("unpack", "round"), _DIV),
    ("sqrt", ("unpack", "round"), _SQRT),
)

# intrinsic stem -> (format, number of operands, internal function, its arguments before the mode). ``$a``, ``$b``
# and ``$c`` are the operands' bits.
_ARITHMETIC = {
    "fadd": ("f32", 2, "add", "$a, $b"),
    "fsub": ("f32", 2, "add", "$a, $b ^ $sign"),
    "fmul": ("f32", 2, "mul", "$a, $b"),
    "fdiv": ("f32", 2, "div", "$a, $b"),
    "frcp": ("f32", 1, "div", "$onebits, $a"),
    "fsqrt": ("f32", 1, "sqrt", "$a"),
    "fmaf": ("f32", 3, "fma", "$a, $b, $c"),
    "fmaf_ieee": ("f32", 3, "fma", "$a, $b, $c"),
    "dadd": ("f64", 2, "add", "$a, $b"),
    "dsub": ("f64", 2, "add", "$a, $b ^ $sign"),
    "dmul": ("f64", 2, "mul", "$a, $b"),
    "ddiv": ("f64", 2, "div", "$a, $b"),
    "drcp": ("f64", 1, "div", "$onebits, $a"),
    "dsqrt": ("f64", 1, "sqrt", "$a"),
    "fma": ("f64", 3, "fma", "$a, $b, $c"),
}

# intrinsic stem -> (operand type, result type, (internal function, its format))
_CONVERSIONS = {
    "int2float": ("i32", "f32", ("from_u32", "f32")),
    "uint2float": ("u32", "f32", ("from_u32", "f32")),
    "ll2float": ("i64", "f32", ("from_u64", "f32")),
    "ull2float": ("u64", "f32", ("from_u64", "f32")),
    "ll2double": ("i64", "f64", ("from_u64", "f64")),
    "ull2double": ("u64", "f64", ("from_u64", "f64")),
    "double2float": ("f64", "f32", ("narrow", "f32")),
}

_INTEGER_MAGNITUDE = {
    "i32": ("x < 0", "~u32(x) + 1u", "u32(x)"),
    "u32": (None, None, "x"),
    "i64": ("x < i64(0)", "~u64(x) + u64(1u)", "u64(x)"),
    "u64": (None, None, "x"),
}

_IDENTIFIER = re.compile(r"(?<![0-9A-Za-z_])[A-Za-z_][A-Za-z0-9_]*")
# Words of the function texts that are not parameter or local names.
_RESERVED_WORDS = frozenset(
    "bool void i32 u32 i64 u64 f32 f64 if else for return out true false findMSB floatBitsToUint "
    "uintBitsToFloat as_type".split()
)

_BASE_NAMES = {
    "msb": "directed_msb_u64",
    "from_u32": "directed_from_u32_f32",
    "from_u64": "directed_from_u64_{fmt}",
    "narrow": "directed_narrow_f64_f32",
}


def _split_intrinsic(name):
    """``('fadd', 'rd')`` for ``__fadd_rd``; None for a name that is not a directed-rounding intrinsic built here."""
    if not isinstance(name, str) or not name.startswith("__") or len(name) < 6:
        return None
    stem, mode = name[2:-3], name[-2:]
    if name[-3] != "_" or mode not in DIRECTED_ROUNDING_MODES:
        return None
    if stem in _ARITHMETIC or stem in _CONVERSIONS:
        return stem, mode
    return None


class DirectedRoundingLibrary:
    """The directed-rounding functions one CrossGL program calls, named apart from the program's identifiers."""

    def __init__(self, used_names=()):
        self.used_names = set(used_names)
        self.names = {}
        self.wrappers = {}

    def handles(self, intrinsic):
        return _split_intrinsic(str(intrinsic or "").lstrip(":")) is not None

    def arity(self, intrinsic):
        stem, _mode = _split_intrinsic(str(intrinsic).lstrip(":"))
        return _ARITHMETIC[stem][1] if stem in _ARITHMETIC else 1

    def call(self, intrinsic, args):
        """The CrossGL call of ``intrinsic`` on the rendered ``args``, converted to its parameter types as C++ does.

        The caller has checked that ``args`` has :meth:`arity` items.
        """
        intrinsic = str(intrinsic).lstrip(":")
        stem, mode = _split_intrinsic(intrinsic)
        if stem in _ARITHMETIC:
            parameter_type = _ARITHMETIC[stem][0]
            parameters = [parameter_type] * _ARITHMETIC[stem][1]
        else:
            parameters = [_CONVERSIONS[stem][0]]
        wrapper = self.wrappers.get(intrinsic)
        if wrapper is None:
            wrapper = self.wrappers[intrinsic] = self.name(intrinsic[2:])
        converted = ", ".join(
            f"{t}({a})" for t, a in zip(parameters, args, strict=True)
        )
        return f"{wrapper}({converted})"

    def name(self, base):
        if base in self.names:
            return self.names[base]
        candidate = base
        while candidate in self.used_names:
            candidate = f"{candidate}_"
        self.used_names.add(candidate)
        self.names[base] = candidate
        return candidate

    def helper_name(self, key, fmt):
        if key in _BASE_NAMES:
            return self.name(_BASE_NAMES[key].format(fmt=fmt))
        return self.name(f"directed_{key}_{fmt}")

    def render(self):
        """CrossGL text of every function the requested calls need, callees first, in a fixed order."""
        needed = []
        wrapper_texts = []
        for intrinsic in sorted(self.wrappers):
            text, dependencies = self.wrapper_text(intrinsic)
            wrapper_texts.append(text)
            needed.extend(dependencies)
        closure = set()
        pending = list(needed)
        while pending:
            helper = pending.pop()
            if helper in closure:
                continue
            closure.add(helper)
            pending.extend(self.dependencies(*helper))
        texts = [self.helper_text(*helper) for helper in self.ordered(closure)]
        return self.rename_locals("\n\n".join(texts + wrapper_texts))

    def rename_locals(self, text):
        """Rename the parameters and locals that the program also uses as names, such as a kernel's buffer ``a``.

        A target can reject a local that shadows a global resource, so none of them may share a program name.
        """
        functions = set(self.names.values())
        words = sorted(set(_IDENTIFIER.findall(text)) - _RESERVED_WORDS - functions)
        taken = self.used_names | set(words)
        renames = {}
        for word in words:
            if word in self.used_names:
                candidate = f"{word}_"
                while candidate in taken:
                    candidate = f"{candidate}_"
                taken.add(candidate)
                renames[word] = candidate
        if not renames:
            return text
        renamed = re.compile(
            r"(?<![0-9A-Za-z_])("
            + "|".join(
                re.escape(word) for word in sorted(renames, key=lambda w: (-len(w), w))
            )
            + r")(?![0-9A-Za-z_])"
        )
        return renamed.sub(lambda match: renames[match.group(1)], text)

    def ordered(self, closure):
        order = [("msb", "f64")]
        for fmt in ("f32", "f64"):
            order.extend((key, fmt) for key, _deps, _text in _GENERIC)
        order.extend(
            [
                ("from_u32", "f32"),
                ("from_u64", "f32"),
                ("from_u64", "f64"),
                ("narrow", "f32"),
            ]
        )
        return [helper for helper in order if helper in closure]

    def dependencies(self, key, fmt):
        if key == "msb":
            return []
        if key in ("from_u32", "from_u64"):
            return [("round", fmt)]
        if key == "narrow":
            return [("unpack", "f64"), ("round", "f32")]
        deps = {k: d for k, d, _t in _GENERIC}[key]
        return [(dep, fmt) for dep in deps if not (dep == "msb" and fmt == "f32")]

    def substitutions(self, fmt):
        values = _FORMATS[fmt].substitutions()
        for key, _deps, _text in _GENERIC:
            values[key] = self.helper_name(key, fmt)
        values["msb"] = "findMSB" if fmt == "f32" else self.helper_name("msb", "f64")
        return values

    def helper_text(self, key, fmt):
        if key == "msb":
            return Template(_MSB64).substitute(self=self.helper_name(key, fmt))
        if key == "narrow":
            values = {
                "unpack": self.helper_name("unpack", "f64"),
                "round": self.helper_name("round", "f32"),
            }
            template = _NARROW
        elif key in ("from_u32", "from_u64"):
            values = {"round": self.helper_name("round", fmt)}
            template = _FROM_U32 if key == "from_u32" else _FROM_U64[fmt]
        else:
            values = self.substitutions(fmt)
            template = {k: t for k, _d, t in _GENERIC}[key]
        values["self"] = self.helper_name(key, fmt)
        return Template(template).substitute(values)

    def wrapper_text(self, intrinsic):
        stem, mode = _split_intrinsic(intrinsic)
        name = self.wrappers[intrinsic]
        mode_value = DIRECTED_ROUNDING_MODES[mode]
        if stem in _ARITHMETIC:
            fmt, arity, key, arguments = _ARITHMETIC[stem]
            values = self.substitutions(fmt)
            parameter_names = "abc"[:arity]
            for parameter in parameter_names:
                values[parameter] = f"{values['to_bits']}({parameter})"
            body = Template(arguments).substitute(values)
            parameters = ", ".join(f"{fmt} {p}" for p in parameter_names)
            text = (
                f"{fmt} {name}({parameters}) {{\n"
                f"    return {values[key]}({body}, {mode_value});\n"
                "}"
            )
            return text, [(key, fmt)]
        parameter_type, result_type, helper = _CONVERSIONS[stem]
        helper_name = self.helper_name(*helper)
        if helper[0] == "narrow":
            body = f"    return {helper_name}(x, {mode_value});\n"
        else:
            negative, negated, magnitude = _INTEGER_MAGNITUDE[parameter_type]
            if negative is None:
                body = f"    return {helper_name}(false, {magnitude}, {mode_value});\n"
            else:
                magnitude_type = "u32" if parameter_type == "i32" else "u64"
                body = (
                    f"    bool neg = {negative};\n"
                    f"    {magnitude_type} magnitude = {magnitude};\n"
                    "    if (neg) {\n"
                    f"        magnitude = {negated};\n"
                    "    }\n"
                    f"    return {helper_name}(neg, magnitude, {mode_value});\n"
                )
        text = f"{result_type} {name}({parameter_type} x) {{\n{body}}}"
        return text, [helper]

"""Reserved-member-name, reserved-type-name and reserved-global-name probes: a struct's members, a struct type, and a
program-scope constant and constant array named with words that the probe's own language allows and others reserve.

reserved-identifier names a helper, its parameter and locals; these probes name what a program declares inside a
struct or at program scope, which a translation has to rename at the declaration and at every use. Each language
takes its words from its reserved-identifier probe (:mod:`.lexical`): ``device``, ``kernel`` and ``thread`` are Metal
keywords; ``fn``, ``let``, ``loop``, ``match``, ``mod`` and ``type`` are Rust keywords that WGSL reserves too; ``var``
belongs to WGSL and Mojo, ``sample`` to GLSL and HLSL, ``half`` to HLSL and Metal, and ``filter``, ``input`` and
``output`` to GLSL or WGSL. CrossGL itself accepts few of these words as a struct name, so its probe names the struct
``type``. Every value the probes compute is a small whole number or a power-of-two multiple of the input, so each
result is exact.
"""

from __future__ import annotations

from .base import Template
from .statements import c_text

NAMES = ("reserved-member-name", "reserved-type-name", "reserved-global-name")

# The member probe's three members, its type probe's struct name and its global probe's constant and array, by
# language, in the order members, type, globals.
_CROSSGL = ("sample", "half", "fn", "type", "device", "output")
_FAMILY = ("device", "kernel", "thread", "match", "loop", "type")
_METAL = ("sample", "filter", "output", "let", "var", "input")
_C = ("fn", "input", "output", "let", "var", "mod")
_WGSL_MOJO = ("device", "kernel", "sample", "thread", "half", "output")
_RUST = ("device", "kernel", "sample", "var", "half", "output")

_MEMBER_BODY = (
    "Rec r;\nr.{0} = float(iin[i] & 7);\nr.{1} = iin[i] & 3;\nr.{2} = r.{0} * 2.0;\n"
    "r.{2} += float(r.{1} * 3 + 1) * 32.0;\noutp[i] = total(r);"
)
_TYPE_BODY = "{0} t = make(float(iin[i] & 7));\nt.b += iin[i] & 3;\noutp[i] = weigh(t);"
_GLOBAL_BODY = (
    "int k = iin[i] & 1;\noutp[i] = fin[i] * {0} + {1}[k] * 32.0 + {1}[2 - k] * 1024.0;"
)
_TABLE = "{0.5, 1.0, 4.0}"


def _record(name: str, fields: str, typedef: bool) -> str:
    if typedef:
        return f"typedef struct {{\n{fields}\n}} {name};"
    return f"struct {name} {{\n{fields}\n}};"


def _family_parts(
    names: tuple[str, ...], typedef: bool = False, device: str = ""
) -> dict[str, tuple[str, str, str]]:
    """Each probe's body, struct and helpers in C-family syntax, before any language's conversions."""
    a, b, c, kind, scale, table = names
    return {
        "reserved-member-name": (
            _MEMBER_BODY.format(a, b, c),
            _record("Rec", f"    float {a};\n    int {b};\n    float {c};", typedef),
            f"{device}float total(Rec r) {{\n    return r.{a} + r.{c};\n}}",
        ),
        "reserved-type-name": (
            _TYPE_BODY.format(kind),
            _record(kind, "    float a;\n    int b;", typedef),
            f"{device}{kind} make(float x) {{\n    {kind} t;\n    t.a = x * 2.0;\n    t.b = 3;\n    return t;\n}}\n\n"
            f"{device}float weigh({kind} t) {{\n    return t.a + float(t.b) * 32.0;\n}}",
        ),
        "reserved-global-name": (_GLOBAL_BODY.format(scale, table), "", ""),
    }


def _globals(names: tuple[str, ...], constant: str, table: str) -> str:
    return (
        f"{constant}float {names[4]} = 2.0;\n{constant}float {names[5]}[3] = {table};"
    )


def _family(
    names: tuple[str, ...], constant: str, table: str = _TABLE
) -> dict[str, Template]:
    """The GLSL, HLSL and Metal probes: ``constant`` qualifies each program-scope constant."""
    probes = {
        name: Template(body, f"{record}\n{helpers}")
        for name, (body, record, helpers) in _family_parts(names).items()
    }
    probes["reserved-global-name"] = Template(
        probes["reserved-global-name"].body, _globals(names, constant, table)
    )
    return probes


GLSL = _family(_FAMILY, "const ")
WEBGL = _family(_FAMILY, "const ", "float[3](0.5, 1.0, 4.0)")
FAMILY = _family(_FAMILY, "static const ")
METAL = _family(_METAL, "constant ")

# CrossGL declares structs and constants ahead of the shader and helpers inside it.
CROSSGL = {
    name: Template(body, helpers, prelude=record)
    for name, (body, record, helpers) in _family_parts(_CROSSGL).items()
}
CROSSGL["reserved-global-name"] = Template(
    CROSSGL["reserved-global-name"].body, prelude=_globals(_CROSSGL, "const ", _TABLE)
)


def _c(unsigned: str, constant: str, typedef: bool, device: str) -> dict[str, Template]:
    """The probes CUDA, HIP and OpenCL C share, with C casts in place of conversions."""
    probes = {
        name: Template(c_text(body, unsigned), c_text(f"{record}\n{helpers}", unsigned))
        for name, (body, record, helpers) in _family_parts(_C, typedef, device).items()
    }
    probes["reserved-global-name"] = Template(
        probes["reserved-global-name"].body,
        c_text(_globals(_C, constant, _TABLE), unsigned),
    )
    return probes


CUDA = _c("unsigned int", "__constant__ ", False, "__device__ ")
OPENCL = _c("uint", "__constant ", True, "")


def _wgsl() -> dict[str, Template]:
    a, b, c, kind, scale, array = _WGSL_MOJO
    return {
        "reserved-member-name": Template(
            f"var r: Rec;\nr.{a} = f32(iin[i] & 7);\nr.{b} = iin[i] & 3;\nr.{c} = r.{a} * 2.0;\n"
            f"r.{c} += f32(r.{b} * 3 + 1) * 32.0;\noutp[i] = total(r);",
            f"struct Rec {{\n    {a}: f32,\n    {b}: i32,\n    {c}: f32,\n}}\n\n"
            f"fn total(r: Rec) -> f32 {{\n    return r.{a} + r.{c};\n}}",
        ),
        "reserved-type-name": Template(
            "var t = make(f32(iin[i] & 7));\nt.b += iin[i] & 3;\noutp[i] = weigh(t);",
            f"struct {kind} {{\n    a: f32,\n    b: i32,\n}}\n\n"
            f"fn make(x: f32) -> {kind} {{\n    var t: {kind};\n    t.a = x * 2.0;\n    t.b = 3;\n    return t;\n}}\n\n"
            f"fn weigh(t: {kind}) -> f32 {{\n    return t.a + f32(t.b) * 32.0;\n}}",
        ),
        "reserved-global-name": Template(
            f"let k = iin[i] & 1;\noutp[i] = fin[i] * {scale} + {array}[k] * 32.0 + {array}[2 - k] * 1024.0;",
            f"const {scale}: f32 = 2.0;\nconst {array} = array<f32, 3>(0.5, 1.0, 4.0);",
        ),
    }


WGSL = _wgsl()

_RUST_DERIVE = "#[derive(Clone, Copy, Default)]\n#[repr(C)]\n"


def _rust() -> dict[str, Template]:
    a, b, c, kind, scale, array = _RUST
    return {
        "reserved-member-name": Template(
            f"let mut r = Rec::default();\nr.{a} = (iin[i] & 7) as f32;\nr.{b} = iin[i] & 3;\nr.{c} = r.{a} * 2.0;\n"
            f"r.{c} += (r.{b} * 3 + 1) as f32 * 32.0;\noutp[i] = total(r);",
            f"{_RUST_DERIVE}pub struct Rec {{\n    {a}: f32,\n    {b}: i32,\n    {c}: f32,\n}}\n\n"
            f"fn total(r: Rec) -> f32 {{\n    r.{a} + r.{c}\n}}",
        ),
        "reserved-type-name": Template(
            "let mut t = make((iin[i] & 7) as f32);\nt.b += iin[i] & 3;\noutp[i] = weigh(t);",
            f"{_RUST_DERIVE}#[allow(non_camel_case_types)]\npub struct {kind} {{\n    a: f32,\n    b: i32,\n}}\n\n"
            f"fn make(x: f32) -> {kind} {{\n    {kind} {{ a: x * 2.0, b: 3 }}\n}}\n\n"
            f"fn weigh(t: {kind}) -> f32 {{\n    t.a + t.b as f32 * 32.0\n}}",
        ),
        "reserved-global-name": Template(
            f"let k = (iin[i] & 1) as usize;\n"
            f"outp[i] = fin[i] * {scale} + {array}[k] * 32.0 + {array}[2 - k] * 1024.0;",
            f"#[allow(non_upper_case_globals)]\nconst {scale}: f32 = 2.0;\n"
            f"#[allow(non_upper_case_globals)]\nconst {array}: [f32; 3] = [0.5, 1.0, 4.0];",
        ),
    }


RUST = _rust()


def _mojo() -> dict[str, Template]:
    a, b, c, kind, scale, array = _WGSL_MOJO
    return {
        "reserved-member-name": Template(
            f"var r = Rec(0.0, 0, 0.0)\nr.{a} = Float32(iin[i] & 7)\nr.{b} = iin[i] & 3\nr.{c} = r.{a} * 2.0\n"
            f"r.{c} += Float32(r.{b} * 3 + 1) * 32.0\noutp[i] = total(r)",
            f"@fieldwise_init\nstruct Rec(Copyable, Movable):\n    var {a}: Float32\n    var {b}: Int32\n"
            f"    var {c}: Float32\n\n\ndef total(r: Rec) -> Float32:\n    return r.{a} + r.{c}",
        ),
        "reserved-type-name": Template(
            "var t = make(Float32(iin[i] & 7))\nt.b += iin[i] & 3\noutp[i] = weigh(t)",
            f"@fieldwise_init\nstruct {kind}(Copyable, Movable):\n    var a: Float32\n    var b: Int32\n\n\n"
            f"def make(x: Float32) -> {kind}:\n    return {kind}(x * 2.0, 3)\n\n\n"
            f"def weigh(t: {kind}) -> Float32:\n    return t.a + Float32(t.b) * 32.0",
        ),
        "reserved-global-name": Template(
            f"var k = Int(iin[i] & 1)\nvar table = materialize[{array}]()\n"
            f"outp[i] = fin[i] * {scale} + table[k] * 32.0 + table[2 - k] * 1024.0",
            f"comptime {scale}: Float32 = 2.0\ncomptime {array}: Array[Float32, 3] = [0.5, 1.0, 4.0]",
        ),
    }


MOJO = _mojo()

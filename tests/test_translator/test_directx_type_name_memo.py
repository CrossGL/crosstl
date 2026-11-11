"""The DirectX bare type-name memo reuses a spelling only while the state it read holds."""

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import (
    DirectXNative16BitUnsupportedError,
    HLSLCodeGen,
)


def test_memo_replays_the_native_16_bit_requirement_on_every_hit():
    generator = HLSLCodeGen()
    assert generator.map_type("half") == "float16_t"
    assert generator.requires_hlsl_native_16_bit_types
    generator.requires_hlsl_native_16_bit_types = False
    assert generator.map_type("half") == "float16_t"
    assert generator.requires_hlsl_native_16_bit_types


def test_memo_raises_no_requirement_for_32_bit_names():
    generator = HLSLCodeGen()
    for _ in range(3):
        assert generator.map_type("int") == "int"
        assert generator.map_type("vec3") == "float3"
    assert not generator.requires_hlsl_native_16_bit_types


def _alias(generator):
    generator.hlsl_type_aliases = {"Payload": "vec2"}


def _type_table(generator):
    generator.type_mapping["Payload"] = "float3"


def _enum(generator):
    generator.enum_type_names = {"Payload"}


def _enum_struct(generator):
    generator.enum_struct_type_names = {"Payload"}


@pytest.mark.parametrize(
    "declare,expected",
    [
        (_alias, "float2"),
        (_type_table, "float3"),
        (_enum, "int"),
        (_enum_struct, "Payload"),
    ],
)
def test_memo_follows_declarations_made_after_the_first_lookup(declare, expected):
    generator = HLSLCodeGen()
    assert generator.map_type("Payload") == "Payload"
    declare(generator)
    assert generator.map_type("Payload") == expected


def test_memo_follows_float16_widening():
    generator = HLSLCodeGen()
    assert generator.map_type("half3") == "float16_t3"
    generator.widen_native_float16 = True
    assert generator.map_type("half3") == "float3"


def test_memo_follows_the_target_profile():
    generator = HLSLCodeGen()
    assert generator.map_type("half") == "float16_t"
    generator.target_profile = "dx11"
    with pytest.raises(DirectXNative16BitUnsupportedError):
        generator.map_type("half")


def _program(statements):
    body = "\n".join(
        f"        int v{index} = (a + {index}) * b - (a >> 1);"
        for index in range(statements)
    )
    total = " + ".join(f"v{index}" for index in range(statements))
    return f"""shader S {{
    int f(int a, int b) {{
{body}
        return {total};
    }}
    half g(half x, half y) {{ return x * y + half(1.0); }}
}}"""


def _bare_computations(monkeypatch, source):
    computed = []
    map_type = HLSLCodeGen.map_type

    def counting(self, vtype, *, memoized=True):
        if not memoized and vtype.__class__ is str:
            computed.append(vtype)
        return map_type(self, vtype, memoized=memoized)

    with monkeypatch.context() as patch:
        patch.setattr(HLSLCodeGen, "map_type", counting)
        HLSLCodeGen().generate(parse(source))
    return computed


def test_bare_type_work_does_not_grow_with_program_size(monkeypatch):
    small = _bare_computations(monkeypatch, _program(4))
    large = _bare_computations(monkeypatch, _program(64))
    assert large.count("int") == small.count("int")
    assert large.count("int") <= 2


@pytest.mark.parametrize(
    "source",
    [
        _program(8),
        """shader S {
    enum Mode { A, B }
    struct Pair { half2 h; int i; };
    half2 scale(Pair p, Mode m) {
        half2 r = p.h * half(2.0);
        if (m == Mode::A) { r = r + half2(p.i); }
        return r;
    }
}""",
    ],
)
def test_memo_output_matches_uncached_mapping(monkeypatch, source):
    memoized = HLSLCodeGen()
    expected_output = memoized.generate(parse(source))
    map_type = HLSLCodeGen.map_type
    monkeypatch.setattr(
        HLSLCodeGen,
        "map_type",
        lambda self, vtype, memoized=True: map_type(self, vtype, memoized=False),
    )
    uncached = HLSLCodeGen()
    assert uncached.generate(parse(source)) == expected_output
    assert (
        uncached.requires_hlsl_native_16_bit_types
        == memoized.requires_hlsl_native_16_bit_types
    )

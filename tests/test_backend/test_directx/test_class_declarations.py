"""HLSL input imports ``class`` definitions as structs, and leaves the other uses of the keyword alone."""

import textwrap

import pytest

import crosstl
from crosstl.backend.DirectX.class_declarations import class_definitions_as_structs
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from tests.test_backend.test_directx.test_codegen import generate_crossgl, parse_crossgl

CLASS_WITH_METHODS = textwrap.dedent("""
    interface Scaler {
        uint scale(uint v);
    };

    class Doubler : Scaler {
        uint bias;
        uint scale(uint v) { return v * 2 + bias; }
        static uint twice(uint v) { return v * 2; }
    };

    RWStructuredBuffer<uint> output : register(u0);

    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {
        class Pair { uint first; uint second; };
        Pair pair = { 3, 4 };
        Doubler doubler;
        doubler.bias = 1;
        output[id.x] = doubler.scale(id.x) + Doubler::twice(pair.second);
    }
    """)


def _tokens(code):
    return HLSLLexer(code).tokenize()


def test_class_definitions_become_struct_definitions():
    output = generate_crossgl(CLASS_WITH_METHODS)

    assert "struct Doubler {" in output
    assert "uint bias;" in output
    assert "uint Doubler_scale(in Doubler __this, uint v) {" in output
    assert "uint Doubler_twice(uint v) {" in output
    assert "struct Pair {" in output
    assert "Doubler_scale(doubler, id.x) + Doubler_twice(pair.second)" in output
    # The interface declares only prototypes, so nothing of it remains.
    assert "Scaler" not in output
    parse_crossgl(output)


@pytest.mark.parametrize("backend", ["metal", "opengl", "vulkan", "directx"])
def test_a_class_used_through_an_instance_translates(tmp_path, backend):
    source = tmp_path / "kernel.hlsl"
    source.write_text(CLASS_WITH_METHODS, encoding="utf-8")

    output = crosstl.translate(str(source), backend=backend)

    assert "Doubler" in output


@pytest.mark.parametrize(
    "code",
    [
        "template <class T> T identity(T v) { return v; }",
        "template <class T, class U = float> struct Box { T a; U b; };",
        "enum class Mode : uint { A, B };",
        "enum class Mode { A, B };",
    ],
)
def test_template_parameters_and_scoped_enumerations_keep_class(code):
    tokens = _tokens(code)

    assert class_definitions_as_structs(tokens) is tokens


def test_class_forward_declarations_and_specializations_become_structs():
    code = "class A; template <class T> class Box { T v; }; template <> class Box<int> { int v; };"
    rewritten = class_definitions_as_structs(_tokens(code))

    keywords = [token for token in rewritten if token[1] == "class"]
    assert keywords == [
        ("STRUCT", "class"),
        ("IDENTIFIER", "class"),
        ("STRUCT", "class"),
        ("STRUCT", "class"),
    ]

"""HLSL namespaces import with qualified declarations and resolved references.

The importer flattened ``namespace ns { ... }`` and dropped its name while references kept their spelling, so
``ns::neg(x)`` called an undeclared function, ``ns::Pair`` named an undeclared ``ns_Pair``, and two namespaces that
declared ``f`` produced two indistinguishable functions. Each case below fails on that importer. The HLSL programs
compile with DXC 1.10 (``-HV 2021``), which also reports the ambiguous call refused below.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.DirectX import DirectxCrossGLCodeGen
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.backend.import_refusals import DirectXImportRefused
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

PROGRAM = textwrap.dedent("""
    float f(float x) { return x + 1.0; }
    namespace a {
        float f(float x) { return x * 2.0; }
        float f(int x) { return float(x) * 4.0; }
        float g(float x) { return f(x) + ::f(x); }
        namespace inner {
            static const float SCALE = 3.0;
            float h(float x) { float SCALE = 0.5; return x * SCALE; }
            float k(float x) { return x * SCALE + f(x); }
        }
        float m(float x) { return inner::k(x) + a::inner::h(x); }
        struct Item { float v; uint id; };
        Item make(float v) { Item it; it.v = v; it.id = 1; return it; }
        float3 tint();
    }
    namespace b {
        float f(float x) { return x - 3.0; }
        float only_b(float x) { return x * 10.0; }
    }
    namespace res {
        RWStructuredBuffer<float> data : register(u0);
        cbuffer Params : register(b0) { float bias; uint count; };
    }
    float3 a::tint() { return float3(inner::SCALE, 0.0, 1.0); }
    StructuredBuffer<a::Item> items : register(t1);
    using namespace a;
    using b::only_b;
    [numthreads(64, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {
        float x = res::data[id.x];
        a::Item it = make(x);
        float r = g(x) + b::f(x) + a::f(2) + m(x) + only_b(x) + ::f(x) + tint().x;
        res::data[id.x] = r + it.v + items[0].v + bias + float(count);
    }
""").strip()


def import_hlsl(code):
    ast = HLSLParser(HLSLLexer(code).tokenize()).parse()
    return DirectxCrossGLCodeGen.HLSLToCrossGLConverter().generate(ast)


def _body(crossgl, signature):
    start = crossgl.index(signature)
    return crossgl[start : crossgl.index("\n    }\n", start)]


def test_namespaced_declarations_take_qualified_names():
    crossgl = import_hlsl(PROGRAM)

    assert "struct a_Item {" in crossgl
    assert "static const float a_inner_SCALE = 3.0;" in crossgl
    for signature in (
        "float f(float x)",
        "float a_f(float x)",
        "float a_f(int x)",
        "float a_g(float x)",
        "float a_inner_h(float x)",
        "float a_inner_k(float x)",
        "a_Item a_make(float v)",
        "vec3 a_tint()",
        "float b_f(float x)",
        "float b_only_b(float x)",
    ):
        assert signature in crossgl, signature
    # Resources and constant-buffer members keep the names a host binds them by.
    assert "RWStructuredBuffer<float> data;" in crossgl
    assert "StructuredBuffer<a_Item> items;" in crossgl
    assert "float bias;" in crossgl
    assert "::" not in crossgl
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()


def test_references_resolve_as_cxx_lookup_does():
    crossgl = import_hlsl(PROGRAM)

    # ``f`` inside ``a`` is ``a::f``; ``::f`` is the file-scope ``f``.
    assert "return a_f(x) + f(x);" in _body(crossgl, "float a_g(float x)")
    # A local shadows the namespace member; the enclosing namespace's members are found from ``inner``.
    assert "return x * SCALE;" in _body(crossgl, "float a_inner_h(float x)")
    assert "(x * a_inner_SCALE) + a_f(x)" in _body(crossgl, "float a_inner_k(float x)")
    # ``inner::k`` is relative to ``a``.
    assert "return a_inner_k(x) + a_inner_h(x);" in _body(crossgl, "float a_m(float x)")
    # An out-of-line definition reads names from its namespace.
    assert "vec3(a_inner_SCALE, 0.0, 1.0)" in _body(crossgl, "vec3 a_tint()")
    main = crossgl[crossgl.index("void main(") :]
    assert "a_Item it = a_make(x);" in main
    assert "a_g(x) + b_f(x)" in main and "a_f(2)" in main and "a_m(x)" in main
    assert "b_only_b(x)" in main and "+ f(x)" in main and "a_tint().x" in main
    assert "data[id.x]" in main and "+ bias)" in main


def test_using_directives_resolve_relative_to_their_namespace():
    crossgl = import_hlsl(textwrap.dedent("""
        namespace a {
            namespace inner { float k(float x) { return x * 3.0; } }
            using namespace inner;
            float z(float x) { return k(x) + 1.0; }
        }
        float a_z(float x) { return x; }
        RWStructuredBuffer<float> data : register(u0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) { data[0] = a::z(data[1]) + a_z(data[2]); }
    """))

    assert "return a_inner_k(x) + 1.0;" in crossgl
    # ``a::z`` takes a fresh name, since the program declares ``a_z`` itself.
    assert "float a_z_(float x)" in crossgl and "float a_z(float x)" in crossgl
    assert "data[0] = a_z_(data[1]) + a_z(data[2]);" in crossgl


def test_methods_find_their_members_before_the_namespace():
    crossgl = import_hlsl(textwrap.dedent("""
        namespace a {
            static const float SCALE = 3.0;
            static const float v = 7.0;
            float helper(float x) { return x + SCALE; }
            struct S {
                float v;
                float get() { return helper(v) * SCALE; }
            };
        }
        RWStructuredBuffer<float> data : register(u0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) { a::S s; s.v = data[1]; data[0] = s.get(); }
    """))

    assert "struct a_S {" in crossgl
    assert "return a_helper(__this.v) * a_SCALE;" in crossgl


ENUMS = textwrap.dedent("""
    enum Level { Low, High = 5 };
    namespace a {
        enum Mode { Off, On = 3 };
        enum class Kind { X, Y };
        Mode pick(uint v) { return v > 0 ? On : Off; }
    }
    RWStructuredBuffer<uint> data : register(u0);
    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {
        a::Mode m = a::pick(data[1]);
        a::Kind k = a::Kind::Y;
        Level l = High;
        data[0] = (uint)m + (uint)a::On + (uint)a::Mode::Off + (uint)k + (uint)l;
    }
""").strip()


def test_enumerators_resolve_through_their_enumeration():
    crossgl = import_hlsl(ENUMS)

    assert "enum a_Mode {" in crossgl and "enum a_Kind {" in crossgl
    # An unscoped enumerator is a member of its enumeration's scope, in a namespace or at file scope.
    assert "return v > 0 ? a_Mode::On : a_Mode::Off;" in crossgl
    assert "a_Kind k = a_Kind::Y;" in crossgl
    assert "Level l = Level::High;" in crossgl
    assert "uint(a_Mode::On)" in crossgl and "uint(a_Mode::Off)" in crossgl


@pytest.mark.parametrize("backend", ["opengl", "vulkan", "metal", "directx"])
def test_namespaced_enumerations_translate_to_valid_targets(tmp_path, backend):
    path = tmp_path / "enums.hlsl"
    path.write_text(ENUMS)

    _validate(tmp_path, backend, crosstl.translate(str(path), backend=backend))


def test_a_name_found_in_two_namespaces_is_refused():
    program = textwrap.dedent("""
        float f(float x) { return x + 1.0; }
        namespace a { float f(float x) { return x * 2.0; } }
        using namespace a;
        RWStructuredBuffer<float> data : register(u0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) { data[0] = f(data[1]); }
    """)

    with pytest.raises(DirectXImportRefused, match="HLSL name 'f' is ambiguous"):
        import_hlsl(program)


def test_resources_of_one_name_in_two_namespaces_are_refused():
    program = textwrap.dedent("""
        namespace p { RWStructuredBuffer<float> data : register(u0); }
        namespace q { RWStructuredBuffer<float> data : register(u1); }
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) { p::data[0] = q::data[1]; }
    """)

    with pytest.raises(DirectXImportRefused, match="'q::data' has the same name"):
        import_hlsl(program)


def _validate(tmp_path, backend, output):
    if backend == "opengl":
        tool = shutil.which("glslangValidator")
        if tool is None:
            pytest.skip("glslangValidator is not installed")
        path = tmp_path / "out.comp"
        path.write_text(output)
        command = [tool, "-S", "comp", str(path)]
    elif backend == "vulkan":
        assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
        if assembler is None or validator is None:
            pytest.skip("SPIRV-Tools are not installed")
        path, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        path.write_text(output)
        subprocess.run(
            [assembler, "--target-env", "vulkan1.1", str(path), "-o", str(binary)],
            check=True,
            capture_output=True,
        )
        command = [validator, "--target-env", "vulkan1.1", str(binary)]
    elif backend == "metal":
        xcrun = shutil.which("xcrun")
        if xcrun is None:
            pytest.skip("xcrun is not installed")
        path = tmp_path / "out.metal"
        path.write_text(output)
        command = [xcrun, "-sdk", "macosx", "metal", "-c", str(path), "-o", "/dev/null"]
    else:
        tool = shutil.which("dxc")
        if tool is None:
            pytest.skip("dxc is not installed")
        path = tmp_path / "out.hlsl"
        path.write_text(output)
        command = [tool, "-HV", "2021", "-T", "cs_6_2", "-E", "CSMain", str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["opengl", "vulkan", "metal", "directx"])
def test_namespaced_programs_translate_to_valid_targets(tmp_path, backend):
    path = tmp_path / "namespaces.hlsl"
    path.write_text(PROGRAM)

    output = crosstl.translate(str(path), backend=backend)

    _validate(tmp_path, backend, output)

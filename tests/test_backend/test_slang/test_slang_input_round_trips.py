"""Slang constructs the Slang import used to drop or misread, so that the Slang round trip compiled differently.

Each case was a wrong row in the Slang input coverage: the import flattened nested blocks (a declaration in one then
clashed with or shadowed another), wrote ``-(-x)`` as ``--x``, kept a prototype as an empty function, wrote Slang's
``E.A`` for an enumerator, dropped member functions, ``packoffset`` and ``[[vk::offset]]`` placements,
``[earlydepthstencil]``, the conservative depth semantics and ``[[vk::shader_record]]``. Each CrossGL program is
generated back to Slang and compiled with slangc when it is installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.import_refusals import SlangImportRefused
from crosstl.translator.semantics.member_placements import MemberPlacementError
from tests.spirv_friendly_names import translate_spirv


def to_crossgl(tmp_path, source):
    path = tmp_path / "probe.slang"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="cgl")


def to_slang(tmp_path, crossgl):
    path = tmp_path / "round.cgl"
    path.write_text(crossgl, encoding="utf-8")
    return crosstl.translate(str(path), backend="slang")


def compile_slang(tmp_path, code):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    source, binary = tmp_path / "out.slang", tmp_path / "out.spv"
    source.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [slangc, str(source), "-target", "spirv", "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def round_trip(tmp_path, source):
    crossgl = to_crossgl(tmp_path, source)
    compile_slang(tmp_path, to_slang(tmp_path, crossgl))
    return crossgl


def squash(code):
    return re.sub(r"\s+", " ", code)


COMPUTE = """RWStructuredBuffer<float> fin;
RWStructuredBuffer<int> iin;
RWStructuredBuffer<float> outp;
"""


def test_nested_blocks_keep_their_scope(tmp_path):
    source = COMPUTE + """static const float scale = 2.0;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    float x = fin[i];
    float r = 0.0;
    { float x = 2.0; r += x; }
    { int t = iin[i] + 1; r += float(t); }
    { float t = float(iin[i]) * 0.5; r += t; }
    { float scale = 8.0; r += scale; }
    outp[i] = r + x + scale;
}
"""
    crossgl = round_trip(tmp_path, source)
    text = squash(crossgl)
    assert "{ float x = 2.0; r += x; }" in text
    assert (
        "{ int t = iin[i] + 1;" in text and "{ float t = float(iin[i]) * 0.5;" in text
    )
    assert "{ float scale = 8.0; r += scale; }" in text


def test_braced_case_bodies_keep_their_scope(tmp_path):
    source = COMPUTE + """[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    float acc = 0.0;
    switch (iin[i] & 3) {
        case 0: { float t = fin[i]; acc = t * 2.0; break; }
        default: { int t = iin[i] * 4; acc = float(t); break; }
    }
    outp[i] = acc;
}
"""
    text = squash(round_trip(tmp_path, source))
    assert "case 0: { float t = fin[i];" in text
    assert "default: { int t = iin[i] * 4;" in text


def test_nested_prefix_operators_keep_their_parentheses(tmp_path):
    source = COMPUTE + """[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    int k = iin[i];
    float x = fin[i];
    bool b = k > 3;
    float a = -(-x) + -(+x);
    int e = ~(~k) + -(~k);
    bool g = !(!b);
    outp[i] = a + float(e) + (g ? 1.0 : 0.0);
}
"""
    crossgl = round_trip(tmp_path, source)
    for expression in ("-(-x)", "-(+x)", "~(~k)", "-(~k)", "!(!b)"):
        assert expression in crossgl
    assert "--x" not in crossgl


def test_prototypes_are_not_written_as_empty_functions(tmp_path):
    source = COMPUTE + """float twice(float x);
float quadruple(float x) { return twice(twice(x)); }
float twice(float x) { return x * 2.0; }
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outp[id.x] = quadruple(fin[id.x]);
}
"""
    crossgl = round_trip(tmp_path, source)
    assert crossgl.count("float twice(float x)") == 1


def test_enumerators_are_scoped_with_double_colons(tmp_path):
    source = COMPUTE + """enum Mode { Off, Low = 2, High };
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    Mode m = (iin[i] & 1) != 0 ? Mode.Low : Mode.High;
    float r = 0.0;
    switch (m) {
    case Mode.Off: r = 1.0; break;
    default: r = 4.0; break;
    }
    outp[i] = r + float(int(m)) + float(int(Mode::Off));
}
"""
    crossgl = round_trip(tmp_path, source)
    assert "Mode::Low : Mode::High" in crossgl
    assert "case Mode::Off:" in crossgl
    assert "Mode.Low" not in crossgl


def test_member_functions_are_kept(tmp_path):
    source = COMPUTE + """struct Span {
    float lo;
    float hi;
    float width() { return hi - lo; }
    float at(float t) { return lo + t * width(); }
    [mutating] void widen(float d) { hi += d; }
};
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    Span s;
    s.lo = fin[id.x];
    s.hi = s.lo * 2.0 + 1.0;
    float w = s.width();
    s.widen(3.0);
    outp[id.x] = w + s.at(0.5) + s.hi;
}
"""
    text = squash(round_trip(tmp_path, source))
    assert "float width() { return hi - lo; }" in text
    assert "void widen(float d) { hi += d; }" in text


def test_generic_member_functions_are_refused(tmp_path):
    source = COMPUTE + """struct Box {
    float v;
    T pick<T>(T a) { return a; }
};
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    Box b;
    b.v = 1.0;
    outp[id.x] = b.v;
}
"""
    with pytest.raises(
        SlangImportRefused, match="'pick' of Slang structure 'Box' is generic"
    ):
        to_crossgl(tmp_path, source)


def test_packoffset_members_keep_their_registers(tmp_path):
    source = COMPUTE + """cbuffer Params {
    float a : packoffset(c0);
    float3 b : packoffset(c0.y);
    float2 g : packoffset(c1.z);
};
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outp[id.x] = fin[id.x] * (a + b.x + g.y);
}
"""
    crossgl = to_crossgl(tmp_path, source)
    text = squash(crossgl)
    assert "@ buffer_layout(hlsl_cbuffer) cbuffer Params {" in text
    assert "@ packoffset(c0.y) vec3 b;" in text
    slang = to_slang(tmp_path, crossgl)
    assert "float3 b : packoffset(c0.y);" in slang
    assert "float2 g : packoffset(c1.z);" in slang
    compile_slang(tmp_path, slang)


def test_vk_offset_members_are_kept(tmp_path):
    source = COMPUTE + """struct Constants {
    [[vk::offset(16)]] float scale;
    float bias;
};
[[vk::push_constant]] ConstantBuffer<Constants> pc;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outp[id.x] = fin[id.x] * pc.scale + pc.bias;
}
"""
    crossgl = to_crossgl(tmp_path, source)
    assert "float scale @offset(16);" in crossgl
    # Slang output refuses a placed push-constant member instead of dropping the placement.
    with pytest.raises(MemberPlacementError):
        to_slang(tmp_path, crossgl)


def test_earlydepthstencil_is_early_fragment_tests(tmp_path):
    source = """[shader("fragment")]
[earlydepthstencil]
float4 main(float2 uv : TEXCOORD0) : SV_Target {
    return float4(uv, 0.0, 1.0);
}
"""
    crossgl = round_trip(tmp_path, source)
    assert "layout(early_fragment_tests) in;" in crossgl


@pytest.mark.parametrize(
    "semantic, mode",
    [("SV_DepthGreaterEqual", "depth_greater"), ("SV_DepthLessEqual", "depth_less")],
)
def test_conservative_depth_semantics(tmp_path, semantic, mode):
    source = f"""struct FragmentOutput {{
    float4 color : SV_Target;
    float depth : {semantic};
}};
[shader("fragment")]
FragmentOutput main(float2 uv : TEXCOORD0) {{
    FragmentOutput output;
    output.color = float4(uv, 0.0, 1.0);
    output.depth = uv.x * 0.5 + 0.25;
    return output;
}}
"""
    crossgl = round_trip(tmp_path, source)
    assert f"float depth @ gl_FragDepth @ {mode};" in crossgl


RECORD = """struct [raypayload] Payload {
    float4 color : read(caller) : write(anyhit, closesthit, miss);
};
struct Record {
    float a;
    float3 b;
    float c[2];
};
"""


def test_shader_records_are_std140_record_blocks(tmp_path):
    source = RECORD + """[[vk::shader_record]] ConstantBuffer<Record> record;
[shader("miss")]
void main(inout Payload payload) {
    payload.color = float4(record.a, record.b.x, record.c[1], 1.0);
}
"""
    crossgl = to_crossgl(tmp_path, source)
    assert "Record record @glsl_buffer_block(shaderRecordEXT, std140);" in crossgl
    path = tmp_path / "record.cgl"
    path.write_text(crossgl, encoding="utf-8")
    spirv = translate_spirv(path)
    # slangc lays the record out by std140: a at 0, b at 16, c at 32 with a 16-byte stride.
    assert re.search(r"OpTypePointer ShaderRecordBufferKHR %Record", spirv)
    offsets = re.findall(r"OpMemberDecorate %Record\w* (\d) Offset (\d+)", spirv)
    assert offsets[:3] == [("0", "0"), ("1", "16"), ("2", "32")]
    assert "ArrayStride 16" in spirv


def test_shader_record_that_is_not_a_constant_buffer_is_refused(tmp_path):
    source = RECORD + """[[vk::shader_record]] StructuredBuffer<Record> record;
[shader("miss")]
void main(inout Payload payload) {
    payload.color = float4(record[0].a, 0.0, 0.0, 1.0);
}
"""
    with pytest.raises(SlangImportRefused, match="not a ConstantBuffer<T>"):
        to_crossgl(tmp_path, source)

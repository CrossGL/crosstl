"""The WGSL importer (crosstl/backend/WGSL): WGSL modules written as CrossGL, and back to WGSL through CrossGL.

Each import is parsed again as CrossGL. When naga is installed, the WGSL that CrossTL then writes from the CrossGL is
validated too, so the round trip is checked end to end.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.WGSL import WgslLexer, WgslParser, WgslToCrossGLConverter
from crosstl.backend.WGSL.wgsl_types import WgslImportError

BUFFERS = (
    "@group(0) @binding(0) var<storage, read_write> fin: array<f32>;\n"
    "@group(0) @binding(1) var<storage, read_write> iin: array<i32>;\n"
    "@group(0) @binding(2) var<storage, read_write> outp: array<f32>;\n"
)


def _import(source):
    return WgslToCrossGLConverter().generate(
        WgslParser(WgslLexer(source).tokenize()).parse()
    )


def _compute(body, helpers=""):
    return (
        f"{BUFFERS}{helpers}\n@compute @workgroup_size(1)\n"
        f"fn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n    let i = id.x;\n{body}\n}}\n"
    )


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _round_trip(source, tmp_path):
    """The CrossGL import of ``source`` and the WGSL CrossTL writes from it, validated with naga if present."""
    path = tmp_path / "probe.wgsl"
    path.write_text(source)
    crossgl = crosstl.translate(str(path), backend="cgl", format_output=False)
    crosstl.translator.parse(crossgl)
    cgl_path = tmp_path / "probe.cgl"
    cgl_path.write_text(crossgl)
    wgsl = crosstl.translate(str(cgl_path), backend="wgsl", format_output=False)
    naga = shutil.which("naga")
    if naga is not None:
        (tmp_path / "out.wgsl").write_text(wgsl)
        result = subprocess.run(
            [naga, "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr + wgsl
    return crossgl, wgsl


def test_compute_entry_keeps_bindings_and_workgroup_size(tmp_path):
    crossgl, wgsl = _round_trip(
        _compute("    outp[i] = fin[i] * 2.0;").replace("(1)", "(8, 2)"), tmp_path
    )
    assert "RWStructuredBuffer<float> fin @set(0) @binding(0);" in crossgl
    assert "layout(local_size_x = 8, local_size_y = 2, local_size_z = 1) in;" in crossgl
    assert "void main(uvec3 id @ gl_GlobalInvocationID)" in crossgl
    assert "@binding(2)" in wgsl and "@workgroup_size(8, 2, 1)" in wgsl


def test_abstract_literals_take_the_type_of_their_context(tmp_path):
    crossgl, _ = _round_trip(
        _compute(
            "    let k: u32 = 3;\n    let f = fin[i] * 2;\n    outp[i] = f + f32(k) + f32(iin[i] << 1);"
        ),
        tmp_path,
    )
    assert "uint k = 3u;" in crossgl
    assert "float f = (fin[i] * 2.0);" in crossgl
    assert "(iin[i] << 1u)" in crossgl


def test_var_without_initializer_is_zero(tmp_path):
    source = _compute(
        "    var v: vec3<f32>;\n    var s: S;\n    var a: array<i32, 2>;\n"
        "    outp[i] = v.x + s.a + f32(a[1]);",
        "struct S { a: f32, b: u32, }\n",
    )
    crossgl, _ = _round_trip(source, tmp_path)
    assert "vec3 v = vec3(0.0);" in crossgl
    assert "S s = S(0.0, 0u);" in crossgl
    assert "int a[2] = int[2](0, 0);" in crossgl


def test_workgroup_variables_start_at_zero(tmp_path):
    source = _compute(
        "    tile[0] = fin[i];\n    workgroupBarrier();\n    outp[i] = tile[0] + f32(big[3]);",
        "var<workgroup> tile: array<f32, 2>;\nvar<workgroup> big: array<i32, 64>;\n",
    )
    crossgl, _ = _round_trip(source, tmp_path)
    flat = _flat(crossgl)
    assert "shared float tile[2];" in crossgl
    assert "if (gl_LocalInvocationIndex == 0u) { tile = float[2](0.0, 0.0);" in flat
    assert (
        "for (int zero_index = 0; zero_index < int(64); zero_index++) { big[zero_index] = 0; }"
        in flat
    )
    assert flat.index("barrier();") < flat.index("uint i = id.x;")


def test_loop_continuing_runs_before_each_continue(tmp_path):
    source = _compute(
        "    var k = 0;\n    var acc = 0.0;\n"
        "    loop {\n        k++;\n        if (k > 4) { acc += 0.5; continue; }\n        acc += f32(k);\n"
        "        continuing { acc *= 1.5; break if k >= 6; }\n    }\n    outp[i] = acc;"
    )
    crossgl, _ = _round_trip(source, tmp_path)
    flat = _flat(crossgl)
    assert "while (true) {" in flat
    assert "acc += 0.5; { acc *= 1.5; if (k >= 6) { break; } continue; }" in flat
    assert flat.count("acc *= 1.5;") == 2


def test_continue_in_a_switch_of_a_loop_with_break_if_is_refused():
    source = _compute(
        "    var k = 0;\n    loop {\n        switch k { case 0: { continue; } default: { } }\n"
        "        continuing { k++; break if k > 2; }\n    }"
    )
    with pytest.raises(WgslImportError, match="continue inside a switch"):
        _import(source)


def test_switch_clauses_end_with_break(tmp_path):
    source = _compute(
        "    var r = 0.0;\n    switch iin[i] { case 1, 2: { r = 1.0; } case 3, default: { r = 2.0; } }\n"
        "    outp[i] = r;"
    )
    crossgl, _ = _round_trip(source, tmp_path)
    flat = _flat(crossgl)
    assert "case 1: case 2: { r = 1.0; break; }" in flat
    assert "case 3: default: { r = 2.0; break; }" in flat


def test_integer_division_by_a_run_time_divisor_has_wgsl_results(tmp_path):
    source = _compute(
        "    let a = iin[i];\n    let b = iin[i + 1u];\n    let u = u32(a);\n"
        "    outp[i] = f32(a / b) + f32(a % b) + f32(u / 3u) + f32(a / 2) + f32(u % u32(b));"
    )
    crossgl, _ = _round_trip(source, tmp_path)
    assert (
        "return (b == 0 || (a == (-2147483647 - 1) && b == -1)) ? a : (a / b);"
        in crossgl
    )
    assert (
        "return (b == 0 || (a == (-2147483647 - 1) && b == -1)) ? 0 : (a % b);"
        in crossgl
    )
    assert "return (b == 0u) ? 0u : (a % b);" in crossgl
    assert "float((u / 3u))" in crossgl and "float((a / 2))" in crossgl


def test_shift_by_a_run_time_amount_is_masked(tmp_path):
    source = _compute(
        "    let s = u32(iin[i]);\n    outp[i] = f32(iin[i] << s) + f32(iin[i] >> 3u);"
    )
    crossgl, _ = _round_trip(source, tmp_path)
    assert "(iin[i] << (s & 31u))" in crossgl
    assert "(iin[i] >> 3u)" in crossgl


def test_builtins_keep_wgsl_results(tmp_path):
    source = _compute(
        "    let x = fin[i];\n    let v = vec2<f32>(x, 1.0);\n"
        "    let a = round(x) + clamp(x, 0.0, 1.0) + inverseSqrt(x) + atan2(x, 1.0);\n"
        "    let b = select(v, vec2<f32>(0.0), v > vec2<f32>(0.5));\n"
        "    let c = countOneBits(u32(iin[i])) + firstLeadingBit(u32(iin[i]));\n"
        "    outp[i] = a + b.x + f32(c) + select(1.0, 2.0, x > 0.5);"
    )
    crossgl, _ = _round_trip(source, tmp_path)
    assert "roundEven(x)" in crossgl
    assert "min(max(x, 0.0), 1.0)" in crossgl
    assert "inversesqrt(x)" in crossgl and "atan(x, 1.0)" in crossgl
    assert "mix(v, vec2(0.0), (v > vec2(0.5)))" in crossgl
    assert (
        "uint(bitCount(uint(iin[i])))" in crossgl
        and "uint(findMSB(uint(iin[i])))" in crossgl
    )
    assert "((x > 0.5) ? 2.0 : 1.0)" in crossgl


def test_pointers_are_the_memory_they_point_to(tmp_path):
    source = _compute(
        "    var k = iin[i] & 1;\n    let p = &outp[i + u32(k)];\n    k = 5;\n    *p = fin[i];\n"
        "    var a = fin[i];\n    bump(&a);\n    outp[i] += a;",
        "fn bump(a: ptr<function, f32>) { *a = *a + 1.0; }\n",
    )
    crossgl, _ = _round_trip(source, tmp_path)
    assert "void bump(inout float a) {" in crossgl
    assert "a = (a + 1.0);" in crossgl
    # The index is read where the let is, before k changes.
    assert re.search(
        r"uint (tmp\w*) = \(i \+ uint\(k\)\);\s+k = 5;\s+outp\[\1\] = fin\[i\];",
        crossgl,
    )
    assert "bump(a);" in crossgl


def test_overrides_are_specialization_constants(tmp_path):
    source = _compute(
        "    outp[i] = fin[i] * f32(scale) * f32(doubled) * gain;",
        "override scale: i32 = 3;\n@id(7) override gain: f32 = 1.5;\noverride doubled: i32 = scale * 2 + 1;\n",
    )
    crossgl, wgsl = _round_trip(source, tmp_path)
    assert "const int scale @constant_id(0) = 3;" in crossgl
    assert "const float gain @constant_id(7) = 1.5;" in crossgl
    assert "const int doubled = ((scale * 2) + 1);" in crossgl
    assert "override scale" in wgsl and "override gain" in wgsl


def test_textures_and_samplers_are_separate(tmp_path):
    source = _compute(
        "    let uv = vec2<f32>(fin[i], 0.5);\n"
        "    let c = textureSampleLevel(t, s, uv, 0.0) + textureLoad(t, vec2<u32>(1u, 2u), 0);\n"
        "    let d = textureDimensions(t);\n"
        "    textureStore(img, vec2<i32>(0, 0), c);\n"
        "    outp[i] = c.x + f32(d.x);",
        "@group(0) @binding(4) var t: texture_2d<f32>;\n@group(0) @binding(5) var s: sampler;\n"
        "@group(0) @binding(6) var img: texture_storage_2d<rgba32float, write>;\n",
    )
    crossgl = _import(source)
    assert "texture2D t @set(0) @binding(4);" in crossgl
    assert "sampler s @set(0) @binding(5);" in crossgl
    assert "image2D img @rgba32f @access(write) @set(0) @binding(6);" in crossgl
    assert "textureLod(t, s, uv, 0.0)" in crossgl
    assert (
        "texelFetch(t, ivec2(vec2(1u, 2u)), 0)".replace("vec2(1u", "uvec2(1u")
        in crossgl
    )
    assert "uvec2(textureSize(t, 0))" in crossgl
    assert "imageStore(img, ivec2(0, 0), c);" in crossgl


def test_fragment_and_vertex_interfaces(tmp_path):
    source = (
        "struct VOut {\n    @builtin(position) @invariant pos: vec4<f32>,\n"
        "    @location(0) @interpolate(flat) n: u32,\n    @location(1) @interpolate(linear, centroid) uv: vec2<f32>,\n}\n"
        "@vertex fn vs(@location(0) p: vec4<f32>, @builtin(vertex_index) v: u32) -> VOut {\n"
        "    var o: VOut;\n    o.pos = p;\n    o.n = v;\n    o.uv = p.xy;\n    return o;\n}\n"
        "@fragment fn fs(@location(0) @interpolate(flat) n: u32, @location(1) @interpolate(linear, centroid)"
        " uv: vec2<f32>, @builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {\n"
        "    return vec4<f32>(uv, frag.x, f32(n));\n}\n"
    )
    crossgl, wgsl = _round_trip(source, tmp_path)
    assert "vec4 pos @ gl_Position @ invariant;" in crossgl
    assert "uint n @location(0) @ flat;" in crossgl
    assert "uint v @ gl_VertexIndex" in crossgl
    assert "vec2 uv @location(1) @ noperspective @ centroid" in crossgl
    assert "vec4 frag @ gl_FragCoord" in crossgl
    assert ") @location(0) {" in crossgl
    assert "@vertex" in wgsl and "@fragment" in wgsl


def test_names_crossgl_reserves_are_renamed(tmp_path):
    source = _compute(
        "    var t: thread;\n    t.input = fin[i];\n    let output = t.input * 2.0;\n    outp[i] = output;",
        "struct thread { input: f32, }\n",
    )
    crossgl, _ = _round_trip(source, tmp_path)
    assert "struct thread_ {" in crossgl
    assert "float input_;" in crossgl
    assert "float output_ = (t.input_ * 2.0);" in crossgl


@pytest.mark.parametrize(
    ("helpers", "body", "message"),
    [
        (
            "@group(0) @binding(3) var<storage> s: S;\nstruct S { a: f32, }\n",
            "    outp[i] = s.a;",
            "must be a runtime-sized array",
        ),
        (
            "struct S { @align(16) a: f32, }\n",
            "    var s: S;\n    outp[i] = s.a;",
            "@align",
        ),
        ("", "    outp[i] = madeUp(fin[i]);", "'madeUp' is not supported"),
    ],
)
def test_constructs_without_a_crossgl_form_are_refused(helpers, body, message):
    with pytest.raises(WgslImportError, match=re.escape(message)):
        _import(_compute(body, helpers))


def test_translate_reads_wgsl_by_extension(tmp_path):
    path = tmp_path / "kernel.wgsl"
    path.write_text(_compute("    outp[i] = fin[i] + 1.0;"))
    metal = crosstl.translate(str(path), backend="metal", format_output=False)
    assert "kernel void" in metal

"""Probes for the memory layout of uniform and storage buffers: arrays, structs, matrices, booleans and packing."""

import pytest

from tools.coverage import constructs, execution, lowering, probes
from tools.coverage.model import Result

from .construct_grammar import GRAMMAR, construct_feature


def test_a_uniform_float_array_reads_one_word_per_row_in_every_language():
    # std140 and a cbuffer give each float its own 16-byte row; languages that pack a float array declare vectors.
    padded = probes.build(construct_feature("glsl", "uniform-array"), "glsl").source
    assert "float w[4];" in padded and "+ w[k])" in padded
    for language, declaration in (
        ("metal", "float4 w[4];"),
        ("wgsl", "w: array<vec4<f32>, 4>,"),
        ("cuda", "float4 w[4];"),
        ("rust", "w: [Vec4; 4],"),
    ):
        source = probes.build(
            construct_feature(language, "uniform-array"), language
        ).source
        assert declaration in source and "w[k].x" in source


def test_a_uniform_struct_is_declared_before_its_block_and_copied_into_a_mutable_local():
    glsl = probes.build(construct_feature("glsl", "uniform-struct"), "glsl").source
    assert glsl.index("struct Light {") < glsl.index("uniform Params {")
    assert "    Light light;\n    vec4 tint;" in glsl
    for language, copy in (
        ("wgsl", "var l = params.light;"),
        ("rust", "let mut l = params.light;"),
        ("mojo", "var l = params.light\n"),
        ("hlsl", "Light l = light;"),
    ):
        source = probes.build(
            construct_feature(language, "uniform-struct-copy"), language
        ).source
        assert copy in source
    # Mojo reads a vector's lanes by index.
    mojo = probes.build(
        construct_feature("mojo", "uniform-struct-array"), "mojo"
    ).source
    assert "params.lights[k].color[0]" in mojo and "var lights: Array[Light, 2]" in mojo


def test_uniform_matrix_probes_read_every_column_from_its_own_16_byte_row():
    crossgl = probes.build(
        construct_feature("glsl", "uniform-matrix3"), "crossgl"
    ).source
    assert "cbuffer Params @binding(3) {\n        mat3 m;\n        vec4 v;" in crossgl
    # Metal, WGSL and Rust give a 2x2 matrix an 8-byte column stride, so they build it from 16-byte vectors.
    for language, member, build in (
        (
            "metal",
            "float4 c[2];",
            "float2x2 m = float2x2(params.c[0].xy, params.c[1].xy);",
        ),
        (
            "wgsl",
            "c: array<vec4<f32>, 2>,",
            "let m = mat2x2<f32>(params.c[0].xy, params.c[1].xy);",
        ),
        (
            "rust",
            "c: [Vec4; 2],",
            "let m = Mat2::from_cols(Vec2::new(params.c[0].x, params.c[0].y)",
        ),
    ):
        source = probes.build(
            construct_feature(language, "uniform-matrix2"), language
        ).source
        assert member in source and build in source, language
    rust = probes.build(construct_feature("rust", "uniform-matrix3"), "rust").source
    assert "c: [Vec4; 3],\n    v: Vec4," in rust and "m.z_axis.y * params.v.w" in rust
    # HLSL names CrossGL's mat2x3 float3x2 and swaps the indices.
    hlsl = probes.build(
        construct_feature("hlsl", "uniform-non-square-matrix"), "hlsl"
    ).source
    assert "float3x2 m;" in hlsl and "mul(v.xyz, m)" in hlsl and "m[2][1] * v.w" in hlsl
    wgsl = probes.build(
        construct_feature("wgsl", "uniform-non-square-matrix"), "wgsl"
    ).source
    assert "m: mat2x3<f32>," in wgsl and "params.m[1][2] * params.v.w" in wgsl
    for language in ("cuda", "hip", "opencl", "mojo"):
        assert not [
            f
            for f in constructs.features(language, GRAMMAR)
            if f.name.startswith("uniform-matrix")
            or f.name == "uniform-non-square-matrix"
        ], language


def test_uniform_bool_probes_store_each_boolean_in_a_32_bit_word():
    crossgl = probes.build(
        construct_feature("glsl", "uniform-bool-vector"), "crossgl"
    ).source
    assert (
        "cbuffer Params @binding(3) {\n        bvec4 f;\n        bvec2 g;\n        bool h;"
        in crossgl
    )
    hlsl = probes.build(construct_feature("hlsl", "uniform-bool-vector"), "hlsl").source
    assert "bool4 f;\n    bool2 g;" in hlsl and "bool4 c = f;" in hlsl
    # Metal's bool is one byte and the others cannot put a boolean in a host-written block, so they test a uint.
    for language, member, test in (
        ("metal", "uint b0;", "bool b0 = bool(params.b0);"),
        ("wgsl", "b0: u32,", "let b0 = bool(params.b0);"),
        ("cuda", "unsigned int b0;", "bool b0 = params.b0 != 0u;"),
        ("opencl", "uint b0;", "bool b0 = params.b0 != 0u;"),
        ("rust", "b0: u32,", "let b0 = params.b0 != 0;"),
        ("mojo", "var b0: UInt32", "var b0 = params.b0 != 0"),
    ):
        source = probes.build(
            construct_feature(language, "uniform-bool"), language
        ).source
        assert member in source and test in source, language
    for language, member, test in (
        ("metal", "uint4 f;\n    uint2 g;", "bool4 f = bool4(params.f);"),
        ("wgsl", "f: vec4<u32>,\n    g: vec2<u32>,", "var c = vec4<bool>(params.f);"),
        ("rust", "f: UVec4,\n    g: UVec2,", "params.f.cmpne(UVec4::ZERO)"),
        ("mojo", "var f: SIMD[DType.uint32, 4]", "params.f.cast[DType.bool]()"),
    ):
        source = probes.build(
            construct_feature(language, "uniform-bool-vector"), language
        ).source
        assert member in source and test in source, language
    for language in ("cuda", "hip", "opencl"):
        names = {f.name for f in constructs.features(language, GRAMMAR)}
        assert "uniform-bool" in names and "uniform-bool-vector" not in names, language


def test_row_major_probes_store_each_matrix_row_on_its_own_16_byte_row():
    crossgl = probes.build(
        construct_feature("glsl", "uniform-row-major-non-square"), "crossgl"
    ).source
    assert (
        "cbuffer Params @binding(3) {\n        @ row_major mat2x3 m;\n        vec4 v;"
        in crossgl
    )
    glsl = probes.build(
        construct_feature("glsl", "uniform-row-major-array"), "glsl"
    ).source
    assert "layout(row_major) mat2 m[2];" in glsl and "m[0][1][0] * 8.0" in glsl
    # HLSL indexes by rows, so CrossGL's element in column 1 and row 2 is m[2][1].
    hlsl = probes.build(
        construct_feature("hlsl", "uniform-row-major-non-square"), "hlsl"
    ).source
    assert "row_major float3x2 m;\n    float4 v;" in hlsl and "m[2][1] * v.w" in hlsl
    # Metal, WGSL and Rust have no row-major matrices, so they transpose the matrix whose columns are the rows.
    for language, member, build in (
        (
            "metal",
            "float4 r[3];\n    float4 v;",
            "float3x3 m = transpose(float3x3(params.r[0].xyz,",
        ),
        (
            "wgsl",
            "r: array<vec4<f32>, 3>,\n    v: vec4<f32>,",
            "let m = transpose(mat3x3<f32>(params.r[0].xyz,",
        ),
        (
            "rust",
            "r: [Vec4; 3],\n    v: Vec4,",
            ".transpose();\n    let t = m * params.v",
        ),
    ):
        source = probes.build(
            construct_feature(language, "uniform-row-major-matrix"), language
        ).source
        assert member in source and build in source, language
    names = {f.name for f in constructs.features("rust", GRAMMAR)}
    assert "uniform-row-major-array" in names
    assert "uniform-row-major-non-square" not in names
    for language in ("cuda", "hip", "opencl", "mojo"):
        assert not [
            f
            for f in constructs.features(language, GRAMMAR)
            if f.name.startswith("uniform-row-major")
        ], language


def test_buffer_row_major_probes_store_struct_matrices_by_rows():
    crossgl = probes.build(
        construct_feature("glsl", "buffer-row-major-non-square"), "crossgl"
    ).source
    assert "struct Rows {\n        @ row_major mat2x3 m;\n        vec2 w;" in crossgl
    assert "buffer Rows* outp;" in crossgl
    # GLSL has no layout qualifier on a structure member, so the buffer's array is row-major.
    glsl = probes.build(
        construct_feature("glsl", "buffer-row-major-array"), "glsl"
    ).source
    assert "{ layout(row_major) Rows outp[]; }" in glsl and "mat2 m[2];" in glsl
    hlsl = probes.build(
        construct_feature("hlsl", "buffer-row-major-matrix"), "hlsl"
    ).source
    assert "row_major float2x2 m;" in hlsl and "RWStructuredBuffer<Rows>" in hlsl
    assert "outp[i].m[1][0] += 0.25;" in hlsl
    # Metal, WGSL and Rust store the rows of the matrix's transpose.
    for language, member, store in (
        ("metal", "float2 r[3];", "outp[i].r[2][1] += 0.25;"),
        ("wgsl", "r: array<vec2<f32>, 3>,", "let t = transpose(mat2x3<f32>("),
    ):
        source = probes.build(
            construct_feature(language, "buffer-row-major-non-square"), language
        ).source
        assert member in source and store in source, language
    rust = probes.build(
        construct_feature("rust", "buffer-row-major-array"), "rust"
    ).source
    assert "r: [[Vec2; 2]; 2]," in rust and "outp[i].r[k][0].y += 0.25;" in rust
    names = {f.name for f in constructs.features("rust", GRAMMAR)}
    assert "buffer-row-major-matrix" in names
    assert "buffer-row-major-non-square" not in names
    for language in ("webgl", "cuda", "hip", "opencl", "mojo"):
        assert not [
            f
            for f in constructs.features(language, GRAMMAR)
            if f.name.startswith("buffer-row-major")
        ], language


def test_packing_probes_name_the_layout_rule_and_declare_its_offsets():
    for name, rule in (
        ("uniform-std140-vector", "std140"),
        ("uniform-cbuffer-vector", "hlsl_cbuffer"),
        ("uniform-std140-array", "std140"),
    ):
        crossgl = probes.build(construct_feature("glsl", name), "crossgl").source
        assert f"cbuffer Params @binding(3) @buffer_layout({rule}) {{" in crossgl, name
    # slangc lays out a plain cbuffer by std140 for SPIR-V, so HLSL and Slang name every member's offset.
    for language in ("hlsl", "slang"):
        std140 = probes.build(
            construct_feature(language, "uniform-std140-vector"), language
        )
        assert (
            "float3 b : packoffset(c1);\n    float c : packoffset(c1.w);"
            in std140.source
        )
        cbuffer = probes.build(
            construct_feature(language, "uniform-cbuffer-vector"), language
        )
        assert (
            "float3 b : packoffset(c0.y);\n    float c : packoffset(c1);"
            in cbuffer.source
        )
        array = probes.build(
            construct_feature(language, "uniform-std140-array"), language
        )
        assert "float d : packoffset(c2);" in array.source
    # No GLSL, WGSL or OpenCL C rule puts a three-component vector at offset 4, so those declare its components.
    for language, member, build in (
        ("glsl", "float b0;", "vec3 b = vec3(b0, b1, b2);"),
        ("wgsl", "b0: f32,", "let b = vec3<f32>(params.b0, params.b1, params.b2);"),
        (
            "opencl",
            "float b0;",
            "float3 b = (float3)(params.b0, params.b1, params.b2);",
        ),
    ):
        source = probes.build(
            construct_feature(language, "uniform-cbuffer-vector"), language
        )
        assert member in source.source and build in source.source, language
    # The languages without std140 rules declare its padding.
    for language, padded in (
        ("metal", "float pad2;\n    packed_float3 b;"),
        ("cuda", "float pad2;\n    float3 b;"),
        ("rust", "pad2: f32,\n    b: Vec3,"),
        ("mojo", "var pad2: Float32\n    var b0: Float32"),
    ):
        source = probes.build(
            construct_feature(language, "uniform-std140-vector"), language
        )
        assert padded in source.source and "pad3" in source.source, language


def test_uniform_integer_probes_keep_each_member_s_signedness_in_every_language():
    crossgl = probes.build(
        construct_feature("glsl", "uniform-integer"), "crossgl"
    ).source
    assert "cbuffer Params @binding(3) {\n        int n0;" in crossgl
    for language, member, use in (
        ("wgsl", "u1: u32,", "f32(params.u2 % 1000u + (params.u3 >> 24))"),
        ("cuda", "unsigned int u1;", "(float)(params.u2 % 1000u + (params.u3 >> 24))"),
        ("opencl", "uint u1;", "(float)(params.u2 % 1000u + (params.u3 >> 24))"),
        ("rust", "u1: u32,", "(params.u2 % 1000 + (params.u3 >> 24)) as f32"),
        ("mojo", "var u1: UInt32", "Float32(params.u2 % 1000 + (params.u3 >> 24))"),
    ):
        source = probes.build(
            construct_feature(language, "uniform-unsigned"), language
        ).source
        assert member in source and use in source, language
    mojo = probes.build(construct_feature("mojo", "uniform-integer"), "mojo").source
    assert "params.n3 // 1000000" in mojo
    for language, local in (
        ("wgsl", "let v = params.iv;"),
        ("mojo", "var v = params.iv"),
    ):
        vector = probes.build(
            construct_feature(language, "uniform-integer-vector"), language
        )
        assert local in vector.source and "$" not in vector.source, language
    assert "(v[0] >> 20) + (v[1] >> 20) + (v[3] >> 24)" in vector.source


# Each HLSL matrix probe with its matrices written as CrossGL lists them, by columns, and transposed into HLSL's rows.
# The indexing probe keeps CrossGL's columns as HLSL rows instead, and indexes them as CrossGL does.
_HLSL_BY_COLUMNS = {
    "matrix-indexing": (
        [("int", "iin"), ("float", "outp")],
        [
            "float f = float(iin[i]);",
            "float2x2 m = float2x2(f, 1.0, 2.0, 3.0);",
            "float2 c = m[iin[i] & 1];",
            "outp[i] = m[1][0] * 100.0 + m[0].y * 10.0 + c.x + 0.5 * c.y;",
        ],
    ),
    "matrix-from-vectors": (
        [("int", "iin"), ("float", "outp")],
        [
            "float f = float(iin[i]);",
            "float2x2 m = transpose(float2x2(float2(f, 1.0), float2(2.0, 3.0)));",
            "outp[i] = m[1][0] * 100.0 + m[0][1] * 10.0 + m[0][0];",
        ],
    ),
    "matrix-vector-product": (
        [("float", "fin"), ("float", "outp")],
        [
            "float2x2 m = transpose(float2x2(fin[i], 1.0, 2.0, 3.0));",
            "float2 v = mul(m, float2(1.0, fin[i]));",
            "outp[i] = v.y;",
        ],
    ),
    "matrix-product": (
        [("int", "iin"), ("float", "outp")],
        [
            "float f = float(iin[i]);",
            "float2x2 m = transpose(float2x2(f, 1.0, 2.0, 3.0));",
            "float2x2 n = transpose(float2x2(1.0, f, 0.5, 2.0));",
            "float2 v = mul(mul(m, n), float2(1.0, f));",
            "outp[i] = v.x + 2.0 * v.y;",
        ],
    ),
    "vector-matrix-product": (
        [("int", "iin"), ("float", "outp")],
        [
            "float f = float(iin[i]);",
            "float2x2 m = transpose(float2x2(f, 1.0, 2.0, 3.0));",
            "float2 v = mul(float2(1.0, f), m);",
            "outp[i] = v.x + 2.0 * v.y;",
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(_HLSL_BY_COLUMNS))
def test_the_hlsl_matrix_probes_compute_what_the_crossgl_probes_compute(name):
    # HLSL fills and indexes a matrix by rows and CrossGL by columns, so the probes must write them differently.
    if not lowering.can_lower("hlsl"):
        pytest.skip("dxc or spirv-dis is not installed")
    feature = construct_feature("hlsl", name)
    native = probes.build(feature, "hlsl").source
    by_columns = probes.DIALECTS["hlsl"].kernel(*_HLSL_BY_COLUMNS[name])
    probe = Result(feature, "output", "supported", sources=(native, by_columns))
    [result], used = execution.run([probe], workers=1)
    if not used:
        pytest.skip("no Vulkan runtime")
    assert (result.outcome, result.runtime) == ("supported", used[0])

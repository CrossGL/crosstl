import re
from typing import List

import pytest

import crosstl.translator
from crosstl.translator.ast import (
    ArrayNode,
    BlockNode,
    ExecutionModel,
    FunctionNode,
    PrimitiveType,
    ShaderNode,
    ShaderStage,
    StructMemberNode,
    StructNode,
    create_legacy_shader_node,
)
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = Lexer(code)
    return lexer.get_tokens()


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST.

    Args:
        tokens (List): The list of tokens to parse
    Returns:
        AST: The abstract syntax tree generated from the parser
    """
    parser = Parser(tokens)
    return parser.parse()


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = HLSLCodeGen()
    return codegen.generate(ast_node)


HLSL_SCALAR_VECTOR_ZERO_DIAGNOSTIC = re.compile(
    r"(?:(?:\b(?:float|double|half|min16float|min10float|int|uint|"
    r"min16int|min16uint|bool)[234]\((?:0(?:\.0)?|0u|false|true)\)"
    r"\s*/\*\s*unsupported\s+(?:HLSL|DirectX)\b)|"
    r"(?:/\*\s*unsupported\s+(?:HLSL|DirectX)\b[^*]*\*/\s*"
    r"\b(?:float|double|half|min16float|min10float|int|uint|"
    r"min16int|min16uint|bool)[234]\((?:0(?:\.0)?|0u|false|true)\)))"
)


def hlsl_scalar_vector_zero_diagnostics(code: str):
    return [
        match.group(0) for match in HLSL_SCALAR_VECTOR_ZERO_DIAGNOSTIC.finditer(code)
    ]


def test_directx_synchronization_builtins_lower_to_hlsl_intrinsics():
    shader = """
    shader SynchronizationBuiltins {
        compute {
            void main() {
                barrier();
                memoryBarrier();
                workgroupBarrier();
                groupMemoryBarrier();
                memoryBarrierShared();
                deviceMemoryBarrier();
                memoryBarrierBuffer();
                memoryBarrierImage();
                allMemoryBarrier();
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert generated_code.count("GroupMemoryBarrierWithGroupSync();") == 2
    assert generated_code.count("GroupMemoryBarrier();") == 2
    assert generated_code.count("DeviceMemoryBarrier();") == 3
    assert generated_code.count("AllMemoryBarrier();") == 2
    assert "barrier();" not in generated_code
    assert "memoryBarrier();" not in generated_code
    assert "workgroupBarrier();" not in generated_code
    assert "groupMemoryBarrier();" not in generated_code
    assert "memoryBarrierShared();" not in generated_code
    assert "deviceMemoryBarrier();" not in generated_code
    assert "memoryBarrierBuffer();" not in generated_code
    assert "memoryBarrierImage();" not in generated_code
    assert "allMemoryBarrier();" not in generated_code


def test_directx_user_defined_synchronization_names_are_not_lowered():
    shader = """
    shader SynchronizationShadowing {
        compute {
            void barrier() {
                return;
            }

            void memoryBarrier() {
                return;
            }

            void workgroupBarrier() {
                return;
            }

            void groupMemoryBarrier() {
                return;
            }

            void memoryBarrierBuffer() {
                return;
            }

            void memoryBarrierImage() {
                return;
            }

            void main() {
                barrier();
                memoryBarrier();
                workgroupBarrier();
                groupMemoryBarrier();
                memoryBarrierBuffer();
                memoryBarrierImage();
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "void barrier()" in generated_code
    assert "void memoryBarrier()" in generated_code
    assert "void workgroupBarrier()" in generated_code
    assert "void groupMemoryBarrier()" in generated_code
    assert "void memoryBarrierBuffer()" in generated_code
    assert "void memoryBarrierImage()" in generated_code
    assert "barrier();" in generated_code
    assert "memoryBarrier();" in generated_code
    assert "workgroupBarrier();" in generated_code
    assert "groupMemoryBarrier();" in generated_code
    assert "memoryBarrierBuffer();" in generated_code
    assert "memoryBarrierImage();" in generated_code
    assert "GroupMemoryBarrierWithGroupSync();" not in generated_code
    assert "GroupMemoryBarrier();" not in generated_code
    assert "DeviceMemoryBarrier();" not in generated_code
    assert "AllMemoryBarrier();" not in generated_code


def test_directx_synchronization_builtins_reject_arguments():
    shader = """
    shader BadSynchronizationBuiltinArgs {
        compute {
            void main() {
                barrier(1);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="synchronization builtin 'barrier' requires 0 argument",
    ):
        generate_code(parse_code(tokenize_code(shader)))


@pytest.mark.parametrize(
    "builtin",
    [
        "barrier",
        "workgroupBarrier",
        "groupMemoryBarrier",
        "memoryBarrierShared",
        "memoryBarrier",
        "allMemoryBarrier",
    ],
)
def test_directx_synchronization_builtins_reject_unsupported_fragment_stages(builtin):
    shader = f"""
    shader BadSynchronizationStage {{
        fragment {{
            void main() {{
                {builtin}();
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            rf"DirectX fragment stage cannot call {re.escape(builtin)}; "
            r".*only valid in compute, mesh, or amplification/task stages"
        ),
    ):
        generate_code(parse_code(tokenize_code(shader)))


def test_directx_device_memory_synchronization_builtins_allow_fragment_stage():
    shader = """
    shader FragmentDeviceMemorySynchronization {
        fragment {
            void main() {
                deviceMemoryBarrier();
                memoryBarrierBuffer();
                memoryBarrierImage();
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert generated_code.count("DeviceMemoryBarrier();") == 3
    assert "deviceMemoryBarrier();" not in generated_code
    assert "memoryBarrierBuffer();" not in generated_code
    assert "memoryBarrierImage();" not in generated_code


@pytest.mark.parametrize(
    "builtin",
    [
        "deviceMemoryBarrier",
        "memoryBarrierBuffer",
        "memoryBarrierImage",
    ],
)
def test_directx_device_memory_synchronization_rejects_vertex_stage(builtin):
    shader = f"""
    shader BadDeviceMemorySynchronizationStage {{
        vertex {{
            void main() {{
                {builtin}();
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            rf"DirectX vertex stage cannot call {re.escape(builtin)}; "
            r".*valid in fragment/pixel, compute, mesh, or amplification/task stages"
        ),
    ):
        generate_code(parse_code(tokenize_code(shader)))


def test_directx_synchronization_builtins_reject_transitive_non_compute_calls():
    shader = """
    shader TransitiveSynchronizationStage {
        void synchronize_workgroup() {
            workgroupBarrier();
        }

        fragment {
            void main() {
                synchronize_workgroup();
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            r"DirectX fragment stage cannot call synchronize_workgroup; "
            r"'synchronize_workgroup' reaches 'workgroupBarrier'.*"
            r"compute, mesh, or amplification/task stages"
        ),
    ):
        generate_code(parse_code(tokenize_code(shader)))


def test_directx_synchronization_helper_calls_remain_compute_compatible():
    shader = """
    shader ComputeSynchronizationHelper {
        void synchronize_workgroup() {
            workgroupBarrier();
        }

        compute {
            void main() {
                synchronize_workgroup();
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "void synchronize_workgroup()" in generated_code
    assert "GroupMemoryBarrierWithGroupSync();" in generated_code
    assert "synchronize_workgroup();" in generated_code


def test_directx_synchronization_helper_calls_remain_mesh_compatible():
    shader = """
    shader MeshSynchronizationHelper {
        void synchronize_workgroup() {
            groupMemoryBarrier();
        }

        mesh {
            void main() @numthreads(32, 1, 1) @outputtopology(triangle) {
                synchronize_workgroup();
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert '[shader("mesh")]' in generated_code
    assert "GroupMemoryBarrier();" in generated_code
    assert "synchronize_workgroup();" in generated_code


def test_directx_interpolation_builtins_lower_to_hlsl_intrinsics():
    shader = """
    shader InterpolationBuiltins {
        struct FSInput {
            vec4 color;
        };

        vec4 shade(vec4 color, uint sampleIndex, ivec2 offset) {
            vec4 atSample = interpolateAtSample(color, sampleIndex);
            vec4 atOffset = interpolateAtOffset(color, offset);
            vec4 atCentroid = interpolateAtCentroid(color);
            return atSample + atOffset + atCentroid;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return shade(input.color, 0u, ivec2(1, -1));
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "EvaluateAttributeAtSample(color, sampleIndex)" in generated_code
    assert "EvaluateAttributeSnapped(color, offset)" in generated_code
    assert "EvaluateAttributeCentroid(color)" in generated_code
    assert "interpolateAtSample(" not in generated_code
    assert "interpolateAtOffset(" not in generated_code
    assert "interpolateAtCentroid(" not in generated_code


def test_directx_user_defined_interpolation_names_are_not_lowered():
    shader = """
    shader InterpolationShadowing {
        vec4 interpolateAtSample(vec4 color, uint sampleIndex) {
            return color + vec4(float(sampleIndex));
        }

        fragment {
            vec4 main() @ gl_FragColor {
                return interpolateAtSample(vec4(1.0), 0u);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert (
        "float4 interpolateAtSample(float4 color, uint sampleIndex)" in generated_code
    )
    assert "interpolateAtSample(float4(1.0, 1.0, 1.0, 1.0), 0u)" in generated_code
    assert "EvaluateAttributeAtSample(" not in generated_code


def test_directx_interpolation_builtins_reject_wrong_arity():
    shader = """
    shader BadInterpolationBuiltinArgs {
        vec4 shade(vec4 color) {
            return interpolateAtCentroid(color, 0);
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "DirectX interpolation builtin 'interpolateAtCentroid' requires "
            "1 argument"
        ),
    ):
        generate_code(parse_code(tokenize_code(shader)))


def test_hlsl_float16_ir_aliases_map_to_half_and_min_precision_names():
    shader = """
    shader Float16IRSmoke {
        float16 tone(float16 input) {
            float16 bias = float16(0.5);
            return input + bias;
        }

        f16vec2 pair(f16vec2 input) {
            f16vec2 scale = f16vec2(1.0, 2.0);
            return input * scale;
        }

        i16vec2 signedPair(i16vec2 input) {
            i16vec2 inc = i16vec2(1, 2);
            return input + inc;
        }

        u16vec2 unsignedPair(u16vec2 input) {
            u16vec2 inc = u16vec2(1u, 2u);
            return input + inc;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "half tone(half input)" in generated_code
    assert "half bias = half(0.5);" in generated_code
    assert "half2 pair(half2 input)" in generated_code
    assert "half2 scale = half2(1.0, 2.0);" in generated_code
    assert "min16int2 signedPair(min16int2 input)" in generated_code
    assert "min16int2 inc = min16int2(1, 2);" in generated_code
    assert "min16uint2 unsignedPair(min16uint2 input)" in generated_code
    assert "min16uint2 inc = min16uint2(1u, 2u);" in generated_code
    for invalid_token in ("float16", "f16vec2", "i16vec2", "u16vec2"):
        assert invalid_token not in generated_code


def test_hlsl_float16_matrix_ir_aliases_map_to_half_matrices():
    shader = """
    shader Float16MatrixIRSmoke {
        f16mat3x2 passMatrix(f16mat3x2 input) {
            f16mat3x2 m = f16mat3x2(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);
            return input;
        }

        f16mat3 passSquare(f16mat3 input) {
            f16mat3 m = f16mat3(1.0);
            return input * m;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "half2x3 passMatrix(half2x3 input)" in generated_code
    assert "half2x3 m = half2x3(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);" in generated_code
    assert "half3x3 passSquare(half3x3 input)" in generated_code
    assert "half3x3 m = half3x3(1.0);" in generated_code
    assert "return (input * m);" in generated_code
    assert "f16mat3x2" not in generated_code
    assert "f16mat3" not in generated_code


def test_hlsl_vector_constructor_scalar_splats_expand_for_dxc():
    shader = """
    shader HLSLVectorSplat {
        compute {
            void main() {
                vec2 uv = vec2(0.5);
                float2 aliasUv = float2(0.25);
                ivec3 cell = ivec3(1);
                uvec4 mask = uvec4(2u);
                bvec2 flags = bvec2(true);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "float2 uv = float2(0.5, 0.5);" in generated_code
    assert "float2 aliasUv = float2(0.25, 0.25);" in generated_code
    assert "int3 cell = int3(1, 1, 1);" in generated_code
    assert "uint4 mask = uint4(2u, 2u, 2u, 2u);" in generated_code
    assert "bool2 flags = bool2(true, true);" in generated_code


def test_hlsl_diagnostic_scalar_vector_zero_scanner_flags_single_arg_fallbacks():
    bad_hlsl = """
    float4 color = float4(0.0) /* unsupported HLSL GLSL buffer block access block */;
    float2 lod = /* unsupported DirectX texture query: textureQueryLod */ float2(0.0);
    uint3 counts = uint3(0u) /* unsupported HLSL GLSL buffer block access block */;
    """

    assert hlsl_scalar_vector_zero_diagnostics(bad_hlsl) == [
        "float4(0.0) /* unsupported HLSL",
        "/* unsupported DirectX texture query: textureQueryLod */ float2(0.0)",
        "uint3(0u) /* unsupported HLSL",
    ]


def test_hlsl_diagnostic_vector_zero_fallbacks_expand_for_dxc():
    shader = """
    shader DiagnosticVectorFallbacks {
        struct UnsupportedDiagnosticBlock {
            double flag;
            vec2 uv;
            dvec2 precise;
            vec3 normal;
            vec4 color;
            ivec2 pixel;
            ivec3 voxel;
            uvec2 counts;
            uvec4 mask;
            bvec2 predicates;
        };

        UnsupportedDiagnosticBlock block @glsl_buffer_block(std430) @binding(77);

        vec3 readNormal(UnsupportedDiagnosticBlock localBlock @glsl_buffer_block(std430)) {
            return localBlock.normal;
        }

        fragment {
            vec4 main() @ gl_FragColor {
                vec2 uv = block.uv;
                dvec2 precise = block.precise;
                vec3 normal = readNormal(block);
                vec4 color = block.color;
                ivec2 pixel = block.pixel;
                ivec3 voxel = block.voxel;
                uvec2 counts = block.counts;
                uvec4 mask = block.mask;
                bvec2 predicates = block.predicates;
                float boolValue = predicates.x ? 1.0 : 0.0;
                return color + vec4(
                    uv.x + float(precise.x) + normal.x +
                    float(pixel.x + voxel.x) + float(counts.x + mask.x) +
                    boolValue
                );
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "float2 uv = float2(0.0, 0.0) /* unsupported HLSL" in generated_code
    assert "double2 precise = double2(0.0, 0.0) /* unsupported HLSL" in generated_code
    assert "float3 normal = float3(0.0, 0.0, 0.0) /* unsupported HLSL" in generated_code
    assert (
        "float4 color = float4(0.0, 0.0, 0.0, 0.0) /* unsupported HLSL"
        in generated_code
    )
    assert "int2 pixel = int2(0, 0) /* unsupported HLSL" in generated_code
    assert "int3 voxel = int3(0, 0, 0) /* unsupported HLSL" in generated_code
    assert "uint2 counts = uint2(0u, 0u) /* unsupported HLSL" in generated_code
    assert "uint4 mask = uint4(0u, 0u, 0u, 0u) /* unsupported HLSL" in generated_code
    assert (
        "bool2 predicates = bool2(false, false) /* unsupported HLSL" in generated_code
    )
    assert hlsl_scalar_vector_zero_diagnostics(generated_code) == []


def test_hlsl_narrow_integer_aliases_map_to_valid_hlsl_integer_types():
    shader = """
    shader NarrowIntegerAliasSmoke {
        int8 signedScalar(int8 input) {
            int8 one = int8(1);
            return input + one;
        }

        uint8 unsignedScalar(uint8 input) {
            uint8 one = uint8(1u);
            return input + one;
        }

        i8vec2 signedPair(i8vec2 input) {
            i8vec2 inc = i8vec2(1, 2);
            return input + inc;
        }

        u8vec3 unsignedTriple(u8vec3 input) {
            u8vec3 inc = u8vec3(1u, 2u, 3u);
            return input + inc;
        }

        short2 signedShort(short2 input) {
            short2 inc = short2(1, 2);
            return input + inc;
        }

        ushort3 unsignedShort(ushort3 input) {
            ushort3 inc = ushort3(1u, 2u, 3u);
            return input + inc;
        }

        char4 signedChar(char4 input) {
            char4 inc = char4(1, 2, 3, 4);
            return input + inc;
        }

        uchar2 unsignedChar(uchar2 input) {
            uchar2 inc = uchar2(1u, 2u);
            return input + inc;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "int signedScalar(int input)" in generated_code
    assert "int one = int(1);" in generated_code
    assert "uint unsignedScalar(uint input)" in generated_code
    assert "uint one = uint(1u);" in generated_code
    assert "int2 signedPair(int2 input)" in generated_code
    assert "int2 inc = int2(1, 2);" in generated_code
    assert "uint3 unsignedTriple(uint3 input)" in generated_code
    assert "uint3 inc = uint3(1u, 2u, 3u);" in generated_code
    assert "min16int2 signedShort(min16int2 input)" in generated_code
    assert "min16int2 inc = min16int2(1, 2);" in generated_code
    assert "min16uint3 unsignedShort(min16uint3 input)" in generated_code
    assert "min16uint3 inc = min16uint3(1u, 2u, 3u);" in generated_code
    assert "int4 signedChar(int4 input)" in generated_code
    assert "int4 inc = int4(1, 2, 3, 4);" in generated_code
    assert "uint2 unsignedChar(uint2 input)" in generated_code
    assert "uint2 inc = uint2(1u, 2u);" in generated_code
    for invalid_token in (
        "int8",
        "uint8",
        "i8vec2",
        "u8vec3",
        "short2",
        "ushort3",
        "char4",
        "uchar2",
    ):
        assert invalid_token not in generated_code


def test_hlsl_packed_vector_aliases_map_to_standard_hlsl_vectors():
    shader = """
    shader PackedVectorAliasSmoke {
        packed_float4 color(packed_float4 input) {
            packed_float4 bias = packed_float4(1.0, 2.0, 3.0, 4.0);
            return input + bias;
        }

        packed_half2 halfPair(packed_half2 input) {
            packed_half2 scale = packed_half2(1.0, 2.0);
            return input * scale;
        }

        packed_int3 signedTriple(packed_int3 input) {
            packed_int3 inc = packed_int3(1, 2, 3);
            return input + inc;
        }

        packed_uint4 unsignedQuad(packed_uint4 input) {
            packed_uint4 inc = packed_uint4(1u, 2u, 3u, 4u);
            return input + inc;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "float4 color(float4 input)" in generated_code
    assert "float4 bias = float4(1.0, 2.0, 3.0, 4.0);" in generated_code
    assert "half2 halfPair(half2 input)" in generated_code
    assert "half2 scale = half2(1.0, 2.0);" in generated_code
    assert "int3 signedTriple(int3 input)" in generated_code
    assert "int3 inc = int3(1, 2, 3);" in generated_code
    assert "uint4 unsignedQuad(uint4 input)" in generated_code
    assert "uint4 inc = uint4(1u, 2u, 3u, 4u);" in generated_code
    assert "packed_" not in generated_code


def test_hlsl_simd_aliases_map_to_standard_hlsl_types():
    shader = """
    shader SimdAliasSmoke {
        simd_float4 color(simd_float4 input) {
            simd_float4 bias = simd_float4(1.0, 2.0, 3.0, 4.0);
            return input + bias;
        }

        simd_int3 signedTriple(simd_int3 input) {
            simd_int3 inc = simd_int3(1, 2, 3);
            return input + inc;
        }

        simd_uint2 unsignedPair(simd_uint2 input) {
            simd_uint2 inc = simd_uint2(1u, 2u);
            return input + inc;
        }

        simd_float4x4 passSquare(simd_float4x4 input) {
            simd_float4x4 m = simd_float4x4(1.0);
            return input * m;
        }

        simd_float3x2 passMatrix(simd_float3x2 input) {
            simd_float3x2 m = simd_float3x2(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);
            return input;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "float4 color(float4 input)" in generated_code
    assert "float4 bias = float4(1.0, 2.0, 3.0, 4.0);" in generated_code
    assert "int3 signedTriple(int3 input)" in generated_code
    assert "int3 inc = int3(1, 2, 3);" in generated_code
    assert "uint2 unsignedPair(uint2 input)" in generated_code
    assert "uint2 inc = uint2(1u, 2u);" in generated_code
    assert "float4x4 passSquare(float4x4 input)" in generated_code
    assert "float4x4 m = float4x4(1.0);" in generated_code
    assert "float3x2 passMatrix(float3x2 input)" in generated_code
    assert "float3x2 m = float3x2(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);" in generated_code
    assert "simd_" not in generated_code


def test_hlsl_fixed_width_scalar_aliases_map_to_valid_hlsl_scalars():
    shader = """
    shader FixedWidthScalarAliasSmoke {
        int8_t signedByte(int8_t input) {
            int8_t one = int8_t(1);
            return input + one;
        }

        uint8_t unsignedByte(uint8_t input) {
            uint8_t one = uint8_t(1u);
            return input + one;
        }

        int16_t signedShortScalar(int16_t input) {
            int16_t one = int16_t(1);
            return input + one;
        }

        uint16_t unsignedShortScalar(uint16_t input) {
            uint16_t one = uint16_t(1u);
            return input + one;
        }

        int32_t signedWord(int32_t input) {
            int32_t one = int32_t(1);
            return input + one;
        }

        uint32_t unsignedWord(uint32_t input) {
            uint32_t one = uint32_t(1u);
            return input + one;
        }

        int64 signedLong(int64 input) {
            int64 one = int64(1);
            return input + one;
        }

        uint64 unsignedLong(uint64 input) {
            uint64 one = uint64(1u);
            return input + one;
        }

        int64_t signedLongT(int64_t input) {
            int64_t one = int64_t(1);
            return input + one;
        }

        uint64_t unsignedLongT(uint64_t input) {
            uint64_t one = uint64_t(1u);
            return input + one;
        }

        size_t sizeValue(size_t input) {
            size_t one = size_t(1u);
            return input + one;
        }

        ptrdiff_t ptrDiff(ptrdiff_t input) {
            ptrdiff_t one = ptrdiff_t(1);
            return input + one;
        }

        long longValue(long input) {
            long one = long(1);
            return input + one;
        }

        ulong ulongValue(ulong input) {
            ulong one = ulong(1u);
            return input + one;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "int signedByte(int input)" in generated_code
    assert "uint unsignedByte(uint input)" in generated_code
    assert "min16int signedShortScalar(min16int input)" in generated_code
    assert "min16uint unsignedShortScalar(min16uint input)" in generated_code
    assert "int signedWord(int input)" in generated_code
    assert "uint unsignedWord(uint input)" in generated_code
    assert "int64_t signedLong(int64_t input)" in generated_code
    assert "uint64_t unsignedLong(uint64_t input)" in generated_code
    assert "int64_t signedLongT(int64_t input)" in generated_code
    assert "uint64_t unsignedLongT(uint64_t input)" in generated_code
    assert "uint64_t sizeValue(uint64_t input)" in generated_code
    assert "int64_t ptrDiff(int64_t input)" in generated_code
    assert "int64_t longValue(int64_t input)" in generated_code
    assert "uint64_t ulongValue(uint64_t input)" in generated_code
    assert "int one = int(1);" in generated_code
    assert "uint one = uint(1u);" in generated_code
    assert "min16int one = min16int(1);" in generated_code
    assert "min16uint one = min16uint(1u);" in generated_code
    assert "int64_t one = int64_t(1);" in generated_code
    assert "uint64_t one = uint64_t(1u);" in generated_code
    for invalid_token in (
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
        "int64 ",
        "uint64 ",
        "size_t",
        "ptrdiff_t",
        "long ",
        "ulong ",
    ):
        assert invalid_token not in generated_code


def test_hlsl_fixed_width_scalar_array_aliases_map_in_aggregate_declarations():
    shader = """
    shader AliasArrayAggregateSmoke {
        int8_t globalBytes[2];
        uint64_t globalCounters[2];

        struct AliasPayload {
            int8_t bytes[2];
            uint16_t words[3];
            int64_t signedValue;
            size_t offsets[2];
        };

        cbuffer AliasConstants @register(b1) {
            int16_t smalls[2];
            uint32_t count;
            ptrdiff_t delta;
        };

        uint64_t bump(uint64_t counters[2], AliasPayload payload) {
            uint64_t localCounters[2];
            uint64_t one = uint64_t(1u);
            localCounters[0] = counters[0] + one;
            return localCounters[0] + payload.offsets[1];
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "int bytes[2];" in generated_code
    assert "min16uint words[3];" in generated_code
    assert "int64_t signedValue;" in generated_code
    assert "uint64_t offsets[2];" in generated_code
    assert "int globalBytes[2];" in generated_code
    assert "uint64_t globalCounters[2];" in generated_code
    assert "min16int smalls[2];" in generated_code
    assert "uint count;" in generated_code
    assert "int64_t delta;" in generated_code
    assert "uint64_t bump(uint64_t counters[2], AliasPayload payload)" in generated_code
    assert "uint64_t localCounters[2];" in generated_code
    assert "uint64_t one = uint64_t(1u);" in generated_code
    for invalid_token in (
        "int8_t",
        "uint16_t",
        "int16_t",
        "uint32_t",
        "int64 ",
        "uint64 ",
        "size_t",
        "ptrdiff_t",
    ):
        assert invalid_token not in generated_code


def test_hlsl_fixed_width_nested_array_aliases_map_to_valid_hlsl_types():
    shader = """
    shader AliasNestedArraySmoke {
        struct AliasGridPayload {
            int8_t bytes[2][3];
            size_t offsets[2][3];
        };

        uint16_t bumpNested(uint16_t values[2][3], int row, int col) {
            uint16_t grid[2][3];
            grid[row][col] = values[row][col] + uint16_t(1u);
            return grid[row][col];
        }

        ptrdiff_t readPayload(AliasGridPayload payload, int row, int col) {
            return ptrdiff_t(payload.bytes[row][col]) +
                ptrdiff_t(payload.offsets[row][col]);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "int bytes[2][3];" in generated_code
    assert "uint64_t offsets[2][3];" in generated_code
    assert (
        "min16uint bumpNested(min16uint values[2][3], int row, int col)"
        in generated_code
    )
    assert "min16uint grid[2][3];" in generated_code
    assert "min16uint(1u)" in generated_code
    assert (
        "int64_t readPayload(AliasGridPayload payload, int row, int col)"
        in generated_code
    )
    assert "int64_t(payload.bytes[row][col])" in generated_code
    assert "int64_t(payload.offsets[row][col])" in generated_code
    assert "int[2] bytes[3]" not in generated_code
    assert "uint64_t[2] offsets[3]" not in generated_code
    for invalid_token in (
        "int8_t",
        "uint16_t",
        "uint64 ",
        "size_t",
        "ptrdiff_t",
    ):
        assert invalid_token not in generated_code


def test_hlsl_array_return_aliases_are_rejected_with_clear_diagnostic():
    shader = """
    shader AliasReturnArraySmoke {
        uint64_t[2] makeCounters() {
            uint64_t counters[2];
            return counters;
        }
    }
    """

    with pytest.raises(
        ValueError, match="DirectX output does not support array return"
    ):
        generate_code(parse_code(tokenize_code(shader)))


def test_hlsl_fixed_width_nested_cbuffer_array_aliases_map_to_valid_types():
    shader = """
    shader AliasNestedCbufferArraySmoke {
        cbuffer AliasNestedConstants @register(b2) {
            int64_t signedGrid[2][3];
            uint16_t unsignedGrid[2][3];
            size_t offsets[2][3];
        };

        uint64_t readConstant(int row, int col) {
            return uint64_t(offsets[row][col]) + uint64_t(unsignedGrid[row][col]);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "cbuffer AliasNestedConstants : register(b2)" in generated_code
    assert "int64_t signedGrid[2][3];" in generated_code
    assert "min16uint unsignedGrid[2][3];" in generated_code
    assert "uint64_t offsets[2][3];" in generated_code
    assert "uint64_t readConstant(int row, int col)" in generated_code
    assert "uint64_t(offsets[row][col])" in generated_code
    assert "uint64_t(unsignedGrid[row][col])" in generated_code
    assert "int64_t[2] signedGrid[3]" not in generated_code
    assert "min16uint[2] unsignedGrid[3]" not in generated_code
    for invalid_token in (
        "uint16_t",
        "uint64 ",
        "size_t",
    ):
        assert invalid_token not in generated_code


def test_hlsl_structured_buffer_fixed_width_aliases_map_resource_generics():
    shader = """
    shader AliasStructuredBufferHLSL {
        RWStructuredBuffer<uint16_t> counts @ binding(3);
        StructuredBuffer<int64_t> signedValues @ binding(4);
        RWStructuredBuffer<size_t> offsets @ binding(5);

        uint64_t loadOffset(uint index) {
            uint64_t offset = buffer_load(offsets, index);
            uint16_t count = buffer_load(counts, index);
            int64_t signedValue = buffer_load(signedValues, index);
            buffer_store(counts, index, count + uint16_t(1u));
            return offset + uint64_t(count) + uint64_t(signedValue);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "RWStructuredBuffer<min16uint> counts : register(u3);" in generated_code
    assert "StructuredBuffer<int64_t> signedValues : register(t4);" in generated_code
    assert "RWStructuredBuffer<uint64_t> offsets : register(u5);" in generated_code
    assert "min16uint count = counts.Load(index);" in generated_code
    assert "counts.Store(index, (count + min16uint(1u)));" in generated_code
    assert "RWStructuredBuffer<uint16_t>" not in generated_code
    assert "RWStructuredBuffer<size_t>" not in generated_code
    assert "size_t" not in generated_code


def test_hlsl_structured_buffer_alias_arrays_infer_helper_parameter_sizes():
    shader = """
    shader AliasStructuredBufferArrayHLSL {
        RWStructuredBuffer<uint16_t> counts[2] @ binding(3);
        StructuredBuffer<size_t> offsets[2] @ binding(5);

        uint16_t readCount(RWStructuredBuffer<uint16_t> localCounts[], uint which, uint index) {
            return buffer_load(localCounts[which], index);
        }

        uint64_t readOffset(StructuredBuffer<size_t> localOffsets[2], uint which, uint index) {
            return buffer_load(localOffsets[which], index);
        }

        uint64_t combine(uint which, uint index) {
            uint16_t count = readCount(counts, which, index);
            uint64_t offset = readOffset(offsets, which, index);
            buffer_store(counts[which], index, count + uint16_t(1u));
            return offset + uint64_t(count);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "RWStructuredBuffer<min16uint> counts[2] : register(u3);" in generated_code
    assert "StructuredBuffer<uint64_t> offsets[2] : register(t5);" in generated_code
    assert (
        "min16uint readCount(RWStructuredBuffer<min16uint> localCounts[2], uint which, uint index)"
        in generated_code
    )
    assert (
        "uint64_t readOffset(StructuredBuffer<uint64_t> localOffsets[2], uint which, uint index)"
        in generated_code
    )
    assert "return localCounts[which].Load(index);" in generated_code
    assert "counts[which].Store(index, (count + min16uint(1u)));" in generated_code
    assert "localCounts[]" not in generated_code
    assert "uint16_t" not in generated_code
    assert "size_t" not in generated_code


def test_hlsl_structured_buffer_array_u_register_overlap_raises():
    shader = """
    shader StructuredBufferArrayRegisterOverlapHLSL {
        RWStructuredBuffer<int> first[2] @ binding(3);
        RWStructuredBuffer<int> second @ binding(4);

        int read(uint index) {
            return buffer_load(first[1], index) + buffer_load(second, index);
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting DirectX resource binding for 'second'",
    ):
        generate_code(parse_code(tokenize_code(shader)))


def test_hlsl_unsized_structured_buffer_global_allows_multiple_fixed_helper_sizes():
    shader = """
    shader UnsizedStructuredBufferMultipleFixedHelpersHLSL {
        RWStructuredBuffer<uint> counts[] @ binding(3);

        uint readThree(RWStructuredBuffer<uint> localCounts[3], uint index) {
            return buffer_load(localCounts[2], index);
        }

        uint readFour(RWStructuredBuffer<uint> localCounts[4], uint index) {
            return buffer_load(localCounts[3], index);
        }

        uint combine(uint index) {
            return readThree(counts, index) + readFour(counts, index);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "RWStructuredBuffer<uint> counts[] : register(u3);" in generated_code
    assert (
        "uint readThree(RWStructuredBuffer<uint> localCounts[3], uint index)"
        in generated_code
    )
    assert (
        "uint readFour(RWStructuredBuffer<uint> localCounts[4], uint index)"
        in generated_code
    )
    assert "return localCounts[2].Load(index);" in generated_code
    assert "return localCounts[3].Load(index);" in generated_code
    assert (
        "return (readThree(counts, index) + readFour(counts, index));" in generated_code
    )


def test_hlsl_structured_buffer_array_helpers_propagate_nested_fixed_sizes():
    shader = """
    shader NestedStructuredBufferArrayHelpersHLSL {
        RWStructuredBuffer<uint16_t> counts[] @ binding(3);
        RWStructuredBuffer<uint16_t> afterCounts;

        uint16_t readLeaf(RWStructuredBuffer<uint16_t> leafCounts[3], uint index) {
            return buffer_load(leafCounts[2], index);
        }

        uint16_t readMid(RWStructuredBuffer<uint16_t> midCounts[], uint index) {
            return readLeaf(midCounts, index);
        }

        uint16_t combine(uint index) {
            return readMid(counts, index) + buffer_load(afterCounts, index);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "RWStructuredBuffer<min16uint> counts[] : register(u3);" in generated_code
    assert "RWStructuredBuffer<min16uint> afterCounts : register(u4);" in generated_code
    assert (
        "min16uint readLeaf(RWStructuredBuffer<min16uint> leafCounts[3], uint index)"
        in generated_code
    )
    assert (
        "min16uint readMid(RWStructuredBuffer<min16uint> midCounts[3], uint index)"
        in generated_code
    )
    assert "return leafCounts[2].Load(index);" in generated_code
    assert "return readLeaf(midCounts, index);" in generated_code
    assert (
        "return (readMid(counts, index) + afterCounts.Load(index));" in generated_code
    )
    assert "RWStructuredBuffer<uint16_t>" not in generated_code


def test_hlsl_structured_buffer_array_nested_helpers_reject_mixed_fixed_sizes():
    shader = """
    shader NestedStructuredBufferArrayConflictHLSL {
        RWStructuredBuffer<uint> counts[] @ binding(3);

        uint readLeaf(RWStructuredBuffer<uint> leafCounts[3], uint index) {
            return buffer_load(leafCounts[2], index);
        }

        uint readMid(RWStructuredBuffer<uint> midCounts[4], uint index) {
            return readLeaf(midCounts, index);
        }

        uint combine(uint index) {
            return readMid(counts, index);
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'midCounts': 4 and 3",
    ):
        generate_code(parse_code(tokenize_code(shader)))


def test_hlsl_glsl_buffer_block_fixed_width_aliases_lower_to_byteaddressbuffer():
    shader = """
    shader AliasGlslBufferBlockHLSL {
        struct AliasBlock {
            uint16_t words[2];
            size_t offsets[];
        };

        AliasBlock aliasBlock @glsl_buffer_block(std430) @binding(7);

        uint64_t readAlias(uint index) {
            uint16_t word = aliasBlock.words[1];
            return uint64_t(word) + uint64_t(aliasBlock.offsets[index]);
        }

        void writeAlias(uint index, uint16_t word, size_t offset) {
            aliasBlock.words[1] = word;
            aliasBlock.offsets[index] = offset;
            aliasBlock.words[0] += uint16_t(1u);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "RWByteAddressBuffer aliasBlock : register(u7);" in generated_code
    assert "min16uint word = aliasBlock.Load(4);" in generated_code
    assert "uint64_t(aliasBlock.Load((8 + index * 4)))" in generated_code
    assert (
        "void writeAlias(uint index, min16uint word, uint64_t offset)" in generated_code
    )
    assert "aliasBlock.Store(4, uint(word));" in generated_code
    assert "aliasBlock.Store((8 + index * 4), uint(offset));" in generated_code
    assert (
        "aliasBlock.Store(0, uint((aliasBlock.Load(0) + min16uint(1u))));"
        in generated_code
    )
    assert "unsupported HLSL GLSL buffer block" not in generated_code
    assert "uint16_t" not in generated_code
    assert "size_t" not in generated_code


def test_directx_byte_address_vector_buffer_helpers_lower_to_load_store_methods():
    shader = """
    shader ByteAddressVectorHelpersHLSL {
        ByteAddressBuffer rawInput @register(t3);
        RWByteAddressBuffer rawOutput @register(u4);

        uvec2 readPair(ByteAddressBuffer raw, uint offset) {
            return buffer_load2(raw, offset);
        }

        uvec3 readTriple(RWByteAddressBuffer raw, uint offset) {
            return buffer_load3(raw, offset + uint(16));
        }

        uvec4 readQuad(RWByteAddressBuffer raw, uint offset) {
            return buffer_load4(raw, offset + uint(32));
        }

        void writeVectors(
            RWByteAddressBuffer raw,
            uint offset,
            uvec2 pair,
            uvec3 triple,
            uvec4 quad
        ) {
            buffer_store2(raw, offset, pair);
            buffer_store3(raw, offset + uint(16), triple);
            buffer_store4(raw, offset + uint(32), quad);
        }

        uvec4 combine(uint offset) {
            uvec2 pair = readPair(rawInput, offset);
            uvec3 triple = readTriple(rawOutput, offset);
            uvec4 quad = readQuad(rawOutput, offset);
            writeVectors(rawOutput, offset, pair, triple, quad);
            return quad;
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "ByteAddressBuffer rawInput : register(t3);" in generated_code
    assert "RWByteAddressBuffer rawOutput : register(u4);" in generated_code
    assert "uint2 readPair(ByteAddressBuffer raw, uint offset)" in generated_code
    assert "return raw.Load2(offset);" in generated_code
    assert "uint3 readTriple(RWByteAddressBuffer raw, uint offset)" in generated_code
    assert "return raw.Load3((offset + uint(16)));" in generated_code
    assert "return raw.Load4((offset + uint(32)));" in generated_code
    assert "raw.Store2(offset, pair);" in generated_code
    assert "raw.Store3((offset + uint(16)), triple);" in generated_code
    assert "raw.Store4((offset + uint(32)), quad);" in generated_code
    assert "buffer_load2(" not in generated_code
    assert "buffer_load3(" not in generated_code
    assert "buffer_load4(" not in generated_code
    assert "buffer_store2(" not in generated_code
    assert "buffer_store3(" not in generated_code
    assert "buffer_store4(" not in generated_code


def test_directx_glsl_buffer_block_atomics_lower_to_byteaddress_helpers():
    shader = """
    shader GlslBufferBlockAtomicsHLSL {
        struct AtomicBlock {
            uint counter;
            uint bins[4];
            int signedCounter;
        };

        AtomicBlock atomicBlock @glsl_buffer_block(std430) @binding(17);

        uint update(uint value) {
            uint oldCounter = atomicAdd(atomicBlock.counter, value);
            uint swapped = atomicCompSwap(atomicBlock.bins[1], oldCounter, 7u);
            uint exchanged = atomicCompareExchange(atomicBlock.bins[2], oldCounter, swapped);
            return oldCounter + swapped + exchanged;
        }

        int updateSigned(int value) {
            return atomicMin(atomicBlock.signedCounter, value);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "RWByteAddressBuffer atomicBlock : register(u17);" in generated_code
    assert (
        "uint __crossgl_byteaddress_atomic_add_uint("
        "RWByteAddressBuffer buffer, uint offset, uint value)"
    ) in generated_code
    assert (
        "uint __crossgl_byteaddress_atomic_compare_exchange_uint("
        "RWByteAddressBuffer buffer, uint offset, uint compareValue, uint value)"
    ) in generated_code
    assert (
        "int __crossgl_byteaddress_atomic_min_int("
        "RWByteAddressBuffer buffer, uint offset, int value)"
    ) in generated_code
    assert (
        "uint oldCounter = __crossgl_byteaddress_atomic_add_uint("
        "atomicBlock, 0, value);"
    ) in generated_code
    assert (
        "uint swapped = __crossgl_byteaddress_atomic_compare_exchange_uint("
        "atomicBlock, 8, oldCounter, 7u);"
    ) in generated_code
    assert (
        "uint exchanged = __crossgl_byteaddress_atomic_compare_exchange_uint("
        "atomicBlock, 12, oldCounter, swapped);"
    ) in generated_code
    assert (
        "return __crossgl_byteaddress_atomic_min_int(atomicBlock, 20, value);"
        in generated_code
    )
    assert "atomicAdd(atomicBlock" not in generated_code
    assert "atomicCompSwap(atomicBlock" not in generated_code
    assert "atomicCompareExchange(atomicBlock" not in generated_code


def test_directx_glsl_buffer_block_atomics_are_expression_safe_in_value_contexts():
    shader = """
    shader GlslBufferBlockAtomicValueContextsHLSL {
        struct AtomicBlock {
            uint counter;
            uint bins[4];
            int signedCounter;
        };

        struct Pair {
            uint oldCounter;
            int oldSigned;
        };

        AtomicBlock atomicBlock @glsl_buffer_block(std430) @binding(17);

        uint consume(uint value) {
            return value + 1u;
        }

        int consumeSigned(int value) {
            return value - 1;
        }

        Pair makePair(uint value) {
            return Pair(
                atomicAdd(atomicBlock.counter, value),
                atomicMin(atomicBlock.signedCounter, -1)
            );
        }

        compute {
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {
                uint value = tid.x;
                uint direct = atomicAdd(atomicBlock.counter, value);
                direct = consume(
                    atomicCompareExchange(atomicBlock.bins[0], 1u, direct)
                );
                uint selected = direct != 0u
                    ? atomicExchange(atomicBlock.bins[1], 9u)
                    : atomicOr(atomicBlock.bins[2], 2u);
                Pair pair = Pair(
                    atomicXor(atomicBlock.bins[3], selected),
                    consumeSigned(atomicMax(atomicBlock.signedCounter, -2))
                );
                uint values[2] = {
                    atomicAnd(atomicBlock.bins[0], 3u),
                    atomicOr(atomicBlock.bins[1], 4u)
                };
                pair = makePair(values[0] + values[1]);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    expected_direct = (
        "uint direct = __crossgl_byteaddress_atomic_add_uint(" "atomicBlock, 0, value);"
    )
    assert expected_direct in generated_code
    assert (
        "direct = consume(__crossgl_byteaddress_atomic_compare_exchange_uint("
        "atomicBlock, 4, 1u, direct));"
    ) in generated_code
    assert (
        "__crossgl_byteaddress_atomic_exchange_uint(atomicBlock, 8, 9u)"
        in generated_code
    )
    assert "__crossgl_byteaddress_atomic_or_uint(atomicBlock, 12, 2u)" in generated_code
    assert (
        "Pair pair = Pair(__crossgl_byteaddress_atomic_xor_uint("
        "atomicBlock, 16, selected), consumeSigned("
        "__crossgl_byteaddress_atomic_max_int(atomicBlock, 20, -2)));"
    ) in generated_code
    assert (
        "uint values[2] = {__crossgl_byteaddress_atomic_and_uint("
        "atomicBlock, 4, 3u), __crossgl_byteaddress_atomic_or_uint("
        "atomicBlock, 8, 4u)};"
    ) in generated_code
    assert (
        "return Pair(__crossgl_byteaddress_atomic_add_uint(atomicBlock, 0, value), "
        "__crossgl_byteaddress_atomic_min_int(atomicBlock, 20, -1));"
    ) in generated_code
    assert "unsupported HLSL GLSL buffer block atomic" not in generated_code
    assert "atomicAdd(atomicBlock" not in generated_code
    assert "atomicCompareExchange(atomicBlock" not in generated_code


@pytest.mark.parametrize(
    ("function_body", "match"),
    [
        (
            "float wrong = atomicAdd(atomicBlock.counter, 1u); return 0u;",
            "atomic 'atomicAdd' requires uint result context.*expected float",
        ),
        (
            "uint wrong = atomicMin(atomicBlock.signedCounter, -1); return wrong;",
            "atomic 'atomicMin' requires int result context.*expected uint",
        ),
        (
            "useFloat(atomicAdd(atomicBlock.counter, 1u)); return 0u;",
            "atomic 'atomicAdd' requires uint result context.*expected float",
        ),
        (
            "float wrong = atomicBlock.counter != 0u "
            "? atomicAdd(atomicBlock.counter, 1u) : 0.0; return 0u;",
            "atomic 'atomicAdd' requires uint result context.*expected float",
        ),
        (
            "float wrong; wrong = atomicBlock.counter != 0u "
            "? 0.0 : atomicAdd(atomicBlock.counter, 1u); return 0u;",
            "atomic 'atomicAdd' requires uint result context.*expected float",
        ),
        (
            "float wrong[1] = { atomicAdd(atomicBlock.counter, 1u) }; return 0u;",
            "atomic 'atomicAdd' requires uint result context.*expected float",
        ),
        (
            "FloatPair wrong = FloatPair(atomicAdd(atomicBlock.counter, 1u)); "
            "return 0u;",
            "atomic 'atomicAdd' requires uint result context.*expected float",
        ),
    ],
)
def test_directx_glsl_buffer_block_atomics_validate_result_contexts(
    function_body, match
):
    shader = f"""
    shader BadGlslBufferBlockAtomicResultHLSL {{
        struct AtomicBlock {{
            uint counter;
            int signedCounter;
        }};

        struct FloatPair {{
            float oldCounter;
        }};

        AtomicBlock atomicBlock @glsl_buffer_block(std430) @binding(17);

        float useFloat(float value) {{
            return value;
        }}

        uint update() {{
            {function_body}
        }}
    }}
    """

    with pytest.raises(ValueError, match=match):
        generate_code(parse_code(tokenize_code(shader)))


@pytest.mark.parametrize(
    ("return_type", "return_expr", "match"),
    [
        (
            "float",
            "atomicAdd(atomicBlock.counter, 1u)",
            "atomic 'atomicAdd' requires uint result context.*expected float",
        ),
        (
            "float",
            "atomicBlock.counter != 0u ? atomicAdd(atomicBlock.counter, 1u) : 0.0",
            "atomic 'atomicAdd' requires uint result context.*expected float",
        ),
        (
            "uint",
            "atomicMin(atomicBlock.signedCounter, -1)",
            "atomic 'atomicMin' requires int result context.*expected uint",
        ),
    ],
)
def test_directx_glsl_buffer_block_atomics_validate_return_result_contexts(
    return_type, return_expr, match
):
    shader = f"""
    shader BadGlslBufferBlockAtomicReturnHLSL {{
        struct AtomicBlock {{
            uint counter;
            int signedCounter;
        }};

        AtomicBlock atomicBlock @glsl_buffer_block(std430) @binding(17);

        {return_type} update() {{
            return {return_expr};
        }}
    }}
    """

    with pytest.raises(ValueError, match=match):
        generate_code(parse_code(tokenize_code(shader)))


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            "atomicAdd(atomicBlock.counter, 1u, 2u)",
            "atomic 'atomicAdd' requires 2 argument\\(s\\), got 3",
        ),
        (
            "atomicCompSwap(atomicBlock.counter, 1u)",
            "atomic 'atomicCompSwap' requires 3 argument\\(s\\), got 2",
        ),
        (
            "atomicCompareExchange(atomicBlock.counter, 1u)",
            "atomic 'atomicCompareExchange' requires 3 argument\\(s\\), got 2",
        ),
        (
            "atomicAdd(atomicBlock.counter, 1.0)",
            "atomic 'atomicAdd' value argument must be scalar uint, got float",
        ),
        (
            "atomicCompSwap(atomicBlock.counter, 1u, -1)",
            "atomic 'atomicCompSwap' replacement argument must be scalar uint, got int",
        ),
    ],
)
def test_directx_glsl_buffer_block_atomics_validate_arguments(call, match):
    shader = f"""
    shader BadGlslBufferBlockAtomicHLSL {{
        struct AtomicBlock {{
            uint counter;
        }};

        AtomicBlock atomicBlock @glsl_buffer_block(std430) @binding(17);

        uint update() {{
            return {call};
        }}
    }}
    """

    with pytest.raises(ValueError, match=match):
        generate_code(parse_code(tokenize_code(shader)))


def test_directx_typed_buffer_atomics_lower_to_interlocked_statements():
    shader = """
    shader TypedBufferAtomicsHLSL {
        struct Counter {
            uint value;
            int signedValue;
        };

        RWBuffer<uint> counters @register(u1);
        RWStructuredBuffer<Counter> structuredCounters @register(u2);
        RWBuffer<uint> counterArrays[2] @register(u4);
        RWStructuredBuffer<int> signedCounters @register(u6);

        uint fetchAndAdd(uint index) {
            return atomicAdd(counters[index], 1u);
        }

        uint compareAndSwap(uint index) {
            return atomicCompareExchange(counters[index], 2u, 3u);
        }

        uint addWithBias(uint index) {
            return atomicAdd(counters[index], 5u) + 3u;
        }

        compute {
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {
                uint original = atomicAdd(counters[tid.x], 1u);
                original = atomicCompareExchange(counters[tid.x], 2u, 3u);
                atomicXor(counters[tid.x], 4u, original);
                atomicMin(structuredCounters[tid.x].signedValue, -1);
                atomicAdd(counterArrays[1][tid.x], 1u, original);
                int oldSigned = atomicMax(signedCounters[tid.x], -1);
                uint combined = atomicAdd(counters[tid.x], 5u)
                    + atomicCompareExchange(counters[tid.x], 2u, 3u);
                original = atomicAdd(counterArrays[1][tid.x], 2u) + 4u;
                original = fetchAndAdd(tid.x) + compareAndSwap(tid.x);
                original += addWithBias(tid.x) + combined;
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "RWBuffer<uint> counters : register(u1);" in generated_code
    assert (
        "RWStructuredBuffer<Counter> structuredCounters : register(u2);"
        in generated_code
    )
    assert "RWBuffer<uint> counterArrays[2] : register(u4);" in generated_code
    assert "RWStructuredBuffer<int> signedCounters : register(u6);" in generated_code
    assert "uint original;" in generated_code
    assert "InterlockedAdd(counters[tid.x], 1u, original);" in generated_code
    assert (
        "InterlockedCompareExchange(counters[tid.x], 2u, 3u, original);"
        in generated_code
    )
    assert "uint __crossgl_atomic_return_0;" in generated_code
    assert (
        "InterlockedAdd(counters[index], 1u, __crossgl_atomic_return_0);"
        in generated_code
    )
    assert "return __crossgl_atomic_return_0;" in generated_code
    assert "uint __crossgl_atomic_return_1;" in generated_code
    assert (
        "InterlockedCompareExchange(counters[index], 2u, 3u, __crossgl_atomic_return_1);"
        in generated_code
    )
    assert "return __crossgl_atomic_return_1;" in generated_code
    assert "uint __crossgl_atomic_expr_2;" in generated_code
    assert (
        "InterlockedAdd(counters[index], 5u, __crossgl_atomic_expr_2);"
        in generated_code
    )
    assert "return (__crossgl_atomic_expr_2 + 3u);" in generated_code
    assert "InterlockedXor(counters[tid.x], 4u, original);" in generated_code
    assert (
        "InterlockedMin(structuredCounters[tid.x].signedValue, -1);" in generated_code
    )
    assert "InterlockedAdd(counterArrays[1][tid.x], 1u, original);" in generated_code
    assert "int oldSigned;" in generated_code
    assert "InterlockedMax(signedCounters[tid.x], -1, oldSigned);" in generated_code
    assert "uint __crossgl_atomic_expr_3;" in generated_code
    assert (
        "InterlockedAdd(counters[tid.x], 5u, __crossgl_atomic_expr_3);"
        in generated_code
    )
    assert "uint __crossgl_atomic_expr_4;" in generated_code
    assert (
        "InterlockedCompareExchange(counters[tid.x], 2u, 3u, "
        "__crossgl_atomic_expr_4);"
    ) in generated_code
    assert (
        "uint combined = (__crossgl_atomic_expr_3 + __crossgl_atomic_expr_4);"
        in generated_code
    )
    assert "uint __crossgl_atomic_expr_5;" in generated_code
    assert (
        "InterlockedAdd(counterArrays[1][tid.x], 2u, __crossgl_atomic_expr_5);"
        in generated_code
    )
    assert "original = (__crossgl_atomic_expr_5 + 4u);" in generated_code
    assert "atomicAdd(counters" not in generated_code
    assert "atomicCompareExchange(counters" not in generated_code
    assert "atomicXor(counters" not in generated_code
    assert "atomicMin(structuredCounters" not in generated_code


@pytest.mark.parametrize(
    ("declaration", "call", "match"),
    [
        (
            "float wrongOriginal = 0.0;",
            "atomicAdd(counters[tid.x], 1u, wrongOriginal);",
            "atomic 'atomicAdd' original argument must be scalar uint, got float",
        ),
        (
            "int wrongOriginal = 0;",
            "atomicXor(counters[tid.x], 1u, wrongOriginal);",
            "atomic 'atomicXor' original argument must be scalar uint, got int",
        ),
        (
            "uint wrongOriginal = 0u;",
            "atomicMin(signedCounters[tid.x], -1, wrongOriginal);",
            "atomic 'atomicMin' original argument must be scalar int, got uint",
        ),
        (
            "float wrongOriginal = 0.0;",
            "atomicCompareExchange(counters[tid.x], 2u, 3u, wrongOriginal);",
            (
                "atomic 'atomicCompareExchange' original argument must be scalar "
                "uint, got float"
            ),
        ),
        (
            "uvec2 wrongOriginal = uvec2(0u, 0u);",
            "atomicAdd(counters[tid.x], 1u, wrongOriginal);",
            "atomic 'atomicAdd' original argument must be scalar uint, got "
            "(uvec2|uint2)",
        ),
    ],
)
def test_directx_typed_buffer_atomics_validate_explicit_original_targets(
    declaration, call, match
):
    shader = f"""
    shader BadTypedBufferAtomicOriginalHLSL {{
        RWBuffer<uint> counters @register(u1);
        RWStructuredBuffer<int> signedCounters @register(u2);

        compute {{
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {{
                {declaration}
                {call}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=match):
        generate_code(parse_code(tokenize_code(shader)))


@pytest.mark.parametrize(
    "function_body",
    [
        "uint oldValue = 0u; "
        "atomicAdd(counters[index], 1u, oldValue + 1u); "
        "return oldValue;",
        "uint oldValue = 0u; "
        "uint result = atomicAdd(counters[index], 1u, oldValue + 1u); "
        "return result;",
        "uint oldValue = 0u; "
        "uint result = index != 0u "
        "? atomicAdd(counters[index], 1u, oldValue + 1u) : 0u; "
        "return result;",
        "uint oldValue = 0u; "
        "return useUint(atomicAdd(counters[index], 1u, oldValue + 1u));",
        "uint oldValue = 0u; " "return atomicAdd(counters[index], 1u, oldValue + 1u);",
    ],
)
def test_directx_typed_buffer_atomics_reject_non_lvalue_explicit_originals(
    function_body,
):
    shader = f"""
    shader BadTypedBufferAtomicOriginalLValueHLSL {{
        RWBuffer<uint> counters @register(u1);

        uint useUint(uint value) {{
            return value;
        }}

        uint update(uint index) {{
            {function_body}
        }}

        compute {{
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {{
                uint result = update(tid.x);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "atomic 'atomicAdd' original argument must be an assignable "
            "scalar uint target"
        ),
    ):
        generate_code(parse_code(tokenize_code(shader)))


@pytest.mark.parametrize(
    ("function_body", "match"),
    [
        (
            "float wrong = atomicAdd(counters[index], 1u); return 0u;",
            "atomic 'atomicAdd' requires uint result context.*expected float",
        ),
        (
            "uint wrong = atomicMin(signedCounters[index], -1); return wrong;",
            "atomic 'atomicMin' requires int result context.*expected uint",
        ),
        (
            "float wrong = index != 0u ? atomicAdd(counters[index], 1u) : 0.0; "
            "return 0u;",
            "atomic 'atomicAdd' requires uint result context.*expected float",
        ),
        (
            "useFloat(atomicAdd(counters[index], 1u)); return 0u;",
            "atomic 'atomicAdd' requires uint result context.*expected float",
        ),
        (
            "uvec2 wrong = atomicAdd(counters[index], 1u); return 0u;",
            "atomic 'atomicAdd' requires uint result context.*expected uint2",
        ),
        (
            "bool wrong = atomicAdd(counters[index], 1u); return 0u;",
            "atomic 'atomicAdd' requires uint result context.*expected bool",
        ),
    ],
)
def test_directx_typed_buffer_atomics_validate_result_contexts(function_body, match):
    shader = f"""
    shader BadTypedBufferAtomicResultHLSL {{
        RWBuffer<uint> counters @register(u1);
        RWStructuredBuffer<int> signedCounters @register(u2);

        float useFloat(float value) {{
            return value;
        }}

        uint update(uint index) {{
            {function_body}
        }}

        compute {{
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {{
                uint result = update(tid.x);
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=match):
        generate_code(parse_code(tokenize_code(shader)))


def test_directx_typed_buffer_atomics_validate_return_result_contexts():
    shader = """
    shader BadTypedBufferAtomicReturnResultHLSL {
        RWBuffer<uint> counters @register(u1);

        float readOld(uint index) {
            return atomicAdd(counters[index], 1u);
        }

        compute {
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {
                float result = readOld(tid.x);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="atomic 'atomicAdd' requires uint result context.*expected float",
    ):
        generate_code(parse_code(tokenize_code(shader)))


def test_directx_typed_buffer_atomics_lift_inside_constructors():
    shader = """
    shader TypedBufferAtomicConstructorsHLSL {
        RWBuffer<uint> counters @register(u1);

        compute {
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {
                uvec2 pair = uvec2(
                    atomicAdd(counters[tid.x], 1u),
                    atomicCompareExchange(counters[tid.x], 2u, 3u)
                );
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "uint __crossgl_atomic_expr_0;" in generated_code
    assert (
        "InterlockedAdd(counters[tid.x], 1u, __crossgl_atomic_expr_0);"
        in generated_code
    )
    assert "uint __crossgl_atomic_expr_1;" in generated_code
    assert (
        "InterlockedCompareExchange(counters[tid.x], 2u, 3u, "
        "__crossgl_atomic_expr_1);"
    ) in generated_code
    assert (
        "uint2 pair = uint2(__crossgl_atomic_expr_0, __crossgl_atomic_expr_1);"
        in generated_code
    )
    assert "unsupported HLSL typed buffer atomic expression" not in generated_code
    assert "atomicAdd(counters" not in generated_code
    assert "atomicCompareExchange(counters" not in generated_code


def test_directx_typed_buffer_atomics_lower_inside_ternary_statements():
    shader = """
    shader TypedBufferAtomicTernariesHLSL {
        RWBuffer<uint> counters @register(u1);

        uint choose(bool useAdd, uint index) {
            return useAdd
                ? atomicAdd(counters[index], 1u)
                : atomicCompareExchange(counters[index], 2u, 3u);
        }

        compute {
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {
                bool useAdd = tid.x != 0u;
                uint selected = useAdd
                    ? atomicAdd(counters[tid.x], 1u)
                    : atomicCompareExchange(counters[tid.x], 2u, 3u);
                selected = useAdd
                    ? atomicAdd(counters[tid.x], 4u) + 1u
                    : atomicExchange(counters[tid.x], 5u);
                selected += useAdd ? atomicXor(counters[tid.x], 6u) : 7u;
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "uint choose(bool useAdd, uint index) {\n    if (useAdd) {" in generated_code
    assert (
        "InterlockedAdd(counters[index], 1u, __crossgl_atomic_expr_0);"
        in generated_code
    )
    assert "return __crossgl_atomic_expr_0;" in generated_code
    assert (
        "InterlockedCompareExchange(counters[index], 2u, 3u, "
        "__crossgl_atomic_expr_1);"
    ) in generated_code
    assert "return __crossgl_atomic_expr_1;" in generated_code
    assert "uint selected;\n    if (useAdd) {" in generated_code
    assert (
        "InterlockedAdd(counters[tid.x], 1u, __crossgl_atomic_expr_2);"
        in generated_code
    )
    assert "selected = __crossgl_atomic_expr_2;" in generated_code
    assert (
        "InterlockedCompareExchange(counters[tid.x], 2u, 3u, "
        "__crossgl_atomic_expr_3);"
    ) in generated_code
    assert "selected = __crossgl_atomic_expr_3;" in generated_code
    assert (
        "InterlockedAdd(counters[tid.x], 4u, __crossgl_atomic_expr_4);"
        in generated_code
    )
    assert "selected = (__crossgl_atomic_expr_4 + 1u);" in generated_code
    assert (
        "InterlockedExchange(counters[tid.x], 5u, __crossgl_atomic_expr_5);"
        in generated_code
    )
    assert "selected = __crossgl_atomic_expr_5;" in generated_code
    assert (
        "InterlockedXor(counters[tid.x], 6u, __crossgl_atomic_expr_6);"
        in generated_code
    )
    assert "selected += __crossgl_atomic_expr_6;" in generated_code
    assert "selected += 7u;" in generated_code
    assert "unsupported HLSL typed buffer atomic expression" not in generated_code
    assert "atomicAdd(counters" not in generated_code
    assert "atomicCompareExchange(counters" not in generated_code
    assert "atomicExchange(counters" not in generated_code
    assert "atomicXor(counters" not in generated_code


def test_directx_typed_buffer_atomics_lift_inside_ternary_conditions():
    shader = """
    shader TypedBufferAtomicTernaryConditionsHLSL {
        RWBuffer<uint> counters @register(u1);

        compute {
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {
                uint selected = atomicAdd(counters[tid.x], 1u) != 0u
                    ? 11u
                    : atomicExchange(counters[tid.x], 2u);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "uint selected;" in generated_code
    assert "uint __crossgl_atomic_expr_0;" in generated_code
    assert (
        "InterlockedAdd(counters[tid.x], 1u, __crossgl_atomic_expr_0);"
        in generated_code
    )
    assert "if ((__crossgl_atomic_expr_0 != 0u)) {" in generated_code
    assert "selected = 11u;" in generated_code
    assert "uint __crossgl_atomic_expr_1;" in generated_code
    assert (
        "InterlockedExchange(counters[tid.x], 2u, __crossgl_atomic_expr_1);"
        in generated_code
    )
    assert "selected = __crossgl_atomic_expr_1;" in generated_code
    assert "unsupported HLSL typed buffer atomic expression" not in generated_code


def test_directx_typed_buffer_atomic_ternaries_lift_inside_larger_expressions():
    shader = """
    shader TypedBufferAtomicNestedTernariesHLSL {
        RWBuffer<uint> counters @register(u1);

        uint pick(bool useAdd, uint index) {
            return (useAdd ? atomicAdd(counters[index], 1u) : 7u) + 2u;
        }

        compute {
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {
                bool useAdd = tid.x != 0u;
                uint selected =
                    (useAdd
                        ? atomicAdd(counters[tid.x], 1u)
                        : atomicExchange(counters[tid.x], 2u)) + 3u;
                selected = uint(
                    useAdd
                        ? atomicCompareExchange(counters[tid.x], 4u, 5u)
                        : 6u
                ) + 1u;
                selected +=
                    (useAdd ? atomicXor(counters[tid.x], 6u) : 7u) + 8u;
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "uint __crossgl_atomic_ternary_0;" in generated_code
    assert (
        "InterlockedAdd(counters[index], 1u, __crossgl_atomic_expr_1);"
        in generated_code
    )
    assert "__crossgl_atomic_ternary_0 = __crossgl_atomic_expr_1;" in generated_code
    assert "__crossgl_atomic_ternary_0 = 7u;" in generated_code
    assert "return (__crossgl_atomic_ternary_0 + 2u);" in generated_code
    assert "uint __crossgl_atomic_ternary_2;" in generated_code
    assert (
        "InterlockedAdd(counters[tid.x], 1u, __crossgl_atomic_expr_3);"
        in generated_code
    )
    assert (
        "InterlockedExchange(counters[tid.x], 2u, __crossgl_atomic_expr_4);"
        in generated_code
    )
    assert "uint selected = (__crossgl_atomic_ternary_2 + 3u);" in generated_code
    assert "uint __crossgl_atomic_ternary_5;" in generated_code
    assert (
        "InterlockedCompareExchange(counters[tid.x], 4u, 5u, "
        "__crossgl_atomic_expr_6);"
    ) in generated_code
    assert "selected = (uint(__crossgl_atomic_ternary_5) + 1u);" in generated_code
    assert "uint __crossgl_atomic_ternary_7;" in generated_code
    assert (
        "InterlockedXor(counters[tid.x], 6u, __crossgl_atomic_expr_8);"
        in generated_code
    )
    assert "selected += (__crossgl_atomic_ternary_7 + 8u);" in generated_code
    assert "unsupported HLSL typed buffer atomic expression" not in generated_code
    assert "atomicAdd(counters" not in generated_code
    assert "atomicExchange(counters" not in generated_code
    assert "atomicCompareExchange(counters" not in generated_code
    assert "atomicXor(counters" not in generated_code


def test_directx_typed_buffer_atomics_lift_in_user_function_arguments():
    shader = """
    shader TypedBufferAtomicUserCallArgsHLSL {
        RWBuffer<uint> counters @register(u1);

        uint consume(uint value) {
            return value + 1u;
        }

        void store(uint value) {
        }

        compute {
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {
                uint result = consume(atomicAdd(counters[tid.x], 1u));
                result =
                    consume(atomicCompareExchange(counters[tid.x], 2u, 3u)) + 4u;
                store(atomicXor(counters[tid.x], 5u));
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "uint __crossgl_atomic_expr_0;" in generated_code
    assert (
        "InterlockedAdd(counters[tid.x], 1u, __crossgl_atomic_expr_0);"
        in generated_code
    )
    assert "uint result = consume(__crossgl_atomic_expr_0);" in generated_code
    assert "uint __crossgl_atomic_expr_1;" in generated_code
    assert (
        "InterlockedCompareExchange(counters[tid.x], 2u, 3u, "
        "__crossgl_atomic_expr_1);"
    ) in generated_code
    assert "result = (consume(__crossgl_atomic_expr_1) + 4u);" in generated_code
    assert "uint __crossgl_atomic_expr_2;" in generated_code
    assert (
        "InterlockedXor(counters[tid.x], 5u, __crossgl_atomic_expr_2);"
        in generated_code
    )
    assert "store(__crossgl_atomic_expr_2);" in generated_code
    assert "unsupported HLSL typed buffer atomic expression" not in generated_code
    assert "atomicAdd(counters" not in generated_code
    assert "atomicCompareExchange(counters" not in generated_code
    assert "atomicXor(counters" not in generated_code


def test_directx_typed_buffer_atomic_unsupported_expression_context_raises():
    shader = """
    shader BadTypedBufferAtomicExpressionContextHLSL {
        RWBuffer<uint> counters @register(u1);

        compute {
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {
                float result = sin(atomicAdd(counters[tid.x], 1u));
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="DirectX typed buffer atomic 'atomicAdd' requires statement lowering",
    ):
        generate_code(parse_code(tokenize_code(shader)))


def test_directx_typed_buffer_atomics_lift_in_struct_constructors():
    shader = """
    shader TypedBufferAtomicStructConstructorsHLSL {
        struct Pair {
            uint oldValue;
            uint fixedValue;
        };

        RWBuffer<uint> counters @register(u1);

        Pair makePair(bool useAdd, uint index) {
            return Pair(
                useAdd ? atomicAdd(counters[index], 1u) : 7u,
                atomicCompareExchange(counters[index], 2u, 3u)
            );
        }

        compute {
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {
                Pair p = Pair(atomicAdd(counters[tid.x], 1u), 7u);
                p = Pair(
                    atomicCompareExchange(counters[tid.x], 2u, 3u),
                    atomicXor(counters[tid.x], 4u)
                );
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "struct Pair" in generated_code
    assert "uint __crossgl_atomic_ternary_0;" in generated_code
    assert (
        "InterlockedAdd(counters[index], 1u, __crossgl_atomic_expr_1);"
        in generated_code
    )
    assert "__crossgl_atomic_ternary_0 = __crossgl_atomic_expr_1;" in generated_code
    assert "__crossgl_atomic_ternary_0 = 7u;" in generated_code
    assert "uint __crossgl_atomic_expr_2;" in generated_code
    assert (
        "InterlockedCompareExchange(counters[index], 2u, 3u, "
        "__crossgl_atomic_expr_2);"
    ) in generated_code
    assert "return Pair(__crossgl_atomic_ternary_0, __crossgl_atomic_expr_2);" in (
        generated_code
    )
    assert "uint __crossgl_atomic_expr_3;" in generated_code
    assert (
        "InterlockedAdd(counters[tid.x], 1u, __crossgl_atomic_expr_3);"
        in generated_code
    )
    assert "Pair p = Pair(__crossgl_atomic_expr_3, 7u);" in generated_code
    assert "uint __crossgl_atomic_expr_4;" in generated_code
    assert (
        "InterlockedCompareExchange(counters[tid.x], 2u, 3u, "
        "__crossgl_atomic_expr_4);"
    ) in generated_code
    assert "uint __crossgl_atomic_expr_5;" in generated_code
    assert (
        "InterlockedXor(counters[tid.x], 4u, __crossgl_atomic_expr_5);"
        in generated_code
    )
    assert "p = Pair(__crossgl_atomic_expr_4, __crossgl_atomic_expr_5);" in (
        generated_code
    )
    assert "unsupported HLSL typed buffer atomic expression" not in generated_code
    assert "atomicAdd(counters" not in generated_code
    assert "atomicCompareExchange(counters" not in generated_code
    assert "atomicXor(counters" not in generated_code


def test_directx_typed_buffer_atomics_lift_in_array_literals():
    shader = """
    shader TypedBufferAtomicArrayLiteralsHLSL {
        RWBuffer<uint> counters @register(u1);

        compute {
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {
                float weights[2] = {1.0, 2.0};
                uint values[2] = {
                    atomicAdd(counters[tid.x], 1u),
                    atomicCompareExchange(counters[tid.x], 2u, 3u)
                };
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "float weights[2] = {1.0, 2.0};" in generated_code
    assert "uint __crossgl_atomic_expr_0;" in generated_code
    assert (
        "InterlockedAdd(counters[tid.x], 1u, __crossgl_atomic_expr_0);"
        in generated_code
    )
    assert "uint __crossgl_atomic_expr_1;" in generated_code
    assert (
        "InterlockedCompareExchange(counters[tid.x], 2u, 3u, "
        "__crossgl_atomic_expr_1);"
    ) in generated_code
    assert (
        "uint values[2] = {__crossgl_atomic_expr_0, __crossgl_atomic_expr_1};"
        in generated_code
    )
    assert "ArrayLiteralNode" not in generated_code
    assert "unsupported HLSL typed buffer atomic expression" not in generated_code
    assert "atomicAdd(counters" not in generated_code
    assert "atomicCompareExchange(counters" not in generated_code


def test_directx_typed_buffer_atomics_reject_non_integer_targets():
    shader = """
    shader BadTypedBufferAtomicHLSL {
        RWBuffer<float> values @register(u1);

        compute {
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {
                float original = atomicAdd(values[tid.x], 1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "DirectX typed buffer atomic 'atomicAdd' requires a scalar "
            "int or uint target, got float"
        ),
    ):
        generate_code(parse_code(tokenize_code(shader)))


@pytest.mark.parametrize(
    ("declaration", "call", "resource_type"),
    [
        (
            "Buffer<uint> values @register(t1);",
            "atomicAdd(values[tid.x], 1u)",
            "Buffer<uint>",
        ),
        (
            "StructuredBuffer<uint> values @register(t1);",
            "atomicAdd(values[tid.x], 1u)",
            "StructuredBuffer<uint>",
        ),
        (
            "Buffer<uint> values[2] @register(t1);",
            "atomicAdd(values[1][tid.x], 1u)",
            "Buffer<uint>",
        ),
    ],
)
def test_directx_typed_buffer_atomics_reject_readonly_targets(
    declaration, call, resource_type
):
    shader = f"""
    shader BadReadonlyTypedBufferAtomicHLSL {{
        {declaration}

        compute {{
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {{
                uint original = {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "DirectX typed buffer atomic 'atomicAdd' cannot write readonly "
            f"{re.escape(resource_type)}"
        ),
    ):
        generate_code(parse_code(tokenize_code(shader)))


@pytest.mark.parametrize(
    ("declaration", "call", "resource_type"),
    [
        (
            "Buffer<uint> values @register(t1);",
            "buffer_store(values, tid.x, 1u);",
            "Buffer<uint>",
        ),
        (
            "StructuredBuffer<uint> values @register(t1);",
            "buffer_store(values, tid.x, 1u);",
            "StructuredBuffer<uint>",
        ),
        (
            "StructuredBuffer<uint> values[2] @register(t1);",
            "buffer_store(values[1], tid.x, 1u);",
            "StructuredBuffer<uint>",
        ),
        (
            "ByteAddressBuffer rawBytes @register(t3);",
            "buffer_store(rawBytes, tid.x * 4u, 1u);",
            "ByteAddressBuffer",
        ),
        (
            "ByteAddressBuffer rawBytes @register(t3);",
            "buffer_store4(rawBytes, tid.x * 16u, uvec4(1u, 2u, 3u, 4u));",
            "ByteAddressBuffer",
        ),
    ],
)
def test_directx_buffer_store_helpers_reject_readonly_resources(
    declaration, call, resource_type
):
    shader = f"""
    shader BadReadonlyBufferStoreHLSL {{
        {declaration}

        compute {{
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {{
                {call}
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX buffer helper 'buffer_store"
            f"{'4' if 'buffer_store4' in call else ''}' cannot write readonly "
            f"{re.escape(resource_type)}"
        ),
    ):
        generate_code(parse_code(tokenize_code(shader)))


@pytest.mark.parametrize(
    ("declaration", "call", "match"),
    [
        (
            "AppendStructuredBuffer<uint> values @register(u1);",
            "uint value = buffer_load(values, tid.x);",
            "DirectX buffer helper 'buffer_load' requires a resource with "
            "indexed read support, got AppendStructuredBuffer<uint>",
        ),
        (
            "ConsumeStructuredBuffer<uint> values @register(u2);",
            "buffer_store(values, tid.x, 1u);",
            "DirectX buffer helper 'buffer_store' requires a resource with "
            "indexed write support, got ConsumeStructuredBuffer<uint>",
        ),
        (
            "ConsumeStructuredBuffer<uint> values @register(u2);",
            "buffer_append(values, 1u);",
            "DirectX buffer helper 'buffer_append' requires "
            "AppendStructuredBuffer, got ConsumeStructuredBuffer<uint>",
        ),
        (
            "AppendStructuredBuffer<uint> values @register(u1);",
            "uint value = buffer_consume(values);",
            "DirectX buffer helper 'buffer_consume' requires "
            "ConsumeStructuredBuffer, got AppendStructuredBuffer<uint>",
        ),
        (
            "StructuredBuffer<uint> values @register(t1);",
            "uint value = buffer_increment_counter(values);",
            "DirectX buffer helper 'buffer_increment_counter' requires "
            "RWStructuredBuffer, got StructuredBuffer<uint>",
        ),
        (
            "AppendStructuredBuffer<uint> values @register(u1);",
            "uint value = buffer_decrement_counter(values);",
            "DirectX buffer helper 'buffer_decrement_counter' requires "
            "RWStructuredBuffer, got AppendStructuredBuffer<uint>",
        ),
    ],
)
def test_directx_append_consume_buffers_reject_wrong_helpers(declaration, call, match):
    shader = f"""
    shader BadAppendConsumeHelperHLSL {{
        {declaration}

        compute {{
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {{
                {call}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=re.escape(match)):
        generate_code(parse_code(tokenize_code(shader)))


@pytest.mark.parametrize(
    ("declaration", "call", "match"),
    [
        (
            "AppendStructuredBuffer<uint> values @register(u1);",
            "buffer_append(values);",
            "DirectX buffer helper 'buffer_append' requires " "2 argument(s), got 1",
        ),
        (
            "AppendStructuredBuffer<uint> values @register(u1);",
            "buffer_append(values, 1u, 2u);",
            "DirectX buffer helper 'buffer_append' requires " "2 argument(s), got 3",
        ),
        (
            "ConsumeStructuredBuffer<uint> values @register(u2);",
            "uint value = buffer_consume();",
            "DirectX buffer helper 'buffer_consume' requires " "1 argument(s), got 0",
        ),
        (
            "ConsumeStructuredBuffer<uint> values @register(u2);",
            "uint value = buffer_consume(values, 1u);",
            "DirectX buffer helper 'buffer_consume' requires " "1 argument(s), got 2",
        ),
        (
            "RWStructuredBuffer<uint> values @register(u3);",
            "uint value = buffer_increment_counter();",
            "DirectX buffer helper 'buffer_increment_counter' requires "
            "1 argument(s), got 0",
        ),
        (
            "RWStructuredBuffer<uint> values @register(u3);",
            "uint value = buffer_decrement_counter(values, 1u);",
            "DirectX buffer helper 'buffer_decrement_counter' requires "
            "1 argument(s), got 2",
        ),
    ],
)
def test_directx_append_consume_buffers_validate_helper_arity(declaration, call, match):
    shader = f"""
    shader BadAppendConsumeHelperArityHLSL {{
        {declaration}

        compute {{
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {{
                {call}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=re.escape(match)):
        generate_code(parse_code(tokenize_code(shader)))


@pytest.mark.parametrize(
    ("declaration", "call", "match"),
    [
        (
            "AppendStructuredBuffer<uint> values @register(u1);",
            "buffer_append(values, uvec2(1u, 2u));",
            "DirectX buffer helper 'buffer_append' requires value matching "
            "AppendStructuredBuffer element type uint, got uint2",
        ),
        (
            "AppendStructuredBuffer<uvec2> values @register(u1);",
            "buffer_append(values, 1u);",
            "DirectX buffer helper 'buffer_append' requires value matching "
            "AppendStructuredBuffer element type uint2, got uint",
        ),
    ],
)
def test_directx_append_buffers_validate_element_shape(declaration, call, match):
    shader = f"""
    shader BadAppendElementShapeHLSL {{
        {declaration}

        compute {{
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {{
                {call}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=re.escape(match)):
        generate_code(parse_code(tokenize_code(shader)))


@pytest.mark.parametrize(
    ("declaration", "body", "match"),
    [
        (
            "ConsumeStructuredBuffer<uint> values @register(u2);",
            """
            void update() {
                uvec2 value = buffer_consume(values);
            }
            """,
            "DirectX buffer helper 'buffer_consume' result requires target matching "
            "ConsumeStructuredBuffer element type uint, got uint2",
        ),
        (
            "ConsumeStructuredBuffer<uint> values @register(u2);",
            """
            void update() {
                uvec2 value;
                value = buffer_consume(values);
            }
            """,
            "DirectX buffer helper 'buffer_consume' result requires target matching "
            "ConsumeStructuredBuffer element type uint, got uint2",
        ),
        (
            "ConsumeStructuredBuffer<uvec2> values @register(u2);",
            """
            uint update() {
                return buffer_consume(values);
            }
            """,
            "DirectX buffer helper 'buffer_consume' result requires target matching "
            "ConsumeStructuredBuffer element type uint2, got uint",
        ),
        (
            "ConsumeStructuredBuffer<uvec2> values @register(u2);",
            """
            uint update() {
                return buffer_consume(values) + 1u;
            }
            """,
            "DirectX buffer helper 'buffer_consume' result requires target matching "
            "ConsumeStructuredBuffer element type uint2, got uint",
        ),
    ],
)
def test_directx_consume_buffers_validate_result_shape(declaration, body, match):
    shader = f"""
    shader BadConsumeResultShapeHLSL {{
        {declaration}

        {body}
    }}
    """

    with pytest.raises(ValueError, match=re.escape(match)):
        generate_code(parse_code(tokenize_code(shader)))


@pytest.mark.parametrize(
    ("declaration", "callee", "call", "match"),
    [
        (
            "ConsumeStructuredBuffer<uint> values @register(u2);",
            "void acceptPair(uvec2 value) {}",
            "acceptPair(buffer_consume(values));",
            "DirectX buffer helper 'buffer_consume' result requires target matching "
            "ConsumeStructuredBuffer element type uint, got uint2",
        ),
        (
            "ConsumeStructuredBuffer<uvec2> values @register(u2);",
            "void acceptScalar(uint value) {}",
            "acceptScalar(buffer_consume(values));",
            "DirectX buffer helper 'buffer_consume' result requires target matching "
            "ConsumeStructuredBuffer element type uint2, got uint",
        ),
    ],
)
def test_directx_consume_buffers_validate_user_function_argument_shape(
    declaration, callee, call, match
):
    shader = f"""
    shader BadConsumeUserCallShapeHLSL {{
        {declaration}

        {callee}

        compute {{
            @numthreads(1, 1, 1)
            void main() {{
                {call}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=re.escape(match)):
        generate_code(parse_code(tokenize_code(shader)))


@pytest.mark.parametrize(
    ("declaration", "call", "match"),
    [
        (
            "RWBuffer<uint> values @register(u1);",
            "uvec2 value = buffer_load2(values, tid.x);",
            "DirectX buffer helper 'buffer_load2' requires "
            "ByteAddressBuffer, RWByteAddressBuffer, or "
            "RasterizerOrderedByteAddressBuffer resource, got RWBuffer<uint>",
        ),
        (
            "RWStructuredBuffer<uint> values @register(u2);",
            "buffer_store3(values, tid.x, uvec3(1u, 2u, 3u));",
            "DirectX buffer helper 'buffer_store3' requires "
            "ByteAddressBuffer, RWByteAddressBuffer, or "
            "RasterizerOrderedByteAddressBuffer resource, got "
            "RWStructuredBuffer<uint>",
        ),
        (
            "RWByteAddressBuffer rawBytes @register(u3);",
            "buffer_store2(rawBytes, tid.x * 8u, 7u);",
            "DirectX buffer helper 'buffer_store2' requires uint2 value, got uint",
        ),
        (
            "RWByteAddressBuffer rawBytes @register(u3);",
            "buffer_store4(rawBytes, tid.x * 16u, uvec3(1u, 2u, 3u));",
            "DirectX buffer helper 'buffer_store4' requires uint4 value, got uint3",
        ),
    ],
)
def test_directx_byte_address_vector_helpers_validate_resource_and_value_shape(
    declaration, call, match
):
    shader = f"""
    shader BadByteAddressVectorHelperHLSL {{
        {declaration}

        compute {{
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {{
                {call}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=re.escape(match)):
        generate_code(parse_code(tokenize_code(shader)))


def test_directx_byte_address_interlocked_member_calls_are_preserved():
    shader = """
    shader ByteAddressInterlockedMemberCallsHLSL {
        RWByteAddressBuffer rawBytes @register(u3);
        RasterizerOrderedByteAddressBuffer orderedBytes @register(u4);

        compute {
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {
                uint offset = tid.x * 4u;
                uint oldValue = 0u;
                rawBytes.InterlockedAdd(offset, 1u);
                rawBytes.InterlockedOr(offset + 4u, 3u, oldValue);
                rawBytes.InterlockedCompareExchange(offset + 8u, oldValue, 7u, oldValue);
                orderedBytes.InterlockedExchange(offset + 12u, 11u, oldValue);
                rawBytes.InterlockedCompareStore(offset + 16u, oldValue, 13u);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(shader)))

    assert "rawBytes.InterlockedAdd(offset, 1u);" in generated_code
    assert "rawBytes.InterlockedOr((offset + 4u), 3u, oldValue);" in generated_code
    assert (
        "rawBytes.InterlockedCompareExchange((offset + 8u), oldValue, 7u, oldValue);"
        in generated_code
    )
    assert (
        "orderedBytes.InterlockedExchange((offset + 12u), 11u, oldValue);"
        in generated_code
    )
    assert (
        "rawBytes.InterlockedCompareStore((offset + 16u), oldValue, 13u);"
        in generated_code
    )


@pytest.mark.parametrize(
    ("declaration", "call", "match"),
    [
        (
            "ByteAddressBuffer rawBytes @register(t3);",
            "uint oldValue = 0u; rawBytes.InterlockedAdd(0u, 1u, oldValue);",
            "DirectX ByteAddressBuffer interlocked member 'InterlockedAdd' "
            "cannot write readonly ByteAddressBuffer",
        ),
        (
            "RWBuffer<uint> rawBytes @register(u3);",
            "uint oldValue = 0u; rawBytes.InterlockedAdd(0u, 1u, oldValue);",
            "DirectX ByteAddressBuffer interlocked member 'InterlockedAdd' requires "
            "ByteAddressBuffer, RWByteAddressBuffer, or "
            "RasterizerOrderedByteAddressBuffer resource, got RWBuffer<uint>",
        ),
        (
            "RWByteAddressBuffer rawBytes @register(u3);",
            "rawBytes.InterlockedAdd(0u);",
            "DirectX ByteAddressBuffer interlocked member 'InterlockedAdd' requires "
            "2 or 3 argument(s), got 1",
        ),
        (
            "RWByteAddressBuffer rawBytes @register(u3);",
            "uint oldValue = 0u; rawBytes.InterlockedCompareExchange(0u, 1u, oldValue);",
            "DirectX ByteAddressBuffer interlocked member "
            "'InterlockedCompareExchange' requires 4 argument(s), got 3",
        ),
        (
            "RWByteAddressBuffer rawBytes @register(u3);",
            "rawBytes.InterlockedCompareStore(0u, 1u);",
            "DirectX ByteAddressBuffer interlocked member "
            "'InterlockedCompareStore' requires 3 argument(s), got 2",
        ),
        (
            "RWByteAddressBuffer rawBytes @register(u3);",
            "uint oldValue = 0u; rawBytes.InterlockedAdd(0.0, 1u, oldValue);",
            "DirectX ByteAddressBuffer interlocked member 'InterlockedAdd' "
            "address argument must be scalar uint, got float",
        ),
        (
            "RWByteAddressBuffer rawBytes @register(u3);",
            "uint oldValue = 0u; rawBytes.InterlockedAdd(0u, 1.0, oldValue);",
            "DirectX ByteAddressBuffer interlocked member 'InterlockedAdd' "
            "value argument must be scalar uint, got float",
        ),
        (
            "RWByteAddressBuffer rawBytes @register(u3);",
            "uint oldValue = 0u; rawBytes.InterlockedAdd(0u, 1u, oldValue + 1u);",
            "DirectX ByteAddressBuffer interlocked member 'InterlockedAdd' "
            "original argument must be an assignable scalar uint target",
        ),
        (
            "RWByteAddressBuffer rawBytes @register(u3);",
            "uvec2 oldValue = uvec2(0u, 0u); rawBytes.InterlockedAdd(0u, 1u, oldValue);",
            "DirectX ByteAddressBuffer interlocked member 'InterlockedAdd' "
            "original argument must be scalar uint, got uint2",
        ),
    ],
)
def test_directx_byte_address_interlocked_member_calls_validate_arguments(
    declaration, call, match
):
    shader = f"""
    shader BadByteAddressInterlockedMemberCallHLSL {{
        {declaration}

        compute {{
            @numthreads(1, 1, 1)
            void main() {{
                {call}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=re.escape(match)):
        generate_code(parse_code(tokenize_code(shader)))


@pytest.mark.parametrize(
    ("function_body", "method_name"),
    [
        (
            "uint direct = rawBytes.InterlockedAdd(0u, 1u, oldValue); "
            "return direct;",
            "InterlockedAdd",
        ),
        (
            "direct = rawBytes.InterlockedOr(4u, 3u, oldValue); return direct;",
            "InterlockedOr",
        ),
        (
            "consume(rawBytes.InterlockedXor(8u, 2u, oldValue)); return direct;",
            "InterlockedXor",
        ),
        (
            "uint selected = direct != 0u "
            "? rawBytes.InterlockedAnd(12u, 1u, oldValue) : 0u; "
            "return selected;",
            "InterlockedAnd",
        ),
        (
            "Pair pair = Pair(rawBytes.InterlockedExchange(16u, 5u, oldValue)); "
            "return pair.value;",
            "InterlockedExchange",
        ),
        (
            "uint values[1] = { "
            "rawBytes.InterlockedCompareExchange(20u, 1u, 2u, oldValue) "
            "}; return values[0];",
            "InterlockedCompareExchange",
        ),
        (
            "return rawBytes.InterlockedMin(24u, 1u, oldValue);",
            "InterlockedMin",
        ),
        (
            "uint direct = rawBytes.InterlockedCompareStore(28u, 1u, 2u); "
            "return direct;",
            "InterlockedCompareStore",
        ),
    ],
)
def test_directx_byte_address_interlocked_member_calls_reject_value_contexts(
    function_body, method_name
):
    shader = f"""
    shader BadByteAddressInterlockedMemberValueContextHLSL {{
        RWByteAddressBuffer rawBytes @register(u3);

        struct Pair {{
            uint value;
        }};

        uint consume(uint value) {{
            return value + 1u;
        }}

        uint update(uint direct) {{
            uint oldValue = 0u;
            {function_body}
        }}
    }}
    """

    match = (
        f"DirectX ByteAddressBuffer interlocked member '{method_name}' "
        "requires standalone statement context"
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        generate_code(parse_code(tokenize_code(shader)))


def test_structured_buffer_dimensions_lower_to_get_dimensions():
    code = """
    shader StructuredBufferDimensionsHLSL {
        RWStructuredBuffer<int> values @ binding(3);
        RWStructuredBuffer<int> buffers[2] @ binding(4);

        compute {
            void main(uint index) {
                uint len;
                uint arrayLen;
                buffer_dimensions(values, len);
                buffer_dimensions(buffers[index], arrayLen);
            }
        }
    }
    """
    ast = parse_code(tokenize_code(code))

    generated = generate_code(ast)

    assert "values.GetDimensions(len);" in generated
    assert "buffers[index].GetDimensions(arrayLen);" in generated
    assert "buffer_dimensions" not in generated


def test_structured_buffer_append_consume_lower_to_native_methods():
    code = """
    shader StructuredBufferAppendConsumeHLSL {
        AppendStructuredBuffer<int> appendValues @ binding(1);
        ConsumeStructuredBuffer<int> consumeValues @ binding(2);

        compute {
            void main(uint value) {
                buffer_append(appendValues, int(value));
                int consumed = buffer_consume(consumeValues);
            }
        }
    }
    """
    ast = parse_code(tokenize_code(code))

    generated = generate_code(ast)

    assert "AppendStructuredBuffer<int> appendValues : register(u1);" in generated
    assert "ConsumeStructuredBuffer<int> consumeValues : register(u2);" in generated
    assert "appendValues.Append(int(value));" in generated
    assert "int consumed = consumeValues.Consume();" in generated
    assert "buffer_append" not in generated
    assert "buffer_consume" not in generated


def test_rwstructured_buffer_counter_helpers_lower_to_native_methods():
    code = """
    shader StructuredBufferCounterHelpersHLSL {
        RWStructuredBuffer<uint> counters @ binding(3);
        RWStructuredBuffer<uint> counterArrays[2] @ binding(4);

        compute {
            void main(uint which) {
                uint nextIndex = buffer_increment_counter(counters);
                uint oldIndex = buffer_decrement_counter(counterArrays[which]);
                buffer_store(counterArrays[which], nextIndex, oldIndex);
            }
        }
    }
    """
    ast = parse_code(tokenize_code(code))

    generated = generate_code(ast)

    assert "RWStructuredBuffer<uint> counters : register(u3);" in generated
    assert "RWStructuredBuffer<uint> counterArrays[2] : register(u4);" in generated
    assert "uint nextIndex = counters.IncrementCounter();" in generated
    assert "uint oldIndex = counterArrays[which].DecrementCounter();" in generated
    assert "counterArrays[which].Store(nextIndex, oldIndex);" in generated
    assert "buffer_increment_counter" not in generated
    assert "buffer_decrement_counter" not in generated


def test_structured_buffer_append_accepts_matching_vector_element_shape():
    code = """
    shader StructuredBufferAppendVectorShapeHLSL {
        AppendStructuredBuffer<uvec2> appendValues @ binding(1);

        compute {
            void main() {
                buffer_append(appendValues, uvec2(1u, 2u));
            }
        }
    }
    """
    ast = parse_code(tokenize_code(code))

    generated = generate_code(ast)

    assert "AppendStructuredBuffer<uint2> appendValues : register(u1);" in generated
    assert "appendValues.Append(uint2(1u, 2u));" in generated
    assert "buffer_append" not in generated


def test_structured_buffer_consume_accepts_matching_vector_result_shape():
    code = """
    shader StructuredBufferConsumeVectorShapeHLSL {
        ConsumeStructuredBuffer<uvec2> consumeValues @ binding(2);

        uvec2 consumePair(bool fallback) {
            return fallback
                ? uvec2(0u, 0u)
                : buffer_consume(consumeValues);
        }

        compute {
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {
                uvec2 consumed = buffer_consume(consumeValues);
                consumed += consumePair(tid.x == 0u);
            }
        }
    }
    """
    ast = parse_code(tokenize_code(code))

    generated = generate_code(ast)

    assert "ConsumeStructuredBuffer<uint2> consumeValues : register(u2);" in generated
    assert "return (fallback ? uint2(0u, 0u) : consumeValues.Consume());" in generated
    assert "uint2 consumed = consumeValues.Consume();" in generated
    assert "buffer_consume" not in generated


def test_structured_buffer_consume_accepts_matching_user_function_argument_shape():
    code = """
    shader StructuredBufferConsumeUserCallShapeHLSL {
        ConsumeStructuredBuffer<uint> scalarValues @ binding(1);
        ConsumeStructuredBuffer<uvec2> pairValues @ binding(2);

        void acceptScalar(uint value) {}
        void acceptPair(uvec2 value) {}

        compute {
            @numthreads(1, 1, 1)
            void main() {
                acceptScalar(buffer_consume(scalarValues));
                acceptPair(buffer_consume(pairValues));
            }
        }
    }
    """
    ast = parse_code(tokenize_code(code))

    generated = generate_code(ast)

    assert "ConsumeStructuredBuffer<uint> scalarValues : register(u1);" in generated
    assert "ConsumeStructuredBuffer<uint2> pairValues : register(u2);" in generated
    assert "acceptScalar(scalarValues.Consume());" in generated
    assert "acceptPair(pairValues.Consume());" in generated
    assert "buffer_consume" not in generated


def test_hlsl_unsigned_integer_literal_suffix_codegen():
    code = """
    shader UIntLiteralCodegen {
        compute {
            void main() {
                uint a = 7u;
                uint b = 0xFu;
                uint c = 0b101U;
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "uint a = 7u;" in generated_code
    assert "uint b = 15u;" in generated_code
    assert "uint c = 5u;" in generated_code
    assert "7, u" not in generated_code


def test_hlsl_resource_binding_attributes_are_not_parameter_semantics():
    code = """
    shader BindingAttributes {
        vec4 sampleBound(sampler2D tex @texture(1), sampler samp @sampler(2), sampler2D registered @register(t3), vec2 uv) {
            return texture(tex, samp, uv) + texture(registered, uv);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "float4 sampleBound(Texture2D tex, SamplerState samp, Texture2D registered, SamplerState registeredSampler, float2 uv)"
        in generated_code
    )
    assert "tex.Sample(samp, uv)" in generated_code
    assert "registered.Sample(registeredSampler, uv)" in generated_code
    assert ": texture" not in generated_code
    assert ": sampler" not in generated_code
    assert ": register" not in generated_code


def test_hlsl_global_resource_binding_attributes_drive_registers():
    code = """
    shader ExplicitGlobalBindings {
        sampler samp @sampler(5);
        sampler2D explicitTex @texture(3);
        image2D storageImage @binding(7);
        sampler2D registerTex @register(t8);
        sampler2D autoTex;

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                vec4 stored = imageLoad(storageImage, ivec2(0, 0));
                return texture(explicitTex, samp, uv) + texture(registerTex, samp, uv) + texture(autoTex, samp, uv) + stored;
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "SamplerState samp : register(s5);" in generated_code
    assert "Texture2D explicitTex : register(t3);" in generated_code
    assert "RWTexture2D<float4> storageImage : register(u7);" in generated_code
    assert "Texture2D registerTex : register(t8);" in generated_code
    assert "Texture2D autoTex : register(t9);" in generated_code


def test_hlsl_stage_local_resources_emit_global_registers():
    code = """
    shader StageLocalResources {
        fragment {
            uniform sampler2D localTex @texture(2);
            uniform image2D localImage @binding(4);

            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                vec4 stored = imageLoad(localImage, ivec2(0, 0));
                return texture(localTex, uv) + stored;
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "Texture2D localTex : register(t2);" in generated_code
    assert "SamplerState localTexSampler : register(s0);" in generated_code
    assert "RWTexture2D<float4> localImage : register(u4);" in generated_code
    assert "localTex.Sample(localTexSampler, uv)" in generated_code


def test_hlsl_stage_local_cube_image_size_uses_storage_image_helper():
    code = """
    shader StageLocalCubeImageSize {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            uniform imageCube cubeImage @texture(3);

            void main() {
                ivec2 size = imageSize(cubeImage);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "compute"
    )

    assert "RWTextureCube<float4> cubeImage : register(u3);" in generated_code
    assert "int2 imageSize(RWTextureCube<float4> image)" in generated_code
    assert "int2 size = imageSize(cubeImage);" in generated_code


def test_hlsl_generic_bindings_are_independent_between_register_namespaces():
    code = """
    shader GenericBindingNamespaces {
        sampler2D colorMap @binding(0);
        sampler linearSampler @binding(0);
        image2D outImage @binding(0);

        cbuffer Constants @binding(0) {
            vec4 tint;
        };

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                vec4 stored = imageLoad(outImage, ivec2(0, 0));
                return texture(colorMap, linearSampler, uv) + stored + tint;
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "RWTexture2D<float4> outImage : register(u0);" in generated_code
    assert "cbuffer Constants : register(b0)" in generated_code


def test_hlsl_generic_binding_overlap_raises_within_uav_namespace():
    code = """
    shader GenericBindingUavOverlap {
        image2D images[2] @binding(3);
        RWStructuredBuffer<int> values @binding(4);
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting DirectX resource binding for 'values': "
            "u4 overlaps 'images' u3-u4"
        ),
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_hlsl_auto_registers_skip_later_explicit_global_bindings():
    code = """
    shader LateExplicitBindings {
        sampler2D autoTexture;
        image2D autoImage;
        sampler autoSampler;

        cbuffer AutoBlock {
            vec4 autoTint;
        };

        sampler2D explicitTexture @register(t0);
        RWStructuredBuffer<int> explicitValues @register(u0);
        sampler explicitSampler @register(s0);

        cbuffer ExplicitBlock @register(b0) {
            vec4 explicitTint;
        };

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                vec4 stored = imageLoad(autoImage, ivec2(0, 0));
                int value = buffer_load(explicitValues, 0);
                return texture(autoTexture, autoSampler, uv) +
                    texture(explicitTexture, explicitSampler, uv) +
                    stored + autoTint + explicitTint + vec4(float(value));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "Texture2D autoTexture : register(t1);" in generated_code
    assert "RWTexture2D<float4> autoImage : register(u1);" in generated_code
    assert "SamplerState autoSampler : register(s1);" in generated_code
    assert "cbuffer AutoBlock : register(b1)" in generated_code
    assert "Texture2D explicitTexture : register(t0);" in generated_code
    assert "RWStructuredBuffer<int> explicitValues : register(u0);" in generated_code
    assert "SamplerState explicitSampler : register(s0);" in generated_code
    assert "cbuffer ExplicitBlock : register(b0)" in generated_code


def test_hlsl_implicit_sampler_skips_later_explicit_sampler_register():
    code = """
    shader LateExplicitSampler {
        sampler2D colorMap;
        sampler linearSampler @register(s0);

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                return texture(colorMap, uv) + texture(colorMap, linearSampler, uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "SamplerState colorMapSampler : register(s1);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "colorMap.Sample(colorMapSampler, uv)" in generated_code
    assert "colorMap.Sample(linearSampler, uv)" in generated_code


def test_hlsl_global_resource_register_spaces_keep_independent_cursors():
    code = """
    shader ExplicitGlobalBindingSpaces {
        sampler spaceSampler @register(s4, space1);
        sampler autoSpaceSampler @space(1);
        sampler defaultSampler;

        sampler2D spaceTex @register(t5, space1);
        sampler2D otherSpaceTex @register(t5, space2);
        sampler2D autoSpaceTex @space(2);
        sampler2D defaultTex;

        uimage2D spaceImage @register(u3, space1);
        image2D otherSpaceImage @register(u3, space2);
        RWStructuredBuffer<int> autoSpaceBuffer @space(2);
        RWStructuredBuffer<int> defaultBuffer;
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "SamplerState spaceSampler : register(s4, space1);" in generated_code
    assert "SamplerState autoSpaceSampler : register(s5, space1);" in generated_code
    assert "SamplerState defaultSampler : register(s0);" in generated_code
    assert "Texture2D spaceTex : register(t5, space1);" in generated_code
    assert "Texture2D otherSpaceTex : register(t5, space2);" in generated_code
    assert "Texture2D autoSpaceTex : register(t6, space2);" in generated_code
    assert "Texture2D defaultTex : register(t0);" in generated_code
    assert "RWTexture2D<uint> spaceImage : register(u3, space1);" in generated_code
    assert (
        "RWTexture2D<float4> otherSpaceImage : register(u3, space2);" in generated_code
    )
    assert (
        "RWStructuredBuffer<int> autoSpaceBuffer : register(u4, space2);"
        in generated_code
    )
    assert "RWStructuredBuffer<int> defaultBuffer : register(u0);" in generated_code


def test_hlsl_implicit_sampler_inherits_global_texture_register_space():
    code = """
    shader ImplicitSamplerSpaces {
        sampler explicitSpaceSampler @register(s4, space1);
        sampler defaultSampler;
        sampler2D spaceTex @register(t3, space1);
        sampler2D defaultTex;

        fragment {
            vec4 main(vec2 uv @TEXCOORD0) @gl_FragColor {
                return texture(spaceTex, uv) +
                    texture(defaultTex, uv) +
                    texture(spaceTex, explicitSpaceSampler, uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "SamplerState explicitSpaceSampler : register(s4, space1);" in generated_code
    assert "SamplerState spaceTexSampler : register(s5, space1);" in generated_code
    assert "SamplerState defaultSampler : register(s0);" in generated_code
    assert "SamplerState defaultTexSampler : register(s1);" in generated_code
    assert "Texture2D spaceTex : register(t3, space1);" in generated_code
    assert "Texture2D defaultTex : register(t0);" in generated_code
    assert "spaceTex.Sample(spaceTexSampler, uv)" in generated_code
    assert "defaultTex.Sample(defaultTexSampler, uv)" in generated_code


def test_hlsl_nonuniform_resource_index_infers_integer_descriptor_indices():
    code = """
    shader NonUniformDescriptorArrayIndex {
        Texture2D<float4> textures[4] @register(t0);
        SamplerState samplers[4] @register(s0);

        fragment {
            vec4 main(
                uint materialIndex @ TEXCOORD0,
                uint samplerIndex @ TEXCOORD1,
                vec2 uv @ TEXCOORD2
            ) @ SV_Target {
                let textureIndex = NonUniformResourceIndex(materialIndex);
                let samplerSlot = NonUniformResourceIndex(samplerIndex);
                return textures[textureIndex].Sample(samplers[samplerSlot], uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "Texture2D<float4> textures[4] : register(t0);" in generated_code
    assert "SamplerState samplers[4] : register(s0);" in generated_code
    assert (
        "uint textureIndex = NonUniformResourceIndex(materialIndex);" in generated_code
    )
    assert "uint samplerSlot = NonUniformResourceIndex(samplerIndex);" in generated_code
    assert "textures[textureIndex].Sample(samplers[samplerSlot], uv)" in generated_code
    assert "float textureIndex" not in generated_code
    assert "float samplerSlot" not in generated_code


def test_hlsl_nonuniform_resource_index_validates_arity_and_scalar_index_type():
    missing_argument_code = """
    shader MissingNonUniformResourceIndexArgument {
        fragment {
            vec4 main(vec2 uv @ TEXCOORD0) @ SV_Target {
                uint index = NonUniformResourceIndex();
                return vec4(float(index));
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="NonUniformResourceIndex requires exactly 1 argument, got 0",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(missing_argument_code), "fragment"
        )

    vector_index_code = """
    shader VectorNonUniformResourceIndexArgument {
        fragment {
            vec4 main(vec2 uv @ TEXCOORD0) @ SV_Target {
                uint index = NonUniformResourceIndex(uv);
                return vec4(float(index));
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="NonUniformResourceIndex index argument.*scalar int or uint.*float2",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(vector_index_code), "fragment"
        )


def test_hlsl_implicit_shadow_sampler_split_inherits_global_texture_register_space():
    code = """
    shader ImplicitShadowSamplerSpaces {
        sampler defaultSampler;
        sampler2DShadow shadowMap @register(t2, space3);

        fragment {
            float main(vec2 uv @TEXCOORD0, float depth) @gl_FragDepth {
                vec2 lod = textureQueryLod(shadowMap, uv);
                float cmp = textureCompare(shadowMap, uv, depth);
                return lod.x + cmp;
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "fragment"
    )

    assert "SamplerState defaultSampler : register(s0);" in generated_code
    assert "Texture2D shadowMap : register(t2, space3);" in generated_code
    assert (
        "SamplerComparisonState shadowMapSampler : register(s0, space3);"
        in generated_code
    )
    assert (
        "SamplerState shadowMapQuerySampler : register(s1, space3);" in generated_code
    )
    assert (
        "shadowMap.CalculateLevelOfDetailUnclamped(shadowMapQuerySampler, uv)"
        in generated_code
    )
    assert "shadowMap.SampleCmp(shadowMapSampler, uv, depth)" in generated_code


def test_hlsl_output_return_semantics_still_emit():
    depth_code = """
    shader ValidReturnSemantics {
        vertex {
            vec4 main() @gl_Position {
                return vec4(0.0);
            }
        }

        fragment {
            float main() @gl_FragDepth {
                return 0.5;
            }
        }
    }
    """
    ast = crosstl.translator.parse(depth_code)

    vertex_code = HLSLCodeGen().generate_stage(ast, "vertex")
    assert "float4 VSMain(): SV_POSITION" in vertex_code

    fragment_code = HLSLCodeGen().generate_stage(ast, "fragment")
    assert "float PSMain(): SV_DEPTH" in fragment_code

    color_code = """
    shader ValidColorReturnSemantic {
        fragment {
            vec4 main() @gl_FragColor1 {
                return vec4(1.0);
            }
        }
    }
    """
    color_output = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(color_code), "fragment"
    )
    assert "float4 PSMain(): SV_TARGET1" in color_output


@pytest.mark.parametrize(
    ("stage", "return_type", "semantic", "value", "hlsl_semantic", "expected_type"),
    [
        ("vertex", "float", "gl_Position", "0.0", "SV_POSITION", "float4"),
        ("fragment", "vec3", "gl_FragColor", "vec3(0.0)", "SV_TARGET", "float4"),
        ("fragment", "vec4", "gl_FragDepth", "vec4(0.0)", "SV_DEPTH", "float"),
        ("fragment", "float", "SV_TARGET1", "0.0", "SV_TARGET1", "float4"),
    ],
)
def test_hlsl_function_return_output_builtin_types_are_validated(
    stage, return_type, semantic, value, hlsl_semantic, expected_type
):
    code = f"""
    shader BadReturnBuiltinType {{
        {stage} {{
            {return_type} main() @{semantic} {{
                return {value};
            }}
        }}
    }}
    """
    with pytest.raises(ValueError, match=f"{hlsl_semantic}.*{expected_type}"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(code), stage)


@pytest.mark.parametrize(
    ("stage", "return_type", "semantic", "value", "hlsl_semantic"),
    [
        ("vertex", "vec4", "gl_FragColor", "vec4(0.0)", "SV_TARGET"),
        ("vertex", "float", "gl_FragDepth", "0.5", "SV_DEPTH"),
        ("fragment", "vec4", "gl_Position", "vec4(0.0)", "SV_POSITION"),
        ("compute", "vec4", "gl_Position", "vec4(0.0)", "SV_POSITION"),
    ],
)
def test_hlsl_function_return_output_builtin_stages_are_validated(
    stage, return_type, semantic, value, hlsl_semantic
):
    code = f"""
    shader BadReturnBuiltinStage {{
        {stage} {{
            {return_type} main() @{semantic} {{
                return {value};
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=f"{stage}.*{hlsl_semantic}"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(code), stage)


@pytest.mark.parametrize(
    ("stage", "semantic"),
    [
        ("vertex", "gl_Position"),
        ("fragment", "SV_TARGET0"),
    ],
)
def test_hlsl_void_function_return_semantics_are_rejected(stage, semantic):
    code = f"""
    shader BadVoidReturnSemantic {{
        {stage} {{
            void main() @{semantic} {{ }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=f"{stage}.*{semantic}.*void return type"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(code), stage)


def test_hlsl_struct_output_builtin_semantics_still_emit():
    code = """
    shader ValidStructOutputSemantics {
        struct VSOutput {
            vec4 position @ gl_Position;
        };

        struct FSOutput {
            vec4 color @ gl_FragColor1;
            float depth @ gl_FragDepth;
        };

        vertex {
            VSOutput main() {
                VSOutput output;
                output.position = vec4(0.0);
                return output;
            }
        }

        fragment {
            FSOutput main() {
                FSOutput output;
                output.color = vec4(1.0);
                output.depth = 0.5;
                return output;
            }
        }
    }
    """

    generated = HLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "float4 position: SV_POSITION;" in generated
    assert "float4 color: SV_TARGET1;" in generated
    assert "float depth: SV_DEPTH;" in generated


def test_hlsl_struct_array_member_semantics_are_preserved():
    code = """
    shader ArrayStructSemantics {
        struct FSInput {
            float weights[4] @ TEXCOORD0;
        };
    }
    """

    generated = HLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "float weights[4]: TEXCOORD0;" in generated


def test_hlsl_fragment_entry_input_structs_get_missing_member_semantics():
    code = """
    shader FragmentInputDefaultSemantics {
        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float lod;
            vec3 normal @ TEXCOORD2;
            float depth;
        };

        struct HelperInput {
            float value;
        };

        float helper(HelperInput input) {
            return input.value;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return vec4(input.uv, input.lod + input.depth + helper(HelperInput(1.0)), 1.0);
            }
        }
    }
    """

    generated = HLSLCodeGen().generate_stage(crosstl.translator.parse(code), "fragment")

    assert "float2 uv: TEXCOORD0;" in generated
    assert "float lod: TEXCOORD1;" in generated
    assert "float3 normal: TEXCOORD2;" in generated
    assert "float depth: TEXCOORD3;" in generated
    assert "float value;" in generated
    assert "float4 PSMain(FSInput input): SV_TARGET" in generated


def test_hlsl_legacy_array_node_struct_member_semantics_are_preserved():
    ast = create_legacy_shader_node(
        structs=[
            StructNode(
                "LegacyInput",
                [ArrayNode(PrimitiveType("float"), "weights", 4, "TEXCOORD0")],
            )
        ],
        functions=[],
        global_variables=[],
        cbuffers=[],
    )

    generated = HLSLCodeGen().generate(ast)

    assert "float weights[4]: TEXCOORD0;" in generated


def test_hlsl_struct_output_builtin_array_member_types_are_validated():
    code = """
    shader BadStructOutputArrayBuiltin {
        struct BadOutput {
            vec4 colors[2] @ gl_FragColor;
        };

        fragment {
            vec4 main() @ gl_FragColor {
                return vec4(0.0);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="SV_TARGET.*float4"):
        HLSLCodeGen().generate(crosstl.translator.parse(code))


@pytest.mark.parametrize(
    ("stage", "struct_name", "member_decl", "hlsl_semantic"),
    [
        ("vertex", "VSOutput", "vec4 color @ gl_FragColor", "SV_TARGET"),
        ("fragment", "FSOutput", "vec4 position @ gl_Position", "SV_POSITION"),
    ],
)
def test_hlsl_struct_return_output_builtin_stages_are_validated(
    stage, struct_name, member_decl, hlsl_semantic
):
    code = f"""
    shader BadStructReturnBuiltinStage {{
        struct {struct_name} {{
            {member_decl};
        }};

        {stage} {{
            {struct_name} main() {{
                {struct_name} output;
                return output;
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=f"{stage}.*{hlsl_semantic}"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(code), stage)


@pytest.mark.parametrize(
    ("member_decl", "hlsl_semantic", "expected_type"),
    [
        ("float position @ gl_Position", "SV_POSITION", "float4"),
        ("vec3 color @ gl_FragColor", "SV_TARGET", "float4"),
        ("vec4 depth @ gl_FragDepth", "SV_DEPTH", "float"),
        ("float rawColor @ SV_TARGET1", "SV_TARGET1", "float4"),
    ],
)
def test_hlsl_struct_output_builtin_types_are_validated(
    member_decl, hlsl_semantic, expected_type
):
    code = f"""
    shader BadStructOutputBuiltin {{
        struct BadOutput {{
            {member_decl};
        }};

        fragment {{
            vec4 main() @ gl_FragColor {{
                return vec4(0.0);
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=f"{hlsl_semantic}.*{expected_type}"):
        HLSLCodeGen().generate(crosstl.translator.parse(code))


@pytest.mark.parametrize(
    ("stage", "return_type", "semantic", "value"),
    [
        ("vertex", "uint", "gl_VertexID", "0"),
        ("vertex", "uint", "SV_VertexID", "0"),
        ("fragment", "vec4", "gl_FragCoord", "vec4(0.0)"),
        ("fragment", "bool", "gl_FrontFacing", "true"),
        ("fragment", "vec2", "gl_PointCoord", "vec2(0.0)"),
        ("compute", "uvec3", "gl_GlobalInvocationID", "uvec3(0u)"),
    ],
)
def test_hlsl_function_return_input_only_builtin_semantics_are_rejected(
    stage, return_type, semantic, value
):
    code = f"""
    shader BadReturnBuiltin {{
        {stage} {{
            {return_type} main() @{semantic} {{
                return {value};
            }}
        }}
    }}
    """
    with pytest.raises(ValueError, match=f"{semantic}.*return semantic"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(code), stage)


def test_hlsl_cbuffer_binding_attributes_drive_registers():
    code = """
    shader CBufferBindings {
        cbuffer Camera @register(b2) {
            mat4 viewProj;
            mat4 bones[2];
            vec4 viewRow;
        };

        uniform Material @binding(4) {
            vec4 tint;
        };

        cbuffer AutoBlock {
            vec4 color;
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "cbuffer Camera : register(b2)" in generated_code
    assert "float4x4 viewProj;" in generated_code
    assert "float4x4 bones[2];" in generated_code
    assert "cbuffer Material : register(b4)" in generated_code
    assert "cbuffer AutoBlock : register(b5)" in generated_code
    assert "MatrixType(" not in generated_code


def test_hlsl_cbuffer_register_overlap_raises():
    code = """
    shader CBufferOverlap {
        cbuffer Camera @register(b2) {
            mat4 viewProj;
        };

        cbuffer Material @binding(2) {
            vec4 tint;
        };
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting DirectX resource binding for 'Material': "
            "b2 overlaps 'Camera' b2"
        ),
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_hlsl_cbuffer_register_spaces_are_independent():
    code = """
    shader CBufferSpaces {
        cbuffer Camera @register(b2, space1) {
            mat4 viewProj;
        };

        cbuffer Material @register(b2, space2) {
            vec4 tint;
        };

        cbuffer AutoBlock {
            vec4 color;
        };
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "cbuffer Camera : register(b2, space1)" in generated_code
    assert "cbuffer Material : register(b2, space2)" in generated_code
    assert "cbuffer AutoBlock : register(b0)" in generated_code


def test_hlsl_default_float_image_scalar_and_vector_load_store():
    code = """
    shader DefaultFloatImageLoadStore {
        image2D storageImage;

        float touchScalar(image2D image, ivec2 pixel, float value) {
            float scalarOld = imageLoad(image, pixel);
            imageStore(image, pixel, scalarOld + value);
            return scalarOld;
        }

        vec4 touchVector(image2D image, ivec2 pixel, vec4 value) {
            vec4 vectorOld = imageLoad(image, pixel);
            imageStore(image, pixel, vectorOld + value);
            return vectorOld;
        }

        compute {
            void main() {
                float a = touchScalar(storageImage, ivec2(0, 1), 0.25);
                vec4 b = touchVector(storageImage, ivec2(2, 3), vec4(1.0));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "compute"
    )

    assert "float scalarOld = image[pixel].x;" in generated_code
    assert (
        "image[pixel] = float4((scalarOld + value), (scalarOld + value), "
        "(scalarOld + value), (scalarOld + value));" in generated_code
    )
    assert "float4 vectorOld = image[pixel];" in generated_code
    assert "image[pixel] = (vectorOld + value);" in generated_code
    assert "float4 vectorOld = image[pixel].x;" not in generated_code
    assert "image[pixel] = float4((vectorOld + value));" not in generated_code


def test_hlsl_rg_image_scalar_and_vector_load_store():
    code = """
    shader RGImageScalarVector {
        image2D rgFloat @rg32f;
        uimage2D rgUnsigned @rg32ui;

        float scalarFloat(image2D image @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloat(image2D image @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsigned(uimage2D image @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsigned(uimage2D image @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float a = scalarFloat(rgFloat, ivec2(0, 1), 0.25);
                vec2 b = vectorFloat(rgFloat, ivec2(2, 3), vec2(1.0));
                uint c = scalarUnsigned(rgUnsigned, ivec2(4, 5), 7u);
                uvec2 d = vectorUnsigned(rgUnsigned, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "compute"
    )

    assert "float oldValue = image[pixel].x;" in generated_code
    assert "uint oldValue = image[pixel].x;" in generated_code
    assert "image[pixel] = float2((oldValue + value), 0.0);" in generated_code
    assert "image[pixel] = uint2((oldValue + value), 0u);" in generated_code
    assert "float2 oldValue = image[pixel];" in generated_code
    assert "uint2 oldValue = image[pixel];" in generated_code
    assert "image[pixel] = (oldValue + value);" in generated_code


def test_struct():
    code = """
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("Struct codegen not implemented.")


def test_if_statement():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;

            if (input.texCoord.x > 0.5) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
            } else {
                output.color = vec4(0.0, 0.0, 0.0, 1.0);
            }

            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
        }
    }

    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            if (bloom > 0.5) {
                bloom = 0.5;
            } else {
                bloom = 0.0;
            }

            // Apply bloom to the texture color
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);

            return vec4(colorWithBloom, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("if statement codegen not implemented.")


def test_for_statement():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;

            for (int i = 0; i < 10; i++) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
            }
            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("for statement codegen not implemented.")


def test_for_statement_preserves_declaration_initializers():
    shader = """
    shader LoopDeclarationInitializers {
        float helper() {
            const float weights[2];
            int i = 0;
            float total = 0.0;
            for (int i = 0; i < 2; i++) {
                total = total + weights[0];
            }
            for (i = 0; i < 4; i++) {
                if (i == 0) {
                    continue;
                }
                break;
            }
            for (const int fixed = 0; fixed < 0;) {
                total = total + 1.0;
            }
            for (;;) {
                break;
            }
            return total;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "const float weights[2];" in generated_code
    assert "for (int i = 0; (i < 2); ++i)" in generated_code
    assert "for (i = 0; (i < 4); ++i)" in generated_code
    assert "for (const int fixed = 0; (fixed < 0); )" in generated_code
    assert "for (; ; )" in generated_code
    assert "continue;" in generated_code
    assert "break;" in generated_code
    assert "for (i; (i < 2); ++i)" not in generated_code
    assert "for (fixed; (fixed < 0); )" not in generated_code
    assert "BreakNode(" not in generated_code
    assert "ContinueNode(" not in generated_code


def test_loop_statement_lowers_to_while_true():
    shader = """
    shader LoopNodeSmoke {
        int helper(int limit) {
            int i = 0;
            loop {
                i = i + 1;
                if (i >= limit) {
                    break;
                }
            }
            return i;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "while (true)" in generated_code
    assert "i = (i + 1);" in generated_code
    assert "if ((i >= limit))" in generated_code
    assert "break;" in generated_code
    assert "return i;" in generated_code
    assert "LoopNode(" not in generated_code


def test_do_while_statement_lowers_to_c_style_syntax():
    shader = """
    shader DoWhileNodeSmoke {
        int helper(int limit) {
            int i = 0;
            do {
                i = i + 1;
                if (i >= limit) {
                    break;
                }
            } while (i < 4);
            return i;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "do {" in generated_code
    assert "i = (i + 1);" in generated_code
    assert "if ((i >= limit))" in generated_code
    assert "break;" in generated_code
    assert "} while ((i < 4));" in generated_code
    assert "return i;" in generated_code
    assert "DoWhileNode(" not in generated_code


def test_for_in_statement_lowers_to_counted_loop():
    shader = """
    shader ForInNodeSmoke {
        int helper(int limit) {
            int total = 0;
            for i in limit {
                total = total + i;
            }
            return total;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "for (int i = 0; i < limit; ++i)" in generated_code
    assert "total = (total + i);" in generated_code
    assert "return total;" in generated_code
    assert "ForInNode(" not in generated_code


def test_for_in_range_statement_lowers_to_counted_loop():
    shader = """
    shader ForInRangeNodeSmoke {
        int helper(int limit) {
            int total = 0;
            for i in 2..5 {
                total = total + i;
            }
            for j in 1..=limit {
                total = total + j;
            }
            return total;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "for (int i = 2; i < 5; ++i)" in generated_code
    assert "for (int j = 1; j <= limit; ++j)" in generated_code
    assert "total = (total + i);" in generated_code
    assert "total = (total + j);" in generated_code
    assert "return total;" in generated_code
    assert "RangeNode(" not in generated_code
    assert "ForInNode(" not in generated_code


def test_while_switch_and_void_return_emit_c_style_syntax():
    shader = """
    shader StatementLeakSmoke {
        void helper() {
            int i = 0;
            while (i < 4) {
                switch (i) {
                    case 0:
                        i = i + 1;
                        continue;
                    default:
                        break;
                }
            }
            return;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "while ((i < 4))" in generated_code
    assert "switch (i)" in generated_code
    assert "case 0:" in generated_code
    assert "default:" in generated_code
    assert "i = (i + 1);" in generated_code
    assert "continue;" in generated_code
    assert "break;" in generated_code
    assert "return;" in generated_code
    assert "WhileNode(" not in generated_code
    assert "SwitchNode(" not in generated_code
    assert "return ;" not in generated_code
    assert "return None;" not in generated_code


def test_switch_fallthrough_and_nested_switch_emit_c_style_syntax():
    shader = """
    shader SwitchEdgeSmoke {
        int helper(int mode, int submode) {
            int value = 0;
            switch (mode) {
                case 0:
                case 1:
                    value = value + 1;
                    break;
                case 2:
                    switch (submode) {
                        case 0:
                            value = value + 2;
                            break;
                        default:
                            value = value + 3;
                            break;
                    }
                    break;
                default:
                    value = value + 4;
                    break;
            }
            return value;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" in generated_code
    assert "case 0:\n        case 1: {" in generated_code
    assert "case 2: {\n            switch (submode)" in generated_code
    assert generated_code.count("default:") == 2
    assert "value = (value + 1);" in generated_code
    assert "value = (value + 2);" in generated_code
    assert "value = (value + 3);" in generated_code
    assert "value = (value + 4);" in generated_code
    assert "return value;" in generated_code
    assert "SwitchNode(" not in generated_code
    assert "CaseNode(" not in generated_code


def test_switch_and_match_case_blocks_scope_local_declarations():
    shader = """
    shader CaseScopeSmoke {
        int switchHelper(int mode) {
            int value = 0;
            switch (mode) {
                case 0:
                case 1:
                    int scoped = value + 1;
                    value = scoped;
                    break;
                default:
                    int scoped = value + 2;
                    value = scoped;
                    break;
            }
            return value;
        }

        int matchHelper(int mode) {
            int value = 0;
            match mode {
                0 => { int scoped = value + 1; value = scoped; }
                1 => { int scoped = value + 2; value = scoped; }
                _ => { int scoped = value + 3; value = scoped; }
            }
            return value;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "case 0:\n        case 1: {" in generated_code
    assert generated_code.count("case 0: {") == 1
    assert generated_code.count("int scoped") == 5
    assert "default: {" in generated_code
    assert "MatchNode(" not in generated_code
    assert "SwitchNode(" not in generated_code


def test_match_literal_and_wildcard_arms_lower_to_switch():
    shader = """
    shader MatchLeakSmoke {
        int helper(int mode) {
            int value = 0;
            match mode {
                0 => { value = 1; }
                1 => { value = 2; }
                _ => { value = 3; }
            }
            return value;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" in generated_code
    assert "case 0:" in generated_code
    assert "case 1:" in generated_code
    assert "default:" in generated_code
    assert "value = 1;" in generated_code
    assert "value = 2;" in generated_code
    assert "value = 3;" in generated_code
    assert generated_code.count("break;") == 3
    assert "return value;" in generated_code
    assert "MatchNode(" not in generated_code
    assert "MatchArmNode(" not in generated_code


def test_match_return_arms_do_not_emit_extra_breaks():
    shader = """
    shader MatchReturnArms {
        int helper(int mode) {
            match mode {
                0 => { return 1; }
                _ => { return 2; }
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" in generated_code
    assert "case 0:" in generated_code
    assert "default:" in generated_code
    assert "return 1;" in generated_code
    assert "return 2;" in generated_code
    assert "break;" not in generated_code
    assert "MatchNode(" not in generated_code


def test_match_guarded_literal_and_wildcard_arms_lower_to_if_chain():
    shader = """
    shader MatchGuardPattern {
        int helper(int mode) {
            int value = 0;
            match mode {
                0 if mode > 0 => { value = 1; }
                0 => { value = 2; }
                _ if mode < 10 => { value = 3; }
                _ => { value = 4; }
            }
            return value;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" not in generated_code
    assert "if (((mode == 0) &&" in generated_code
    assert "mode > 0" in generated_code
    assert "else if ((mode == 0))" in generated_code
    assert "else if" in generated_code
    assert "mode < 10" in generated_code
    assert "else {" in generated_code
    assert "value = 1;" in generated_code
    assert "value = 2;" in generated_code
    assert "value = 3;" in generated_code
    assert "value = 4;" in generated_code
    assert "MatchNode(" not in generated_code


def test_match_identifier_binding_arm_lowers_to_scoped_else_body():
    shader = """
    shader MatchBindingPattern {
        int helper(int mode) {
            int value = 0;
            match mode {
                0 => { value = 1; }
                other => { value = other; }
            }
            return value;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" not in generated_code
    assert "if ((mode == 0))" in generated_code
    assert "else {" in generated_code
    assert "int other = mode;" in generated_code
    assert "value = other;" in generated_code
    assert "MatchNode(" not in generated_code


def test_match_guarded_identifier_binding_falls_through_to_later_arm():
    shader = """
    shader MatchGuardedBindingPattern {
        int helper(int mode) {
            int value = 0;
            match mode {
                0 => { value = 1; }
                candidate if candidate > 2 => { value = candidate; }
                _ => { value = 7; }
            }
            return value;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "switch (mode)" not in generated_code
    assert "if ((mode == 0))" in generated_code
    assert "else {" in generated_code
    assert "int candidate = mode;" in generated_code
    assert "candidate > 2" in generated_code
    assert "value = candidate;" in generated_code
    assert "value = 7;" in generated_code
    assert "MatchNode(" not in generated_code


def test_match_plain_struct_pattern_binds_fields():
    shader = """
    shader MatchStructPattern {
        struct Pair {
            int left;
            int right;
        };

        int helper(Pair pair) {
            int value = 0;
            match pair {
                Pair { left, right } => { value = left + right; }
            }
            return value;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "int left = pair.left;" in generated_code
    assert "int right = pair.right;" in generated_code
    assert "value = (left + right);" in generated_code
    assert "MatchNode(" not in generated_code


def test_match_enum_path_pattern_lowers_to_integer_constants():
    shader = """
    shader MatchEnumPathPattern {
        enum Mode {
            Add,
            Multiply = 4,
            Divide
        }

        int helper(Mode mode) {
            int value = Mode::Divide;
            match mode {
                Mode::Add => { value = 1; }
                Mode::Multiply => { value = 2; }
                _ => { value = 3; }
            }
            return value;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int Mode_Add = 0;" in generated_code
    assert "static const int Mode_Multiply = 4;" in generated_code
    assert "static const int Mode_Divide = 5;" in generated_code
    assert "int helper(int mode)" in generated_code
    assert "int value = Mode_Divide;" in generated_code
    assert "switch (mode)" not in generated_code
    assert "if ((mode == Mode_Add))" in generated_code
    assert "else if ((mode == Mode_Multiply))" in generated_code
    assert "value = 3;" in generated_code
    assert "MatchNode(" not in generated_code


def test_match_struct_enum_pattern_binds_named_payload_fields():
    shader = """
    shader MatchStructEnumPattern {
        enum LightingModel {
            Phong {
                ambient: vec3,
                diffuse: vec3,
                shininess: float
            },
            Toon {
                base_color: vec3,
                levels: int
            }
        }

        vec3 shade(LightingModel model) {
            vec3 result = vec3(0.0);
            match model {
                LightingModel::Phong { ambient, diffuse, shininess } => {
                    result = ambient + diffuse * shininess;
                },
                LightingModel::Toon { base_color, .. } => {
                    result = base_color;
                }
            }
            return result;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int LightingModel_Phong = 0;" in generated_code
    assert "static const int LightingModel_Toon = 1;" in generated_code
    assert "struct LightingModel {" in generated_code
    assert "int variant;" in generated_code
    assert "float3 ambient;" in generated_code
    assert "float3 base_color;" in generated_code
    assert "float3 shade(LightingModel model)" in generated_code
    assert "if ((model.variant == LightingModel_Phong))" in generated_code
    assert "else if ((model.variant == LightingModel_Toon))" in generated_code
    assert "float3 ambient = model.ambient;" in generated_code
    assert "float shininess = model.shininess;" in generated_code
    assert "float3 base_color = model.base_color;" in generated_code
    assert "LightingModel::" not in generated_code
    assert "MatchNode(" not in generated_code


def test_match_tuple_enum_pattern_binds_payload_fields():
    shader = """
    shader MatchTupleEnumPattern {
        enum MaybeInt {
            Value(int),
            Pair(int, float),
            Missing
        }

        int read(MaybeInt item) {
            match item {
                MaybeInt::Value(value) => { return value; },
                MaybeInt::Pair(left, scale) => { return left + int(scale); },
                MaybeInt::Missing => { return 0; }
            }
        }

        MaybeInt make(int value) {
            return MaybeInt::Value(value);
        }

        MaybeInt none() {
            return MaybeInt::Missing;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int MaybeInt_Value = 0;" in generated_code
    assert "static const int MaybeInt_Pair = 1;" in generated_code
    assert "static const int MaybeInt_Missing = 2;" in generated_code
    assert "struct MaybeInt {" in generated_code
    assert "int Value_0;" in generated_code
    assert "int Pair_0;" in generated_code
    assert "float Pair_1;" in generated_code
    assert "MaybeInt MaybeInt_Value_make(int payload0)" in generated_code
    assert "MaybeInt MaybeInt_Missing_make()" in generated_code
    assert "if ((item.variant == MaybeInt_Value))" in generated_code
    assert "else if ((item.variant == MaybeInt_Pair))" in generated_code
    assert "int value = item.Value_0;" in generated_code
    assert "int left = item.Pair_0;" in generated_code
    assert "float scale = item.Pair_1;" in generated_code
    assert "return MaybeInt_Value_make(value);" in generated_code
    assert "return MaybeInt_Missing_make();" in generated_code
    assert "MaybeInt::" not in generated_code
    assert "MatchNode(" not in generated_code


def test_string_payload_enum_maps_to_shader_token_type():
    shader = """
    shader StringPayloadEnum {
        enum ShaderError {
            TextureError(str),
            BufferError(str),
            InvalidState
        }

        ShaderError texture_error(int code) {
            return ShaderError::TextureError(code);
        }

        ShaderError invalid_state() {
            return ShaderError::InvalidState;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct ShaderError {" in generated_code
    assert "int TextureError_0;" in generated_code
    assert "int BufferError_0;" in generated_code
    assert "ShaderError ShaderError_TextureError_make(int payload0)" in generated_code
    assert "ShaderError ShaderError_InvalidState_make()" in generated_code
    assert "return ShaderError_TextureError_make(code);" in generated_code
    assert "return ShaderError_InvalidState_make();" in generated_code
    assert re.search(r"\bstr\b", generated_code) is None


def test_tuple_enum_constructor_wrong_arity_raises():
    shader = """
    shader TupleEnumConstructorArity {
        enum MaybeInt {
            Value(int),
            Missing
        }

        MaybeInt make() {
            return MaybeInt::Value();
        }
    }
    """

    with pytest.raises(
        ValueError, match="Enum constructor MaybeInt::Value expects 1 arguments, got 0"
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_generic_enum_struct_concrete_match_and_constructors():
    shader = """
    shader GenericEnumConcrete {
        generic<T> struct Option {
            enum OptionType { Some(T), None }
            OptionType variant;
        }

        generic<T, E> struct Result {
            enum ResultType { Ok(T), Err(E) }
            ResultType variant;
        }

        enum MathError {
            DivisionByZero
        }

        Option<int> some(int value) {
            return Option::Some(value);
        }

        Option<int> none() {
            return Option::None;
        }

        int read_option(Option<int> item) {
            match item {
                Option::Some(value) => { return value; },
                Option::None => { return 0; }
            }
        }

        Result<int, MathError> make_result(bool ok) {
            match ok {
                true => { return Result::Ok(1); },
                false => { return Result::Err(MathError::DivisionByZero); }
            }
        }

        int read_result(Result<int, MathError> item) {
            match item {
                Result::Ok(value) => { return value; },
                Result::Err(_) => { return 0; }
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int Option_Some = 0;" in generated_code
    assert "static const int Result_Ok = 0;" in generated_code
    assert "struct Option_int {" in generated_code
    assert "struct Result_int_MathError {" in generated_code
    assert "struct Option {" not in generated_code
    assert "struct Result {" not in generated_code
    assert "Option_int Option_int_Some_make(int payload0)" in generated_code
    assert "Option_int Option_int_None_make()" in generated_code
    assert (
        "Result_int_MathError Result_int_MathError_Ok_make(int payload0)"
        in generated_code
    )
    assert (
        "Result_int_MathError Result_int_MathError_Err_make(int payload0)"
        in generated_code
    )
    assert "Option_int some(int value)" in generated_code
    assert "Option_int none()" in generated_code
    assert "Result_int_MathError make_result(bool ok)" in generated_code
    assert "int read_result(Result_int_MathError item)" in generated_code
    assert "return Option_int_Some_make(value);" in generated_code
    assert "return Option_int_None_make();" in generated_code
    assert "return Result_int_MathError_Ok_make(1);" in generated_code
    assert (
        "return Result_int_MathError_Err_make(MathError_DivisionByZero);"
        in generated_code
    )
    assert "if ((item.variant == Option_Some))" in generated_code
    assert "else if ((item.variant == Option_None))" in generated_code
    assert "if ((item.variant == Result_Ok))" in generated_code
    assert "else if ((item.variant == Result_Err))" in generated_code
    assert "int value = item.Some_0;" in generated_code
    assert "int value = item.Ok_0;" in generated_code
    assert "Option<" not in generated_code
    assert "Result<" not in generated_code
    assert "Option::" not in generated_code
    assert "Result::" not in generated_code


def test_generic_enum_function_call_match_infers_result_type_once():
    shader = """
    shader FunctionCallResultMatch {
        generic<T, E> struct Result {
            enum ResultType { Ok(T), Err(E) }
            ResultType variant;
        }

        enum MathError {
            DivisionByZero
        }

        Result<int, MathError> make_result(bool ok) {
            match ok {
                true => { return Result::Ok(7); },
                false => { return Result::Err(MathError::DivisionByZero); }
            }
        }

        int read(bool ok) {
            match make_result(ok) {
                Result::Ok(value) => { return value; },
                Result::Err(_) => { return -1; }
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Result_int_MathError make_result(bool ok)" in generated_code
    assert (
        "Result_int_MathError __crossgl_match_subject_0 = make_result(ok);"
        in generated_code
    )
    assert generated_code.count("make_result(ok)") == 1
    assert "if ((__crossgl_match_subject_0.variant == Result_Ok))" in generated_code
    assert (
        "else if ((__crossgl_match_subject_0.variant == Result_Err))" in generated_code
    )
    assert "int value = __crossgl_match_subject_0.Ok_0;" in generated_code
    assert "Result::" not in generated_code
    assert "MatchNode(" not in generated_code


def test_generic_enum_local_function_call_type_infers_match_subject():
    shader = """
    shader LocalResultMatch {
        generic<T, E> struct Result {
            enum ResultType { Ok(T), Err(E) }
            ResultType variant;
        }

        enum MathError {
            DivisionByZero
        }

        Result<int, MathError> make_result(bool ok) {
            match ok {
                true => { return Result::Ok(7); },
                false => { return Result::Err(MathError::DivisionByZero); }
            }
        }

        int read(bool ok) {
            let item = make_result(ok);
            match item {
                Result::Ok(value) => { return value; },
                Result::Err(_) => { return -1; }
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Result_int_MathError item = make_result(ok);" in generated_code
    assert "float item = make_result(ok);" not in generated_code
    assert "if ((item.variant == Result_Ok))" in generated_code
    assert "else if ((item.variant == Result_Err))" in generated_code
    assert "int value = item.Ok_0;" in generated_code
    assert "Result<" not in generated_code
    assert "ConstructorNode(" not in generated_code
    assert "MatchNode(" not in generated_code


def test_generic_enum_match_expression_initializes_vector_local():
    shader = """
    shader MatchExpressionValue {
        generic<T, E> struct Result {
            enum ResultType { Ok(T), Err(E) }
            ResultType variant;
        }

        enum MathError {
            DivisionByZero
        }

        Result<vec3, MathError> make_result(bool ok) {
            match ok {
                true => { return Result::Ok(vec3(1.0, 2.0, 3.0)); },
                false => { return Result::Err(MathError::DivisionByZero); }
            }
        }

        vec3 read(bool ok, vec3 fallback) {
            let value = match make_result(ok) {
                Result::Ok(actual) => actual,
                Result::Err(_) => fallback
            };
            return value;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Result_float3_MathError make_result(bool ok)" in generated_code
    assert "float3 read(bool ok, float3 fallback)" in generated_code
    assert "float3 value;" in generated_code
    assert (
        "Result_float3_MathError __crossgl_match_subject_0 = make_result(ok);"
        in generated_code
    )
    assert "if ((__crossgl_match_subject_0.variant == Result_Ok))" in generated_code
    assert (
        "else if ((__crossgl_match_subject_0.variant == Result_Err))" in generated_code
    )
    assert "float3 actual = __crossgl_match_subject_0.Ok_0;" in generated_code
    assert "value = actual;" in generated_code
    assert "value = fallback;" in generated_code
    assert "float value = MatchNode" not in generated_code
    assert "MatchNode(" not in generated_code


def test_enum_struct_variant_constructor_expression():
    shader = """
    shader EnumStructVariantConstructor {
        enum RenderOutput {
            Clear { color: vec4, depth: float },
            StateSet
        }

        RenderOutput clear(vec4 color, float depth) {
            return RenderOutput::Clear { color, depth };
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "RenderOutput RenderOutput_Clear_make(float4 payload0, float payload1)"
        in generated_code
    )
    assert "RenderOutput clear(float4 color, float depth)" in generated_code
    assert "return RenderOutput_Clear_make(color, depth);" in generated_code
    assert "ConstructorNode(" not in generated_code
    assert "RenderOutput::" not in generated_code


def test_tail_expression_returns_struct_constructor_and_vector():
    shader = """
    shader TailExpressionReturn {
        struct Pair {
            value: vec3,
            weight: float
        }

        Pair make_pair(vec3 value, float weight) {
            Pair { value, weight }
        }

        vec4 make_color(vec3 color) {
            vec4(color, 1.0)
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Pair make_pair(float3 value, float weight)" in generated_code
    assert "return Pair(value, weight);" in generated_code
    assert "float4 make_color(float3 color)" in generated_code
    assert "return float4(color, 1.0);" in generated_code
    assert "ConstructorNode(" not in generated_code


def test_stage_tail_struct_constructor_returns_stage_output():
    shader = """
    shader StageTailReturn {
        vertex {
            struct VertexInput {
                position: vec3,
                color: vec4
            }
            struct VertexOutput {
                position: vec4,
                color: vec4
            }
            VertexOutput main(VertexInput input) {
                VertexOutput {
                    position: vec4(input.position, 1.0),
                    color: input.color
                }
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct VertexInput {" in generated_code
    assert "struct VertexOutput {" in generated_code
    assert "VertexOutput VSMain(VertexInput input)" in generated_code
    assert (
        "return VertexOutput(float4(input.position, 1.0), input.color);"
        in generated_code
    )
    assert "ConstructorNode(" not in generated_code


def test_trait_self_return_does_not_emit_generic_enum_specialization():
    shader = """
    shader TraitSelfOption {
        generic<T> struct Option {
            enum OptionType { Some(T), None }
            OptionType variant;
        }

        trait VectorOps {
            fn normalize(self) -> Option<Self>;
        }

        int passthrough(int value) {
            return value;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct Option_Self" not in generated_code
    assert "Option_Self" not in generated_code
    assert "Self Some_0" not in generated_code
    assert "int passthrough(int value)" in generated_code


def test_generic_function_call_emits_concrete_specialization():
    shader = """
    shader GenericFunctionSpecialization {
        generic<T> fn add_one(value: T) -> T {
            return value.add(T::one());
        }

        float use_add_one(float value) {
            return add_one(value);
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "float add_one_float(float value)" in generated_code
    assert "return (value + 1.0);" in generated_code
    assert "return add_one_float(value);" in generated_code
    assert "T add_one(T value)" not in generated_code
    assert "return add_one(value);" not in generated_code
    assert ".add(" not in generated_code
    assert "T::one" not in generated_code


def test_generic_struct_concrete_constructor_and_member_access():
    shader = """
    shader GenericStructConcrete {
        generic<T> struct Box {
            value: T;
        }

        generic<T> struct PairBox {
            first: Box<T>;
            second: T;
        }

        Box<int> make(int value) {
            return Box { value: value };
        }

        int read(Box<int> item) {
            return item.value;
        }

        PairBox<int> make_pair(int value) {
            return PairBox {
                first: Box { value: value },
                second: value
            };
        }

        int read_pair(PairBox<int> item) {
            return item.first.value + item.second;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct Box_int {" in generated_code
    assert "int value;" in generated_code
    assert "struct PairBox_int {" in generated_code
    assert "Box_int first;" in generated_code
    assert "int second;" in generated_code
    assert "struct Box {" not in generated_code
    assert "struct PairBox {" not in generated_code
    assert "Box_int make(int value)" in generated_code
    assert "return Box_int(value);" in generated_code
    assert "int read(Box_int item)" in generated_code
    assert "return item.value;" in generated_code
    assert "PairBox_int make_pair(int value)" in generated_code
    assert "return PairBox_int(Box_int(value), value);" in generated_code
    assert "int read_pair(PairBox_int item)" in generated_code
    assert "item.first.value + item.second" in generated_code
    assert "Box<" not in generated_code
    assert "PairBox<" not in generated_code
    assert "ConstructorNode(" not in generated_code


def test_struct_constructor_and_inferred_generic_constructor_local():
    shader = """
    shader StructConstructors {
        struct Geometry {
            center: vec3;
            normal: vec3;
        }

        generic<T> struct Box {
            value: T;
        }

        Geometry make_geometry(vec3 center, vec3 normal) {
            return Geometry { center: center, normal: normal };
        }

        Box<float> make_box(vec3 value) {
            let normalized = normalize(value);
            let wrapped = Box { value: normalized.x };
            return wrapped;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct Geometry {" in generated_code
    assert "struct Box_float {" in generated_code
    assert "Geometry make_geometry(float3 center, float3 normal)" in generated_code
    assert "return Geometry(center, normal);" in generated_code
    assert "Box_float make_box(float3 value)" in generated_code
    assert "float3 normalized = normalize(value);" in generated_code
    assert "Box_float wrapped = Box_float(normalized.x);" in generated_code
    assert "return wrapped;" in generated_code
    assert "Option_Self normalized" not in generated_code
    assert "Box<" not in generated_code
    assert "ConstructorNode(" not in generated_code


def test_ray_payload_semantics():
    code = """
    shader rt {
        struct Payload {
            vec3 color;
        };
        ray_miss {
            void main(Payload payload @ payload) {
                payload.color = vec3(1.0, 0.0, 0.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert '[shader("miss")]' in generated
    assert "void MissMain(inout Payload payload)" in generated
    assert ": payload" not in generated


def test_ray_and_mesh_shader_attributes():
    code = """
    shader rt {
        ray_generation {
            void main() { }
        }
        mesh {
            void main() { }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)
    assert '[shader("raygeneration")]' in generated
    assert '[shader("mesh")]' in generated
    assert "void RayGenMain()" in generated
    assert "void MSMain()" in generated


def test_directx_ray_stage_semantic_parameters_emit_and_validate():
    shader = """
    shader RayStageSemantics {
        struct RayPayload {
            vec3 color;
        };

        struct CallableData {
            uint value;
        };

        ray_closest_hit {
            void main(
                RayPayload payload @ payload,
                BuiltInTriangleIntersectionAttributes attributes @ hit_attribute
            ) {
                payload.color = vec3(attributes.barycentrics, 1.0);
            }
        }

        ray_callable {
            void main(CallableData data @ callable_data) {
                data.value = 1u;
            }
        }

        ray_miss {
            void main(RayPayload payload @ payload) {
                payload.color = vec3(0.0, 0.0, 0.0);
            }
        }
    }
    """

    generated = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert '[shader("closesthit")]' in generated
    assert (
        "void ClosestHitMain(inout RayPayload payload, "
        "in BuiltInTriangleIntersectionAttributes attributes)"
    ) in generated
    assert '[shader("callable")]' in generated
    assert "void CallableMain(inout CallableData data)" in generated
    assert '[shader("miss")]' in generated
    assert "void MissMain(inout RayPayload payload)" in generated
    assert ": payload" not in generated
    assert ": hit_attribute" not in generated
    assert ": callable_data" not in generated
    assert "struct BuiltInTriangleIntersectionAttributes" not in generated


def test_directx_ray_stage_semantic_parameters_reject_invalid_stages_and_types():
    raygen_payload_code = """
    shader BadRaygenPayload {
        struct RayPayload {
            vec3 color;
        };

        ray_generation {
            void main(RayPayload payload @ payload) { }
        }
    }
    """
    with pytest.raises(ValueError, match="ray_generation.*payload"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(raygen_payload_code), "ray_generation"
        )

    callable_payload_code = """
    shader BadCallablePayload {
        struct RayPayload {
            vec3 color;
        };

        ray_callable {
            void main(RayPayload payload @ payload) { }
        }
    }
    """
    with pytest.raises(ValueError, match="ray_callable.*payload"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(callable_payload_code), "ray_callable"
        )

    miss_hit_attribute_code = """
    shader BadMissHitAttribute {
        struct HitAttributes {
            vec2 barycentrics;
        };

        ray_miss {
            void main(HitAttributes attributes @ hit_attribute) { }
        }
    }
    """
    with pytest.raises(ValueError, match="ray_miss.*hit_attribute"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(miss_hit_attribute_code), "ray_miss"
        )

    scalar_payload_code = """
    shader BadScalarRayPayload {
        ray_miss {
            void main(float payload @ payload) { }
        }
    }
    """
    with pytest.raises(ValueError, match="payload.*user-defined struct"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(scalar_payload_code), "ray_miss"
        )

    builtin_payload_code = """
    shader BadBuiltinRayPayload {
        ray_miss {
            void main(BuiltInTriangleIntersectionAttributes payload @ payload) { }
        }
    }
    """
    with pytest.raises(ValueError, match="payload.*user-defined struct"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(builtin_payload_code), "ray_miss"
        )

    builtin_callable_data_code = """
    shader BadBuiltinCallableData {
        ray_callable {
            void main(BuiltInTriangleIntersectionAttributes data @ callable_data) { }
        }
    }
    """
    with pytest.raises(ValueError, match="callable_data.*user-defined struct"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(builtin_callable_data_code), "ray_callable"
        )


def test_directx_ray_stage_semantic_parameters_reject_duplicates_order_and_extras():
    duplicate_payload_code = """
    shader DuplicateRayPayload {
        struct RayPayload {
            vec3 color;
        };

        ray_miss {
            void main(RayPayload first @ payload, RayPayload second @ payload) { }
        }
    }
    """
    with pytest.raises(ValueError, match="at most one payload"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(duplicate_payload_code), "ray_miss"
        )

    wrong_order_code = """
    shader WrongRayHitOrder {
        struct RayPayload {
            vec3 color;
        };

        struct HitAttributes {
            vec2 barycentrics;
        };

        ray_closest_hit {
            void main(
                HitAttributes attributes @ hit_attribute,
                RayPayload payload @ payload
            ) { }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="parameters must be declared as payload, hit_attribute",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(wrong_order_code), "ray_closest_hit"
        )

    extra_closest_hit_parameter_code = """
    shader ExtraClosestHitParameter {
        struct RayPayload {
            vec3 color;
        };

        struct HitAttributes {
            vec2 barycentrics;
        };

        ray_closest_hit {
            void main(
                RayPayload payload @ payload,
                HitAttributes attributes @ hit_attribute,
                uint primitiveId
            ) { }
        }
    }
    """
    with pytest.raises(ValueError, match="primitiveId.*payload, hit_attribute"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(extra_closest_hit_parameter_code),
            "ray_closest_hit",
        )

    intersection_parameter_code = """
    shader IntersectionWithParameter {
        ray_intersection {
            void main(uint primitiveId) { }
        }
    }
    """
    with pytest.raises(ValueError, match="ray_intersection.*must not declare"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(intersection_parameter_code), "ray_intersection"
        )


def test_directx_ray_stage_semantic_parameters_require_stage_roles():
    miss_missing_payload_code = """
    shader MissMissingPayload {
        ray_miss {
            void main() { }
        }
    }
    """
    with pytest.raises(ValueError, match="ray_miss.*requires.*payload"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(miss_missing_payload_code), "ray_miss"
        )

    closest_missing_attribute_code = """
    shader ClosestHitMissingAttribute {
        struct RayPayload {
            vec3 color;
        };

        ray_closest_hit {
            void main(RayPayload payload @ payload) { }
        }
    }
    """
    with pytest.raises(ValueError, match="ray_closest_hit.*hit_attribute"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(closest_missing_attribute_code),
            "ray_closest_hit",
        )

    callable_missing_data_code = """
    shader CallableMissingData {
        ray_callable {
            void main() { }
        }
    }
    """
    with pytest.raises(ValueError, match="ray_callable.*callable_data"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(callable_missing_data_code), "ray_callable"
        )


def test_directx_ray_tracing_intrinsics_validate_stage_and_arity():
    wrong_stage_code = """
    shader BadTraceRayStage {
        compute {
            void main() {
                TraceRay(1, 2, 3, 4, 5, 6, 7, 8);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="compute.*cannot call TraceRay"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(wrong_stage_code), "compute"
        )

    wrong_arity_code = """
    shader BadTraceRayArity {
        ray_generation {
            void main() {
                TraceRay(1, 2, 3);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="TraceRay requires 8 or 11"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(wrong_arity_code), "ray_generation"
        )

    report_hit_stage_code = """
    shader BadReportHitStage {
        ray_any_hit {
            void main() {
                ReportHit(1.0, 0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="ray_any_hit.*cannot call ReportHit"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(report_hit_stage_code), "ray_any_hit"
        )

    any_hit_ops_code = """
    shader ValidAnyHitOps {
        struct RayPayload {
            vec3 color;
        };

        struct HitAttributes {
            vec2 barycentrics;
        };

        ray_any_hit {
            void main(
                RayPayload payload @ payload,
                HitAttributes attributes @ hit_attribute
            ) {
                IgnoreHit();
                AcceptHitAndEndSearch();
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(any_hit_ops_code), "ray_any_hit"
    )
    assert "IgnoreHit();" in generated
    assert "AcceptHitAndEndSearch();" in generated


def test_directx_ray_tracing_intrinsics_validate_payload_arguments():
    valid_trace_ray_code = """
    shader ValidTraceRayPayload {
        struct RayPayload {
            vec3 color;
        };

        ray_generation {
            void main() {
                RaytracingAccelerationStructure accel;
                RayDesc ray;
                RayPayload payload;
                TraceRay(accel, 0, 0xFF, 0, 1, 0, ray, payload);
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(valid_trace_ray_code), "ray_generation"
    )
    assert "TraceRay(accel, 0, 255, 0, 1, 0, ray, payload);" in generated

    valid_expanded_trace_ray_code = """
    shader ValidExpandedTraceRayPayload {
        struct RayPayload {
            vec3 color;
        };

        ray_generation {
            void main() {
                RaytracingAccelerationStructure accel;
                RayPayload payload;
                TraceRay(
                    accel,
                    0,
                    0xFF,
                    0,
                    1,
                    0,
                    vec3(0.0, 0.0, 0.0),
                    0.0,
                    vec3(0.0, 0.0, 1.0),
                    100.0,
                    payload
                );
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(valid_expanded_trace_ray_code), "ray_generation"
    )
    assert (
        "TraceRay(accel, 0, 255, 0, 1, 0, float3(0.0, 0.0, 0.0), 0.0, "
        "float3(0.0, 0.0, 1.0), 100.0, payload);"
    ) in generated

    bad_acceleration_structure_code = """
    shader BadTraceRayAccelerationStructure {
        struct RayPayload {
            vec3 color;
        };

        ray_generation {
            void main() {
                uint accel;
                RayDesc ray;
                RayPayload payload;
                TraceRay(accel, 0, 0xFF, 0, 1, 0, ray, payload);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="TraceRay acceleration structure.*RaytracingAccelerationStructure",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(bad_acceleration_structure_code),
            "ray_generation",
        )

    bad_ray_descriptor_code = """
    shader BadTraceRayDescriptor {
        struct RayPayload {
            vec3 color;
        };

        ray_generation {
            void main() {
                RaytracingAccelerationStructure accel;
                vec3 ray;
                RayPayload payload;
                TraceRay(accel, 0, 0xFF, 0, 1, 0, ray, payload);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="TraceRay ray descriptor.*RayDesc"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(bad_ray_descriptor_code),
            "ray_generation",
        )

    scalar_trace_ray_payload_code = """
    shader BadTraceRayScalarPayload {
        ray_generation {
            void main() {
                RaytracingAccelerationStructure accel;
                RayDesc ray;
                uint payload;
                TraceRay(accel, 0, 0xFF, 0, 1, 0, ray, payload);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="TraceRay payload.*user-defined struct"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(scalar_trace_ray_payload_code),
            "ray_generation",
        )

    expanded_trace_ray_payload_code = """
    shader BadExpandedTraceRayScalarPayload {
        ray_generation {
            void main() {
                RaytracingAccelerationStructure accel;
                uint payload;
                TraceRay(
                    accel,
                    0,
                    0xFF,
                    0,
                    1,
                    0,
                    vec3(0.0, 0.0, 0.0),
                    0.0,
                    vec3(0.0, 0.0, 1.0),
                    100.0,
                    payload
                );
            }
        }
    }
    """
    with pytest.raises(ValueError, match="TraceRay payload.*user-defined struct"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(expanded_trace_ray_payload_code),
            "ray_generation",
        )


def test_directx_ray_tracing_intrinsics_validate_trace_ray_expanded_shape():
    bad_origin_code = """
    shader BadExpandedTraceRayOrigin {
        struct RayPayload {
            vec3 color;
        };

        ray_generation {
            void main() {
                RaytracingAccelerationStructure accel;
                RayPayload payload;
                TraceRay(
                    accel,
                    0,
                    0xFF,
                    0,
                    1,
                    0,
                    vec2(0.0, 0.0),
                    0.0,
                    vec3(0.0, 0.0, 1.0),
                    100.0,
                    payload
                );
            }
        }
    }
    """
    with pytest.raises(ValueError, match="TraceRay origin.*float3"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(bad_origin_code),
            "ray_generation",
        )

    bad_min_distance_code = """
    shader BadExpandedTraceRayMinDistance {
        struct RayPayload {
            vec3 color;
        };

        ray_generation {
            void main() {
                RaytracingAccelerationStructure accel;
                uint minDistance;
                RayPayload payload;
                TraceRay(
                    accel,
                    0,
                    0xFF,
                    0,
                    1,
                    0,
                    vec3(0.0, 0.0, 0.0),
                    minDistance,
                    vec3(0.0, 0.0, 1.0),
                    100.0,
                    payload
                );
            }
        }
    }
    """
    with pytest.raises(ValueError, match="TraceRay minimum distance.*scalar floating"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(bad_min_distance_code),
            "ray_generation",
        )

    bad_direction_code = """
    shader BadExpandedTraceRayDirection {
        struct RayPayload {
            vec3 color;
        };

        ray_generation {
            void main() {
                RaytracingAccelerationStructure accel;
                RayPayload payload;
                TraceRay(
                    accel,
                    0,
                    0xFF,
                    0,
                    1,
                    0,
                    vec3(0.0, 0.0, 0.0),
                    0.0,
                    vec2(0.0, 1.0),
                    100.0,
                    payload
                );
            }
        }
    }
    """
    with pytest.raises(ValueError, match="TraceRay direction.*float3"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(bad_direction_code),
            "ray_generation",
        )

    bad_max_distance_code = """
    shader BadExpandedTraceRayMaxDistance {
        struct RayPayload {
            vec3 color;
        };

        ray_generation {
            void main() {
                RaytracingAccelerationStructure accel;
                uint maxDistance;
                RayPayload payload;
                TraceRay(
                    accel,
                    0,
                    0xFF,
                    0,
                    1,
                    0,
                    vec3(0.0, 0.0, 0.0),
                    0.0,
                    vec3(0.0, 0.0, 1.0),
                    maxDistance,
                    payload
                );
            }
        }
    }
    """
    with pytest.raises(ValueError, match="TraceRay maximum distance.*scalar floating"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(bad_max_distance_code),
            "ray_generation",
        )


def test_directx_ray_tracing_intrinsics_validate_callable_and_hit_arguments():
    valid_callable_and_hit_code = """
    shader ValidCallableAndHitArguments {
        struct CallableData {
            uint value;
        };

        struct HitAttributes {
            vec2 barycentrics;
        };

        ray_generation {
            void main() {
                CallableData data;
                CallShader(0, data);
            }
        }

        ray_intersection {
            void main() {
                HitAttributes attributes;
                ReportHit(1.0, 0, attributes);
            }
        }
    }
    """
    generated = HLSLCodeGen().generate(
        crosstl.translator.parse(valid_callable_and_hit_code)
    )
    assert "CallShader(0, data);" in generated
    assert "ReportHit(1.0, 0, attributes);" in generated

    valid_report_hit_without_attributes_code = """
    shader ValidReportHitWithoutAttributes {
        ray_intersection {
            void main() {
                ReportHit(1.0, 0);
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(valid_report_hit_without_attributes_code),
        "ray_intersection",
    )
    assert "ReportHit(1.0, 0);" in generated

    float_callable_index_code = """
    shader BadCallShaderFloatIndex {
        struct CallableData {
            uint value;
        };

        ray_generation {
            void main() {
                CallableData data;
                CallShader(0.5, data);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="CallShader shader index.*scalar int or uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(float_callable_index_code), "ray_generation"
        )

    scalar_callable_code = """
    shader BadCallableScalarArgument {
        ray_generation {
            void main() {
                uint data;
                CallShader(0, data);
            }
        }
    }
    """
    with pytest.raises(
        ValueError, match="CallShader callable data.*user-defined struct"
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(scalar_callable_code), "ray_generation"
        )

    uint_hit_distance_code = """
    shader BadReportHitUintDistance {
        struct HitAttributes {
            vec2 barycentrics;
        };

        ray_intersection {
            void main() {
                HitAttributes attributes;
                uint hitDistance;
                ReportHit(hitDistance, 0, attributes);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="ReportHit hit distance.*scalar floating"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(uint_hit_distance_code), "ray_intersection"
        )

    float_hit_kind_code = """
    shader BadReportHitFloatKind {
        struct HitAttributes {
            vec2 barycentrics;
        };

        ray_intersection {
            void main() {
                HitAttributes attributes;
                ReportHit(1.0, 0.5, attributes);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="ReportHit hit kind.*scalar int or uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(float_hit_kind_code), "ray_intersection"
        )

    scalar_hit_attribute_code = """
    shader BadReportHitScalarAttribute {
        ray_intersection {
            void main() {
                uint attributes;
                ReportHit(1.0, 0, attributes);
            }
        }
    }
    """
    with pytest.raises(
        ValueError, match="ReportHit hit attribute.*user-defined struct"
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(scalar_hit_attribute_code), "ray_intersection"
        )


def test_directx_mesh_task_stages_emit_numthreads_layouts():
    shader = """
    shader MeshTaskLocalSizes {
        task {
            layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
            void main() { }
        }

        mesh {
            void main() @numthreads(32, 1, 1) @outputtopology(triangle) { }
        }
    }
    """

    generated = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        '[numthreads(8, 4, 1)]\n[shader("amplification")]\nvoid ASMain()' in generated
    )
    assert (
        '[numthreads(32, 1, 1)]\n[outputtopology("triangle")]\n'
        '[shader("mesh")]\nvoid MSMain()'
    ) in generated


def test_directx_mesh_stage_validates_outputtopology_values():
    invalid_code = """
    shader BadMeshTopology {
        mesh {
            void main() @outputtopology(triangle_cw) { }
        }
    }
    """
    with pytest.raises(ValueError, match="mesh stage outputtopology.*triangle_cw"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(invalid_code), "mesh")

    point_code = """
    shader BadPointMeshTopology {
        mesh {
            void main() @outputtopology(point) { }
        }
    }
    """
    with pytest.raises(ValueError, match="mesh stage outputtopology.*point"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(point_code), "mesh")

    valid_code = """
    shader ValidMeshTopology {
        mesh {
            void main() @outputtopology(line) { }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(valid_code), "mesh"
    )
    assert '[outputtopology("line")]' in generated


def test_directx_mesh_task_stage_validates_system_value_parameter_types():
    bad_mesh_group_index_code = """
    shader BadMeshGroupIndex {
        mesh {
            void main(uvec2 groupIndex @ SV_GroupIndex)
                @numthreads(32, 1, 1)
                @outputtopology(triangle) { }
        }
    }
    """
    with pytest.raises(ValueError, match="mesh.*SV_GroupIndex.*scalar uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(bad_mesh_group_index_code), "mesh"
        )

    bad_task_dispatch_id_code = """
    shader BadTaskDispatchId {
        task {
            void main(ivec3 dispatchId @ SV_DispatchThreadID)
                @numthreads(8, 1, 1) {
                DispatchMesh(1, 1, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="task.*SV_DispatchThreadID.*uint3"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(bad_task_dispatch_id_code), "task"
        )

    valid_code = """
    shader ValidMeshTaskSystemValues {
        task {
            void main(
                uvec3 groupId @ SV_GroupID,
                uvec3 groupThreadId @ SV_GroupThreadID,
                uvec3 dispatchId @ SV_DispatchThreadID,
                uint groupIndex @ SV_GroupIndex
            ) @numthreads(8, 1, 1) {
                DispatchMesh(1, 1, 1);
            }
        }

        mesh {
            void main(
                uvec3 groupId @ SV_GroupID,
                uvec3 groupThreadId @ SV_GroupThreadID,
                uvec3 dispatchId @ SV_DispatchThreadID,
                uint groupIndex @ SV_GroupIndex
            ) @numthreads(32, 1, 1) @outputtopology(triangle) { }
        }
    }
    """
    generated = HLSLCodeGen().generate(crosstl.translator.parse(valid_code))

    assert "uint3 groupId : SV_GroupID" in generated
    assert "uint3 groupThreadId : SV_GroupThreadID" in generated
    assert "uint3 dispatchId : SV_DispatchThreadID" in generated
    assert "uint groupIndex : SV_GroupIndex" in generated


def test_directx_mesh_output_signature_roles_emit_validator_compatible_hlsl():
    shader = """
    shader MeshOutputSignature {
        struct MeshVertex {
            vec4 position @ SV_Position;
            vec2 uv @ TEXCOORD0;
        };

        struct MeshPrimitive {
            bool culled @ SV_CullPrimitive;
            uint layer @ SV_RenderTargetArrayIndex;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1],
                @primitives out MeshPrimitive prims[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                verts[1].position = vec4(1.0, 0.0, 0.0, 1.0);
                verts[2].position = vec4(0.0, 1.0, 0.0, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
                prims[0].culled = false;
                prims[0].layer = 0u;
            }
        }
    }
    """

    generated = HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")

    assert "out vertices MeshVertex verts[3]" in generated
    assert "out indices uint3 tris[1]" in generated
    assert "out primitives MeshPrimitive prims[1]" in generated
    assert "bool culled: SV_CullPrimitive;" in generated
    assert "uint layer: SV_RenderTargetArrayIndex;" in generated
    assert ": vertices" not in generated
    assert ": indices" not in generated
    assert ": primitives" not in generated


def test_directx_mesh_output_signature_validates_required_roles_and_counts():
    missing_vertices_code = """
    shader MissingMeshVertices {
        mesh {
            void main(@indices out uvec3 tris[1])
                @numthreads(32, 1, 1)
                @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="out vertices"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(missing_vertices_code), "mesh"
        )

    wrong_index_type_code = """
    shader WrongMeshIndexType {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec2 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="triangle.*uint3"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(wrong_index_type_code), "mesh"
        )

    mismatched_primitives_code = """
    shader MismatchedMeshPrimitiveCount {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        struct MeshPrimitive {
            bool culled @ SV_CullPrimitive;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[2],
                @primitives out MeshPrimitive prims[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="primitives.*match.*indices"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(mismatched_primitives_code), "mesh"
        )


def test_directx_mesh_set_output_counts_validates_arguments_and_bounds():
    missing_argument_code = """
    shader MeshOutputCountMissingArgument {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SetMeshOutputCounts requires exactly"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(missing_argument_code), "mesh"
        )

    float_count_code = """
    shader MeshOutputCountFloatArgument {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                float vertexCount @ TEXCOORD0,
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(vertexCount, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="numVertices.*scalar int or uint"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(float_count_code), "mesh")

    vertex_bound_code = """
    shader MeshOutputCountVertexBound {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(4, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="numVertices.*cannot exceed vertices"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(vertex_bound_code), "mesh"
        )

    primitive_bound_code = """
    shader MeshOutputCountPrimitiveBound {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 2);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="numPrimitives.*cannot exceed indices"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(primitive_bound_code), "mesh"
        )


def test_directx_set_mesh_output_counts_rejects_non_mesh_stages():
    shader = """
    shader SetMeshOutputCountsWrongStage {
        compute {
            void main() {
                SetMeshOutputCounts(1, 1);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="compute.*cannot call SetMeshOutputCounts"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_set_mesh_output_counts_requires_mesh_outputs():
    shader = """
    shader SetMeshOutputCountsWithoutOutputs {
        mesh {
            void main() @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(1, 1);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="requires mesh output"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "mesh")


def test_directx_set_mesh_output_counts_validates_control_flow_placement():
    loop_code = """
    shader MeshOutputCountInLoop {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                for (int i = 0; i < 1; i++) {
                    SetMeshOutputCounts(3, 1);
                }
            }
        }
    }
    """
    with pytest.raises(ValueError, match="loop control flow"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(loop_code), "mesh")

    thread_varying_branch_code = """
    shader MeshOutputCountInThreadVaryingBranch {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                uint groupIndex @ SV_GroupIndex,
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                if (groupIndex == 0u) {
                    SetMeshOutputCounts(3, 1);
                }
            }
        }
    }
    """
    with pytest.raises(ValueError, match="thread-varying control flow"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(thread_varying_branch_code), "mesh"
        )

    uniform_branch_code = """
    shader MeshOutputCountInUniformBranch {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                if (true) {
                    SetMeshOutputCounts(3, 1);
                    verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                    tris[0] = uvec3(0u, 1u, 2u);
                }
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(uniform_branch_code), "mesh"
    )
    assert "SetMeshOutputCounts(3, 1);" in generated


def test_directx_mesh_output_writes_validate_order_and_indices():
    write_before_count_code = """
    shader MeshOutputWriteBeforeCount {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                SetMeshOutputCounts(3, 1);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="verts.*after SetMeshOutputCounts"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(write_before_count_code), "mesh"
        )

    component_index_write_code = """
    shader MeshOutputPartialIndexWrite {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
                tris[0].x = 0u;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="indices output array.*whole"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(component_index_write_code), "mesh"
        )


def test_directx_mesh_output_writes_track_literal_branch_dominance():
    true_branch_code = """
    shader MeshOutputCountDominatesAfterTrueBranch {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                if (true) {
                    SetMeshOutputCounts(3, 1);
                }
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(true_branch_code), "mesh"
    )
    assert "SetMeshOutputCounts(3, 1);" in generated

    false_branch_code = """
    shader MeshOutputCountSkippedByFalseBranch {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                if (false) {
                    SetMeshOutputCounts(3, 1);
                }
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="verts.*after SetMeshOutputCounts"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(false_branch_code), "mesh"
        )

    else_branch_code = """
    shader MeshOutputCountDominatesThroughElseBranch {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                if (false) {
                    verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                } else {
                    SetMeshOutputCounts(3, 1);
                }
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(else_branch_code), "mesh"
    )
    assert "SetMeshOutputCounts(3, 1);" in generated


def test_directx_mesh_output_writes_validate_literal_bounds():
    declared_bound_code = """
    shader MeshOutputWriteDeclaredBound {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
                verts[3].position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="verts.*declared array size"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(declared_bound_code), "mesh"
        )

    set_count_bound_code = """
    shader MeshOutputWriteSetCountBound {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(2, 1);
                verts[2].position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="verts.*numVertices"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(set_count_bound_code), "mesh"
        )


def test_directx_mesh_output_signature_validates_struct_semantics():
    missing_position_code = """
    shader MeshMissingPosition {
        struct MeshVertex {
            vec2 uv @ TEXCOORD0;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_Position"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(missing_position_code), "mesh"
        )

    missing_semantic_code = """
    shader MeshMissingSemantic {
        struct MeshVertex {
            vec4 position @ SV_Position;
            vec2 uv;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="member 'uv'.*semantic"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(missing_semantic_code), "mesh"
        )

    primitive_only_vertex_semantic_code = """
    shader MeshPrimitiveOnlyVertexSemantic {
        struct MeshVertex {
            vec4 position @ SV_Position;
            bool culled @ SV_CullPrimitive;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="per-primitive semantic"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(primitive_only_vertex_semantic_code), "mesh"
        )


def test_directx_amplification_mesh_payload_signature_emits_and_validates():
    shader = """
    shader MeshPayloadPipeline {
        struct MeshPayload {
            uint meshlet;
        };

        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        groupshared MeshPayload payload;

        task {
            void main() @numthreads(1, 1, 1) {
                payload.meshlet = 7u;
                DispatchMesh(1, 1, 1, payload);
            }
        }

        mesh {
            void main(
                @mesh_payload in MeshPayload payload,
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
                verts[0].position = vec4(float(payload.meshlet), 0.0, 0.0, 1.0);
                verts[1].position = vec4(1.0, 0.0, 0.0, 1.0);
                verts[2].position = vec4(0.0, 1.0, 0.0, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """

    generated = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "groupshared MeshPayload payload;" in generated
    assert generated.index("groupshared MeshPayload payload;") < generated.index(
        "void ASMain()"
    )
    as_body = generated[
        generated.index("void ASMain()") : generated.index("[numthreads(32, 1, 1)]")
    ]
    assert "groupshared MeshPayload payload;" not in as_body
    assert "DispatchMesh(1, 1, 1, payload);" in generated
    assert "in payload MeshPayload payload" in generated
    assert ": mesh_payload" not in generated


def test_directx_amplification_mesh_payload_type_mismatch_is_rejected():
    shader = """
    shader MeshPayloadMismatch {
        struct TaskPayload {
            uint meshlet;
        };

        struct MeshPayload {
            uint meshlet;
        };

        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        groupshared TaskPayload payload;

        task {
            void main() @numthreads(1, 1, 1) {
                DispatchMesh(1, 1, 1, payload);
            }
        }

        mesh {
            void main(
                @mesh_payload in MeshPayload payload,
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="DispatchMesh payload type.*MeshPayload"):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_mesh_payload_requires_amplification_dispatch_payload():
    shader = """
    shader MeshPayloadMissingDispatchArgument {
        struct MeshPayload {
            uint meshlet;
        };

        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        task {
            void main() @numthreads(1, 1, 1) {
                DispatchMesh(1, 1, 1);
            }
        }

        mesh {
            void main(
                @mesh_payload in MeshPayload payload,
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="DispatchMesh call must pass a payload"):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_mesh_payload_does_not_capture_ray_payload_semantic():
    shader = """
    shader RayPayloadStillSemantic {
        struct RayPayload {
            vec3 color;
        };

        ray_miss {
            void main(RayPayload payload @ payload) {
                payload.color = vec3(1.0, 0.0, 0.0);
            }
        }
    }
    """

    generated = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "void MissMain(inout RayPayload payload)" in generated
    assert "in payload RayPayload" not in generated
    assert ": payload" not in generated


def test_directx_dispatch_mesh_validates_argument_count_and_group_count_types():
    too_many_args_code = """
    shader DispatchMeshTooManyArgs {
        task {
            void main() @numthreads(1, 1, 1) {
                DispatchMesh(1, 1, 1, 1, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="DispatchMesh requires exactly"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(too_many_args_code), "task"
        )

    float_count_code = """
    shader DispatchMeshFloatCount {
        task {
            void main() @numthreads(1, 1, 1) {
                DispatchMesh(1.5, 1, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="ThreadGroupCountX.*scalar int or uint"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(float_count_code), "task")

    vector_count_code = """
    shader DispatchMeshVectorCount {
        task {
            void main(uvec3 count @ SV_GroupID) @numthreads(1, 1, 1) {
                DispatchMesh(count, 1, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="ThreadGroupCountX.*uint3"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(vector_count_code), "task"
        )

    negative_count_code = """
    shader DispatchMeshNegativeCount {
        task {
            void main() @numthreads(1, 1, 1) {
                DispatchMesh(-1, 1, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="ThreadGroupCountX.*non-negative"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(negative_count_code), "task"
        )

    too_large_count_code = """
    shader DispatchMeshTooLargeCount {
        task {
            void main() @numthreads(1, 1, 1) {
                DispatchMesh(65536, 1, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="ThreadGroupCountX.*less than 65536"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(too_large_count_code), "task"
        )

    product_too_large_code = """
    shader DispatchMeshProductTooLarge {
        task {
            void main() @numthreads(1, 1, 1) {
                DispatchMesh(2048, 2048, 2);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="thread group count product"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(product_too_large_code), "task"
        )


def test_directx_dispatch_mesh_rejects_multiple_calls_per_amplification_stage():
    shader = """
    shader DispatchMeshMultipleCalls {
        task {
            void main() @numthreads(1, 1, 1) {
                DispatchMesh(1, 1, 1);
                DispatchMesh(1, 1, 1);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="DispatchMesh at most once"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "task")


def test_directx_dispatch_mesh_rejects_non_uniform_control_flow():
    loop_code = """
    shader DispatchMeshInLoop {
        task {
            void main() @numthreads(1, 1, 1) {
                for (int i = 0; i < 1; i++) {
                    DispatchMesh(1, 1, 1);
                }
            }
        }
    }
    """
    with pytest.raises(ValueError, match="DispatchMesh.*loop control flow"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(loop_code), "task")

    thread_varying_branch_code = """
    shader DispatchMeshInThreadVaryingBranch {
        task {
            void main(uint groupIndex @ SV_GroupIndex) @numthreads(1, 1, 1) {
                if (groupIndex == 0u) {
                    DispatchMesh(1, 1, 1);
                }
            }
        }
    }
    """
    with pytest.raises(ValueError, match="DispatchMesh.*thread-varying control flow"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(thread_varying_branch_code), "task"
        )


def test_directx_dispatch_mesh_payload_must_be_groupshared():
    shader = """
    shader DispatchMeshLocalPayload {
        struct MeshPayload {
            uint meshlet;
        };

        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        task {
            void main() @numthreads(1, 1, 1) {
                MeshPayload payload;
                payload.meshlet = 7u;
                DispatchMesh(1, 1, 1, payload);
            }
        }

        mesh {
            void main(
                @mesh_payload in MeshPayload payload,
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="payload argument.*groupshared"):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_groupshared_variables_must_be_global_scope():
    shader = """
    shader LocalGroupShared {
        task {
            void main() @numthreads(1, 1, 1) {
                groupshared uint scratch;
                scratch = 1u;
                DispatchMesh(1, 1, 1);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="groupshared variables.*global scope"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "task")


def test_directx_dispatch_mesh_rejects_calls_outside_amplification_stages():
    shader = """
    shader DispatchMeshWrongStage {
        struct MeshPayload {
            uint meshlet;
        };

        compute {
            void main() {
                MeshPayload payload;
                DispatchMesh(1, 1, 1, payload);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="compute.*cannot call DispatchMesh"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_advanced_stage_entries_use_stage_specific_names():
    code = """
    shader advanced {
        geometry {
            void main() @maxvertexcount(3) { }
        }

        tessellation_control {
            void main()
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) { }
        }

        tessellation_evaluation {
            void main() @domain(tri) { }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert '[shader("geometry")]' in generated
    assert "void GSMain()" in generated
    assert '[shader("hull")]' in generated
    assert "void HSMain()" in generated
    assert '[shader("domain")]' in generated
    assert "void DSMain()" in generated
    assert "void main()" not in generated


def test_directx_advanced_stage_entry_name_avoids_local_helper_collision():
    code = """
    shader geometry_collision {
        geometry {
            void GSMain() { }

            void main() @maxvertexcount(3) {
                GSMain();
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(crosstl.translator.parse(code), "geometry")

    assert "void GSMain() {" in generated
    assert '[shader("geometry")]' in generated
    assert "void GSMain_2()" in generated
    assert "GSMain();" in generated


def test_directx_geometry_stage_lowers_hlsl_stage_attributes():
    code = """
    shader geometry_attributes {
        geometry {
            void main() @maxvertexcount(3) { }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(crosstl.translator.parse(code), "geometry")

    assert "[maxvertexcount(3)]" in generated
    assert generated.index("[maxvertexcount(3)]") < generated.index(
        '[shader("geometry")]'
    )
    assert "void GSMain()" in generated


def test_directx_geometry_stage_validates_positive_maxvertexcount():
    code = """
    shader bad_geometry_maxvertexcount {
        geometry {
            void main() @maxvertexcount(0) { }
        }
    }
    """

    with pytest.raises(ValueError, match="maxvertexcount.*positive"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(code), "geometry")


def test_directx_tessellation_stages_lower_hlsl_stage_attributes():
    code = """
    shader tessellation_attributes {
        tessellation_control {
            void main()
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) { }
        }

        tessellation_evaluation {
            void main() @domain(tri) { }
        }
    }
    """
    generated = HLSLCodeGen().generate(crosstl.translator.parse(code))

    assert '[domain("tri")]' in generated
    assert '[partitioning("fractional_odd")]' in generated
    assert '[outputtopology("triangle_cw")]' in generated
    assert "[outputcontrolpoints(3)]" in generated
    assert '[patchconstantfunc("HSConst")]' in generated
    assert generated.index('[domain("tri")]') < generated.index('[shader("hull")]')
    assert generated.count('[domain("tri")]') == 2
    assert "void HSMain()" in generated
    assert "void DSMain()" in generated


def test_directx_advanced_stage_signatures_accept_hlsl_patch_and_stream_types():
    code = """
    shader advanced_signatures {
        struct GSOutput {
            vec4 position @ SV_Position;
        };

        struct GSInput {
            vec3 position @ POSITION;
        };

        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec3 edges @ SV_TessFactor;
            float inside @ SV_InsideTessFactor;
        };

        geometry {
            void main(triangle GSInput input[3], inout TriangleStream<GSOutput> stream)
                @maxvertexcount(3) { }
        }

        tessellation_control {
            HSConstData HSConst(
                InputPatch<HSInput, 3> patch,
                uint patchId @ SV_PrimitiveID
            ) {
                HSConstData constants;
                return constants;
            }

            void main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) { }
        }

        tessellation_evaluation {
            vec4 main(OutputPatch<HSOutput, 3> patch, vec3 bary @ SV_DomainLocation)
                @domain(tri) {
                return vec4(0.0);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void GSMain(triangle GSInput input[3], inout TriangleStream<GSOutput> stream)"
    ) in generated
    assert (
        "void HSMain(InputPatch<HSInput, 3> patch, uint id : SV_OutputControlPointID)"
    ) in generated
    assert (
        "float4 DSMain(OutputPatch<HSOutput, 3> patch, float3 bary : SV_DomainLocation)"
    ) in generated
    assert (
        "HSConstData HSConst(InputPatch<HSInput, 3> patch, "
        "uint patchId : SV_PrimitiveID)"
    ) in generated
    assert "float edges[3] : SV_TessFactor;" in generated
    assert "float3 edges : SV_TessFactor;" not in generated


def test_directx_tessellation_accepts_maximum_control_point_count():
    code = """
    shader maximum_tessellation_control_points {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec3 edges @ SV_TessFactor;
            float inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(InputPatch<HSInput, 32> patch) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 32> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(integer)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(32)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }

        tessellation_evaluation {
            vec4 main(OutputPatch<HSOutput, 32> patch, vec3 bary @ SV_DomainLocation)
                @domain(tri) {
                return vec4(0.0);
            }
        }
    }
    """
    generated = HLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "[outputcontrolpoints(32)]" in generated
    assert "HSMain(InputPatch<HSInput, 32> patch" in generated
    assert "DSMain(OutputPatch<HSOutput, 32> patch" in generated
    assert "HSConst(InputPatch<HSInput, 32> patch)" in generated


def test_directx_tessellation_signatures_preserve_extra_hlsl_parameters():
    code = """
    shader tessellation_signature_parameters {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec3 edges @ SV_TessFactor;
            float inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(InputPatch<HSInput, 3> patch) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(
                InputPatch<HSInput, 3> patch,
                uint id @ SV_OutputControlPointID,
                uint patchId @ SV_PrimitiveID
            )
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }

        tessellation_evaluation {
            vec4 main(
                HSConstData constants,
                vec3 bary @ SV_DomainLocation,
                const OutputPatch<HSOutput, 3> patch
            ) @domain(tri) {
                return vec4(0.0);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    assert (
        "HSOutput HSMain(InputPatch<HSInput, 3> patch, "
        "uint id : SV_OutputControlPointID, uint patchId : SV_PrimitiveID)"
    ) in generated
    assert (
        "float4 DSMain(HSConstData constants, float3 bary : SV_DomainLocation, "
        "const OutputPatch<HSOutput, 3> patch)"
    ) in generated
    assert "HSConstData HSConst(InputPatch<HSInput, 3> patch)" in generated
    assert "float edges[3] : SV_TessFactor;" in generated
    assert "float3 edges : SV_TessFactor;" not in generated


def test_directx_tessellation_factor_vectors_emit_hlsl_arrays():
    code = """
    shader tessellation_factor_layout {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec4 edges @ SV_TessFactor;
            vec2 inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(InputPatch<HSInput, 4> patch) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 4> patch, uint id @ SV_OutputControlPointID)
                @domain(quad)
                @partitioning(fractional_even)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(4)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "tessellation_control"
    )

    assert "float edges[4] : SV_TessFactor;" in generated
    assert "float inside[2] : SV_InsideTessFactor;" in generated
    assert "float4 edges : SV_TessFactor;" not in generated
    assert "float2 inside : SV_InsideTessFactor;" not in generated


@pytest.mark.parametrize(
    ("patch_constant_members", "message"),
    [
        (
            "ivec3 edges @ SV_TessFactor;\n"
            "            float inside @ SV_InsideTessFactor;",
            "SV_TessFactor.*floating-point",
        ),
        (
            "uvec3 edges @ SV_TessFactor;\n"
            "            float inside @ SV_InsideTessFactor;",
            "SV_TessFactor.*floating-point",
        ),
        (
            "vec3 edges @ SV_TessFactor;\n"
            "            bool inside @ SV_InsideTessFactor;",
            "SV_InsideTessFactor.*floating-point",
        ),
    ],
)
def test_directx_tessellation_factor_semantics_require_floating_members(
    patch_constant_members, message
):
    code = f"""
    shader invalid_tess_factor_member_type {{
        struct HSInput {{
            vec3 position @ POSITION;
        }};

        struct HSOutput {{
            vec3 position @ POSITION;
        }};

        struct HSConstData {{
            {patch_constant_members}
        }};

        tessellation_control {{
            HSConstData HSConst(InputPatch<HSInput, 3> patch) {{
                HSConstData constants;
                return constants;
            }}

            HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {{
                HSOutput output;
                return output;
            }}
        }}
    }}
    """
    with pytest.raises(ValueError, match=message):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


def test_directx_advanced_stage_signature_validation_rejects_partial_hlsl_shapes():
    geometry_code = """
    shader bad_geometry_signature {
        geometry {
            void main(vec3 position @ POSITION) @maxvertexcount(3) { }
        }
    }
    """
    with pytest.raises(ValueError, match="TriangleStream"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(geometry_code), "geometry"
        )

    stream_only_geometry_code = """
    shader bad_geometry_stream_only_signature {
        geometry {
            void main(inout TriangleStream<GSOutput> stream) @maxvertexcount(3) { }
        }
    }
    """
    with pytest.raises(ValueError, match="input primitive"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(stream_only_geometry_code), "geometry"
        )

    hull_code = """
    shader bad_hull_signature {
        tessellation_control {
            void main(uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) { }
        }
    }
    """
    with pytest.raises(ValueError, match="InputPatch"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(hull_code), "tessellation_control"
        )

    domain_code = """
    shader bad_domain_signature {
        tessellation_evaluation {
            vec4 main(vec3 bary @ SV_DomainLocation) @domain(tri) {
                return vec4(0.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="OutputPatch"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(domain_code), "tessellation_evaluation"
        )


@pytest.mark.parametrize(
    ("patch_parameter", "message"),
    [
        (
            "InputPatch<HSInput> patch",
            r"tessellation_control.*InputPatch.*InputPatch<T, N>",
        ),
        (
            "InputPatch<HSInput, PATCH_SIZE> patch",
            r"tessellation_control.*InputPatch.*integer literal",
        ),
        (
            "InputPatch<HSInput, 0> patch",
            r"tessellation_control.*InputPatch.*positive",
        ),
        (
            "InputPatch<HSInput, 33> patch",
            r"tessellation_control.*InputPatch.*at most 32",
        ),
    ],
)
def test_directx_tessellation_control_validates_inputpatch_generic_signature(
    patch_parameter, message
):
    code = f"""
    shader bad_hull_patch_signature {{
        struct HSInput {{
            vec3 position @ POSITION;
        }};

        struct HSOutput {{
            vec3 position @ POSITION;
        }};

        tessellation_control {{
            HSOutput main({patch_parameter}, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {{
                HSOutput output;
                return output;
            }}
        }}
    }}
    """
    with pytest.raises(ValueError, match=message):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


@pytest.mark.parametrize(
    ("patch_parameter", "message"),
    [
        (
            "OutputPatch<HSOutput> patch",
            r"tessellation_evaluation.*OutputPatch.*OutputPatch<T, N>",
        ),
        (
            "OutputPatch<HSOutput, PATCH_SIZE> patch",
            r"tessellation_evaluation.*OutputPatch.*integer literal",
        ),
        (
            "OutputPatch<HSOutput, 0> patch",
            r"tessellation_evaluation.*OutputPatch.*positive",
        ),
        (
            "OutputPatch<HSOutput, 33> patch",
            r"tessellation_evaluation.*OutputPatch.*at most 32",
        ),
    ],
)
def test_directx_tessellation_evaluation_validates_outputpatch_generic_signature(
    patch_parameter, message
):
    code = f"""
    shader bad_domain_patch_signature {{
        struct HSOutput {{
            vec3 position @ POSITION;
        }};

        tessellation_evaluation {{
            vec4 main({patch_parameter}, vec3 bary @ SV_DomainLocation)
                @domain(tri) {{
                return vec4(0.0);
            }}
        }}
    }}
    """
    with pytest.raises(ValueError, match=message):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_evaluation"
        )


def test_directx_geometry_stream_output_builtin_stages_are_validated():
    code = """
    shader bad_geometry_stream_output_semantic {
        struct GSInput {
            vec3 position @ POSITION;
        };

        struct GSOutput {
            vec4 color @ gl_FragColor;
        };

        geometry {
            void main(triangle GSInput input[3], inout TriangleStream<GSOutput> stream)
                @maxvertexcount(3) { }
        }
    }
    """

    with pytest.raises(ValueError, match="geometry.*SV_TARGET"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(code), "geometry")


@pytest.mark.parametrize(
    ("stage", "return_decl", "stage_body", "hlsl_semantic"),
    [
        (
            "tessellation_evaluation",
            "HSOutput main(OutputPatch<HSOutput, 3> patch, vec3 bary @ SV_DomainLocation) "
            "@domain(tri)",
            "HSOutput output; return output;",
            "SV_TARGET",
        ),
        (
            "tessellation_control",
            "HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID) "
            "@domain(tri) @partitioning(fractional_odd) @outputtopology(triangle_cw) "
            "@outputcontrolpoints(3) @patchconstantfunc(HSConst)",
            "HSOutput output; return output;",
            "SV_TARGET",
        ),
    ],
)
def test_directx_tessellation_output_builtin_stages_are_validated(
    stage, return_decl, stage_body, hlsl_semantic
):
    code = f"""
    shader bad_tessellation_output_semantic {{
        struct HSInput {{
            vec3 position @ POSITION;
        }};

        struct HSOutput {{
            vec4 color @ gl_FragColor;
        }};

        struct HSConstData {{
            vec3 edges @ SV_TessFactor;
            float inside @ SV_InsideTessFactor;
        }};

        HSConstData HSConst(InputPatch<HSInput, 3> patch) {{
            HSConstData constants;
            return constants;
        }}

        {stage} {{
            {return_decl} {{
                {stage_body}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=f"{stage}.*{hlsl_semantic}"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(code), stage)


def test_directx_tessellation_direct_return_semantic_ignores_stage_attributes():
    code = """
    shader domain_direct_return_semantic {
        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_evaluation {
            vec4 main(
                OutputPatch<HSOutput, 3> patch,
                vec3 bary @ SV_DomainLocation
            ) @domain(tri) @gl_Position {
                return vec4(0.0);
            }
        }
    }
    """

    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "tessellation_evaluation"
    )

    assert '[domain("tri")]' in generated
    assert (
        "float4 DSMain(OutputPatch<HSOutput, 3> patch, "
        "float3 bary : SV_DomainLocation): SV_POSITION"
    ) in generated


def test_directx_tessellation_direct_return_builtin_stage_is_validated():
    code = """
    shader bad_domain_direct_return_semantic {
        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_evaluation {
            vec4 main(
                OutputPatch<HSOutput, 3> patch,
                vec3 bary @ SV_DomainLocation
            ) @domain(tri) @gl_FragColor {
                return vec4(0.0);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="tessellation_evaluation.*SV_TARGET"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_evaluation"
        )


def test_directx_tessellation_custom_direct_return_semantic_emits():
    code = """
    shader domain_custom_direct_return_semantic {
        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_evaluation {
            vec2 main(
                OutputPatch<HSOutput, 3> patch,
                vec3 bary @ SV_DomainLocation
            ) @domain(tri) @TEXCOORD0 {
                return vec2(0.0);
            }
        }
    }
    """

    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "tessellation_evaluation"
    )

    assert (
        "float2 DSMain(OutputPatch<HSOutput, 3> patch, "
        "float3 bary : SV_DomainLocation): TEXCOORD0"
    ) in generated


def test_directx_tessellation_void_direct_return_semantic_is_rejected():
    code = """
    shader bad_domain_void_direct_return_semantic {
        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_evaluation {
            void main(
                OutputPatch<HSOutput, 3> patch,
                vec3 bary @ SV_DomainLocation
            ) @domain(tri) @TEXCOORD0 { }
        }
    }
    """

    with pytest.raises(ValueError, match="tessellation_evaluation.*TEXCOORD0.*void"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_evaluation"
        )


def test_directx_geometry_stage_validates_input_primitive_array_arity():
    wrong_triangle_count_code = """
    shader bad_geometry_triangle_arity {
        struct GSInput {
            vec3 position @ POSITION;
        };

        struct GSOutput {
            vec4 position @ SV_Position;
        };

        geometry {
            void main(triangle GSInput input[2], inout TriangleStream<GSOutput> stream)
                @maxvertexcount(3) { }
        }
    }
    """
    with pytest.raises(ValueError, match="triangle.*3"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(wrong_triangle_count_code), "geometry"
        )

    missing_array_code = """
    shader bad_geometry_missing_primitive_array {
        struct GSInput {
            vec3 position @ POSITION;
        };

        struct GSOutput {
            vec4 position @ SV_Position;
        };

        geometry {
            void main(line GSInput input, inout LineStream<GSOutput> stream)
                @maxvertexcount(2) { }
        }
    }
    """
    with pytest.raises(ValueError, match="line.*array.*2"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(missing_array_code), "geometry"
        )

    lineadj_code = """
    shader geometry_lineadj_arity {
        struct GSInput {
            vec3 position @ POSITION;
        };

        struct GSOutput {
            vec4 position @ SV_Position;
        };

        geometry {
            void main(lineadj GSInput input[4], inout LineStream<GSOutput> stream)
                @maxvertexcount(2) { }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(lineadj_code), "geometry"
    )
    assert "void GSMain(lineadj GSInput input[4]" in generated


def test_directx_geometry_stage_validates_primitive_id_type():
    float_id_code = """
    shader bad_geometry_float_primitive_id {
        struct GSInput {
            vec3 position @ POSITION;
        };

        struct GSOutput {
            vec4 position @ SV_Position;
        };

        geometry {
            void main(
                triangle GSInput input[3],
                float primitiveId @ SV_PrimitiveID,
                inout TriangleStream<GSOutput> stream
            ) @maxvertexcount(3) { }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_PrimitiveID.*int or uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(float_id_code), "geometry"
        )

    vector_id_code = """
    shader bad_geometry_vector_primitive_id {
        struct GSInput {
            vec3 position @ POSITION;
        };

        struct GSOutput {
            vec4 position @ SV_Position;
        };

        geometry {
            void main(
                triangle GSInput input[3],
                ivec2 primitiveId @ SV_PrimitiveID,
                inout TriangleStream<GSOutput> stream
            ) @maxvertexcount(3) { }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_PrimitiveID.*int or uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(vector_id_code), "geometry"
        )

    int_id_code = """
    shader valid_geometry_int_primitive_id {
        struct GSInput {
            vec3 position @ POSITION;
        };

        struct GSOutput {
            vec4 position @ SV_Position;
        };

        geometry {
            void main(
                triangle GSInput input[3],
                int primitiveId @ SV_PrimitiveID,
                inout TriangleStream<GSOutput> stream
            ) @maxvertexcount(3) { }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(int_id_code), "geometry"
    )
    assert "int primitiveId : SV_PrimitiveID" in generated


def test_directx_geometry_stage_validates_gs_instance_id_type():
    float_id_code = """
    shader bad_geometry_float_gs_instance_id {
        struct GSInput {
            vec3 position @ POSITION;
        };

        struct GSOutput {
            vec4 position @ SV_Position;
        };

        geometry {
            void main(
                triangle GSInput input[3],
                float instanceId @ SV_GSInstanceID,
                inout TriangleStream<GSOutput> stream
            ) @maxvertexcount(3) { }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_GSInstanceID.*int or uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(float_id_code), "geometry"
        )

    vector_id_code = """
    shader bad_geometry_vector_gs_instance_id {
        struct GSInput {
            vec3 position @ POSITION;
        };

        struct GSOutput {
            vec4 position @ SV_Position;
        };

        geometry {
            void main(
                triangle GSInput input[3],
                uvec2 instanceId @ SV_GSInstanceID,
                inout TriangleStream<GSOutput> stream
            ) @maxvertexcount(3) { }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_GSInstanceID.*int or uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(vector_id_code), "geometry"
        )

    uint_id_code = """
    shader valid_geometry_uint_gs_instance_id {
        struct GSInput {
            vec3 position @ POSITION;
        };

        struct GSOutput {
            vec4 position @ SV_Position;
        };

        geometry {
            void main(
                triangle GSInput input[3],
                uint instanceId @ SV_GSInstanceID,
                inout TriangleStream<GSOutput> stream
            ) @maxvertexcount(3) { }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(uint_id_code), "geometry"
    )
    assert "uint instanceId : SV_GSInstanceID" in generated


def test_directx_tessellation_control_validates_patch_constant_helper():
    missing_helper_code = """
    shader missing_patch_constant_helper {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_control {
            HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="patchconstantfunc 'HSConst'"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(missing_helper_code), "tessellation_control"
        )

    void_helper_code = """
    shader void_patch_constant_helper {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_control {
            void HSConst(InputPatch<HSInput, 3> patch) { }

            HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="return patch-constant data"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(void_helper_code), "tessellation_control"
        )

    parameter_shape_code = """
    shader bad_patch_constant_helper_parameters {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec3 edges @ SV_TessFactor;
            float inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(uint patchId @ SV_PrimitiveID) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="InputPatch"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(parameter_shape_code), "tessellation_control"
        )


@pytest.mark.parametrize(
    ("patch_parameter", "message"),
    [
        (
            "InputPatch<OtherInput, 3> patch",
            "patchconstantfunc 'HSConst'.*InputPatch element type.*HSInput",
        ),
        (
            "InputPatch<HSInput, 4> patch",
            "patchconstantfunc 'HSConst'.*InputPatch control point count.*3",
        ),
        (
            "InputPatch<HSInput> patch",
            r"patchconstantfunc 'HSConst'.*InputPatch.*InputPatch<T, N>",
        ),
        (
            "InputPatch<HSInput, PATCH_SIZE> patch",
            r"patchconstantfunc 'HSConst'.*InputPatch.*integer literal",
        ),
        (
            "InputPatch<HSInput, 0> patch",
            r"patchconstantfunc 'HSConst'.*InputPatch.*positive",
        ),
        (
            "InputPatch<HSInput, 33> patch",
            r"patchconstantfunc 'HSConst'.*InputPatch.*at most 32",
        ),
    ],
)
def test_directx_tessellation_control_validates_patch_constant_inputpatch_signature(
    patch_parameter, message
):
    code = f"""
    shader bad_patch_constant_inputpatch_signature {{
        struct HSInput {{
            vec3 position @ POSITION;
        }};

        struct OtherInput {{
            vec3 position @ POSITION;
        }};

        struct HSOutput {{
            vec3 position @ POSITION;
        }};

        struct HSConstData {{
            vec3 edges @ SV_TessFactor;
            float inside @ SV_InsideTessFactor;
        }};

        tessellation_control {{
            HSConstData HSConst({patch_parameter}) {{
                HSConstData constants;
                return constants;
            }}

            HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {{
                HSOutput output;
                return output;
            }}
        }}
    }}
    """
    with pytest.raises(ValueError, match=message):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


def test_directx_tessellation_control_validates_patch_constant_primitive_id_type():
    float_id_code = """
    shader bad_patch_constant_float_primitive_id {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec3 edges @ SV_TessFactor;
            float inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(
                InputPatch<HSInput, 3> patch,
                float patchId @ SV_PrimitiveID
            ) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_PrimitiveID.*int or uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(float_id_code), "tessellation_control"
        )

    vector_id_code = """
    shader bad_patch_constant_vector_primitive_id {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec3 edges @ SV_TessFactor;
            float inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(
                InputPatch<HSInput, 3> patch,
                ivec2 patchId @ SV_PrimitiveID
            ) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_PrimitiveID.*int or uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(vector_id_code), "tessellation_control"
        )

    int_id_code = """
    shader valid_patch_constant_int_primitive_id {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec3 edges @ SV_TessFactor;
            float inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(
                InputPatch<HSInput, 3> patch,
                int patchId @ SV_PrimitiveID
            ) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(int_id_code), "tessellation_control"
    )
    assert "int patchId : SV_PrimitiveID" in generated


def test_directx_tessellation_control_validates_parameterless_patch_constant_helper():
    void_helper_code = """
    shader parameterless_void_patch_constant_helper {
        tessellation_control {
            void HSConst() { }

            void main()
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) { }
        }
    }
    """
    with pytest.raises(ValueError, match="return patch-constant data"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(void_helper_code), "tessellation_control"
        )

    missing_semantics_code = """
    shader parameterless_bad_patch_constant_semantics {
        struct HSConstData {
            float inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst() {
                HSConstData constants;
                return constants;
            }

            void main()
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) { }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_TessFactor"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(missing_semantics_code),
            "tessellation_control",
        )


def test_directx_tessellation_control_rejects_direct_patch_constant_factor_return():
    code = """
    shader direct_patch_constant_tess_factor_return {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_control {
            vec3 HSConst(InputPatch<HSInput, 3> patch) @ SV_TessFactor {
                return vec3(1.0);
            }

            HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(integer)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="patchconstantfunc.*return a struct"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


def test_directx_tessellation_control_validates_patch_constant_semantics():
    missing_outer_code = """
    shader missing_tess_factor_semantic {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            float inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(InputPatch<HSInput, 3> patch) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_TessFactor"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(missing_outer_code), "tessellation_control"
        )

    missing_inside_code = """
    shader missing_inside_tess_factor_semantic {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec3 edges @ SV_TessFactor;
        };

        tessellation_control {
            HSConstData HSConst(InputPatch<HSInput, 3> patch) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_InsideTessFactor"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(missing_inside_code), "tessellation_control"
        )

    isoline_code = """
    shader isoline_patch_constant_semantics {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec2 edges @ SV_TessFactor;
        };

        tessellation_control {
            HSConstData HSConst(InputPatch<HSInput, 2> patch) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 2> patch, uint id @ SV_OutputControlPointID)
                @domain(isoline)
                @partitioning(fractional_even)
                @outputtopology(line)
                @outputcontrolpoints(2)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(isoline_code), "tessellation_control"
    )
    assert '[domain("isoline")]' in generated
    assert "HSConstData HSConst(InputPatch<HSInput, 2> patch)" in generated
    assert "float edges[2] : SV_TessFactor;" in generated


def test_directx_tessellation_control_validates_patch_constant_factor_counts():
    wrong_tri_outer_code = """
    shader wrong_tri_outer_tess_factor_count {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec4 edges @ SV_TessFactor;
            float inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(InputPatch<HSInput, 3> patch) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="3 SV_TessFactor"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(wrong_tri_outer_code), "tessellation_control"
        )

    wrong_quad_inner_code = """
    shader wrong_quad_inner_tess_factor_count {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec4 edges @ SV_TessFactor;
            float inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(InputPatch<HSInput, 4> patch) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 4> patch, uint id @ SV_OutputControlPointID)
                @domain(quad)
                @partitioning(fractional_even)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(4)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="2 SV_InsideTessFactor"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(wrong_quad_inner_code), "tessellation_control"
        )

    wrong_isoline_inside_code = """
    shader wrong_isoline_inside_tess_factor {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec2 edges @ SV_TessFactor;
            float inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(InputPatch<HSInput, 2> patch) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 2> patch, uint id @ SV_OutputControlPointID)
                @domain(isoline)
                @partitioning(fractional_even)
                @outputtopology(line)
                @outputcontrolpoints(2)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="must not return SV_InsideTessFactor"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(wrong_isoline_inside_code),
            "tessellation_control",
        )


def test_directx_tessellation_control_validates_domain_outputtopology_pairs():
    tri_line_code = """
    shader bad_tri_line_topology {
        tessellation_control {
            void main()
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(line)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) { }
        }
    }
    """
    with pytest.raises(ValueError, match="triangle_cw or triangle_ccw"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(tri_line_code), "tessellation_control"
        )

    isoline_triangle_code = """
    shader bad_isoline_triangle_topology {
        tessellation_control {
            void main()
                @domain(isoline)
                @partitioning(fractional_even)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(2)
                @patchconstantfunc(HSConst) { }
        }
    }
    """
    with pytest.raises(ValueError, match="requires outputtopology line"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(isoline_triangle_code), "tessellation_control"
        )

    quad_triangle_code = """
    shader valid_quad_triangle_topology {
        tessellation_control {
            void main()
                @domain(quad)
                @partitioning(fractional_even)
                @outputtopology(triangle_ccw)
                @outputcontrolpoints(4)
                @patchconstantfunc(HSConst) { }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(quad_triangle_code), "tessellation_control"
    )
    assert '[domain("quad")]' in generated
    assert '[outputtopology("triangle_ccw")]' in generated


def test_directx_tessellation_validates_domain_values():
    invalid_hull_code = """
    shader bad_hull_domain {
        tessellation_control {
            void main()
                @domain(trianglee)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) { }
        }
    }
    """
    with pytest.raises(ValueError, match="domain.*trianglee"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(invalid_hull_code), "tessellation_control"
        )

    invalid_domain_code = """
    shader bad_domain_domain {
        tessellation_evaluation {
            void main() @domain(patch) { }
        }
    }
    """
    with pytest.raises(ValueError, match="domain.*patch"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(invalid_domain_code), "tessellation_evaluation"
        )

    triangle_alias_code = """
    shader triangle_alias_domain {
        tessellation_control {
            void main()
                @domain(triangle)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) { }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(triangle_alias_code), "tessellation_control"
    )
    assert '[domain("tri")]' in generated
    assert '[domain("triangle")]' not in generated


def test_directx_tessellation_control_validates_outputtopology_values():
    invalid_code = """
    shader bad_outputtopology {
        tessellation_control {
            void main()
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangles_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) { }
        }
    }
    """
    with pytest.raises(ValueError, match="outputtopology.*triangles_cw"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(invalid_code), "tessellation_control"
        )

    point_code = """
    shader valid_point_outputtopology {
        tessellation_control {
            void main()
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(point)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) { }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(point_code), "tessellation_control"
    )
    assert '[outputtopology("point")]' in generated


def test_directx_tessellation_control_validates_partitioning_values():
    invalid_code = """
    shader bad_partitioning {
        tessellation_control {
            void main()
                @domain(tri)
                @partitioning(fractional)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) { }
        }
    }
    """
    with pytest.raises(ValueError, match="partitioning.*fractional"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(invalid_code), "tessellation_control"
        )

    for partitioning in ("integer", "pow2"):
        valid_code = f"""
        shader valid_{partitioning}_partitioning {{
            tessellation_control {{
                void main()
                    @domain(tri)
                    @partitioning({partitioning})
                    @outputtopology(triangle_cw)
                    @outputcontrolpoints(3)
                    @patchconstantfunc(HSConst) {{ }}
            }}
        }}
        """
        generated = HLSLCodeGen().generate_stage(
            crosstl.translator.parse(valid_code), "tessellation_control"
        )
        assert f'[partitioning("{partitioning}")]' in generated


def test_directx_tessellation_control_validates_outputcontrolpoints_positive():
    code = """
    shader bad_output_control_point_count {
        tessellation_control {
            void main()
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(0)
                @patchconstantfunc(HSConst) { }
        }
    }
    """
    with pytest.raises(ValueError, match="outputcontrolpoints.*positive"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


def test_directx_tessellation_control_validates_outputcontrolpoints_upper_bound():
    code = """
    shader bad_output_control_point_limit {
        tessellation_control {
            void main()
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(33)
                @patchconstantfunc(HSConst) { }
        }
    }
    """
    with pytest.raises(ValueError, match="outputcontrolpoints.*at most 32"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


def test_directx_tessellation_control_validates_outputcontrolpoints_inputpatch_count():
    mismatched_code = """
    shader mismatched_hull_control_point_count {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_control {
            HSOutput main(InputPatch<HSInput, 3> patch, uint id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(4)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="outputcontrolpoints.*InputPatch"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(mismatched_code), "tessellation_control"
        )


@pytest.mark.parametrize(
    ("maxtessfactor", "message"),
    [
        ("0.0", r"maxtessfactor.*range 1.0..64.0"),
        ("65.0", r"maxtessfactor.*range 1.0..64.0"),
        ("MAX_TESS", r"maxtessfactor.*numeric literal"),
    ],
)
def test_directx_tessellation_control_validates_maxtessfactor_value(
    maxtessfactor, message
):
    code = f"""
    shader bad_max_tess_factor {{
        tessellation_control {{
            void main()
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @maxtessfactor({maxtessfactor})
                @patchconstantfunc(HSConst) {{ }}
        }}
    }}
    """
    with pytest.raises(ValueError, match=message):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


def test_directx_tessellation_control_accepts_maximum_maxtessfactor_value():
    valid_code = """
    shader valid_max_tess_factor {
        tessellation_control {
            void main()
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @maxtessfactor(64.0)
                @patchconstantfunc(HSConst) { }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(valid_code), "tessellation_control"
    )
    assert "[maxtessfactor(64.0)]" in generated


def test_directx_tessellation_rejects_maxtessfactor_outside_hull_stage():
    code = """
    shader bad_domain_max_tess_factor {
        tessellation_evaluation {
            void main(
                OutputPatch<float4, 3> patch,
                vec3 bary @ SV_DomainLocation
            ) @domain(tri) @maxtessfactor(8.0) { }
        }
    }
    """
    with pytest.raises(ValueError, match="maxtessfactor.*tessellation_control"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_evaluation"
        )


def test_directx_tessellation_control_validates_output_control_point_id_type():
    float_id_code = """
    shader bad_output_control_point_float_id {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_control {
            HSOutput main(InputPatch<HSInput, 3> patch, float id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_OutputControlPointID.*int or uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(float_id_code), "tessellation_control"
        )

    vector_id_code = """
    shader bad_output_control_point_vector_id {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_control {
            HSOutput main(InputPatch<HSInput, 3> patch, uvec2 id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_OutputControlPointID.*int or uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(vector_id_code), "tessellation_control"
        )

    int_id_code = """
    shader valid_output_control_point_int_id {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec3 edges @ SV_TessFactor;
            float inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(InputPatch<HSInput, 3> patch) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 3> patch, int id @ SV_OutputControlPointID)
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(int_id_code), "tessellation_control"
    )
    assert "int id : SV_OutputControlPointID" in generated


def test_directx_tessellation_control_validates_primitive_id_type():
    float_id_code = """
    shader bad_hull_float_primitive_id {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_control {
            HSOutput main(
                InputPatch<HSInput, 3> patch,
                uint id @ SV_OutputControlPointID,
                float patchId @ SV_PrimitiveID
            )
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_PrimitiveID.*int or uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(float_id_code), "tessellation_control"
        )

    vector_id_code = """
    shader bad_hull_vector_primitive_id {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_control {
            HSOutput main(
                InputPatch<HSInput, 3> patch,
                uint id @ SV_OutputControlPointID,
                uvec2 patchId @ SV_PrimitiveID
            )
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_PrimitiveID.*int or uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(vector_id_code), "tessellation_control"
        )

    int_id_code = """
    shader valid_hull_int_primitive_id {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec3 edges @ SV_TessFactor;
            float inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(InputPatch<HSInput, 3> patch) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(
                InputPatch<HSInput, 3> patch,
                uint id @ SV_OutputControlPointID,
                int patchId @ SV_PrimitiveID
            )
                @domain(tri)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(int_id_code), "tessellation_control"
    )
    assert "int patchId : SV_PrimitiveID" in generated


def test_directx_tessellation_evaluation_validates_domain_location_components():
    tri_vec2_code = """
    shader bad_tri_domain_location_components {
        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_evaluation {
            vec4 main(OutputPatch<HSOutput, 3> patch, vec2 uv @ SV_DomainLocation)
                @domain(tri) {
                return vec4(0.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_DomainLocation.*3"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(tri_vec2_code), "tessellation_evaluation"
        )

    quad_vec3_code = """
    shader bad_quad_domain_location_components {
        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_evaluation {
            vec4 main(OutputPatch<HSOutput, 4> patch, vec3 uvw @ SV_DomainLocation)
                @domain(quad) {
                return vec4(0.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_DomainLocation.*2"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(quad_vec3_code), "tessellation_evaluation"
        )

    isoline_vec2_code = """
    shader valid_isoline_domain_location_components {
        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_evaluation {
            vec4 main(OutputPatch<HSOutput, 2> patch, vec2 uv @ SV_DomainLocation)
                @domain(isoline) {
                return vec4(0.0);
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(isoline_vec2_code), "tessellation_evaluation"
    )
    assert "float2 uv : SV_DomainLocation" in generated


def test_directx_tessellation_evaluation_validates_domain_location_type():
    int_vector_code = """
    shader bad_domain_location_integer_type {
        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_evaluation {
            vec4 main(OutputPatch<HSOutput, 4> patch, ivec2 uv @ SV_DomainLocation)
                @domain(quad) {
                return vec4(0.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_DomainLocation.*floating-point"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(int_vector_code), "tessellation_evaluation"
        )

    bool_vector_code = """
    shader bad_domain_location_bool_type {
        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_evaluation {
            vec4 main(OutputPatch<HSOutput, 3> patch, bvec3 bary @ SV_DomainLocation)
                @domain(tri) {
                return vec4(0.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_DomainLocation.*floating-point"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(bool_vector_code), "tessellation_evaluation"
        )

    valid_code = """
    shader valid_domain_location_float_type {
        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_evaluation {
            vec4 main(OutputPatch<HSOutput, 3> patch, vec3 bary @ SV_DomainLocation)
                @domain(tri) {
                return vec4(0.0);
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(valid_code), "tessellation_evaluation"
    )
    assert "float3 bary : SV_DomainLocation" in generated


def test_directx_tessellation_evaluation_validates_primitive_id_type():
    float_id_code = """
    shader bad_domain_float_primitive_id {
        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_evaluation {
            vec4 main(
                OutputPatch<HSOutput, 3> patch,
                vec3 bary @ SV_DomainLocation,
                float primitiveId @ SV_PrimitiveID
            ) @domain(tri) {
                return vec4(0.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_PrimitiveID.*int or uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(float_id_code), "tessellation_evaluation"
        )

    vector_id_code = """
    shader bad_domain_vector_primitive_id {
        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_evaluation {
            vec4 main(
                OutputPatch<HSOutput, 4> patch,
                vec2 uv @ SV_DomainLocation,
                ivec2 primitiveId @ SV_PrimitiveID
            ) @domain(quad) {
                return vec4(0.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_PrimitiveID.*int or uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(vector_id_code), "tessellation_evaluation"
        )

    uint_id_code = """
    shader valid_domain_uint_primitive_id {
        struct HSOutput {
            vec3 position @ POSITION;
        };

        tessellation_evaluation {
            vec4 main(
                OutputPatch<HSOutput, 2> patch,
                vec2 uv @ SV_DomainLocation,
                uint primitiveId @ SV_PrimitiveID
            ) @domain(isoline) {
                return vec4(0.0);
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(uint_id_code), "tessellation_evaluation"
    )
    assert "uint primitiveId : SV_PrimitiveID" in generated


def test_directx_tessellation_evaluation_validates_outputpatch_count_against_hull():
    mismatched_code = """
    shader mismatched_domain_patch_count {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec4 edges @ SV_TessFactor;
            vec2 inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(InputPatch<HSInput, 4> patch) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 4> patch, uint id @ SV_OutputControlPointID)
                @domain(quad)
                @partitioning(fractional_even)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(4)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }

        tessellation_evaluation {
            vec4 main(OutputPatch<HSOutput, 3> patch, vec2 uv @ SV_DomainLocation)
                @domain(quad) {
                return vec4(0.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="OutputPatch.*outputcontrolpoints"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(mismatched_code), "tessellation_evaluation"
        )

    matching_code = mismatched_code.replace(
        "OutputPatch<HSOutput, 3>", "OutputPatch<HSOutput, 4>"
    )
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(matching_code), "tessellation_evaluation"
    )
    assert "float4 DSMain(OutputPatch<HSOutput, 4> patch" in generated


def test_directx_tessellation_evaluation_validates_outputpatch_element_type_against_hull():
    mismatched_code = """
    shader mismatched_domain_patch_element_type {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct OtherOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec4 edges @ SV_TessFactor;
            vec2 inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(InputPatch<HSInput, 4> patch) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 4> patch, uint id @ SV_OutputControlPointID)
                @domain(quad)
                @partitioning(fractional_even)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(4)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }

        tessellation_evaluation {
            vec4 main(OutputPatch<OtherOutput, 4> patch, vec2 uv @ SV_DomainLocation)
                @domain(quad) {
                return vec4(0.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="OutputPatch element type.*HSOutput"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(mismatched_code), "tessellation_evaluation"
        )

    matching_code = mismatched_code.replace(
        "OutputPatch<OtherOutput, 4>", "OutputPatch<HSOutput, 4>"
    )
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(matching_code), "tessellation_evaluation"
    )
    assert "float4 DSMain(OutputPatch<HSOutput, 4> patch" in generated


def test_directx_tessellation_evaluation_validates_domain_matches_hull_domain():
    mismatched_code = """
    shader mismatched_tessellation_domain {
        struct HSInput {
            vec3 position @ POSITION;
        };

        struct HSOutput {
            vec3 position @ POSITION;
        };

        struct HSConstData {
            vec4 edges @ SV_TessFactor;
            vec2 inside @ SV_InsideTessFactor;
        };

        tessellation_control {
            HSConstData HSConst(InputPatch<HSInput, 4> patch) {
                HSConstData constants;
                return constants;
            }

            HSOutput main(InputPatch<HSInput, 4> patch, uint id @ SV_OutputControlPointID)
                @domain(quad)
                @partitioning(fractional_even)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(4)
                @patchconstantfunc(HSConst) {
                HSOutput output;
                return output;
            }
        }

        tessellation_evaluation {
            vec4 main(OutputPatch<HSOutput, 4> patch, vec2 uv @ SV_DomainLocation)
                @domain(tri) {
                return vec4(0.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="domain.*tessellation_control domain"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(mismatched_code), "tessellation_evaluation"
        )

    matching_code = mismatched_code.replace("@domain(tri) {", "@domain(quad) {")
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(matching_code), "tessellation_evaluation"
    )
    assert '[domain("quad")]' in generated
    assert "float4 DSMain(OutputPatch<HSOutput, 4> patch" in generated


def test_directx_geometry_stage_requires_maxvertexcount_attribute():
    code = """
    shader missing_geometry_attribute {
        geometry {
            void main() { }
        }
    }
    """

    with pytest.raises(ValueError, match="maxvertexcount"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(code), "geometry")


def test_directx_tessellation_control_requires_hlsl_stage_attributes():
    code = """
    shader missing_hull_attributes {
        tessellation_control {
            void main() @domain(tri) @outputcontrolpoints(3) { }
        }
    }
    """

    with pytest.raises(ValueError, match="outputtopology"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_control"
        )


def test_directx_tessellation_evaluation_requires_domain_attribute():
    code = """
    shader missing_domain_attribute {
        tessellation_evaluation {
            void main() { }
        }
    }
    """

    with pytest.raises(ValueError, match="domain"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(code), "tessellation_evaluation"
        )


def test_generate_stage_filters_combined_vertex_fragment_units():
    code = """
    shader combined {
        struct VSInput {
            vec3 position @ POSITION;
            vec2 uv @ TEXCOORD0;
        };

        struct VSOutput {
            vec4 position @ gl_Position;
            vec2 uv @ TEXCOORD0;
        };

        float adjust(float value) {
            return value + 1.0;
        }

        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.position = vec4(input.position, 1.0);
                output.uv = input.uv;
                return output;
            }
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.uv, 0.0, 1.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generator = HLSLCodeGen()

    vertex_code = generator.generate_stage(ast, "vertex")
    fragment_code = generator.generate_stage(ast, "fragment")

    assert "float adjust(float value)" in vertex_code
    assert "float adjust(float value)" in fragment_code
    assert "float3 position: POSITION;" in vertex_code
    assert "float4 position: SV_POSITION;" in vertex_code
    assert "float2 uv: TEXCOORD0;" in fragment_code
    assert "VSOutput VSMain" in vertex_code
    assert "float4 PSMain" not in vertex_code
    assert "float4 PSMain(VSOutput input): SV_TARGET" in fragment_code
    assert "VSOutput VSMain" not in fragment_code


def test_directx_stage_local_helpers_order_by_dependencies_before_entrypoint():
    shader = """
    shader StageLocalHelperDependencyOrder {
        fragment {
            vec4 first(vec2 uv) {
                return second(uv);
            }

            vec4 second(vec2 uv) {
                return vec4(uv, 0.0, 1.0);
            }

            vec4 main(vec2 uv @ TEXCOORD0) @ SV_Target {
                return first(uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    second_index = generated_code.index("float4 second(float2 uv)")
    first_index = generated_code.index("float4 first(float2 uv)")
    entry_index = generated_code.index("float4 PSMain(float2 uv : TEXCOORD0)")

    assert second_index < first_index < entry_index
    assert "return second(uv);" in generated_code
    assert "return first(uv);" in generated_code


def test_directx_stage_entry_name_avoids_global_helper_collision():
    shader = """
    shader StageEntryGlobalHelperCollision {
        vec4 PSMain(vec2 uv) {
            return vec4(uv, 0.0, 1.0);
        }

        fragment {
            vec4 main(vec2 uv @ TEXCOORD0) @ SV_Target {
                return vec4(uv, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "float4 PSMain(float2 uv)" in generated_code
    assert "float4 PSMain_2(float2 uv : TEXCOORD0): SV_Target" in generated_code


def test_directx_stage_entry_name_avoids_local_helper_collision():
    shader = """
    shader StageEntryLocalHelperCollision {
        fragment {
            vec4 PSMain(vec2 uv) {
                return vec4(uv, 0.0, 1.0);
            }

            vec4 main(vec2 uv @ TEXCOORD0) @ SV_Target {
                return PSMain(uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "float4 PSMain(float2 uv)" in generated_code
    assert "float4 PSMain_2(float2 uv : TEXCOORD0): SV_Target" in generated_code
    assert "return PSMain(uv);" in generated_code


def test_directx_qualified_non_graphics_entries_use_stage_specific_names():
    ast = ShaderNode(
        "QualifiedAdvancedEntries",
        ExecutionModel.GRAPHICS_PIPELINE,
        functions=[
            FunctionNode(
                "MSMain",
                PrimitiveType("void"),
                [],
                BlockNode([]),
            ),
            FunctionNode(
                "meshEntry",
                PrimitiveType("void"),
                [],
                BlockNode([]),
                qualifiers=["mesh"],
            ),
            FunctionNode(
                "rayEntry",
                PrimitiveType("void"),
                [],
                BlockNode([]),
                qualifiers=["ray_generation"],
            ),
        ],
    )

    generated_code = HLSLCodeGen().generate(ast)

    assert "void MSMain()" in generated_code
    assert '[shader("mesh")]\nvoid MSMain_2()' in generated_code
    assert '[shader("raygeneration")]\nvoid RayGenMain()' in generated_code
    assert "void meshEntry()" not in generated_code
    assert "void rayEntry()" not in generated_code


def test_compute_stage_emits_default_numthreads_attribute():
    shader = """
    shader ComputeNumthreadsSmoke {
        compute {
            void main() {
                int value = 1;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    compute_code = HLSLCodeGen().generate_stage(ast, "compute")

    assert "[numthreads(1, 1, 1)]" in compute_code
    assert compute_code.index("[numthreads(1, 1, 1)]") < compute_code.index("CSMain")


def test_compute_stage_uses_execution_config_numthreads():
    shader = """
    shader ComputeConfiguredNumthreadsSmoke {
        compute {
            void main() {
                int value = 1;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    ast.stages[ShaderStage.COMPUTE].execution_config = {"numthreads": (8, 4, 2)}

    compute_code = HLSLCodeGen().generate_stage(ast, "compute")

    assert "[numthreads(8, 4, 2)]" in compute_code


def test_compute_stage_emits_wave_size_attribute():
    shader = """
    shader ComputeWaveSize {
        compute {
            void main() @ WaveSize(32) {
                int value = 1;
            }
        }
    }
    """

    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "[numthreads(1, 1, 1)]\n[WaveSize(32)]\nvoid CSMain" in generated


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("WaveSize(16, 64)", "[WaveSize(16, 64)]"),
        ("WaveSize(16, 64, 32)", "[WaveSize(16, 64, 32)]"),
    ],
)
def test_compute_stage_emits_wave_size_range_attribute(attribute, expected):
    shader = f"""
    shader ComputeWaveSizeRange {{
        compute {{
            void main() @ {attribute} {{
                int value = 1;
            }}
        }}
    }}
    """

    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert f"[numthreads(1, 1, 1)]\n{expected}\nvoid CSMain" in generated


def test_compute_wave_size_attribute_validates_arguments_and_stage():
    missing_argument_code = """
    shader BadWaveSizeMissingArgument {
        compute {
            void main() @ WaveSize { }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveSize.*requires 1, 2, or 3 arguments",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(missing_argument_code), "compute"
        )

    too_many_arguments_code = """
    shader BadWaveSizeTooManyArguments {
        compute {
            void main() @ WaveSize(16, 32, 64, 128) { }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveSize.*requires 1, 2, or 3 arguments",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(too_many_arguments_code), "compute"
        )

    non_literal_code = """
    shader BadWaveSizeNonLiteral {
        compute {
            void main() @ WaveSize(WAVE_SIZE) { }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveSize.*immediate integer",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(non_literal_code), "compute"
        )

    invalid_lane_count_code = """
    shader BadWaveSizeLaneCount {
        compute {
            void main() @ WaveSize(12) { }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveSize.*4, 8, 16, 32, 64, or 128",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(invalid_lane_count_code), "compute"
        )

    invalid_minimum_code = """
    shader BadWaveSizeMinimumLaneCount {
        compute {
            void main() @ WaveSize(12, 32) { }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveSize.*minimum.*4, 8, 16, 32, 64, or 128",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(invalid_minimum_code), "compute"
        )

    invalid_range_code = """
    shader BadWaveSizeRange {
        compute {
            void main() @ WaveSize(64, 32) { }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveSize.*minimum lane count.*less than or equal",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(invalid_range_code), "compute"
        )

    invalid_preferred_lane_count_code = """
    shader BadWaveSizePreferredLaneCount {
        compute {
            void main() @ WaveSize(16, 64, 12) { }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveSize.*preferred.*4, 8, 16, 32, 64, or 128",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(invalid_preferred_lane_count_code), "compute"
        )

    invalid_preferred_range_code = """
    shader BadWaveSizePreferredRange {
        compute {
            void main() @ WaveSize(16, 32, 64) { }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveSize.*preferred lane count.*between minimum and maximum",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(invalid_preferred_range_code), "compute"
        )

    fragment_code = """
    shader BadWaveSizeStage {
        fragment {
            vec4 main() @ SV_Target @ WaveSize(32) {
                return vec4(1.0);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveSize.*compute",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(fragment_code), "fragment"
        )


def test_fragment_stage_emits_waveops_include_helper_lanes_attribute():
    shader = """
    shader FragmentWaveOpsHelperLanes {
        fragment {
            vec4 main(bool predicate @ TEXCOORD0)
                @ SV_Target
                @ WaveOpsIncludeHelperLanes {
                return QuadAny(predicate) ? vec4(1.0) : vec4(0.0);
            }
        }
    }
    """

    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "[WaveOpsIncludeHelperLanes]\nfloat4 PSMain" in generated
    assert "QuadAny(predicate)" in generated


def test_waveops_include_helper_lanes_rejects_arguments_and_non_fragment_stages():
    argument_code = """
    shader BadWaveOpsHelperLanesArgument {
        fragment {
            vec4 main() @ SV_Target @ WaveOpsIncludeHelperLanes(true) {
                return vec4(1.0);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveOpsIncludeHelperLanes.*does not accept argument",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(argument_code), "fragment"
        )

    compute_code = """
    shader BadWaveOpsHelperLanesStage {
        compute {
            void main() @ WaveOpsIncludeHelperLanes { }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveOpsIncludeHelperLanes.*fragment",
    ):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(compute_code), "compute")


def test_compute_stage_validates_system_value_parameter_types():
    float_group_index_code = """
    shader BadComputeGroupIndex {
        compute {
            void main(float groupIndex @ SV_GroupIndex) { }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_GroupIndex.*scalar uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(float_group_index_code), "compute"
        )

    vector_group_index_code = """
    shader BadComputeVectorGroupIndex {
        compute {
            void main(uvec2 groupIndex @ SV_GroupIndex) { }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_GroupIndex.*scalar uint"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(vector_group_index_code), "compute"
        )

    dispatch_vec2_code = """
    shader BadComputeDispatchThreadId {
        compute {
            void main(uvec2 dispatchId @ SV_DispatchThreadID) { }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_DispatchThreadID.*uint3"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(dispatch_vec2_code), "compute"
        )

    signed_group_id_code = """
    shader BadComputeSignedGroupId {
        compute {
            void main(ivec3 groupId @ SV_GroupID) { }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_GroupID.*uint3"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(signed_group_id_code), "compute"
        )

    valid_code = """
    shader ValidComputeSystemValues {
        compute {
            void main(
                uvec3 groupId @ SV_GroupID,
                uvec3 groupThreadId @ SV_GroupThreadID,
                uvec3 dispatchId @ SV_DispatchThreadID,
                uint groupIndex @ SV_GroupIndex
            ) { }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(valid_code), "compute"
    )
    assert "uint3 groupId : SV_GroupID" in generated
    assert "uint3 groupThreadId : SV_GroupThreadID" in generated
    assert "uint3 dispatchId : SV_DispatchThreadID" in generated
    assert "uint groupIndex : SV_GroupIndex" in generated


def test_directx_thread_system_crossgl_semantics_lower_and_validate():
    invalid_code = """
    shader BadCrossGLThreadSemantic {
        compute {
            void main(vec3 dispatchId @ gl_GlobalInvocationID) { }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_DispatchThreadID.*uint3"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(invalid_code), "compute")

    valid_code = """
    shader CrossGLThreadSemantics {
        compute {
            void main(
                uvec3 groupId @ gl_WorkGroupID,
                uvec3 groupThreadId @ gl_LocalInvocationID,
                uvec3 dispatchId @ gl_GlobalInvocationID,
                uint groupIndex @ gl_LocalInvocationIndex
            ) { }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(valid_code), "compute"
    )

    assert "uint3 groupId : SV_GroupID" in generated
    assert "uint3 groupThreadId : SV_GroupThreadID" in generated
    assert "uint3 dispatchId : SV_DispatchThreadID" in generated
    assert "uint groupIndex : SV_GroupIndex" in generated
    assert "gl_GlobalInvocationID" not in generated
    assert "gl_LocalInvocationID" not in generated
    assert "gl_WorkGroupID" not in generated
    assert "gl_LocalInvocationIndex" not in generated


def test_directx_mesh_dispatch_mesh_id_semantic_lowers_and_validates():
    invalid_code = """
    shader BadMeshDispatchMeshID {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                vec3 dispatchMeshId @ mesh_DispatchMeshID,
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_DispatchMeshID.*uint3"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(invalid_code), "mesh")

    valid_code = """
    shader MeshDispatchMeshID {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                uvec3 dispatchMeshId @ mesh_DispatchMeshID,
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
                verts[0].position = vec4(float(dispatchMeshId.x), 0.0, 0.0, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(valid_code), "mesh"
    )

    assert "uint3 dispatchMeshId : SV_DispatchMeshID" in generated
    assert "mesh_DispatchMeshID" not in generated


def test_directx_mesh_view_id_semantic_lowers_and_validates():
    wrong_type_code = """
    shader BadMeshViewIDType {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                vec2 viewId @ gl_ViewID,
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SV_ViewID.*scalar uint"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(wrong_type_code), "mesh")

    wrong_stage_code = """
    shader BadTaskViewID {
        task {
            void main(uint viewId @ gl_ViewID) @numthreads(1, 1, 1) {
                DispatchMesh(1, 1, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="task.*SV_ViewID.*mesh"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(wrong_stage_code), "task")

    valid_code = """
    shader MeshViewID {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                uint viewId @ gl_ViewID,
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
                verts[0].position = vec4(float(viewId), 0.0, 0.0, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """
    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(valid_code), "mesh"
    )

    assert "uint viewId : SV_ViewID" in generated
    assert "gl_ViewID" not in generated


def test_wave_and_rayquery_intrinsics_codegen():
    code = """
    shader WaveIntrinsicCoverage {
        compute {
            uint combine(uint value, bool predicate, uint lane, uvec4 mask) {
                uint laneCount = WaveGetLaneCount();
                uint laneIndex = WaveGetLaneIndex();
                bool firstLane = WaveIsFirstLane();
                uint sum = WaveActiveSum(value);
                uint product = WaveActiveProduct(value);
                uint andValue = WaveActiveBitAnd(value);
                uint orValue = WaveActiveBitOr(value);
                uint xorValue = WaveActiveBitXor(value);
                uint minValue = WaveActiveMin(value);
                uint maxValue = WaveActiveMax(value);
                bool allTrue = WaveActiveAllTrue(predicate);
                bool anyTrue = WaveActiveAnyTrue(predicate);
                uvec4 ballot = WaveActiveBallot(predicate);
                uint laneValue = WaveReadLaneAt(value, lane);
                uint firstValue = WaveReadLaneFirst(value);
                uint prefixSum = WavePrefixSum(value);
                uint prefixProduct = WavePrefixProduct(value);
                uvec4 matchMask = WaveMatch(value);
                uint multiSum = WaveMultiPrefixSum(value, mask);
                uint multiCount = WaveMultiPrefixCountBits(predicate, mask);
                uint multiProduct = WaveMultiPrefixProduct(value, mask);
                uint multiAnd = WaveMultiPrefixBitAnd(value, mask);
                uint multiOr = WaveMultiPrefixBitOr(value, mask);
                uint multiXor = WaveMultiPrefixBitXor(value, mask);
                uint quadX = QuadReadAcrossX(value);
                uint quadY = QuadReadAcrossY(value);
                uint quadDiagonal = QuadReadAcrossDiagonal(value);
                uint quadLane = QuadReadLaneAt(value, lane);
                bool quadAny = QuadAny(predicate);
                bool quadAll = QuadAll(predicate);
                return laneCount + laneIndex + sum + product + andValue + orValue
                    + xorValue + minValue + maxValue + laneValue + firstValue
                    + prefixSum + prefixProduct + matchMask.x + multiSum
                    + multiCount + multiProduct + multiAnd + multiOr + multiXor
                    + quadX + quadY + quadDiagonal + quadLane + ballot.x
                    + (firstLane ? 1u : 0u) + (allTrue ? 1u : 0u)
                    + (anyTrue ? 1u : 0u) + (quadAny ? 1u : 0u)
                    + (quadAll ? 1u : 0u);
            }

            void main() {
                uint result = combine(3u, true, 1u, uvec4(1u, 0u, 1u, 0u));
                RayQuery rq;
                rq.Proceed();
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated = generate_code(ast)

    for intrinsic in [
        "WaveGetLaneCount()",
        "WaveGetLaneIndex()",
        "WaveIsFirstLane()",
        "WaveActiveSum(value)",
        "WaveActiveProduct(value)",
        "WaveActiveBitAnd(value)",
        "WaveActiveBitOr(value)",
        "WaveActiveBitXor(value)",
        "WaveActiveMin(value)",
        "WaveActiveMax(value)",
        "WaveActiveAllTrue(predicate)",
        "WaveActiveAnyTrue(predicate)",
        "WaveActiveBallot(predicate)",
        "WaveReadLaneAt(value, lane)",
        "WaveReadLaneFirst(value)",
        "WavePrefixSum(value)",
        "WavePrefixProduct(value)",
        "WaveMatch(value)",
        "WaveMultiPrefixSum(value, mask)",
        "WaveMultiPrefixCountBits(predicate, mask)",
        "WaveMultiPrefixProduct(value, mask)",
        "WaveMultiPrefixBitAnd(value, mask)",
        "WaveMultiPrefixBitOr(value, mask)",
        "WaveMultiPrefixBitXor(value, mask)",
        "QuadReadAcrossX(value)",
        "QuadReadAcrossY(value)",
        "QuadReadAcrossDiagonal(value)",
        "QuadReadLaneAt(value, lane)",
        "QuadAny(predicate)",
        "QuadAll(predicate)",
    ]:
        assert intrinsic in generated
    assert "rq.Proceed" in generated


def test_directx_wave_intrinsics_reject_wrong_arity():
    code = """
    shader BadWaveIntrinsicArity {
        compute {
            uint main(uint value) {
                return WaveReadLaneAt(value);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="DirectX wave intrinsic 'WaveReadLaneAt' requires 2 argument",
    ):
        generate_code(parse_code(tokenize_code(code)))

    no_arg_code = code.replace("WaveReadLaneAt(value)", "WaveGetLaneIndex(value)")
    with pytest.raises(
        ValueError,
        match="DirectX wave intrinsic 'WaveGetLaneIndex' requires 0 argument",
    ):
        generate_code(parse_code(tokenize_code(no_arg_code)))

    quad_no_arg_code = code.replace("WaveReadLaneAt(value)", "QuadAny()")
    with pytest.raises(
        ValueError,
        match="DirectX wave intrinsic 'QuadAny' requires 1 argument",
    ):
        generate_code(parse_code(tokenize_code(quad_no_arg_code)))


def test_directx_wave_intrinsics_validate_argument_types():
    bad_vote_predicate_code = """
    shader BadWaveVotePredicate {
        compute {
            uint main(uint value) {
                return WaveActiveCountBits(value);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveActiveCountBits.*predicate.*scalar bool",
    ):
        generate_code(parse_code(tokenize_code(bad_vote_predicate_code)))

    bad_lane_index_code = """
    shader BadWaveLaneIndex {
        compute {
            uint main(uint value, float lane) {
                return WaveReadLaneAt(value, lane);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveReadLaneAt.*lane index.*scalar int or uint",
    ):
        generate_code(parse_code(tokenize_code(bad_lane_index_code)))

    bad_quad_lane_index_code = """
    shader BadQuadLaneIndexRange {
        compute {
            uint main(uint value) {
                return QuadReadLaneAt(value, 4u);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="QuadReadLaneAt.*quad lane index.*0.*3",
    ):
        generate_code(parse_code(tokenize_code(bad_quad_lane_index_code)))

    bad_quad_vote_predicate_code = """
    shader BadQuadVotePredicate {
        compute {
            bool main(uint value) {
                return QuadAny(value);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="QuadAny.*predicate.*scalar bool",
    ):
        generate_code(parse_code(tokenize_code(bad_quad_vote_predicate_code)))

    bad_multi_prefix_mask_code = """
    shader BadWaveMultiPrefixMask {
        compute {
            uint main(uint value, uint mask) {
                return WaveMultiPrefixSum(value, mask);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveMultiPrefixSum.*partition mask.*uint4",
    ):
        generate_code(parse_code(tokenize_code(bad_multi_prefix_mask_code)))

    bad_multi_prefix_count_predicate_code = """
    shader BadWaveMultiPrefixCountBitsPredicate {
        compute {
            uint main(uint value, uvec4 mask) {
                return WaveMultiPrefixCountBits(value, mask);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveMultiPrefixCountBits.*predicate.*scalar bool",
    ):
        generate_code(parse_code(tokenize_code(bad_multi_prefix_count_predicate_code)))

    bad_bitwise_value_code = """
    shader BadWaveBitwiseValue {
        compute {
            float main(float value) {
                return WaveActiveBitAnd(value);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="WaveActiveBitAnd.*value.*integer scalar or vector",
    ):
        generate_code(parse_code(tokenize_code(bad_bitwise_value_code)))

    valid_code = """
    shader ValidWaveTypeArguments {
        compute {
            uint main(bool predicate, uint value, uint lane, uvec4 mask) {
                uint count = WaveActiveCountBits(predicate);
                uint prefix = WavePrefixCountBits(true);
                uint laneValue = WaveReadLaneAt(value, lane);
                uint quadValue = QuadReadLaneAt(value, 3u);
                bool quadAny = QuadAny(predicate);
                bool quadAll = QuadAll(predicate);
                uint multi = WaveMultiPrefixSum(value, mask);
                uint multiCount = WaveMultiPrefixCountBits(predicate, mask);
                return count + prefix + laneValue + quadValue + multi + multiCount
                    + (quadAny ? 1u : 0u) + (quadAll ? 1u : 0u);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(valid_code)))

    assert "WaveActiveCountBits(predicate)" in generated
    assert "WavePrefixCountBits(true)" in generated
    assert "WaveReadLaneAt(value, lane)" in generated
    assert "QuadReadLaneAt(value, 3u)" in generated
    assert "QuadAny(predicate)" in generated
    assert "QuadAll(predicate)" in generated
    assert "WaveMultiPrefixSum(value, mask)" in generated
    assert "WaveMultiPrefixCountBits(predicate, mask)" in generated


def test_directx_wave_active_all_equal_preserves_input_shape():
    code = """
    shader WaveAllEqualShapes {
        compute {
            uint main(float value, vec2 pair, ivec3 lanes, bvec4 flags, mat2 matrix) {
                let scalarEqual = WaveActiveAllEqual(value);
                bvec2 pairEqual = WaveActiveAllEqual(pair);
                let lanesEqual = WaveActiveAllEqual(lanes);
                let flagsEqual = WaveActiveAllEqual(flags);
                let matrixEqual = WaveActiveAllEqual(matrix);
                return (scalarEqual ? 1u : 0u)
                    + (pairEqual.x ? 1u : 0u)
                    + (lanesEqual.y ? 1u : 0u)
                    + (flagsEqual.z ? 1u : 0u);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    for declaration in [
        "bool scalarEqual = WaveActiveAllEqual(value);",
        "bool2 pairEqual = WaveActiveAllEqual(pair);",
        "bool3 lanesEqual = WaveActiveAllEqual(lanes);",
        "bool4 flagsEqual = WaveActiveAllEqual(flags);",
        "bool2x2 matrixEqual = WaveActiveAllEqual(matrix);",
    ]:
        assert declaration in generated


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            "bool wrong = WaveActiveAllEqual(pair);",
            "DirectX wave intrinsic 'WaveActiveAllEqual' requires "
            "bool2 result context, got bool",
        ),
        (
            "bvec2 wrong = WaveActiveAllEqual(value);",
            "DirectX wave intrinsic 'WaveActiveAllEqual' requires "
            "bool result context, got bool2",
        ),
        (
            "vec2 wrong = WaveActiveAllEqual(pair);",
            "DirectX wave intrinsic 'WaveActiveAllEqual' requires "
            "bool2 result context, got float2",
        ),
        (
            "bool wrong = WaveActiveAllEqual(values);",
            "DirectX wave intrinsic 'WaveActiveAllEqual' value argument must be "
            "basic scalar, vector, or matrix, got float[2]",
        ),
    ],
)
def test_directx_wave_active_all_equal_validates_shape_and_argument(body, match):
    code = f"""
    shader BadWaveAllEqualShape {{
        compute {{
            uint main(float value, vec2 pair) {{
                float values[2];
                {body}
                return 0u;
            }}
        }}
    }}
    """
    with pytest.raises(ValueError, match=re.escape(match)):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            """
                uint wrong = WaveActiveBallot(predicate);
                return wrong;
            """,
            "DirectX wave intrinsic 'WaveActiveBallot' requires "
            "uint4 result context, got uint",
        ),
        (
            """
                uvec4 wrong = WaveGetLaneIndex();
                return wrong.x;
            """,
            "DirectX wave intrinsic 'WaveGetLaneIndex' requires "
            "uint result context, got uint4",
        ),
        (
            """
                bool wrong = WaveActiveSum(value);
                return wrong ? 1u : 0u;
            """,
            "DirectX wave intrinsic 'WaveActiveSum' requires "
            "uint result context, got bool",
        ),
        (
            """
                float wrong = QuadReadAcrossX(pair);
                return uint(wrong);
            """,
            "DirectX wave intrinsic 'QuadReadAcrossX' requires "
            "float2 result context, got float",
        ),
        (
            "return WaveActiveBallot(predicate);",
            "DirectX wave intrinsic 'WaveActiveBallot' requires "
            "uint4 result context, got uint",
        ),
        (
            "return take(WaveActiveBallot(predicate));",
            "DirectX wave intrinsic 'WaveActiveBallot' requires "
            "uint4 result context, got uint",
        ),
    ],
)
def test_directx_wave_intrinsics_validate_result_contexts(body, match):
    code = f"""
    shader BadWaveResultContext {{
        compute {{
            uint take(uint value) {{
                return value;
            }}

            uint main(uint value, bool predicate, uint lane, uvec4 mask, vec2 pair) {{
                {body}
            }}
        }}
    }}
    """
    with pytest.raises(ValueError, match=re.escape(match)):
        generate_code(parse_code(tokenize_code(code)))


def test_directx_wave_intrinsics_infer_let_result_types():
    code = """
    shader WaveIntrinsicLetTypes {
        compute {
            uint main(uint value, int bits, bool predicate, uint lane, uvec4 mask, vec2 pair) {
                let laneCount = WaveGetLaneCount();
                let laneIndex = WaveGetLaneIndex();
                let firstLane = WaveIsFirstLane();
                let allTrue = WaveActiveAllTrue(predicate);
                let anyTrue = WaveActiveAnyTrue(predicate);
                let allEqual = WaveActiveAllEqual(value);
                let count = WaveActiveCountBits(predicate);
                let prefixCount = WavePrefixCountBits(predicate);
                let ballot = WaveActiveBallot(predicate);
                let matchMask = WaveMatch(value);
                let reduced = WaveActiveSum(value);
                let bitAnd = WaveActiveBitAnd(bits);
                let laneValue = WaveReadLaneAt(value, lane);
                let firstValue = WaveReadLaneFirst(value);
                let prefixValue = WavePrefixSum(value);
                let multiValue = WaveMultiPrefixSum(value, mask);
                let multiCount = WaveMultiPrefixCountBits(predicate, mask);
                let quadPair = QuadReadAcrossX(pair);
                let quadAny = QuadAny(predicate);
                let quadAll = QuadAll(predicate);

                return laneCount + laneIndex + count + prefixCount + ballot.x
                    + matchMask.x + reduced + uint(bitAnd) + laneValue
                    + firstValue + prefixValue + multiValue + multiCount
                    + uint(quadPair.x) + (firstLane ? 1u : 0u) + (allTrue ? 1u : 0u)
                    + (anyTrue ? 1u : 0u) + (allEqual ? 1u : 0u)
                    + (quadAny ? 1u : 0u) + (quadAll ? 1u : 0u);
            }
        }
    }
    """
    generated = generate_code(parse_code(tokenize_code(code)))

    for declaration in [
        "uint laneCount = WaveGetLaneCount();",
        "uint laneIndex = WaveGetLaneIndex();",
        "bool firstLane = WaveIsFirstLane();",
        "bool allTrue = WaveActiveAllTrue(predicate);",
        "bool anyTrue = WaveActiveAnyTrue(predicate);",
        "bool allEqual = WaveActiveAllEqual(value);",
        "uint count = WaveActiveCountBits(predicate);",
        "uint prefixCount = WavePrefixCountBits(predicate);",
        "uint4 ballot = WaveActiveBallot(predicate);",
        "uint4 matchMask = WaveMatch(value);",
        "uint reduced = WaveActiveSum(value);",
        "int bitAnd = WaveActiveBitAnd(bits);",
        "uint laneValue = WaveReadLaneAt(value, lane);",
        "uint firstValue = WaveReadLaneFirst(value);",
        "uint prefixValue = WavePrefixSum(value);",
        "uint multiValue = WaveMultiPrefixSum(value, mask);",
        "uint multiCount = WaveMultiPrefixCountBits(predicate, mask);",
        "float2 quadPair = QuadReadAcrossX(pair);",
        "bool quadAny = QuadAny(predicate);",
        "bool quadAll = QuadAll(predicate);",
    ]:
        assert declaration in generated


def test_directx_ray_query_methods_validate_trace_and_infer_results():
    code = """
    shader RayQueryInlineTrace {
        compute {
            void main() {
                RaytracingAccelerationStructure accel;
                RayDesc ray;
                RayQuery<RAY_FLAG_NONE> rq;
                rq.TraceRayInline(accel, 0, 0xFF, ray);
                let advanced = rq.Proceed();
                let candidateType = rq.CandidateType();
                let origin = rq.CandidateObjectRayOrigin();
                let barycentrics = rq.CandidateTriangleBarycentrics();
                let committedT = rq.CommittedRayT();
                rq.CommitProceduralPrimitiveHit(1.0);
                rq.CommitNonOpaqueTriangleHit();
                rq.Abort();
            }
        }
    }
    """

    generated = HLSLCodeGen().generate_stage(crosstl.translator.parse(code), "compute")

    assert "RayQuery<RAY_FLAG_NONE> rq;" in generated
    assert "rq.TraceRayInline(accel, 0, 255, ray);" in generated
    assert "bool advanced = rq.Proceed();" in generated
    assert "uint candidateType = rq.CandidateType();" in generated
    assert "float3 origin = rq.CandidateObjectRayOrigin();" in generated
    assert "float2 barycentrics = rq.CandidateTriangleBarycentrics();" in generated
    assert "float committedT = rq.CommittedRayT();" in generated
    assert "rq.CommitProceduralPrimitiveHit(1.0);" in generated
    assert "rq.CommitNonOpaqueTriangleHit();" in generated
    assert "rq.Abort();" in generated


def test_directx_ray_query_result_methods_infer_all_hlsl_result_types():
    result_methods = [
        ("CommittedStatus", "uint"),
        ("CandidatePrimitiveIndex", "uint"),
        ("CommittedPrimitiveIndex", "uint"),
        ("CandidateInstanceID", "uint"),
        ("CommittedInstanceID", "uint"),
        ("CandidateInstanceIndex", "uint"),
        ("CommittedInstanceIndex", "uint"),
        ("CandidateGeometryIndex", "uint"),
        ("CommittedGeometryIndex", "uint"),
        ("CandidateObjectRayDirection", "float3"),
        ("CommittedObjectRayOrigin", "float3"),
        ("CommittedObjectRayDirection", "float3"),
        ("CandidateRayT", "float"),
        ("CandidateObjectRayTMin", "float"),
        ("CommittedTriangleBarycentrics", "float2"),
        ("CandidateTriangleFrontFace", "bool"),
        ("CommittedTriangleFrontFace", "bool"),
        ("CandidateObjectToWorld3x4", "float3x4"),
        ("CandidateWorldToObject3x4", "float3x4"),
        ("CommittedObjectToWorld3x4", "float3x4"),
        ("CommittedWorldToObject3x4", "float3x4"),
    ]
    declarations = "\n".join(
        f"                let value{index} = rq.{method}();"
        for index, (method, _expected_type) in enumerate(result_methods)
    )
    code = f"""
    shader RayQueryResultTypes {{
        compute {{
            void main() {{
                RayQuery<RAY_FLAG_NONE> rq;
{declarations}
            }}
        }}
    }}
    """

    generated = HLSLCodeGen().generate_stage(crosstl.translator.parse(code), "compute")

    for index, (method, expected_type) in enumerate(result_methods):
        assert f"{expected_type} value{index} = rq.{method}();" in generated
    assert "None value" not in generated


def test_directx_ray_query_validates_receiver_and_trace_ray_inline_arguments():
    bad_receiver_code = """
    shader BadRayQueryReceiver {
        compute {
            void main() {
                uint rq;
                rq.Proceed();
            }
        }
    }
    """
    with pytest.raises(ValueError, match="RayQuery.Proceed receiver.*RayQuery"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(bad_receiver_code), "compute"
        )

    wrong_arity_code = """
    shader BadTraceRayInlineArity {
        compute {
            void main() {
                RaytracingAccelerationStructure accel;
                RayQuery<RAY_FLAG_NONE> rq;
                rq.TraceRayInline(accel, 0, 0xFF);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="TraceRayInline requires 4"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(wrong_arity_code), "compute"
        )

    bad_acceleration_structure_code = """
    shader BadTraceRayInlineAccelerationStructure {
        compute {
            void main() {
                uint accel;
                RayDesc ray;
                RayQuery<RAY_FLAG_NONE> rq;
                rq.TraceRayInline(accel, 0, 0xFF, ray);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="TraceRayInline acceleration structure.*RaytracingAccelerationStructure",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(bad_acceleration_structure_code), "compute"
        )

    bad_ray_descriptor_code = """
    shader BadTraceRayInlineRayDescriptor {
        compute {
            void main() {
                RaytracingAccelerationStructure accel;
                vec3 ray;
                RayQuery<RAY_FLAG_NONE> rq;
                rq.TraceRayInline(accel, 0, 0xFF, ray);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="TraceRayInline ray descriptor.*RayDesc"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(bad_ray_descriptor_code), "compute"
        )


def test_directx_ray_query_validates_commit_arguments():
    bad_hit_distance_code = """
    shader BadCommitProceduralPrimitiveHitDistance {
        compute {
            void main() {
                RayQuery<RAY_FLAG_NONE> rq;
                uint hitDistance;
                rq.CommitProceduralPrimitiveHit(hitDistance);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="CommitProceduralPrimitiveHit hit distance.*scalar floating",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(bad_hit_distance_code), "compute"
        )

    wrong_arity_code = """
    shader BadCommitNonOpaqueTriangleHitArity {
        compute {
            void main() {
                RayQuery<RAY_FLAG_NONE> rq;
                rq.CommitNonOpaqueTriangleHit(1.0);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="CommitNonOpaqueTriangleHit requires 0"):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(wrong_arity_code), "compute"
        )


def test_directx_ray_acceleration_structure_globals_use_srv_registers_and_spaces():
    code = """
    shader RayAccelerationStructureBindings {
        const int ACCEL_COUNT = 2;
        Texture2D color @binding(0) @space(1);
        RaytracingAccelerationStructure scene @binding(2) @space(1);
        RaytracingAccelerationStructure gapFill @space(1);
        RaytracingAccelerationStructure instances[ACCEL_COUNT] @binding(5) @space(1);
        Texture2D after @space(1);

        compute {
            void main() { }
        }
    }
    """

    generated = HLSLCodeGen().generate(crosstl.translator.parse(code))

    assert "Texture2D color : register(t0, space1);" in generated
    assert "RaytracingAccelerationStructure scene : register(t2, space1);" in generated
    assert (
        "RaytracingAccelerationStructure gapFill : register(t3, space1);" in generated
    )
    assert (
        "RaytracingAccelerationStructure instances[ACCEL_COUNT] : "
        "register(t5, space1);"
    ) in generated
    assert "Texture2D after : register(t7, space1);" in generated


def test_directx_ray_acceleration_structure_parameters_keep_array_sizes():
    code = """
    shader RayAccelerationStructureParameters {
        const int ACCEL_COUNT = 2;

        void traceOne(RaytracingAccelerationStructure accels[ACCEL_COUNT], RayDesc ray) {
            RayQuery<RAY_FLAG_NONE> rq;
            rq.TraceRayInline(accels[0], 0, 0xFF, ray);
        }
    }
    """

    generated = HLSLCodeGen().generate(crosstl.translator.parse(code))

    assert (
        "void traceOne(RaytracingAccelerationStructure accels[ACCEL_COUNT], "
        "RayDesc ray)"
    ) in generated
    assert "rq.TraceRayInline(accels[0], 0, 255, ray);" in generated


def test_directx_ray_acceleration_structure_register_conflicts_are_rejected():
    code = """
    shader RayAccelerationStructureRegisterConflict {
        RaytracingAccelerationStructure accel @register(t0, space2);
        Texture2D tex @register(t0, space2);

        compute {
            void main() { }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting DirectX resource binding.*t0, space2.*accel",
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(code))


def test_directx_ray_acceleration_structure_rejects_non_srv_register_prefix():
    code = """
    shader RayAccelerationStructureWrongRegisterClass {
        RaytracingAccelerationStructure accel @register(u0);

        compute {
            void main() { }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="RaytracingAccelerationStructure resource 'accel'.*t-register.*u",
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(code))


def test_directx_trace_ray_validates_global_acceleration_structure_arguments():
    code = """
    shader GlobalTraceRayAccelerationStructure {
        RaytracingAccelerationStructure scene @binding(2);

        struct RayPayload {
            vec3 color;
        };

        ray_generation {
            void main() {
                RayDesc ray;
                RayPayload payload;
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
            }
        }
    }
    """

    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(code), "ray_generation"
    )

    assert "RaytracingAccelerationStructure scene : register(t2);" in generated
    assert "TraceRay(scene, 0, 255, 0, 1, 0, ray, payload);" in generated

    bad_code = """
    shader BadGlobalTraceRayAccelerationStructure {
        Texture2D scene;

        struct RayPayload {
            vec3 color;
        };

        ray_generation {
            void main() {
                RayDesc ray;
                RayPayload payload;
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="TraceRay acceleration structure.*RaytracingAccelerationStructure",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(bad_code), "ray_generation"
        )


def test_directx_ray_query_validates_global_acceleration_structure_arguments():
    code = """
    shader GlobalRayQueryAccelerationStructure {
        RaytracingAccelerationStructure scene @binding(1);

        compute {
            void main() {
                RayDesc ray;
                RayQuery<RAY_FLAG_NONE> rq;
                rq.TraceRayInline(scene, 0, 0xFF, ray);
            }
        }
    }
    """

    generated = HLSLCodeGen().generate_stage(crosstl.translator.parse(code), "compute")

    assert "RaytracingAccelerationStructure scene : register(t1);" in generated
    assert "rq.TraceRayInline(scene, 0, 255, ray);" in generated

    bad_code = """
    shader BadGlobalRayQueryAccelerationStructure {
        Texture2D scene;

        compute {
            void main() {
                RayDesc ray;
                RayQuery<RAY_FLAG_NONE> rq;
                rq.TraceRayInline(scene, 0, 0xFF, ray);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="TraceRayInline acceleration structure.*RaytracingAccelerationStructure",
    ):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(bad_code), "compute")


def test_directx_ray_tracing_validates_literal_instance_inclusion_masks():
    trace_ray_code = """
    shader BadTraceRayInstanceMask {
        RaytracingAccelerationStructure scene;

        struct RayPayload {
            vec3 color;
        };

        ray_generation {
            void main() {
                RayDesc ray;
                RayPayload payload;
                TraceRay(scene, 0, 256, 0, 1, 0, ray, payload);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="TraceRay instance inclusion mask argument.*range 0 to 255.*256",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(trace_ray_code), "ray_generation"
        )

    ray_query_code = """
    shader BadRayQueryInstanceMask {
        compute {
            void main() {
                RaytracingAccelerationStructure accel;
                RayDesc ray;
                RayQuery<RAY_FLAG_NONE> rq;
                rq.TraceRayInline(accel, 0, 0x100, ray);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "RayQuery.TraceRayInline instance inclusion mask argument"
            ".*range 0 to 255.*256"
        ),
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(ray_query_code), "compute"
        )

    valid_code = """
    shader ValidRayInstanceMaskBounds {
        RaytracingAccelerationStructure scene;

        struct RayPayload {
            vec3 color;
        };

        ray_generation {
            void main() {
                RayDesc ray;
                RayPayload payload;
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
            }
        }
    }
    """

    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(valid_code), "ray_generation"
    )

    assert "TraceRay(scene, 0, 255, 0, 1, 0, ray, payload);" in generated


def test_directx_ray_tracing_validates_literal_ray_flags():
    unknown_flag_code = """
    shader BadTraceRayUnknownFlag {
        RaytracingAccelerationStructure scene;

        struct RayPayload {
            vec3 color;
        };

        ray_generation {
            void main() {
                RayDesc ray;
                RayPayload payload;
                TraceRay(scene, 0x400, 0xFF, 0, 1, 0, ray, payload);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="TraceRay ray flags argument.*known RAY_FLAG bits.*1024",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(unknown_flag_code), "ray_generation"
        )

    mutually_exclusive_query_code = """
    shader BadRayQueryMutuallyExclusiveFlags {
        compute {
            void main() {
                RaytracingAccelerationStructure accel;
                RayDesc ray;
                RayQuery<RAY_FLAG_NONE> rq;
                rq.TraceRayInline(
                    accel,
                    RAY_FLAG_SKIP_TRIANGLES | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES,
                    0xFF,
                    ray
                );
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "RayQuery.TraceRayInline ray flags argument.*mutually exclusive"
            ".*RAY_FLAG_SKIP_TRIANGLES.*RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES"
        ),
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(mutually_exclusive_query_code), "compute"
        )

    valid_code = """
    shader ValidRayFlags {
        RaytracingAccelerationStructure scene;

        struct RayPayload {
            vec3 color;
        };

        ray_generation {
            void main() {
                RayDesc ray;
                RayPayload payload;
                TraceRay(
                    scene,
                    RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
                        | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
                    0xFF,
                    0,
                    1,
                    0,
                    ray,
                    payload
                );
            }
        }
    }
    """

    generated = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(valid_code), "ray_generation"
    )

    assert (
        "TraceRay(scene, "
        "(RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | "
        "RAY_FLAG_SKIP_CLOSEST_HIT_SHADER), 255, 0, 1, 0, ray, payload);"
    ) in generated


def test_else_if_statement():
    code = """
    shader main {

    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };

    struct VSOutput {
        vec4 color @ COLOR;
    };

    sampler2D iChannel0;

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;

            if (input.texCoord.x > 0.5) {
                output.color = vec4(1.0, 1.0, 1.0, 1.0);
            } else if (input.texCoord.x < 0.5) {
                output.color = vec4(0.0, 0.0, 0.0, 1.0);
            } else {
                output.color = vec4(0.5, 0.5, 0.5, 1.0);

            // Pass through texture coordinates as color
            output.color = vec4(input.texCoord, 0.0, 1.0);

            return output;
        }
    }
}

fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Sample brightness and calculate bloom
            float brightness = texture(iChannel0, input.color.xy).r;
            float bloom = max(0.0, brightness - 0.5);
            if (bloom > 0.5) {
                bloom = 0.5;
            } else if (bloom < 0.5) {
                bloom = 0.0;
            } else {
                bloom = 0.5;
            }
            // Apply bloom to the texture color
            vec3 texColor = texture(iChannel0, input.color.xy).rgb;
            vec3 colorWithBloom = texColor + vec3(bloom);

            return vec4(colorWithBloom, 1.0);
        }
    }
}

    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        print(code)
    except SyntaxError:
        pytest.fail("else if codegen not implemented.")


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    float result = add(1.0, 2.0);
                }
                
                float add(float a, float b) {
                    return a + b;
                }
            }
            """,
            "add(1.0, 2.0)",
        )
    ],
)
def test_function_call(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = HLSLCodeGen()
    generated_code = code_gen.generate(ast)

    assert expected_output in generated_code


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    a |= 2;
                }
            }
            """,
            "a |= 2",
        )
    ],
)
def test_assignment_or_operator(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = HLSLCodeGen()
    generated_code = code_gen.generate(ast)

    assert expected_output in generated_code


def test_assignment_modulus_operator():
    code = """
    shader main {
        vertex {
            void main() {
                int a = 10;
                a %= 3;  // Assignment modulus operator
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "a %= 3" in generated_code or "a = a % 3" in generated_code
    except SyntaxError:
        pytest.fail("Assignment modulus operator codegen not implemented.")


def test_assignment_xor_operator():
    code = """
    shader main {
        vertex {
            void main() {
                int a = 5;
                a ^= 3;  // Assignment XOR operator
            }
        }
    }

    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        assert "a ^= 3" in generated_code or "a = a ^ 3" in generated_code
    except SyntaxError:
        pytest.fail("Assignment XOR operator codegen not implemented.")


@pytest.mark.parametrize(
    "shader, expected_output",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    a <<= 2;
                    a >>= 1;
                }
            }
            """,
            ["a <<= 2", "a >>= 1"],
        )
    ],
)
def test_assignment_shift_operators(shader, expected_output):
    ast = crosstl.translator.parse(shader)
    code_gen = HLSLCodeGen()
    generated_code = code_gen.generate(ast)

    for output in expected_output:
        assert output in generated_code


@pytest.mark.parametrize(
    "shader, expected_outputs",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    int b = 2;
                    int c = a | b;
                    int d = a & b;
                    int e = a ^ b;
                }
            }
            """,
            ["a | b", "a & b", "a ^ b"],
        )
    ],
)
def test_bitwise_operators(shader, expected_outputs):
    ast = crosstl.translator.parse(shader)
    code_gen = HLSLCodeGen()
    generated_code = code_gen.generate(ast)

    for expected in expected_outputs:
        assert expected in generated_code


def test_bitwise_and_operator():
    code = """
    shader main {
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };
    struct VSOutput {
        vec4 color @ COLOR;
    };
    sampler2D iChannel0;
    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            // Use bitwise AND on texture coordinates (for testing purposes)
            output.color = vec4(float(int(input.texCoord.x * 100.0) & 15), 
                                float(int(input.texCoord.y * 100.0) & 15), 
                                0.0, 1.0);
            return output;
        }
    }
    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Simple fragment shader to display the result of the AND operation
            return vec4(input.color.rgb, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise AND codegen not implemented")


def test_double_data_type():
    code = """
    shader DoubleShader {
        struct VSInput {
            double texCoord @ TEXCOORD0;
        };

        struct VSOutput {
            double color @ COLOR;
        };

        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = input.texCoord * 2.0;
                return output;
            }
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return vec4(input.color, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
        assert "double" in generated_code
    except SyntaxError:
        pytest.fail("Double data type not supported.")


# Test the codegen for the shift operators("<<", ">>")
@pytest.mark.parametrize(
    "shader, expected_outputs",
    [
        (
            """
            shader TestShader {
                void main() {
                    int a = 1;
                    int b = 2;
                    int c = a << b;
                    int d = a >> b;
                }
            }
            """,
            ["a << b", "a >> b"],
        )
    ],
)
def test_shift_operators(shader, expected_outputs):
    ast = crosstl.translator.parse(shader)
    code_gen = HLSLCodeGen()
    generated_code = code_gen.generate(ast)

    for expected in expected_outputs:
        assert expected in generated_code


def test_multiview_and_viewport_semantics_roundtrip():
    shader = """
    shader ViewShader {
        struct VSOut {
            vec4 position @ gl_Position;
            uint view @ gl_ViewID;
            uint layer @ gl_Layer;
            uint viewport @ gl_ViewportIndex;
        };

        vertex {
            VSOut main() {
                VSOut o;
                o.position = vec4(0.0, 0.0, 0.0, 1.0);
                o.view = 1;
                o.layer = 2;
                o.viewport = 3;
                return o;
            }
        }
    }
    """
    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)
    for semantic in ["SV_ViewID", "SV_RenderTargetArrayIndex", "SV_ViewportArrayIndex"]:
        assert semantic in generated_code


def test_bitwise_or_operator():
    code = """
    shader main {
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };
    struct VSOutput {
        vec4 color @ COLOR;
    };
    sampler2D iChannel0;
    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            // Use bitwise OR on texture coordinates (for testing purposes)
            output.color = vec4(float(int(input.texCoord.x * 100.0) | 15), 
                                float(int(input.texCoord.y * 100.0) | 15), 
                                0.0, 1.0);
            return output;
        }
    }
    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            // Simple fragment shader to display the result of the AND operation
            return vec4(input.color.rgb, 1.0);
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)
    except SyntaxError:
        pytest.fail("Bitwise OR codegen not implemented")


def test_directx_array_handling(array_test_data):
    """Test the DirectX code generator's handling of array types and array access."""
    code = """
    shader main {
    struct Particle {
        vec3 position;
        vec3 velocity;
    };

    struct Material {
        float values[4];  // Fixed-size array
        vec3 colors[];    // Dynamic array
    };

    cbuffer Constants {
        float weights[8];
        int indices[10];
    };

    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            
            // Array access in various forms
            float value = weights[2];
            int index = indices[5];
            
            // Array member access
            Material material;
            float x = material.values[0];
            vec3 color = material.colors[index];
            
            // Nested array access
            Particle particles[10];
            vec3 pos = particles[3].position;
            
            // Array access in expressions
            float sum = weights[0] + weights[1] + weights[2];
            
            return output;
        }
    }
}
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        generated_code = generate_code(ast)
        print(generated_code)

        # Use the fixture data for verification
        for expected in array_test_data["hlsl"]["array_type_declarations"]:
            assert (
                expected in generated_code
                or expected.replace("[", "<").replace("]", ">") in generated_code
            )

        for expected in array_test_data["hlsl"]["array_access"]:
            assert expected in generated_code

    except SyntaxError as e:
        pytest.fail(f"DirectX array codegen failed: {e}")


def test_directx_cbuffer_members_are_not_global_variables():
    code = """
    shader CBufferScope {
        cbuffer Constants {
            float weights[8];
            vec3 colors[2];
        };

        compute {
            void main() {
                float value = weights[2];
                vec3 color = colors[1];
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    generated_code = HLSLCodeGen().generate_stage(ast, "compute")

    assert [buffer.name for buffer in getattr(ast, "cbuffers", [])] == ["Constants"]
    assert [var.name for var in ast.global_variables] == []
    assert "cbuffer Constants : register(b0)" in generated_code
    assert "float weights[8];" in generated_code
    assert "float3 colors[2];" in generated_code
    assert "float value = weights[2];" in generated_code
    assert "float3 color = colors[1];" in generated_code
    assert "\nfloat weights[8];" not in generated_code


def test_directx_duplicate_cbuffer_members_error():
    code = """
    shader CBufferScope {
        cbuffer Camera {
            float value;
        };

        compute {
            void main() {
                float x = value;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    ast.cbuffers.append(
        StructNode(
            name="Lighting",
            members=[
                StructMemberNode(name="value", member_type=PrimitiveType("float"))
            ],
        )
    )

    with pytest.raises(
        ValueError,
        match="Ambiguous cbuffer member name\\(s\\) in DirectX output: value",
    ):
        HLSLCodeGen().generate_stage(ast, "compute")


def test_directx_duplicate_cbuffer_names_error():
    code = """
    shader CBufferScope {
        cbuffer Camera {
            float exposure;
        };

        compute {
            void main() {
                float x = exposure;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    ast.cbuffers.append(
        StructNode(
            name="Camera",
            members=[
                StructMemberNode(name="gamma", member_type=PrimitiveType("float"))
            ],
        )
    )

    with pytest.raises(
        ValueError,
        match="Duplicate cbuffer name\\(s\\) in DirectX output: Camera",
    ):
        HLSLCodeGen().generate_stage(ast, "compute")


def test_directx_cbuffer_name_conflicts_with_struct_error():
    code = """
    shader CBufferScope {
        struct Camera {
            float exposure;
        };

        cbuffer Lighting {
            float gamma;
        };

        compute {
            void main() {
                float x = gamma;
            }
        }
    }
    """

    ast = parse_code(tokenize_code(code))
    ast.cbuffers[0].name = "Camera"

    with pytest.raises(
        ValueError,
        match="Cbuffer name\\(s\\) conflict with existing DirectX declaration\\(s\\): Camera",
    ):
        HLSLCodeGen().generate_stage(ast, "compute")


def test_directx_local_array_declarations_use_hlsl_order():
    shader = """
    shader TestShader {
        void main() {
            vec3 localColors[4];
            float weights[8];
            localColors[0] = vec3(1.0, 0.0, 0.0);
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "float3 localColors[4];" in generated_code
    assert "float weights[8];" in generated_code
    assert "float3[4] localColors" not in generated_code
    assert "float[8] weights" not in generated_code


def test_directx_array_parameters_use_hlsl_order():
    shader = """
    shader TestShader {
        float accumulate(float weights[4], vec3 colors[2]) {
            return weights[0] + colors[1].x;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "float accumulate(float weights[4], float3 colors[2])" in generated_code
    assert "float[4] weights" not in generated_code
    assert "float3[2] colors" not in generated_code


def test_directx_non_resource_arrays_preserve_expression_sizes():
    shader = """
    shader ArrayExpressionSizes {
        struct Payload {
            vec3 colors[(2 + 1) * 2];
            float weights[+6];
        };

        float accumulate(float values[(2 + 1) * 2], vec3 normals[+6]) {
            float localWeights[(2 + 1) * 2];
            vec3 localNormals[+6];
            return values[2] + normals[2].x + localWeights[2] + localNormals[2].x;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "float3 colors[((2 + 1) * 2)];" in generated_code
    assert "float weights[+6];" in generated_code
    assert (
        "float accumulate(float values[((2 + 1) * 2)], float3 normals[+6])"
        in generated_code
    )
    assert "float localWeights[((2 + 1) * 2)];" in generated_code
    assert "float3 localNormals[+6];" in generated_code
    assert "float values[]" not in generated_code
    assert "float localWeights[]" not in generated_code


def test_directx_texture_resources_and_sampling():
    shader = """
    shader TextureShader {
        sampler2D colorMap;
        samplerCube envMap;

        struct VSOutput {
            vec2 uv;
            vec3 normal;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 color = texture(colorMap, input.uv);
                vec4 env = texture(envMap, input.normal);
                return color + env;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "TextureCube envMap : register(t1);" in generated_code
    assert "SamplerState colorMapSampler : register(s0);" in generated_code
    assert "SamplerState envMapSampler : register(s1);" in generated_code
    assert "colorMap.Sample(colorMapSampler, input.uv)" in generated_code
    assert "envMap.Sample(envMapSampler, input.normal)" in generated_code
    assert "sampler2D colorMap" not in generated_code
    assert "VectorType(" not in generated_code
    assert generated_code.count("// Fragment Shader") == 1


def test_directx_rejects_non_resource_shadow_of_global_resource():
    shader = """
    shader ResourceShadow {
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv;
        };

        vec4 shade(float colorMap, FSInput input) {
            float linearSampler = 1.0;
            return vec4(colorMap + linearSampler);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return shade(1.0, input);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "Non-resource local declaration\\(s\\) shadow DirectX global "
            "resource\\(s\\): colorMap, linearSampler"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_rejects_texture_call_with_non_resource_argument():
    shader = """
    shader InvalidTextureArgument {
        struct FSInput {
            vec2 uv;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float value = 1.0;
                return texture(value, input.uv);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "DirectX texture operation 'texture' requires a declared texture "
            "or image resource argument: value"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            "texture(colorMap)",
            "DirectX texture operation 'texture' requires at least 2 "
            "argument\\(s\\), got 1",
        ),
        (
            "textureLod(colorMap, input.uv)",
            "DirectX texture operation 'textureLod' requires at least 3 "
            "argument\\(s\\), got 2",
        ),
        (
            "textureGrad(colorMap, input.uv, input.uv)",
            "DirectX texture operation 'textureGrad' requires at least 4 "
            "argument\\(s\\), got 3",
        ),
        (
            "texture(colorMap, linearSampler)",
            "DirectX texture operation 'texture' requires at least 3 "
            "argument\\(s\\), got 2",
        ),
    ],
)
def test_directx_rejects_texture_call_with_too_few_arguments(call, match):
    shader = f"""
    shader InvalidTextureArity {{
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {{
            vec2 uv;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=match):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expression", "operation", "max_count", "arg_count"),
    [
        (
            "vec4(float(textureSamples(msTex, input.layer)))",
            "textureSamples",
            1,
            2,
        ),
        (
            "vec4(float(imageSamples(msTex, input.layer)))",
            "imageSamples",
            1,
            2,
        ),
        (
            "vec4(float(textureQueryLevels(colorMap, input.layer)))",
            "textureQueryLevels",
            1,
            2,
        ),
        (
            "vec4(vec2(imageSize(colorImage, input.layer)), 0.0, 1.0)",
            "imageSize",
            1,
            2,
        ),
        (
            "vec4(textureSize(colorMap, input.layer, input.layer), 0.0, 1.0)",
            "textureSize",
            2,
            3,
        ),
    ],
)
def test_directx_rejects_resource_query_call_with_too_many_arguments(
    return_expression, operation, max_count, arg_count
):
    shader = f"""
    shader InvalidTextureQueryArity {{
        sampler2D colorMap;
        sampler2DMS msTex;
        image2D colorImage;

        struct FSInput {{
            ivec2 pixel;
            int layer;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expression};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX texture operation '{operation}' accepts at most "
            f"{max_count} argument\\(s\\), got {arg_count}"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("statement", "operation", "max_count", "arg_count"),
    [
        (
            "vec4 fetched = texelFetch(colorMap, input.pixel, input.layer, input.layer);",
            "texelFetch",
            3,
            4,
        ),
        (
            "vec4 fetched = texelFetchOffset(colorMap, input.pixel, input.layer, input.offset, input.layer);",
            "texelFetchOffset",
            4,
            5,
        ),
        (
            "vec4 color = imageLoad(colorImage, input.pixel, input.layer);",
            "imageLoad",
            2,
            3,
        ),
        (
            "imageStore(colorImage, input.pixel, vec4(1.0), input.layer);",
            "imageStore",
            3,
            4,
        ),
        (
            "uint oldValue = imageAtomicAdd(counterImage, input.pixel, input.amount, input.amount);",
            "imageAtomicAdd",
            3,
            4,
        ),
        (
            "uint oldValue = imageAtomicCompSwap(counterImage, input.pixel, input.amount, input.amount, input.amount);",
            "imageAtomicCompSwap",
            4,
            5,
        ),
    ],
)
def test_directx_rejects_fixed_resource_call_with_too_many_arguments(
    statement, operation, max_count, arg_count
):
    shader = f"""
    shader InvalidFixedResourceArity {{
        sampler2D colorMap;
        image2D colorImage;
        uimage2D counterImage;

        struct FSInput {{
            ivec2 pixel;
            ivec2 offset;
            int layer;
            uint amount;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                {statement}
                return vec4(1.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX texture operation '{operation}' accepts at most "
            f"{max_count} argument\\(s\\), got {arg_count}"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expression", "operation", "max_count", "arg_count"),
    [
        (
            "texture(colorMap, input.uv, input.bias, input.bias)",
            "texture",
            3,
            4,
        ),
        (
            "textureLod(colorMap, input.uv, input.bias, input.bias)",
            "textureLod",
            3,
            4,
        ),
        (
            "textureGrad(colorMap, input.uv, input.ddx, input.ddy, input.bias)",
            "textureGrad",
            4,
            5,
        ),
        (
            "textureOffset(colorMap, input.uv, input.offset, input.bias, input.bias)",
            "textureOffset",
            4,
            5,
        ),
        (
            "vec4(textureCompare(shadowMap, compareSampler, input.uv, input.depth, input.bias))",
            "textureCompare",
            4,
            5,
        ),
        (
            "textureGather(colorMap, input.uv, input.component, input.component)",
            "textureGather",
            3,
            4,
        ),
        (
            "textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depth, input.offset, input.component)",
            "textureGatherCompareOffset",
            5,
            6,
        ),
    ],
)
def test_directx_rejects_texture_sampling_call_with_too_many_arguments(
    return_expression, operation, max_count, arg_count
):
    shader = f"""
    shader InvalidTextureSamplingArity {{
        sampler2D colorMap;
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec2 ddx;
            vec2 ddy;
            ivec2 offset;
            float bias;
            float depth;
            int component;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expression};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX texture operation '{operation}' accepts at most "
            f"{max_count} argument\\(s\\), got {arg_count}"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_rejects_texture_gather_offsets_with_ambiguous_argument_count():
    shader = """
    shader InvalidTextureGatherOffsetsArity {
        sampler2D colorMap;

        struct FSInput {
            vec2 uv;
            ivec2 offset;
            int component;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return textureGatherOffsets(
                    colorMap,
                    input.uv,
                    input.offset,
                    input.offset,
                    input.component
                );
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "DirectX texture operation 'textureGatherOffsets' accepts "
            "3, 4, 6, 7 argument\\(s\\), got 5"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("statement", "operation"),
    [
        ("vec4 color = imageLoad(colorMap, input.pixel);", "imageLoad"),
        ("imageStore(colorMap, input.pixel, vec4(1.0));", "imageStore"),
        ("ivec2 size = imageSize(colorMap);", "imageSize"),
        (
            "uint oldValue = imageAtomicAdd(colorMap, input.pixel, input.amount);",
            "imageAtomicAdd",
        ),
    ],
)
def test_directx_rejects_image_call_with_sampled_texture_argument(statement, operation):
    shader = f"""
    shader InvalidImageResource {{
        sampler2D colorMap;

        struct FSInput {{
            ivec2 pixel;
            uint amount;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                {statement}
                return vec4(1.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX image operation '{operation}' requires a storage "
            "image resource argument: colorMap"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("helper", "helper_call", "match"),
    [
        (
            "vec4 misuse(sampler sampleState, vec2 uv) { return texture(sampleState, uv); }",
            "misuse(linearSampler, input.uv)",
            "DirectX texture operation 'texture' requires a declared texture "
            "or image resource argument: sampleState",
        ),
        (
            "vec4 misuse(sampler sampleState, ivec2 pixel) { return imageLoad(sampleState, pixel); }",
            "misuse(linearSampler, input.pixel)",
            "DirectX image operation 'imageLoad' requires a storage image "
            "resource argument: sampleState",
        ),
    ],
)
def test_directx_rejects_sampler_parameter_as_texture_or_image_operand(
    helper, helper_call, match
):
    shader = f"""
    shader InvalidSamplerOperand {{
        sampler linearSampler;

        struct FSInput {{
            vec2 uv;
            ivec2 pixel;
        }};

        {helper}

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {helper_call};
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=match):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        ("texelFetch(colorMap, input.uv, 0)", "texelFetch"),
        ("imageLoad(colorImage, input.uv)", "imageLoad"),
    ],
)
def test_directx_rejects_float_coordinate_for_integer_resource_operation(
    call, operation
):
    shader = f"""
    shader InvalidResourceCoordinate {{
        sampler2D colorMap;
        image2D colorImage;

        struct FSInput {{
            vec2 uv;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX resource operation '{operation}' requires an integer "
            "coordinate argument: input.uv has type float2"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation", "dimension"),
    [
        ("texelFetch(arrayMap, input.pixel2, 0)", "texelFetch", 3),
        ("imageLoad(volumeImage, input.pixel2)", "imageLoad", 3),
        ("imageLoad(colorImage, input.pixel3)", "imageLoad", 2),
    ],
)
def test_directx_rejects_wrong_coordinate_dimension_for_resource_operation(
    call, operation, dimension
):
    shader = f"""
    shader InvalidResourceCoordinateDimension {{
        sampler2DArray arrayMap;
        image2D colorImage;
        image3D volumeImage;

        struct FSInput {{
            ivec2 pixel2;
            ivec3 pixel3;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX resource operation '{operation}' requires a "
            f"{dimension}D integer coordinate"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        ("textureOffset(arrayMap, input.uvLayer, input.offset3)", "textureOffset"),
        (
            "texelFetchOffset(arrayMap, input.pixelLayer, 0, input.offset3)",
            "texelFetchOffset",
        ),
    ],
)
def test_directx_rejects_wrong_offset_dimension_for_resource_operation(call, operation):
    shader = f"""
    shader InvalidResourceOffsetDimension {{
        sampler2DArray arrayMap;

        struct FSInput {{
            vec3 uvLayer;
            ivec3 pixelLayer;
            ivec3 offset3;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX resource operation '{operation}' requires a 2D integer offset"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        (
            "textureProjOffset(arrayMap, input.projLayer, input.offset3)",
            "textureProjOffset",
        ),
        (
            "textureGatherOffset(arrayMap, input.uvLayer, input.offset3)",
            "textureGatherOffset",
        ),
        (
            "textureGatherOffsets(arrayMap, input.uvLayer, input.offset3, input.offset3, input.offset3, input.offset3)",
            "textureGatherOffsets",
        ),
        (
            "vec4(textureCompareOffset(shadowMap, compareSampler, input.uv, input.depth, input.offset3))",
            "textureCompareOffset",
        ),
        (
            "vec4(textureCompareProjGradOffset(shadowMap, compareSampler, input.projCoord, input.depth, input.ddx, input.ddy, input.offset3))",
            "textureCompareProjGradOffset",
        ),
        (
            "textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depth, input.offset3)",
            "textureGatherCompareOffset",
        ),
    ],
)
def test_directx_rejects_wrong_offset_dimension_for_extended_resource_operation(
    call, operation
):
    shader = f"""
    shader InvalidExtendedResourceOffsetDimension {{
        sampler2DArray arrayMap;
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec3 uvLayer;
            vec3 projCoord;
            vec4 projLayer;
            vec2 ddx;
            vec2 ddy;
            float depth;
            ivec3 offset3;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX resource operation '{operation}' requires a 2D integer offset"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_rejects_float_offset_for_resource_operation():
    shader = """
    shader InvalidResourceOffsetType {
        sampler2D colorMap;

        struct FSInput {
            vec2 uv;
            vec2 offset;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return textureOffset(colorMap, input.uv, input.offset);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "DirectX resource operation 'textureOffset' requires an integer "
            "offset argument"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation"),
    [
        (
            "textureGatherOffset(colorMap, input.uv, input.offset)",
            "textureGatherOffset",
        ),
        (
            "textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depth, input.offset)",
            "textureGatherCompareOffset",
        ),
    ],
)
def test_directx_rejects_float_offset_for_extended_resource_operation(call, operation):
    shader = f"""
    shader InvalidExtendedResourceOffsetType {{
        sampler2D colorMap;
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec2 offset;
            float depth;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX resource operation '{operation}' requires an integer "
            "offset argument"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation", "dimension"),
    [
        ("textureGrad(colorMap, input.uv, input.ddx3, input.ddy3)", "textureGrad", 2),
        (
            "textureGrad(cubeMap, input.direction, input.ddx2, input.ddy2)",
            "textureGrad",
            3,
        ),
        (
            "textureGradOffset(colorMap, input.uv, input.ddx3, input.ddy3, input.offset2)",
            "textureGradOffset",
            2,
        ),
        (
            "textureProjGrad(colorMap, input.projCoord, input.ddx3, input.ddy3)",
            "textureProjGrad",
            2,
        ),
        (
            "vec4(textureCompareGrad(shadowMap, compareSampler, input.uv, input.depth, input.ddx3, input.ddy3))",
            "textureCompareGrad",
            2,
        ),
        (
            "vec4(textureCompareProjGradOffset(shadowMap, compareSampler, input.projCoord, input.depth, input.ddx3, input.ddy3, input.offset2))",
            "textureCompareProjGradOffset",
            2,
        ),
    ],
)
def test_directx_rejects_wrong_gradient_dimension_for_resource_operation(
    call, operation, dimension
):
    shader = f"""
    shader InvalidResourceGradientDimension {{
        sampler2D colorMap;
        samplerCube cubeMap;
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec3 direction;
            vec3 projCoord;
            vec2 ddx2;
            vec2 ddy2;
            vec3 ddx3;
            vec3 ddy3;
            ivec2 offset2;
            float depth;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX resource operation '{operation}' requires a "
            f"{dimension}D floating gradient"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_rejects_integer_gradient_for_resource_operation():
    shader = """
    shader InvalidResourceGradientType {
        sampler2D colorMap;

        struct FSInput {
            vec2 uv;
            ivec2 pixel;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return textureGrad(colorMap, input.uv, input.pixel, input.pixel);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "DirectX resource operation 'textureGrad' requires a floating "
            "gradient argument"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation", "type_name"),
    [
        (
            "textureLod(colorMap, linearSampler, input.uv, input.lodVec)",
            "textureLod",
            "float2",
        ),
        (
            "textureLodOffset(colorMap, linearSampler, input.uv, input.lodVec, input.offset)",
            "textureLodOffset",
            "float2",
        ),
        (
            "textureProjLod(colorMap, linearSampler, input.projCoord, input.lodVec)",
            "textureProjLod",
            "float2",
        ),
        (
            "vec4(textureCompareLod(shadowMap, compareSampler, input.uv, input.depth, input.lodVec))",
            "textureCompareLod",
            "float2",
        ),
        (
            "vec4(textureCompareProjLodOffset(shadowMap, compareSampler, input.projCoord, input.depth, input.lodVec, input.offset))",
            "textureCompareProjLodOffset",
            "float2",
        ),
        (
            "textureLod(colorMap, linearSampler, input.uv, colorMap)",
            "textureLod",
            "Texture2D",
        ),
    ],
)
def test_directx_rejects_non_scalar_numeric_lod_argument(call, operation, type_name):
    shader = f"""
    shader InvalidTextureLodArgument {{
        sampler2D colorMap;
        sampler2DShadow shadowMap;
        sampler linearSampler;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec3 projCoord;
            vec2 lodVec;
            ivec2 offset;
            float depth;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX texture LOD operation '{operation}' requires a scalar "
            f"numeric lod argument: .* has type {type_name}"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expr", "operation", "type_name"),
    [
        (
            "texelFetch(colorMap, input.pixel, input.floatLevel)",
            "texelFetch",
            "float",
        ),
        (
            "texelFetchOffset(colorMap, input.pixel, input.levelVec, input.offset)",
            "texelFetchOffset",
            "int2",
        ),
        (
            "vec4(textureSize(colorMap, input.floatLevel), 0.0, 1.0)",
            "textureSize",
            "float",
        ),
        (
            "texelFetch(colorMap, input.pixel, colorMap)",
            "texelFetch",
            "Texture2D",
        ),
    ],
)
def test_directx_rejects_non_scalar_integer_mip_level_argument(
    return_expr, operation, type_name
):
    shader = f"""
    shader InvalidTextureMipLevelArgument {{
        sampler2D colorMap;

        struct FSInput {{
            ivec2 pixel;
            ivec2 offset;
            ivec2 levelVec;
            float floatLevel;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expr};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX resource operation '{operation}' requires a scalar integer "
            f"mip/sample level argument: .* has type {type_name}"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expr", "type_name"),
    [
        ("texelFetch(msTex, input.pixel, input.floatSample)", "float"),
        ("texelFetch(msArray, input.pixelLayer, input.sampleVec)", "int2"),
        ("texelFetch(msTex, input.pixel, msTex)", "Texture2DMS<float4>"),
    ],
)
def test_directx_rejects_non_scalar_integer_multisample_sample_index(
    return_expr, type_name
):
    shader = f"""
    shader InvalidMultisampleTexelFetchSampleIndex {{
        sampler2DMS msTex;
        sampler2DMSArray msArray;

        struct FSInput {{
            ivec2 pixel;
            ivec3 pixelLayer;
            ivec2 sampleVec;
            float floatSample;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expr};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "DirectX multisample texel fetch operation 'texelFetch' requires a "
            f"scalar integer sample index argument: .* has type {type_name}"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expr", "type_name"),
    [
        ("textureSamplePosition(msTex, input.floatSample)", "float"),
        ("textureSamplePosition(msArray, input.sampleVec)", "int2"),
        ("textureSamplePosition(msTex, msTex)", "Texture2DMS<float4>"),
    ],
)
def test_directx_rejects_non_scalar_integer_texture_sample_position_index(
    return_expr, type_name
):
    shader = f"""
    shader InvalidTextureSamplePositionIndex {{
        sampler2DMS msTex;
        sampler2DMSArray msArray;

        struct FSInput {{
            ivec2 sampleVec;
            float floatSample;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return vec4({return_expr}, 0.0, 1.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "DirectX texture sample-position query operation "
            "'textureSamplePosition' requires a scalar integer sample index "
            f"argument: .* has type {type_name}"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expr", "helper", "operation", "type_name"),
    [
        (
            "textureGather(colorMap, linearSampler, input.uv, input.floatComponent)",
            "",
            "textureGather",
            "float",
        ),
        (
            "textureGatherOffset(colorMap, linearSampler, input.uv, input.offset, input.componentVec)",
            "",
            "textureGatherOffset",
            "int2",
        ),
        (
            "textureGatherOffsets(colorMap, linearSampler, input.uv, input.offset, input.offset, input.offset, input.offset, colorMap)",
            "",
            "textureGatherOffsets",
            "Texture2D",
        ),
        (
            "vec4(1.0)",
            "vec4 invalidArrayGather(sampler2D tex, sampler s, vec2 uv, ivec2 offsets[4], float component) { return textureGatherOffsets(tex, s, uv, offsets, component); }",
            "textureGatherOffsets",
            "float",
        ),
    ],
)
def test_directx_rejects_non_scalar_integer_gather_component_argument(
    return_expr, helper, operation, type_name
):
    shader = f"""
    shader InvalidTextureGatherComponentArgument {{
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {{
            vec2 uv;
            ivec2 offset;
            ivec2 componentVec;
            float floatComponent;
        }};

        {helper}

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expr};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX texture gather operation '{operation}' requires a scalar "
            f"integer component argument: .* has type {type_name}"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("return_expr", "operation", "type_name"),
    [
        (
            "texture(colorMap, linearSampler, input.uv, input.biasVec)",
            "texture",
            "float2",
        ),
        (
            "textureOffset(colorMap, linearSampler, input.uv, input.offset, input.biasVec)",
            "textureOffset",
            "float2",
        ),
        (
            "textureProj(colorMap, linearSampler, input.projCoord, input.biasVec)",
            "textureProj",
            "float2",
        ),
        (
            "textureProjOffset(colorMap, linearSampler, input.projCoord, input.offset, colorMap)",
            "textureProjOffset",
            "Texture2D",
        ),
    ],
)
def test_directx_rejects_non_scalar_numeric_bias_argument(
    return_expr, operation, type_name
):
    shader = f"""
    shader InvalidTextureBiasArgument {{
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {{
            vec2 uv;
            vec3 projCoord;
            vec2 biasVec;
            ivec2 offset;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {return_expr};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX texture bias operation '{operation}' requires a scalar "
            f"numeric bias argument: .* has type {type_name}"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_texture_bias_variants_use_sample_bias():
    shader = """
    shader TextureBiasVariants {
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv;
            ivec2 offset;
            float bias;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 biased = texture(colorMap, linearSampler, input.uv, input.bias);
                vec4 offsetBiased = textureOffset(
                    colorMap,
                    linearSampler,
                    input.uv,
                    input.offset,
                    input.bias
                );
                return biased + offsetBiased;
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "colorMap.SampleBias(linearSampler, input.uv, input.bias)" in generated_code
    assert (
        "colorMap.SampleBias(linearSampler, input.uv, input.bias, input.offset)"
        in generated_code
    )
    assert "colorMap.Sample(linearSampler, input.uv, input.bias)" not in generated_code


def test_directx_rejects_non_floating_query_lod_coordinate():
    shader = """
    shader InvalidTextureQueryLodCoordinate {
        sampler2D colorMap;
        sampler linearSampler;

        struct FSInput {
            ivec2 pixel;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return vec4(textureQueryLod(colorMap, linearSampler, input.pixel), 0.0, 1.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "DirectX texture query operation 'textureQueryLod' requires a "
            "floating coordinate argument: .* has type int2"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("query_call", "dimension", "resource_type", "type_name"),
    [
        (
            "textureQueryLod(layerMap, linearSampler, input.uv)",
            3,
            "Texture2DArray",
            "float2",
        ),
        (
            "textureQueryLod(cubeMap, linearSampler, input.uv)",
            3,
            "TextureCube",
            "float2",
        ),
        (
            "textureQueryLod(cubeArray, linearSampler, input.direction)",
            4,
            "TextureCubeArray",
            "float3",
        ),
    ],
)
def test_directx_rejects_wrong_query_lod_coordinate_dimension(
    query_call, dimension, resource_type, type_name
):
    shader = f"""
    shader InvalidTextureQueryLodCoordinateDimension {{
        sampler2DArray layerMap;
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler linearSampler;

        struct FSInput {{
            vec2 uv;
            vec3 direction;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return vec4({query_call}, 0.0, 1.0);
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            "DirectX texture query operation 'textureQueryLod' requires a "
            f"{dimension}D floating coordinate for {resource_type}: "
            f".* has type {type_name}"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("call", "operation", "type_name"),
    [
        (
            "vec4(textureCompare(shadowMap, compareSampler, input.uv, input.depthVec))",
            "textureCompare",
            "float2",
        ),
        (
            "vec4(textureCompareProjGradOffset(shadowMap, compareSampler, input.projCoord, input.depthVec, input.ddx, input.ddy, input.offset))",
            "textureCompareProjGradOffset",
            "float2",
        ),
        (
            "textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depthVec, input.offset)",
            "textureGatherCompareOffset",
            "float2",
        ),
        (
            "vec4(textureCompare(shadowMap, compareSampler, input.uv, input.layer))",
            "textureCompare",
            "int",
        ),
    ],
)
def test_directx_rejects_non_scalar_float_compare_argument(call, operation, type_name):
    shader = f"""
    shader InvalidTextureCompareArgument {{
        sampler2DShadow shadowMap;
        sampler compareSampler;

        struct FSInput {{
            vec2 uv;
            vec3 projCoord;
            vec2 depthVec;
            vec2 ddx;
            vec2 ddy;
            ivec2 offset;
            int layer;
        }};

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {call};
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=(
            f"DirectX texture compare operation '{operation}' requires a scalar "
            f"floating compare argument: .* has type {type_name}"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_storage_image_load_store():
    shader = """
    shader StorageImages {
        image2D outputImage;
        image3D volumeImage;
        image2DArray layerImage;

        vec4 touchImages(image2D outImg, image3D volume, image2DArray layers, ivec2 pixel, ivec3 voxel, ivec3 pixelLayer) {
            vec4 color = imageLoad(outImg, pixel);
            vec4 volumeColor = imageLoad(volume, voxel);
            vec4 layerColor = imageLoad(layers, pixelLayer);
            imageStore(outImg, pixel, color + layerColor);
            imageStore(volume, voxel, volumeColor);
            imageStore(layers, pixelLayer, color);
            return color + volumeColor + layerColor;
        }

        compute {
            void main() {
                vec4 result = touchImages(outputImage, volumeImage, layerImage, ivec2(0, 1), ivec3(0, 1, 2), ivec3(3, 4, 5));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float4> outputImage : register(u0);" in generated_code
    assert "RWTexture3D<float4> volumeImage : register(u1);" in generated_code
    assert "RWTexture2DArray<float4> layerImage : register(u2);" in generated_code
    assert (
        "float4 touchImages(RWTexture2D<float4> outImg, RWTexture3D<float4> volume, RWTexture2DArray<float4> layers, int2 pixel, int3 voxel, int3 pixelLayer)"
        in generated_code
    )
    assert "float4 color = outImg[pixel];" in generated_code
    assert "float4 volumeColor = volume[voxel];" in generated_code
    assert "float4 layerColor = layers[pixelLayer];" in generated_code
    assert "outImg[pixel] = (color + layerColor);" in generated_code
    assert "volume[voxel] = volumeColor;" in generated_code
    assert "layers[pixelLayer] = color;" in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_direct_stage_image_load_store_and_atomics_use_input_members():
    shader = """
    shader DirectStageImageOps {
        image2D outImg;
        image3D volume;
        image2DArray layers;
        uimage2D counters;
        iimage2DArray signedLayers;

        struct FSInput {
            ivec2 pixel @ TEXCOORD0;
            ivec3 voxel @ TEXCOORD1;
            ivec3 pixelLayer @ TEXCOORD2;
            uint amount;
            int signedAmount;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 color = imageLoad(outImg, input.pixel);
                vec4 volumeColor = imageLoad(volume, input.voxel);
                vec4 layerColor = imageLoad(layers, input.pixelLayer);
                imageStore(outImg, input.pixel, color + layerColor);
                imageStore(volume, input.voxel, volumeColor);
                imageStore(layers, input.pixelLayer, color);
                uint previous = imageAtomicAdd(counters, input.pixel, input.amount);
                int previousSigned = imageAtomicMin(signedLayers, input.pixelLayer, input.signedAmount);
                return color + volumeColor + layerColor + vec4(float(previous + uint(previousSigned)));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float4> outImg : register(u0);" in generated_code
    assert "RWTexture3D<float4> volume : register(u1);" in generated_code
    assert "RWTexture2DArray<float4> layers : register(u2);" in generated_code
    assert "RWTexture2D<uint> counters : register(u3);" in generated_code
    assert "RWTexture2DArray<int> signedLayers : register(u4);" in generated_code
    assert (
        "uint imageAtomicAdd_uimage2D(RWTexture2D<uint> image, int2 coord, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicMin_iimage2DArray(RWTexture2DArray<int> image, int3 coord, int value)"
        in generated_code
    )
    assert "float4 color = outImg[input.pixel];" in generated_code
    assert "float4 volumeColor = volume[input.voxel];" in generated_code
    assert "float4 layerColor = layers[input.pixelLayer];" in generated_code
    assert "outImg[input.pixel] = (color + layerColor);" in generated_code
    assert "volume[input.voxel] = volumeColor;" in generated_code
    assert "layers[input.pixelLayer] = color;" in generated_code
    assert (
        "uint previous = imageAtomicAdd_uimage2D(counters, input.pixel, input.amount);"
        in generated_code
    )
    assert (
        "int previousSigned = imageAtomicMin_iimage2DArray(signedLayers, input.pixelLayer, input.signedAmount);"
        in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code
    assert "imageAtomicAdd(counters" not in generated_code
    assert "imageAtomicMin(signedLayers" not in generated_code


def test_directx_image_atomic_helper_descriptor_for_integer_storage_textures():
    codegen = HLSLCodeGen()

    uint_descriptor = codegen.image_atomic_helper_descriptor(
        "imageAtomicAdd", "RWTexture1DArray<uint>"
    )
    assert uint_descriptor == {
        "helper_name": "imageAtomicAdd_uimage1DArray",
        "return_type": "uint",
        "coord_type": "int2",
        "intrinsic": "InterlockedAdd",
    }

    int_descriptor = codegen.image_atomic_helper_descriptor(
        "imageAtomicCompSwap", "RWTexture2DArray<int>"
    )
    assert int_descriptor == {
        "helper_name": "imageAtomicCompSwap_iimage2DArray",
        "return_type": "int",
        "coord_type": "int3",
        "intrinsic": "InterlockedCompareExchange",
    }

    assert (
        codegen.image_atomic_helper_descriptor("imageAtomicAdd", "RWTexture2D<float4>")
        is None
    )
    assert (
        codegen.image_atomic_helper_descriptor(
            "imageAtomicUnsupported", "RWTexture2D<uint>"
        )
        is None
    )


def test_directx_direct_stage_explicit_image_formats_use_input_members():
    shader = """
    shader DirectStageImageFormats {
        image2D scalarFloat @r32f;
        image2D rgFloat @rg32f;
        uimage2D unsignedScalar @r32ui;
        uimage2DArray unsignedLayers @rg32ui;
        iimage3D signedVolume @r32i;

        struct FSInput {
            ivec2 pixel @ TEXCOORD0;
            ivec3 voxel @ TEXCOORD1;
            ivec3 pixelLayer @ TEXCOORD2;
            float amount;
            uint unsignedAmount;
            int signedAmount;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float scalarValue = imageLoad(scalarFloat, input.pixel);
                vec2 rgValue = imageLoad(rgFloat, input.pixel);
                uint unsignedValue = imageLoad(unsignedScalar, input.pixel);
                uvec2 layerValue = imageLoad(unsignedLayers, input.pixelLayer);
                int signedValue = imageLoad(signedVolume, input.voxel);
                imageStore(scalarFloat, input.pixel, scalarValue + input.amount);
                imageStore(rgFloat, input.pixel, rgValue + vec2(input.amount));
                imageStore(unsignedScalar, input.pixel, unsignedValue + input.unsignedAmount);
                imageStore(unsignedLayers, input.pixelLayer, layerValue + uvec2(input.unsignedAmount));
                imageStore(signedVolume, input.voxel, signedValue + input.signedAmount);
                return vec4(scalarValue + rgValue.x + float(unsignedValue + layerValue.x) + float(signedValue));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float> scalarFloat : register(u0);" in generated_code
    assert "RWTexture2D<float2> rgFloat : register(u1);" in generated_code
    assert "RWTexture2D<uint> unsignedScalar : register(u2);" in generated_code
    assert "RWTexture2DArray<uint2> unsignedLayers : register(u3);" in generated_code
    assert "RWTexture3D<int> signedVolume : register(u4);" in generated_code
    assert "float scalarValue = scalarFloat[input.pixel];" in generated_code
    assert "float2 rgValue = rgFloat[input.pixel];" in generated_code
    assert "uint unsignedValue = unsignedScalar[input.pixel];" in generated_code
    assert "uint2 layerValue = unsignedLayers[input.pixelLayer];" in generated_code
    assert "int signedValue = signedVolume[input.voxel];" in generated_code
    assert "scalarFloat[input.pixel] = (scalarValue + input.amount);" in generated_code
    assert (
        "rgFloat[input.pixel] = (rgValue + float2(input.amount, input.amount));"
        in generated_code
    )
    assert (
        "unsignedScalar[input.pixel] = (unsignedValue + input.unsignedAmount);"
        in generated_code
    )
    assert (
        "unsignedLayers[input.pixelLayer] = "
        "(layerValue + uint2(input.unsignedAmount, input.unsignedAmount));"
        in generated_code
    )
    assert (
        "signedVolume[input.voxel] = (signedValue + input.signedAmount);"
        in generated_code
    )
    assert "RWTexture2D<float4> scalarFloat" not in generated_code
    assert "RWTexture2D<float4> rgFloat" not in generated_code
    assert "RWTexture2DArray<uint4> unsignedLayers" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_direct_stage_image_compare_swap_use_input_members():
    shader = """
    shader DirectStageImageCompareSwap {
        uimage3D volumeCounters;
        iimage2DArray layerCounters;

        struct FSInput {
            ivec3 voxel @ TEXCOORD0;
            ivec3 pixelLayer @ TEXCOORD1;
            uint expected;
            uint replacement;
            int signedExpected;
            int signedReplacement;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                uint volumeOld = imageAtomicCompSwap(volumeCounters, input.voxel, input.expected, input.replacement);
                int layerOld = imageAtomicCompSwap(layerCounters, input.pixelLayer, input.signedExpected, input.signedReplacement);
                return vec4(float(volumeOld), float(layerOld), 0.0, 1.0);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture3D<uint> volumeCounters : register(u0);" in generated_code
    assert "RWTexture2DArray<int> layerCounters : register(u1);" in generated_code
    assert (
        "uint imageAtomicCompSwap_uimage3D(RWTexture3D<uint> image, int3 coord, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicCompSwap_iimage2DArray(RWTexture2DArray<int> image, int3 coord, int compareValue, int value)"
        in generated_code
    )
    assert (
        "InterlockedCompareExchange(image[coord], compareValue, value, original);"
        in generated_code
    )
    assert (
        "uint volumeOld = imageAtomicCompSwap_uimage3D(volumeCounters, input.voxel, input.expected, input.replacement);"
        in generated_code
    )
    assert (
        "int layerOld = imageAtomicCompSwap_iimage2DArray(layerCounters, input.pixelLayer, input.signedExpected, input.signedReplacement);"
        in generated_code
    )
    assert "imageAtomicCompSwap(volumeCounters" not in generated_code
    assert "imageAtomicCompSwap(layerCounters" not in generated_code


def test_directx_integer_image_atomic_add():
    shader = """
    shader AtomicImages {
        uimage2D counters;
        iimage2D signedCounters;

        uint addCounter(uimage2D image, ivec2 pixel, uint value) {
            uint previous = imageAtomicAdd(image, pixel, value);
            return previous;
        }

        int addSignedCounter(iimage2D image, ivec2 pixel, int value) {
            int previous = imageAtomicAdd(image, pixel, value);
            return previous;
        }

        compute {
            void main() {
                uint oldValue = addCounter(counters, ivec2(0, 1), 2);
                int oldSigned = addSignedCounter(signedCounters, ivec2(2, 3), -1);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<uint> counters : register(u0);" in generated_code
    assert "RWTexture2D<int> signedCounters : register(u1);" in generated_code
    assert (
        "uint imageAtomicAdd_uimage2D(RWTexture2D<uint> image, int2 coord, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicAdd_iimage2D(RWTexture2D<int> image, int2 coord, int value)"
        in generated_code
    )
    assert "InterlockedAdd(image[coord], value, original);" in generated_code
    assert (
        "uint addCounter(RWTexture2D<uint> image, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "int addSignedCounter(RWTexture2D<int> image, int2 pixel, int value)"
        in generated_code
    )
    assert (
        "uint previous = imageAtomicAdd_uimage2D(image, pixel, value);"
        in generated_code
    )
    assert (
        "int previous = imageAtomicAdd_iimage2D(image, pixel, value);" in generated_code
    )
    assert "imageAtomicAdd(image" not in generated_code


def test_directx_integer_image_atomic_operations():
    shader = """
    shader AtomicOps {
        uimage2D counters;
        iimage2D signedCounters;

        uint unsignedOps(uimage2D image, ivec2 pixel, uint value) {
            uint minValue = imageAtomicMin(image, pixel, value);
            uint maxValue = imageAtomicMax(image, pixel, value);
            uint andValue = imageAtomicAnd(image, pixel, value);
            uint orValue = imageAtomicOr(image, pixel, value);
            uint xorValue = imageAtomicXor(image, pixel, value);
            uint exchanged = imageAtomicExchange(image, pixel, value);
            return minValue + maxValue + andValue + orValue + xorValue + exchanged;
        }

        int signedOps(iimage2D image, ivec2 pixel, int value) {
            int minValue = imageAtomicMin(image, pixel, value);
            int maxValue = imageAtomicMax(image, pixel, value);
            int exchanged = imageAtomicExchange(image, pixel, value);
            return minValue + maxValue + exchanged;
        }

        compute {
            void main() {
                uint unsignedResult = unsignedOps(counters, ivec2(0, 1), 3);
                int signedResult = signedOps(signedCounters, ivec2(2, 3), -4);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    for intrinsic in [
        "InterlockedMin",
        "InterlockedMax",
        "InterlockedAnd",
        "InterlockedOr",
        "InterlockedXor",
        "InterlockedExchange",
    ]:
        assert f"{intrinsic}(image[coord], value, original);" in generated_code

    for operation in [
        "imageAtomicMin",
        "imageAtomicMax",
        "imageAtomicAnd",
        "imageAtomicOr",
        "imageAtomicXor",
        "imageAtomicExchange",
    ]:
        assert f"{operation}_uimage2D(RWTexture2D<uint> image" in generated_code
        assert f"{operation}_uimage2D(image, pixel, value)" in generated_code
        assert f"{operation}(image" not in generated_code

    for operation in ["imageAtomicMin", "imageAtomicMax", "imageAtomicExchange"]:
        assert f"{operation}_iimage2D(RWTexture2D<int> image" in generated_code
        assert f"{operation}_iimage2D(image, pixel, value)" in generated_code


def test_directx_integer_image_atomic_compare_swap():
    shader = """
    shader AtomicCompareSwap {
        uimage2D counters;
        iimage2D signedCounters;

        uint compareUnsigned(uimage2D image, ivec2 pixel, uint expected, uint replacement) {
            uint previous = imageAtomicCompSwap(image, pixel, expected, replacement);
            return previous;
        }

        int compareSigned(iimage2D image, ivec2 pixel, int expected, int replacement) {
            int previous = imageAtomicCompSwap(image, pixel, expected, replacement);
            return previous;
        }

        compute {
            void main() {
                uint oldValue = compareUnsigned(counters, ivec2(0, 1), 2, 3);
                int oldSigned = compareSigned(signedCounters, ivec2(2, 3), -1, 4);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert (
        "uint imageAtomicCompSwap_uimage2D(RWTexture2D<uint> image, int2 coord, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicCompSwap_iimage2D(RWTexture2D<int> image, int2 coord, int compareValue, int value)"
        in generated_code
    )
    assert (
        "InterlockedCompareExchange(image[coord], compareValue, value, original);"
        in generated_code
    )
    assert (
        "uint previous = imageAtomicCompSwap_uimage2D(image, pixel, expected, replacement);"
        in generated_code
    )
    assert (
        "int previous = imageAtomicCompSwap_iimage2D(image, pixel, expected, replacement);"
        in generated_code
    )
    assert "imageAtomicCompSwap(image" not in generated_code


def test_directx_image_atomic_result_type_drives_nested_expression_codegen():
    shader = """
    shader AtomicImageExpressionTyping {
        uimage2D counters;

        uint consume(uint value) {
            return value + 1u;
        }

        uint touch(uimage2D image, ivec2 pixel, uint value, bool choose) {
            uint2 splat = uint2(imageAtomicAdd(image, pixel, value));
            uint callValue = consume(imageAtomicCompSwap(image, pixel, value, splat.x));
            uint selected = choose
                ? imageAtomicMin(image, pixel, callValue)
                : imageAtomicMax(image, pixel, value);
            return splat.y + callValue + selected;
        }

        compute {
            void main() {
                uint result = touch(counters, ivec2(0, 1), 2u, true);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert (
        "uint2 splat = ((uint2)(imageAtomicAdd_uimage2D(image, pixel, value)));"
        in generated_code
    )
    assert (
        "uint callValue = consume(imageAtomicCompSwap_uimage2D(image, pixel, value, splat.x));"
        in generated_code
    )
    assert (
        "uint selected = (choose ? imageAtomicMin_uimage2D(image, pixel, callValue) : imageAtomicMax_uimage2D(image, pixel, value));"
        in generated_code
    )
    assert "imageAtomicAdd(image" not in generated_code
    assert "imageAtomicCompSwap(image" not in generated_code
    assert "imageAtomicMin(image" not in generated_code
    assert "imageAtomicMax(image" not in generated_code


def test_directx_integer_image_dimension_atomics():
    shader = """
    shader TypedImageDimensions {
        uimage3D volumeCounters;
        iimage3D signedVolumeCounters;
        iimage2DArray layerCounters;
        uimage2DArray unsignedLayerCounters;

        uint touchVolume(uimage3D image, ivec3 voxel, uint value) {
            uint oldValue = imageAtomicAdd(image, voxel, value);
            uint swapped = imageAtomicCompSwap(image, voxel, oldValue, value);
            return oldValue + swapped;
        }

        int touchLayers(iimage2DArray image, ivec3 pixelLayer, int value) {
            int oldValue = imageAtomicMin(image, pixelLayer, value);
            int swapped = imageAtomicCompSwap(image, pixelLayer, oldValue, value);
            return oldValue + swapped;
        }

        compute {
            void main() {
                uint volumeResult = touchVolume(volumeCounters, ivec3(0, 1, 2), 3);
                int layerResult = touchLayers(layerCounters, ivec3(4, 5, 6), -7);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture3D<uint> volumeCounters : register(u0);" in generated_code
    assert "RWTexture3D<int> signedVolumeCounters : register(u1);" in generated_code
    assert "RWTexture2DArray<int> layerCounters : register(u2);" in generated_code
    assert (
        "RWTexture2DArray<uint> unsignedLayerCounters : register(u3);" in generated_code
    )
    assert (
        "uint imageAtomicAdd_uimage3D(RWTexture3D<uint> image, int3 coord, uint value)"
        in generated_code
    )
    assert (
        "uint imageAtomicCompSwap_uimage3D(RWTexture3D<uint> image, int3 coord, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicMin_iimage2DArray(RWTexture2DArray<int> image, int3 coord, int value)"
        in generated_code
    )
    assert (
        "int imageAtomicCompSwap_iimage2DArray(RWTexture2DArray<int> image, int3 coord, int compareValue, int value)"
        in generated_code
    )
    assert (
        "uint oldValue = imageAtomicAdd_uimage3D(image, voxel, value);"
        in generated_code
    )
    assert (
        "uint swapped = imageAtomicCompSwap_uimage3D(image, voxel, oldValue, value);"
        in generated_code
    )
    assert (
        "int oldValue = imageAtomicMin_iimage2DArray(image, pixelLayer, value);"
        in generated_code
    )
    assert (
        "int swapped = imageAtomicCompSwap_iimage2DArray(image, pixelLayer, oldValue, value);"
        in generated_code
    )
    assert "imageAtomicAdd(image" not in generated_code
    assert "imageAtomicMin(image" not in generated_code
    assert "imageAtomicCompSwap(image" not in generated_code


def test_directx_integer_image_scalar_load_store():
    shader = """
    shader IntegerImageLoadStore {
        uimage2D counters;
        iimage3D signedVolume;
        uimage2DArray layerCounters;

        uint touch2D(uimage2D image, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        int touch3D(iimage3D image, ivec3 voxel, int value) {
            int oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uint touchLayer(uimage2DArray image, ivec3 pixelLayer, uint value) {
            uint oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touch2D(counters, ivec2(0, 1), 2);
                int b = touch3D(signedVolume, ivec3(1, 2, 3), -4);
                uint c = touchLayer(layerCounters, ivec3(4, 5, 6), 7);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<uint> counters : register(u0);" in generated_code
    assert "RWTexture3D<int> signedVolume : register(u1);" in generated_code
    assert "RWTexture2DArray<uint> layerCounters : register(u2);" in generated_code
    assert "uint oldValue = image[pixel];" in generated_code
    assert "image[pixel] = (oldValue + value);" in generated_code
    assert "int oldValue = image[voxel];" in generated_code
    assert "image[voxel] = (oldValue + value);" in generated_code
    assert "uint oldValue = image[pixelLayer];" in generated_code
    assert "image[pixelLayer] = (oldValue + value);" in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_explicit_scalar_image_formats():
    shader = """
    shader ExplicitScalarImageFormats {
        image2D scalarFloat @r32f;
        image3D signedVolume @ r32i;
        image2DArray unsignedLayers @format(r32ui);

        float touchFloat(image2D image @r32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        int touchSigned(image3D image @r32i, ivec3 voxel, int value) {
            int oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uint touchUnsigned(image2DArray image @format(r32ui), ivec3 pixelLayer, uint value) {
            uint oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float a = touchFloat(scalarFloat, ivec2(0, 1), 0.5);
                int b = touchSigned(signedVolume, ivec3(1, 2, 3), -4);
                uint c = touchUnsigned(unsignedLayers, ivec3(4, 5, 6), 7);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float> scalarFloat : register(u0);" in generated_code
    assert "RWTexture3D<int> signedVolume : register(u1);" in generated_code
    assert "RWTexture2DArray<uint> unsignedLayers : register(u2);" in generated_code
    assert (
        "float touchFloat(RWTexture2D<float> image, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "int touchSigned(RWTexture3D<int> image, int3 voxel, int value)"
        in generated_code
    )
    assert (
        "uint touchUnsigned(RWTexture2DArray<uint> image, int3 pixelLayer, uint value)"
        in generated_code
    )
    assert "float oldValue = image[pixel];" in generated_code
    assert "image[pixel] = (oldValue + value);" in generated_code
    assert "int oldValue = image[voxel];" in generated_code
    assert "image[voxel] = (oldValue + value);" in generated_code
    assert "uint oldValue = image[pixelLayer];" in generated_code
    assert "image[pixelLayer] = (oldValue + value);" in generated_code
    assert ": r32" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_explicit_rg_image_formats():
    shader = """
    shader ExplicitRGImageFormats {
        image2D rgFloat @rg32f;
        image3D rgSigned @format(rg32i);
        image2DArray rgUnsigned @ rg32ui;

        vec2 touchFloat(image2D image @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        ivec2 touchSigned(image3D image @rg32i, ivec3 voxel, ivec2 value) {
            ivec2 oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uvec2 touchUnsigned(image2DArray image @rg32ui, ivec3 pixelLayer, uvec2 value) {
            uvec2 oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float2> rgFloat : register(u0);" in generated_code
    assert "RWTexture3D<int2> rgSigned : register(u1);" in generated_code
    assert "RWTexture2DArray<uint2> rgUnsigned : register(u2);" in generated_code
    assert (
        "float2 touchFloat(RWTexture2D<float2> image, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "int2 touchSigned(RWTexture3D<int2> image, int3 voxel, int2 value)"
        in generated_code
    )
    assert (
        "uint2 touchUnsigned(RWTexture2DArray<uint2> image, int3 pixelLayer, uint2 value)"
        in generated_code
    )
    assert "float2 oldValue = image[pixel];" in generated_code
    assert "int2 oldValue = image[voxel];" in generated_code
    assert "uint2 oldValue = image[pixelLayer];" in generated_code
    assert "image[pixel] = (oldValue + value);" in generated_code
    assert "image[voxel] = (oldValue + value);" in generated_code
    assert "image[pixelLayer] = (oldValue + value);" in generated_code
    assert "RWTexture2D<float4> rgFloat" not in generated_code
    assert "RWTexture3D<float4> rgSigned" not in generated_code
    assert "RWTexture2DArray<float4> rgUnsigned" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_explicit_narrow_rg_image_formats():
    shader = """
    shader ExplicitNarrowRGImageFormats {
        image2D rg8Float @rg8;
        image2D rg8Snorm @rg8_snorm;
        image3D rg16Float @format(rg16);
        image2D rg16Snorm @rg16_snorm;
        image2DArray rg16Half @ rg16f;
        image2D rg8Signed @rg8i;
        image3D rg16Signed @format(rg16i);
        image2D rg8Unsigned @rg8ui;
        image2DArray rg16Unsigned @format(rg16ui);

        vec2 touchFloat(image2D image @rg8, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchHalf(image2DArray image @rg16f, ivec3 pixelLayer, vec2 value) {
            vec2 oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }

        ivec2 touchSigned(image3D image @rg16i, ivec3 voxel, ivec2 value) {
            ivec2 oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uvec2 touchUnsigned(image2D image @rg8ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float2> rg8Float : register(u0);" in generated_code
    assert "RWTexture2D<float2> rg8Snorm : register(u1);" in generated_code
    assert "RWTexture3D<float2> rg16Float : register(u2);" in generated_code
    assert "RWTexture2D<float2> rg16Snorm : register(u3);" in generated_code
    assert "RWTexture2DArray<float2> rg16Half : register(u4);" in generated_code
    assert "RWTexture2D<int2> rg8Signed : register(u5);" in generated_code
    assert "RWTexture3D<int2> rg16Signed : register(u6);" in generated_code
    assert "RWTexture2D<uint2> rg8Unsigned : register(u7);" in generated_code
    assert "RWTexture2DArray<uint2> rg16Unsigned : register(u8);" in generated_code
    assert (
        "float2 touchFloat(RWTexture2D<float2> image, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float2 touchHalf(RWTexture2DArray<float2> image, int3 pixelLayer, float2 value)"
        in generated_code
    )
    assert (
        "int2 touchSigned(RWTexture3D<int2> image, int3 voxel, int2 value)"
        in generated_code
    )
    assert (
        "uint2 touchUnsigned(RWTexture2D<uint2> image, int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float2 oldValue = image[pixel];" in generated_code
    assert "float2 oldValue = image[pixelLayer];" in generated_code
    assert "int2 oldValue = image[voxel];" in generated_code
    assert "uint2 oldValue = image[pixel];" in generated_code
    assert "RWTexture2D<float4> rg8Float" not in generated_code
    assert "RWTexture3D<float4> rg16Float" not in generated_code
    assert "RWTexture2D<float4> rg8Signed" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_explicit_rgba_float_image_formats():
    shader = """
    shader ExplicitRGBAFloatFormats {
        image2D rgba8Color @rgba8;
        image2D rgba8Snorm @rgba8_snorm;
        image3D rgba16Color @format(rgba16);
        image2D rgba16Snorm @rgba16_snorm;
        image2DArray rgba16Half @ rgba16f;
        image3D rgba32Float @format(rgba32f);

        vec4 touchColor(image2D image @rgba8, ivec2 pixel, vec4 value) {
            vec4 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        vec4 touchHalf(image2DArray image @rgba16f, ivec3 pixelLayer, vec4 value) {
            vec4 oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }

        vec4 touchFloat(image3D image @rgba32f, ivec3 voxel, vec4 value) {
            vec4 oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        vec4 typedOverride(iimage2D image @rgba16f, ivec2 pixel, vec4 value) {
            vec4 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float4> rgba8Color : register(u0);" in generated_code
    assert "RWTexture2D<float4> rgba8Snorm : register(u1);" in generated_code
    assert "RWTexture3D<float4> rgba16Color : register(u2);" in generated_code
    assert "RWTexture2D<float4> rgba16Snorm : register(u3);" in generated_code
    assert "RWTexture2DArray<float4> rgba16Half : register(u4);" in generated_code
    assert "RWTexture3D<float4> rgba32Float : register(u5);" in generated_code
    assert (
        "float4 touchColor(RWTexture2D<float4> image, int2 pixel, float4 value)"
        in generated_code
    )
    assert (
        "float4 touchHalf(RWTexture2DArray<float4> image, int3 pixelLayer, float4 value)"
        in generated_code
    )
    assert (
        "float4 touchFloat(RWTexture3D<float4> image, int3 voxel, float4 value)"
        in generated_code
    )
    assert (
        "float4 typedOverride(RWTexture2D<float4> image, int2 pixel, float4 value)"
        in generated_code
    )
    assert "float4 oldValue = image[pixel];" in generated_code
    assert "float4 oldValue = image[pixelLayer];" in generated_code
    assert "float4 oldValue = image[voxel];" in generated_code
    assert "image[pixel] = (oldValue + value);" in generated_code
    assert "image[pixelLayer] = (oldValue + value);" in generated_code
    assert "image[voxel] = (oldValue + value);" in generated_code
    assert "RWTexture2D<int> image" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_formatted_image_arrays_preserve_format_metadata():
    shader = """
    shader FormattedImageArrays {
        image2D counters @r32ui[2];
        image2D rgPairs @rg16f[3];
        image3D rgbaVolumes @rgba16f[2];
        image2D afterCounters @r32ui;
        sampler2D sampled;

        uint touchCounters(image2D images[2] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[1], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[3] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec4 touchVolumes(image3D images[2] @rgba16f, ivec3 voxel, vec4 value) {
            vec4 oldValue = imageLoad(images[1], voxel);
            imageStore(images[0], voxel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
                vec2 b = touchPairs(rgPairs, ivec2(2, 3), vec2(0.5));
                vec4 c = touchVolumes(rgbaVolumes, ivec3(1, 2, 3), vec4(1.0));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<uint> counters[2] : register(u0);" in generated_code
    assert "RWTexture2D<float2> rgPairs[3] : register(u2);" in generated_code
    assert "RWTexture3D<float4> rgbaVolumes[2] : register(u5);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u7);" in generated_code
    assert "Texture2D sampled : register(t0);" in generated_code
    assert "SamplerState sampledSampler" not in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[2], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(RWTexture2D<float2> images[3], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float4 touchVolumes(RWTexture3D<float4> images[2], int3 voxel, float4 value)"
        in generated_code
    )
    assert "uint oldValue = images[1][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "images[1][pixel] = (oldValue + value);" in generated_code
    assert "float4 oldValue = images[1][voxel];" in generated_code
    assert "images[0][voxel] = (oldValue + value);" in generated_code
    assert "RWTexture2D<float4> counters" not in generated_code
    assert "RWTexture2D<float4> rgPairs" not in generated_code
    assert "RWTexture3D<int" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_resource_arrays_account_register_spaces_independently():
    shader = """
    shader RegisterSpaceResourceArrays {
        sampler2D colorMaps[2] @register(t0, space1);
        sampler colorSamplers[2] @register(s0, space1);
        sampler2D spaceTwoMaps[3] @register(t0, space2);
        image2D counters @r32ui @register(u0, space2)[3];
        sampler2D afterSpaceOne @space(space1);
        sampler2D afterSpaceTwo @space(space2);
        image2D afterCounter @r32ui @space(space2);

        struct FSInput {
            vec2 uv;
            ivec2 pixel;
            int layer;
        };

        uint touchCounters(image2D images[3] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 explicitSample = texture(colorMaps[input.layer], colorSamplers[input.layer], input.uv);
                vec4 implicitSample = texture(spaceTwoMaps[0], input.uv);
                uint oldValue = touchCounters(counters, input.pixel, 1u);
                return explicitSample + implicitSample + vec4(float(oldValue));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D colorMaps[2] : register(t0, space1);" in generated_code
    assert "SamplerState colorSamplers[2] : register(s0, space1);" in generated_code
    assert "Texture2D spaceTwoMaps[3] : register(t0, space2);" in generated_code
    assert "SamplerState spaceTwoMapsSampler : register(s0, space2);" in generated_code
    assert "RWTexture2D<uint> counters[3] : register(u0, space2);" in generated_code
    assert "Texture2D afterSpaceOne : register(t2, space1);" in generated_code
    assert "Texture2D afterSpaceTwo : register(t3, space2);" in generated_code
    assert "RWTexture2D<uint> afterCounter : register(u3, space2);" in generated_code
    assert (
        "colorMaps[input.layer].Sample(colorSamplers[input.layer], input.uv)"
        in generated_code
    )
    assert "spaceTwoMaps[0].Sample(spaceTwoMapsSampler, input.uv)" in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[3], int2 pixel, uint value)"
        in generated_code
    )
    assert "uint oldValue = images[2][pixel];" in generated_code
    assert "images[1][pixel] = (oldValue + value);" in generated_code
    assert "Texture2D afterSpaceOne : register(t2, space2);" not in generated_code
    assert "Texture2D afterSpaceTwo : register(t3, space1);" not in generated_code
    assert (
        "RWTexture2D<uint> afterCounter : register(u1, space2);" not in generated_code
    )
    assert "colorMapsSampler" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_resource_array_register_space_conflicts_use_full_ranges():
    shader = """
    shader ConflictingResourceArraySpaces {
        sampler2D first[3] @register(t2, space1);
        sampler2D second[2] @register(t4, space1);

        fragment {
            vec4 main() @ gl_FragColor {
                return vec4(0.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Conflicting DirectX resource binding for 'second': "
            "t4-t5, space1 overlaps 'first' t2-t4, space1"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_register_space0_aliases_default_resource_space():
    conflict_shader = """
    shader RegisterSpaceZeroConflict {
        sampler2D implicitSpace @register(t0);
        sampler2D explicitSpaceZero @register(t0, space0);

        fragment {
            vec4 main() @ gl_FragColor {
                return vec4(0.0);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Conflicting DirectX resource binding for 'explicitSpaceZero': "
            "t0 overlaps 'implicitSpace' t0"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(conflict_shader))

    allocation_shader = """
    shader RegisterSpaceZeroAllocation {
        sampler2D explicitSpaceZero @register(t0, space0);
        sampler2D autoSpaceZero @space(space0);
        sampler2D autoImplicit;

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return texture(autoSpaceZero, uv) + texture(autoImplicit, uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(allocation_shader))

    assert "Texture2D explicitSpaceZero : register(t0);" in generated_code
    assert "Texture2D autoSpaceZero : register(t1);" in generated_code
    assert "SamplerState autoSpaceZeroSampler : register(s0);" in generated_code
    assert "Texture2D autoImplicit : register(t2);" in generated_code
    assert "SamplerState autoImplicitSampler : register(s1);" in generated_code
    assert "space0" not in generated_code


def test_directx_rg_image_arrays_respect_scalar_and_vector_context():
    shader = """
    shader RGImageArrayContext {
        image2D rgFloatImages @rg32f[3];
        uimage2D rgUnsignedImages @rg32ui[2];

        float scalarFloat(image2D images[3] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[1], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloat(image2D images[3] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsigned(uimage2D images[2] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[1], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsigned(uimage2D images[2] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[1], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloat(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloat(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsigned(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsigned(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "RWTexture2D<float2> rgFloatImages[3] : register(u0);" in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[2] : register(u3);" in generated_code
    assert (
        "float scalarFloat(RWTexture2D<float2> images[3], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloat(RWTexture2D<float2> images[3], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsigned(RWTexture2D<uint2> images[2], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsigned(RWTexture2D<uint2> images[2], int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[1][pixel].x;" in generated_code
    assert "images[0][pixel] = float2((oldValue + value), 0.0);" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "images[1][pixel] = (oldValue + value);" in generated_code
    assert "uint oldValue = images[1][pixel].x;" in generated_code
    assert "images[0][pixel] = uint2((oldValue + value), 0u);" in generated_code
    assert "uint2 oldValue = images[1][pixel];" in generated_code
    assert "float oldValue = images[1][pixel];" not in generated_code
    assert "uint oldValue = images[1][pixel];" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_inferred_rg_image_arrays_respect_scalar_context():
    shader = """
    shader RGImageArrayInferredScalarContext {
        const int COUNT = 3;
        const int LAYER = COUNT - 1;
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[COUNT];
        image2D afterImages @rg32f;

        float scalarFloat(image2D images[] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloat(image2D images[] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsigned(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsigned(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloat(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloat(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsigned(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsigned(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "static const int COUNT = 3;" in generated_code
    assert "static const int LAYER = (COUNT - 1);" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[3] : register(u0);" in generated_code
    assert (
        "RWTexture2D<uint2> rgUnsignedImages[COUNT] : register(u3);" in generated_code
    )
    assert "RWTexture2D<float2> afterImages : register(u6);" in generated_code
    assert (
        "float scalarFloat(RWTexture2D<float2> images[3], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloat(RWTexture2D<float2> images[3], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsigned(RWTexture2D<uint2> images[COUNT], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsigned(RWTexture2D<uint2> images[COUNT], int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[LAYER][pixel].x;" in generated_code
    assert "images[0][pixel] = float2((oldValue + value), 0.0);" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "images[1][pixel] = (oldValue + value);" in generated_code
    assert "uint oldValue = images[LAYER][pixel].x;" in generated_code
    assert "images[0][pixel] = uint2((oldValue + value), 0u);" in generated_code
    assert "uint2 oldValue = images[2][pixel];" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in generated_code
    assert "RWTexture2D<float2> afterImages : register(u3);" not in generated_code
    assert "float oldValue = images[LAYER][pixel];" not in generated_code
    assert "uint oldValue = images[LAYER][pixel];" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_transitive_rg_image_arrays_share_call_site_size():
    shader = """
    shader TransitiveRGImageArrayScalarContext {
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];
        image2D afterImages @rg32f;

        float scalarFloatDeep(image2D images[] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        float scalarFloatMid(image2D images[] @rg32f, ivec2 pixel, float value) {
            return scalarFloatDeep(images, pixel, value);
        }

        vec2 vectorFloatDeep(image2D images[] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloatMid(image2D images[] @rg32f, ivec2 pixel, vec2 value) {
            return vectorFloatDeep(images, pixel, value);
        }

        uint scalarUnsignedDeep(uimage2D images[] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsignedMid(uimage2D images[] @rg32ui, ivec2 pixel, uint value) {
            return scalarUnsignedDeep(images, pixel, value);
        }

        uvec2 vectorUnsignedDeep(uimage2D images[] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsignedMid(uimage2D images[] @rg32ui, ivec2 pixel, uvec2 value) {
            return vectorUnsignedDeep(images, pixel, value);
        }

        compute {
            void main() {
                float sf = scalarFloatMid(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloatMid(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsignedMid(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsignedMid(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[4] : register(u4);" in generated_code
    assert "RWTexture2D<float2> afterImages : register(u8);" in generated_code
    assert (
        "float scalarFloatDeep(RWTexture2D<float2> images[4], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float scalarFloatMid(RWTexture2D<float2> images[4], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatDeep(RWTexture2D<float2> images[4], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatMid(RWTexture2D<float2> images[4], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedDeep(RWTexture2D<uint2> images[4], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedDeep(RWTexture2D<uint2> images[4], int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[3][pixel].x;" in generated_code
    assert "images[1][pixel] = float2((oldValue + value), 0.0);" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "uint oldValue = images[3][pixel].x;" in generated_code
    assert "images[1][pixel] = uint2((oldValue + value), 0u);" in generated_code
    assert "uint2 oldValue = images[2][pixel];" in generated_code
    assert "float2 vectorFloatDeep(RWTexture2D<float2> images[3]" not in generated_code
    assert "uint2 vectorUnsignedDeep(RWTexture2D<uint2> images[3]" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_fixed_rg_image_array_parameters_size_unsized_globals():
    shader = """
    shader FixedParamRGImageArrayContext {
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];

        float scalarFloatFixed(image2D images[4] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloatFixed(image2D images[4] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsignedFixed(uimage2D images[4] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsignedFixed(uimage2D images[4] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloatFixed(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloatFixed(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsignedFixed(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsignedFixed(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[4] : register(u4);" in generated_code
    assert (
        "float scalarFloatFixed(RWTexture2D<float2> images[4], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatFixed(RWTexture2D<float2> images[4], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(RWTexture2D<uint2> images[4], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedFixed(RWTexture2D<uint2> images[4], int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[3][pixel].x;" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "uint oldValue = images[3][pixel].x;" in generated_code
    assert "uint2 oldValue = images[2][pixel];" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[] : register(u1);" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_const_sized_rg_image_array_parameters_size_unsized_globals():
    shader = """
    shader FixedConstParamRGImageArrayContext {
        const int COUNT = 4;
        const int LAST = COUNT - 1;
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];

        float scalarFloatFixed(image2D images[COUNT] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[LAST], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloatFixed(image2D images[COUNT] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsignedFixed(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[LAST], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsignedFixed(uimage2D images[COUNT] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloatFixed(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloatFixed(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsignedFixed(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsignedFixed(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "static const int COUNT = 4;" in generated_code
    assert "static const int LAST = (COUNT - 1);" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[4] : register(u4);" in generated_code
    assert (
        "float scalarFloatFixed(RWTexture2D<float2> images[COUNT], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatFixed(RWTexture2D<float2> images[COUNT], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(RWTexture2D<uint2> images[COUNT], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedFixed(RWTexture2D<uint2> images[COUNT], int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[LAST][pixel].x;" in generated_code
    assert "uint oldValue = images[LAST][pixel].x;" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "uint2 oldValue = images[2][pixel];" in generated_code
    assert "float scalarFloatFixed(RWTexture2D<float2> images[4]" not in generated_code
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_expr_sized_rg_image_array_parameters_size_unsized_globals():
    shader = """
    shader FixedExprParamRGImageArrayContext {
        const int COUNT = 3;
        const int UINT_COUNT = 2;
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];

        float scalarFloatFixed(image2D images[(COUNT + 1)] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[COUNT], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorFloatFixed(image2D images[(COUNT + 1)] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsignedFixed(uimage2D images[(UINT_COUNT * 2)] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uvec2 vectorUnsignedFixed(uimage2D images[(UINT_COUNT * 2)] @rg32ui, ivec2 pixel, uvec2 value) {
            uvec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float sf = scalarFloatFixed(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 vf = vectorFloatFixed(rgFloatImages, ivec2(2, 3), vec2(1.0));
                uint su = scalarUnsignedFixed(rgUnsignedImages, ivec2(4, 5), 7u);
                uvec2 vu = vectorUnsignedFixed(rgUnsignedImages, ivec2(6, 7), uvec2(8u, 9u));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "static const int COUNT = 3;" in generated_code
    assert "static const int UINT_COUNT = 2;" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[4] : register(u4);" in generated_code
    assert (
        "float scalarFloatFixed(RWTexture2D<float2> images[(COUNT + 1)], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorFloatFixed(RWTexture2D<float2> images[(COUNT + 1)], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint scalarUnsignedFixed(RWTexture2D<uint2> images[(UINT_COUNT * 2)], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint2 vectorUnsignedFixed(RWTexture2D<uint2> images[(UINT_COUNT * 2)], int2 pixel, uint2 value)"
        in generated_code
    )
    assert "float oldValue = images[COUNT][pixel].x;" in generated_code
    assert "uint oldValue = images[3][pixel].x;" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "uint2 oldValue = images[2][pixel];" in generated_code
    assert "float scalarFloatFixed(RWTexture2D<float2> images[4]" not in generated_code
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_conflicting_fixed_rg_image_array_sizes_raise():
    shader = """
    shader ConflictingFixedRGImageArrayContext {
        image2D rgFloatImages @rg32f[];

        float touchFour(image2D images[4] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[3], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchThree(image2D images[3] @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float a = touchFour(rgFloatImages, ivec2(0, 1), 0.25);
                vec2 b = touchThree(rgFloatImages, ivec2(2, 3), vec2(1.0));
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_direct_rg_image_array_index_conflicts_with_fixed_parameter_size():
    shader = """
    shader DirectIndexFixedConflict {
        image2D rgFloatImages @rg32f[];

        float touchFour(image2D images[4] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[3], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float direct = imageLoad(rgFloatImages[5], pixel);
                float helper = touchFour(rgFloatImages, pixel, direct);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_direct_rg_image_array_index_within_fixed_parameter_size():
    shader = """
    shader DirectIndexWithinFixedSize {
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];

        float touchFour(image2D images[4] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[3], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uint touchUnsignedFour(uimage2D images[4] @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float directFloat = imageLoad(rgFloatImages[2], pixel);
                uint directUint = imageLoad(rgUnsignedImages[1], pixel);
                float helperFloat = touchFour(rgFloatImages, pixel, directFloat);
                uint helperUint = touchUnsignedFour(rgUnsignedImages, pixel, directUint);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[4] : register(u4);" in generated_code
    assert (
        "float touchFour(RWTexture2D<float2> images[4], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "uint touchUnsignedFour(RWTexture2D<uint2> images[4], int2 pixel, uint value)"
        in generated_code
    )
    assert "float directFloat = rgFloatImages[2][pixel].x;" in generated_code
    assert "uint directUint = rgUnsignedImages[1][pixel].x;" in generated_code
    assert "float oldValue = images[3][pixel].x;" in generated_code
    assert "uint oldValue = images[3][pixel].x;" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[3] : register(u0);" not in generated_code
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_fixed_rg_image_array_global_conflicts_with_fixed_parameter_size():
    shader = """
    shader FixedGlobalToMismatchedFixedHelper {
        image2D rgFloatImages @rg32f[4];

        float touchThree(image2D images[3] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float a = touchThree(rgFloatImages, ivec2(0, 1), 0.25);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_fixed_rg_image_array_global_widens_unsized_parameter_size():
    shader = """
    shader FixedGlobalToUnsizedHelper {
        image2D rgFloatImages @rg32f[4];

        float touchUnsized(image2D images[] @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                float a = touchUnsized(rgFloatImages, ivec2(0, 1), 0.25);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert (
        "float touchUnsized(RWTexture2D<float2> images[4], int2 pixel, float value)"
        in generated_code
    )
    assert "float oldValue = images[2][pixel].x;" in generated_code
    assert (
        "float touchUnsized(RWTexture2D<float2> images[3], int2 pixel, float value)"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_fixed_rg_image_array_global_direct_index_out_of_bounds_raises():
    shader = """
    shader FixedGlobalDirectIndexOutOfBounds {
        image2D rgFloatImages @rg32f[4];

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float value = imageLoad(rgFloatImages[4], pixel);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_fixed_rg_image_array_parameter_direct_index_out_of_bounds_raises():
    shader = """
    shader FixedParameterDirectIndexOutOfBounds {
        float touch(image2D images[4] @rg32f, ivec2 pixel) {
            return imageLoad(images[4], pixel);
        }

        compute {
            void main() {
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_fixed_rg_image_array_global_const_index_out_of_bounds_raises():
    shader = """
    shader FixedGlobalConstIndexOutOfBounds {
        const int COUNT = 4;
        image2D rgFloatImages @rg32f[4];

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float value = imageLoad(rgFloatImages[COUNT], pixel);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_fixed_rg_image_array_parameter_const_index_out_of_bounds_raises():
    shader = """
    shader FixedParameterConstIndexOutOfBounds {
        const int COUNT = 4;

        float touch(image2D images[4] @rg32f, ivec2 pixel) {
            return imageLoad(images[COUNT], pixel);
        }

        compute {
            void main() {
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_fixed_rg_image_array_const_index_within_bounds_generates():
    shader = """
    shader FixedConstIndexWithinBounds {
        const int COUNT = 4;
        image2D rgFloatImages @rg32f[4];

        float touch(image2D images[4] @rg32f, ivec2 pixel) {
            return imageLoad(images[COUNT - 1], pixel);
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float direct = imageLoad(rgFloatImages[COUNT - 1], pixel);
                float helper = touch(rgFloatImages, pixel);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "static const int COUNT = 4;" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "float touch(RWTexture2D<float2> images[4], int2 pixel)" in generated_code
    assert "return images[(COUNT - 1)][pixel].x;" in generated_code
    assert "float direct = rgFloatImages[(COUNT - 1)][pixel].x;" in generated_code
    assert (
        "float touch(RWTexture2D<float2> images[5], int2 pixel)" not in generated_code
    )
    assert "imageLoad(" not in generated_code


def test_directx_fixed_rg_image_array_shadowed_const_index_stays_dynamic():
    shader = """
    shader FixedShadowedConstIndex {
        const int COUNT = 4;
        image2D rgFloatImages @rg32f[4];

        float touch(image2D images[4] @rg32f, ivec2 pixel) {
            int COUNT = 0;
            return imageLoad(images[COUNT], pixel);
        }

        compute {
            void main() {
                int COUNT = 0;
                ivec2 pixel = ivec2(0, 1);
                float direct = imageLoad(rgFloatImages[COUNT], pixel);
                float helper = touch(rgFloatImages, pixel);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "static const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "float touch(RWTexture2D<float2> images[4], int2 pixel)" in generated_code
    assert "return images[COUNT][pixel].x;" in generated_code
    assert "float direct = rgFloatImages[COUNT][pixel].x;" in generated_code
    assert (
        "float touch(RWTexture2D<float2> images[5], int2 pixel)" not in generated_code
    )
    assert "imageLoad(" not in generated_code


def test_directx_transitive_rg_image_array_shadowed_const_index_stays_dynamic():
    shader = """
    shader TransitiveShadowedConstIndex {
        const int COUNT = 4;
        image2D rgFloatImages @rg32f[4];

        float leaf(image2D images[] @rg32f, ivec2 pixel) {
            int COUNT = 0;
            return imageLoad(images[COUNT], pixel);
        }

        float passThrough(image2D images[] @rg32f, ivec2 pixel) {
            int COUNT = 0;
            float sampled = imageLoad(images[COUNT], pixel);
            return sampled + leaf(images, pixel);
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float value = passThrough(rgFloatImages, pixel);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "compute"
    )

    assert "static const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" in generated_code
    assert "float leaf(RWTexture2D<float2> images[4], int2 pixel)" in generated_code
    assert (
        "float passThrough(RWTexture2D<float2> images[4], int2 pixel)" in generated_code
    )
    assert "return images[COUNT][pixel].x;" in generated_code
    assert "float sampled = images[COUNT][pixel].x;" in generated_code
    assert "float leaf(RWTexture2D<float2> images[5], int2 pixel)" not in generated_code
    assert (
        "float passThrough(RWTexture2D<float2> images[5], int2 pixel)"
        not in generated_code
    )
    assert "imageLoad(" not in generated_code


def test_directx_transitive_rg_image_array_unshadowed_const_index_conflict_raises():
    shader = """
    shader TransitiveUnshadowedConstIndexConflict {
        const int COUNT = 4;
        image2D rgFloatImages @rg32f[4];

        float leaf(image2D images[] @rg32f, ivec2 pixel) {
            return imageLoad(images[COUNT], pixel);
        }

        float passThrough(image2D images[] @rg32f, ivec2 pixel) {
            int COUNT = 0;
            return leaf(images, pixel);
        }

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 1);
                float value = passThrough(rgFloatImages, pixel);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "compute")


def test_directx_shadowed_rg_image_array_constant_keeps_scalar_context():
    shader = """
    shader ShadowedRGImageArrayScalarContext {
        const int LAYER = 3;
        image2D rgFloatImages @rg32f[];
        uimage2D rgUnsignedImages @rg32ui[];
        image2D afterImages @rg32f;

        float scalarFloat(image2D images[] @rg32f, ivec2 pixel, float value) {
            int LAYER = 0;
            float oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        uint scalarUnsigned(uimage2D images[] @rg32ui, ivec2 pixel, uint value) {
            int LAYER = 0;
            uint oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int LAYER = 3;" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[] : register(u0);" in generated_code
    assert "RWTexture2D<uint2> rgUnsignedImages[] : register(u1);" in generated_code
    assert "RWTexture2D<float2> afterImages : register(u2);" in generated_code
    assert (
        "float scalarFloat(RWTexture2D<float2> images[], int2 pixel, float value)"
        in generated_code
    )
    assert (
        "uint scalarUnsigned(RWTexture2D<uint2> images[], int2 pixel, uint value)"
        in generated_code
    )
    assert "int LAYER = 0;" in generated_code
    assert "float oldValue = images[LAYER][pixel].x;" in generated_code
    assert "images[0][pixel] = float2((oldValue + value), 0.0);" in generated_code
    assert "uint oldValue = images[LAYER][pixel].x;" in generated_code
    assert "images[0][pixel] = uint2((oldValue + value), 0u);" in generated_code
    assert "RWTexture2D<float2> rgFloatImages[4] : register(u0);" not in generated_code
    assert "RWTexture2D<float2> afterImages : register(u4);" not in generated_code
    assert "float oldValue = images[LAYER][pixel];" not in generated_code
    assert "uint oldValue = images[LAYER][pixel];" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_formatted_image_arrays_preserve_expression_sizes():
    shader = """
    shader ExprFormattedImageArrays {
        image2D counters @r32ui[(1 + 1) * 2];
        image2D rgPairs @rg16f[+3];
        image2D afterCounters @r32ui;
        sampler2D sampled;

        uint touchCounters(image2D images[(1 + 1) * 2] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[+3] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
                vec2 b = touchPairs(rgPairs, ivec2(2, 3), vec2(0.5));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "RWTexture2D<uint> counters[((1 + 1) * 2)] : register(u0);" in generated_code
    assert "RWTexture2D<float2> rgPairs[+3] : register(u4);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u7);" in generated_code
    assert "Texture2D sampled : register(t0);" in generated_code
    assert "SamplerState sampledSampler" not in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[((1 + 1) * 2)], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(RWTexture2D<float2> images[+3], int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[2][pixel];" in generated_code
    assert "images[1][pixel] = (oldValue + value);" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert (
        "float2 b = touchPairs(rgPairs, int2(2, 3), float2(0.5, 0.5));"
        in generated_code
    )
    assert "RWTexture2D<uint> afterCounters : register(u4);" not in generated_code
    assert "RWTexture2D<float4> counters" not in generated_code
    assert "RWTexture2D<float4> rgPairs" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_unsized_formatted_image_arrays_preserve_format_metadata():
    shader = """
    shader UnsizedFormattedImageArrays {
        image2D counters @r32ui[];
        image2D rgPairs @rg16f[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[0], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[0], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<uint> counters[] : register(u0);" in generated_code
    assert "RWTexture2D<float2> rgPairs[] : register(u1);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u2);" in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(RWTexture2D<float2> images[], int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[0][pixel];" in generated_code
    assert "float2 oldValue = images[0][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "RWTexture2D<float4> counters" not in generated_code
    assert "RWTexture2D<float4> rgPairs" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_formatted_image_arrays_infer_named_constant_size():
    shader = """
    shader ConstSizedFormattedImageArrays {
        const int COUNT = 3;
        const int LAYER = COUNT - 1;
        image2D counters @r32ui[COUNT];
        image2D rgPairs @rg16f[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[COUNT] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
                vec2 b = touchPairs(rgPairs, ivec2(2, 3), vec2(0.5));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int COUNT = 3;" in generated_code
    assert "static const int LAYER = (COUNT - 1);" in generated_code
    assert "RWTexture2D<uint> counters[COUNT] : register(u0);" in generated_code
    assert "RWTexture2D<float2> rgPairs[3] : register(u3);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u6);" in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[COUNT], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairs(RWTexture2D<float2> images[3], int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[LAYER][pixel];" in generated_code
    assert "images[1][pixel] = (oldValue + value);" in generated_code
    assert "float2 oldValue = images[LAYER][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert (
        "float2 b = touchPairs(rgPairs, int2(2, 3), float2(0.5, 0.5));"
        in generated_code
    )
    assert "RWTexture2D<float2> rgPairs[] : register(u3);" not in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u2);" not in generated_code
    assert "RWTexture2D<float4> counters" not in generated_code
    assert "RWTexture2D<float4> rgPairs" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_formatted_image_arrays_ignore_shadowed_local_constant():
    shader = """
    shader ShadowedImageConstIndex {
        const int LAYER = 3;
        image2D counters @r32ui[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            int LAYER = 0;
            uint oldValue = imageLoad(images[LAYER], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int LAYER = 3;" in generated_code
    assert "RWTexture2D<uint> counters[] : register(u0);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u1);" in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[], int2 pixel, uint value)"
        in generated_code
    )
    assert "int LAYER = 0;" in generated_code
    assert "uint oldValue = images[LAYER][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert "RWTexture2D<uint> counters[4] : register(u0);" not in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u4);" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_formatted_image_arrays_infer_transitive_helper_size():
    shader = """
    shader TransitiveFormattedImageArrays {
        image2D counters @r32ui[];
        image2D rgPairs @rg16f[];
        image2D afterCounters @r32ui;

        uint touchCountersDeep(image2D images[] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[3], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        uint touchCountersMid(image2D images[] @r32ui, ivec2 pixel, uint value) {
            return touchCountersDeep(images, pixel, value);
        }

        vec2 touchPairsDeep(image2D images[] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairsMid(image2D images[] @rg16f, ivec2 pixel, vec2 value) {
            return touchPairsDeep(images, pixel, value);
        }

        compute {
            void main() {
                uint a = touchCountersMid(counters, ivec2(1, 2), 3);
                vec2 b = touchPairsMid(rgPairs, ivec2(2, 3), vec2(0.5));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "RWTexture2D<uint> counters[4] : register(u0);" in generated_code
    assert "RWTexture2D<float2> rgPairs[3] : register(u4);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u7);" in generated_code
    assert (
        "uint touchCountersDeep(RWTexture2D<uint> images[4], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint touchCountersMid(RWTexture2D<uint> images[4], int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float2 touchPairsDeep(RWTexture2D<float2> images[3], int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "float2 touchPairsMid(RWTexture2D<float2> images[3], int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[3][pixel];" in generated_code
    assert "images[1][pixel] = (oldValue + value);" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "return touchCountersDeep(images, pixel, value);" in generated_code
    assert "return touchPairsDeep(images, pixel, value);" in generated_code
    assert "uint a = touchCountersMid(counters, int2(1, 2), 3);" in generated_code
    assert (
        "float2 b = touchPairsMid(rgPairs, int2(2, 3), float2(0.5, 0.5));"
        in generated_code
    )
    assert "RWTexture2D<uint> counters[] : register(u0);" not in generated_code
    assert "RWTexture2D<float2> rgPairs[] : register(u4);" not in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u2);" not in generated_code
    assert "RWTexture2D<float4> counters" not in generated_code
    assert "RWTexture2D<float4> rgPairs" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_formatted_image_arrays_ignore_unsupported_indices():
    dynamic_shader = """
    shader DynamicOnlyFormattedImageArrays {
        image2D counters @r32ui[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, int layer, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[layer], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, 0, ivec2(1, 2), 3);
            }
        }
    }
    """
    negative_shader = """
    shader NegativeIndexedFormattedImageArrays {
        image2D counters @r32ui[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[-1], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
            }
        }
    }
    """

    dynamic_code = HLSLCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = HLSLCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "RWTexture2D<uint> counters[] : register(u0);" in dynamic_code
    assert "RWTexture2D<uint> afterCounters : register(u1);" in dynamic_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[], int layer, int2 pixel, uint value)"
        in dynamic_code
    )
    assert "uint oldValue = images[layer][pixel];" in dynamic_code
    assert "images[0][pixel] = (oldValue + value);" in dynamic_code
    assert "uint a = touchCounters(counters, 0, int2(1, 2), 3);" in dynamic_code
    assert "RWTexture2D<uint> counters[2] : register(u0);" not in dynamic_code
    assert "RWTexture2D<uint> afterCounters : register(u2);" not in dynamic_code
    assert "imageLoad(" not in dynamic_code
    assert "imageStore(" not in dynamic_code

    assert "RWTexture2D<uint> counters[] : register(u0);" in negative_code
    assert "RWTexture2D<uint> afterCounters : register(u1);" in negative_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[], int2 pixel, uint value)"
        in negative_code
    )
    assert "uint oldValue = images[-1][pixel];" in negative_code
    assert "images[0][pixel] = (oldValue + value);" in negative_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in negative_code
    assert "RWTexture2D<uint> counters[0] : register(u0);" not in negative_code
    assert "RWTexture2D<uint> afterCounters : register(u0);" not in negative_code
    assert "imageLoad(" not in negative_code
    assert "imageStore(" not in negative_code


def test_directx_formatted_image_arrays_ignore_function_call_indices():
    shader = """
    shader FunctionIndexedFormattedImageArrays {
        image2D counters @r32ui[];
        image2D afterCounters @r32ui;

        int getLayer() {
            return 0;
        }

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[getLayer()], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "RWTexture2D<uint> counters[] : register(u0);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u1);" in generated_code
    assert "int getLayer()" in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[], int2 pixel, uint value)"
        in generated_code
    )
    assert "uint oldValue = images[getLayer()][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert "RWTexture2D<uint> counters[1] : register(u0);" not in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u2);" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_formatted_image_arrays_infer_local_constant_alias_size():
    shader = """
    shader LocalConstAliasFormattedImageArrays {
        const int GLOBAL = 2;
        image2D counters @r32ui[];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[] @r32ui, ivec2 pixel, uint value) {
            const int LOCAL = GLOBAL + 1;
            uint oldValue = imageLoad(images[LOCAL], pixel);
            imageStore(images[0], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int GLOBAL = 2;" in generated_code
    assert "RWTexture2D<uint> counters[4] : register(u0);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u4);" in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[4], int2 pixel, uint value)"
        in generated_code
    )
    assert "const int LOCAL = (GLOBAL + 1);" in generated_code
    assert "uint oldValue = images[LOCAL][pixel];" in generated_code
    assert "images[0][pixel] = (oldValue + value);" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3);" in generated_code
    assert "RWTexture2D<uint> counters[] : register(u0);" not in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u1);" not in generated_code
    assert "    int LOCAL = (GLOBAL + 1);" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_explicit_scalar_float_image_formats():
    shader = """
    shader ExplicitScalarFloatFormats {
        image2D normalized8 @r8;
        image3D normalized16 @format(r16);
        image2DArray halfLayers @ r16f;
        image2D signedNormalized @r8_snorm;

        float touchR8(image2D image @r8, ivec2 pixel, float value) {
            float oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        float touchR16(image3D image @r16, ivec3 voxel, float value) {
            float oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        float touchR16f(image2DArray image @r16f, ivec3 pixelLayer, float value) {
            float oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float> normalized8 : register(u0);" in generated_code
    assert "RWTexture3D<float> normalized16 : register(u1);" in generated_code
    assert "RWTexture2DArray<float> halfLayers : register(u2);" in generated_code
    assert "RWTexture2D<float> signedNormalized : register(u3);" in generated_code
    assert (
        "float touchR8(RWTexture2D<float> image, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float touchR16(RWTexture3D<float> image, int3 voxel, float value)"
        in generated_code
    )
    assert (
        "float touchR16f(RWTexture2DArray<float> image, int3 pixelLayer, float value)"
        in generated_code
    )
    assert "float oldValue = image[pixel];" in generated_code
    assert "float oldValue = image[voxel];" in generated_code
    assert "float oldValue = image[pixelLayer];" in generated_code
    assert "image[pixel] = (oldValue + value);" in generated_code
    assert "image[voxel] = (oldValue + value);" in generated_code
    assert "image[pixelLayer] = (oldValue + value);" in generated_code
    assert "RWTexture2D<float4> normalized8" not in generated_code
    assert "RWTexture3D<float4> normalized16" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_explicit_narrow_integer_image_formats():
    shader = """
    shader ExplicitNarrowIntegerFormats {
        image2D signed8 @r8i;
        image3D signed16 @format(r16i);
        image2D unsigned8 @ r8ui;
        image2DArray unsigned16 @format(r16ui);

        int loadSigned8(image2D image @r8i, ivec2 pixel, int value) {
            int oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        int loadSigned16(image3D image @r16i, ivec3 voxel, int value) {
            int oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uint loadUnsigned8(image2D image @r8ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        uint loadUnsigned16(image2DArray image @r16ui, ivec3 pixelLayer, uint value) {
            uint oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<int> signed8 : register(u0);" in generated_code
    assert "RWTexture3D<int> signed16 : register(u1);" in generated_code
    assert "RWTexture2D<uint> unsigned8 : register(u2);" in generated_code
    assert "RWTexture2DArray<uint> unsigned16 : register(u3);" in generated_code
    assert (
        "int loadSigned8(RWTexture2D<int> image, int2 pixel, int value)"
        in generated_code
    )
    assert (
        "int loadSigned16(RWTexture3D<int> image, int3 voxel, int value)"
        in generated_code
    )
    assert (
        "uint loadUnsigned8(RWTexture2D<uint> image, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "uint loadUnsigned16(RWTexture2DArray<uint> image, int3 pixelLayer, uint value)"
        in generated_code
    )
    assert "int oldValue = image[pixel];" in generated_code
    assert "int oldValue = image[voxel];" in generated_code
    assert "uint oldValue = image[pixel];" in generated_code
    assert "uint oldValue = image[pixelLayer];" in generated_code
    assert "RWTexture2D<float4> signed8" not in generated_code
    assert "RWTexture3D<float4> signed16" not in generated_code
    assert "RWTexture2D<float4> unsigned8" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_directx_explicit_integer_image_formats_use_atomic_helpers():
    shader = """
    shader ExplicitAtomicFormats {
        image2D unsignedCounters @r32ui;
        image2D signedCounters @r32i;
        image3D unsignedVolume @format(r32ui);
        image2DArray signedLayers @format(r32i);

        uint addUnsigned(image2D image @r32ui, ivec2 pixel, uint value) {
            return imageAtomicAdd(image, pixel, value);
        }

        int minSigned(image2D image @r32i, ivec2 pixel, int value) {
            return imageAtomicMin(image, pixel, value);
        }

        uint swapVolume(image3D image @r32ui, ivec3 voxel, uint expected, uint value) {
            return imageAtomicCompSwap(image, voxel, expected, value);
        }

        int exchangeLayer(image2DArray image @r32i, ivec3 pixelLayer, int value) {
            return imageAtomicExchange(image, pixelLayer, value);
        }

        compute {
            void main() {
                uint a = addUnsigned(unsignedCounters, ivec2(0, 0), 1);
                int b = minSigned(signedCounters, ivec2(1, 0), -1);
                uint c = swapVolume(unsignedVolume, ivec3(0, 1, 2), 3, 4);
                int d = exchangeLayer(signedLayers, ivec3(2, 3, 4), 5);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<uint> unsignedCounters : register(u0);" in generated_code
    assert "RWTexture2D<int> signedCounters : register(u1);" in generated_code
    assert "RWTexture3D<uint> unsignedVolume : register(u2);" in generated_code
    assert "RWTexture2DArray<int> signedLayers : register(u3);" in generated_code
    assert (
        "uint imageAtomicAdd_uimage2D(RWTexture2D<uint> image, int2 coord, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicMin_iimage2D(RWTexture2D<int> image, int2 coord, int value)"
        in generated_code
    )
    assert (
        "uint imageAtomicCompSwap_uimage3D(RWTexture3D<uint> image, int3 coord, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "int imageAtomicExchange_iimage2DArray(RWTexture2DArray<int> image, int3 coord, int value)"
        in generated_code
    )
    assert "return imageAtomicAdd_uimage2D(image, pixel, value);" in generated_code
    assert "return imageAtomicMin_iimage2D(image, pixel, value);" in generated_code
    assert (
        "return imageAtomicCompSwap_uimage3D(image, voxel, expected, value);"
        in generated_code
    )
    assert (
        "return imageAtomicExchange_iimage2DArray(image, pixelLayer, value);"
        in generated_code
    )
    assert "imageAtomicAdd(image" not in generated_code
    assert "imageAtomicMin(image" not in generated_code
    assert "imageAtomicCompSwap(image" not in generated_code
    assert "imageAtomicExchange(image" not in generated_code


def test_directx_float_image_atomic_exchange_emits_unsupported_fallback():
    shader = """
    shader FloatImageAtomicExchange {
        image2D floatCounters @r32f;

        float exchangeFloat(image2D image @r32f, ivec2 pixel, float value) {
            return imageAtomicExchange(image, pixel, value);
        }

        compute {
            void main() {
                float oldValue = exchangeFloat(floatCounters, ivec2(0, 1), 2.0);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float> floatCounters : register(u0);" in generated_code
    assert (
        "float exchangeFloat(RWTexture2D<float> image, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "return /* unsupported DirectX image atomic resource call: "
        "imageAtomicExchange on RWTexture2D<float> */ 0.0;" in generated_code
    )
    assert "imageAtomicExchange_" not in generated_code
    assert "InterlockedExchange" not in generated_code


@pytest.mark.parametrize(
    ("shader", "match"),
    [
        (
            """
            shader InvalidAtomicVectorFormat {
                image2D color @rgba32ui;

                uint addColor(image2D image @rgba32ui, ivec2 pixel, uint value) {
                    return imageAtomicAdd(image, pixel, value);
                }

                compute {
                    void main() {
                    }
                }
            }
            """,
            "DirectX image atomic operation 'imageAtomicAdd' requires r32i or "
            "r32ui image format, got rgba32ui",
        ),
        (
            """
            shader InvalidAtomicFloatAdd {
                image2D values @r32f;

                float addValue(image2D image @r32f, ivec2 pixel, float value) {
                    return imageAtomicAdd(image, pixel, value);
                }

                compute {
                    void main() {
                    }
                }
            }
            """,
            "DirectX image atomic operation 'imageAtomicAdd' requires r32i or "
            "r32ui image format, got r32f",
        ),
        (
            """
            shader InvalidAtomicValueKind {
                uimage2D counters @r32ui;

                uint addCounter(uimage2D image @r32ui, ivec2 pixel, int value) {
                    return imageAtomicAdd(image, pixel, value);
                }

                compute {
                    void main() {
                    }
                }
            }
            """,
            "DirectX image atomic operation 'imageAtomicAdd' requires uint "
            "data argument for r32ui images: value has type int",
        ),
        (
            """
            shader InvalidAtomicResultContext {
                uimage2D counters @r32ui;

                int addCounter(uimage2D image @r32ui, ivec2 pixel, uint value) {
                    return imageAtomicAdd(image, pixel, value);
                }

                compute {
                    void main() {
                    }
                }
            }
            """,
            "DirectX image atomic operation 'imageAtomicAdd' requires uint "
            "result context for r32ui images: expected int",
        ),
        (
            """
            shader InvalidFloatExchangeValueKind {
                image2D values @r32f;

                float exchangeValue(image2D image @r32f, ivec2 pixel, uint value) {
                    return imageAtomicExchange(image, pixel, value);
                }

                compute {
                    void main() {
                    }
                }
            }
            """,
            "DirectX image atomic operation 'imageAtomicExchange' requires float "
            "data argument for r32f images: value has type uint",
        ),
    ],
)
def test_directx_rejects_invalid_image_atomic_format_and_type_context(shader, match):
    ast = crosstl.translator.parse(shader)

    with pytest.raises(ValueError, match=re.escape(match)):
        HLSLCodeGen().generate(ast)


@pytest.mark.parametrize(
    ("declaration", "match"),
    [
        (
            "uvec2 wrong = imageAtomicAdd(counters, pixel, value);",
            "DirectX image atomic operation 'imageAtomicAdd' requires uint "
            "result context for r32ui images: expected uint2",
        ),
        (
            "bool wrong = imageAtomicAdd(counters, pixel, value);",
            "DirectX image atomic operation 'imageAtomicAdd' requires uint "
            "result context for r32ui images: expected bool",
        ),
    ],
)
def test_directx_image_atomics_reject_non_scalar_result_contexts(declaration, match):
    shader = f"""
    shader InvalidImageAtomicNonScalarResult {{
        uimage2D counters @r32ui;

        compute {{
            void main(ivec2 pixel @ TEXCOORD0) {{
                uint value = 1u;
                {declaration}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=re.escape(match)):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_explicit_vector_integer_image_formats():
    shader = """
    shader ExplicitVectorIntegerFormats {
        image2D unsignedColor @rgba32ui;
        image3D signedVolume @format(rgba32i);
        image2DArray unsignedLayers @ rgba16ui;

        uvec4 touchUnsigned(image2D image @rgba32ui, ivec2 pixel, uvec4 value) {
            uvec4 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        ivec4 touchSigned(image3D image @format(rgba32i), ivec3 voxel, ivec4 value) {
            ivec4 oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uvec4 touchLayers(image2DArray image @rgba16ui, ivec3 pixelLayer, uvec4 value) {
            uvec4 oldValue = imageLoad(image, pixelLayer);
            imageStore(image, pixelLayer, oldValue + value);
            return oldValue;
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<uint4> unsignedColor : register(u0);" in generated_code
    assert "RWTexture3D<int4> signedVolume : register(u1);" in generated_code
    assert "RWTexture2DArray<uint4> unsignedLayers : register(u2);" in generated_code
    assert (
        "uint4 touchUnsigned(RWTexture2D<uint4> image, int2 pixel, uint4 value)"
        in generated_code
    )
    assert (
        "int4 touchSigned(RWTexture3D<int4> image, int3 voxel, int4 value)"
        in generated_code
    )
    assert (
        "uint4 touchLayers(RWTexture2DArray<uint4> image, int3 pixelLayer, uint4 value)"
        in generated_code
    )
    assert "uint4 oldValue = image[pixel];" in generated_code
    assert "int4 oldValue = image[voxel];" in generated_code
    assert "uint4 oldValue = image[pixelLayer];" in generated_code
    assert "image[pixel] = (oldValue + value);" in generated_code
    assert "image[voxel] = (oldValue + value);" in generated_code
    assert "image[pixelLayer] = (oldValue + value);" in generated_code
    assert "RWTexture2D<float4> unsignedColor" not in generated_code
    assert "RWTexture3D<float4> signedVolume" not in generated_code


def test_directx_texture_array_resources_and_indexed_sampling():
    shader = """
    shader TextureArrayShader {
        sampler2D textures[4];
        samplerCube envMap;

        struct VSOutput {
            vec2 uv;
            vec3 normal;
            int layer;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 color = texture(textures[input.layer], input.uv);
                vec4 env = texture(envMap, input.normal);
                return color + env;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "TextureCube envMap : register(t4);" in generated_code
    assert "SamplerState texturesSampler : register(s0);" in generated_code
    assert "SamplerState envMapSampler : register(s1);" in generated_code
    assert "textures[input.layer].Sample(texturesSampler, input.uv)" in generated_code
    assert "textures[input.layer]Sampler" not in generated_code


def test_directx_fixed_texture_and_sampler_arrays_keep_declared_size_with_constant_indices():
    shader = """
    shader FixedArrayConstantIndex {
        const int LAYER = 2;
        sampler2D textures[6];
        sampler samplers[6];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[6], sampler samplers[6], vec2 uv) {
            return texture(textures[LAYER], samplers[LAYER], uv) + texture(textures[1 + 2], samplers[1 + 2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int LAYER = 2;" in generated_code
    assert "Texture2D textures[6] : register(t0);" in generated_code
    assert "SamplerState samplers[6] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t6);" in generated_code
    assert "SamplerState afterTextureSampler : register(s6);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[6], SamplerState samplers[6], float2 uv)"
        in generated_code
    )
    assert "textures[LAYER].Sample(samplers[LAYER], uv)" in generated_code
    assert "textures[(1 + 2)].Sample(samplers[(1 + 2)], uv)" in generated_code
    assert "Texture2D textures[4] : register(t0);" not in generated_code
    assert "Texture2D afterTexture : register(t4);" not in generated_code


def test_directx_fixed_texture_and_sampler_arrays_resolve_constant_declared_size_for_bindings():
    shader = """
    shader ConstSizedResourceArrays {
        const int BASE_COUNT = 2;
        const int TEXTURE_COUNT = BASE_COUNT * 3;
        sampler2D textures[TEXTURE_COUNT];
        sampler samplers[TEXTURE_COUNT];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[TEXTURE_COUNT], sampler samplers[TEXTURE_COUNT], vec2 uv) {
            return texture(textures[2], samplers[2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int BASE_COUNT = 2;" in generated_code
    assert "static const int TEXTURE_COUNT = (BASE_COUNT * 3);" in generated_code
    assert "Texture2D textures[TEXTURE_COUNT] : register(t0);" in generated_code
    assert "SamplerState samplers[TEXTURE_COUNT] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t6);" in generated_code
    assert "SamplerState afterTextureSampler : register(s6);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[TEXTURE_COUNT], SamplerState samplers[TEXTURE_COUNT], float2 uv)"
        in generated_code
    )
    assert "textures[2].Sample(samplers[2], uv)" in generated_code
    assert "Texture2D afterTexture : register(t1);" not in generated_code


def test_directx_fixed_texture_and_sampler_arrays_resolve_inline_declared_size_expression_for_bindings():
    shader = """
    shader ExprSizedResourceArrays {
        sampler2D textures[2 * 3];
        sampler samplers[2 * 3];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[2 * 3], sampler samplers[2 * 3], vec2 uv) {
            return texture(textures[2], samplers[2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D textures[(2 * 3)] : register(t0);" in generated_code
    assert "SamplerState samplers[(2 * 3)] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t6);" in generated_code
    assert "SamplerState afterTextureSampler : register(s6);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[(2 * 3)], SamplerState samplers[(2 * 3)], float2 uv)"
        in generated_code
    )
    assert "textures[2].Sample(samplers[2], uv)" in generated_code
    assert "Texture2D afterTexture : register(t1);" not in generated_code
    assert "[None]" not in generated_code


def test_directx_fixed_texture_arrays_preserve_parenthesized_and_unary_declared_sizes():
    shader = """
    shader ParenthesizedSizedResourceArrays {
        sampler2D textures[(2 + 1) * 2];
        sampler samplers[(2 + 1) * 2];
        sampler2D unaryTextures[+6];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[(2 + 1) * 2], sampler samplers[(2 + 1) * 2], sampler2D unaryTextures[+6], vec2 uv) {
            return texture(textures[2], samplers[2], uv) + texture(unaryTextures[2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, unaryTextures, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D textures[((2 + 1) * 2)] : register(t0);" in generated_code
    assert "SamplerState samplers[((2 + 1) * 2)] : register(s0);" in generated_code
    assert "Texture2D unaryTextures[+6] : register(t6);" in generated_code
    assert "SamplerState unaryTexturesSampler : register(s6);" in generated_code
    assert "Texture2D afterTexture : register(t12);" in generated_code
    assert "SamplerState afterTextureSampler : register(s7);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[((2 + 1) * 2)], SamplerState samplers[((2 + 1) * 2)], Texture2D unaryTextures[+6], SamplerState unaryTexturesSampler, float2 uv)"
        in generated_code
    )
    assert (
        "sampleLayer(textures, samplers, unaryTextures, unaryTexturesSampler, input.uv)"
        in generated_code
    )
    assert "textures[2].Sample(samplers[2], uv)" in generated_code
    assert "unaryTextures[2].Sample(unaryTexturesSampler, uv)" in generated_code
    assert "Texture2D afterTexture : register(t6);" not in generated_code
    assert "[None]" not in generated_code


def test_directx_texture_array_explicit_sampler():
    shader = """
    shader TextureArrayShader {
        sampler2D textures[4];
        sampler linearSampler;

        struct VSOutput {
            vec2 uv;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return texture(textures[0], linearSampler, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "textures[0].Sample(linearSampler, input.uv)" in generated_code
    assert "texturesSampler" not in generated_code


def test_directx_texture_array_helper_parameter_uses_implicit_sampler():
    shader = """
    shader TextureArrayHelper {
        sampler2D textures[4];

        struct VSOutput {
            vec2 uv;
            int layer;
        };

        vec4 sampleLayer(sampler2D textures[4], int layer, vec2 uv) {
            return texture(textures[layer], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, input.layer, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert (
        "float4 sampleLayer(Texture2D textures[4], SamplerState texturesSampler, int layer, float2 uv)"
        in generated_code
    )
    assert "textures[layer].Sample(texturesSampler, uv)" in generated_code
    assert (
        "sampleLayer(textures, texturesSampler, input.layer, input.uv)"
        in generated_code
    )
    assert "textures[layer]Sampler" not in generated_code


def test_directx_texture_array_helper_parameter_uses_indexed_sampler_array():
    shader = """
    shader SamplerArrayHelper {
        sampler2D textures[4];
        sampler samplers[4];

        struct VSOutput {
            vec2 uv;
            int layer;
        };

        vec4 sampleLayer(sampler2D textures[4], sampler samplers[4], int layer, vec2 uv) {
            return texture(textures[layer], samplers[layer], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.layer, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState samplers[4] : register(s0);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[4], SamplerState samplers[4], int layer, float2 uv)"
        in generated_code
    )
    assert "textures[layer].Sample(samplers[layer], uv)" in generated_code
    assert "sampleLayer(textures, samplers, input.layer, input.uv)" in generated_code
    assert "texturesSampler" not in generated_code


def test_directx_unsized_texture_and_sampler_arrays_infer_helper_size():
    shader = """
    shader UnsizedSampledResourceArrays {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            vec4 color = texture(textures[2], samplers[2], uv);
            vec4 other = texture(textures[1], samplers[1], uv);
            return color + other;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D textures[3] : register(t0);" in generated_code
    assert "SamplerState samplers[3] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t3);" in generated_code
    assert "SamplerState afterTextureSampler : register(s3);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[3], SamplerState samplers[3], float2 uv)"
        in generated_code
    )
    assert "textures[2].Sample(samplers[2], uv)" in generated_code
    assert "textures[1].Sample(samplers[1], uv)" in generated_code
    assert "Texture2D textures[] : register(t0);" not in generated_code
    assert "SamplerState samplers[] : register(s0);" not in generated_code


def test_directx_unsized_texture_and_sampler_arrays_infer_transitive_helper_size():
    shader = """
    shader MultiHopUnsizedSampledResources {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleDeep(sampler2D textures[], sampler samplers[], vec2 uv) {
            vec4 high = texture(textures[4], samplers[4], uv);
            vec4 low = texture(textures[1], samplers[1], uv);
            return high + low;
        }

        vec4 sampleMid(sampler2D textures[], sampler samplers[], vec2 uv) {
            return sampleDeep(textures, samplers, uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleMid(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D textures[5] : register(t0);" in generated_code
    assert "SamplerState samplers[5] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t5);" in generated_code
    assert "SamplerState afterTextureSampler : register(s5);" in generated_code
    assert (
        "float4 sampleDeep(Texture2D textures[5], SamplerState samplers[5], float2 uv)"
        in generated_code
    )
    assert (
        "float4 sampleMid(Texture2D textures[5], SamplerState samplers[5], float2 uv)"
        in generated_code
    )
    assert "textures[4].Sample(samplers[4], uv)" in generated_code
    assert "textures[1].Sample(samplers[1], uv)" in generated_code
    assert "sampleDeep(textures, samplers, uv)" in generated_code
    assert "sampleMid(textures, samplers, input.uv)" in generated_code
    assert "Texture2D textures[] : register(t0);" not in generated_code
    assert "SamplerState samplers[] : register(s0);" not in generated_code


def test_directx_unsized_texture_and_sampler_arrays_preserve_dynamic_indexing():
    shader = """
    shader MixedIndexedUnsizedSampledResources {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
            int layer;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], int layer, vec2 uv) {
            vec4 dynamicColor = texture(textures[layer], samplers[layer], uv);
            vec4 fixedColor = texture(textures[3], samplers[3], uv);
            return dynamicColor + fixedColor;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.layer, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState samplers[4] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t4);" in generated_code
    assert "SamplerState afterTextureSampler : register(s4);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[4], SamplerState samplers[4], int layer, float2 uv)"
        in generated_code
    )
    assert "textures[layer].Sample(samplers[layer], uv)" in generated_code
    assert "textures[3].Sample(samplers[3], uv)" in generated_code
    assert "sampleLayer(textures, samplers, input.layer, input.uv)" in generated_code
    assert "Texture2D textures[] : register(t0);" not in generated_code
    assert "SamplerState samplers[] : register(s0);" not in generated_code


def test_directx_unsized_texture_and_sampler_arrays_ignore_unsupported_indices():
    dynamic_shader = """
    shader DynamicOnlyUnsizedSampledResources {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
            int layer;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], int layer, vec2 uv) {
            return texture(textures[layer], samplers[layer], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.layer, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """
    negative_shader = """
    shader NegativeIndexedUnsizedSampledResources {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[-1], samplers[-1], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    dynamic_code = HLSLCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = HLSLCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "Texture2D textures[] : register(t0);" in dynamic_code
    assert "SamplerState samplers[] : register(s0);" in dynamic_code
    assert "Texture2D afterTexture : register(t1);" in dynamic_code
    assert "SamplerState afterTextureSampler : register(s1);" in dynamic_code
    assert (
        "float4 sampleLayer(Texture2D textures[], SamplerState samplers[], int layer, float2 uv)"
        in dynamic_code
    )
    assert "textures[layer].Sample(samplers[layer], uv)" in dynamic_code
    assert "Texture2D textures[1] : register(t0);" not in dynamic_code
    assert "Texture2D afterTexture : register(t2);" not in dynamic_code

    assert "Texture2D textures[] : register(t0);" in negative_code
    assert "SamplerState samplers[] : register(s0);" in negative_code
    assert "Texture2D afterTexture : register(t1);" in negative_code
    assert "SamplerState afterTextureSampler : register(s1);" in negative_code
    assert "textures[-1].Sample(samplers[-1], uv)" in negative_code
    assert "Texture2D textures[0] : register(t0);" not in negative_code
    assert "Texture2D afterTexture : register(t0);" not in negative_code


def test_directx_unsized_texture_and_sampler_arrays_infer_constant_expression_size():
    shader = """
    shader ExprIndexedUnsizedSampledResources {
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[1 + 2], samplers[1 + 2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState samplers[4] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t4);" in generated_code
    assert "SamplerState afterTextureSampler : register(s4);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[4], SamplerState samplers[4], float2 uv)"
        in generated_code
    )
    assert "textures[(1 + 2)].Sample(samplers[(1 + 2)], uv)" in generated_code
    assert "Texture2D textures[] : register(t0);" not in generated_code


def test_directx_unsized_texture_and_sampler_arrays_infer_named_constant_size():
    shader = """
    shader ConstIndexedUnsizedSampledResources {
        const int BASE = 1;
        const int LAYER = BASE + 2;
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[LAYER], samplers[LAYER], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int BASE = 1;" in generated_code
    assert "static const int LAYER = (BASE + 2);" in generated_code
    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState samplers[4] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t4);" in generated_code
    assert "SamplerState afterTextureSampler : register(s4);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[4], SamplerState samplers[4], float2 uv)"
        in generated_code
    )
    assert "textures[LAYER].Sample(samplers[LAYER], uv)" in generated_code
    assert "Texture2D textures[] : register(t0);" not in generated_code


def test_directx_unsized_texture_and_sampler_arrays_ignore_shadowed_constant_name():
    shader = """
    shader ShadowedConstIndex {
        const int LAYER = 3;
        sampler2D textures[];
        sampler samplers[];
        sampler2D afterTexture;

        struct VSOutput {
            vec2 uv;
            int layer;
        };

        vec4 sampleLayer(sampler2D textures[], sampler samplers[], int LAYER, vec2 uv) {
            return texture(textures[LAYER], samplers[LAYER], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.layer, input.uv) + texture(afterTexture, input.uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int LAYER = 3;" in generated_code
    assert "Texture2D textures[] : register(t0);" in generated_code
    assert "SamplerState samplers[] : register(s0);" in generated_code
    assert "Texture2D afterTexture : register(t1);" in generated_code
    assert "SamplerState afterTextureSampler : register(s1);" in generated_code
    assert (
        "float4 sampleLayer(Texture2D textures[], SamplerState samplers[], int LAYER, float2 uv)"
        in generated_code
    )
    assert "textures[LAYER].Sample(samplers[LAYER], uv)" in generated_code
    assert "Texture2D textures[4] : register(t0);" not in generated_code
    assert "Texture2D afterTexture : register(t4);" not in generated_code


def test_directx_fixed_texture_array_global_conflicts_with_fixed_parameter_size():
    shader = """
    shader FixedSampledGlobalMismatch {
        sampler2D textures[4];
        sampler samplers[4];

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleThree(sampler2D textures[3], sampler samplers[3], vec2 uv) {
            return texture(textures[2], samplers[2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleThree(textures, samplers, input.uv);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_directx_fixed_texture_array_global_widens_unsized_parameter_size():
    shader = """
    shader FixedSampledGlobalToUnsizedHelper {
        sampler2D textures[4];
        sampler samplers[4];

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleUnsized(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[2], samplers[2], uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleUnsized(textures, samplers, input.uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState samplers[4] : register(s0);" in generated_code
    assert (
        "float4 sampleUnsized(Texture2D textures[4], SamplerState samplers[4], float2 uv)"
        in generated_code
    )
    assert "textures[2].Sample(samplers[2], uv)" in generated_code
    assert "sampleUnsized(textures, samplers, input.uv)" in generated_code
    assert (
        "float4 sampleUnsized(Texture2D textures[3], SamplerState samplers[3], float2 uv)"
        not in generated_code
    )


def test_directx_fixed_texture_array_global_direct_index_out_of_bounds_raises():
    shader = """
    shader FixedSampledGlobalDirectOutOfBounds {
        sampler2D textures[4];
        sampler samplers[4];

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return texture(textures[4], samplers[4], uv);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_directx_fixed_texture_array_parameter_direct_index_out_of_bounds_raises():
    shader = """
    shader FixedSampledParamDirectOutOfBounds {
        vec4 sampleLayer(sampler2D textures[4], sampler samplers[4], vec2 uv) {
            return texture(textures[4], samplers[4], uv);
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return vec4(0.0);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_directx_fixed_texture_array_global_const_index_out_of_bounds_raises():
    shader = """
    shader FixedSampledGlobalConstIndexOutOfBounds {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return texture(textures[COUNT], samplers[COUNT], uv);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_directx_fixed_texture_array_parameter_const_index_out_of_bounds_raises():
    shader = """
    shader FixedSampledParamConstIndexOutOfBounds {
        const int COUNT = 4;

        vec4 sampleLayer(sampler2D textures[4], sampler samplers[4], vec2 uv) {
            return texture(textures[COUNT], samplers[COUNT], uv);
        }

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return vec4(0.0);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_directx_fixed_texture_array_const_index_and_shadowing_generate():
    shader = """
    shader FixedSampledConstIndexWithinBounds {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        vec4 sampleConst(sampler2D textures[4], sampler samplers[4], vec2 uv) {
            return texture(textures[COUNT - 1], samplers[COUNT - 1], uv);
        }

        vec4 sampleShadowed(sampler2D textures[4], sampler samplers[4], vec2 uv) {
            int COUNT = 0;
            return texture(textures[COUNT], samplers[COUNT], uv);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                int COUNT = 0;
                vec4 direct = texture(textures[COUNT], samplers[COUNT], input.uv);
                return direct + sampleConst(textures, samplers, input.uv) + sampleShadowed(textures, samplers, input.uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "static const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState samplers[4] : register(s0);" in generated_code
    assert (
        "float4 sampleConst(Texture2D textures[4], SamplerState samplers[4], float2 uv)"
        in generated_code
    )
    assert (
        "float4 sampleShadowed(Texture2D textures[4], SamplerState samplers[4], float2 uv)"
        in generated_code
    )
    assert "textures[(COUNT - 1)].Sample(samplers[(COUNT - 1)], uv)" in generated_code
    assert "textures[COUNT].Sample(samplers[COUNT], uv)" in generated_code
    assert "sampleConst(textures, samplers, input.uv)" in generated_code
    assert "sampleShadowed(textures, samplers, input.uv)" in generated_code
    assert "Texture2D textures[5]" not in generated_code
    assert "SamplerState samplers[5]" not in generated_code


def test_directx_transitive_texture_array_shadowed_const_index_stays_dynamic():
    shader = """
    shader TransitiveSampledShadowedConstIndex {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv) {
            int COUNT = 0;
            return texture(textures[COUNT], samplers[COUNT], uv);
        }

        vec4 passThrough(sampler2D textures[], sampler samplers[], vec2 uv) {
            int COUNT = 0;
            vec4 sampled = texture(textures[COUNT], samplers[COUNT], uv);
            return sampled + leaf(textures, samplers, uv);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return passThrough(textures, samplers, input.uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "static const int COUNT = 4;" in generated_code
    assert generated_code.count("int COUNT = 0;") == 2
    assert "Texture2D textures[4] : register(t0);" in generated_code
    assert "SamplerState samplers[4] : register(s0);" in generated_code
    assert (
        "float4 leaf(Texture2D textures[4], SamplerState samplers[4], float2 uv)"
        in generated_code
    )
    assert (
        "float4 passThrough(Texture2D textures[4], SamplerState samplers[4], float2 uv)"
        in generated_code
    )
    assert "return textures[COUNT].Sample(samplers[COUNT], uv);" in generated_code
    assert (
        "float4 sampled = textures[COUNT].Sample(samplers[COUNT], uv);"
        in generated_code
    )
    assert "return (sampled + leaf(textures, samplers, uv));" in generated_code
    assert "return passThrough(textures, samplers, input.uv);" in generated_code
    assert "Texture2D textures[5]" not in generated_code
    assert "SamplerState samplers[5]" not in generated_code


def test_directx_transitive_texture_array_unshadowed_const_index_conflict_raises():
    shader = """
    shader TransitiveSampledUnshadowedConstIndexConflict {
        const int COUNT = 4;
        sampler2D textures[4];
        sampler samplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
        };

        vec4 leaf(sampler2D textures[], sampler samplers[], vec2 uv) {
            return texture(textures[COUNT], samplers[COUNT], uv);
        }

        vec4 passThrough(sampler2D textures[], sampler samplers[], vec2 uv) {
            int COUNT = 0;
            return leaf(textures, samplers, uv);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return passThrough(textures, samplers, input.uv);
            }
        }
    }
    """

    with pytest.raises(ValueError, match="Conflicting fixed resource array sizes"):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_directx_texture_array_helper_operation_variants():
    shader = """
    shader TextureArrayOps {
        sampler2D textures[4];
        sampler samplers[4];

        struct VSOutput {
            vec2 uv;
            ivec2 pixel;
            int layer;
        };

        vec4 sampleOps(sampler2D textures[4], sampler samplers[4], int layer, vec2 uv, ivec2 pixel) {
            vec4 lodColor = textureLod(textures[layer], samplers[layer], uv, 2.0);
            vec4 gradColor = textureGrad(textures[layer], samplers[layer], uv, vec2(0.1), vec2(0.2));
            vec4 gathered = textureGather(textures[layer], samplers[layer], uv);
            vec4 fetched = texelFetch(textures[layer], pixel, 0);
            return lodColor + gradColor + gathered + fetched;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleOps(textures, samplers, input.layer, input.uv, input.pixel);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert (
        "float4 sampleOps(Texture2D textures[4], SamplerState samplers[4], int layer, float2 uv, int2 pixel)"
        in generated_code
    )
    assert "textures[layer].SampleLevel(samplers[layer], uv, 2.0)" in generated_code
    assert (
        "textures[layer].SampleGrad(samplers[layer], uv, float2(0.1, 0.1), float2(0.2, 0.2))"
        in generated_code
    )
    assert "textures[layer].Gather(samplers[layer], uv)" in generated_code
    assert "textures[layer].Load(int3(pixel, 0))" in generated_code
    assert (
        "sampleOps(textures, samplers, input.layer, input.uv, input.pixel)"
        in generated_code
    )
    assert "texturesSampler" not in generated_code


def test_directx_array_texture_types_and_shadow_compares():
    shader = """
    shader ArrayTextureTypes {
        sampler2DArray colorArray;
        sampler2DArrayShadow shadowArray;
        samplerCubeShadow cubeShadow;
        sampler arraySampler;
        sampler shadowSampler;

        struct VSOutput {
            vec3 uvLayer;
            ivec3 pixelLayer;
            vec3 direction;
            float depth;
        };

        vec4 sampleArray(sampler2DArray tex, sampler s, vec3 uvLayer, ivec3 pixelLayer) {
            vec4 color = texture(tex, s, uvLayer);
            vec4 lodColor = textureLod(tex, s, uvLayer, 1.0);
            vec4 gradColor = textureGrad(tex, s, uvLayer, vec2(0.1), vec2(0.2));
            vec4 gathered = textureGather(tex, s, uvLayer);
            vec4 fetched = texelFetch(tex, pixelLayer, 0);
            return color + lodColor + gradColor + gathered + fetched;
        }

        float sampleShadowArray(sampler2DArrayShadow tex, sampler s, vec3 uvLayer, float depth) {
            return textureCompare(tex, s, uvLayer, depth);
        }

        float sampleCubeShadow(samplerCubeShadow tex, sampler s, vec3 direction, float depth) {
            return textureCompare(tex, s, direction, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float shadow = sampleShadowArray(shadowArray, shadowSampler, input.uvLayer, input.depth);
                float cube = sampleCubeShadow(cubeShadow, shadowSampler, input.direction, input.depth);
                return sampleArray(colorArray, arraySampler, input.uvLayer, input.pixelLayer) * shadow * cube;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2DArray colorArray : register(t0);" in generated_code
    assert "Texture2DArray shadowArray : register(t1);" in generated_code
    assert "TextureCube cubeShadow : register(t2);" in generated_code
    assert "SamplerState arraySampler : register(s0);" in generated_code
    assert "SamplerComparisonState shadowSampler : register(s1);" in generated_code
    assert (
        "float4 sampleArray(Texture2DArray tex, SamplerState s, float3 uvLayer, int3 pixelLayer)"
        in generated_code
    )
    assert "tex.Sample(s, uvLayer)" in generated_code
    assert "tex.SampleLevel(s, uvLayer, 1.0)" in generated_code
    assert (
        "tex.SampleGrad(s, uvLayer, float2(0.1, 0.1), float2(0.2, 0.2))"
        in generated_code
    )
    assert "tex.Gather(s, uvLayer)" in generated_code
    assert "tex.Load(int4(pixelLayer, 0))" in generated_code
    assert (
        "float sampleShadowArray(Texture2DArray tex, SamplerComparisonState s, float3 uvLayer, float depth)"
        in generated_code
    )
    assert "tex.SampleCmp(s, uvLayer, depth)" in generated_code
    assert (
        "float sampleCubeShadow(TextureCube tex, SamplerComparisonState s, float3 direction, float depth)"
        in generated_code
    )
    assert "tex.SampleCmp(s, direction, depth)" in generated_code


def test_directx_cube_array_and_multisample_texture_types():
    shader = """
    shader CubeMsResources {
        samplerCubeArray cubeArray;
        samplerCubeArrayShadow cubeArrayShadow;
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler cubeSampler;
        sampler shadowSampler;

        struct VSOutput {
            vec4 cubeLayer;
            ivec2 pixel;
            ivec3 pixelLayer;
            int sampleIndex;
            float depth;
        };

        vec4 sampleCubeArray(samplerCubeArray tex, sampler s, vec4 cubeLayer) {
            return texture(tex, s, cubeLayer) + textureLod(tex, s, cubeLayer, 2.0);
        }

        float sampleCubeArrayShadow(samplerCubeArrayShadow tex, sampler s, vec4 cubeLayer, float depth) {
            return textureCompare(tex, s, cubeLayer, depth);
        }

        vec4 fetchMs(sampler2DMS tex, sampler2DMSArray texArray, ivec2 pixel, ivec3 pixelLayer, int sampleIndex) {
            return texelFetch(tex, pixel, sampleIndex) + texelFetch(texArray, pixelLayer, sampleIndex);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float shadow = sampleCubeArrayShadow(cubeArrayShadow, shadowSampler, input.cubeLayer, input.depth);
                return sampleCubeArray(cubeArray, cubeSampler, input.cubeLayer) * shadow + fetchMs(msTex, msArray, input.pixel, input.pixelLayer, input.sampleIndex);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "TextureCubeArray cubeArray : register(t0);" in generated_code
    assert "TextureCubeArray cubeArrayShadow : register(t1);" in generated_code
    assert "Texture2DMS<float4> msTex : register(t2);" in generated_code
    assert "Texture2DMSArray<float4> msArray : register(t3);" in generated_code
    assert "SamplerState cubeSampler : register(s0);" in generated_code
    assert "SamplerComparisonState shadowSampler : register(s1);" in generated_code
    assert "msTexSampler" not in generated_code
    assert "msArraySampler" not in generated_code
    assert (
        "float4 sampleCubeArray(TextureCubeArray tex, SamplerState s, float4 cubeLayer)"
        in generated_code
    )
    assert "tex.Sample(s, cubeLayer)" in generated_code
    assert "tex.SampleLevel(s, cubeLayer, 2.0)" in generated_code
    assert (
        "float sampleCubeArrayShadow(TextureCubeArray tex, SamplerComparisonState s, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert "tex.SampleCmp(s, cubeLayer, depth)" in generated_code
    assert (
        "float4 fetchMs(Texture2DMS<float4> tex, Texture2DMSArray<float4> texArray, int2 pixel, int3 pixelLayer, int sampleIndex)"
        in generated_code
    )
    assert "tex.Load(pixel, sampleIndex)" in generated_code
    assert "texArray.Load(pixelLayer, sampleIndex)" in generated_code


def test_directx_cube_array_texture_grad_gather_keep_sampler_arguments():
    shader = """
    shader CubeArrayGradGather {
        samplerCubeArray cubeArray;
        samplerCubeArray cubeArrays[4];
        sampler cubeSampler;
        sampler cubeSamplers[4];

        struct FSInput {
            vec4 cubeLayer @ TEXCOORD0;
            vec3 ddx @ TEXCOORD1;
            vec3 ddy @ TEXCOORD2;
        };

        vec4 sampleCubeArrayOps(samplerCubeArray tex, sampler s, vec4 cubeLayer, vec3 ddx, vec3 ddy) {
            vec4 gradColor = textureGrad(tex, s, cubeLayer, ddx, ddy);
            vec4 gathered = textureGather(tex, s, cubeLayer);
            return gradColor + gathered;
        }

        vec4 sampleCubeArrayElements(samplerCubeArray cubeArrays[], sampler cubeSamplers[], vec4 cubeLayer, vec3 ddx, vec3 ddy) {
            vec4 gradColor = textureGrad(cubeArrays[2], cubeSamplers[2], cubeLayer, ddx, ddy);
            vec4 gathered = textureGather(cubeArrays[3], cubeSamplers[3], cubeLayer);
            return gradColor + gathered;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return sampleCubeArrayOps(cubeArray, cubeSampler, input.cubeLayer, input.ddx, input.ddy)
                    + sampleCubeArrayElements(cubeArrays, cubeSamplers, input.cubeLayer, input.ddx, input.ddy);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "TextureCubeArray cubeArray : register(t0);" in generated_code
    assert "TextureCubeArray cubeArrays[4] : register(t1);" in generated_code
    assert "SamplerState cubeSampler : register(s0);" in generated_code
    assert "SamplerState cubeSamplers[4] : register(s1);" in generated_code
    assert (
        "float4 sampleCubeArrayOps(TextureCubeArray tex, SamplerState s, float4 cubeLayer, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert "tex.SampleGrad(s, cubeLayer, ddx, ddy)" in generated_code
    assert "tex.Gather(s, cubeLayer)" in generated_code
    assert (
        "float4 sampleCubeArrayElements(TextureCubeArray cubeArrays[4], SamplerState cubeSamplers[4], float4 cubeLayer, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "cubeArrays[2].SampleGrad(cubeSamplers[2], cubeLayer, ddx, ddy)"
        in generated_code
    )
    assert "cubeArrays[3].Gather(cubeSamplers[3], cubeLayer)" in generated_code
    assert (
        "sampleCubeArrayOps(cubeArray, cubeSampler, input.cubeLayer, input.ddx, input.ddy)"
        in generated_code
    )
    assert (
        "sampleCubeArrayElements(cubeArrays, cubeSamplers, input.cubeLayer, input.ddx, input.ddy)"
        in generated_code
    )
    assert "cubeArraySampler" not in generated_code
    assert "cubeArraysSampler" not in generated_code


def test_directx_texture_sampling_capability_descriptors():
    codegen = HLSLCodeGen()

    assert codegen.texture_sampling_capabilities("Texture2D") == {
        "texture_type": "Texture2D",
        "gather": True,
        "gather_offset": True,
        "sample_offset": True,
        "compare_offset": True,
        "gather_compare_offset": True,
    }
    assert codegen.texture_sampling_capabilities("TextureCube") == {
        "texture_type": "TextureCube",
        "gather": True,
        "gather_offset": False,
        "sample_offset": False,
        "compare_offset": False,
        "gather_compare_offset": False,
    }
    assert codegen.texture_sampling_capabilities("Texture2DArray[4]") == {
        "texture_type": "Texture2DArray",
        "gather": True,
        "gather_offset": True,
        "sample_offset": True,
        "compare_offset": True,
        "gather_compare_offset": True,
    }
    assert codegen.texture_sampling_capabilities("Texture3D") == {
        "texture_type": "Texture3D",
        "gather": False,
        "gather_offset": False,
        "sample_offset": True,
        "compare_offset": False,
        "gather_compare_offset": False,
    }


def test_directx_texture_dimension_descriptors():
    codegen = HLSLCodeGen()

    def expect(texture_type, **overrides):
        expected = {
            "texture_type": codegen.resource_base_type(texture_type),
            "coordinate_dimension": None,
            "offset_dimension": None,
            "sample_offset_dimension": None,
            "texel_fetch_offset_dimension": None,
            "gather_offset_dimension": None,
            "compare_offset_dimension": None,
            "compare_lod_offset_dimension": None,
            "compare_grad_offset_dimension": None,
            "gather_compare_offset_dimension": None,
            "gradient_dimension": None,
            "query_lod_coordinate_dimension": None,
        }
        expected.update(overrides)
        assert codegen.texture_dimension_descriptor(texture_type) == expected

    expect(
        "Texture2DArray[4]",
        coordinate_dimension=3,
        offset_dimension=2,
        sample_offset_dimension=2,
        texel_fetch_offset_dimension=2,
        gather_offset_dimension=2,
        compare_offset_dimension=2,
        compare_lod_offset_dimension=2,
        compare_grad_offset_dimension=2,
        gather_compare_offset_dimension=2,
        gradient_dimension=2,
        query_lod_coordinate_dimension=3,
    )
    expect(
        "Texture1DArray",
        coordinate_dimension=2,
        offset_dimension=1,
        sample_offset_dimension=1,
        texel_fetch_offset_dimension=1,
        gradient_dimension=1,
        query_lod_coordinate_dimension=2,
    )
    expect(
        "TextureCubeArray",
        gradient_dimension=3,
        query_lod_coordinate_dimension=4,
    )
    expect("Texture2DMSArray<float4>", coordinate_dimension=3)
    expect("RWTexture3D<float4>", coordinate_dimension=3)


def test_directx_texture_gather_offset_variants_keep_sampler_arguments():
    shader = """
    shader GatherOffsetVariants {
        sampler2D colorMap;
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            ivec2 offset @ TEXCOORD2;
            ivec2 offset0 @ TEXCOORD3;
            ivec2 offset1 @ TEXCOORD4;
            ivec2 offset2 @ TEXCOORD5;
            ivec2 offset3 @ TEXCOORD6;
            int component @ TEXCOORD7;
        };

        vec4 implicitGatherOffset(sampler2D tex, vec2 uv, ivec2 offset) {
            return textureGatherOffset(tex, uv, offset);
        }

        vec4 gatherArrayOffsets(
            sampler2DArray layers,
            sampler s,
            vec3 uvLayer,
            ivec2 offsets[4]
        ) {
            return textureGatherOffsets(layers, s, uvLayer, offsets, 2);
        }

        vec4 gatherOps(
            sampler2D tex,
            sampler2DArray layers,
            sampler s,
            vec2 uv,
            vec3 uvLayer,
            ivec2 offset,
            ivec2 offset0,
            ivec2 offset1,
            ivec2 offset2,
            ivec2 offset3,
            int component
        ) {
            vec4 green = textureGather(tex, s, uv, 1);
            vec4 dynamic = textureGather(tex, s, uv, component);
            vec4 offsetGather = textureGatherOffset(tex, s, uv, offset, 3);
            vec4 dynamicOffset = textureGatherOffset(tex, s, uv, offset, component);
            vec4 offsetsGather = textureGatherOffsets(
                layers,
                s,
                uvLayer,
                offset0,
                offset1,
                offset2,
                offset3,
                component
            );
            return green + dynamic + offsetGather + dynamicOffset + offsetsGather;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return implicitGatherOffset(colorMap, input.uv, input.offset)
                    + gatherOps(
                        colorMap,
                        layerMap,
                        linearSampler,
                        input.uv,
                        input.uvLayer,
                        input.offset,
                        input.offset0,
                        input.offset1,
                        input.offset2,
                        input.offset3,
                        input.component
                    );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "Texture2DArray layerMap : register(t1);" in generated_code
    assert "SamplerState colorMapSampler : register(s0);" in generated_code
    assert "SamplerState linearSampler : register(s1);" in generated_code
    assert (
        "float4 implicitGatherOffset(Texture2D tex, SamplerState texSampler, float2 uv, int2 offset)"
        in generated_code
    )
    assert "return tex.Gather(texSampler, uv, offset);" in generated_code
    assert (
        "float4 gatherArrayOffsets(Texture2DArray layers, SamplerState s, float3 uvLayer, int2 offsets[4])"
        in generated_code
    )
    assert (
        "return layers.GatherBlue(s, uvLayer, offsets[0], offsets[1], offsets[2], offsets[3]);"
        in generated_code
    )
    assert (
        "float4 gatherOps(Texture2D tex, Texture2DArray layers, SamplerState s, float2 uv, float3 uvLayer, int2 offset, int2 offset0, int2 offset1, int2 offset2, int2 offset3, int component)"
        in generated_code
    )
    assert "float4 green = tex.GatherGreen(s, uv);" in generated_code
    assert (
        "float4 dynamic = (component == 0 ? tex.GatherRed(s, uv) : "
        "component == 1 ? tex.GatherGreen(s, uv) : "
        "component == 2 ? tex.GatherBlue(s, uv) : tex.GatherAlpha(s, uv));"
        in generated_code
    )
    assert "float4 offsetGather = tex.GatherAlpha(s, uv, offset);" in generated_code
    assert (
        "component == 0 ? tex.GatherRed(s, uv, offset) : "
        "component == 1 ? tex.GatherGreen(s, uv, offset) : "
        "component == 2 ? tex.GatherBlue(s, uv, offset) : "
        "tex.GatherAlpha(s, uv, offset)" in generated_code
    )
    assert (
        "component == 0 ? layers.GatherRed(s, uvLayer, offset0, offset1, offset2, offset3) : "
        "component == 1 ? layers.GatherGreen(s, uvLayer, offset0, offset1, offset2, offset3) : "
        "component == 2 ? layers.GatherBlue(s, uvLayer, offset0, offset1, offset2, offset3) : "
        "layers.GatherAlpha(s, uvLayer, offset0, offset1, offset2, offset3)"
        in generated_code
    )
    assert (
        "implicitGatherOffset(colorMap, colorMapSampler, input.uv, input.offset)"
        in generated_code
    )
    assert (
        "gatherOps(colorMap, layerMap, linearSampler, input.uv, input.uvLayer, input.offset, input.offset0, input.offset1, input.offset2, input.offset3, input.component)"
        in generated_code
    )
    assert "textureGather(" not in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code


def test_directx_texture_gather_offsets_mix_literal_and_dynamic_offsets():
    shader = """
    shader GatherOffsetMixed {
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            ivec2 dynamic0 @ TEXCOORD1;
            ivec2 dynamic1 @ TEXCOORD2;
            int component @ TEXCOORD3;
        };

        vec4 gatherOps(
            sampler2DArray layers,
            sampler s,
            vec3 uvLayer,
            ivec2 dynamic0,
            ivec2 dynamic1,
            int component
        ) {
            return textureGatherOffsets(
                layers,
                s,
                uvLayer,
                ivec2(-1, 0),
                dynamic0,
                ivec2(1, -1),
                dynamic1,
                component
            );
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherOps(
                    layerMap,
                    linearSampler,
                    input.uvLayer,
                    input.dynamic0,
                    input.dynamic1,
                    input.component
                );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2DArray layerMap : register(t0);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert (
        "float4 gatherOps(Texture2DArray layers, SamplerState s, float3 uvLayer, int2 dynamic0, int2 dynamic1, int component)"
        in generated_code
    )
    assert "textureGatherOffsets(" not in generated_code
    methods = {
        0: "GatherRed",
        1: "GatherGreen",
        2: "GatherBlue",
        3: "GatherAlpha",
    }
    for component, method in methods.items():
        call = (
            f"layers.{method}(s, uvLayer, int2(-1, 0), dynamic0, int2(1, -1), dynamic1)"
        )
        assert call in generated_code
        if component < 3:
            assert f"component == {component} ? {call}" in generated_code
    assert (
        "gatherOps(layerMap, linearSampler, input.uvLayer, input.dynamic0, input.dynamic1, input.component)"
        in generated_code
    )


def test_directx_direct_stage_gather_offsets_use_input_members():
    shader = """
    shader DirectStageGatherOffsets {
        sampler2D colorMap;
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            ivec2 offset @ TEXCOORD2;
            ivec2 offset0 @ TEXCOORD3;
            ivec2 offset1 @ TEXCOORD4;
            ivec2 offset2 @ TEXCOORD5;
            ivec2 offset3 @ TEXCOORD6;
            int component @ TEXCOORD7;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 dynamic = textureGather(colorMap, linearSampler, input.uv, input.component);
                vec4 dynamicOffset = textureGatherOffset(colorMap, linearSampler, input.uv, input.offset, input.component);
                vec4 offsetsGather = textureGatherOffsets(
                    layerMap,
                    linearSampler,
                    input.uvLayer,
                    input.offset0,
                    input.offset1,
                    input.offset2,
                    input.offset3,
                    input.component
                );
                return dynamic + dynamicOffset + offsetsGather;
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "Texture2DArray layerMap : register(t1);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert (
        "float4 dynamic = (input.component == 0 ? colorMap.GatherRed(linearSampler, input.uv) : "
        "input.component == 1 ? colorMap.GatherGreen(linearSampler, input.uv) : "
        "input.component == 2 ? colorMap.GatherBlue(linearSampler, input.uv) : "
        "colorMap.GatherAlpha(linearSampler, input.uv));" in generated_code
    )
    assert (
        "input.component == 0 ? colorMap.GatherRed(linearSampler, input.uv, input.offset) : "
        "input.component == 1 ? colorMap.GatherGreen(linearSampler, input.uv, input.offset) : "
        "input.component == 2 ? colorMap.GatherBlue(linearSampler, input.uv, input.offset) : "
        "colorMap.GatherAlpha(linearSampler, input.uv, input.offset)" in generated_code
    )
    assert (
        "input.component == 0 ? layerMap.GatherRed(linearSampler, input.uvLayer, input.offset0, input.offset1, input.offset2, input.offset3) : "
        "input.component == 1 ? layerMap.GatherGreen(linearSampler, input.uvLayer, input.offset0, input.offset1, input.offset2, input.offset3) : "
        "input.component == 2 ? layerMap.GatherBlue(linearSampler, input.uvLayer, input.offset0, input.offset1, input.offset2, input.offset3) : "
        "layerMap.GatherAlpha(linearSampler, input.uvLayer, input.offset0, input.offset1, input.offset2, input.offset3)"
        in generated_code
    )
    assert "textureGather(" not in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code


def test_directx_cube_texture_gather_offsets_emit_diagnostics_without_samplers():
    shader = """
    shader UnsupportedCubeGatherOffsets {
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler cubeSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            ivec2 offset @ TEXCOORD2;
            ivec2 offset0 @ TEXCOORD3;
            ivec2 offset1 @ TEXCOORD4;
            ivec2 offset2 @ TEXCOORD5;
            ivec2 offset3 @ TEXCOORD6;
            int component @ TEXCOORD7;
        };

        vec4 gatherCube(
            samplerCube tex,
            sampler s,
            vec3 direction,
            ivec2 offset,
            ivec2 offset0,
            ivec2 offset1,
            ivec2 offset2,
            ivec2 offset3,
            int component
        ) {
            vec4 offsetGather = textureGatherOffset(tex, s, direction, offset, component);
            vec4 offsetsGather = textureGatherOffsets(tex, s, direction, offset0, offset1, offset2, offset3, component);
            return offsetGather + offsetsGather;
        }

        vec4 gatherCubeArray(
            samplerCubeArray tex,
            sampler s,
            vec4 cubeLayer,
            ivec2 offset,
            ivec2 offset0,
            ivec2 offset1,
            ivec2 offset2,
            ivec2 offset3,
            int component
        ) {
            vec4 offsetGather = textureGatherOffset(tex, s, cubeLayer, offset, component);
            vec4 offsetsGather = textureGatherOffsets(tex, s, cubeLayer, offset0, offset1, offset2, offset3, component);
            return offsetGather + offsetsGather;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherCube(
                    cubeMap,
                    cubeSampler,
                    input.direction,
                    input.offset,
                    input.offset0,
                    input.offset1,
                    input.offset2,
                    input.offset3,
                    input.component
                ) + gatherCubeArray(
                    cubeArray,
                    cubeSampler,
                    input.cubeLayer,
                    input.offset,
                    input.offset0,
                    input.offset1,
                    input.offset2,
                    input.offset3,
                    input.component
                ) + textureGatherOffset(cubeMap, input.direction, input.offset, input.component)
                    + textureGatherOffset(cubeArray, input.cubeLayer, input.offset, input.component)
                    + textureGatherOffsets(cubeMap, input.direction, input.offset0, input.offset1, input.offset2, input.offset3, input.component)
                    + textureGatherOffsets(cubeArray, input.cubeLayer, input.offset0, input.offset1, input.offset2, input.offset3, input.component);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    offset_diagnostic = (
        "/* unsupported DirectX texture gather: textureGatherOffset "
        "offsets require 2D or 2D-array textures */ float4(0.0, 0.0, 0.0, 0.0)"
    )
    offsets_diagnostic = (
        "/* unsupported DirectX texture gather: textureGatherOffsets "
        "offsets require 2D or 2D-array textures */ float4(0.0, 0.0, 0.0, 0.0)"
    )
    assert "TextureCube cubeMap : register(t0);" in generated_code
    assert "TextureCubeArray cubeArray : register(t1);" in generated_code
    assert "SamplerState cubeSampler : register(s0);" in generated_code
    assert "SamplerState cubeMapSampler" not in generated_code
    assert "SamplerState cubeArraySampler" not in generated_code
    assert (
        "float4 gatherCube(TextureCube tex, SamplerState s, float3 direction, int2 offset, int2 offset0, int2 offset1, int2 offset2, int2 offset3, int component)"
        in generated_code
    )
    assert (
        "float4 gatherCubeArray(TextureCubeArray tex, SamplerState s, float4 cubeLayer, int2 offset, int2 offset0, int2 offset1, int2 offset2, int2 offset3, int component)"
        in generated_code
    )
    assert generated_code.count(offset_diagnostic) == 4
    assert generated_code.count(offsets_diagnostic) == 4
    assert ".Gather" not in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code


def test_directx_unsupported_dimension_texture_gather_emits_diagnostics_without_samplers():
    shader = """
    shader UnsupportedDimensionGather {
        sampler1D lineMap;
        sampler3D volumeMap;
        sampler linearSampler;

        struct FSInput {
            float u @ TEXCOORD0;
            vec3 volumeUv @ TEXCOORD1;
            int component @ TEXCOORD2;
        };

        vec4 gatherLine(sampler1D tex, sampler s, float u, int component) {
            vec4 fixedGather = textureGather(tex, s, u, 2);
            vec4 dynamicGather = textureGather(tex, s, u, component);
            return fixedGather + dynamicGather;
        }

        vec4 gatherVolume(sampler3D tex, sampler s, vec3 volumeUv, int component) {
            vec4 fixedGather = textureGather(tex, s, volumeUv, 1);
            vec4 dynamicGather = textureGather(tex, s, volumeUv, component);
            return fixedGather + dynamicGather;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherLine(lineMap, linearSampler, input.u, input.component)
                    + gatherVolume(volumeMap, linearSampler, input.volumeUv, input.component)
                    + textureGather(lineMap, input.u, input.component)
                    + textureGather(volumeMap, input.volumeUv, input.component);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported DirectX texture gather: textureGather requires 2D, "
        "2D-array, cube, or cube-array textures */ float4(0.0, 0.0, 0.0, 0.0)"
    )
    assert "Texture1D lineMap : register(t0);" in generated_code
    assert "Texture3D volumeMap : register(t1);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "SamplerState lineMapSampler" not in generated_code
    assert "SamplerState volumeMapSampler" not in generated_code
    assert (
        "float4 gatherLine(Texture1D tex, SamplerState s, float u, int component)"
        in generated_code
    )
    assert (
        "float4 gatherVolume(Texture3D tex, SamplerState s, float3 volumeUv, int component)"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 6
    assert ".Gather" not in generated_code
    assert "textureGather(" not in generated_code


def test_directx_unsupported_dimension_texture_gather_offsets_emit_diagnostics_without_samplers():
    shader = """
    shader UnsupportedDimensionGatherOffsets {
        sampler1D lineMap;
        sampler3D volumeMap;
        sampler linearSampler;

        struct FSInput {
            float u @ TEXCOORD0;
            vec3 volumeUv @ TEXCOORD1;
            ivec2 offset @ TEXCOORD2;
            int component @ TEXCOORD3;
        };

        vec4 gatherLine(
            sampler1D tex,
            sampler s,
            float u,
            ivec2 offset,
            int component
        ) {
            return textureGatherOffset(tex, s, u, offset, component);
        }

        vec4 gatherVolume(
            sampler3D tex,
            sampler s,
            vec3 volumeUv,
            ivec2 offset,
            int component
        ) {
            return textureGatherOffsets(
                tex,
                s,
                volumeUv,
                offset,
                offset,
                offset,
                offset,
                component
            );
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherLine(
                    lineMap,
                    linearSampler,
                    input.u,
                    input.offset,
                    input.component
                ) + gatherVolume(
                    volumeMap,
                    linearSampler,
                    input.volumeUv,
                    input.offset,
                    input.component
                ) + textureGatherOffset(
                    lineMap,
                    input.u,
                    input.offset,
                    input.component
                ) + textureGatherOffsets(
                    volumeMap,
                    input.volumeUv,
                    input.offset,
                    input.offset,
                    input.offset,
                    input.offset,
                    input.component
                );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    offset_diagnostic = (
        "/* unsupported DirectX texture gather: textureGatherOffset offsets "
        "require 2D or 2D-array textures */ float4(0.0, 0.0, 0.0, 0.0)"
    )
    offsets_diagnostic = (
        "/* unsupported DirectX texture gather: textureGatherOffsets offsets "
        "require 2D or 2D-array textures */ float4(0.0, 0.0, 0.0, 0.0)"
    )
    assert "Texture1D lineMap : register(t0);" in generated_code
    assert "Texture3D volumeMap : register(t1);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "SamplerState lineMapSampler" not in generated_code
    assert "SamplerState volumeMapSampler" not in generated_code
    assert (
        "float4 gatherLine(Texture1D tex, SamplerState s, float u, int2 offset, int component)"
        in generated_code
    )
    assert (
        "float4 gatherVolume(Texture3D tex, SamplerState s, float3 volumeUv, int2 offset, int component)"
        in generated_code
    )
    assert generated_code.count(offset_diagnostic) == 2
    assert generated_code.count(offsets_diagnostic) == 2
    assert ".Gather" not in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code


def test_directx_unsupported_dimension_texture_gather_compare_emits_diagnostics_without_samplers():
    shader = """
    shader UnsupportedDimensionGatherCompare {
        sampler1D lineMap;
        sampler3D volumeMap;
        sampler2D colorMap;
        sampler sharedSampler;

        struct FSInput {
            float u @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            vec3 volumeUv @ TEXCOORD2;
            float depth;
        };

        vec4 gatherLine(sampler1D tex, sampler s, float u, float depth) {
            return textureGatherCompare(tex, s, u, depth);
        }

        vec4 gatherVolume(sampler3D tex, sampler s, vec3 volumeUv, float depth) {
            return textureGatherCompare(tex, s, volumeUv, depth);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 sampled = texture(colorMap, sharedSampler, input.uv);
                return sampled
                    + gatherLine(lineMap, sharedSampler, input.u, input.depth)
                    + gatherVolume(volumeMap, sharedSampler, input.volumeUv, input.depth)
                    + textureGatherCompare(lineMap, input.u, input.depth)
                    + textureGatherCompare(volumeMap, input.volumeUv, input.depth);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported DirectX texture gather compare: textureGatherCompare "
        "requires 2D, 2D-array, cube, or cube-array textures */ "
        "float4(0.0, 0.0, 0.0, 0.0)"
    )
    assert "Texture1D lineMap : register(t0);" in generated_code
    assert "Texture3D volumeMap : register(t1);" in generated_code
    assert "Texture2D colorMap : register(t2);" in generated_code
    assert "SamplerState sharedSampler : register(s0);" in generated_code
    assert "SamplerComparisonState sharedSampler" not in generated_code
    assert "lineMapSampler" not in generated_code
    assert "volumeMapSampler" not in generated_code
    assert (
        "float4 gatherLine(Texture1D tex, SamplerState s, float u, float depth)"
        in generated_code
    )
    assert (
        "float4 gatherVolume(Texture3D tex, SamplerState s, float3 volumeUv, float depth)"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 4
    assert (
        "float4 sampled = colorMap.Sample(sharedSampler, input.uv);" in generated_code
    )
    assert ".GatherCmp(" not in generated_code
    assert "textureGatherCompare(" not in generated_code


def test_directx_texture_sample_offset_variants_use_sample_offsets():
    shader = """
    shader TextureSampleOffsets {
        sampler2D colorMap;
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 implicitOffsetOps(
            sampler2D tex,
            vec2 uv,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec4 plain = textureOffset(tex, uv, offset);
            vec4 lodSample = textureLodOffset(tex, uv, lod, offset);
            vec4 gradSample = textureGradOffset(tex, uv, ddx, ddy, offset);
            return plain + lodSample + gradSample;
        }

        vec4 offsetOps(
            sampler2D tex,
            sampler2DArray layers,
            sampler s,
            vec2 uv,
            vec3 uvLayer,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec4 plain = textureOffset(tex, s, uv, offset);
            vec4 lodSample = textureLodOffset(tex, s, uv, lod, offset);
            vec4 gradSample = textureGradOffset(tex, s, uv, ddx, ddy, offset);
            vec4 arrayLod = textureLodOffset(layers, s, uvLayer, lod, offset);
            vec4 arrayGrad = textureGradOffset(layers, s, uvLayer, ddx, ddy, offset);
            return plain + lodSample + gradSample + arrayLod + arrayGrad;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return implicitOffsetOps(
                    colorMap,
                    input.uv,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + offsetOps(
                    colorMap,
                    layerMap,
                    linearSampler,
                    input.uv,
                    input.uvLayer,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "Texture2DArray layerMap : register(t1);" in generated_code
    assert "SamplerState colorMapSampler : register(s0);" in generated_code
    assert "SamplerState linearSampler : register(s1);" in generated_code
    assert (
        "float4 implicitOffsetOps(Texture2D tex, SamplerState texSampler, float2 uv, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "float4 plain = tex.Sample(texSampler, uv, offset);" in generated_code
    assert (
        "float4 lodSample = tex.SampleLevel(texSampler, uv, lod, offset);"
        in generated_code
    )
    assert (
        "float4 gradSample = tex.SampleGrad(texSampler, uv, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float4 offsetOps(Texture2D tex, Texture2DArray layers, SamplerState s, float2 uv, float3 uvLayer, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "float4 plain = tex.Sample(s, uv, offset);" in generated_code
    assert "float4 lodSample = tex.SampleLevel(s, uv, lod, offset);" in generated_code
    assert (
        "float4 gradSample = tex.SampleGrad(s, uv, ddx, ddy, offset);" in generated_code
    )
    assert (
        "float4 arrayLod = layers.SampleLevel(s, uvLayer, lod, offset);"
        in generated_code
    )
    assert (
        "float4 arrayGrad = layers.SampleGrad(s, uvLayer, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "implicitOffsetOps(colorMap, colorMapSampler, input.uv, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "offsetOps(colorMap, layerMap, linearSampler, input.uv, input.uvLayer, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code


def test_directx_cube_texture_sample_offsets_emit_diagnostics_without_samplers():
    shader = """
    shader CubeTextureSampleOffsetDiagnostics {
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        samplerCubeShadow shadowMap;
        samplerCubeArrayShadow shadowArray;
        sampler linearSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 cubeOffsets(
            samplerCube tex,
            sampler s,
            vec3 direction,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            vec4 plain = textureOffset(tex, s, direction, offset);
            vec4 lodSample = textureLodOffset(tex, s, direction, lod, offset);
            vec4 gradSample = textureGradOffset(tex, s, direction, ddx, ddy, offset);
            return plain + lodSample + gradSample;
        }

        vec4 cubeArrayOffsets(
            samplerCubeArray tex,
            sampler s,
            vec4 cubeLayer,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            vec4 plain = textureOffset(tex, s, cubeLayer, offset);
            vec4 lodSample = textureLodOffset(tex, s, cubeLayer, lod, offset);
            vec4 gradSample = textureGradOffset(tex, s, cubeLayer, ddx, ddy, offset);
            return plain + lodSample + gradSample;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 sampled = cubeOffsets(
                    cubeMap,
                    linearSampler,
                    input.direction,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + cubeArrayOffsets(
                    cubeArray,
                    linearSampler,
                    input.cubeLayer,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                )
                    + textureOffset(cubeMap, input.direction, input.offset)
                    + textureLodOffset(cubeMap, input.direction, input.lod, input.offset)
                    + textureGradOffset(cubeMap, input.direction, input.ddx, input.ddy, input.offset)
                    + textureOffset(cubeArray, input.cubeLayer, input.offset)
                    + textureLodOffset(cubeArray, input.cubeLayer, input.lod, input.offset)
                    + textureGradOffset(cubeArray, input.cubeLayer, input.ddx, input.ddy, input.offset);
                float compared = textureCompareOffset(shadowMap, input.direction, input.depth, input.offset)
                    + textureCompareLodOffset(shadowMap, input.direction, input.depth, input.lod, input.offset)
                    + textureCompareGradOffset(shadowMap, input.direction, input.depth, input.ddx, input.ddy, input.offset)
                    + textureCompareOffset(shadowArray, input.cubeLayer, input.depth, input.offset)
                    + textureCompareLodOffset(shadowArray, input.cubeLayer, input.depth, input.lod, input.offset)
                    + textureCompareGradOffset(shadowArray, input.cubeLayer, input.depth, input.ddx, input.ddy, input.offset);
                return sampled + vec4(compared);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "TextureCube cubeMap : register(t0);" in generated_code
    assert "TextureCubeArray cubeArray : register(t1);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "cubeMapSampler" not in generated_code
    assert "cubeArraySampler" not in generated_code
    assert "shadowMapSampler" not in generated_code
    assert "shadowArraySampler" not in generated_code
    assert (
        generated_code.count(
            "/* unsupported DirectX texture offset: textureOffset offsets require 1D, 2D, 2D-array, or 3D textures */ float4(0.0, 0.0, 0.0, 0.0)"
        )
        == 4
    )
    assert (
        generated_code.count(
            "/* unsupported DirectX texture offset: textureLodOffset offsets require 1D, 2D, 2D-array, or 3D textures */ float4(0.0, 0.0, 0.0, 0.0)"
        )
        == 4
    )
    assert (
        generated_code.count(
            "/* unsupported DirectX texture offset: textureGradOffset offsets require 1D, 2D, 2D-array, or 3D textures */ float4(0.0, 0.0, 0.0, 0.0)"
        )
        == 4
    )
    assert (
        generated_code.count(
            "/* unsupported DirectX texture compare: textureCompareOffset offsets require 2D or 2D-array textures */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported DirectX texture compare: textureCompareLodOffset offsets require 2D or 2D-array textures */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported DirectX texture compare: textureCompareGradOffset offsets require 2D or 2D-array textures */ 0.0"
        )
        == 2
    )
    assert ".Sample(" not in generated_code
    assert ".SampleLevel(" not in generated_code
    assert ".SampleGrad(" not in generated_code
    assert ".SampleCmp" not in generated_code
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code
    assert "textureCompareOffset(" not in generated_code
    assert "textureCompareLodOffset(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_directx_direct_stage_sample_offsets_and_texel_fetch_offset_use_input_members():
    shader = """
    shader DirectStageSampleOffsets {
        sampler2D colorMap;
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            ivec2 pixel @ TEXCOORD2;
            ivec3 pixelLayer @ TEXCOORD3;
            float lod;
            vec2 ddx @ TEXCOORD4;
            vec2 ddy @ TEXCOORD5;
            ivec2 offset @ TEXCOORD6;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 plain = textureOffset(colorMap, linearSampler, input.uv, input.offset);
                vec4 lodSample = textureLodOffset(colorMap, linearSampler, input.uv, input.lod, input.offset);
                vec4 gradSample = textureGradOffset(colorMap, linearSampler, input.uv, input.ddx, input.ddy, input.offset);
                vec4 arrayPlain = textureOffset(layerMap, linearSampler, input.uvLayer, input.offset);
                vec4 arrayLod = textureLodOffset(layerMap, linearSampler, input.uvLayer, input.lod, input.offset);
                vec4 arrayGrad = textureGradOffset(layerMap, linearSampler, input.uvLayer, input.ddx, input.ddy, input.offset);
                vec4 fetched = texelFetchOffset(colorMap, input.pixel, int(input.lod), input.offset);
                vec4 fetchedLayer = texelFetchOffset(layerMap, input.pixelLayer, int(input.lod), input.offset);
                return plain + lodSample + gradSample + arrayPlain + arrayLod + arrayGrad + fetched + fetchedLayer;
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "Texture2DArray layerMap : register(t1);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert (
        "float4 plain = colorMap.Sample(linearSampler, input.uv, input.offset);"
        in generated_code
    )
    assert (
        "float4 lodSample = colorMap.SampleLevel(linearSampler, input.uv, input.lod, input.offset);"
        in generated_code
    )
    assert (
        "float4 gradSample = colorMap.SampleGrad(linearSampler, input.uv, input.ddx, input.ddy, input.offset);"
        in generated_code
    )
    assert (
        "float4 arrayPlain = layerMap.Sample(linearSampler, input.uvLayer, input.offset);"
        in generated_code
    )
    assert (
        "float4 arrayLod = layerMap.SampleLevel(linearSampler, input.uvLayer, input.lod, input.offset);"
        in generated_code
    )
    assert (
        "float4 arrayGrad = layerMap.SampleGrad(linearSampler, input.uvLayer, input.ddx, input.ddy, input.offset);"
        in generated_code
    )
    assert (
        "float4 fetched = colorMap.Load(int3((input.pixel + input.offset), int(input.lod)));"
        in generated_code
    )
    assert (
        "float4 fetchedLayer = layerMap.Load(int4((input.pixelLayer.xy + input.offset), input.pixelLayer.z, int(input.lod)));"
        in generated_code
    )
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code
    assert "texelFetchOffset(" not in generated_code


def test_directx_projected_texture_variants_use_sample_projection():
    shader = """
    shader TextureProjectionVariants {
        sampler2D colorMap;
        sampler3D volumeMap;
        sampler linearSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 xyzq @ TEXCOORD2;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        vec4 implicitProjectedOps(sampler2D tex, vec3 uvq, vec2 ddx, vec2 ddy) {
            vec4 projected = textureProj(tex, uvq);
            vec4 projectedGrad = textureProjGrad(tex, uvq, ddx, ddy);
            return projected + projectedGrad;
        }

        vec4 projectedOps(
            sampler2D tex,
            sampler3D volume,
            sampler s,
            vec3 uvq,
            vec4 uvqw,
            vec4 xyzq,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec4 projected = textureProj(tex, s, uvq);
            vec4 projectedBias = textureProj(tex, s, uvqw, 0.25);
            vec4 volumeProjected = textureProj(volume, s, xyzq);
            vec4 projectedOffset = textureProjOffset(tex, s, uvq, offset);
            vec4 projectedOffsetBias = textureProjOffset(tex, s, uvq, offset, 0.5);
            vec4 projectedLod = textureProjLod(tex, s, uvq, 2.0);
            vec4 projectedLodOffset = textureProjLodOffset(tex, s, uvq, 3.0, offset);
            vec4 projectedGrad = textureProjGrad(tex, s, uvq, ddx, ddy);
            vec4 projectedGradOffset = textureProjGradOffset(tex, s, uvq, ddx, ddy, offset);
            return projected
                + projectedBias
                + volumeProjected
                + projectedOffset
                + projectedOffsetBias
                + projectedLod
                + projectedLodOffset
                + projectedGrad
                + projectedGradOffset;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return implicitProjectedOps(colorMap, input.uvq, input.ddx, input.ddy)
                    + projectedOps(
                        colorMap,
                        volumeMap,
                        linearSampler,
                        input.uvq,
                        input.uvqw,
                        input.xyzq,
                        input.ddx,
                        input.ddy,
                        input.offset
                    );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "Texture3D volumeMap : register(t1);" in generated_code
    assert "SamplerState colorMapSampler : register(s0);" in generated_code
    assert "SamplerState linearSampler : register(s1);" in generated_code
    assert (
        "float4 implicitProjectedOps(Texture2D tex, SamplerState texSampler, float3 uvq, float2 ddx, float2 ddy)"
        in generated_code
    )
    assert (
        "float4 projected = tex.Sample(texSampler, uvq.xy / uvq.z);" in generated_code
    )
    assert (
        "float4 projectedGrad = tex.SampleGrad(texSampler, uvq.xy / uvq.z, ddx, ddy);"
        in generated_code
    )
    assert (
        "float4 projectedOps(Texture2D tex, Texture3D volume, SamplerState s, float3 uvq, float4 uvqw, float4 xyzq, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "float4 projected = tex.Sample(s, uvq.xy / uvq.z);" in generated_code
    assert (
        "float4 projectedBias = tex.SampleBias(s, uvqw.xy / uvqw.w, 0.25);"
        in generated_code
    )
    assert (
        "float4 volumeProjected = volume.Sample(s, xyzq.xyz / xyzq.w);"
        in generated_code
    )
    assert (
        "float4 projectedOffset = tex.Sample(s, uvq.xy / uvq.z, offset);"
        in generated_code
    )
    assert (
        "float4 projectedOffsetBias = tex.SampleBias(s, uvq.xy / uvq.z, 0.5, offset);"
        in generated_code
    )
    assert (
        "float4 projectedLod = tex.SampleLevel(s, uvq.xy / uvq.z, 2.0);"
        in generated_code
    )
    assert (
        "float4 projectedLodOffset = tex.SampleLevel(s, uvq.xy / uvq.z, 3.0, offset);"
        in generated_code
    )
    assert (
        "float4 projectedGrad = tex.SampleGrad(s, uvq.xy / uvq.z, ddx, ddy);"
        in generated_code
    )
    assert (
        "float4 projectedGradOffset = tex.SampleGrad(s, uvq.xy / uvq.z, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "implicitProjectedOps(colorMap, colorMapSampler, input.uvq, input.ddx, input.ddy)"
        in generated_code
    )
    assert (
        "projectedOps(colorMap, volumeMap, linearSampler, input.uvq, input.uvqw, input.xyzq, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureProj(" not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLod(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGrad(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_directx_direct_projected_texture_stage_input_members():
    shader = """
    shader DirectProjectedTexture {
        sampler2D colorMap;
        sampler3D volumeMap;
        sampler linearSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 xyzq @ TEXCOORD2;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 projected = textureProj(colorMap, linearSampler, input.uvq);
                vec4 projectedBias = textureProj(colorMap, linearSampler, input.uvqw, 0.25);
                vec4 volumeProjected = textureProj(volumeMap, linearSampler, input.xyzq);
                vec4 projectedOffset = textureProjOffset(colorMap, linearSampler, input.uvq, input.offset);
                vec4 projectedLod = textureProjLod(colorMap, linearSampler, input.uvq, input.lod);
                vec4 projectedLodOffset = textureProjLodOffset(colorMap, linearSampler, input.uvq, input.lod, input.offset);
                vec4 projectedGrad = textureProjGrad(colorMap, linearSampler, input.uvq, input.ddx, input.ddy);
                vec4 projectedGradOffset = textureProjGradOffset(colorMap, linearSampler, input.uvq, input.ddx, input.ddy, input.offset);
                return projected
                    + projectedBias
                    + volumeProjected
                    + projectedOffset
                    + projectedLod
                    + projectedLodOffset
                    + projectedGrad
                    + projectedGradOffset;
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "Texture3D volumeMap : register(t1);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert (
        "float4 projected = colorMap.Sample(linearSampler, input.uvq.xy / input.uvq.z);"
        in generated_code
    )
    assert (
        "float4 projectedBias = colorMap.SampleBias(linearSampler, input.uvqw.xy / input.uvqw.w, 0.25);"
        in generated_code
    )
    assert (
        "float4 volumeProjected = volumeMap.Sample(linearSampler, input.xyzq.xyz / input.xyzq.w);"
        in generated_code
    )
    assert (
        "float4 projectedOffset = colorMap.Sample(linearSampler, input.uvq.xy / input.uvq.z, input.offset);"
        in generated_code
    )
    assert (
        "float4 projectedLod = colorMap.SampleLevel(linearSampler, input.uvq.xy / input.uvq.z, input.lod);"
        in generated_code
    )
    assert (
        "float4 projectedLodOffset = colorMap.SampleLevel(linearSampler, input.uvq.xy / input.uvq.z, input.lod, input.offset);"
        in generated_code
    )
    assert (
        "float4 projectedGrad = colorMap.SampleGrad(linearSampler, input.uvq.xy / input.uvq.z, input.ddx, input.ddy);"
        in generated_code
    )
    assert (
        "float4 projectedGradOffset = colorMap.SampleGrad(linearSampler, input.uvq.xy / input.uvq.z, input.ddx, input.ddy, input.offset);"
        in generated_code
    )
    assert "unsupported DirectX projected texture" not in generated_code
    assert "textureProj(" not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLod(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGrad(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_directx_direct_projected_array_texture_stage_input_members():
    shader = """
    shader DirectProjectedArrayTexture {
        sampler2DArray layerMap;
        sampler linearSampler;

        struct FSInput {
            vec4 uvLayerQ @ TEXCOORD0;
            float lod;
            vec2 ddx @ TEXCOORD1;
            vec2 ddy @ TEXCOORD2;
            ivec2 offset @ TEXCOORD3;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 projected = textureProj(layerMap, linearSampler, input.uvLayerQ);
                vec4 projectedOffset = textureProjOffset(layerMap, linearSampler, input.uvLayerQ, input.offset);
                vec4 projectedLod = textureProjLod(layerMap, linearSampler, input.uvLayerQ, input.lod);
                vec4 projectedLodOffset = textureProjLodOffset(layerMap, linearSampler, input.uvLayerQ, input.lod, input.offset);
                vec4 projectedGrad = textureProjGrad(layerMap, linearSampler, input.uvLayerQ, input.ddx, input.ddy);
                vec4 projectedGradOffset = textureProjGradOffset(layerMap, linearSampler, input.uvLayerQ, input.ddx, input.ddy, input.offset);
                return projected + projectedOffset + projectedLod + projectedLodOffset + projectedGrad + projectedGradOffset;
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    projected_coord = "float3(input.uvLayerQ.xy / input.uvLayerQ.w, input.uvLayerQ.z)"
    assert "Texture2DArray layerMap : register(t0);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert (
        f"float4 projected = layerMap.Sample(linearSampler, {projected_coord});"
        in generated_code
    )
    assert (
        f"float4 projectedOffset = layerMap.Sample(linearSampler, {projected_coord}, input.offset);"
        in generated_code
    )
    assert (
        f"float4 projectedLod = layerMap.SampleLevel(linearSampler, {projected_coord}, input.lod);"
        in generated_code
    )
    assert (
        f"float4 projectedLodOffset = layerMap.SampleLevel(linearSampler, {projected_coord}, input.lod, input.offset);"
        in generated_code
    )
    assert (
        f"float4 projectedGrad = layerMap.SampleGrad(linearSampler, {projected_coord}, input.ddx, input.ddy);"
        in generated_code
    )
    assert (
        f"float4 projectedGradOffset = layerMap.SampleGrad(linearSampler, {projected_coord}, input.ddx, input.ddy, input.offset);"
        in generated_code
    )
    assert "unsupported DirectX projected texture" not in generated_code
    assert "textureProj(" not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLod(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGrad(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code


def test_directx_implicit_projected_array_texture_stage_input_members_generate_sampler():
    shader = """
    shader DirectImplicitProjectedArrayTexture {
        sampler2DArray layerMap;

        struct FSInput {
            vec4 uvLayerQ @ TEXCOORD0;
            float lod;
            vec2 ddx @ TEXCOORD1;
            vec2 ddy @ TEXCOORD2;
            ivec2 offset @ TEXCOORD3;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 projected = textureProj(layerMap, input.uvLayerQ);
                vec4 projectedOffset = textureProjOffset(layerMap, input.uvLayerQ, input.offset);
                vec4 projectedLod = textureProjLod(layerMap, input.uvLayerQ, input.lod);
                vec4 projectedLodOffset = textureProjLodOffset(layerMap, input.uvLayerQ, input.lod, input.offset);
                vec4 projectedGrad = textureProjGrad(layerMap, input.uvLayerQ, input.ddx, input.ddy);
                vec4 projectedGradOffset = textureProjGradOffset(layerMap, input.uvLayerQ, input.ddx, input.ddy, input.offset);
                return projected + projectedOffset + projectedLod + projectedLodOffset + projectedGrad + projectedGradOffset;
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    projected_coord = "float3(input.uvLayerQ.xy / input.uvLayerQ.w, input.uvLayerQ.z)"
    assert "Texture2DArray layerMap : register(t0);" in generated_code
    assert "SamplerState layerMapSampler : register(s0);" in generated_code
    assert (
        f"float4 projected = layerMap.Sample(layerMapSampler, {projected_coord});"
        in generated_code
    )
    assert (
        f"float4 projectedOffset = layerMap.Sample(layerMapSampler, {projected_coord}, input.offset);"
        in generated_code
    )
    assert (
        f"float4 projectedLod = layerMap.SampleLevel(layerMapSampler, {projected_coord}, input.lod);"
        in generated_code
    )
    assert (
        f"float4 projectedLodOffset = layerMap.SampleLevel(layerMapSampler, {projected_coord}, input.lod, input.offset);"
        in generated_code
    )
    assert (
        f"float4 projectedGrad = layerMap.SampleGrad(layerMapSampler, {projected_coord}, input.ddx, input.ddy);"
        in generated_code
    )
    assert (
        f"float4 projectedGradOffset = layerMap.SampleGrad(layerMapSampler, {projected_coord}, input.ddx, input.ddy, input.offset);"
        in generated_code
    )
    assert "unsupported DirectX projected texture" not in generated_code
    assert "textureProj" not in generated_code


def test_directx_projected_array_texture_resource_arrays_forward_samplers():
    shader = """
    shader ProjectedArrayTextureResourceArrays {
        sampler2DArray layerMaps[4];
        sampler linearSamplers[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec4 uvLayerQ @ TEXCOORD1;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 projectedLeaf(
            sampler2DArray maps[],
            sampler samplers[],
            int layer,
            vec4 uvLayerQ,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec4 fixedProj = textureProj(maps[2], samplers[2], uvLayerQ);
            vec4 dynamicOffset = textureProjOffset(maps[layer], samplers[layer], uvLayerQ, offset);
            vec4 fixedLod = textureProjLod(maps[1], samplers[1], uvLayerQ, lod);
            vec4 dynamicLodOffset = textureProjLodOffset(maps[layer], samplers[layer], uvLayerQ, lod, offset);
            vec4 fixedGrad = textureProjGrad(maps[3], samplers[3], uvLayerQ, ddx, ddy);
            vec4 dynamicGradOffset = textureProjGradOffset(maps[layer], samplers[layer], uvLayerQ, ddx, ddy, offset);
            return fixedProj + dynamicOffset + fixedLod + dynamicLodOffset + fixedGrad + dynamicGradOffset;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return projectedLeaf(
                    layerMaps,
                    linearSamplers,
                    input.layer,
                    input.uvLayerQ,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    projected_coord = "float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z)"
    assert "Texture2DArray layerMaps[4] : register(t0);" in generated_code
    assert "SamplerState linearSamplers[4] : register(s0);" in generated_code
    assert (
        "float4 projectedLeaf(Texture2DArray maps[4], SamplerState samplers[4], int layer, float4 uvLayerQ, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        f"float4 fixedProj = maps[2].Sample(samplers[2], {projected_coord});"
        in generated_code
    )
    assert (
        f"float4 dynamicOffset = maps[layer].Sample(samplers[layer], {projected_coord}, offset);"
        in generated_code
    )
    assert (
        f"float4 fixedLod = maps[1].SampleLevel(samplers[1], {projected_coord}, lod);"
        in generated_code
    )
    assert (
        f"float4 dynamicLodOffset = maps[layer].SampleLevel(samplers[layer], {projected_coord}, lod, offset);"
        in generated_code
    )
    assert (
        f"float4 fixedGrad = maps[3].SampleGrad(samplers[3], {projected_coord}, ddx, ddy);"
        in generated_code
    )
    assert (
        f"float4 dynamicGradOffset = maps[layer].SampleGrad(samplers[layer], {projected_coord}, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "projectedLeaf(layerMaps, linearSamplers, input.layer, input.uvLayerQ, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "unsupported DirectX projected texture" not in generated_code
    assert "textureProj" not in generated_code


def test_directx_implicit_projected_array_texture_resource_arrays_thread_sampler():
    shader = """
    shader ImplicitProjectedArrayTextureResourceArrays {
        sampler2DArray layerMaps[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec4 uvLayerQ @ TEXCOORD1;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 projectedLeaf(
            sampler2DArray maps[],
            int layer,
            vec4 uvLayerQ,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec4 fixedProj = textureProj(maps[2], uvLayerQ);
            vec4 dynamicOffset = textureProjOffset(maps[layer], uvLayerQ, offset);
            vec4 fixedLod = textureProjLod(maps[1], uvLayerQ, lod);
            vec4 dynamicLodOffset = textureProjLodOffset(maps[layer], uvLayerQ, lod, offset);
            vec4 fixedGrad = textureProjGrad(maps[3], uvLayerQ, ddx, ddy);
            vec4 dynamicGradOffset = textureProjGradOffset(maps[layer], uvLayerQ, ddx, ddy, offset);
            return fixedProj + dynamicOffset + fixedLod + dynamicLodOffset + fixedGrad + dynamicGradOffset;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return projectedLeaf(
                    layerMaps,
                    input.layer,
                    input.uvLayerQ,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    projected_coord = "float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z)"
    assert "Texture2DArray layerMaps[4] : register(t0);" in generated_code
    assert "SamplerState layerMapsSampler : register(s0);" in generated_code
    assert (
        "float4 projectedLeaf(Texture2DArray maps[4], SamplerState mapsSampler, int layer, float4 uvLayerQ, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        f"float4 fixedProj = maps[2].Sample(mapsSampler, {projected_coord});"
        in generated_code
    )
    assert (
        f"float4 dynamicOffset = maps[layer].Sample(mapsSampler, {projected_coord}, offset);"
        in generated_code
    )
    assert (
        f"float4 fixedLod = maps[1].SampleLevel(mapsSampler, {projected_coord}, lod);"
        in generated_code
    )
    assert (
        f"float4 dynamicLodOffset = maps[layer].SampleLevel(mapsSampler, {projected_coord}, lod, offset);"
        in generated_code
    )
    assert (
        f"float4 fixedGrad = maps[3].SampleGrad(mapsSampler, {projected_coord}, ddx, ddy);"
        in generated_code
    )
    assert (
        f"float4 dynamicGradOffset = maps[layer].SampleGrad(mapsSampler, {projected_coord}, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "projectedLeaf(layerMaps, layerMapsSampler, input.layer, input.uvLayerQ, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "unsupported DirectX projected texture" not in generated_code
    assert "textureProj" not in generated_code


def test_directx_implicit_projected_stage_input_members_generate_samplers():
    shader = """
    shader DirectImplicitProjection {
        sampler2D colorMap;
        sampler3D volumeMap;
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 xyzq @ TEXCOORD2;
            vec4 uvLayerQ @ TEXCOORD3;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD4;
            vec2 ddy @ TEXCOORD5;
            ivec2 offset @ TEXCOORD6;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 color = textureProj(colorMap, input.uvq);
                vec4 colorLod = textureProjLod(colorMap, input.uvqw, input.lod);
                vec4 volume = textureProj(volumeMap, input.xyzq);
                float shadow = textureCompareProj(shadowMap, input.uvq, input.depth);
                float shadowGrad = textureCompareProjGrad(shadowArray, input.uvLayerQ, input.depth, input.ddx, input.ddy);
                return color + colorLod + volume + vec4(shadow + shadowGrad);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "SamplerState colorMapSampler : register(s0);" in generated_code
    assert "Texture3D volumeMap : register(t1);" in generated_code
    assert "SamplerState volumeMapSampler : register(s1);" in generated_code
    assert "Texture2D shadowMap : register(t2);" in generated_code
    assert "SamplerComparisonState shadowMapSampler : register(s2);" in generated_code
    assert "Texture2DArray shadowArray : register(t3);" in generated_code
    assert "SamplerComparisonState shadowArraySampler : register(s3);" in generated_code
    assert (
        "float4 color = colorMap.Sample(colorMapSampler, input.uvq.xy / input.uvq.z);"
        in generated_code
    )
    assert (
        "float4 colorLod = colorMap.SampleLevel(colorMapSampler, input.uvqw.xy / input.uvqw.w, input.lod);"
        in generated_code
    )
    assert (
        "float4 volume = volumeMap.Sample(volumeMapSampler, input.xyzq.xyz / input.xyzq.w);"
        in generated_code
    )
    assert (
        "float shadow = shadowMap.SampleCmp(shadowMapSampler, input.uvq.xy / input.uvq.z, input.depth);"
        in generated_code
    )
    assert (
        "float shadowGrad = shadowArray.SampleCmpGrad(shadowArraySampler, float3(input.uvLayerQ.xy / input.uvLayerQ.w, input.uvLayerQ.z), input.depth, input.ddx, input.ddy);"
        in generated_code
    )
    assert "unsupported DirectX projected texture" not in generated_code
    assert "unsupported DirectX texture compare" not in generated_code
    assert "textureProj" not in generated_code
    assert "textureCompareProj" not in generated_code


def test_directx_projected_shadow_compare_variants_use_sample_cmp_projection():
    shader = """
    shader ProjectedShadowCompareVariants {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 uvLayerQ @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        float implicitProjectedShadow(
            sampler2DShadow tex,
            vec3 uvq,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, uvq, depth);
            float lodOffsetProjected = textureCompareProjLodOffset(tex, uvq, depth, lod, offset);
            float gradProjected = textureCompareProjGrad(tex, uvq, depth, ddx, ddy);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, uvq, depth, ddx, ddy, offset);
            return projected + lodOffsetProjected + gradProjected + gradOffsetProjected;
        }

        float projectedShadow(
            sampler2DShadow tex,
            sampler s,
            vec3 uvq,
            vec4 uvqw,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, s, uvq, depth);
            float projectedW = textureCompareProj(tex, s, uvqw, depth);
            float offsetProjected = textureCompareProjOffset(tex, s, uvq, depth, offset);
            float lodProjected = textureCompareProjLod(tex, s, uvq, depth, lod);
            float lodOffsetProjected = textureCompareProjLodOffset(tex, s, uvq, depth, lod, offset);
            float gradProjected = textureCompareProjGrad(tex, s, uvq, depth, ddx, ddy);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, s, uvq, depth, ddx, ddy, offset);
            return projected + projectedW + offsetProjected + lodProjected + lodOffsetProjected + gradProjected + gradOffsetProjected;
        }

        float projectedArrayShadow(
            sampler2DArrayShadow tex,
            sampler s,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, s, uvLayerQ, depth);
            float offsetProjected = textureCompareProjOffset(tex, s, uvLayerQ, depth, offset);
            float lodProjected = textureCompareProjLod(tex, s, uvLayerQ, depth, lod);
            float lodOffsetProjected = textureCompareProjLodOffset(tex, s, uvLayerQ, depth, lod, offset);
            float gradProjected = textureCompareProjGrad(tex, s, uvLayerQ, depth, ddx, ddy);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, s, uvLayerQ, depth, ddx, ddy, offset);
            return projected + offsetProjected + lodProjected + lodOffsetProjected + gradProjected + gradOffsetProjected;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float implicitValue = implicitProjectedShadow(
                    shadowMap,
                    input.uvq,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float explicitValue = projectedShadow(
                    shadowMap,
                    compareSampler,
                    input.uvq,
                    input.uvqw,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float arrayValue = projectedArrayShadow(
                    shadowArray,
                    compareSampler,
                    input.uvLayerQ,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                return vec4(implicitValue + explicitValue + arrayValue);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "Texture2DArray shadowArray : register(t1);" in generated_code
    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert "SamplerComparisonState compareSampler : register(s1);" in generated_code
    assert (
        "float implicitProjectedShadow(Texture2D tex, SamplerComparisonState texSampler, float3 uvq, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float projected = tex.SampleCmp(texSampler, uvq.xy / uvq.z, depth);"
        in generated_code
    )
    assert (
        "float lodOffsetProjected = tex.SampleCmpLevel(texSampler, uvq.xy / uvq.z, depth, lod, offset);"
        in generated_code
    )
    assert (
        "float gradProjected = tex.SampleCmpGrad(texSampler, uvq.xy / uvq.z, depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetProjected = tex.SampleCmpGrad(texSampler, uvq.xy / uvq.z, depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float projectedShadow(Texture2D tex, SamplerComparisonState s, float3 uvq, float4 uvqw, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float projected = tex.SampleCmp(s, uvq.xy / uvq.z, depth);" in generated_code
    )
    assert (
        "float projectedW = tex.SampleCmp(s, uvqw.xy / uvqw.w, depth);"
        in generated_code
    )
    assert (
        "float offsetProjected = tex.SampleCmp(s, uvq.xy / uvq.z, depth, offset);"
        in generated_code
    )
    assert (
        "float lodProjected = tex.SampleCmpLevel(s, uvq.xy / uvq.z, depth, lod);"
        in generated_code
    )
    assert (
        "float lodOffsetProjected = tex.SampleCmpLevel(s, uvq.xy / uvq.z, depth, lod, offset);"
        in generated_code
    )
    assert (
        "float gradProjected = tex.SampleCmpGrad(s, uvq.xy / uvq.z, depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetProjected = tex.SampleCmpGrad(s, uvq.xy / uvq.z, depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "implicitProjectedShadow(shadowMap, shadowMapSampler, input.uvq, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "projectedShadow(shadowMap, compareSampler, input.uvq, input.uvqw, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "float projectedArrayShadow(Texture2DArray tex, SamplerComparisonState s, float4 uvLayerQ, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float projected = tex.SampleCmp(s, float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth);"
        in generated_code
    )
    assert (
        "float offsetProjected = tex.SampleCmp(s, float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, offset);"
        in generated_code
    )
    assert (
        "float lodProjected = tex.SampleCmpLevel(s, float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, lod);"
        in generated_code
    )
    assert (
        "float lodOffsetProjected = tex.SampleCmpLevel(s, float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, lod, offset);"
        in generated_code
    )
    assert (
        "float gradProjected = tex.SampleCmpGrad(s, float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetProjected = tex.SampleCmpGrad(s, float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "projectedArrayShadow(shadowArray, compareSampler, input.uvLayerQ, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureCompareProj" not in generated_code


def test_directx_direct_projected_shadow_compare_stage_input_members():
    shader = """
    shader DirectProjectedShadowCompare {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvqw @ TEXCOORD1;
            vec4 uvLayerQ @ TEXCOORD2;
            float depth;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float planar = textureCompareProj(shadowMap, compareSampler, input.uvq, input.depth);
                float planarOffset = textureCompareProjOffset(shadowMap, compareSampler, input.uvqw, input.depth, input.offset);
                float arrayGrad = textureCompareProjGrad(shadowArray, compareSampler, input.uvLayerQ, input.depth, input.ddx, input.ddy);
                return vec4(planar + planarOffset + arrayGrad);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "Texture2DArray shadowArray : register(t1);" in generated_code
    assert "SamplerComparisonState compareSampler : register(s0);" in generated_code
    assert (
        "float planar = shadowMap.SampleCmp(compareSampler, input.uvq.xy / input.uvq.z, input.depth);"
        in generated_code
    )
    assert (
        "float planarOffset = shadowMap.SampleCmp(compareSampler, input.uvqw.xy / input.uvqw.w, input.depth, input.offset);"
        in generated_code
    )
    assert (
        "float arrayGrad = shadowArray.SampleCmpGrad(compareSampler, float3(input.uvLayerQ.xy / input.uvLayerQ.w, input.uvLayerQ.z), input.depth, input.ddx, input.ddy);"
        in generated_code
    )
    assert "unsupported DirectX texture compare" not in generated_code
    assert "textureCompareProj" not in generated_code


def test_directx_projected_shadow_compare_resource_arrays_forward_comparison_samplers():
    shader = """
    shader ProjectedShadowResourceArrays {
        sampler2DShadow shadowMaps[4];
        sampler2DArrayShadow shadowArrays[4];
        sampler shadowSamplers[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec3 uvq @ TEXCOORD1;
            vec4 uvLayerQ @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        float projectedLeaf(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            sampler shadowSamplers[],
            int layer,
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float planar = textureCompareProj(shadowMaps[layer], shadowSamplers[layer], uvq, depth);
            float planarOffset = textureCompareProjOffset(shadowMaps[1], shadowSamplers[1], uvq, depth, offset);
            float planarLod = textureCompareProjLod(shadowMaps[2], shadowSamplers[2], uvq, depth, lod);
            float planarGradOffset = textureCompareProjGradOffset(shadowMaps[layer], shadowSamplers[layer], uvq, depth, ddx, ddy, offset);
            float arrayProjected = textureCompareProj(shadowArrays[2], shadowSamplers[2], uvLayerQ, depth);
            float arrayOffset = textureCompareProjOffset(shadowArrays[layer], shadowSamplers[layer], uvLayerQ, depth, offset);
            float arrayGrad = textureCompareProjGrad(shadowArrays[1], shadowSamplers[1], uvLayerQ, depth, ddx, ddy);
            return planar + planarOffset + planarLod + planarGradOffset + arrayProjected + arrayOffset + arrayGrad;
        }

        float projectedWrapper(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            sampler shadowSamplers[],
            int layer,
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            return projectedLeaf(shadowMaps, shadowArrays, shadowSamplers, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return projectedWrapper(
                    shadowMaps,
                    shadowArrays,
                    shadowSamplers,
                    input.layer,
                    input.uvq,
                    input.uvLayerQ,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "Texture2DArray shadowArrays[4] : register(t4);" in generated_code
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in generated_code
    assert (
        "float projectedLeaf(Texture2D shadowMaps[4], Texture2DArray shadowArrays[4], SamplerComparisonState shadowSamplers[4], int layer, float3 uvq, float4 uvLayerQ, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float planar = shadowMaps[layer].SampleCmp(shadowSamplers[layer], uvq.xy / uvq.z, depth);"
        in generated_code
    )
    assert (
        "float planarOffset = shadowMaps[1].SampleCmp(shadowSamplers[1], uvq.xy / uvq.z, depth, offset);"
        in generated_code
    )
    assert (
        "float planarLod = shadowMaps[2].SampleCmpLevel(shadowSamplers[2], uvq.xy / uvq.z, depth, lod);"
        in generated_code
    )
    assert (
        "float planarGradOffset = shadowMaps[layer].SampleCmpGrad(shadowSamplers[layer], uvq.xy / uvq.z, depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float arrayProjected = shadowArrays[2].SampleCmp(shadowSamplers[2], float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth);"
        in generated_code
    )
    assert (
        "float arrayOffset = shadowArrays[layer].SampleCmp(shadowSamplers[layer], float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, offset);"
        in generated_code
    )
    assert (
        "float arrayGrad = shadowArrays[1].SampleCmpGrad(shadowSamplers[1], float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "float projectedWrapper(Texture2D shadowMaps[4], Texture2DArray shadowArrays[4], SamplerComparisonState shadowSamplers[4], int layer, float3 uvq, float4 uvLayerQ, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "projectedLeaf(shadowMaps, shadowArrays, shadowSamplers, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedWrapper(shadowMaps, shadowArrays, shadowSamplers, input.layer, input.uvq, input.uvLayerQ, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureCompareProj" not in generated_code


def test_directx_implicit_projected_shadow_compare_resource_arrays_thread_samplers():
    shader = """
    shader ImplicitProjectedShadowResourceArrays {
        sampler2DShadow shadowMaps[4];
        sampler2DArrayShadow shadowArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec3 uvq @ TEXCOORD1;
            vec4 uvLayerQ @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        float projectedLeaf(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            int layer,
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float planar = textureCompareProj(shadowMaps[layer], uvq, depth);
            float planarOffset = textureCompareProjOffset(shadowMaps[1], uvq, depth, offset);
            float planarLod = textureCompareProjLod(shadowMaps[2], uvq, depth, lod);
            float planarGradOffset = textureCompareProjGradOffset(shadowMaps[layer], uvq, depth, ddx, ddy, offset);
            float arrayProjected = textureCompareProj(shadowArrays[2], uvLayerQ, depth);
            float arrayOffset = textureCompareProjOffset(shadowArrays[layer], uvLayerQ, depth, offset);
            float arrayGrad = textureCompareProjGrad(shadowArrays[1], uvLayerQ, depth, ddx, ddy);
            return planar + planarOffset + planarLod + planarGradOffset + arrayProjected + arrayOffset + arrayGrad;
        }

        float projectedWrapper(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            int layer,
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            return projectedLeaf(shadowMaps, shadowArrays, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return projectedWrapper(
                    shadowMaps,
                    shadowArrays,
                    input.layer,
                    input.uvq,
                    input.uvLayerQ,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowMapsSampler : register(s0);" in generated_code
    assert "Texture2DArray shadowArrays[4] : register(t4);" in generated_code
    assert (
        "SamplerComparisonState shadowArraysSampler : register(s1);" in generated_code
    )
    assert (
        "float projectedLeaf(Texture2D shadowMaps[4], SamplerComparisonState shadowMapsSampler, Texture2DArray shadowArrays[4], SamplerComparisonState shadowArraysSampler, int layer, float3 uvq, float4 uvLayerQ, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float planar = shadowMaps[layer].SampleCmp(shadowMapsSampler, uvq.xy / uvq.z, depth);"
        in generated_code
    )
    assert (
        "float planarOffset = shadowMaps[1].SampleCmp(shadowMapsSampler, uvq.xy / uvq.z, depth, offset);"
        in generated_code
    )
    assert (
        "float planarLod = shadowMaps[2].SampleCmpLevel(shadowMapsSampler, uvq.xy / uvq.z, depth, lod);"
        in generated_code
    )
    assert (
        "float planarGradOffset = shadowMaps[layer].SampleCmpGrad(shadowMapsSampler, uvq.xy / uvq.z, depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float arrayProjected = shadowArrays[2].SampleCmp(shadowArraysSampler, float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth);"
        in generated_code
    )
    assert (
        "float arrayOffset = shadowArrays[layer].SampleCmp(shadowArraysSampler, float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, offset);"
        in generated_code
    )
    assert (
        "float arrayGrad = shadowArrays[1].SampleCmpGrad(shadowArraysSampler, float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "float projectedWrapper(Texture2D shadowMaps[4], SamplerComparisonState shadowMapsSampler, Texture2DArray shadowArrays[4], SamplerComparisonState shadowArraysSampler, int layer, float3 uvq, float4 uvLayerQ, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "projectedLeaf(shadowMaps, shadowMapsSampler, shadowArrays, shadowArraysSampler, layer, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "projectedWrapper(shadowMaps, shadowMapsSampler, shadowArrays, shadowArraysSampler, input.layer, input.uvq, input.uvLayerQ, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "SamplerComparisonState shadowMapsSampler[" not in generated_code
    assert "SamplerComparisonState shadowArraysSampler[" not in generated_code
    assert "textureCompareProj" not in generated_code


def test_directx_unsized_projected_shadow_compare_arrays_infer_transitive_constant_size():
    shader = """
    shader UnsizedProjectedShadowResources {
        const int LAYER = 4;
        sampler2DShadow shadowMaps[];
        sampler2DArrayShadow shadowArrays[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct FSInput {
            vec3 uvq @ TEXCOORD0;
            vec4 uvLayerQ @ TEXCOORD1;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        float shadowDeep(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            sampler shadowSamplers[],
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float planarHigh = textureCompareProj(shadowMaps[LAYER], shadowSamplers[LAYER], uvq, depth);
            float planarLow = textureCompareProjOffset(shadowMaps[1], shadowSamplers[1], uvq, depth, offset);
            float arrayHigh = textureCompareProjGrad(shadowArrays[2 * 2], shadowSamplers[2 * 2], uvLayerQ, depth, ddx, ddy);
            float arrayLow = textureCompareProjOffset(shadowArrays[3], shadowSamplers[3], uvLayerQ, depth, offset);
            return planarHigh + planarLow + arrayHigh + arrayLow;
        }

        float shadowMid(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            sampler shadowSamplers[],
            vec3 uvq,
            vec4 uvLayerQ,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            return shadowDeep(shadowMaps, shadowArrays, shadowSamplers, uvq, uvLayerQ, depth, lod, ddx, ddy, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float arrayShadow = shadowMid(
                    shadowMaps,
                    shadowArrays,
                    shadowSamplers,
                    input.uvq,
                    input.uvLayerQ,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uvq.xy, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMaps[5] : register(t0);" in generated_code
    assert "Texture2DArray shadowArrays[5] : register(t5);" in generated_code
    assert "SamplerComparisonState shadowSamplers[5] : register(s0);" in generated_code
    assert "Texture2D afterShadow : register(t10);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s5);" in generated_code
    assert (
        "float shadowDeep(Texture2D shadowMaps[5], Texture2DArray shadowArrays[5], SamplerComparisonState shadowSamplers[5], float3 uvq, float4 uvLayerQ, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float planarHigh = shadowMaps[LAYER].SampleCmp(shadowSamplers[LAYER], uvq.xy / uvq.z, depth);"
        in generated_code
    )
    assert (
        "float planarLow = shadowMaps[1].SampleCmp(shadowSamplers[1], uvq.xy / uvq.z, depth, offset);"
        in generated_code
    )
    assert (
        "float arrayHigh = shadowArrays[(2 * 2)].SampleCmpGrad(shadowSamplers[(2 * 2)], float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "float arrayLow = shadowArrays[3].SampleCmp(shadowSamplers[3], float3(uvLayerQ.xy / uvLayerQ.w, uvLayerQ.z), depth, offset);"
        in generated_code
    )
    assert (
        "float shadowMid(Texture2D shadowMaps[5], Texture2DArray shadowArrays[5], SamplerComparisonState shadowSamplers[5], float3 uvq, float4 uvLayerQ, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "shadowDeep(shadowMaps, shadowArrays, shadowSamplers, uvq, uvLayerQ, depth, lod, ddx, ddy, offset)"
        in generated_code
    )
    assert (
        "shadowMid(shadowMaps, shadowArrays, shadowSamplers, input.uvq, input.uvLayerQ, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "float singleShadow = afterShadow.SampleCmp(afterSampler, input.uvq.xy, input.depth);"
        in generated_code
    )
    assert "Texture2D shadowMaps[] : register(t0);" not in generated_code
    assert "SamplerState shadowSamplers[]" not in generated_code
    assert "textureCompareProj" not in generated_code


def test_directx_projected_cube_shadow_compare_lowers_supported_forms():
    shader = """
    shader ProjectedCubeShadowDiagnostics {
        samplerCubeShadow cubeMap;
        samplerCubeArrayShadow cubeArray;
        sampler compareSampler;

        struct FSInput {
            vec4 cubeProj @ TEXCOORD0;
            vec4 cubeLayerProj @ TEXCOORD1;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        float cubeProjected(
            samplerCubeShadow tex,
            sampler s,
            vec4 cubeProj,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, s, cubeProj, depth);
            float offsetProjected = textureCompareProjOffset(tex, s, cubeProj, depth, offset);
            float lodOffsetProjected = textureCompareProjLodOffset(tex, s, cubeProj, depth, lod, offset);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, s, cubeProj, depth, ddx, ddy, offset);
            return projected + offsetProjected + lodOffsetProjected + gradOffsetProjected;
        }

        float cubeArrayProjected(
            samplerCubeArrayShadow tex,
            sampler s,
            vec4 cubeLayerProj,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, s, cubeLayerProj, depth);
            float offsetProjected = textureCompareProjOffset(tex, s, cubeLayerProj, depth, offset);
            float lodOffsetProjected = textureCompareProjLodOffset(tex, s, cubeLayerProj, depth, lod, offset);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, s, cubeLayerProj, depth, ddx, ddy, offset);
            return projected + offsetProjected + lodOffsetProjected + gradOffsetProjected;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return cubeProjected(
                    cubeMap,
                    compareSampler,
                    input.cubeProj,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + cubeArrayProjected(
                    cubeArray,
                    compareSampler,
                    input.cubeLayerProj,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "TextureCube cubeMap : register(t0);" in generated_code
    assert "TextureCubeArray cubeArray : register(t1);" in generated_code
    assert "SamplerComparisonState compareSampler : register(s0);" in generated_code
    assert "SamplerState compareSampler" not in generated_code
    assert (
        "float cubeProjected(TextureCube tex, SamplerComparisonState s, float4 cubeProj, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float cubeArrayProjected(TextureCubeArray tex, SamplerComparisonState s, float4 cubeLayerProj, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float projected = tex.SampleCmp(s, cubeProj.xyz / cubeProj.w, depth);"
        in generated_code
    )
    for func_name in {
        "textureCompareProjOffset",
        "textureCompareProjLodOffset",
        "textureCompareProjGradOffset",
    }:
        assert (
            generated_code.count(
                f"/* unsupported DirectX texture compare: {func_name} offsets require 2D or 2D-array textures */ 0.0"
            )
            == 1
        )
    coordinate_diagnostic_counts = {
        "textureCompareProj": 1,
        "textureCompareProjOffset": 1,
        "textureCompareProjLodOffset": 1,
        "textureCompareProjGradOffset": 1,
    }
    for func_name, count in coordinate_diagnostic_counts.items():
        assert (
            generated_code.count(
                f"/* unsupported DirectX texture compare: {func_name} requires Texture2D vec3/vec4 or Texture2DArray vec4 projection coordinates */ 0.0"
            )
            == count
        )
    assert "textureCompareProj(" not in generated_code
    assert "textureCompareProjOffset(" not in generated_code
    assert "textureCompareProjLodOffset(" not in generated_code
    assert "textureCompareProjGradOffset(" not in generated_code
    assert ".SampleCmp(s, cubeProj" in generated_code
    assert ".SampleCmp(s, cubeLayerProj" not in generated_code
    assert ".SampleCmpLevel(s, cubeProj" not in generated_code
    assert ".SampleCmpGrad(s, cubeProj" not in generated_code


def test_directx_projected_cube_shadow_compare_implicit_samplers_lowers_supported_forms():
    shader = """
    shader ProjectedCubeShadowImplicitDiagnostics {
        samplerCubeShadow cubeMap;
        samplerCubeArrayShadow cubeArray;

        struct FSInput {
            vec4 cubeProj @ TEXCOORD0;
            vec4 cubeArrayProj @ TEXCOORD1;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 cubeProjected(
            samplerCubeShadow tex,
            vec4 cubeProj,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, cubeProj, depth);
            float lodProjected = textureCompareProjLod(tex, cubeProj, depth, lod);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, cubeProj, depth, ddx, ddy, offset);
            return vec4(projected + lodProjected + gradOffsetProjected);
        }

        vec4 cubeArrayProjected(
            samplerCubeArrayShadow tex,
            vec4 cubeArrayProj,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float projected = textureCompareProj(tex, cubeArrayProj, depth);
            float lodProjected = textureCompareProjLod(tex, cubeArrayProj, depth, lod);
            float gradOffsetProjected = textureCompareProjGradOffset(tex, cubeArrayProj, depth, ddx, ddy, offset);
            return vec4(projected + lodProjected + gradOffsetProjected);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float globalProjected = textureCompareProj(cubeMap, input.cubeProj, input.depth);
                float globalArrayProjected = textureCompareProjLod(cubeArray, input.cubeArrayProj, input.depth, input.lod);
                return vec4(globalProjected + globalArrayProjected)
                    + cubeProjected(
                        cubeMap,
                        input.cubeProj,
                        input.depth,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    )
                    + cubeArrayProjected(
                        cubeArray,
                        input.cubeArrayProj,
                        input.depth,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "TextureCube cubeMap : register(t0);" in generated_code
    assert "SamplerComparisonState cubeMapSampler : register(s0);" in generated_code
    assert "TextureCubeArray cubeArray : register(t1);" in generated_code
    assert (
        "float4 cubeProjected(TextureCube tex, SamplerComparisonState texSampler, float4 cubeProj, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float4 cubeArrayProjected(TextureCubeArray tex, float4 cubeArrayProj, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float projected = tex.SampleCmp(texSampler, cubeProj.xyz / cubeProj.w, depth);"
        in generated_code
    )
    assert (
        "float lodProjected = tex.SampleCmpLevel(texSampler, cubeProj.xyz / cubeProj.w, depth, lod);"
        in generated_code
    )
    assert (
        "float globalProjected = cubeMap.SampleCmp(cubeMapSampler, input.cubeProj.xyz / input.cubeProj.w, input.depth);"
        in generated_code
    )
    assert (
        "/* unsupported DirectX texture compare: textureCompareProjGradOffset offsets require 2D or 2D-array textures */ 0.0"
        in generated_code
    )
    coordinate_diagnostic_counts = {
        "textureCompareProj": 1,
        "textureCompareProjLod": 2,
        "textureCompareProjGradOffset": 1,
    }
    for func_name, count in coordinate_diagnostic_counts.items():
        assert (
            generated_code.count(
                f"/* unsupported DirectX texture compare: {func_name} requires Texture2D vec3/vec4 or Texture2DArray vec4 projection coordinates */ 0.0"
            )
            == count
        )
    assert (
        "cubeProjected(cubeMap, cubeMapSampler, input.cubeProj, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "cubeArrayProjected(cubeArray, input.cubeArrayProj, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "SamplerComparisonState cubeArraySampler" not in generated_code


def test_directx_projected_cube_shadow_compare_resource_arrays_thread_supported_sampler():
    shader = """
    shader ProjectedCubeShadowArrayDiagnostics {
        samplerCubeShadow cubeMaps[4];
        samplerCubeArrayShadow cubeArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec4 cubeProj @ TEXCOORD1;
            vec4 cubeArrayProj @ TEXCOORD2;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD3;
            vec3 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        vec4 cubeProjected(
            samplerCubeShadow maps[],
            int layer,
            vec4 cubeProj,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float fixedProjected = textureCompareProj(maps[2], cubeProj, depth);
            float dynamicLodProjected = textureCompareProjLod(maps[layer], cubeProj, depth, lod);
            float dynamicGradOffsetProjected = textureCompareProjGradOffset(maps[layer], cubeProj, depth, ddx, ddy, offset);
            return vec4(fixedProjected + dynamicLodProjected + dynamicGradOffsetProjected);
        }

        vec4 cubeArrayProjected(
            samplerCubeArrayShadow maps[],
            int layer,
            vec4 cubeArrayProj,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float fixedProjected = textureCompareProj(maps[2], cubeArrayProj, depth);
            float dynamicLodProjected = textureCompareProjLod(maps[layer], cubeArrayProj, depth, lod);
            float dynamicGradOffsetProjected = textureCompareProjGradOffset(maps[layer], cubeArrayProj, depth, ddx, ddy, offset);
            return vec4(fixedProjected + dynamicLodProjected + dynamicGradOffsetProjected);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float globalProjected = textureCompareProj(cubeMaps[input.layer], input.cubeProj, input.depth);
                float globalArrayProjected = textureCompareProjLod(cubeArrays[2], input.cubeArrayProj, input.depth, input.lod);
                return vec4(globalProjected + globalArrayProjected)
                    + cubeProjected(
                        cubeMaps,
                        input.layer,
                        input.cubeProj,
                        input.depth,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    )
                    + cubeArrayProjected(
                        cubeArrays,
                        input.layer,
                        input.cubeArrayProj,
                        input.depth,
                        input.lod,
                        input.ddx,
                        input.ddy,
                        input.offset
                    );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "TextureCube cubeMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState cubeMapsSampler : register(s0);" in generated_code
    assert "TextureCubeArray cubeArrays[4] : register(t4);" in generated_code
    assert (
        "float4 cubeProjected(TextureCube maps[4], SamplerComparisonState mapsSampler, int layer, float4 cubeProj, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float4 cubeArrayProjected(TextureCubeArray maps[4], int layer, float4 cubeArrayProj, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float fixedProjected = maps[2].SampleCmp(mapsSampler, cubeProj.xyz / cubeProj.w, depth);"
        in generated_code
    )
    assert (
        "float dynamicLodProjected = maps[layer].SampleCmpLevel(mapsSampler, cubeProj.xyz / cubeProj.w, depth, lod);"
        in generated_code
    )
    assert (
        "float globalProjected = cubeMaps[input.layer].SampleCmp(cubeMapsSampler, input.cubeProj.xyz / input.cubeProj.w, input.depth);"
        in generated_code
    )
    assert (
        "/* unsupported DirectX texture compare: textureCompareProjGradOffset offsets require 2D or 2D-array textures */ 0.0"
        in generated_code
    )
    coordinate_diagnostic_counts = {
        "textureCompareProj": 1,
        "textureCompareProjLod": 2,
        "textureCompareProjGradOffset": 1,
    }
    for func_name, count in coordinate_diagnostic_counts.items():
        assert (
            generated_code.count(
                f"/* unsupported DirectX texture compare: {func_name} requires Texture2D vec3/vec4 or Texture2DArray vec4 projection coordinates */ 0.0"
            )
            == count
        )
    assert (
        "cubeProjected(cubeMaps, cubeMapsSampler, input.layer, input.cubeProj, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "cubeArrayProjected(cubeArrays, input.layer, input.cubeArrayProj, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "SamplerComparisonState cubeArraysSampler" not in generated_code


def test_directx_direct_projected_cube_texture_lowers_supported_color_forms():
    shader = """
    shader DirectProjectedCubeDiagnostics {
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler linearSampler;

        struct FSInput {
            vec4 cubeProj @ TEXCOORD0;
            vec4 cubeArrayProj @ TEXCOORD1;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 cubeProjected = textureProj(cubeMap, linearSampler, input.cubeProj);
                vec4 cubeLod = textureProjLod(cubeMap, linearSampler, input.cubeProj, input.lod);
                vec4 cubeGrad = textureProjGrad(cubeMap, linearSampler, input.cubeProj, input.ddx, input.ddy);
                vec4 cubeLodOffset = textureProjLodOffset(cubeMap, linearSampler, input.cubeProj, input.lod, input.offset);
                vec4 cubeGradOffset = textureProjGradOffset(cubeMap, linearSampler, input.cubeProj, input.ddx, input.ddy, input.offset);
                vec4 cubeArrayProjected = textureProj(cubeArray, linearSampler, input.cubeArrayProj);
                vec4 cubeArrayLodOffset = textureProjLodOffset(cubeArray, linearSampler, input.cubeArrayProj, input.lod, input.offset);
                vec4 cubeArrayGradOffset = textureProjGradOffset(cubeArray, linearSampler, input.cubeArrayProj, input.ddx, input.ddy, input.offset);
                vec4 implicitCube = textureProj(cubeMap, input.cubeProj);
                vec4 implicitCubeGrad = textureProjGrad(cubeMap, input.cubeProj, input.ddx, input.ddy);
                vec4 implicitCubeArray = textureProjLod(cubeArray, input.cubeArrayProj, input.lod);
                return cubeProjected + cubeLod + cubeGrad + cubeLodOffset + cubeGradOffset + cubeArrayProjected + cubeArrayLodOffset + cubeArrayGradOffset + implicitCube + implicitCubeGrad + implicitCubeArray;
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "TextureCube cubeMap : register(t0);" in generated_code
    assert "TextureCubeArray cubeArray : register(t1);" in generated_code
    assert "SamplerState cubeMapSampler : register(s0);" in generated_code
    assert "SamplerState linearSampler : register(s1);" in generated_code
    assert "SamplerState cubeArraySampler" not in generated_code
    assert (
        "float4 cubeProjected = cubeMap.Sample(linearSampler, input.cubeProj.xyz / input.cubeProj.w);"
        in generated_code
    )
    assert (
        "float4 cubeLod = cubeMap.SampleLevel(linearSampler, input.cubeProj.xyz / input.cubeProj.w, input.lod);"
        in generated_code
    )
    assert (
        "float4 cubeGrad = cubeMap.SampleGrad(linearSampler, input.cubeProj.xyz / input.cubeProj.w, input.ddx, input.ddy);"
        in generated_code
    )
    assert (
        "float4 implicitCube = cubeMap.Sample(cubeMapSampler, input.cubeProj.xyz / input.cubeProj.w);"
        in generated_code
    )
    assert (
        "float4 implicitCubeGrad = cubeMap.SampleGrad(cubeMapSampler, input.cubeProj.xyz / input.cubeProj.w, input.ddx, input.ddy);"
        in generated_code
    )
    assert (
        "/* unsupported DirectX projected texture: textureProjLodOffset offsets require 1D, 2D, 2D-array, or 3D textures */ float4(0.0, 0.0, 0.0, 0.0)"
        in generated_code
    )
    assert (
        "/* unsupported DirectX projected texture: textureProjGradOffset offsets require 1D, 2D, 2D-array, or 3D textures */ float4(0.0, 0.0, 0.0, 0.0)"
        in generated_code
    )
    coordinate_diagnostic_counts = {
        "textureProj": 1,
        "textureProjLodOffset": 1,
        "textureProjGradOffset": 1,
        "textureProjLod": 1,
    }
    for func_name, count in coordinate_diagnostic_counts.items():
        assert (
            generated_code.count(
                f"/* unsupported DirectX projected texture: {func_name} requires 1D, 2D, or 3D projection coordinates */ float4(0.0, 0.0, 0.0, 0.0)"
            )
            == count
        )
    assert "textureProj(cubeMap" not in generated_code
    assert "textureProj(cubeArray" not in generated_code
    assert ".Sample(linearSampler, input.cubeArrayProj" not in generated_code


def test_directx_projected_cube_texture_parameters_forward_supported_implicit_sampler():
    shader = """
    shader ProjectedCubeParameterDiagnostics {
        samplerCube cubeMap;
        samplerCubeArray cubeArray;

        struct FSInput {
            vec4 cubeProj @ TEXCOORD0;
            vec4 cubeArrayProj @ TEXCOORD1;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 cubeProjected(
            samplerCube tex,
            vec4 cubeProj,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            vec4 projected = textureProj(tex, cubeProj);
            vec4 lodProjected = textureProjLod(tex, cubeProj, lod);
            vec4 gradOffsetProjected = textureProjGradOffset(tex, cubeProj, ddx, ddy, offset);
            return projected + lodProjected + gradOffsetProjected;
        }

        vec4 cubeArrayProjected(
            samplerCubeArray tex,
            vec4 cubeArrayProj,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            vec4 projected = textureProj(tex, cubeArrayProj);
            vec4 lodProjected = textureProjLod(tex, cubeArrayProj, lod);
            vec4 gradOffsetProjected = textureProjGradOffset(tex, cubeArrayProj, ddx, ddy, offset);
            return projected + lodProjected + gradOffsetProjected;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return cubeProjected(
                    cubeMap,
                    input.cubeProj,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + cubeArrayProjected(
                    cubeArray,
                    input.cubeArrayProj,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "TextureCube cubeMap : register(t0);" in generated_code
    assert "TextureCubeArray cubeArray : register(t1);" in generated_code
    assert (
        "float4 cubeProjected(TextureCube tex, SamplerState texSampler, float4 cubeProj, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float4 cubeArrayProjected(TextureCubeArray tex, float4 cubeArrayProj, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert "SamplerState cubeMapSampler : register(s0);" in generated_code
    assert (
        "float4 projected = tex.Sample(texSampler, cubeProj.xyz / cubeProj.w);"
        in generated_code
    )
    assert (
        "float4 lodProjected = tex.SampleLevel(texSampler, cubeProj.xyz / cubeProj.w, lod);"
        in generated_code
    )
    assert (
        "/* unsupported DirectX projected texture: textureProjGradOffset offsets require 1D, 2D, 2D-array, or 3D textures */ float4(0.0, 0.0, 0.0, 0.0)"
        in generated_code
    )
    coordinate_diagnostic_counts = {
        "textureProj": 1,
        "textureProjLod": 1,
        "textureProjGradOffset": 1,
    }
    for func_name, count in coordinate_diagnostic_counts.items():
        assert (
            generated_code.count(
                f"/* unsupported DirectX projected texture: {func_name} requires 1D, 2D, or 3D projection coordinates */ float4(0.0, 0.0, 0.0, 0.0)"
            )
            == count
        )
    assert (
        "cubeProjected(cubeMap, cubeMapSampler, input.cubeProj, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "cubeArrayProjected(cubeArray, input.cubeArrayProj, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "SamplerState cubeArraySampler" not in generated_code


def test_directx_projected_cube_sampler_role_conflicts_and_compare_diagnostics():
    regular_diagnostic_shader = """
    shader DiagnosticRegularSamplerRoleConflict {
        samplerCube cubeMap;
        sampler2DShadow shadowMap;
        sampler sharedSampler;

        struct FSInput {
            vec4 cubeProj @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            float depth;
        };

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                vec4 bad = textureProj(cubeMap, sharedSampler, input.cubeProj);
                float ok = textureCompare(shadowMap, sharedSampler, input.uv, input.depth);
                return ok + bad.x;
            }
        }
    }
    """
    parameter_diagnostic_shader = """
    shader DiagnosticSamplerParamRoleConflict {
        samplerCube cubeMap;
        sampler2DShadow shadowMap;
        sampler sharedSampler;

        struct FSInput {
            vec4 cubeProj @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            float depth;
        };

        float inspect(sampler s, vec4 cubeProj, vec2 uv, float depth) {
            vec4 bad = textureProj(cubeMap, s, cubeProj);
            float ok = textureCompare(shadowMap, s, uv, depth);
            return ok + bad.x;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return inspect(sharedSampler, input.cubeProj, input.uv, input.depth);
            }
        }
    }
    """
    comparison_diagnostic_shader = """
    shader DiagnosticCompareSamplerRoleConflict {
        samplerCubeShadow cubeShadow;
        sampler2D colorMap;
        sampler sharedSampler;

        struct FSInput {
            vec4 cubeProj @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            float depth;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float bad = textureCompareProj(
                    cubeShadow,
                    sharedSampler,
                    input.cubeProj,
                    input.depth
                );
                vec4 ok = texture(colorMap, sharedSampler, input.uv);
                return ok + vec4(bad);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="DirectX sampler\\(s\\) used for both regular sampling and shadow comparison: sharedSampler",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(regular_diagnostic_shader), "fragment"
        )

    with pytest.raises(
        ValueError,
        match=r"DirectX sampler\(s\) used for both regular sampling and shadow comparison: (sharedSampler|inspect\.s)",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(parameter_diagnostic_shader), "fragment"
        )

    with pytest.raises(
        ValueError,
        match="DirectX sampler\\(s\\) used for both regular sampling and shadow comparison: sharedSampler",
    ):
        HLSLCodeGen().generate_stage(
            crosstl.translator.parse(comparison_diagnostic_shader), "fragment"
        )


def test_directx_projected_cube_struct_sampler_member_role_conflict_is_reported():
    shader = """
    shader DiagnosticStructSamplerRoleConflict {
        samplerCube cubeMap;
        sampler2DShadow shadowMap;

        struct SamplerPack {
            sampler samplers[2];
        };

        struct FSInput {
            vec4 cubeProj @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            float depth;
            int index;
        };

        float inspect(SamplerPack pack, FSInput input) {
            vec4 bad = textureProj(cubeMap, pack.samplers[input.index], input.cubeProj);
            float ok = textureCompare(
                shadowMap,
                pack.samplers[input.index],
                input.uv,
                input.depth
            );
            return ok + bad.x;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                SamplerPack pack;
                return inspect(pack, input);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=r"DirectX sampler\(s\) used for both regular sampling and shadow comparison: SamplerPack\.samplers",
    ):
        HLSLCodeGen().generate_stage(crosstl.translator.parse(shader), "fragment")


def test_directx_unsupported_gather_diagnostics_do_not_create_sampler_role_conflicts():
    regular_diagnostic_shader = """
    shader UnsupportedGatherSamplerRoleConflict {
        sampler1D lineMap;
        sampler2DShadow shadowMap;
        sampler sharedSampler;

        struct FSInput {
            float u @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            float depth;
        };

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                vec4 bad = textureGather(lineMap, sharedSampler, input.u);
                float ok = textureCompare(shadowMap, sharedSampler, input.uv, input.depth);
                return ok + bad.x;
            }
        }
    }
    """
    compare_diagnostic_shader = """
    shader UnsupportedGatherCompareSamplerRoleConflict {
        samplerCubeShadow cubeShadow;
        sampler2D colorMap;
        sampler sharedSampler;

        struct FSInput {
            vec4 cubeLayer @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            float depth;
            ivec2 offset @ TEXCOORD2;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 bad = textureGatherCompareOffset(
                    cubeShadow,
                    sharedSampler,
                    input.cubeLayer,
                    input.depth,
                    input.offset
                );
                vec4 ok = texture(colorMap, sharedSampler, input.uv);
                return ok + bad;
            }
        }
    }
    """
    parameter_diagnostic_shader = """
    shader UnsupportedGatherParamRoleConflict {
        sampler1D lineMap;
        sampler2DShadow shadowMap;
        sampler sharedSampler;

        struct FSInput {
            float u @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            float depth;
        };

        float inspect(sampler s, float u, vec2 uv, float depth) {
            vec4 bad = textureGather(lineMap, s, u);
            float ok = textureCompare(shadowMap, s, uv, depth);
            return ok + bad.x;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return inspect(sharedSampler, input.u, input.uv, input.depth);
            }
        }
    }
    """

    regular_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(regular_diagnostic_shader), "fragment"
    )
    compare_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(compare_diagnostic_shader), "fragment"
    )
    parameter_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(parameter_diagnostic_shader), "fragment"
    )

    gather_diagnostic = (
        "/* unsupported DirectX texture gather: textureGather requires 2D, "
        "2D-array, cube, or cube-array textures */ float4(0.0, 0.0, 0.0, 0.0)"
    )
    gather_compare_offset_diagnostic = (
        "/* unsupported DirectX texture gather compare: "
        "textureGatherCompareOffset offsets require 2D or 2D-array textures */ "
        "float4(0.0, 0.0, 0.0, 0.0)"
    )

    assert "SamplerComparisonState sharedSampler : register(s0);" in regular_code
    assert "SamplerState sharedSampler" not in regular_code
    assert gather_diagnostic in regular_code
    assert (
        "float ok = shadowMap.SampleCmp(sharedSampler, input.uv, input.depth);"
        in regular_code
    )
    assert ".Gather(sharedSampler" not in regular_code

    assert "SamplerState sharedSampler : register(s0);" in compare_code
    assert "SamplerComparisonState sharedSampler" not in compare_code
    assert gather_compare_offset_diagnostic in compare_code
    assert "float4 ok = colorMap.Sample(sharedSampler, input.uv);" in compare_code
    assert "GatherCmp(sharedSampler" not in compare_code

    assert "SamplerComparisonState sharedSampler : register(s0);" in parameter_code
    assert (
        "float inspect(SamplerComparisonState s, float u, float2 uv, float depth)"
        in parameter_code
    )
    assert gather_diagnostic in parameter_code
    assert "float ok = shadowMap.SampleCmp(s, uv, depth);" in parameter_code
    assert (
        "return inspect(sharedSampler, input.u, input.uv, input.depth);"
        in parameter_code
    )


def test_directx_storage_image_texture_diagnostics_do_not_create_sampler_role_conflicts():
    regular_diagnostic_shader = """
    shader StorageRegularSamplerRoleConflict {
        image2D colorImage;
        sampler2DShadow shadowMap;
        sampler sharedSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                vec4 bad = texture(colorImage, sharedSampler, input.uv);
                float ok = textureCompare(shadowMap, sharedSampler, input.uv, input.depth);
                return ok + bad.x;
            }
        }
    }
    """
    comparison_diagnostic_shader = """
    shader StorageCompareSamplerRoleConflict {
        image2D colorImage;
        sampler2D colorMap;
        sampler sharedSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float bad = textureCompare(colorImage, sharedSampler, input.uv, input.depth);
                vec4 ok = texture(colorMap, sharedSampler, input.uv);
                return ok + vec4(bad);
            }
        }
    }
    """

    regular_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(regular_diagnostic_shader), "fragment"
    )
    comparison_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(comparison_diagnostic_shader), "fragment"
    )

    storage_texture_diagnostic = (
        "/* unsupported DirectX storage image texture operation: "
        "texture on RWTexture2D<float4> */ float4(0.0, 0.0, 0.0, 0.0)"
    )
    storage_compare_diagnostic = (
        "/* unsupported DirectX storage image texture comparison: "
        "textureCompare on RWTexture2D<float4> */ 0.0"
    )

    assert "RWTexture2D<float4> colorImage : register(u0);" in regular_code
    assert "SamplerComparisonState sharedSampler : register(s0);" in regular_code
    assert "SamplerState sharedSampler" not in regular_code
    assert storage_texture_diagnostic in regular_code
    assert (
        "float ok = shadowMap.SampleCmp(sharedSampler, input.uv, input.depth);"
        in regular_code
    )
    assert ".Sample(sharedSampler, input.uv)" not in regular_code

    assert "RWTexture2D<float4> colorImage : register(u0);" in comparison_code
    assert "SamplerState sharedSampler : register(s0);" in comparison_code
    assert "SamplerComparisonState sharedSampler" not in comparison_code
    assert storage_compare_diagnostic in comparison_code
    assert "float4 ok = colorMap.Sample(sharedSampler, input.uv);" in comparison_code
    assert "SampleCmp(sharedSampler" not in comparison_code


def test_directx_shadow_gather_compare_offsets_use_comparison_samplers():
    shader = """
    shader ShadowGatherCompareOffsets {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float depth;
            ivec2 offset @ TEXCOORD3;
        };

        vec4 implicitShadow(sampler2DShadow tex, vec2 uv, float depth, ivec2 offset) {
            vec4 gathered = textureGatherCompare(tex, uv, depth);
            float compared = textureCompareOffset(tex, uv, depth, offset);
            return gathered + vec4(compared);
        }

        vec4 gatherShadow(sampler2DShadow tex, sampler s, vec2 uv, float depth, ivec2 offset) {
            vec4 gathered = textureGatherCompare(tex, s, uv, depth);
            vec4 offsetGathered = textureGatherCompareOffset(tex, s, uv, depth, offset);
            float offsetCompared = textureCompareOffset(tex, s, uv, depth, offset);
            return gathered + offsetGathered + vec4(offsetCompared);
        }

        vec4 gatherShadowArray(sampler2DArrayShadow tex, sampler s, vec3 uvLayer, float depth, ivec2 offset) {
            vec4 gathered = textureGatherCompare(tex, s, uvLayer, depth);
            vec4 offsetGathered = textureGatherCompareOffset(tex, s, uvLayer, depth, offset);
            float offsetCompared = textureCompareOffset(tex, s, uvLayer, depth, offset);
            return gathered + offsetGathered + vec4(offsetCompared);
        }

        vec4 gatherCubeShadowArray(samplerCubeArrayShadow tex, sampler s, vec4 cubeLayer, float depth) {
            return textureGatherCompare(tex, s, cubeLayer, depth);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return implicitShadow(shadowMap, input.uv, input.depth, input.offset)
                    + gatherShadow(shadowMap, compareSampler, input.uv, input.depth, input.offset)
                    + gatherShadowArray(shadowArray, compareSampler, input.uvLayer, input.depth, input.offset)
                    + gatherCubeShadowArray(cubeShadowArray, compareSampler, input.cubeLayer, input.depth);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "Texture2DArray shadowArray : register(t1);" in generated_code
    assert "TextureCubeArray cubeShadowArray : register(t2);" in generated_code
    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert "SamplerComparisonState compareSampler : register(s1);" in generated_code
    assert (
        "float4 implicitShadow(Texture2D tex, SamplerComparisonState texSampler, float2 uv, float depth, int2 offset)"
        in generated_code
    )
    assert "float4 gathered = tex.GatherCmp(texSampler, uv, depth);" in generated_code
    assert (
        "float compared = tex.SampleCmp(texSampler, uv, depth, offset);"
        in generated_code
    )
    assert (
        "float4 gatherShadow(Texture2D tex, SamplerComparisonState s, float2 uv, float depth, int2 offset)"
        in generated_code
    )
    assert "float4 gathered = tex.GatherCmp(s, uv, depth);" in generated_code
    assert (
        "float4 offsetGathered = tex.GatherCmp(s, uv, depth, offset);" in generated_code
    )
    assert (
        "float offsetCompared = tex.SampleCmp(s, uv, depth, offset);" in generated_code
    )
    assert (
        "float4 gatherShadowArray(Texture2DArray tex, SamplerComparisonState s, float3 uvLayer, float depth, int2 offset)"
        in generated_code
    )
    assert "tex.GatherCmp(s, uvLayer, depth, offset)" in generated_code
    assert "tex.SampleCmp(s, uvLayer, depth, offset)" in generated_code
    assert (
        "float4 gatherCubeShadowArray(TextureCubeArray tex, SamplerComparisonState s, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert "return tex.GatherCmp(s, cubeLayer, depth);" in generated_code
    assert (
        "implicitShadow(shadowMap, shadowMapSampler, input.uv, input.depth, input.offset)"
        in generated_code
    )
    assert (
        "gatherShadow(shadowMap, compareSampler, input.uv, input.depth, input.offset)"
        in generated_code
    )
    assert (
        "gatherShadowArray(shadowArray, compareSampler, input.uvLayer, input.depth, input.offset)"
        in generated_code
    )
    assert (
        "gatherCubeShadowArray(cubeShadowArray, compareSampler, input.cubeLayer, input.depth)"
        in generated_code
    )
    assert "textureGatherCompare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code
    assert "textureCompareOffset(" not in generated_code


def test_directx_cube_shadow_gather_compare_supports_cube_and_cube_array():
    shader = """
    shader CubeShadowGatherCompare {
        samplerCubeShadow cubeShadow;
        samplerCubeArrayShadow cubeShadowArray;
        sampler compareSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
        };

        vec4 gatherCubeShadow(samplerCubeShadow tex, sampler s, vec3 direction, float depth) {
            return textureGatherCompare(tex, s, direction, depth);
        }

        vec4 gatherCubeArrayShadow(samplerCubeArrayShadow tex, sampler s, vec4 cubeLayer, float depth) {
            return textureGatherCompare(tex, s, cubeLayer, depth);
        }

        vec4 implicitCubeShadow(samplerCubeShadow tex, vec3 direction, float depth) {
            return textureGatherCompare(tex, direction, depth);
        }

        vec4 implicitCubeArrayShadow(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth) {
            return textureGatherCompare(tex, cubeLayer, depth);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherCubeShadow(cubeShadow, compareSampler, input.direction, input.depth)
                    + gatherCubeArrayShadow(cubeShadowArray, compareSampler, input.cubeLayer, input.depth)
                    + implicitCubeShadow(cubeShadow, input.direction, input.depth)
                    + implicitCubeArrayShadow(cubeShadowArray, input.cubeLayer, input.depth)
                    + textureGatherCompare(cubeShadow, compareSampler, input.direction, input.depth)
                    + textureGatherCompare(cubeShadowArray, compareSampler, input.cubeLayer, input.depth)
                    + textureGatherCompare(cubeShadow, input.direction, input.depth)
                    + textureGatherCompare(cubeShadowArray, input.cubeLayer, input.depth);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "TextureCube cubeShadow : register(t0);" in generated_code
    assert "SamplerComparisonState cubeShadowSampler : register(s0);" in generated_code
    assert "TextureCubeArray cubeShadowArray : register(t1);" in generated_code
    assert (
        "SamplerComparisonState cubeShadowArraySampler : register(s1);"
        in generated_code
    )
    assert "SamplerComparisonState compareSampler : register(s2);" in generated_code
    assert (
        "float4 gatherCubeShadow(TextureCube tex, SamplerComparisonState s, float3 direction, float depth)"
        in generated_code
    )
    assert "return tex.GatherCmp(s, direction, depth);" in generated_code
    assert (
        "float4 gatherCubeArrayShadow(TextureCubeArray tex, SamplerComparisonState s, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert "return tex.GatherCmp(s, cubeLayer, depth);" in generated_code
    assert (
        "float4 implicitCubeShadow(TextureCube tex, SamplerComparisonState texSampler, float3 direction, float depth)"
        in generated_code
    )
    assert "return tex.GatherCmp(texSampler, direction, depth);" in generated_code
    assert (
        "float4 implicitCubeArrayShadow(TextureCubeArray tex, SamplerComparisonState texSampler, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert "return tex.GatherCmp(texSampler, cubeLayer, depth);" in generated_code
    assert (
        "gatherCubeShadow(cubeShadow, compareSampler, input.direction, input.depth)"
        in generated_code
    )
    assert (
        "implicitCubeShadow(cubeShadow, cubeShadowSampler, input.direction, input.depth)"
        in generated_code
    )
    assert (
        "cubeShadow.GatherCmp(compareSampler, input.direction, input.depth)"
        in generated_code
    )
    assert (
        "cubeShadowArray.GatherCmp(cubeShadowArraySampler, input.cubeLayer, input.depth)"
        in generated_code
    )
    assert "unsupported DirectX texture gather compare" not in generated_code
    assert "textureGatherCompare" not in generated_code


def test_directx_implicit_shadow_gather_compare_offsets_cover_arrays_and_cube_arrays():
    shader = """
    shader ImplicitShadowGatherCompare {
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            ivec2 offset @ TEXCOORD2;
        };

        vec4 implicitArray(sampler2DArrayShadow tex, vec3 uvLayer, float depth, ivec2 offset) {
            return textureGatherCompare(tex, uvLayer, depth)
                + textureGatherCompareOffset(tex, uvLayer, depth, offset)
                + vec4(textureCompareOffset(tex, uvLayer, depth, offset));
        }

        vec4 implicitCubeArray(samplerCubeArrayShadow tex, vec4 cubeLayer, float depth) {
            return textureGatherCompare(tex, cubeLayer, depth);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return implicitArray(shadowArray, input.uvLayer, input.depth, input.offset)
                    + implicitCubeArray(cubeShadowArray, input.cubeLayer, input.depth);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2DArray shadowArray : register(t0);" in generated_code
    assert "SamplerComparisonState shadowArraySampler : register(s0);" in generated_code
    assert "TextureCubeArray cubeShadowArray : register(t1);" in generated_code
    assert (
        "SamplerComparisonState cubeShadowArraySampler : register(s1)" in generated_code
    )
    assert (
        "float4 implicitArray(Texture2DArray tex, SamplerComparisonState texSampler, float3 uvLayer, float depth, int2 offset)"
        in generated_code
    )
    assert "tex.GatherCmp(texSampler, uvLayer, depth)" in generated_code
    assert "tex.GatherCmp(texSampler, uvLayer, depth, offset)" in generated_code
    assert "tex.SampleCmp(texSampler, uvLayer, depth, offset)" in generated_code
    assert (
        "float4 implicitCubeArray(TextureCubeArray tex, SamplerComparisonState texSampler, float4 cubeLayer, float depth)"
        in generated_code
    )
    assert "return tex.GatherCmp(texSampler, cubeLayer, depth);" in generated_code
    assert (
        "implicitArray(shadowArray, shadowArraySampler, input.uvLayer, input.depth, input.offset)"
        in generated_code
    )
    assert (
        "implicitCubeArray(cubeShadowArray, cubeShadowArraySampler, input.cubeLayer, input.depth)"
        in generated_code
    )
    assert "textureGatherCompare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code
    assert "textureCompareOffset(" not in generated_code


def test_directx_direct_shadow_gather_compare_stage_input_members():
    explicit_shader = """
    shader DirectShadowGatherCompare {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float depth;
            ivec2 offset @ TEXCOORD3;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 planar = textureGatherCompare(shadowMap, compareSampler, input.uv, input.depth);
                vec4 planarOffset = textureGatherCompareOffset(shadowMap, compareSampler, input.uv, input.depth, input.offset);
                vec4 arrayGather = textureGatherCompare(shadowArray, compareSampler, input.uvLayer, input.depth);
                vec4 arrayOffset = textureGatherCompareOffset(shadowArray, compareSampler, input.uvLayer, input.depth, input.offset);
                vec4 cubeArrayGather = textureGatherCompare(cubeShadowArray, compareSampler, input.cubeLayer, input.depth);
                return planar + planarOffset + arrayGather + arrayOffset + cubeArrayGather;
            }
        }
    }
    """
    implicit_shader = """
    shader DirectImplicitShadowGatherCompare {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float depth;
            ivec2 offset @ TEXCOORD3;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 planar = textureGatherCompare(shadowMap, input.uv, input.depth);
                vec4 planarOffset = textureGatherCompareOffset(shadowMap, input.uv, input.depth, input.offset);
                vec4 arrayGather = textureGatherCompare(shadowArray, input.uvLayer, input.depth);
                vec4 arrayOffset = textureGatherCompareOffset(shadowArray, input.uvLayer, input.depth, input.offset);
                vec4 cubeArrayGather = textureGatherCompare(cubeShadowArray, input.cubeLayer, input.depth);
                return planar + planarOffset + arrayGather + arrayOffset + cubeArrayGather;
            }
        }
    }
    """

    explicit_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(explicit_shader), "fragment"
    )
    implicit_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(implicit_shader), "fragment"
    )

    assert "Texture2D shadowMap : register(t0);" in explicit_code
    assert "Texture2DArray shadowArray : register(t1);" in explicit_code
    assert "TextureCubeArray cubeShadowArray : register(t2);" in explicit_code
    assert "SamplerComparisonState compareSampler : register(s0);" in explicit_code
    assert (
        "float4 planar = shadowMap.GatherCmp(compareSampler, input.uv, input.depth);"
        in explicit_code
    )
    assert (
        "float4 planarOffset = shadowMap.GatherCmp(compareSampler, input.uv, input.depth, input.offset);"
        in explicit_code
    )
    assert (
        "float4 arrayGather = shadowArray.GatherCmp(compareSampler, input.uvLayer, input.depth);"
        in explicit_code
    )
    assert (
        "float4 arrayOffset = shadowArray.GatherCmp(compareSampler, input.uvLayer, input.depth, input.offset);"
        in explicit_code
    )
    assert (
        "float4 cubeArrayGather = cubeShadowArray.GatherCmp(compareSampler, input.cubeLayer, input.depth);"
        in explicit_code
    )
    assert "unsupported DirectX texture gather compare" not in explicit_code
    assert "textureGatherCompare" not in explicit_code

    assert "Texture2D shadowMap : register(t0);" in implicit_code
    assert "SamplerComparisonState shadowMapSampler : register(s0);" in implicit_code
    assert "Texture2DArray shadowArray : register(t1);" in implicit_code
    assert "SamplerComparisonState shadowArraySampler : register(s1);" in implicit_code
    assert "TextureCubeArray cubeShadowArray : register(t2);" in implicit_code
    assert (
        "SamplerComparisonState cubeShadowArraySampler : register(s2);" in implicit_code
    )
    assert (
        "float4 planar = shadowMap.GatherCmp(shadowMapSampler, input.uv, input.depth);"
        in implicit_code
    )
    assert (
        "float4 planarOffset = shadowMap.GatherCmp(shadowMapSampler, input.uv, input.depth, input.offset);"
        in implicit_code
    )
    assert (
        "float4 arrayGather = shadowArray.GatherCmp(shadowArraySampler, input.uvLayer, input.depth);"
        in implicit_code
    )
    assert (
        "float4 arrayOffset = shadowArray.GatherCmp(shadowArraySampler, input.uvLayer, input.depth, input.offset);"
        in implicit_code
    )
    assert (
        "float4 cubeArrayGather = cubeShadowArray.GatherCmp(cubeShadowArraySampler, input.cubeLayer, input.depth);"
        in implicit_code
    )
    assert "unsupported DirectX texture gather compare" not in implicit_code
    assert "textureGatherCompare" not in implicit_code


def test_directx_shadow_gather_compare_resource_arrays_forward_samplers():
    explicit_shader = """
    shader ShadowGatherCompareResourceArrays {
        sampler2DShadow shadowMaps[4];
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeArrays[4];
        sampler compareSamplers[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            vec4 cubeLayer @ TEXCOORD3;
            float depth;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 gatherLayer(
            sampler2DShadow maps[],
            sampler2DArrayShadow arrays[],
            samplerCubeArrayShadow cubes[],
            sampler samplers[],
            int layer,
            vec2 uv,
            vec3 uvLayer,
            vec4 cubeLayer,
            float depth,
            ivec2 offset
        ) {
            vec4 fixedPlanar = textureGatherCompare(maps[2], samplers[2], uv, depth);
            vec4 dynamicPlanarOffset = textureGatherCompareOffset(maps[layer], samplers[layer], uv, depth, offset);
            vec4 fixedArray = textureGatherCompare(arrays[1], samplers[1], uvLayer, depth);
            vec4 dynamicArrayOffset = textureGatherCompareOffset(arrays[layer], samplers[layer], uvLayer, depth, offset);
            vec4 fixedCubeArray = textureGatherCompare(cubes[3], samplers[3], cubeLayer, depth);
            vec4 dynamicCubeArray = textureGatherCompare(cubes[layer], samplers[layer], cubeLayer, depth);
            return fixedPlanar + dynamicPlanarOffset + fixedArray + dynamicArrayOffset + fixedCubeArray + dynamicCubeArray;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherLayer(
                    shadowMaps,
                    shadowArrays,
                    cubeArrays,
                    compareSamplers,
                    input.layer,
                    input.uv,
                    input.uvLayer,
                    input.cubeLayer,
                    input.depth,
                    input.offset
                );
            }
        }
    }
    """
    implicit_shader = """
    shader ImplicitShadowGatherCompareResourceArrays {
        sampler2DShadow shadowMaps[4];
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            vec4 cubeLayer @ TEXCOORD3;
            float depth;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 gatherLayer(
            sampler2DShadow maps[],
            sampler2DArrayShadow arrays[],
            samplerCubeArrayShadow cubes[],
            int layer,
            vec2 uv,
            vec3 uvLayer,
            vec4 cubeLayer,
            float depth,
            ivec2 offset
        ) {
            vec4 fixedPlanar = textureGatherCompare(maps[2], uv, depth);
            vec4 dynamicPlanarOffset = textureGatherCompareOffset(maps[layer], uv, depth, offset);
            vec4 fixedArray = textureGatherCompare(arrays[1], uvLayer, depth);
            vec4 dynamicArrayOffset = textureGatherCompareOffset(arrays[layer], uvLayer, depth, offset);
            vec4 fixedCubeArray = textureGatherCompare(cubes[3], cubeLayer, depth);
            vec4 dynamicCubeArray = textureGatherCompare(cubes[layer], cubeLayer, depth);
            return fixedPlanar + dynamicPlanarOffset + fixedArray + dynamicArrayOffset + fixedCubeArray + dynamicCubeArray;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return gatherLayer(
                    shadowMaps,
                    shadowArrays,
                    cubeArrays,
                    input.layer,
                    input.uv,
                    input.uvLayer,
                    input.cubeLayer,
                    input.depth,
                    input.offset
                );
            }
        }
    }
    """

    explicit_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(explicit_shader), "fragment"
    )
    implicit_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(implicit_shader), "fragment"
    )

    assert "Texture2D shadowMaps[4] : register(t0);" in explicit_code
    assert "Texture2DArray shadowArrays[4] : register(t4);" in explicit_code
    assert "TextureCubeArray cubeArrays[4] : register(t8);" in explicit_code
    assert "SamplerComparisonState compareSamplers[4] : register(s0);" in explicit_code
    assert (
        "float4 gatherLayer(Texture2D maps[4], Texture2DArray arrays[4], TextureCubeArray cubes[4], SamplerComparisonState samplers[4], int layer, float2 uv, float3 uvLayer, float4 cubeLayer, float depth, int2 offset)"
        in explicit_code
    )
    assert "maps[2].GatherCmp(samplers[2], uv, depth)" in explicit_code
    assert "maps[layer].GatherCmp(samplers[layer], uv, depth, offset)" in explicit_code
    assert "arrays[1].GatherCmp(samplers[1], uvLayer, depth)" in explicit_code
    assert (
        "arrays[layer].GatherCmp(samplers[layer], uvLayer, depth, offset)"
        in explicit_code
    )
    assert "cubes[3].GatherCmp(samplers[3], cubeLayer, depth)" in explicit_code
    assert "cubes[layer].GatherCmp(samplers[layer], cubeLayer, depth)" in explicit_code
    assert (
        "gatherLayer(shadowMaps, shadowArrays, cubeArrays, compareSamplers, input.layer, input.uv, input.uvLayer, input.cubeLayer, input.depth, input.offset)"
        in explicit_code
    )
    assert "textureGatherCompare" not in explicit_code

    assert "Texture2D shadowMaps[4] : register(t0);" in implicit_code
    assert "SamplerComparisonState shadowMapsSampler : register(s0);" in implicit_code
    assert "Texture2DArray shadowArrays[4] : register(t4);" in implicit_code
    assert "SamplerComparisonState shadowArraysSampler : register(s1);" in implicit_code
    assert "TextureCubeArray cubeArrays[4] : register(t8);" in implicit_code
    assert "SamplerComparisonState cubeArraysSampler : register(s2);" in implicit_code
    assert (
        "float4 gatherLayer(Texture2D maps[4], SamplerComparisonState mapsSampler, Texture2DArray arrays[4], SamplerComparisonState arraysSampler, TextureCubeArray cubes[4], SamplerComparisonState cubesSampler, int layer, float2 uv, float3 uvLayer, float4 cubeLayer, float depth, int2 offset)"
        in implicit_code
    )
    assert "maps[2].GatherCmp(mapsSampler, uv, depth)" in implicit_code
    assert "maps[layer].GatherCmp(mapsSampler, uv, depth, offset)" in implicit_code
    assert "arrays[1].GatherCmp(arraysSampler, uvLayer, depth)" in implicit_code
    assert (
        "arrays[layer].GatherCmp(arraysSampler, uvLayer, depth, offset)"
        in implicit_code
    )
    assert "cubes[3].GatherCmp(cubesSampler, cubeLayer, depth)" in implicit_code
    assert "cubes[layer].GatherCmp(cubesSampler, cubeLayer, depth)" in implicit_code
    assert (
        "gatherLayer(shadowMaps, shadowMapsSampler, shadowArrays, shadowArraysSampler, cubeArrays, cubeArraysSampler, input.layer, input.uv, input.uvLayer, input.cubeLayer, input.depth, input.offset)"
        in implicit_code
    )
    assert "textureGatherCompare" not in implicit_code


def test_directx_unsupported_cube_shadow_gather_compare_offsets_skip_implicit_samplers():
    shader = """
    shader UnsupportedCubeGatherCompareOffset {
        samplerCubeShadow cubeMap;
        samplerCubeArrayShadow cubeArray;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            ivec2 offset @ TEXCOORD2;
        };

        vec4 cubeOffset(
            samplerCubeShadow tex,
            vec3 direction,
            float depth,
            ivec2 offset
        ) {
            return textureGatherCompareOffset(tex, direction, depth, offset);
        }

        vec4 cubeArrayOffset(
            samplerCubeArrayShadow tex,
            vec4 cubeLayer,
            float depth,
            ivec2 offset
        ) {
            return textureGatherCompareOffset(tex, cubeLayer, depth, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return cubeOffset(cubeMap, input.direction, input.depth, input.offset)
                    + cubeArrayOffset(cubeArray, input.cubeLayer, input.depth, input.offset)
                    + textureGatherCompareOffset(cubeArray, input.cubeLayer, input.depth, input.offset);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported DirectX texture gather compare: "
        "textureGatherCompareOffset offsets require 2D or 2D-array textures */ "
        "float4(0.0, 0.0, 0.0, 0.0)"
    )
    assert "TextureCube cubeMap : register(t0);" in generated_code
    assert "TextureCubeArray cubeArray : register(t1);" in generated_code
    assert "SamplerComparisonState" not in generated_code
    assert "cubeMapSampler" not in generated_code
    assert "cubeArraySampler" not in generated_code
    assert (
        "float4 cubeOffset(TextureCube tex, float3 direction, float depth, int2 offset)"
        in generated_code
    )
    assert (
        "float4 cubeArrayOffset(TextureCubeArray tex, float4 cubeLayer, float depth, int2 offset)"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 3
    assert (
        "cubeOffset(cubeMap, input.direction, input.depth, input.offset)"
        in generated_code
    )
    assert (
        "cubeArrayOffset(cubeArray, input.cubeLayer, input.depth, input.offset)"
        in generated_code
    )
    assert ".GatherCmp(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code


def test_directx_shadow_compare_lod_grad_use_comparison_samplers():
    shader = """
    shader ShadowCompareLodGrad {
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
            vec4 cubeLayer @ TEXCOORD5;
            vec3 cubeDdx @ TEXCOORD6;
            vec3 cubeDdy @ TEXCOORD7;
        };

        float implicitShadow(
            sampler2DShadow tex,
            vec2 uv,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float lodValue = textureCompareLod(tex, uv, depth, lod);
            float lodOffsetValue = textureCompareLodOffset(tex, uv, depth, lod, offset);
            float gradValue = textureCompareGrad(tex, uv, depth, ddx, ddy);
            float gradOffsetValue = textureCompareGradOffset(tex, uv, depth, ddx, ddy, offset);
            return lodValue + lodOffsetValue + gradValue + gradOffsetValue;
        }

        float compareShadow(
            sampler2DShadow tex,
            sampler s,
            vec2 uv,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float lodValue = textureCompareLod(tex, s, uv, depth, lod);
            float lodOffsetValue = textureCompareLodOffset(tex, s, uv, depth, lod, offset);
            float gradValue = textureCompareGrad(tex, s, uv, depth, ddx, ddy);
            float gradOffsetValue = textureCompareGradOffset(tex, s, uv, depth, ddx, ddy, offset);
            return lodValue + lodOffsetValue + gradValue + gradOffsetValue;
        }

        float compareShadowArray(
            sampler2DArrayShadow tex,
            sampler s,
            vec3 uvLayer,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float lodValue = textureCompareLod(tex, s, uvLayer, depth, lod);
            float lodOffsetValue = textureCompareLodOffset(tex, s, uvLayer, depth, lod, offset);
            float gradValue = textureCompareGrad(tex, s, uvLayer, depth, ddx, ddy);
            float gradOffsetValue = textureCompareGradOffset(tex, s, uvLayer, depth, ddx, ddy, offset);
            return lodValue + lodOffsetValue + gradValue + gradOffsetValue;
        }

        float compareCubeArrayShadow(
            samplerCubeArrayShadow tex,
            sampler s,
            vec4 cubeLayer,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy
        ) {
            float lodValue = textureCompareLod(tex, s, cubeLayer, depth, lod);
            float gradValue = textureCompareGrad(tex, s, cubeLayer, depth, ddx, ddy);
            return lodValue + gradValue;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float implicitValue = implicitShadow(
                    shadowMap,
                    input.uv,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float explicitValue = compareShadow(
                    shadowMap,
                    compareSampler,
                    input.uv,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float arrayValue = compareShadowArray(
                    shadowArray,
                    compareSampler,
                    input.uvLayer,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float cubeArrayValue = compareCubeArrayShadow(
                    cubeShadowArray,
                    compareSampler,
                    input.cubeLayer,
                    input.depth,
                    input.lod,
                    input.cubeDdx,
                    input.cubeDdy
                );
                return vec4(implicitValue + explicitValue + arrayValue + cubeArrayValue);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "Texture2DArray shadowArray : register(t1);" in generated_code
    assert "TextureCubeArray cubeShadowArray : register(t2);" in generated_code
    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert "SamplerComparisonState compareSampler : register(s1);" in generated_code
    assert (
        "float implicitShadow(Texture2D tex, SamplerComparisonState texSampler, float2 uv, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float lodValue = tex.SampleCmpLevel(texSampler, uv, depth, lod);"
        in generated_code
    )
    assert (
        "float lodOffsetValue = tex.SampleCmpLevel(texSampler, uv, depth, lod, offset);"
        in generated_code
    )
    assert (
        "float gradValue = tex.SampleCmpGrad(texSampler, uv, depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "float gradOffsetValue = tex.SampleCmpGrad(texSampler, uv, depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float compareShadow(Texture2D tex, SamplerComparisonState s, float2 uv, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "float lodValue = tex.SampleCmpLevel(s, uv, depth, lod);" in generated_code
    assert (
        "float lodOffsetValue = tex.SampleCmpLevel(s, uv, depth, lod, offset);"
        in generated_code
    )
    assert (
        "float gradValue = tex.SampleCmpGrad(s, uv, depth, ddx, ddy);" in generated_code
    )
    assert (
        "float gradOffsetValue = tex.SampleCmpGrad(s, uv, depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float compareShadowArray(Texture2DArray tex, SamplerComparisonState s, float3 uvLayer, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "tex.SampleCmpLevel(s, uvLayer, depth, lod)" in generated_code
    assert "tex.SampleCmpLevel(s, uvLayer, depth, lod, offset)" in generated_code
    assert "tex.SampleCmpGrad(s, uvLayer, depth, ddx, ddy)" in generated_code
    assert "tex.SampleCmpGrad(s, uvLayer, depth, ddx, ddy, offset)" in generated_code
    assert (
        "float compareCubeArrayShadow(TextureCubeArray tex, SamplerComparisonState s, float4 cubeLayer, float depth, float lod, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert "tex.SampleCmpLevel(s, cubeLayer, depth, lod)" in generated_code
    assert "tex.SampleCmpGrad(s, cubeLayer, depth, ddx, ddy)" in generated_code
    assert (
        "implicitShadow(shadowMap, shadowMapSampler, input.uv, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "compareShadow(shadowMap, compareSampler, input.uv, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "compareShadowArray(shadowArray, compareSampler, input.uvLayer, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert (
        "compareCubeArrayShadow(cubeShadowArray, compareSampler, input.cubeLayer, input.depth, input.lod, input.cubeDdx, input.cubeDdy)"
        in generated_code
    )
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareLodOffset(" not in generated_code
    assert "textureCompareGrad(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_directx_cube_shadow_compare_offsets_report_unsupported():
    shader = """
    shader CubeShadowCompareOffsetDiagnostics {
        samplerCubeShadow cubeMap;
        samplerCubeArrayShadow cubeArray;
        sampler compareSampler;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        float cubeOffsets(
            samplerCubeShadow tex,
            sampler s,
            vec3 direction,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float compareOffset = textureCompareOffset(tex, s, direction, depth, offset);
            float lodOffset = textureCompareLodOffset(tex, s, direction, depth, lod, offset);
            float gradOffset = textureCompareGradOffset(tex, s, direction, depth, ddx, ddy, offset);
            return compareOffset + lodOffset + gradOffset;
        }

        float cubeArrayOffsets(
            samplerCubeArrayShadow tex,
            sampler s,
            vec4 cubeLayer,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy,
            ivec2 offset
        ) {
            float compareOffset = textureCompareOffset(tex, s, cubeLayer, depth, offset);
            float lodOffset = textureCompareLodOffset(tex, s, cubeLayer, depth, lod, offset);
            float gradOffset = textureCompareGradOffset(tex, s, cubeLayer, depth, ddx, ddy, offset);
            return compareOffset + lodOffset + gradOffset;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return cubeOffsets(
                    cubeMap,
                    compareSampler,
                    input.direction,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                ) + cubeArrayOffsets(
                    cubeArray,
                    compareSampler,
                    input.cubeLayer,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "TextureCube cubeMap : register(t0);" in generated_code
    assert "TextureCubeArray cubeArray : register(t1);" in generated_code
    assert "SamplerState compareSampler : register(s0);" in generated_code
    assert "SamplerComparisonState compareSampler" not in generated_code
    assert (
        "float cubeOffsets(TextureCube tex, SamplerState s, float3 direction, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float cubeArrayOffsets(TextureCubeArray tex, SamplerState s, float4 cubeLayer, float depth, float lod, float3 ddx, float3 ddy, int2 offset)"
        in generated_code
    )
    assert (
        generated_code.count(
            "/* unsupported DirectX texture compare: textureCompareOffset offsets require 2D or 2D-array textures */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported DirectX texture compare: textureCompareLodOffset offsets require 2D or 2D-array textures */ 0.0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported DirectX texture compare: textureCompareGradOffset offsets require 2D or 2D-array textures */ 0.0"
        )
        == 2
    )
    assert "textureCompareOffset(" not in generated_code
    assert "textureCompareLodOffset(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code
    assert ".SampleCmp(s, direction, depth, offset)" not in generated_code
    assert ".SampleCmpLevel(s, direction, depth, lod, offset)" not in generated_code
    assert ".SampleCmpGrad(s, direction, depth, ddx, ddy, offset)" not in generated_code


def test_directx_array_shadow_texture_resource_arrays_keep_compare_coordinates():
    shader = """
    shader ArrayShadowTextureResourceArrays {
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeShadowArrays[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
        };

        float sampleArrayLayer(sampler2DArrayShadow shadowArrays[], sampler shadowSamplers[], vec3 uvLayer, float depth) {
            return textureCompare(shadowArrays[2], shadowSamplers[2], uvLayer, depth);
        }

        float sampleCubeLayer(samplerCubeArrayShadow cubeShadowArrays[], sampler shadowSamplers[], vec4 cubeLayer, float depth) {
            return textureCompare(cubeShadowArrays[3], shadowSamplers[3], cubeLayer, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return sampleArrayLayer(shadowArrays, shadowSamplers, input.uvLayer, input.depth) + sampleCubeLayer(cubeShadowArrays, shadowSamplers, input.cubeLayer, input.depth);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2DArray shadowArrays[4] : register(t0);" in generated_code
    assert "TextureCubeArray cubeShadowArrays[4] : register(t4);" in generated_code
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in generated_code
    assert (
        "float sampleArrayLayer(Texture2DArray shadowArrays[4], SamplerComparisonState shadowSamplers[4], float3 uvLayer, float depth)"
        in generated_code
    )
    assert (
        "shadowArrays[2].SampleCmp(shadowSamplers[2], uvLayer, depth)" in generated_code
    )
    assert (
        "float sampleCubeLayer(TextureCubeArray cubeShadowArrays[4], SamplerComparisonState shadowSamplers[4], float4 cubeLayer, float depth)"
        in generated_code
    )
    assert (
        "cubeShadowArrays[3].SampleCmp(shadowSamplers[3], cubeLayer, depth)"
        in generated_code
    )
    assert (
        "sampleArrayLayer(shadowArrays, shadowSamplers, input.uvLayer, input.depth)"
        in generated_code
    )
    assert (
        "sampleCubeLayer(cubeShadowArrays, shadowSamplers, input.cubeLayer, input.depth)"
        in generated_code
    )
    assert "Texture2DArray shadowArrays[] : register(t0);" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_array_shadow_compare_lod_grad_resource_arrays():
    shader = """
    shader ShadowCompareResourceArrays {
        sampler2DShadow shadowMaps[4];
        sampler2DArrayShadow shadowArrays[4];
        sampler shadowSamplers[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec2 uv @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
        };

        float shadowLayer(
            sampler2DShadow shadowMaps[],
            sampler2DArrayShadow shadowArrays[],
            sampler shadowSamplers[],
            int layer,
            vec2 uv,
            vec3 uvLayer,
            float depth,
            float lod,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            float planarLod = textureCompareLod(shadowMaps[layer], shadowSamplers[layer], uv, depth, lod);
            float planarGrad = textureCompareGradOffset(shadowMaps[1], shadowSamplers[1], uv, depth, ddx, ddy, offset);
            float arrayLod = textureCompareLod(shadowArrays[2], shadowSamplers[2], uvLayer, depth, lod);
            float arrayGrad = textureCompareGradOffset(shadowArrays[layer], shadowSamplers[layer], uvLayer, depth, ddx, ddy, offset);
            return planarLod + planarGrad + arrayLod + arrayGrad;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return shadowLayer(
                    shadowMaps,
                    shadowArrays,
                    shadowSamplers,
                    input.layer,
                    input.uv,
                    input.uvLayer,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "Texture2DArray shadowArrays[4] : register(t4);" in generated_code
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[4], Texture2DArray shadowArrays[4], SamplerComparisonState shadowSamplers[4], int layer, float2 uv, float3 uvLayer, float depth, float lod, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float planarLod = shadowMaps[layer].SampleCmpLevel(shadowSamplers[layer], uv, depth, lod);"
        in generated_code
    )
    assert (
        "float planarGrad = shadowMaps[1].SampleCmpGrad(shadowSamplers[1], uv, depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float arrayLod = shadowArrays[2].SampleCmpLevel(shadowSamplers[2], uvLayer, depth, lod);"
        in generated_code
    )
    assert (
        "float arrayGrad = shadowArrays[layer].SampleCmpGrad(shadowSamplers[layer], uvLayer, depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "shadowLayer(shadowMaps, shadowArrays, shadowSamplers, input.layer, input.uv, input.uvLayer, input.depth, input.lod, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_directx_cube_shadow_compare_lod_grad_resource_arrays():
    shader = """
    shader CubeShadowCompareResourceArrays {
        samplerCubeShadow cubeMaps[4];
        samplerCubeArrayShadow cubeArrays[4];
        sampler shadowSamplers[4];

        struct FSInput {
            int layer @ TEXCOORD0;
            vec3 direction @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD3;
            vec3 ddy @ TEXCOORD4;
        };

        float cubeShadowLayer(
            samplerCubeShadow cubeMaps[],
            samplerCubeArrayShadow cubeArrays[],
            sampler shadowSamplers[],
            int layer,
            vec3 direction,
            vec4 cubeLayer,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy
        ) {
            float cubeLod = textureCompareLod(cubeMaps[layer], shadowSamplers[layer], direction, depth, lod);
            float cubeGrad = textureCompareGrad(cubeMaps[1], shadowSamplers[1], direction, depth, ddx, ddy);
            float cubeArrayLod = textureCompareLod(cubeArrays[2], shadowSamplers[2], cubeLayer, depth, lod);
            float cubeArrayGrad = textureCompareGrad(cubeArrays[layer], shadowSamplers[layer], cubeLayer, depth, ddx, ddy);
            return cubeLod + cubeGrad + cubeArrayLod + cubeArrayGrad;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return cubeShadowLayer(
                    cubeMaps,
                    cubeArrays,
                    shadowSamplers,
                    input.layer,
                    input.direction,
                    input.cubeLayer,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy
                );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "TextureCube cubeMaps[4] : register(t0);" in generated_code
    assert "TextureCubeArray cubeArrays[4] : register(t4);" in generated_code
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in generated_code
    assert (
        "float cubeShadowLayer(TextureCube cubeMaps[4], TextureCubeArray cubeArrays[4], SamplerComparisonState shadowSamplers[4], int layer, float3 direction, float4 cubeLayer, float depth, float lod, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "float cubeLod = cubeMaps[layer].SampleCmpLevel(shadowSamplers[layer], direction, depth, lod);"
        in generated_code
    )
    assert (
        "float cubeGrad = cubeMaps[1].SampleCmpGrad(shadowSamplers[1], direction, depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "float cubeArrayLod = cubeArrays[2].SampleCmpLevel(shadowSamplers[2], cubeLayer, depth, lod);"
        in generated_code
    )
    assert (
        "float cubeArrayGrad = cubeArrays[layer].SampleCmpGrad(shadowSamplers[layer], cubeLayer, depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "cubeShadowLayer(cubeMaps, cubeArrays, shadowSamplers, input.layer, input.direction, input.cubeLayer, input.depth, input.lod, input.ddx, input.ddy)"
        in generated_code
    )
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGrad(" not in generated_code


def test_directx_array_shadow_texture_resource_arrays_reject_mismatched_fixed_helper_size():
    shader = """
    shader ArrayShadowTextureResourceArrayMismatch {
        sampler2DArrayShadow shadowArrays[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            float depth;
        };

        float sampleArrayLayer(sampler2DArrayShadow shadowArrays[3], sampler shadowSamplers[3], vec3 uvLayer, float depth) {
            return textureCompare(shadowArrays[2], shadowSamplers[2], uvLayer, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return sampleArrayLayer(shadowArrays, shadowSamplers, input.uvLayer, input.depth);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'shadowArrays': 4 and 3",
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_image_size_helper_descriptor_for_storage_images():
    codegen = HLSLCodeGen()

    assert codegen.image_size_helper_descriptor("RWTexture1D<float4>") == {
        "return_type": "int",
        "dimensions": ("width",),
        "return_expr": "int(width)",
    }
    assert codegen.image_size_helper_descriptor("RWTexture2DArray<int>") == {
        "return_type": "int3",
        "dimensions": ("width", "height", "elements"),
        "return_expr": "int3(width, height, elements)",
    }
    assert codegen.image_size_helper_descriptor("RWTexture3D<uint>") == {
        "return_type": "int3",
        "dimensions": ("width", "height", "depth"),
        "return_expr": "int3(width, height, depth)",
    }
    assert codegen.image_size_helper_descriptor("RWTexture2DMS<float4>") == {
        "return_type": "int2",
        "dimensions": ("width", "height", "samples"),
        "return_expr": "int2(width, height)",
    }
    assert codegen.image_size_helper_descriptor("RWTexture2DMSArray<uint>") == {
        "return_type": "int3",
        "dimensions": ("width", "height", "elements", "samples"),
        "return_expr": "int3(width, height, elements)",
    }
    assert codegen.image_size_helper_descriptor("Texture2D") is None


def test_directx_texture_query_helper_descriptors():
    codegen = HLSLCodeGen()

    assert codegen.texture_size_helper_descriptor("Texture1D") == {
        "return_type": "int",
        "function_params": "int lod",
        "dimensions": ("width", "levels"),
        "get_dimensions_args": ("lod", "width", "levels"),
        "return_expr": "int(width)",
    }
    assert codegen.texture_size_helper_descriptor("Texture2DMSArray<float4>") == {
        "return_type": "int3",
        "function_params": "",
        "dimensions": ("width", "height", "elements", "samples"),
        "get_dimensions_args": ("width", "height", "elements", "samples"),
        "return_expr": "int3(width, height, elements)",
    }
    assert codegen.texture_query_levels_helper_descriptor("Texture3D") == {
        "return_type": "int",
        "function_params": "",
        "dimensions": ("width", "height", "depth", "levels"),
        "get_dimensions_args": ("0", "width", "height", "depth", "levels"),
        "return_expr": "int(levels)",
    }
    assert codegen.texture_query_levels_helper_descriptor("Texture2DMS<float4>") == {
        "return_type": "int",
        "function_params": "",
        "dimensions": (),
        "get_dimensions_args": (),
        "return_expr": "1",
    }
    assert codegen.texture_samples_helper_descriptor("Texture2DMSArray<float4>") == {
        "return_type": "int",
        "function_params": "",
        "dimensions": ("width", "height", "elements", "samples"),
        "get_dimensions_args": ("width", "height", "elements", "samples"),
        "return_expr": "int(samples)",
    }
    assert codegen.texture_samples_helper_descriptor("Texture2D") is None


def test_directx_texture_query_resource_descriptors():
    codegen = HLSLCodeGen()
    codegen.texture_variable_types = {
        "colorImage": "RWTexture2D<float4>",
        "colorMap": "Texture2D",
        "msTex": "Texture2DMS<float4>",
        "msArray": "Texture2DMSArray<float4>",
    }

    assert codegen.texture_query_resource_descriptor("colorImage") == {
        "texture_type": "RWTexture2D<float4>",
        "storage_image": True,
        "multisample": False,
        "size_descriptor": {
            "return_type": "int2",
            "dimensions": ("width", "height"),
            "return_expr": "int2(width, height)",
        },
        "levels_descriptor": None,
        "samples_descriptor": None,
    }
    assert codegen.texture_query_resource_descriptor("msArray") == {
        "texture_type": "Texture2DMSArray<float4>",
        "storage_image": False,
        "multisample": True,
        "size_descriptor": {
            "return_type": "int3",
            "function_params": "",
            "dimensions": ("width", "height", "elements", "samples"),
            "get_dimensions_args": ("width", "height", "elements", "samples"),
            "return_expr": "int3(width, height, elements)",
        },
        "levels_descriptor": {
            "return_type": "int",
            "function_params": "",
            "dimensions": (),
            "get_dimensions_args": (),
            "return_expr": "1",
        },
        "samples_descriptor": {
            "return_type": "int",
            "function_params": "",
            "dimensions": ("width", "height", "elements", "samples"),
            "get_dimensions_args": ("width", "height", "elements", "samples"),
            "return_expr": "int(samples)",
        },
    }
    assert (
        codegen.texture_query_size_expression("colorImage") == "imageSize(colorImage)"
    )
    assert codegen.texture_query_size_expression("msTex") == "textureSize(msTex)"
    assert (
        codegen.texture_query_size_expression("colorMap") == "textureSize(colorMap, 0)"
    )
    assert (
        codegen.texture_query_levels_expression("colorImage")
        == "/* unsupported DirectX texture query: textureQueryLevels on RWTexture2D<float4> */ 0"
    )
    assert (
        codegen.texture_query_levels_expression("colorMap")
        == "textureQueryLevels(colorMap)"
    )
    assert (
        codegen.texture_samples_expression("colorMap")
        == "/* unsupported DirectX texture samples query: requires multisample texture */ 0"
    )
    assert codegen.texture_samples_expression("msArray") == "textureSamples(msArray)"
    assert (
        codegen.texture_sample_position_expression("colorMap", "sampleIndex")
        == "/* unsupported DirectX texture sample-position query: "
        "textureSamplePosition on Texture2D requires sampled multisample texture */ "
        "float2(0.0, 0.0)"
    )
    assert (
        codegen.texture_sample_position_expression("msTex", "sampleIndex")
        == "msTex.GetSamplePosition(sampleIndex)"
    )
    assert (
        "imageSize",
        "RWTexture2D<float4>",
    ) in codegen.required_texture_query_helpers
    assert ("textureSize", "Texture2D") in codegen.required_texture_query_helpers
    assert (
        "textureSize",
        "Texture2DMS<float4>",
    ) in codegen.required_texture_query_helpers
    assert (
        "textureQueryLevels",
        "Texture2D",
    ) in codegen.required_texture_query_helpers
    assert (
        "textureSamples",
        "Texture2DMSArray<float4>",
    ) in codegen.required_texture_query_helpers


def test_directx_storage_image_size_queries_use_get_dimensions_helpers():
    shader = """
    shader StorageImageSizeQueries {
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;
        uimage2D uintImage;
        iimage3D intVolume;

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        ivec2 queryImage2D(image2D image) {
            return textureSize(image, 0) + imageSize(image);
        }

        ivec3 queryImage3D(image3D image) {
            return textureSize(image, 0) + imageSize(image);
        }

        ivec3 queryImageArray(image2DArray image) {
            return textureSize(image, 0) + imageSize(image);
        }

        ivec2 queryUintImage(uimage2D image) {
            return textureSize(image, 0) + imageSize(image);
        }

        ivec3 queryIntVolume(iimage3D image) {
            return textureSize(image, 0) + imageSize(image);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                ivec2 a = textureSize(colorImage, 0) + imageSize(colorImage);
                ivec3 b = textureSize(volumeImage, 0) + imageSize(volumeImage);
                ivec3 c = textureSize(layerImage, 0) + imageSize(layerImage);
                ivec2 d = textureSize(uintImage, 0) + imageSize(uintImage);
                ivec3 e = textureSize(intVolume, 0) + imageSize(intVolume);
                return vec4(a.x + b.x + c.z + d.x + e.z);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "RWTexture2D<float4> colorImage : register(u0);" in generated_code
    assert "RWTexture3D<float4> volumeImage : register(u1);" in generated_code
    assert "RWTexture2DArray<float4> layerImage : register(u2);" in generated_code
    assert "RWTexture2D<uint> uintImage : register(u3);" in generated_code
    assert "RWTexture3D<int> intVolume : register(u4);" in generated_code
    assert "int2 imageSize(RWTexture2D<float4> image)" in generated_code
    assert "int2 imageSize(RWTexture2D<uint> image)" in generated_code
    assert "int3 imageSize(RWTexture3D<float4> image)" in generated_code
    assert "int3 imageSize(RWTexture3D<int> image)" in generated_code
    assert "int3 imageSize(RWTexture2DArray<float4> image)" in generated_code
    assert "image.GetDimensions(width, height);" in generated_code
    assert "image.GetDimensions(width, height, depth);" in generated_code
    assert "image.GetDimensions(width, height, elements);" in generated_code
    assert "int2 queryImage2D(RWTexture2D<float4> image)" in generated_code
    assert "int3 queryImageArray(RWTexture2DArray<float4> image)" in generated_code
    assert "return (imageSize(image) + imageSize(image));" in generated_code
    assert "int2 a = (imageSize(colorImage) + imageSize(colorImage));" in generated_code
    assert (
        "int3 b = (imageSize(volumeImage) + imageSize(volumeImage));" in generated_code
    )
    assert "int3 c = (imageSize(layerImage) + imageSize(layerImage));" in generated_code
    assert "int2 d = (imageSize(uintImage) + imageSize(uintImage));" in generated_code
    assert "int3 e = (imageSize(intVolume) + imageSize(intVolume));" in generated_code
    assert "textureSize(" not in generated_code


def test_directx_storage_image_levels_and_lod_queries_emit_diagnostics():
    shader = """
    shader StorageImageInvalidQueries {
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;
        uimage2D uintImage;
        iimage3D intVolume;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvw @ TEXCOORD1;
        };

        int imageLevels(image2D image) {
            return textureQueryLevels(image);
        }

        vec2 imageLod(image2D image, vec2 uv) {
            return textureQueryLod(image, uv);
        }

        int volumeLevels(image3D image) {
            return textureQueryLevels(image);
        }

        vec2 volumeLod(image3D image, vec3 uvw) {
            return textureQueryLod(image, uvw);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                int levels = textureQueryLevels(colorImage)
                    + textureQueryLevels(volumeImage)
                    + textureQueryLevels(layerImage)
                    + textureQueryLevels(uintImage)
                    + textureQueryLevels(intVolume);
                vec2 lod = textureQueryLod(colorImage, input.uv)
                    + textureQueryLod(volumeImage, input.uvw)
                    + imageLod(colorImage, input.uv)
                    + volumeLod(volumeImage, input.uvw);
                return vec4(float(levels) + lod.x);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert (
        generated_code.count(
            "/* unsupported DirectX texture query: textureQueryLevels on RWTexture2D<float4> */ 0"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported DirectX texture query: textureQueryLevels on RWTexture3D<float4> */ 0"
        )
        == 2
    )
    assert (
        "/* unsupported DirectX texture query: textureQueryLevels on RWTexture2DArray<float4> */ 0"
        in generated_code
    )
    assert (
        "/* unsupported DirectX texture query: textureQueryLevels on RWTexture2D<uint> */ 0"
        in generated_code
    )
    assert (
        "/* unsupported DirectX texture query: textureQueryLevels on RWTexture3D<int> */ 0"
        in generated_code
    )
    assert (
        generated_code.count(
            "/* unsupported DirectX texture query: textureQueryLod on RWTexture2D<float4> */ float2(0.0, 0.0)"
        )
        == 2
    )
    assert (
        generated_code.count(
            "/* unsupported DirectX texture query: textureQueryLod on RWTexture3D<float4> */ float2(0.0, 0.0)"
        )
        == 2
    )
    assert "textureQueryLevels(" not in generated_code
    assert "textureQueryLod(" not in generated_code
    assert "CalculateLevelOfDetail" not in generated_code
    assert "imageSampler" not in generated_code
    assert "SamplerState" not in generated_code


def test_directx_texture_query_functions():
    shader = """
    shader TextureQueries {
        sampler2D colorMap;
        sampler2DArray layerMap;
        sampler2DMS msMap;
        sampler linearSampler;

        struct VSOutput {
            vec2 uv;
        };

        ivec2 query2D(sampler2D tex, sampler s, vec2 uv) {
            ivec2 size = textureSize(tex, 1);
            int levels = textureQueryLevels(tex);
            vec2 lod = textureQueryLod(tex, s, uv);
            return size + ivec2(levels) + ivec2(lod);
        }

        ivec3 queryArray(sampler2DArray tex) {
            return textureSize(tex, 0);
        }

        ivec2 queryMs(sampler2DMS tex) {
            return textureSize(tex);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                ivec2 q = query2D(colorMap, linearSampler, input.uv);
                ivec3 qa = queryArray(layerMap);
                ivec2 qm = queryMs(msMap);
                return vec4(q.x + qa.z + qm.x);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "int textureQueryLevels(Texture2D tex)" in generated_code
    assert "tex.GetDimensions(0, width, height, levels);" in generated_code
    assert "int2 textureSize(Texture2D tex, int lod)" in generated_code
    assert "tex.GetDimensions(lod, width, height, levels);" in generated_code
    assert "int3 textureSize(Texture2DArray tex, int lod)" in generated_code
    assert "tex.GetDimensions(lod, width, height, elements, levels);" in generated_code
    assert "int2 textureSize(Texture2DMS<float4> tex)" in generated_code
    assert "tex.GetDimensions(width, height, samples);" in generated_code
    assert "int2 size = textureSize(tex, 1);" in generated_code
    assert "int levels = textureQueryLevels(tex);" in generated_code
    assert (
        "float2 lod = float2(tex.CalculateLevelOfDetailUnclamped(s, uv), tex.CalculateLevelOfDetail(s, uv));"
        in generated_code
    )
    assert "return textureSize(tex, 0);" in generated_code
    assert "return textureSize(tex);" in generated_code


def test_directx_multisample_texture_samples_queries_use_helpers():
    shader = """
    shader MultisampleSamplesQuery {
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler2DMS msTextures[4];
        sampler2DMSArray msArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        int querySamples(sampler2DMS tex, sampler2DMSArray texArray) {
            return textureSamples(tex) + textureSamples(texArray);
        }

        int queryArraySamples(sampler2DMS textures[], sampler2DMSArray arrays[], int layer) {
            return textureSamples(textures[2]) + textureSamples(arrays[layer]);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return float(querySamples(msTex, msArray)
                    + queryArraySamples(msTextures, msArrays, input.layer));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2DMS<float4> msTex : register(t0);" in generated_code
    assert "Texture2DMSArray<float4> msArray : register(t1);" in generated_code
    assert "Texture2DMS<float4> msTextures[4] : register(t2);" in generated_code
    assert "Texture2DMSArray<float4> msArrays[4] : register(t6);" in generated_code
    assert "int textureSamples(Texture2DMS<float4> tex)" in generated_code
    assert "tex.GetDimensions(width, height, samples);" in generated_code
    assert "int textureSamples(Texture2DMSArray<float4> tex)" in generated_code
    assert "tex.GetDimensions(width, height, elements, samples);" in generated_code
    assert (
        "int querySamples(Texture2DMS<float4> tex, Texture2DMSArray<float4> texArray)"
        in generated_code
    )
    assert "return (textureSamples(tex) + textureSamples(texArray));" in generated_code
    assert (
        "int queryArraySamples(Texture2DMS<float4> textures[4], Texture2DMSArray<float4> arrays[4], int layer)"
        in generated_code
    )
    assert (
        "return (textureSamples(textures[2]) + textureSamples(arrays[layer]));"
        in generated_code
    )
    assert (
        "querySamples(msTex, msArray) + queryArraySamples(msTextures, msArrays, input.layer)"
        in generated_code
    )
    assert "SamplerState msTexSampler" not in generated_code
    assert "SamplerState msArraySampler" not in generated_code


def test_directx_multisample_texture_sample_position_queries_use_hlsl_member():
    shader = """
    shader MultisampleSamplePositionQuery {
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler2DMS msTextures[4];
        sampler2DMSArray msArrays[4];
        sampler2D colorMap;

        struct FSInput {
            int sampleIndex @ SV_SampleIndex;
            int layer @ TEXCOORD0;
        };

        vec2 queryPosition(sampler2DMS tex, int sampleIndex) {
            return textureSamplePosition(tex, sampleIndex);
        }

        vec2 queryArrayPosition(sampler2DMSArray arrays[], int layer, int sampleIndex) {
            return textureSamplePosition(arrays[layer], sampleIndex);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 direct = textureSamplePosition(msTex, input.sampleIndex);
                vec2 arrayDirect = textureSamplePosition(msArray, input.sampleIndex);
                vec2 indexed = textureSamplePosition(msTextures[2], input.sampleIndex)
                    + queryArrayPosition(msArrays, input.layer, input.sampleIndex);
                vec2 invalid = textureSamplePosition(colorMap, input.sampleIndex);
                return vec4(direct + arrayDirect + indexed + invalid, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2DMS<float4> msTex : register(t0);" in generated_code
    assert "Texture2DMSArray<float4> msArray : register(t1);" in generated_code
    assert (
        "float2 queryPosition(Texture2DMS<float4> tex, int sampleIndex)"
        in generated_code
    )
    assert "return tex.GetSamplePosition(sampleIndex);" in generated_code
    assert (
        "float2 queryArrayPosition(Texture2DMSArray<float4> arrays[4], int layer, int sampleIndex)"
        in generated_code
    )
    assert "return arrays[layer].GetSamplePosition(sampleIndex);" in generated_code
    assert (
        "float2 direct = msTex.GetSamplePosition(input.sampleIndex);" in generated_code
    )
    assert (
        "float2 arrayDirect = msArray.GetSamplePosition(input.sampleIndex);"
        in generated_code
    )
    assert "msTextures[2].GetSamplePosition(input.sampleIndex)" in generated_code
    assert "textureSamplePosition(" not in generated_code
    assert (
        "/* unsupported DirectX texture sample-position query: "
        "textureSamplePosition on Texture2D requires sampled multisample texture */ "
        "float2(0.0, 0.0)"
    ) in generated_code
    assert "SamplerState msTexSampler" not in generated_code
    assert "SamplerState msArraySampler" not in generated_code


def test_directx_struct_member_resource_queries_use_hlsl_members():
    shader = """
    shader StructMemberResourceQueries {
        struct ResourceBundle {
            sampler2D color;
            sampler2DMS ms;
            image2D image;
        };

        ResourceBundle resources;
        sampler linearSampler;

        fragment {
            vec4 main(vec2 uv @ TEXCOORD0, ivec2 pixel @ TEXCOORD1, int sampleIndex @ SV_SampleIndex) @ gl_FragColor {
                ivec2 dims = textureSize(resources.color, 0);
                int levels = textureQueryLevels(resources.color);
                vec2 lod = textureQueryLod(resources.color, linearSampler, uv);
                vec2 pos = textureSamplePosition(resources.ms, sampleIndex);
                vec4 stored = imageLoad(resources.image, pixel);
                return texture(resources.color, linearSampler, uv)
                    + stored
                    + vec4(float(dims.x + dims.y + levels) + lod.y + pos.x);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D color;" in generated_code
    assert "Texture2DMS<float4> ms;" in generated_code
    assert "RWTexture2D<float4> image;" in generated_code
    assert "int2 textureSize(Texture2D tex, int lod)" in generated_code
    assert "int textureQueryLevels(Texture2D tex)" in generated_code
    assert "int2 dims = textureSize(resources.color, 0);" in generated_code
    assert "int levels = textureQueryLevels(resources.color);" in generated_code
    assert (
        "float2 lod = float2(resources.color.CalculateLevelOfDetailUnclamped("
        "linearSampler, uv), resources.color.CalculateLevelOfDetail(linearSampler, uv));"
    ) in generated_code
    assert "float2 pos = resources.ms.GetSamplePosition(sampleIndex);" in generated_code
    assert "float4 stored = resources.image[pixel];" in generated_code
    assert "resources.color.Sample(linearSampler, uv)" in generated_code
    assert "unsupported DirectX texture sample-position query" not in generated_code


def test_directx_indexed_struct_member_resource_queries_use_hlsl_members():
    shader = """
    shader IndexedStructMemberResourceQueries {
        struct ResourceBundle {
            sampler2D color;
            sampler2DMS ms;
            image2D image;
        };

        ResourceBundle resources[2];
        sampler linearSampler;

        fragment {
            vec4 main(vec2 uv @ TEXCOORD0, ivec2 pixel @ TEXCOORD1, int layer @ TEXCOORD2, int sampleIndex @ SV_SampleIndex) @ gl_FragColor {
                ivec2 dims = textureSize(resources[layer].color, 0);
                int levels = textureQueryLevels(resources[layer].color);
                vec2 lod = textureQueryLod(resources[layer].color, linearSampler, uv);
                vec2 pos = textureSamplePosition(resources[layer].ms, sampleIndex);
                vec4 stored = imageLoad(resources[layer].image, pixel);
                return texture(resources[layer].color, linearSampler, uv)
                    + stored
                    + vec4(float(dims.x + dims.y + levels) + lod.y + pos.x);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "ResourceBundle resources[2];" in generated_code
    assert "int2 textureSize(Texture2D tex, int lod)" in generated_code
    assert "int textureQueryLevels(Texture2D tex)" in generated_code
    assert "int2 dims = textureSize(resources[layer].color, 0);" in generated_code
    assert "int levels = textureQueryLevels(resources[layer].color);" in generated_code
    assert (
        "float2 lod = float2(resources[layer].color.CalculateLevelOfDetailUnclamped("
        "linearSampler, uv), resources[layer].color.CalculateLevelOfDetail(linearSampler, uv));"
    ) in generated_code
    assert (
        "float2 pos = resources[layer].ms.GetSamplePosition(sampleIndex);"
        in generated_code
    )
    assert "float4 stored = resources[layer].image[pixel];" in generated_code
    assert "resources[layer].color.Sample(linearSampler, uv)" in generated_code
    assert "unsupported DirectX texture sample-position query" not in generated_code


def test_directx_typed_multisample_texture_queries_and_fetches_use_hlsl_ms_methods():
    shader = """
    shader DirectXTypedMultisampleTextures {
        Texture2DMS<uint> msUint @register(t0);
        Texture2DMSArray<int> msIntArray @register(t1);

        struct FSInput {
            ivec2 pixel @ TEXCOORD0;
            ivec3 pixelLayer @ TEXCOORD1;
            int sampleIndex @ SV_SampleIndex;
        };

        int queryArray(Texture2DMSArray<int> arrayTex, ivec3 pixelLayer, int sampleIndex) {
            ivec3 size = textureSize(arrayTex);
            int samples = textureSamples(arrayTex);
            int levels = textureQueryLevels(arrayTex);
            vec2 pos = textureSamplePosition(arrayTex, sampleIndex);
            int fetched = texelFetch(arrayTex, pixelLayer, sampleIndex);
            return fetched + size.z + samples + levels + int(pos.x);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                ivec2 size = textureSize(msUint);
                int samples = textureSamples(msUint);
                int levels = textureQueryLevels(msUint);
                vec2 pos = textureSamplePosition(msUint, input.sampleIndex);
                uint fetched = texelFetch(msUint, input.pixel, input.sampleIndex);
                return vec4(float(fetched + uint(size.x + samples + levels + int(pos.x)
                    + queryArray(msIntArray, input.pixelLayer, input.sampleIndex))));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2DMS<uint> msUint : register(t0);" in generated_code
    assert "Texture2DMSArray<int> msIntArray : register(t1);" in generated_code
    assert "int2 textureSize(Texture2DMS<uint> tex)" in generated_code
    assert "int3 textureSize(Texture2DMSArray<int> tex)" in generated_code
    assert "int textureSamples(Texture2DMS<uint> tex)" in generated_code
    assert "int textureSamples(Texture2DMSArray<int> tex)" in generated_code
    assert "int textureQueryLevels(Texture2DMS<uint> tex)" in generated_code
    assert "int textureQueryLevels(Texture2DMSArray<int> tex)" in generated_code
    assert (
        "uint fetched = msUint.Load(input.pixel, input.sampleIndex);" in generated_code
    )
    assert "int fetched = arrayTex.Load(pixelLayer, sampleIndex);" in generated_code
    assert "msUint.GetSamplePosition(input.sampleIndex)" in generated_code
    assert "arrayTex.GetSamplePosition(sampleIndex)" in generated_code
    assert "unsupported DirectX texture sample-position query" not in generated_code
    assert "unsupported DirectX texture samples query" not in generated_code
    assert "unsupported DirectX texel fetch" not in generated_code
    assert ".Load(int3(" not in generated_code
    assert ".Load(int4(" not in generated_code


def test_directx_typed_sampled_texture_queries_gather_and_fetches_use_resource_shape():
    shader = """
    shader DirectXTypedSampledTextures {
        Texture2D<float4> color : register(t0);
        Texture2DArray<uint4> layers : register(t1);
        Texture3D<int4> volume : register(t2);
        TextureCube<float4> cube : register(t3);
        sampler linearSampler : register(s0);

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec3 uvw @ TEXCOORD2;
            vec3 direction @ TEXCOORD3;
            ivec2 pixel @ TEXCOORD4;
            ivec3 pixelLayer @ TEXCOORD5;
            ivec3 voxel @ TEXCOORD6;
            ivec3 voxelOffset @ TEXCOORD7;
            int lod @ TEXCOORD8;
            int component @ TEXCOORD9;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                ivec2 colorSize = textureSize(color, input.lod);
                int colorLevels = textureQueryLevels(color);
                ivec3 layerSize = textureSize(layers, input.lod);
                uvec4 gathered = textureGather(
                    layers, linearSampler, input.uvLayer, input.component
                );
                uvec4 fetchedLayer = texelFetch(layers, input.pixelLayer, input.lod);
                ivec4 fetchedVolume = texelFetch(volume, input.voxel, input.lod);
                ivec4 offsetVolume = texelFetchOffset(
                    volume, input.voxel, input.lod, input.voxelOffset
                );
                vec4 cubeColor = texture(cube, linearSampler, input.direction);
                return cubeColor
                    + vec4(colorSize, colorLevels, 1.0)
                    + vec4(layerSize, 1.0)
                    + vec4(gathered)
                    + vec4(fetchedLayer)
                    + vec4(fetchedVolume + offsetVolume);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D<float4> color : register(t0);" in generated_code
    assert "Texture2DArray<uint4> layers : register(t1);" in generated_code
    assert "Texture3D<int4> volume : register(t2);" in generated_code
    assert "TextureCube<float4> cube : register(t3);" in generated_code
    assert "int2 textureSize(Texture2D<float4> tex, int lod)" in generated_code
    assert "int3 textureSize(Texture2DArray<uint4> tex, int lod)" in generated_code
    assert "int textureQueryLevels(Texture2D<float4> tex)" in generated_code
    assert "int2 colorSize = textureSize(color, input.lod);" in generated_code
    assert "int colorLevels = textureQueryLevels(color);" in generated_code
    assert "int3 layerSize = textureSize(layers, input.lod);" in generated_code
    assert (
        "input.component == 0 ? layers.GatherRed(linearSampler, input.uvLayer) : "
        "input.component == 1 ? layers.GatherGreen(linearSampler, input.uvLayer) : "
        "input.component == 2 ? layers.GatherBlue(linearSampler, input.uvLayer) : "
        "layers.GatherAlpha(linearSampler, input.uvLayer)"
    ) in generated_code
    assert (
        "uint4 fetchedLayer = layers.Load(int4(input.pixelLayer, input.lod));"
        in generated_code
    )
    assert (
        "int4 fetchedVolume = volume.Load(int4(input.voxel, input.lod));"
        in generated_code
    )
    assert (
        "int4 offsetVolume = volume.Load(int4((input.voxel + input.voxelOffset), input.lod));"
        in generated_code
    )
    assert (
        "float4 cubeColor = cube.Sample(linearSampler, input.direction);"
        in generated_code
    )
    assert "unsupported DirectX texture gather" not in generated_code
    assert "textureGather(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert ".Load(int3(input.pixelLayer" not in generated_code
    assert ".Load(int3(input.voxel" not in generated_code


def test_directx_typed_comparison_texture_operations_use_resource_shape():
    shader = """
    shader DirectXTypedComparisonTextures {
        Texture2D<float> shadow2D : register(t0);
        Texture2DArray<float> shadowArray : register(t1);
        TextureCube<float> cubeShadow : register(t2);
        TextureCubeArray<float> cubeArray : register(t3);
        sampler compareSampler : register(s0);

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec3 direction @ TEXCOORD2;
            vec4 cubeLayer @ TEXCOORD3;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD4;
            vec2 ddy @ TEXCOORD5;
            vec3 cubeDdx @ TEXCOORD6;
            vec3 cubeDdy @ TEXCOORD7;
            ivec2 offset @ TEXCOORD8;
        };

        float inspectImplicit(
            Texture2D<float> tex,
            vec2 uv,
            float depth,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec2 lodInfo = textureQueryLod(tex, uv);
            float cmp = textureCompare(tex, uv, depth);
            float grad = textureCompareGradOffset(tex, uv, depth, ddx, ddy, offset);
            return lodInfo.x + lodInfo.y + cmp + grad;
        }

        float explicitArray(
            Texture2DArray<float> tex,
            sampler s,
            vec3 uvLayer,
            float depth,
            float lod,
            ivec2 offset
        ) {
            float cmp = textureCompareLod(tex, s, uvLayer, depth, lod);
            vec4 gathered = textureGatherCompareOffset(
                tex, s, uvLayer, depth, offset
            );
            return cmp + gathered.x;
        }

        float explicitCube(
            TextureCube<float> tex,
            sampler s,
            vec3 direction,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy
        ) {
            float cmp = textureCompareLod(tex, s, direction, depth, lod);
            float grad = textureCompareGrad(tex, s, direction, depth, ddx, ddy);
            return cmp + grad;
        }

        float unsupportedCubeArrayOffset(
            TextureCubeArray<float> tex,
            sampler s,
            vec4 cubeLayer,
            float depth,
            ivec2 offset
        ) {
            return textureCompareOffset(tex, s, cubeLayer, depth, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float implicitValue = inspectImplicit(
                    shadow2D,
                    input.uv,
                    input.depth,
                    input.ddx,
                    input.ddy,
                    input.offset
                );
                float arrayValue = explicitArray(
                    shadowArray,
                    compareSampler,
                    input.uvLayer,
                    input.depth,
                    input.lod,
                    input.offset
                );
                float cubeValue = explicitCube(
                    cubeShadow,
                    compareSampler,
                    input.direction,
                    input.depth,
                    input.lod,
                    input.cubeDdx,
                    input.cubeDdy
                );
                float diagnosticValue = unsupportedCubeArrayOffset(
                    cubeArray,
                    compareSampler,
                    input.cubeLayer,
                    input.depth,
                    input.offset
                );
                return vec4(implicitValue + arrayValue + cubeValue + diagnosticValue);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D<float> shadow2D : register(t0);" in generated_code
    assert "Texture2DArray<float> shadowArray : register(t1);" in generated_code
    assert "TextureCube<float> cubeShadow : register(t2);" in generated_code
    assert "TextureCubeArray<float> cubeArray : register(t3);" in generated_code
    assert "SamplerComparisonState compareSampler : register(s0);" in generated_code
    assert "SamplerComparisonState shadow2DSampler : register(s1);" in generated_code
    assert "SamplerState shadow2DQuerySampler : register(s2);" in generated_code
    assert (
        "float inspectImplicit(Texture2D<float> tex, SamplerComparisonState texSampler, "
        "SamplerState texQuerySampler, float2 uv, float depth, float2 ddx, float2 ddy, "
        "int2 offset)"
    ) in generated_code
    assert (
        "float2 lodInfo = float2(tex.CalculateLevelOfDetailUnclamped(texQuerySampler, uv), "
        "tex.CalculateLevelOfDetail(texQuerySampler, uv));"
    ) in generated_code
    assert "float cmp = tex.SampleCmp(texSampler, uv, depth);" in generated_code
    assert (
        "float grad = tex.SampleCmpGrad(texSampler, uv, depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float explicitArray(Texture2DArray<float> tex, SamplerComparisonState s, "
        "float3 uvLayer, float depth, float lod, int2 offset)"
    ) in generated_code
    assert "float cmp = tex.SampleCmpLevel(s, uvLayer, depth, lod);" in generated_code
    assert "tex.GatherCmp(s, uvLayer, depth, offset)" in generated_code
    assert (
        "float explicitCube(TextureCube<float> tex, SamplerComparisonState s, "
        "float3 direction, float depth, float lod, float3 ddx, float3 ddy)"
    ) in generated_code
    assert "tex.SampleCmpLevel(s, direction, depth, lod)" in generated_code
    assert "tex.SampleCmpGrad(s, direction, depth, ddx, ddy)" in generated_code
    assert (
        "float unsupportedCubeArrayOffset(TextureCubeArray<float> tex, "
        "SamplerComparisonState s, float4 cubeLayer, float depth, int2 offset)"
    ) in generated_code
    assert (
        "/* unsupported DirectX texture compare: textureCompareOffset offsets require "
        "2D or 2D-array textures */ 0.0"
    ) in generated_code
    assert (
        "inspectImplicit(shadow2D, shadow2DSampler, shadow2DQuerySampler, input.uv, "
        "input.depth, input.ddx, input.ddy, input.offset)"
    ) in generated_code
    assert (
        "explicitArray(shadowArray, compareSampler, input.uvLayer, input.depth, "
        "input.lod, input.offset)"
    ) in generated_code
    assert (
        "explicitCube(cubeShadow, compareSampler, input.direction, input.depth, "
        "input.lod, input.cubeDdx, input.cubeDdy)"
    ) in generated_code
    assert (
        "unsupportedCubeArrayOffset(cubeArray, compareSampler, input.cubeLayer, "
        "input.depth, input.offset)"
    ) in generated_code
    assert "SamplerState s, float4 cubeLayer" not in generated_code
    assert "Texture2D<float4> shadow2D" not in generated_code
    assert "Texture2D shadowArray : register" not in generated_code
    assert "textureCompare(" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGrad(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code
    assert "textureQueryLod(" not in generated_code


def test_directx_multisample_image_samples_queries_use_helpers():
    shader = """
    shader MultisampleImageSamplesQuery {
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler2DMS msTextures[4];
        sampler2DMSArray msArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        int querySamples(sampler2DMS tex, sampler2DMSArray texArray) {
            return imageSamples(tex) + imageSamples(texArray);
        }

        int queryArraySamples(sampler2DMS textures[], sampler2DMSArray arrays[], int layer) {
            return imageSamples(textures[2]) + imageSamples(arrays[layer]);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return float(imageSamples(msTex)
                    + imageSamples(msArray)
                    + querySamples(msTex, msArray)
                    + queryArraySamples(msTextures, msArrays, input.layer));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2DMS<float4> msTex : register(t0);" in generated_code
    assert "Texture2DMSArray<float4> msArray : register(t1);" in generated_code
    assert "Texture2DMS<float4> msTextures[4] : register(t2);" in generated_code
    assert "Texture2DMSArray<float4> msArrays[4] : register(t6);" in generated_code
    assert "int textureSamples(Texture2DMS<float4> tex)" in generated_code
    assert "int textureSamples(Texture2DMSArray<float4> tex)" in generated_code
    assert (
        "int querySamples(Texture2DMS<float4> tex, Texture2DMSArray<float4> texArray)"
        in generated_code
    )
    assert "return (textureSamples(tex) + textureSamples(texArray));" in generated_code
    assert (
        "int queryArraySamples(Texture2DMS<float4> textures[4], Texture2DMSArray<float4> arrays[4], int layer)"
        in generated_code
    )
    assert (
        "return (textureSamples(textures[2]) + textureSamples(arrays[layer]));"
        in generated_code
    )
    assert "textureSamples(msTex) + textureSamples(msArray)" in generated_code
    assert "queryArraySamples(msTextures, msArrays, input.layer)" in generated_code
    assert "imageSamples(" not in generated_code
    assert "unsupported DirectX texture samples query" not in generated_code
    assert "SamplerState msTexSampler" not in generated_code
    assert "SamplerState msArraySampler" not in generated_code


def test_directx_non_multisample_texture_samples_emit_diagnostics_without_samplers():
    shader = """
    shader NonMultisampleSamplesQuery {
        sampler2D colorMap;
        sampler2DArray layerMap;
        samplerCube cubeMap;
        samplerCubeArray cubeArray;
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        sampler2D textures[4];

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        int querySamples(
            sampler2D tex,
            sampler2DArray arrayTex,
            samplerCube cubeTex,
            samplerCubeArray cubeArrayTex,
            sampler2DShadow shadowTex,
            sampler2DArrayShadow shadowArrayTex
        ) {
            return textureSamples(tex)
                + textureSamples(arrayTex)
                + textureSamples(cubeTex)
                + textureSamples(cubeArrayTex)
                + textureSamples(shadowTex)
                + textureSamples(shadowArrayTex);
        }

        int queryResourceArray(sampler2D textures[], int layer) {
            return textureSamples(textures[2]) + textureSamples(textures[layer]);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                int direct = textureSamples(colorMap) + textureSamples(shadowArray);
                return float(querySamples(
                    colorMap,
                    layerMap,
                    cubeMap,
                    cubeArray,
                    shadowMap,
                    shadowArray
                ) + queryResourceArray(textures, input.layer) + direct);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported DirectX texture samples query: "
        "requires multisample texture */ 0"
    )
    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "Texture2DArray layerMap : register(t1);" in generated_code
    assert "TextureCube cubeMap : register(t2);" in generated_code
    assert "TextureCubeArray cubeArray : register(t3);" in generated_code
    assert "Texture2D textures[4]" in generated_code
    assert generated_code.count(diagnostic) == 10
    assert "textureSamples(" not in generated_code
    assert "GetDimensions(width, height, samples)" not in generated_code
    assert "SamplerState" not in generated_code
    assert "SamplerComparisonState" not in generated_code


def test_directx_storage_image_samples_queries_emit_diagnostics():
    shader = """
    shader StorageImageSamplesQueries {
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;
        uimage2D uintImage;
        iimage3D intVolume;

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        int samples2D(image2D image) {
            return imageSamples(image) + textureSamples(image);
        }

        int samples3D(image3D image) {
            return imageSamples(image) + textureSamples(image);
        }

        int samplesArray(image2DArray image) {
            return imageSamples(image) + textureSamples(image);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                int samples = imageSamples(colorImage)
                    + imageSamples(volumeImage)
                    + imageSamples(layerImage)
                    + imageSamples(uintImage)
                    + imageSamples(intVolume)
                    + textureSamples(colorImage)
                    + textureSamples(volumeImage);
                return float(samples
                    + samples2D(colorImage)
                    + samples3D(volumeImage)
                    + samplesArray(layerImage));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported DirectX texture samples query: "
        "requires multisample texture */ 0"
    )
    assert "RWTexture2D<float4> colorImage : register(u0);" in generated_code
    assert "RWTexture3D<float4> volumeImage : register(u1);" in generated_code
    assert "RWTexture2DArray<float4> layerImage : register(u2);" in generated_code
    assert "RWTexture2D<uint> uintImage : register(u3);" in generated_code
    assert "RWTexture3D<int> intVolume : register(u4);" in generated_code
    assert generated_code.count(diagnostic) == 13
    assert "imageSamples(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "GetDimensions(width, height, samples)" not in generated_code
    assert "SamplerState" not in generated_code
    assert "SamplerComparisonState" not in generated_code


def test_directx_storage_image_sampling_and_fetch_emit_diagnostics():
    shader = """
    shader StorageImageInvalidSampleFetch {
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvw @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            ivec2 pixel @ TEXCOORD3;
            ivec3 voxel @ TEXCOORD4;
            ivec3 pixelLayer @ TEXCOORD5;
            ivec2 offset2 @ TEXCOORD6;
            ivec3 offset3 @ TEXCOORD7;
            int lod;
        };

        vec4 invalid2D(image2D image, vec2 uv, ivec2 pixel, int lod, ivec2 offset) {
            return texture(image, uv)
                + textureLod(image, uv, float(lod))
                + textureGather(image, uv)
                + texelFetch(image, pixel, lod)
                + texelFetchOffset(image, pixel, lod, offset);
        }

        vec4 invalid3D(image3D image, vec3 uvw, ivec3 voxel, int lod, ivec3 offset) {
            return texture(image, uvw)
                + textureLod(image, uvw, float(lod))
                + textureGather(image, uvw)
                + texelFetch(image, voxel, lod)
                + texelFetchOffset(image, voxel, lod, offset);
        }

        vec4 invalidArray(image2DArray image, vec3 uvLayer, ivec3 pixelLayer, int lod, ivec2 offset) {
            return texture(image, uvLayer)
                + textureLod(image, uvLayer, float(lod))
                + textureGather(image, uvLayer)
                + texelFetch(image, pixelLayer, lod)
                + texelFetchOffset(image, pixelLayer, lod, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return invalid2D(colorImage, input.uv, input.pixel, input.lod, input.offset2)
                    + invalid3D(volumeImage, input.uvw, input.voxel, input.lod, input.offset3)
                    + invalidArray(layerImage, input.uvLayer, input.pixelLayer, input.lod, input.offset2)
                    + texture(colorImage, input.uv)
                    + textureLod(volumeImage, input.uvw, float(input.lod))
                    + textureGather(layerImage, input.uvLayer)
                    + texelFetch(colorImage, input.pixel, input.lod)
                    + texelFetchOffset(layerImage, input.pixelLayer, input.lod, input.offset2);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = "unsupported DirectX storage image texture operation"
    assert "RWTexture2D<float4> colorImage : register(u0);" in generated_code
    assert "RWTexture3D<float4> volumeImage : register(u1);" in generated_code
    assert "RWTexture2DArray<float4> layerImage : register(u2);" in generated_code
    assert generated_code.count(diagnostic) == 20
    for operation in {
        "texture",
        "textureLod",
        "textureGather",
        "texelFetch",
        "texelFetchOffset",
    }:
        assert f"{operation} on RWTexture2D<float4>" in generated_code
    assert ".Sample" not in generated_code
    assert ".Load(" not in generated_code
    assert "texture(" not in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGather(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "texelFetchOffset(" not in generated_code
    assert "SamplerState" not in generated_code
    assert "imageSampler" not in generated_code
    assert "colorImageSampler" not in generated_code


def test_directx_storage_image_compare_calls_emit_diagnostics_without_comparison_samplers():
    shader = """
    shader StorageImageInvalidCompare {
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;
        sampler compareSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvw @ TEXCOORD1;
            vec3 uvLayer @ TEXCOORD2;
            vec2 ddx @ TEXCOORD3;
            vec2 ddy @ TEXCOORD4;
            ivec2 offset @ TEXCOORD5;
            float depth;
            float lod;
        };

        float compareImplicit(image2D image, vec2 uv, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset) {
            return textureCompare(image, uv, depth)
                + textureCompareOffset(image, uv, depth, offset)
                + textureCompareLod(image, uv, depth, lod)
                + textureCompareLodOffset(image, uv, depth, lod, offset)
                + textureCompareGrad(image, uv, depth, ddx, ddy)
                + textureCompareGradOffset(image, uv, depth, ddx, ddy, offset);
        }

        float compareExplicit(image3D image, sampler s, vec3 uvw, float depth, float lod, vec2 ddx, vec2 ddy, ivec2 offset) {
            return textureCompare(image, s, uvw, depth)
                + textureCompareOffset(image, s, uvw, depth, offset)
                + textureCompareLod(image, s, uvw, depth, lod)
                + textureCompareLodOffset(image, s, uvw, depth, lod, offset)
                + textureCompareGrad(image, s, uvw, depth, ddx, ddy)
                + textureCompareGradOffset(image, s, uvw, depth, ddx, ddy, offset);
        }

        vec4 gatherExplicit(image2DArray image, sampler s, vec3 uvLayer, float depth, ivec2 offset) {
            return textureGatherCompare(image, s, uvLayer, depth)
                + textureGatherCompareOffset(image, s, uvLayer, depth, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float cmp = compareImplicit(colorImage, input.uv, input.depth, input.lod, input.ddx, input.ddy, input.offset)
                    + compareExplicit(volumeImage, compareSampler, input.uvw, input.depth, input.lod, input.ddx, input.ddy, input.offset)
                    + textureCompare(colorImage, input.uv, input.depth)
                    + textureCompare(volumeImage, compareSampler, input.uvw, input.depth);
                return vec4(cmp)
                    + gatherExplicit(layerImage, compareSampler, input.uvLayer, input.depth, input.offset)
                    + textureGatherCompare(layerImage, compareSampler, input.uvLayer, input.depth)
                    + textureGatherCompareOffset(layerImage, compareSampler, input.uvLayer, input.depth, input.offset);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    comparison_diagnostic = "unsupported DirectX storage image texture comparison"
    gather_diagnostic = "unsupported DirectX storage image texture operation"
    assert "RWTexture2D<float4> colorImage : register(u0);" in generated_code
    assert "RWTexture3D<float4> volumeImage : register(u1);" in generated_code
    assert "RWTexture2DArray<float4> layerImage : register(u2);" in generated_code
    assert "SamplerState compareSampler : register(s0);" in generated_code
    assert (
        "float compareExplicit(RWTexture3D<float4> image, SamplerState s"
        in generated_code
    )
    assert (
        "float4 gatherExplicit(RWTexture2DArray<float4> image, SamplerState s"
        in generated_code
    )
    assert generated_code.count(comparison_diagnostic) == 14
    assert generated_code.count(gather_diagnostic) == 4
    assert "SamplerComparisonState" not in generated_code
    assert "SampleCmp" not in generated_code
    assert "textureCompare(" not in generated_code
    assert "textureGatherCompare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code


def test_directx_multisample_texture_query_levels_use_single_level_helpers():
    shader = """
    shader MultisampleLevelQueries {
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler2DMS msTextures[4];
        sampler2DMSArray msArrays[4];

        struct FSInput {
            int layer @ TEXCOORD0;
        };

        int levels2D(sampler2DMS tex) {
            return textureQueryLevels(tex);
        }

        int levelsArray(sampler2DMSArray tex) {
            return textureQueryLevels(tex);
        }

        int levelsResourceArrays(sampler2DMS textures[], sampler2DMSArray arrays[], int layer) {
            return textureQueryLevels(textures[2]) + textureQueryLevels(arrays[layer]);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                int c = textureQueryLevels(msTex);
                int d = textureQueryLevels(msArray);
                int e = textureQueryLevels(msTextures[2]);
                int f = textureQueryLevels(msArrays[input.layer]);
                return float(c + d + e + f + levels2D(msTex) + levelsArray(msArray)
                    + levelsResourceArrays(msTextures, msArrays, input.layer));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2DMS<float4> msTex : register(t0);" in generated_code
    assert "Texture2DMSArray<float4> msArray : register(t1);" in generated_code
    assert "Texture2DMS<float4> msTextures[4] : register(t2);" in generated_code
    assert "Texture2DMSArray<float4> msArrays[4] : register(t6);" in generated_code
    assert "int textureQueryLevels(Texture2DMS<float4> tex)" in generated_code
    assert "int textureQueryLevels(Texture2DMSArray<float4> tex)" in generated_code
    assert generated_code.count("return 1;") == 2
    assert (
        "int levelsResourceArrays(Texture2DMS<float4> textures[4], Texture2DMSArray<float4> arrays[4], int layer)"
        in generated_code
    )
    assert (
        "return (textureQueryLevels(textures[2]) + textureQueryLevels(arrays[layer]));"
        in generated_code
    )
    assert "int c = textureQueryLevels(msTex);" in generated_code
    assert "int d = textureQueryLevels(msArray);" in generated_code
    assert "int e = textureQueryLevels(msTextures[2]);" in generated_code
    assert "int f = textureQueryLevels(msArrays[input.layer]);" in generated_code
    assert "GetDimensions(0" not in generated_code
    assert "SamplerState" not in generated_code


def test_directx_multisample_sampling_operations_emit_diagnostics_without_samplers():
    shader = """
    shader UnsupportedMultisampleSampling {
        sampler2DMS msTex;
        sampler2DMSArray msArray;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        vec4 invalid2D(sampler2DMS tex, vec2 uv, vec2 ddx, vec2 ddy, ivec2 offset) {
            vec4 sampled = texture(tex, uv);
            vec4 lod = textureLod(tex, uv, 0.0);
            vec4 grad = textureGrad(tex, uv, ddx, ddy);
            vec4 gathered = textureGather(tex, uv);
            vec4 offsetGathered = textureGatherOffset(tex, uv, offset);
            return sampled + lod + grad + gathered + offsetGathered;
        }

        vec4 invalidArray(sampler2DMSArray tex, vec3 uvLayer, vec2 ddx, vec2 ddy, ivec2 offset) {
            vec4 sampled = texture(tex, uvLayer);
            vec4 lod = textureLod(tex, uvLayer, 0.0);
            vec4 grad = textureGrad(tex, uvLayer, ddx, ddy);
            vec4 gathered = textureGather(tex, uvLayer);
            vec4 offsetGathered = textureGatherOffset(tex, uvLayer, offset);
            return sampled + lod + grad + gathered + offsetGathered;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return invalid2D(msTex, input.uv, input.ddx, input.ddy, input.offset)
                    + invalidArray(msArray, input.uvLayer, input.ddx, input.ddy, input.offset);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2DMS<float4> msTex : register(t0);" in generated_code
    assert "Texture2DMSArray<float4> msArray : register(t1);" in generated_code
    assert (
        "float4 invalid2D(Texture2DMS<float4> tex, float2 uv, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float4 invalidArray(Texture2DMSArray<float4> tex, float3 uvLayer, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert "SamplerState" not in generated_code

    for func_name in {
        "texture",
        "textureLod",
        "textureGrad",
        "textureGather",
        "textureGatherOffset",
    }:
        assert (
            f"unsupported DirectX multisample texture call: {func_name} on Texture2DMS<float4>"
            in generated_code
        )
        assert (
            f"unsupported DirectX multisample texture call: {func_name} on Texture2DMSArray<float4>"
            in generated_code
        )

    for invalid_call in {".Sample(", ".SampleLevel(", ".SampleGrad(", ".Gather("}:
        assert invalid_call not in generated_code


def test_directx_multisample_texture_query_lod_emits_diagnostics_without_samplers():
    shader = """
    shader UnsupportedMultisampleQueryLod {
        sampler2DMS msTex;
        sampler2DMSArray msArray;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
        };

        vec2 query2D(sampler2DMS tex, vec2 uv) {
            return textureQueryLod(tex, uv);
        }

        vec2 queryArray(sampler2DMSArray tex, vec3 uvLayer) {
            return textureQueryLod(tex, uvLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = textureQueryLod(msTex, input.uv);
                vec2 b = textureQueryLod(msArray, input.uvLayer);
                vec2 c = query2D(msTex, input.uv);
                vec2 d = queryArray(msArray, input.uvLayer);
                return vec4(a + b + c + d, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2DMS<float4> msTex : register(t0);" in generated_code
    assert "Texture2DMSArray<float4> msArray : register(t1);" in generated_code
    assert "float2 query2D(Texture2DMS<float4> tex, float2 uv)" in generated_code
    assert (
        "float2 queryArray(Texture2DMSArray<float4> tex, float3 uvLayer)"
        in generated_code
    )
    assert "SamplerState" not in generated_code
    assert "CalculateLevelOfDetail" not in generated_code
    assert "textureQueryLod(" not in generated_code
    assert (
        generated_code.count(
            "unsupported DirectX multisample texture query: textureQueryLod on Texture2DMS<float4>"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported DirectX multisample texture query: textureQueryLod on Texture2DMSArray<float4>"
        )
        == 2
    )


def test_directx_multisample_compare_operations_emit_diagnostics():
    shader = """
    shader UnsupportedMultisampleCompare {
        sampler2DMS msTex;
        sampler2DMSArray msArray;
        sampler sharedSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            float depth;
            float lod;
            vec2 ddx @ TEXCOORD2;
            vec2 ddy @ TEXCOORD3;
            ivec2 offset @ TEXCOORD4;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float cmp = textureCompare(msTex, sharedSampler, input.uv, input.depth)
                    + textureCompareOffset(msTex, sharedSampler, input.uv, input.depth, input.offset)
                    + textureCompareLod(msTex, sharedSampler, input.uv, input.depth, input.lod)
                    + textureCompareLodOffset(msTex, sharedSampler, input.uv, input.depth, input.lod, input.offset)
                    + textureCompareGrad(msTex, sharedSampler, input.uv, input.depth, input.ddx, input.ddy)
                    + textureCompareGradOffset(msTex, sharedSampler, input.uv, input.depth, input.ddx, input.ddy, input.offset);
                vec4 gathered = textureGatherCompare(msArray, sharedSampler, input.uvLayer, input.depth)
                    + textureGatherCompareOffset(msArray, sharedSampler, input.uvLayer, input.depth, input.offset);
                return gathered + vec4(cmp);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2DMS<float4> msTex : register(t0);" in generated_code
    assert "Texture2DMSArray<float4> msArray : register(t1);" in generated_code
    assert "SamplerState sharedSampler : register(s0);" in generated_code
    assert "SamplerComparisonState sharedSampler" not in generated_code
    for func_name in {
        "textureCompare",
        "textureCompareOffset",
        "textureCompareLod",
        "textureCompareLodOffset",
        "textureCompareGrad",
        "textureCompareGradOffset",
    }:
        assert (
            f"unsupported DirectX multisample texture comparison: {func_name} on Texture2DMS<float4>"
            in generated_code
        )
    for func_name in {"textureGatherCompare", "textureGatherCompareOffset"}:
        assert (
            f"unsupported DirectX multisample texture gather comparison: {func_name} on Texture2DMSArray<float4>"
            in generated_code
        )
    assert "SampleCmp" not in generated_code
    assert "GatherCmp" not in generated_code
    assert "textureCompare(" not in generated_code
    assert "textureGatherCompare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code


def test_directx_multisample_diagnostics_do_not_create_sampler_role_conflicts():
    regular_diagnostic_shader = """
    shader MultisampleSamplerRoleConflict {
        sampler2DMS msTex;
        sampler2DShadow shadowMap;
        sampler sharedSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
            vec2 ddx @ TEXCOORD1;
            vec2 ddy @ TEXCOORD2;
        };

        float inspect(sampler s, vec2 uv, float depth, vec2 ddx, vec2 ddy) {
            vec4 bad = textureGrad(msTex, s, uv, ddx, ddy);
            float ok = textureCompare(shadowMap, s, uv, depth);
            return ok + bad.x;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return inspect(sharedSampler, input.uv, input.depth, input.ddx, input.ddy);
            }
        }
    }
    """
    compare_diagnostic_shader = """
    shader MultisampleCompareSamplerRoleConflict {
        sampler2DMS msTex;
        sampler2D colorMap;
        sampler sharedSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                float bad = textureCompare(msTex, sharedSampler, input.uv, input.depth);
                vec4 ok = texture(colorMap, sharedSampler, input.uv);
                return ok + vec4(bad);
            }
        }
    }
    """
    struct_diagnostic_shader = """
    shader MultisampleStructSamplerRoleConflict {
        sampler2DMS msTex;
        sampler2DShadow shadowMap;

        struct SamplerPack {
            sampler samplers[2];
        };

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
            int index;
        };

        float inspect(SamplerPack pack, FSInput input) {
            vec4 bad = texture(msTex, pack.samplers[input.index], input.uv);
            float ok = textureCompare(
                shadowMap,
                pack.samplers[input.index],
                input.uv,
                input.depth
            );
            return ok + bad.x;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                SamplerPack pack;
                return inspect(pack, input);
            }
        }
    }
    """

    regular_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(regular_diagnostic_shader), "fragment"
    )
    compare_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(compare_diagnostic_shader), "fragment"
    )
    struct_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(struct_diagnostic_shader), "fragment"
    )

    regular_diagnostic = (
        "/* unsupported DirectX multisample texture call: "
        "textureGrad on Texture2DMS<float4> */ float4(0.0, 0.0, 0.0, 0.0)"
    )
    compare_diagnostic = (
        "/* unsupported DirectX multisample texture comparison: "
        "textureCompare on Texture2DMS<float4> */ 0.0"
    )
    sample_diagnostic = (
        "/* unsupported DirectX multisample texture call: "
        "texture on Texture2DMS<float4> */ float4(0.0, 0.0, 0.0, 0.0)"
    )

    assert "SamplerComparisonState sharedSampler : register(s0);" in regular_code
    assert (
        "float inspect(SamplerComparisonState s, float2 uv, float depth, float2 ddx, float2 ddy)"
        in regular_code
    )
    assert regular_diagnostic in regular_code
    assert "float ok = shadowMap.SampleCmp(s, uv, depth);" in regular_code
    assert (
        "return inspect(sharedSampler, input.uv, input.depth, input.ddx, input.ddy);"
        in regular_code
    )

    assert "SamplerState sharedSampler : register(s0);" in compare_code
    assert "SamplerComparisonState sharedSampler" not in compare_code
    assert compare_diagnostic in compare_code
    assert "float4 ok = colorMap.Sample(sharedSampler, input.uv);" in compare_code
    assert "SampleCmp(sharedSampler" not in compare_code

    assert "SamplerComparisonState samplers[2];" in struct_code
    assert "SamplerState samplers[2];" not in struct_code
    assert sample_diagnostic in struct_code
    assert (
        "float ok = shadowMap.SampleCmp(pack.samplers[input.index], input.uv, input.depth);"
        in struct_code
    )
    assert ".Sample(pack.samplers" not in struct_code


def test_directx_multisample_texel_fetch_offsets_emit_diagnostics():
    shader = """
    shader UnsupportedMultisampleTexelFetchOffset {
        sampler2DMS msTex;
        sampler2DMSArray msArray;

        struct FSInput {
            ivec2 pixel @ TEXCOORD0;
            ivec3 pixelLayer @ TEXCOORD1;
            ivec2 offset @ TEXCOORD2;
            int sampleIndex;
        };

        vec4 offset2D(sampler2DMS tex, ivec2 pixel, int sampleIndex, ivec2 offset) {
            return texelFetchOffset(tex, pixel, sampleIndex, offset);
        }

        vec4 offsetArray(sampler2DMSArray tex, ivec3 pixelLayer, int sampleIndex, ivec2 offset) {
            return texelFetchOffset(tex, pixelLayer, sampleIndex, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return offset2D(msTex, input.pixel, input.sampleIndex, input.offset)
                    + offsetArray(msArray, input.pixelLayer, input.sampleIndex, input.offset)
                    + texelFetchOffset(msTex, input.pixel, input.sampleIndex, input.offset)
                    + texelFetchOffset(msArray, input.pixelLayer, input.sampleIndex, input.offset);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    diagnostic = (
        "/* unsupported DirectX texel fetch offset: "
        "multisample textures do not support offsets */ float4(0.0, 0.0, 0.0, 0.0)"
    )
    assert "Texture2DMS<float4> msTex : register(t0);" in generated_code
    assert "Texture2DMSArray<float4> msArray : register(t1);" in generated_code
    assert (
        "float4 offset2D(Texture2DMS<float4> tex, int2 pixel, int sampleIndex, int2 offset)"
        in generated_code
    )
    assert (
        "float4 offsetArray(Texture2DMSArray<float4> tex, int3 pixelLayer, int sampleIndex, int2 offset)"
        in generated_code
    )
    assert generated_code.count(diagnostic) == 4
    assert "texelFetchOffset(" not in generated_code
    assert ".Load(" not in generated_code
    assert "SamplerState" not in generated_code


def test_directx_cube_texel_fetches_emit_diagnostics():
    shader = """
    shader UnsupportedCubeTexelFetch {
        samplerCube cubeMap;
        samplerCubeArray cubeArray;

        struct FSInput {
            ivec3 cubeCoord @ TEXCOORD0;
            ivec4 cubeLayerCoord @ TEXCOORD1;
            ivec3 offset @ TEXCOORD2;
            int lod;
        };

        vec4 fetchCube(samplerCube tex, ivec3 cubeCoord, int lod, ivec3 offset) {
            vec4 plain = texelFetch(tex, cubeCoord, lod);
            vec4 offsetFetch = texelFetchOffset(tex, cubeCoord, lod, offset);
            return plain + offsetFetch;
        }

        vec4 fetchCubeArray(samplerCubeArray tex, ivec4 cubeLayerCoord, int lod, ivec3 offset) {
            vec4 plain = texelFetch(tex, cubeLayerCoord, lod);
            vec4 offsetFetch = texelFetchOffset(tex, cubeLayerCoord, lod, offset);
            return plain + offsetFetch;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return fetchCube(cubeMap, input.cubeCoord, input.lod, input.offset)
                    + fetchCubeArray(cubeArray, input.cubeLayerCoord, input.lod, input.offset)
                    + texelFetch(cubeMap, input.cubeCoord, input.lod)
                    + texelFetch(cubeArray, input.cubeLayerCoord, input.lod)
                    + texelFetchOffset(cubeMap, input.cubeCoord, input.lod, input.offset)
                    + texelFetchOffset(cubeArray, input.cubeLayerCoord, input.lod, input.offset);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "TextureCube cubeMap : register(t0);" in generated_code
    assert "TextureCubeArray cubeArray : register(t1);" in generated_code
    assert (
        "float4 fetchCube(TextureCube tex, int3 cubeCoord, int lod, int3 offset)"
        in generated_code
    )
    assert (
        "float4 fetchCubeArray(TextureCubeArray tex, int4 cubeLayerCoord, int lod, int3 offset)"
        in generated_code
    )
    assert (
        generated_code.count(
            "unsupported DirectX texel fetch: texelFetch on TextureCube */ float4(0.0, 0.0, 0.0, 0.0)"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported DirectX texel fetch: texelFetchOffset on TextureCube */ float4(0.0, 0.0, 0.0, 0.0)"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported DirectX texel fetch: texelFetch on TextureCubeArray"
        )
        == 2
    )
    assert (
        generated_code.count(
            "unsupported DirectX texel fetch: texelFetchOffset on TextureCubeArray"
        )
        == 2
    )
    assert ".Load(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "texelFetchOffset(" not in generated_code
    assert "SamplerState" not in generated_code


def test_directx_array_shadow_texture_query_functions_prune_implicit_samplers():
    shader = """
    shader ArrayShadowTextureQueries {
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeShadowArrays[4];

        ivec3 query2DArrayShadow(sampler2DArrayShadow tex) {
            ivec3 size = textureSize(tex, 1);
            int levels = textureQueryLevels(tex);
            return size + ivec3(levels);
        }

        ivec3 queryCubeArrayShadow(samplerCubeArrayShadow tex) {
            ivec3 size = textureSize(tex, 0);
            int levels = textureQueryLevels(tex);
            return size + ivec3(levels);
        }

        ivec3 queryArrayElements(sampler2DArrayShadow shadowArrays[], samplerCubeArrayShadow cubeShadowArrays[]) {
            ivec3 arraySize = textureSize(shadowArrays[2], 1);
            ivec3 cubeSize = textureSize(cubeShadowArrays[3], 0);
            int arrayLevels = textureQueryLevels(shadowArrays[2]);
            int cubeLevels = textureQueryLevels(cubeShadowArrays[3]);
            return arraySize + cubeSize + ivec3(arrayLevels + cubeLevels);
        }

        fragment {
            vec4 main() @ gl_FragColor {
                ivec3 a = query2DArrayShadow(shadowArray);
                ivec3 b = queryCubeArrayShadow(cubeShadowArray);
                ivec3 c = queryArrayElements(shadowArrays, cubeShadowArrays);
                return vec4(float(a.x + b.y + c.z));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2DArray shadowArray : register(t0);" in generated_code
    assert "TextureCubeArray cubeShadowArray : register(t1);" in generated_code
    assert "Texture2DArray shadowArrays[4] : register(t2);" in generated_code
    assert "TextureCubeArray cubeShadowArrays[4] : register(t6);" in generated_code
    assert "SamplerComparisonState shadowArraySampler" not in generated_code
    assert "SamplerComparisonState cubeShadowArraySampler" not in generated_code
    assert "SamplerComparisonState shadowArraysSampler" not in generated_code
    assert "SamplerComparisonState cubeShadowArraysSampler" not in generated_code
    assert "int3 textureSize(Texture2DArray tex, int lod)" in generated_code
    assert "int3 textureSize(TextureCubeArray tex, int lod)" in generated_code
    assert "int textureQueryLevels(Texture2DArray tex)" in generated_code
    assert "int textureQueryLevels(TextureCubeArray tex)" in generated_code
    assert "tex.GetDimensions(lod, width, height, elements, levels);" in generated_code
    assert "tex.GetDimensions(0, width, height, elements, levels);" in generated_code
    assert "int3 query2DArrayShadow(Texture2DArray tex)" in generated_code
    assert "int3 queryCubeArrayShadow(TextureCubeArray tex)" in generated_code
    assert (
        "int3 queryArrayElements(Texture2DArray shadowArrays[4], TextureCubeArray cubeShadowArrays[4])"
        in generated_code
    )
    assert "int3 arraySize = textureSize(shadowArrays[2], 1);" in generated_code
    assert "int3 cubeSize = textureSize(cubeShadowArrays[3], 0);" in generated_code
    assert "int arrayLevels = textureQueryLevels(shadowArrays[2]);" in generated_code
    assert "int cubeLevels = textureQueryLevels(cubeShadowArrays[3]);" in generated_code
    assert "SampleCmp" not in generated_code


def test_directx_direct_stage_texture_queries_mix_size_levels_and_lod():
    shader = """
    shader DirectTextureQueries {
        sampler2D colorMap;
        sampler2DArray layerMap;
        samplerCubeArray cubeArray;
        sampler2DShadow shadowMap;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler linearSampler;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            vec3 uvLayer @ TEXCOORD1;
            vec4 cubeLayer @ TEXCOORD2;
            float lod;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                ivec2 colorSize = textureSize(colorMap, 1);
                ivec3 layerSize = textureSize(layerMap, 2);
                ivec3 cubeSize = textureSize(cubeArray, 0);
                ivec2 shadowSize = textureSize(shadowMap, 0);
                ivec3 shadowArraySize = textureSize(shadowArray, 1);
                ivec3 cubeShadowSize = textureSize(cubeShadowArray, 0);
                int colorLevels = textureQueryLevels(colorMap);
                int shadowLevels = textureQueryLevels(shadowArray);
                vec2 colorLod = textureQueryLod(colorMap, linearSampler, input.uv);
                vec2 layerLod = textureQueryLod(layerMap, linearSampler, input.uvLayer);
                vec2 cubeLod = textureQueryLod(cubeArray, linearSampler, input.cubeLayer);
                vec2 implicitLayerLod = textureQueryLod(layerMap, input.uvLayer);
                vec2 implicitCubeShadowLod = textureQueryLod(cubeShadowArray, input.cubeLayer);
                float total = float(colorSize.x + layerSize.z + cubeSize.z + shadowSize.x + shadowArraySize.z + cubeShadowSize.z + colorLevels + shadowLevels);
                return vec4(total + colorLod.x + layerLod.y + cubeLod.x + implicitLayerLod.x + implicitCubeShadowLod.y);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "Texture2DArray layerMap : register(t1);" in generated_code
    assert "SamplerState layerMapSampler : register(s0);" in generated_code
    assert "TextureCubeArray cubeArray : register(t2);" in generated_code
    assert "Texture2D shadowMap : register(t3);" in generated_code
    assert "Texture2DArray shadowArray : register(t4);" in generated_code
    assert "TextureCubeArray cubeShadowArray : register(t5);" in generated_code
    assert "SamplerState cubeShadowArraySampler : register(s1);" in generated_code
    assert "SamplerState linearSampler : register(s2);" in generated_code
    assert "SamplerComparisonState" not in generated_code
    assert "int2 textureSize(Texture2D tex, int lod)" in generated_code
    assert "int3 textureSize(Texture2DArray tex, int lod)" in generated_code
    assert "int3 textureSize(TextureCubeArray tex, int lod)" in generated_code
    assert "int textureQueryLevels(Texture2D tex)" in generated_code
    assert "int textureQueryLevels(Texture2DArray tex)" in generated_code
    assert "int2 colorSize = textureSize(colorMap, 1);" in generated_code
    assert "int3 layerSize = textureSize(layerMap, 2);" in generated_code
    assert "int3 cubeSize = textureSize(cubeArray, 0);" in generated_code
    assert "int2 shadowSize = textureSize(shadowMap, 0);" in generated_code
    assert "int3 shadowArraySize = textureSize(shadowArray, 1);" in generated_code
    assert "int3 cubeShadowSize = textureSize(cubeShadowArray, 0);" in generated_code
    assert "int colorLevels = textureQueryLevels(colorMap);" in generated_code
    assert "int shadowLevels = textureQueryLevels(shadowArray);" in generated_code
    assert (
        "float2 colorLod = float2(colorMap.CalculateLevelOfDetailUnclamped(linearSampler, input.uv), colorMap.CalculateLevelOfDetail(linearSampler, input.uv));"
        in generated_code
    )
    assert (
        "float2 layerLod = float2(layerMap.CalculateLevelOfDetailUnclamped(linearSampler, input.uvLayer.xy), layerMap.CalculateLevelOfDetail(linearSampler, input.uvLayer.xy));"
        in generated_code
    )
    assert (
        "float2 cubeLod = float2(cubeArray.CalculateLevelOfDetailUnclamped(linearSampler, input.cubeLayer.xyz), cubeArray.CalculateLevelOfDetail(linearSampler, input.cubeLayer.xyz));"
        in generated_code
    )
    assert (
        "float2 implicitLayerLod = float2(layerMap.CalculateLevelOfDetailUnclamped(layerMapSampler, input.uvLayer.xy), layerMap.CalculateLevelOfDetail(layerMapSampler, input.uvLayer.xy));"
        in generated_code
    )
    assert (
        "float2 implicitCubeShadowLod = float2(cubeShadowArray.CalculateLevelOfDetailUnclamped(cubeShadowArraySampler, input.cubeLayer.xyz), cubeShadowArray.CalculateLevelOfDetail(cubeShadowArraySampler, input.cubeLayer.xyz));"
        in generated_code
    )
    assert "textureQueryLod(" not in generated_code


def test_directx_mixed_explicit_and_implicit_texture_sampling_keeps_synthetic_sampler():
    shader = """
    shader MixedExplicitImplicitSampling {
        sampler2D colorMap;
        sampler linearSampler;

        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return texture(colorMap, linearSampler, uv) + texture(colorMap, uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "SamplerState colorMapSampler : register(s0);" in generated_code
    assert "SamplerState linearSampler : register(s1);" in generated_code
    assert "colorMap.Sample(linearSampler, uv)" in generated_code
    assert "colorMap.Sample(colorMapSampler, uv)" in generated_code


def test_directx_rejects_global_sampler_used_for_regular_and_shadow_compare():
    shader = """
    shader MixedSamplerUse {
        sampler2D colorMap;
        sampler2DShadow shadowMap;
        sampler sharedSampler;

        struct FSInput {
            vec2 uv;
            float depth;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 color = texture(colorMap, sharedSampler, input.uv);
                float shadow = textureCompare(
                    shadowMap,
                    sharedSampler,
                    input.uv,
                    input.depth
                );
                return color * shadow;
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "DirectX sampler\\(s\\) used for both regular sampling and "
            "shadow comparison: sharedSampler"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_rejects_sampler_parameter_used_for_regular_and_shadow_compare():
    shader = """
    shader MixedSamplerParameter {
        sampler2D colorMap;
        sampler2DShadow shadowMap;

        struct FSInput {
            vec2 uv;
            float depth;
        };

        vec4 shade(sampler sampleState, FSInput input) {
            vec4 color = texture(colorMap, sampleState, input.uv);
            float shadow = textureCompare(
                shadowMap,
                sampleState,
                input.uv,
                input.depth
            );
            return color * shadow;
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "DirectX sampler\\(s\\) used for both regular sampling and "
            "shadow comparison: shade.sampleState"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_rejects_struct_sampler_member_used_for_regular_and_shadow_compare():
    shader = """
    shader MixedStructSamplerUse {
        sampler2D colorMap;
        sampler2DShadow shadowMap;

        struct SamplerPack {
            sampler samplers[2];
        };

        struct FSInput {
            vec2 uv;
            float depth;
            int index;
        };

        vec4 shade(SamplerPack pack, FSInput input) {
            vec4 color = texture(colorMap, pack.samplers[input.index], input.uv);
            float shadow = textureCompare(
                shadowMap,
                pack.samplers[input.index],
                input.uv,
                input.depth
            );
            return color * shadow;
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "DirectX sampler\\(s\\) used for both regular sampling and "
            "shadow comparison: SamplerPack.samplers"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_array_texture_query_lod_uses_non_layer_coordinates():
    shader = """
    shader ArrayTextureQueryLod {
        sampler2DArray layerMap;
        samplerCubeArray cubeArray;
        sampler2DArray layerMaps[4];
        samplerCubeArray cubeArrays[4];
        sampler linearSampler;
        sampler linearSamplers[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
        };

        vec2 queryArrayLod(sampler2DArray tex, sampler s, vec3 uvLayer) {
            return textureQueryLod(tex, s, uvLayer);
        }

        vec2 queryCubeArrayLod(samplerCubeArray tex, sampler s, vec4 cubeLayer) {
            return textureQueryLod(tex, s, cubeLayer);
        }

        vec2 queryArrayElementLod(sampler2DArray layerMaps[], sampler linearSamplers[], vec3 uvLayer) {
            return textureQueryLod(layerMaps[2], linearSamplers[2], uvLayer);
        }

        vec2 queryCubeArrayElementLod(samplerCubeArray cubeArrays[], sampler linearSamplers[], vec4 cubeLayer) {
            return textureQueryLod(cubeArrays[3], linearSamplers[3], cubeLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = queryArrayLod(layerMap, linearSampler, input.uvLayer);
                vec2 b = queryCubeArrayLod(cubeArray, linearSampler, input.cubeLayer);
                vec2 c = queryArrayElementLod(layerMaps, linearSamplers, input.uvLayer);
                vec2 d = queryCubeArrayElementLod(cubeArrays, linearSamplers, input.cubeLayer);
                return vec4(a.x + b.y, c.x + d.y, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2DArray layerMap : register(t0);" in generated_code
    assert "TextureCubeArray cubeArray : register(t1);" in generated_code
    assert "Texture2DArray layerMaps[4] : register(t2);" in generated_code
    assert "TextureCubeArray cubeArrays[4] : register(t6);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "SamplerState linearSamplers[4] : register(s1);" in generated_code
    assert (
        "float2 queryArrayLod(Texture2DArray tex, SamplerState s, float3 uvLayer)"
        in generated_code
    )
    assert "tex.CalculateLevelOfDetailUnclamped(s, uvLayer.xy)" in generated_code
    assert "tex.CalculateLevelOfDetail(s, uvLayer.xy)" in generated_code
    assert (
        "float2 queryCubeArrayLod(TextureCubeArray tex, SamplerState s, float4 cubeLayer)"
        in generated_code
    )
    assert "tex.CalculateLevelOfDetailUnclamped(s, cubeLayer.xyz)" in generated_code
    assert "tex.CalculateLevelOfDetail(s, cubeLayer.xyz)" in generated_code
    assert (
        "float2 queryArrayElementLod(Texture2DArray layerMaps[4], SamplerState linearSamplers[4], float3 uvLayer)"
        in generated_code
    )
    assert (
        "layerMaps[2].CalculateLevelOfDetailUnclamped(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "layerMaps[2].CalculateLevelOfDetail(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "float2 queryCubeArrayElementLod(TextureCubeArray cubeArrays[4], SamplerState linearSamplers[4], float4 cubeLayer)"
        in generated_code
    )
    assert (
        "cubeArrays[3].CalculateLevelOfDetailUnclamped(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert (
        "cubeArrays[3].CalculateLevelOfDetail(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert "CalculateLevelOfDetailUnclamped(s, uvLayer)" not in generated_code
    assert "CalculateLevelOfDetailUnclamped(s, cubeLayer)" not in generated_code


def test_directx_shadow_array_texture_query_lod_uses_non_layer_coordinates():
    shader = """
    shader ShadowArrayTextureQueryLod {
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeShadowArrays[4];
        sampler linearSampler;
        sampler linearSamplers[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
        };

        vec2 queryArrayLod(sampler2DArrayShadow tex, sampler s, vec3 uvLayer) {
            return textureQueryLod(tex, s, uvLayer);
        }

        vec2 queryCubeArrayLod(samplerCubeArrayShadow tex, sampler s, vec4 cubeLayer) {
            return textureQueryLod(tex, s, cubeLayer);
        }

        vec2 queryArrayElementLod(sampler2DArrayShadow shadowArrays[], sampler linearSamplers[], vec3 uvLayer) {
            return textureQueryLod(shadowArrays[2], linearSamplers[2], uvLayer);
        }

        vec2 queryCubeArrayElementLod(samplerCubeArrayShadow cubeShadowArrays[], sampler linearSamplers[], vec4 cubeLayer) {
            return textureQueryLod(cubeShadowArrays[3], linearSamplers[3], cubeLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = queryArrayLod(shadowArray, linearSampler, input.uvLayer);
                vec2 b = queryCubeArrayLod(cubeShadowArray, linearSampler, input.cubeLayer);
                vec2 c = queryArrayElementLod(shadowArrays, linearSamplers, input.uvLayer);
                vec2 d = queryCubeArrayElementLod(cubeShadowArrays, linearSamplers, input.cubeLayer);
                return vec4(a.x + b.y, c.x + d.y, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2DArray shadowArray : register(t0);" in generated_code
    assert "TextureCubeArray cubeShadowArray : register(t1);" in generated_code
    assert "Texture2DArray shadowArrays[4] : register(t2);" in generated_code
    assert "TextureCubeArray cubeShadowArrays[4] : register(t6);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "SamplerState linearSamplers[4] : register(s1);" in generated_code
    assert "SamplerComparisonState linearSampler" not in generated_code
    assert "SamplerComparisonState linearSamplers" not in generated_code
    assert (
        "float2 queryArrayLod(Texture2DArray tex, SamplerState s, float3 uvLayer)"
        in generated_code
    )
    assert "tex.CalculateLevelOfDetailUnclamped(s, uvLayer.xy)" in generated_code
    assert "tex.CalculateLevelOfDetail(s, uvLayer.xy)" in generated_code
    assert (
        "float2 queryCubeArrayLod(TextureCubeArray tex, SamplerState s, float4 cubeLayer)"
        in generated_code
    )
    assert "tex.CalculateLevelOfDetailUnclamped(s, cubeLayer.xyz)" in generated_code
    assert "tex.CalculateLevelOfDetail(s, cubeLayer.xyz)" in generated_code
    assert (
        "float2 queryArrayElementLod(Texture2DArray shadowArrays[4], SamplerState linearSamplers[4], float3 uvLayer)"
        in generated_code
    )
    assert (
        "shadowArrays[2].CalculateLevelOfDetailUnclamped(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "shadowArrays[2].CalculateLevelOfDetail(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "float2 queryCubeArrayElementLod(TextureCubeArray cubeShadowArrays[4], SamplerState linearSamplers[4], float4 cubeLayer)"
        in generated_code
    )
    assert (
        "cubeShadowArrays[3].CalculateLevelOfDetailUnclamped(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert (
        "cubeShadowArrays[3].CalculateLevelOfDetail(linearSamplers[3], cubeLayer.xyz)"
        in generated_code
    )
    assert "CalculateLevelOfDetailUnclamped(s, uvLayer)" not in generated_code
    assert "CalculateLevelOfDetailUnclamped(s, cubeLayer)" not in generated_code


def test_directx_implicit_array_texture_query_lod_synthesizes_regular_samplers():
    shader = """
    shader ImplicitArrayTextureQueryLodGlobals {
        sampler2DArray layerMap;
        samplerCubeArray cubeArray;
        sampler2DArrayShadow shadowArray;
        samplerCubeArrayShadow cubeShadowArray;

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = textureQueryLod(layerMap, input.uvLayer);
                vec2 b = textureQueryLod(cubeArray, input.cubeLayer);
                vec2 c = textureQueryLod(shadowArray, input.uvLayer);
                vec2 d = textureQueryLod(cubeShadowArray, input.cubeLayer);
                return vec4(a.x + b.y, c.x + d.y, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2DArray layerMap : register(t0);" in generated_code
    assert "SamplerState layerMapSampler : register(s0);" in generated_code
    assert "TextureCubeArray cubeArray : register(t1);" in generated_code
    assert "SamplerState cubeArraySampler : register(s1);" in generated_code
    assert "Texture2DArray shadowArray : register(t2);" in generated_code
    assert "SamplerState shadowArraySampler : register(s2);" in generated_code
    assert "TextureCubeArray cubeShadowArray : register(t3);" in generated_code
    assert "SamplerState cubeShadowArraySampler : register(s3);" in generated_code
    assert "SamplerComparisonState" not in generated_code
    assert (
        "layerMap.CalculateLevelOfDetailUnclamped(layerMapSampler, input.uvLayer.xy)"
        in generated_code
    )
    assert (
        "cubeArray.CalculateLevelOfDetailUnclamped(cubeArraySampler, input.cubeLayer.xyz)"
        in generated_code
    )
    assert (
        "shadowArray.CalculateLevelOfDetailUnclamped(shadowArraySampler, input.uvLayer.xy)"
        in generated_code
    )
    assert (
        "cubeShadowArray.CalculateLevelOfDetailUnclamped(cubeShadowArraySampler, input.cubeLayer.xyz)"
        in generated_code
    )


def test_directx_nested_array_texture_query_lod_threads_resource_arrays():
    shader = """
    shader NestedArrayTextureQueryLod {
        sampler2DArray layerMaps[4];
        samplerCubeArray cubeArrays[4];
        sampler linearSamplers[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
        };

        vec2 explicitLeaf(sampler2DArray layerMaps[], sampler linearSamplers[], vec3 uvLayer) {
            return textureQueryLod(layerMaps[2], linearSamplers[2], uvLayer);
        }

        vec2 explicitMid(sampler2DArray layerMaps[], sampler linearSamplers[], vec3 uvLayer) {
            return explicitLeaf(layerMaps, linearSamplers, uvLayer);
        }

        vec2 implicitLeaf(samplerCubeArray cubeArrays[], vec4 cubeLayer) {
            return textureQueryLod(cubeArrays[3], cubeLayer);
        }

        vec2 implicitMid(samplerCubeArray cubeArrays[], vec4 cubeLayer) {
            return implicitLeaf(cubeArrays, cubeLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 a = explicitMid(layerMaps, linearSamplers, input.uvLayer);
                vec2 b = implicitMid(cubeArrays, input.cubeLayer);
                return vec4(a.x, a.y, b.x, b.y);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2DArray layerMaps[4] : register(t0);" in generated_code
    assert "TextureCubeArray cubeArrays[4] : register(t4);" in generated_code
    assert "SamplerState cubeArraysSampler : register(s0);" in generated_code
    assert "SamplerState linearSamplers[4] : register(s1);" in generated_code
    assert (
        "float2 explicitLeaf(Texture2DArray layerMaps[4], SamplerState linearSamplers[4], float3 uvLayer)"
        in generated_code
    )
    assert (
        "float2 explicitMid(Texture2DArray layerMaps[4], SamplerState linearSamplers[4], float3 uvLayer)"
        in generated_code
    )
    assert "return explicitLeaf(layerMaps, linearSamplers, uvLayer);" in generated_code
    assert (
        "layerMaps[2].CalculateLevelOfDetailUnclamped(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "layerMaps[2].CalculateLevelOfDetail(linearSamplers[2], uvLayer.xy)"
        in generated_code
    )
    assert (
        "float2 implicitLeaf(TextureCubeArray cubeArrays[4], SamplerState cubeArraysSampler, float4 cubeLayer)"
        in generated_code
    )
    assert (
        "float2 implicitMid(TextureCubeArray cubeArrays[4], SamplerState cubeArraysSampler, float4 cubeLayer)"
        in generated_code
    )
    assert (
        "return implicitLeaf(cubeArrays, cubeArraysSampler, cubeLayer);"
        in generated_code
    )
    assert (
        "cubeArrays[3].CalculateLevelOfDetailUnclamped(cubeArraysSampler, cubeLayer.xyz)"
        in generated_code
    )
    assert (
        "cubeArrays[3].CalculateLevelOfDetail(cubeArraysSampler, cubeLayer.xyz)"
        in generated_code
    )
    assert (
        "float2 a = explicitMid(layerMaps, linearSamplers, input.uvLayer);"
        in generated_code
    )
    assert (
        "float2 b = implicitMid(cubeArrays, cubeArraysSampler, input.cubeLayer);"
        in generated_code
    )
    assert (
        "CalculateLevelOfDetailUnclamped(linearSamplers[2], uvLayer)"
        not in generated_code
    )
    assert (
        "CalculateLevelOfDetailUnclamped(cubeArraysSampler, cubeLayer)"
        not in generated_code
    )
    assert "textureQueryLod(" not in generated_code


def test_directx_nested_shadow_array_query_lod_and_compare_split_samplers():
    shader = """
    shader NestedShadowArrayQueryCompare {
        sampler2DArrayShadow shadowArrays[4];
        samplerCubeArrayShadow cubeShadowArrays[4];

        struct FSInput {
            vec3 uvLayer @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            float lod;
            vec2 ddxLayer @ TEXCOORD2;
            vec2 ddyLayer @ TEXCOORD3;
            vec3 ddxCube @ TEXCOORD4;
            vec3 ddyCube @ TEXCOORD5;
        };

        float layerLeaf(sampler2DArrayShadow shadowArrays[], vec3 uvLayer, float depth, float lod, vec2 ddx, vec2 ddy) {
            vec2 query = textureQueryLod(shadowArrays[2], uvLayer);
            float cmp = textureCompare(shadowArrays[2], uvLayer, depth);
            float cmpLod = textureCompareLod(shadowArrays[1], uvLayer, depth, lod);
            float cmpGrad = textureCompareGrad(shadowArrays[3], uvLayer, depth, ddx, ddy);
            return query.x + cmp + cmpLod + cmpGrad;
        }

        float layerMid(sampler2DArrayShadow shadowArrays[], vec3 uvLayer, float depth, float lod, vec2 ddx, vec2 ddy) {
            return layerLeaf(shadowArrays, uvLayer, depth, lod, ddx, ddy);
        }

        float cubeLeaf(samplerCubeArrayShadow cubeShadowArrays[], vec4 cubeLayer, float depth, float lod, vec3 ddx, vec3 ddy) {
            vec2 query = textureQueryLod(cubeShadowArrays[2], cubeLayer);
            float cmp = textureCompare(cubeShadowArrays[2], cubeLayer, depth);
            float cmpLod = textureCompareLod(cubeShadowArrays[1], cubeLayer, depth, lod);
            float cmpGrad = textureCompareGrad(cubeShadowArrays[3], cubeLayer, depth, ddx, ddy);
            return query.y + cmp + cmpLod + cmpGrad;
        }

        float cubeMid(samplerCubeArrayShadow cubeShadowArrays[], vec4 cubeLayer, float depth, float lod, vec3 ddx, vec3 ddy) {
            return cubeLeaf(cubeShadowArrays, cubeLayer, depth, lod, ddx, ddy);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return layerMid(shadowArrays, input.uvLayer, input.depth, input.lod, input.ddxLayer, input.ddyLayer)
                    + cubeMid(cubeShadowArrays, input.cubeLayer, input.depth, input.lod, input.ddxCube, input.ddyCube);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2DArray shadowArrays[4] : register(t0);" in generated_code
    assert (
        "SamplerComparisonState shadowArraysSampler : register(s0);" in generated_code
    )
    assert "SamplerState shadowArraysQuerySampler : register(s1);" in generated_code
    assert "TextureCubeArray cubeShadowArrays[4] : register(t4);" in generated_code
    assert (
        "SamplerComparisonState cubeShadowArraysSampler : register(s2);"
        in generated_code
    )
    assert "SamplerState cubeShadowArraysQuerySampler : register(s3);" in generated_code
    assert (
        "float layerLeaf(Texture2DArray shadowArrays[4], SamplerComparisonState shadowArraysSampler, SamplerState shadowArraysQuerySampler, float3 uvLayer, float depth, float lod, float2 ddx, float2 ddy)"
        in generated_code
    )
    assert (
        "float2 query = float2(shadowArrays[2].CalculateLevelOfDetailUnclamped(shadowArraysQuerySampler, uvLayer.xy), shadowArrays[2].CalculateLevelOfDetail(shadowArraysQuerySampler, uvLayer.xy));"
        in generated_code
    )
    assert (
        "float cmp = shadowArrays[2].SampleCmp(shadowArraysSampler, uvLayer, depth);"
        in generated_code
    )
    assert (
        "float cmpLod = shadowArrays[1].SampleCmpLevel(shadowArraysSampler, uvLayer, depth, lod);"
        in generated_code
    )
    assert (
        "float cmpGrad = shadowArrays[3].SampleCmpGrad(shadowArraysSampler, uvLayer, depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "float layerMid(Texture2DArray shadowArrays[4], SamplerComparisonState shadowArraysSampler, SamplerState shadowArraysQuerySampler, float3 uvLayer, float depth, float lod, float2 ddx, float2 ddy)"
        in generated_code
    )
    assert (
        "return layerLeaf(shadowArrays, shadowArraysSampler, shadowArraysQuerySampler, uvLayer, depth, lod, ddx, ddy);"
        in generated_code
    )
    assert (
        "float cubeLeaf(TextureCubeArray cubeShadowArrays[4], SamplerComparisonState cubeShadowArraysSampler, SamplerState cubeShadowArraysQuerySampler, float4 cubeLayer, float depth, float lod, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "float2 query = float2(cubeShadowArrays[2].CalculateLevelOfDetailUnclamped(cubeShadowArraysQuerySampler, cubeLayer.xyz), cubeShadowArrays[2].CalculateLevelOfDetail(cubeShadowArraysQuerySampler, cubeLayer.xyz));"
        in generated_code
    )
    assert (
        "float cmp = cubeShadowArrays[2].SampleCmp(cubeShadowArraysSampler, cubeLayer, depth);"
        in generated_code
    )
    assert (
        "float cmpLod = cubeShadowArrays[1].SampleCmpLevel(cubeShadowArraysSampler, cubeLayer, depth, lod);"
        in generated_code
    )
    assert (
        "float cmpGrad = cubeShadowArrays[3].SampleCmpGrad(cubeShadowArraysSampler, cubeLayer, depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "float cubeMid(TextureCubeArray cubeShadowArrays[4], SamplerComparisonState cubeShadowArraysSampler, SamplerState cubeShadowArraysQuerySampler, float4 cubeLayer, float depth, float lod, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "return cubeLeaf(cubeShadowArrays, cubeShadowArraysSampler, cubeShadowArraysQuerySampler, cubeLayer, depth, lod, ddx, ddy);"
        in generated_code
    )
    assert (
        "layerMid(shadowArrays, shadowArraysSampler, shadowArraysQuerySampler, input.uvLayer, input.depth, input.lod, input.ddxLayer, input.ddyLayer)"
        in generated_code
    )
    assert (
        "cubeMid(cubeShadowArrays, cubeShadowArraysSampler, cubeShadowArraysQuerySampler, input.cubeLayer, input.depth, input.lod, input.ddxCube, input.ddyCube)"
        in generated_code
    )
    assert "CalculateLevelOfDetailUnclamped(shadowArraysSampler" not in generated_code
    assert (
        "CalculateLevelOfDetailUnclamped(cubeShadowArraysSampler" not in generated_code
    )
    assert "SampleCmp(shadowArraysQuerySampler" not in generated_code
    assert "SampleCmp(cubeShadowArraysQuerySampler" not in generated_code
    assert "textureQueryLod(" not in generated_code
    assert "textureCompare(" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGrad(" not in generated_code


def test_directx_implicit_texture_query_lod_parameter_threads_regular_sampler():
    shader = """
    shader ImplicitQueryLodParameter {
        samplerCubeArrayShadow cubeShadowArray;

        struct FSInput {
            vec4 cubeLayer @ TEXCOORD0;
        };

        vec2 queryCubeShadow(samplerCubeArrayShadow tex, vec4 cubeLayer) {
            return textureQueryLod(tex, cubeLayer);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec2 lod = queryCubeShadow(cubeShadowArray, input.cubeLayer);
                return vec4(lod, 0.0, 1.0);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "TextureCubeArray cubeShadowArray : register(t0);" in generated_code
    assert "SamplerState cubeShadowArraySampler : register(s0);" in generated_code
    assert "SamplerComparisonState" not in generated_code
    assert (
        "float2 queryCubeShadow(TextureCubeArray tex, SamplerState texSampler, float4 cubeLayer)"
        in generated_code
    )
    assert (
        "tex.CalculateLevelOfDetailUnclamped(texSampler, cubeLayer.xyz)"
        in generated_code
    )
    assert "tex.CalculateLevelOfDetail(texSampler, cubeLayer.xyz)" in generated_code
    assert (
        "queryCubeShadow(cubeShadowArray, cubeShadowArraySampler, input.cubeLayer)"
        in generated_code
    )


def test_directx_mixed_implicit_shadow_regular_sample_and_compare_split_samplers():
    shader = """
    shader MixedImplicitShadowSampler {
        sampler2DShadow shadowMap;

        struct FSInput {
            vec2 uv;
            float depth;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 raw = texture(shadowMap, input.uv);
                float shadow = textureCompare(shadowMap, input.uv, input.depth);
                return raw * shadow;
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert "SamplerState shadowMapRegularSampler : register(s1);" in generated_code
    assert (
        "float4 raw = shadowMap.Sample(shadowMapRegularSampler, input.uv);"
        in generated_code
    )
    assert (
        "float shadow = shadowMap.SampleCmp(shadowMapSampler, input.uv, input.depth);"
        in generated_code
    )


def test_directx_mixed_implicit_shadow_parameter_regular_sample_and_compare_threads_split_samplers():
    shader = """
    shader MixedImplicitShadowSamplerParameter {
        sampler2DShadow shadowMap;

        struct FSInput {
            vec2 uv;
            float depth;
        };

        vec4 inspect(sampler2DShadow tex, vec2 uv, float depth) {
            vec4 raw = texture(tex, uv);
            float shadow = textureCompare(tex, uv, depth);
            return raw * shadow;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return inspect(shadowMap, input.uv, input.depth);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert "SamplerState shadowMapRegularSampler : register(s1);" in generated_code
    assert (
        "float4 inspect(Texture2D tex, SamplerComparisonState texSampler, SamplerState texRegularSampler, float2 uv, float depth)"
        in generated_code
    )
    assert "float4 raw = tex.Sample(texRegularSampler, uv);" in generated_code
    assert "float shadow = tex.SampleCmp(texSampler, uv, depth);" in generated_code
    assert (
        "inspect(shadowMap, shadowMapSampler, shadowMapRegularSampler, input.uv, input.depth)"
        in generated_code
    )


def test_directx_declared_implicit_shadow_sampler_name_is_comparison_sampler():
    shader = """
    shader DeclaredImplicitShadowSamplerName {
        sampler2DShadow shadowMap;
        sampler shadowMapSampler;

        struct FSInput {
            vec2 uv;
            float depth;
        };

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return textureCompare(shadowMap, input.uv, input.depth);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert "SamplerState shadowMapSampler" not in generated_code
    assert (
        "return shadowMap.SampleCmp(shadowMapSampler, input.uv, input.depth);"
        in generated_code
    )


def test_directx_declared_implicit_shadow_sampler_names_split_mixed_regular_and_compare():
    shader = """
    shader DeclaredImplicitMixedShadowSamplerNames {
        sampler2DShadow shadowMap;
        sampler shadowMapSampler;
        sampler shadowMapRegularSampler;

        struct FSInput {
            vec2 uv;
            float depth;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 raw = texture(shadowMap, input.uv);
                float shadow = textureCompare(shadowMap, input.uv, input.depth);
                return raw * shadow;
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert "SamplerState shadowMapRegularSampler : register(s1);" in generated_code
    assert generated_code.count("shadowMapSampler : register") == 1
    assert generated_code.count("shadowMapRegularSampler : register") == 1
    assert (
        "float4 raw = shadowMap.Sample(shadowMapRegularSampler, input.uv);"
        in generated_code
    )
    assert (
        "float shadow = shadowMap.SampleCmp(shadowMapSampler, input.uv, input.depth);"
        in generated_code
    )


def test_directx_existing_implicit_sampler_parameter_is_not_inserted_twice():
    shader = """
    shader ExistingImplicitSamplerParameter {
        sampler2D colorMap;
        sampler colorMapSampler;

        struct FSInput {
            vec2 uv;
        };

        vec4 inspect(sampler2D tex, sampler texSampler, vec2 uv) {
            return texture(tex, uv);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return inspect(colorMap, colorMapSampler, input.uv);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "SamplerState colorMapSampler : register(s0);" in generated_code
    assert (
        "float4 inspect(Texture2D tex, SamplerState texSampler, float2 uv)"
        in generated_code
    )
    assert "return tex.Sample(texSampler, uv);" in generated_code
    assert "return inspect(colorMap, colorMapSampler, input.uv);" in generated_code
    assert "inspect(colorMap, colorMapSampler, colorMapSampler" not in generated_code


def test_directx_existing_implicit_comparison_sampler_parameter_is_not_inserted_twice():
    shader = """
    shader ExistingImplicitComparisonSamplerParameter {
        sampler2DShadow shadowMap;
        sampler shadowMapSampler;

        struct FSInput {
            vec2 uv;
            float depth;
        };

        float inspect(sampler2DShadow tex, sampler texSampler, vec2 uv, float depth) {
            return textureCompare(tex, uv, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return inspect(shadowMap, shadowMapSampler, input.uv, input.depth);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert (
        "float inspect(Texture2D tex, SamplerComparisonState texSampler, float2 uv, float depth)"
        in generated_code
    )
    assert "return tex.SampleCmp(texSampler, uv, depth);" in generated_code
    assert (
        "return inspect(shadowMap, shadowMapSampler, input.uv, input.depth);"
        in generated_code
    )
    assert "inspect(shadowMap, shadowMapSampler, shadowMapSampler" not in generated_code


def test_directx_declared_implicit_shadow_array_sampler_name_is_comparison_sampler():
    shader = """
    shader DeclaredImplicitShadowArraySamplerName {
        sampler2DShadow shadowMaps[4];
        sampler shadowMapsSampler;

        struct FSInput {
            vec2 uv;
            float depth;
            int layer;
        };

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return textureCompare(shadowMaps[input.layer], input.uv, input.depth);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowMapsSampler : register(s0);" in generated_code
    assert "SamplerState shadowMapsSampler" not in generated_code
    assert generated_code.count("shadowMapsSampler : register") == 1
    assert (
        "return shadowMaps[input.layer].SampleCmp(shadowMapsSampler, input.uv, input.depth);"
        in generated_code
    )
    assert "shadowMaps[input.layer]Sampler" not in generated_code


def test_directx_existing_implicit_shadow_array_sampler_parameter_is_not_inserted_twice():
    shader = """
    shader ExistingImplicitShadowArraySamplerParameter {
        sampler2DShadow shadowMaps[4];
        sampler shadowMapsSampler;

        struct FSInput {
            vec2 uv;
            float depth;
            int layer;
        };

        float inspect(sampler2DShadow maps[4], sampler mapsSampler, int layer, vec2 uv, float depth) {
            return textureCompare(maps[layer], uv, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return inspect(shadowMaps, shadowMapsSampler, input.layer, input.uv, input.depth);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "SamplerComparisonState shadowMapsSampler : register(s0);" in generated_code
    assert (
        "float inspect(Texture2D maps[4], SamplerComparisonState mapsSampler, int layer, float2 uv, float depth)"
        in generated_code
    )
    assert "return maps[layer].SampleCmp(mapsSampler, uv, depth);" in generated_code
    assert (
        "return inspect(shadowMaps, shadowMapsSampler, input.layer, input.uv, input.depth);"
        in generated_code
    )
    assert (
        "inspect(shadowMaps, shadowMapsSampler, shadowMapsSampler" not in generated_code
    )


def test_directx_unsized_implicit_shadow_array_helper_threads_synthetic_sampler_once():
    shader = """
    shader ImplicitUnsizedShadowArrayHelper {
        sampler2DShadow shadowMaps[];

        struct FSInput {
            vec2 uv;
            float depth;
        };

        float inspect(sampler2DShadow maps[], vec2 uv, float depth) {
            return textureCompare(maps[3], uv, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return inspect(shadowMaps, input.uv, input.depth);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowMapsSampler : register(s0);" in generated_code
    assert (
        "float inspect(Texture2D maps[4], SamplerComparisonState mapsSampler, float2 uv, float depth)"
        in generated_code
    )
    assert "return maps[3].SampleCmp(mapsSampler, uv, depth);" in generated_code
    assert (
        "return inspect(shadowMaps, shadowMapsSampler, input.uv, input.depth);"
        in generated_code
    )
    assert (
        "inspect(shadowMaps, shadowMapsSampler, shadowMapsSampler" not in generated_code
    )


def test_directx_mixed_unsized_implicit_shadow_array_helper_splits_regular_and_compare_samplers():
    shader = """
    shader MixedImplicitUnsizedShadowArrayHelper {
        sampler2DShadow shadowMaps[];

        struct FSInput {
            vec2 uv;
            float depth;
            int layer;
        };

        vec4 inspect(sampler2DShadow maps[], int layer, vec2 uv, float depth) {
            vec4 raw = texture(maps[layer], uv);
            float shadow = textureCompare(maps[3], uv, depth);
            return raw * shadow;
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return inspect(shadowMaps, input.layer, input.uv, input.depth);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowMapsSampler : register(s0);" in generated_code
    assert "SamplerState shadowMapsRegularSampler : register(s1);" in generated_code
    assert (
        "float4 inspect(Texture2D maps[4], SamplerComparisonState mapsSampler, SamplerState mapsRegularSampler, int layer, float2 uv, float depth)"
        in generated_code
    )
    assert "float4 raw = maps[layer].Sample(mapsRegularSampler, uv);" in generated_code
    assert "float shadow = maps[3].SampleCmp(mapsSampler, uv, depth);" in generated_code
    assert (
        "return inspect(shadowMaps, shadowMapsSampler, shadowMapsRegularSampler, input.layer, input.uv, input.depth);"
        in generated_code
    )


def test_directx_mixed_implicit_cube_shadow_regular_sample_and_compare_split_samplers():
    shader = """
    shader MixedImplicitCubeShadowSampler {
        samplerCubeShadow cubeShadow;
        samplerCubeArrayShadow cubeArrayShadow;

        struct FSInput {
            vec3 direction;
            vec4 cubeLayer;
            float depth;
        };

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                vec4 rawCube = texture(cubeShadow, input.direction);
                float cmpCube = textureCompare(cubeShadow, input.direction, input.depth);
                vec4 rawArray = texture(cubeArrayShadow, input.cubeLayer);
                float cmpArray = textureCompare(cubeArrayShadow, input.cubeLayer, input.depth);
                return rawCube * cmpCube + rawArray * cmpArray;
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "TextureCube cubeShadow : register(t0);" in generated_code
    assert "SamplerComparisonState cubeShadowSampler : register(s0);" in generated_code
    assert "SamplerState cubeShadowRegularSampler : register(s1);" in generated_code
    assert "TextureCubeArray cubeArrayShadow : register(t1);" in generated_code
    assert (
        "SamplerComparisonState cubeArrayShadowSampler : register(s2);"
        in generated_code
    )
    assert (
        "SamplerState cubeArrayShadowRegularSampler : register(s3);" in generated_code
    )
    assert (
        "float4 rawCube = cubeShadow.Sample(cubeShadowRegularSampler, input.direction);"
        in generated_code
    )
    assert (
        "float cmpCube = cubeShadow.SampleCmp(cubeShadowSampler, input.direction, input.depth);"
        in generated_code
    )
    assert (
        "float4 rawArray = cubeArrayShadow.Sample(cubeArrayShadowRegularSampler, input.cubeLayer);"
        in generated_code
    )
    assert (
        "float cmpArray = cubeArrayShadow.SampleCmp(cubeArrayShadowSampler, input.cubeLayer, input.depth);"
        in generated_code
    )


def test_directx_mixed_implicit_shadow_query_lod_and_compare_split_samplers():
    shader = """
    shader MixedShadowQueryLodCompare {
        sampler2DShadow shadowMap;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        float inspectShadow(sampler2DShadow tex, vec2 uv, float depth) {
            vec2 lod = textureQueryLod(tex, uv);
            float cmp = textureCompare(tex, uv, depth);
            return lod.x + cmp;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return inspectShadow(shadowMap, input.uv, input.depth);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert "SamplerState shadowMapQuerySampler : register(s1);" in generated_code
    assert (
        "float inspectShadow(Texture2D tex, SamplerComparisonState texSampler, SamplerState texQuerySampler, float2 uv, float depth)"
        in generated_code
    )
    assert (
        "float2 lod = float2(tex.CalculateLevelOfDetailUnclamped(texQuerySampler, uv), tex.CalculateLevelOfDetail(texQuerySampler, uv));"
        in generated_code
    )
    assert "float cmp = tex.SampleCmp(texSampler, uv, depth);" in generated_code
    assert (
        "inspectShadow(shadowMap, shadowMapSampler, shadowMapQuerySampler, input.uv, input.depth)"
        in generated_code
    )


def test_directx_mixed_implicit_cube_shadow_query_lod_and_compare_split_samplers():
    shader = """
    shader MixedCubeShadowQueryLodCompare {
        samplerCubeShadow cubeShadow;
        samplerCubeArrayShadow cubeShadowArray;

        struct FSInput {
            vec3 direction @ TEXCOORD0;
            vec4 cubeLayer @ TEXCOORD1;
            float depth;
            float lod;
            vec3 ddx @ TEXCOORD2;
            vec3 ddy @ TEXCOORD3;
        };

        float inspectCubeShadow(
            samplerCubeShadow tex,
            vec3 direction,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy
        ) {
            vec2 lodValue = textureQueryLod(tex, direction);
            float cmp = textureCompare(tex, direction, depth);
            float cmpLod = textureCompareLod(tex, direction, depth, lod);
            float grad = textureCompareGrad(tex, direction, depth, ddx, ddy);
            return lodValue.x + cmp + cmpLod + grad;
        }

        float inspectCubeArrayShadow(
            samplerCubeArrayShadow tex,
            vec4 cubeLayer,
            float depth,
            float lod,
            vec3 ddx,
            vec3 ddy
        ) {
            vec2 lodValue = textureQueryLod(tex, cubeLayer);
            float cmp = textureCompare(tex, cubeLayer, depth);
            float cmpLod = textureCompareLod(tex, cubeLayer, depth, lod);
            float grad = textureCompareGrad(tex, cubeLayer, depth, ddx, ddy);
            return lodValue.y + cmp + cmpLod + grad;
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return inspectCubeShadow(
                    cubeShadow,
                    input.direction,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy
                ) + inspectCubeArrayShadow(
                    cubeShadowArray,
                    input.cubeLayer,
                    input.depth,
                    input.lod,
                    input.ddx,
                    input.ddy
                );
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "TextureCube cubeShadow : register(t0);" in generated_code
    assert "SamplerComparisonState cubeShadowSampler : register(s0);" in generated_code
    assert "SamplerState cubeShadowQuerySampler : register(s1);" in generated_code
    assert "TextureCubeArray cubeShadowArray : register(t1);" in generated_code
    assert (
        "SamplerComparisonState cubeShadowArraySampler : register(s2);"
        in generated_code
    )
    assert "SamplerState cubeShadowArrayQuerySampler : register(s3);" in generated_code
    assert (
        "float inspectCubeShadow(TextureCube tex, SamplerComparisonState texSampler, SamplerState texQuerySampler, float3 direction, float depth, float lod, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "float2 lodValue = float2(tex.CalculateLevelOfDetailUnclamped(texQuerySampler, direction), tex.CalculateLevelOfDetail(texQuerySampler, direction));"
        in generated_code
    )
    assert "float cmp = tex.SampleCmp(texSampler, direction, depth);" in generated_code
    assert (
        "float cmpLod = tex.SampleCmpLevel(texSampler, direction, depth, lod);"
        in generated_code
    )
    assert (
        "float grad = tex.SampleCmpGrad(texSampler, direction, depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "float inspectCubeArrayShadow(TextureCubeArray tex, SamplerComparisonState texSampler, SamplerState texQuerySampler, float4 cubeLayer, float depth, float lod, float3 ddx, float3 ddy)"
        in generated_code
    )
    assert (
        "float2 lodValue = float2(tex.CalculateLevelOfDetailUnclamped(texQuerySampler, cubeLayer.xyz), tex.CalculateLevelOfDetail(texQuerySampler, cubeLayer.xyz));"
        in generated_code
    )
    assert "float cmp = tex.SampleCmp(texSampler, cubeLayer, depth);" in generated_code
    assert (
        "float cmpLod = tex.SampleCmpLevel(texSampler, cubeLayer, depth, lod);"
        in generated_code
    )
    assert (
        "float grad = tex.SampleCmpGrad(texSampler, cubeLayer, depth, ddx, ddy);"
        in generated_code
    )
    assert (
        "inspectCubeShadow(cubeShadow, cubeShadowSampler, cubeShadowQuerySampler, input.direction, input.depth, input.lod, input.ddx, input.ddy)"
        in generated_code
    )
    assert (
        "inspectCubeArrayShadow(cubeShadowArray, cubeShadowArraySampler, cubeShadowArrayQuerySampler, input.cubeLayer, input.depth, input.lod, input.ddx, input.ddy)"
        in generated_code
    )
    assert "textureQueryLod(" not in generated_code
    assert "textureCompare(" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGrad(" not in generated_code


def test_directx_nested_implicit_shadow_compare_lod_grad_threads_split_samplers():
    shader = """
    shader NestedShadowCompareQuery {
        sampler2DShadow shadowMap;

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
            vec2 ddx @ TEXCOORD1;
            vec2 ddy @ TEXCOORD2;
            ivec2 offset @ TEXCOORD3;
        };

        float shadowOps(
            sampler2DShadow tex,
            vec2 uv,
            float depth,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            vec2 lod = textureQueryLod(tex, uv);
            float cmp = textureCompareLod(tex, uv, depth, lod.x);
            float grad = textureCompareGradOffset(tex, uv, depth, ddx, ddy, offset);
            return cmp + grad + lod.y;
        }

        float wrappedShadow(
            sampler2DShadow tex,
            vec2 uv,
            float depth,
            vec2 ddx,
            vec2 ddy,
            ivec2 offset
        ) {
            return shadowOps(tex, uv, depth, ddx, ddy, offset);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return vec4(wrappedShadow(
                    shadowMap,
                    input.uv,
                    input.depth,
                    input.ddx,
                    input.ddy,
                    input.offset
                ));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate_stage(
        crosstl.translator.parse(shader), "fragment"
    )

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert "SamplerState shadowMapQuerySampler : register(s1);" in generated_code
    assert (
        "float shadowOps(Texture2D tex, SamplerComparisonState texSampler, SamplerState texQuerySampler, float2 uv, float depth, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "float2 lod = float2(tex.CalculateLevelOfDetailUnclamped(texQuerySampler, uv), tex.CalculateLevelOfDetail(texQuerySampler, uv));"
        in generated_code
    )
    assert (
        "float cmp = tex.SampleCmpLevel(texSampler, uv, depth, lod.x);"
        in generated_code
    )
    assert (
        "float grad = tex.SampleCmpGrad(texSampler, uv, depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "float wrappedShadow(Texture2D tex, SamplerComparisonState texSampler, SamplerState texQuerySampler, float2 uv, float depth, float2 ddx, float2 ddy, int2 offset)"
        in generated_code
    )
    assert (
        "return shadowOps(tex, texSampler, texQuerySampler, uv, depth, ddx, ddy, offset);"
        in generated_code
    )
    assert (
        "wrappedShadow(shadowMap, shadowMapSampler, shadowMapQuerySampler, input.uv, input.depth, input.ddx, input.ddy, input.offset)"
        in generated_code
    )
    assert "textureQueryLod(" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGradOffset(" not in generated_code


def test_directx_texture_operation_variants():
    shader = """
    shader TextureOps {
        sampler2D colorMap;

        struct VSOutput {
            vec2 uv;
            ivec2 pixel;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                vec4 lodColor = textureLod(colorMap, input.uv, 1.0);
                vec4 gradColor = textureGrad(colorMap, input.uv, vec2(0.1), vec2(0.2));
                vec4 fetched = texelFetch(colorMap, input.pixel, 0);
                vec4 gathered = textureGather(colorMap, input.uv);
                return lodColor + gradColor + fetched + gathered;
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "colorMap.SampleLevel(colorMapSampler, input.uv, 1.0)" in generated_code
    assert (
        "colorMap.SampleGrad(colorMapSampler, input.uv, float2(0.1, 0.1), float2(0.2, 0.2))"
        in generated_code
    )
    assert "colorMap.Load(int3(input.pixel, 0))" in generated_code
    assert "colorMap.Gather(colorMapSampler, input.uv)" in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGrad(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "textureGather(" not in generated_code


def test_directx_explicit_sampler_argument():
    shader = """
    shader ExplicitSampler {
        sampler2D colorMap;
        sampler linearSampler;

        struct VSOutput {
            vec2 uv;
        };

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return texture(colorMap, linearSampler, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "colorMap.Sample(linearSampler, input.uv)" in generated_code
    assert "colorMapSampler" not in generated_code


def test_directx_sampler_parameter_texture_call():
    shader = """
    shader SamplerParameter {
        sampler2D colorMap;
        sampler linearSampler;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleColor(sampler sampleState, vec2 uv) {
            return texture(colorMap, sampleState, uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleColor(linearSampler, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "float4 sampleColor(SamplerState sampleState, float2 uv)" in generated_code
    assert "colorMap.Sample(sampleState, uv)" in generated_code
    assert "sampleColor(linearSampler, input.uv)" in generated_code
    assert "colorMapSampler" not in generated_code


def test_directx_texture_and_sampler_parameters():
    shader = """
    shader TextureParameter {
        sampler2D colorMap;
        sampler linearSampler;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleColor(sampler2D tex, sampler sampleState, vec2 uv) {
            return texture(tex, sampleState, uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleColor(colorMap, linearSampler, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert (
        "float4 sampleColor(Texture2D tex, SamplerState sampleState, float2 uv)"
        in generated_code
    )
    assert "tex.Sample(sampleState, uv)" in generated_code
    assert "sampleColor(colorMap, linearSampler, input.uv)" in generated_code
    assert "colorMapSampler" not in generated_code


def test_directx_struct_member_sampler_expression_is_explicit_sampler():
    shader = """
    shader StructSamplerExpression {
        sampler2D colorMap;

        struct SamplerPack {
            sampler samplers[2];
        };

        vec4 samplePacked(sampler2D tex, SamplerPack pack, int index, vec2 uv) {
            return texture(tex, pack.samplers[index], uv);
        }

        fragment {
            vec4 main() @ gl_FragColor {
                SamplerPack pack;
                return samplePacked(colorMap, pack, 0, vec2(0.5));
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct SamplerPack" in generated_code
    assert "SamplerState samplers[2];" in generated_code
    assert (
        "float4 samplePacked(Texture2D tex, SamplerPack pack, int index, float2 uv)"
        in generated_code
    )
    assert "return tex.Sample(pack.samplers[index], uv);" in generated_code
    assert "return samplePacked(colorMap, pack, 0, float2(0.5, 0.5));" in generated_code
    assert "texSampler" not in generated_code
    assert "SampleBias(tex" not in generated_code


def test_directx_struct_member_sampler_expression_used_for_compare_is_comparison_sampler():
    shader = """
    shader StructCompareSamplerExpression {
        sampler2DShadow shadowMap;

        struct SamplerPack {
            sampler samplers[2];
        };

        float samplePacked(sampler2DShadow tex, SamplerPack pack, int index, vec2 uv, float depth) {
            return textureCompare(tex, pack.samplers[index], uv, depth);
        }

        fragment {
            float main() @ gl_FragDepth {
                SamplerPack pack;
                return samplePacked(shadowMap, pack, 0, vec2(0.5), 0.5);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "struct SamplerPack" in generated_code
    assert "SamplerComparisonState samplers[2];" in generated_code
    assert "SamplerState samplers[2];" not in generated_code
    assert (
        "float samplePacked(Texture2D tex, SamplerPack pack, int index, float2 uv, float depth)"
        in generated_code
    )
    assert "return tex.SampleCmp(pack.samplers[index], uv, depth);" in generated_code
    assert (
        "return samplePacked(shadowMap, pack, 0, float2(0.5, 0.5), 0.5);"
        in generated_code
    )
    assert "texSampler" not in generated_code


def test_directx_struct_member_sampler_forwarded_to_compare_helper_is_comparison_sampler():
    shader = """
    shader StructCompareSamplerForwarded {
        sampler2DShadow shadowMap;

        struct SamplerPack {
            sampler samplers[2];
        };

        float compareWithSampler(
            sampler2DShadow tex,
            sampler compareSampler,
            vec2 uv,
            float depth
        ) {
            return textureCompare(tex, compareSampler, uv, depth);
        }

        float samplePacked(SamplerPack pack, int index, vec2 uv, float depth) {
            return compareWithSampler(
                shadowMap,
                pack.samplers[index],
                uv,
                depth
            );
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "SamplerComparisonState samplers[2];" in generated_code
    assert "SamplerState samplers[2];" not in generated_code
    assert (
        "float compareWithSampler(Texture2D tex, SamplerComparisonState compareSampler, float2 uv, float depth)"
        in generated_code
    )
    assert "return tex.SampleCmp(compareSampler, uv, depth);" in generated_code
    assert (
        "return compareWithSampler(shadowMap, pack.samplers[index], uv, depth);"
        in generated_code
    )


def test_directx_texture_and_sampler_parameters_transitive():
    shader = """
    shader TextureParameter {
        sampler2D colorMap;
        sampler linearSampler;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleColor(sampler2D tex, sampler sampleState, vec2 uv) {
            return texture(tex, sampleState, uv);
        }

        vec4 sampleInput(sampler2D tex, sampler sampleState, VSOutput input) {
            return sampleColor(tex, sampleState, input.uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleInput(colorMap, linearSampler, input);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert (
        "float4 sampleColor(Texture2D tex, SamplerState sampleState, float2 uv)"
        in generated_code
    )
    assert (
        "float4 sampleInput(Texture2D tex, SamplerState sampleState, VSOutput input)"
        in generated_code
    )
    assert "sampleColor(tex, sampleState, input.uv)" in generated_code
    assert "sampleInput(colorMap, linearSampler, input)" in generated_code
    assert "colorMapSampler" not in generated_code


def test_directx_implicit_sampler_for_texture_parameter():
    shader = """
    shader TextureParameter {
        sampler2D colorMap;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleColor(sampler2D tex, vec2 uv) {
            return texture(tex, uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleColor(colorMap, input.uv);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D colorMap : register(t0);" in generated_code
    assert "SamplerState colorMapSampler : register(s0);" in generated_code
    assert (
        "float4 sampleColor(Texture2D tex, SamplerState texSampler, float2 uv)"
        in generated_code
    )
    assert "tex.Sample(texSampler, uv)" in generated_code
    assert "sampleColor(colorMap, colorMapSampler, input.uv)" in generated_code


def test_directx_implicit_sampler_for_texture_parameter_transitive():
    shader = """
    shader TextureParameter {
        sampler2D colorMap;

        struct VSOutput {
            vec2 uv;
        };

        vec4 sampleColor(sampler2D tex, vec2 uv) {
            return texture(tex, uv);
        }

        vec4 sampleInput(sampler2D tex, VSOutput input) {
            return sampleColor(tex, input.uv);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return sampleInput(colorMap, input);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert (
        "float4 sampleColor(Texture2D tex, SamplerState texSampler, float2 uv)"
        in generated_code
    )
    assert (
        "float4 sampleInput(Texture2D tex, SamplerState texSampler, VSOutput input)"
        in generated_code
    )
    assert "sampleColor(tex, texSampler, input.uv)" in generated_code
    assert "sampleInput(colorMap, colorMapSampler, input)" in generated_code


def test_directx_shadow_texture_compare():
    shader = """
    shader ShadowTexture {
        sampler2DShadow shadowMap;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return textureCompare(shadowMap, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert (
        "shadowMap.SampleCmp(shadowMapSampler, input.uv, input.depth)" in generated_code
    )
    assert "textureCompare(" not in generated_code


def test_directx_shadow_texture_array_compare():
    shader = """
    shader ShadowTextureArray {
        sampler2DShadow shadowMaps[4];

        struct VSOutput {
            vec2 uv;
            float depth;
            int layer;
        };

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return textureCompare(shadowMaps[input.layer], input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowMapsSampler : register(s0);" in generated_code
    assert (
        "shadowMaps[input.layer].SampleCmp(shadowMapsSampler, input.uv, input.depth)"
        in generated_code
    )
    assert "shadowMaps[input.layer]Sampler" not in generated_code


def test_directx_shadow_texture_array_compare_with_indexed_sampler_array():
    shader = """
    shader ShadowSamplerArrayHelper {
        sampler2DShadow shadowMaps[4];
        sampler shadowSamplers[4];

        struct VSOutput {
            vec2 uv;
            float depth;
            int layer;
        };

        float shadowLayer(sampler2DShadow shadowMaps[4], sampler shadowSamplers[4], int layer, vec2 uv, float depth) {
            return textureCompare(shadowMaps[layer], shadowSamplers[layer], uv, depth);
        }

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return shadowLayer(shadowMaps, shadowSamplers, input.layer, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[4], SamplerComparisonState shadowSamplers[4], int layer, float2 uv, float depth)"
        in generated_code
    )
    assert (
        "shadowMaps[layer].SampleCmp(shadowSamplers[layer], uv, depth)"
        in generated_code
    )
    assert (
        "shadowLayer(shadowMaps, shadowSamplers, input.layer, input.uv, input.depth)"
        in generated_code
    )
    assert "shadowMapsSampler" not in generated_code


def test_directx_fixed_shadow_texture_and_sampler_arrays_keep_declared_size_with_constant_indices():
    shader = """
    shader FixedShadowArrayConstantIndex {
        const int LAYER = 2;
        sampler2DShadow shadowMaps[6];
        sampler shadowSamplers[6];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[6], sampler shadowSamplers[6], vec2 uv, float depth) {
            return textureCompare(shadowMaps[LAYER], shadowSamplers[LAYER], uv, depth) + textureCompare(shadowMaps[1 + 2], shadowSamplers[1 + 2], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int LAYER = 2;" in generated_code
    assert "Texture2D shadowMaps[6] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[6] : register(s0);" in generated_code
    assert "Texture2D afterShadow : register(t6);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s6);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[6], SamplerComparisonState shadowSamplers[6], float2 uv, float depth)"
        in generated_code
    )
    assert (
        "shadowMaps[LAYER].SampleCmp(shadowSamplers[LAYER], uv, depth)"
        in generated_code
    )
    assert (
        "shadowMaps[(1 + 2)].SampleCmp(shadowSamplers[(1 + 2)], uv, depth)"
        in generated_code
    )
    assert "Texture2D shadowMaps[4] : register(t0);" not in generated_code
    assert "Texture2D afterShadow : register(t4);" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_fixed_shadow_texture_and_sampler_arrays_resolve_constant_declared_size_for_bindings():
    shader = """
    shader ConstSizedShadowResourceArrays {
        const int BASE_COUNT = 2;
        const int SHADOW_COUNT = BASE_COUNT * 3;
        sampler2DShadow shadowMaps[SHADOW_COUNT];
        sampler shadowSamplers[SHADOW_COUNT];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[SHADOW_COUNT], sampler shadowSamplers[SHADOW_COUNT], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2], shadowSamplers[2], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int BASE_COUNT = 2;" in generated_code
    assert "static const int SHADOW_COUNT = (BASE_COUNT * 3);" in generated_code
    assert "Texture2D shadowMaps[SHADOW_COUNT] : register(t0);" in generated_code
    assert (
        "SamplerComparisonState shadowSamplers[SHADOW_COUNT] : register(s0);"
        in generated_code
    )
    assert "Texture2D afterShadow : register(t6);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s6);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[SHADOW_COUNT], SamplerComparisonState shadowSamplers[SHADOW_COUNT], float2 uv, float depth)"
        in generated_code
    )
    assert "shadowMaps[2].SampleCmp(shadowSamplers[2], uv, depth)" in generated_code
    assert "Texture2D afterShadow : register(t1);" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_fixed_shadow_texture_and_sampler_arrays_resolve_inline_declared_size_expression_for_bindings():
    shader = """
    shader ExprSizedShadowResourceArrays {
        sampler2DShadow shadowMaps[2 * 3];
        sampler shadowSamplers[2 * 3];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[2 * 3], sampler shadowSamplers[2 * 3], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2], shadowSamplers[2], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMaps[(2 * 3)] : register(t0);" in generated_code
    assert (
        "SamplerComparisonState shadowSamplers[(2 * 3)] : register(s0);"
        in generated_code
    )
    assert "Texture2D afterShadow : register(t6);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s6);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[(2 * 3)], SamplerComparisonState shadowSamplers[(2 * 3)], float2 uv, float depth)"
        in generated_code
    )
    assert "shadowMaps[2].SampleCmp(shadowSamplers[2], uv, depth)" in generated_code
    assert "Texture2D afterShadow : register(t1);" not in generated_code
    assert "[None]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_fixed_shadow_texture_arrays_preserve_parenthesized_and_unary_declared_sizes():
    shader = """
    shader ParenthesizedSizedShadowResourceArrays {
        sampler2DShadow shadowMaps[(2 + 1) * 2];
        sampler shadowSamplers[(2 + 1) * 2];
        sampler2DShadow unaryShadowMaps[+6];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[(2 + 1) * 2], sampler shadowSamplers[(2 + 1) * 2], sampler2DShadow unaryShadowMaps[+6], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2], shadowSamplers[2], uv, depth) + textureCompare(unaryShadowMaps[2], shadowSamplers[2], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, unaryShadowMaps, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMaps[((2 + 1) * 2)] : register(t0);" in generated_code
    assert (
        "SamplerComparisonState shadowSamplers[((2 + 1) * 2)] : register(s0);"
        in generated_code
    )
    assert "Texture2D unaryShadowMaps[+6] : register(t6);" in generated_code
    assert "Texture2D afterShadow : register(t12);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s6);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[((2 + 1) * 2)], SamplerComparisonState shadowSamplers[((2 + 1) * 2)], Texture2D unaryShadowMaps[+6], float2 uv, float depth)"
        in generated_code
    )
    assert (
        "shadowLayer(shadowMaps, shadowSamplers, unaryShadowMaps, input.uv, input.depth)"
        in generated_code
    )
    assert "shadowMaps[2].SampleCmp(shadowSamplers[2], uv, depth)" in generated_code
    assert (
        "unaryShadowMaps[2].SampleCmp(shadowSamplers[2], uv, depth)" in generated_code
    )
    assert "Texture2D afterShadow : register(t6);" not in generated_code
    assert "[None]" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_unsized_shadow_texture_and_sampler_arrays_infer_helper_size():
    shader = """
    shader UnsizedShadowSamplerArrayHelper {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            float high = textureCompare(shadowMaps[3], shadowSamplers[3], uv, depth);
            float low = textureCompare(shadowMaps[1], shadowSamplers[1], uv, depth);
            return high + low;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in generated_code
    assert "Texture2D afterShadow : register(t4);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s4);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[4], SamplerComparisonState shadowSamplers[4], float2 uv, float depth)"
        in generated_code
    )
    assert "shadowMaps[3].SampleCmp(shadowSamplers[3], uv, depth)" in generated_code
    assert "shadowMaps[1].SampleCmp(shadowSamplers[1], uv, depth)" in generated_code
    assert (
        "afterShadow.SampleCmp(afterSampler, input.uv, input.depth)" in generated_code
    )
    assert "Texture2D shadowMaps[] : register(t0);" not in generated_code
    assert "SamplerState shadowSamplers[]" not in generated_code
    assert "shadowMapsSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_unsized_shadow_texture_and_sampler_arrays_infer_transitive_helper_size():
    shader = """
    shader MultiHopUnsizedShadowResources {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowDeep(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            float high = textureCompare(shadowMaps[4], shadowSamplers[4], uv, depth);
            float low = textureCompare(shadowMaps[1], shadowSamplers[1], uv, depth);
            return high + low;
        }

        float shadowMid(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return shadowDeep(shadowMaps, shadowSamplers, uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowMid(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMaps[5] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[5] : register(s0);" in generated_code
    assert "Texture2D afterShadow : register(t5);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s5);" in generated_code
    assert (
        "float shadowDeep(Texture2D shadowMaps[5], SamplerComparisonState shadowSamplers[5], float2 uv, float depth)"
        in generated_code
    )
    assert (
        "float shadowMid(Texture2D shadowMaps[5], SamplerComparisonState shadowSamplers[5], float2 uv, float depth)"
        in generated_code
    )
    assert "shadowMaps[4].SampleCmp(shadowSamplers[4], uv, depth)" in generated_code
    assert "shadowMaps[1].SampleCmp(shadowSamplers[1], uv, depth)" in generated_code
    assert "shadowDeep(shadowMaps, shadowSamplers, uv, depth)" in generated_code
    assert (
        "shadowMid(shadowMaps, shadowSamplers, input.uv, input.depth)" in generated_code
    )
    assert (
        "afterShadow.SampleCmp(afterSampler, input.uv, input.depth)" in generated_code
    )
    assert "Texture2D shadowMaps[] : register(t0);" not in generated_code
    assert "SamplerState shadowSamplers[]" not in generated_code
    assert "shadowMapsSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_unsized_shadow_texture_and_sampler_arrays_preserve_dynamic_indexing():
    shader = """
    shader MixedIndexedUnsizedShadowResources {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
            int layer;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], int layer, vec2 uv, float depth) {
            float dynamicShadow = textureCompare(shadowMaps[layer], shadowSamplers[layer], uv, depth);
            float fixedShadow = textureCompare(shadowMaps[3], shadowSamplers[3], uv, depth);
            return dynamicShadow + fixedShadow;
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.layer, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in generated_code
    assert "Texture2D afterShadow : register(t4);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s4);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[4], SamplerComparisonState shadowSamplers[4], int layer, float2 uv, float depth)"
        in generated_code
    )
    assert (
        "shadowMaps[layer].SampleCmp(shadowSamplers[layer], uv, depth)"
        in generated_code
    )
    assert "shadowMaps[3].SampleCmp(shadowSamplers[3], uv, depth)" in generated_code
    assert (
        "shadowLayer(shadowMaps, shadowSamplers, input.layer, input.uv, input.depth)"
        in generated_code
    )
    assert (
        "afterShadow.SampleCmp(afterSampler, input.uv, input.depth)" in generated_code
    )
    assert "Texture2D shadowMaps[] : register(t0);" not in generated_code
    assert "SamplerState shadowSamplers[]" not in generated_code
    assert "shadowMapsSampler" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_unsized_shadow_texture_and_sampler_arrays_ignore_unsupported_indices():
    dynamic_shader = """
    shader DynamicOnlyUnsizedShadowResources {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
            int layer;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], int layer, vec2 uv, float depth) {
            return textureCompare(shadowMaps[layer], shadowSamplers[layer], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.layer, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """
    negative_shader = """
    shader NegativeIndexedUnsizedShadowResources {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return textureCompare(shadowMaps[-1], shadowSamplers[-1], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    dynamic_code = HLSLCodeGen().generate(crosstl.translator.parse(dynamic_shader))
    negative_code = HLSLCodeGen().generate(crosstl.translator.parse(negative_shader))

    assert "Texture2D shadowMaps[] : register(t0);" in dynamic_code
    assert "SamplerComparisonState shadowSamplers[] : register(s0);" in dynamic_code
    assert "Texture2D afterShadow : register(t1);" in dynamic_code
    assert "SamplerComparisonState afterSampler : register(s1);" in dynamic_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[], SamplerComparisonState shadowSamplers[], int layer, float2 uv, float depth)"
        in dynamic_code
    )
    assert (
        "shadowMaps[layer].SampleCmp(shadowSamplers[layer], uv, depth)" in dynamic_code
    )
    assert "Texture2D shadowMaps[1] : register(t0);" not in dynamic_code
    assert "Texture2D afterShadow : register(t2);" not in dynamic_code

    assert "Texture2D shadowMaps[] : register(t0);" in negative_code
    assert "SamplerComparisonState shadowSamplers[] : register(s0);" in negative_code
    assert "Texture2D afterShadow : register(t1);" in negative_code
    assert "SamplerComparisonState afterSampler : register(s1);" in negative_code
    assert "shadowMaps[-1].SampleCmp(shadowSamplers[-1], uv, depth)" in negative_code
    assert "Texture2D shadowMaps[0] : register(t0);" not in negative_code
    assert "Texture2D afterShadow : register(t0);" not in negative_code


def test_directx_unsized_shadow_texture_and_sampler_arrays_infer_constant_expression_size():
    shader = """
    shader ExprIndexedUnsizedShadowResources {
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2 * 2], shadowSamplers[2 * 2], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMaps[5] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[5] : register(s0);" in generated_code
    assert "Texture2D afterShadow : register(t5);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s5);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[5], SamplerComparisonState shadowSamplers[5], float2 uv, float depth)"
        in generated_code
    )
    assert (
        "shadowMaps[(2 * 2)].SampleCmp(shadowSamplers[(2 * 2)], uv, depth)"
        in generated_code
    )
    assert "Texture2D shadowMaps[] : register(t0);" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_unsized_shadow_texture_and_sampler_arrays_infer_named_constant_size():
    shader = """
    shader ConstIndexedUnsizedShadowResources {
        const int BASE = 2;
        const int LAYER = BASE * 2;
        sampler2DShadow shadowMaps[];
        sampler shadowSamplers[];
        sampler2DShadow afterShadow;
        sampler afterSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float shadowLayer(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return textureCompare(shadowMaps[LAYER], shadowSamplers[LAYER], uv, depth);
        }

        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                float arrayShadow = shadowLayer(shadowMaps, shadowSamplers, input.uv, input.depth);
                float singleShadow = textureCompare(afterShadow, afterSampler, input.uv, input.depth);
                return vec4(arrayShadow + singleShadow);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int BASE = 2;" in generated_code
    assert "static const int LAYER = (BASE * 2);" in generated_code
    assert "Texture2D shadowMaps[5] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[5] : register(s0);" in generated_code
    assert "Texture2D afterShadow : register(t5);" in generated_code
    assert "SamplerComparisonState afterSampler : register(s5);" in generated_code
    assert (
        "float shadowLayer(Texture2D shadowMaps[5], SamplerComparisonState shadowSamplers[5], float2 uv, float depth)"
        in generated_code
    )
    assert (
        "shadowMaps[LAYER].SampleCmp(shadowSamplers[LAYER], uv, depth)"
        in generated_code
    )
    assert "Texture2D shadowMaps[] : register(t0);" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_fixed_shadow_texture_array_rejects_mismatched_fixed_helper_size():
    shader = """
    shader FixedShadowGlobalMismatch {
        sampler2DShadow shadowMaps[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        float shadowThree(sampler2DShadow shadowMaps[3], sampler shadowSamplers[3], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2], shadowSamplers[2], uv, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return shadowThree(shadowMaps, shadowSamplers, input.uv, input.depth);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'shadowMaps': 4 and 3",
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_fixed_shadow_texture_array_widens_unsized_helper():
    shader = """
    shader FixedShadowGlobalToUnsizedHelper {
        sampler2DShadow shadowMaps[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        float shadowUnsized(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return textureCompare(shadowMaps[2], shadowSamplers[2], uv, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return shadowUnsized(shadowMaps, shadowSamplers, input.uv, input.depth);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in generated_code
    assert (
        "float shadowUnsized(Texture2D shadowMaps[4], SamplerComparisonState shadowSamplers[4], float2 uv, float depth)"
        in generated_code
    )
    assert "shadowMaps[2].SampleCmp(shadowSamplers[2], uv, depth)" in generated_code
    assert (
        "shadowUnsized(shadowMaps, shadowSamplers, input.uv, input.depth)"
        in generated_code
    )
    assert "Texture2D shadowMaps[] : register(t0);" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_transitive_shadow_texture_array_shadowed_const_index_stays_dynamic():
    shader = """
    shader TransitiveShadowSamplerShadowedConstIndex {
        const int COUNT = 4;
        sampler2DShadow shadowMaps[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        float leaf(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            int COUNT = 0;
            return textureCompare(shadowMaps[COUNT], shadowSamplers[COUNT], uv, depth);
        }

        float passThrough(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            int COUNT = 0;
            float sampled = textureCompare(shadowMaps[COUNT], shadowSamplers[COUNT], uv, depth);
            return sampled + leaf(shadowMaps, shadowSamplers, uv, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return passThrough(shadowMaps, shadowSamplers, input.uv, input.depth);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "static const int COUNT = 4;" in generated_code
    assert "Texture2D shadowMaps[4] : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSamplers[4] : register(s0);" in generated_code
    assert (
        "float leaf(Texture2D shadowMaps[4], SamplerComparisonState shadowSamplers[4], float2 uv, float depth)"
        in generated_code
    )
    assert (
        "float passThrough(Texture2D shadowMaps[4], SamplerComparisonState shadowSamplers[4], float2 uv, float depth)"
        in generated_code
    )
    assert generated_code.count("int COUNT = 0;") == 2
    assert (
        "shadowMaps[COUNT].SampleCmp(shadowSamplers[COUNT], uv, depth)"
        in generated_code
    )
    assert "leaf(shadowMaps, shadowSamplers, uv, depth)" in generated_code
    assert (
        "passThrough(shadowMaps, shadowSamplers, input.uv, input.depth)"
        in generated_code
    )
    assert "Texture2D shadowMaps[] : register(t0);" not in generated_code
    assert "textureCompare(" not in generated_code


def test_directx_transitive_shadow_texture_array_unshadowed_const_index_conflict_raises():
    shader = """
    shader TransitiveShadowSamplerUnshadowedConstIndexConflict {
        const int COUNT = 4;
        sampler2DShadow shadowMaps[4];
        sampler shadowSamplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            float depth;
        };

        float leaf(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            return textureCompare(shadowMaps[COUNT], shadowSamplers[COUNT], uv, depth);
        }

        float passThrough(sampler2DShadow shadowMaps[], sampler shadowSamplers[], vec2 uv, float depth) {
            int COUNT = 0;
            return leaf(shadowMaps, shadowSamplers, uv, depth);
        }

        fragment {
            float main(FSInput input) @ gl_FragDepth {
                return passThrough(shadowMaps, shadowSamplers, input.uv, input.depth);
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match="Conflicting fixed resource array sizes for 'shadowMaps': 4 and 5",
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_shadow_compare_sampler_parameter():
    shader = """
    shader ShadowHelper {
        sampler2DShadow shadowMap;
        sampler shadowSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float sampleShadow(sampler compareSampler, vec2 uv, float depth) {
            return textureCompare(shadowMap, compareSampler, uv, depth);
        }

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return sampleShadow(shadowSampler, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSampler : register(s0);" in generated_code
    assert (
        "float sampleShadow(SamplerComparisonState compareSampler, float2 uv, float depth)"
        in generated_code
    )
    assert "shadowMap.SampleCmp(compareSampler, uv, depth)" in generated_code
    assert "sampleShadow(shadowSampler, input.uv, input.depth)" in generated_code
    assert "shadowMapSampler" not in generated_code


def test_directx_shadow_texture_and_sampler_parameters():
    shader = """
    shader ShadowParameter {
        sampler2DShadow shadowMap;
        sampler shadowSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float sampleShadow(sampler2DShadow tex, sampler compareSampler, vec2 uv, float depth) {
            return textureCompare(tex, compareSampler, uv, depth);
        }

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return sampleShadow(shadowMap, shadowSampler, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "SamplerComparisonState shadowSampler : register(s0);" in generated_code
    assert (
        "float sampleShadow(Texture2D tex, SamplerComparisonState compareSampler, float2 uv, float depth)"
        in generated_code
    )
    assert "tex.SampleCmp(compareSampler, uv, depth)" in generated_code
    assert (
        "sampleShadow(shadowMap, shadowSampler, input.uv, input.depth)"
        in generated_code
    )
    assert "shadowMapSampler" not in generated_code


def test_directx_implicit_comparison_sampler_for_shadow_texture_parameter():
    shader = """
    shader ShadowParameter {
        sampler2DShadow shadowMap;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float sampleShadow(sampler2DShadow tex, vec2 uv, float depth) {
            return textureCompare(tex, uv, depth);
        }

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return sampleShadow(shadowMap, input.uv, input.depth);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2D shadowMap : register(t0);" in generated_code
    assert "SamplerComparisonState shadowMapSampler : register(s0);" in generated_code
    assert (
        "float sampleShadow(Texture2D tex, SamplerComparisonState texSampler, float2 uv, float depth)"
        in generated_code
    )
    assert "tex.SampleCmp(texSampler, uv, depth)" in generated_code
    assert (
        "sampleShadow(shadowMap, shadowMapSampler, input.uv, input.depth)"
        in generated_code
    )


def test_directx_shadow_compare_sampler_parameter_transitive():
    shader = """
    shader ShadowHelper {
        sampler2DShadow shadowMap;
        sampler shadowSampler;

        struct VSOutput {
            vec2 uv;
            float depth;
        };

        float compareShadow(sampler compareSampler, vec2 uv, float depth) {
            return textureCompare(shadowMap, compareSampler, uv, depth);
        }

        float sampleShadow(sampler compareSampler, VSOutput input) {
            return compareShadow(compareSampler, input.uv, input.depth);
        }

        fragment {
            float main(VSOutput input) @ gl_FragDepth {
                return sampleShadow(shadowSampler, input);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "SamplerComparisonState shadowSampler : register(s0);" in generated_code
    assert (
        "float compareShadow(SamplerComparisonState compareSampler, float2 uv, float depth)"
        in generated_code
    )
    assert (
        "float sampleShadow(SamplerComparisonState compareSampler, VSOutput input)"
        in generated_code
    )
    assert "compareShadow(compareSampler, input.uv, input.depth)" in generated_code
    assert "sampleShadow(shadowSampler, input)" in generated_code


def test_directx_sampler_1d_array_sampling_and_queries():
    shader = """
    shader OneDArrayTexture {
        sampler1DArray lineArray;
        sampler linearSampler;

        vec4 sampleLineArray(
            sampler1DArray tex,
            sampler samp,
            vec2 uvLayer,
            ivec2 pixelLayer,
            int lod,
            int offset
        ) {
            ivec2 dims = textureSize(tex, lod);
            int levels = textureQueryLevels(tex);
            vec2 lodInfo = textureQueryLod(tex, samp, uvLayer);
            return texture(tex, samp, uvLayer)
                + textureLod(tex, samp, uvLayer, lod)
                + texelFetch(tex, pixelLayer, lod)
                + texelFetchOffset(tex, pixelLayer, lod, offset)
                + vec4(dims.x, dims.y, levels, lodInfo.x + lodInfo.y);
        }

        fragment {
            vec4 main() @gl_FragColor {
                return sampleLineArray(
                    lineArray,
                    linearSampler,
                    vec2(0.5, 0.0),
                    ivec2(4, 0),
                    0,
                    1
                );
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture1DArray lineArray : register(t0);" in generated_code
    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "int2 textureSize(Texture1DArray tex, int lod)" in generated_code
    assert "tex.GetDimensions(lod, width, elements, levels);" in generated_code
    assert "int textureQueryLevels(Texture1DArray tex)" in generated_code
    assert (
        "float4 sampleLineArray(Texture1DArray tex, SamplerState samp, float2 uvLayer, int2 pixelLayer, int lod, int offset)"
        in generated_code
    )
    assert (
        "float2 lodInfo = float2(tex.CalculateLevelOfDetailUnclamped(samp, uvLayer.x), tex.CalculateLevelOfDetail(samp, uvLayer.x));"
        in generated_code
    )
    assert "tex.Sample(samp, uvLayer)" in generated_code
    assert "tex.SampleLevel(samp, uvLayer, lod)" in generated_code
    assert "tex.Load(int3(pixelLayer, lod))" in generated_code
    assert (
        "tex.Load(int3((pixelLayer.x + offset), pixelLayer.y, lod))" in generated_code
    )
    assert "CalculateLevelOfDetailUnclamped(samp, uvLayer)" not in generated_code


def test_directx_image_1d_and_1d_array_storage_operations():
    shader = """
    shader OneDStorageImages {
        image1D line;
        image1DArray layers;
        uimage1D counters @r32ui;
        uimage1DArray layerCounters @r32ui;

        float touchLine(image1D image, int x, float value) {
            float oldValue = imageLoad(image, x);
            imageStore(image, x, oldValue + value);
            return oldValue;
        }

        vec4 touchLayer(image1DArray image, ivec2 coord, vec4 value) {
            vec4 oldValue = imageLoad(image, coord);
            imageStore(image, coord, oldValue + value);
            return oldValue;
        }

        uint addCounter(uimage1D image @r32ui, int x, uint value) {
            return imageAtomicAdd(image, x, value);
        }

        uint exchangeLayer(uimage1DArray image @r32ui, ivec2 coord, uint value) {
            return imageAtomicExchange(image, coord, value);
        }

        uint compareLayer(uimage1DArray image @r32ui, ivec2 coord, uint expected, uint value) {
            return imageAtomicCompSwap(image, coord, expected, value);
        }

        compute {
            void main() {
                int sizeLine = imageSize(line);
                ivec2 sizeLayer = imageSize(layers);
                float a = touchLine(line, 1, 0.25);
                vec4 b = touchLayer(layers, ivec2(2, 3), vec4(1.0));
                uint c = addCounter(counters, 4, 7u);
                uint d = exchangeLayer(layerCounters, ivec2(5, 6), 8u);
                uint e = compareLayer(layerCounters, ivec2(7, 8), d, c);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture1D<float4> line : register(u0);" in generated_code
    assert "RWTexture1DArray<float4> layers : register(u1);" in generated_code
    assert "RWTexture1D<uint> counters : register(u2);" in generated_code
    assert "RWTexture1DArray<uint> layerCounters : register(u3);" in generated_code
    assert "int imageSize(RWTexture1D<float4> image)" in generated_code
    assert "int2 imageSize(RWTexture1DArray<float4> image)" in generated_code
    assert "float oldValue = image[x].x;" in generated_code
    assert (
        "image[x] = float4((oldValue + value), (oldValue + value), "
        "(oldValue + value), (oldValue + value));" in generated_code
    )
    assert "float4 oldValue = image[coord];" in generated_code
    assert "image[coord] = (oldValue + value);" in generated_code
    assert (
        "uint imageAtomicAdd_uimage1D(RWTexture1D<uint> image, int coord, uint value)"
        in generated_code
    )
    assert (
        "uint imageAtomicExchange_uimage1DArray(RWTexture1DArray<uint> image, int2 coord, uint value)"
        in generated_code
    )
    assert (
        "uint imageAtomicCompSwap_uimage1DArray(RWTexture1DArray<uint> image, int2 coord, uint compareValue, uint value)"
        in generated_code
    )
    assert (
        "InterlockedCompareExchange(image[coord], compareValue, value, original);"
        in generated_code
    )
    assert (
        "return imageAtomicCompSwap_uimage1DArray(image, coord, expected, value);"
        in generated_code
    )


def test_directx_multisample_storage_images_emit_srv_reads_and_diagnostics():
    shader = """
    shader DirectXMultisampleStorageImages {
        image2DMS colorImage @rgba16f;
        uimage2DMS counters @r32ui;

        vec4 touch(image2DMS image @rgba16f, uimage2DMS counterImage @r32ui, ivec2 pixel, int sampleIndex, vec4 value, uint count) {
            vec4 oldColor = imageLoad(image, pixel, sampleIndex);
            uint oldCount = imageLoad(counterImage, pixel, sampleIndex);
            imageStore(image, pixel, sampleIndex, oldColor + value);
            imageStore(counterImage, pixel, sampleIndex, oldCount + count);
            uint atomicOld = imageAtomicAdd(counterImage, pixel, sampleIndex, count);
            return oldColor + vec4(float(oldCount + atomicOld));
        }

        fragment {
            vec4 main() @ gl_FragColor {
                return touch(colorImage, counters, ivec2(0), 1, vec4(1.0), 2u);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2DMS<float4> colorImage : register(t0);" in generated_code
    assert "Texture2DMS<uint> counters : register(t1);" in generated_code
    assert (
        "float4 touch(Texture2DMS<float4> image, Texture2DMS<uint> "
        "counterImage, int2 pixel, int sampleIndex, float4 value, uint count)"
        in generated_code
    )
    assert "float4 oldColor = image.Load(pixel, sampleIndex);" in generated_code
    assert "uint oldCount = counterImage.Load(pixel, sampleIndex);" in generated_code
    assert (
        "unsupported DirectX multisample image store: imageStore on "
        "RWTexture2DMS<float4>" in generated_code
    )
    assert (
        "unsupported DirectX multisample image store: imageStore on "
        "RWTexture2DMS<uint>" in generated_code
    )
    assert (
        "uint atomicOld = /* unsupported DirectX multisample image atomic: "
        "imageAtomicAdd on RWTexture2DMS<uint> */ 0u;" in generated_code
    )
    assert "RWTexture2DMS<float4> colorImage" not in generated_code
    assert "RWTexture2DMS<uint> counters" not in generated_code
    assert "RWTexture2DMS<float4> image" not in generated_code
    assert "RWTexture2DMS<uint> counterImage" not in generated_code
    assert "InterlockedAdd" not in generated_code


def test_directx_multisample_storage_image_arrays_emit_srv_reads_and_diagnostics():
    shader = """
    shader DirectXMultisampleStorageImageArrays {
        image2DMSArray layered @rgba16f;
        uimage2DMSArray counters @r32ui;

        vec4 touchLayer(
            image2DMSArray image @rgba16f,
            uimage2DMSArray counterImage @r32ui,
            ivec3 pixelLayer,
            int sampleIndex,
            vec4 value,
            uint count
        ) {
            vec4 oldColor = imageLoad(image, pixelLayer, sampleIndex);
            uint oldCount = imageLoad(counterImage, pixelLayer, sampleIndex);
            imageStore(image, pixelLayer, sampleIndex, oldColor + value);
            imageStore(counterImage, pixelLayer, sampleIndex, oldCount + count);
            uint atomicOld = imageAtomicAdd(counterImage, pixelLayer, sampleIndex, count);
            return oldColor + vec4(float(oldCount + atomicOld));
        }

        fragment {
            vec4 main() @ gl_FragColor {
                return touchLayer(layered, counters, ivec3(0, 1, 2), 3, vec4(1.0), 2u);
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "Texture2DMSArray<float4> layered : register(t0);" in generated_code
    assert "Texture2DMSArray<uint> counters : register(t1);" in generated_code
    assert (
        "float4 touchLayer(Texture2DMSArray<float4> image, "
        "Texture2DMSArray<uint> counterImage, int3 pixelLayer, "
        "int sampleIndex, float4 value, uint count)" in generated_code
    )
    assert "float4 oldColor = image.Load(pixelLayer, sampleIndex);" in generated_code
    assert (
        "uint oldCount = counterImage.Load(pixelLayer, sampleIndex);" in generated_code
    )
    diagnostic_prefix = "un" + "supported DirectX multisample image"
    assert (
        f"{diagnostic_prefix} store: imageStore on RWTexture2DMSArray<float4>"
        in generated_code
    )
    assert (
        f"{diagnostic_prefix} store: imageStore on RWTexture2DMSArray<uint>"
        in generated_code
    )
    assert (
        "uint atomicOld = /* "
        f"{diagnostic_prefix} atomic: imageAtomicAdd on "
        "RWTexture2DMSArray<uint> */ 0u;" in generated_code
    )
    assert "RWTexture2DMSArray<float4> layered" not in generated_code
    assert "RWTexture2DMSArray<uint> counters" not in generated_code
    assert "RWTexture2DMSArray<float4> image" not in generated_code
    assert "RWTexture2DMSArray<uint> counterImage" not in generated_code
    assert "InterlockedAdd" not in generated_code


def test_directx_uav_metadata_attributes_emit_coherency_and_register_space():
    shader = """
    shader UavMetadata {
        uimage2D counters @r32ui @globallycoherent @register(u4, space2);
        image2D outImage @register(u5, space2);

        compute {
            void main() {
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert (
        "globallycoherent RWTexture2D<uint> counters : register(u4, space2);"
        in generated_code
    )
    assert "RWTexture2D<float4> outImage : register(u5, space2);" in generated_code


def test_directx_generic_image_memory_attributes_do_not_become_semantics():
    shader = """
    shader UavMemoryAttrs {
        uimage2D counters @r32ui @coherent @register(u4, space2);

        uint readCounter(uimage2D image @r32ui @coherent, ivec2 pixel) {
            return imageLoad(image, pixel);
        }

        compute {
            void main() {
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert (
        "globallycoherent RWTexture2D<uint> counters : register(u4, space2);"
        in generated_code
    )
    assert "uint readCounter(RWTexture2D<uint> image, int2 pixel)" in generated_code
    assert ": coherent" not in generated_code


def test_directx_storage_image_access_attributes_validate_without_semantics():
    shader = """
    shader UavAccessAttrs {
        image2D outImage @writeonly;

        float readPixel(image2D image @readonly, ivec2 pixel) {
            return imageLoad(image, pixel);
        }

        void writePixel(image2D image @writeonly, ivec2 pixel, vec4 value) {
            imageStore(image, pixel, value);
        }

        compute {
            void main() {
                imageStore(outImage, ivec2(0, 0), vec4(1.0));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "RWTexture2D<float4> outImage : register(u0);" in generated_code
    assert "float readPixel(RWTexture2D<float4> image, int2 pixel)" in generated_code
    assert (
        "void writePixel(RWTexture2D<float4> image, int2 pixel, float4 value)"
        in generated_code
    )
    assert ": readonly" not in generated_code
    assert ": writeonly" not in generated_code


@pytest.mark.parametrize(
    ("shader", "match"),
    [
        (
            """
            shader UavAccessInvalidLoad {
                float readPixel(image2D image @writeonly, ivec2 pixel) {
                    return imageLoad(image, pixel);
                }

                compute {
                    void main() {
                    }
                }
            }
            """,
            "requires read-capable storage image access",
        ),
        (
            """
            shader UavAccessInvalidStore {
                void writePixel(image2D image @readonly, ivec2 pixel, vec4 value) {
                    imageStore(image, pixel, value);
                }

                compute {
                    void main() {
                    }
                }
            }
            """,
            "requires write-capable storage image access",
        ),
        (
            """
            shader UavAccessInvalidAtomic {
                uint addCounter(uimage2D image @r32ui @readonly, ivec2 pixel, uint value) {
                    return imageAtomicAdd(image, pixel, value);
                }

                compute {
                    void main() {
                    }
                }
            }
            """,
            "requires read-write storage image access",
        ),
        (
            """
            shader UavAccessInvalidGlobal {
                image2D outImage @writeonly;

                compute {
                    void main() {
                        float value = imageLoad(outImage, ivec2(0, 0));
                    }
                }
            }
            """,
            "requires read-capable storage image access",
        ),
    ],
)
def test_directx_storage_image_access_rejects_invalid_operations(shader, match):
    ast = crosstl.translator.parse(shader)

    with pytest.raises(ValueError, match=match):
        HLSLCodeGen().generate(ast)


def test_directx_storage_image_access_allows_compatible_helper_calls():
    shader = """
    shader UavAccessHelperValid {
        image2D source @readonly;
        image2D target @writeonly;

        float readPixel(image2D image, ivec2 pixel) {
            return imageLoad(image, pixel);
        }

        void writePixel(image2D image, ivec2 pixel, vec4 value) {
            imageStore(image, pixel, value);
        }

        compute {
            void main() {
                float value = readPixel(source, ivec2(0, 0));
                writePixel(target, ivec2(0, 0), vec4(value));
            }
        }
    }
    """

    ast = crosstl.translator.parse(shader)
    generated_code = HLSLCodeGen().generate(ast)

    assert "float value = readPixel(source, int2(0, 0));" in generated_code
    assert (
        "writePixel(target, int2(0, 0), float4(value, value, value, value));"
        in generated_code
    )


@pytest.mark.parametrize(
    ("shader", "match"),
    [
        (
            """
            shader UavAccessHelperInvalidRead {
                image2D target @writeonly;

                float readPixel(image2D image, ivec2 pixel) {
                    return imageLoad(image, pixel);
                }

                compute {
                    void main() {
                        float value = readPixel(target, ivec2(0, 0));
                    }
                }
            }
            """,
            "function call 'readPixel' requires read-capable storage image access",
        ),
        (
            """
            shader UavAccessHelperInvalidWrite {
                image2D source @readonly;

                void writePixel(image2D image, ivec2 pixel, vec4 value) {
                    imageStore(image, pixel, value);
                }

                compute {
                    void main() {
                        writePixel(source, ivec2(0, 0), vec4(1.0));
                    }
                }
            }
            """,
            "function call 'writePixel' requires write-capable storage image access",
        ),
        (
            """
            shader UavAccessHelperInvalidAtomic {
                uimage2D source @r32ui @readonly;

                uint addCounter(uimage2D image @r32ui, ivec2 pixel, uint value) {
                    return imageAtomicAdd(image, pixel, value);
                }

                compute {
                    void main() {
                        uint value = addCounter(source, ivec2(0, 0), 1u);
                    }
                }
            }
            """,
            "function call 'addCounter' requires read-write storage image access",
        ),
        (
            """
            shader UavAccessHelperInvalidTransitive {
                image2D target @writeonly;

                float leaf(image2D image, ivec2 pixel) {
                    return imageLoad(image, pixel);
                }

                float mid(image2D image, ivec2 pixel) {
                    return leaf(image, pixel);
                }

                compute {
                    void main() {
                        float value = mid(target, ivec2(0, 0));
                    }
                }
            }
            """,
            "function call 'mid' requires read-capable storage image access",
        ),
    ],
)
def test_directx_storage_image_access_rejects_incompatible_helper_calls(shader, match):
    ast = crosstl.translator.parse(shader)

    with pytest.raises(ValueError, match=match):
        HLSLCodeGen().generate(ast)


def test_directx_feedback_texture_helpers_lower_to_native_methods():
    shader = """
    shader FeedbackTextureWrites {
        sampler2D pairedTexture @ register(t0, space10);
        sampler2DArray pairedLayers @ register(t1, space10);
        sampler pairedSampler @ register(s0, space10);
        feedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMin @ register(u0, space10);
        feedbackTexture2DArray<SAMPLER_FEEDBACK_MIP_REGION_USED> feedbackUsed @ register(u1, space10);

        fragment {
            void main(vec2 uv @ TexCoord0, float layer @ TexCoord1) {
                vec2 ddxValue = vec2(0.25, 0.0);
                vec2 ddyValue = vec2(0.0, 0.25);
                vec3 uvLayer = vec3(uv, layer);
                write_sampler_feedback(feedbackMin, pairedTexture, pairedSampler, uv);
                write_sampler_feedback_bias(feedbackMin, pairedTexture, pairedSampler, uv, 0.5);
                write_sampler_feedback_grad(feedbackMin, pairedTexture, pairedSampler, uv, ddxValue, ddyValue);
                write_sampler_feedback_level(feedbackMin, pairedTexture, pairedSampler, uv, 2.0);
                write_sampler_feedback_level(feedbackUsed, pairedLayers, pairedSampler, uvLayer, 1.0);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert (
        "FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMin : "
        "register(u0, space10);" in generated_code
    )
    assert (
        "FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIP_REGION_USED> feedbackUsed : "
        "register(u1, space10);" in generated_code
    )
    assert (
        "feedbackMin.WriteSamplerFeedback(pairedTexture, pairedSampler, uv);"
        in generated_code
    )
    assert (
        "feedbackMin.WriteSamplerFeedbackBias("
        "pairedTexture, pairedSampler, uv, 0.5);" in generated_code
    )
    assert (
        "feedbackMin.WriteSamplerFeedbackGrad("
        "pairedTexture, pairedSampler, uv, ddxValue, ddyValue);" in generated_code
    )
    assert (
        "feedbackMin.WriteSamplerFeedbackLevel("
        "pairedTexture, pairedSampler, uv, 2.0);" in generated_code
    )
    assert (
        "feedbackUsed.WriteSamplerFeedbackLevel("
        "pairedLayers, pairedSampler, uvLayer, 1.0);" in generated_code
    )
    assert "write_sampler_feedback" not in generated_code


def test_directx_feedback_texture_grad_and_level_helpers_are_compute_compatible():
    shader = """
    shader FeedbackTextureComputeWrites {
        sampler2D pairedTexture @ register(t0, space11);
        sampler pairedSampler @ register(s0, space11);
        feedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMin @ register(u0, space11);

        compute {
            void main() {
                vec2 uv = vec2(0.25, 0.75);
                vec2 ddxValue = vec2(0.25, 0.0);
                vec2 ddyValue = vec2(0.0, 0.25);
                write_sampler_feedback_grad(feedbackMin, pairedTexture, pairedSampler, uv, ddxValue, ddyValue);
                write_sampler_feedback_level(feedbackMin, pairedTexture, pairedSampler, uv, 2.0);
            }
        }
    }
    """

    generated_code = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "[numthreads(1, 1, 1)]" in generated_code
    assert (
        "FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMin : "
        "register(u0, space11);" in generated_code
    )
    assert (
        "feedbackMin.WriteSamplerFeedbackGrad("
        "pairedTexture, pairedSampler, uv, ddxValue, ddyValue);" in generated_code
    )
    assert (
        "feedbackMin.WriteSamplerFeedbackLevel("
        "pairedTexture, pairedSampler, uv, 2.0);" in generated_code
    )
    assert "write_sampler_feedback" not in generated_code


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            "write_sampler_feedback(feedbackMin, pairedTexture, pairedSampler, uv);",
            "DirectX compute stage cannot call write_sampler_feedback; "
            "WriteSamplerFeedback is only valid in fragment/pixel stages",
        ),
        (
            "write_sampler_feedback_bias(feedbackMin, pairedTexture, pairedSampler, uv, 0.5);",
            "DirectX compute stage cannot call write_sampler_feedback_bias; "
            "WriteSamplerFeedbackBias is only valid in fragment/pixel stages",
        ),
    ],
)
def test_directx_feedback_texture_pixel_only_helpers_reject_compute_stage(body, match):
    shader = f"""
    shader InvalidComputeFeedbackTextureWrites {{
        sampler2D pairedTexture;
        sampler pairedSampler;
        feedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMin;

        compute {{
            void main() {{
                vec2 uv = vec2(0.25, 0.75);
                {body}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=match):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_directx_feedback_texture_pixel_only_helpers_reject_transitive_compute_call():
    shader = """
    shader InvalidTransitiveComputeFeedbackTextureWrite {
        sampler2D pairedTexture;
        sampler pairedSampler;
        feedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMin;

        void recordFeedback(vec2 uv) {
            write_sampler_feedback(feedbackMin, pairedTexture, pairedSampler, uv);
        }

        compute {
            void main() {
                recordFeedback(vec2(0.25, 0.75));
            }
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=(
            "DirectX compute stage cannot call recordFeedback; "
            "'recordFeedback' reaches WriteSamplerFeedback via "
            "write_sampler_feedback, which is only valid in fragment/pixel stages"
        ),
    ):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            "write_sampler_feedback(pairedTexture, pairedTexture, pairedSampler, uv);",
            "requires FeedbackTexture2D or FeedbackTexture2DArray receiver",
        ),
        (
            "write_sampler_feedback(feedbackMin, pairedLayers, pairedSampler, uv);",
            "receiver FeedbackTexture2D requires paired Texture2D, got Texture2DArray",
        ),
        (
            "write_sampler_feedback(feedbackMin, pairedTexture, pairedSampler, uvLayer);",
            "location argument must be float2",
        ),
        (
            "write_sampler_feedback_grad(feedbackMin, pairedTexture, pairedSampler, uv, ddxLayer, ddyLayer);",
            "ddx argument must be float2",
        ),
        (
            "write_sampler_feedback_bias(feedbackMin, pairedTexture, pairedSampler, uv);",
            "requires 5 or 6 argument",
        ),
    ],
)
def test_directx_feedback_texture_helpers_validate_resources_and_shapes(body, match):
    shader = f"""
    shader InvalidFeedbackTextureWrites {{
        sampler2D pairedTexture;
        sampler2DArray pairedLayers;
        sampler pairedSampler;
        feedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMin;

        fragment {{
            void main(vec2 uv @ TexCoord0, float layer @ TexCoord1) {{
                vec3 uvLayer = vec3(uv, layer);
                vec3 ddxLayer = vec3(0.25, 0.0, 0.0);
                vec3 ddyLayer = vec3(0.0, 0.25, 0.0);
                {body}
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=match):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


if __name__ == "__main__":
    pytest.main()

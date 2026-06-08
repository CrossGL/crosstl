from __future__ import annotations

import textwrap
from dataclasses import dataclass

import pytest

from crosstl.backend.DirectX.DirectxCrossGLCodeGen import HLSLToCrossGLConverter
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser


@dataclass(frozen=True)
class ExternalFixture:
    name: str
    repo: str
    commit: str
    path: str
    code: str
    contains: tuple[str, ...]

    @property
    def source_url(self):
        return f"{self.repo}/blob/{self.commit}/{self.path}"


DIRECTX_GRAPHICS_SAMPLES_REPO = "https://github.com/microsoft/DirectX-Graphics-Samples"
DIRECTX_GRAPHICS_SAMPLES_COMMIT = "31ae3c91160d8634264004cdaf4e41a99c41243e"
DIRECTX_SHADER_COMPILER_REPO = "https://github.com/microsoft/DirectXShaderCompiler"
DIRECTX_SHADER_COMPILER_COMMIT = "517dd5eb5d8cbb46c15fc1230acac1d2f4779092"
DIRECTX_SHADER_COMPILER_NESTED_ENUM_COMMIT = "8ed708842c1ccb24bd914eff03125c837a01be71"
DIRECTX_SHADER_COMPILER_FLOAT64_COMMIT = "8ed708842c1ccb24bd914eff03125c837a01be71"
DIRECTX_SHADER_COMPILER_BUFFER_ACCESS_COMMIT = (
    "8ed708842c1ccb24bd914eff03125c837a01be71"
)
DIRECTX_SHADER_COMPILER_CONTEXTUAL_RESOURCE_COMMIT = (
    "8ed708842c1ccb24bd914eff03125c837a01be71"
)
DIRECTX_SHADER_COMPILER_REWRITER_COMMIT = "8ed708842c1ccb24bd914eff03125c837a01be71"
DIRECTX_SHADER_COMPILER_BINARY_OP_SUGAR_COMMIT = (
    "8ed708842c1ccb24bd914eff03125c837a01be71"
)
DIRECTX_SHADER_COMPILER_NAMESPACE_COMMIT = "8ed708842c1ccb24bd914eff03125c837a01be71"
FIDELITYFX_FSR_REPO = "https://github.com/GPUOpen-Effects/FidelityFX-FSR"
FIDELITYFX_FSR_COMMIT = "a21ffb8f6c13233ba336352bdff293894c706575"
FIDELITYFX_SDK_REPO = "https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK"
FIDELITYFX_SDK_COMMIT = "e236f2304dcda35f282fdddd085f41e2ff48c86a"
WICKED_ENGINE_REPO = "https://github.com/turanszkij/WickedEngine"
WICKED_ENGINE_COMMIT = "9df7a530aed53cc59b345f751939e513170ddf3c"


EXTERNAL_FIXTURES = [
    ExternalFixture(
        name="directx_graphics_samples_hello_triangle",
        repo=DIRECTX_GRAPHICS_SAMPLES_REPO,
        commit=DIRECTX_GRAPHICS_SAMPLES_COMMIT,
        path="Samples/Desktop/D3D12HelloWorld/src/HelloTriangle/shaders.hlsl",
        code=textwrap.dedent("""
            struct PSInput
            {
                float4 position : SV_POSITION;
                float4 color : COLOR;
            };

            PSInput VSMain(float4 position : POSITION, float4 color : COLOR)
            {
                PSInput result;

                result.position = position;
                result.color = color;

                return result;
            }

            float4 PSMain(PSInput input) : SV_TARGET
            {
                return input.color;
            }
        """).strip(),
        contains=(
            "vec4 position @ gl_Position",
            "vec4 PSMain(PSInput input) @ gl_FragColor",
        ),
    ),
    ExternalFixture(
        name="directx_graphics_samples_hello_texture_sample",
        repo=DIRECTX_GRAPHICS_SAMPLES_REPO,
        commit=DIRECTX_GRAPHICS_SAMPLES_COMMIT,
        path="Samples/Desktop/D3D12HelloWorld/src/HelloTexture/shaders.hlsl",
        code=textwrap.dedent("""
            struct PSInput
            {
                float4 position : SV_POSITION;
                float2 uv : TEXCOORD;
            };

            Texture2D g_texture : register(t0);
            SamplerState g_sampler : register(s0);

            PSInput VSMain(float4 position : POSITION, float2 uv : TEXCOORD)
            {
                PSInput result;

                result.position = position;
                result.uv = uv;

                return result;
            }

            float4 PSMain(PSInput input) : SV_TARGET
            {
                return g_texture.Sample(g_sampler, input.uv);
            }
        """).strip(),
        contains=(
            "sampler2D g_texture;",
            "sampler g_sampler;",
            "vec2 uv @ TexCoord;",
            "return texture(g_texture, g_sampler, input.uv);",
        ),
    ),
    ExternalFixture(
        name="directx_graphics_samples_miniengine_present_sdr",
        repo=DIRECTX_GRAPHICS_SAMPLES_REPO,
        commit=DIRECTX_GRAPHICS_SAMPLES_COMMIT,
        path="MiniEngine/Core/Shaders/PresentSDRPS.hlsl",
        code=textwrap.dedent("""
            #include "ShaderUtility.hlsli"
            #include "PresentRS.hlsli"

            Texture2D<float3> ColorTex : register(t0);

            [RootSignature(Present_RootSig)]
            float3 main(float4 position : SV_Position) : SV_Target0
            {
                float3 LinearRGB = ColorTex[(int2)position.xy];
                return ApplyDisplayProfile(LinearRGB, DISPLAY_PLANE_FORMAT);
            }
        """).strip(),
        contains=(
            "@ RootSignature(Present_RootSig)",
            "sampler2D ColorTex;",
            "vec3 LinearRGB = texelFetch(ColorTex, ivec2(position.xy), 0);",
        ),
    ),
    ExternalFixture(
        name="directx_graphics_samples_miniengine_screen_quad_common_vs",
        repo=DIRECTX_GRAPHICS_SAMPLES_REPO,
        commit=DIRECTX_GRAPHICS_SAMPLES_COMMIT,
        path="MiniEngine/Core/Shaders/ScreenQuadCommonVS.hlsl",
        code=textwrap.dedent("""
            #include "CommonRS.hlsli"

            [RootSignature(Common_RootSig)]
            void main(
                in uint VertID : SV_VertexID,
                out float4 Pos : SV_Position,
                out float2 Tex : TexCoord0
            )
            {
                Tex = float2(uint2(VertID, VertID << 1) & 2);
                Pos = float4(lerp(float2(-1, 1), float2(1, -1), Tex), 0, 1);
            }
        """).strip(),
        contains=(
            "vertex {",
            "uint VertID @ gl_VertexID",
            "out vec4 Pos @ gl_Position",
            "Tex = vec2(uvec2(VertID, VertID << 1) & 2);",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_groupshared_splat",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenHLSL/SplatGroupSharedScalar.hlsl",
        code=textwrap.dedent("""
            groupshared int a;
            [numthreads(64, 1, 1)]
            void main() {
              a = 123;
              int4 x = (a).xxxx;
            }
        """).strip(),
        contains=(
            "groupshared int a;",
            "@ numthreads(64, 1, 1)",
            "ivec4 x = a.xxxx;",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_groupshared_array_parameter_reserved_fn",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenHLSL/groupsharedArgs/ArrTest.hlsl",
        code=textwrap.dedent("""
            groupshared float4 SharedArr[64];

            void fn(groupshared float4 Arr[64], float F) {
              float4 tmp = F.xxxx;
              Arr[5] = tmp;
            }

            [numthreads(4,1,1)]
            void main() {
              fn(SharedArr, 6.0);
            }
        """).strip(),
        contains=(
            "groupshared vec4 SharedArr[64];",
            "void fn_(vec4 Arr[64], float F)",
            "fn_(SharedArr, 6.0);",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_precise_struct_member",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenHLSL/precise/precise_gvn.hlsl",
        code=textwrap.dedent("""
            struct VSIn
            {
                float4 Pos : P;
                float4 A   : A;
            };

            struct VSOut
            {
                precise float4 Pos : SV_Position;
                float4 N : A;
            };

            [RootSignature("")]
            VSOut main(VSIn input)
            {
                float4 X  = input.A * input.A;
                float4 Y  = input.A + input.A;
                float4 R1 = mul(X, Y);
                float4 R2 = mul(X, Y);

                VSOut O;
                O.Pos = R1 * R1;
                O.N   = R2;
                return O;
            }
        """).strip(),
        contains=(
            "precise vec4 Pos @ gl_Position;",
            '@ RootSignature("")',
            "return O;",
        ),
    ),
    ExternalFixture(
        name="directx_graphics_samples_emulated_pointer_reserved_buffer_identifier",
        repo=DIRECTX_GRAPHICS_SAMPLES_REPO,
        commit=DIRECTX_GRAPHICS_SAMPLES_COMMIT,
        path="Libraries/D3D12RaytracingFallback/src/EmulatedPointer.hlsli",
        code=textwrap.dedent("""
            struct RWByteAddressBufferPointer
            {
                RWByteAddressBuffer buffer;
                uint offsetInBytes;
            };

            static
            RWByteAddressBufferPointer CreateRWByteAddressBufferPointer(in RWByteAddressBuffer buffer, uint offsetInBytes)
            {
                RWByteAddressBufferPointer pointer;
                pointer.buffer = buffer;
                pointer.offsetInBytes = offsetInBytes;
                return pointer;
            }
        """).strip(),
        contains=(
            "RWByteAddressBuffer buffer;",
            "RWByteAddressBufferPointer CreateRWByteAddressBufferPointer(in RWByteAddressBuffer buffer_, uint offsetInBytes)",
            "pointer.buffer = buffer_;",
        ),
    ),
    ExternalFixture(
        name="directx_graphics_samples_build_bvh_splits_unsigned_int_parameter",
        repo=DIRECTX_GRAPHICS_SAMPLES_REPO,
        commit=DIRECTX_GRAPHICS_SAMPLES_COMMIT,
        path="Libraries/D3D12RaytracingFallback/src/BuildBVHSplits.hlsli",
        code=textwrap.dedent("""
            uint2 DetermineRange(uint idx)
            {
                return uint2(idx, idx);
            }

            void GenerateHierarchy(unsigned int idx)
            {
                uint2 range = DetermineRange(idx);
                uint first = range.x;
            }
        """).strip(),
        contains=(
            "uvec2 DetermineRange(uint idx)",
            "void GenerateHierarchy(uint idx)",
            "uvec2 range = DetermineRange(idx);",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_nbody_gravity_compute",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenHLSL/Samples/D12/d12_nBodyGravityCS.hlsl",
        code=textwrap.dedent("""
            static float softeningSquared = 0.0012500000f * 0.0012500000f;
            static float g_fG = 6.67300e-11f * 10000.0f;
            static float g_fParticleMass = g_fG * 10000.0f * 10000.0f;

            #define blocksize 128
            groupshared float4 sharedPos[blocksize];

            void bodyBodyInteraction(inout float3 ai, float4 bj, float4 bi, float mass, int particles)
            {
                float3 r = bj.xyz - bi.xyz;
                float distSqr = dot(r, r);
                distSqr += softeningSquared;

                float invDist = 1.0f / sqrt(distSqr);
                float invDistCube = invDist * invDist * invDist;
                float s = mass * invDistCube * particles;

                ai += r * s;
            }

            cbuffer cbCS : register(b0)
            {
                uint4 g_param;
                float4 g_paramf;
            };

            struct PosVelo
            {
                float4 pos;
                float4 velo;
            };

            StructuredBuffer<PosVelo> oldPosVelo : register(t0);
            RWStructuredBuffer<PosVelo> newPosVelo : register(u0);

            [numthreads(blocksize, 1, 1)]
            void main(uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex)
            {
                float4 pos = oldPosVelo[DTid.x].pos;
                float4 vel = oldPosVelo[DTid.x].velo;
                float3 accel = 0;
                float mass = g_fParticleMass;

                [loop]
                for (uint tile = 0; tile < g_param.y; tile++)
                {
                    sharedPos[GI] = oldPosVelo[tile * blocksize + GI].pos;
                    GroupMemoryBarrierWithGroupSync();

                    [unroll]
                    for (uint counter = 0; counter < blocksize; counter += 8)
                    {
                        bodyBodyInteraction(accel, sharedPos[counter], pos, mass, 1);
                    }

                    GroupMemoryBarrierWithGroupSync();
                }

                const int tooManyParticles = g_param.y * blocksize - g_param.x;
                bodyBodyInteraction(accel, float4(0, 0, 0, 0), pos, mass, -tooManyParticles);

                vel.xyz += accel.xyz * g_paramf.x;
                vel.xyz *= g_paramf.y;
                pos.xyz += vel.xyz * g_paramf.x;

                if (DTid.x < g_param.x)
                {
                    newPosVelo[DTid.x].pos = pos;
                    newPosVelo[DTid.x].velo = float4(vel.xyz, length(accel));
                }
            }
        """).strip(),
        contains=(
            "static float softeningSquared = 0.00125 * 0.00125;",
            "groupshared vec4 sharedPos[128];",
            "void bodyBodyInteraction(inout vec3 ai, vec4 bj, vec4 bi, float mass, int particles)",
            "@ numthreads(128, 1, 1)",
            "void main(uvec3 Gid @ gl_WorkGroupID, uvec3 DTid @ gl_GlobalInvocationID, uvec3 GTid @ gl_LocalInvocationID, uint GI @ gl_LocalInvocationIndex)",
            "workgroupBarrier();",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_texture_cube_dimensions_lod",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenHLSL/GetDimCalcLOD.hlsl",
        code=textwrap.dedent("""
            TextureCube<float4> cube;

            SamplerState    g_sam;

            float4 main(float2 uv : UV) : SV_TARGET
            {
                uint w;
                uint h;

                cube.GetDimensions(w,h);
                float lod = cube.CalculateLevelOfDetail(g_sam, float3(uv,1));
                return float4(w, h, lod, 1.0);
            }
        """).strip(),
        contains=(
            "samplerCube cube;",
            "w = uint(textureSize(cube, 0).x);",
            "float lod = textureQueryLod(cube, g_sam, vec3(uv, 1)).x;",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_local_function_prototype",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenSPIRV/fn.forward-declaration.hlsl",
        code=textwrap.dedent("""
            float4 main() : SV_Target
            {
              float MulBy2(float f);
              return float4(MulBy2(.25), 1, 0, 1);
            }

            float MulBy2(float f) {
              return f*2;
            }
        """).strip(),
        contains=(
            "return vec4(MulBy2(0.25), 1, 0, 1);",
            "float MulBy2(float f)",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_global_scope_enum_type",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenSPIRV/type.enum.hlsl",
        code=textwrap.dedent("""
            enum Number {
              First,
              Second,
              Third = 3,
              Fourth = -1,
            };

            static ::Number a = Second;

            [numthreads(1, 1, 1)]
            void main() {
              static ::Number e = Fourth;
              Number d;
              d = Number::Third;
            }
        """).strip(),
        contains=(
            "enum Number {",
            "static Number a = Second;",
            "@ numthreads(1, 1, 1)",
            "static Number e = Fourth;",
            "d = Number::Third;",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_const_external_enum_alias",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_BINARY_OP_SUGAR_COMMIT,
        path="tools/clang/test/CodeGenSPIRV/binary-op-sugar.hlsl",
        code=textwrap.dedent("""
            enum LuminairePlanesSymmetry : uint
            {
                ISOTROPIC = 0u,
                NO_LATERAL_SYMMET
            };

            using symmetry_t = LuminairePlanesSymmetry;
            const symmetry_t symmetry;

            [numthreads(1,1,1)]
            [shader("compute")]
            void main(uint3 id : SV_DispatchThreadID)
            {
              const uint i0 = (symmetry == symmetry_t::ISOTROPIC) ? 0u : 1u;
            }
        """).strip(),
        contains=(
            "type symmetry_t = LuminairePlanesSymmetry;",
            "symmetry_t symmetry;",
            "const uint i0 = symmetry == symmetry_t::ISOTROPIC ? 0 : 1;",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_struct_nested_enum",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_NESTED_ENUM_COMMIT,
        path="tools/clang/test/CodeGenSPIRV/class.enum.hlsl",
        code=textwrap.dedent("""
            struct TestStruct {
                enum EnumInTestStruct {
                    A = 1,
                };
            };

            TestStruct::EnumInTestStruct testFunc() {
                return TestStruct::A;
            }

            uint PSMain() : SV_TARGET
            {
                TestStruct::EnumInTestStruct i = testFunc();
                return i;
            }
        """).strip(),
        contains=(
            "struct TestStruct {",
            "enum EnumInTestStruct {",
            "TestStruct_EnumInTestStruct testFunc()",
            "return TestStruct::A;",
            "TestStruct_EnumInTestStruct i = testFunc();",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_static_member_definition",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenSPIRV/static_member_var.hlsl",
        code=textwrap.dedent("""
            struct PSInput
            {
                static const uint Val;
                uint Func() { return Val; }
            };

            static const uint PSInput::Val = 3;

            uint PSMain(PSInput input) : SV_Target0
            {
                return input.Func();
            }
        """).strip(),
        contains=(
            "static const uint PSInput_Val = 3;",
            "uint PSMain(PSInput input) @ gl_FragData[0]",
            "return input.Func();",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_native_16bit_scalar_types",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenSPIRV/constant.scalar.16bit.enabled.hlsl",
        code=textwrap.dedent("""
            void main() {
              float16_t c_float16t = 1.5;
              uint16_t c_uint16_16 = 16;
              int16_t c_int16_n16 = -16;
            }
        """).strip(),
        contains=(
            "float16 c_float16t = 1.5;",
            "uint16 c_uint16_16 = 16;",
            "int16 c_int16_n16 = -16;",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_const_eval_numeric_scalar_swizzle",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/HLSLFileCheck/hlsl/types/vector/vec_elem_const_eval.hlsl",
        code=textwrap.dedent("""
            static const float4 a = (1.5).xxxx;
            static const float4 b = (2).xxxx;
            static const int4   c = (3.5).xxxx;

            float4 main() : SV_Target {
              return a + b + c;
            }
        """).strip(),
        contains=(
            "static const vec4 a = (1.5).xxxx;",
            "static const vec4 b = (2).xxxx;",
            "static const ivec4 c = (3.5).xxxx;",
            "return (a + b) + c;",
        ),
    ),
    # Source repo: https://github.com/microsoft/DirectXShaderCompiler
    # Source commit: 8ed708842c1ccb24bd914eff03125c837a01be71
    # Source path: tools/clang/test/CodeGenSPIRV/constant.scalar.64bit.hlsl
    ExternalFixture(
        name="directx_shader_compiler_float64_scalar_alias",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_FLOAT64_COMMIT,
        path="tools/clang/test/CodeGenSPIRV/constant.scalar.64bit.hlsl",
        code=textwrap.dedent("""
            void main() {
              float64_t c_double_4_5 = 4.5;
            }
        """).strip(),
        contains=("double c_double_4_5 = 4.5;",),
    ),
    ExternalFixture(
        name="directx_shader_compiler_fixed_width_vector_typedef",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenSPIRV/initializelist.undefined.hlsl",
        code=textwrap.dedent("""
            typedef vector<uint32_t,3> uint32_t3;
            typedef vector<uint16_t,3> uint16_t3;
            uint32_t3 gl_WorkGroupSize();

            [numthreads(1, 1, 1)]
            void main() {
              const uint16_t3 dims = uint16_t3(gl_WorkGroupSize());
            }
        """).strip(),
        contains=(
            "type uint32_t3 = uvec3;",
            "type uint16_t3 = u16vec3;",
            "@ numthreads(1, 1, 1)",
            "u16vec3 dims = u16vec3(gl_WorkGroupSize());",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_layout_qualified_matrix_casts",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenSPIRV/cast.vec-to-mat.explicit.hlsl",
        code=textwrap.dedent("""
            float4 main(float4 input : A) : SV_Target
            {
                float2x2 baseMatrix = (float2x2)input;
                float2x2 rowMajorMatrix = (row_major float2x2)input;
                float2x2 columnMajorMatrix = (column_major float2x2)input;

                return float4(
                    baseMatrix[0][0],
                    rowMajorMatrix[0][1],
                    columnMajorMatrix[1][0],
                    baseMatrix[1][1]);
            }
        """).strip(),
        contains=(
            "mat2 baseMatrix = mat2(input);",
            "mat2 rowMajorMatrix = mat2(input);",
            "mat2 columnMajorMatrix = mat2(input);",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_struct_bitfield_members",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/HLSLFileCheck/hlsl/operators/swizzle/swizzleBitfield.hlsl",
        code=textwrap.dedent("""
            struct MyStruct
            {
                uint v0: 5;
                uint v1: 15;
                uint v2: 12;
            };

            ConstantBuffer<MyStruct> myConstBuff;
            StructuredBuffer<MyStruct> myStructBuff;

            [RootSignature("RootFlags(0), DescriptorTable(CBV(b0), SRV(t0))")]
            float4 main(uint addr: TEXCOORD): SV_Target
            {
                float4 temp1 = myStructBuff[0].v2.xxxx;
                float4 temp2 = myConstBuff.v2.xxxx;
                return temp1 + temp2;
            }
        """).strip(),
        contains=(
            "uint v0;",
            "uint v1;",
            "uint v2;",
            "ConstantBuffer<MyStruct> myConstBuff;",
            "StructuredBuffer<MyStruct> myStructBuff;",
            "vec4 temp1 = myStructBuff[0].v2.xxxx;",
            "vec4 temp2 = myConstBuff.v2.xxxx;",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_block_scope_using_linalg_aliases",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenDXIL/hlsl/linalg/api/matrix-multiply.hlsl",
        code=textwrap.dedent("""
            #include <dx/linalg.h>
            using namespace dx::linalg;

            [numthreads(4, 4, 4)]
            void main()
            {
              using MatrixAF16WTy = Matrix<ComponentType::F16, 3, 4, MatrixUse::A, MatrixScope::Wave>;
              using MatrixBI32WTy = Matrix<ComponentType::I32, 4, 5, MatrixUse::B, MatrixScope::Wave>;
              using MatrixAccF32WTy = Matrix<ComponentType::F32, 3, 5, MatrixUse::Accumulator, MatrixScope::Wave>;

              MatrixAF16WTy MatA1 = MatrixAF16WTy::Splat(1.5f);
              MatrixBI32WTy MatB1 = MatrixBI32WTy::Splat(13);
              MatrixAccF32WTy MatCFlt1 = Multiply<ComponentType::F32>(MatA1, MatB1);
            }
        """).strip(),
        contains=(
            "@ numthreads(4, 4, 4)",
            "MatrixAF16WTy MatA1 = MatrixAF16WTy::Splat(1.5);",
            "MatrixAccF32WTy MatCFlt1 = Multiply<ComponentType::F32>(MatA1, MatB1);",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_block_scope_typedef_linalg_vector",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenDXIL/hlsl/linalg/api/vectors.hlsl",
        code=textwrap.dedent("""
            #include <dx/linalg.h>
            using namespace dx::linalg;

            ByteAddressBuffer BAB : register(t0);

            [numthreads(4, 4, 4)]
            void main(uint ID : SV_GroupID) {
              typedef vector<half, 16> half16;
              half16 srcF16 = BAB.Load<half16>(128);
              InterpretedVector<uint, 4, ComponentEnum::F8_E4M3FN> convertedPacked =
                  Convert<ComponentEnum::F8_E4M3FN, ComponentEnum::F16>(srcF16);
            }
        """).strip(),
        contains=(
            "half16 srcF16 = buffer_load(BAB, 128);",
            "convertedPacked = Convert<ComponentEnum::F8_E4M3FN, ComponentEnum::F16>(srcF16);",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_native_16bit_vector_aliases",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenSPIRV/intrinsics.asuint16.hlsl",
        code=textwrap.dedent("""
            void main() {
              uint16_t4 result4;
              int16_t4 d;
              uint16_t4 e;
              float16_t4 f;
              float16_t2x3 floatMat;

              result4 = asuint16(d);
              result4 = asuint16(e);
              result4 = asuint16(f);
            }
        """).strip(),
        contains=(
            "u16vec4 result4;",
            "i16vec4 d;",
            "u16vec4 e;",
            "f16vec4 f;",
            "f16mat2x3 floatMat;",
            "result4 = asuint16(d);",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_vector_size_one_initializer",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenSPIRV/var.init.hlsl",
        code=textwrap.dedent("""
            float4 main(float component: COLOR) : SV_TARGET {
                uint1 x = uint1(1);
                return float4(component, component, component, component);
            }
        """).strip(),
        contains=(
            "uint x = uint(1);",
            "return vec4(component, component, component, component);",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_global_const_array_initializer",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/HLSLFileCheck/hlsl/types/vector/imm0.hlsl",
        code=textwrap.dedent("""
            float4 m = float4(4, 5, 6, 7);
            const float2 m3[2] = {12, 13, 14, 15};

            float4 main(float4 a : A) : SV_TARGET
            {
                return a + m + m3[0].xyxy + m3[1].xyxy;
            }
        """).strip(),
        contains=(
            "vec4 m = vec4(4, 5, 6, 7);",
            "const vec2[2] m3 = {12, 13, 14, 15};",
            "return ((a + m) + m3[0].xyxy) + m3[1].xyxy;",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_scoped_function_definitions",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_NAMESPACE_COMMIT,
        path="tools/clang/test/CodeGenSPIRV/namespace.functions.hlsl",
        code=textwrap.dedent("""
            namespace A {
              float3 AddGreen();

              namespace B {
                float3 AddBlue();
              }
            }

            float4 main(float4 PosCS : SV_Position) : SV_Target
            {
              float3 blue = A::B::AddBlue();
              float3 green = A::AddGreen();
              return float4(blue + green, 1);
            }

            float3 A::B::AddBlue() { return float3(0, 0, 1); }
            float3 A::AddGreen() { return float3(0, 1, 0); }
        """).strip(),
        contains=(
            "vec3 blue = A_B_AddBlue();",
            "vec3 green = A_AddGreen();",
            "vec3 A_B_AddBlue()",
            "vec3 A_AddGreen()",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_buffer_access_type_like_local_names",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_BUFFER_ACCESS_COMMIT,
        path="tools/clang/test/CodeGenSPIRV/op.buffer.access.hlsl",
        code=textwrap.dedent("""
            Buffer<int> intbuf;
            RWBuffer<int2> int2buf;
            Buffer<uint3> uint3buf;
            RWBuffer<float4> float4buf;

            void main() {
              int address;

              int int1 = intbuf[address];
              int2 int2 = int2buf[address];
              uint3 uint3 = uint3buf[address];
              float b = float4buf[address][2];
            }
        """).strip(),
        contains=(
            "Buffer<int> intbuf;",
            "RWBuffer<int2> int2buf;",
            "int int1 = intbuf[address];",
            "ivec2 int2 = int2buf[address];",
            "uvec3 uint3 = uint3buf[address];",
            "float b = float4buf[address][2];",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_contextual_resource_member_name",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_CONTEXTUAL_RESOURCE_COMMIT,
        path="tools/clang/test/CodeGenSPIRV/type.rwstructured-buffer.baseclass.counter.hlsl",
        code=textwrap.dedent("""
            RWStructuredBuffer<uint> Buf1;

            struct A
            {
              RWStructuredBuffer<uint> Buffer;
            };

            [numthreads(1, 1, 1)]
            void main()
            {
              A a;
              a.Buffer = Buf1;
            }
        """).strip(),
        contains=(
            "RWStructuredBuffer<uint> Buffer;",
            "@ numthreads(1, 1, 1)",
            "a.Buffer = Buf1;",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_rewriter_anonymous_struct_global",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_REWRITER_COMMIT,
        path="tools/clang/test/HLSL/rewriter/anonymous_struct.hlsl",
        code=textwrap.dedent("""
            SamplerState ss;

            static const struct {
              float a;
              SamplerState s;
            } A = {1.2, ss};

            float4 main() : SV_Target {
                return A.a;
            }
        """).strip(),
        contains=(
            "struct AnonymousStruct_A {",
            "sampler ss;",
            "static const AnonymousStruct_A A = {1.2, ss};",
            "return A.a;",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_center_modifier_identifier",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/HLSLFileCheck/hlsl/types/modifiers/center/center_kwd.hlsl",
        code=textwrap.dedent("""
            float main(center float t : T) : SV_TARGET
            {
                float center = 10.0f;
                return center * 2;
            }
        """).strip(),
        contains=(
            "float main(float t @ T) @ gl_FragColor",
            "float center = 10.0;",
            "return center * 2;",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_non_uniform_dynamic_resource_heap",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path=(
            "tools/clang/test/HLSLFileCheck/hlsl/intrinsics/"
            "createHandleFromHeap/NonUniformDynamic.hlsl"
        ),
        code=textwrap.dedent("""
            float read(uint ID) {
              Buffer<float> buf = ResourceDescriptorHeap[NonUniformResourceIndex(ID)];
              return buf[0];
            }

            void write(uint ID, float f) {
              RWBuffer<float> buf = ResourceDescriptorHeap[NonUniformResourceIndex(ID)];
              buf[0] = f;
            }

            [numthreads(8, 8, 1)]
            void main( uint2 ID : SV_DispatchThreadID) {
              float v = read(ID.x);
              write(ID.y, v);
            }
        """).strip(),
        contains=(
            "Buffer<float> buf = ResourceDescriptorHeap[NonUniformResourceIndex(ID)];",
            "RWBuffer<float> buf = ResourceDescriptorHeap[NonUniformResourceIndex(ID)];",
            "buf[0] = f;",
            "@ numthreads(8, 8, 1)",
            "void main(uvec2 ID @ gl_GlobalInvocationID)",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_contextual_keyword_identifiers",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/HLSLFileCheck/hlsl/objects/Texture/sample_kwd.hlsl",
        code=textwrap.dedent("""
            float3 foo(float3 sample) {
                return sample;
            }

            struct S {
              float4 center;
              float4 precise;
              float4 sample;
              float4 globallycoherent;
            };

            float4 main(float4 input : SV_POSITION) : SV_TARGET
            {
                float precise = 1.0f;
                int globallycoherent = 1;
                float sample;

                sample = 1.0f;
                globallycoherent += 10;

                return float4(foo(float3(precise, globallycoherent, sample)), input.x);
            }
        """).strip(),
        contains=(
            "vec4 precise;",
            "vec4 globallycoherent;",
            "vec3 foo(vec3 sample)",
            "float precise = 1.0;",
            "int globallycoherent = 1;",
            "sample = 1.0;",
            "globallycoherent += 10;",
        ),
    ),
    ExternalFixture(
        name="directx_shader_compiler_sampler_kind_modifiers",
        repo=DIRECTX_SHADER_COMPILER_REPO,
        commit=DIRECTX_SHADER_COMPILER_COMMIT,
        path="tools/clang/test/CodeGenHLSL/SamplerKind.hlsl",
        code=textwrap.dedent("""
            cbuffer cbPerObject : register( b0 )
            {
                float4 g_vObjectColor : packoffset( c0 );
            };

            cbuffer cbPerFrame : register( b1 )
            {
                float3 g_vLightDir : packoffset( c0 );
                float g_fAmbient : packoffset( c0.w );
            };

            Texture2D g_txDiffuse : register( t0 );
            SamplerState g_samLinear : register( s0 );
            SamplerComparisonState g_samLinearC : register( s1 );
            RWTexture2D<float4> uav1 : register( u3 );

            struct PS_INPUT
            {
              sample float3 vNormal : NORMAL;
              noperspective float2 vTexcoord : TEXCOORD0;
            };

            float cmpVal;

            float4 main( PS_INPUT Input) : SV_TARGET
            {
                float4 vDiffuse = g_txDiffuse.Sample( g_samLinear, Input.vTexcoord );
                vDiffuse += g_txDiffuse.CalculateLevelOfDetail(g_samLinear, Input.vTexcoord);
                vDiffuse += g_txDiffuse.Gather(g_samLinear, Input.vTexcoord);
                vDiffuse += g_txDiffuse.SampleCmp(g_samLinearC, Input.vTexcoord, cmpVal);
                vDiffuse += g_txDiffuse.GatherCmp(g_samLinearC, Input.vTexcoord, cmpVal);

                float fLighting = saturate( dot( g_vLightDir, Input.vNormal ) );
                fLighting = max( fLighting, g_fAmbient );

                return vDiffuse * fLighting * uav1.Load(int2(0,0));
            }
        """).strip(),
        contains=(
            "vec3 vNormal @ Normal @ sample;",
            "vec2 vTexcoord @ TexCoord0 @ noperspective;",
            "@ packoffset(c0.w)",
            "textureGather(g_txDiffuse, g_samLinear, Input.vTexcoord)",
            "textureGatherCompare(g_txDiffuse, g_samLinearC, Input.vTexcoord, cmpVal)",
        ),
    ),
    ExternalFixture(
        name="directx_graphics_samples_meshlet_render_pixel",
        repo=DIRECTX_GRAPHICS_SAMPLES_REPO,
        commit=DIRECTX_GRAPHICS_SAMPLES_COMMIT,
        path="Samples/Desktop/D3D12MeshShaders/src/MeshletRender/MeshletPS.hlsl",
        code=textwrap.dedent("""
            struct Constants
            {
                float4x4 World;
                float4x4 WorldView;
                float4x4 WorldViewProj;
                uint     DrawMeshlets;
            };

            struct VertexOut
            {
                float4 PositionHS   : SV_Position;
                float3 PositionVS   : POSITION0;
                float3 Normal       : NORMAL0;
                uint   MeshletIndex : COLOR0;
            };

            ConstantBuffer<Constants> Globals : register(b0);

            float4 main(VertexOut input) : SV_TARGET
            {
                float ambientIntensity = 0.1;
                float3 lightColor = float3(1, 1, 1);
                float3 lightDir = -normalize(float3(1, -1, 1));

                float3 diffuseColor;
                float shininess;
                if (Globals.DrawMeshlets)
                {
                    uint meshletIndex = input.MeshletIndex;
                    diffuseColor = float3(
                        float(meshletIndex & 1),
                        float(meshletIndex & 3) / 4,
                        float(meshletIndex & 7) / 8);
                    shininess = 16.0;
                }
                else
                {
                    diffuseColor = 0.8;
                    shininess = 64.0;
                }

                float3 normal = normalize(input.Normal);

                float cosAngle = saturate(dot(normal, lightDir));
                float3 viewDir = -normalize(input.PositionVS);
                float3 halfAngle = normalize(lightDir + viewDir);

                float blinnTerm = saturate(dot(normal, halfAngle));
                blinnTerm = cosAngle != 0.0 ? blinnTerm : 0.0;
                blinnTerm = pow(blinnTerm, shininess);

                float3 finalColor = (cosAngle + blinnTerm + ambientIntensity) * diffuseColor;

                return float4(finalColor, 1);
            }
        """).strip(),
        contains=(
            "ConstantBuffer<Constants> Globals;",
            "float cosAngle = clamp(dot(normal, lightDir), 0.0, 1.0);",
            "return vec4(finalColor, 1);",
        ),
    ),
    ExternalFixture(
        name="fidelityfx_fsr_dx12_pass_dispatch_filter",
        repo=FIDELITYFX_FSR_REPO,
        commit=FIDELITYFX_FSR_COMMIT,
        path="sample/src/DX12/FSR_Pass.hlsl",
        code=textwrap.dedent("""
            cbuffer cb : register(b0)
            {
                uint4 Const0;
                uint4 Sample;
            };

            SamplerState samLinearClamp : register(s0);

            Texture2D InputTexture : register(t0);
            RWTexture2D<float4> OutputTexture : register(u0);

            void CurrFilter(int2 pos)
            {
                float2 pp = (float2(pos) * float2(Const0.xy));
                OutputTexture[pos] = InputTexture.SampleLevel(samLinearClamp, pp, 0.0);
                float3 c;
                if (Sample.x == 1)
                    c *= c;
                OutputTexture[pos] = float4(c, 1);
            }

            [numthreads(8, 8, 1)]
            void mainCS(uint3 LocalThreadId : SV_GroupThreadID, uint3 WorkGroupId : SV_GroupID, uint3 Dtid : SV_DispatchThreadID)
            {
                uint2 gxy = LocalThreadId.xy + uint2(WorkGroupId.x << 4u, WorkGroupId.y << 4u);
                CurrFilter(gxy);
                gxy.x += 8u;
                CurrFilter(gxy);
            }
        """).strip(),
        contains=(
            "@ register(u0)",
            "imageStore(OutputTexture, pos, textureLod(InputTexture, samLinearClamp, pp, 0.0));",
            "uvec2 gxy = LocalThreadId.xy + uvec2(WorkGroupId.x << 4, WorkGroupId.y << 4);",
        ),
    ),
    ExternalFixture(
        name="fidelityfx_sdk_autoexposure_atomic_counter",
        repo=FIDELITYFX_SDK_REPO,
        commit=FIDELITYFX_SDK_COMMIT,
        path="Kits/Cauldron2/dx12/framework/shaders/autoexposure.hlsl",
        code=textwrap.dedent("""
            RWTexture2D<uint> AutomaticExposureSpdAtomicCounter : register(u0);

            groupshared uint spdCounter;

            void SpdIncreaseAtomicCounter(uint slice)
            {
                InterlockedAdd(AutomaticExposureSpdAtomicCounter[int2(0, 0)], 1, spdCounter);
            }

            uint SpdGetAtomicCounter()
            {
                return spdCounter;
            }

            void SpdResetAtomicCounter(uint slice)
            {
                AutomaticExposureSpdAtomicCounter[int2(0, 0)] = 0;
            }

            [numthreads(64, 1, 1)]
            void MainCS(uint3 WorkGroupId : SV_GroupID, uint LocalThreadIndex : SV_GroupIndex)
            {
                SpdIncreaseAtomicCounter(WorkGroupId.z);
            }
        """).strip(),
        contains=(
            "uimage2D AutomaticExposureSpdAtomicCounter;",
            "groupshared uint spdCounter;",
            "spdCounter = imageAtomicAdd(AutomaticExposureSpdAtomicCounter, ivec2(0, 0), 1u);",
            "imageStore(AutomaticExposureSpdAtomicCounter, ivec2(0, 0), 0);",
            "@ numthreads(64, 1, 1)",
        ),
    ),
    ExternalFixture(
        name="fidelityfx_sdk_fullscreen_triangle_vertex",
        repo=FIDELITYFX_SDK_REPO,
        commit=FIDELITYFX_SDK_COMMIT,
        path="Kits/Cauldron2/dx12/framework/shaders/fullscreen.hlsl",
        code=textwrap.dedent("""
            #define FAR_DEPTH 1.0

            float2 GetUV(uint2 coord, float2 texelSize)
            {
                return (coord + 0.5f) * texelSize;
            }

            int2 GetScreenCoordinates(float2 uv, int2 textureDims)
            {
                return floor(uv * textureDims);
            }

            struct VertexOut
            {
                float4 PosOut   : SV_Position;
                float2 UVOut    : TEXCOORD;
            };

            static const float4 FullScreenVertsPos[3] = { float4(-1.0, 1.0, FAR_DEPTH, 1.0), float4(3.0, 1.0, FAR_DEPTH, 1.0), float4(-1.0, -3.0, FAR_DEPTH, 1.0) };
            static const float2 FullScreenVertsUVs[3] = { float2(0.0, 0.0), float2(2.0, 0.0), float2(0.0, 2.0) };

            VertexOut FullscreenVS(uint vertexId : SV_VertexID)
            {
                VertexOut outVert;
                outVert.PosOut = FullScreenVertsPos[vertexId];
                outVert.UVOut = FullScreenVertsUVs[vertexId];
                return outVert;
            }
        """).strip(),
        contains=(
            "static const vec4 FullScreenVertsPos[3] = {vec4(-1.0, 1.0, 1.0, 1.0), vec4(3.0, 1.0, 1.0, 1.0), vec4(-1.0, -3.0, 1.0, 1.0)};",
            "VertexOut FullscreenVS(uint vertexId @ gl_VertexID)",
            "outVert.UVOut = FullScreenVertsUVs[vertexId];",
        ),
    ),
    ExternalFixture(
        name="fidelityfx_sdk_fpslimiter_rwstructuredbuffer",
        repo=FIDELITYFX_SDK_REPO,
        commit=FIDELITYFX_SDK_COMMIT,
        path="Kits/Cauldron2/dx12/framework/shaders/fpslimiter.hlsl",
        code=textwrap.dedent("""
            RWStructuredBuffer<float> DataBuffer : register(u0);

            cbuffer cb : register(b0)
            {
                uint NumLoops;
            }

            [numthreads(32,1,1)]
            void CSMain(uint dtID : SV_DispatchThreadID)
            {
                float tmp = DataBuffer[dtID];
                for (uint i = 0; i < NumLoops; i++)
                {
                    tmp = sin(tmp) + 1.5f;
                }
                DataBuffer[dtID] = tmp;
            }
        """).strip(),
        contains=(
            "RWStructuredBuffer<float> DataBuffer;",
            "@ numthreads(32, 1, 1)",
            "for (uint i = 0; i < NumLoops; i++)",
            "DataBuffer[dtID] = tmp;",
        ),
    ),
    ExternalFixture(
        name="fidelityfx_sdk_taa_tile_cache",
        repo=FIDELITYFX_SDK_REPO,
        commit=FIDELITYFX_SDK_COMMIT,
        path="Kits/Cauldron2/dx12/rendermodules/taa/shaders/taa.hlsl",
        code=textwrap.dedent("""
            #define RADIUS 1
            #define GROUP_SIZE 16
            #define TILE_DIM (2 * RADIUS + GROUP_SIZE)

            Texture2D ColorBuffer : register(t0);
            RWTexture2D<float4> OutputBuffer : register(u0);

            groupshared float3 Tile[TILE_DIM * TILE_DIM];

            float3 Reinhard(in float3 hdr)
            {
                return hdr / (hdr + 1.0f);
            }

            float3 Tap(in float2 pos)
            {
                return Tile[int(pos.x) + TILE_DIM * int(pos.y)];
            }

            [numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
            void FirstCS(uint3 globalID : SV_DispatchThreadID, uint3 localID : SV_GroupThreadID, uint localIndex : SV_GroupIndex, uint3 groupID : SV_GroupID)
            {
                const float2 tilePos = localID.xy + RADIUS + 0.5f;

                if (localIndex < TILE_DIM * TILE_DIM / 4)
                {
                    const int2 anchor = groupID.xy * GROUP_SIZE - RADIUS;
                    const int2 coord1 = anchor + int2(localIndex % TILE_DIM, localIndex / TILE_DIM);
                    const float3 color0 = ColorBuffer[coord1].xyz;
                    Tile[localIndex] = Reinhard(color0);
                }

                GroupMemoryBarrierWithGroupSync();
                const float3 center = Tap(tilePos);
                OutputBuffer[globalID.xy] = float4(center, 1.0);
            }
        """).strip(),
        contains=(
            "groupshared vec3 Tile[((2 * 1) + 16) * ((2 * 1) + 16)];",
            "@ numthreads(16, 16, 1)",
            "vec3 color0 = texelFetch(ColorBuffer, coord1, 0).xyz;",
            "workgroupBarrier();",
            "imageStore(OutputBuffer, globalID.xy, vec4(center, 1.0));",
        ),
    ),
    ExternalFixture(
        name="wickedengine_rtao_unorm_uav_component_write",
        repo=WICKED_ENGINE_REPO,
        commit=WICKED_ENGINE_COMMIT,
        path="WickedEngine/shaders/rtao_denoise_filterCS.hlsl",
        code=textwrap.dedent("""
            RWTexture2D<unorm float> output : register(u1);
            float rtao_power;

            bool FFX_DNSR_Shadows_IsShadowReciever(uint2 did)
            {
                return did.x > 0;
            }

            [numthreads(8, 8, 1)]
            void main(uint2 did : SV_DispatchThreadID)
            {
                const float mean = 0.5f;
                output[did].x = FFX_DNSR_Shadows_IsShadowReciever(did) ? pow(mean, rtao_power) : 1;
            }
        """).strip(),
        contains=(
            "@ register(u1)",
            "image2D output;",
            "imageStore(output, did, FFX_DNSR_Shadows_IsShadowReciever(did) ? pow(mean, rtao_power) : 1);",
        ),
    ),
]


def parse_hlsl(code):
    tokens = HLSLLexer(code).tokenize()
    return HLSLParser(tokens).parse()


def generate_crossgl(code):
    ast = parse_hlsl(code)
    return HLSLToCrossGLConverter().generate(ast)


def parse_crossgl(code):
    tokens = CrossGLLexer(code).get_tokens()
    return CrossGLParser(tokens).parse()


def test_external_fixture_metadata_records_repositories_and_commits():
    assert all(
        fixture.repo.startswith("https://github.com/") for fixture in EXTERNAL_FIXTURES
    )
    assert all(len(fixture.commit) == 40 for fixture in EXTERNAL_FIXTURES)
    assert all(
        fixture.path.endswith((".hlsl", ".hlsli")) for fixture in EXTERNAL_FIXTURES
    )


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda fixture: fixture.name)
def test_parse_external_directx_fixture(fixture):
    ast = parse_hlsl(fixture.code)

    assert ast is not None
    assert ast.functions
    assert fixture.source_url.startswith(fixture.repo)


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda fixture: fixture.name)
def test_codegen_external_directx_fixture_to_parseable_crossgl(fixture):
    crossgl = generate_crossgl(fixture.code)

    for expected in fixture.contains:
        assert expected in crossgl
    assert parse_crossgl(crossgl) is not None

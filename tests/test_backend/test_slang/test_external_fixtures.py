import pytest

import crosstl.translator as cgl_translator
from crosstl.backend.slang import SlangCrossGLCodeGen, SlangLexer, SlangParser

EXTERNAL_REPOS = {
    "shader-slang/slang": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "adc996670ec281aa8a4ee131f30b324648cbbe60",
    },
    "shader-slang/slang-current": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "726e0973b3f547c7729b86f122ff7aef8322bace",
    },
    "shader-slang/slang-current-2026-06-04": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "8c4e02e4021d73091a4f1d4eba842c0dd986997e",
    },
    "shader-slang/slang-current-2026-06-07": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "5230a81f2fe68afe5cb8d04a1b09d56476f6b960",
    },
    "shader-slang/slang-property-2026-06-04": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "564ac9f050d6569efd773e2f74e7d067a4e54baa",
    },
    "shader-slang/slang-layout-2026-06-04": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "564ac9f050d6569efd773e2f74e7d067a4e54baa",
    },
    "shader-slang/slang-subscript-2026-06-04": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "564ac9f050d6569efd773e2f74e7d067a4e54baa",
    },
    "shader-slang/slang-types-interface-2026-06-05": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "564ac9f050d6569efd773e2f74e7d067a4e54baa",
    },
    "shader-slang/slang-gfx-tools-2026": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "c6f104ca76a54ca1565dac54363ea763dd906de6",
    },
    "shader-slang/slang-examples": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "bbc4b02787d427065d713c1820a33b66dc6c4117",
    },
    "shader-slang/slang-generated": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "3dacd6028f380f3f2ff735516d39a1bdc05aeed2",
    },
    "shader-slang/slang-generated-2026": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "a49a125809f7b491bfd54a3014021fa7d716bbdc",
    },
    "shader-slang/slang-wgsl-2026": {
        "url": "https://github.com/shader-slang/slang",
        "commit": "726e0973b3f547c7729b86f122ff7aef8322bace",
    },
    "shader-slang/optix-examples": {
        "url": "https://github.com/shader-slang/optix-examples",
        "commit": "02fa85ae2f39400ced9a602531aa096589055076",
    },
    "shader-slang/slang-rhi": {
        "url": "https://github.com/shader-slang/slang-rhi",
        "commit": "9aa67530649c52b4a057a2fba185cf9247c33ec0",
    },
    "NVIDIAGameWorks/Falcor": {
        "url": "https://github.com/NVIDIAGameWorks/Falcor",
        "commit": "eb540f6748774680ce0039aaf3ac9279266ec521",
    },
    "HenriMichelon/vireo_samples": {
        "url": "https://github.com/HenriMichelon/vireo_samples",
        "commit": "e2d788909cc73a7f515380792796cebaaed53a7e",
    },
    "NVIDIAGameWorks/RTXGI": {
        "url": "https://github.com/NVIDIAGameWorks/RTXGI",
        "commit": "10b5770b8eaddfc1faab82b65f799ac6f47dcc44",
        "note": "searched; current tree has no .slang/.slangh files",
    },
    "nvpro-samples/vk_slang_editor": {
        "url": "https://github.com/nvpro-samples/vk_slang_editor",
        "commit": "620f60e4f724c4d06b1e7251c3fac6ee4d96cb54",
    },
}


EXTERNAL_FIXTURES = [
    {
        "id": "slang_default_parameter",
        "repo": "shader-slang/slang",
        "path": "tests/compute/default-parameter.slang",
        "source": (
            """
            RWStructuredBuffer<int> outputBuffer;

            int helper(int val, int a = 16)
            {
                return val + a;
            }

            int test(int val)
            {
                return helper(val) + helper(val, 256);
            }

            [numthreads(4, 1, 1)]
            void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
            {
                outputBuffer[dispatchThreadID.x] = test((int)dispatchThreadID.x);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "int helper(int val, int a)",
            "return helper(val) + helper(val, 256);",
        ],
        "not_contains": ["int a, = , 16", "int a = 16"],
    },
    {
        "id": "slang_autodiff_generic_where_clause",
        "repo": "shader-slang/slang",
        "path": "tests/autodiff/autodiff-generic-where-clause.slang",
        "source": (
            """
            [ForwardDifferentiable]
            T genericCalc<T : IDifferentiable & IFloat>(T val, T x)
            {
                return val * x * x + x;
            }

            void main()
            {
                let result = fwd_diff(genericCalc<float>)(
                    DifferentialPair<float>(2.0, 0.0),
                    DifferentialPair<float>(3.0, 1.0));
                printf("%f\\n", result.d);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "T genericCalc(T val, T x)",
            "let result = fwd_diff(genericCalc<float>)",
            'printf("%f\\n", result.d);',
        ],
    },
    {
        # Source: https://github.com/shader-slang/slang
        # Path: tests/cooperative-matrix/
        # mat-mul-add-spirv-matrix-operands.slang
        "id": "slang_cooperative_matrix_generic_prefix_typealias",
        "repo": "shader-slang/slang-current",
        "path": "tests/cooperative-matrix/mat-mul-add-spirv-matrix-operands.slang",
        "source": (
            """
            using namespace linalg;

            __generic<T : __BuiltinArithmeticType>
            typealias CoopMatAType =
                CoopMat<T, MemoryScope.Subgroup, 16, 16,
                        CoopMatMatrixUse::MatrixA>;

            RWStructuredBuffer<uint32_t> outputBuffer;

            [numthreads(32, 1, 1)]
            void computeMain()
            {
                coopMatMulAdd<uint32_t, false>(
                    CoopMatAType<uint16_t>(2),
                    CoopMatAType<uint32_t>(3),
                    CoopMatAType<uint16_t>(4)
                ).Store<CoopMatMatrixLayout::RowMajor>(outputBuffer, 0, 16);
            }
        """
        ),
        "crossgl": False,
        "contains": [
            (
                "typedef CoopMat<T, MemoryScope.Subgroup, 16, 16, "
                "CoopMatMatrixUse::MatrixA> CoopMatAType<T>;"
            ),
            "coopMatMulAdd<uint32_t, false>",
            ".Store<CoopMatMatrixLayout::RowMajor>",
        ],
        "not_contains": ["__generic", "__BuiltinArithmeticType"],
    },
    {
        "id": "slang_empty_switch_unlabeled_statement",
        "repo": "shader-slang/slang",
        "path": "tests/bugs/empty-switch.slang",
        "source": (
            """
            RWStructuredBuffer<int> outputBuffer;

            [numthreads(4, 1, 1)]
            void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
            {
                int index = int(dispatchThreadID.x);
                int a = index;

                switch (++a)
                {
                }

                switch (index)
                {
                    a += 10;
                }

                outputBuffer[index] = a;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "switch (++a) {",
            "switch (index) {",
            "a += 10;",
            "outputBuffer[index] = a;",
        ],
    },
    {
        # Source: https://github.com/shader-slang/slang
        # Commit: 5230a81f2fe68afe5cb8d04a1b09d56476f6b960
        # Path: tests/compute/comma-operator.slang
        "id": "slang_compute_return_comma_operator",
        "repo": "shader-slang/slang-current-2026-06-07",
        "path": "tests/compute/comma-operator.slang",
        "source": (
            """
            int test(int inVal)
            {
                int a = inVal;
                return a*=2, a+1;
            }

            RWStructuredBuffer<int> outputBuffer;

            [numthreads(4, 1, 1)]
            void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
            {
                uint tid = dispatchThreadID.x;
                int inVal = outputBuffer[tid];
                int outVal = test(inVal);
                outputBuffer[tid] = outVal;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "return (a *= 2, a + 1);",
            "layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;",
            "outputBuffer[tid] = outVal;",
        ],
        "not_contains": ["return a *= 2, a + 1;"],
    },
    {
        # Source: https://github.com/shader-slang/slang
        # Commit: 5230a81f2fe68afe5cb8d04a1b09d56476f6b960
        # Path: tests/bugs/c-style-cast-overload.slang
        "id": "slang_c_style_cast_overload_unnamed_parameters",
        "repo": "shader-slang/slang-current-2026-06-07",
        "path": "tests/bugs/c-style-cast-overload.slang",
        "source": (
            """
            RWStructuredBuffer<int> outputBuffer;

            struct S {};

            int f(S)
            {
                return 1;
            }

            int f(float)
            {
                return 2;
            }

            [shader("compute")]
            [numthreads(1,1,1)]
            void computeMain()
            {
                outputBuffer[0] = f(float(0));
                outputBuffer[1] = f((float)1);
                outputBuffer[2] = f((float)0);
                outputBuffer[3] = f((S)0);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "int f(S _param0)",
            "int f(float _param0)",
            "outputBuffer[3] = f(S(0));",
        ],
        "not_contains": ["int f(S )", "int f(float )"],
    },
    {
        # Source: https://github.com/shader-slang/slang
        # Commit: 5230a81f2fe68afe5cb8d04a1b09d56476f6b960
        # Path: tests/language-feature/multi-level-break.slang
        "id": "slang_labeled_break_from_multi_level_break_sample",
        "repo": "shader-slang/slang-current-2026-06-07",
        "path": "tests/language-feature/multi-level-break.slang",
        "source": (
            """
            int test(int r)
            {
                int result = 0;
            outer:
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        result++;
                        if (r == 0)
                        {
                            break outer;
                        }
                        break;
                    }
                }
                return result;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "outer:",
            "break outer;",
            "break;",
            "return result;",
        ],
    },
    {
        # Source: https://github.com/shader-slang/slang
        # Commit: 5230a81f2fe68afe5cb8d04a1b09d56476f6b960
        # Path: tests/autodiff/arithmetic-jvp.slang
        "id": "slang_autodiff_scalar_differential_typedef_reparse",
        "repo": "shader-slang/slang-current-2026-06-07",
        "path": "tests/autodiff/arithmetic-jvp.slang",
        "source": (
            """
            RWStructuredBuffer<float> outputBuffer;

            typedef DifferentialPair<float> dpfloat;
            typedef float.Differential dfloat;

            [numthreads(1, 1, 1)]
            void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
            {
                dfloat x = 1.0;
                outputBuffer[0] = x;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "typedef DifferentialPair<float> dpfloat;",
            "typedef float dfloat;",
            "dfloat x = 1.0;",
            "outputBuffer[0] = x;",
        ],
        "not_contains": ["float.Differential"],
    },
    {
        # Source: https://github.com/shader-slang/slang
        # Commit: 5230a81f2fe68afe5cb8d04a1b09d56476f6b960
        # Path: tests/autodiff/constref-param.slang
        "id": "slang_autodiff_constref_parameter_qualifier_reparse",
        "repo": "shader-slang/slang-current-2026-06-07",
        "path": "tests/autodiff/constref-param.slang",
        "source": (
            """
            RWStructuredBuffer<float> outputBuffer;

            struct NonDiff
            {
                float a;
            }

            [Differentiable]
            float myFunc(__constref NonDiff fIn, float x, __constref no_diff float y)
            {
                return x * fIn.a + y;
            }

            [Differentiable]
            float myFunc2(__constref NonDiff fIn, float x, no_diff __constref float y)
            {
                return x * fIn.a + y;
            }

            [numthreads(1, 1, 1)]
            void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
            {
                float a = 10.0;
                NonDiff fIn = { a };
                DifferentialPair<float> dpx = DifferentialPair<float>(4.0, 1.0);
                float rs = __fwd_diff(myFunc)(fIn, dpx, 1.0).d;
                float rs2 = __fwd_diff(myFunc2)(fIn, dpx, 1.0).d;

                outputBuffer[0] = rs;
                outputBuffer[1] = rs2;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "float myFunc(NonDiff fIn, float x, float y)",
            "float myFunc2(NonDiff fIn, float x, float y)",
            "float rs = __fwd_diff(myFunc)(fIn, dpx, 1.0).d;",
            "outputBuffer[1] = rs2;",
        ],
        "not_contains": [
            "__constref fIn NonDiff",
            "__constref no_diff",
            "no_diff __constref",
        ],
    },
    {
        "id": "slang_generated_defer_scope_exit",
        "repo": "shader-slang/slang-current-2026-06-07",
        "path": (
            "docs/generated/tests/design/pipeline/04b-pre-link-passes/"
            "phase-b-lower-defer-inlines-body-at-scope-exit.slang"
        ),
        "source": (
            """
            uniform RWStructuredBuffer<int> buf;

            [numthreads(1, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID)
            {
                defer { buf[1] = 100; }
                buf[0] = 1;
            }
        """
        ),
        "crossgl": True,
        "ordered_contains": [
            "buf[0] = 1;",
            "buf[1] = 100;",
        ],
        "not_contains": ["defer"],
    },
    {
        "id": "slang_generated_sizeof_type_operand",
        "repo": "shader-slang/slang-current",
        "path": (
            "docs/generated/tests/syntax-reference/keywords-and-builtins/"
            "expr-sizeof-int.slang"
        ),
        "source": (
            """
            void main()
            {
                printf("s=%d\\n", int(sizeof(int)));
            }
        """
        ),
        "crossgl": True,
        "contains": [
            'printf("s=%d\\n", int(sizeof(int)));',
        ],
    },
    {
        "id": "slang_tbuffer",
        "repo": "shader-slang/slang",
        "path": "tests/hlsl/tbuffer.slang",
        "source": (
            """
            tbuffer tbuf : register(t0)
            {
                float4 tb_val1;
            }

            tbuffer tbuf2 : register(t1)
            {
                Texture2D<float4> texture2D;
                float4 tb_val2;
            }

            RWStructuredBuffer<float4> outputBuffer;

            [numthreads(1, 1, 1)]
            void computeMain()
            {
                outputBuffer[0] = tb_val1 + texture2D[0] + tb_val2;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "cbuffer tbuf @register(t0)",
            "cbuffer tbuf2 @register(t1)",
            "sampler2D texture2D;",
        ],
    },
    {
        "id": "slang_func_keyword_default_parameter_from_current_docs",
        "repo": "shader-slang/slang-current-2026-06-04",
        "path": (
            "docs/generated/tests/conformance/declarations/"
            "func-default-param-functional.slang"
        ),
        "source": (
            """
            func add(x: int, y: float = 1.0f) -> float
            {
                return float(x) + y;
            }

            void main()
            {
                printf("%g\\n", add(5));
                printf("%g\\n", add(5, 2.0));
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "float add(int x, float y)",
            'printf("%g\\n", add(5));',
            'printf("%g\\n", add(5, 2.0));',
        ],
        "not_contains": ["func add", "float y = 1.0"],
    },
    {
        "id": "falcor_texture_load",
        "repo": "NVIDIAGameWorks/Falcor",
        "path": "Source/Tools/FalcorTest/Tests/Core/TextureLoadTests.cs.slang",
        "source": (
            """
            Texture2D<float4> gTex;
            RWStructuredBuffer<float4> result;

            [numthreads(1, 1, 1)]
            void main(uint3 dispatchThreadID : SV_DispatchThreadID)
            {
                result[dispatchThreadID.x] = gTex.Load(int3(0, 0, 0));
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "sampler2D gTex;",
            "texelFetch(gTex, ivec2(0, 0), 0)",
        ],
        "not_contains": ["gTex.Load"],
    },
    {
        "id": "falcor_enum_class_underlying_type",
        "repo": "NVIDIAGameWorks/Falcor",
        "path": "Source/RenderPasses/DebugPasses/ColorMapPass/ColorMapParams.slang",
        "source": (
            """
            BEGIN_NAMESPACE_FALCOR

            enum class ColorMap : uint32_t
            {
                Grey,
                Jet,
                Viridis,
                Plasma,
                Magma,
                Inferno,
            };

            FALCOR_ENUM_INFO(
                ColorMap,
                {
                    { ColorMap::Grey, "Grey" },
                    { ColorMap::Jet, "Jet" },
                    { ColorMap::Viridis, "Viridis" },
                    { ColorMap::Plasma, "Plasma" },
                    { ColorMap::Magma, "Magma" },
                    { ColorMap::Inferno, "Inferno" },
                }
            );
            FALCOR_ENUM_REGISTER(ColorMap);

            struct ColorMapParams
            {
                float minValue = 0.f;
                float maxValue = 1.f;
            };

            END_NAMESPACE_FALCOR
        """
        ),
        "crossgl": True,
        "enum_underlying_types": {"ColorMap": "uint32_t"},
        "contains": [
            "enum ColorMap {",
            "Grey,",
            "Inferno,",
            "struct ColorMapParams {",
            "float minValue = 0.0;",
        ],
        "not_contains": ["BEGIN_NAMESPACE_FALCOR", "FALCOR_ENUM_INFO"],
    },
    {
        "id": "slang_gfx_tool_public_enums",
        "repo": "shader-slang/slang-gfx-tools-2026",
        "path": "tools/gfx/gfx.slang",
        "source": (
            """
            public enum AccelerationStructureBuildFlags
            {
                None,
                AllowUpdate = 1,
                AllowCompaction = 2,
            };

            public enum class GeometryType
            {
                Triangles,
                ProcedurePrimitives
            };
        """
        ),
        "crossgl": True,
        "contains": [
            "enum AccelerationStructureBuildFlags {",
            "AllowUpdate = 1,",
            "enum GeometryType {",
            "ProcedurePrimitives,",
        ],
        "not_contains": ["public enum"],
    },
    {
        "id": "falcor_texture_brace_vector_initializer",
        "repo": "NVIDIAGameWorks/Falcor",
        "path": "Source/Tools/FalcorTest/Tests/Core/TextureLoadTests.cs.slang",
        "source": (
            """
            RWStructuredBuffer<uint4> result;

            Texture2D<float4> texUnorm;
            Texture2D<uint4> texUnormAsUint;
            Texture2D<uint4> texUint;

            [numthreads(256, 1, 1)]
            void testLoadFormat(uint3 threadId: SV_DispatchThreadID)
            {
                float4 f = texUnorm[threadId.xy];
                uint4 u = texUnormAsUint[threadId.xy];
                uint4 v = texUint[threadId.xy];

                result[threadId.x] =
                    { asuint(f.x), u.x, (uint)(f.x * 255.f), v.x };
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "sampler2D texUnorm;",
            (
                "result[threadId.x] = "
                "{floatBitsToUint(f.x), u.x, uint(f.x * 255.0), v.x};"
            ),
        ],
        "not_contains": ["asuint(", "255.f"],
    },
    {
        "id": "slang_texture2darray_load_int4",
        "repo": "shader-slang/slang",
        "path": (
            "docs/generated/tests/cross-cutting/core-module/texture2darray-load.slang"
        ),
        "source": (
            """
            Texture2DArray<float4> tex;
            RWStructuredBuffer<float4> outBuf;

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID)
            {
                outBuf[tid.x] = tex.Load(int4(int(tid.x), int(tid.y), 0, 0));
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "sampler2DArray tex;",
            "texelFetch(tex, ivec3(int(tid.x), int(tid.y), 0), 0)",
        ],
        "not_contains": [
            "tex.Load",
            "ivec4(int(tid.x), int(tid.y), 0, 0)",
        ],
    },
    {
        "id": "slang_texture3d_load_int4",
        "repo": "shader-slang/slang-generated-2026",
        "path": "docs/generated/tests/cross-cutting/core-module/texture3d-load.slang",
        "source": (
            """
            Texture3D<float4> tex;
            RWStructuredBuffer<float4> outBuf;

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID)
            {
                outBuf[tid.x] =
                    tex.Load(int4(int(tid.x), int(tid.y), int(tid.z), 0));
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "sampler3D tex;",
            "texelFetch(tex, ivec3(int(tid.x), int(tid.y), int(tid.z)), 0)",
        ],
        "not_contains": [
            "tex.Load",
            "ivec4(int(tid.x), int(tid.y), int(tid.z), 0)",
        ],
    },
    {
        "id": "slang_wgsl_texture1d_load_int2",
        "repo": "shader-slang/slang-wgsl-2026",
        "path": "tests/wgsl/texture-load.slang",
        "source": (
            """
            Texture1D<float4> tex;
            RWStructuredBuffer<float4> outBuf;

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID)
            {
                outBuf[tid.x] = tex.Load(int2(int(tid.x), 0));
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "sampler1D tex;",
            "texelFetch(tex, int(tid.x), 0)",
        ],
        "not_contains": [
            "tex.Load",
            "ivec2(int(tid.x), 0)",
        ],
    },
    {
        "id": "slang_texturecubearray_samplelevel",
        "repo": "shader-slang/slang-generated",
        "path": (
            "docs/generated/tests/cross-cutting/core-module/"
            "texturecubearray-samplelevel.slang"
        ),
        "source": (
            """
            TextureCubeArray<float4> tex;
            SamplerState samp;
            RWStructuredBuffer<float4> outBuf;

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID)
            {
                outBuf[tid.x] =
                    tex.SampleLevel(samp, float4(1.0, 0.0, 0.0, 0.0), 0.0);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "samplerCubeArray tex;",
            "sampler samp;",
            "textureLod(tex, samp, vec4(1.0, 0.0, 0.0, 0.0), 0.0)",
        ],
        "not_contains": ["tex.SampleLevel"],
    },
    {
        "id": "slang_texture2darray_gatherred_from_core_module",
        "repo": "shader-slang/slang-generated-2026",
        "path": (
            "docs/generated/tests/cross-cutting/core-module/"
            "texture2darray-gatherred.slang"
        ),
        "source": (
            """
            Texture2DArray<float4> tex;
            SamplerState samp;
            RWStructuredBuffer<float4> outBuf;

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID)
            {
                outBuf[tid.x] =
                    tex.GatherRed(samp, float3(0.5, 0.5, 0.0));
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "sampler2DArray tex;",
            "sampler samp;",
            "textureGather(tex, samp, vec3(0.5, 0.5, 0.0), 0)",
        ],
        "not_contains": ["tex.GatherRed"],
    },
    {
        # Source: https://github.com/shader-slang/slang
        # Commit: 5230a81f2fe68afe5cb8d04a1b09d56476f6b960
        # Path: tests/type/texture-sampler/combined-texture-sampler-array.slang
        "id": "slang_combined_sampler_array_dual_registers",
        "repo": "shader-slang/slang-current-2026-06-07",
        "path": "tests/type/texture-sampler/combined-texture-sampler-array.slang",
        "source": (
            """
            Sampler2D tex2D[16] : register(t0, space2) : register(s0, space2);
            SamplerCube texCube[8] : register(t5, space1) : register(s10, space1);
            Sampler3D tex3D[4] : register(t20) : register(s25);
            Sampler1D tex1D[2] : register(t30, space3) : register(s30, space3);
            Sampler2D singleTex[1] : register(t100, space5) : register(s100, space5);
            Sampler2D largeTex[64] : register(t200, space10) : register(s200, space10);
            Sampler2D singleSampler : register(t300) : register(s300);
            Sampler2D partialTex[2] : register(t400);

            float4 fragMain() : SV_Target
            {
                float4 result = tex2D[0].Sample(float2(0.5, 0.5));
                result += texCube[1].Sample(float3(0.2, 0.3, 0.4));
                result += tex3D[2].Sample(float3(0.1, 0.9, 0.6));
                result += tex1D[1].Sample(0.7);
                result += singleTex[0].Sample(float2(0.7, 0.4));
                result += largeTex[5].Sample(float2(0.8, 0.6));
                result += singleSampler.Sample(float2(0.3, 0.7));
                result += partialTex[1].Sample(float2(0.4, 0.8));

                return result;
            }
        """
        ),
        "crossgl": True,
        "registers": {
            "tex2D": ["t0,space2", "s0,space2"],
            "texCube": ["t5,space1", "s10,space1"],
            "tex3D": ["t20", "s25"],
            "tex1D": ["t30,space3", "s30,space3"],
            "singleSampler": ["t300", "s300"],
            "partialTex": ["t400"],
        },
        "contains": [
            "sampler2D tex2D[16] @register(t0, space2);",
            "samplerCube texCube[8] @register(t5, space1);",
            "sampler3D tex3D[4] @register(t20);",
            "sampler1D tex1D[2] @register(t30, space3);",
            "sampler2D singleSampler @register(t300);",
            "vec4 result = texture(tex2D[0], vec2(0.5, 0.5));",
            "result += texture(texCube[1], vec3(0.2, 0.3, 0.4));",
            "result += texture(tex1D[1], 0.7);",
            "return result;",
        ],
        "not_contains": [
            "Expected semantic name",
            ".Sample(",
            "@register(s0, space2)",
            "@register(s300)",
        ],
    },
    {
        "id": "slang_hlsl_intrinsic_sample_cmp_bias",
        "repo": "shader-slang/slang",
        "path": "tests/hlsl-intrinsic/texture/sample-cmp.slang",
        "source": (
            """
            SamplerComparisonState shadowSampler;
            Texture2D<float> shadowMap2D;
            Texture2DArray<float> shadowMap2DArray;
            TextureCube<float> shadowMapCube;

            struct PSInput
            {
                float4 position : SV_Position;
                float2 texCoord : TEXCOORD0;
                float depth : DEPTH;
            };

            struct PSOutput
            {
                float4 color : SV_Target0;
            };

            PSOutput fragmentMain(PSInput input)
            {
                PSOutput output;
                float bias = 0.5;
                int2 offset2 = int2(1, 0);
                float3 cubeDir = float3(input.texCoord, 1.0);
                float shadow = 0;

                shadow += shadowMap2D.SampleCmpBias(
                    shadowSampler, input.texCoord, input.depth, bias);
                shadow += shadowMap2DArray.SampleCmpBias(
                    shadowSampler, float3(input.texCoord, 0),
                    input.depth, bias, offset2);
                shadow += shadowMapCube.SampleCmpBias(
                    shadowSampler, cubeDir, input.depth, bias);

                output.color = float4(shadow, shadow, shadow, 1.0);
                return output;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "sampler shadowSampler;",
            "sampler2D shadowMap2D;",
            "sampler2DArray shadowMap2DArray;",
            "samplerCube shadowMapCube;",
            "textureCompare(shadowMap2D, shadowSampler, input.texCoord, input.depth)",
            (
                "textureCompareOffset(shadowMap2DArray, shadowSampler, "
                "vec3(input.texCoord, 0), input.depth, offset2)"
            ),
            "textureCompare(shadowMapCube, shadowSampler, cubeDir, input.depth)",
            (
                "unsupported Slang texture overload extras for SampleCmpBias: "
                "dropped LOD bias"
            ),
        ],
        "not_contains": ["SampleCmpBias("],
    },
    {
        "id": "slang_texture2d_getdimensions_querysize_lod_from_spirv_generated",
        "repo": "shader-slang/slang-generated-2026",
        "path": (
            "docs/generated/tests/target-pipelines/spirv/"
            "texture2d-getdimensions-querysize-lod.slang"
        ),
        "source": (
            """
            Texture2D<float4> tex;
            RWStructuredBuffer<uint> outBuf;

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID)
            {
                uint w, h, levels;
                tex.GetDimensions(0, w, h, levels);
                outBuf[tid.x] = w + h + levels;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "sampler2D tex;",
            "ivec2 _cgl_getDimensionsSize0 = textureSize(tex, 0);",
            "w = uint(_cgl_getDimensionsSize0.x);",
            "h = uint(_cgl_getDimensionsSize0.y);",
            "levels = uint(textureQueryLevels(tex));",
        ],
        "not_contains": ["tex.GetDimensions"],
    },
    {
        "id": "slang_spirv_matrix_layout_modifier_members",
        "repo": "shader-slang/slang-layout-2026-06-04",
        "path": "tests/spirv/optypematrix-layout-modifier.slang",
        "source": (
            """
            struct MatrixData
            {
                MATRIX_LAYOUT float4x4 m1;
                MATRIX_LAYOUT float2x4 m2;
                MATRIX_LAYOUT float3x2 m3;
            };

            RWStructuredBuffer<float> outputBuffer;

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void computeMain()
            {
                outputBuffer[0] = MatrixData().m1[0][0];
                outputBuffer[1] = MatrixData().m2[1][3];
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "struct MatrixData {",
            "mat4 m1;",
            "float2x4 m2;",
            "float3x2 m3;",
            "outputBuffer[0] = MatrixData().m1[0][0];",
        ],
        "not_contains": ["MATRIX_LAYOUT", "row_major", "column_major"],
    },
    {
        "id": "slang_generated_subscript_get_accessor",
        "repo": "shader-slang/slang-subscript-2026-06-04",
        "path": (
            "docs/generated/tests/conformance/declarations/"
            "subscript-get-functional.slang"
        ),
        "source": (
            """
            struct MyVec
            {
                float x, y;
                __subscript(int index) -> float
                {
                    get { return index == 0 ? x : y; }
                }
            }

            void main()
            {
                MyVec v;
                v.x = 5.0;
                v.y = 9.0;
                printf("%g\\n%g\\n", v[0], v[1]);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "struct MyVec {",
            "float x;",
            "float y;",
            "v.x = 5.0;",
            "v.y = 9.0;",
            'printf("%g\\n%g\\n", v[0], v[1]);',
        ],
        "not_contains": ["__subscript", "operator[]", "FunctionNode("],
    },
    {
        "id": "slang_generated_interface_constructor_requirement_codegen",
        "repo": "shader-slang/slang-types-interface-2026-06-05",
        "path": (
            "docs/generated/tests/conformance/types-interface/"
            "interface-constructor-requirement.slang"
        ),
        "source": (
            """
            interface IInitializable
            {
                __init(int v);
                int getValue();
            }

            struct InitStruct : IInitializable
            {
                int stored;

                __init(int v)
                {
                    stored = v * 2;
                }

                int getValue() { return stored; }
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "struct InitStruct {",
            "int stored;",
        ],
        "not_contains": [
            "interface IInitializable",
            ": IInitializable",
            "__init",
        ],
    },
    {
        "id": "slang_generated_interface_subscript_requirement_codegen",
        "repo": "shader-slang/slang-types-interface-2026-06-05",
        "path": (
            "docs/generated/tests/conformance/types-interface/"
            "interface-subscript-requirement.slang"
        ),
        "source": (
            """
            interface IIndexable
            {
                __subscript(uint i) -> int { get; set; }
            }

            struct FixedArray : IIndexable
            {
                int data[4];

                __subscript(uint i) -> int
                {
                    get { return data[i]; }
                    set { data[i] = newValue; }
                }
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "struct FixedArray {",
            "int data[4];",
        ],
        "not_contains": [
            "interface IIndexable",
            ": IIndexable",
            "__subscript",
            "operator[]",
        ],
    },
    {
        "id": "slang_hlsl_intrinsic_mul_matrix_vector",
        "repo": "shader-slang/slang-generated-2026",
        "path": (
            "docs/generated/tests/cross-cutting/core-module/"
            "hlsl-intrinsic-mul-matrix-lowers-per-target.slang"
        ),
        "source": (
            """
            RWStructuredBuffer<float3> outBuf;

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID)
            {
                float3x3 m = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
                float3 v = float3(float(tid.x), float(tid.y), float(tid.z));
                outBuf[tid.x] = mul(m, v);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "mat3 m = mat3(1, 0, 0, 0, 1, 0, 0, 0, 1);",
            "vec3 v = vec3(float(tid.x), float(tid.y), float(tid.z));",
            "outBuf[tid.x] = (m * v);",
        ],
        "not_contains": ["mul("],
    },
    {
        "id": "slang_groupshared_atomic_add",
        "repo": "shader-slang/slang-generated",
        "path": (
            "docs/generated/tests/ir-reference/resources-and-atomics/"
            "atomic-groupshared-add.slang"
        ),
        "source": (
            """
            groupshared Atomic<int> gsa;
            uniform RWStructuredBuffer<int> rwbuf;

            [numthreads(8, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID,
                      uint3 gid : SV_GroupThreadID)
            {
                int prev = gsa.add((int)gid.x + 1);
                rwbuf[tid.x] = prev;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "Atomic<int> gsa;",
            "RWStructuredBuffer<int> rwbuf;",
            "int prev = gsa.add(int(gid.x) + 1);",
            "rwbuf[tid.x] = prev;",
        ],
    },
    {
        "id": "optix_examples_raygeneration_stages",
        "repo": "shader-slang/optix-examples",
        "path": "example02_pipelineAndRayGen/devicePrograms.slang",
        "source": (
            """
            int frameID;
            RWStructuredBuffer<uint> colorBuffer;
            int2 fbSize;

            [shader("closesthit")]
            void closesthit_radiance()
            {
            }

            [shader("anyhit")]
            void anyhit_radiance()
            {
            }

            [shader("miss")]
            void miss_radiance()
            {
            }

            [shader("raygeneration")]
            void renderFrame()
            {
                const int ix = DispatchRaysIndex().x;
                const int iy = DispatchRaysIndex().y;
                const int r = (ix % 256);
                const int g = (iy % 256);
                const int b = ((ix + iy) % 256);
                const uint rgba = 0xff000000
                    | (r << 0) | (g << 8) | (b << 16);
                const uint fbIndex = ix + iy * fbSize.x;
                colorBuffer[fbIndex] = rgba;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "ray_closest_hit {",
            "ray_any_hit {",
            "ray_miss {",
            "ray_generation {",
            "int b = (ix + iy) % 256;",
            "uint rgba = 0xff000000 | r << 0 | g << 8 | b << 16;",
            "colorBuffer[fbIndex] = rgba;",
        ],
    },
    {
        "id": "slang_examples_ray_payload_access_semantics",
        "repo": "shader-slang/slang-examples",
        "path": "examples/ray-tracing-pipeline/shaders.slang",
        "source": (
            """
            [raypayload] struct RayPayload
            {
                float4 color : read(caller) : write(caller, closesthit, miss);
            };

            RWTexture2D resultTexture;

            [shader("raygeneration")]
            void rayGenShader()
            {
                RayPayload payload = { float4(0, 0, 0, 0) };
                resultTexture[uint2(0, 0)] = payload.color;
            }

            [shader("miss")]
            void missShader(inout RayPayload payload)
            {
                payload.color = float4(0, 0, 0, 1);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "vec4 color @ ray_payload_read(caller) @ ray_payload_write(caller,closesthit,miss);",
            "ray_generation {",
            "resultTexture[uvec2(0, 0)] = payload.color;",
            "payload.color = vec4(0, 0, 0, 1);",
        ],
        "not_contains": ["@ read(caller):write"],
    },
    {
        "id": "falcor_pixel_stats_resource_array",
        "repo": "NVIDIAGameWorks/Falcor",
        "path": "Source/Falcor/Rendering/Utils/PixelStats.cs.slang",
        "source": (
            """
            import PixelStatsShared;

            cbuffer CB
            {
                uint2 gFrameDim;
            }

            Texture2D<uint> gStatsRayCount[(uint)PixelStatsRayType::Count];
            RWTexture2D<uint> gStatsRayCountTotal;

            [numthreads(16, 16, 1)]
            void main(uint3 dispatchThreadId : SV_DispatchThreadID)
            {
                uint2 pixel = dispatchThreadId.xy;
                if (any(pixel > gFrameDim)) return;

                uint totalRays = 0;
                for (uint i = 0; i < (uint)PixelStatsRayType::Count; i++)
                {
                    totalRays += gStatsRayCount[i][pixel];
                }
                gStatsRayCountTotal[pixel] = totalRays;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "import PixelStatsShared;",
            "sampler2D gStatsRayCount[uint(PixelStatsRayType::Count)];",
            "image2D gStatsRayCountTotal;",
            "layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;",
            "totalRays += gStatsRayCount[i][pixel];",
        ],
    },
    {
        "id": "vireo_samples_constantbuffer_gbuffer_sampling",
        "repo": "HenriMichelon/vireo_samples",
        "path": "src/shaders/deferred_lighting.frag.slang",
        "source": (
            """
            struct VertexOutput {
                float4 position : SV_POSITION;
                float2 uv : TEXCOORD;
            };

            struct Global {
                float exposure;
            };

            struct Light {
                float3 color;
            };

            ConstantBuffer<Global> global : register(b0);
            ConstantBuffer<Light> light : register(b1);
            Texture2D positionBuffer : register(t2);
            SamplerState sampler : register(SAMPLER_NEAREST_BORDER, space1);

            float3 calcLighting(Global global, Light light, float3 worldPos)
            {
                return light.color * global.exposure;
            }

            float4 fragmentMain(VertexOutput input) : SV_TARGET
            {
                float3 worldPos =
                    positionBuffer.Sample(sampler, input.uv).rgb;
                float3 lit = calcLighting(global, light, worldPos);
                return float4(lit, 1.0);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "ConstantBuffer<Global> global_ @register(b0);",
            "ConstantBuffer<Light> light @register(b1);",
            "sampler2D positionBuffer @register(t2);",
            "sampler sampler_;",
            "vec3 worldPos = texture(positionBuffer, sampler_, input.uv).rgb;",
            "vec3 lit = calcLighting(global_, light, worldPos);",
            "return vec4(lit, 1.0);",
        ],
        "not_contains": [
            "@register(SAMPLER_NEAREST_BORDER, space1)",
            "positionBuffer.Sample",
            "global.exposure",
        ],
    },
    {
        "id": "vk_slang_editor_mandelbrot_inout_texture_write",
        "repo": "nvpro-samples/vk_slang_editor",
        "path": "examples/Basics/Mandelbrot.slang",
        "source": (
            """
            RWTexture2D<float4> texFrame;
            uniform float2 iResolution;

            bool mandelbrot(inout float2 z, out int i)
            {
                const float2 c = z;
                for(i = 0; i < 64; i++)
                {
                    if(dot(z, z) > 256.0)
                    {
                        return true;
                    }
                    z = c + float2(z.x * z.x - z.y * z.y, 2 * z.x * z.y);
                }
                return false;
            }

            [shader("compute")]
            [numthreads(16, 16, 1)]
            void render(uint2 thread: SV_DispatchThreadID)
            {
                float2 uv =
                    (float2(thread) - .5 * iResolution.xy) / iResolution.y;
                float2 z = 2.5 * uv - float2(.5, 0);
                int i;
                float3 color = float3(0.0);
                if(mandelbrot(z, i))
                {
                    color = 0.5 + 0.5 * sin(
                        i + float3(0, .5, 1) - log2(log2(dot(z,z))));
                }
                texFrame[thread] = float4(color, 1.0);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "image2D texFrame;",
            "bool mandelbrot(vec2 z, int i)",
            "layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;",
            "texFrame[thread] = vec4(color, 1.0);",
        ],
    },
    {
        "id": "vk_slang_editor_rasterization_semantics",
        "repo": "nvpro-samples/vk_slang_editor",
        "path": "examples/Basics/Rasterization.slang",
        "source": (
            """
            uniform float4x4 iViewProjection;

            struct Vertex {
                float3 position : POSITION;
                float3 normal : NORMAL;
                float4 tangent : TANGENT;
                float2 uv0 : TEXCOORD0;
            };
            struct VsOutput {
                Vertex vertex;
                float4 svPosition : SV_Position;
            };

            [shader("vertex")]
            VsOutput vertexMain(Vertex input)
            {
                VsOutput o;
                o.vertex = input;
                o.svPosition = mul(float4(o.vertex.position, 1.0),
                                   iViewProjection);
                return o;
            }

            [shader("fragment")]
            float4 fragmentMain(Vertex v,
                                float2 fragCoord : SV_Position,
                                float3 barycentrics : SV_Barycentrics,
                                uint triangleId : SV_PrimitiveID)
            {
                uint2 xy = uint2(fragCoord);
                bool sierpinski = ((xy.x & xy.y) == 0);
                float3 color = float3(sierpinski);
                return float4(color, 1.0);
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "struct Vertex {",
            "vec3 position @ in_Position;",
            "vec4 svPosition @ Out_Position;",
            "vec3 barycentrics @ gl_BaryCoordEXT",
            "bool sierpinski = (xy.x & xy.y) == 0;",
        ],
    },
    {
        "id": "slang_rhi_root_parameter_block_attributes",
        "repo": "shader-slang/slang-rhi",
        "path": "tests/test-root-shader-parameter.slang",
        "source": (
            """
            [__AttributeUsage(_AttributeTargets.Var)]
            struct rootAttribute {};

            struct S1
            {
                StructuredBuffer<uint> c0;
                [root] RWStructuredBuffer<uint> c1;
            }

            struct S0
            {
                StructuredBuffer<uint> b0;
                [root] StructuredBuffer<uint> b1;
                ParameterBlock<S1> s1;
                ConstantBuffer<S1> s2;
            }

            ParameterBlock<S0> g;
            [root] RWStructuredBuffer<uint> buffer;

            [shader("compute")]
            [numthreads(1,1,1)]
            void computeMain(uint3 sv_dispatchThreadID : SV_DispatchThreadID)
            {
                buffer[0] = g.b0[0] - g.b1[0] + g.s1.c0[0]
                    - g.s1.c1[0] + g.s2.c0[0];
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "struct rootAttribute {",
            "ParameterBlock<S0> g;",
            "RWStructuredBuffer<uint> buffer_;",
            "buffer_[0] = g.b0[0] - g.b1[0] + g.s1.c0[0] - g.s1.c1[0] + g.s2.c0[0];",
        ],
        "not_contains": ["[root]", "__AttributeUsage"],
    },
    {
        # Source: https://github.com/shader-slang/slang
        # Commit: 5230a81f2fe68afe5cb8d04a1b09d56476f6b960
        # Path: tests/bugs/gh-3802.slang
        "id": "slang_cpp_style_namespace_path",
        "repo": "shader-slang/slang-current-2026-06-07",
        "path": "tests/bugs/gh-3802.slang",
        "source": (
            """
            namespace foo::bar::baz {}
            namespace foo::bar {}

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void computeMain()
            {}
        """
        ),
        "crossgl": True,
        "contains": [
            "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;",
            "void computeMain()",
        ],
        "not_contains": ["namespace foo", "foo::bar"],
    },
    {
        # Source: https://github.com/shader-slang/slang
        # Commit: 5230a81f2fe68afe5cb8d04a1b09d56476f6b960
        # Path: docs/generated/tests/design/ast-reference/statements/
        # targetswitch-static-dispatch.slang
        "id": "slang_target_switch_default_arm",
        "repo": "shader-slang/slang-current-2026-06-07",
        "path": (
            "docs/generated/tests/design/ast-reference/statements/"
            "targetswitch-static-dispatch.slang"
        ),
        "source": (
            """
            float helper()
            {
                __target_switch
                {
                    case hlsl:
                        return 1.0;
                    default:
                        return 9.0;
                }
            }

            RWStructuredBuffer<float> output;

            [shader("compute")]
            [numthreads(1, 1, 1)]
            void computeMain()
            {
                output[0] = helper();
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "float helper()",
            "return 9.0;",
            "output[0] = helper();",
        ],
        "not_contains": ["__target_switch", "return 1.0;"],
    },
    {
        # Source: https://github.com/shader-slang/slang
        # Commit: 5230a81f2fe68afe5cb8d04a1b09d56476f6b960
        # Path: docs/generated/tests/conformance/basics-program-execution/
        # dispatch-3d-group-coords-functional.slang
        "id": "slang_dispatch_group_thread_relational_condition",
        "repo": "shader-slang/slang-current-2026-06-07",
        "path": (
            "docs/generated/tests/conformance/basics-program-execution/"
            "dispatch-3d-group-coords-functional.slang"
        ),
        "source": (
            """
            RWStructuredBuffer<uint> outputBuffer;

            [numthreads(2, 3, 1)]
            void computeMain(uint3 groupID     : SV_GroupID,
                             uint3 groupThread : SV_GroupThreadID)
            {
                uint idx = groupThread.y * 2u + groupThread.x;
                uint groupInRange =
                    (groupID.x == 0u && groupID.y == 0u && groupID.z == 0u)
                        ? 1u : 0u;
                uint threadInRange =
                    (groupThread.x < 2u && groupThread.y < 3u &&
                     groupThread.z == 0u)
                        ? 1u : 0u;

                outputBuffer[idx] = groupInRange & threadInRange;
            }
        """
        ),
        "crossgl": True,
        "contains": [
            "layout(local_size_x = 2, local_size_y = 3, local_size_z = 1) in;",
            "uvec3 groupID @ gl_WorkGroupID",
            "uvec3 groupThread @ gl_LocalInvocationID",
            (
                "uint threadInRange = (groupThread.x < 2u && "
                "groupThread.y < 3u && groupThread.z == 0u ? 1u : 0u);"
            ),
            "outputBuffer[idx] = groupInRange & threadInRange;",
        ],
    },
]


def parse_slang(source):
    tokens = SlangLexer(source).tokenize()
    return SlangParser(tokens).parse()


def generate_crossgl(ast):
    return SlangCrossGLCodeGen.SlangToCrossGLConverter().generate(ast)


@pytest.mark.parametrize("fixture", EXTERNAL_FIXTURES, ids=lambda item: item["id"])
def test_external_fixture_codegen_crossgl_reparse(fixture):
    ast = parse_slang(fixture["source"])
    generated = generate_crossgl(ast)

    enum_underlying_types = fixture.get("enum_underlying_types", {})
    if enum_underlying_types:
        enum_types = {
            enum.name: getattr(enum, "underlying_type", None)
            for enum in getattr(ast, "enums", [])
        }
        assert enum_types == enum_underlying_types

    expected_registers = fixture.get("registers", {})
    if expected_registers:
        actual_registers = {}
        for node in getattr(ast, "global_vars", []):
            declaration = getattr(node, "left", node)
            name = getattr(declaration, "name", None)
            if name in expected_registers:
                actual_registers[name] = getattr(declaration, "registers", [])
        assert actual_registers == expected_registers

    for expected in fixture.get("contains", []):
        assert expected in generated
    previous_position = -1
    for expected in fixture.get("ordered_contains", []):
        position = generated.find(expected, previous_position + 1)
        assert position != -1
        previous_position = position
    for rejected in fixture.get("not_contains", []):
        assert rejected not in generated

    if fixture.get("crossgl", False):
        cgl_translator.parse(generated)


def test_gfx_public_namespace_static_constants_codegen_crossgl_reparse():
    # Source: shader-slang/slang tools/gfx/gfx.slang at
    # c6f104ca76a54ca1565dac54363ea763dd906de6.
    source = """
        import slang;

        public namespace gfx
        {
        public static const uint64_t kTimeoutInfinite = 0xFFFFFFFFFFFFFFFF;
        public static const uint kWholeSize = 0xFFFFFFFF;
        }
    """

    ast = parse_slang(source)
    generated = generate_crossgl(ast)

    assert [
        getattr(getattr(node, "left", node), "name", None) for node in ast.global_vars
    ] == [
        "gfx.kTimeoutInfinite",
        "gfx.kWholeSize",
    ]
    assert (
        "static const uint64_t gfx_kTimeoutInfinite = 0xFFFFFFFFFFFFFFFF;" in generated
    )
    assert "static const uint gfx_kWholeSize = 0xFFFFFFFF;" in generated
    assert "public namespace" not in generated
    cgl_translator.parse(generated)


def test_wgsl_texture_load_shorthand_where_conformance_constraints_parse():
    # Source: shader-slang/slang tests/wgsl/texture-load.slang at
    # 726e0973b3f547c7729b86f122ff7aef8322bace.
    source = """
        bool TEST_textureLoad<T>(
            Texture1D<T> t1D,
            Texture2D<T> t2D,
            Texture3D<T> t3D,
            Texture2DMS<T> t2DMS,
            Texture2DArray<T> t2DArray)
            where T : ITexelElement, IArithmetic
        {
            typealias Tvn = T;
            return true && all(Tvn(T.Element(0)) == t1D.Load(int2(0, 0)));
        }
    """

    ast = parse_slang(source)
    function = ast.functions[0]

    assert [
        (constraint.parameter, constraint.relation, constraint.constraint_type)
        for constraint in function.generic_constraints
    ] == [
        ("T", ":", "ITexelElement"),
        ("T", ":", "IArithmetic"),
    ]


def test_falcor_exported_import_and_default_interface_parameter_parse():
    source = """
        __exported import Rendering.Materials.IMaterialInstance;
        __exported import Scene.Material.TextureSampler;
        import Rendering.Volumes.PhaseFunction;

        [anyValueSize(128)]
        interface IMaterial
        {
            associatedtype MaterialInstance : IMaterialInstance;

            MaterialInstance setupMaterialInstance(
                const MaterialSystem ms,
                const ShadingData sd,
                const ITextureSampler lod,
                const uint hints = (uint)MaterialInstanceHints::None);
        }
    """

    ast = parse_slang(source)
    method = ast.interfaces[0].methods[0]
    hints = method.params[-1]

    assert [node.module_name for node in ast.imports] == [
        "Rendering.Materials.IMaterialInstance",
        "Scene.Material.TextureSampler",
        "Rendering.Volumes.PhaseFunction",
    ]
    assert ast.imports[0].qualifiers == ["__exported"]
    assert ast.imports[1].qualifiers == ["__exported"]
    assert hints.vtype == "uint"
    assert hints.name == "hints"
    assert hints.value.target_type == "uint"
    assert hints.value.expression.name == "MaterialInstanceHints::None"


def test_generated_type_equality_property_syntax_parse():
    source = """
        interface IFace
        {
            associatedtype PropertyType;
            property prop : PropertyType { get; }
        }

        struct IntProperty : IFace
        {
            typealias PropertyType = int;
            int _val;
            property prop : int { get { return _val; } }
        }

        int addTwoInts<T : IFace>(T a, T b) where T.PropertyType == int
        {
            return a.prop + b.prop;
        }
    """

    ast = parse_slang(source)
    interface_property = ast.interfaces[0].properties[0]
    struct_property = ast.structs[0].members[1]
    function = ast.functions[0]

    assert interface_property.name == "prop"
    assert interface_property.vtype == "PropertyType"
    assert interface_property.property_accessors == {"get": []}
    assert struct_property.name == "prop"
    assert struct_property.vtype == "int"
    assert list(struct_property.property_accessors) == ["get"]
    assert function.generic_constraints[0].parameter == "T.PropertyType"
    assert function.generic_constraints[0].relation == "=="
    assert function.generic_constraints[0].constraint_type == "int"


def test_generated_interface_constructor_requirement_parse():
    # Source: shader-slang/slang@564ac9f050d6569efd773e2f74e7d067a4e54baa
    # docs/generated/tests/conformance/types-interface/
    # interface-constructor-requirement.slang
    source = """
        interface IInitializable
        {
            __init(int v);
            int getValue();
        }

        struct InitStruct : IInitializable
        {
            int stored;

            __init(int v)
            {
                stored = v * 2;
            }

            int getValue() { return stored; }
        }
    """

    ast = parse_slang(source)
    interface = ast.interfaces[0]
    constructor = interface.methods[0]
    get_value = interface.methods[1]
    struct_constructor = ast.structs[0].methods[0]

    assert interface.name == "IInitializable"
    assert constructor.name == "__init"
    assert constructor.return_type == "void"
    assert constructor.is_declaration is True
    assert [(param.vtype, param.name) for param in constructor.params] == [("int", "v")]
    assert get_value.name == "getValue"
    assert get_value.is_declaration is True
    assert struct_constructor.name == "__init"
    assert struct_constructor.is_declaration is False


def test_generated_interface_subscript_requirement_parse():
    # Source: shader-slang/slang@564ac9f050d6569efd773e2f74e7d067a4e54baa
    # docs/generated/tests/conformance/types-interface/
    # interface-subscript-requirement.slang
    source = """
        interface IIndexable
        {
            __subscript(uint i) -> int { get; set; }
        }

        struct FixedArray : IIndexable
        {
            int data[4];

            __subscript(uint i) -> int
            {
                get { return data[i]; }
                set { data[i] = newValue; }
            }
        }
    """

    ast = parse_slang(source)
    requirement = ast.interfaces[0].methods[0]
    implementation = ast.structs[0].methods[0]

    assert requirement.name == "operator[]"
    assert requirement.slang_name == "__subscript"
    assert requirement.is_declaration is True
    assert requirement.return_type == "int"
    assert [(param.vtype, param.name) for param in requirement.params] == [
        ("uint", "i")
    ]
    assert requirement.property_accessors == {"get": [], "set": []}
    assert implementation.name == "operator[]"
    assert implementation.is_declaration is False
    assert set(implementation.property_accessors) == {"get", "set"}


def test_core_meta_interface_static_const_value_requirements_parse():
    # Source: shader-slang/slang source/slang/core.meta.slang at
    # 564ac9f050d6569efd773e2f74e7d067a4e54baa.
    source = """
        interface IRangedValue
        {
            static const This maxValue;
            static const This minValue;
        }
    """

    ast = parse_slang(source)
    interface = ast.interfaces[0]

    assert interface.name == "IRangedValue"
    assert [
        (requirement.qualifiers, requirement.vtype, requirement.name)
        for requirement in interface.value_requirements
    ] == [
        (["static", "const"], "This", "maxValue"),
        (["static", "const"], "This", "minValue"),
    ]
    assert interface.methods == []


def test_core_meta_generic_vector_conversion_constructor_parse():
    # Source: shader-slang/slang source/slang/core.meta.slang at
    # 564ac9f050d6569efd773e2f74e7d067a4e54baa.
    source = """
        extension vector<ToType,N>
        {
            __implicit_conversion(constraint)
            __intrinsic_op(BuiltinCast)
            __init<FromType>(vector<FromType,N> value)
                where ToType(FromType) implicit;

            __implicit_conversion(constraint+)
            [__unsafeForceInlineEarly]
            [__readNone]
            [TreatAsDifferentiable]
            __init<FromType>(FromType value) where ToType(FromType) implicit
            {
                this = __builtin_cast<vector<ToType,N>>(
                    vector<FromType,N>(value));
            }
        }
    """

    ast = parse_slang(source)
    extension = ast.extensions[0]

    assert extension.extended_type == "vector<ToType, N>"
    assert len(extension.methods) == 2
    assert [method.generic_parameters for method in extension.methods] == [
        "<FromType>",
        "<FromType>",
    ]
    assert [
        (
            constraint.parameter,
            constraint.relation,
            constraint.constraint_type,
        )
        for constraint in extension.methods[0].generic_constraints
    ] == [("ToType", "implicit", "FromType")]
    assert extension.methods[0].is_declaration is True
    assert extension.methods[1].is_declaration is False

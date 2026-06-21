import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

import crosstl
import crosstl.translator
import crosstl.translator.codegen as codegen
from crosstl.project import translate_project
from crosstl.translator.ast import ShaderStage
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources
from tests.test_translator.spirv_wgsl_contract import (
    SPIRV_VERTEX_POSITION_OUTPUT_SOURCE,
    assert_spirv_position_output_wgsl_contract,
)

ROOT = Path(__file__).resolve().parents[2]

NATIVE_SOURCE_SNIPPETS = {
    "directx": (
        "shader.hlsl",
        "float4 main(float4 pos : SV_Position) : SV_Target { return pos; }",
    ),
    "opengl": (
        "shader.glsl",
        """
        #version 450 core
        layout(location = 0) out vec4 fragColor;
        void main() { fragColor = vec4(1.0); }
        """,
    ),
    "metal": (
        "shader.metal",
        """
        #include <metal_stdlib>
        using namespace metal;
        fragment float4 fragment_main() { return float4(1.0); }
        """,
    ),
    "slang": (
        "shader.slang",
        "float4 main(float4 pos : SV_Position) : SV_Target { return pos; }",
    ),
    "cuda": (
        "shader.cu",
        "__global__ void main(float* out) { int i = threadIdx.x; out[i] = 1.0f; }",
    ),
    "hip": (
        "shader.hip",
        "__global__ void main(float* out) { int i = threadIdx.x; out[i] = 1.0f; }",
    ),
    "opencl": (
        "shader.cl",
        "kernel void main(global float* out) { uint i = get_global_id(0); out[i] = 1.0f; }",
    ),
    "mojo": ("shader.mojo", "fn main() -> None:\n    return\n"),
    "rust": ("shader.rs", "fn main() { let x: f32 = 1.0; }"),
    "vulkan": (
        "shader.spvasm",
        """
        void main() {
            vec4 color = vec4(1.0, 0.0, 0.0, 1.0);
            gl_FragColor = color;
        }
        """,
    ),
}

NATIVE_SOURCE_EXTENSION_ALIAS_SNIPPETS = {
    "cuda_header": (
        "shader.cuh",
        NATIVE_SOURCE_SNIPPETS["cuda"][1],
    ),
    "cuda_long": (
        "shader.cuda",
        NATIVE_SOURCE_SNIPPETS["cuda"][1],
    ),
    "rust_long": (
        "shader.rust",
        NATIVE_SOURCE_SNIPPETS["rust"][1],
    ),
    "slang_header": (
        "shader.slangh",
        NATIVE_SOURCE_SNIPPETS["slang"][1],
    ),
    "metal_msl": (
        "shader.msl",
        NATIVE_SOURCE_SNIPPETS["metal"][1],
    ),
    "opencl_long": (
        "shader.opencl",
        NATIVE_SOURCE_SNIPPETS["opencl"][1],
    ),
}

WEBGL_NATIVE_COMPUTE_SOURCE_DIAGNOSTICS = frozenset({"cuda", "hip", "opencl"})
WEBGL_NATIVE_COMPUTE_EXTENSION_ALIAS_DIAGNOSTICS = frozenset(
    {"cuda_header", "cuda_long", "opencl_long"}
)
NATIVE_SOURCE_TO_TARGET_PIPELINE_CASES = tuple(
    (source_name, target_backend)
    for source_name in sorted(NATIVE_SOURCE_SNIPPETS)
    for target_backend in codegen.backend_names()
    if not (
        target_backend == "webgl"
        and source_name in WEBGL_NATIVE_COMPUTE_SOURCE_DIAGNOSTICS
    )
)
NATIVE_SOURCE_EXTENSION_ALIAS_TO_TARGET_PIPELINE_CASES = tuple(
    (alias_name, target_backend)
    for alias_name in sorted(NATIVE_SOURCE_EXTENSION_ALIAS_SNIPPETS)
    for target_backend in codegen.backend_names()
    if not (
        target_backend == "webgl"
        and alias_name in WEBGL_NATIVE_COMPUTE_EXTENSION_ALIAS_DIAGNOSTICS
    )
)

GLSL_FRAGMENT_INVOCATION_DENSITY_SOURCE = """
#version 450 core
#extension GL_EXT_fragment_invocation_density : require
layout(location = 0) out vec4 fragColor;

void main() {
    float h = (
        clamp(
            1.0 - 1.0 / float(gl_FragSizeEXT.x * gl_FragSizeEXT.y),
            0.0,
            1.0
        )
    ) / 1.35;
    fragColor = vec4(h);
}
"""

GLSL_FRAGMENT_INVOCATION_DENSITY_HELPER_SOURCE = """
#version 450 core
#extension GL_EXT_fragment_invocation_density : require
layout(location = 0) out vec4 fragColor;

vec4 FragmentDensityToColor() {
    float h = (
        clamp(
            1.0 - 1.0 / float(gl_FragSizeEXT.x * gl_FragSizeEXT.y),
            0.0,
            1.0
        )
    ) / 1.35;
    return vec4(h, h, h, 1.0);
}

void main() {
    fragColor = FragmentDensityToColor();
}
"""


def _write_source(tmp_path, filename, source):
    path = tmp_path / filename
    path.write_text(source, encoding="utf-8")
    return path


def _assert_generated_output_is_usable(generated):
    assert isinstance(generated, str)
    assert generated.strip()
    assert "Traceback" not in generated
    assert "NotImplemented" not in generated
    assert "<crosstl." not in generated


def _compile_glslang_if_available(source: str, stage: str) -> None:
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        return

    stage_name = {"vertex": "vert", "fragment": "frag", "compute": "comp"}[stage]
    suffix = {"vertex": ".vert", "fragment": ".frag", "compute": ".comp"}[stage]
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = Path(temp_dir) / f"shader{suffix}"
        source_path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [glslang, "-S", stage_name, str(source_path)],
            capture_output=True,
            check=False,
            text=True,
        )

    assert result.returncode == 0, result.stderr or result.stdout


def _compile_metal_if_available(source: str) -> None:
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return

    lookup = subprocess.run(
        [xcrun, "-f", "metal"],
        capture_output=True,
        check=False,
        text=True,
    )
    if lookup.returncode != 0:
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = Path(temp_dir) / "shader.metal"
        output_path = Path(temp_dir) / "shader.air"
        source_path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [xcrun, "metal", str(source_path), "-o", str(output_path)],
            capture_output=True,
            check=False,
            text=True,
        )

    assert result.returncode == 0, result.stderr


def _compile_with_metal_if_available(source: str, tmp_path: Path):
    _ = tmp_path
    _compile_metal_if_available(source)


def test_cgl_translate_save_shader_preserves_source_line_endings(tmp_path):
    source_text = (
        "shader LineEndingSmoke {\r\n"
        "    float helper(float value) {\r\n"
        "        return value + 1.0;\r\n"
        "    }\r\n"
        "}\r\n"
    )
    source_path = tmp_path / "line-ending-smoke.cgl"
    output_path = tmp_path / "out.cgl"
    source_path.write_bytes(source_text.encode("utf-8"))

    generated = crosstl.translate(
        str(source_path),
        backend="cgl",
        save_shader=str(output_path),
        format_output=False,
    )

    assert generated == source_text
    assert output_path.read_bytes() == source_path.read_bytes()


def test_root_package_exposes_documented_registry_helpers():
    assert {"supported_backends", "supported_sources", "translate", "project"}.issubset(
        crosstl.__all__
    )
    assert "project" in dir(crosstl)
    assert crosstl.supported_backends() == crosstl.translator.supported_backends()
    assert crosstl.supported_sources() == crosstl.translator.supported_sources()
    assert crosstl.project.translate_project is translate_project
    assert "opengl" in crosstl.supported_backends()
    assert "cgl" in crosstl.supported_sources()


def test_registered_native_sources_have_reverse_codegen_factories():
    register_default_sources()

    for backend in codegen.source_backend_names():
        backend_spec = codegen.get_backend(backend)
        source_name = backend_spec.source_registry_name if backend_spec else backend
        spec = SOURCE_REGISTRY.get(source_name)
        assert spec is not None
        assert spec.reverse_codegen_factory is not None

        converter = spec.reverse_codegen_factory()
        assert callable(getattr(converter, "generate", None))


def test_source_registry_reports_lexer_option_support():
    register_default_sources()

    expected_define_support = {
        "cgl": True,
        "cuda": True,
        "directx": True,
        "hip": True,
        "metal": True,
        "mojo": False,
        "opengl": True,
        "rust": False,
        "slang": True,
        "vulkan": True,
    }
    expected_include_support = {
        "cgl": True,
        "cuda": True,
        "directx": True,
        "hip": True,
        "metal": True,
        "mojo": False,
        "opengl": True,
        "rust": False,
        "slang": True,
        "vulkan": True,
    }

    for source_name, supports_defines in expected_define_support.items():
        spec = SOURCE_REGISTRY.get(source_name)
        assert spec is not None
        assert spec.supports_lexer_keyword("defines") is supports_defines
        assert (
            spec.supports_lexer_keyword("include_paths")
            is expected_include_support[source_name]
        )


@pytest.mark.parametrize("backend", codegen.backend_names())
def test_cgl_translate_api_accepts_registered_backend_aliases(tmp_path, backend):
    spec = codegen.get_backend(backend)
    aliases = spec.aliases if spec else ()
    if not aliases:
        pytest.skip(f"{backend} has no aliases")

    source_path = _write_source(
        tmp_path,
        "alias-smoke.cgl",
        """
        shader AliasSmoke {
            float helper(float value) {
                return value + 1.0;
            }
        }
        """,
    )

    for alias in aliases:
        generated = crosstl.translate(
            str(source_path), backend=alias, format_output=False
        )
        _assert_generated_output_is_usable(generated)


@pytest.mark.parametrize(
    ("backend_argument", "expected_snippet"),
    (
        (".hlsl", "float helper(float value)"),
        ("out.HLSL", "float helper(float value)"),
        (".glsl", "float helper(float value)"),
        ("kernel.CU", "__device__ float helper(float value)"),
        (".rs", "fn helper(value: f32) -> f32"),
    ),
)
def test_cgl_translate_api_accepts_target_extension_backend_spellings(
    tmp_path, backend_argument, expected_snippet
):
    source_path = _write_source(
        tmp_path,
        "extension-target.cgl",
        """
        shader ExtensionTarget {
            float helper(float value) {
                return value + 1.0;
            }
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend=backend_argument, format_output=False
    )

    _assert_generated_output_is_usable(generated)
    assert expected_snippet in generated


@pytest.mark.parametrize("source_name", sorted(NATIVE_SOURCE_SNIPPETS))
def test_native_sources_translate_to_parseable_crossgl(tmp_path, source_name):
    filename, source = NATIVE_SOURCE_SNIPPETS[source_name]
    source_path = _write_source(tmp_path, filename, source)

    generated = crosstl.translate(str(source_path), backend="cgl", format_output=False)

    _assert_generated_output_is_usable(generated)
    crosstl.translator.parse(generated)


@pytest.mark.parametrize("alias_name", sorted(NATIVE_SOURCE_EXTENSION_ALIAS_SNIPPETS))
def test_native_source_extension_aliases_translate_to_parseable_crossgl(
    tmp_path, alias_name
):
    filename, source = NATIVE_SOURCE_EXTENSION_ALIAS_SNIPPETS[alias_name]
    source_path = _write_source(tmp_path, filename, source)

    generated = crosstl.translate(str(source_path), backend="cgl", format_output=False)

    _assert_generated_output_is_usable(generated)
    crosstl.translator.parse(generated)


def test_hip_bit_extract_kernel_body_survives_metal_and_spirv_targets(tmp_path):
    source_path = _write_source(
        tmp_path,
        "bit_extract.hip",
        """
        #include <hip/hip_runtime.h>
        #include <iostream>

        __global__ void bit_extract_kernel(
            uint32_t* d_output,
            const uint32_t* d_input,
            size_t size
        ) {
            const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
            const size_t stride = blockDim.x * gridDim.x;
            for (size_t i = offset; i < size; i += stride) {
                d_output[i] = ((d_input[i] >> 8) & 0xfu);
            }
        }

        int main() {
            constexpr size_t size = 1024;
            uint32_t *d_input, *d_output;
            hipMalloc(&d_input, size * sizeof(uint32_t));
            hipMalloc(&d_output, size * sizeof(uint32_t));
            hipMemcpy(d_input, d_output, size, hipMemcpyHostToDevice);
            bit_extract_kernel<<<dim3(1), dim3(256), 0, hipStreamDefault>>>(
                d_output,
                d_input,
                size
            );
            hipDeviceSynchronize();
            std::cout << "done" << std::endl;
            hipFree(d_input);
            hipFree(d_output);
        }
        """,
    )

    metal = crosstl.translate(
        str(source_path),
        backend="metal",
        source_backend="hip",
        format_output=False,
    )
    spirv = crosstl.translate(
        str(source_path),
        backend="vulkan",
        source_backend="hip",
        format_output=False,
    )

    _assert_generated_output_is_usable(metal)
    assert "device uint* d_output [[buffer(0)]]" in metal
    assert "const device uint* d_input [[buffer(1)]]" in metal
    assert "constant uint& size [[buffer(2)]]" in metal
    assert "for (uint i = offset; i < size; i += stride)" in metal
    assert "d_output[i] = d_input[i] >> 8 & 15u;" in metal

    _assert_generated_output_is_usable(spirv)
    assert "OpEntryPoint GLCompute" in spirv
    assert "OpName %" in spirv and '"d_outputBuffer"' in spirv
    assert '"d_inputBuffer"' in spirv
    assert "OpLoopMerge" in spirv
    assert "OpShiftRightLogical" in spirv
    assert "OpBitwiseAnd" in spirv
    assert spirv.index("OpLoopMerge") < spirv.index("OpShiftRightLogical")
    assert spirv.index("OpShiftRightLogical") < spirv.index("OpBitwiseAnd")
    assert spirv.rindex("OpStore") < spirv.rindex("OpReturn")

    for shader_artifact in (metal, spirv):
        assert "hipMalloc" not in shader_artifact
        assert "hipMemcpy" not in shader_artifact
        assert "hipDeviceSynchronize" not in shader_artifact
        assert "std::cout" not in shader_artifact


def test_metal_vertex_struct_return_preserves_view_dir_outputs(tmp_path):
    source_path = _write_source(
        tmp_path,
        "xhalf-view-dir.metal",
        """
        struct Camera { float4x4 invViewMatrix; };
        struct Input { float3 position; };
        struct Output { xhalf3 viewDir; };

        vertex Output main_vertex(Input in [[stage_in]],
                                  constant Camera& camera [[buffer(0)]]) {
            Output out;
            out.viewDir = (xhalf3)normalize(
                camera.invViewMatrix[3].xyz - in.position
            );
            return out;
        }
        """,
    )

    opengl = crosstl.translate(str(source_path), backend="opengl", format_output=False)
    directx = crosstl.translate(
        str(source_path), backend="directx", format_output=False
    )
    spirv = crosstl.translate(str(source_path), backend="vulkan", format_output=False)

    _assert_generated_output_is_usable(opengl)
    assert "out vec3 viewDir;" in opengl
    assert "viewDir = vec3(normalize" in opengl
    assert "xhalf" not in opengl
    assert "f16vec3" not in opengl

    _assert_generated_output_is_usable(directx)
    assert "half3 viewDir: TEXCOORD0;" in directx
    assert "out_.viewDir = half3(normalize" in directx
    assert "return out_;" in directx
    assert "return Output(half3(0));" not in directx

    _assert_generated_output_is_usable(spirv)
    output_match = re.search(
        r'OpName %(\d+) "CrossGL_vertex_output_main_vertex_viewDir"', spirv
    )
    assert output_match is not None
    output_id = output_match.group(1)
    stored_values = re.findall(rf"OpStore %{output_id} %(\d+)", spirv)
    assert stored_values
    assert re.search(rf"%{stored_values[-1]} = OpCompositeExtract %\d+ %\d+ 0", spirv)
    assert "Normalize" in spirv
    assert "Unknown type f16vec3" not in spirv
    assert "cannot lower unknown function 'f16vec3'" not in spirv


@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan"])
def test_metal_numeric_heavy_generated_identifier_translates(tmp_path, backend):
    source_path = _write_source(
        tmp_path,
        "numeric-generated-helper.metal",
        """
        #include <metal_stdlib>
        using namespace metal;

        float nvfp4_quantize_float_gs_16_b_4(float value) {
            return value + 1.0;
        }

        kernel void k(device float* out [[buffer(0)]]) {
            out[0] = nvfp4_quantize_float_gs_16_b_4(2.0);
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend=backend, format_output=False
    )

    _assert_generated_output_is_usable(generated)
    if backend != "vulkan":
        assert "nvfp4_quantize_float_gs_16_b_4" in generated


def test_metal_template_helper_specializes_to_concrete_opengl(tmp_path):
    source_path = _write_source(
        tmp_path,
        "template-helper.metal",
        """
        #include <metal_stdlib>
        using namespace metal;

        template <typename T, typename U>
        METAL_FUNC T ceildiv(T N, U M) {
            T local = N;
            return (local + M - 1) / M;
        }

        kernel void k(device uint* out [[buffer(0)]]) {
            out[0] = ceildiv(4u, 2u);
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend="opengl", format_output=False
    )

    _assert_generated_output_is_usable(generated)
    assert "uint ceildiv_uint_uint(uint N, uint M)" in generated
    assert "uint local = N;" in generated
    assert "ceildiv_uint_uint(4u, 2u)" in generated
    assert not re.search(r"\b(?:T|U|IdxT)\b", generated)


def test_metal_explicit_template_helper_specializes_to_concrete_opengl(tmp_path):
    source_path = _write_source(
        tmp_path,
        "explicit-template-helper.metal",
        """
        #include <metal_stdlib>
        using namespace metal;

        template <typename T, typename IdxT, int Offset>
        METAL_FUNC T add_offset(T value, IdxT index) {
            return value + T(index + Offset);
        }

        kernel void k(device float* out [[buffer(0)]],
                      uint gid [[thread_position_in_grid]]) {
            out[gid] = add_offset<float, uint, 7>(1.0, gid);
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend="opengl", format_output=False
    )

    _assert_generated_output_is_usable(generated)
    assert "float add_offset_float_uint_7(float value, uint index)" in generated
    assert "return (value + float((index + 7)));" in generated
    assert "add_offset_float_uint_7(1.0, gid)" in generated
    assert not re.search(r"\b(?:T|IdxT|Offset)\b", generated)


def test_metal_uint2_dispatch_id_promotes_to_directx_uint3(tmp_path):
    source_path = _write_source(
        tmp_path,
        "mat-mul-dispatch-id.metal",
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void main(uint2 id [[thread_position_in_grid]]) {
            uint row = id.y;
            uint col = id.x;
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend="directx", format_output=False
    )

    _assert_generated_output_is_usable(generated)
    assert "uint3 id_dispatchThreadID : SV_DispatchThreadID" in generated
    assert "uint2 id = id_dispatchThreadID.xy;" in generated
    assert "uint row = id.y;" in generated
    assert "uint col = id.x;" in generated
    assert "uint2 id : SV_DispatchThreadID" not in generated


def test_metal_scalar_vector_constructor_lowers_to_directx_splat(tmp_path):
    source_path = _write_source(
        tmp_path,
        "metal-scalar-vector-constructor.metal",
        """
        #include <metal_stdlib>
        using namespace metal;

        struct Input {
            float3 position [[attribute(0)]];
        };

        struct Output {
            half3 viewDir [[user(TEXCOORD0)]];
        };

        vertex Output VSMain(Input in [[stage_in]]) {
            half scalar = half(0);
            Output literalValue = Output(half3(0));
            Output scalarValue = Output(half3(scalar));
            Output nestedValue = Output(half3(half(0)));
            return scalarValue;
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend="directx", format_output=False
    )

    _assert_generated_output_is_usable(generated)
    assert "Output literalValue = Output(half3(0, 0, 0));" in generated
    assert "Output scalarValue = Output(half3(scalar, scalar, scalar));" in generated
    assert "Output nestedValue = Output(half3(half(0), half(0), half(0)));" in generated
    assert "half3(0))" not in generated
    assert "half3(scalar))" not in generated
    assert "half3(half(0)))" not in generated


def test_metal_threadgroup_scratch_lowers_to_directx_groupshared(tmp_path):
    source_path = _write_source(
        tmp_path,
        "mlx-threadgroup-scratch.metal",
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void mat_mul(
            device float* out [[buffer(0)]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            threadgroup float scratch[256];
            scratch[tid] = 1.0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            out[tid] = scratch[tid];
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend="directx", format_output=False
    )

    _assert_generated_output_is_usable(generated)
    assert "groupshared float mat_mul_scratch[256];" in generated
    assert "threadgroup float" not in generated
    assert "    groupshared float scratch[256];" not in generated
    assert "mat_mul_scratch[tid] = 1.0;" in generated
    assert "out_[tid] = mat_mul_scratch[tid];" in generated
    assert "GroupMemoryBarrierWithGroupSync();" in generated
    assert generated.index("groupshared float mat_mul_scratch[256];") < generated.index(
        "void CSMain"
    )


def test_metal_constant_reference_parameter_lowers_to_directx_constant_buffer(
    tmp_path,
):
    source_path = _write_source(
        tmp_path,
        "mat-mul-params.metal",
        """
        #include <metal_stdlib>
        using namespace metal;

        struct MatMulParams {
            uint row_dim_x;
            uint col_dim_x;
            uint inner_dim;
        };

        kernel void mat_mul_simple1(
            device float* A [[buffer(0)]],
            device float* B [[buffer(1)]],
            device float* X [[buffer(2)]],
            constant MatMulParams& params [[buffer(3)]],
            uint2 id [[thread_position_in_grid]]
        ) {
            const uint row_dim_x = params.row_dim_x;
            const uint col_dim_x = params.col_dim_x;
            const uint inner_dim = params.inner_dim;
            uint row = id.y;
            uint col = id.x;
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend="directx", format_output=False
    )

    _assert_generated_output_is_usable(generated)
    assert "#include <metal_stdlib>" not in generated
    assert "RWStructuredBuffer<float> A : register(u0);" in generated
    assert "RWStructuredBuffer<float> B : register(u1);" in generated
    assert "RWStructuredBuffer<float> X : register(u2);" in generated
    assert "ConstantBuffer<MatMulParams> params : register(b3);" in generated
    assert "void CSMain(uint3 id_dispatchThreadID : SV_DispatchThreadID)" in generated
    assert "void CSMain(float* A" not in generated
    assert "RWStructuredBuffer<float> A," not in generated
    assert "uint3 id_dispatchThreadID : SV_DispatchThreadID" in generated
    assert "uint2 id = id_dispatchThreadID.xy;" in generated
    assert "const uint row_dim_x = params.row_dim_x;" in generated
    assert "const uint col_dim_x = params.col_dim_x;" in generated
    assert "const uint inner_dim = params.inner_dim;" in generated
    assert "ReferenceType" not in generated
    assert "MatMulParams&" not in generated


def test_metal_constant_reference_parameter_lowers_to_opengl_uniform_block(
    tmp_path,
):
    source_path = _write_source(
        tmp_path,
        "mat-mul-params.metal",
        """
        #include <metal_stdlib>
        using namespace metal;

        struct MatMulParams {
            uint row_dim_x;
            uint col_dim_x;
            uint inner_dim;
        };

        kernel void mat_mul_simple1(
            device float* A [[buffer(0)]],
            device float* B [[buffer(1)]],
            device float* X [[buffer(2)]],
            constant MatMulParams& params [[buffer(3)]],
            uint2 id [[thread_position_in_grid]]
        ) {
            const uint row_dim_x = params.row_dim_x;
            const uint col_dim_x = params.col_dim_x;
            const uint inner_dim = params.inner_dim;
            uint row = id.y;
            uint col = id.x;
            X[(row * col_dim_x) + col] = row_dim_x + inner_dim;
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend="opengl", format_output=False
    )

    _assert_generated_output_is_usable(generated)
    assert "layout(std430, binding = 0) buffer ABuffer { float A[]; };" in generated
    assert "layout(std430, binding = 1) buffer BBuffer { float B[]; };" in generated
    assert "layout(std430, binding = 2) buffer XBuffer { float X[]; };" in generated
    assert "layout(std140, binding = 3) uniform MatMulParams {" in generated
    assert "} params;" in generated
    assert "void main()" in generated
    assert "const uint row_dim_x = params.row_dim_x;" in generated
    assert "const uint col_dim_x = params.col_dim_x;" in generated
    assert "const uint inner_dim = params.inner_dim;" in generated
    assert "ReferenceType" not in generated
    assert "MatMulParams&" not in generated
    assert "constant MatMulParams" not in generated
    _compile_glslang_if_available(generated, "compute")


def test_metal_scalar_dispatch_id_promotes_to_directx_uint3(tmp_path):
    source_path = _write_source(
        tmp_path,
        "scalar-dispatch-id.metal",
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void main(uint index [[thread_position_in_grid]]) {
            uint value = index;
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend="directx", format_output=False
    )

    _assert_generated_output_is_usable(generated)
    assert "uint3 index_dispatchThreadID : SV_DispatchThreadID" in generated
    assert "uint index = index_dispatchThreadID.x;" in generated
    assert "uint value = index;" in generated
    assert "uint index : SV_DispatchThreadID" not in generated


def test_metal_multi_entry_resource_names_are_scoped_for_directx(tmp_path):
    source_path = _write_source(
        tmp_path,
        "mlx-multi-entry-resource-scope.metal",
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void read_kernel(
            device const float* C [[buffer(0)]],
            device const float* inp [[buffer(1)]],
            uint index [[thread_position_in_grid]]) {
            float v = C[index] + inp[index];
        }

        kernel void write_kernel(
            device uint* C [[buffer(2)]],
            device uint* inp [[buffer(3)]],
            uint index [[thread_position_in_grid]]) {
            C[index] = inp[index];
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend="directx", format_output=False
    )

    _assert_generated_output_is_usable(generated)
    assert "StructuredBuffer<float> C" in generated
    assert "StructuredBuffer<float> inp" in generated
    assert "RWStructuredBuffer<uint> write_kernel_C" in generated
    assert "RWStructuredBuffer<uint> write_kernel_inp" in generated
    assert "float v = (C.Load(index) + inp.Load(index));" in generated
    assert "write_kernel_C[index] = write_kernel_inp.Load(index);" in generated
    assert "RWStructuredBuffer<uint> C" not in generated
    assert "Conflicting DirectX resource declaration" not in generated


def test_metal_multi_entry_resource_names_are_scoped_for_opengl(tmp_path):
    source_path = _write_source(
        tmp_path,
        "mlx-multi-entry-resource-scope.metal",
        """
        #include <metal_stdlib>
        using namespace metal;

        kernel void read_kernel(
            device const float* C [[buffer(0)]],
            device const float* inp [[buffer(1)]],
            uint index [[thread_position_in_grid]]) {
            float v = C[index] + inp[index];
        }

        kernel void write_kernel(
            device uint* C [[buffer(2)]],
            device uint* inp [[buffer(3)]],
            uint index [[thread_position_in_grid]]) {
            C[index] = inp[index];
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend="opengl", format_output=False
    )

    _assert_generated_output_is_usable(generated)
    assert "readonly buffer CBuffer { float C[]; };" in generated
    assert "readonly buffer inpBuffer { float inp[]; };" in generated
    assert "buffer write_kernel_CBuffer { uint write_kernel_C[]; };" in generated
    assert "buffer write_kernel_inpBuffer { uint write_kernel_inp[]; };" in generated
    assert "float v = (C[index] + inp[index]);" in generated
    assert "write_kernel_C[index] = write_kernel_inp[index];" in generated
    assert "buffer CBuffer { uint C[]; };" not in generated
    assert "Conflicting OpenGL resource declaration" not in generated


def test_slang_non_main_compute_entry_lowers_dispatch_id_to_opengl(tmp_path):
    source_path = _write_source(
        tmp_path,
        "hello-world.slang",
        """
        StructuredBuffer<float> buffer0;
        StructuredBuffer<float> buffer1;
        RWStructuredBuffer<float> result;

        [shader("compute")]
        [numthreads(1,1,1)]
        void computeMain(uint3 threadId : SV_DispatchThreadID)
        {
            uint index = threadId.x;
            result[index] = buffer0[index] + buffer1[index];
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend="opengl", format_output=False
    )

    _assert_generated_output_is_usable(generated)
    assert (
        "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;" in generated
    )
    assert "void main()" in generated
    assert "uint index = gl_GlobalInvocationID.x;" in generated
    assert "result[index] = (buffer0[index] + buffer1[index]);" in generated
    assert "threadId gl_GlobalInvocationID" not in generated
    assert "SV_DispatchThreadID" not in generated


@pytest.mark.parametrize(
    ("target_backend", "expected_call", "rejected_call"),
    (
        ("opengl", "helper(val, 16)", "helper(val);"),
        ("metal", "helper(val, 16)", "helper(val);"),
        ("directx", "helper(val, 16)", "helper(val);"),
    ),
)
def test_slang_default_parameter_calls_lower_for_target_codegen(
    tmp_path, target_backend, expected_call, rejected_call
):
    source_path = _write_source(
        tmp_path,
        "default-parameter.slang",
        """
        RWStructuredBuffer<int> result;

        int helper(int val, int a = 16)
        {
            return val + a;
        }

        [shader("compute")]
        [numthreads(1,1,1)]
        void computeMain(uint3 threadId : SV_DispatchThreadID)
        {
            int val = int(threadId.x);
            result[threadId.x] = helper(val);
            result[threadId.x + 1] = helper(val, 4);
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend=target_backend, format_output=False
    )

    _assert_generated_output_is_usable(generated)
    assert expected_call in generated
    assert rejected_call not in generated
    assert "int helper(int val, int a = 16)" not in generated


def test_cgl_perlin_noise_translates_to_rust_after_default_argument_lowering():
    source_path = ROOT / "examples" / "graphics" / "PerlinNoise.cgl"

    generated = crosstl.translate(str(source_path), backend="rust", format_output=False)

    _assert_generated_output_is_usable(generated)
    assert "pub fn fragment_main(" in generated
    assert "pub fn perlinNoise(" in generated


def test_glsl_frag_source_path_translates_to_fragment_crossgl(tmp_path):
    source_path = _write_source(
        tmp_path,
        "shader.frag",
        """
        #version 450 core
        layout(location = 0) in vec2 vUV;
        layout(location = 0) out vec4 fragColor;
        void main() { fragColor = vec4(vUV, 0.0, 1.0); }
        """,
    )

    generated = crosstl.translate(str(source_path), backend="cgl", format_output=False)

    _assert_generated_output_is_usable(generated)
    shader_ast = crosstl.translator.parse(generated)
    assert ShaderStage.FRAGMENT in shader_ast.stages
    assert ShaderStage.VERTEX not in shader_ast.stages
    assert "fragment {" in generated
    assert "VertexOutput" not in generated
    assert "gl_Position" not in generated


def test_translate_accepts_pathlike_source_paths(tmp_path):
    source_path = _write_source(
        tmp_path,
        "pathlike.FRAG.GLSL",
        """
        #version 450 core
        layout(location = 0) out vec4 fragColor;
        void main() { fragColor = vec4(1.0); }
        """,
    )

    generated = crosstl.translate(source_path, backend="cgl", format_output=False)

    _assert_generated_output_is_usable(generated)
    shader_ast = crosstl.translator.parse(generated)
    assert ShaderStage.FRAGMENT in shader_ast.stages
    assert "fragment {" in generated


def test_spirv_assembly_vertex_position_output_lowers_to_wgsl(tmp_path):
    source_path = _write_source(
        tmp_path, "position.spvasm", SPIRV_VERTEX_POSITION_OUTPUT_SOURCE
    )

    generated = crosstl.translate(str(source_path), backend="wgsl", format_output=False)

    _assert_generated_output_is_usable(generated)
    assert_spirv_position_output_wgsl_contract(generated)


@pytest.mark.parametrize(
    ("filename", "source", "stage", "expected_crossgl"),
    (
        (
            "shader.vertex",
            """
            #version 450 core
            layout(location = 0) in vec2 position;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
            """,
            ShaderStage.VERTEX,
            "VertexOutput main(VertexInput input)",
        ),
        (
            "gbuffers_terrain.vsh",
            """
            #version 450 core
            layout(location = 0) in vec3 position;
            void main() {
                gl_Position = vec4(position, 1.0);
            }
            """,
            ShaderStage.VERTEX,
            "VertexOutput main(VertexInput input)",
        ),
        (
            "deferred8.fsh",
            """
            #version 450 core
            layout(location = 0) out vec4 fragColor;
            void main() {
                fragColor = vec4(1.0);
            }
            """,
            ShaderStage.FRAGMENT,
            "fragment {",
        ),
        (
            "eevee_film_frag.glsl",
            """
            #version 450 core
            layout(location = 0) in vec2 vUV;
            layout(location = 0) out vec4 fragColor;
            void main() {
                fragColor = vec4(vUV, 0.0, 1.0);
            }
            """,
            ShaderStage.FRAGMENT,
            "fragment {",
        ),
    ),
)
def test_glsl_real_world_stage_filename_conventions_translate_to_crossgl(
    tmp_path, filename, source, stage, expected_crossgl
):
    source_path = _write_source(tmp_path, filename, source)

    generated = crosstl.translate(str(source_path), backend="cgl", format_output=False)

    _assert_generated_output_is_usable(generated)
    shader_ast = crosstl.translator.parse(generated)
    assert stage in shader_ast.stages
    assert expected_crossgl in generated


UNSUPPORTED_SOURCE_EXTENSION_DIAGNOSTICS = {
    "shader.spv": "Binary SPIR-V input files",
    "shader.spirv": "Binary SPIR-V input files",
    "shader.air": "Compiled Metal artifacts",
    "shader.metallib": "Compiled Metal artifacts",
    "shader.wgsl": "WGSL/WebGPU source files",
    "shader.wesl": "WGSL/WebGPU source files",
    "shader.wgsl.json": "WGSL/WebGPU source files",
    "shader.spv.json": "Binary SPIR-V input files",
    "shader.metallib.json": "Compiled Metal artifacts",
    "shader.dxil.json": "Compiled DirectX shader binaries",
    "shader.fatbin.json": "Generated CUDA/NVIDIA artifacts",
    "shader.hsaco.json": "Compiled HIP/ROCm artifacts",
    "shader.cso": "Compiled DirectX shader binaries",
    "shader.dxbc": "Compiled DirectX shader binaries",
    "shader.dxil": "Compiled DirectX shader binaries",
    "shader.ptx": "Generated CUDA/NVIDIA artifacts",
    "shader.cubin": "Generated CUDA/NVIDIA artifacts",
    "shader.fatbin": "Generated CUDA/NVIDIA artifacts",
    "shader.hsaco": "Compiled HIP/ROCm artifacts",
}


def test_spirv_source_inference_distinguishes_assembly_from_binary():
    register_default_sources()

    assert SOURCE_REGISTRY.get_by_extension("shader.spvasm").name == "vulkan"
    with pytest.raises(ValueError, match="Binary SPIR-V input files"):
        SOURCE_REGISTRY.get_by_extension("shader.spv")


def test_source_registry_exposes_known_unsupported_extension_diagnostics():
    register_default_sources()

    assert (
        SOURCE_REGISTRY.unsupported_extension_message("shader.spv")
        == "Binary SPIR-V input files (.spv) are not supported; provide SPIR-V "
        "assembly (.spvasm) or disassemble the binary with spirv-dis first."
    )
    assert (
        SOURCE_REGISTRY.unsupported_extension_message("shader.spv.json")
        == "Binary SPIR-V input files (.spv) are not supported; provide SPIR-V "
        "assembly (.spvasm) or disassemble the binary with spirv-dis first."
    )
    assert SOURCE_REGISTRY.unsupported_extension_message("shader.cgl") is None


@pytest.mark.parametrize(
    ("filename", "diagnostic"),
    sorted(UNSUPPORTED_SOURCE_EXTENSION_DIAGNOSTICS.items()),
)
def test_known_unsupported_source_extensions_are_not_source_files(filename, diagnostic):
    register_default_sources()

    with pytest.raises(ValueError, match=re.escape(diagnostic)):
        SOURCE_REGISTRY.get_by_extension(filename)
    assert filename[filename.rfind(".") :] not in SOURCE_REGISTRY.extensions()


@pytest.mark.parametrize(
    ("filename", "diagnostic"),
    sorted(UNSUPPORTED_SOURCE_EXTENSION_DIAGNOSTICS.items()),
)
def test_translate_reports_specific_diagnostic_for_unsupported_source_extensions(
    tmp_path, filename, diagnostic
):
    artifact_path = tmp_path / filename
    artifact_path.write_bytes(b"\x03\x02#shader-binary")

    with pytest.raises(ValueError, match=re.escape(diagnostic)):
        crosstl.translate(str(artifact_path), backend="cgl", format_output=False)


def test_translate_decodes_legacy_hlsl_comment_bytes_with_replacement(tmp_path):
    source_path = tmp_path / "legacy-comment.hlsl"
    source_path.write_bytes(
        b"// Matthias M\xfcller\n"
        b"float4 main(float4 pos : SV_Position) : SV_Target { return pos; }\n"
    )

    generated = crosstl.translate(str(source_path), backend="cgl", format_output=False)

    _assert_generated_output_is_usable(generated)
    crosstl.translator.parse(generated)


@pytest.mark.parametrize("filename", ("library.hlsli", "effect.fx", "effect.fxh"))
def test_directx_real_world_extensions_translate_to_crossgl(tmp_path, filename):
    source_path = _write_source(
        tmp_path,
        filename,
        "float4 main(float4 pos : SV_Position) : SV_Target { return pos; }",
    )

    generated = crosstl.translate(str(source_path), backend="cgl", format_output=False)

    _assert_generated_output_is_usable(generated)
    crosstl.translator.parse(generated)
    assert "fragment {" in generated
    assert "vec4 main(vec4 pos @ gl_FragCoord) @ gl_FragColor" in generated


def test_hlsl_compute_scalar_splat_swizzle_lowers_for_vulkan_and_metal(tmp_path):
    source_path = _write_source(
        tmp_path,
        "scalar-splat.hlsl",
        """
        groupshared int a;
        [numthreads(64, 1, 1)]
        void main() {
            a = 123;
            int4 x = (a).xxxx;
        }
        """,
    )

    crossgl = crosstl.translate(str(source_path), backend="cgl", format_output=False)
    vulkan = crosstl.translate(str(source_path), backend="vulkan", format_output=False)
    metal = crosstl.translate(str(source_path), backend="metal", format_output=False)

    assert "ivec4 x = ivec4(a);" in crossgl
    assert "a.xxxx" not in crossgl
    loaded_scalar = re.search(r"(%\d+) = OpLoad %\d+ %\d+", vulkan)
    assert loaded_scalar is not None
    assert re.search(
        rf"OpCompositeConstruct %\d+ {loaded_scalar.group(1)} "
        rf"{loaded_scalar.group(1)} {loaded_scalar.group(1)} "
        rf"{loaded_scalar.group(1)}",
        vulkan,
    )
    assert "Could not find member xxxx" not in vulkan
    assert "threadgroup int a;" in metal
    assert "int4 x = int4(a);" in metal
    assert "a.xxxx" not in metal
    assert "unsupported Metal program-scope groupshared" not in metal


def test_hlsl_legacy_sampler_register_lowers_to_opengl_binding(tmp_path):
    source_path = _write_source(
        tmp_path,
        "sprite.fx",
        """
        sampler2D Texture : register(s0);

        float4 main(float2 uv : TEXCOORD0) : SV_Target
        {
            return tex2D(Texture, uv);
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend="opengl", format_output=False
    )

    assert "layout(binding = 0) uniform sampler2D Texture;" in generated
    assert "fragColor = texture(Texture, uv);" in generated


def test_hlsl_pixel_shader_user_semantic_input_lowers_to_metal_stage_in(tmp_path):
    source_path = _write_source(
        tmp_path,
        "neg1.hlsl",
        """
        float4 main(float4 a : A) : SV_Target
        {
            int4 x = int4(a);
            return float4(x);
        }
        """,
    )

    metal = crosstl.translate(str(source_path), backend="metal", format_output=False)

    assert "struct fragment_main_Input" in metal
    assert "float4 a [[attribute(0)]];" in metal
    assert (
        "fragment float4 fragment_main("
        "fragment_main_Input _crossglInput [[stage_in]])"
    ) in metal
    assert "float4 a = _crossglInput.a;" in metal
    assert "[[A]]" not in metal
    _compile_with_metal_if_available(metal, tmp_path)


def test_hlsl_fx_macro_texture_resource_survives_metal_translation(tmp_path):
    _write_source(
        tmp_path,
        "Macros.fxh",
        """
        #define DECLARE_TEXTURE(Name, index) \\
            sampler2D Name : register(s##index);
        #define SAMPLE_TEXTURE(Name, texCoord) tex2D(Name, texCoord)
        """,
    )
    source_path = _write_source(
        tmp_path,
        "SpriteEffect.fx",
        """
        #include "Macros.fxh"

        DECLARE_TEXTURE(Texture, 0);

        struct VSOutput {
            float4 position : SV_Position;
            float4 color : COLOR0;
            float2 texCoord : TEXCOORD0;
        };

        float4 SpritePixelShader(VSOutput input) : SV_Target0 {
            return SAMPLE_TEXTURE(Texture, input.texCoord) * input.color;
        }
        """,
    )

    crossgl = crosstl.translate(str(source_path), backend="cgl", format_output=False)
    metal = crosstl.translate(str(source_path), backend="metal", format_output=False)

    assert "@ register(s0)\n    sampler2D Texture;" in crossgl
    assert (
        "@ hlsl_program_constant\n    @ register(s0)\n    sampler2D Texture;"
        not in crossgl
    )
    assert "texture2d<float> Texture [[texture(0)]]" in metal
    assert (
        "Texture.sample(sampler(mag_filter::linear, min_filter::linear), input.texCoord)"
        in metal
    )
    assert "[[gl_FragData]]" not in metal
    _compile_with_metal_if_available(metal, tmp_path)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            "directx",
            (
                "SV_ShadingRate",
                "CrossGLFragmentSizeFromShadingRate",
                "uint2 _crossglFragSize",
            ),
        ),
        (
            "vulkan",
            (
                "OpCapability FragmentDensityEXT",
                'OpExtension "SPV_EXT_fragment_invocation_density"',
                "BuiltIn FragSizeEXT",
            ),
        ),
    ],
)
def test_glsl_fragment_invocation_density_lowers_supported_targets(
    tmp_path, target, expected
):
    source_path = _write_source(
        tmp_path, "CubeFDM_fs.glsl", GLSL_FRAGMENT_INVOCATION_DENSITY_SOURCE
    )
    output_path = tmp_path / f"CubeFDM_fs.{target}.out"

    generated = crosstl.translate(
        str(source_path),
        backend=target,
        save_shader=str(output_path),
        format_output=False,
    )

    assert output_path.exists()
    for snippet in expected:
        assert snippet in generated


def test_glsl_fragment_invocation_density_reports_metal_target_diagnostic(tmp_path):
    source_path = _write_source(
        tmp_path, "CubeFDM_fs.glsl", GLSL_FRAGMENT_INVOCATION_DENSITY_SOURCE
    )
    output_path = tmp_path / "CubeFDM_fs.metal.out"

    with pytest.raises(ValueError, match="GL_EXT_fragment_invocation_density"):
        crosstl.translate(
            str(source_path),
            backend="metal",
            save_shader=str(output_path),
            format_output=False,
        )

    assert not output_path.exists()


def test_glsl_fragment_invocation_density_helper_lowers_to_directx(tmp_path):
    source_path = _write_source(
        tmp_path,
        "CubeFDM_fs.glsl",
        GLSL_FRAGMENT_INVOCATION_DENSITY_HELPER_SOURCE,
    )
    output_path = tmp_path / "CubeFDM_fs.hlsl"

    generated = crosstl.translate(
        str(source_path),
        backend="directx",
        save_shader=str(output_path),
        format_output=False,
    )

    assert output_path.exists()
    assert "SV_ShadingRate" in generated
    assert "CrossGLFragmentSizeFromShadingRate" in generated
    assert "float4 FragmentDensityToColor(uint2 _crossglFragSize)" in generated
    assert "FragmentDensityToColor(_crossglFragSize)" in generated
    assert "gl_FragSizeEXT" not in generated


def test_glsl_fragment_invocation_density_helper_reports_metal_diagnostic(tmp_path):
    source_path = _write_source(
        tmp_path,
        "CubeFDM_fs.glsl",
        GLSL_FRAGMENT_INVOCATION_DENSITY_HELPER_SOURCE,
    )
    output_path = tmp_path / "CubeFDM_fs.metal"

    with pytest.raises(ValueError, match="GL_EXT_fragment_invocation_density"):
        crosstl.translate(
            str(source_path),
            backend="metal",
            save_shader=str(output_path),
            format_output=False,
        )

    assert not output_path.exists()


def test_metal_max_total_threads_metadata_translates_to_vulkan(tmp_path):
    source_path = _write_source(
        tmp_path,
        "mlx-max-total-threads.metal",
        """
        #include <metal_stdlib>
        using namespace metal;

        [[max_total_threads_per_threadgroup(1024)]]
        kernel void pinned_kernel(
            device float* out [[buffer(0)]],
            uint index [[thread_position_in_grid]]) {
            out[index] = 1.0;
        }
        """,
    )

    generated = crosstl.translate(
        str(source_path), backend="vulkan", format_output=False
    )

    _assert_generated_output_is_usable(generated)
    assert "OpEntryPoint GLCompute" in generated
    assert '"pinned_kernel"' in generated
    assert "return semantic" not in generated


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_metal_max_total_threads_metadata_is_not_return_semantic_for_void_compute(
    tmp_path, target
):
    source_path = _write_source(
        tmp_path,
        "mlx-void-compute-max-total-threads.metal",
        """
        #include <metal_stdlib>
        using namespace metal;

        [[max_total_threads_per_threadgroup(1024)]]
        kernel void pinned_kernel(
            device float* out [[buffer(0)]],
            uint index [[thread_position_in_grid]]) {
            out[index] = 1.0;
        }
        """,
    )

    generated = crosstl.translate(str(source_path), backend=target, format_output=False)

    _assert_generated_output_is_usable(generated)
    assert "return semantic" not in generated
    assert "max_total_threads_per_threadgroup" not in generated


def test_hlsl_hello_const_buffers_vertex_semantics_lower_to_metal_attributes(
    tmp_path,
):
    source_path = _write_source(
        tmp_path,
        "hello-const-buffers.hlsl",
        """
        cbuffer SceneConstantBuffer : register(b0)
        {
            float4 offset;
        };

        struct PSInput
        {
            float4 position : SV_POSITION;
            float4 color : COLOR;
        };

        PSInput VSMain(float4 position : POSITION, float4 color : COLOR)
        {
            PSInput result;
            result.position = position + offset;
            result.color = color;
            return result;
        }

        float4 PSMain(PSInput input) : SV_TARGET
        {
            return input.color;
        }
        """,
    )

    metal = crosstl.translate(str(source_path), backend="metal", format_output=False)

    assert "float4 position [[attribute(0)]];" in metal
    assert "float4 color [[attribute(1)]];" in metal
    assert "float4 position [[position]];" in metal
    assert "[[Color]]" not in metal
    assert "[[COLOR]]" not in metal
    _compile_with_metal_if_available(metal, tmp_path)


def test_hlsl_struct_inout_vertex_entry_generates_valid_glsl_and_metal(tmp_path):
    source_path = _write_source(
        tmp_path,
        "diligent-cube-vs.hlsl",
        """
        cbuffer Constants
        {
            float4x4 g_WorldViewProj;
        };

        struct VSInput
        {
            float3 Pos   : ATTRIB0;
            float4 Color : ATTRIB1;
        };

        struct PSInput
        {
            float4 Pos   : SV_POSITION;
            float4 Color : COLOR0;
        };

        [shader("vertex")]
        void main(in VSInput VSIn, out PSInput PSIn)
        {
            PSIn.Pos   = mul(float4(VSIn.Pos, 1.0), g_WorldViewProj);
            PSIn.Color = VSIn.Color;
        }
        """,
    )

    opengl = crosstl.translate(str(source_path), backend="opengl", format_output=False)
    metal = crosstl.translate(str(source_path), backend="metal", format_output=False)

    assert "in vec3 VSIn_Pos;" in opengl
    assert "in vec4 VSIn_Color;" in opengl
    assert "out vec4 PSIn_Color;" in opengl
    assert "PSIn_Color = VSIn_Color;" in opengl
    assert "VSIn." not in opengl
    assert "PSIn." not in opengl
    assert "in vec4 Color;" not in opengl
    assert "out vec4 Color;" not in opengl

    assert "vertex vertex_main_Return vertex_main" in metal
    assert "float3 VSIn_Pos [[attribute(0)]];" in metal
    assert "float4 VSIn_Color [[attribute(1)]];" in metal
    assert "float4 PSIn_Pos [[position]];" in metal
    assert "float4 PSIn_Color [[user(Color0)]];" in metal
    assert "[[ATTRIB" not in metal
    assert "[[COLOR" not in metal
    assert " PSIn." not in metal

    _compile_glslang_if_available(opengl, "vertex")
    _compile_metal_if_available(metal)


def test_hlsl_struct_inout_pixel_entry_generates_valid_glsl_and_metal(tmp_path):
    source_path = _write_source(
        tmp_path,
        "diligent-cube-ps.hlsl",
        """
        struct PSInput
        {
            float4 Pos   : SV_POSITION;
            float4 Color : COLOR0;
        };

        struct PSOutput
        {
            float4 Color : SV_TARGET;
        };

        [shader("pixel")]
        void main(in PSInput PSIn, out PSOutput PSOut)
        {
            float4 Color = PSIn.Color;
            PSOut.Color = Color;
        }
        """,
    )

    opengl = crosstl.translate(str(source_path), backend="opengl", format_output=False)
    metal = crosstl.translate(str(source_path), backend="metal", format_output=False)

    assert "in vec4 PSIn_Color;" in opengl
    assert "layout(location = 0) out vec4 fragColor;" in opengl
    assert "vec4 Color = PSIn_Color;" in opengl
    assert "fragColor = Color;" in opengl
    assert "PSIn." not in opengl
    assert "PSOut." not in opengl

    assert "fragment fragment_main_Return fragment_main" in metal
    assert "float4 PSIn_Pos [[position]]" in metal
    assert "float4 PSIn_Color [[user(Color0)]]" in metal
    assert "float4 PSOut_Color [[color(0)]];" in metal
    assert "[[COLOR" not in metal
    assert " PSIn." not in metal
    assert " PSOut." not in metal

    _compile_glslang_if_available(opengl, "fragment")
    _compile_metal_if_available(metal)


def test_glsl_es_legacy_fragcolor_lowers_to_non_reserved_opengl_output(tmp_path):
    source_path = _write_source(
        tmp_path,
        "Cube_cube.frag",
        """
        precision lowp float;
        varying vec3 vv3colour;
        void main() { gl_FragColor = vec4(vv3colour, 1.0); }
        """,
    )

    opengl = crosstl.translate(
        str(source_path),
        backend="opengl",
        source_backend="opengl",
        format_output=False,
    )

    assert "layout(location = 0) out vec4 fragColor;" in opengl
    assert "fragColor = vec4(vv3colour, 1.0);" in opengl
    assert "vec4 gl_FragColor;" not in opengl
    assert "gl_FragColor" not in opengl

    _compile_glslang_if_available(opengl, "fragment")


@pytest.mark.parametrize(
    "source_name,target_backend", NATIVE_SOURCE_TO_TARGET_PIPELINE_CASES
)
def test_native_source_to_registered_target_pipeline_is_total(
    tmp_path, source_name, target_backend
):
    filename, source = NATIVE_SOURCE_SNIPPETS[source_name]
    source_path = _write_source(tmp_path, filename, source)

    generated = crosstl.translate(
        str(source_path), backend=target_backend, format_output=False
    )

    _assert_generated_output_is_usable(generated)


@pytest.mark.parametrize(
    "alias_name,target_backend",
    NATIVE_SOURCE_EXTENSION_ALIAS_TO_TARGET_PIPELINE_CASES,
)
def test_native_source_extension_alias_to_registered_target_pipeline_is_total(
    tmp_path, alias_name, target_backend
):
    filename, source = NATIVE_SOURCE_EXTENSION_ALIAS_SNIPPETS[alias_name]
    source_path = _write_source(tmp_path, filename, source)

    generated = crosstl.translate(
        str(source_path), backend=target_backend, format_output=False
    )

    _assert_generated_output_is_usable(generated)


@pytest.mark.parametrize("source_name", sorted(WEBGL_NATIVE_COMPUTE_SOURCE_DIAGNOSTICS))
def test_native_compute_sources_report_webgl_diagnostics(tmp_path, source_name):
    filename, source = NATIVE_SOURCE_SNIPPETS[source_name]
    source_path = _write_source(tmp_path, filename, source)

    with pytest.raises(
        ValueError,
        match="WebGL target does not support shader stage\\(s\\): compute",
    ):
        crosstl.translate(str(source_path), backend="webgl", format_output=False)


@pytest.mark.parametrize(
    "alias_name", sorted(WEBGL_NATIVE_COMPUTE_EXTENSION_ALIAS_DIAGNOSTICS)
)
def test_native_compute_extension_aliases_report_webgl_diagnostics(
    tmp_path, alias_name
):
    filename, source = NATIVE_SOURCE_EXTENSION_ALIAS_SNIPPETS[alias_name]
    source_path = _write_source(tmp_path, filename, source)

    with pytest.raises(
        ValueError,
        match="WebGL target does not support shader stage\\(s\\): compute",
    ):
        crosstl.translate(str(source_path), backend="webgl", format_output=False)


def test_metal_template_placeholder_scan_ignores_comments():
    # Regression: the residual-template placeholder scan treated comment words as
    # identifiers, so a comment captured in an extracted type (e.g. a trailing
    # `// Get the max ...`) produced a spurious "missing template parameter 'Get'"
    # and wrongly blocked otherwise-translatable kernels (MLX softmax/logsumexp).
    from crosstl.project.pipeline import (
        _normalize_metal_type_text,
        _unresolved_metal_template_placeholders_in_type,
    )

    # Comments are stripped from normalized type text.
    assert (
        _normalize_metal_type_text("void // Get the max and the normalizer") == "void"
    )
    assert _normalize_metal_type_text("float /* T */ x").replace(" ", "") == "floatx"

    # No spurious placeholder is reported for comment words.
    assert (
        _unresolved_metal_template_placeholders_in_type(
            "void // Get the max and the normalizer", set()
        )
        == []
    )
    # A genuine unresolved template parameter is still detected.
    assert _unresolved_metal_template_placeholders_in_type("T", {"T"}) == ["T"]


def test_metal_function_header_masks_comments_spanning_the_header_start():
    # Regression: a function span can begin inside a preceding comment (e.g. a
    # license block), so masking only the already-sliced header missed the
    # opening `/*` and leaked license prose (e.g. "OR"/"AND") into extracted
    # return/parameter types. _metal_function_header masks with full-source
    # context so the sliced header is comment-free regardless of where it starts.
    from types import SimpleNamespace

    from crosstl.project.pipeline import _metal_function_header

    source = "/* ... OR BUSINESS INTERRUPTION ... */ float foo(int x) { return x; }"
    body_start = source.index("{")
    # Span deliberately begins mid-comment (after the opening `/*`).
    function = SimpleNamespace(
        span=(3, len(source)), body_span=(body_start, len(source))
    )

    header = _metal_function_header(source, function)

    assert "OR" not in header
    assert "BUSINESS" not in header
    assert "float foo(int x)" in header

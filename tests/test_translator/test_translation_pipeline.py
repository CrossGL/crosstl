import re

import pytest

import crosstl
import crosstl.translator
import crosstl.translator.codegen as codegen
from crosstl.project import translate_project
from crosstl.translator.ast import ShaderStage
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

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


def test_metal_xhalf_vectors_do_not_leak_to_opengl(tmp_path):
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

    generated = crosstl.translate(
        str(source_path), backend="opengl", format_output=False
    )

    _assert_generated_output_is_usable(generated)
    assert "out vec3 viewDir;" in generated
    assert "xhalf" not in generated
    assert "f16vec3" not in generated


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
    assert "ConstantBuffer<MatMulParams> params" in generated
    assert "uint3 id_dispatchThreadID : SV_DispatchThreadID" in generated
    assert "uint2 id = id_dispatchThreadID.xy;" in generated
    assert "const uint row_dim_x = params.row_dim_x;" in generated
    assert "const uint col_dim_x = params.col_dim_x;" in generated
    assert "const uint inner_dim = params.inner_dim;" in generated
    assert "ReferenceType" not in generated
    assert "MatMulParams&" not in generated


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


@pytest.mark.parametrize("source_name", sorted(NATIVE_SOURCE_SNIPPETS))
@pytest.mark.parametrize("target_backend", codegen.backend_names())
def test_native_source_to_registered_target_pipeline_is_total(
    tmp_path, source_name, target_backend
):
    filename, source = NATIVE_SOURCE_SNIPPETS[source_name]
    source_path = _write_source(tmp_path, filename, source)

    generated = crosstl.translate(
        str(source_path), backend=target_backend, format_output=False
    )

    _assert_generated_output_is_usable(generated)


@pytest.mark.parametrize("alias_name", sorted(NATIVE_SOURCE_EXTENSION_ALIAS_SNIPPETS))
@pytest.mark.parametrize("target_backend", codegen.backend_names())
def test_native_source_extension_alias_to_registered_target_pipeline_is_total(
    tmp_path, alias_name, target_backend
):
    filename, source = NATIVE_SOURCE_EXTENSION_ALIAS_SNIPPETS[alias_name]
    source_path = _write_source(tmp_path, filename, source)

    generated = crosstl.translate(
        str(source_path), backend=target_backend, format_output=False
    )

    _assert_generated_output_is_usable(generated)

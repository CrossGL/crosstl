import pytest

import crosstl
import crosstl.translator
import crosstl.translator.codegen as codegen
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
    "mojo": ("shader.mojo", "fn main() -> None:\n    return\n"),
    "rust": ("shader.rs", "fn main() { let x: f32 = 1.0; }"),
    "vulkan": (
        "shader.spirv",
        """
        void main() {
            vec4 color = vec4(1.0, 0.0, 0.0, 1.0);
            gl_FragColor = color;
        }
        """,
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


def test_registered_native_sources_have_reverse_codegen_factories():
    register_default_sources()

    for backend in codegen.backend_names():
        spec = SOURCE_REGISTRY.get(backend)
        assert spec is not None
        assert spec.reverse_codegen_factory is not None

        converter = spec.reverse_codegen_factory()
        assert callable(getattr(converter, "generate", None))


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


@pytest.mark.parametrize("source_name", sorted(NATIVE_SOURCE_SNIPPETS))
def test_native_sources_translate_to_parseable_crossgl(tmp_path, source_name):
    filename, source = NATIVE_SOURCE_SNIPPETS[source_name]
    source_path = _write_source(tmp_path, filename, source)

    generated = crosstl.translate(str(source_path), backend="cgl", format_output=False)

    _assert_generated_output_is_usable(generated)
    crosstl.translator.parse(generated)


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

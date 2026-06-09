import os
import re
from pathlib import Path

import pytest

import crosstl.translator.codegen as codegen
from crosstl.formatter import CodeFormatter, ShaderLanguage
from crosstl.translator import parse
from crosstl.translator.plugin_loader import discover_backend_plugins
from crosstl.translator.source_registry import (
    BINARY_SPIRV_UNSUPPORTED_MESSAGE,
    CUDA_ARTIFACT_UNSUPPORTED_MESSAGE,
    DIRECTX_BINARY_UNSUPPORTED_MESSAGE,
    HIP_ARTIFACT_UNSUPPORTED_MESSAGE,
    METAL_BINARY_UNSUPPORTED_MESSAGE,
    SOURCE_REGISTRY,
    WGSL_SOURCE_UNSUPPORTED_MESSAGE,
    register_default_sources,
)

SMOKE_SHADER = """
shader main {
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };
    struct VSOutput {
        vec4 color @ COLOR;
    };
    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            output.color = vec4(input.texCoord, 0.0, 1.0);
            return output;
        }
    }
    fragment {
        vec4 main(VSOutput input) @ gl_FragColor {
            return input.color;
        }
    }
}
"""

ADVANCED_SMOKE_SHADER = """
shader main {
    struct Payload {
        vec3 color;
    };
    struct VSInput {
        vec3 position @ POSITION;
    };
    struct VSOutput {
        vec4 position @ gl_Position;
    };
    vertex {
        VSOutput main(VSInput input) {
            VSOutput output;
            Payload payload;
            visible_function_table vft;
            indirect_command_buffer icb;
            vft[0](payload);
            icb.reset();
            output.position = vec4(input.position, 1.0);
            return output;
        }
    }
}
"""

TRANSLATE_API_ROUNDTRIP_BACKENDS = (
    "metal",
    "directx",
    "opengl",
    "cuda",
    "hip",
    "mojo",
    "rust",
    "spirv",
)

REAL_WORLD_TARGET_EXTENSION_BACKENDS = (
    ("cuda", (".cu", ".cuh", ".cuda"), ShaderLanguage.CUDA),
    ("rust", (".rs", ".rust"), ShaderLanguage.RUST),
)

SOURCE_ONLY_BACKEND_DIRS = {"OpenCL"}


def _backend_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "crosstl", "backend")
    )


def _backend_dirs():
    backend_root = _backend_root()
    if not os.path.isdir(backend_root):
        return []
    return sorted(
        [
            name
            for name in os.listdir(backend_root)
            if os.path.isdir(os.path.join(backend_root, name))
            and not name.startswith(".")
            and not name.startswith("__")
        ]
    )


def _normalize_backend_dir(name: str) -> str:
    normalized = codegen.normalize_backend_name(name)
    return normalized if normalized else name.strip().lower()


def _backend_test_files():
    test_dir = os.path.dirname(__file__)
    return [
        name
        for name in os.listdir(test_dir)
        if name.startswith("test_") and name.endswith(".py")
    ]


def test_backend_registry_covers_backend_dirs():
    discover_backend_plugins()
    missing = []
    for name in _backend_dirs():
        if name in SOURCE_ONLY_BACKEND_DIRS:
            continue
        normalized = _normalize_backend_dir(name)
        if not codegen.get_backend(normalized):
            missing.append(name)
    assert not missing, f"Unregistered backends: {missing}"


def test_source_registry_covers_backend_dirs():
    register_default_sources()
    discover_backend_plugins()
    missing = []
    for name in _backend_dirs():
        normalized = _normalize_backend_dir(name)
        if not SOURCE_REGISTRY.get(normalized):
            missing.append(name)
    assert not missing, f"Unregistered source backends: {missing}"


@pytest.mark.parametrize(
    "extension",
    (
        ".glsl",
        ".vs",
        ".vsh",
        ".fs",
        ".fsh",
        ".vert",
        ".vertex",
        ".frag",
        ".fragment",
        ".comp",
        ".cs",
        ".csh",
        ".compute",
        ".geom",
        ".gs",
        ".gsh",
        ".geometry",
        ".tesc",
        ".tcs",
        ".tese",
        ".tes",
    ),
)
def test_source_registry_recognizes_glsl_stage_extensions(extension):
    register_default_sources()

    assert SOURCE_REGISTRY.get_by_extension(extension).name == "opengl"
    assert (
        SOURCE_REGISTRY.get_by_extension(f"shader{extension.upper()}").name == "opengl"
    )


@pytest.mark.parametrize("extension", (".rgen.glsl", ".mesh.glsl", ".frag.glsl"))
def test_source_registry_recognizes_compound_glsl_extension_strings(extension):
    register_default_sources()

    assert SOURCE_REGISTRY.get_by_extension(extension).name == "opengl"


@pytest.mark.parametrize("extension", (".hlsl", ".hlsli", ".fx", ".fxh"))
def test_source_registry_recognizes_directx_real_world_extensions(extension):
    register_default_sources()

    assert SOURCE_REGISTRY.get_by_extension(extension).name == "directx"
    assert (
        SOURCE_REGISTRY.get_by_extension(f"shader{extension.upper()}").name == "directx"
    )


@pytest.mark.parametrize("extension", (".slang", ".slangh"))
def test_source_registry_recognizes_slang_real_world_extensions(extension):
    register_default_sources()

    assert SOURCE_REGISTRY.get_by_extension(extension).name == "slang"
    assert (
        SOURCE_REGISTRY.get_by_extension(f"shader{extension.upper()}").name == "slang"
    )


@pytest.mark.parametrize("extension", (".cl", ".opencl"))
def test_source_registry_recognizes_opencl_real_world_extensions(extension):
    register_default_sources()

    assert SOURCE_REGISTRY.get_by_extension(extension).name == "opencl"
    assert (
        SOURCE_REGISTRY.get_by_extension(f"shader{extension.upper()}").name == "opencl"
    )


@pytest.mark.parametrize(
    ("extension", "message"),
    (
        (".spv", BINARY_SPIRV_UNSUPPORTED_MESSAGE),
        (".spirv", BINARY_SPIRV_UNSUPPORTED_MESSAGE),
        (".air", METAL_BINARY_UNSUPPORTED_MESSAGE),
        (".metallib", METAL_BINARY_UNSUPPORTED_MESSAGE),
        (".wgsl", WGSL_SOURCE_UNSUPPORTED_MESSAGE),
        (".wesl", WGSL_SOURCE_UNSUPPORTED_MESSAGE),
        (".cso", DIRECTX_BINARY_UNSUPPORTED_MESSAGE),
        (".dxbc", DIRECTX_BINARY_UNSUPPORTED_MESSAGE),
        (".dxil", DIRECTX_BINARY_UNSUPPORTED_MESSAGE),
        (".ptx", CUDA_ARTIFACT_UNSUPPORTED_MESSAGE),
        (".cubin", CUDA_ARTIFACT_UNSUPPORTED_MESSAGE),
        (".fatbin", CUDA_ARTIFACT_UNSUPPORTED_MESSAGE),
        (".hsaco", HIP_ARTIFACT_UNSUPPORTED_MESSAGE),
    ),
)
def test_source_registry_known_unsupported_extensions_raise_clear_diagnostic(
    extension, message
):
    register_default_sources()

    with pytest.raises(ValueError, match=re.escape(message)):
        SOURCE_REGISTRY.get_by_extension(f"shader{extension}")


@pytest.mark.parametrize(
    ("filename", "message"),
    (
        ("shader.wgsl.json", WGSL_SOURCE_UNSUPPORTED_MESSAGE),
        ("shader.spv.json", BINARY_SPIRV_UNSUPPORTED_MESSAGE),
        ("shader.metallib.json", METAL_BINARY_UNSUPPORTED_MESSAGE),
        ("shader.dxil.json", DIRECTX_BINARY_UNSUPPORTED_MESSAGE),
        ("shader.fatbin.json", CUDA_ARTIFACT_UNSUPPORTED_MESSAGE),
        ("shader.hsaco.json", HIP_ARTIFACT_UNSUPPORTED_MESSAGE),
    ),
)
def test_source_registry_compound_artifact_extensions_raise_clear_diagnostic(
    filename, message
):
    register_default_sources()

    with pytest.raises(ValueError, match=re.escape(message)):
        SOURCE_REGISTRY.get_by_extension(filename)


def test_source_registry_accepts_pathlike_inputs_for_source_and_artifact_inference():
    register_default_sources()

    assert (
        SOURCE_REGISTRY.get_by_extension(Path("shaders") / "deferred.FRAG.GLSL").name
        == "opengl"
    )
    with pytest.raises(ValueError, match=re.escape(BINARY_SPIRV_UNSUPPORTED_MESSAGE)):
        SOURCE_REGISTRY.get_by_extension(Path("build") / "deferred.FRAG.SPV")


@pytest.mark.parametrize("extension", (".metal", ".msl"))
def test_source_registry_recognizes_metal_real_world_extensions(extension):
    register_default_sources()

    assert SOURCE_REGISTRY.get_by_extension(extension).name == "metal"
    assert (
        SOURCE_REGISTRY.get_by_extension(f"shader{extension.upper()}").name == "metal"
    )


@pytest.mark.parametrize(
    ("backend", "extensions", "formatter_language"),
    REAL_WORLD_TARGET_EXTENSION_BACKENDS,
)
def test_target_registry_real_world_extensions_match_source_and_formatter(
    backend, extensions, formatter_language
):
    register_default_sources()
    formatter = CodeFormatter()
    spec = codegen.get_backend(backend)

    assert spec is not None
    assert tuple(spec.file_extensions) == extensions
    assert codegen.get_backend_extension(backend) == extensions[0]

    for extension in extensions:
        assert SOURCE_REGISTRY.get_by_extension(extension).name == backend
        assert formatter.detect_language(f"shader{extension}") == formatter_language


@pytest.mark.parametrize(
    ("backend", "extensions"),
    (
        ("cuda", (".cu", ".cuh", ".cuda")),
        ("directx", (".hlsl",)),
        ("opengl", (".glsl",)),
        ("hip", (".hip",)),
        ("metal", (".metal",)),
        ("mojo", (".mojo",)),
        ("rust", (".rs", ".rust")),
        ("vulkan", (".spvasm",)),
    ),
)
def test_target_registry_resolves_file_extension_backend_spellings(backend, extensions):
    for extension in extensions:
        assert codegen.normalize_backend_name(extension) == backend
        assert codegen.normalize_backend_name(f"target{extension.upper()}") == backend
        assert codegen.get_backend(extension).name == backend
        assert (
            codegen.get_codegen(f"target{extension}").__class__
            is codegen.get_codegen(backend).__class__
        )


def test_each_backend_has_codegen_tests():
    backend_files = [name.lower() for name in _backend_test_files()]
    missing = []
    for backend_name in codegen.backend_names():
        spec = codegen.get_backend(backend_name)
        identifiers = {backend_name.lower()}
        if spec:
            identifiers.update(alias.lower() for alias in spec.aliases)
        has_test = any(
            any(identifier in filename for identifier in identifiers)
            for filename in backend_files
        )
        if not has_test:
            missing.append(backend_name)
    assert not missing, f"Missing codegen tests for: {missing}"


@pytest.mark.parametrize("backend", codegen.backend_names())
def test_backend_codegen_smoke(backend):
    ast = parse(SMOKE_SHADER)
    generator = codegen.get_codegen(backend)
    generated = generator.generate(ast)
    assert isinstance(generated, str)
    assert generated.strip()


@pytest.mark.parametrize("backend", codegen.backend_names())
def test_backend_codegen_advanced_smoke(backend):
    ast = parse(ADVANCED_SMOKE_SHADER)
    generator = codegen.get_codegen(backend)
    generated = generator.generate(ast)
    assert isinstance(generated, str)
    assert generated.strip()


@pytest.mark.parametrize("backend", codegen.backend_names())
def test_backend_extension_is_available(backend):
    ext = codegen.get_backend_extension(backend)
    assert ext is not None
    assert ext.startswith(".")


@pytest.mark.parametrize("backend", TRANSLATE_API_ROUNDTRIP_BACKENDS)
def test_translate_api_roundtrip(tmp_path, backend):
    """Verify the crosstl.translate() API works end-to-end with a CGL file."""
    import crosstl

    cgl_file = tmp_path / "test_shader.cgl"
    cgl_file.write_text(SMOKE_SHADER, encoding="utf-8")

    output = crosstl.translate(str(cgl_file), backend=backend)
    assert isinstance(output, str)
    assert len(output) > 50, f"{backend} output too small: {len(output)}"

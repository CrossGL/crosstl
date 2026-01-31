import os
import pytest

import crosstl.translator.codegen as codegen
from crosstl.translator import parse
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources
from crosstl.translator.plugin_loader import discover_backend_plugins


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

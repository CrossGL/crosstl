"""Shader record buffers and anonymous buffer blocks keep their storage through CrossGL.

GLSL puts the members of a block without an instance name in global scope, but they stay members of the buffer
(GLSL 4.60 section 4.3.9). The importer declared such members as plain globals unless the block ended in a runtime
array, so ``buffer B { float x; float y; };`` became two private variables on SPIR-V and two uniforms on GLSL, and
``layout(shaderRecordEXT) buffer SBT { ... };`` lost its record. See
:mod:`crosstl.translator.semantics.shader_records`.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.shader_records import (
    ShaderRecordBufferUnsupportedError,
)

ANONYMOUS_BUFFER = """#version 460
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Data { float x; float y; };
void main() { y = x * 2.0; }
"""

SHADER_RECORD_MISS = """#version 460
#extension GL_EXT_ray_tracing : enable
layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(shaderRecordEXT, std430) buffer SBT { float r; float g; float b; };
void main() { hitValue = vec3(r, g, b); }
"""

NAMED_SHADER_RECORD_MISS = SHADER_RECORD_MISS.replace(
    "};\nvoid", "} sbt;\nvoid"
).replace("vec3(r, g, b)", "vec3(sbt.r, sbt.g, sbt.b)")


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def test_anonymous_buffer_members_stay_in_the_buffer(tmp_path):
    source = _write(tmp_path, "data.comp", ANONYMOUS_BUFFER)
    crossgl = crosstl.translate(source, backend="cgl", format_output=False)
    assert re.search(r"buffer\s+Data\s*\{", crossgl)
    assert re.search(r"\w+\.y\s*=\s*\(?\w+\.x", crossgl)

    spirv = crosstl.translate(source, backend="vulkan", format_output=False)
    assert "Private" not in spirv
    hlsl = crosstl.translate(source, backend="directx", format_output=False)
    assert re.search(r"\bRW\w*Buffer\b", hlsl)


def test_anonymous_buffer_glsl_output_compiles(tmp_path):
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    glsl = crosstl.translate(
        _write(tmp_path, "data.comp", ANONYMOUS_BUFFER),
        backend="opengl",
        format_output=False,
    )
    assert "uniform float" not in glsl
    path = _write(tmp_path, "out.comp", glsl)
    result = subprocess.run(
        ["glslangValidator", "-S", "comp", path], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("source", [SHADER_RECORD_MISS, NAMED_SHADER_RECORD_MISS])
def test_shader_record_reaches_spirv_storage_class(tmp_path, source):
    spirv = crosstl.translate(
        _write(tmp_path, "sbt.rmiss", source), backend="vulkan", format_output=False
    )
    variable = re.search(r"%(\w+) = OpVariable %\w+ ShaderRecordBufferKHR", spirv)
    assert variable is not None
    assert not re.search(
        rf"OpDecorate %{variable.group(1)} (?:DescriptorSet|Binding)\b", spirv
    )
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly = _write(tmp_path, "sbt.spvasm", spirv)
        binary = str(tmp_path / "sbt.spv")
        subprocess.run(
            ["spirv-as", "--target-env", "vulkan1.2", assembly, "-o", binary],
            check=True,
        )
        subprocess.run(["spirv-val", "--target-env", "vulkan1.2", binary], check=True)


@pytest.mark.parametrize("source", [SHADER_RECORD_MISS, NAMED_SHADER_RECORD_MISS])
def test_shader_record_reaches_glsl_without_a_binding(tmp_path, source):
    glsl = crosstl.translate(
        _write(tmp_path, "sbt.rmiss", source), backend="opengl", format_output=False
    )
    declaration = re.search(r"layout\(([^)]*)\)\s*buffer\s+SBT", glsl)
    assert declaration is not None
    assert "shaderRecordEXT" in declaration.group(1)
    assert "binding" not in declaration.group(1)
    if shutil.which("glslangValidator"):
        path = _write(tmp_path, "out.rmiss", glsl)
        result = subprocess.run(
            [
                "glslangValidator",
                "--target-env",
                "vulkan1.2",
                "-o",
                str(tmp_path / "out.spv"),
                "-S",
                "rmiss",
                path,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("backend", ["directx", "metal"])
def test_shader_record_is_refused_without_a_shader_binding_table(tmp_path, backend):
    with pytest.raises(ShaderRecordBufferUnsupportedError, match="shaderRecordEXT"):
        crosstl.translate(
            _write(tmp_path, "sbt.rmiss", SHADER_RECORD_MISS), backend=backend
        )

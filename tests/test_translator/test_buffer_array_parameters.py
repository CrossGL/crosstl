"""A helper's ``buffer T name[]`` parameter becomes the structured buffer its caller passes.

CrossGL lets a helper take a storage buffer as ``buffer T name[]`` (``readonly`` for a read-only one). The
HLSL, GLSL, SPIR-V and Slang generators printed it as an unsized array parameter, which DXC, glslang, spirv-val
and slangc reject, because none of those languages can pass an unsized array. The unsized-array pass now gives
such a parameter the structured-buffer form the HLSL importer uses for the same parameter, so HLSL and Slang pass
the resource, Metal a device pointer, and GLSL and SPIR-V specialize the helper for each buffer it receives.
Elements without one stride across std430, HLSL structured buffers and MSL are refused.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import StorageBufferParameterUnsupportedError

TARGETS = ("directx", "opengl", "vulkan", "metal", "slang")

COMPUTE = """
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        void main() {
            %s
        }
    }
"""

HELPERS = """
shader BufferParameters {
    struct Pair { float a; float b; }
    buffer uint data[] @binding(0);
    readonly buffer float src[] @binding(1);
    buffer Pair pairs[] @binding(2);
    void put(buffer uint values[], uint i);
    void put(buffer uint values[], uint i) {
        values[i] = uint(values.length());
    }
    float get(readonly buffer float s[], uint i) {
        return s[i] + float(s.length());
    }
    void swap(buffer Pair p[], uint i) {
        float t = p[i].a;
        p[i].a = p[i].b;
        p[i].b = t;
    }
%s}
""" % (
    COMPUTE % "uint i = gl_GlobalInvocationID.x; put(data, i); "
    "data[i + 1u] = uint(get(src, i)); swap(pairs, i);"
)

SIZED = """
shader SizedParameter {
    buffer uint data[4] @binding(0);
    uint count(buffer uint values[4]) { return uint(values.length()); }
%s}
""" % (COMPUTE % "data[0] = count(data);")

VECTOR3 = """
shader Vector3Parameter {
    buffer vec3 data[] @binding(0);
    void put(buffer vec3 values[], uint i) { values[i] = vec3(1.0); }
%s}
""" % (COMPUTE % "data[0] = vec3(0.0);")

STD140 = """
shader Std140Parameter {
    buffer float data[] @binding(0);
    void put(buffer float values[] @std140, uint i) { values[i] = 1.0; }
%s}
""" % (COMPUTE % "data[0] = 0.0;")

SHADOWED = """
shader ShadowedParameter {
    buffer uint data[] @binding(0);
    void put(buffer uint values[], uint i) {
        values[i] = 1u;
        {
            uint values[2];
            values[0] = uint(values.length());
        }
    }
%s}
""" % (COMPUTE % "put(data, 0u);")


PROTOTYPE = """
shader Prototype {
    buffer float data[] @binding(0);
    float twice(float x);
    float twice(float x) { return x * 2.0; }
%s}
""" % (COMPUTE % "data[0] = twice(data[1]);")


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _run(command):
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    output = "\n".join(p for p in (completed.stdout, completed.stderr) if p.strip())
    assert completed.returncode == 0, output[-3000:]


def _tool(name):
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name} is not installed")
    return path


def test_hlsl_helpers_take_structured_buffers(tmp_path):
    code = _flat(_translate(tmp_path, HELPERS, "directx"))
    assert code.count("void put(RWStructuredBuffer<uint> values, uint i)") >= 2
    assert "float get(StructuredBuffer<float> s, uint i)" in code
    assert "void swap(RWStructuredBuffer<Pair> p, uint i)" in code
    assert "put(data, i);" in code and "swap(pairs, i);" in code
    assert not re.search(r"\w+ \w+\[\]\s*[,)]", code)


def test_slang_helpers_take_structured_buffers(tmp_path):
    code = _flat(_translate(tmp_path, HELPERS, "slang"))
    assert "void put(RWStructuredBuffer<uint> values, uint i)" in code
    assert "float get(StructuredBuffer<float> s, uint i)" in code
    assert "void swap(RWStructuredBuffer<Pair> p, uint i)" in code
    assert "buffer.GetDimensions(count, stride);" in code
    assert not re.search(r"\w+ \w+\[\]\s*[,)]", code)


def test_slang_prototype_is_a_forward_declaration(tmp_path):
    code = _flat(_translate(tmp_path, PROTOTYPE, "slang"))
    assert "float twice(float x);" in code
    assert code.count("float twice(float x) {") == 1


def test_glsl_specializes_helpers_for_the_buffer_they_receive(tmp_path):
    code = _flat(_translate(tmp_path, HELPERS, "opengl"))
    assert not re.search(r"\w+ \w+\[\]\s*[,)]", code)
    assert re.search(r"data\[\(values_offset \+ int\(i\)\)\] = ", code)
    assert "data.length()" in code and "src.length()" in code
    assert re.search(r"pairs\[\(p_offset \+ int\(i\)\)\]\.a = ", code)


def test_metal_helpers_take_device_pointers(tmp_path):
    code = _flat(_translate(tmp_path, HELPERS, "metal"))
    assert re.search(r"void put\(device uint\s*\*\s*values,", code)
    assert re.search(r"float get\(const device float\s*\*\s*s,", code)
    assert re.search(r"void swap\(device Pair\s*\*\s*p, uint i\)", code)


def test_sized_buffer_parameter_length_is_its_declared_count(tmp_path):
    code = _flat(_translate(tmp_path, SIZED, "directx"))
    assert "uint count(RWStructuredBuffer<uint> values)" in code
    assert "return uint(int(4));" in code or "return uint(4);" in code


@pytest.mark.parametrize("backend", TARGETS)
@pytest.mark.parametrize(
    ("source", "why"),
    [
        (VECTOR3, "holds elements of another type"),
        (STD140, "has an std140 layout"),
        (SHADOWED, "where a local declaration shadows it"),
    ],
    ids=["vec3", "std140", "shadowed-length"],
)
def test_buffer_parameters_without_one_layout_are_refused(
    tmp_path, backend, source, why
):
    with pytest.raises(StorageBufferParameterUnsupportedError, match=why) as caught:
        _translate(tmp_path, source, backend)
    assert caught.value.code.endswith("-storage-buffer-parameter-unsupported")


def test_hlsl_output_compiles_with_dxc(tmp_path):
    dxc = _tool("dxc")
    source = tmp_path / "shader.hlsl"
    source.write_text(_translate(tmp_path, HELPERS, "directx"), encoding="utf-8")
    _run([dxc, "-T", "cs_6_0", "-E", "CSMain", str(source)])


def test_glsl_output_compiles_with_glslang(tmp_path):
    glslang = _tool("glslangValidator")
    source = tmp_path / "shader.comp"
    source.write_text(_translate(tmp_path, HELPERS, "opengl"), encoding="utf-8")
    _run([glslang, str(source)])


def test_spirv_output_validates(tmp_path):
    spirv_as, spirv_val = _tool("spirv-as"), _tool("spirv-val")
    source = tmp_path / "shader.spvasm"
    source.write_text(_translate(tmp_path, HELPERS, "vulkan"), encoding="utf-8")
    binary = tmp_path / "shader.spv"
    _run([spirv_as, "--target-env", "vulkan1.1", str(source), "-o", str(binary)])
    _run([spirv_val, "--target-env", "vulkan1.1", str(binary)])


def test_slang_output_compiles_with_slangc(tmp_path):
    slangc = _tool("slangc")
    source = tmp_path / "shader.slang"
    source.write_text(_translate(tmp_path, HELPERS, "slang"), encoding="utf-8")
    _run(
        [
            slangc,
            str(source),
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            "compute",
            "-o",
            str(tmp_path / "shader.spv"),
        ]
    )


def test_slang_prototype_compiles_with_slangc(tmp_path):
    slangc = _tool("slangc")
    source = tmp_path / "prototype.slang"
    source.write_text(_translate(tmp_path, PROTOTYPE, "slang"), encoding="utf-8")
    _run(
        [
            slangc,
            str(source),
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            "compute",
            "-o",
            str(tmp_path / "prototype.spv"),
        ]
    )


def test_metal_output_compiles(tmp_path):
    xcrun = _tool("xcrun")
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        pytest.skip("the Metal compiler is not available")
    source = tmp_path / "shader.metal"
    source.write_text(_translate(tmp_path, HELPERS, "metal"), encoding="utf-8")
    _run(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.0",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "shader.air"),
        ]
    )

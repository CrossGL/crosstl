"""Slang declares a shader record as a ``[[vk::shader_record]]`` constant buffer in the record's layout (#3889).

A CrossGL shader record block (``Record record @glsl_buffer_block(shaderRecordEXT, std430);``) is
``[[vk::shader_record]] ConstantBuffer<Record, Std430DataLayout> record;`` in Slang, ``ConstantBuffer<Record>`` for
std140 and ``ConstantBuffer<Record, ScalarDataLayout>`` for scalar; slangc places it in the
``ShaderRecordBufferKHR`` storage class with those offsets. Slang input reads the same declarations back. A record
that a stage without a shader binding table reads is refused on SPIR-V, GLSL and Slang
(VUID-StandaloneSpirv-ShaderRecordBufferKHR-07119). When slangc and spirv-val are installed, the Slang output is
compiled and the SPIR-V checked for the record's storage class and offsets.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.import_refusals import SlangImportRefused
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused
from crosstl.translator.semantics.shader_records import ShaderRecordStageError

HLSL_RECORD = """
struct Record {
    float4 color;
};

[[vk::shader_record_ext]] ConstantBuffer<Record> record;

struct [raypayload] Payload {
    float4 color : read(caller) : write(closesthit, miss);
};

[shader("miss")]
void main(inout Payload payload) {
    payload.color = record.color;
}
"""


def crossgl_record(layout, stage="ray_miss", record_type="Record"):
    """A program whose ``stage`` entry reads a record with a float array before a vec3, laid out as ``layout``."""
    if stage == "ray_miss":
        entry = """    ray_miss {
        void main(inout Payload payload @ payload) {
            payload.color = vec4(record.weights[1], record.tint.y, record.scale, 0.0);
        }
    }"""
    else:
        entry = """    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer vec4* o;
        vec4 read_record() {
            return vec4(record.weights[1], record.tint.y, record.scale, 0.0);
        }
        void main() {
            o[gl_GlobalInvocationID.x] = read_record();
        }
    }"""
    return f"""
shader main {{
    struct Record {{
        float weights[2];
        vec3 tint;
        float scale;
    }}
    struct Payload {{
        vec4 color;
    }}
    {record_type} record @glsl_buffer_block(shaderRecordEXT, {layout});
{entry}
}}
"""


def translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def spirv_of(tmp_path, slang, stage="miss"):
    """The validated SPIR-V assembly slangc writes for ``slang``; skips when slangc is not installed."""
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    source = tmp_path / "probe.slang"
    source.write_text(slang, encoding="utf-8")
    binary = tmp_path / "probe.spv"
    completed = subprocess.run(
        [slangc, str(source), "-target", "spirv", "-entry", "main", "-stage", stage]
        + ["-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for tool in ("spirv-val", "spirv-dis"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} is not installed")
    validated = subprocess.run(
        ["spirv-val", "--target-env", "vulkan1.2", str(binary)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert validated.returncode == 0, validated.stderr
    return subprocess.run(
        ["spirv-dis", str(binary)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    ).stdout


def record_offsets(assembly):
    """The member offsets of the struct the ShaderRecordBufferKHR variable points to, by member index."""
    pointee = re.search(
        r"OpTypePointer ShaderRecordBufferKHR (%\w+)\n(?:.|\n)*?OpVariable", assembly
    )
    assert pointee, assembly
    struct = pointee.group(1)
    return {
        int(index): int(offset)
        for index, offset in re.findall(
            rf"OpMemberDecorate {re.escape(struct)} (\d+) Offset (\d+)", assembly
        )
    }


def test_hlsl_shader_record_is_a_slang_shader_record_constant_buffer(tmp_path):
    slang = translate(tmp_path, HLSL_RECORD, "slang", suffix=".hlsl")

    assert (
        "[[vk::shader_record]] ConstantBuffer<Record, Std430DataLayout> record;"
        in slang
    )
    assert "payload.color = record.color;" in slang
    assembly = spirv_of(tmp_path, slang)
    assert "ShaderRecordBufferKHR" in assembly
    assert "DescriptorSet" not in assembly


@pytest.mark.parametrize(
    "layout,argument,offsets",
    [
        ("std430", ", Std430DataLayout", {0: 0, 1: 16, 2: 28}),
        ("std140", "", {0: 0, 1: 32, 2: 44}),
        ("scalar", ", ScalarDataLayout", {0: 0, 1: 8, 2: 20}),
    ],
)
def test_slang_shader_record_keeps_the_record_layout(
    tmp_path, layout, argument, offsets
):
    slang = translate(tmp_path, crossgl_record(layout), "slang")

    assert f"[[vk::shader_record]] ConstantBuffer<Record{argument}> record;" in slang
    assert "struct Record" in slang
    assert record_offsets(spirv_of(tmp_path, slang)) == offsets


def test_slang_refuses_an_array_of_shader_records(tmp_path):
    with pytest.raises(SlangLoweringRefused, match="not an array of them"):
        translate(tmp_path, crossgl_record("std430", record_type="Record[2]"), "slang")


@pytest.mark.parametrize("backend", ["slang", "vulkan", "opengl"])
def test_a_compute_stage_that_reads_a_shader_record_is_refused(tmp_path, backend):
    with pytest.raises(ShaderRecordStageError) as raised:
        translate(tmp_path, crossgl_record("std430", stage="compute"), backend)

    assert "compute entry point 'main'" in str(raised.value)
    assert raised.value.code.endswith(f"{backend}-shader-record-stage-unsupported")


def test_a_compute_local_named_like_the_record_is_not_a_record_read(tmp_path):
    source = crossgl_record("std430", stage="compute").replace(
        "            o[gl_GlobalInvocationID.x] = read_record();",
        "            vec4 record = vec4(1.0);\n            o[gl_GlobalInvocationID.x] = record;",
    )

    slang = translate(tmp_path, source, "slang")

    assert "o[gl_GlobalInvocationID.x] = record;" in slang


@pytest.mark.parametrize(
    "argument,layout",
    [
        ("", "std140"),
        (", DefaultDataLayout", "std140"),
        (", Std430DataLayout", "std430"),
        (", ScalarDataLayout", "scalar"),
    ],
)
def test_slang_input_reads_the_shader_record_layout(tmp_path, argument, layout):
    source = (
        HLSL_RECORD.replace(
            "[[vk::shader_record_ext]] ConstantBuffer<Record> record;",
            f"[[vk::shader_record]] ConstantBuffer<Record{argument}> record;",
        )
        .replace("struct [raypayload] Payload", "struct Payload")
        .replace(" : read(caller) : write(closesthit, miss)", "")
    )

    crossgl = translate(tmp_path, source, "cgl", suffix=".slang")

    assert f"Record record @glsl_buffer_block(shaderRecordEXT, {layout});" in crossgl


def test_slang_input_refuses_an_unknown_shader_record_layout(tmp_path):
    source = (
        "struct Record { float4 color; };\n"
        "[[vk::shader_record]] ConstantBuffer<Record, CDataLayout> record;\n"
        "struct Payload { float4 color; };\n"
        '[shader("miss")]\n'
        "void main(inout Payload payload) { payload.color = record.color; }\n"
    )

    with pytest.raises(SlangImportRefused, match="CDataLayout"):
        translate(tmp_path, source, "cgl", suffix=".slang")

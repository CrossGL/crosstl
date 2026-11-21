"""HLSL output declares a shader record block for SPIR-V and, through a local root signature, for Direct3D 12.

See :mod:`crosstl.translator.codegen.hlsl_shader_records`.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.hlsl_shader_records import HlslShaderRecordError

HEADER = "#version 460\n#extension GL_EXT_ray_tracing : require\n"
PAYLOAD = "struct Payload {\n    vec4 color;\n};\nlayout(location = 0) rayPayloadInEXT Payload payload;\n"
MISS = (
    HEADER
    + PAYLOAD
    + "layout(shaderRecordEXT, %(layout)s) buffer Record {\n%(members)s\n} record;\n"
    + "void main() {\n    payload.color = %(use)s;\n}\n"
)


def _miss(tmp_path, members="    vec4 color;", use="record.color", layout="std430"):
    path = tmp_path / "record.rmiss"
    path.write_text(MISS % {"members": members, "use": use, "layout": layout})
    return str(path)


def _hlsl(path):
    return crosstl.translate(path, backend="directx", format_output=False)


def _crossgl(tmp_path, source, name="program.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def _dxc(tmp_path, hlsl, *arguments, warnings=False):
    dxc = shutil.which("dxc")
    if not dxc:
        return None
    (tmp_path / "record.hlsl").write_text(hlsl, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", "lib_6_7", *arguments, "record.hlsl", "-Fc", "record.txt"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert warnings or "warning" not in result.stderr, result.stderr
    return (tmp_path / "record.txt").read_text(encoding="utf-8")


def test_record_is_a_shader_record_buffer_for_spirv_and_root_constants_for_direct3d(
    tmp_path,
):
    hlsl = _hlsl(_miss(tmp_path))

    assert (
        "#ifdef __spirv__\n[[vk::shader_record_ext]] ConstantBuffer<Record> record;\n#else\n"
        "ConstantBuffer<Record> record : register(b0, space1);\n"
    ) in hlsl
    signature = re.search(
        r'LocalRootSignature (\w+) = \{\s*"RootConstants\(num32BitConstants=4, b0, space=1\)"\s*\};',
        hlsl,
    )
    assert signature, hlsl
    assert re.search(
        r'SubobjectToExportsAssociation \w+ = \{\s*"%s", "MissMain"\s*\};'
        % signature.group(1),
        hlsl,
    ), hlsl

    dxil = _dxc(tmp_path, hlsl)
    if dxil is not None:
        assert "LocalRootSignature" in dxil and '{ "MissMain" }' in dxil
        assert "Size:    16" in dxil
        spirv = _dxc(tmp_path, hlsl, "-spirv", "-fspv-target-env=vulkan1.2")
        assert "OpVariable %_ptr_ShaderRecordBufferKHR" in spirv


def test_native_hlsl_record_round_trips_with_its_local_root_signature(tmp_path):
    path = tmp_path / "record.hlsl"
    path.write_text(
        "struct [raypayload] Payload {\n    float4 color : read(caller) : write(miss);\n};\n"
        "struct Record {\n    float4 color;\n    uint2 ids;\n};\n"
        "[[vk::shader_record_ext]] ConstantBuffer<Record> record;\n"
        '[shader("miss")]\nvoid main(inout Payload payload) {\n'
        "    payload.color = record.color + float(record.ids.y);\n}\n"
    )
    hlsl = crosstl.translate(
        str(path), backend="directx", source_backend="directx", format_output=False
    )

    # float4 then uint2 ends at byte 24: six root constants.
    assert "RootConstants(num32BitConstants=6, b0, space=1)" in hlsl
    _dxc(tmp_path, hlsl)


def test_record_space_is_above_every_space_the_program_names(tmp_path):
    program = _crossgl(
        tmp_path,
        """shader main {
    struct Payload { vec4 color; };
    struct Record { vec4 color; };
    Payload payload @location(0) @rayPayloadInEXT;
    Record record @glsl_buffer_block(shaderRecordEXT, std430);
    @set(3) @binding(0)
    texture2D image;
    ray_miss {
        void main() {
            payload.color = record.color;
        }
    }
}
""",
    )
    hlsl = _hlsl(program)

    assert "ConstantBuffer<Record> record : register(b0, space4);" in hlsl
    assert "RootConstants(num32BitConstants=4, b0, space=4)" in hlsl


def test_only_the_entries_that_read_a_record_take_its_local_root_signature(tmp_path):
    program = _crossgl(
        tmp_path,
        """shader main {
    struct Payload { vec4 color; };
    struct Record { vec4 color; };
    Payload payload @location(0) @rayPayloadInEXT;
    Record record @glsl_buffer_block(shaderRecordEXT, std430);
    buffer float* outp;
    vec4 tint() {
        return record.color;
    }
    ray_closest_hit {
        void main() {
            payload.color = tint();
        }
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            outp[0] = 1.0;
        }
    }
}
""",
    )
    hlsl = _hlsl(program)

    association = re.search(r"SubobjectToExportsAssociation [^;]*;", hlsl).group(0)
    assert association.endswith('"ClosestHitMain" };'), association
    assert "CSMain" not in association
    # A library ignores the compute entry's [numthreads] without [shader("compute")], and warns.
    _dxc(tmp_path, hlsl, warnings=True)


def test_unread_record_has_no_local_root_signature(tmp_path):
    hlsl = _hlsl(_miss(tmp_path, use="vec4(1.0)"))

    assert "ConstantBuffer<Record> record : register(b0, space1);" in hlsl
    assert "LocalRootSignature" not in hlsl
    assert "SubobjectToExportsAssociation" not in hlsl


@pytest.mark.parametrize(
    "members, use, layout, reason",
    [
        # std430 puts the vec3 at 16; DXC's relaxed rule and constant-buffer packing put it at 4.
        (
            "    float s;\n    vec3 v;",
            "vec4(record.v, record.s)",
            "std430",
            r"offset of 'v' is 16 bytes under std430 and 4 under DXC's relaxed std430",
        ),
        # Constant-buffer array elements are 16 bytes apart; std430 and DXC keep 4.
        (
            "    float t[2];",
            "vec4(record.t[1])",
            "std430",
            r"stride of 't' is 4 bytes under std430 and 16 under HLSL constant-buffer packing",
        ),
        (
            "    mat4 m;",
            "record.m[0]",
            "std430",
            r"member 'm' is a matrix",
        ),
        # 255 vec4s take 4080 bytes; a record holds 4064 bytes of local root arguments.
        (
            "    vec4 data[255];",
            "record.data[3]",
            "std430",
            r"takes 4080 bytes, and a Direct3D 12 shader record holds at most 4064 bytes",
        ),
    ],
)
def test_record_whose_bytes_hlsl_cannot_reproduce_is_refused(
    tmp_path, members, use, layout, reason
):
    with pytest.raises(HlslShaderRecordError, match=reason) as error:
        _hlsl(_miss(tmp_path, members=members, use=use, layout=layout))
    assert error.value.code == "project.translate.directx-shader-record-unsupported"


def test_largest_record_fits_the_local_root_arguments(tmp_path):
    hlsl = _hlsl(_miss(tmp_path, members="    vec4 data[254];", use="record.data[3]"))

    assert "RootConstants(num32BitConstants=1016, b0, space=1)" in hlsl


def test_record_read_outside_a_ray_tracing_stage_is_refused(tmp_path):
    program = _crossgl(
        tmp_path,
        """shader main {
    struct Record { vec4 color; };
    Record record @glsl_buffer_block(shaderRecordEXT, std430);
    buffer float* outp;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            outp[0] = record.color.x;
        }
    }
}
""",
    )
    with pytest.raises(HlslShaderRecordError, match="only ray tracing stages"):
        _hlsl(program)

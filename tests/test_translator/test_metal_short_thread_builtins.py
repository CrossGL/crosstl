"""Metal kernels may declare their thread-index attributes with 16-bit types.

The Metal Shading Language Specification, section 5.2.3.6, lets ``[[thread_position_in_grid]]``,
``[[thread_position_in_threadgroup]]``, ``[[threadgroup_position_in_grid]]`` and ``[[threads_per_threadgroup]]``
take ``ushort``, ``ushort2`` or ``ushort3``, and the lane and index attributes take ``ushort``. Such a kernel keeps
its own entry point. Previously the importer accepted only the ``uint`` forms, dropped ``@stage_entry`` and every
such kernel became an unnamed ``main``, so a library of them produced duplicate SPIR-V entry-point names.

HLSL declares its thread-index system values as 32-bit, so the entry receives the 32-bit value and narrows it to the
declared 16-bit type in the body.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.host_workgroup_size import HostDefinedWorkgroupSizeError
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

METAL_LIBRARY = """
#include <metal_stdlib>
using namespace metal;

kernel void scale_rows(device float *data [[buffer(0)]],
                       ushort3 tpig [[thread_position_in_grid]],
                       ushort tiitg [[thread_index_in_threadgroup]]) {
    data[tpig.x] = data[tpig.x] * 2.0f + float(tiitg);
}

kernel void offset_rows(device float *data [[buffer(0)]],
                        uint3 gid [[thread_position_in_grid]]) {
    data[gid.x] = data[gid.x] + 1.0f;
}
"""

METAL_LANE = """
#include <metal_stdlib>
using namespace metal;

kernel void lanes(device float *data [[buffer(0)]],
                  ushort2 tpig [[thread_position_in_grid]],
                  ushort tiisg [[thread_index_in_simdgroup]]) {
    data[tpig.x] = float(tiisg);
}
"""

CROSSGL_COMPUTE = """
shader main {
    compute {
        RWStructuredBuffer<uint> data @buffer(0);
        @stage_entry
        @workgroup_size(64, 1, 1)
        void k(u16vec3 tpig @gl_GlobalInvocationID, uint16 index @gl_LocalInvocationIndex) {
            buffer_store(data, uint(tpig.x), uint(index));
        }
    }
}
"""


def _metal(tmp_path, source, name="kernels.metal"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _translate(path, backend, **options):
    return crosstl.translate(str(path), backend=backend, source_options=options or None)


def _flat(code):
    """``code`` with every whitespace run as one space: the formatter may wrap long signatures."""
    return " ".join(code.split())


def test_short_thread_attributes_keep_their_kernel_entry(tmp_path):
    crossgl = _translate(_metal(tmp_path, METAL_LIBRARY), "cgl")

    assert len(re.findall(r"@\s*stage_entry", crossgl)) == 2
    assert re.search(r"void scale_rows\(", crossgl)
    assert re.search(r"u16vec3 tpig @gl_GlobalInvocationID", crossgl)
    assert re.search(r"uint16 tiitg @gl_LocalInvocationIndex", crossgl)


def test_short_thread_attributes_give_distinct_spirv_entry_points(tmp_path):
    spirv = _translate(
        _metal(tmp_path, METAL_LIBRARY), "spirv", threadgroup_size=(64, 1, 1)
    )

    names = re.findall(r'OpEntryPoint GLCompute %\w+ "(\w+)"', spirv)
    assert sorted(names) == ["offset_rows", "scale_rows"]
    assert "LocalSize 64 1 1" in spirv
    assert not re.search(r'OpEntryPoint GLCompute %\w+ "main"', spirv)

    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly = tmp_path / "kernels.spvasm"
        module = tmp_path / "kernels.spv"
        assembly.write_text(spirv, encoding="utf-8")
        assembled = subprocess.run(
            ["spirv-as", str(assembly), "-o", str(module)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert assembled.returncode == 0, assembled.stderr
        validated = subprocess.run(
            ["spirv-val", str(module)], capture_output=True, text=True, timeout=60
        )
        assert validated.returncode == 0, validated.stdout + validated.stderr


def test_hlsl_entry_narrows_the_32_bit_system_values(tmp_path):
    hlsl = _translate(
        _metal(tmp_path, METAL_LIBRARY), "directx", threadgroup_size=(64, 1, 1)
    )

    assert "uint3 tpig_wide : SV_DispatchThreadID" in _flat(hlsl)
    assert "uint tiitg_wide : SV_GroupIndex" in _flat(hlsl)
    assert "uint16_t3 tpig = (uint16_t3)tpig_wide.xyz;" in _flat(hlsl)
    assert "uint16_t tiitg = (uint16_t)tiitg_wide;" in _flat(hlsl)
    assert not re.search(r"uint16_t3? \w+ : SV_", _flat(hlsl))

    dxc = shutil.which("dxc")
    if dxc is None:
        return
    artifact = tmp_path / "kernels.hlsl"
    artifact.write_text(hlsl, encoding="utf-8")
    result = subprocess.run(
        [
            dxc,
            "-T",
            "cs_6_2",
            "-E",
            "CSMain",
            "-WX",
            "-enable-16bit-types",
            str(artifact),
            "-Fo",
            str(tmp_path / "kernels.dxil"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_hlsl_wave_lane_index_narrows_explicitly(tmp_path):
    hlsl = _translate(
        _metal(tmp_path, METAL_LANE), "directx", threadgroup_size=(32, 1, 1)
    )

    assert "uint16_t tiisg = (uint16_t)(WaveGetLaneIndex());" in _flat(hlsl)
    assert "uint16_t2 tpig = (uint16_t2)tpig_wide.xy;" in _flat(hlsl)


def test_lane_index_without_threadgroup_size_is_refused(tmp_path):
    with pytest.raises(HostDefinedWorkgroupSizeError, match="lanes"):
        _translate(_metal(tmp_path, METAL_LANE), "spirv")


def test_crossgl_16_bit_compute_system_values_are_accepted_by_hlsl():
    ast = Parser(Lexer(CROSSGL_COMPUTE).get_tokens()).parse()
    hlsl = HLSLCodeGen().generate(ast)

    assert "uint3 tpig_wide : SV_DispatchThreadID" in _flat(hlsl)
    assert "uint index_wide : SV_GroupIndex" in _flat(hlsl)
    assert "uint16_t index = (uint16_t)index_wide;" in _flat(hlsl)


@pytest.mark.parametrize(
    "parameter",
    ["u16vec3 tpig @gl_LocalInvocationIndex", "int16 tpig @gl_LocalInvocationIndex"],
)
def test_hlsl_rejects_mismatched_16_bit_system_values(parameter):
    program = CROSSGL_COMPUTE.replace(
        "u16vec3 tpig @gl_GlobalInvocationID, uint16 index @gl_LocalInvocationIndex",
        parameter,
    ).replace("uint(tpig.x), uint(index)", "0u, 0u")
    ast = Parser(Lexer(program).get_tokens()).parse()

    with pytest.raises(Exception, match="SV_GroupIndex"):
        HLSLCodeGen().generate(ast)

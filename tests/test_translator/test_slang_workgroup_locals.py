"""Slang declares a function's workgroup variables ``static groupshared`` and writes structured buffers by subscript.

slangc refuses ``groupshared`` on a function-scope variable with automatic storage (E31201) and accepts a
``static groupshared`` local, which it gives one instance per workgroup. Slang's ``RWStructuredBuffer`` has no
``Store`` method (E30027), so ``buffer_store`` on one is a subscript assignment. Scalar ``uint`` thread and
workgroup ids, which the Metal importer keeps from ``uint tid [[thread_position_in_threadgroup]]``, bind their
system values' first component, as HLSL does.
"""

import os
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def _generate(code):
    return SlangCodeGen().generate(Parser(Lexer(code).get_tokens()).parse())


def _compute(body, helpers="", parameters="uint3 tid @gl_LocalInvocationID"):
    return f"""
shader main {{
    compute {{
        RWStructuredBuffer<float> output @buffer(0);
        {helpers}
        @numthreads(64, 1, 1)
        void main({parameters}) {{
            {body}
        }}
    }}
}}
"""


def _slangc():
    configured = os.environ.get("SLANGC")
    if configured and os.path.exists(configured):
        return configured
    return shutil.which("slangc")


REDUCE_METAL = """
#include <metal_stdlib>
using namespace metal;

kernel void reduce_sum(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint gid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[64];
    scratch[tid] = input[gid * 64 + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0;
        for (uint i = 0; i < 64; i++) {
            total += scratch[i];
        }
        output[gid] = total;
    }
}
"""


@pytest.mark.parametrize(
    "qualifier", ["threadgroup", "shared", "groupshared", "workgroup"]
)
def test_entry_workgroup_local_is_static_groupshared(qualifier):
    code = _generate(
        _compute(
            f"{qualifier} float scratch[64];\n"
            "scratch[tid.x] = float(tid.x);\n"
            "workgroupBarrier();\n"
            "buffer_store(output, tid.x, scratch[63u - tid.x]);"
        )
    )
    assert "    static groupshared float scratch[64];\n" in code
    assert "    groupshared float scratch" not in code


def test_helper_workgroup_local_is_static_groupshared():
    helper = """
        float rotate(uint lane) {
            threadgroup float lanes[64];
            lanes[lane] = float(lane);
            workgroupBarrier();
            return lanes[63u - lane];
        }
    """
    code = _generate(
        _compute("buffer_store(output, tid.x, rotate(tid.x));", helpers=helper)
    )
    assert "    static groupshared float lanes[64];\n" in code


def test_workgroup_local_sized_by_a_local_constant_stays_in_the_function():
    code = _generate(
        _compute(
            "const int count = 64;\n"
            "threadgroup float scratch[count];\n"
            "scratch[tid.x] = 1.0;\n"
            "buffer_store(output, tid.x, scratch[tid.x]);"
        )
    )
    assert "static groupshared float scratch[count];" in code


def test_workgroup_local_with_an_initializer_is_refused():
    with pytest.raises(SlangLoweringRefused, match="cannot have an initializer"):
        _generate(
            _compute(
                "threadgroup float total = 0.0;\n" "buffer_store(output, tid.x, total);"
            )
        )


def test_threadgroup_pointer_local_is_refused():
    with pytest.raises(SlangLoweringRefused, match="threadgroup pointer 'cursor'"):
        _generate(
            _compute(
                "threadgroup float scratch[64];\n"
                "threadgroup float* cursor = scratch;\n"
                "cursor[tid.x] = 1.0;\n"
                "buffer_store(output, tid.x, scratch[tid.x]);"
            )
        )


def test_structured_buffer_store_is_a_subscript_assignment():
    code = _generate(
        _compute(
            "buffer_store(output, tid.x, 1.0);\n"
            "for (uint i = 0u; i < 4u; buffer_store(output, i, 2.0)) { i++; }"
        )
    )
    assert "output[tid.x] = 1.0;" in code
    assert "for (uint i = 0u; i < 4u; output[i] = 2.0)" in code
    assert ".Store(" not in code


@pytest.mark.parametrize(
    "parameter, semantic",
    [
        ("uint tid @gl_LocalInvocationID", "uint tid : SV_GroupThreadID"),
        ("uint2 tid @gl_LocalInvocationID", "uint2 tid : SV_GroupThreadID"),
        ("uint tid @gl_WorkGroupID", "uint tid : SV_GroupID"),
        ("uint2 tid @gl_WorkGroupID", "uint2 tid : SV_GroupID"),
    ],
)
def test_scalar_and_pair_thread_ids_bind_their_system_value(parameter, semantic):
    code = _generate(
        _compute("buffer_store(output, 0u, float(tid.x));", parameters=parameter)
        if "uint2" in parameter
        else _compute("buffer_store(output, 0u, float(tid));", parameters=parameter)
    )
    assert semantic in code


@pytest.mark.parametrize(
    "parameter", ["float tid @gl_LocalInvocationID", "uint4 tid @gl_WorkGroupID"]
)
def test_other_thread_id_types_are_still_refused(parameter):
    with pytest.raises(ValueError, match="expected uint3 type"):
        _generate(_compute("buffer_store(output, 0u, 1.0);", parameters=parameter))


def _metal_reduce_to_slang(tmp_path):
    source = tmp_path / "reduce.metal"
    source.write_text(REDUCE_METAL)
    crossgl = crosstl.translate(str(source), backend="cgl", format_output=False)
    sized = tmp_path / "reduce.cgl"
    sized.write_text(
        crossgl.replace("@ host_workgroup_size", "@numthreads(64, 1, 1)").replace(
            "#include <metal_stdlib>", ""
        )
    )
    return crosstl.translate(str(sized), backend="slang", format_output=False)


def test_metal_threadgroup_reduction_translates_to_slang(tmp_path):
    code = _metal_reduce_to_slang(tmp_path)
    assert "uint tid : SV_GroupThreadID, uint gid : SV_GroupID" in code
    assert "    static groupshared float scratch[64];\n" in code
    assert "output[gid] = total;" in code


@pytest.mark.skipif(_slangc() is None, reason="slangc is required")
@pytest.mark.parametrize("target", ["spirv", "hlsl", "metal"])
def test_metal_threadgroup_reduction_compiles_with_slangc(tmp_path, target):
    shader = tmp_path / "reduce.slang"
    shader.write_text(_metal_reduce_to_slang(tmp_path))
    output = tmp_path / ("reduce.spv" if target == "spirv" else f"reduce.{target}")
    result = subprocess.run(
        [
            _slangc(),
            str(shader),
            "-target",
            target,
            "-entry",
            "reduce_sum",
            "-stage",
            "compute",
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    if target == "spirv" and shutil.which("spirv-val"):
        validation = subprocess.run(
            ["spirv-val", str(output)], capture_output=True, text=True, check=False
        )
        assert validation.returncode == 0, validation.stdout + validation.stderr

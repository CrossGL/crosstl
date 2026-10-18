"""Software collectives use immutable invocation identity, not source variables."""

import pytest

from crosstl.project import ProjectConfig, translate_project
from crosstl.translator import parse
from tests.test_translator.test_directx_float_atomics import _compile
from tests.test_translator.test_directx_software_reductions import (
    _codegen,
    _execute_words,
)
from tests.test_translator.test_directx_software_reductions import (
    directx_runtime as _directx_runtime,
)

directx_runtime = _directx_runtime
CASES = ("absent", "scalar", "vec2", "vec3", "index", "inout", "helper", "collision")


def _source(case):
    parameter = ""
    body = "uint observed = 0u;"
    helpers = ""
    call = "WaveShuffleDown(value, 1u)"
    if case in {"scalar", "vec2", "vec3"}:
        value_type = {"scalar": "uint", "vec2": "uvec2", "vec3": "uvec3"}[case]
        parameter = f", {value_type} localID @gl_LocalInvocationID"
        value = "localID" + (".x" if case != "scalar" else "")
        body = f"uint before = {value}; {value} += 1000u; uint observed = {value} - before;"
    if case in {"index", "inout"}:
        parameter = ", uint localIndex @gl_LocalInvocationIndex"
        body = "uint before = localIndex; localIndex += 1000u; uint observed = localIndex - before;"
        if case == "inout":
            helpers = "void change(inout uint index) { index += 1000u; }"
            body = "uint before = localIndex; change(localIndex); uint observed = localIndex - before;"
    if case == "helper":
        helpers = """uint nested(uint value, uint index @gl_LocalInvocationIndex) {
            index += 1000u;
            return WaveShuffleDown(value, 1u) + index;
        }
        uint outer(uint value) { return nested(value, 9999u); }
        """
        call = "outer(value)"
    if case == "collision":
        body = "uint groupIndex = 1000u; uint __crossgl_software_subgroup_invocation = 2000u; uint observed = groupIndex + __crossgl_software_subgroup_invocation;"
    return f"""shader SubgroupIdentity {{
        StructuredBuffer<uint> inputWords @register(t0);
        RWStructuredBuffer<uint> outputWords @register(u1);
        {helpers}
        compute {{
            @numthreads(32, 4, 1)
            void main(uvec3 globalID @gl_GlobalInvocationID {parameter}) {{
                uint destination = globalID.x + globalID.y * 96u;
                uint value = inputWords[destination];
                {body}
                uint neighbor = {call};
                uint total = WaveActiveSum(value);
                outputWords[destination * 3u] = neighbor;
                outputWords[destination * 3u + 1u] = total;
                outputWords[destination * 3u + 2u] = observed;
            }}
        }}
    }}"""


@pytest.mark.parametrize("case", CASES)
def test_immutable_subgroup_identity_compiles(tmp_path, case):
    generated = _codegen().generate_stage(parse(_source(case)), "compute")
    assert generated.count(": SV_GroupIndex") == 1
    assert "static uint __crossgl_software_subgroup_invocation" in generated
    assert "uint(None)" not in generated
    if case in {"index", "inout"}:
        assert "__crossgl_software_subgroup_invocation = uint(localIndex);" in generated
    if case == "scalar":
        assert "localID.y" not in generated and "localID.z" not in generated
    if case == "vec2":
        assert "localID.z" not in generated
    if case == "collision":
        assert "static uint __crossgl_software_subgroup_invocation_;" in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize("case", CASES)
def test_immutable_subgroup_identity_executes(tmp_path, directx_runtime, case):
    generated = _codegen().generate_stage(parse(_source(case)), "compute")
    words = [17 * index + 3 for index in range(384)]
    expected = []
    for index in range(len(words)):
        base = index // 32 * 32
        neighbor = min(index + 1, base + 31)
        observed = (
            1000
            if case in {"scalar", "vec2", "vec3", "index", "inout"}
            else 3000 if case == "collision" else 0
        )
        expected.extend(
            [
                words[neighbor] + (10999 if case == "helper" else 0),
                sum(words[base : base + 32]),
                observed,
            ]
        )
    assert _execute_words(tmp_path, generated, words, expected) == expected


@pytest.mark.parametrize("local_type", [None, "uint", "uint2", "uint3"])
def test_metal_project_partial_local_ids_compile(tmp_path, local_type):
    parameter = (
        f", {local_type} localID [[thread_position_in_threadgroup]]"
        if local_type
        else ""
    )
    statement = ""
    if local_type:
        component = "" if local_type == "uint" else ".x"
        statement = f"localID{component} += 1000u;"
    path = tmp_path / "kernel.metal"
    path.write_text(f"""#include <metal_stdlib>
    using namespace metal;
    kernel void transfer(device uint* output [[buffer(0)]],
                         uint3 globalID [[thread_position_in_grid]] {parameter}) {{
        uint index = globalID.x + globalID.y * 96u;
        {statement}
        output[index] = simd_shuffle_down(index, 1u) + simd_sum(index);
    }}""")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=("directx",),
            output_dir="out",
            include_patterns=(path.name,),
            entry_points={path.name: ("transfer",)},
            workgroup_size=(32, 4, 1),
            source_options={
                "metal": {
                    "target_options": {
                        "directx": {
                            "software_subgroup_width": 32,
                            "relative_wave_shuffle_out_of_range": "self",
                        }
                    }
                }
            },
        ),
        format_output=False,
    )
    report.write_json(tmp_path / "report.json")
    payload = report.to_json()
    assert payload["diagnostics"] == []
    assert payload["summary"]["translatedCount"] == 1
    generated = (tmp_path / payload["artifacts"][0]["path"]).read_text()
    _compile(generated, tmp_path)

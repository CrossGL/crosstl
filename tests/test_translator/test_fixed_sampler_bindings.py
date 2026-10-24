"""Fixed sampler state on Direct3D and Vulkan: a bound sampler plus a ``crosstl-fixed-samplers`` record.

Direct3D and Vulkan fix sampler state in host objects that the shader names through a binding: a static sampler or
sampler descriptor at a ``SamplerState`` register, or an immutable sampler or sampler descriptor at a Vulkan sampler
binding. A Metal ``constexpr sampler`` therefore becomes such a sampler, and the artifact records the state the host
must give it. See :mod:`crosstl.translator.semantics.fixed_samplers`.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.fixed_sampler_contracts import (
    parse_fixed_sampler_contract,
)
from crosstl.translator.semantics.fixed_samplers import (
    FixedSamplerCoordinatesUnsupportedError,
)
from crosstl.translator.semantics.sampler_states import (
    FixedSamplerStateUnsupportedError,
)

SAMPLERS = """#include <metal_stdlib>
using namespace metal;
constexpr sampler linearRepeat(coord::normalized, s_address::repeat,
                               t_address::mirrored_repeat, filter::linear,
                               mip_filter::linear, max_anisotropy(8),
                               lod_clamp(0.0f, 4.0f));
struct VOut { float4 position [[position]]; float2 uv; };
fragment float4 shade(VOut in [[stage_in]], texture2d<float> tex [[texture(0)]],
                      depth2d<float> depth [[texture(1)]],
                      sampler user [[sampler(0)]]) {
    constexpr sampler fallback;
    constexpr sampler compare(compare_func::less, address::clamp_to_border,
                              border_color::opaque_white);
    float4 color = tex.sample(linearRepeat, in.uv) + tex.sample(fallback, in.uv);
    float lit = depth.sample_compare(compare, in.uv, 0.5);
    return color * lit + tex.sample(sampler(metal::filter::nearest), in.uv)
           + tex.sample(user, in.uv);
}
"""

SHADOWED = """#include <metal_stdlib>
using namespace metal;
float4 blurred(texture2d<float> tex, float2 uv) {
    constexpr sampler s(filter::linear);
    return tex.sample(s, uv);
}
fragment float4 shade(float4 position [[position]], texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::repeat);
    float2 uv = position.xy / 64.0;
    return tex.sample(s, uv) + blurred(tex, uv);
}
"""

MIXED = """#include <metal_stdlib>
using namespace metal;
fragment float4 shade(float4 position [[position]], texture2d<float> tex [[texture(0)]],
                      texture2d<float> second [[texture(1)]], sampler user [[sampler(0)]]) {
    constexpr sampler s(filter::linear);
    float2 uv = position.xy / 64.0;
    return tex.sample(user, uv) + SAMPLE_SECOND;
}
"""

PIXEL = """#include <metal_stdlib>
using namespace metal;
fragment float4 shade(float4 position [[position]], texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(coord::pixel);
    return tex.sample(s, position.xy);
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _records(artifact):
    return {record["name"]: record for record in parse_fixed_sampler_contract(artifact)}


def _compile_hlsl(tmp_path, hlsl):
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    entry = re.search(r"(\w+)\([^)]*\)\s*:\s*SV_TARGET", hlsl, re.I).group(1)
    path = tmp_path / "shader.hlsl"
    path.write_text(hlsl, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", "ps_6_0", "-E", entry, str(path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


def _assemble(tmp_path, spirv):
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        return None
    assembly, module = tmp_path / "shader.spvasm", tmp_path / "shader.spv"
    assembly.write_text(spirv, encoding="utf-8")
    for command in (
        [spirv_as, "--target-env", "vulkan1.1", str(assembly), "-o", str(module)],
        [spirv_val, "--target-env", "vulkan1.1", str(module)],
    ):
        completed = subprocess.run(command, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    return module


def _spirv_bindings(spirv):
    names = dict(re.findall(r'OpName (%\w+) "(\w+)"', spirv))
    slots = {}
    for variable, decoration, value in re.findall(
        r"OpDecorate (%\w+) (DescriptorSet|Binding) (\d+)", spirv
    ):
        slots.setdefault(names.get(variable, variable), {})[decoration] = int(value)
    return slots


def test_hlsl_binds_each_fixed_sampler_to_the_next_free_register(tmp_path):
    hlsl = _translate(tmp_path, SAMPLERS, "directx")
    records = _records(hlsl)

    assert hlsl.startswith("// crosstl-fixed-samplers: ")
    assert set(records) == {
        "linearRepeat",
        "fallback",
        "compare",
        "crosstl_fixed_sampler_0",
    }
    # The sampler the host already binds keeps register s0; the fixed ones follow it.
    assert re.search(r"SamplerState user : register\(s0\);", hlsl)
    registers = {name: record["register"] for name, record in records.items()}
    assert sorted(registers.values()) == ["s1", "s2", "s3", "s4"]
    for name, register in registers.items():
        kind = "SamplerComparisonState" if name == "compare" else "SamplerState"
        assert re.search(rf"{kind} {name} : register\({register}\);", hlsl), name
    assert "depth.SampleCmp(compare, in_.uv, 0.5)" in hlsl
    assert "tex.Sample(crosstl_fixed_sampler_0, in_.uv)" in hlsl
    assert "sampler(" not in hlsl.split("\n", 1)[1]
    _compile_hlsl(tmp_path, hlsl)


def test_records_carry_every_parameter_of_the_source_state(tmp_path):
    records = _records(_translate(tmp_path, SAMPLERS, "directx"))

    linear = records["linearRepeat"]["state"]
    assert (linear["s_address"], linear["t_address"]) == ("repeat", "mirrored_repeat")
    assert (linear["mag_filter"], linear["min_filter"], linear["mip_filter"]) == (
        "linear",
        "linear",
        "linear",
    )
    assert linear["max_anisotropy"] == 8
    assert linear["lod_clamp"] == [0.0, 4.0]
    compare = records["compare"]["state"]
    assert compare["compare_func"] == "less"
    assert compare["border_color"] == "opaque_white"
    assert {compare[f"{axis}_address"] for axis in "str"} == {"clamp_to_border"}
    default = records["fallback"]["state"]
    assert (default["mag_filter"], default["mip_filter"], default["s_address"]) == (
        "nearest",
        "none",
        "clamp_to_edge",
    )
    assert records["crosstl_fixed_sampler_0"]["state"]["mag_filter"] == "nearest"


def test_spirv_puts_fixed_samplers_in_their_own_descriptor_set(tmp_path):
    spirv = _translate(tmp_path, SAMPLERS, "vulkan")
    records = _records(spirv)
    slots = _spirv_bindings(spirv)

    assert len(records) == 4
    assert sorted(
        (record["set"], record["binding"]) for record in records.values()
    ) == [(1, 0), (1, 1), (1, 2), (1, 3)]
    for name, record in records.items():
        assert slots[name] == {"DescriptorSet": 1, "Binding": record["binding"]}
    # The resources of the source keep set 0.
    for name in ("tex", "depth", "user"):
        assert slots[name].get("DescriptorSet", 0) == 0
    module = _assemble(tmp_path, spirv)
    if module is not None and shutil.which("spirv-dis"):
        disassembled = subprocess.run(
            ["spirv-dis", str(module)], capture_output=True, text=True, check=True
        ).stdout
        # The record is a debug instruction, so it survives assembly.
        assert _records(disassembled) == records


def test_fixed_samplers_leave_the_other_bindings_where_they_were(tmp_path):
    fixed = MIXED.replace("SAMPLE_SECOND", "second.sample(s, uv)")
    bound = MIXED.replace("SAMPLE_SECOND", "second.sample(user, uv)")
    hlsl = _translate(tmp_path, fixed, "directx")
    pattern = r"(\w+) (tex|second|user) : register\((\w+)\)"
    assert re.findall(pattern, hlsl) == re.findall(
        pattern, _translate(tmp_path, bound, "directx")
    )
    assert [r["register"] for r in parse_fixed_sampler_contract(hlsl)] == ["s1"]
    spirv = _spirv_bindings(_translate(tmp_path, fixed, "vulkan"))
    plain = _spirv_bindings(_translate(tmp_path, bound, "vulkan"))
    for name in ("tex", "second", "user"):
        assert spirv[name] == plain[name], name


def test_same_named_local_samplers_get_distinct_bindings(tmp_path):
    for backend in ("directx", "vulkan"):
        artifact = _translate(tmp_path, SHADOWED, backend)
        records = parse_fixed_sampler_contract(artifact)
        filters = sorted(record["state"]["mag_filter"] for record in records)
        assert filters == ["linear", "nearest"], backend
        names = [record["name"] for record in records]
        assert len(set(names)) == 2, names
        if backend == "directx":
            by_filter = {r["state"]["mag_filter"]: r["name"] for r in records}
            helper = hlsl_function(artifact, "blurred")
            assert f"tex.Sample({by_filter['linear']}, uv)" in helper
            assert by_filter["nearest"] not in helper
            _compile_hlsl(tmp_path, artifact)
        else:
            _assemble(tmp_path, artifact)


def hlsl_function(hlsl, name):
    start = re.search(rf"\b{name}\([^;{{]*\)\s*{{", hlsl)
    assert start, hlsl
    depth, index = 0, start.end() - 1
    for index in range(start.end() - 1, len(hlsl)):
        depth += {"{": 1, "}": -1}.get(hlsl[index], 0)
        if depth == 0:
            break
    return hlsl[start.start() : index + 1]


@pytest.mark.parametrize("backend", ["directx", "vulkan"])
def test_pixel_coordinates_are_refused(tmp_path, backend):
    with pytest.raises(FixedSamplerCoordinatesUnsupportedError) as raised:
        _translate(tmp_path, PIXEL, backend)
    assert (
        raised.value.project_diagnostic_code
        == f"project.translate.{backend}-unnormalized-sampler-coordinates-unsupported"
    )


def test_opengl_still_refuses_fixed_sampler_state(tmp_path):
    with pytest.raises(FixedSamplerStateUnsupportedError):
        _translate(tmp_path, SAMPLERS, "opengl")


KEYWORD_NAMED_SAMPLER = """#include <metal_stdlib>
using namespace metal;
constexpr sampler linear(filter::linear);
struct VOut { float4 position [[position]]; float2 uv; };
fragment float4 shade(VOut in [[stage_in]], texture2d<float> colour [[texture(0)]]) {
    return colour.sample(linear, in.uv);
}
"""


def test_hlsl_keeps_the_record_of_a_sampler_renamed_from_a_keyword(tmp_path):
    # `linear` is an HLSL interpolation modifier, so the sampler is renamed;
    # its record must name the declaration the host binds.
    hlsl = _translate(tmp_path, KEYWORD_NAMED_SAMPLER, "directx")
    records = _records(hlsl)
    assert set(records) == {"linear_"}
    assert re.search(r"SamplerState linear_ : register\(s0\);", hlsl), hlsl
    assert records["linear_"]["register"] == "s0"
    assert records["linear_"]["state"]["mag_filter"] == "linear"

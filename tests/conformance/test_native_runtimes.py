"""Native execution of one source fixture per runtime; opt in with CROSSTL_CONFORMANCE_NATIVE=1.

Each available runtime executes a source kernel of its own language unchanged and must reproduce the source-rule
oracle bit for bit, so a broken adapter, binding or driver identity is caught before translated results are trusted.
With CROSSTL_CONFORMANCE_REQUIRE naming runtimes, an unavailable required runtime fails instead of skipping.
"""

import json
import os
from dataclasses import replace

import pytest

from . import runner
from .process import run
from .runtimes import RUNTIMES, provision
from .test_multiple_inputs import make_case
from .toolchains import tool

# The HLSL check binds a 12-byte structured-buffer stride. It is not h_precise: DXC emits that kernel's
# a * b - c as an unfused fmul and fsub without fast-math flags, and WARP 1.0.19 and 1.0.21 still fuse them,
# so it measures the WARP JIT rather than the adapter. Its scoreboard cells keep recording the divergence.
# The second kernel of each language binds a constants block (Metal constant buffer, D3D12 constant-buffer
# view, OpenGL uniform block, Vulkan push constants), so the adapter's constants path is checked as well.
REFERENCE = {
    "metal": ("m_ptr_arith", "m_constant_struct"),
    "hlsl": ("h_float3_stride", "h_cbuffer_packing"),
    "glsl": ("g_std430_vec3_float", "g_std140_uniform"),
    "spirv": ("s_nocontraction", "s_push_constant_std430", "s_std140_uniform"),
}
REQUIRED = set(
    filter(None, os.environ.get("CROSSTL_CONFORMANCE_REQUIRE", "").split(","))
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CROSSTL_CONFORMANCE_NATIVE") != "1",
    reason="set CROSSTL_CONFORMANCE_NATIVE=1 to execute on native runtimes",
)


def test_required_runtimes_are_known():
    assert REQUIRED <= set(RUNTIMES)


@pytest.mark.parametrize("name", sorted(RUNTIMES))
def test_reference_kernel_reproduces_its_oracle(name, tmp_path):
    runtime = RUNTIMES[name]
    provided = provision(runtime)
    if not provided.available:
        if name in REQUIRED:
            pytest.fail(f"{name} is required but unavailable: {provided.reason}")
        pytest.skip(provided.reason)
    output = tmp_path / "run"
    assert (
        runner.main(
            [
                "--output",
                str(output),
                "--cases",
                ",".join(REFERENCE[runtime.target]),
                "--targets",
                runtime.target,
                "--runtimes",
                name,
                "--reference",
            ]
        )
        == 0
    )
    records = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    # Name the failing stage and its message, so a CI log explains a runtime failure without an artifact.
    reasons = [
        {
            "outcome": record["outcome"],
            "validation": [
                (
                    os.path.basename((item.get("command") or ("?",))[0]),
                    item.get("status"),
                    item.get("log", "")[-600:],
                )
                for item in record.get("validation") or ()
                if item.get("status") != "accepted"
            ],
            "execution": {
                key: value
                for key, value in (record.get("execution") or {}).items()
                if key in {"status", "exception", "message", "device"}
            },
            "comparison": record.get("comparison"),
        }
        for record in records
    ]
    assert [(record["runtime"], record["outcome"]) for record in records] == [
        (name, "preserved")
    ] * len(REFERENCE[runtime.target]), json.dumps(reasons, indent=2, default=str)
    assert all(runtime.expect in record["execution"]["device"] for record in records)


D3D12_MODELS = tuple(f"cs_6_{minor}" for minor in range(10))


def test_d3d12_device_runs_the_harness_shader_model(tmp_path):
    """The HLSL profile the harness validates with must also create pipelines on the D3D12 device."""
    runtime = RUNTIMES["d3d12-warp"]
    provided = provision(runtime)
    if not provided.available:
        if "d3d12-warp" in REQUIRED:
            pytest.fail(f"d3d12-warp is required but unavailable: {provided.reason}")
        pytest.skip(provided.reason)
    from .native import d3d12_pipeline_models
    from .toolchains import tool
    from .validators import HLSL_PROFILE

    dxc = tool("dxc")
    assert dxc.available, dxc.reason
    device, models = d3d12_pipeline_models(dxc.command, tmp_path, D3D12_MODELS)
    report = "\n".join(f"{model}: {result}" for model, result in models.items())
    print(f"{device}\n{report}")
    assert models[HLSL_PROFILE] == "ok", f"{device}\n{report}"


@pytest.mark.parametrize(
    "name",
    [name for name, runtime in RUNTIMES.items() if runtime.target != "hlsl"],
)
@pytest.mark.parametrize("input_count", [2, 3])
def test_native_mixed_inputs_reproduce_the_oracle(tmp_path, name, input_count):
    runtime = RUNTIMES[name]
    provided = provision(runtime)
    if not provided.available:
        if name in os.environ.get("CROSSTL_CONFORMANCE_REQUIRE", "").split(","):
            pytest.fail(provided.reason)
        pytest.skip(provided.reason)
    case = make_case(
        tmp_path, "metal" if runtime.target == "metal" else "glsl", input_count
    )
    if runtime.target == "spirv":
        compiler, disassembler = tool("glslangValidator"), tool("spirv-dis")
        if not compiler.available or not disassembler.available:
            pytest.skip(
                "glslangValidator and spirv-dis are needed for the mixed-input fixture"
            )
        binary, assembly = tmp_path / "probe.spv", tmp_path / "probe.spvasm"
        compiled = run([*compiler.command, "-V", case.src, "-o", binary])
        assert compiled.status == "accepted", compiled.log
        disassembled = run([*disassembler.command, binary, "-o", assembly])
        assert disassembled.status == "accepted", disassembled.log
        case = replace(case, src=assembly, lang="spirv")
    [record] = runner.run_suite(
        [case],
        [runtime.target],
        ["strict"],
        {runtime.target: [(runtime, provided)]},
        tmp_path / "run",
        reference=True,
    )
    assert record["outcome"] == "preserved", record
    assert record["inputUnchanged"] and record["outputGuardIntact"]


@pytest.mark.parametrize(
    "change, expected",
    [("same", "supported"), ("different", "wrong"), ("input-write", "wrong")],
)
def test_coverage_executes_mixed_input_probes(change, expected):
    from tools.coverage import execution, probes
    from tools.coverage.model import Feature, Param, Result, Signature, Type

    runtime = RUNTIMES["opengl-llvmpipe"]
    provided = provision(runtime)
    if not provided.available:
        if runtime.id in REQUIRED:
            pytest.fail(provided.reason)
        pytest.skip(provided.reason)
    feature = Feature(
        "glsl",
        "function",
        "ldexp",
        Signature(Type("float"), (Param(Type("float")), Param(Type("int")))),
    )
    source = probes.build(feature, "glsl").source.replace(
        "#version 460", "#version 450"
    )
    translated = source
    if change == "different":
        translated = source.replace("ldexp(a0, a1)", "ldexp(a0, a1 + 1)")
    elif change == "input-write":
        translated = source.replace("outp[i] = r;", "outp[i] = r; iin[i] = -1;")
    [result], used = execution.run(
        [Result(feature, "output", "supported", sources=(source, translated))],
        runtimes=runtime.id,
    )
    assert used == [runtime.id]
    assert result.outcome == expected, result
    assert result.runtime == runtime.id
    if change == "input-write":
        assert "into its input buffer" in result.detail

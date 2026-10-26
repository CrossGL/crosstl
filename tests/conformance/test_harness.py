import json
import sys
from dataclasses import replace

import numpy as np
import pytest

from . import runner, runtimes, scoreboard, toolchains, validators
from .cases import BY_ID, CASES, compare
from .heldout import heldout_cases
from .interface import InterfaceChanged, hlsl_reflected_strides, interface
from .process import CommandResult, require_artifact, run
from .scoreboard import check, improvements, merge, results_scope, snapshot
from .validators import dxil_is_signed, validate


def test_hazard_inventory_and_oracles():
    # The 66 evidence kernels, the binary16 division, square-root, fma and conversion cases, the OpenCL
    # saturated conversions to uint (round to nearest even) and to char, the HLSL and GLSL workgroup-memory
    # exchanges, the CUDA warp shuffles called from CrossGL, GLSL and SPIR-V runtime-array lengths, and GLSL matrix
    # orientation in values, in matrix-valued expressions and in storage blocks, the GLSL packing, bit-field, boolean-mix and frexp/ldexp/modf
    # built-ins, SPIR-V's FrexpStruct and ModfStruct, GLSL and SPIR-V extended integer arithmetic, and SPIR-V
    # integer instructions whose opcode signedness differs from their operand types, Metal clz/ctz and GLSL
    # findMSB/findLSB bit scans, SPIR-V structured control
    # flow with function variables and with phis, GLSL short-circuit operators, an HLSL structured buffer of structs with vector members, a
    # Metal buffer of structs with packed vector members, Metal buffers of float3, float3x3 and structs in the
    # natural MSL layout, Metal helper functions that take packed and natural buffer pointers, a CUDA buffer of
    # structs with a three-component vector member, and constants
    # blocks in GLSL std140, SPIR-V std140 and std430 push constants, HLSL cbuffer packing, HLSL row_major cbuffer
    # matrices and MSL layout.
    assert len(CASES) == len(BY_ID) == 115
    assert {case.lang for case in CASES} == {
        "metal",
        "hlsl",
        "glsl",
        "spirv",
        "cuda",
        "opencl",
        "cgl",
    }
    for case in CASES:
        expected = case.expected()
        if expected is not None:
            assert compare(case, expected.astype(case.out_dtype).tobytes())[0], case.id
            assert not compare(case, b"")[0], case.id


def test_catalogue_undefined_controls_are_explicit():
    assert not any(BY_ID["m_srem_sign"].readings.values())
    assert not any(BY_ID["s_nmin_nan"].readings.values())


def test_bit_comparison_preserves_zero_sign_and_nan_class():
    case = replace(
        BY_ID["m_fmin_nan"],
        oracle=lambda _: np.array([0.0] * 64, np.float32),
        out_count=64,
    )
    expected = case.expected().astype(np.float32)
    actual = expected.copy()
    index = np.flatnonzero(expected == 0)[0]
    actual[index] = -expected[index]
    assert not compare(case, actual.tobytes())[0]
    actual[index] = np.nan
    assert not compare(case, actual.tobytes())[0]


def test_heldout_generation_is_deterministic_and_does_not_mutate_fixtures(tmp_path):
    from . import cases

    original = cases.CASES_DIR
    first = heldout_cases(tmp_path / "first", n_per_family=1)
    second = heldout_cases(tmp_path / "second", n_per_family=1)
    assert len(first) == len(second) == 17
    assert cases.CASES_DIR == original
    for a, b in zip(first, second):
        assert a.src.read_bytes() == b.src.read_bytes()
        np.testing.assert_array_equal(a.expected(), b.expected())
        assert compare(a, a.expected().tobytes())[0]
    with pytest.raises(FileExistsError):
        heldout_cases(tmp_path / "first")


def test_process_distinguishes_timeout_missing_tool_and_missing_output(tmp_path):
    assert run(["crosstl-absent-test-executable"]).status == "unavailable"
    assert (
        run([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.1).status
        == "timeout"
    )
    success = run([sys.executable, "-c", "pass"])
    assert require_artifact(success, tmp_path / "missing.bin").status == "harness-error"


# ---------------------------------------------------------------------------------------------- interface
def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_every_source_fixture_exposes_the_two_slot_interface():
    for case in CASES:
        if case.lang in validators.TARGETS:
            found = interface(
                case.src, case.lang, case.group, case.constants is not None
            )
            assert found["bindings"] == (0, 1), case.id
            assert found["group"][1:] == (1, 1), case.id
            assert (found["constants"] is None) == (case.constants is None), case.id


def test_constants_cases_cover_each_source_language_and_document_their_offsets():
    from . import constants_cases

    with_constants = {case.lang: case for case in CASES if case.constants is not None}
    assert set(with_constants) == set(validators.TARGETS)
    words = {
        layout: np.frombuffer(constants_cases.block(fields), dtype=np.float32)
        for layout, fields in {
            "std140": constants_cases.STD140,
            "std430": constants_cases.STD430,
            "hlsl": constants_cases.HLSL,
            "msl": constants_cases.MSL,
        }.items()
    }
    canary = np.uint32(constants_cases.CANARY).view(np.float32)
    # bias follows the vec3 in its 16-byte slot except in MSL, where float3 takes all 16 bytes.
    assert [words[name][3] for name in ("std140", "std430", "hlsl")] == [11.0] * 3
    assert words["msl"][3] == canary and words["msl"][4] == 11.0
    # tail: after a slot-aligned array (std140), after the 4-byte-stride array (std430), in the last array
    # element's register (HLSL) and at its own 8-byte alignment (MSL).
    assert [
        words[name][index]
        for name, index in (("std140", 41), ("std430", 29), ("hlsl", 15), ("msl", 33))
    ] == [53.0] * 4
    assert all(len(constants_cases.block(f)) == 256 for f in (constants_cases.HLSL,))
    with pytest.raises(ValueError, match="overlaps"):
        constants_cases.block(((0, (1.0, 2.0), np.float32), (4, (3.0,), np.float32)))


def test_constants_blocks_are_discovered_in_every_language(tmp_path):
    metal = (
        "kernel void k(device const float* x [[buffer(0)]], device float* o [[buffer(1)]],\n"
        "              constant P& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {}"
    )
    found = interface(write(tmp_path, "a.metal", metal), "metal", 8, True)
    assert found["bindings"] == (0, 1)
    assert found["constants"] == {"slot": 3, "kind": "uniform"}
    with pytest.raises(InterfaceChanged, match="host does not provide"):
        interface(write(tmp_path, "a.metal", metal), "metal", 8)
    with pytest.raises(InterfaceChanged, match="shares a storage-buffer index"):
        shared = metal.replace(
            "constant P& p [[buffer(3)]]", "constant P& p [[buffer(1)]]"
        )
        interface(write(tmp_path, "b.metal", shared), "metal", 8, True)
    pointer = metal.replace("device const float* x", "constant float* x")
    assert interface(write(tmp_path, "c.metal", pointer), "metal", 8, True)[
        "bindings"
    ] == (0, 1)
    with pytest.raises(InterfaceChanged, match="Expected one constants block"):
        interface(
            write(
                tmp_path,
                "d.metal",
                metal.replace(
                    "constant P& p [[buffer(3)]]", "device P& p [[buffer(3)]]"
                ),
            ),
            "metal",
            8,
            True,
        )

    hlsl = """StructuredBuffer<float> x : register(t0);
RWStructuredBuffer<float> o : register(u1);
cbuffer P : register(b2) { float a; };
[numthreads(64, 1, 1)] void CSMain(uint3 id : SV_DispatchThreadID) {}
"""
    found = interface(write(tmp_path, "a.hlsl", hlsl), "hlsl", 1, True)
    assert found["constants"] == {"slot": 2, "kind": "uniform", "register": "b2"}
    typed = hlsl.replace(
        "cbuffer P : register(b2) { float a; };",
        "struct Q { float a; };\n" "ConstantBuffer<Q> p : register(b0);",
    )
    assert (
        interface(write(tmp_path, "b.hlsl", typed), "hlsl", 1, True)["constants"][
            "slot"
        ]
        == 0
    )
    for text, message in (
        (hlsl.replace(" : register(b2)", ""), "no explicit register"),
        (hlsl.replace("register(b2)", "register(b2, space1)"), "space1"),
        (
            hlsl + "cbuffer R : register(b3) { float b; };\n",
            "Expected one constants block",
        ),
    ):
        with pytest.raises(InterfaceChanged, match=message):
            interface(write(tmp_path, "c.hlsl", text), "hlsl", 1, True)

    glsl = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer In { float a[]; };
layout(std430, binding = 1) buffer Out { float o[]; };
layout(std140, binding = 3) uniform P { float k; } p;
void main() {}
"""
    assert interface(write(tmp_path, "a.comp", glsl), "glsl", 1, True)["constants"] == {
        "slot": 3,
        "kind": "uniform",
    }
    push = glsl.replace("std140, binding = 3", "push_constant")
    assert interface(write(tmp_path, "b.comp", push), "glsl", 1, True)["constants"] == {
        "slot": None,
        "kind": "push",
    }
    for text, message in (
        (
            glsl.replace("std140, binding = 3", "std140"),
            "uniform block has no explicit binding",
        ),
        (glsl + "uniform float loose;\n", "uniforms the host does not provide"),
        (glsl, "host does not provide"),
    ):
        with pytest.raises(InterfaceChanged, match=message):
            interface(write(tmp_path, "c.comp", text), "glsl", 1, text != glsl)

    spirv = """OpEntryPoint GLCompute %main "main" %id
OpExecutionMode %main LocalSize 8 1 1
OpDecorate %In Block
OpDecorate %P Block
OpDecorate %a DescriptorSet 0
OpDecorate %a Binding 0
OpDecorate %b DescriptorSet 0
OpDecorate %b Binding 1
OpDecorate %p DescriptorSet 0
OpDecorate %p Binding 2
%ptr_ssbo = OpTypePointer StorageBuffer %In
%ptr_ubo = OpTypePointer Uniform %P
%a = OpVariable %ptr_ssbo StorageBuffer
%b = OpVariable %ptr_ssbo StorageBuffer
%p = OpVariable %ptr_ubo Uniform
"""
    found = interface(write(tmp_path, "a.spvasm", spirv), "spirv", 1, True)
    assert found["bindings"] == (0, 1)
    assert found["constants"] == {"slot": 2, "kind": "uniform"}
    # A BufferBlock struct in Uniform storage is a storage buffer, so it takes a storage slot.
    legacy = spirv.replace("OpDecorate %P Block", "OpDecorate %P BufferBlock")
    with pytest.raises(InterfaceChanged, match=r"slots \[0, 1\], found \[0, 1, 2\]"):
        interface(write(tmp_path, "b.spvasm", legacy), "spirv", 1, True)
    pushed = (
        spirv.replace("OpDecorate %p DescriptorSet 0\nOpDecorate %p Binding 2\n", "")
        .replace(
            "%ptr_ubo = OpTypePointer Uniform %P",
            "%ptr_pc = OpTypePointer PushConstant %P",
        )
        .replace(
            "%p = OpVariable %ptr_ubo Uniform", "%p = OpVariable %ptr_pc PushConstant"
        )
    )
    assert interface(write(tmp_path, "c.spvasm", pushed), "spirv", 1, True)[
        "constants"
    ] == {
        "slot": None,
        "kind": "push",
    }
    with pytest.raises(InterfaceChanged, match="host does not provide"):
        interface(write(tmp_path, "c.spvasm", pushed), "spirv", 1)


def test_staged_jobs_carry_the_constants_block(tmp_path, monkeypatch):
    from . import native

    case = BY_ID["g_std140_uniform"]
    cell = runner.Cell(case, "spirv", "strict", tmp_path, {})
    cell.artifact, cell.binary = tmp_path / "shader.spvasm", tmp_path / "shader.spv"
    cell.abi = {
        "entry": "main",
        "group": (64, 1, 1),
        "bindings": (0, 1),
        "constants": {"slot": 2, "kind": "uniform"},
    }
    path, size = runner.stage(cell, runtimes.RUNTIMES["vulkan-lavapipe"])
    job = json.loads(path.read_text())
    assert (path.parent / "constants.bin").read_bytes() == case.constants
    assert job["constants"] == {"slot": 2, "kind": "uniform"} and size == 512 * 4
    seen = []

    def vkrun(args, **_):
        seen.append(args)
        return CommandResult(
            tuple(args),
            "accepted",
            0,
            "ok gpu_ns=1\ndriver=llvmpipe\ndevice=llvmpipe\n",
        )

    monkeypatch.setattr(native, "run", vkrun)
    native.execute_vulkan(job)
    assert seen[-1][-1] == f"2:uniform:{path.parent / 'constants.bin'}"
    native.execute_vulkan(dict(job, constants={"slot": None, "kind": "push"}))
    assert seen[-1][-1] == f"push:{path.parent / 'constants.bin'}"
    with pytest.raises(RuntimeError, match="cannot bind a push constants block"):
        native._constants(
            dict(job, constants={"slot": None, "kind": "push"}), "uniform"
        )


def test_metal_slots_come_from_buffer_indices_not_names(tmp_path):
    kernel = "kernel void k(device const float* xy [[buffer(0)]], device float* anything [[buffer(1)]]) {}"
    assert interface(write(tmp_path, "a.metal", kernel), "metal", 32) == {
        "entry": "k",
        "group": (32, 1, 1),
        "bindings": (0, 1),
        "constants": None,
    }
    with pytest.raises(InterfaceChanged, match="slots"):
        interface(
            write(tmp_path, "b.metal", kernel.replace("buffer(1)", "buffer(3)")),
            "metal",
            32,
        )
    extra = kernel.replace(") {}", ", texture2d<float> t [[texture(0)]]) {}")
    with pytest.raises(InterfaceChanged, match="resource kinds"):
        interface(write(tmp_path, "c.metal", extra), "metal", 32)
    with pytest.raises(InterfaceChanged, match="one compute entry"):
        interface(write(tmp_path, "d.metal", kernel + "\n" + kernel), "metal", 32)


def test_hlsl_interface_reports_register_kinds_and_strides(tmp_path):
    source = """struct Pair { int a; float b[3]; };
StructuredBuffer<Pair> src : register(t0);
RWByteAddressBuffer dst : register(u1, space0);
[numthreads(64, 1, 1)] void CSMain(uint3 id : SV_DispatchThreadID) {}
"""
    found = interface(write(tmp_path, "a.hlsl", source), "hlsl", 1)
    assert found["entry"] == "CSMain" and found["group"] == (64, 1, 1)
    assert found["resources"] == [
        {"register": "t0", "name": "src", "kind": "structured", "stride": 16},
        {"register": "u1", "name": "dst", "kind": "raw", "stride": 0},
    ]
    with pytest.raises(
        InterfaceChanged, match="constants block the host does not provide"
    ):
        interface(
            write(
                tmp_path,
                "b.hlsl",
                "cbuffer Info : register(b0) { uint n; };\n" + source,
            ),
            "hlsl",
            1,
        )
    with pytest.raises(InterfaceChanged, match="Register class"):
        interface(
            write(tmp_path, "c.hlsl", source.replace("register(t0)", "register(u0)")),
            "hlsl",
            1,
        )


def test_glsl_and_spirv_interfaces_refuse_extra_host_state(tmp_path):
    glsl = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer In { int a[]; };
layout(std430, binding = 1) buffer Out { int o[]; };
void main() {}
"""
    assert interface(write(tmp_path, "a.comp", glsl), "glsl", 1)["group"] == (64, 1, 1)
    with pytest.raises(InterfaceChanged, match="uniforms"):
        interface(write(tmp_path, "b.comp", glsl + "uniform uint n;\n"), "glsl", 1)
    spirv = """OpEntryPoint GLCompute %main "main" %id
OpExecutionMode %main LocalSize 8 1 1
OpDecorate %a DescriptorSet 0
OpDecorate %a Binding 0
OpDecorate %b DescriptorSet 0
OpDecorate %b Binding 1
"""
    assert interface(write(tmp_path, "a.spvasm", spirv), "spirv", 1)["group"] == (
        8,
        1,
        1,
    )
    with pytest.raises(InterfaceChanged, match="descriptor sets"):
        interface(
            write(
                tmp_path,
                "b.spvasm",
                spirv.replace("%b DescriptorSet 0", "%b DescriptorSet 1"),
            ),
            "spirv",
            1,
        )
    with pytest.raises(InterfaceChanged, match="one-dimensional"):
        interface(
            write(tmp_path, "c.spvasm", spirv.replace("8 1 1", "8 2 1")), "spirv", 1
        )


def test_dxc_reflection_strides_are_parsed():
    dump = """; Buffer Definitions:
;
; Resource bind info for value_22
; {
;
;   struct struct.Inp
;   {
;       int x;                                        ; Offset:    0
;   } $Element;                                       ; Offset:    0 Size:     4
;
; }
;
; Resource bind info for o
; {
;
;   float4 $Element;                                  ; Offset:    0 Size:    16
;
; }
"""
    assert hlsl_reflected_strides(dump) == {"value_22": 4, "o": 16}


# ---------------------------------------------------------------------------------------------- validators
def test_dxil_signature_detects_unvalidated_containers(tmp_path):
    signed, unsigned = tmp_path / "signed.dxil", tmp_path / "unsigned.dxil"
    signed.write_bytes(b"DXBC" + bytes(range(1, 17)) + bytes(8))
    unsigned.write_bytes(b"DXBC" + bytes(24))
    assert dxil_is_signed(signed) and not dxil_is_signed(unsigned)


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    commands = []

    def observed(command, **_):
        commands.append([str(word) for word in command])
        if "-Fo" in command or "-o" in command:
            (tmp_path / "binary").write_bytes(b"DXBC" + bytes(range(1, 17)) + bytes(8))
        return CommandResult(tuple(map(str, command)), "accepted", 0, "")

    monkeypatch.setattr(validators, "run", observed)
    monkeypatch.setattr(
        validators, "tool", lambda name: toolchains.Tool(name, (name,), "test")
    )
    return commands


def test_native_command_configurations_are_retained(tmp_path, recorded):
    source, binary = tmp_path / "src", tmp_path / "binary"
    results, _ = validate(source, "hlsl", binary, "main")
    assert [item.status for item in results] == ["accepted", "accepted"]
    assert (
        "-Gis" in recorded[-2]
        and "2021" in recorded[-2]
        and recorded[-1][1] == "-dumpbin"
    )
    validate(source, "hlsl", binary, "main", config="relaxed")
    assert "-Gis" not in recorded[-2]
    validate(source, "spirv", binary, "main", environment="vulkan1.3")
    assert recorded[-1][0] == "spirv-val" and "vulkan1.3" in recorded[-1]
    validate(source, "metal", binary, "main", config="relaxed")
    assert "-ffast-math" in recorded[-1]
    with pytest.raises(ValueError):
        validate(source, "metal", binary, "main", config="fast")


def test_unsigned_dxil_is_invalid(tmp_path, recorded, monkeypatch):
    monkeypatch.setattr(validators, "dxil_is_signed", lambda _: False)
    results, _ = validate(tmp_path / "src", "hlsl", tmp_path / "binary", "main")
    assert results[-1].status == "rejected" and "unsigned" in results[-1].log


def test_missing_validator_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        validators,
        "tool",
        lambda name: toolchains.Tool(name, (), None, "absent for the test"),
    )
    results, _ = validate(tmp_path / "src", "glsl", tmp_path / "binary", "main")
    assert results[0].status == "unavailable" and "absent" in results[0].log


# ---------------------------------------------------------------------------------------------- toolchains
def test_tool_override_and_sdk_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("CROSSTL_CONFORMANCE_DXC", str(tmp_path / "missing-dxc"))
    toolchains.tool.cache_clear()
    try:
        assert not toolchains.tool("dxc").available
    finally:
        toolchains.tool.cache_clear()
    for version in ("1.3.290.0", "1.4.363.0", "1.4.9.0"):
        (tmp_path / "VulkanSDK" / version / "macOS" / "bin").mkdir(parents=True)
    monkeypatch.delenv("VULKAN_SDK", raising=False)
    monkeypatch.setattr(toolchains.Path, "home", lambda: tmp_path)
    toolchains.vulkan_sdk.cache_clear()
    try:
        assert toolchains.vulkan_sdk() == tmp_path / "VulkanSDK" / "1.4.363.0" / "macOS"
    finally:
        toolchains.vulkan_sdk.cache_clear()


def test_inventory_markdown_lists_every_runtime():
    data = {
        "host": {
            "system": "Linux",
            "release": "6",
            "machine": "x86_64",
            "python": "3.12",
            "vulkanSDK": None,
        },
        "tools": [toolchains.Tool("dxc", (), None, "absent").to_json()],
        "runtimes": [
            runtimes.Provision(name, runtime.target, False, None, "absent").to_json()
            for name, runtime in runtimes.RUNTIMES.items()
        ],
    }
    text = toolchains.markdown(data)
    assert all(f"`{name}`" in text for name in runtimes.RUNTIMES)
    assert "unavailable: absent" in text


# ---------------------------------------------------------------------------------------------- runtimes
def test_runtime_registry_covers_two_vulkan_implementations_and_each_target():
    by_target = {}
    for runtime in runtimes.RUNTIMES.values():
        by_target.setdefault(runtime.target, []).append(runtime.id)
    assert set(by_target) == set(validators.TARGETS)
    assert len(by_target["spirv"]) >= 2


def test_unavailable_runtime_reports_every_job(tmp_path):
    absent = runtimes.Provision("vulkan-lavapipe", "spirv", False, None, "no driver")
    assert (
        runtimes.execute(absent, [tmp_path / "a.json", tmp_path / "b.json"], tmp_path)
        == [{"status": "unavailable", "message": "no driver"}] * 2
    )


def test_container_batch_is_offline_and_classifies_each_job(tmp_path, monkeypatch):
    jobs = [
        tmp_path / "ok" / "execute.json",
        tmp_path / "hang" / "execute.json",
        tmp_path / "lost" / "execute.json",
    ]
    for job in jobs:
        job.parent.mkdir()
    seen = []

    def batch(command, **_):
        seen.append(command)
        jobs[0].with_suffix(".result.json").write_text(
            json.dumps({"status": "executed"})
        )
        (tmp_path / "ok" / "execute.json.rc").write_text("0\n")
        (tmp_path / "hang" / "execute.json.rc").write_text("137\n")
        return CommandResult(tuple(command), "accepted", 0, "")

    monkeypatch.setattr(runtimes, "run", batch)
    container = runtimes.Provision(
        "vulkan-lavapipe",
        "spirv",
        True,
        "container",
        "",
        "llvmpipe",
        (),
        ("docker", "image:tag"),
    )
    results = runtimes.execute(container, jobs, tmp_path, timeout=5)
    assert [result["status"] for result in results] == [
        "executed",
        "timeout",
        "worker-error",
    ]
    command = seen[0]
    assert (
        command[:3] == ["docker", "run", "--rm"]
        and "--network" in command
        and "none" in command
    )
    assert f"{tmp_path}:{tmp_path}" in command and "image:tag" in command
    script = (tmp_path / "execute-vulkan-lavapipe.sh").read_text()
    assert script.count("timeout -s KILL 5") == 3


def test_adapters_refuse_results_from_unexpected_drivers():
    from . import native

    with pytest.raises(RuntimeError, match="Expected a llvmpipe device"):
        native._expect({"expect": "llvmpipe"}, "MoltenVK on Apple M2")
    assert (
        native._expect({"expect": "llvmpipe"}, "llvmpipe (LLVM 20)")
        == "llvmpipe (LLVM 20)"
    )


def test_explicit_runtimes_are_mandatory(monkeypatch):
    with pytest.raises(ValueError, match="Unknown runtimes"):
        runner.select_runtimes("vulkan-imaginary", ["spirv"])
    absent = runtimes.Provision("vulkan-lavapipe", "spirv", False, None, "absent")
    monkeypatch.setattr(runner, "provision", lambda runtime: absent)
    selection = runner.select_runtimes("vulkan-lavapipe", ["spirv", "glsl"])
    assert [runtime.id for runtime, _ in selection["spirv"]] == [
        "vulkan-lavapipe"
    ] and selection["glsl"] == []
    assert runner.select_runtimes("available", ["spirv"]) == {"spirv": []}


# ---------------------------------------------------------------------------------------------- execution
def test_evaluation_detects_divergence_and_memory_corruption(tmp_path):
    case = BY_ID["g_add_wrap"]
    cell = runner.Cell(case, "glsl", "strict", tmp_path, {})
    job = tmp_path / "execute.json"
    expected = case.expected().astype(case.out_dtype).tobytes()
    guard = np.full(runner.GUARD_WORDS, runner.CANARY, dtype=np.uint32).tobytes()

    def outcome(output, inputs=case.inputs.tobytes()):
        (tmp_path / "input.bin").write_bytes(inputs)
        (tmp_path / "output.bin").write_bytes(output)
        return runner.evaluate(cell, job, len(expected), {"status": "executed"})[
            "outcome"
        ]

    assert outcome(expected + guard) == "preserved"
    assert outcome(bytes(len(expected)) + guard) == "diverged"
    assert outcome(expected + bytes(len(guard))) == "memory-corruption"
    assert (
        outcome(expected + guard, inputs=bytes(len(case.inputs.tobytes())))
        == "memory-corruption"
    )
    assert (
        runner.evaluate(cell, job, len(expected), {"status": "timeout"})["outcome"]
        == "timeout"
    )


# ---------------------------------------------------------------------------------------------- scoreboard
def record(outcome="preserved", runtime="vulkan-lavapipe", case="sample"):
    return dict(
        case=case,
        language="metal",
        target="spirv",
        config="strict",
        route="translated",
        environment="vulkan1.1",
        runtime=runtime,
        outcome=outcome,
        sourceSHA256="source",
        inputSHA256="inputs",
    )


@pytest.mark.parametrize(
    "outcome",
    [
        "diverged",
        "invalid",
        "refused",
        "unavailable",
        "compiled-only",
        "timeout",
        "interface-changed",
        "driver-rejected",
        "harness-error",
    ],
)
def test_scoreboard_rejects_lost_execution(outcome):
    assert check(snapshot([record()]), [record(outcome)])


def test_scoreboard_accepts_a_valid_artifact_only_where_nothing_executes():
    compile_only = "compile-only@darwin"
    baseline = snapshot([record("invalid", runtime=compile_only)])
    fixed = [record("compiled-only", runtime=compile_only)]
    assert not check(baseline, fixed)
    assert improvements(baseline, fixed)
    assert check(
        snapshot([record("compiled-only", runtime=compile_only)]),
        [record("invalid", runtime=compile_only)],
    )
    assert check(snapshot([record("invalid")]), [record("compiled-only")])


def test_scoreboard_rejects_missing_duplicate_and_changed_cases():
    baseline = snapshot([record()])
    assert check(baseline, [])
    with pytest.raises(ValueError, match="Duplicate"):
        snapshot([record(), record()])
    changed = record()
    changed["sourceSHA256"] = "different"
    assert check(baseline, [changed])
    assert not check(snapshot([record("diverged")]), [record()])
    assert improvements(snapshot([record("diverged")]), [record()])
    assert not check(baseline, [record()])


def test_scoreboard_scope_is_explicit_per_runtime():
    baseline = snapshot([record(), record(runtime="vulkan-moltenvk")])
    assert not check(baseline, [record()], scope={"runtime": {"vulkan-lavapipe"}})
    assert check(
        baseline, [record()], scope={"runtime": {"vulkan-lavapipe", "vulkan-moltenvk"}}
    )
    assert check(baseline, [record()], scope={"runtime": {"d3d12-warp"}}) == [
        "The baseline has no cells in this run's scope"
    ]
    with pytest.raises(ValueError, match="scope fields"):
        check(baseline, [record()], scope={"device": {"x"}})


def test_scoreboard_results_scope_keeps_other_routes_and_configs():
    reference = record("diverged")
    reference["route"] = "reference"
    relaxed = record("diverged")
    relaxed["config"] = "relaxed"
    baseline = snapshot([record("diverged"), reference, relaxed])
    fresh = [record()]
    merged = merge(baseline, fresh, results_scope(fresh))
    assert set(merged["cells"]) == set(baseline["cells"])
    assert merged["counts"] == {"diverged": 2, "preserved": 1}


def test_scoreboard_merge_replaces_only_the_runs_scope():
    baseline = snapshot(
        [record("diverged"), record("diverged", runtime="vulkan-moltenvk")]
    )
    merged = merge(
        baseline, [record()], {"runtime": {"vulkan-lavapipe"}, "case": {"sample"}}
    )
    assert {cell["outcome"] for cell in merged["cells"].values()} == {
        "preserved",
        "diverged",
    }
    assert merged["counts"] == {"diverged": 1, "preserved": 1}
    assert scoreboard.SCHEMA == merged["schemaVersion"]


def test_native_16_bit_hlsl_is_compiled_with_16_bit_types(tmp_path):
    from .validators import hlsl_arguments

    native = tmp_path / "native.hlsl"
    native.write_text("float16_t2 f(uint16_t x) { return float16_t2(x, x); }\n")
    plain = tmp_path / "plain.hlsl"
    plain.write_text("min16float f(float x) { return x; } // float16_tx\n")
    assert hlsl_arguments(native) == ["-enable-16bit-types"]
    assert hlsl_arguments(plain) == []

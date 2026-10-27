"""Compute workgroups whose total fits a target but whose shape does not run linearized.

Direct3D 12 and OpenGL cap ``numthreads``/``local_size_z`` at 64 while Metal bounds only the total, so a Metal
kernel dispatched in (1, 2, 256) threadgroups has no direct HLSL or GLSL shape. Project translation declares
(512, 1, 1) instead, rebuilds the shape-dependent built-ins from the flattened local index, and records the declared
size as ``targetWorkgroupSize`` beside the source ``workgroupSize``.
"""

import itertools
import json
import re
import shutil
import struct
import subprocess
import sys

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    translate_project,
    validate_project_report,
)
from crosstl.project.directx_toolchain import dxc_compiler_arguments_for_source
from crosstl.project.workgroup_linearization import (
    _local_components,
    linearize_compute_workgroups,
    linearized_workgroup_size,
)
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

PLAIN = """
#include <metal_stdlib>
using namespace metal;

[[kernel]] void scale(
    device float* out [[buffer(0)]],
    const device float* in [[buffer(1)]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint lindex [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[1024];
  scratch[lindex] = in[gid.z + tptg.z * (gid.y + 4 * gid.x)];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint other = (lid.z + 1) % tptg.z + tptg.z * (lid.y + tptg.y * lid.x);
  out[gid.z + tptg.z * (gid.y + 4 * gid.x)] = scratch[other] + float(tg.x);
}
"""

TEMPLATE = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void rows(
    device T* out [[buffer(0)]],
    const device T* in [[buffer(1)]],
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]) {
  threadgroup T shared[1024];
  uint flat = lid.z + tptg.z * (lid.y + tptg.y * lid.x);
  shared[flat] = in[elem.z + grid.z * (elem.y + grid.y * elem.x)];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint mirrored = (tptg.z - 1 - lid.z) + tptg.z * (lid.y + tptg.y * lid.x);
  out[elem.z + grid.z * (elem.y + grid.y * elem.x)] = shared[mirrored];
}

template [[host_name("rows_float")]] [[kernel]] decltype(rows<float>) rows<float>;
"""

NARROW = """
#include <metal_stdlib>
using namespace metal;

[[kernel]] void narrow(
    device float* out [[buffer(0)]],
    ushort3 lid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]],
    uint gx [[thread_position_in_grid]],
    ushort3 tptg [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float s[512];
  s[lid.z] = float(gid.y + gx);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  out[gid.y * 2 + gid.x] = s[tptg.z - 1 - lid.z] + float(lane);
}
"""

# Every thread records the positions Metal gives it, so a device run can compare them with Metal's definitions.
IDS = """
#include <metal_stdlib>
using namespace metal;
[[kernel]] void ids(device uint* out [[buffer(0)]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 grid [[threads_per_grid]],
    uint li [[thread_index_in_threadgroup]]) {
  threadgroup uint seen[1024];
  seen[li] = lid.x + tptg.x * (lid.y + tptg.y * lid.z);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint total = tptg.x * tptg.y * tptg.z;
  uint flat = gid.x + grid.x * (gid.y + grid.y * gid.z);
  uint base = flat * 16u;
  out[base + 0] = lid.x; out[base + 1] = lid.y; out[base + 2] = lid.z;
  out[base + 3] = gid.x; out[base + 4] = gid.y; out[base + 5] = gid.z;
  out[base + 6] = tptg.x; out[base + 7] = tptg.y; out[base + 8] = tptg.z;
  out[base + 9] = grid.x; out[base + 10] = grid.y; out[base + 11] = grid.z;
  out[base + 12] = tg.x; out[base + 13] = tg.y; out[base + 14] = tg.z;
  out[base + 15] = seen[(li + 1u) % total];
}
"""

SOURCES = {"scale": PLAIN, "rows_float": TEMPLATE, "narrow": NARROW, "ids": IDS}
ARTIFACT_PATTERNS = {"directx": "*.hlsl", "opengl": "*.glsl"}
SHAPE_DEPENDENT = {
    "directx": ("SV_GroupThreadID", "SV_DispatchThreadID"),
    "opengl": ("gl_LocalInvocationID", "gl_GlobalInvocationID", "gl_WorkGroupSize"),
}


def _translate(tmp_path, entry, target, size, *, rule):
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    source = f"{entry}.metal"
    (root / source).write_text(SOURCES[entry], encoding="utf-8")
    sizing = (
        {"entry_workgroup_size_rules": {source: {entry: list(size)}}}
        if rule
        else {"workgroup_size": tuple(size)}
    )
    config = ProjectConfig(
        root=root,
        source_roots=(".",),
        include_patterns=(source,),
        targets=(target,),
        output_dir="out",
        entry_points={source: (entry,)},
        **sizing,
    )
    report = translate_project(config, format_output=False)
    report_path = root / "out" / "report.json"
    report.write_json(report_path)
    return report.to_json(), report_path


def _artifact(report_path, target):
    (artifact,) = sorted(report_path.parent.rglob(ARTIFACT_PATTERNS[target]))
    return artifact


def _errors(payload):
    return [d for d in payload["diagnostics"] if d["severity"] == "error"]


def _validate_with_toolchain(artifact, target):
    code = artifact.read_text(encoding="utf-8")
    if target == "directx" and shutil.which("dxc"):
        completed = subprocess.run(
            [
                "dxc",
                *dxc_compiler_arguments_for_source(code),
                "-T",
                "cs_6_2",
                "-E",
                "CSMain",
                str(artifact),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
    if target == "opengl" and shutil.which("glslangValidator"):
        completed = subprocess.run(
            ["glslangValidator", "-S", "comp", str(artifact)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stdout[-2000:]


def _declared_size(code, target):
    if target == "directx":
        match = re.search(r"\[numthreads\((\d+), (\d+), (\d+)\)\]", code)
    else:
        match = re.search(
            r"local_size_x = (\d+), local_size_y = (\d+), local_size_z = (\d+)", code
        )
    assert match, code
    return [int(value) for value in match.groups()]


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("size", [(1, 1, 128), (2, 1, 512), (1, 4, 256)])
def test_project_workgroup_size_linearizes_oversized_dimension(tmp_path, target, size):
    payload, report_path = _translate(tmp_path, "scale", target, size, rule=False)
    assert not _errors(payload), _errors(payload)
    total = size[0] * size[1] * size[2]
    execution = payload["artifacts"][0]["execution"]
    assert execution["workgroupSize"] == list(size)
    assert execution["targetWorkgroupSize"] == [total, 1, 1]

    artifact = _artifact(report_path, target)
    code = artifact.read_text(encoding="utf-8")
    assert _declared_size(code, target) == [total, 1, 1]
    entry = code[code.index("CSMain" if target == "directx" else "void main") :]
    for builtin in SHAPE_DEPENDENT[target]:
        assert builtin not in entry
    assert (
        f"uvec3({size[0]}u, {size[1]}u, {size[2]}u)".replace(
            "uvec3", "uint3" if target == "directx" else "uvec3"
        )
        in code
    )

    validation = validate_project_report(report_path)
    assert validation["success"], validation["diagnostics"]
    _validate_with_toolchain(artifact, target)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_entry_rule_linearizes_template_entry_through_native_package(tmp_path, target):
    payload, report_path = _translate(
        tmp_path, "rows_float", target, (1, 2, 256), rule=True
    )
    assert not _errors(payload), _errors(payload)
    execution = payload["artifacts"][0]["execution"]
    (entry,) = execution["entryPoints"]
    assert entry["workgroupSize"] == [1, 2, 256]
    assert entry["targetWorkgroupSize"] == [512, 1, 1]
    assert "targetWorkgroupSize" not in execution

    artifact = _artifact(report_path, target)
    code = artifact.read_text(encoding="utf-8")
    assert _declared_size(code, target) == [512, 1, 1]
    # threads_per_grid is rebuilt from the dispatch's workgroup count and the source shape.
    assert (
        "uvec3(1u, 2u, 256u)".replace(
            "uvec3", "uint3" if target == "directx" else "uvec3"
        )
        in code
    )

    validation = validate_project_report(report_path)
    assert validation["success"], validation["diagnostics"]
    _validate_with_toolchain(artifact, target)

    work = report_path.parent
    manifest = build_runtime_artifact_manifest(report_path)
    assert manifest["success"], manifest["diagnostics"]
    (work / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
    package = build_runtime_package(work / "artifacts.json", work / "package")
    assert package["success"], package["diagnostics"]
    loader = build_runtime_loader_manifest(work / "package" / "runtime-package.json")
    assert loader["success"], loader["diagnostics"]
    descriptor = build_native_loader_abi_descriptor(loader)
    config = descriptor["entryPoint"]["executionConfig"]
    key = "numthreads" if target == "directx" else "local_size"
    assert config[key] == [512, 1, 1]


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_narrow_and_partial_builtin_parameters_are_rebuilt(tmp_path, target):
    payload, report_path = _translate(
        tmp_path, "narrow", target, (1, 1, 512), rule=False
    )
    assert not _errors(payload), _errors(payload)
    assert payload["artifacts"][0]["execution"]["targetWorkgroupSize"] == [512, 1, 1]
    artifact = _artifact(report_path, target)
    code = artifact.read_text(encoding="utf-8")
    assert _declared_size(code, target) == [512, 1, 1]
    if target == "directx":
        assert "uint16_t3 lid = uint16_t3(" in code
        assert "uint2 gid = uint2(" in code
        assert "uint gx = uint(" in code
    else:
        assert "uvec2 gid = uvec2(" in code
        assert "uint gx = uint(" in code
    _validate_with_toolchain(artifact, target)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_shapes_the_target_declares_are_unchanged(tmp_path, target):
    payload, report_path = _translate(tmp_path, "scale", target, (1, 2, 64), rule=False)
    assert not _errors(payload), _errors(payload)
    assert "targetWorkgroupSize" not in payload["artifacts"][0]["execution"]
    code = _artifact(report_path, target).read_text(encoding="utf-8")
    assert _declared_size(code, target) == [1, 2, 64]
    assert "crosstl_linear" not in code


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_workgroups_over_the_invocation_limit_still_fail_closed(tmp_path, target):
    payload, _report_path = _translate(
        tmp_path, "scale", target, (1, 2, 1024), rule=False
    )
    codes = {d["code"] for d in _errors(payload)}
    assert "project.translate.workgroup-size-invalid" in codes


def test_report_validation_rejects_a_target_size_that_is_not_the_linearization(
    tmp_path,
):
    payload, report_path = _translate(
        tmp_path, "rows_float", "directx", (1, 2, 256), rule=True
    )
    assert not _errors(payload), _errors(payload)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifacts"][0]["execution"]["entryPoints"][0]["targetWorkgroupSize"] = [
        256,
        2,
        1,
    ]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    validation = validate_project_report(report_path)
    assert not validation["success"]
    messages = " ".join(d["message"] for d in validation["diagnostics"])
    assert "targetWorkgroupSize must be the linearized workgroupSize" in messages


def _evaluate_components(expression, index):
    python = re.sub(r"(\d+)u\b", r"\1", expression).replace("/", "//")
    return eval(python, {"uvec3": lambda *values: values, "i": index})  # noqa: S307


@pytest.mark.parametrize("size", [(1, 1, 128), (1, 2, 256), (2, 4, 65), (3, 5, 66)])
def test_local_id_decomposition_matches_row_major_flattening(size):
    x, y, z = size
    expression = _local_components("i", size)
    for index in range(x * y * z):
        assert _evaluate_components(expression, index) == (
            index % x,
            (index // x) % y,
            index // (x * y),
        )


def test_linearized_size_is_defined_only_for_shapes_the_target_cannot_declare():
    assert linearized_workgroup_size("directx", (1, 1, 128)) == (128, 1, 1)
    assert linearized_workgroup_size("opengl", (2, 2, 256)) == (1024, 1, 1)
    assert linearized_workgroup_size("directx", (1, 2, 64)) is None
    assert linearized_workgroup_size("directx", (1, 2, 1024)) is None
    assert linearized_workgroup_size("metal", (1, 1, 512)) is None


def _parse_crossgl(source):
    register_default_sources()
    return SOURCE_REGISTRY.get("cgl").parse(
        source, source_options={"strict_function_bodies": True}
    )


def test_programs_naming_shape_builtins_directly_are_left_to_the_generator():
    program = _parse_crossgl("""
shader direct {
    compute {
        void main(uint i @gl_LocalInvocationIndex) {
            uint z = gl_LocalInvocationID.z + i;
        }
    }
}
""")
    for stage in program.stages.values():
        stage.execution_config.update(
            {"local_size_x": "1", "local_size_y": "1", "local_size_z": "128"}
        )
    assert linearize_compute_workgroups(program, target="directx") == ()
    assert linearize_compute_workgroups(program, target="metal") == ()


def _expected_ids(size, groups):
    """Metal's thread positions for every thread of ``groups`` threadgroups of ``size``, keyed by grid offset."""
    (x_size, y_size, z_size), grid = size, [g * s for g, s in zip(groups, size)]
    total = x_size * y_size * z_size
    expected = {}
    for group in itertools.product(*(range(count) for count in groups)):
        for local in itertools.product(range(x_size), range(y_size), range(z_size)):
            position = [g * s + i for g, s, i in zip(group, size, local)]
            flat = position[0] + grid[0] * (position[1] + grid[1] * position[2])
            index = local[0] + x_size * (local[1] + y_size * local[2])
            expected[flat] = (
                *local,
                *position,
                *size,
                *grid,
                *group,
                (index + 1) % total,
            )
    return expected


@pytest.mark.parametrize("size", [(1, 1, 128), (2, 4, 64), (1, 2, 256)])
def test_linearized_thread_positions_execute_on_opengl(tmp_path, size):
    if sys.platform == "darwin":
        pytest.skip("macOS OpenGL has no compute shaders")
    moderngl = pytest.importorskip("moderngl")
    try:
        context = moderngl.create_standalone_context(require=430, backend="egl")
    except Exception as error:  # noqa: BLE001 - no EGL device on this runner
        pytest.skip(f"no OpenGL 4.3 EGL context: {error}")
    payload, report_path = _translate(tmp_path, "ids", "opengl", size, rule=False)
    assert not _errors(payload), _errors(payload)
    groups = (3, 2, 2)
    expected = _expected_ids(size, groups)
    output = None
    try:
        shader = context.compute_shader(_artifact(report_path, "opengl").read_text())
        output = context.buffer(b"\xff" * (len(expected) * 16 * 4))
        output.bind_to_storage_buffer(0)
        shader.run(*groups)
        context.finish()
        words = struct.unpack(f"<{len(expected) * 16}I", output.read())
    finally:
        if output is not None:
            output.release()
        context.release()
    observed = {
        flat: tuple(words[flat * 16 : flat * 16 + 16]) for flat in range(len(expected))
    }
    assert observed == expected

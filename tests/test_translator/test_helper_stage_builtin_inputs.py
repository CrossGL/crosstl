"""Compute and vertex built-in inputs a helper reads reach it as parameters threaded from the entry point.

GLSL reads ``gl_GlobalInvocationID``, ``gl_VertexID`` and the other stage inputs as globals anywhere in the stage.
HLSL, MSL, Slang and WGSL deliver them only as entry-point parameters, so a helper that read one by name was emitted
with an undeclared identifier.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

COMPUTE = """
shader Probe {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        uint lane() {
            return gl_GlobalInvocationID.x + gl_LocalInvocationIndex;
        }
        uint slot() {
            return lane() + gl_WorkGroupID.x * gl_NumWorkGroups.y;
        }
        void main() {
            outp[slot()] = float(gl_GlobalInvocationID.y + gl_LocalInvocationID.x);
        }
    }
}
"""

ENTRY_PARAMETER = """
shader Probe {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        uint invocation() {
            return gl_GlobalInvocationID.x;
        }
        void main(uvec3 gid @ gl_GlobalInvocationID) {
            outp[invocation()] = float(gid.y);
        }
    }
}
"""

VERTEX = """
shader Probe {
    vertex {
        float index() {
            return float(gl_VertexID + gl_InstanceID * 3);
        }
        vec4 main() @ gl_Position {
            return vec4(index(), 0.0, 0.0, 1.0);
        }
    }
}
"""

FRAGMENT = """
shader Probe {
    fragment {
        float shade() {
            return gl_FrontFacing ? gl_FragCoord.x : 0.0;
        }
        vec4 main() @ gl_FragColor {
            return vec4(shade());
        }
    }
}
"""

ENTRY_NAMES = {"directx": "CSMain", "slang": "main"}


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, cwd):
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    assert result.returncode == 0, result.stdout + result.stderr


def _helper(code, name):
    """The signature and body of the one-statement helper ``name``."""
    match = re.search(rf"\b{name}\(([^)]*)\)[^{{;]*\{{([^}}]*)\}}", code)
    assert match, code
    return match.group(1), match.group(2)


def _validate(tmp_path, backend, code, profile):
    if backend == "directx" and shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(code)
        entry = {"cs": "CSMain", "vs": "VSMain", "ps": "PSMain"}[profile]
        _run(["dxc", "-T", f"{profile}_6_0", "-E", entry, "out.hlsl"], tmp_path)
    elif backend == "wgsl" and shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(code)
        _run(["naga", "out.wgsl"], tmp_path)
    elif backend == "metal" and shutil.which("xcrun"):
        found = subprocess.run(
            ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
        )
        if found.returncode == 0:
            (tmp_path / "out.metal").write_text(code)
            _run(
                [
                    "xcrun",
                    "-sdk",
                    "macosx",
                    "metal",
                    "-c",
                    "out.metal",
                    "-o",
                    "out.air",
                ],
                tmp_path,
            )


@pytest.mark.parametrize("backend", ["directx", "metal", "slang", "wgsl"])
def test_compute_helpers_receive_the_invocation_ids_they_read(tmp_path, backend):
    code = _translate(tmp_path, COMPUTE, backend)

    lane, lane_body = _helper(code, "lane")
    slot, slot_body = _helper(code, "slot")
    assert not re.search(r"\bgl_\w+", lane + lane_body + slot + slot_body), code
    # lane reads two ids; slot reads two and passes on lane's two.
    assert lane.count(",") == 1, code
    assert slot.count(",") == 3, code
    for name in ("GlobalInvocationID", "LocalInvocationIndex"):
        assert f"cgl_{name}" in lane, code
    # The entry point passes its own parameters on.
    assert re.search(r"\bslot\(cgl_\w+, cgl_\w+, cgl_\w+, cgl_\w+\)", code), code
    _validate(tmp_path, backend, code, "cs")


@pytest.mark.parametrize(
    ("backend", "binding"),
    [
        ("directx", r"uint3 gid : SV_DispatchThreadID"),
        ("metal", r"uint3 gid\s+\[\[thread_position_in_grid\]\]"),
        ("slang", r"uint3 gid : SV_DispatchThreadID"),
        ("wgsl", r"@builtin\(global_invocation_id\) gid: vec3<u32>"),
    ],
)
def test_a_helper_reuses_the_entry_parameter_carrying_the_builtin(
    tmp_path, backend, binding
):
    code = _translate(tmp_path, ENTRY_PARAMETER, backend)

    assert not re.search(r"\bgl_\w+", code), code
    assert re.search(binding, code), code
    assert "invocation(gid)" in code, code
    assert "cgl_GlobalInvocationID" not in code.split("invocation(gid)")[1], code
    _validate(tmp_path, backend, code, "cs")


@pytest.mark.parametrize(
    ("backend", "vertex_id"),
    [
        ("directx", r"uint (\w+) : SV_VertexID"),
        ("metal", r"uint (\w+) \[\[vertex_id\]\]"),
        ("slang", r"uint (\w+) : SV_VertexID"),
        ("wgsl", r"@builtin\(vertex_index\) (\w+): u32"),
    ],
)
def test_vertex_helpers_receive_the_vertex_and_instance_index(
    tmp_path, backend, vertex_id
):
    code = _translate(tmp_path, VERTEX, backend)

    assert not re.search(r"\bgl_\w+", code), code
    raw = re.search(vertex_id, code).group(1)
    # The uint system value is converted to GLSL's int once, in the entry point.
    assert re.search(rf"\b(int|i32)\(\s*{raw}\s*\)", code), code
    assert re.search(r"\bindex\(\s*(int|i32)?\s*:?\s*\w+", code), code
    _validate(tmp_path, backend, code, "vs")


@pytest.mark.parametrize(
    ("backend", "front_facing"),
    [
        ("slang", r"bool \w+ : SV_IsFrontFace"),
        ("wgsl", r"@builtin\(front_facing\) \w+: bool"),
    ],
)
def test_fragment_helpers_receive_builtins_on_slang_and_wgsl(
    tmp_path, backend, front_facing
):
    code = _translate(tmp_path, FRAGMENT, backend)

    assert not re.search(r"\bgl_\w+", code), code
    assert re.search(front_facing, code), code
    assert re.search(r"\bshade\(\w+", code), code
    _validate(tmp_path, backend, code, "ps")


@pytest.mark.parametrize("backend", ["opengl", "vulkan"])
@pytest.mark.parametrize("source", [COMPUTE, VERTEX])
def test_targets_with_builtin_globals_keep_reading_them(tmp_path, backend, source):
    code = _translate(tmp_path, source, backend)

    assert "cgl_" not in code

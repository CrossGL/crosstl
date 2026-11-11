"""SPIR-V output passes ``out`` and ``inout`` arguments by value-result, and Metal references by reference.

GLSL 4.60 section 6.1.1: the arguments of a call are evaluated in order, left to right, exactly once; an ``inout``
argument is copied in at its turn, the l-value of an ``out`` or ``inout`` argument is fixed at its turn, and the
parameter's value is copied back when the function returns. HLSL ``out`` and ``inout`` copy in the same way. SPIR-V
output used to bind the parameter of every inlined function straight to the caller's variable, so
``scale(m, bump(m))`` read ``m`` after ``bump`` had changed it (174 instead of 64), and an ``inout`` parameter the
callee never reads did not undo a later argument's write.

A Metal ``thread T&`` parameter reaches CrossGL as ``inout thread T``. It is a reference (C++ [dcl.ref]): both
parameters of ``twice(m, m)`` name ``m``, so it stays bound to the caller's variable.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from tests.spirv_friendly_names import friendly_spirv

REQUIRE_ENV = "CROSSTL_REQUIRE_VALUE_RESULT_EXECUTION"

SOURCE = """
shader Probe {
    int bump(inout int t) { t = t * 3 + 1; return t - 2; }
    void scale(inout int p, int q) { p = p * 10 + q; }
    void keep(inout int p, int q) { }
    void put(out int dst, int v) { dst = v; }
    void swap(inout int a, inout int b) { int t = a; a = b; b = t; }
    void twice(inout thread int a, inout thread int b) { a = a + 1; b = b + 1; }
    void both(inout int a, inout int b) { a = a + 1; b = b + 1; }

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* outp;
        void main() {
            int x = int(gl_GlobalInvocationID.x) + 5;
            int m = x;
            scale(m, bump(m));
            outp[0] = m;
            m = x;
            keep(m, bump(m));
            outp[1] = m;
            int k = x & 1;
            int slots[2];
            slots[0] = 0;
            slots[1] = 0;
            put(slots[k], bump(k));
            outp[2] = slots[1] * 16 + slots[0] + k * 1000;
            outp[3] = x;
            outp[4] = 7;
            swap(outp[3], outp[4]);
            m = x;
            twice(m, m);
            outp[5] = m;
            m = x;
            both(m, m);
            outp[6] = m;
        }
    }
}
"""

# x = 5: bump(5) leaves 16 and returns 14; bump(1) leaves 4 and returns 2.
EXPECTED = [
    5 * 10 + 14,  # p copied in before bump(m) runs, then copied back over bump's write
    5,  # keep copies its unread inout parameter back
    2 * 16 + 0 + 4 * 1000,  # slots[k] is fixed while k is 1, before bump(k) makes it 4
    7,  # storage-buffer elements copied in and back
    5,
    5 + 2,  # Metal references: a and b both name m
    5 + 1,  # value-result: each parameter has its own copy, both copied back as 6
]


def _translate(tmp_path):
    path = tmp_path / "probe.cgl"
    path.write_text(SOURCE)
    return friendly_spirv(
        crosstl.translate(str(path), backend="vulkan", format_output=False)
    )


def _assemble(tmp_path, spirv):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not (assembler and validator):
        pytest.skip("SPIRV-Tools is not installed")
    source = tmp_path / "probe.spvasm"
    binary = tmp_path / "probe.spv"
    source.write_text(spirv)
    for command in (
        [assembler, "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        [validator, "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr or result.stdout
    return binary


def _vulkan_runtime():
    """A host Vulkan runtime and ``vkrun``, or ``None``: lavapipe, else MoltenVK or KosmicKrisp."""
    try:
        from tests.conformance.runtimes import provision
    except ImportError:
        return None
    for runtime in ("vulkan-lavapipe", "vulkan-moltenvk", "vulkan-kosmickrisp"):
        try:
            provisioned = provision(runtime)
        except Exception:
            continue
        if provisioned.available and provisioned.mode == "host":
            env = dict(provisioned.env)
            if env.get("CROSSTL_CONFORMANCE_VKRUN"):
                return env
    return None


def test_value_result_arguments_have_their_own_variables(tmp_path):
    spirv = _translate(tmp_path)

    # Every function with an out or inout parameter is inlined, so the copies are in main.
    assert "OpFunctionCall" not in spirv
    main = spirv[spirv.index("OpFunction %void") :]
    temporaries = re.findall(r"(%\d+) = OpVariable %_ptr_Function_int Function", main)
    # bump x3, scale, keep, put, swap x2, both x2; twice binds m itself, twice.
    assert len(temporaries) == 10
    _assemble(tmp_path, spirv)


def test_value_result_arguments_execute_in_order(tmp_path):
    env = _vulkan_runtime()
    if env is None:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(
                f"{REQUIRE_ENV}=1 but no host Vulkan runtime with vkrun is available"
            )
        pytest.skip(f"no host Vulkan runtime; set {REQUIRE_ENV}=1 to require one")
    spirv = _translate(tmp_path)
    binary = _assemble(tmp_path, spirv)
    entry = re.search(r'OpEntryPoint GLCompute %\w+ "(\w+)"', spirv).group(1)
    (binding,) = re.findall(r"OpDecorate %outp\w* Binding (\d+)", spirv)
    result_path = tmp_path / "result.bin"
    command = [env["CROSSTL_CONFORMANCE_VKRUN"], str(binary), entry, "1", "1", "1"]
    command.append(f"{binding}:out:{result_path}:{4 * len(EXPECTED)}")

    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, env={**os.environ, **env}
    )

    assert result.returncode == 0 and result.stdout.startswith("ok"), (
        result.stdout + result.stderr
    )
    values = struct.unpack(f"<{len(EXPECTED)}i", result_path.read_bytes())
    assert list(values) == EXPECTED

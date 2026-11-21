"""SPIR-V output of ``++`` and ``--`` on parameters.

SPIR-V output holds a by-value parameter as the ``OpFunctionParameter`` value and gives it a Function variable when
an assignment first writes it. An increment looked the target up without that step, so ``s.c++`` on a member of an
``inout`` structure parameter was refused ("increment target is not assignable") while ``s.c += 1`` translated, and
``x++`` on a by-value scalar parameter wrote ``OpLoad %int %x`` on the value, which spirv-val rejects. CUDA struct
member functions such as ``int take() { return a++; }`` import as functions with an ``inout`` receiver and reach the
first form.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess

import pytest

import crosstl

PREAMBLE = """
struct S {
    i32 a;
    i32 c;
};
"""

ENTRY = """
@compute
@workgroup_size(64, 1, 1)
fn k(
    @group(0) @binding(0) var<storage, read_write> o: array<i32>
) {{
    var t: i32 = gl_LocalInvocationID.x;
{body}
}}
"""

MEMBER_INCREMENTS = (
    PREAMBLE
    + """
i32 take(inout S s) {
    return s.a++;
}

i32 drop(inout S s) {
    return --s.c;
}
"""
    + ENTRY.format(body="""
    var s: S;
    s.a = t;
    s.c = 3 * t;
    i32 first = take(s);
    i32 second = take(s);
    i32 third = drop(s);
    o[4 * t] = first;
    o[4 * t + 1] = second + 100 * s.a;
    o[4 * t + 2] = third + 1000 * s.c;
    o[4 * t + 3] = 0;
""")
)

STATEMENT_INCREMENTS = (
    PREAMBLE
    + """
void grow(inout S s) {
    s.c++;
    ++s.a;
}
"""
    + ENTRY.format(body="""
    var s: S;
    s.a = t;
    s.c = 2 * t;
    grow(s);
    o[4 * t] = s.a;
    o[4 * t + 1] = s.c;
    o[4 * t + 2] = 0;
    o[4 * t + 3] = 0;
""")
)

VALUE_PARAMETER_INCREMENTS = """
i32 bump(i32 x) {
    x++;
    i32 y = x--;
    return ++x + 10 * y;
}
""" + ENTRY.format(
    body="""
    o[4 * t] = bump(t);
    o[4 * t + 1] = t;
    o[4 * t + 2] = 0;
    o[4 * t + 3] = 0;
"""
)

CASES = {
    "member": (
        MEMBER_INCREMENTS,
        [
            v
            for t in range(64)
            for v in (t, t + 1 + 100 * (t + 2), 3 * t - 1 + 1000 * (3 * t - 1), 0)
        ],
    ),
    "statement": (
        STATEMENT_INCREMENTS,
        [v for t in range(64) for v in (t + 1, 2 * t + 1, 0, 0)],
    ),
    "value": (
        VALUE_PARAMETER_INCREMENTS,
        [v for t in range(64) for v in (t + 1 + 10 * (t + 1), t, 0, 0)],
    ),
}


def _translate(tmp_path, source):
    path = tmp_path / "probe.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="vulkan", format_output=False)


def _assemble(tmp_path, assembly):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not (assembler and validator):
        pytest.skip("SPIRV-Tools is not installed")
    source = tmp_path / "probe.spvasm"
    binary = tmp_path / "probe.spv"
    source.write_text(assembly, encoding="utf-8")
    for command in (
        [assembler, "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        [validator, "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr or result.stdout
    return binary


def _value_parameters(assembly):
    """Names of the ``OpFunctionParameter`` results whose type is not a pointer."""
    return set(re.findall(r"%(\w+) = OpFunctionParameter %(?!_ptr_)\w+", assembly))


@pytest.mark.parametrize("case", sorted(CASES))
def test_increment_of_a_parameter_validates(tmp_path, case):
    assembly = _translate(tmp_path, CASES[case][0])

    _assemble(tmp_path, assembly)
    for name in _value_parameters(assembly):
        assert not re.search(rf"Op(Load|AccessChain) %\w+ %{name}\b", assembly)
        assert not re.search(rf"OpStore %{name}\b", assembly)


def _vulkan_runtime():
    try:
        from tests.conformance.runtimes import provision
    except ImportError:
        return None
    for runtime in ("vulkan-lavapipe", "vulkan-moltenvk"):
        try:
            provisioned = provision(runtime)
        except Exception:
            continue
        if provisioned.available and provisioned.mode == "host":
            env = dict(provisioned.env)
            if env.get("CROSSTL_CONFORMANCE_VKRUN"):
                return env
    return None


@pytest.mark.parametrize("case", sorted(CASES))
def test_increment_of_a_parameter_computes_the_crossgl_result(tmp_path, case):
    env = _vulkan_runtime()
    if env is None:
        pytest.skip("no host Vulkan runtime with vkrun")
    source, expected = CASES[case]
    assembly = _translate(tmp_path, source)
    binary = _assemble(tmp_path, assembly)
    entry = re.search(r'OpEntryPoint GLCompute %\w+ "(\w+)"', assembly).group(1)
    result = tmp_path / "o.bin"
    result.write_bytes(struct.pack("<256i", *([-7] * 256)))

    run = subprocess.run(
        [
            env["CROSSTL_CONFORMANCE_VKRUN"],
            str(binary),
            entry,
            "1",
            "1",
            "1",
            f"0:inout:{result}:1024",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **env},
    )

    assert run.returncode == 0 and run.stdout.startswith("ok"), run.stdout + run.stderr
    assert list(struct.unpack("<256i", result.read_bytes())) == expected

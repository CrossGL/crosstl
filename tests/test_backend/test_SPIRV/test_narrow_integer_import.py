"""SPIR-V input keeps 8- and 16-bit integer arithmetic at its declared width.

SPIR-V computes ``OpIMul %char`` modulo 2^8, while CrossGL promotes a scalar ``i8`` to ``int`` before arithmetic, so
the importer wraps a narrow scalar ``OpIAdd``, ``OpISub``, ``OpIMul``, ``OpShiftLeftLogical`` and ``OpSNegate``, and
an unsigned ``OpNot``, in a constructor of the result type (``narrow_integer_import.py``). ``OpBitcast`` between
integer types of one width, which glslang emits for ``uint8_t(int8_t(x))``, imports as the same-width constructor.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess

import pytest

import crosstl

REQUIRE_ENV = "CROSSTL_REQUIRE_VULKAN_RUNTIME"
INVOCATIONS = 8
INPUTS = [13, -100, 127, -128, 200, 1000, 40000, -7]

TYPES = {
    "char": ("8 1", "Int8"),
    "uchar": ("8 0", "Int8"),
    "short": ("16 1", "Int16"),
    "ushort": ("16 0", "Int16"),
    "long": ("64 1", "Int64"),
    "ulong": ("64 0", "Int64"),
}


def module(narrow, body):
    """A compute module that reads ``int``, converts it to ``narrow`` as ``%v`` and writes ``body``'s ``%r``."""
    width_sign, capability = TYPES[narrow]
    signed = width_sign.endswith("1")
    if signed:
        convert = f"%v = OpSConvert %{narrow} %x"
    else:
        signed_name = {"uchar": "char", "ushort": "short", "ulong": "long"}[narrow]
        convert = f"%vs = OpSConvert %{signed_name} %x\n%v = OpBitcast %{narrow} %vs"
    extra_type = ""
    if not signed:
        extra_type = f"%{signed_name} = OpTypeInt {width_sign.split()[0]} 1"
    to_float = "OpConvertSToF" if signed else "OpConvertUToF"
    return f"""OpCapability Shader
OpCapability {capability}
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gid
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %ints ArrayStride 4
OpDecorate %In Block
OpMemberDecorate %In 0 Offset 0
OpDecorate %inp Binding 0
OpDecorate %inp DescriptorSet 0
OpDecorate %floats ArrayStride 4
OpDecorate %Out Block
OpMemberDecorate %Out 0 Offset 0
OpDecorate %outp Binding 1
OpDecorate %outp DescriptorSet 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%{narrow} = OpTypeInt {width_sign}
{extra_type}
%v3uint = OpTypeVector %uint 3
%ptr_in_v3uint = OpTypePointer Input %v3uint
%ptr_in_uint = OpTypePointer Input %uint
%gid = OpVariable %ptr_in_v3uint Input
%uint_0 = OpConstant %uint 0
%int_0 = OpConstant %int 0
%uint_3 = OpConstant %uint 3
%ints = OpTypeRuntimeArray %int
%In = OpTypeStruct %ints
%ptr_sb_In = OpTypePointer StorageBuffer %In
%inp = OpVariable %ptr_sb_In StorageBuffer
%ptr_sb_int = OpTypePointer StorageBuffer %int
%floats = OpTypeRuntimeArray %float
%Out = OpTypeStruct %floats
%ptr_sb_Out = OpTypePointer StorageBuffer %Out
%outp = OpVariable %ptr_sb_Out StorageBuffer
%ptr_sb_float = OpTypePointer StorageBuffer %float
%main = OpFunction %void None %fn
%entry = OpLabel
%gx = OpAccessChain %ptr_in_uint %gid %uint_0
%i = OpLoad %uint %gx
%xp = OpAccessChain %ptr_sb_int %inp %int_0 %i
%x = OpLoad %int %xp
{convert}
{body}
%f = {to_float} %float %r
%op = OpAccessChain %ptr_sb_float %outp %int_0 %i
OpStore %op %f
OpReturn
OpFunctionEnd
"""


def import_crossgl(tmp_path, code):
    path = tmp_path / "narrow.spvasm"
    path.write_text(code, encoding="utf-8")
    return crosstl.translate(str(path), backend="cgl")


def wrap(value, bits, signed):
    value &= (1 << bits) - 1
    if signed and value >= 1 << (bits - 1):
        value -= 1 << bits
    return value


CONSTRUCTORS = {"char": "i8", "uchar": "u8", "short": "i16", "ushort": "u16"}


def stored_value(code):
    """The expression the module stores to its output, as the imported CrossGL writes it."""
    match = re.search(r"outp\[0\]\.\w+\[[^=]*\] = float\((.*)\);", code)
    assert match, code
    return match.group(1)


@pytest.mark.parametrize(
    "narrow, body, operator",
    [
        ("char", "%r = OpIMul %char %v %v", "*"),
        ("uchar", "%r = OpIAdd %uchar %v %v", "+"),
        ("short", "%r = OpISub %short %v %v", "-"),
        ("ushort", "%r = OpShiftLeftLogical %ushort %v %uint_3", "<<"),
        ("char", "%r = OpSNegate %char %v", "-"),
        ("uchar", "%r = OpNot %uchar %v", "~"),
    ],
)
def test_narrow_scalar_result_is_converted_to_its_type(
    tmp_path, narrow, body, operator
):
    value = stored_value(import_crossgl(tmp_path, module(narrow, body)))
    assert value.startswith(f"{CONSTRUCTORS[narrow]}("), value
    assert operator in value, value


@pytest.mark.parametrize(
    "narrow, body",
    [
        ("char", "%r = OpSDiv %char %v %v"),
        ("uchar", "%r = OpBitwiseAnd %uchar %v %v"),
        ("short", "%r = OpShiftRightArithmetic %short %v %uint_3"),
        ("char", "%r = OpNot %char %v"),
        ("long", "%r = OpIMul %long %v %v"),
    ],
)
def test_operations_that_stay_in_range_are_not_converted(tmp_path, narrow, body):
    value = stored_value(import_crossgl(tmp_path, module(narrow, body)))
    assert value.startswith("("), value


@pytest.mark.parametrize(
    "narrow, constructor",
    [("uchar", "u8(i8("), ("ushort", "u16(i16("), ("ulong", "u64(i64(")],
)
def test_same_width_integer_bitcast_is_a_constructor(tmp_path, narrow, constructor):
    code = import_crossgl(tmp_path, module(narrow, f"%r = OpIMul %{narrow} %v %v"))
    assert constructor in code, code
    assert "spirvBitcast" not in code, code


def vulkan_runtime():
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


def run_tool(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize(
    "narrow, body, compute",
    [
        ("char", "%r = OpIMul %char %v %v", lambda v: wrap(v * v, 8, True)),
        ("uchar", "%r = OpIMul %uchar %v %v", lambda v: wrap(v * v, 8, False)),
        ("short", "%r = OpIMul %short %v %v", lambda v: wrap(v * v, 16, True)),
        ("ushort", "%r = OpIAdd %ushort %v %v", lambda v: wrap(v + v, 16, False)),
        ("uchar", "%r = OpNot %uchar %v", lambda v: wrap(~v, 8, False)),
    ],
)
def test_round_trip_computes_the_spirv_result(tmp_path, narrow, body, compute):
    env = vulkan_runtime()
    if env is None:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(f"{REQUIRE_ENV}=1 but no host Vulkan runtime is available")
        pytest.skip(f"no host Vulkan runtime; set {REQUIRE_ENV}=1 to require one")
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools is not installed")
    crossgl = tmp_path / "narrow.cgl"
    crossgl.write_text(import_crossgl(tmp_path, module(narrow, body)), encoding="utf-8")
    source, binary = tmp_path / "round.spvasm", tmp_path / "round.spv"
    source.write_text(
        crosstl.translate(str(crossgl), backend="vulkan"), encoding="utf-8"
    )
    run_tool(["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)])
    run_tool(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
    entry = re.search(
        r'OpEntryPoint GLCompute %\w+ "(\w+)"',
        run_tool(["spirv-dis", str(binary)]).stdout,
    ).group(1)
    inputs, outputs = tmp_path / "in.bin", tmp_path / "out.bin"
    inputs.write_bytes(struct.pack(f"<{INVOCATIONS}i", *INPUTS))
    outputs.write_bytes(bytes(4 * INVOCATIONS))
    command = [
        env["CROSSTL_CONFORMANCE_VKRUN"],
        str(binary),
        entry,
        str(INVOCATIONS),
        "1",
        "1",
        f"0:inout:{inputs}",
        f"1:inout:{outputs}",
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, env={**os.environ, **env}
    )
    assert result.returncode == 0, result.stdout + result.stderr
    bits = int(TYPES[narrow][0].split()[0])
    signed = TYPES[narrow][0].endswith("1")
    expected = [float(compute(wrap(x, bits, signed))) for x in INPUTS]
    assert list(struct.unpack(f"<{INVOCATIONS}f", outputs.read_bytes())) == expected

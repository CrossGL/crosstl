"""SPIR-V lowering of MSL math builtins, bfloat16 operands, generic struct templates and buffer array parameters.

Before the fail-closed refusals these constructs were dropped from the module (a call became a zero constant, a
generic member became ``float``, a buffer argument a warning comment). Each test checks the instruction that now
carries the operation and validates the module with ``spirv-val`` when SPIRV-Tools is installed.
"""

import re
import shutil
import subprocess

import pytest

import crosstl


def _translate(tmp_path, source):
    path = tmp_path / "kernel.metal"
    path.write_text("#include <metal_stdlib>\nusing namespace metal;\n" + source, encoding="utf-8")
    return crosstl.translate(str(path), backend="vulkan", format_output=False)


def _validate(spv_code, tmp_path):
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        pytest.skip("spirv-as and spirv-val are not installed")
    asm, binary = tmp_path / "k.spvasm", tmp_path / "k.spv"
    asm.write_text(spv_code, encoding="utf-8")
    env = ["--target-env", "vulkan1.1"]
    for command in ([spirv_as, *env, str(asm), "-o", str(binary)], [spirv_val, *env, str(binary)]):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr


def _ext(spv_code, name):
    return re.findall(rf"OpExtInst %\w+ %\w+ {name}\b", spv_code)


def test_fabs_lowers_to_fabs_on_scalars_and_vectors(tmp_path):
    spv = _translate(
        tmp_path,
        "kernel void k(device const float4* a [[buffer(0)]], device float* o [[buffer(1)]],\n"
        "              uint i [[thread_position_in_grid]]) {\n"
        "  float4 v = metal::fabs(a[i]);\n"
        "  o[i] = metal::fmax(metal::fabs(v.x), fabs(v.y - v.z));\n"
        "}\n",
    )
    assert len(_ext(spv, "FAbs")) == 3
    fmax = re.search(r"OpExtInst %\w+ %\w+ FMax %(\w+) %(\w+)", spv)
    assert fmax is not None
    for operand in fmax.groups():
        assert re.search(rf"%{operand} = OpExtInst %\w+ %\w+ FAbs\b", spv)
    _validate(spv, tmp_path)


def test_fabs_refuses_integer_operands(tmp_path):
    with pytest.raises(Exception, match="fabs requires a floating-point"):
        _translate(
            tmp_path,
            "kernel void k(device const int* a [[buffer(0)]], device int* o [[buffer(1)]],\n"
            "              uint i [[thread_position_in_grid]]) {\n"
            "  int v = a[i];\n"
            "  o[i] = int(fabs(v));\n"
            "}\n",
        )


def test_select_lowers_to_opselect_with_false_value_first(tmp_path):
    spv = _translate(
        tmp_path,
        "kernel void k(device const float* a [[buffer(0)]], device float* o [[buffer(1)]],\n"
        "              uint i [[thread_position_in_grid]]) {\n"
        "  float x = a[i];\n"
        "  float y = x * 2.0f;\n"
        "  o[i] = metal::select(x, y, x > 1.0f);\n"
        "}\n",
    )
    selects = re.findall(r"= OpSelect %\w+ %(\w+) %(\w+) %(\w+)", spv)
    assert len(selects) == 1
    _condition, true_id, false_id = selects[0]
    # select(a, b, c) is c ? b : a, so the true operand loads the variable holding x * 2.
    product = re.search(r"%(\w+) = OpFMul", spv).group(1)
    y_variable = re.search(rf"OpStore %(\w+) %{product}\b", spv).group(1)
    assert re.search(rf"%{true_id} = OpLoad %\w+ %{y_variable}\b", spv)
    assert not re.search(rf"%{false_id} = OpLoad %\w+ %{y_variable}\b", spv)
    _validate(spv, tmp_path)


def test_generic_struct_with_parameter_typed_members_is_not_emitted(tmp_path):
    spv = _translate(
        tmp_path,
        "template <typename T> struct pair_t { T first; T second; };\n"
        "struct pair_f { float first; float second; };\n"
        "kernel void k(device const float* a [[buffer(0)]], device float* o [[buffer(1)]],\n"
        "              uint i [[thread_position_in_grid]]) {\n"
        "  pair_f p = {a[i], a[i] + 1.0f};\n"
        "  o[i] = p.first * p.second;\n"
        "}\n",
    )
    assert not re.search(r'OpName %\w+ "pair_t"', spv)
    assert re.search(r'OpName %\w+ "pair_f"', spv)
    _validate(spv, tmp_path)


def test_bfloat16_math_builtins_compute_in_binary32_and_round_back(tmp_path):
    spv = _translate(
        tmp_path,
        "kernel void k(device const bfloat* a [[buffer(0)]], device bfloat* o [[buffer(1)]],\n"
        "              device bool* n [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
        "  bfloat x = a[i];\n"
        "  o[i] = metal::max(metal::exp(x), x);\n"
        "  n[i] = metal::isnan(x);\n"
        "}\n",
    )
    exp = re.search(r"OpExtInst (%\w+) %\w+ Exp\b", spv)
    assert exp is not None
    assert re.search(rf"{exp.group(1)} = OpTypeFloat 32", spv)
    assert _ext(spv, "FMax")
    # Rounding back to bfloat16 storage is a round-to-nearest-even on the binary32 bits.
    assert spv.count("OpShiftRightLogical") >= 2
    _validate(spv, tmp_path)


def test_bfloat16_math_refuses_mixed_floating_operands(tmp_path):
    from crosstl.translator.codegen.spirv_bfloat16_math import SpirvBfloat16MathMixin

    class Probe(SpirvBfloat16MathMixin):
        BFLOAT16_TYPE_NAME = "bfloat16"

        def _bfloat16_math_component(self, value):
            return value

    assert Probe().call_bfloat16_math_function("max", ["bfloat16", "float"]) is None
    assert Probe().call_bfloat16_math_function("length", ["bfloat16"]) is None
    assert Probe().call_bfloat16_math_function("exp", ["float"]) is None


def test_constant_array_parameter_binds_the_callers_buffer(tmp_path):
    spv = _translate(
        tmp_path,
        "void impl(device float* out, constant const int strides[3], uint pos) {\n"
        "  out[pos] = float(strides[0] + strides[2]);\n"
        "}\n"
        "kernel void k(device float* out [[buffer(0)]], constant const int* strides [[buffer(1)]],\n"
        "              uint pos [[thread_position_in_grid]]) {\n"
        "  impl(out, strides, pos);\n"
        "}\n",
    )
    assert "runtime-array aggregate" not in spv
    chains = re.findall(r"= OpAccessChain %\w+ %\w+ %\w+ %(\w+)", spv)
    assert len(chains) >= 2
    _validate(spv, tmp_path)

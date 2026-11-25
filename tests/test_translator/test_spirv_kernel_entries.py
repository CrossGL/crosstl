"""SPIR-V modules of CUDA and OpenCL C files with several kernels.

Each kernel binds its own parameters from binding 0, and its plain values become a ``<kernel>_Args`` block, so two
kernels of one file give different buffers, or blocks with the same member names, one descriptor slot. A Vulkan
compute pipeline holds one entry point and only lays out the resources that entry point uses, so the module keeps
both kernels: their resources may share a slot, a block member resolves in its own kernel, and a parameter whose
name the other kernel declares differently gets a name of its own.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

import crosstl
from crosstl.translator.ast import VariableNode
from crosstl.translator.codegen.spirv_entry_resources import (
    SPIRV_ENTRY_SCOPE_ANNOTATION,
)
from crosstl.translator.codegen.spirv_hlsl_registers import reserve_exact_binding
from crosstl.translator.refusals import VulkanRefusal

SAME_PARAMETERS = """
__global__ void a(float* o, const float* i, int w) { o[threadIdx.x] = i[threadIdx.x] * w; }
__global__ void b(float* o, const float* i, int h) { o[threadIdx.x] = i[threadIdx.x] + h; }
"""

DIFFERENT_PARAMETERS = """
__global__ void a(float* o, const float* i, int n) {
    if (threadIdx.x < n) o[threadIdx.x] = i[threadIdx.x] * 2.0f;
}
__global__ void b(int* o, int n) { if (threadIdx.x < n) o[threadIdx.x] = n; }
"""

OPENCL_KERNELS = """
__kernel void a(__global float* o, __global const float* i, int n) {
    size_t g = get_global_id(0);
    if (g < n) o[g] = i[g] * 2.0f;
}
__kernel void b(__global int* o, int n) { size_t g = get_global_id(0); if (g < n) o[g] = n; }
"""


def _translate(tmp_path, source, suffix=".cu", **options):
    path = tmp_path / f"kernels{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="vulkan", **options)


def _validate(tmp_path, code):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler is None or validator is None:
        return
    asm = tmp_path / "module.spvasm"
    asm.write_text(code, encoding="utf-8")
    binary = tmp_path / "module.spv"
    subprocess.run(
        [assembler, "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        capture_output=True,
        check=True,
    )
    result = subprocess.run(
        [validator, "--target-env", "vulkan1.1", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def _entry_points(code):
    return re.findall(r'OpEntryPoint GLCompute %(\w+) "(\w+)"', code)


def _slots(code):
    """``{variable: (set, binding)}`` of the module's descriptor-bound variables."""
    sets = dict(re.findall(r"OpDecorate %(\w+) DescriptorSet (\d+)", code))
    bindings = dict(re.findall(r"OpDecorate %(\w+) Binding (\d+)", code))
    return {name: (int(sets[name]), int(bindings[name])) for name in bindings}


def _function_body(code, function_id):
    match = re.search(
        rf"^\s*%{function_id} = OpFunction .*?^\s*OpFunctionEnd",
        code,
        re.MULTILINE | re.DOTALL,
    )
    assert match, function_id
    return match.group(0)


def _accessed_variables(body):
    return set(re.findall(r"OpAccessChain %\w+ %(\w+)", body))


def test_kernels_with_the_same_parameters_share_their_resources(tmp_path):
    code = _translate(tmp_path, SAME_PARAMETERS)
    entries = dict((name, function) for function, name in _entry_points(code))
    assert sorted(entries) == ["a", "b"]
    slots = _slots(code)
    assert slots["o"] == (0, 0) and slots["i"] == (0, 1)
    # One buffer variable for both kernels, and each kernel's own constant block, both at binding 2.
    assert code.count('OpName %o "o"') == 1
    assert slots["a_Args_0"] == slots["b_Args_0"] == (0, 2)
    assert "b_Args_0" not in _accessed_variables(_function_body(code, entries["a"]))
    assert "a_Args_0" not in _accessed_variables(_function_body(code, entries["b"]))
    _validate(tmp_path, code)


@pytest.mark.parametrize(
    "source, suffix",
    [(DIFFERENT_PARAMETERS, ".cu"), (OPENCL_KERNELS, ".cl")],
    ids=["cuda", "opencl"],
)
def test_kernels_with_different_parameters_bind_their_own_resources(
    tmp_path, source, suffix
):
    code = _translate(tmp_path, source, suffix)
    entries = dict((name, function) for function, name in _entry_points(code))
    slots = _slots(code)
    a_body = _function_body(code, entries["a"])
    b_body = _function_body(code, entries["b"])
    assert _accessed_variables(a_body) == {"a_Args_0", "i", "o"}
    # b's int buffer named o is not a's float buffer o: it is its own variable at the same slot.
    (b_buffer,) = _accessed_variables(b_body) - {"b_Args_0"}
    assert b_buffer != "o" and slots[b_buffer] == slots["o"] == (0, 0)
    assert re.search(rf"OpAccessChain %_ptr_\w+_int %{b_buffer} ", b_body)
    # Both blocks declare n; each kernel reads its own.
    assert slots["a_Args_0"] == (0, 2) and slots["b_Args_0"] == (0, 1)
    _validate(tmp_path, code)


def test_one_slot_is_shared_only_by_resources_no_entry_takes_together():
    codegen = SimpleNamespace(
        binding_occupants={},
        shared_binding_nodes=set(),
        reserved_resource_bindings=set(),
    )

    def resource(name, entries):
        node = VariableNode(name, "RWStructuredBuffer<float>")
        if entries is not None:
            node.add_annotation(SPIRV_ENTRY_SCOPE_ANNOTATION, frozenset(entries))
        return node

    first, other_kernel = resource("o", {"a"}), resource("p", {"b"})
    reserve_exact_binding(codegen, (0, 0), first, "o")
    reserve_exact_binding(codegen, (0, 0), other_kernel, "p")
    assert codegen.shared_binding_nodes == {id(first), id(other_kernel)}
    for clash in (resource("q", {"a"}), resource("r", {"a", "b"}), resource("s", None)):
        with pytest.raises(VulkanRefusal, match="Duplicate SPIR-V resource binding"):
            reserve_exact_binding(codegen, (0, 0), clash, clash.name)

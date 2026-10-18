"""SPIR-V copy and reference semantics of locals initialised from storage elements.

A value-typed local (``int s = a[i];``, ``S s = b[i];``) is a copy: the C++ and
HLSL rules give it its own storage, so later reads of ``s`` see the value at the
declaration and writes to ``s`` never reach the buffer. A reference local
(``device int& r = a[i];``) binds the element, so writes through ``r`` must reach
the buffer and ``r += 1`` adds to the element, not to an address.
"""

import re
import shutil
import subprocess

import pytest

import crosstl


def translate(tmp_path, suffix: str, source: str, backend: str = "vulkan") -> str:
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def validate(spv_code: str, tmp_path) -> None:
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        pytest.skip("spirv-as and spirv-val are not installed")
    asm_path = tmp_path / "shader.spvasm"
    spv_path = tmp_path / "shader.spv"
    asm_path.write_text(spv_code)
    for command in (
        [spirv_as, "--target-env", "vulkan1.0", str(asm_path), "-o", str(spv_path)],
        [spirv_val, "--target-env", "vulkan1.0", str(spv_path)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr


INSTRUCTION = re.compile(r"^\s*(?:(%\w+)\s*=\s*)?(Op\w+)\s*(.*)$")


def instructions(spv: str):
    parsed = []
    for line in spv.splitlines():
        match = INSTRUCTION.match(line)
        if match:
            result, opcode, operands = match.groups()
            parsed.append((result, opcode, operands.split()))
    return parsed


def store_roots(spv: str):
    """Return the root variable of every OpStore target, in program order."""
    defs = {
        result: (opcode, ops) for result, opcode, ops in instructions(spv) if result
    }
    roots = []
    for _, opcode, operands in instructions(spv):
        if opcode != "OpStore":
            continue
        pointer = operands[0]
        while defs.get(pointer, ("",))[0] in {"OpAccessChain", "OpInBoundsAccessChain"}:
            pointer = defs[pointer][1][1]
        roots.append(pointer)
    return roots


def function_variables(spv: str):
    return [
        (result, operands)
        for result, opcode, operands in instructions(spv)
        if opcode == "OpVariable" and "Function" in operands
    ]


HLSL_LOOP_FROM_BUFFER = """
StructuredBuffer<int> a : register(t0);
RWStructuredBuffer<int> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int start = a[id.x];
    int count = 0;
    for (int k = start; k < start + 4; ++k) { count += 1; }
    o[id.x] = count;
}
"""


def test_value_local_from_readonly_buffer_is_a_copy(tmp_path):
    spv = translate(tmp_path, ".hlsl", HLSL_LOOP_FROM_BUFFER)
    names = [result for result, _ in function_variables(spv)]
    assert "%start" in names and "%k" in names
    # The induction variable is its own storage, initialised from `start`, and
    # incremented in place; nothing writes to the read-only buffer.
    roots = store_roots(spv)
    assert roots.count("%k") == 2
    assert "%a" not in roots
    # No anonymous zero-initialised variable stands in for a write to an alias.
    assert not any(len(ops) > 3 for _, ops in function_variables(spv))
    condition = re.search(r"(%\w+) = OpSLessThan %bool (%\w+) (%\w+)", spv)
    defs = {r: (op, ops) for r, op, ops in instructions(spv) if r}
    assert defs[condition.group(2)] == ("OpLoad", ["%int", "%k"])
    validate(spv, tmp_path)


def test_struct_local_from_buffer_is_a_copy(tmp_path):
    spv = translate(
        tmp_path,
        ".hlsl",
        """
struct S { int x; int y; };
RWStructuredBuffer<S> b : register(u0);
RWStructuredBuffer<int> o : register(u1);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    S s = b[id.x];
    s.x = 7;
    int v = o[id.x];
    v += 1;
    o[id.x] = b[id.x].x + s.x + v;
}
""",
    )
    roots = store_roots(spv)
    assert "%b" not in roots
    assert roots.count("%s") == 2
    assert roots.count("%v") == 2
    assert roots[-1] == "%o"
    validate(spv, tmp_path)


CROSSGL_REFERENCE = """
shader ReferenceToElement {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main(RWStructuredBuffer<int> a @buffer(0),
                  RWStructuredBuffer<int> o @buffer(1),
                  uint id @gl_GlobalInvocationID) {
            device int& r = a[id];
            r += 1;
            r = r * 3;
            int v = a[id];
            v += 2;
            buffer_store(o, id, v);
        }
    }
}
"""


def test_reference_local_writes_through_to_the_element(tmp_path):
    spv = translate(tmp_path, ".cgl", CROSSGL_REFERENCE)
    names = [result for result, _ in function_variables(spv)]
    assert "%r" not in names
    # Both writes through `r` store to the buffer element; `v` stays a copy.
    assert store_roots(spv)[-5:] == ["%a", "%a", "%v", "%v", "%o"]
    assert not any(len(ops) > 3 for _, ops in function_variables(spv))
    assert "WARNING" not in spv
    validate(spv, tmp_path)


METAL_REFERENCE = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device int* a [[buffer(0)]], device int* o [[buffer(1)]],
              uint id [[thread_position_in_grid]]) {
    device int& r = a[id];
    r += 1;
    int v = a[id];
    v += 2;
    o[id] = v;
}
"""


def test_metal_importer_binds_reference_to_the_element(tmp_path):
    crossgl = translate(tmp_path, ".metal", METAL_REFERENCE, backend="cgl")
    assert re.search(r"device int& r = a\[id\];", crossgl)
    assert "int v = buffer_load(a, id);" in crossgl

    spv = translate(tmp_path, ".metal", METAL_REFERENCE)
    assert store_roots(spv)[-4:] == ["%a", "%v", "%v", "%o"]
    validate(spv, tmp_path)

    metal = translate(tmp_path, ".metal", METAL_REFERENCE, backend="metal")
    assert re.search(r"device int &r = a\[id\];\s*r \+= 1;", metal)


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_reference_to_element_is_refused_where_aliasing_is_not_expressible(
    backend, tmp_path
):
    with pytest.raises(ValueError, match="reference"):
        translate(tmp_path, ".metal", METAL_REFERENCE, backend=backend)

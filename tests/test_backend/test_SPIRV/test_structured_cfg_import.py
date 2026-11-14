"""SPIR-V structured control flow is imported as nested CrossGL statements, or refused with a typed diagnostic.

SPIR-V 1.6 section 2.11 defines the constructs: a loop header's ``OpLoopMerge`` names its merge block and continue
target, a selection header's ``OpSelectionMerge`` names its merge block, and a branch may leave only its innermost
loop or switch. The importer rebuilds ``while``, ``if`` and ``switch`` statements from those instructions, turns
loop-carried ``OpPhi`` values into variables assigned on their incoming edges, and refuses a shape that structured
statements cannot express instead of emitting it as straight-line code. Execution on every runtime is covered by the
``s_structured_control_flow`` and ``s_structured_control_flow_phi`` conformance cases; these tests check the shape
of the imported CrossGL and that the outputs compile.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.SPIRV.structured_cfg import SpirvUnstructuredControlFlowError

HEADER = """               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid
               OpExecutionMode %main LocalSize 64 1 1
               OpName %main "main"
               OpName %Buf "Buf"
               OpMemberName %Buf 0 "v"
               OpName %buf "buf"
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %rt ArrayStride 4
               OpMemberDecorate %Buf 0 Offset 0
               OpDecorate %Buf Block
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
       %bool = OpTypeBool
     %v3uint = OpTypeVector %uint 3
         %rt = OpTypeRuntimeArray %uint
        %Buf = OpTypeStruct %rt
    %ptr_buf = OpTypePointer StorageBuffer %Buf
   %ptr_uint = OpTypePointer StorageBuffer %uint
    %ptr_gid = OpTypePointer Input %v3uint
     %ptr_in = OpTypePointer Input %uint
      %int_0 = OpConstant %int 0
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
     %uint_3 = OpConstant %uint 3
    %uint_10 = OpConstant %uint 10
        %buf = OpVariable %ptr_buf StorageBuffer
        %gid = OpVariable %ptr_gid Input
       %main = OpFunction %void None %fn
      %entry = OpLabel
         %gp = OpAccessChain %ptr_in %gid %uint_0
          %i = OpLoad %uint %gp
         %xp = OpAccessChain %ptr_uint %buf %int_0 %i
          %x = OpLoad %uint %xp
"""

FOOTER = """               OpFunctionEnd
"""

# Two loop-carried values that swap every iteration: the back edge assigns both phis at once.
SWAP = """               OpBranch %header
     %header = OpLabel
          %p = OpPhi %uint %x %entry %q %cont
          %q = OpPhi %uint %uint_1 %entry %p %cont
          %k = OpPhi %uint %uint_0 %entry %kn %cont
               OpLoopMerge %merge %cont None
               OpBranch %check
      %check = OpLabel
          %c = OpULessThan %bool %k %uint_3
               OpBranchConditional %c %body %merge
       %body = OpLabel
               OpBranch %cont
       %cont = OpLabel
         %kn = OpIAdd %uint %k %uint_1
               OpBranch %header
      %merge = OpLabel
         %o0 = OpAccessChain %ptr_uint %buf %int_0 %uint_0
               OpStore %o0 %p
         %o1 = OpAccessChain %ptr_uint %buf %int_0 %uint_1
               OpStore %o1 %q
               OpReturn
"""

# Nested loops: the inner loop breaks from inside a selection and stores to a word the loop bound may alias.
NESTED = """               OpBranch %h1
         %h1 = OpLabel
          %r = OpPhi %uint %uint_0 %entry %rn %c1
        %acc = OpPhi %uint %uint_0 %entry %acc2 %c1
         %t1 = OpULessThan %bool %r %uint_3
               OpLoopMerge %m1 %c1 None
               OpBranchConditional %t1 %b1 %m1
         %b1 = OpLabel
               OpBranch %h2
         %h2 = OpLabel
          %c = OpPhi %uint %uint_0 %b1 %cn %c2
       %acc1 = OpPhi %uint %acc %b1 %accn %c2
         %t2 = OpULessThan %bool %c %x
               OpLoopMerge %m2 %c2 None
               OpBranchConditional %t2 %b2 %m2
         %b2 = OpLabel
         %cr = OpIMul %uint %c %r
         %t3 = OpUGreaterThan %bool %cr %uint_10
               OpSelectionMerge %s3 None
               OpBranchConditional %t3 %m2 %s3
         %s3 = OpLabel
       %accn = OpIAdd %uint %acc1 %c
         %sp = OpAccessChain %ptr_uint %buf %int_0 %c
               OpStore %sp %accn
               OpBranch %c2
         %c2 = OpLabel
         %cn = OpIAdd %uint %c %uint_1
               OpBranch %h2
         %m2 = OpLabel
       %acc2 = OpPhi %uint %acc1 %h2 %acc1 %b2
               OpBranch %c1
         %c1 = OpLabel
         %rn = OpIAdd %uint %r %uint_1
               OpBranch %h1
         %m1 = OpLabel
         %o1 = OpAccessChain %ptr_uint %buf %int_0 %uint_1
               OpStore %o1 %acc
               OpReturn
"""

# A continue from inside a selection: the continue construct's statements run before the jump.
CONTINUE = """               OpBranch %h
          %h = OpLabel
          %k = OpPhi %uint %uint_0 %entry %kn %cont
          %c = OpULessThan %bool %k %x
               OpLoopMerge %m %cont None
               OpBranchConditional %c %body %m
       %body = OpLabel
        %bit = OpBitwiseAnd %uint %k %uint_1
        %odd = OpIEqual %bool %bit %uint_1
               OpSelectionMerge %sm None
               OpBranchConditional %odd %cont %sm
         %sm = OpLabel
         %kp = OpAccessChain %ptr_uint %buf %int_0 %k
               OpStore %kp %uint_3
               OpBranch %cont
       %cont = OpLabel
         %kn = OpIAdd %uint %k %uint_1
         %o9 = OpAccessChain %ptr_uint %buf %int_0 %uint_10
               OpStore %o9 %kn
               OpBranch %h
          %m = OpLabel
               OpReturn
"""

# Case 0 falls through to the default, which falls through to case 1; SPIR-V lists case 0 immediately before case 1.
SWITCH = """          %s = OpBitwiseAnd %uint %x %uint_3
               OpSelectionMerge %merge None
               OpSwitch %s %dflt 0 %c0 1 %c1 2 %c2
         %c0 = OpLabel
         %o0 = OpAccessChain %ptr_uint %buf %int_0 %uint_0
               OpStore %o0 %uint_0
               OpBranch %dflt
       %dflt = OpLabel
         %o3 = OpAccessChain %ptr_uint %buf %int_0 %uint_3
               OpStore %o3 %uint_3
               OpBranch %c1
         %c1 = OpLabel
         %o1 = OpAccessChain %ptr_uint %buf %int_0 %uint_1
               OpStore %o1 %uint_1
               OpBranch %merge
         %c2 = OpLabel
         %o2 = OpAccessChain %ptr_uint %buf %int_0 %uint_2
               OpStore %o2 %uint_2
               OpBranch %merge
      %merge = OpLabel
               OpReturn
"""

# A value the loop header loads is used after the loop, after a store to the word it was loaded from.
HEADER_VALUE = """               OpBranch %h
          %h = OpLabel
          %k = OpPhi %uint %uint_0 %entry %kn %cont
         %kp = OpAccessChain %ptr_uint %buf %int_0 %k
          %v = OpLoad %uint %kp
          %c = OpULessThan %bool %v %x
               OpLoopMerge %m %cont None
               OpBranchConditional %c %body %m
       %body = OpLabel
               OpStore %kp %uint_0
               OpBranch %cont
       %cont = OpLabel
         %kn = OpIAdd %uint %k %uint_1
               OpBranch %h
          %m = OpLabel
               OpStore %kp %uint_3
         %o1 = OpAccessChain %ptr_uint %buf %int_0 %uint_1
               OpStore %o1 %v
               OpReturn
"""

# The selection condition reads a word that its true branch overwrites; the merge phi selects on that condition.
SELECT_PHI = """         %o1 = OpAccessChain %ptr_uint %buf %int_0 %uint_1
         %l1 = OpLoad %uint %o1
          %c = OpULessThan %bool %l1 %uint_3
               OpSelectionMerge %m None
               OpBranchConditional %c %a %b
          %a = OpLabel
         %t = OpLoad %uint %o1
               OpStore %o1 %uint_10
               OpBranch %m
          %b = OpLabel
               OpBranch %m
          %m = OpLabel
          %w = OpPhi %uint %t %a %x %b
         %o2 = OpAccessChain %ptr_uint %buf %int_0 %uint_2
               OpStore %o2 %w
               OpReturn
"""

# An atomic whose result is used twice, after a store that follows it.
ATOMIC = """         %o1 = OpAccessChain %ptr_uint %buf %int_0 %uint_1
          %a = OpAtomicIAdd %uint %o1 %uint_1 %uint_0 %uint_1
         %o2 = OpAccessChain %ptr_uint %buf %int_0 %uint_2
               OpStore %o2 %uint_3
          %b = OpIAdd %uint %a %a
               OpStore %o2 %b
               OpReturn
"""

VALID = {
    "swap": SWAP,
    "nested": NESTED,
    "continue": CONTINUE,
    "switch": SWITCH,
    "header_value": HEADER_VALUE,
    "select_phi": SELECT_PHI,
    "atomic": ATOMIC,
}

# Shapes outside SPIR-V's structured control flow rules, which no structured statement expresses.
UNSTRUCTURED = {
    "switch_without_merge": (
        """          %s = OpBitwiseAnd %uint %x %uint_3
               OpSwitch %s %merge 1 %c1
         %c1 = OpLabel
               OpBranch %merge
      %merge = OpLabel
               OpReturn
""",
        "OpSwitch in block %entry has no OpSelectionMerge",
    ),
    "conditional_without_merge": (
        """          %c = OpULessThan %bool %x %uint_3
               OpBranchConditional %c %a %b
          %a = OpLabel
               OpBranch %m
          %b = OpLabel
               OpBranch %m
          %m = OpLabel
               OpReturn
""",
        "has no merge instruction and neither target leaves the construct",
    ),
    "break_out_of_two_loops": (
        """               OpBranch %h1
         %h1 = OpLabel
               OpLoopMerge %m1 %c1 None
               OpBranch %h2
         %h2 = OpLabel
               OpLoopMerge %m2 %c2 None
               OpBranch %b2
         %b2 = OpLabel
          %c = OpULessThan %bool %x %uint_3
               OpBranchConditional %c %m1 %c2
         %c2 = OpLabel
               OpBranch %h2
         %m2 = OpLabel
               OpBranch %c1
         %c1 = OpLabel
               OpBranch %h1
         %m1 = OpLabel
               OpReturn
""",
        "branch to %m1 leaves more than one construct",
    ),
    "case_entered_from_a_non_preceding_case": (
        """          %s = OpBitwiseAnd %uint %x %uint_3
               OpSelectionMerge %merge None
               OpSwitch %s %merge 1 %c1 2 %c2 3 %c3
         %c1 = OpLabel
               OpBranch %c3
         %c2 = OpLabel
               OpBranch %c3
         %c3 = OpLabel
               OpBranch %merge
      %merge = OpLabel
               OpReturn
""",
        "two switch cases fall through to case %c3",
    ),
}


def _write(tmp_path, body, name="shader.spvasm"):
    path = tmp_path / name
    path.write_text(HEADER + body + FOOTER, encoding="utf-8")
    return str(path)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def _crossgl(tmp_path, body):
    crossgl = crosstl.translate(
        _write(tmp_path, body), backend="cgl", format_output=False
    )
    return re.sub(r"\s+", " ", crossgl[crossgl.index("void main()") :])


def _ordered(text, *parts):
    """Each part occurs in ``text``, after the one before it."""
    position = 0
    for part in parts:
        found = text.find(part, position)
        assert found >= 0, f"{part!r} not found after offset {position} in {text}"
        position = found + len(part)


@pytest.mark.parametrize("name", sorted(VALID))
def test_fixture_is_valid_spirv(tmp_path, name):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    binary = tmp_path / "shader.spv"
    _run(
        [
            "spirv-as",
            "--target-env",
            "vulkan1.1",
            _write(tmp_path, VALID[name]),
            "-o",
            str(binary),
        ]
    )
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def test_loop_carried_swap_reads_the_old_value_through_a_temporary(tmp_path):
    crossgl = _crossgl(tmp_path, SWAP)
    temporary = re.search(r"uint (q_next\d*) = p;", crossgl)
    assert temporary, crossgl
    _ordered(
        crossgl,
        "uint p; uint q; uint k;",
        "p = buf[0].v[gl_GlobalInvocationID[0]]; q = 1; k = 0;",
        "while ((k < 3)) {",
        temporary.group(0),
        "p = q;",
        f"q = {temporary.group(1)};",
        "k = (k + 1);",
        "}",
        "buf[0].v[0] = p; buf[0].v[1] = q;",
    )


def test_nested_loops_keep_their_nesting_and_inner_break(tmp_path):
    crossgl = _crossgl(tmp_path, NESTED)
    # The inner loop stores to buf.v[c], which may alias buf.v[gid.x], so the bound is read once, before the loops.
    _ordered(
        crossgl,
        "uint x = buf[0].v[gl_GlobalInvocationID[0]];",
        "r = 0; acc = 0;",
        "while ((r < 3)) {",
        "c = 0; acc1 = acc;",
        "while ((c < x)) {",
        "if (((c * r) > 10)) { break; }",
        "buf[0].v[c] = (acc1 + c);",
        "c = (c + 1);",
        "}",
        "r = (r + 1); acc = acc1; }",
        "buf[0].v[1] = acc;",
    )
    assert crossgl.count("while (") == 2
    # Both inner-loop phis change on the back edge, and acc1's next value reads c, which is assigned first.
    temporary = re.search(r"uint (acc1_next\d*) = \(acc1 \+ c\);", crossgl)
    assert temporary, crossgl
    _ordered(
        crossgl, temporary.group(0), "c = (c + 1);", f"acc1 = {temporary.group(1)};"
    )


def test_continue_from_a_selection_runs_the_continue_construct_first(tmp_path):
    crossgl = _crossgl(tmp_path, CONTINUE)
    _ordered(
        crossgl,
        "while ((k < x)) {",
        "if (((k & 1) == 1)) {",
        "buf[0].v[10] = (k + 1); k = (k + 1); continue; }",
        "buf[0].v[k] = 3;",
        "buf[0].v[10] = (k + 1); k = (k + 1); }",
    )
    assert crossgl.count("continue;") == 1


def test_switch_cases_fall_through_the_default_in_source_order(tmp_path):
    crossgl = _crossgl(tmp_path, SWITCH)
    _ordered(
        crossgl,
        "switch ((buf[0].v[gl_GlobalInvocationID[0]] & 3)) {",
        "case 0: buf[0].v[0] = 0;",
        "default: buf[0].v[3] = 3;",
        "case 1: buf[0].v[1] = 1; break;",
        "case 2: buf[0].v[2] = 2; break;",
    )
    # Neither case 0 nor the default has a break of its own.
    assert crossgl.count("break;") == 2


def test_header_value_used_after_the_loop_is_declared_outside_it(tmp_path):
    crossgl = _crossgl(tmp_path, HEADER_VALUE)
    _ordered(
        crossgl,
        "uint v;",
        "while (true) {",
        "v = buf[0].v[k];",
        "if (!(v < x)) { break; }",
        "buf[0].v[k] = 0;",
        "}",
        "buf[0].v[k] = 3;",
        "buf[0].v[1] = v;",
    )
    assert "uint v =" not in crossgl


def test_selection_phi_reads_the_condition_and_value_taken_before_the_store(
    tmp_path,
):
    crossgl = _crossgl(tmp_path, SELECT_PHI)
    _ordered(
        crossgl,
        "uint t;",
        "uint l1 = buf[0].v[1];",
        "if ((l1 < 3)) {",
        "t = buf[0].v[1];",
        "buf[0].v[1] = 10; }",
        "buf[0].v[2] = ((l1 < 3) ? t : x);",
    )


def test_atomic_result_is_evaluated_once(tmp_path):
    crossgl = _crossgl(tmp_path, ATOMIC)
    _ordered(
        crossgl,
        "uint a = atomicAdd(buf[0].v[1], 1);",
        "buf[0].v[2] = 3;",
        "buf[0].v[2] = (a + a);",
    )
    assert crossgl.count("atomicAdd") == 1


@pytest.mark.parametrize("name", sorted(UNSTRUCTURED))
def test_unstructured_control_flow_is_refused(tmp_path, name):
    body, detail = UNSTRUCTURED[name]
    with pytest.raises(SpirvUnstructuredControlFlowError) as error:
        crosstl.translate(_write(tmp_path, body), backend="cgl", format_output=False)
    assert detail in str(error.value)
    assert error.value.missing_capabilities == (
        "spirv.import.unstructured_control_flow",
    )


@pytest.mark.parametrize("name", sorted(UNSTRUCTURED))
def test_unstructured_control_flow_is_refused_for_every_target(tmp_path, name):
    body, _detail = UNSTRUCTURED[name]
    with pytest.raises(SpirvUnstructuredControlFlowError):
        crosstl.translate(_write(tmp_path, body), backend="opengl", format_output=False)


@pytest.mark.parametrize("name", sorted(VALID))
def test_glsl_output_compiles(tmp_path, name):
    code = crosstl.translate(
        _write(tmp_path, VALID[name]), backend="opengl", format_output=False
    )
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / "shader.comp"
    path.write_text(code, encoding="utf-8")
    _run(["glslangValidator", "-S", "comp", str(path)])


@pytest.mark.parametrize("name", sorted(VALID))
def test_spirv_output_validates(tmp_path, name):
    assembly = crosstl.translate(
        _write(tmp_path, VALID[name]), backend="vulkan", format_output=False
    )
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    path = tmp_path / "round_trip.spvasm"
    path.write_text(assembly, encoding="utf-8")
    binary = tmp_path / "round_trip.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", str(path), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])

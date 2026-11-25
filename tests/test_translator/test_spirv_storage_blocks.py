"""SPIR-V storage blocks and runtime-array lengths reach every target in a form its checker accepts.

The SPIR-V importer writes a storage block as a one-element structured buffer of the block struct, accessed through
``v[0]``, and ``OpArrayLength`` as ``spirvArrayLength(v, k)``. The DirectX emitter gave the runtime array 1024
elements, which DXC rejects, and no emitter knew ``spirvArrayLength``. Separately, HLSL, Metal, WGSL and SPIR-V
printed GLSL's ``block.member.length()`` or the hub's ``buffer_dimensions(b)`` unchanged. Execution evidence is in
the ``g_buffer_length``, ``g_buffer_count``, ``s_buffer_length`` and ``s_buffer_count`` conformance cases.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.buffer_length import BufferLengthUnsupportedError
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.semantics import (
    SpirvArrayLengthUnsupportedError,
    normalize_spirv_storage_blocks,
)
from tests.conformance.validators import validate

CASES = Path(__file__).resolve().parents[1] / "conformance" / "cases"
COUNT = CASES / "s_buffer_count" / "src.spvasm"  # block { int a[]; }
LENGTH = (
    CASES / "s_buffer_length" / "src.spvasm"
)  # block { int head; ivec2 pad; int a[]; }
GLSL_LENGTH = CASES / "g_buffer_length" / "src.comp"
GLSL_COUNT = CASES / "g_buffer_count" / "src.comp"
WRAP = CASES / "s_add_wrap" / "src.spvasm"  # blocks without a length query

HLSL_ENTRY = re.compile(r"\[numthreads\([^)]*\)\]\s*void\s+(\w+)")
SUFFIX = {"hlsl": ".hlsl", "glsl": ".comp", "spirv": ".spvasm", "metal": ".metal"}
BACKEND = {"hlsl": "directx", "glsl": "opengl", "spirv": "vulkan", "metal": "metal"}


def translate(source, target):
    return crosstl.translate(str(source), backend=BACKEND.get(target, target))


def check_natively(tmp_path, target, code):
    """Run the target's checker when it is installed; a missing tool is not a failure here."""
    if target == "wgsl":
        if not shutil.which("naga"):
            return
        (tmp_path / "out.wgsl").write_text(code)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
        return
    path = tmp_path / f"out{SUFFIX[target]}"
    path.write_text(code)
    entry = HLSL_ENTRY.search(code).group(1) if target == "hlsl" else "main"
    results, _ = validate(path, target, tmp_path / "out.bin", entry)
    for result in results:
        assert result.status in ("accepted", "unavailable"), result.log


@pytest.mark.parametrize("source", [COUNT, LENGTH, WRAP])
@pytest.mark.parametrize("target", ["hlsl", "glsl", "spirv", "wgsl"])
def test_spirv_storage_blocks_translate_to_checked_output(tmp_path, source, target):
    code = translate(source, target)
    assert "spirvArrayLength" not in code
    assert "[1024]" not in code
    check_natively(tmp_path, target, code)


def test_single_array_block_becomes_a_structured_buffer_of_its_element():
    hlsl = translate(COUNT, "hlsl")
    assert re.search(r"\bStructuredBuffer<int> inp\b", hlsl)
    assert "inp[0]" not in hlsl
    assert "uint cgl_buffer_count(StructuredBuffer<int> b)" in hlsl
    assert "struct Inp" not in hlsl


def test_block_with_header_becomes_a_std430_byte_address_buffer():
    hlsl = translate(LENGTH, "hlsl")
    assert re.search(r"\bByteAddressBuffer inp\b", hlsl)
    # The runtime array starts after int head (0) and ivec2 pad (8): byte 16, stride 4.
    assert "int((cgl_buffer_bytes(inp) - 16u) / 4u)" in hlsl
    assert "inp.Load((16 + " in hlsl


def test_glsl_block_lengths_reach_hlsl_spirv_and_wgsl(tmp_path):
    for source in (GLSL_LENGTH, GLSL_COUNT):
        for target in ("hlsl", "spirv", "wgsl"):
            code = translate(source, target)
            assert ".length()" not in code
            check_natively(tmp_path, target, code)
    # Unformatted output keeps numeric ids whether or not SPIRV-Tools is installed to add friendly names.
    raw = crosstl.translate(str(GLSL_LENGTH), backend="vulkan", format_output=False)
    inp = re.search(r'^OpName %(\w+) "inp"$', raw, re.M).group(1)
    uint = re.search(r"^%(\w+) = OpTypeInt 32 0$", raw, re.M).group(1)
    assert re.search(rf"= OpArrayLength %{uint} %{inp} 2$", raw, re.M)
    assert "i32(arrayLength(&inp.a))" in translate(GLSL_LENGTH, "wgsl")


@pytest.mark.parametrize("source", [LENGTH, GLSL_LENGTH])
def test_metal_refuses_runtime_array_length(source):
    # MSL buffer arguments are pointers and carry no size (MSL 3.2 section 2.8).
    with pytest.raises(BufferLengthUnsupportedError) as raised:
        translate(source, "metal")
    assert raised.value.code == "project.translate.metal-buffer-length-unsupported"


def test_metal_translates_blocks_without_length_queries(tmp_path):
    check_natively(tmp_path, "metal", translate(WRAP, "metal"))


def spirv_like(
    body,
    declaration="RWStructuredBuffer<Block> v @set(0) @binding(0);",
    members="uint head; int a[];",
):
    return parse(f"""
        shader S {{
            struct Block {{ {members} }};
            {declaration}
            compute {{
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                void main() {{ {body} }}
            }}
        }}
        """)


def test_other_uses_leave_the_variable_unchanged():
    program = spirv_like("v[1].head = 0u;")
    assert not normalize_spirv_storage_blocks(program, "vulkan", "directx")
    assert program.global_variables[0].var_type.name == "RWStructuredBuffer"


@pytest.mark.parametrize("source", ["vulkan", "cgl", "opengl"])
def test_runtime_array_blocks_are_rewritten_for_every_source(source):
    # A struct ending in a runtime array has no element count, so the buffer can only hold one block.
    program = spirv_like("v[0].head = 0u;")
    assert normalize_spirv_storage_blocks(program, source, "directx")
    variable = program.global_variables[0]
    assert variable.var_type.name == "Block"
    assert [a.name for a in variable.attributes][0] == "glsl_buffer_block"


@pytest.mark.parametrize(
    "members,var_type",
    [("float a[];", "RWStructuredBuffer"), ("uint head; int a[];", "Block")],
    ids=["flat", "header"],
)
def test_unused_runtime_array_blocks_are_rewritten(members, var_type):
    # A module keeps the binding of a buffer its entry point never reads; its block struct ends in a runtime
    # array, which no target can declare as a plain struct.
    program = spirv_like("return;", members=members)
    assert normalize_spirv_storage_blocks(program, "vulkan", "metal")
    assert program.global_variables[0].var_type.name == var_type
    if var_type == "RWStructuredBuffer":
        assert not program.structs


def test_unused_fixed_size_blocks_are_left_alone():
    program = spirv_like("return;", members="uint head; int a[2];")
    assert not normalize_spirv_storage_blocks(program, "vulkan", "directx")
    assert program.global_variables[0].var_type.name == "RWStructuredBuffer"


@pytest.mark.parametrize("target", ["metal", "glsl", "hlsl"])
def test_unused_runtime_array_block_keeps_its_binding(tmp_path, target):
    source = tmp_path / "unused.cgl"
    source.write_text(
        """
        shader S {
            struct Read { float fin[]; };
            RWStructuredBuffer<Read> unused @set(0) @binding(0);
            struct Write { uint outp[]; };
            RWStructuredBuffer<Write> used @set(0) @binding(1);
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                uint3 gl_GlobalInvocationID @input @gl_GlobalInvocationID;
                void main() { used[0].outp[gl_GlobalInvocationID.x] = 1u; }
            }
        }
        """,
        encoding="utf-8",
    )
    code = crosstl.translate(str(source), backend=BACKEND[target], format_output=False)
    assert "fin[]" not in code
    check_natively(tmp_path, target, code)


def test_only_spirv_sources_rewrite_fixed_size_blocks():
    program = spirv_like("v[0].head = 0u;", members="uint head; int a[2];")
    assert not normalize_spirv_storage_blocks(program, "opengl", "directx")
    assert program.global_variables[0].var_type.name == "RWStructuredBuffer"
    assert normalize_spirv_storage_blocks(program, "vulkan", "directx")
    assert program.global_variables[0].var_type.name == "Block"


def test_array_length_of_anything_else_is_refused():
    program = spirv_like(
        "uint n = spirvArrayLength(w, 0);", "RWStructuredBuffer<int> w @binding(0);"
    )
    with pytest.raises(SpirvArrayLengthUnsupportedError) as raised:
        normalize_spirv_storage_blocks(program, "vulkan", "metal")
    assert raised.value.code == "project.translate.metal-spirv-array-length-unsupported"


def test_hlsl_buffer_dimensions_of_a_byte_address_buffer_is_refused():
    program = parse("""
        shader S {
            ByteAddressBuffer raw @binding(0);
            RWStructuredBuffer<uint> o @binding(1);
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                void main() { o[0] = buffer_dimensions(raw); }
            }
        }
        """)
    with pytest.raises(BufferLengthUnsupportedError, match="element count"):
        HLSLCodeGen().generate(program)


# slangc names the block of every RWStructuredBuffer<T> after the resource type (#4531):
# ``RWStructuredBuffer<float> outp; [numthreads(1, 1, 1)] void main(uint3 id :
# SV_DispatchThreadID) { outp[id.x] = 1.0; }`` disassembled (``slangc min.slang -target spirv``).
# ``{block}`` is the block's OpName and type id; the second parameter covers the read-only spelling.
RWSTRUCTUREDBUFFER_BLOCK_SPIRV = """; SPIR-V
; Version: 1.6
; Generator: Slang; 1
; Bound: 20
; Schema: 0
               OpCapability Shader
               OpExtension "SPV_KHR_storage_buffer_storage_class"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %outp %gl_GlobalInvocationID
               OpExecutionMode %main LocalSize 1 1 1
               OpSource Slang 1
               OpName %{block} "{block}"
               OpMemberName %{block} 0 "__member0"
               OpName %outp "outp"
               OpName %main "main"
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %_ptr_StorageBuffer_float ArrayStride 4
               OpDecorate %_runtimearr_float ArrayStride 4
               OpDecorate %{block} Block
               OpMemberDecorate %{block} 0 Offset 0
               OpDecorate %outp Binding 0
               OpDecorate %outp DescriptorSet 0{extra}
               %void = OpTypeVoid
                  %3 = OpTypeFunction %void
               %uint = OpTypeInt 32 0
             %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
                %int = OpTypeInt 32 1
             %int_0 = OpConstant %int 0
              %float = OpTypeFloat 32
%_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
   %_runtimearr_float = OpTypeRuntimeArray %float
%{block} = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_{block} = OpTypePointer StorageBuffer %{block}
            %float_1 = OpConstant %float 1
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
               %outp = OpVariable %_ptr_StorageBuffer_{block} StorageBuffer
               %main = OpFunction %void None %3
                  %4 = OpLabel
                  %7 = OpLoad %v3uint %gl_GlobalInvocationID
                 %10 = OpCompositeExtract %uint %7 0
                 %15 = OpAccessChain %_ptr_StorageBuffer_float %outp %int_0 %10
               OpStore %15 %float_1
               OpReturn
               OpFunctionEnd
"""


def _rw_block_source(tmp_path, block_name, extra=""):
    source = tmp_path / "min.spvasm"
    source.write_text(
        RWSTRUCTUREDBUFFER_BLOCK_SPIRV.format(block=block_name, extra=extra)
    )
    return source


@pytest.mark.parametrize("block_name", ["RWStructuredBuffer", "StructuredBuffer"])
@pytest.mark.parametrize("target", ["hlsl", "spirv", "metal", "wgsl"])
def test_spirv_block_named_like_a_resource_type_translates(
    tmp_path, block_name, target
):
    # The block name is only an OpName, so it must not change what is imported: the variable stays a
    # structured buffer of the runtime array's element, never a buffer of the block struct.
    code = translate(_rw_block_source(tmp_path, block_name), target)
    assert f"{block_name}<{block_name}>" not in code
    if target == "hlsl":
        assert re.search(r"\bRWStructuredBuffer<float> outp\b", code)
    if target == "wgsl":
        assert "var<storage, read_write> outp: array<f32>;" in code
    check_natively(tmp_path, target, code)


def test_spirv_block_named_like_a_resource_type_is_renamed_in_crossgl(tmp_path):
    code = crosstl.translate(
        str(_rw_block_source(tmp_path, "RWStructuredBuffer")), backend="cgl"
    )
    assert "RWStructuredBuffer<RWStructuredBuffer>" not in code
    # The importer renames the block at parse time (type_declaration_names.py): a struct
    # named like a built-in type takes a trailing underscore, so the wrapper no longer
    # reads as a buffer of buffers.
    assert "RWStructuredBuffer<RWStructuredBuffer_>" in code
    assert "struct RWStructuredBuffer_" in code


@pytest.mark.parametrize("target", ["cgl", "hlsl"])
def test_spirv_readonly_block_named_like_a_resource_type_translates(tmp_path, target):
    # The same collision on the read-only side: a read-only block is declared as StructuredBuffer<B>, so a
    # block named ``StructuredBuffer`` used to become ``StructuredBuffer<StructuredBuffer> outp``. Only the
    # CrossGL and HLSL forms are checked here: the reproducer stores to the buffer, which the SPIR-V, Metal
    # and WGSL generators refuse for a read-only buffer.
    source = _rw_block_source(
        tmp_path,
        "StructuredBuffer",
        extra="\n               OpDecorate %outp NonWritable",
    )
    code = crosstl.translate(str(source), backend=target)
    assert "StructuredBuffer<StructuredBuffer>" not in code
    if target == "cgl":
        assert "StructuredBuffer<StructuredBuffer_>" in code
    if target == "hlsl":
        assert re.search(r"\bStructuredBuffer<float> outp\b", code)


# The coverage probe for SPIR-V input ``pointer-arithmetic`` (tools/coverage/templates/pointers.py), as slangc
# compiles it: ``float* p = &outp[0] + i; *p = fin[i]; float* q = p - i; q[i] = *(q + i) * 4.0f;``. Both
# buffers share the one block type, which slangc names ``RWStructuredBuffer`` (#4531), so the importer must
# rename the block *and* translate the pointer arithmetic. ``OpPtrAccessChain`` advances its base pointer by
# its Element operand; it is not another traversal index: appending it as one used to import ``*p`` as
# ``outp[0].__member0[0][i]`` — a subscript on a float — and the SPIR-V generator re-emitted it as an
# ``OpAccessChain`` that reaches the float while indexes remain (spirv-val: "OpAccessChain reached
# non-composite type while indexes still remain to be traversed"), turning a clean refusal into invalid
# SPIR-V on the round trip.
POINTER_ARITHMETIC_SPIRV = """; SPIR-V
; Version: 1.5
; Generator: Slang; 1
; Bound: 35
; Schema: 0
               OpCapability VariablePointersStorageBuffer
               OpCapability Shader
               OpExtension "SPV_KHR_storage_buffer_storage_class"
               OpExtension "SPV_KHR_variable_pointers"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %outp %fin %gl_GlobalInvocationID
               OpExecutionMode %main LocalSize 1 1 1
               OpSource Slang 1
               OpName %RWStructuredBuffer "RWStructuredBuffer"
               OpMemberName %RWStructuredBuffer 0 "__member0"
               OpName %outp "outp"
               OpName %fin "fin"
               OpName %main "main"
               OpDecorate %_runtimearr_float ArrayStride 4
               OpDecorate %RWStructuredBuffer Block
               OpMemberDecorate %RWStructuredBuffer 0 Offset 0
               OpDecorate %outp Binding 1
               OpDecorate %outp DescriptorSet 0
               OpDecorate %fin Binding 0
               OpDecorate %fin DescriptorSet 0
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %_ptr_StorageBuffer_float ArrayStride 4
               %void = OpTypeVoid
                  %3 = OpTypeFunction %void
               %uint = OpTypeInt 32 0
             %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
                %int = OpTypeInt 32 1
             %int_0 = OpConstant %int 0
              %float = OpTypeFloat 32
%_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
   %_runtimearr_float = OpTypeRuntimeArray %float
%RWStructuredBuffer = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_RWStructuredBuffer = OpTypePointer StorageBuffer %RWStructuredBuffer
            %float_4 = OpConstant %float 4
            %float_1 = OpConstant %float 1
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
               %outp = OpVariable %_ptr_StorageBuffer_RWStructuredBuffer StorageBuffer
               %fin = OpVariable %_ptr_StorageBuffer_RWStructuredBuffer StorageBuffer
               %main = OpFunction %void None %3
                 %14 = OpLabel
                 %15 = OpLoad %v3uint %gl_GlobalInvocationID
                  %i = OpCompositeExtract %uint %15 0
                 %20 = OpAccessChain %_ptr_StorageBuffer_float %outp %int_0 %int_0
                  %p = OpPtrAccessChain %_ptr_StorageBuffer_float %20 %i
                 %22 = OpAccessChain %_ptr_StorageBuffer_float %fin %int_0 %i
                 %23 = OpLoad %float %22
               OpStore %p %23
                 %25 = OpSNegate %uint %i
                 %26 = OpPtrAccessChain %_ptr_StorageBuffer_float %p %25
                 %27 = OpPtrAccessChain %_ptr_StorageBuffer_float %26 %i
                 %28 = OpLoad %float %27
                 %29 = OpFMul %float %28 %float_4
                 %31 = OpFAdd %float %29 %float_1
               OpStore %27 %31
               OpReturn
               OpFunctionEnd
"""


def _pointer_arithmetic_source(tmp_path):
    source = tmp_path / "pointer-arithmetic.spvasm"
    source.write_text(POINTER_ARITHMETIC_SPIRV)
    return source


def _assert_access_chains_traverse_composites(assembly):
    """Every ``OpAccessChain``/``OpPtrAccessChain`` index must traverse a composite (SPIR-V 1.6 section
    3.49.8): an index left over after reaching a scalar is the invalid output spirv-val reports as
    "OpAccessChain reached non-composite type while indexes still remain to be traversed". A pointer
    arithmetic element advances the base pointer, so it consumes no composite level."""
    types = {}
    constants = {}
    variables = {}
    for line in assembly.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].startswith("%") or parts[1] != "=":
            continue
        result, opcode, operands = parts[0], parts[2], parts[3:]
        if opcode == "OpTypePointer":
            types[result] = ("pointer", operands[1])
        elif opcode == "OpTypeStruct":
            types[result] = ("struct", operands)
        elif opcode in {"OpTypeRuntimeArray", "OpTypeArray"}:
            types[result] = ("array", operands[0])
        elif opcode == "OpTypeVector":
            types[result] = ("vector", operands[0])
        elif opcode == "OpTypeMatrix":
            types[result] = ("matrix", operands[0])
        elif opcode == "OpTypeInt" or opcode == "OpTypeFloat" or opcode == "OpTypeBool":
            types[result] = ("scalar",)
        elif opcode == "OpConstant" and len(operands) == 2:
            try:
                constants[result] = int(operands[1])
            except ValueError:
                pass  # a float constant: never a struct index
        elif opcode == "OpVariable":
            variables[result] = operands[0]

    def pointee(type_id):
        info = types.get(type_id)
        return info[1] if info and info[0] == "pointer" else None

    def composite_step(type_id, index_id):
        info = types.get(type_id)
        if info is None:
            return None
        kind = info[0]
        if kind == "struct":
            index = constants.get(index_id)
            assert index is not None, f"struct index {index_id} is not a constant"
            assert 0 <= index < len(info[1]), f"struct index {index} out of range"
            return info[1][index]
        if kind in {"array", "vector", "matrix"}:
            return info[1]
        return False  # scalar: no index may remain

    for line in assembly.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[1] != "=":
            continue
        opcode = parts[2]
        if opcode not in {
            "OpAccessChain",
            "OpInBoundsAccessChain",
            "OpPtrAccessChain",
            "OpInBoundsPtrAccessChain",
        }:
            continue
        operands = parts[3:]
        base = operands[1]
        indexes = operands[2:]
        if opcode in {"OpPtrAccessChain", "OpInBoundsPtrAccessChain"}:
            indexes = indexes[
                1:
            ]  # the Element advances the pointer; it indexes nothing
        base_type = variables.get(base) or _chain_result_type(
            assembly, base, types, variables
        )
        current = pointee(base_type)
        assert current is not None, f"{opcode} base {base} has no pointer type"
        for index_id in indexes:
            stepped = composite_step(current, index_id)
            assert stepped is not False, (
                f"{opcode} in {line.strip()}: reached non-composite type "
                "while indexes still remain to be traversed"
            )
            current = stepped


def _chain_result_type(assembly, result_id, types, variables):
    """The result type id of the instruction defining ``result_id`` (for chained access results)."""
    for line in assembly.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == result_id and parts[1] == "=":
            return (
                parts[3]
                if parts[2]
                in {
                    "OpAccessChain",
                    "OpInBoundsAccessChain",
                    "OpPtrAccessChain",
                    "OpInBoundsPtrAccessChain",
                }
                else None
            )
    return None


def test_spirv_pointer_arithmetic_advances_the_last_index(tmp_path):
    code = crosstl.translate(str(_pointer_arithmetic_source(tmp_path)), backend="cgl")
    i = "gl_GlobalInvocationID[0]"
    # ``*p`` is element ``i`` of the runtime array, not a subscript on element 0: the
    # Element folds into the base index, and a zero base index is dropped.
    assert "__member0[0][" not in code
    assert f"outp[0].__member0[{i}] = fin[0].__member0[{i}];" in code
    # ``q = p - i`` then ``q[i]`` is element ``((i + -i) + i)``: still one subscript each.
    assert f"__member0[(({i} + -{i}) + {i})]" in code
    assert code.count("__member0[(") == 2


def test_spirv_pointer_arithmetic_round_trip_has_valid_access_chains(tmp_path):
    code = crosstl.translate(
        str(_pointer_arithmetic_source(tmp_path)), backend="vulkan"
    )
    _assert_access_chains_traverse_composites(code)

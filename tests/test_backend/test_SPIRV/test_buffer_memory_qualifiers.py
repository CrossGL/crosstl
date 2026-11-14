"""The SPIR-V importer keeps a storage block's memory qualifiers as attributes of the imported buffer.

``Coherent``, ``Volatile``, ``Restrict`` and ``NonReadable`` were dropped, and ``NonWritable`` was honoured only on the
variable, so a block whose members alone were non-writable (CrossTL's own SPIR-V output decorates only members) was
imported writable. A plain store into a ``writeonly`` buffer was also refused by the OpenGL emitter, which demanded
read-write access for every store.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.SPIRV.buffer_memory_qualifiers import (
    storage_block_memory_qualifiers,
)
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused

# Four storage blocks of one runtime float array each, and one two-member block:
#   %src  binding 0: NonWritable and Restrict on the member only (CrossTL's own output form);
#   %dst  binding 1: Coherent on the variable and the member (glslang's form);
#   %vol  binding 2: Volatile on the variable;
#   %wo   binding 3: NonReadable on the member only;
#   %two  binding 4: Coherent and Restrict on member 0 alone.
MODULE = """               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid
               OpExecutionMode %main LocalSize 1 1 1
               OpName %Src "Src"
               OpMemberName %Src 0 "fin"
               OpName %src "src"
               OpName %Dst "Dst"
               OpMemberName %Dst 0 "outp"
               OpName %dst "dst"
               OpName %Vol "Vol"
               OpMemberName %Vol 0 "v"
               OpName %vol "vol"
               OpName %Wo "Wo"
               OpMemberName %Wo 0 "w"
               OpName %wo "wo"
               OpName %Two "Two"
               OpMemberName %Two 0 "a"
               OpMemberName %Two 1 "b"
               OpName %two "two"
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %rt ArrayStride 4
               OpDecorate %Src Block
               OpMemberDecorate %Src 0 Offset 0
               OpMemberDecorate %Src 0 NonWritable
               OpMemberDecorate %Src 0 Restrict
               OpDecorate %src DescriptorSet 0
               OpDecorate %src Binding 0
               OpDecorate %Dst Block
               OpMemberDecorate %Dst 0 Offset 0
               OpMemberDecorate %Dst 0 Coherent
               OpDecorate %dst Coherent
               OpDecorate %dst DescriptorSet 0
               OpDecorate %dst Binding 1
               OpDecorate %Vol Block
               OpMemberDecorate %Vol 0 Offset 0
               OpDecorate %vol Volatile
               OpDecorate %vol DescriptorSet 0
               OpDecorate %vol Binding 2
               OpDecorate %Wo Block
               OpMemberDecorate %Wo 0 Offset 0
               OpMemberDecorate %Wo 0 NonReadable
               OpDecorate %wo DescriptorSet 0
               OpDecorate %wo Binding 3
               OpDecorate %Two Block
               OpMemberDecorate %Two 0 Offset 0
               OpMemberDecorate %Two 0 Coherent
               OpMemberDecorate %Two 0 Restrict
               OpMemberDecorate %Two 1 Offset 4
               OpDecorate %two DescriptorSet 0
               OpDecorate %two Binding 4
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
     %v3uint = OpTypeVector %uint 3
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
     %uint_0 = OpConstant %uint 0
    %float_1 = OpConstant %float 1
    %float_2 = OpConstant %float 2
         %rt = OpTypeRuntimeArray %float
        %Src = OpTypeStruct %rt
        %Dst = OpTypeStruct %rt
        %Vol = OpTypeStruct %rt
         %Wo = OpTypeStruct %rt
        %Two = OpTypeStruct %float %float
    %ptr_src = OpTypePointer StorageBuffer %Src
    %ptr_dst = OpTypePointer StorageBuffer %Dst
    %ptr_vol = OpTypePointer StorageBuffer %Vol
     %ptr_wo = OpTypePointer StorageBuffer %Wo
    %ptr_two = OpTypePointer StorageBuffer %Two
     %ptr_in = OpTypePointer Input %v3uint
    %ptr_flt = OpTypePointer StorageBuffer %float
        %src = OpVariable %ptr_src StorageBuffer
        %dst = OpVariable %ptr_dst StorageBuffer
        %vol = OpVariable %ptr_vol StorageBuffer
         %wo = OpVariable %ptr_wo StorageBuffer
        %two = OpVariable %ptr_two StorageBuffer
        %gid = OpVariable %ptr_in Input
       %main = OpFunction %void None %fn
      %entry = OpLabel
         %id = OpLoad %v3uint %gid
          %i = OpCompositeExtract %uint %id 0
         %ps = OpAccessChain %ptr_flt %src %int_0 %i
          %x = OpLoad %float %ps
         %y2 = OpFMul %float %x %float_2
         %pd = OpAccessChain %ptr_flt %dst %int_0 %i
               OpStore %pd %y2
         %pv = OpAccessChain %ptr_flt %vol %int_0 %i
               OpStore %pv %float_1
         %pw = OpAccessChain %ptr_flt %wo %int_0 %i
               OpStore %pw %float_2
         %pa = OpAccessChain %ptr_flt %two %int_0
         %pb = OpAccessChain %ptr_flt %two %int_1
          %a = OpLoad %float %pa
               OpStore %pb %a
               OpReturn
               OpFunctionEnd
"""

MEMORY = ("Coherent", "Volatile", "Restrict", "NonReadable", "NonWritable")


@pytest.fixture
def module(tmp_path):
    path = tmp_path / "buffers.spvasm"
    path.write_text(MODULE)
    return str(path)


def _declaration(code, name):
    lines = [
        line.strip() for line in code.splitlines() if re.search(rf"\b{name}\b", line)
    ]
    assert lines, code
    return lines[0]


def _memory_decorations(spirv, block):
    """The memory decorations of the block type that ``OpName`` calls ``block``."""
    ids = re.findall(rf'OpName (%\w+) "{block}"', spirv)
    found = set()
    for decoration in re.findall(r"Op(?:Member)?Decorate (%\w+)(?: \d+)? (\w+)", spirv):
        if decoration[0] in ids and decoration[1] in MEMORY:
            found.add(decoration[1])
    return found


def test_qualifier_on_the_variable_holds_for_the_block():
    qualifiers = storage_block_memory_qualifiers(
        [("Coherent", []), ("NonReadable", []), ("Binding", ["1"])], [], 1
    )
    assert qualifiers == ["writeonly", "coherent"]


def test_member_decorations_lift_only_in_the_safe_direction():
    members = [("0", "Coherent", []), ("0", "Restrict", []), ("0", "NonWritable", [])]
    # One member of two: coherence covers the block; restrict and readonly would be promises the module never made.
    assert storage_block_memory_qualifiers([], members, 2) == ["coherent"]
    # Every member: all of them hold for the block.
    every = members + [("1", "Restrict", []), ("1", "NonWritable", [])]
    assert storage_block_memory_qualifiers([], every, 2) == [
        "readonly",
        "coherent",
        "restrict",
    ]


def test_import_writes_the_memory_qualifiers(module):
    crossgl = crosstl.translate(module, backend="cgl")
    assert _declaration(crossgl, "src").startswith("StructuredBuffer<")
    assert "@restrict" in _declaration(crossgl, "src")
    assert "@coherent" in _declaration(crossgl, "dst")
    assert "@volatile" in _declaration(crossgl, "vol")
    assert _declaration(crossgl, "wo").startswith("RWStructuredBuffer<")
    assert "@writeonly" in _declaration(crossgl, "wo")
    two = _declaration(crossgl, "two")
    assert "@coherent" in two and "@restrict" not in two


def test_spirv_round_trip_keeps_the_memory_decorations(module, tmp_path):
    spirv = crosstl.translate(module, backend="vulkan")
    assert {"NonWritable", "Restrict"} <= _memory_decorations(spirv, "srcBuffer")
    assert "Coherent" in _memory_decorations(spirv, "dstBuffer")
    assert "Volatile" in _memory_decorations(spirv, "volBuffer")
    assert "NonReadable" in _memory_decorations(spirv, "woBuffer")
    _validate_spirv(tmp_path, spirv)


def test_glsl_output_declares_the_memory_qualifiers(module, tmp_path):
    glsl = crosstl.translate(module, backend="opengl")
    blocks = {
        name: re.search(rf"(\w[\w ]*) buffer {name}Buffer\b", glsl)
        for name in ("src", "dst", "vol", "wo")
    }
    assert all(blocks.values()), glsl
    assert set(blocks["src"].group(1).split()) >= {"restrict", "readonly"}
    assert "coherent" in blocks["dst"].group(1).split()
    assert "volatile" in blocks["vol"].group(1).split()
    assert "writeonly" in blocks["wo"].group(1).split()
    _validate_glsl(tmp_path, glsl)


def test_hlsl_output_keeps_coherence_and_read_only_registers(module, tmp_path):
    # HLSL has no volatile resource, so the module's volatile buffer is refused ...
    with pytest.raises(DirectXLoweringRefused, match="resource 'vol' is volatile"):
        crosstl.translate(module, backend="directx")
    # ... and without it the other qualifiers carry over.
    path = tmp_path / "no_volatile.spvasm"
    path.write_text(MODULE.replace("OpDecorate %vol Volatile\n", ""))
    hlsl = crosstl.translate(str(path), backend="directx")
    assert re.search(r"\bStructuredBuffer<float> src : register\(t0\)", hlsl), hlsl
    assert re.search(
        r"globallycoherent RWStructuredBuffer<float> dst : register\(u1\)", hlsl
    ), hlsl


def test_opengl_store_into_writeonly_buffer_is_emitted(tmp_path):
    path = tmp_path / "store.cgl"
    path.write_text(_writeonly_program("wb[id.x] = 2.0;"))
    glsl = crosstl.translate(str(path), backend="opengl")
    assert "writeonly buffer" in glsl
    assert "wb[gl_GlobalInvocationID.x] = 2.0;" in glsl
    _validate_glsl(tmp_path, glsl)


def test_opengl_compound_assignment_into_writeonly_buffer_is_refused(tmp_path):
    path = tmp_path / "update.cgl"
    path.write_text(_writeonly_program("wb[id.x] += 2.0;"))
    with pytest.raises(Exception, match=r"read_write|read-write|writeonly"):
        crosstl.translate(str(path), backend="opengl")


def _writeonly_program(statement):
    return f"""shader main {{
    @ writeonly
    @ set(0)
    @ binding(3)
    RWStructuredBuffer<float> wb;
    compute {{
    @ numthreads(1, 1, 1)
    void main(uvec3 id @ gl_GlobalInvocationID) {{
        {statement}
    }}
    }}
}}
"""


def _validate_spirv(tmp_path, spirv):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler is None or validator is None:
        return
    source, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    source.write_text(spirv)
    assembled = subprocess.run(
        [assembler, "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert assembled.returncode == 0, assembled.stderr
    validated = subprocess.run(
        [validator, "--target-env", "vulkan1.1", str(binary)],
        capture_output=True,
        text=True,
    )
    assert validated.returncode == 0, validated.stderr + spirv


def _validate_glsl(tmp_path, glsl):
    validator = shutil.which("glslangValidator")
    if validator is None:
        return
    source = tmp_path / "out.comp"
    source.write_text(glsl)
    result = subprocess.run([validator, str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + glsl

"""GL_KHR_memory_scope_semantics coherence qualifiers on storage buffers and images.

The extension's scoped qualifiers (``devicecoherent``, ``queuefamilycoherent``, ``workgroupcoherent``,
``subgroupcoherent``, ``shadercallcoherent``) and ``nonprivate`` reach CrossGL as bare attributes or as
buffer-block qualifiers. Every generator used to drop them, and the DirectX generator wrote an unknown attribute
as an array size (``RWStructuredBuffer<uint> b[workgroupcoherent]``). The GLSL importer refused
``devicecoherent``, and the CrossGL parser refused the importer's own ``workgroupcoherent buffer B { ... }``.

Device scope holds every scope the extension names, so targets without scoped coherence make the resource
coherent across the device:

* HLSL: ``globallycoherent``.
* SPIR-V (GLSL450 memory model): ``Coherent``.
* GLSL without ``#pragma use_vulkan_memory_model``: ``coherent`` (glslang ignores the scoped names there).
* GLSL under the Vulkan memory model: the scoped names themselves.
* MSL 3.2 buffers: ``coherent(threadgroup)`` for workgroup and subgroup scope, ``coherent(device)`` otherwise.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.ast import AttributeNode, PreprocessorNode, VariableNode
from crosstl.translator.vulkan_memory_scope import (
    declares_vulkan_memory_model,
    glsl_memory_scope_qualifiers,
    metal_coherence_scope,
    vulkan_memory_scope_names,
)

ATTRIBUTES = """shader MemoryModel {
    RWStructuredBuffer<uint> b @workgroupcoherent;
    RWStructuredBuffer<uint> d @devicecoherent;
    RWStructuredBuffer<uint> n @nonprivate;
    RWStructuredBuffer<uint> s @subgroupcoherent;
    compute {
        void main() {
            b[0] = d[0] + n[0] + s[0];
        }
    }
}
"""

MIXED = """shader Mixed {
    RWStructuredBuffer<uint> b @coherent @workgroupcoherent;
    compute {
        void main() {
            b[0] = b[1];
        }
    }
}
"""

GLSL_BLOCKS = """#version 450
#extension GL_KHR_memory_scope_semantics : require
layout(local_size_x = 1) in;
layout(std430, binding = 0) workgroupcoherent buffer B { uint b[]; };
layout(std430, binding = 1) devicecoherent buffer D { uint d[]; };
layout(std430, binding = 2) nonprivate buffer N { uint n[]; };
layout(std430, binding = 3) subgroupcoherent buffer S { uint x; uint s[]; };
layout(binding = 4, r32ui) uniform queuefamilycoherent uimage2D img;
void main() { b[0] = d[0] + n[0] + x + s[0] + imageLoad(img, ivec2(0)).x; }
"""

GLSL_VULKAN_MEMORY_MODEL = GLSL_BLOCKS.replace(
    "#extension GL_KHR_memory_scope_semantics : require\n",
    "#extension GL_KHR_memory_scope_semantics : require\n"
    "#pragma use_vulkan_memory_model\n",
)

SCOPED_NAMES = (
    "workgroupcoherent",
    "devicecoherent",
    "nonprivate",
    "subgroupcoherent",
    "queuefamilycoherent",
)


def translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def spirv_ids(code):
    """Map OpName strings to the ids that name them.

    Output is disassembled with friendly names only when SPIRV-Tools are installed, so tests look ids up.
    """
    ids = {}
    for ident, name in re.findall(r'OpName %(\w+) "(\w+)"', code):
        ids.setdefault(name, ident)
    return ids


def run_tool(tmp_path, *command):
    if shutil.which(command[0]) is None:
        pytest.skip(f"{command[0]} is not installed")
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    diagnostics = result.stdout + result.stderr
    if result.returncode != 0 and "missing Metal Toolchain" in diagnostics:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, diagnostics
    return result.stdout


def glslang_disassembly(tmp_path, code, name, *options):
    (tmp_path / f"{name}.comp").write_text(code)
    run_tool(
        tmp_path,
        "glslangValidator",
        "-V",
        *options,
        f"{name}.comp",
        "-o",
        f"{name}.spv",
    )
    return run_tool(tmp_path, "spirv-dis", f"{name}.spv")


def test_scope_names_come_from_attributes_and_block_qualifiers():
    node = VariableNode(
        "b",
        "RWStructuredBuffer<uint>",
        qualifiers=["nonprivate", "buffer"],
        attributes=[AttributeNode("workgroupcoherent"), AttributeNode("binding", [])],
    )
    assert vulkan_memory_scope_names(node) == ("workgroupcoherent", "nonprivate")
    assert metal_coherence_scope(node) == "device"
    assert glsl_memory_scope_qualifiers(node, False) == ("coherent",)
    assert glsl_memory_scope_qualifiers(node, True) == (
        "workgroupcoherent",
        "nonprivate",
    )

    workgroup = VariableNode("w", "uimage2D", qualifiers=["subgroupcoherent"])
    assert metal_coherence_scope(workgroup) == "threadgroup"
    nontemporal = VariableNode("t", "uimage2D", qualifiers=["nontemporal"])
    assert vulkan_memory_scope_names(nontemporal) == ()
    assert metal_coherence_scope(nontemporal) is None


def test_vulkan_memory_model_needs_the_pragma_and_the_extension():
    pragma = PreprocessorNode("pragma", "use_vulkan_memory_model")
    extension = {"GL_KHR_memory_scope_semantics"}
    assert declares_vulkan_memory_model([pragma], extension)
    assert not declares_vulkan_memory_model([pragma], set())
    assert not declares_vulkan_memory_model([], extension)
    assert not declares_vulkan_memory_model(
        [PreprocessorNode("pragma", "optimize(off)")], extension
    )


def test_hlsl_output_makes_scoped_resources_globally_coherent(tmp_path):
    code = translate(tmp_path, ATTRIBUTES, "directx")

    for name in "bdns":
        assert re.search(
            rf"^globallycoherent RWStructuredBuffer<uint> {name} : register\(u\d\);$",
            code,
            re.MULTILINE,
        ), code
    assert not any(qualifier in code for qualifier in SCOPED_NAMES)
    (tmp_path / "out.hlsl").write_text(code)
    run_tool(tmp_path, "dxc", "-T", "cs_6_0", "-E", "CSMain", "out.hlsl")


def test_hlsl_output_of_imported_glsl_blocks_is_globally_coherent(tmp_path):
    code = translate(tmp_path, GLSL_BLOCKS, "directx", suffix=".comp")

    declarations = re.findall(r"^(.*) : register\(u\d\);$", code, re.MULTILINE)
    assert len(declarations) == 5, code
    assert all(line.startswith("globallycoherent ") for line in declarations), code
    assert "[" not in "".join(declarations)
    (tmp_path / "out.hlsl").write_text(code)
    run_tool(tmp_path, "dxc", "-T", "cs_6_0", "-E", "CSMain", "out.hlsl")


def test_glsl_output_writes_coherent_without_the_vulkan_memory_model(tmp_path):
    code = translate(tmp_path, ATTRIBUTES, "opengl")

    for name in "bdns":
        assert f"coherent buffer {name}Buffer {{ uint {name}[]; }};" in code, code
    assert not any(qualifier in code for qualifier in SCOPED_NAMES)
    spirv = glslang_disassembly(tmp_path, code, "out")
    assert len(re.findall(r"OpMemberDecorate %\w+ 0 Coherent", spirv)) == 4


def test_glsl_round_trip_keeps_scoped_coherence_as_coherent(tmp_path):
    crossgl = translate(tmp_path, GLSL_BLOCKS, "cgl", suffix=".comp")
    assert "devicecoherent buffer D {" in crossgl
    assert "subgroupcoherent buffer S {" in crossgl
    # The block header carries the qualifier once; members do not repeat it.
    assert not re.search(r"uint \w+(?:\[\])? @\w+coherent;", crossgl), crossgl

    code = translate(tmp_path, crossgl, "opengl")
    assert len(re.findall(r"\bcoherent buffer [BDNS] \{", code)) == 4, code
    assert "coherent uniform uimage2D img;" in code
    spirv = glslang_disassembly(tmp_path, code, "out")
    assert "OpDecorate %img Coherent" in spirv


def test_glsl_round_trip_keeps_scopes_under_the_vulkan_memory_model(tmp_path):
    crossgl = translate(tmp_path, GLSL_VULKAN_MEMORY_MODEL, "cgl", suffix=".comp")
    code = translate(tmp_path, crossgl, "opengl")

    assert "#pragma use_vulkan_memory_model" in code
    for qualifier, block in zip(SCOPED_NAMES[:4], "BDNS"):
        assert f"{qualifier} buffer {block} {{" in code, code
    assert "queuefamilycoherent uniform uimage2D img;" in code

    options = ("--target-env", "vulkan1.3")
    source = glslang_disassembly(tmp_path, GLSL_VULKAN_MEMORY_MODEL, "src", *options)
    output = glslang_disassembly(tmp_path, code, "out", *options)
    operands = r"MakePointer(?:Available|Visible)|NonPrivate(?:Pointer|Texel)"
    assert "OpMemoryModel Logical Vulkan" in output
    assert sorted(re.findall(operands, output)) == sorted(re.findall(operands, source))


def test_spirv_output_decorates_scoped_resources_coherent(tmp_path):
    code = translate(tmp_path, ATTRIBUTES, "vulkan")
    ids = spirv_ids(code)

    for name in "bdns":
        assert f"OpMemberDecorate %{ids[name + 'Buffer']} 0 Coherent" in code, code
    (tmp_path / "out.spvasm").write_text(code)
    run_tool(
        tmp_path, "spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"
    )
    run_tool(tmp_path, "spirv-val", "--target-env", "vulkan1.1", "out.spv")


def test_spirv_output_of_imported_glsl_blocks_is_coherent(tmp_path):
    crossgl = translate(tmp_path, GLSL_BLOCKS, "cgl", suffix=".comp")
    code = translate(tmp_path, crossgl, "vulkan")
    ids = spirv_ids(code)

    assert f"OpDecorate %{ids['img']} Coherent" in code, code
    for block in "BDN":
        assert f"OpMemberDecorate %{ids[block]} 0 Coherent" in code, code
    assert f"OpMemberDecorate %{ids['S']} 1 Coherent" in code, code


def test_metal_output_scopes_buffer_coherence(tmp_path):
    code = translate(tmp_path, ATTRIBUTES, "metal")

    for name, scope in zip("bdns", ("threadgroup", "device", "device", "threadgroup")):
        assert re.search(
            rf"coherent\({scope}\) device uint\s*\*\s*{name} \[\[buffer\(\d\)\]\]",
            code,
        ), code
    (tmp_path / "out.metal").write_text(code)
    run_tool(
        tmp_path,
        "xcrun",
        "-sdk",
        "macosx",
        "metal",
        "-std=metal3.2",
        "-c",
        "out.metal",
        "-o",
        "out.air",
    )


def test_metal_output_writes_one_coherence_qualifier(tmp_path):
    code = translate(tmp_path, MIXED, "metal")

    assert re.search(r"\bcoherent device uint\s*\*\s*b \[\[buffer\(0\)\]\]", code), code
    assert "coherent(" not in code

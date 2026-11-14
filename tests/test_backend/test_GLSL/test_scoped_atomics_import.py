"""GL_KHR_memory_scope_semantics atomics import as CrossGL's relaxed atomics, or refuse.

The importer passed the scope and semantics trailer through, so OpenGL and DirectX refused the arity, SPIR-V named
``gl_ScopeDevice`` as an unknown variable and Metal dropped the trailer unread. Relaxed calls at device scope or
narrower now import as the plain built-ins; acquire, release, availability, visibility, volatile, cross-device and
non-constant operands, and image loads and stores, raise a typed diagnostic.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.GLSL.scoped_atomics import ScopedAtomicUnsupportedError
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

HEADER = """#version 450
#extension GL_KHR_memory_scope_semantics : require
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer B { uint head; int tail; uint data[]; } b;
layout(binding = 1, r32ui) uniform uimage2D img;
shared uint g;
"""

READ_MODIFY_WRITE = """
    uint a = atomicAdd(b.head, 1u, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
    int c = atomicCompSwap(b.tail, 0, 1, gl_ScopeWorkgroup, gl_StorageSemanticsNone, gl_SemanticsRelaxed,
                           gl_StorageSemanticsNone, gl_SemanticsRelaxed);
    uint d = atomicOr(g, 2u, gl_ScopeWorkgroup, gl_StorageSemanticsShared, gl_SemanticsRelaxed);
    uint e = imageAtomicMax(img, ivec2(gl_GlobalInvocationID.xy), 3u, gl_ScopeDevice,
                            gl_StorageSemanticsImage, gl_SemanticsRelaxed);
    uint f = atomicExchange(b.head, 4u, gl_ScopeSubgroup,
                            gl_StorageSemanticsBuffer | gl_StorageSemanticsShared, 0);
    b.data[gl_GlobalInvocationID.x] = a + uint(c) + d + e + f;
"""

LOADS_AND_STORES = """
    atomicStore(g, 0u, gl_ScopeWorkgroup, gl_StorageSemanticsShared, gl_SemanticsRelaxed);
    uint a = atomicLoad(b.head, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
    atomicStore(b.tail, int(a), gl_ScopeQueueFamily, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
    b.data[gl_GlobalInvocationID.x] = a + atomicLoad(g, gl_ScopeInvocation, gl_StorageSemanticsNone, 0u);
"""

PLAIN_CALLS = [
    "atomicAdd(b.head, 1u)",
    "atomicCompSwap(b.tail, 0, 1)",
    "atomicOr(g, 2u)",
    "imageAtomicMax(img, ivec2(gl_GlobalInvocationID.xy), 3u)",
    "atomicExchange(b.head, 4u)",
]


def _source(tmp_path, body, helpers=""):
    path = tmp_path / "shader.comp"
    path.write_text(
        HEADER + helpers + "\nvoid main() {" + body + "}\n", encoding="utf-8"
    )
    return path


def _translate(tmp_path, backend, body, helpers=""):
    path = _source(tmp_path, body, helpers)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def _glslang(tmp_path, code, stage="comp"):
    if shutil.which("glslangValidator") is None:
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / f"out.{stage}"
    path.write_text(code, encoding="utf-8")
    _run(["glslangValidator", str(path)])


def _spirv_val(tmp_path, assembly):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    path = tmp_path / "shader.spvasm"
    path.write_text(assembly, encoding="utf-8")
    binary = tmp_path / "shader.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", str(path), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def _dxc(tmp_path, code):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    path = tmp_path / "shader.hlsl"
    path.write_text(code, encoding="utf-8")
    entry = re.search(r"\]\s*void\s+(\w+)\s*\(", code).group(1)
    _run([dxc, "-T", "cs_6_0", "-E", entry, str(path), "-Fo", str(tmp_path / "o")])


def _metal(tmp_path, code):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not on PATH")
    probe = subprocess.run(
        ["xcrun", "--find", "metal"], capture_output=True, text=True, check=False
    )
    if probe.returncode != 0:
        pytest.skip("the Metal compiler is not installed")
    path = tmp_path / "shader.metal"
    path.write_text(code, encoding="utf-8")
    _run(["xcrun", "metal", "-c", str(path), "-o", str(tmp_path / "shader.air")])


def test_crossgl_import_drops_relaxed_scope_and_semantics(tmp_path):
    flat = _flat(_translate(tmp_path, "cgl", READ_MODIFY_WRITE))
    for call in PLAIN_CALLS:
        assert call in flat, call
    assert "gl_Scope" not in flat
    assert "gl_StorageSemantics" not in flat
    assert "gl_Semantics" not in flat


def test_crossgl_import_maps_relaxed_loads_and_stores(tmp_path):
    flat = _flat(_translate(tmp_path, "cgl", LOADS_AND_STORES))
    assert "atomicStore(g, 0u);" in flat
    assert "atomicLoad(b.head)" in flat
    assert "atomicStore(b.tail, int(a));" in flat
    assert "atomicLoad(g)" in flat
    assert "gl_Scope" not in flat


def test_opengl_output_uses_plain_atomics_and_compiles(tmp_path):
    code = _translate(tmp_path, "opengl", READ_MODIFY_WRITE)
    flat = _flat(code)
    for call in PLAIN_CALLS:
        assert call in flat, call
    assert "gl_Scope" not in flat
    _glslang(tmp_path, code)


def test_opengl_output_lowers_loads_and_stores_and_compiles(tmp_path):
    code = _translate(tmp_path, "opengl", LOADS_AND_STORES)
    flat = _flat(code)
    assert "atomicExchange(g, 0u);" in flat
    assert "atomicOr(b.head, 0u)" in flat
    assert "gl_Scope" not in flat
    _glslang(tmp_path, code)


def test_spirv_output_uses_device_or_workgroup_scope_and_validates(tmp_path):
    assembly = _translate(tmp_path, "vulkan", READ_MODIFY_WRITE)
    opcodes = re.findall(r"= (OpAtomic\w+)", assembly)
    assert sorted(opcodes) == sorted(
        [
            "OpAtomicIAdd",
            "OpAtomicCompareExchange",
            "OpAtomicOr",
            "OpAtomicUMax",
            "OpAtomicExchange",
        ]
    )
    assert "gl_Scope" not in assembly
    _spirv_val(tmp_path, assembly)


@pytest.mark.parametrize(
    "body, functions",
    [
        (
            READ_MODIFY_WRITE,
            ("atomic_fetch_add_explicit(", "atomic_fetch_or_explicit("),
        ),
        (LOADS_AND_STORES, ("atomic_load_explicit(", "atomic_store_explicit(")),
    ],
    ids=["read-modify-write", "load-store"],
)
def test_metal_output_uses_relaxed_atomics_and_compiles(tmp_path, body, functions):
    code = _translate(tmp_path, "metal", body)
    flat = _flat(code)
    for function in functions:
        assert function in flat, function
    assert "memory_order_relaxed" in flat
    for name in ("acquire", "release", "acq_rel", "seq_cst"):
        assert f"memory_order_{name}" not in flat
    assert "gl_Scope" not in flat
    _metal(tmp_path, code)


def test_directx_output_uses_interlocked_operations_and_compiles(tmp_path):
    code = _translate(tmp_path, "directx", READ_MODIFY_WRITE)
    flat = _flat(code)
    assert "InterlockedAdd(" in flat
    assert "InterlockedCompareExchange(" in flat
    assert "InterlockedOr(g, 2u," in flat
    assert "InterlockedMax(" in flat
    assert "InterlockedExchange(" in flat
    assert "gl_Scope" not in flat
    _dxc(tmp_path, code)


def test_a_user_function_with_the_scoped_arity_shadows_the_built_in(tmp_path):
    helpers = (
        "uint atomicAdd(uint x, uint y, int s, int t, int u) "
        "{ return x + y + uint(s + t + u); }\n"
    )
    body = "b.head = atomicAdd(1u, 2u, 3, 4, 5);"
    flat = _flat(_translate(tmp_path, "cgl", body, helpers))
    assert "atomicAdd(1u, 2u, 3, 4, 5)" in flat


@pytest.mark.parametrize(
    "body, reason",
    [
        (
            "atomicAdd(b.head, 1u, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);",
            "memory semantics gl_SemanticsAcquire have no CrossGL form",
        ),
        (
            "atomicStore(b.head, 1u, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);",
            "memory semantics gl_SemanticsRelease have no CrossGL form",
        ),
        (
            "b.head = atomicLoad(b.head, gl_ScopeDevice, gl_StorageSemanticsBuffer, "
            "gl_SemanticsAcquire | gl_SemanticsMakeVisible);",
            "memory semantics gl_SemanticsAcquire | gl_SemanticsMakeVisible have no CrossGL form",
        ),
        (
            "atomicOr(g, 1u, gl_ScopeWorkgroup, gl_StorageSemanticsShared, gl_SemanticsVolatile);",
            "memory semantics gl_SemanticsVolatile have no CrossGL form",
        ),
        (
            "atomicCompSwap(b.tail, 0, 1, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed, "
            "gl_StorageSemanticsBuffer, gl_SemanticsAcquireRelease);",
            "memory semantics gl_SemanticsAcquireRelease have no CrossGL form",
        ),
        (
            "atomicAdd(b.head, 1u, gl_ScopeCrossDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);",
            "gl_ScopeCrossDevice is wider than any target's atomics",
        ),
        (
            "int s = 1; atomicAdd(b.head, 1u, s, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);",
            "the memory scope is not a gl_Scope* constant",
        ),
        (
            "int m = 0; atomicAdd(b.head, 1u, gl_ScopeDevice, gl_StorageSemanticsBuffer, m);",
            "must be gl_StorageSemantics* and gl_Semantics* constants",
        ),
        (
            "atomicAdd(b.head, 1u, 9, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);",
            "the memory scope is not a gl_Scope* constant",
        ),
        (
            "b.head = imageAtomicLoad(img, ivec2(0), gl_ScopeDevice, gl_StorageSemanticsImage, "
            "gl_SemanticsRelaxed);",
            "CrossGL has no atomic image load or store",
        ),
        (
            "imageAtomicStore(img, ivec2(0), 1u, gl_ScopeDevice, gl_StorageSemanticsImage, "
            "gl_SemanticsRelaxed);",
            "CrossGL has no atomic image load or store",
        ),
    ],
    ids=[
        "acquire",
        "release",
        "acquire-make-visible",
        "volatile",
        "compswap-unequal-acq-rel",
        "cross-device",
        "variable-scope",
        "variable-semantics",
        "unknown-scope",
        "image-load",
        "image-store",
    ],
)
@pytest.mark.parametrize("backend", ["cgl", "opengl", "vulkan", "metal", "directx"])
def test_scoped_atomics_without_a_crossgl_form_are_refused(
    tmp_path, backend, body, reason
):
    with pytest.raises(ScopedAtomicUnsupportedError, match=re.escape(reason)) as info:
        _translate(tmp_path, backend, body)
    assert info.value.code.endswith("opengl-import-scoped-atomic-unsupported")
    assert info.value.missing_capabilities[0].startswith("opengl.import.")


METAL_EXTRA_OPERAND = """shader main {
    struct B { uint head; };
    B b @glsl_buffer_block(std430) @binding(0);
    compute {
    layout(local_size_x = 64) in;
    void main() {
        uint v = atomicAdd(b.head, 1u, 2u);
        b.head = v;
    }
    }
}
"""


def test_metal_refuses_extra_buffer_block_atomic_operands():
    ast = CrossGLParser(CrossGLLexer(METAL_EXTRA_OPERAND).get_tokens()).parse()
    with pytest.raises(Exception, match=r"requires 2 argument\(s\), got 3"):
        MetalCodeGen().generate(ast)

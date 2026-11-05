"""Relaxed atomic loads and stores reach SPIR-V and DirectX on buffers and group-shared memory.

SPIR-V had no lowering for ``atomicLoad`` and ``atomicStore`` at all; they now become ``OpAtomicLoad`` and
``OpAtomicStore`` with relaxed semantics, at ``Device`` scope on storage buffers and ``Workgroup`` scope on
workgroup memory. DirectX refused ``atomicLoad`` on byte-address buffer members and printed atomics on group-shared
variables that the semantic pass left in expression form under their CrossGL names. Assigning an atomic result to a
GLSL buffer block member on DirectX printed the member's ``Load`` call as an assignment target or an ``out``
argument, which DXC rejects.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

PROGRAM = """shader main {
    struct B { uint head; int tail; uint data[]; };
    struct R { uint ro; };
    B b @glsl_buffer_block(std430) @binding(0);
    R r @glsl_buffer_block(std430) @binding(1) @readonly;
    shared uint g;
    shared int h[2];
    shared float gf;
    HELPERS
    compute {
    layout(local_size_x = 64) in;
    void main() {
        BODY
    }
    }
}
"""

LOADS_AND_STORES = """
        if (gl_LocalInvocationIndex == 0u) {
            atomicStore(g, 0u);
            atomicStore(h[1], 5);
        }
        barrier();
        uint a = atomicLoad(b.head);
        atomicStore(b.tail, int(a));
        int s = atomicLoad(h[1]);
        atomicExchange(g, a);
        b.data[gl_GlobalInvocationID.x] = a + uint(s) + atomicAdd(g, 1u) + atomicLoad(g);
"""

# Scope 1 is Device and scope 2 Workgroup (SPIR-V 1.6, section 3.27); semantics 0 is Relaxed (section 3.28).
DEVICE, WORKGROUP, RELAXED = "1", "2", "0"


def _translate(tmp_path, backend, body, helpers=""):
    path = tmp_path / "shader.cgl"
    path.write_text(
        PROGRAM.replace("HELPERS", helpers).replace("BODY", body), encoding="utf-8"
    )
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


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


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _spirv_atomics(assembly):
    """``(opcode, pointer, scope, semantics)`` of every load and store, with scope and semantics as numbers."""
    constants = dict(re.findall(r"(%\w+) = OpConstant %\w+ (\d+)", assembly))
    found = re.findall(r"= OpAtomicLoad %\w+ (%\w+) (%\w+) (%\w+)", assembly)
    loads = [("load", p, constants[s], constants[m]) for p, s, m in found]
    found = re.findall(r"OpAtomicStore (%\w+) (%\w+) (%\w+) %\w+", assembly)
    stores = [("store", p, constants[s], constants[m]) for p, s, m in found]
    return loads + stores


def _pointer_storage(assembly, pointer):
    """The storage class of the access chain or variable ``pointer``."""
    match = re.search(rf"{pointer} = Op(?:AccessChain|Variable) (%\w+)", assembly)
    pointer_type = match.group(1)
    return re.search(rf"{pointer_type} = OpTypePointer (\w+)", assembly).group(1)


def test_spirv_lowers_atomic_loads_and_stores_at_their_storage_scope(tmp_path):
    assembly = _translate(tmp_path, "vulkan", LOADS_AND_STORES)
    accesses = _spirv_atomics(assembly)
    assert sorted(kind for kind, *_ in accesses) == ["load"] * 3 + ["store"] * 3
    for _kind, pointer, scope, semantics in accesses:
        storage = _pointer_storage(assembly, pointer)
        assert scope == (WORKGROUP if storage == "Workgroup" else DEVICE), storage
        assert semantics == RELAXED
    # Storage buffers are Uniform-class BufferBlock variables before SPIR-V 1.3.
    storages = {_pointer_storage(assembly, pointer) for _k, pointer, *_ in accesses}
    assert "Workgroup" in storages and storages - {"Workgroup"} <= {
        "Uniform",
        "StorageBuffer",
    }
    assert len(storages) == 2
    _spirv_val(tmp_path, assembly)


def test_spirv_atomic_load_from_a_helper_reaches_workgroup_memory(tmp_path):
    assembly = _translate(
        tmp_path,
        "vulkan",
        "atomicStore(g, 2u); b.head = helper();",
        "uint helper() { return atomicLoad(g); }",
    )
    assert [kind for kind, *_ in _spirv_atomics(assembly)] == ["load", "store"]
    _spirv_val(tmp_path, assembly)


@pytest.mark.parametrize(
    "body, helpers, reason",
    [
        (
            "uint p = 0u; b.head = atomicLoad(p);",
            "",
            "atomicLoad requires a storage buffer or workgroup target",
        ),
        (
            "b.head = 0u; { uint g = 3u; b.head = atomicLoad(g); }",
            "",
            "atomicLoad requires a storage buffer or workgroup target",
        ),
        (
            "b.head = helper(1u);",
            "uint helper(uint g) { return atomicLoad(g); }",
            "atomicLoad requires an addressable target",
        ),
        (
            "atomicStore(gf, 1.0);",
            "",
            "atomicStore currently supports only int or uint",
        ),
        (
            "atomicStore(r.ro, 1u);",
            "",
            "atomicStore requires a writable storage buffer",
        ),
        ("atomicStore(g);", "", "atomicStore requires exactly a target and a value"),
    ],
    ids=[
        "private",
        "shadowing-local",
        "shadowing-parameter",
        "float",
        "readonly",
        "arity",
    ],
)
def test_spirv_refuses_atomic_loads_and_stores_it_cannot_express(
    tmp_path, body, helpers, reason
):
    with pytest.raises(Exception, match=re.escape(reason)):
        _translate(tmp_path, "vulkan", body, helpers)


def test_spirv_atomic_load_reads_a_readonly_storage_buffer(tmp_path):
    # OpAtomicLoad does not write, so a NonWritable storage buffer is a valid target, which spirv-val confirms.
    assembly = _translate(tmp_path, "vulkan", "b.head = atomicLoad(r.ro);")
    assert [kind for kind, *_ in _spirv_atomics(assembly)] == ["load"]
    _spirv_val(tmp_path, assembly)


def test_directx_lowers_group_shared_and_byte_address_loads_and_stores(tmp_path):
    code = _translate(tmp_path, "directx", LOADS_AND_STORES)
    flat = _flat(code)
    assert re.search(
        r"InterlockedExchange\(g, 0u, __crossgl_atomic_original_\d+\);", flat
    )
    assert re.search(
        r"InterlockedExchange\(h\[1\], 5, __crossgl_atomic_original_\d+\);", flat
    )
    assert "uint a = __crossgl_byteaddress_atomic_or_uint(b, 0, 0u);" in flat
    assert "__crossgl_byteaddress_atomic_exchange_int(b, 4, int(a));" in flat
    assert "InterlockedOr(h[1], 0, s);" in flat
    assert re.search(
        r"InterlockedExchange\(g, a, __crossgl_atomic_original_\d+\);", flat
    )
    assert re.search(r"InterlockedAdd\(g, 1u, __crossgl_atomic_expr_\d+\);", flat)
    assert re.search(r"InterlockedOr\(g, 0u, __crossgl_atomic_expr_\d+\);", flat)
    assert not re.search(r"\batomic(Load|Store|Add|Exchange)\(", code)
    assert not re.search(r"\.Load\([^;]*\)\s*[-+*/]?=", code)
    _dxc(tmp_path, code)


@pytest.mark.parametrize(
    "statement, store",
    [
        ("b.head = atomicAdd(g, 1u);", r"b\.Store\(0, (__crossgl_atomic_value_\d+)\);"),
        (
            "b.tail = int(atomicAdd(g, 1u));",
            r"b\.Store\(4, asuint\((__crossgl_atomic_value_\d+)\)\);",
        ),
        (
            "b.head += atomicLoad(g);",
            r"b\.Store\(0, \(b\.Load\(0\) \+ (__crossgl_atomic_value_\d+)\)\);",
        ),
        (
            "b.data[gl_GlobalInvocationID.x] = atomicLoad(g);",
            r"b\.Store\(\(8 \+ \(\w+\.x\) \* 4\), (__crossgl_atomic_value_\d+)\);",
        ),
        (
            "b.data[gl_GlobalInvocationID.x] = b.head + atomicAdd(g, 1u) + atomicLoad(g);",
            r"b\.Store\(\(8 \+ \(\w+\.x\) \* 4\), (__crossgl_atomic_value_\d+)\);",
        ),
        (
            "b.data[0] = b.head > 1u ? atomicLoad(g) : 2u;",
            r"b\.Store\(8, (__crossgl_atomic_value_\d+)\);",
        ),
    ],
    ids=["direct", "converted", "compound", "array", "nested", "conditional"],
)
def test_directx_stores_atomic_results_into_buffer_block_members(
    tmp_path, statement, store
):
    code = _translate(tmp_path, "directx", statement)
    flat = _flat(code)
    match = re.search(store, flat)
    assert match, flat[flat.index("void CSMain") :]
    temporary = match.group(1)
    # The temporary is declared and written by the atomic lowering before the store reads it.
    before = flat[: match.start()]
    assert f" {temporary};" in before
    assert re.search(
        rf"(Interlocked\w+\([^;]*\b{temporary}\)|\b{temporary} = )", before
    )
    assert not re.search(r"\.Load\([^;]*\)\s*[-+*/]?=", code)
    assert not re.search(r"Interlocked\w+\([^;]*\bb\.Load\(", code)
    _dxc(tmp_path, code)


def test_directx_buffer_block_atomic_store_keeps_the_readonly_refusal(tmp_path):
    with pytest.raises(Exception, match="readonly"):
        _translate(tmp_path, "directx", "r.ro = atomicAdd(g, 1u);")


@pytest.mark.parametrize(
    "body, helpers",
    [
        ("uint p = 0u; b.head = atomicLoad(p);", ""),
        ("b.head = 0u; { uint g = 3u; b.head = atomicLoad(g); }", ""),
        ("b.head = helper(1u);", "uint helper(uint g) { return atomicLoad(g); }"),
    ],
    ids=["private", "shadowing-local", "shadowing-parameter"],
)
def test_directx_refuses_atomic_loads_of_private_storage(tmp_path, body, helpers):
    with pytest.raises(ValueError, match="requires writable buffer or groupshared"):
        _translate(tmp_path, "directx", body, helpers)


def test_directx_refuses_interlocked_reads_of_readonly_byte_address_buffers(tmp_path):
    with pytest.raises(Exception, match="cannot read readonly ByteAddressBuffer"):
        _translate(tmp_path, "directx", "b.head = atomicLoad(r.ro);")


def test_directx_atomic_load_from_a_helper_reads_group_shared_memory(tmp_path):
    code = _translate(
        tmp_path,
        "directx",
        "atomicStore(g, 2u); b.head = helper();",
        "uint helper() { return atomicLoad(g); }",
    )
    assert re.search(r"InterlockedOr\(g, 0u, __crossgl_atomic_return_\d+\);", code)
    _dxc(tmp_path, code)


def test_shared_variable_scan_accepts_statement_list_bodies_and_foreign_nodes():
    """Group-shared discovery walks a bare statement-list body and ignores nodes from other ASTs."""
    from types import SimpleNamespace

    from crosstl.translator import ast as A
    from crosstl.translator.semantics.atomic_forms import shared_variable_names

    shared = A.VariableNode(
        "tile", A.PrimitiveType("float"), qualifiers=["groupshared"]
    )
    plain = A.VariableNode("value", A.PrimitiveType("float"))
    foreign = SimpleNamespace(name="ignored")
    function = SimpleNamespace(name="main", body=[shared, plain, foreign])
    program = SimpleNamespace(global_variables=[], stages={}, functions=[function])

    assert shared_variable_names(program, frozenset({"groupshared"})) == {"tile"}


def test_hlsl_codegen_accepts_a_native_metal_ast():
    """The HLSL generator still accepts a Metal front-end AST whose nodes are not CrossGL nodes."""
    from crosstl.backend.Metal.MetalLexer import MetalLexer
    from crosstl.backend.Metal.MetalParser import MetalParser
    from crosstl.translator.codegen.directx_codegen import HLSLCodeGen

    code = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void add(device float* a [[buffer(0)]], uint3 id [[thread_position_in_grid]]) {
        a[id.x] = a[id.x] + 1.0;
    }
    """
    ast = MetalParser(MetalLexer(code).tokenize()).parse()
    assert "RWStructuredBuffer<float> a" in HLSLCodeGen().generate(ast)

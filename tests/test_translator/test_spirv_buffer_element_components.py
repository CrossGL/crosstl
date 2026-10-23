"""Components of structured-buffer elements are addressed one by one.

``buf[i]`` on a structured buffer is the ``i``-th element (HLSL ``StructuredBuffer<T>``, a SPIR-V storage block's
runtime array). A further subscript applies to that element's value: ``buf[i][1]`` on a ``float4`` element is its
``y`` component (HLSL vector subscripts; GLSL 4.60 section 5.5), on a matrix it is a row in HLSL and a column in
CrossGL, and the SPIR-V emitter lowers it to one more ``OpAccessChain`` index on the element pointer (SPIR-V 1.6,
``OpAccessChain``). The emitter used to index the element pointer as if it were the buffer again, which gave a
``float4`` result for ``buf[i][1]`` and refused ``uint(buf[i][1])`` with "Constructor uint requires a scalar operand".

A write to some components of a vector in a buffer or in workgroup memory must not store the others: another
invocation may write them at the same time (Vulkan 1.3 appendix B, "Memory Model"). The SPIR-V emitter loaded the
vector, inserted the new components and stored the whole vector; it now stores each component through its own
pointer, as glslang and DXC do. One component is also addressable, so ``v.y`` is an atomic's memory, an
increment's operand and an ``inout`` argument.

The GLSL importer wrote an element of a converted storage block as ``buffer_load(b, i)``, a value, also where the
element is storage: ``b.v[i].y = x`` became ``buffer_load(b, i).y = x``, which DirectX emitted as
``b.Load(i).y = x`` and SPIR-V refused. Those positions now read ``b[i]``.

``spirv_buffer_components/vec4_component.spvasm`` is glslang 11's ``-V --target-env vulkan1.1`` output for the GLSL
at its top.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import crosstl
from tests.spirv_friendly_names import translate_spirv

FIXTURES = Path(__file__).parent / "spirv_buffer_components"

COMPONENT_HLSL = """
StructuredBuffer<float4> inb : register(t0);
RWStructuredBuffer<uint> outb : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outb[id.x] = (uint)inb[id.x][1] + (uint)inb[id.x][id.y & 3];
}
"""

STORE_HLSL = """
RWStructuredBuffer<float4> rw : register(u0);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    rw[id.x][2] = 5.0;
}
"""

MATRIX_HLSL = """
StructuredBuffer<float4x4> mb : register(t0);
RWStructuredBuffer<float> outb : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outb[id.x] = mb[id.x][3][0];
}
"""


def _translate(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source)
    return translate_spirv(path)


def _spirv_val(tmp_path, code):
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        return
    asm = tmp_path / "out.spvasm"
    asm.write_text(code)
    binary = tmp_path / "out.spv"
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def _element_pointers(code, buffer, element_type):
    """Ids of ``OpAccessChain`` results that select one element of ``buffer``'s runtime array."""
    pattern = rf"%(\w+) = OpAccessChain %_ptr_\w+_{element_type} %{buffer} %int_0 %\w+"
    return set(re.findall(pattern, code))


def _subscripts(code, base_ids, result_type):
    """``(base, index)`` pairs of access chains of type ``result_type`` taken on one of ``base_ids``."""
    pattern = rf"%\w+ = OpAccessChain %_ptr_\w+_{result_type} %(\w+) %(\w+)\n"
    return [
        (base, index) for base, index in re.findall(pattern, code) if base in base_ids
    ]


def test_hlsl_vector_element_subscripts_select_one_component(tmp_path):
    code = _translate(tmp_path, COMPONENT_HLSL, "component.hlsl")

    elements = _element_pointers(code, "inb", "v4float")
    subscripts = _subscripts(code, elements, "float")
    assert len(subscripts) == 2, code
    assert subscripts[0][1] == "int_1"
    assert not subscripts[1][1].startswith("int_"), "the dynamic index is an operand"
    assert "OpConvertFToU %uint" in code
    _spirv_val(tmp_path, code)


def test_hlsl_vector_element_component_store_writes_one_component(tmp_path):
    code = _translate(tmp_path, STORE_HLSL, "store.hlsl")

    elements = _element_pointers(code, "rw", "v4float")
    subscripts = _subscripts(code, elements, "float")
    assert [index for _, index in subscripts] == ["int_2"], code
    component = re.search(
        rf"%(\w+) = OpAccessChain %_ptr_\w+_float %{subscripts[0][0]} %int_2", code
    ).group(1)
    assert re.search(rf"OpStore %{component} %float_5\b", code), code
    _spirv_val(tmp_path, code)


def test_hlsl_matrix_element_subscripts_select_a_column_component(tmp_path):
    code = _translate(tmp_path, MATRIX_HLSL, "matrix.hlsl")

    elements = _element_pointers(code, "mb", "mat4v4float")
    columns = _subscripts(code, elements, "v4float")
    # HLSL ``m[3][0]`` is row 3, column 0: CrossGL column 0, component 3.
    assert columns and {index for _, index in columns} == {"int_0"}, code
    column_ids = {
        match
        for base, _ in columns
        for match in re.findall(
            rf"%(\w+) = OpAccessChain %_ptr_\w+_v4float %{base} %int_0", code
        )
    }
    assert {index for _, index in _subscripts(code, column_ids, "float")} == {
        "int_3"
    }, code
    _spirv_val(tmp_path, code)


def test_spirv_storage_block_vector_component_round_trips_to_spirv(tmp_path):
    source = (FIXTURES / "vec4_component.spvasm").read_text()
    code = _translate(tmp_path, source, "component.spvasm")

    elements = _element_pointers(code, "inb", "v4float")
    assert [index for _, index in _subscripts(code, elements, "float")] == [
        "int_1"
    ], code
    _spirv_val(tmp_path, code)


@pytest.mark.parametrize("backend", ["directx", "metal", "opengl"])
def test_spirv_storage_block_vector_component_reaches_every_emitter(tmp_path, backend):
    path = tmp_path / "component.spvasm"
    path.write_text((FIXTURES / "vec4_component.spvasm").read_text())
    code = crosstl.translate(str(path), backend=backend)

    assert re.search(r"\binb\[i\]\[1\]", code), code


# Each invocation writes one component of a vector other invocations write too.
SHARED_COMPONENT_HLSL = """
RWStructuredBuffer<float4> rw : register(u0);
RWStructuredBuffer<int4> ri : register(u1);
groupshared float4 gs[64];
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    rw[id.x].y = 2.0;
    rw[id.x].zw = float2(3.0, 4.0);
    rw[id.x].x += 1.0;
    rw[id.x].yz *= float2(2.0, 0.5);
    ri[id.x].w++;
    gs[gi].x = 1.0;
    GroupMemoryBarrierWithGroupSync();
    rw[id.x].w -= gs[(gi + 1) % 64].x;
    float4 p = float4(0.0, 0.0, 0.0, 0.0);
    p.x = 7.0;
    p.yz += float2(1.0, 2.0);
    ri[id.x].x = int(p.x + p.y);
}
"""

COMPONENT_REFERENCE_HLSL = """
RWStructuredBuffer<uint4> b : register(u0);
groupshared uint4 g[4];
void bump(inout uint x) { x += 1u; }
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    uint old;
    InterlockedAdd(b[i].y, 1u, old);
    b[i].z++;
    bump(b[i].w);
    InterlockedAdd(g[i & 3].x, 1u);
    g[i & 3].y++;
    GroupMemoryBarrierWithGroupSync();
    b[i].x = old + g[0].x;
}
"""

GLSL_ELEMENT_REFERENCES = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer B { uvec4 v[]; } b;
layout(std430, binding = 1) buffer C { uint c[]; } cc;
void bump(inout uint x) { x += 1u; }
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint whole = cc.c[i];
    atomicAdd(cc.c[i], 1u);
    atomicAdd(b.v[i].y, whole);
    b.v[i].z++;
    ++b.v[i].x;
    b.v[i][3] = b.v[i].y;
    b.v[i].w += 2u;
    bump(b.v[i].w);
    bump(cc.c[i]);
    cc.c[i]++;
}
"""


def _vector_stores(code, storage_classes):
    """``OpStore`` instructions whose pointer points to a vector in one of ``storage_classes``."""
    vector_pointers = set()
    for name, storage in re.findall(
        r"%(\w+) = OpAccessChain %_ptr_(\w+?)_v\d\w+ ", code
    ):
        if storage in storage_classes:
            vector_pointers.add(name)
    return [
        line
        for line in code.splitlines()
        if (match := re.match(r"\s*OpStore %(\w+) ", line))
        and match.group(1) in vector_pointers
    ]


def test_partial_vector_stores_to_shared_storage_write_only_their_components(tmp_path):
    code = _translate(tmp_path, SHARED_COMPONENT_HLSL, "shared.hlsl")

    assert _vector_stores(code, {"Uniform", "StorageBuffer", "Workgroup"}) == [], code
    component_stores = re.findall(
        r"OpAccessChain %_ptr_(?:Uniform|Workgroup)_(?:float|int) %\w+ %int_(\d)", code
    )
    # .y, .zw, .x +=, .yz *=, .w++, gs.x, .w -=, ri.x: one pointer per component (loads and stores).
    assert set(component_stores) == {"0", "1", "2", "3"}, code
    # A vector in invocation-private storage is still updated as one value.
    assert "OpCompositeInsert %v4float %float_7" in code, code
    _spirv_val(tmp_path, code)


def test_vector_components_are_atomic_increment_and_inout_targets(tmp_path):
    code = _translate(tmp_path, COMPONENT_REFERENCE_HLSL, "reference.hlsl")

    atomic_pointers = re.findall(r"OpAtomicIAdd %uint %(\w+) ", code)
    assert len(atomic_pointers) == 2, code
    for pointer in atomic_pointers:
        assert re.search(
            rf"%{pointer} = OpAccessChain %_ptr_(?:Uniform|Workgroup)_uint %\w+ %int_[01]\n",
            code,
        ), code
    assert _vector_stores(code, {"Uniform", "Workgroup"}) == [], code
    _spirv_val(tmp_path, code)


def test_glsl_storage_block_elements_are_written_through_subscripts(tmp_path):
    path = tmp_path / "references.comp"
    path.write_text(GLSL_ELEMENT_REFERENCES)
    crossgl = crosstl.translate(str(path), backend="cgl")

    for expected in (
        "uint whole = buffer_load(cc, i);",
        "atomicAdd(cc[i], 1u);",
        "atomicAdd(b[i].y, whole);",
        "(b[i].z++);",
        "(++b[i].x);",
        "b[i][3] = b[i].y;",
        "b[i].w += 2u;",
        "bump(b[i].w);",
        "bump(cc[i]);",
        "(cc[i]++);",
    ):
        assert expected in crossgl, crossgl
    assert not re.search(r"buffer_load\([^;]*\)\s*[.\[]", crossgl), crossgl
    assert "atomicAdd(buffer_load" not in crossgl, crossgl


@pytest.mark.parametrize("backend", ["vulkan", "directx", "metal", "opengl"])
def test_glsl_storage_block_element_references_reach_every_emitter(tmp_path, backend):
    path = tmp_path / "references.comp"
    path.write_text(GLSL_ELEMENT_REFERENCES)
    if backend == "vulkan":
        code = translate_spirv(path)
    else:
        code = crosstl.translate(str(path), backend=backend)

    # A loaded element is a value: nothing may select from, increment or assign to it.
    assert not re.search(r"\.Load\([^;]*\)\s*(\.|\[|\+\+|--)", code), code
    if backend == "vulkan":
        assert _vector_stores(code, {"Uniform", "StorageBuffer"}) == [], code
        assert len(re.findall(r"OpAtomicIAdd", code)) == 2, code
        _spirv_val(tmp_path, code)
    elif backend == "opengl" and shutil.which("glslangValidator"):
        out = tmp_path / "out.comp"
        out.write_text(code)
        result = subprocess.run(
            ["glslangValidator", "-S", "comp", str(out)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr

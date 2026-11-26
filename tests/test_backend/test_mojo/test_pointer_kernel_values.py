"""Mojo pointer kernels: what the importer maps, and what it refuses instead of passing Mojo text to CrossGL.

``stack_allocation[N, T, address_space=AddressSpace.SHARED]()`` is the block's workgroup array
(crosstl/backend/Mojo/mojo_stack_allocations.py); ``Array[T, N]``/``InlineArray[T, N]`` are CrossGL ``T[N]`` and
``value.copy()`` is the value (crosstl/backend/Mojo/mojo_arrays.py, pointer_kernels.py). A method of a local value, a
parameterized call and any other ``stack_allocation`` have no CrossGL form and are refused by name.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.backend.import_refusals import MojoImportRefused

_HEADER = """from max.gpu import global_idx, barrier
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace

"""


def _import(tmp_path, body, prelude=""):
    path = tmp_path / "kernel.mojo"
    path.write_text(
        _HEADER
        + prelude
        + "def probe(fin: UnsafePointer[Float32, MutAnyOrigin], outp: UnsafePointer[Float32, MutAnyOrigin]):\n"
        + "    var i: Int = global_idx.x\n"
        + body
    )
    return crosstl.translate(str(path), backend="cgl")


def _onward(tmp_path, crossgl, backend):
    path = tmp_path / "imported.cgl"
    path.write_text(crossgl)
    return crosstl.translate(str(path), backend=backend)


def _flat(code):
    return " ".join(code.split())


_SHARED = """    var tile = stack_allocation[2, Float32, address_space=AddressSpace.SHARED]()
    tile[0] = fin[i]
    barrier()
    outp[i] = tile[0]
"""


def test_shared_stack_allocation_is_a_workgroup_array(tmp_path):
    crossgl = _import(tmp_path, _SHARED)
    assert "var<workgroup> tile: array<float, 2>;" in crossgl
    assert "stack_allocation" not in crossgl

    mojo = _flat(_onward(tmp_path, crossgl, "mojo"))
    assert (
        "var tile = stack_allocation[2, Float32, address_space=AddressSpace.SHARED]()"
        in mojo
    )
    hlsl = _flat(_onward(tmp_path, crossgl, "directx"))
    assert "groupshared float probe_tile[2];" in hlsl


def test_shared_stack_allocation_of_vectors_and_structures(tmp_path):
    prelude = (
        "@fieldwise_init\nstruct Pair(TrivialRegisterPassable):\n"
        "    var a: Float32\n    var b: Float32\n\n"
    )
    body = (
        "    var pairs = stack_allocation[2, Pair, address_space=AddressSpace.SHARED]()\n"
        "    var lanes = stack_allocation[2, SIMD[DType.float32, 4], address_space=AddressSpace.SHARED]()\n"
        "    pairs[0] = Pair(fin[i], 1.0)\n"
        "    lanes[0] = SIMD[DType.float32, 4](fin[i])\n"
        "    barrier()\n"
        "    outp[i] = pairs[0].a + lanes[0][1]\n"
    )
    crossgl = _import(tmp_path, body, prelude)
    assert "var<workgroup> pairs: array<Pair, 2>;" in crossgl
    assert "var<workgroup> lanes: array<vec4, 2>;" in crossgl


@pytest.mark.parametrize(
    "allocation",
    [
        "stack_allocation[2, Float32]()",
        "stack_allocation[2, Float32, address_space=AddressSpace.LOCAL]()",
        "stack_allocation[2, Float32, alignment=16]()",
    ],
)
def test_other_stack_allocations_are_refused(tmp_path, allocation):
    body = f"    var tile = {allocation}\n    tile[0] = fin[i]\n    outp[i] = tile[0]\n"
    with pytest.raises(
        MojoImportRefused, match="only a stack_allocation in AddressSpace.SHARED"
    ):
        _import(tmp_path, body)


def test_copy_is_the_value_and_array_types_are_crossgl_arrays(tmp_path):
    prelude = (
        "@fieldwise_init\nstruct Pair(Copyable, Movable):\n"
        "    var a: Float32\n    var b: Float32\n\n"
    )
    body = (
        "    var p = Pair(fin[i], 2.0)\n"
        "    var q = p.copy()\n"
        "    q.a = 9.0\n"
        "    var values: InlineArray[Float32, 3] = [1.0, 2.0, 4.0]\n"
        "    outp[i] = p.a + q.a + q.b + values[Int(i) % 3]\n"
    )
    crossgl = _import(tmp_path, body, prelude)
    assert "var q = p;" in crossgl
    assert "float[3] values = {1.0, 2.0, 4.0};" in crossgl
    assert "copy" not in crossgl
    assert "InlineArray" not in crossgl


def test_a_method_of_a_local_structure_is_refused(tmp_path):
    prelude = (
        "@fieldwise_init\nstruct Span(Copyable, Movable):\n"
        "    var lo: Float32\n    var hi: Float32\n\n"
        "    def width(self) -> Float32:\n        return self.hi - self.lo\n\n"
    )
    body = "    var s = Span(fin[i], 0.0)\n    outp[i] = s.width()\n"
    with pytest.raises(MojoImportRefused, match=r"method call 's\.width\(\.\.\.\)'"):
        _import(tmp_path, body, prelude)


def test_a_parameterized_call_is_refused(tmp_path):
    body = "    outp[i] = fin[i].cast[DType.float16]().cast[DType.float32]()\n"
    with pytest.raises(MojoImportRefused, match="parameterized Mojo call"):
        _import(tmp_path, body)

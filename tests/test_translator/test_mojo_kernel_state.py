"""Kernel state the Mojo generator writes without global variables, and real Mojo atomics.

Mojo 1.2 has no global variables. A private variable declared outside a function becomes a local of the kernel that
helpers take as a ``mut`` argument; a GPU index input variable becomes a local of each function that reads it; a
storage block holding one runtime-sized array becomes an ``UnsafePointer`` kernel argument. Atomic built-ins write
``std.atomic.Atomic`` operations on the target's address instead of a plain read-modify-write, which would race.
"""

from __future__ import annotations

import subprocess

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.mojo_codegen import MojoCodeGen
from crosstl.translator.refusals import MojoRefusal
from tests.test_translator.test_codegen.test_mojo_codegen import find_mojo_compiler

PRIVATE_GLOBALS = """
shader Private {
    buffer float* outp;
    float scale = 2.0;
    static int counter;
    void bump() { counter = counter + 1; }
    float scaled(float v) { return v * scale; }
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        float bias = 0.5;
        float shifted(float v) { return scaled(v) + bias; }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            bump();
            outp[i] = shifted(outp[i]) + float(counter);
        }
    }
}
"""

BUILTIN_INPUTS = """
shader Inputs {
    buffer uint* outp;
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        uint3 gid @input @gl_GlobalInvocationID;
        uint3 lid @input @gl_LocalInvocationID;
        uint lane() { return lid.x; }
        void main() { outp[gid.x] = lane(); }
    }
}
"""

STORAGE_BLOCKS = """
shader Blocks {
    struct Particle { vec4 pos; float mass; };
    layout(std430, binding = 1) readonly buffer Masses { float masses[]; } massBuf;
    layout(std430, binding = 2) buffer Particles { Particle items[]; } particles;
    float massAt(uint i) { return massBuf.masses[i]; }
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        layout(std430, binding = 0) buffer Data { float data[]; };
        void main() {
            uint i = gl_GlobalInvocationID.x;
            particles.items[i].mass = massAt(i);
            data[i] = particles.items[i].pos.x;
        }
    }
}
"""

ATOMICS = """
shader Atomics {
    buffer int* counts;
    buffer uint* flags;
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        shared int tile[64];
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int before = atomicAdd(counts[0], 1);
            atomicMax(counts[1], int(i));
            uint old = atomicOr(flags[0], 4u);
            uint swapped = atomicCompSwap(flags[1], 0u, i);
            atomicAdd(tile[0], before);
            counts[2] = int(old + swapped);
        }
    }
}
"""


def _generate(source):
    return MojoCodeGen().generate(parse(source))


def _build(tmp_path, code):
    mojo = find_mojo_compiler()
    source = tmp_path / "kernel.mojo"
    source.write_text(code)
    return subprocess.run(
        [
            mojo,
            "build",
            "--emit",
            "llvm",
            "--disable-warnings",
            str(source),
            "-o",
            str(tmp_path / "kernel.ll"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_private_globals_become_kernel_locals_passed_by_reference():
    code = _generate(PRIVATE_GLOBALS)

    assert "def bump(mut counter: Int32) -> None:" in code
    assert "def scaled(v: Float32, mut scale: Float32) -> Float32:" in code
    assert (
        "def shifted(v: Float32, mut scale: Float32, mut bias: Float32) -> Float32:"
        in code
    )
    assert (
        "def compute_main(outp: UnsafePointer[Float32, MutAnyOrigin]) -> None:" in code
    )
    assert "    var scale: Float32 = 2.0\n" in code
    assert "    var counter: Int32 = 0\n" in code
    assert "    var bias: Float32 = 0.5\n" in code
    assert "    bump(counter)\n" in code
    assert "shifted(outp[Int(i)], scale, bias)" in code
    # No module-scope variable: Mojo 1.2 rejects one.
    assert not [line for line in code.splitlines() if line.startswith("var ")]


def test_builtin_input_variables_read_gpu_indices():
    code = _generate(BUILTIN_INPUTS)

    assert "def lane() -> UInt32:" in code
    assert (
        "    var lid = SIMD[DType.uint32, 4](UInt32(thread_idx.x), UInt32(thread_idx.y), "
        "UInt32(thread_idx.z), UInt32(0))\n" in code
    )
    assert (
        "    var gid = SIMD[DType.uint32, 4](UInt32(global_idx.x), UInt32(global_idx.y), "
        "UInt32(global_idx.z), UInt32(0))\n" in code
    )
    assert not [line for line in code.splitlines() if line.startswith("var ")]


def test_single_array_storage_blocks_become_pointer_arguments():
    code = _generate(STORAGE_BLOCKS)

    assert (
        "def massAt(i: UInt32, massBuf: UnsafePointer[Float32, MutAnyOrigin]) -> Float32:"
        in code
    )
    assert "    return massBuf[Int(i)]\n" in code
    assert (
        "def compute_main(massBuf: UnsafePointer[Float32, MutAnyOrigin], "
        "particles: UnsafePointer[Particle, MutAnyOrigin], "
        "data_: UnsafePointer[Float32, MutAnyOrigin]) -> None:" in code
    )
    assert "    particles[Int(i)].mass = massAt(i, massBuf)\n" in code
    assert "    data_[Int(i)] = particles[Int(i)].pos[0]\n" in code
    # The blocks' structures name nothing once the blocks are pointers.
    assert "struct Masses" not in code
    assert "struct Particles" not in code
    assert "struct Data" not in code
    assert "struct Particle(" in code


def test_readonly_single_array_storage_block_keeps_its_access_check():
    source = STORAGE_BLOCKS.replace(
        "return massBuf.masses[i];", "massBuf.masses[i] = 1.0; return 0.0;"
    )
    with pytest.raises(ValueError, match="readonly"):
        _generate(source)


def test_storage_block_with_other_members_is_refused():
    source = """
    shader Counted {
        layout(std430, binding = 0) buffer Values { uint count; float values[]; } vals;
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() { vals.values[0] = float(vals.count); }
        }
    }
    """
    with pytest.raises(
        MojoRefusal,
        match=r"storage block declared outside a function \(vals: Values\)",
    ):
        _generate(source)


def test_atomics_write_mojo_atomic_operations_on_the_target_address():
    code = _generate(ATOMICS)

    assert "from std.atomic import Atomic\n" in code
    assert (
        "var __cgl_atomic_0: Int32 = Atomic[Int32].fetch_add("
        "UnsafePointer(to=counts[0]), 1)" in code
    )
    # Mojo's Atomic.max returns nothing and Atomic has no fetch_or, so these retry a compare-and-exchange.
    assert "var __cgl_atomic_1_ptr = UnsafePointer(to=counts[1])" in code
    assert (
        "while not Atomic[Int32].compare_exchange(__cgl_atomic_1_ptr, __cgl_atomic_1, "
        "max(__cgl_atomic_1, __cgl_atomic_1_value)):" in code
    )
    assert (
        "while not Atomic[UInt32].compare_exchange(__cgl_atomic_2_ptr, __cgl_atomic_2, "
        "(__cgl_atomic_2 | __cgl_atomic_2_value)):" in code
    )
    assert "var __cgl_atomic_3: UInt32 = 0" in code
    assert (
        "_ = Atomic[UInt32].compare_exchange(UnsafePointer(to=flags[1]), __cgl_atomic_3, i)"
        in code
    )
    assert "Atomic[Int32].fetch_add(UnsafePointer(to=tile[0]), before)" in code
    # A plain read-modify-write of the target would race between invocations.
    assert "counts[0] = (__cgl_atomic_0" not in code


@pytest.mark.parametrize(
    "source", [PRIVATE_GLOBALS, BUILTIN_INPUTS, STORAGE_BLOCKS, ATOMICS]
)
def test_kernel_state_compiles_with_mojo(tmp_path, source):
    result = _build(tmp_path, _generate(source))

    assert result.returncode == 0, result.stderr

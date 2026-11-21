"""Mojo 1.2 kernel parameters, generic struct parameters and scalar conversions written by the Mojo generator.

A ``var<storage>`` entry parameter holding a runtime-sized array is the buffer itself, so it becomes an
``UnsafePointer`` kernel argument, like a module-scope ``buffer``; a writable storage parameter of any other type
would be copied into the kernel and lose its stores, so it is refused. A struct type parameter is written
``Self.T`` inside the struct, as Mojo 1.2 requires, and is bounded by ``TrivialRegisterPassable`` because the
struct is. A scalar constructor converts its argument once.
"""

from __future__ import annotations

import subprocess

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.mojo_codegen import MojoCodeGen
from crosstl.translator.refusals import MojoRefusal
from tests.test_translator.test_codegen.test_mojo_codegen import find_mojo_compiler

STORAGE_PARAMETERS = """
@compute
fn saxpy(
    @group(0) @binding(0) var<storage, read_write> dst: array<f32>,
    @group(0) @binding(1) var<storage, read> x: array<f32>,
    f32 a
) {
    var gid: i32 = gl_GlobalInvocationID.x;
    dst[gid] = a * x[gid];
}
"""

WRITABLE_STORAGE_STRUCT = """
struct Counter {
    u32 value;
};

@compute
fn count(@group(0) @binding(0) var<storage, read_write> counter: Counter) {
    counter.value = counter.value + 1u;
}
"""

GENERIC_STRUCT = """
shader Generic {
    struct Box<T> { T value; };
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* out_values;
        void main() {
            Box<float> box;
            box.value = 2.0;
            out_values[0] = box.value;
        }
    }
}
"""

SCALAR_CONSTRUCTOR = """
shader Convert {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer uint* counts;
        buffer float* out_values;
        void main() {
            out_values[0] = float(counts[0]);
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


def test_storage_array_parameters_become_pointer_arguments():
    code = _generate(STORAGE_PARAMETERS)
    assert (
        "def saxpy(dst: UnsafePointer[Float32, MutAnyOrigin], "
        "x: UnsafePointer[Float32, MutAnyOrigin], a: Float32) -> None:"
    ) in code
    assert "dst[Int(gid)] = (a * x[Int(gid)])" in code
    assert "List[" not in code
    assert "var dst" not in code
    assert "f32" not in code


def test_writable_storage_parameter_of_a_struct_is_refused():
    with pytest.raises(MojoRefusal, match="writable storage buffer"):
        _generate(WRITABLE_STORAGE_STRUCT)


def test_generic_struct_parameters_are_register_passable_and_qualified():
    code = _generate(GENERIC_STRUCT)
    assert "struct Box[T: TrivialRegisterPassable](TrivialRegisterPassable):" in code
    assert "    var value: Self.T\n" in code


def test_scalar_constructor_converts_its_argument_once():
    code = _generate(SCALAR_CONSTRUCTOR)
    assert "out_values[0] = Float32(counts[0])" in code
    assert "Float32(Float32(" not in code


@pytest.mark.parametrize(
    "source", [STORAGE_PARAMETERS, GENERIC_STRUCT, SCALAR_CONSTRUCTOR]
)
def test_generated_kernels_build_with_mojo(tmp_path, source):
    result = _build(tmp_path, _generate(source))
    assert result.returncode == 0, result.stderr[-2000:]

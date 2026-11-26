"""Workgroup scalars, vectors and structures in Rust output are read and written through their reference.

rust-gpu declares workgroup memory as ``#[spirv(workgroup)] name: &mut T`` and the emitter threads that reference to
the helpers that use it. Rust output used to read such a scalar as the reference itself (``s.v + a0`` with
``a0: &mut f32``, which rustc rejects: no ``Add<&mut f32>`` for ``f32``) and to assign ``a0 = &mut 1.0``, which
rebinds the parameter instead of writing the shared memory. See ``rust_workgroup_references``.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import crosstl

from .test_rust_codegen import assert_generated_rust_smoke_compiles

HELPER_READS_SHARED_SCALAR = """
shader P {
    struct S { float v; int k[2]; };
    compute {
        layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
        shared float a0;
        shared S s[3];
        buffer float* outp;
        float loadA(int i) {
            workgroupBarrier();
            float v = s[i].v + a0;
            workgroupBarrier();
            return v;
        }
        void main() {
            a0 = 1.0;
            s[1].v = 2.0;
            float r = loadA(1);
            outp[0] = r;
        }
    }
}
"""

VALUES_AND_PLACES = """
shader P {
    struct S { float v; int k[2]; };
    compute {
        layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
        shared float a0;
        shared vec3 v3;
        shared S st;
        shared float arr[4];
        buffer float* outp;
        float helper(int i) {
            float v = st.v + a0 + v3.y + arr[i];
            S copy = st;
            a0 += 1.0;
            v3 = v3 * 2.0;
            return v + copy.v;
        }
        void main() {
            uint i = gl_LocalInvocationID.x;
            a0 = 1.0;
            v3 = vec3(1.0, 2.0, 3.0);
            st.v = 2.0;
            arr[i] = a0;
            workgroupBarrier();
            vec3 w = v3 + vec3(a0);
            S local = st;
            outp[i] = helper(int(i)) + w.x + local.v;
        }
    }
}
"""


def rust(source: str) -> str:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "program.cgl")
        path.write_text(source, encoding="utf-8")
        return crosstl.translate(str(path), backend="rust", format_output=False)


def flat(code: str) -> str:
    return re.sub(r"\s+", " ", code)


def function(code: str, name: str) -> str:
    match = re.search(rf"pub fn {name}\(.*?\n}}\n", code, re.S)
    assert match, code
    return flat(match.group(0))


def test_helper_reads_a_shared_scalar_through_its_reference():
    code = rust(HELPER_READS_SHARED_SCALAR)
    helper = function(code, "loadA")
    assert "a0: &mut f32" in helper and "mut a0" not in helper
    assert "let v: f32 = (s[i as usize].v + (*a0));" in helper
    entry = function(code, "main")
    assert "(*a0) = 1.0;" in entry
    assert "&mut 1.0" not in code
    assert "loadA(1, a0, s)" in entry


def test_scalars_vectors_and_structures_are_places_and_arrays_index_the_reference():
    code = rust(VALUES_AND_PLACES)
    entry = function(code, "main")
    for statement in (
        "(*a0) = 1.0;",
        "(*v3) = Vec3::new(1.0, 2.0, 3.0);",
        "(*st).v = 2.0;",
        "arr[i as usize] = (*a0);",
        "let local: S = (*st);",
        "helper((i as i32), a0, v3, st, arr)",
    ):
        assert statement in entry, entry
    helper = function(code, "helper")
    assert "a0: &mut f32, v3: &mut Vec3, st: &mut S, arr: &mut [f32; 4]" in helper
    for statement in (
        "((((*st).v + (*a0)) + (*v3).y) + arr[i as usize])",
        "let copy: S = (*st);",
        "(*a0) += 1.0;",
    ):
        assert statement in helper, helper
    assert re.search(r"\(\*v3\) = ", helper), helper


def test_workgroup_references_compile_with_rust_gpu(tmp_path):
    for index, source in enumerate((HELPER_READS_SHARED_SCALAR, VALUES_AND_PLACES)):
        directory = tmp_path / str(index)
        directory.mkdir()
        assert_generated_rust_smoke_compiles(rust(source), directory)

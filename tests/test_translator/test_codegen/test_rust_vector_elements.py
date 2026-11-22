"""Rust output selects vector components and matrix columns without glam's ``Index`` and ``IndexMut``.

glam's ``Index``/``IndexMut`` and ``Mat::col_mut`` select a component reference with a ``match`` that rust-gpu writes
as an ``OpPhi`` of ``Function`` pointers, which spirv-val rejects for Vulkan (``Instruction may only have a logical
pointer operand in the StorageBuffer or Workgroup storage classes``). A constant index is now the component or column
name, a dynamic read is ``vector_extract_dynamic``/``BVecN::test``/``Mat::col`` and a dynamic store is a ``match`` on
the index. See ``rust_vector_elements``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.codegen.rust_vector_elements import constant_component_index

from .test_rust_codegen import assert_generated_rust_smoke_compiles

COMPONENTS = """
shader P {
    compute {
        layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int k = iin[i] & 1;
            mat2 m = mat2(float(iin[i]), 2.0, 1.0, 3.0);
            vec2 c = vec2(m[0][k], m[1][k]);
            vec3 v = vec3(1.0, 2.0, 3.0);
            v[k] = 5.0;
            v[2] += 1.0;
            v[k + 1] *= 2.0;
            v[k]++;
            m[1][0] = 4.0;
            m[k] = vec2(7.0, 8.0);
            m[k][k] -= 1.0;
            bvec2 b = bvec2(true, false);
            b[k] = false;
            b[1] = true;
            vec2 col = m[k];
            outp[i] = m[0][1] * 100.0 + m[1].x * 10.0 + c.x + 0.5 * c.y + v[k] + v[1] + col.y
                + (b[k] ? 1.0 : 0.0) + (b[1] ? 2.0 : 0.0);
        }
    }
}
"""

OUT_OF_RANGE = """
shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            vec2 v = vec2(1.0, 2.0);
            outp[0] = v[2];
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


def test_constant_component_index_reads_only_non_negative_integer_literals():
    assert constant_component_index(2) == 2
    assert constant_component_index("3") == 3
    assert constant_component_index(-1) is None
    assert constant_component_index(True) is None
    assert constant_component_index("k") is None


def test_components_are_named_extracted_or_stored_through_a_match():
    code = flat(rust(COMPONENTS))
    for fragment in (
        "vector_extract_dynamic(m.x_axis, k as usize)",
        "vector_extract_dynamic(m.y_axis, k as usize)",
        "match k as usize { 0 => { v.x = 5.0; } 1 => { v.y = 5.0; } 2 => { v.z = 5.0; } _ => {} }",
        "v.z += 1.0;",
        "match (k + 1) as usize { 0 => { v.x *= 2.0; }",
        "match k as usize { 0 => { v.x += 1.0; }",
        "m.y_axis.x = 4.0;",
        "match k as usize { 0 => { m.x_axis = Vec2::new(7.0, 8.0); } 1 => { m.y_axis = Vec2::new(7.0, 8.0); } "
        "_ => {} }",
        "match k as usize { 0 => { match k as usize { 0 => { m.x_axis.x -= 1.0; } 1 => { m.x_axis.y -= 1.0; }",
        "match k as usize { 0 => { b.x = false; } 1 => { b.y = false; } _ => {} }",
        "b.y = true;",
        "let col: Vec2 = m.col(k as usize);",
        "b.test(k as usize)",
        "m.x_axis.y * 100.0",
        "vector_extract_dynamic(v, k as usize)",
        "+ v.y)",
    ):
        assert fragment in code, (fragment, code)
    assert "col_mut" not in code
    assert not re.search(r"\b(?:v|m|b|c)\[", code), code


def test_a_constant_index_past_the_last_component_is_refused():
    with pytest.raises(Exception, match="index 2 is past the last of the 2 components"):
        rust(OUT_OF_RANGE)


def test_components_compile_with_rust_gpu_to_valid_vulkan_spirv(tmp_path):
    validator = shutil.which("spirv-val")
    if validator is None:
        pytest.skip("spirv-val is not on PATH")
    assert_generated_rust_smoke_compiles(rust(COMPONENTS), tmp_path)
    modules = sorted(tmp_path.rglob("*.spv"))
    assert modules, sorted(tmp_path.iterdir())
    for module in modules:
        result = subprocess.run(
            [validator, "--target-env", "vulkan1.3", str(module)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

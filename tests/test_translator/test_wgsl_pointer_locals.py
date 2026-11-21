"""WGSL output writes a CrossGL pointer local ``T* p = &r;`` as ``let p = &r;`` and refuses what WGSL cannot hold.

WGSL keeps a pointer only in a ``let`` declaration initialized with an address-of expression, and it cannot take
the address of a vector component, re-point a pointer, index one or do arithmetic on it
(``crosstl/translator/codegen/wgsl_pointer_locals.py``). Each supported case is validated with naga when it is on
``PATH``.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.refusals import WGSLRefusal

PROGRAM = """shader Probe {
    struct Pair { float a; vec2 b; };
    struct R { float ro[4]; };
    R r @glsl_buffer_block(std430) @binding(2) @readonly;
    shared float tile[4];
    buffer float* fin;
    buffer float* outp;
    HELPERS
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            BODY
        }
    }
}
"""


def translate(tmp_path, body, helpers=""):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM.replace("HELPERS", helpers).replace("BODY", body))
    output = crosstl.translate(str(path), backend="wgsl", format_output=False)
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(output)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr + output
    return output


def refusal(tmp_path, body, helpers=""):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM.replace("HELPERS", helpers).replace("BODY", body))
    with pytest.raises(WGSLRefusal) as raised:
        crosstl.translate(str(path), backend="wgsl", format_output=False)
    return str(raised.value)


def test_a_pointer_to_a_storage_buffer_element_is_a_let(tmp_path):
    output = translate(
        tmp_path,
        """float* p = &outp[i];
            *p = fin[i] * 2.0;
            *p += *p * 4.0 + 1.0;""",
    )
    assert "let p = &outp[i];" in output
    assert "*p = (fin[i] * 2.0);" in output
    assert "*p += ((*p * 4.0) + 1.0);" in output


def test_a_pointer_to_a_local_variable_writes_the_variable(tmp_path):
    output = translate(
        tmp_path,
        """float x = fin[i];
            float* p = &x;
            *p = *p * 3.0;
            outp[i] = x;""",
    )
    assert "let p = &x;" in output
    assert "*p = (*p * 3.0);" in output


def test_a_pointer_to_workgroup_memory(tmp_path):
    output = translate(
        tmp_path,
        """float* t = &tile[0];
            *t = fin[i];
            outp[i] = *t;""",
    )
    assert "let t = &tile[0];" in output


def test_a_pointer_to_a_structure_member_and_a_matrix_column(tmp_path):
    output = translate(
        tmp_path,
        """Pair pair;
            float* a = &pair.a;
            *a = fin[i];
            mat2 m = mat2(1.0);
            vec2* c = &m[1];
            *c = vec2(*a, 2.0);
            outp[i] = m[1].x + pair.a;""",
    )
    assert "let a = &pair.a;" in output
    assert "let c = &m[1];" in output


def test_a_parameter_whose_address_is_taken_becomes_a_variable(tmp_path):
    output = translate(
        tmp_path,
        "outp[i] = twice(fin[i]);",
        helpers="""float twice(float value) {
        float* p = &value;
        *p = *p * 2.0;
        return value;
    }""",
    )
    assert "var value = value_in;" in output or "var value: f32 = value_in;" in output
    assert "let p = &value;" in output


def test_a_read_only_buffer_element_may_be_read_through_a_pointer(tmp_path):
    output = translate(
        tmp_path,
        """float* p = &r.ro[1];
            outp[i] = *p;""",
    )
    assert "let p = &r.ro[1];" in output


def test_a_vector_component_has_no_address(tmp_path):
    message = refusal(
        tmp_path,
        """vec2 v = vec2(1.0);
            float* p = &v.x;
            *p = 2.0;""",
    )
    assert "cannot take the address of a vector component" in message


def test_an_indexed_vector_component_has_no_address(tmp_path):
    message = refusal(
        tmp_path,
        """vec4 v = vec4(1.0);
            float* p = &v[2];
            *p = 2.0;""",
    )
    assert "vector component" in message


def test_re_pointing_a_pointer_is_refused(tmp_path):
    message = refusal(
        tmp_path,
        """float* p = &outp[i];
            p = &outp[i + 1u];
            *p = 1.0;""",
    )
    assert "pointer local 'p'" in message
    assert "re-pointing" in message


def test_pointer_arithmetic_is_refused(tmp_path):
    message = refusal(
        tmp_path,
        """float* p = &outp[i];
            *(p + 1) = 1.0;""",
    )
    assert "pointer arithmetic" in message


def test_indexing_a_pointer_is_refused(tmp_path):
    message = refusal(
        tmp_path,
        """float* p = &outp[i];
            p[1] = 1.0;""",
    )
    assert "pointer local 'p'" in message


def test_passing_a_pointer_on_is_refused(tmp_path):
    message = refusal(
        tmp_path,
        """float* p = &outp[i];
            outp[i] = read(p);""",
        helpers="""float read(float value) { return value; }""",
    )
    assert "pointer local 'p'" in message


def test_an_uninitialized_pointer_is_refused(tmp_path):
    message = refusal(
        tmp_path,
        """float* p;
            outp[i] = 1.0;""",
    )
    assert "without an address-of initializer" in message


def test_the_address_of_a_let_value_is_refused(tmp_path):
    message = refusal(
        tmp_path,
        """const float c = fin[i];
            const float* p = &c;
            outp[i] = *p;""",
    )
    assert "take the address of 'c'" in message


def test_a_pointer_to_a_pointer_is_refused(tmp_path):
    message = refusal(
        tmp_path,
        """float* p = &outp[i];
            float* q = &p;
            outp[i] = 1.0;""",
    )
    assert "pointer local" in message


def test_writing_a_read_only_buffer_through_a_pointer_is_refused(tmp_path):
    message = refusal(
        tmp_path,
        """float* p = &r.ro[1];
            *p = 1.0;""",
    )
    assert "read-only storage buffer r" in message

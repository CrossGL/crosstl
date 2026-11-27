"""A dereference of a storage buffer becomes a WGSL element access.

WGSL declares a CrossGL ``buffer T*`` as ``var<storage, read_write> name: array<T>``, which is not a pointer, so
``*name`` and ``*(name + i)`` do not validate: the operand of ``*`` must be a pointer (WGSL section 8.14). The
generator printed both unchanged and reported success. ``*buf`` now becomes ``buf[0]``, and ``*(buf + i)`` or
``*(i + buf)`` becomes ``buf[i]``, as the GLSL generator already does. A dereference of a buffer that has no
element-access form is refused instead of being printed (#4751).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.refusals import WGSLRefusal

PROGRAM = """shader Deref {
    HELPERS
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* x;
        buffer float* o;

        void main() {
            BODY
        }
    }
}
"""


def _translate(tmp_path, body, helpers=""):
    path = tmp_path / "deref.cgl"
    path.write_text(
        PROGRAM.replace("HELPERS", helpers).replace("BODY", body), encoding="utf-8"
    )
    return crosstl.translate(str(path), backend="wgsl")


def _validate(tmp_path, output):
    naga = shutil.which("naga")
    if naga is None:
        pytest.skip("naga is not installed")
    path = tmp_path / "out.wgsl"
    path.write_text(output, encoding="utf-8")
    result = subprocess.run(
        [naga, str(path)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr + output


def test_issue_probe_reads_elements(tmp_path):
    output = _translate(tmp_path, "o[0] = *(x + 1);\n            o[1] = *x;")
    assert "o[0] = x[1];" in output
    assert "o[1] = x[0];" in output
    assert "*x" not in output and "*(x" not in output
    _validate(tmp_path, output)


def test_offset_on_either_side(tmp_path):
    output = _translate(tmp_path, "o[0] = *(2 + x);")
    assert "o[0] = x[2];" in output
    _validate(tmp_path, output)


def test_variable_and_nested_offsets(tmp_path):
    output = _translate(
        tmp_path,
        "int i = 3;\n            o[0] = *(x + i);\n            o[1] = *(x + i + 1);",
    )
    assert "o[0] = x[i];" in output
    assert "o[1] = x[(i + 1)];" in output
    _validate(tmp_path, output)


def test_offsets_keep_source_order(tmp_path):
    output = _translate(tmp_path, "int i = 3;\n            o[0] = *(1 + (i + x));")
    assert "o[0] = x[(1 + i)];" in output
    _validate(tmp_path, output)


def test_dereference_as_assignment_target(tmp_path):
    output = _translate(tmp_path, "*o = 2.0;\n            *(o + 3) = *x;")
    assert "o[0] = 2.0;" in output
    assert "o[3] = x[0];" in output
    _validate(tmp_path, output)


def test_buffer_parameter_dereference_matches_indexing(tmp_path):
    # A buffer parameter is a ptr<storage, array<T>>, so a dereference writes the same
    # element access as indexing it. naga does not accept storage pointer parameters
    # without the unrestricted_pointer_parameters extension, so this checks the text.
    deref = _translate(
        tmp_path,
        "o[0] = first(x);",
        helpers="float first(buffer float* values) { return *(values + 1); }",
    )
    index = _translate(
        tmp_path,
        "o[0] = first(x);",
        helpers="float first(buffer float* values) { return values[1]; }",
    )
    assert "return (*values)[1];" in deref
    assert deref == index


@pytest.mark.parametrize(
    "body",
    [
        "o[0] = *(x - 1);",
        "o[0] = *(x + -1);",
        "o[0] = *(x + 1.5);",
        "o[0] = *(x + x[0]);",
    ],
)
def test_dereference_without_element_access_is_refused(tmp_path, body):
    with pytest.raises(WGSLRefusal, match="dereference"):
        _translate(tmp_path, body)

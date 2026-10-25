"""Named braced struct constructors must preserve every field in member order."""

import re

import pytest

import crosstl

SOURCE = """
shader Probe {
    struct Pair { float a; float b; };
    compute {
        layout(local_size_x = 1) in;
        buffer float* outp;
        void main() {
            Pair q = Pair { FIELDS };
            outp[0] = q.a + q.b;
        }
    }
}
"""


def translate(tmp_path, backend, fields):
    source = tmp_path / "named-struct.cgl"
    source.write_text(SOURCE.replace("FIELDS", fields))
    return crosstl.translate(str(source), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", ["wgsl", "slang"])
@pytest.mark.parametrize("fields", ["a: 1.0, b: 2.0", "b: 2.0, a: 1.0", "1.0, b: 2.0"])
def test_named_fields_follow_declaration_order(tmp_path, backend, fields):
    output = translate(tmp_path, backend, fields)
    assert re.search(r"Pair\(1\.0, 2\.0\)", output), output


@pytest.mark.parametrize("backend", ["wgsl", "slang"])
def test_positional_struct_constructor_is_preserved(tmp_path, backend):
    output = translate(tmp_path, backend, "1.0, 2.0")
    assert "Pair(1.0, 2.0)" in output


@pytest.mark.parametrize("backend", ["wgsl", "slang"])
@pytest.mark.parametrize(
    "fields,reason",
    [
        ("a: 1.0", "missing.*b"),
        ("a: 1.0, c: 2.0", "no field.*c"),
        ("1.0, a: 2.0, b: 3.0", "more than once.*a"),
    ],
)
def test_incomplete_or_invalid_named_constructor_is_refused(
    tmp_path, backend, fields, reason
):
    with pytest.raises(ValueError, match=reason):
        translate(tmp_path, backend, fields)


@pytest.mark.parametrize("backend", ["wgsl", "slang"])
def test_named_initializer_keeps_dynamic_buffer_values(tmp_path, backend):
    source = tmp_path / "buffer-values.cgl"
    source.write_text(
        SOURCE.replace("buffer float* outp;", "buffer float* fin; buffer float* outp;")
        .replace(
            "Pair q = Pair { FIELDS };",
            "uint i = gl_GlobalInvocationID.x; Pair q = Pair { b: fin[i], a: 1.0 };",
        )
        .replace("outp[0]", "outp[i]")
    )
    output = crosstl.translate(str(source), backend=backend, format_output=False)
    assert re.search(r"Pair\(1\.0, fin\[i\]\)", output), output


@pytest.mark.parametrize(
    "replace,expected",
    [
        # WGSL reserves ``filter``, so the structure is emitted as ``filter_``.
        (("Pair", "filter"), "filter_(1.0, 2.0)"),
        # A builtin Option specialization adds member types of its own.
        (("Pair q =", "Option<int> m = Option::None; Pair q ="), "Pair(1.0, 2.0)"),
    ],
)
def test_wgsl_named_initializer_reads_the_declared_fields(tmp_path, replace, expected):
    source = tmp_path / "declared-fields.cgl"
    source.write_text(SOURCE.replace(*replace).replace("FIELDS", "b: 2.0, a: 1.0"))
    output = crosstl.translate(str(source), backend="wgsl", format_output=False)
    assert expected in output, output

"""Metal output never contains a double-precision type.

MSL has no double type (Metal Shading Language Specification, section 2.1), so a program that needs binary64 is
refused with a typed error instead of being printed with ``double`` values the Metal compiler rejects.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.translator.codegen.metal_double_refusal import refuse_metal_double_values
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

COMPUTE = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            BODY
        }
    }
}
"""


def _reasons(output):
    reasons = []
    refuse_metal_double_values(output, reasons.append)
    return reasons


def _metal(tmp_path, body):
    path = tmp_path / "probe.cgl"
    path.write_text(COMPUTE.replace("BODY", body), encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


@pytest.mark.parametrize(
    "body",
    (
        "double d = double(fin[i]) * 3.0; outp[i] = float(d);",
        "dvec2 h = dvec2(double(fin[i]), 1.0); outp[i] = float(h.x);",
        "outp[i] = float(fma(dvec2(fin[i]), dvec2(2.0), dvec2(1.0)).y);",
        "dmat2 m = dmat2(1.0); outp[i] = float(m[0][0]);",
        "outp[i] = float(double(fin[i]));",
    ),
)
def test_double_values_are_refused(tmp_path, body):
    with pytest.raises(MetalLoweringRefused, match="double-precision value"):
        _metal(tmp_path, body)


def test_float_programs_are_unaffected(tmp_path):
    code = _metal(
        tmp_path, "float double_value = fin[i] * 2.0; outp[i] = double_value;"
    )
    assert "double_value" in code


def test_comments_strings_and_longer_identifiers_are_not_types():
    output = (
        "// double precision is not needed\n"
        "/* double2 */ float doubled = 2.0;\n"
        'constant char* label = "double";\n'
        "float double_value = doubled;\n"
    )
    assert _reasons(output) == []


def test_the_refusal_names_every_spelling_and_the_first_line():
    output = "float a = 1.0;\ndouble2 b = double2(1.0);\nlong double c = 0.0;\n"
    (reason,) = _reasons(output)
    assert "uses double2, long double" in reason
    assert "first at line 2: double2 b = double2(1.0);" in reason

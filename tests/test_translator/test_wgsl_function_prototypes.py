"""WGSL keeps actual definitions when CrossGL includes prototypes."""

import re

import pytest

import crosstl


@pytest.mark.parametrize("prototype_first", [True, False])
def test_prototype_never_hides_the_matching_wgsl_definition(tmp_path, prototype_first):
    declaration = "float twice(float x);"
    definition = "float twice(float x) { return x * 2.0; }"
    helpers = (
        declaration + "\n" + definition
        if prototype_first
        else definition + "\n" + declaration
    )
    source = tmp_path / "prototype.cgl"
    source.write_text(
        "shader Proto { compute {\n"
        "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n"
        "buffer float* data;\n"
        + helpers
        + "\nfloat quadruple(float x) { return twice(twice(x)); }\n"
        "void main() { uint i = gl_GlobalInvocationID.x; data[i] = quadruple(data[i]); }\n"
        "} }\n"
    )
    output = crosstl.translate(str(source), backend="wgsl", format_output=False)
    assert output.count("fn twice(") == 1, output
    assert re.search(r"fn twice\([^)]*\)\s*->\s*f32\s*\{[^}]*return", output), output
    assert "twice(twice(x))" in output, output


def test_empty_void_definition_is_not_treated_as_a_prototype(tmp_path):
    source = tmp_path / "empty.cgl"
    source.write_text(
        "shader Empty { compute { void noop() {} void main() { noop(); } } }"
    )
    output = crosstl.translate(str(source), backend="wgsl", format_output=False)
    assert "fn noop()" in output, output
    assert "noop();" in output, output

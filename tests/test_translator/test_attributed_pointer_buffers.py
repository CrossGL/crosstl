"""A pointer buffer declaration with post-declaration attributes, ``buffer float* data @binding(2);``.

The declaration lookahead accepted a pointer declarator only before ``[``, ``=``, ``;`` or ``,``, so a pointer buffer
followed by an attribute was not a declaration: the module and stage parsers skipped it token by token, and every
target read an undeclared buffer.
"""

from __future__ import annotations

import re

import pytest

import crosstl
import crosstl.translator

MODULE_SCOPE = """shader P {
    buffer float* source @binding(2);
    buffer float* target @binding(3);
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            target[i] = source[i] * 2.0;
        }
    }
}
"""

STAGE_SCOPE = """shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* source @binding(2);
        buffer float* target @binding(3);
        void main() {
            uint i = gl_GlobalInvocationID.x;
            target[i] = source[i] * 2.0;
        }
    }
}
"""


def _variables(ast):
    stage = next(iter(ast.stages.values()))
    return {
        variable.name: [attribute.name for attribute in variable.attributes or []]
        for variable in list(ast.global_variables or []) + list(
            stage.local_variables or []
        )
    }


@pytest.mark.parametrize("source", [MODULE_SCOPE, STAGE_SCOPE], ids=["module", "stage"])
def test_attributed_pointer_buffers_are_declarations(source):
    variables = _variables(crosstl.translator.parse(source))
    assert variables["source"] == ["binding"]
    assert variables["target"] == ["binding"]


@pytest.mark.parametrize("source", [MODULE_SCOPE, STAGE_SCOPE], ids=["module", "stage"])
@pytest.mark.parametrize(
    ("backend", "pattern"),
    [
        (
            "opengl",
            r"layout\(std430, binding = {binding}\) buffer \w+ {{ float {name}\[\]; }};",
        ),
        ("metal", r"device (?:const )?float ?\* ?{name} \[\[buffer\({binding}\)\]\]"),
        ("directx", r"StructuredBuffer<float> {name} : register\([tu]{binding}\);"),
    ],
)
def test_targets_declare_the_buffers_at_their_bindings(
    tmp_path, source, backend, pattern
):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    code = crosstl.translate(str(path), backend=backend, format_output=False)
    for name, binding in (("source", 2), ("target", 3)):
        assert re.search(pattern.format(name=name, binding=binding), code), code


def test_spirv_binds_the_buffers(tmp_path):
    path = tmp_path / "shader.cgl"
    path.write_text(MODULE_SCOPE)
    assembly = crosstl.translate(str(path), backend="vulkan", format_output=False)
    assert sorted(map(int, re.findall(r"OpDecorate %\w+ Binding (\d+)", assembly))) == [
        2,
        3,
    ]

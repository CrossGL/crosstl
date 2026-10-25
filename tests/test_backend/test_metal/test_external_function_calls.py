"""A Metal call to a function the translation unit declares but never defines is refused.

MSL links the functions of a library across source files, so a file may call a function it only declares. The parser
dropped the prototype and every target was given a call to an undeclared function.
"""

import pytest

import crosstl
from crosstl.backend.Metal.external_functions import MetalExternalFunctionError

EXTERNAL = """
#include <metal_stdlib>
using namespace metal;

half3 tonemap(half3 inColor);

kernel void k(device half4 *out [[buffer(0)]], uint tid [[thread_position_in_grid]])
{
    out[tid] = half4(tonemap(out[tid].xyz), 1.0h);
}
"""

FORWARD = """
#include <metal_stdlib>
using namespace metal;

namespace tone { half3 tonemap(half3 inColor); }

kernel void k(device half4 *out [[buffer(0)]], uint tid [[thread_position_in_grid]])
{
    out[tid] = half4(tone::tonemap(out[tid].xyz), 1.0h);
}

namespace tone {
half3 tonemap(half3 inColor)
{
    return inColor / (inColor + half3(1.0h));
}
}
"""

UNUSED = EXTERNAL.replace(
    "half4(tonemap(out[tid].xyz), 1.0h)", "half4(out[tid].xyz, 1.0h)"
)


def _translate(tmp_path, source, backend):
    path = tmp_path / "unit.metal"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("backend", ["metal", "opengl", "directx", "cgl"])
def test_calls_to_declared_only_functions_are_refused(tmp_path, backend):
    with pytest.raises(
        MetalExternalFunctionError, match="'tonemap' is declared but not defined"
    ) as raised:
        _translate(tmp_path, EXTERNAL, backend)
    assert (
        raised.value.project_diagnostic_code
        == "project.translate.metal-external-function"
    )


@pytest.mark.parametrize("backend", ["metal", "opengl"])
def test_forward_declarations_of_defined_functions_translate(tmp_path, backend):
    assert "tonemap" in _translate(tmp_path, FORWARD, backend)


def test_unused_prototypes_do_not_block_translation(tmp_path):
    assert "tonemap" not in _translate(tmp_path, UNUSED, "opengl")

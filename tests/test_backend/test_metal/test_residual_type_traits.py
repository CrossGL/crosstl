"""A type trait left unevaluated in concrete Metal code is refused for every target.

No stage after the Metal preprocessor folds ``is_same<A, B>::value`` or ``is_same_v<A, B>``. A trait whose operands
cannot be resolved in a non-template function used to reach the parser as an identifier and was emitted as a mangled
name (``is_same_u3c...``) for template-friendly targets such as CrossGL. It now raises ``MetalTypeTraitError``.
"""

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import MetalPreprocessor, MetalTypeTraitError

_KERNEL = """
#include <metal_stdlib>
using namespace metal;

kernel void pick(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {{
    if ({trait}) {{
        out[tid] = 1.0f;
    }} else {{
        out[tid] = 2.0f;
    }}
}}
"""


def _translate(tmp_path, trait, backend):
    source = tmp_path / "trait.metal"
    source.write_text(_KERNEL.format(trait=trait))
    return crosstl.translate(str(source), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", ["cgl", "metal", "directx", "opengl"])
@pytest.mark.parametrize(
    "trait",
    [
        "is_same<sampler, float>::value",
        "is_same_v<texture2d<float>, float>",
        "metal::is_same_v<MissingType, int>",
    ],
)
def test_unresolved_concrete_trait_is_refused_for_every_target(
    tmp_path, trait, backend
):
    with pytest.raises(MetalTypeTraitError) as exc_info:
        _translate(tmp_path, trait, backend)

    error = exc_info.value
    assert error.reason == "operand-unresolved"
    assert error.owner == "pick"
    assert error.project_diagnostic_code == (
        "project.translate.metal-type-trait-unresolved"
    )


def test_resolved_concrete_trait_still_folds(tmp_path):
    output = _translate(tmp_path, "is_same<float, int>::value", "cgl")

    assert "is_same" not in output
    assert "if (false)" in output


def test_resolved_concrete_trait_selects_the_branch_for_glsl(tmp_path):
    output = _translate(tmp_path, "is_same<float, float>::value", "opengl")

    assert "is_same" not in output
    assert "out_[tid] = 1.0;" in output
    assert "2.0" not in output


def test_preprocess_refuses_residual_trait_without_fail_closed_assertions():
    code = """
    kernel void residual(device int* out [[buffer(0)]]) {
        out[0] = is_same<sampler, int>::value ? 1 : 0;
    }
    """

    with pytest.raises(MetalTypeTraitError) as exc_info:
        MetalPreprocessor().preprocess(code)

    assert exc_info.value.operands == ("sampler", "int")


def test_preprocess_keeps_traits_on_template_parameters():
    code = """
    template <typename T>
    int pick(T value) {
        return is_same<T, int>::value ? 1 : 0;
    }

    kernel void concrete(device int* out [[buffer(0)]]) {
        out[0] = is_same<float, float>::value ? 1 : 0;
    }
    """

    output = MetalPreprocessor().preprocess(code)

    assert "return is_same<T, int>::value ? 1 : 0;" in output
    assert "out[0] = true ? 1 : 0;" in output

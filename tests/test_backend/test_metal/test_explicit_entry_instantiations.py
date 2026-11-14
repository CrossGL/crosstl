"""Explicit kernel instantiations that the preprocessor leaves unmaterialized are refused.

A source with more ``template [[host_name(...)]] [[kernel]] decltype(f<T>) f<T>;`` declarations than the
specialization limit (512) kept them all unmaterialized, the parser skipped them, and the translation had no
kernel at all. Each refusal case below translated to an empty program on that code.
"""

import textwrap

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import (
    DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT,
    MetalTemplateSpecializationError,
)

KERNEL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    template <typename T>
    T twice(T value) { return value + value; }
    template float twice<float>(float);

    template <typename T>
    [[kernel]] void fill(device T* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
        out[gid] = twice(T(gid));
    }
""")


def _source(tmp_path, count):
    declarations = "\n".join(
        f'template [[host_name("fill{index}")]] [[kernel]] decltype(fill<float>) fill<float>;'
        for index in range(count)
    )
    path = tmp_path / "fill.metal"
    path.write_text(KERNEL + declarations + "\n")
    return str(path)


@pytest.mark.parametrize("backend", ["metal", "cgl"])
def test_instantiations_over_the_limit_are_refused(tmp_path, backend):
    source = _source(tmp_path, DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT + 1)

    with pytest.raises(MetalTemplateSpecializationError) as raised:
        crosstl.translate(source, backend=backend, format_output=False)

    assert (
        raised.value.project_diagnostic_code
        == "project.translate.metal-template-specialization"
    )
    assert 'host_name("fill0")' in str(raised.value)


def test_instantiations_within_the_limit_keep_every_kernel(tmp_path):
    output = crosstl.translate(
        _source(tmp_path, 3), backend="metal", format_output=False
    )

    for index in range(3):
        assert f"kernel void fill{index}(" in output
    # The explicit instantiation of a helper template is not an entry point and is accepted.
    assert "decltype" not in output

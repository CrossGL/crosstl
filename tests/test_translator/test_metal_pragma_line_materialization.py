"""Template materialization of Metal kernels whose bodies hold preserved ``#pragma`` lines.

A ``#pragma`` line (written as one, or produced by the ``_Pragma`` operator) ends without a semicolon, so the
statement scan that finds local declarations and statement blocks must step over it rather than take it as
the start of the statement that follows.
"""

import textwrap

import pytest

import crosstl

KERNEL = """
template <bool Negate, typename T>
T apply(T value) {{
    if constexpr (Negate) {{
        return -value;
    }}
    return value;
}}

template <typename T>
[[kernel]] void plain_helper(device T* results [[buffer(0)]]) {{
    constexpr bool negate = false;
    {before_loop}
    for (int index = 0; index < 1; ++index) {{
        {in_loop}
        T value = results[index];
        results[index] = apply<negate>(value);
    }}
}}

instantiate_kernel("plain_helper_float", plain_helper, float)
"""

PRAGMA = "#pragma clang loop unroll(disable)"
OPERATOR = '_Pragma("clang loop unroll(disable)")'


@pytest.mark.parametrize(
    "before_loop, in_loop",
    [
        (PRAGMA, ""),
        ("", PRAGMA),
        (OPERATOR, ""),
        ("", OPERATOR),
    ],
    ids=[
        "pragma-before-loop",
        "pragma-in-loop",
        "operator-before-loop",
        "operator-in-loop",
    ],
)
def test_template_call_after_a_pragma_line_is_materialized(
    tmp_path, before_loop, in_loop
):
    path = tmp_path / "plain_helper.metal"
    path.write_text(
        textwrap.dedent(KERNEL.format(before_loop=before_loop, in_loop=in_loop)),
        encoding="utf-8",
    )
    output = crosstl.translate(str(path), backend="directx", format_output=False)
    assert "apply<" not in output

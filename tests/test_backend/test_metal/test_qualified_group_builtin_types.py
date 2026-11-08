"""``metal::``-qualified SIMD-group built-ins type a call argument like the unqualified spelling.

``::metal::simd_shuffle_and_fill_down(val, fill, i)`` returns the type of ``val`` (MSL 4.1 section 6.10.2), so a
constrained call that receives it, such as ``mul(val, ::metal::simd_shuffle_down(val, i))``, can be resolved.
"""

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor


@pytest.mark.parametrize(
    "expression,expected",
    (
        ("simd_shuffle_down(val, i)", "float2"),
        ("metal::simd_shuffle_down(val, i)", "float2"),
        ("::metal::simd_shuffle_and_fill_down(val, float2(1, 0), i)", "float2"),
        (":: metal :: simd_sum(count)", "uint"),
        ("metal::simd_broadcast_first(count)", "uint"),
        # Only the Metal standard namespace names the built-in.
        ("other::simd_shuffle_down(val, i)", None),
        ("metal::not_a_group_builtin(val)", None),
        ("metal::simd_shuffle_down(missing, i)", None),
    ),
)
def test_qualified_group_builtin_returns_first_argument_type(expression, expected):
    preprocessor = MetalPreprocessor()
    locals_ = {"val": "float2", "count": "uint", "i": "ushort"}
    assert preprocessor._infer_argument_type(expression, {}, locals_) == expected

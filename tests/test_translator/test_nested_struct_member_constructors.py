"""A struct member given as a brace list must be built with its own type."""

import re

import pytest

import crosstl

SOURCE = """
shader NestedBrace {
    struct I { uint a; uint b; };
    struct N { uint a; I i; };
    compute {
        layout(local_size_x = 1) in;
        buffer uint* o;
        void main() {
            uint t = gl_GlobalInvocationID.x;
            N n = INITIALIZER;
            o[t] = n.a + n.i.b;
        }
    }
}
"""


def translate(tmp_path, backend, initializer):
    source = tmp_path / "nested.cgl"
    source.write_text(SOURCE.replace("INITIALIZER", initializer))
    return crosstl.translate(str(source), backend=backend, format_output=False)


@pytest.mark.parametrize("initializer", ["N(1u, {3u, 4u})", "{1u, {3u, 4u}}"])
def test_wgsl_builds_member_with_struct_constructor(tmp_path, initializer):
    output = translate(tmp_path, "wgsl", initializer)
    assert "N(1u, I(3u, 4u))" in output
    assert "array(" not in output


@pytest.mark.parametrize("initializer", ["N(1u, {3u, 4u})", "{1u, {3u, 4u}}"])
def test_spirv_builds_member_with_struct_type(tmp_path, initializer):
    output = translate(tmp_path, "vulkan", initializer)
    assert not re.search(
        r"OpCompositeConstruct %_arr_\w+ %\w+_3 %\w+_4", output
    ), output
    member = re.search(r"OpName %(\w+) \"I\"", output)
    assert member is not None, output
    assert re.search(
        rf"OpCompositeConstruct %{member.group(1)} %\w+ %\w+", output
    ), output


def test_wgsl_keeps_an_argument_past_the_last_member(tmp_path):
    # Each argument is matched to its member's type only when the counts agree; an extra argument stays in the
    # call, where naga reports it, instead of being dropped.
    output = translate(tmp_path, "wgsl", "N(1u, {3u, 4u}, 5u)")
    assert re.search(r"N\(1u, [^;]*, 5u\)", output), output

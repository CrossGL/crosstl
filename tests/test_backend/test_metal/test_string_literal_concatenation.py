"""Adjacent Metal string literals are one literal.

C++ joins ``"a" "b"`` into ``"ab"`` (translation phase 6). ``static_assert`` only accepted a single literal
as its message, so a message split across literals, as macro-built diagnostics produce, stopped the parse
with ``Expected RPAREN, got STRING``. The parser now folds a run of literals into one ordinary literal and
keeps escapes from merging across the boundary.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.Metal.MetalAst import StaticAssertNode
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.backend.Metal.string_literals import concatenate_string_literals

KERNEL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    #define CHECK_SIZE(T, N) static_assert(sizeof(T) == N, "size of " #T " is not " #N)

    CHECK_SIZE(float, 4);

    kernel void k(device float *out [[buffer(0)]],
                  uint i [[thread_position_in_grid]])
    {
        static_assert(sizeof(int) == 4, "int is" " four bytes");
        out[i] = out[i] * 2.0f;
    }
    """)


@pytest.mark.parametrize(
    ("spellings", "expected"),
    [
        (['"a"'], '"a"'),
        (['"size of "', '"float"', '" is 4"'], '"size of float is 4"'),
        (['"a"', '""', '"b"'], '"ab"'),
        # A hexadecimal escape would absorb the next piece's leading hex digit.
        (['"\\x4"', '"1"'], '"\\0041"'),
        (['"\\x4"', '""', '"f"'], '"\\004f"'),
        (['"\\x4"', '"g"'], '"\\x4g"'),
        # A short octal escape would absorb a leading octal digit; a full one cannot.
        (['"\\1"', '"2"'], '"\\0012"'),
        (['"\\101"', '"2"'], '"\\1012"'),
        (['"\\1"', '"8"'], '"\\18"'),
        # An escaped backslash before a digit is not an escape that can grow.
        (['"\\\\"', '"1"'], '"\\\\1"'),
        # A raw literal contributes its characters, escaped for an ordinary literal.
        (['R"(a"b\\c)"', '"d"'], '"a\\"b\\\\cd"'),
        (['R"x(line\nnext)x"', '"!"'], '"line\\nnext!"'),
    ],
)
def test_concatenation_preserves_characters(spellings, expected):
    assert concatenate_string_literals(spellings) == expected


def test_concatenation_rejects_wide_escape_before_hex_digit():
    with pytest.raises(ValueError, match="narrow character"):
        concatenate_string_literals(['"\\x100"', '"1"'])


def _static_asserts(code):
    ast = MetalParser(MetalLexer(code, preprocess=False).tokenize()).parse()
    found = []

    def visit(node, depth=0):
        if depth > 40:
            return
        if isinstance(node, StaticAssertNode):
            found.append(node)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                visit(item, depth + 1)
        elif hasattr(node, "__dict__"):
            for value in vars(node).values():
                visit(value, depth + 1)

    visit(ast)
    return found


def test_static_assert_message_is_one_literal():
    asserts = _static_asserts(
        'static_assert(true, "first " "second");\n'
        "kernel void k() {\n"
        '    static_assert(true, R"(raw)" " tail");\n'
        "}\n"
    )
    assert sorted({node.message for node in asserts}) == [
        '"first second"',
        '"raw tail"',
    ]


def test_macro_built_message_translates_to_crossgl(tmp_path):
    source = tmp_path / "messages.metal"
    source.write_text(KERNEL, encoding="utf-8")
    code = crosstl.translate(str(source), backend="cgl")
    assert '"size of float is not 4"' in code
    assert '"int is four bytes"' in code


_VALIDATORS = {
    "opengl": (
        "messages.comp",
        [["glslangValidator", "-S", "comp", "messages.comp"]],
    ),
    "vulkan": (
        "messages.spvasm",
        [
            ["spirv-as", "--target-env", "vulkan1.1", "messages.spvasm", "-o", "m.spv"],
            ["spirv-val", "--target-env", "vulkan1.1", "m.spv"],
        ],
    ),
    "directx": (
        "messages.hlsl",
        [["dxc", "-T", "cs_6_0", "-E", "CSMain", "messages.hlsl"]],
    ),
    "metal": (
        "messages.metal",
        [["xcrun", "-sdk", "macosx", "metal", "-c", "messages.metal", "-o", "m.air"]],
    ),
}


@pytest.mark.parametrize("backend", sorted(_VALIDATORS))
def test_macro_built_message_reaches_every_target(tmp_path, backend):
    file_name, commands = _VALIDATORS[backend]
    source = tmp_path / "source.metal"
    source.write_text(KERNEL, encoding="utf-8")
    code = crosstl.translate(str(source), backend=backend)
    missing = [command[0] for command in commands if shutil.which(command[0]) is None]
    if missing:
        pytest.skip(f"{', '.join(missing)} not installed")
    (tmp_path / file_name).write_text(code, encoding="utf-8")
    for command in commands:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300, cwd=tmp_path
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]

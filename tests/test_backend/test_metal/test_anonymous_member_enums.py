"""Anonymous enumerations inside a structure import as constants of that structure.

``struct traits { enum { qk = 32, b1 = qk / 4 }; };`` read through a template parameter (``T::qk``) produced a
CrossGL member ``enum;`` that no parser accepts and a reference to an undeclared ``traits::qk``. Each case below
fails on that code.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser, MetalParserError

SOURCE = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    struct traits {
        typedef ushort2 quants;
        enum { qk = 32, b1 = qk / 4, next, last = 0x10 };
        enum : uint { mask = 7u };
    };

    template <typename T>
    kernel void scale(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
        out[i] = out[i] * float(T::qk) + float(T::b1) + float(T::next) + float(T::last + T::mask);
    }

    template [[host_name("scale_traits")]] kernel void scale<traits>(device float*, uint);
""")


def _members(source):
    ast = MetalParser(
        list(MetalLexer(source, preprocess=False).token_generator())
    ).parse()
    struct = next(node for node in ast.structs if node.name == "traits")
    return {member.name: member for member in struct.members}


def test_enumerators_become_static_constant_members():
    members = _members(SOURCE.split("template <typename T>")[0])

    assert set(members) == {"qk", "b1", "next", "last", "mask"}
    for member in members.values():
        assert member.qualifiers == ["static", "constant", "constexpr"]
    assert members["qk"].vtype == "int" and members["mask"].vtype == "uint"
    # ``next`` has no value: it is one more than ``b1``.
    assert repr(members["next"].default_value).startswith(
        "BinaryOpNode(left=VariableNode"
    )


def test_an_enumerator_wider_than_int_is_refused():
    with pytest.raises(MetalParserError, match="does not fit int"):
        _members("struct wide { enum { big = 0x80000000 }; };")


def test_a_declarator_after_the_enumeration_is_not_lowered():
    # A member of the unnamed enumeration type is not a constant; it stays a typed parse failure.
    with pytest.raises(MetalParserError):
        _members("struct flags { enum { on = 1 } state; };")


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl", "vulkan", "cgl"])
def test_reads_through_a_template_parameter_fold_to_the_values(tmp_path, backend):
    path = tmp_path / "scale.metal"
    path.write_text(SOURCE)

    output = crosstl.translate(str(path), backend=backend, format_output=False)

    assert "enum;" not in output and "traits::qk" not in output
    if backend in {"metal", "directx", "opengl", "cgl"}:
        assert "float(32)" in output and "float(8)" in output and "float(9)" in output
    if backend == "metal" and shutil.which("xcrun"):
        artifact = tmp_path / "out.metal"
        artifact.write_text(output)
        result = subprocess.run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(artifact),
                "-o",
                "/dev/null",
            ],
            capture_output=True,
            text=True,
        )
        if "unable to find utility" not in result.stderr:
            assert result.returncode == 0, result.stderr

"""The extent of ``array<T, N>`` is a value, wherever the template spelling is written.

A bare name in the extent position parsed as a type argument, so ``array<f32, N>`` with ``const int N = 4`` was
refused as the undeclared type ``N``, and a generic value parameter substituted into it became the type ``3``
(CUDA output printed ``NamedType(name=3)``). In a ``var<...>`` declaration the extent was a single token, so
``var<workgroup> s: array<f32, (A + B)>`` did not parse. ``T name[N]`` handled all three.

The shared integer-constant collector that sizes arrays knew C spellings (``int``, ``int32_t``) but not CrossGL's
``i8``..``i64`` and ``u8``..``u64``, so SPIR-V refused ``const i32 N = 4; var t: f32[N];``.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.ast import ArrayType, IdentifierNode
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

CONSTANT_EXTENT = """
const int N = 4;
@compute @workgroup_size(1, 1, 1)
fn k(@group(0) @binding(0) var<storage, read_write> o: array<f32>) {
    var t: array<f32, N>;
    t[0] = 1.0;
    t[3] = 2.0;
    o[0] = t[0] + t[3];
}
"""

WORKGROUP_EXPRESSION_EXTENT = """
const int A = 2;
const int B = 1;
@compute @workgroup_size(1, 1, 1)
fn k(@group(0) @binding(0) var<storage, read_write> o: array<f32>) {
    var<workgroup> s: array<f32, (A + B)>;
    s[2] = 1.0;
    o[0] = s[2];
}
"""

GENERIC_VALUE_EXTENT = """
generic<N> f32 first(f32 x) {
    var local: array<f32, N>;
    local[0] = x;
    return local[0];
}
@compute @workgroup_size(1, 1, 1)
fn k(@group(0) @binding(0) var<storage, read_write> out: array<f32>) {
    out[0] = first<3>(1.0f);
}
"""

TARGETS = ("opengl", "directx", "metal", "cuda", "vulkan")


def _parse(code):
    return Parser(Lexer(code).tokens).parse()


def _local_types(ast):
    function = next(f for f in ast.functions if f.name == "k")
    return {
        s.name: s.var_type
        for s in function.body.statements
        if getattr(s, "var_type", None) is not None
    }


def _translate(tmp_path, source, backend):
    path = tmp_path / "extent.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_a_named_extent_is_a_value():
    local = _local_types(_parse(CONSTANT_EXTENT))["t"]
    assert isinstance(local, ArrayType)
    assert isinstance(local.size, IdentifierNode) and local.size.name == "N"


def test_a_named_extent_inside_a_nested_template_is_a_value():
    node = Parser(Lexer("array<array<int, M>, N>").tokens).parse_type()
    assert isinstance(node, ArrayType) and isinstance(node.size, IdentifierNode)
    assert isinstance(node.element_type, ArrayType)
    assert isinstance(node.element_type.size, IdentifierNode)
    assert node.element_type.size.name == "M"


def test_an_address_space_declaration_takes_an_expression_extent():
    shared = _local_types(_parse(WORKGROUP_EXPRESSION_EXTENT))["s"]
    assert isinstance(shared, ArrayType)
    assert not isinstance(shared.size, (str, IdentifierNode))
    assert {"A", "B"} <= {getattr(node, "name", None) for node in _walk(shared.size)}


def test_an_address_space_declaration_keeps_a_single_token_extent():
    source = WORKGROUP_EXPRESSION_EXTENT.replace("(A + B)", "64")
    shared = _local_types(_parse(source))["s"]
    assert isinstance(shared, ArrayType) and str(shared.size) == "64"


def _assert_valid_spirv(tmp_path, code):
    assert "OpTypeArray" in code
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        return
    asm = tmp_path / "out.spvasm"
    asm.write_text(code)
    binary = tmp_path / "out.spv"
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def _walk(node):
    yield node
    for child in node.child_nodes():
        yield from _walk(child)


@pytest.mark.parametrize("backend", TARGETS)
def test_a_constant_extent_translates(tmp_path, backend):
    code = _translate(tmp_path, CONSTANT_EXTENT, backend)
    if backend == "vulkan":
        _assert_valid_spirv(tmp_path, code)
    else:
        assert re.search(r"\bt\[(N|4)\]", code), code


@pytest.mark.parametrize("backend", TARGETS)
def test_an_expression_extent_of_workgroup_memory_translates(tmp_path, backend):
    code = _translate(tmp_path, WORKGROUP_EXPRESSION_EXTENT, backend)
    if backend == "vulkan":
        assert "Workgroup" in code
        _assert_valid_spirv(tmp_path, code)
    else:
        assert re.search(r"s\[\(?(A \+ B|2 \+ 1|3)\)?\]", code), code


@pytest.mark.parametrize("backend", TARGETS)
def test_a_generic_value_extent_is_substituted(tmp_path, backend):
    code = _translate(tmp_path, GENERIC_VALUE_EXTENT, backend)
    assert "NamedType" not in code
    if backend == "vulkan":
        _assert_valid_spirv(tmp_path, code)
    else:
        assert re.search(r"\blocal\[3\]", code), code


@pytest.mark.parametrize(
    ("declaration", "value"),
    [("i32", "4"), ("u32", "4u"), ("i64", "4"), ("u16", "4")],
)
def test_a_sized_integer_constant_is_a_known_extent(declaration, value):
    # The shared integer-constant collector knew C spellings (int, int32_t, uint) but not CrossGL's own i32/u32,
    # so SPIR-V refused ``const i32 N = 4; var t: f32[N];`` as an unresolved extent.
    from crosstl.translator.codegen.array_utils import collect_literal_int_constants

    source = CONSTANT_EXTENT.replace(
        "const int N = 4;", f"const {declaration} N = {value};"
    )
    constants = collect_literal_int_constants(_parse(source).constants)
    assert int(constants["N"]) == 4


@pytest.mark.parametrize("declaration", ["i32", "u32"])
def test_spirv_sizes_an_array_by_a_sized_integer_constant(tmp_path, declaration):
    source = CONSTANT_EXTENT.replace(
        "const int N = 4;",
        f"const {declaration} N = 4{'u' if declaration == 'u32' else ''};",
    ).replace("array<f32, N>", "f32[N]")
    _assert_valid_spirv(tmp_path, _translate(tmp_path, source, "vulkan"))

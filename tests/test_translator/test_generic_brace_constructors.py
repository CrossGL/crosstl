"""Braced constructors of generic types: ``vec<T, 2>{x, y}`` and ``Pair<float>{a, b}``.

MSL writes a vector or structure value as a braced list after its type, and the Metal importer keeps a defaulted
template such as ``template <typename IdxT = int> vec<IdxT, 2> f(...)`` as a CrossGL generic, so CrossGL gets
``vec<IdxT, 2>{...}``. The CrossGL parser read ``vec<IdxT, 2>`` there as a relational chain and refused the
function ("Expected binding name, got NUMBER"); for ``Pair<float>{...}`` it read ``Pair < float`` and emitted the
comparison. A generic vector's braced list now reads as its constructor call, and any other generic type's as an
aggregate constructor of that type.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.ast import ConstructorNode, FunctionCallNode, NamedType
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

GENERIC_VECTOR = """
shader S {
    generic<IdxT = float> vec<IdxT, 2> twice(int x) {
        vec<IdxT, 2> loc = vec<IdxT, 2>{IdxT(x), IdxT(x + 1)};
        return loc;
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            vec<float, 2> a = vec<float, 2>{1.0, 2.0};
            float2 b = twice<float>(3) + a;
        }
    }
}
"""

GENERIC_STRUCT = """
shader S {
    generic<T> struct Pair {
        T a;
        T b;
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            Pair<float> p = Pair<float>{1.0, 2.0};
            float s = p.a + p.b;
        }
    }
}
"""

# A defaulted template the Metal importer keeps as a CrossGL generic, as MLX's elem_to_loc helpers are.
METAL_DEFAULTED_TEMPLATE = """#include <metal_stdlib>
using namespace metal;
template <typename IdxT = int>
vec<IdxT, 2> loc2(uint3 elem, constant int* strides) {
    vec<IdxT, 2> loc = vec<IdxT, 2>{IdxT(elem.x) * IdxT(strides[0]), IdxT(elem.y) * IdxT(strides[1])};
    return loc;
}
kernel void locs(device int2* out [[buffer(0)]], constant int* strides [[buffer(1)]],
                 uint3 id [[thread_position_in_grid]]) {
    out[id.x] = loc2(id, strides);
}
"""

TOOLS = {
    "directx": "dxc",
    "opengl": "glslangValidator",
    "metal": "xcrun",
    "vulkan": "spirv-val",
}


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, cwd):
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _compile(tmp_path, code, backend):
    """Compile the output with the native tool when it is installed."""
    if shutil.which(TOOLS[backend]) is None:
        return
    if backend == "directx":
        (tmp_path / "out.hlsl").write_text(code)
        _run(["dxc", "-T", "cs_6_0", "-E", "CSMain", "out.hlsl"], tmp_path)
    elif backend == "opengl":
        (tmp_path / "out.comp").write_text(code)
        _run(["glslangValidator", "-S", "comp", "out.comp"], tmp_path)
    elif backend == "metal":
        (tmp_path / "out.metal").write_text(code)
        _run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", "out.metal", "-o", "out.air"],
            tmp_path,
        )
    elif shutil.which("spirv-as"):
        (tmp_path / "out.spvasm").write_text(code)
        _run(
            ["spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"],
            tmp_path,
        )
        _run(["spirv-val", "--target-env", "vulkan1.1", "out.spv"], tmp_path)


def _initializer(function, index=0):
    return function.body.statements[index].initial_value


def test_generic_vector_braced_list_parses_as_its_constructor_call():
    program = _parse(GENERIC_VECTOR)
    value = _initializer(program.functions[0])
    assert isinstance(value, FunctionCallNode)
    assert value.function.name == "vec<IdxT, 2>"
    assert len(value.arguments) == 2


def test_generic_struct_braced_list_parses_as_an_aggregate_constructor():
    program = _parse(GENERIC_STRUCT)
    stage = next(iter(program.stages.values()))
    value = stage.entry_point.body.statements[0].initial_value
    assert isinstance(value, ConstructorNode)
    assert isinstance(value.constructor_type, NamedType)
    assert value.constructor_type.name == "Pair"
    assert [arg.name for arg in value.constructor_type.generic_args] == ["float"]
    assert len(value.arguments) == 2


def test_relational_chains_still_parse_as_comparisons():
    source = GENERIC_VECTOR.replace(
        "float2 b = twice<float>(3) + a;", "bool c = 1 < 2 && 3 > 2;"
    )
    stage = next(iter(_parse(source).stages.values()))
    assert stage.entry_point.body.statements[1].initial_value.op == "&&"


@pytest.mark.parametrize(
    ("backend", "spellings"),
    [
        (
            "metal",
            ("vec<float, 2> a = vec<float, 2>(1.0, 2.0);", "vec<float, 2>(float(x)"),
        ),
        ("directx", ("float2 a = float2(1.0, 2.0);", "float2(float(x)")),
        ("opengl", ("vec2 a = vec2(1.0, 2.0);", "vec2(float(x)")),
        ("vulkan", ("OpCompositeConstruct",)),
    ],
)
def test_generic_vector_braced_lists_compile_for_each_target(
    tmp_path, backend, spellings
):
    code = _translate(tmp_path, GENERIC_VECTOR, backend)
    for spelling in spellings:
        assert spelling in code
    assert "ConstructorNode" not in code
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize(
    ("backend", "spelling"),
    [
        ("metal", "Pair_float p = Pair_float{1.0, 2.0};"),
        ("directx", "p.b = 2.0;"),
        ("opengl", "Pair_float p = Pair_float(1.0, 2.0);"),
    ],
)
def test_generic_struct_braced_lists_compile(tmp_path, backend, spelling):
    code = _translate(tmp_path, GENERIC_STRUCT, backend)
    assert spelling in code
    assert "Pair <" not in code and "Pair<" not in code
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize(
    ("backend", "spelling"),
    [
        ("metal", "vec<int, 2>(int(elem.x) * int(strides[0])"),
        ("directx", "int2 loc = int2("),
        ("opengl", "ivec2 loc = ivec2("),
        ("vulkan", "OpCompositeConstruct"),
    ],
)
def test_metal_defaulted_template_with_braced_vector_translates(
    tmp_path, backend, spelling
):
    code = _translate(tmp_path, METAL_DEFAULTED_TEMPLATE, backend, suffix=".metal")
    assert spelling in " ".join(code.split()) or spelling in code
    _compile(tmp_path, code, backend)

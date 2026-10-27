"""``array<T, N>`` with a nonzero extent is the array type ``T name[N]`` declares, wherever a type is written.

The CUDA and HIP importers spell C++ arrays ``array<T, N>``. CrossGL parsed that spelling as an array only for
resource parameters and ``var<...>`` declarations; for parameters, locals, struct members and return types it became
a ``NamedType`` named ``array``, so GLSL, HLSL and Slang wrote ``array<i32, 2>`` verbatim and SPIR-V refused it.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.ast import ArrayType, NamedType, PrimitiveType
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

TEMPLATE_CGL = """
shader S {
    struct Cell {
        array<float, 2> v;
        array<array<int, 3>, 2> grid;
    }
    compute {
        buffer float outBuf[];
        float total(array<float, 2> v) { return v[0] + v[1]; }
        array<float, 2> pair(float x) {
            array<float, 2> r;
            r[0] = x;
            r[1] = x * 2.0;
            return r;
        }
        layout(local_size_x = 1) in;
        void main() {
            Cell c;
            array<float, 2> w = pair(1.0);
            c.v = w;
            c.grid[1][2] = 5;
            outBuf[0] = total(c.v) + float(c.grid[1][2]);
        }
    }
}
"""

CUDA_SOURCE = """
struct Window { int taps[3]; };
__device__ int sum3(const int v[3]) { return v[0] + v[1] + v[2]; }
__global__ void k(const int* inp, int* outp) {
    int a[3];
    a[0] = inp[0]; a[1] = inp[1]; a[2] = inp[2];
    Window w;
    w.taps[0] = a[2];
    w.taps[1] = a[1];
    w.taps[2] = a[0];
    outp[0] = sum3(a) + sum3(w.taps);
}
"""


def _parse(code):
    return Parser(Lexer(code).tokens).parse()


def _shape(type_node):
    sizes = []
    while isinstance(type_node, ArrayType):
        size = type_node.size
        sizes.append(getattr(size, "value", size))
        type_node = type_node.element_type
    return sizes, type_node


def _translate(tmp_path, source, name, backend, **options):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, **options)


def test_template_spelling_parses_to_array_type_in_every_position():
    ast = _parse(TEMPLATE_CGL)
    cell = next(s for s in ast.structs if s.name == "Cell")
    members = {m.name: m.member_type for m in cell.members}
    assert _shape(members["v"])[0] == [2]
    grid_sizes, grid_element = _shape(members["grid"])
    assert grid_sizes == [2, 3]
    assert isinstance(grid_element, PrimitiveType)

    stage = next(iter(ast.stages.values()))
    functions = {f.name: f for f in stage.local_functions}
    parameter_type = functions["total"].parameters[0].param_type
    assert _shape(parameter_type)[0] == [2]
    assert _shape(functions["pair"].return_type)[0] == [2]
    local_type = functions["pair"].body.statements[0].var_type
    assert _shape(local_type)[0] == [2]


def test_unsized_template_and_declarator_suffixes_compose():
    ast = _parse("""
    shader S {
        compute {
            void main() {
                array<float, 3> b[2];
            }
        }
    }
    """)
    entry = next(iter(ast.stages.values())).entry_point
    sizes, element = _shape(entry.body.statements[0].var_type)
    # A declarator's suffixes enclose the written type: two arrays of array<float, 3>.
    assert sizes == [2, 3]
    assert isinstance(element, PrimitiveType)


@pytest.mark.parametrize(
    "spelling", ["array<float, 0>", "array<float, 3 - 3>", "array<uint>"]
)
def test_zero_extent_and_unsized_templates_keep_the_standard_array_type(spelling):
    # A zero-extent standard array (MSL array<T, 0>) has no C-style declarator, and array<T> takes its meaning
    # from where it is declared, so both stay the ``array`` template type.
    node = Parser(Lexer(spelling).tokens).parse_type()
    assert isinstance(node, NamedType) and node.name == "array"


def test_qualified_pointer_elements_parse_inside_the_template():
    node = Parser(Lexer("array<const device float*, 2>").tokens).parse_type()
    assert isinstance(node, ArrayType)
    assert getattr(node.size, "value", None) == 2
    assert getattr(node.element_type, "address_space", None) == "device"


def test_a_type_named_like_array_without_arguments_stays_named():
    parser = Parser(Lexer("array value").tokens)
    assert isinstance(parser.parse_type(), NamedType)


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (
            "opengl",
            ["float total(float v[2])", "float[2] pair(float x)", "int grid[2][3];"],
        ),
        ("directx", ["float total(float v[2])", "int grid[2][3];"]),
        ("slang", ["float total(float v[2])", "float[2] pair(float x)"]),
        ("metal", ["float total(float v[2])", "int grid[2][3];"]),
    ],
)
def test_generators_declare_template_arrays(tmp_path, backend, expected):
    code = _translate(
        tmp_path, TEMPLATE_CGL, "template.cgl", backend, format_output=False
    )
    if backend != "metal":
        # MSL spells a returned array metal::array<T, N>; the other targets have no array<> type.
        assert "array<" not in code
    compact = " ".join(code.split())
    for fragment in expected:
        assert fragment in compact, fragment
    if backend == "metal" and shutil.which("xcrun"):
        source = tmp_path / "template.metal"
        source.write_text(code)
        result = subprocess.run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(source),
                "-o",
                str(tmp_path / "t.air"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_spirv_lowers_template_arrays(tmp_path):
    code = _translate(
        tmp_path, TEMPLATE_CGL, "template.cgl", "vulkan", format_output=False
    )
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


def test_cuda_arrays_reach_glsl_and_hlsl_as_c_arrays(tmp_path):
    crossgl = _translate(tmp_path, CUDA_SOURCE, "arrays.cu", "cgl")
    assert "array<i32, 3>" in crossgl
    (tmp_path / "arrays.cgl").write_text(crossgl)
    glsl = crosstl.translate(
        str(tmp_path / "arrays.cgl"), backend="opengl", format_output=False
    )
    hlsl = crosstl.translate(
        str(tmp_path / "arrays.cgl"), backend="directx", format_output=False
    )
    for code in (glsl, hlsl):
        assert "array<" not in code
        compact = " ".join(code.split())
        assert "int sum3(int v[3])" in compact
        assert "int taps[3];" in compact
    if shutil.which("glslangValidator"):
        shader = tmp_path / "arrays.comp"
        shader.write_text(glsl)
        result = subprocess.run(
            ["glslangValidator", "-S", "comp", str(shader)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


MEMBER_ARGUMENT_CGL = """
shader S {
    struct Cell { float v[2]; }
    compute {
        buffer float outBuf[];
        float total(float v[2]) { v[0] = v[0] * 2.0; return v[0] + v[1]; }
        layout(local_size_x = 1) in;
        void main() {
            Cell c;
            float rows[2][2];
            c.v[0] = 1.0;
            c.v[1] = 2.0;
            rows[1][0] = 3.0;
            rows[1][1] = 4.0;
            outBuf[0] = total(c.v) + total(rows[1]) + c.v[0];
        }
    }
}
"""


def test_spirv_passes_member_and_element_arrays_through_their_own_variable(tmp_path):
    # SPIR-V logical addressing takes only memory object declarations as OpFunctionCall pointer operands, so a
    # by-value array held in a member or an element is copied to a Function variable before the call.
    code = _translate(
        tmp_path, MEMBER_ARGUMENT_CGL, "member.cgl", "vulkan", format_output=False
    )
    access_chains = set()
    variables = set()
    call_operands = []
    for line in code.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "=":
            if parts[2] == "OpAccessChain":
                access_chains.add(parts[0])
            elif parts[2] == "OpVariable":
                variables.add(parts[0])
            elif parts[2] == "OpFunctionCall":
                call_operands.extend(parts[5:])
    assert len(call_operands) == 2
    assert all(operand in variables for operand in call_operands)
    assert not access_chains.intersection(call_operands)
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        return
    asm = tmp_path / "member.spvasm"
    asm.write_text(code)
    binary = tmp_path / "member.spv"
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

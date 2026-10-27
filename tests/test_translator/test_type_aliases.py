"""A type alias resolves to the type it names, at program, stage and block scope, for every target."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import (
    TypeAliasUnsupportedError,
    apply_rule_carrying,
)
from crosstl.translator.semantics.type_aliases import (
    resolve_type_aliases,
    type_constructor_name,
)

ALL_TARGETS = [
    "metal",
    "directx",
    "opengl",
    "vulkan",
    "wgsl",
    "slang",
    "cuda",
    "hip",
    "rust",
    "mojo",
]
RESOLVING_TARGETS = ["vulkan", "wgsl", "slang", "cuda", "hip", "rust", "mojo"]

# The reproducer of the WGSL and SPIR-V failures: a program-scope alias as a declaration type and a conversion.
PROGRAM_ALIAS = """
shader P {
    typedef float real;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            real x = real(iin[i]);
            outp[i] = x * 2.0;
        }
    }
}
"""

# Aliases at every scope: chains, a structure alias, a stage alias, a block alias and a cast through an alias.
SCOPED_ALIASES = """
shader P {
    typedef float real;
    typedef vec3 real3;
    typedef real3 pt3;
    struct Pair { real a; real b; }
    typedef Pair Couple;
    real twice(real v) { return v * 2.0; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;
        typedef uint index;

        void main() {
            index i = gl_GlobalInvocationID.x;
            typedef int count;
            count c = count(iin[i]);
            pt3 p = pt3(real(c), 1.0, 2.0);
            Couple k = Couple(p.x, twice(p.y));
            real x = real(iin[i]);
            outp[i] = x * 2.0 + k.a + k.b + (real)c;
        }
    }
}
"""

# Sibling blocks declare the same alias name for different types; an array alias is sized and constructed.
BLOCK_ALIASES = """
shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            typedef float arr4[4];
            uint n = sizeof(arr4);
            arr4 a = arr4(1.0, 2.0, 3.0, 4.0);
            if (n > 0u) {
                typedef int T;
                T q = T(1.5);
                outp[2] = float(q);
            }
            if (n > 1u) {
                typedef float T;
                T q = T(1.5);
                outp[3] = q;
            }
            outp[0] = a[1] + float(n);
        }
    }
}
"""

QUALIFIED_ALIAS = """
shader P {
    typedef const float cf;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            cf x = 1.0;
            outp[0] = x;
        }
    }
}
"""

ALIAS_NAMES = ("real", "real3", "pt3", "Couple", "index", "count", "arr4", "T")


def _write(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _code(text):
    """``text`` without comments and helper preludes the generators emit before the program."""
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def _alias_mentions(text, names=ALIAS_NAMES):
    code = _code(text)
    return sorted(name for name in names if re.search(rf"\b{name}\b", code))


def _program_body(text):
    """The part of a CUDA or HIP translation after its matrix and vector helper prelude."""
    marker = text.rfind("struct Pair")
    if marker < 0:
        marker = text.rfind("int* iin")
    return text[marker:] if marker >= 0 else text


@pytest.mark.parametrize("target", RESOLVING_TARGETS)
def test_program_alias_resolves_where_the_generator_declares_none(tmp_path, target):
    output = crosstl.translate(_write(tmp_path, "alias.cgl", PROGRAM_ALIAS), backend=target)
    assert _alias_mentions(_program_body(output), ("real",)) == []


def test_wgsl_program_alias_is_not_a_private_variable(tmp_path):
    output = crosstl.translate(_write(tmp_path, "alias.cgl", PROGRAM_ALIAS), backend="wgsl")
    assert "var<private> real" not in output
    assert re.search(r"var x\s*:\s*f32\s*=\s*f32\(", output)


def test_spirv_translates_a_conversion_through_a_program_alias(tmp_path):
    output = crosstl.translate(_write(tmp_path, "alias.cgl", PROGRAM_ALIAS), backend="vulkan")
    assert "OpConvertSToF" in output


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_scoped_aliases_leave_no_alias_name_in_code(tmp_path, target):
    output = crosstl.translate(_write(tmp_path, "scoped.cgl", SCOPED_ALIASES), backend=target)
    assert _alias_mentions(_program_body(output), ALIAS_NAMES) == []


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_stage_and_block_aliases_are_not_variables(tmp_path, target):
    output = crosstl.translate(_write(tmp_path, "scoped.cgl", SCOPED_ALIASES), backend=target)
    assert not re.search(r"\buint\s+index\s*;", output)
    assert not re.search(r"\bint\s+count\s*;", output)


def test_hlsl_resolves_an_alias_chain(tmp_path):
    output = crosstl.translate(_write(tmp_path, "scoped.cgl", SCOPED_ALIASES), backend="directx")
    assert re.search(r"\bfloat3\s+p\s*=\s*float3\(", output)
    assert re.search(r"\bPair\s+k\b", output)


@pytest.mark.parametrize("target", ["metal", "wgsl", "opengl"])
def test_sibling_block_aliases_keep_their_own_types(tmp_path, target):
    output = crosstl.translate(_write(tmp_path, "blocks.cgl", BLOCK_ALIASES), backend=target)
    integer = {"metal": "int", "opengl": "int", "wgsl": "i32"}[target]
    real = {"metal": "float", "opengl": "float", "wgsl": "f32"}[target]
    if target == "wgsl":
        assert re.search(rf"var q\s*:\s*{integer}\s*=\s*{integer}\(1\.5\)", output)
        assert re.search(rf"var q\s*:\s*{real}\s*=\s*{real}\(1\.5\)", output)
    else:
        assert re.search(rf"\b{integer}\s+q\s*=\s*{integer}\(1\.5\)", output)
        assert re.search(rf"\b{real}\s+q\s*=\s*{real}\(1\.5\)", output)
    assert _alias_mentions(output, ("arr4", "T")) == []


@pytest.mark.parametrize("target", RESOLVING_TARGETS)
def test_qualified_alias_is_refused(tmp_path, target):
    with pytest.raises(TypeAliasUnsupportedError) as raised:
        crosstl.translate(_write(tmp_path, "qualified.cgl", QUALIFIED_ALIAS), backend=target)
    assert raised.value.code.endswith("type-alias-unsupported")


def test_alias_of_a_type_without_a_constructor_name_is_refused_as_a_callee():
    program = _parse(PROGRAM_ALIAS)
    alias = program.global_variables[0]
    alias.var_type = A.NamedType("Box", generic_args=[A.PrimitiveType("float")])
    alias.vtype = alias.var_type
    with pytest.raises(TypeAliasUnsupportedError):
        resolve_type_aliases(program, "wgsl")


def test_crossgl_output_keeps_aliases():
    program = _parse(PROGRAM_ALIAS)
    assert resolve_type_aliases(program, "cgl") == 0
    assert program.global_variables[0].is_type_alias


def test_resolution_drops_every_alias_declaration():
    program = _parse(SCOPED_ALIASES)
    assert resolve_type_aliases(program, "wgsl") == 6
    assert not any(
        getattr(variable, "is_type_alias", False) for variable in program.global_variables
    )
    stage = next(iter(program.stages.values()))
    assert [variable.name for variable in stage.local_variables] == ["iin", "outp"]


def test_native_targets_keep_program_aliases_with_resolved_chains():
    program = _parse(SCOPED_ALIASES)
    apply_rule_carrying(program, "cgl", "directx")
    aliases = {
        variable.name: variable.var_type
        for variable in program.global_variables
        if getattr(variable, "is_type_alias", False)
    }
    assert set(aliases) == {"real", "real3", "pt3", "Couple"}
    assert isinstance(aliases["pt3"], A.VectorType)
    assert aliases["pt3"] is program.global_variables[2].vtype


@pytest.mark.parametrize(
    ("type_value", "expected"),
    [
        (A.PrimitiveType("float"), "float"),
        (A.VectorType(A.PrimitiveType("float"), 3), "vec3"),
        (A.VectorType(A.PrimitiveType("int"), 2), "ivec2"),
        (A.VectorType(A.PrimitiveType("uint"), 4), "uvec4"),
        (A.VectorType(A.PrimitiveType("bool"), 3), "bvec3"),
        (A.VectorType(A.PrimitiveType("double"), 2), "dvec2"),
        (A.MatrixType(A.PrimitiveType("float"), 3, 3), "mat3"),
        (A.MatrixType(A.PrimitiveType("float"), 4, 3), "mat3x4"),
        (A.MatrixType(A.PrimitiveType("double"), 2, 2), "dmat2"),
        (A.NamedType("Pair"), "Pair"),
        (A.ArrayType(A.PrimitiveType("float"), A.LiteralNode(4, A.PrimitiveType("int"))), "float[4]"),
        (A.VectorType(A.PrimitiveType("half"), 3), None),
        (A.NamedType("Box", generic_args=[A.PrimitiveType("float")]), None),
        (A.PointerType(A.PrimitiveType("float")), None),
    ],
)
def test_constructor_names_round_trip_through_the_parser(type_value, expected):
    assert type_constructor_name(type_value) == expected
    if expected is not None and "[" not in expected:
        parsed = _parse(f"shader P {{ typedef {expected} t; }}").global_variables[0].var_type
        assert repr(parsed) == repr(type_value)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("source", [PROGRAM_ALIAS, SCOPED_ALIASES, BLOCK_ALIASES])
def test_wgsl_output_validates_with_naga(tmp_path, source):
    if shutil.which("naga") is None:
        pytest.skip("naga is not installed")
    output = crosstl.translate(_write(tmp_path, "alias.cgl", source), backend="wgsl")
    path = tmp_path / "alias.wgsl"
    path.write_text(output)
    _run(["naga", str(path)])


@pytest.mark.parametrize("source", [PROGRAM_ALIAS, SCOPED_ALIASES, BLOCK_ALIASES])
def test_spirv_output_validates(tmp_path, source):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools is not installed")
    output = crosstl.translate(_write(tmp_path, "alias.cgl", source), backend="vulkan")
    assembly = tmp_path / "alias.spvasm"
    binary = tmp_path / "alias.spv"
    assembly.write_text(output)
    _run(["spirv-as", "--target-env", "vulkan1.0", str(assembly), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.0", str(binary)])


@pytest.mark.parametrize("source", [SCOPED_ALIASES, BLOCK_ALIASES])
def test_metal_output_compiles(tmp_path, source):
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    output = crosstl.translate(_write(tmp_path, "alias.cgl", source), backend="metal")
    path = tmp_path / "alias.metal"
    path.write_text(output)
    probe = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True, check=False
    )
    if probe.returncode != 0:
        pytest.skip("the Metal compiler is not installed")
    _run(["xcrun", "-sdk", "macosx", "metal", "-c", str(path), "-o", str(tmp_path / "alias.air")])


@pytest.mark.parametrize("source", [SCOPED_ALIASES, BLOCK_ALIASES])
def test_glsl_output_validates(tmp_path, source):
    if shutil.which("glslangValidator") is None:
        pytest.skip("glslangValidator is not installed")
    output = crosstl.translate(_write(tmp_path, "alias.cgl", source), backend="opengl")
    path = tmp_path / "alias.comp"
    path.write_text(output)
    _run(["glslangValidator", str(path)])

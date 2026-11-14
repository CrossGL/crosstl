"""SPIR-V output of module-scope constants that convert a specialization constant.

A constant that follows a specialization constant is a constant instruction, so each conversion in it must be an
``OpSpecConstantOp``. Under the ``Shader`` capability that covers integer signedness and width, floating-point width
and integer/Boolean conversions, but no integer/floating-point conversion (SPIR-V specification, ``OpSpecConstantOp``).
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused
from tests.spirv_friendly_names import friendly_spirv

PROGRAM = """shader Probe {
    const int SPEC_I @constant_id(0) = 3;
    const bool SPEC_B @constant_id(1) = true;
    const uint SPEC_U @constant_id(2) = 5u;
    const float SPEC_F @constant_id(3) = 1.5;
    PRELUDE
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = READ;
        }
    }
}
"""


def _spirv(tmp_path, prelude, read):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM.replace("PRELUDE", prelude).replace("READ", read))
    return friendly_spirv(
        crosstl.translate(str(path), backend="vulkan", format_output=False)
    )


def _definition(code, name):
    lines = [line.strip() for line in code.splitlines()]
    return next(line for line in lines if line.startswith(f"%{name} = "))


def _validate(tmp_path, code):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("spirv-as and spirv-val are not installed")
    assembly = tmp_path / "probe.spvasm"
    binary = tmp_path / "probe.spv"
    assembly.write_text(code)
    for command in (
        ["spirv-as", "--target-env", "vulkan1.0", str(assembly), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.0", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.parametrize(
    "prelude, read",
    [
        ("const vec2 v = vec2(SPEC_I, 1.0);", "v.x + v.y"),
        ("const float f = float(SPEC_I);", "f"),
        ("const vec2 v = vec2(SPEC_B, 1.0);", "v.x"),
        ("const int n = int(SPEC_F);", "float(n)"),
        ("struct S { float a; float b; };\n    const S s = S(SPEC_U, 2.0);", "s.a"),
        ("const float arr[2] = {SPEC_I, 1.0};", "arr[0]"),
    ],
)
def test_integer_floating_point_conversion_is_refused(tmp_path, prelude, read):
    with pytest.raises(SpirvLoweringRefused, match="Kernel-only"):
        _spirv(tmp_path, prelude, read)


def test_unsigned_widening_is_refused(tmp_path):
    with pytest.raises(SpirvLoweringRefused, match="OpUConvert"):
        _spirv(tmp_path, "const u64 w = u64(SPEC_U);", "float(w)")


def test_vector_constituent_changes_signedness_with_an_operation(tmp_path):
    code = _spirv(tmp_path, "const uvec2 v = uvec2(SPEC_I, 1u);", "float(v.x + v.y)")
    composite = _definition(code, "v").split()
    assert composite[2:4] == ["OpSpecConstantComposite", "%v2uint"]
    converted = composite[4].lstrip("%")
    assert _definition(code, converted).endswith(
        "OpSpecConstantOp %uint IAdd %SPEC_I %uint_0"
    )
    assert composite[5] == "%uint_1"
    _validate(tmp_path, code)


def test_explicit_scalar_conversions_follow_the_specialization_constant(tmp_path):
    prelude = "\n    ".join(
        [
            "const uint u = uint(SPEC_I);",
            "const int s = int(SPEC_U) * 2;",
            "const bool nz = bool(SPEC_I);",
            "const int b = int(SPEC_B);",
            "const i64 wide = i64(SPEC_I);",
            "const uint narrow = uint(u64(SPEC_I));",
            "const double d = double(SPEC_F);",
        ]
    )
    read = "float(u) + float(s) + (nz ? 1.0 : 0.0) + float(b) + float(wide) + float(narrow) + float(d)"
    code = _spirv(tmp_path, prelude, read)
    assert _definition(code, "u").endswith(
        "OpSpecConstantOp %uint IAdd %SPEC_I %uint_0"
    )
    assert "OpSpecConstantOp %int IMul" in _definition(code, "s")
    assert _definition(code, "nz").endswith(
        "OpSpecConstantOp %bool INotEqual %SPEC_I %int_0"
    )
    assert _definition(code, "b").endswith(
        "OpSpecConstantOp %int Select %SPEC_B %int_1 %int_0"
    )
    assert _definition(code, "wide").endswith("OpSpecConstantOp %long SConvert %SPEC_I")
    assert "OpSpecConstantOp %uint SConvert" in _definition(code, "narrow")
    assert _definition(code, "d").endswith("OpSpecConstantOp %double FConvert %SPEC_F")
    _validate(tmp_path, code)


def test_struct_and_array_constituents_are_converted(tmp_path):
    prelude = (
        "struct S { uint a; int b; };\n"
        "    const S s = S(SPEC_I, SPEC_I);\n"
        "    const uint arr[2] = {SPEC_I, 1u};\n"
        "    const uvec3 mixed = uvec3(SPEC_B, SPEC_I, SPEC_U);"
    )
    code = _spirv(
        tmp_path,
        prelude,
        "float(s.a) + float(s.b) + float(arr[0]) + float(mixed.x + mixed.y + mixed.z)",
    )
    for name in ("s", "arr", "mixed"):
        assert "OpSpecConstantComposite" in _definition(code, name)
    struct_members = _definition(code, "s").split()[4:]
    assert struct_members[1] == "%SPEC_I"
    assert _definition(code, struct_members[0].lstrip("%")).endswith(
        "IAdd %SPEC_I %uint_0"
    )
    _validate(tmp_path, code)


def test_function_scope_conversion_still_converts_at_run_time(tmp_path):
    path = tmp_path / "local.cgl"
    path.write_text(
        PROGRAM.replace("PRELUDE", "").replace(
            "outp[i] = READ;", "vec2 v = vec2(SPEC_I, 1.0);\n            outp[i] = v.x;"
        )
    )
    code = friendly_spirv(
        crosstl.translate(str(path), backend="vulkan", format_output=False)
    )
    assert "OpConvertSToF %float %SPEC_I" in code
    _validate(tmp_path, code)

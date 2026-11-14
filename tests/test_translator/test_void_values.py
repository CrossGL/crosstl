"""A ``void`` call has no value, and a cast to ``void`` lowers to the evaluation it stands for."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.semantics import (
    VoidCastError,
    VoidValueError,
    apply_rule_carrying,
)

TARGETS = ["metal", "directx", "opengl", "vulkan", "wgsl"]


def _compute(helpers, body):
    return f"""
shader S {{
    compute {{
        buffer int values[];
        int counter;
        void bump() {{ counter = counter + 1; }}
        void drop() {{ counter = counter - 1; }}
        {helpers}
        @compute @workgroup_size(1, 1, 1)
        void main() {{
            int x = values[1];
            {body}
            values[0] = counter + x;
        }}
    }}
}}
"""


def _write(tmp_path, source, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return str(path)


# Void values -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "body",
    [
        "x = x + bump();",
        "int y = bump();",
        "values[2] = x > 0 ? bump() : 1;",
        "x = (drop(), bump());",
    ],
)
@pytest.mark.parametrize("target", TARGETS)
def test_a_void_call_used_as_a_value_is_refused(body, target):
    with pytest.raises(VoidValueError) as raised:
        apply_rule_carrying(parse(_compute("", body)), "cgl", target)
    assert isinstance(raised.value, TranslationDiagnostic)
    assert raised.value.code.endswith("void-value")
    assert "'bump' returns void" in str(raised.value)


def test_a_void_call_returned_from_a_valued_function_is_refused():
    source = _compute("int wrap() { return bump(); }", "x = wrap();")
    with pytest.raises(VoidValueError, match="'bump' returns void"):
        apply_rule_carrying(parse(source), "cgl", "opengl")


@pytest.mark.parametrize(
    "body",
    [
        "bump();",
        "x > 0 ? bump() : drop();",
        "bump(), drop();",
        "for (int i = 0; i < 2; bump()) { i = i + 1; }",
    ],
)
def test_discarded_void_calls_are_accepted(body):
    apply_rule_carrying(
        parse(_compute("void relay() { return bump(); }", body)), "cgl", "opengl"
    )


def test_an_overload_set_with_a_valued_member_is_not_checked():
    helpers = "int bump(int step) { counter = counter + step; return counter; }"
    apply_rule_carrying(parse(_compute(helpers, "x = x + bump(2);")), "cgl", "vulkan")


# Casts to void -----------------------------------------------------------------------------------------------------
CASTS = _compute(
    "",
    """
            (void)x;
            void(x + 1);
            (void)bump();
            if (x > 2) void(bump());
            x > 0 ? bump() : void(drop());
            for (int i = 0; i < 2; void(i++)) { bump(); }
            int y = (void(drop()), x);
            values[3] = y;
    """,
)


@pytest.mark.parametrize("target", TARGETS)
def test_casts_to_void_lower_to_their_evaluations(tmp_path, target):
    output = crosstl.translate(_write(tmp_path, CASTS), backend=target)
    assert not re.search(r"\bvoid\s*\(", output.replace("(void)", "")), output
    if target == "vulkan":
        # Resolve the void type id: without SPIRV-Tools the ids are numeric.
        void_id = re.search(r"(%\w+) = OpTypeVoid\b", output).group(1)
        calls = re.findall(rf"OpFunctionCall {re.escape(void_id)}\s", output)
        assert len(calls) == 6, output
        return
    lines = [
        line
        for line in output.splitlines()
        if not re.match(r"\s*(void|fn)\s+(bump|drop)\b", line)
    ]
    calls = re.findall(r"\b(bump|drop)\(", "\n".join(lines))
    # Two statements, the conditional's two branches, the loop body and the comma operand.
    assert calls.count("bump") == 4, output
    assert calls.count("drop") == 2, output
    assert "x + 1" not in output


def test_a_void_statement_conditional_becomes_an_if(tmp_path):
    source = _compute("", "x > 0 ? bump() : drop();")
    output = crosstl.translate(_write(tmp_path, source), backend="wgsl")
    assert re.search(
        r"if \(+x > 0\)+\s*\{\s*bump\(\);\s*\}\s*else\s*\{\s*drop\(\);", output
    )


def test_crossgl_keeps_casts_to_void(tmp_path):
    output = crosstl.translate(_write(tmp_path, CASTS), backend="cgl")
    assert "void(bump())" in output


def test_a_cast_to_void_inside_an_expression_is_refused_without_metal():
    source = _compute("", "x = (x > 0 ? void(bump()) : void(drop()), x);")
    for target in ("opengl", "directx", "vulkan", "wgsl"):
        with pytest.raises(VoidCastError) as raised:
            apply_rule_carrying(parse(source), "cgl", target)
        assert raised.value.code.endswith(f"{target}-void-cast")
    apply_rule_carrying(parse(source), "cgl", "metal")


def _tool(name):
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name} is not installed")
    return path


def test_lowered_casts_validate_as_glsl(tmp_path):
    validator = _tool("glslangValidator")
    output = crosstl.translate(_write(tmp_path, CASTS), backend="opengl")
    shader = tmp_path / "casts.comp"
    shader.write_text(output)
    result = subprocess.run([validator, str(shader)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


def test_lowered_casts_validate_as_spirv(tmp_path):
    assembler, validator = _tool("spirv-as"), _tool("spirv-val")
    output = crosstl.translate(_write(tmp_path, CASTS), backend="vulkan")
    assembly, binary = tmp_path / "casts.spvasm", tmp_path / "casts.spv"
    assembly.write_text(output)
    subprocess.run(
        [assembler, "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)],
        check=True,
    )
    result = subprocess.run(
        [validator, "--target-env", "vulkan1.1", str(binary)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# SPIR-V operands without a value -----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expression, operator",
    [
        ("p.missing + 1", "'+'"),
        ("-p.missing", "'-'"),
    ],
)
def test_spirv_refuses_an_operand_without_a_value(expression, operator):
    source = f"""
shader S {{
    struct Pair {{ int a; }}
    compute {{
        buffer int values[];
        @compute @workgroup_size(1, 1, 1)
        void main() {{
            Pair p;
            p.a = 1;
            values[0] = {expression};
        }}
    }}
}}
"""
    with pytest.raises(SpirvLoweringRefused) as raised:
        VulkanSPIRVCodeGen().generate(parse(source))
    assert f"Operand of {operator} has no value" in str(raised.value)

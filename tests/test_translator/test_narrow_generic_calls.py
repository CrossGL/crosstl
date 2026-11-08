"""Generic calls deduce narrow-integer type arguments before narrow registers are widened.

SPIR-V and WGSL compute 8- and 16-bit integers in 32-bit registers. ``bump(a)`` with ``uint8 a`` deduces
``T = uint8`` (C++ [temp.deduct.call]), so it runs ``@explicit_specialization(bump, uint8)`` when the program
declares one ([temp.expl.spec]/1) and otherwise an instantiation of the primary whose ``T`` arithmetic wraps at
8 bits. The generators deduced from the widened registers instead and silently ran the primary for ``int``.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import NarrowIntegerUnsupportedError
from crosstl.translator.semantics.narrow import NARROW_TARGETS
from tests.spirv_friendly_names import friendly_spirv

TARGETS = sorted(NARROW_TARGETS)

KERNEL = """shader S {{
{helpers}
    compute {{
        @ stage_entry
        void k(RWStructuredBuffer<uint> q @buffer(0), uint id @gl_GlobalInvocationID) {{
{body}
        }}
    }}
}}
"""

SPECIALIZED = """
    generic<T> T bump(T x) { return x + T(1); }
    uint8 bump_u8(uint8 x) @explicit_specialization(bump, uint8) { return x + uint8(2); }
"""

NESTED = """
    generic<T> T bump(T x) { T y = x + T(1); return y; }
    generic<T> T outer(T x) { return bump(x) * T(2); }
"""


def translate(tmp_path, helpers, body, target):
    path = tmp_path / "kernel.cgl"
    path.write_text(KERNEL.format(helpers=helpers, body=body))
    output = crosstl.translate(str(path), backend=target, format_output=False)
    if target == "vulkan":
        # Friendly ids (``%int_255``) whether or not SPIRV-Tools formats the output.
        output = friendly_spirv(output)
    validate(tmp_path, output, target)
    return output


def validate(tmp_path, output, target):
    tools = {"vulkan": ("spirv-as", "spirv-val"), "wgsl": ("naga",)}[target]
    paths = [shutil.which(tool) for tool in tools]
    if None in paths:
        return
    if target == "wgsl":
        source = tmp_path / "out.wgsl"
        source.write_text(output)
        commands = [[paths[0], str(source)]]
    else:
        source, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        source.write_text(output)
        commands = [
            [paths[0], "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
            [paths[1], "--target-env", "vulkan1.1", str(binary)],
        ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr + output


def wgsl_functions(output):
    """``{name: body}`` of each WGSL function."""
    return {
        match.group(1): match.group(2)
        for match in re.finditer(r"^fn (\w+)\(.*?\{\n(.*?)^\}", output, re.M | re.S)
    }


def spirv_functions(output):
    """``{result id: instruction lines}`` of each SPIR-V function."""
    functions, current = {}, None
    for line in output.splitlines():
        words = line.split()
        if len(words) > 2 and words[2] == "OpFunction":
            current = functions.setdefault(words[0], [])
        elif current is not None:
            current.append(line.strip())
            if words and words[0] == "OpFunctionEnd":
                current = None
    return functions


def spirv_callees(lines):
    return [line.split()[4] for line in lines if " OpFunctionCall " in line]


def assert_spirv_callee_wraps(output, mask):
    """The entry point calls one helper, and that helper reduces its result modulo ``mask + 1``."""
    functions = spirv_functions(output)
    assert len(functions) == 2, list(functions)
    (entry,) = (lines for lines in functions.values() if spirv_callees(lines))
    (callee,) = spirv_callees(entry)
    assert any(f"%int_{mask}" in line for line in functions[callee]), functions[callee]


@pytest.mark.parametrize("target", TARGETS)
def test_narrow_call_runs_the_explicit_specialization(tmp_path, target):
    body = """
            uint8 a = uint8(q[id]);
            uint8 b = bump(a);
            q[id] = uint(b);"""
    output = translate(tmp_path, SPECIALIZED, body, target)
    if target == "wgsl":
        assert list(wgsl_functions(output)) == ["bump_u8", "compute_main"]
        assert "bump_u8(a)" in output
        return
    functions = spirv_functions(output)
    # The specialization and the entry point; the primary is not instantiated for ``int``.
    assert len(functions) == 2, list(functions)
    (entry,) = (lines for lines in functions.values() if spirv_callees(lines))
    (callee,) = spirv_callees(entry)
    assert any("%int_2" in line for line in functions[callee])


@pytest.mark.parametrize("target", TARGETS)
def test_narrow_call_instantiates_the_primary_at_its_width(tmp_path, target):
    body = """
            uint8 a = uint8(q[id]);
            int16 s = int16(q[id]);
            uint8 b = bump(a);
            uint8 c = outer(a);
            int16 d = outer(s);
            q[id] = uint(b) + uint(c) + uint(int(d) + 70000);"""
    output = translate(tmp_path, NESTED, body, target)
    if target == "wgsl":
        functions = wgsl_functions(output)
        assert sorted(functions) == [
            "bump_int16",
            "bump_uint8",
            "compute_main",
            "outer_int16",
            "outer_uint8",
        ]
        # ``T y = x + T(1)`` and the narrow return wrap at the width of ``T``.
        assert "& 255" in functions["bump_uint8"]
        assert "& 65535" in functions["bump_int16"]
        assert "bump_uint8(x)" in functions["outer_uint8"]
        assert "bump_int16(x)" in functions["outer_int16"]
        assert "& 255" in functions["outer_uint8"]
        return
    functions = spirv_functions(output)
    assert len(functions) == 5, list(functions)
    masks = {
        mask: sum(f"%int_{mask}" in " ".join(lines) for lines in functions.values())
        for mask in (255, 65535)
    }
    # ``bump`` and ``outer`` of each width, plus the entry point's conversions.
    assert masks[255] >= 2 and masks[65535] >= 2, masks


@pytest.mark.parametrize("target", TARGETS)
def test_explicit_narrow_type_argument_converts_the_argument(tmp_path, target):
    # Only the template argument names a narrow type; ``w`` converts to it ([temp.arg.explicit]/6).
    helpers = "    generic<T> T bump(T x) { return x + T(1); }"
    body = """
            int w = int(q[id]);
            q[id] = uint(bump<uint8>(w));"""
    output = translate(tmp_path, helpers, body, target)
    if target == "wgsl":
        functions = wgsl_functions(output)
        assert sorted(functions) == ["bump_uint8", "compute_main"]
        assert "bump_uint8((i32(w) & 255))" in output
        assert "& 255" in functions["bump_uint8"]
        return
    assert_spirv_callee_wraps(output, 255)


@pytest.mark.parametrize("target", TARGETS)
def test_vector_parameter_deduces_the_narrow_element(tmp_path, target):
    helpers = "    generic<T> T first(vec2<T> v) { return v.x + T(1); }"
    body = """
            u8vec2 v = u8vec2(uint8(q[id]), uint8(3));
            uint8 c = first(v);
            q[id] = uint(c);"""
    output = translate(tmp_path, helpers, body, target)
    if target == "wgsl":
        assert "fn first_uint8(v: vec2<i32>) -> i32" in output
        assert "first_uint8(v)" in output
        return
    assert_spirv_callee_wraps(output, 255)


@pytest.mark.parametrize("target", TARGETS)
def test_instantiation_names_avoid_declared_functions(tmp_path, target):
    helpers = """
    generic<T> T bump(T x) { return x + T(1); }
    uint bump_uint8(uint x) { return x; }"""
    body = """
            uint8 a = uint8(q[id]);
            q[id] = bump_uint8(uint(bump(a)));"""
    output = translate(tmp_path, helpers, body, target)
    if target == "wgsl":
        assert "fn bump_uint8_1(x: i32) -> i32" in output
        assert "bump_uint8(u32(bump_uint8_1(a)))" in output


@pytest.mark.parametrize("target", TARGETS)
def test_wide_generic_calls_stay_with_the_generator(tmp_path, target):
    helpers = "    generic<T> T bump(T x) { return x + T(1); }"
    body = """
            uint8 a = uint8(q[id]);
            int i = int(q[id]);
            q[id] = uint(bump(i)) + uint(bump(a));"""
    output = translate(tmp_path, helpers, body, target)
    if target == "wgsl":
        assert sorted(wgsl_functions(output)) == [
            "bump_i32",
            "bump_uint8",
            "compute_main",
        ]
        return
    assert len(spirv_functions(output)) == 3


REFUSED = {
    "conflicting-deduction": (
        "    generic<T> T add(T a, T b) { return a + b; }",
        "q[id] = uint(add(uint8(q[id]), int16(q[id])));",
        "deduce both uint8 and int16",
    ),
    "declaration-only-primary": (
        """
    generic<T> T f(T x);
    float f_float(float x) @explicit_specialization(f, float) { return x; }""",
        "q[id] = uint(f(uint8(q[id])));",
        "declared without a definition",
    ),
    "overloaded-name": (
        """
    generic<T> T bump(T x) { return x + T(1); }
    int bump(int x) { return x + 7; }""",
        "q[id] = uint(bump(uint8(q[id])));",
        "overloaded generic 'bump'",
    ),
}


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", sorted(REFUSED))
def test_unresolvable_narrow_generic_calls_are_refused(tmp_path, case, target):
    helpers, body, message = REFUSED[case]
    with pytest.raises(NarrowIntegerUnsupportedError, match=re.escape(message)):
        translate(tmp_path, helpers, "            " + body, target)

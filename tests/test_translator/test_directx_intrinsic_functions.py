"""HLSL output renames user functions that an HLSL intrinsic of the same name would capture.

DXC resolves a call against the intrinsic overloads and the program's functions together and prefers the intrinsic
where both take the arguments: beside ``float frac(float x)``, ``frac(1.5)`` reads 0.5. Where DXC is installed, these
tests compile each program and read the constant DXC folds into the store, which shows the function the call reached.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Mapping

import pytest

import crosstl.translator
from crosstl.translator.ast import ASTNode, FunctionCallNode
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.hlsl_intrinsic_functions import shadows_hlsl_intrinsic
from crosstl.translator.semantics.function_name_collisions import rename_functions
from crosstl.translator.semantics.hub_types import call_name


def _hlsl(source):
    return HLSLCodeGen().generate(crosstl.translator.parse(source))


def _stored_constants(generated, tmp_path, profile="cs_6_2"):
    """The float constants the compiled shader stores, or ``None`` without DXC."""
    compiler = shutil.which("dxc")
    if compiler is None:
        return None
    source_path = tmp_path / "intrinsic_names.hlsl"
    source_path.write_text(generated, encoding="utf-8")
    compiled = subprocess.run(
        [compiler, "-T", profile, "-E", "CSMain", "-HV", "2021", str(source_path)],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    stores = [line for line in compiled.stdout.splitlines() if "Store" in line]
    return [
        float(value)
        for line in stores
        for value in re.findall(r"float (-?\d+\.\d+e[+-]\d+)", line)
    ]


def _compute(helpers, body):
    return f"""
    shader Probe {{
        {helpers}
        compute {{
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer float* outp;
            void main() {{
                {body}
            }}
        }}
    }}
    """


def test_helpers_named_like_hlsl_intrinsics_reach_their_own_bodies(tmp_path):
    # The coverage probe intrinsic-name: DXC sent frac(x) and rsqrt(x) to the intrinsics.
    generated = _hlsl(
        _compute(
            "float fabs(float x) { return x * 0.5 + 1.0; }\n"
            "float lerp(float a, float b) { return a * 2.0 - b; }\n"
            "float frac(float x) { return x + 0.25; }\n"
            "float rsqrt(float x) { return x * x; }",
            "outp[0] = frac(1.5); outp[1] = rsqrt(4.0); outp[2] = lerp(3.0, 1.0); "
            "outp[3] = fabs(2.0);",
        )
    )
    for name in ("frac", "rsqrt", "lerp"):
        assert re.search(rf"\bfloat {name}_fn\(float", generated)
        assert not re.search(rf"\b{name}\(", generated)
    # fabs is no HLSL intrinsic, so it keeps its name.
    assert "float fabs(float x)" in generated
    stored = _stored_constants(generated, tmp_path)
    if stored is not None:
        assert stored == [1.75, 16.0, 5.0, 2.0]


def test_crossgl_builtins_written_under_an_intrinsic_name_keep_the_intrinsic(
    tmp_path,
):
    # CrossGL fract is HLSL frac: the helper frac no longer captures it.
    generated = _hlsl(
        _compute(
            "float frac(float x) { return x + 0.25; }",
            "outp[0] = fract(1.5); outp[1] = frac(1.5);",
        )
    )
    assert "frac_fn(1.5)" in generated
    assert re.search(r"(?<!_)\bfrac\(1\.5\)", generated)
    stored = _stored_constants(generated, tmp_path)
    if stored is not None:
        assert stored == [0.5, 1.75]


def test_calls_only_the_intrinsic_takes_keep_its_name(tmp_path):
    # max(x) has one parameter, the intrinsic two; dot(v, v) on vectors cannot reach a scalar helper.
    generated = _hlsl(
        _compute(
            "float max(float x) { return x + 1.0; }\n"
            "float dot(float a, float b) { return a - b; }",
            "vec2 v = vec2(1.0, 2.0); outp[0] = max(2.0); outp[1] = max(2.0, 3.0); "
            "outp[2] = dot(v, v); outp[3] = dot(5.0, 1.0);",
        )
    )
    assert "float max_fn(float x)" in generated
    assert "max_fn(2.0)" in generated
    assert re.search(r"(?<!_)\bmax\(2\.0, 3\.0\)", generated)
    assert "float dot_fn(float a, float b)" in generated
    assert re.search(r"(?<!_)\bdot\(v, v\)", generated)
    assert "dot_fn(5.0, 1.0)" in generated
    stored = _stored_constants(generated, tmp_path)
    if stored is not None:
        assert stored == [3.0, 3.0, 5.0, 4.0]


def test_signatures_dxc_sent_to_the_intrinsic_reach_the_helper(tmp_path):
    # Each helper takes what an intrinsic overload takes; DXC chose the intrinsic for every one.
    generated = _hlsl(
        _compute(
            "uint asuint(float value) { return uint(value); }\n"
            "bool isnormal(float v) { return v > 0.0; }\n"
            "bool min(bool l, bool r) { return l; }\n"
            "float abs(float x) { return x + 9.0; }",
            "outp[0] = float(asuint(1.0)); outp[1] = isnormal(-1.0) ? 1.0 : 0.0; "
            "outp[2] = min(true, false) ? 1.0 : 0.0; outp[3] = abs(-1.0);",
        )
    )
    for name in ("asuint", "isnormal", "min", "abs"):
        assert f"{name}_fn(" in generated
    stored = _stored_constants(generated, tmp_path)
    if stored is not None:
        assert stored == [1.0, 0.0, 1.0, 8.0]


def test_generic_primary_named_like_an_intrinsic_keeps_its_specializations():
    generated = _hlsl("""
        shader S {
            generic<T> T clip(T x) { return x; }
            vec4 clip_v(vec4 x) @explicit_specialization(clip, vec4) { return max(x, vec4(0.0)); }
            compute {
                void main() {
                    vec4 b = clip(vec4(1.0));
                    vec4 c = clip<vec4>(b);
                }
            }
        }
        """)
    assert generated.count("clip_v(") >= 3
    assert "clip<" not in generated and "clip_fn" not in generated


def _calls(node, seen=None):
    seen = set() if seen is None else seen
    if id(node) in seen:
        return
    seen.add(id(node))
    if isinstance(node, FunctionCallNode):
        yield node
    for value in vars(node).values():
        if isinstance(value, Mapping):
            items = list(value.values())
        elif isinstance(value, (list, tuple)):
            items = value
        else:
            items = [value]
        for item in items:
            if isinstance(item, ASTNode):
                yield from _calls(item, seen)


def test_renamed_functions_take_their_calls_in_module_scope_initializers():
    # A constant's initializer kept frac(1.5) after the helper was renamed, and so read the intrinsic.
    program = crosstl.translator.parse("""
        shader S {
            float frac(float x) { return x + 0.25; }
            const float folded = frac(1.5);
            compute {
                float staged = frac(2.5);
                void main() {}
            }
        }
        """)
    assert rename_functions(program, {"frac"}, by_arity=True) == {"frac": "frac_fn"}
    assert sorted(call_name(call) for call in _calls(program)) == ["frac_fn"] * 2


@pytest.mark.parametrize(
    "name,expected", [("frac", True), ("asfloat16", True), ("fabs", False)]
)
def test_shadowed_names_are_dxc_global_intrinsics(name, expected):
    assert shadows_hlsl_intrinsic(name) is expected

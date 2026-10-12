"""Precise helper returns retain arithmetic order after target compilation."""

import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from functools import partial

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from tests.test_translator.test_fused_math import _dispatch
from tests.test_translator.test_metal_additive_profile import GUARD, _oracle
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_ADDITIVE_PROFILE"

SOURCE = """shader PreciseReturns {
    StructuredBuffer<uint> values @register(t0);
    RWStructuredBuffer<uint> results @register(u1);
    @precise float add(float a, float b) { return a + b; }
    @precise float subtract(float a, float b) { return a - b; }
    @precise float selected(float a, float b, bool positive) {
        if (positive) { return (a + b) - a; }
        return (a - b) + b;
    }
    @precise float2 vector_value(float a, float b) {
        return (float2(a, b) + float2(b, a)) - float2(a, b);
    }
    float observe(inout uint count, float value) { count += 1u; return value; }
    @precise float counted(float a, float b, inout uint count) {
        return (observe(count, a) + b) - a;
    }
    compute {
        @numthreads(1, 1, 1)
        void main(uint3 id @gl_GlobalInvocationID) {
            float a = asfloat(values[2u * id.x]);
            float b = asfloat(values[2u * id.x + 1u]);
            uint offset = 4u + 7u * id.x;
            results[offset] = asuint(subtract(add(a, b), a));
            results[offset + 1u] = asuint(selected(a, b, true));
            results[offset + 2u] = asuint(selected(a, b, false));
            float2 pair = vector_value(a, b);
            results[offset + 3u] = asuint(pair.x);
            results[offset + 4u] = asuint(pair.y);
            uint count = 0u;
            results[offset + 5u] = asuint(counted(a, b, count));
            results[offset + 6u] = count;
        }
    }
}"""


def test_precise_returns_materialize_values_without_mutating_source():
    tree = parse(SOURCE)
    before = pickle.dumps(tree)
    generator = HLSLCodeGen()
    generated = generator.generate_stage(tree, "compute")
    assert pickle.dumps(tree) == before
    assert len(re.findall(r"precise float(?:2)? crossglPreciseReturn", generated)) == 6
    assert "precise float add(float a, float b)" in generated
    assert generated.count("observe(count, a)") == 1
    assert generator.generate_stage(tree, "compute") == generated
    ordinary = generator.generate(
        parse("shader Plain { float plain(float x) { return x + 1.0; } }")
    )
    assert "crossglPreciseReturn" not in ordinary


def test_precise_return_names_avoid_parameters_locals_and_globals():
    generated = HLSLCodeGen().generate(parse("""shader Names {
        float crossglPreciseReturn_2 = 3.0;
        @precise float selected(float crossglPreciseReturn, float b) {
            float crossglPreciseReturn_1 = b;
            if (b > 0.0) { return crossglPreciseReturn + b; }
            return crossglPreciseReturn_1 + crossglPreciseReturn_2;
        }
    }"""))
    names = re.findall(r"precise float (crossglPreciseReturn\w*) =", generated)
    assert len(names) == len(set(names)) == 2
    assert not set(names) & {
        "crossglPreciseReturn",
        "crossglPreciseReturn_1",
        "crossglPreciseReturn_2",
    }


@pytest.mark.parametrize("target", ("directx", "opengl", "metal"))
def test_precise_returns_compile_on_all_targets(tmp_path, target):
    generator = {"directx": HLSLCodeGen, "opengl": GLSLCodeGen, "metal": MetalCodeGen}[
        target
    ]()
    generated = generator.generate_stage(parse(SOURCE), "compute")
    _compile(
        generated,
        target,
        tmp_path,
        metal_compile_flags=("-std=metal3.2", "-fno-fast-math"),
    )


@pytest.mark.parametrize("optimization", ("-O3", "-Od"))
def test_precise_returns_retain_ordered_dxil_arithmetic(tmp_path, optimization):
    if not shutil.which("dxc"):
        pytest.skip("DXC is required to inspect precise return arithmetic")
    source = tmp_path / "precise.hlsl"
    source.write_text(HLSLCodeGen().generate_stage(parse(SOURCE), "compute"))
    listing = tmp_path / "precise.ll"
    result = subprocess.run(
        [
            "dxc",
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            optimization,
            "-WX",
            str(source),
            "-Fo",
            str(tmp_path / "precise.dxil"),
            "-Fc",
            str(listing),
        ],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    arithmetic = re.findall(r"= f(?:add|sub) (.+)", listing.read_text())
    assert len(arithmetic) >= 4
    assert all(item.startswith("float ") for item in arithmetic), arithmetic


@pytest.mark.parametrize(
    "declaration",
    (
        "@precise float selected(float a, float b) { return (a + b) - a; }",
        "generic<T> @precise fn selected(a: T, b: T) -> T { return (a + b) - a; }",
        "@precise fn selected(a: float, b: float) -> float { (a + b) - a }",
        "@precise float selected(float a, float b) { if (a < b) { return (a + b) - a; } else { return (a - b) + b; } }",
    ),
)
def test_precise_return_specializations_compile(tmp_path, declaration):
    invocation = "selected"
    source = (
        """shader Specialized {
        StructuredBuffer<float> values @register(t0);
        RWStructuredBuffer<float> results @register(u0);
    """
        + declaration
        + """
        compute { @numthreads(1, 1, 1) void main() {
            results[0] = """
        + invocation
        + """(values[0], values[1]);
        } }
    }"""
    )
    generated = HLSLCodeGen().generate_stage(parse(source), "compute")
    assert "precise float crossglPreciseReturn" in generated
    _compile(generated, "directx", tmp_path)


def test_precise_return_prototypes_and_fallthrough_restore_function_state():
    source = """shader Fallthrough {
        @precise float prototype(float value);
        @precise float conditional(float value) {
            if (value > 0.0) { return value + 1.0; }
        }
        float ordinary(float value) { return value + 1.0; }
    }"""
    generated = HLSLCodeGen().generate(parse(source))
    assert "precise float prototype(float value);" in generated
    assert generated.count("precise float crossglPreciseReturn") == 2
    assert "float ordinary(float value) {\n    return (value + 1.0);" in generated


def test_precise_return_local_does_not_shadow_called_function():
    generated = HLSLCodeGen().generate(parse("""shader Names {
        float crossglPreciseReturn(float x) { return x; }
        @precise float selected(float x) { return crossglPreciseReturn(x); }
    }"""))
    assert "precise float crossglPreciseReturn_ = crossglPreciseReturn(x);" in generated


def test_precise_returns_execute_cancellation_and_side_effects(tmp_path, monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_ADDITIVE_PROFILE=1 for precise return execution"
        )
    from tests.test_translator import test_fused_math as native

    monkeypatch.setattr(
        native,
        "_compile",
        partial(_compile, metal_compile_flags=("-std=metal3.2", "-fno-fast-math")),
    )
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    generator = {"directx": HLSLCodeGen, "opengl": GLSLCodeGen, "metal": MetalCodeGen}[
        target
    ]()
    pairs = [
        (0x4B800000, 0x3F800000),
        (0xCB800000, 0xBF800000),
        (0x4B800001, 0x3F800000),
        (0x3F800000, 0x33800000),
        (0x3F800001, 0x33800000),
        (0x3F000000, 0xBF000000),
        (0x80000000, 0),
        (0x80000000, 0x80000000),
    ]
    expected = [GUARD] * 4
    for a, b in pairs:
        total = _oracle(a, b, True)
        first = _oracle(total, a, True, True)
        other = _oracle(_oracle(a, b, True, True), b, True)
        expected.extend(
            (first, first, other, first, _oracle(total, b, True, True), first, 1)
        )
    expected.extend([GUARD] * 4)
    generated = generator.generate_stage(parse(SOURCE), "compute")
    actual, evidence = _dispatch(
        tmp_path,
        target,
        generated,
        pairs,
        len(expected),
        entry="kernel_main" if target == "metal" else None,
        initial_output=[GUARD] * len(expected),
    )
    evidence.update({"expected": expected, "actual": actual})
    (tmp_path / "evidence.json").write_text(json.dumps(evidence, indent=2))
    assert actual == expected


@pytest.mark.parametrize(
    "expression",
    (
        "Pair(a + b, a - b)",
        "a > b ? Pair(a + b, a - b) : Pair(b - a, b + a)",
    ),
)
def test_precise_aggregate_return_paths_compile(tmp_path, expression):
    source = (
        """shader Aggregate {
        struct Pair { float first; float second; };
        StructuredBuffer<float> values @register(t0);
        RWStructuredBuffer<float> results @register(u0);
        @precise Pair selected(float a, float b) { return """
        + expression
        + """; }
        compute { @numthreads(1,1,1) void main() {
            Pair pair = selected(values[0], values[1]);
            results[0] = pair.first; results[1] = pair.second;
        } }
    }"""
    )
    generated = HLSLCodeGen().generate_stage(parse(source), "compute")
    assert "precise Pair crossglPreciseReturn" in generated
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize(
    "expression",
    (
        "float(atomicAdd(counts[0], 1u)) + a",
        "choose ? float(atomicAdd(counts[0], 1u)) + a : a - b",
    ),
)
def test_precise_atomic_return_paths_evaluate_once(tmp_path, expression):
    source = (
        """shader AtomicReturn {
        RWBuffer<uint> counts @register(u0);
        StructuredBuffer<float> values @register(t0);
        RWStructuredBuffer<float> results @register(u1);
        @precise float selected(float a, float b, bool choose) { return """
        + expression
        + """; }
        compute { @numthreads(1,1,1) void main() {
            results[0] = selected(values[0], values[1], values[2] > 0.0);
        } }
    }"""
    )
    generated = HLSLCodeGen().generate_stage(parse(source), "compute")
    assert generated.count("InterlockedAdd(") == 1
    assert "precise float crossglPreciseReturn" in generated
    _compile(generated, "directx", tmp_path)

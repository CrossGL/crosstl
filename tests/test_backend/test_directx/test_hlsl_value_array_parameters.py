"""HLSL array parameters without ``out`` or ``inout`` are passed by value.

A helper that writes to such a parameter changes only its own copy. Metal array
parameters decay to pointers and the OpenGL backend preserves pointer-like
writeback for unqualified fixed arrays, so the HLSL frontend states the copy-in
contract with ``in`` and the Metal backend copies a written ``in`` array into a
local that keeps the source name.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import crosstl
from crosstl.backend.DirectX import DirectxCrossGLCodeGen
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser

REQUIRE_ENV = "CROSSTL_REQUIRE_HLSL_VALUE_ARRAY_EXECUTION"
ROOT = Path(__file__).resolve().parents[3]

VALUE_ARRAYS_HLSL = """
    struct Pair { float a; float b; };
    RWStructuredBuffer<float> results : register(u0);

    float total(float values[4]) { return values[0] + values[1] + values[2] + values[3]; }
    float limit(const float values[4]) { return values[3]; }
    void twice(inout float values[4]) { values[2] *= 2.0; }
    void reset(out float values[4]) { values[0] = 1.0; values[1] = 2.0; values[2] = 3.0; values[3] = 4.0; }
    float forward(float values[4]) { twice(values); return values[2]; }
    float counter(int counts[2]) { counts[1]++; return counts[1]; }
    float angles(float values[4]) { sincos(0.0, values[0], values[1]); return values[0] + values[1]; }
    float grid(float cells[2][3]) { cells[1][2] = 9.0; return cells[1][2] + cells[0][1]; }
    float pairs(Pair items[2]) { items[0].a = 5.0; return items[0].a + items[1].b; }

    [numthreads(1, 1, 1)]
    void CSMain(uint3 id : SV_DispatchThreadID)
    {
        float local[4];
        reset(local);
        int counts[2] = { 1, 2 };
        float cells[2][3] = { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
        Pair items[2];
        items[0].a = 1.0; items[0].b = 2.0; items[1].a = 3.0; items[1].b = 4.0;
        results[0] = total(local);
        results[1] = forward(local);
        results[2] = local[2];
        results[3] = counter(counts);
        results[4] = counts[1];
        results[5] = angles(local);
        results[6] = local[0] + local[1];
        results[7] = grid(cells);
        results[8] = cells[1][2];
        results[9] = pairs(items);
        results[10] = items[0].a + limit(local);
    }
"""

# HLSL semantics: callee writes stay in the callee's copy, while ``out`` and
# ``inout`` arguments are written back.
EXPECTED = [10.0, 6.0, 3.0, 3.0, 2.0, 1.0, 3.0, 11.0, 6.0, 9.0, 5.0]


def translate(tmp_path, backend):
    path = tmp_path / "value_arrays.hlsl"
    path.write_text(textwrap.dedent(VALUE_ARRAYS_HLSL).strip() + "\n", "utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def generate_crossgl():
    source = textwrap.dedent(VALUE_ARRAYS_HLSL).strip()
    ast = HLSLParser(HLSLLexer(source).tokenize()).parse()
    return DirectxCrossGLCodeGen.HLSLToCrossGLConverter().generate(ast)


def run(command):
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def metal_compiler():
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return None
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
    )
    return xcrun if probe.returncode == 0 else None


def compile_metal(source, tmp_path):
    xcrun = metal_compiler()
    if xcrun is None:
        return None
    shader = tmp_path / "value_arrays.metal"
    shader.write_text(source, encoding="utf-8")
    air = tmp_path / "value_arrays.air"
    library = tmp_path / "value_arrays.metallib"
    run([xcrun, "-sdk", "macosx", "metal", "-Werror", "-c", str(shader), "-o", air])
    run([xcrun, "-sdk", "macosx", "metallib", str(air), "-o", str(library)])
    return library


def validate_glsl(source, tmp_path):
    validator = shutil.which("glslangValidator")
    if validator is None:
        return
    shader = tmp_path / "value_arrays.comp"
    shader.write_text(source, encoding="utf-8")
    run([validator, "-S", "comp", str(shader)])


def test_helper_array_parameters_state_copy_in_semantics():
    crossgl = generate_crossgl()

    assert "float total(in float values[4])" in crossgl
    assert "float forward(in float values[4])" in crossgl
    assert "float counter(in int counts[2])" in crossgl
    assert "float grid(in float cells[2][3])" in crossgl
    assert "float pairs(in Pair items[2])" in crossgl
    assert "float limit(const float values[4])" in crossgl
    assert "void twice(inout float values[4])" in crossgl
    assert "void reset(out float values[4])" in crossgl
    assert "void CSMain(uvec3 id @ gl_GlobalInvocationID)" in crossgl


def test_groupshared_array_parameters_keep_reference_semantics():
    # microsoft/DirectXShaderCompiler
    # tools/clang/test/CodeGenHLSL/groupsharedArgs/ArrTest.hlsl
    source = """
        groupshared float4 SharedArr[64];
        void Store(groupshared float4 Arr[64], float F) { Arr[5] = F.xxxx; }
        [numthreads(4, 1, 1)]
        void main() { Store(SharedArr, 6.0); }
    """
    ast = HLSLParser(HLSLLexer(textwrap.dedent(source).strip()).tokenize()).parse()
    crossgl = DirectxCrossGLCodeGen.HLSLToCrossGLConverter().generate(ast)

    assert "void Store(vec4 Arr[64], float F)" in crossgl
    assert "in vec4 Arr" not in crossgl


def test_metal_copies_written_input_arrays(tmp_path):
    metal = translate(tmp_path, "metal")

    copies = {
        "forward": ("float", "values", ["4"]),
        "counter": ("int", "counts", ["2"]),
        "angles": ("float", "values", ["4"]),
        "grid": ("float", "cells", ["2", "3"]),
        "pairs": ("Pair", "items", ["2"]),
    }
    for function, (element, name, dimensions) in copies.items():
        suffix = "".join(f"[{dimension}]" for dimension in dimensions)
        signature = re.search(
            rf"float {function}\({element} {name}_crosstl_input{re.escape(suffix)}\)"
            r" \{(?P<body>.*?)\n\}",
            metal,
            re.S,
        )
        assert signature is not None, function
        body = signature.group("body")
        assert f"{element} {name}{suffix};" in body
        indices = "".join(
            f"[crosstl_{name}_i{level}]" for level in range(len(dimensions))
        )
        assert f"{name}{indices} = {name}_crosstl_input{indices};" in body

    # Read-only inputs keep the pointer parameter; out/inout keep aliasing.
    assert "float total(float values[4])" in metal
    assert "void twice(float values[4])" in metal
    assert "void reset(float values[4])" in metal
    assert "unsupported" not in metal
    compile_metal(metal, tmp_path)


def test_opengl_passes_input_arrays_by_value(tmp_path):
    glsl = translate(tmp_path, "opengl")

    for declaration in (
        "float total(float values[4])",
        "float forward(float values[4])",
        "float counter(int counts[2])",
        "float angles(float values[4])",
        "float grid(float cells[2][3])",
        "float pairs(Pair items[2])",
        "void twice(inout float values[4])",
        "void reset(out float values[4])",
    ):
        assert declaration in glsl
    assert glsl.count("inout") == 2
    validate_glsl(glsl, tmp_path)


def test_value_arrays_execute_natively(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 to require native value-array checks")
    if sys.platform == "darwin":
        metal = translate(tmp_path, "metal")
        library = compile_metal(metal, tmp_path)
        assert library is not None, "the Metal toolchain is required"
        runner = tmp_path / "readback"
        run(
            [
                "swiftc",
                str(
                    ROOT
                    / "tests/fixtures/runtime_verification/metal_float_readback.swift"
                ),
                "-o",
                str(runner),
            ]
        )
        entry = re.search(r"\bkernel\s+void\s+(\w+)", metal).group(1)
        outputs = json.loads(
            run([str(runner), str(library), entry, str(len(EXPECTED))])
        )
        assert outputs["values"] == EXPECTED, outputs
        return

    from crosstl.project.native_runtime_drivers import OpenGLComputeRuntime
    from crosstl.project.runtime_verification import (
        NativeRuntimeBufferBinding,
        NativeRuntimeDispatchRequest,
        RuntimeArtifactSelector,
        RuntimeDispatchGeometry,
        RuntimeExecutionRequest,
        RuntimeFixture,
        RuntimeResourceBinding,
    )

    glsl = translate(tmp_path, "opengl")
    artifact = tmp_path / "value_arrays.comp"
    artifact.write_text(glsl, encoding="utf-8")
    runtime = OpenGLComputeRuntime(context_backends=("egl",))
    request = NativeRuntimeDispatchRequest(
        target="opengl",
        artifact={"target": "opengl"},
        artifact_path=artifact,
        module_path=artifact,
        loaded_artifact=glsl,
        buffers={
            "results": NativeRuntimeBufferBinding(
                name="results",
                binding=RuntimeResourceBinding(
                    name="results",
                    kind="buffer",
                    type_name="RWStructuredBuffer<float>",
                    set=0,
                    binding=0,
                    access="read_write",
                ),
                source="expectedOutput",
                dtype="float32",
                shape=(len(EXPECTED),),
                value=[-1234.0] * len(EXPECTED),
            )
        },
        constants={},
        entry_point="main",
        dispatch=RuntimeDispatchGeometry(
            entry_point="main", workgroup_size=(1, 1, 1), workgroup_count=(1, 1, 1)
        ),
    )
    availability = runtime.is_available(
        None,
        RuntimeExecutionRequest(
            fixture=RuntimeFixture(
                id="hlsl-value-arrays",
                selector=RuntimeArtifactSelector(target="opengl"),
                entry_point="main",
            ),
            artifact=request.artifact,
            artifact_path=artifact,
            project_root=tmp_path,
        ),
    )
    assert availability.available, availability.reason
    outputs = runtime.dispatch(None, None, request)
    assert outputs["results"]["values"] == EXPECTED, outputs

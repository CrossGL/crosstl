"""CrossGL specialization constants become Metal function constants that keep their id and default.

``const T name @constant_id(N) = value;`` may be replaced by the host when it builds the pipeline. Metal output used
to freeze ``value`` under a "cannot preserve" comment; it now declares ``[[function_constant(N)]]`` and reads it
through ``is_function_constant_defined`` so the default still applies when the host sets nothing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

ROOT = Path(__file__).resolve().parents[2]
REQUIRE_ENV = "CROSSTL_REQUIRE_FUNCTION_CONSTANT_EXECUTION"

SPECIALIZED = """
shader SpecializedScale {
    const int scale @constant_id(0) = 3;
    const int offset = scale * 2 + 1;
    const bool negate @constant_id(1) = false;
    const float shift @constant_id(2) = 0.5;
    compute {
        layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
        buffer int* fin;
        buffer int* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int value = fin[i] * scale + offset + int(shift * 2.0);
            outp[i] = negate ? -value : value;
        }
    }
}
"""


def translate(tmp_path, source, backend="metal"):
    path = tmp_path / "specialized.cgl"
    path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def metal_compiler():
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return None
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
    )
    return xcrun if probe.returncode == 0 else None


def run(command):
    completed = subprocess.run(
        [str(item) for item in command], capture_output=True, text=True, timeout=180
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_specialization_constants_declare_function_constants_with_their_ids(tmp_path):
    metal = translate(tmp_path, SPECIALIZED)
    assert "CrossGL fallback" not in metal
    for name, type_name, index, default in (
        ("scale", "int", 0, "3"),
        ("negate", "bool", 1, "false"),
        ("shift", "float", 2, "0.5"),
    ):
        raw = f"__crossgl_function_constant_{name}"
        assert f"{type_name} {raw} [[function_constant({index})]];" in metal, metal
        assert (
            f"{type_name} {name} = is_function_constant_defined({raw}) ? {raw} : "
            f"{type_name}({default});" in metal
        ), metal
    # A constant expression over a specialization constant reads the specialized value.
    assert "constant int offset = scale * 2 + 1;" in metal, metal
    assert metal.count("[[function_constant(") == 3, metal


def test_double_specialization_constant_is_refused(tmp_path):
    source = SPECIALIZED.replace(
        "const float shift @constant_id(2) = 0.5;",
        "const double shift @constant_id(2) = 0.5;",
    )
    with pytest.raises(MetalLoweringRefused, match="specialization constant 'shift'"):
        translate(tmp_path, source)


def test_specialization_constants_compile_into_air_function_constants(tmp_path):
    xcrun = metal_compiler()
    if xcrun is None:
        pytest.skip("requires the Metal toolchain")
    shader = tmp_path / "specialized.metal"
    shader.write_text(translate(tmp_path, SPECIALIZED), encoding="utf-8")
    llvm = tmp_path / "specialized.ll"
    run(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-Werror",
            "-S",
            "-emit-llvm",
            shader,
            "-o",
            llvm,
        ]
    )
    module = llvm.read_text(encoding="utf-8")
    assert re.search(r"^!air\.function_constants = ", module, re.M), module[-2000:]


def _dispatch(tmp_path, library, entry, slots, inputs, tag, ints=(), bools=()):
    files = []
    payloads = {
        "fin": struct.pack(f"<{len(inputs)}i", *inputs),
        "outp": struct.pack(f"<{len(inputs)}i", *([-12345] * len(inputs))),
    }
    assert sorted(slots) == sorted(payloads), slots
    for name, index in sorted(slots.items(), key=lambda item: item[1]):
        path = tmp_path / f"{tag}-buffer-{index}.in"
        path.write_bytes(payloads[name])
        files.append(str(path))
    request = tmp_path / f"{tag}-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": [1, 1, 1],
                "workgroupSize": [len(inputs), 1, 1],
                "simdWidth": 32,
                "intConstants": [{"index": i, "value": v} for i, v in ints],
                "boolConstants": [{"index": i, "value": v} for i, v in bools],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / f"{tag}-output"
    runner = tmp_path / "runner"
    run([runner, library, entry, request, output])
    data = (output / f"buffer-{slots['outp']}.bin").read_bytes()
    return list(struct.unpack(f"<{len(inputs)}i", data))


def test_specialization_constants_take_host_values_on_metal(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1" or sys.platform != "darwin":
        pytest.skip(f"set {REQUIRE_ENV}=1 on macOS to require the Metal dispatch")
    xcrun = metal_compiler()
    assert xcrun is not None, "the Metal toolchain is required"
    metal = translate(tmp_path, SPECIALIZED)
    shader = tmp_path / "specialized.metal"
    shader.write_text(metal, encoding="utf-8")
    air, library = tmp_path / "specialized.air", tmp_path / "specialized.metallib"
    run([xcrun, "-sdk", "macosx", "metal", "-Werror", "-c", shader, "-o", air])
    run([xcrun, "-sdk", "macosx", "metallib", air, "-o", library])
    run(
        [
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            tmp_path / "runner",
        ]
    )
    kernel = re.search(r"kernel void (\w+)\((.*?)\) \{", metal, re.S)
    assert kernel is not None, metal
    slots = {
        name: int(index)
        for name, index in re.findall(r"(\w+) \[\[buffer\((\d+)\)\]\]", kernel.group(2))
    }
    inputs = [0, 1, 2, -5]

    # Only ``negate`` is set: ``scale`` and ``shift`` keep their defaults 3 and 0.5.
    defaults = _dispatch(
        tmp_path,
        library,
        kernel.group(1),
        slots,
        inputs,
        "defaults",
        bools=[(1, False)],
    )
    assert defaults == [3 * x + 7 + 1 for x in inputs]

    # The host values replace every default, and ``offset`` follows ``scale``.
    specialized = _dispatch(
        tmp_path,
        library,
        kernel.group(1),
        slots,
        inputs,
        "specialized",
        ints=[(0, 5)],
        bools=[(1, True)],
    )
    assert specialized == [-(5 * x + 11 + 1) for x in inputs]


SIZED = """
shader SizedBySpecialization {
    const int count @constant_id(3) = 4;
    const int padded = count + 1;
    const int fixedCount = 4;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            float values[SIZE];
            for (int j = 0; j < count; j++) {
                values[j] = float(j);
            }
            outp[0] = values[0];
        }
    }
}
"""


@pytest.mark.parametrize("size", ["count", "padded"])
def test_array_sized_by_a_specialization_constant_reads_a_compile_time_macro(
    tmp_path, size
):
    # A function constant is not a constant expression to the Metal front end, so ``count`` is specialized when the
    # source is compiled: it reads a macro named after its id that defaults to 4. ``padded`` follows ``count``
    # through its initializer, so it moves ``count`` to the macro as well.
    metal = translate(tmp_path, SIZED.replace("SIZE", size))
    assert "#ifndef CROSSGL_SPECIALIZATION_CONSTANT_ID_3\n" in metal, metal
    assert "#define CROSSGL_SPECIALIZATION_CONSTANT_ID_3 4\n" in metal, metal
    assert " count = int(CROSSGL_SPECIALIZATION_CONSTANT_ID_3);" in metal, metal
    assert " padded = count + 1;" in metal, metal
    assert f"float values[{size}];" in metal, metal
    assert "[[function_constant(3)]]" not in metal, metal
    xcrun = metal_compiler()
    if xcrun is None:
        return
    shader = tmp_path / "sized.metal"
    shader.write_text(metal, encoding="utf-8")
    for tag, defines in (
        ("default", []),
        ("specialized", ["-D", "CROSSGL_SPECIALIZATION_CONSTANT_ID_3=9"]),
    ):
        run(
            [
                xcrun,
                "-sdk",
                "macosx",
                "metal",
                "-Werror",
                *defines,
                "-c",
                shader,
                "-o",
                tmp_path / f"sized-{tag}.air",
            ]
        )


def test_only_the_constants_that_size_arrays_leave_function_constants(tmp_path):
    source = (
        SIZED.replace("SIZE", "count")
        .replace(
            "    const int fixedCount = 4;\n",
            "    const int fixedCount = 4;\n    const int scale @constant_id(4) = 2;\n",
        )
        .replace("values[j] = float(j);", "values[j] = float(j * scale);")
    )
    metal = translate(tmp_path, source)
    assert "#define CROSSGL_SPECIALIZATION_CONSTANT_ID_3 4\n" in metal, metal
    assert "[[function_constant(4)]];" in metal, metal
    assert "CROSSGL_SPECIALIZATION_CONSTANT_ID_4" not in metal, metal


def test_host_reflection_reports_both_kinds_of_metal_specialization(tmp_path):
    # The host specializes a macro constant through MTLCompileOptions.preprocessorMacros and a function constant
    # through MTLFunctionConstantValues, so the reflection of the artifact names both, each with its id.
    from crosstl.project.host_reflection import reflect_target_host_interface

    source = (
        SIZED.replace("SIZE", "count")
        .replace(
            "    const int fixedCount = 4;\n",
            "    const int fixedCount = 4;\n    const int scale @constant_id(4) = 2;\n",
        )
        .replace("values[j] = float(j);", "values[j] = float(j * scale);")
    )
    artifact = tmp_path / "sized.metal"
    artifact.write_text(translate(tmp_path, source), encoding="utf-8")
    reflection = reflect_target_host_interface(artifact, target="metal")
    constants = {item["id"]: item for item in reflection["specializationConstants"]}
    assert sorted(constants) == [3, 4], reflection["specializationConstants"]
    assert constants[3]["kind"] == "preprocessor-macro"
    assert constants[3]["macro"] == "CROSSGL_SPECIALIZATION_CONSTANT_ID_3"
    assert constants[3]["name"] == "count"
    assert constants[3]["dtype"] == "int"
    assert constants[3]["default"] == 4
    assert constants[3]["required"] is False
    assert constants[4]["kind"] == "function-constant"
    assert constants[4]["source"] == "metal.function_constant"


SIZED_DISPATCH = """
shader SizedDispatch {
    const int count @constant_id(3) = 4;
    compute {
        layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
        buffer int* fin;
        buffer int* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int values[count];
            for (int j = 0; j < count; j++) {
                values[j] = j + 1;
            }
            int total = 0;
            for (int j = 0; j < count; j++) {
                total += values[j];
            }
            outp[i] = fin[i] * 100 + total;
        }
    }
}
"""


def test_array_sizing_specialization_constant_takes_the_host_macro_on_metal(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1" or sys.platform != "darwin":
        pytest.skip(f"set {REQUIRE_ENV}=1 on macOS to require the Metal dispatch")
    xcrun = metal_compiler()
    assert xcrun is not None, "the Metal toolchain is required"
    metal = translate(tmp_path, SIZED_DISPATCH)
    shader = tmp_path / "sized_dispatch.metal"
    shader.write_text(metal, encoding="utf-8")
    run(
        [
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            tmp_path / "runner",
        ]
    )
    kernel = re.search(r"kernel void (\w+)\((.*?)\) \{", metal, re.S)
    assert kernel is not None, metal
    slots = {
        name: int(index)
        for name, index in re.findall(r"(\w+) \[\[buffer\((\d+)\)\]\]", kernel.group(2))
    }
    inputs = [0, 1, 2, -5]
    results = {}
    for tag, defines, count in (
        ("defaults", [], 4),
        ("specialized", ["-D", "CROSSGL_SPECIALIZATION_CONSTANT_ID_3=7"], 7),
    ):
        air = tmp_path / f"{tag}.air"
        library = tmp_path / f"{tag}.metallib"
        run(
            [
                xcrun,
                "-sdk",
                "macosx",
                "metal",
                "-Werror",
                *defines,
                "-c",
                shader,
                "-o",
                air,
            ]
        )
        run([xcrun, "-sdk", "macosx", "metallib", air, "-o", library])
        results[tag] = _dispatch(tmp_path, library, kernel.group(1), slots, inputs, tag)
        # The array holds ``count`` elements 1..count, so their sum follows the macro.
        assert results[tag] == [x * 100 + count * (count + 1) // 2 for x in inputs], tag


def test_array_sized_by_a_plain_constant_still_compiles(tmp_path):
    metal = translate(tmp_path, SIZED.replace("SIZE", "fixedCount"))
    assert "float values[fixedCount];" in metal, metal
    assert "[[function_constant(3)]];" in metal, metal
    xcrun = metal_compiler()
    if xcrun is None:
        return
    shader = tmp_path / "sized.metal"
    shader.write_text(metal, encoding="utf-8")
    run(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-Werror",
            "-c",
            shader,
            "-o",
            tmp_path / "sized.air",
        ]
    )


STAGE_INPUT_ARRAY = """
shader SpecializedStageInputArray {
    const int count @constant_id(3) = 3;
    struct VertexInput {
        vec4 weights[count] @location(0);
        vec2 uv @location(3);
    }
    struct VertexOutput {
        vec4 position @gl_Position;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            vec4 total = vec4(input.uv, 0.0, 0.0);
            for (int i = 0; i < count; i++) {
                total += input.weights[i];
            }
            output.position = total;
            return output;
        }
    }
}
"""


def test_stage_input_array_sized_by_a_specialization_constant_is_flattened(tmp_path):
    # Interface locations follow the default size, and Metal declares one attribute per element instead of an
    # array, so the array never needs a constant-expression size; the loop bound still follows the host value.
    metal = translate(tmp_path, STAGE_INPUT_ARRAY)
    assert "[[function_constant(3)]];" in metal, metal
    for index in range(3):
        assert f"float4 weights_{index} [[attribute({index})]];" in metal, metal
    assert "float2 uv [[attribute(3)]];" in metal, metal
    assert "weights[count]" not in metal, metal
    xcrun = metal_compiler()
    if xcrun is None:
        return
    shader = tmp_path / "stage_input.metal"
    shader.write_text(metal, encoding="utf-8")
    run(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-Werror",
            "-c",
            shader,
            "-o",
            tmp_path / "stage_input.air",
        ]
    )

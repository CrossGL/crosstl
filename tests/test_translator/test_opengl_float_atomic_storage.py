"""Logical float values retain their bits across word-backed atomic allocations."""

import copy
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.glsl_float_atomic_storage import (
    OpenGLFloatAtomicStorageError,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_float_atomic_compare_exchange import (
    CASES as COMPARE_CASES,
)
from tests.test_translator.test_float_atomic_compare_exchange import (
    _request as _compare_request,
)
from tests.test_translator.test_float_atomic_memory import CASES, _request
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_mlx_current_binary_shapes import _prepare_native_package
from tests.test_translator.test_mlx_float_scatter_runtime import CASES as SCATTER_CASES

REQUIRE_ENV = "CROSTL_REQUIRE_OPENGL_FLOAT_ATOMIC_STORAGE"


def test_float_storage_requires_native_linux_execution():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/mlx-gather-roundtrip.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate indexed OpenGL gather and resource aggregates"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_opengl_float_atomic_storage.py" in step
    assert "-n auto" in step and "--timeout-seconds 1200" in step
    assert "--basetemp=" in step and "--junitxml=" in step
    assert "if:" not in step and "continue-on-error" not in workflow


@pytest.mark.parametrize("operation", ("load", "store"))
@pytest.mark.parametrize("case", CASES)
def test_float_memory_has_physical_word_storage(tmp_path, operation, case):
    _, request, expected = _request(tmp_path, "opengl", operation, case)
    generated = request.artifact_path.read_text()
    assert "crossgl_pending_float_atomic" not in generated
    assert "GL_EXT_shader_atomic_float" not in generated
    assert "GL_NV_shader_atomic_float" not in generated
    assert "uintBitsToFloat(atomicOr(" in generated
    if shutil.which("glslangValidator"):
        _, module = _compile(generated, "opengl", tmp_path)
        assert module.stat().st_size
    if os.environ.get(REQUIRE_ENV) == "1":
        assert sys.platform == "linux"
        _execute(request, expected, tmp_path)


@pytest.mark.parametrize("case", COMPARE_CASES)
def test_float_compare_has_physical_word_storage(tmp_path, case):
    _, request, expected = _compare_request(tmp_path, "opengl", case)
    generated = request.artifact_path.read_text()
    assert "crossgl_pending_float_atomic" not in generated
    assert "atomicCompSwap(" in generated
    assert "== floatBitsToUint(" in generated
    if shutil.which("glslangValidator"):
        _, module = _compile(generated, "opengl", tmp_path)
        assert module.stat().st_size
    if os.environ.get(REQUIRE_ENV) == "1":
        assert sys.platform == "linux"
        _execute(request, expected, tmp_path)


def _canonical(body, helpers="", element="float"):
    return f"""shader Storage {{
        {helpers}
        RWStructuredBuffer<{element}> values @binding(0);
        RWStructuredBuffer<float> results @binding(1);
        compute {{
            @numthreads(1, 1, 1)
            void main() {{ {body} }}
        }}
    }}"""


def _canonical_request(
    work, source, initial, final, output, *, components=1, operation="struct"
):
    (work / "workload.json").write_text(
        json.dumps({"case": "ordinary-storage", "operation": operation}),
        encoding="utf-8",
    )
    (work / "storage.cgl").write_text(source, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=work,
            include_patterns=("storage.cgl",),
            targets=("opengl",),
            output_dir="out",
        ),
        format_output=False,
    )
    descriptor, package = _prepare_native_package(report, work)
    guard = 0x3F7A4321

    def words(values):
        return [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]

    shape = (
        [len(initial) // components + 2, components]
        if components > 1
        else [len(initial) + 2]
    )

    def typed(values, shape):
        return {"dtype": "uint32", "shape": shape, "values": values}

    inputs = {
        "values": typed(initial + [guard] * (2 * components), shape),
        "results": {
            "dtype": "float32",
            "shape": [len(output) + 2],
            "values": [guard] * (len(output) + 2),
            "encoding": FLOAT32_BITS,
        },
    }
    outputs = {
        "values": typed(final + [guard] * (2 * components), shape),
        "results": {**inputs["results"], "values": words(output) + [guard] * 2},
    }
    expected = _bound_values(descriptor, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target="opengl",
    )
    assert not request.execution_plan.diagnostics
    return request, expected


@pytest.mark.parametrize(
    "update", ("= 2.5", "+= 1.25", "-= 1.25", "*= 1.25", "/= 1.25")
)
def test_float_nonatomic_updates_keep_logical_value_and_single_index(tmp_path, update):
    source = _canonical(f"""uint index = 0u;
        atomicLoad(values[0]);
        results[0] = (values[index++] {update});
        results[1] = values[0];
        results[2] = float(index);""")
    generated = GLSLCodeGen().generate_stage(parse(source), "compute")
    assert generated.count("index++") == 1
    _compile(generated, "opengl", tmp_path)
    value = {
        "= 2.5": 2.5,
        "+= 1.25": 5.25,
        "-= 1.25": 2.75,
        "*= 1.25": 5.0,
        "/= 1.25": 3.2,
    }[update]
    final = struct.unpack("<I", struct.pack("<f", value))[0]
    request, expected = _canonical_request(
        tmp_path, source, [0x40800000], [final], [value, value, 1.0], operation=update
    )
    if os.environ.get(REQUIRE_ENV) == "1":
        _execute(request, expected, tmp_path)


@pytest.mark.parametrize(
    "update",
    (
        "values[index++]++",
        "++values[index++]",
        "values[index++]--",
        "--values[index++]",
    ),
)
def test_float_increment_preserves_expression_value(tmp_path, update):
    source = _canonical(f"""
        uint index = 0u;
        atomicLoad(values[0]);
        results[0] = {update};
        results[1] = values[0];
        results[2] = float(index);""")
    generated = GLSLCodeGen().generate_stage(parse(source), "compute")
    assert generated.count("index++") == 1
    _compile(generated, "opengl", tmp_path)
    value = 5.0 if "++values" in update or update.endswith("++") else 3.0
    returned = value if update.startswith(("++", "--")) else 4.0
    final = struct.unpack("<I", struct.pack("<f", value))[0]
    request, expected = _canonical_request(
        tmp_path,
        source,
        [0x40800000],
        [final],
        [returned, value, 1.0],
        operation=update,
    )
    if os.environ.get(REQUIRE_ENV) == "1":
        _execute(request, expected, tmp_path)


def test_mixed_storage_codec_preserves_private_types_and_array_members(tmp_path):
    source = _canonical(
        """
        atomicLoad(values[0].value);
        uint index = 0u;
        Counter local = values[index++];
        local.offset += 7;
        local.lanes += vec2(0.5);
        local.samples[1] = 1.5;
        values[0] = local;
        values[0].offset++;
        values[0].mask |= 2u;
        values[0].lanes += vec2(1.0);
        results[0] = values[0].value;
        results[1] = float(index);
        results[2] = values[0].samples[1];
    """,
        "struct Counter { float value; int offset; uint mask; vec2 lanes; float samples[2]; };",
        "Counter",
    )
    ast = parse(source)
    original = copy.deepcopy(ast)
    generator = GLSLCodeGen()
    generated = generator.generate_stage(ast, "compute")
    assert generator.generate_stage(ast, "compute") == generated
    assert GLSLCodeGen().generate_stage(original, "compute") == generated
    assert "Counter local" in generated and "crossgl_words_Counter" in generated
    assert generated.count("index++") == 1
    _compile(generated, "opengl", tmp_path)
    unrelated = generator.generate_stage(
        parse(_canonical("values[0] = 1.5;")), "compute"
    )
    assert "float values[]" in unrelated and "crossgl_words_" not in unrelated


def test_word_storage_rejects_escaping_logical_reference():
    source = _canonical(
        "atomicLoad(values[0]); modify(values[0]);",
        "void modify(inout float value) { value += 1.0; }",
    )
    with pytest.raises(
        OpenGLFloatAtomicStorageError, match="logical-storage-reference-escape"
    ):
        GLSLCodeGen().generate_stage(parse(source), "compute")


def test_private_parameter_shadowing_storage_remains_logical(tmp_path):
    source = _canonical(
        "atomicLoad(values[0]); results[0] = inspect(Counter(2.5));",
        "struct Counter { float value; }; float inspect(Counter values) { return values.value; }",
    )
    generated = GLSLCodeGen().generate_stage(parse(source), "compute")
    assert "crossgl_decode_Counter" not in generated
    _compile(generated, "opengl", tmp_path)


def test_float_atomic_struct_members_use_sanitized_names(tmp_path):
    source = _canonical(
        "results[0] = atomicAdd(values[0].input, 1.0);",
        "struct Counter { float input; };",
        "Counter",
    )
    generated = GLSLCodeGen().generate_stage(parse(source), "compute")
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize("kind", ("half", "half2", "half[2]"))
def test_float_storage_does_not_change_neighbor_half_rounding(kind):
    source = _canonical(
        "atomicLoad(values[0].value);",
        f"struct Counter {{ float value; {kind} neighbor; }};",
        "Counter",
    )
    with pytest.raises(
        OpenGLFloatAtomicStorageError, match="narrow-float-storage-allocation"
    ):
        GLSLCodeGen().generate_stage(parse(source), "compute")


def test_float_atomic_does_not_widen_half_payload():
    with pytest.raises(
        OpenGLFloatAtomicStorageError, match="float-atomic-payload-width"
    ):
        GLSLCodeGen().generate_stage(
            parse(_canonical("atomicLoad(values[0]);", element="half")), "compute"
        )


def test_float_storage_rejects_whole_array_reference():
    source = _canonical(
        "atomicLoad(values[0].value); inspect(values[0].samples);",
        "struct Counter { float value; float samples[2]; }; void inspect(inout float items[2]) { items[0] += 1.0; }",
        "Counter",
    )
    with pytest.raises(
        OpenGLFloatAtomicStorageError, match="whole-array-storage-transfer"
    ):
        GLSLCodeGen().generate_stage(parse(source), "compute")


def test_mixed_struct_transfer_and_neighbor_updates(tmp_path):
    source = _canonical(
        """
        uint index = 0u;
        Counter local = values[index++];
        local.offset += 7u;
        local.before = -2.5;
        values[0] = local;
        values[0].offset++;
        values[0].before += 1.25;
        values[0].after = -values[0].before;
        results[0] = atomicAdd(values[0].value, 0.5);
        results[1] = values[0].value;
        results[2] = values[0].before;
        results[3] = values[0].after;
        results[4] = float(values[0].offset);
        results[5] = float(index);
    """,
        "struct Counter { float value; uint offset; float before; float after; };",
        "Counter",
    )
    request, expected = _canonical_request(
        tmp_path,
        source,
        [0x40800000, 3, 0x7FA01234, 0x80000000],
        [0x40900000, 11, 0xBFA00000, 0x3FA00000],
        [4.0, 4.5, -1.25, 1.25, 11.0, 1.0],
        components=4,
    )
    generated = request.artifact_path.read_text()
    assert generated.count("index++") == 1
    assert (
        "crossgl_decode_Counter" in generated and "crossgl_encode_Counter" in generated
    )
    _compile(generated, "opengl", tmp_path)
    if os.environ.get(REQUIRE_ENV) == "1":
        _execute(request, expected, tmp_path)


@pytest.mark.parametrize(
    "case", ("load", "store", "compare", "multiply", "minimum", "maximum", "contention")
)
def test_unchanged_mlx_atomic_helpers_use_word_storage(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native unchanged MLX helpers")
    assert sys.platform == "linux"
    from demos.integrations.mlx.portable_host.prepare import COMMIT
    from tests.test_translator.test_mlx_atomic_load_runtime import HEADER
    from tests.test_translator.test_mlx_float_atomic_compare_exchange import (
        _request as compare_request,
    )
    from tests.test_translator.test_mlx_float_atomic_memory import (
        _request as memory_request,
    )
    from tests.test_translator.test_mlx_general_scatter_runtime import _verify_source

    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    hashes = _verify_source(root, (HEADER,))
    try:
        request_builder = (
            memory_request if case in {"load", "store"} else compare_request
        )
        _, request, expected = request_builder(root, "opengl", tmp_path, case)
        (tmp_path / "workload.json").write_text(
            json.dumps({"commit": COMMIT, "headers": hashes, "case": case}),
            encoding="utf-8",
        )
        _execute(request, expected, tmp_path)
    finally:
        assert _verify_source(root, (HEADER,)) == hashes


@pytest.mark.parametrize("operation,layout", SCATTER_CASES)
def test_unchanged_mlx_float_scatter_uses_word_storage(tmp_path, operation, layout):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native unchanged MLX scatter")
    assert sys.platform == "linux"
    from demos.integrations.mlx.portable_host.prepare import COMMIT
    from tests.test_translator.test_mlx_atomic_load_runtime import HEADER
    from tests.test_translator.test_mlx_float_scatter_runtime import (
        _request as scatter_request,
    )
    from tests.test_translator.test_mlx_general_scatter_runtime import (
        HEADERS,
        JIT,
        _verify_source,
    )

    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    hashes = _verify_source(root, (*HEADERS, JIT, HEADER))
    try:
        entry, _, request, expected = scatter_request(
            root, "opengl", tmp_path, operation, layout
        )
        (tmp_path / "workload.json").write_text(
            json.dumps(
                {
                    "commit": COMMIT,
                    "headers": hashes,
                    "entry": entry,
                    "operation": operation,
                    "layout": layout,
                }
            ),
            encoding="utf-8",
        )
        _execute(request, expected, tmp_path)
    finally:
        assert _verify_source(root, (*HEADERS, JIT, HEADER)) == hashes

"""Translate floating-point buffer atomics without copying their destination."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from crosstl.project import ProjectConfig, translate_project
from crosstl.project.native_runtime_drivers import DirectXComputeRuntime
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeDispatchGeometry,
    RuntimeResourceBinding,
)
from crosstl.translator import parse
from crosstl.translator.ast import FunctionCallNode, IdentifierNode
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from tests.test_translator.test_metal_builtin_ownership import (
    _compile as _compile_metal,
)
from tests.test_translator.test_metal_float_atomics import (
    _bits,
    _source,
)
from tests.test_translator.test_metal_float_atomics import (
    metal_word_runner as _metal_word_runner,
)

REQUIRE_ENV = "CROSTL_REQUIRE_DIRECTX_FLOAT_ATOMICS"
metal_word_runner = _metal_word_runner


def _translate(tmp_path, source):
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=("directx",),
            include_patterns=(path.name,),
            entry_points={path.name: ("update_values",)},
            workgroup_size=(1, 1, 1),
            output_dir="out",
        ),
        format_output=False,
    )
    report.write_json(tmp_path / "report.json")
    payload = report.to_json()
    assert payload["diagnostics"] == []
    assert payload["summary"]["translatedCount"] == 1
    return (tmp_path / payload["artifacts"][0]["path"]).read_text(encoding="utf-8")


def _compile(source, tmp_path, *, profile="cs_6_0", flags=()):
    compiler = shutil.which("dxc")
    if compiler is None:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail("DXC is required")
        pytest.skip("DXC is not installed")
    artifact = tmp_path / "translated.hlsl"
    module = tmp_path / "translated.dxil"
    artifact.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [
            compiler,
            "-T",
            profile,
            "-E",
            "CSMain",
            "-WX",
            *flags,
            str(artifact),
            "-Fo",
            str(module),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (tmp_path / "compiler.json").write_text(
        json.dumps(
            {
                "command": result.args,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert module.stat().st_size > 0
    return artifact, module


@pytest.mark.parametrize("operation", ["fetch_add", "exchange"])
@pytest.mark.parametrize("aggregate", [False, True])
def test_project_float_atomics_compile_with_original_value(
    tmp_path, operation, aggregate
):
    generated = _translate(tmp_path, _source(operation, aggregate, False))
    assert "InterlockedAdd(" not in generated
    assert "__crossgl_float_atomic_" in generated
    assert generated.count("index++") == 2
    _compile(generated, tmp_path)


@pytest.mark.parametrize("operation", ["fetch_add", "exchange"])
def test_float_atomic_helper_keeps_resource_pointer_offset(tmp_path, operation):
    generated = _translate(tmp_path, _pointer_source(operation))
    assert "InterlockedAdd(" not in generated
    assert "Interlocked" in generated
    _compile(generated, tmp_path)


def _pointer_source(operation):
    return f"""#include <metal_stdlib>
    using namespace metal;
    struct Counter {{ uint before; atomic_float value; uint after; }};
    float update(device Counter* counters, uint offset, float value) {{
        return atomic_{operation}_explicit(&counters[offset].value, value, memory_order_relaxed);
    }}
    kernel void update_values(device Counter* counters [[buffer(0)]],
                              device uint* results [[buffer(1)]],
                              uint tid [[thread_position_in_grid]]) {{
        device Counter* base = counters + tid * 2;
        results[tid] = as_type<uint>(update(base, 1u, 1.5f));
    }}"""


def _crossgl(
    body, declaration="RWStructuredBuffer<float> counters @register(u0);", helpers=""
):
    return f"""shader FloatAtomics {{
        {declaration}
        {helpers}
        compute {{ @numthreads(1, 1, 1) void main(uvec3 tid @gl_GlobalInvocationID) {{
            {body}
        }} }}
    }}"""


@pytest.mark.parametrize(
    "body",
    [
        "atomicAdd(counters[tid.x], 1.5);",
        "float old; atomicExchange(counters[tid.x], 1.5, old);",
        "float old = atomicAdd(counters[tid.x], 1.5) + atomicExchange(counters[tid.x], 2.0);",
        "float old = tid.x == 0u ? atomicAdd(counters[tid.x], 1.5) : atomicExchange(counters[tid.x], 2.0);",
        "bool old = tid.x == 0u && atomicAdd(counters[tid.x], 1.5) > 0.0;",
        "float old = sqrt(atomicAdd(counters[tid.x], 1.5));",
    ],
)
def test_float_atomic_expression_contexts_compile(tmp_path, body):
    generated = HLSLCodeGen().generate_stage(parse(_crossgl(body)), "compute")
    assert "atomicAdd(" not in generated
    assert "atomicExchange(" not in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize("element", ["float", "uint"])
def test_canonical_buffer_load_atomic_uses_storage(tmp_path, element):
    source = _crossgl(
        f"{element} old = atomicAdd(buffer_load(counters, tid.x).value, {('1.5' if element == 'float' else '1u')});",
        f"struct Counter {{ uint before; {element} value; }}; RWStructuredBuffer<Counter> counters @register(u0);",
    )
    generated = HLSLCodeGen().generate_stage(parse(source), "compute")
    assert ".Load(" not in generated
    _compile(generated, tmp_path)


def test_source_buffer_load_is_not_reinterpreted_as_storage():
    codegen = HLSLCodeGen()
    codegen.global_variable_types["counters"] = "RWStructuredBuffer<float>"
    call = FunctionCallNode("buffer_load", [IdentifierNode("counters"), 0])
    assert codegen.hlsl_typed_buffer_atomic_load_access(call) is not None
    codegen.function_return_types["buffer_load"] = "float"
    assert codegen.hlsl_typed_buffer_atomic_load_access(call) is None


@pytest.mark.parametrize(
    "declaration,target",
    [
        ("RWBuffer<float> counters @register(u0);", "counters[tid.x]"),
        ("RWStructuredBuffer<float4> counters @register(u0);", "counters[tid.x].y"),
        (
            "struct Counter { float values[4]; }; RWStructuredBuffer<Counter> counters @register(u0);",
            "counters[tid.x].values[1]",
        ),
        ("RWStructuredBuffer<float> counters[2] @register(u0);", "counters[1][tid.x]"),
    ],
)
def test_float_atomic_storage_shapes_compile(tmp_path, declaration, target):
    generated = HLSLCodeGen().generate_stage(
        parse(_crossgl(f"float old = atomicAdd({target}, 1.5);", declaration)),
        "compute",
    )
    _compile(generated, tmp_path)


def test_float_atomic_helper_avoids_identifiers_in_later_functions(tmp_path):
    body = "float old = atomicAdd(counters[tid.x], 1.5);"
    codegen = HLSLCodeGen()
    generated = codegen.generate_stage(parse(_crossgl(body)), "compute")
    name = re.search(r"void (__crossgl_float_atomic_\w+)\(", generated).group(1)
    source = _crossgl(body, helpers=f"float later(float {name}) {{ return {name}; }}")
    generated = codegen.generate_stage(parse(source), "compute")
    assert f"void {name}_(" in generated
    assert f"void {name}(" not in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize(
    "declaration,body,diagnostic",
    [
        (
            "StructuredBuffer<float> counters @register(t0);",
            "float old = atomicAdd(counters[0], 1.5);",
            "cannot write readonly",
        ),
        (
            "RWStructuredBuffer<float> counters @register(u0);",
            "float old = atomicMin(counters[0], 1.5);",
            "int or uint target",
        ),
        (
            "RWStructuredBuffer<float> counters @register(u0);",
            "float old = atomicAdd(counters[0], 1u);",
            "must be scalar float",
        ),
        (
            "RWStructuredBuffer<float> counters @register(u0);",
            "uint old; atomicAdd(counters[0], 1.5, old);",
            "original argument must be scalar float",
        ),
    ],
)
def test_float_atomic_invalid_operands_fail(declaration, body, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        HLSLCodeGen().generate_stage(parse(_crossgl(body, declaration)), "compute")


def _case(operation, aggregate, contended):
    if contended:
        initial = [0]
        values = [
            _bits(1.5 if operation == "fetch_add" else (i + 1) / 2) for i in range(513)
        ]
        expected = None
    else:
        # Binary32 round-to-nearest with flushed denormals, as in the native control.
        triples = [
            (0, 0, 0),
            (0x80000000, 0x80000000, 0x80000000),
            (0, 0x80000000, 0),
            (0x3F800000, 0x3FC00000, 0x40200000),
            (0xBF800000, 0x3FC00000, 0x3F000000),
            (0x7F800000, 0xFF800000, None),
            (0xFF800000, 0x3F800000, 0xFF800000),
            (0x7FC12345, 0x3F800000, None),
            (0x3F800000, 0xFFC54321, None),
            (0x7F7FFFFF, 0x7F7FFFFF, 0x7F800000),
            (0x00800000, 0x80800000, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0xFF7FFFFF, 0xFF7FFFFF, 0xFF800000),
        ]
        initial, values, expected = map(list, zip(*triples))
    storage = (
        [
            word
            for i, word in enumerate(initial)
            for word in (0x12340000 + i, word, 0x56780000 + i)
        ]
        if aggregate
        else initial[:]
    )
    return initial, values, storage, expected


def _check_case(
    outputs, initial, values, storage, expected, operation, aggregate, contended
):
    previous = outputs["results"][::3]
    assert outputs["results"][1::3] == (
        [1] * len(values) if contended else list(range(1, len(values) + 1))
    )
    assert outputs["results"][2::3] == list(range(1, len(values) + 1))
    final = outputs["counters"]
    assert len(final) == len(storage)
    if aggregate:
        assert final[::3] == storage[::3]
        assert final[2::3] == storage[2::3]
        final = final[1::3]
    if contended:
        if operation == "fetch_add":
            assert sorted(previous) == [_bits(i * 1.5) for i in range(len(values))]
            assert final == [_bits(len(values) * 1.5)]
        else:
            assert sorted(previous + final) == sorted(initial + values)
            assert final[0] in values
    else:
        assert previous == initial
        expected = values if operation == "exchange" else expected
        for got, want in zip(final, expected):
            if want is None:
                assert got & 0x7F800000 == 0x7F800000 and got & 0x007FFFFF
            else:
                assert got == want


def _dispatch(tmp_path, generated, specifications, count):
    artifact, module = _compile(generated, tmp_path)
    buffers = {}
    for name, slot, element, words, stride, readonly in specifications:
        buffers[name] = NativeRuntimeBufferBinding(
            name=name,
            binding=RuntimeResourceBinding(
                name=name,
                kind="buffer",
                set=0,
                binding=slot,
                type_name=("" if readonly else "RW") + f"StructuredBuffer<{element}>",
                access="read" if readonly else "read_write",
                metadata={"byteStride": stride},
            ),
            source="input" if readonly else "expectedOutput",
            dtype="uint32",
            shape=(len(words),),
            value=words,
        )
    request = NativeRuntimeDispatchRequest(
        target="directx",
        artifact={"target": "directx"},
        artifact_path=artifact,
        module_path=module,
        loaded_artifact=module.read_bytes(),
        buffers=buffers,
        constants={},
        entry_point="CSMain",
        dispatch=RuntimeDispatchGeometry(
            entry_point="CSMain",
            workgroup_size=(1, 1, 1),
            workgroup_count=(count, 1, 1),
        ),
    )
    state = SimpleNamespace(details={})
    (tmp_path / "inputs.json").write_text(
        json.dumps(specifications, indent=2), encoding="utf-8"
    )
    try:
        outputs = DirectXComputeRuntime().dispatch(None, state, request)
    except Exception as error:
        (tmp_path / "runtime-error.json").write_text(
            json.dumps(
                {
                    "message": str(error),
                    "details": getattr(error, "details", {}),
                    "runtime": state.details,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    outputs = {name: data["values"] for name, data in outputs.items()}
    (tmp_path / "readback.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8"
    )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
                "runtime": state.details,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return outputs


@pytest.fixture
def directx_runtime():
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for DirectX numerical execution")
    assert sys.platform == "win32", "DirectX numerical execution requires Windows"


@pytest.mark.parametrize("operation", ["fetch_add", "exchange"])
@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("contended", [False, True])
def test_directx_float_atomics_execute(
    tmp_path, directx_runtime, operation, aggregate, contended
):
    initial, values, storage, expected = _case(operation, aggregate, contended)
    generated = _translate(tmp_path, _source(operation, aggregate, contended))
    outputs = _dispatch(
        tmp_path,
        generated,
        [
            (
                "counters",
                0,
                "Counter" if aggregate else "float",
                storage,
                12 if aggregate else 4,
                False,
            ),
            ("values", 1, "uint", values, 4, True),
            ("results", 2, "uint", [0xDEADBEEF] * (3 * len(values)), 4, False),
        ],
        len(values),
    )
    _check_case(
        outputs, initial, values, storage, expected, operation, aggregate, contended
    )


@pytest.mark.parametrize("operation", ["fetch_add", "exchange"])
def test_directx_atomic_pointer_offsets_execute(tmp_path, directx_runtime, operation):
    count = 97
    storage = [
        word
        for i in range(2 * count)
        for word in (0x12340000 + i, _bits(i / 2), 0x56780000 + i)
    ]
    generated = _translate(tmp_path, _pointer_source(operation))
    outputs = _dispatch(
        tmp_path,
        generated,
        [
            ("counters", 0, "Counter", storage, 12, False),
            ("results", 1, "uint", [0xDEADBEEF] * count, 4, False),
        ],
        count,
    )
    expected = storage[:]
    for i in range(count):
        expected[6 * i + 4] = _bits(
            (2 * i + 1) / 2 + 1.5 if operation == "fetch_add" else 1.5
        )
    assert outputs["results"] == [_bits((2 * i + 1) / 2) for i in range(count)]
    assert outputs["counters"] == expected


def _conditional_source():
    return _crossgl(
        """
        uint i = tid.x;
        bool take = (i % 2u) == 0u;
        bool a = take && atomicAdd(counters[i], 1.5) > 0.0;
        bool b = take || atomicAdd(counters[i], 2.5) > 0.0;
        float old = take ? atomicExchange(counters[i], 4.0) : atomicAdd(counters[i], 3.5);
        float root = sqrt(atomicAdd(counters[i], 0.0));
        results[i * 4] = a ? 1u : 0u;
        results[i * 4 + 1] = b ? 1u : 0u;
        results[i * 4 + 2] = asuint(old);
        results[i * 4 + 3] = asuint(root);
    """,
        "RWStructuredBuffer<float> counters @register(u0); RWStructuredBuffer<uint> results @register(u1);",
    )


def test_float_atomic_control_flow_compiles(tmp_path):
    generated = HLSLCodeGen().generate_stage(parse(_conditional_source()), "compute")
    assert generated.count("bool __crossgl_atomic_condition_") == 2
    _compile(generated, tmp_path)


def test_directx_atomic_control_flow_executes(tmp_path, directx_runtime):
    count = 97
    generated = HLSLCodeGen().generate_stage(parse(_conditional_source()), "compute")
    outputs = _dispatch(
        tmp_path,
        generated,
        [
            ("counters", 0, "float", [_bits(3.0)] * count, 4, False),
            ("results", 1, "uint", [0xDEADBEEF] * (count * 4), 4, False),
        ],
        count,
    )
    assert outputs["counters"] == [
        _bits(4.0 if i % 2 == 0 else 9.0) for i in range(count)
    ]
    assert outputs["results"] == [
        word
        for i in range(count)
        for word in (
            int(i % 2 == 0),
            1,
            _bits(4.5 if i % 2 == 0 else 5.5),
            _bits(2.0 if i % 2 == 0 else 3.0),
        )
    ]


@pytest.mark.parametrize("operation", ["fetch_add", "exchange"])
@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("contended", [False, True])
def test_original_metal_matches_atomic_oracle(
    tmp_path, metal_word_runner, operation, aggregate, contended
):
    initial, values, storage, expected = _case(operation, aggregate, contended)
    source = _source(operation, aggregate, contended)
    artifact, module = _compile_metal(source, "metal", tmp_path)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "inputs": [storage, values],
                "outputCount": len(values) * 3,
                "threadCount": len(values),
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [str(metal_word_runner), str(module), "update_values", str(request)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    (tmp_path / "readback.json").write_text(result.stdout, encoding="utf-8")
    data = json.loads(result.stdout)
    assert data["buffers"][1] == values
    outputs = {"counters": data["buffers"][0], "results": data["buffers"][2]}
    _check_case(
        outputs, initial, values, storage, expected, operation, aggregate, contended
    )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
                "device": data["device"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

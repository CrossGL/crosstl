"""Isolate WARP wave-width effects without changing arithmetic or dispatch extents."""

import hashlib
import json
import os
import re
import struct
import sys
from types import SimpleNamespace

import pytest

from crosstl.translator import parse
from tests.test_translator.test_directx_cooperative_matrix import (
    PRECISE_OPERANDS,
    PRECISE_SCALAR,
    _case,
    _codegen,
    _source,
)
from tests.test_translator.test_directx_float_atomics import _compile
from tests.test_translator.test_directx_pipeline_debug import _query_warp_adapter
from tests.test_translator.test_directx_software_reductions import _execute_words

REQUIRE_ENV = "CROSTL_REQUIRE_DIRECTX_WAVE_SIZE_COMPARISON"
WAVE_SIZES = (4, 8, 16, 32, 64, 128)
PROGRAMS = ("scalar", "matrix", "rounding-matrix")
WITNESS = """
StructuredBuffer<uint> inputWords : register(t0);
RWStructuredBuffer<uint> outputWords : register(u1);
[numthreads(32, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    outputWords[id.x] = WaveGetLaneCount();
}
"""


def _with_wave_size(source, width):
    if width is not None and (type(width) is not int or width not in WAVE_SIZES):
        raise ValueError("Unsupported wave size")
    if "WaveSize" in source or source.count("[numthreads(") != 1:
        raise ValueError("Expected one unannotated compute entry")
    if width is None:
        return source
    return source.replace("[numthreads(", f"[WaveSize({width})]\n[numthreads(")


def _program(name):
    if name == "scalar":
        return PRECISE_SCALAR, PRECISE_OPERANDS * 32, [0] * 32, (32, 1, 1), (1, 1, 1)
    if name not in PROGRAMS:
        raise ValueError("Unknown arithmetic control")
    rounding = name == "rounding-matrix"
    shape = (16, 8, 1) if rounding else (32, 4, 1)
    source = _codegen().generate_stage(
        parse(_source(width=shape[0], height=shape[1])), "compute"
    )
    words, expected = _case(rounding=rounding)
    return source, words, expected, shape, (3, 1, 1)


def _compile_wave(source, work, *, width, flags=()):
    assembly = work / "translated.ll"
    artifact, module = _compile(
        source, work, profile="cs_6_6", flags=(*flags, "-Fc", str(assembly))
    )
    text = assembly.read_text(encoding="utf-8")
    widths = re.findall(r"^; WaveSize=(\d+)\s*$", text, re.MULTILINE)
    assert widths == ([] if width is None else [str(width)]), widths
    if source == _with_wave_size(PRECISE_SCALAR, width):
        operations = re.findall(r"^\s*%\d+ = (fmul .+|fadd .+)$", text, re.MULTILINE)
        assert len(operations) == 2, operations
        assert operations[0].startswith("fmul float "), operations
        assert operations[1].startswith("fadd float "), operations
    return artifact, module


def _word_hash(words):
    return hashlib.sha256(struct.pack(f"<{len(words)}I", *words)).hexdigest()


def _compare(directory, source, words, expected, *, width, shape, groups, device):
    from crosstl.project.directx_runtime import _device_identity, execute_buffer_views

    generated = _with_wave_size(source, width)
    report = {
        "profile": "cs_6_6",
        "requestedWaveSize": width,
        "replacesRequiredGate": False,
        "unannotatedSourceSHA256": hashlib.sha256(source.encode()).hexdigest(),
        "sourceTextSHA256": hashlib.sha256(generated.encode()).hexdigest(),
        "inputsSHA256": _word_hash(words),
        "referenceSHA256": _word_hash(expected),
        "workgroupSize": shape,
        "workgroupCount": groups,
        "bufferByteOffset": 16,
        "requestedAdapter": _device_identity(device),
        "adapterSelection": "EnumWarpAdapter",
        "adapterLuid": device.adapter_luid,
    }

    def compile_source(text, work, **options):
        assert text == generated and options == {"flags": ()}
        artifact, module = _compile_wave(text, work, width=width)
        assert artifact.read_text(encoding="utf-8") == generated
        report["sourceSHA256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        report["moduleSHA256"] = hashlib.sha256(module.read_bytes()).hexdigest()
        return artifact, module

    def execute(*args):
        return execute_buffer_views(*args, device=device)

    try:
        actual = _execute_words(
            directory,
            generated,
            words,
            expected,
            workgroup_size=shape,
            workgroup_count=groups,
            buffer_byte_offset=16,
            buffer_view_executor=execute,
            module_compiler=compile_source,
        )
        report.update(values=actual, matchesReference=actual == expected)
        evidence = json.loads((directory / "evidence.json").read_text())
        assert evidence["artifactSha256"] == report["sourceSHA256"]
        assert evidence["moduleSha256"] == report["moduleSHA256"]
        native = evidence["runtime"]["directxRuntime"]
        assert native["adapterIdentity"] == report["requestedAdapter"]
        assert native["requestedPipelineStateFlags"] == 0
        assert actual == expected, "Wave-width control differs from exact reference"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        details = getattr(error, "details", None)
        if details is not None:
            report["error"]["details"] = details
        raise
    finally:
        (directory / "wave-size-comparison.json").write_text(
            json.dumps(report, indent=2)
        )


@pytest.fixture(scope="session")
def wave_adapter(tmp_path_factory):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for WARP wave-size comparison")
    assert sys.platform == "win32", "WARP wave-size comparison requires Windows"
    return _query_warp_adapter(tmp_path_factory.mktemp("wave-adapter"))


@pytest.fixture
def native_wave_comparison(monkeypatch, wave_adapter):
    monkeypatch.setenv("CROSTL_REQUIRE_DIRECTX_FLOAT_ATOMICS", "1")
    return wave_adapter


@pytest.mark.parametrize("width", (None, *WAVE_SIZES))
@pytest.mark.parametrize("program", PROGRAMS)
def test_wave_size_arithmetic_executes(
    tmp_path, native_wave_comparison, program, width
):
    source, words, expected, shape, groups = _program(program)
    _compare(
        tmp_path,
        source,
        words,
        expected,
        width=width,
        shape=shape,
        groups=groups,
        device=native_wave_comparison,
    )


@pytest.mark.parametrize("width", WAVE_SIZES)
def test_requested_wave_size_executes(tmp_path, native_wave_comparison, width):
    _compare(
        tmp_path,
        WITNESS,
        [0] * 32,
        [width] * 32,
        width=width,
        shape=(32, 1, 1),
        groups=(1, 1, 1),
        device=native_wave_comparison,
    )


@pytest.mark.parametrize("width", (None, *WAVE_SIZES))
@pytest.mark.parametrize("program", (*PROGRAMS, "witness"))
def test_wave_size_annotation_compiles_without_arithmetic_changes(
    tmp_path, program, width
):
    source = WITNESS if program == "witness" else _program(program)[0]
    generated = _with_wave_size(source, width)
    assert generated.replace(f"[WaveSize({width})]\n", "") == source
    _compile_wave(generated, tmp_path, width=width)


@pytest.mark.parametrize("width", (True, False, 0, 1, 3, 256, 4.0, "4"))
def test_wave_size_rejects_invalid_width(width):
    with pytest.raises(ValueError, match="Unsupported wave size"):
        _with_wave_size(PRECISE_SCALAR, width)


@pytest.mark.parametrize(
    "source", ("", "[WaveSize(4)]" + PRECISE_SCALAR, PRECISE_SCALAR * 2)
)
def test_wave_size_rejects_ambiguous_entry(source):
    with pytest.raises(ValueError, match="one unannotated compute entry"):
        _with_wave_size(source, 4)


@pytest.mark.parametrize("width", (None, 4))
@pytest.mark.parametrize(
    "damage", (None, "wave", "duplicate-wave", "multiply", "add", "missing-add")
)
def test_wave_size_compiler_contract_rejects_changed_listing(
    tmp_path, monkeypatch, width, damage
):
    source = _with_wave_size(PRECISE_SCALAR, width)
    listing = "" if width is None else "; WaveSize=4\n"
    if damage == "wave":
        listing = "; WaveSize=8\n"
    if damage == "duplicate-wave":
        listing = "; WaveSize=4\n; WaveSize=4\n"
    listing += "  %1 = fmul float %2, %3\n  %4 = fadd float %1, %5\n"
    if damage == "multiply":
        listing = listing.replace("fmul float", "fmul fast float")
    if damage == "add":
        listing = listing.replace("fadd float", "fadd fast float")
    if damage == "missing-add":
        listing = listing.replace("  %4 = fadd float %1, %5\n", "")

    def compile_source(text, work, **options):
        assert text == source and work == tmp_path
        assert options == {
            "profile": "cs_6_6",
            "flags": ("-Fc", str(work / "translated.ll")),
        }
        artifact, module = work / "translated.hlsl", work / "translated.dxil"
        artifact.write_text(source)
        module.write_bytes(b"DXBC module")
        (work / "translated.ll").write_text(listing)
        return artifact, module

    monkeypatch.setattr(sys.modules[__name__], "_compile", compile_source)
    if damage:
        with pytest.raises(AssertionError):
            _compile_wave(source, tmp_path, width=width)
    else:
        _compile_wave(source, tmp_path, width=width)


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize(
    "failure",
    (None, "compile", "dispatch", "numerical", "adapter", "flags", "source", "module"),
)
def test_wave_size_comparison_preserves_inputs_and_failure_receipts(
    tmp_path, monkeypatch, failure, newline
):
    from crosstl.project import directx_runtime

    device = SimpleNamespace(
        name="Explicit WARP",
        is_hardware=False,
        vendor_id=0x1414,
        dedicated_video_memory=0,
        dedicated_system_memory=0,
        shared_system_memory=4096,
        adapter_luid=[17, 0],
    )
    module = sys.modules[__name__]
    generated = _with_wave_size(PRECISE_SCALAR, 4)
    calls = []

    def compile_source(source, work, *, width):
        calls.append("compile")
        assert source == generated and width == 4 and work == tmp_path
        if failure == "compile":
            raise RuntimeError("Compiler failed")
        artifact, binary = work / "translated.hlsl", work / "translated.dxil"
        artifact.write_bytes(source.replace("\n", newline).encode())
        binary.write_bytes(b"DXBC compiled module")
        return artifact, binary

    def native_views(*args, **options):
        calls.append("dispatch")
        assert args == ("unchanged request",) and options == {"device": device}
        if failure == "dispatch":
            raise RuntimeError("Dispatch failed")

    def execute(work, source, words, expected, **options):
        assert work == tmp_path and source == generated
        assert words == PRECISE_OPERANDS * 32 and expected == [0] * 32
        assert options["workgroup_size"] == (32, 1, 1)
        assert options["workgroup_count"] == (1, 1, 1)
        assert options["buffer_byte_offset"] == 16
        artifact, binary = options["module_compiler"](source, work, flags=())
        options["buffer_view_executor"]("unchanged request")
        (work / "evidence.json").write_text(
            json.dumps(
                {
                    "artifactSha256": (
                        "different"
                        if failure == "source"
                        else hashlib.sha256(artifact.read_bytes()).hexdigest()
                    ),
                    "moduleSha256": (
                        "different"
                        if failure == "module"
                        else hashlib.sha256(binary.read_bytes()).hexdigest()
                    ),
                    "runtime": {
                        "directxRuntime": {
                            "adapterIdentity": (
                                {}
                                if failure == "adapter"
                                else directx_runtime._device_identity(device)
                            ),
                            "requestedPipelineStateFlags": int(failure == "flags"),
                        }
                    },
                }
            )
        )
        return [1 if failure == "numerical" else 0] * 32

    monkeypatch.setattr(module, "_compile_wave", compile_source)
    monkeypatch.setattr(module, "_execute_words", execute)
    monkeypatch.setattr(directx_runtime, "execute_buffer_views", native_views)
    arguments = (tmp_path, PRECISE_SCALAR, PRECISE_OPERANDS * 32, [0] * 32)
    options = {"width": 4, "shape": (32, 1, 1), "groups": (1, 1, 1), "device": device}
    if failure:
        with pytest.raises((AssertionError, RuntimeError)):
            _compare(*arguments, **options)
    else:
        _compare(*arguments, **options)
    assert calls == (["compile"] if failure == "compile" else ["compile", "dispatch"])
    report = json.loads((tmp_path / "wave-size-comparison.json").read_text())
    assert report["profile"] == "cs_6_6" and report["requestedWaveSize"] == 4
    assert report["inputsSHA256"] == _word_hash(PRECISE_OPERANDS * 32)
    assert report["referenceSHA256"] == _word_hash([0] * 32)
    assert report["replacesRequiredGate"] is False
    assert report["sourceTextSHA256"] == hashlib.sha256(generated.encode()).hexdigest()
    if failure != "compile":
        assert (
            report["sourceSHA256"]
            == hashlib.sha256(generated.replace("\n", newline).encode()).hexdigest()
        )
    assert report["adapterSelection"] == "EnumWarpAdapter"
    assert report["adapterLuid"] == [17, 0]
    assert ("error" in report) == bool(failure)
    if failure == "numerical":
        assert report["values"] == [1] * 32 and report["matchesReference"] is False

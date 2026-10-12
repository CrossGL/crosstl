"""Compare WARP pipeline modes without recompiling the shader or changing inputs."""

import hashlib
import json
import os
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
from tests.test_translator.test_directx_software_reductions import _execute_words

REQUIRE_ENV = "CROSTL_REQUIRE_DIRECTX_TOOL_DEBUG_COMPARISON"

WARP_ADAPTER_SOURCE = r"""
#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

void check(HRESULT result, const char* operation) {
    if (FAILED(result)) throw std::runtime_error(operation);
}
void integer(std::ostream& output, uint64_t value, unsigned bytes) {
    for (unsigned index = 0; index < bytes; ++index)
        output.put(static_cast<char>((value >> (index * 8)) & 255));
}
int wmain(int argc, wchar_t** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected adapter receipt path");
        Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
        check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "EnumWarpAdapter");
        DXGI_ADAPTER_DESC1 description = {};
        check(adapter->GetDesc1(&description), "GetDesc1 WARP");
        unsigned length = 0;
        while (length < 128 && description.Description[length]) ++length;
        if (!length || length == 128) throw std::runtime_error("invalid WARP name");
        std::ofstream output(std::filesystem::path(argv[1]), std::ios::binary);
        if (!output) throw std::runtime_error("cannot open adapter receipt");
        integer(output, 0x31505257u, 4);
        integer(output, description.Flags, 4);
        integer(output, description.VendorId, 4);
        integer(output, description.DedicatedVideoMemory, 8);
        integer(output, description.DedicatedSystemMemory, 8);
        integer(output, description.SharedSystemMemory, 8);
        integer(output, description.AdapterLuid.LowPart, 4);
        integer(output, static_cast<uint32_t>(description.AdapterLuid.HighPart), 4);
        integer(output, length, 4);
        for (unsigned index = 0; index < length; ++index)
            integer(output, static_cast<uint16_t>(description.Description[index]), 2);
        output.close();
        if (!output) throw std::runtime_error("cannot write adapter receipt");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
"""


def _decode_warp_adapter(payload):
    header = struct.Struct("<IIIQQQIII")
    if len(payload) < header.size:
        raise ValueError("Truncated WARP adapter receipt")
    magic, flags, vendor, video, system, shared, low, high, length = header.unpack_from(
        payload
    )
    if (
        magic != 0x31505257
        or not 0 < length < 128
        or len(payload) != header.size + length * 2
    ):
        raise ValueError("Invalid WARP adapter receipt")
    name = payload[header.size :].decode("utf-16-le")
    if "\0" in name:
        raise ValueError("Invalid WARP adapter name")
    return SimpleNamespace(
        name=name,
        is_hardware=not bool(flags & 2),
        vendor_id=vendor,
        dedicated_video_memory=video,
        dedicated_system_memory=system,
        shared_system_memory=shared,
        adapter_luid=[low, high],
    )


@pytest.fixture(scope="session")
def warp_adapter(tmp_path_factory):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for WARP pipeline comparison")
    assert sys.platform == "win32", "WARP pipeline comparison requires Windows"
    from crosstl.project.directx_runtime import _compiler_environment, _run

    work = tmp_path_factory.mktemp("warp-adapter")
    source, executable, receipt = (
        work / "adapter.cpp",
        work / "adapter.exe",
        work / "adapter.bin",
    )
    source.write_text(WARP_ADAPTER_SOURCE, encoding="utf-8")
    compiler, environment = _compiler_environment()
    for label, command in (
        (
            "compile",
            [
                compiler,
                "/nologo",
                "/std:c++17",
                "/EHsc",
                "/W4",
                "/WX",
                "/O2",
                str(source),
                f"/Fe{executable}",
                f"/Fo{work / 'adapter.obj'}",
                "/link",
                "dxgi.lib",
            ],
        ),
        ("query", [str(executable), str(receipt)]),
    ):
        result = _run(command, timeout=60, env=environment)
        (work / f"{label}.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "returncode": result.returncode,
                    "stdout": result.stdout.decode(errors="replace"),
                    "stderr": result.stderr.decode(errors="replace"),
                },
                indent=2,
            )
        )
        assert result.returncode == 0, (label, result.stdout, result.stderr)
    return _decode_warp_adapter(receipt.read_bytes())


def _compare(
    directory, source, words, expected, *, shape, groups, flags=(), device=None
):
    compiled = directory / "compiled"
    compiled.mkdir()
    artifact, module = _compile(source, compiled, flags=flags)
    source_bytes, module_bytes = artifact.read_bytes(), module.read_bytes()
    report = {
        "sourceSHA256": hashlib.sha256(source_bytes).hexdigest(),
        "moduleSHA256": hashlib.sha256(module_bytes).hexdigest(),
        "replacesRequiredGate": False,
        "modes": {},
    }
    if device is not None:
        from crosstl.project.directx_runtime import _device_identity

        report["requestedAdapter"] = _device_identity(device)
        report["adapterSelection"] = "EnumWarpAdapter"
        report["adapterLuid"] = device.adapter_luid

    def same_module(text, work, **options):
        assert text == source and options == {"flags": flags}
        assert artifact.read_bytes() == source_bytes
        assert module.read_bytes() == module_bytes
        return artifact, module

    for mode, debug in (("default", False), ("tool-debug", True)):
        work = directory / mode
        work.mkdir()

        def execute(*args, enabled=debug):
            import compushady

            from crosstl.project.directx_runtime import execute_buffer_views

            return execute_buffer_views(
                *args,
                device=(
                    device if device is not None else compushady.get_current_device()
                ),
                tool_debug=enabled,
            )

        outcome = {"requestedPipelineStateFlags": int(debug)}
        report["modes"][mode] = outcome
        try:
            actual = _execute_words(
                work,
                source,
                words,
                expected,
                workgroup_size=shape,
                workgroup_count=groups,
                compile_flags=flags,
                buffer_byte_offset=16,
                buffer_view_executor=execute,
                module_compiler=same_module,
            )
            evidence = json.loads((work / "evidence.json").read_text())
            assert evidence["moduleSha256"] == report["moduleSHA256"]
            assert evidence["artifactSha256"] == report["sourceSHA256"]
            native = evidence["runtime"]["directxRuntime"]
            if device is not None:
                assert native["adapterIdentity"] == report["requestedAdapter"]
            assert native["requestedPipelineStateFlags"] == int(debug)
            if debug:
                assert native["workerPipelineStateFlags"] == 1
            outcome.update(
                values=actual,
                matchesReference=actual == expected,
                requestSHA256=native["requestSHA256"],
            )
        except Exception as error:
            outcome["error"] = {"type": type(error).__name__, "message": str(error)}
            details = getattr(error, "details", None)
            if details is not None:
                outcome["error"]["details"] = details
        finally:
            (directory / "comparison.json").write_text(json.dumps(report, indent=2))
    assert artifact.read_bytes() == source_bytes and module.read_bytes() == module_bytes
    modes = list(report["modes"].values())
    assert not any("error" in mode for mode in modes), report
    assert modes[0]["requestSHA256"] == modes[1]["requestSHA256"], report
    assert all(mode["matchesReference"] for mode in modes), report


@pytest.fixture
def native_comparison(monkeypatch, warp_adapter):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for WARP pipeline comparison")
    assert sys.platform == "win32", "WARP pipeline comparison requires Windows"
    monkeypatch.setenv("CROSTL_REQUIRE_DIRECTX_FLOAT_ATOMICS", "1")
    return warp_adapter


@pytest.mark.parametrize("flags", [(), ("-Gis",), ("-Gis", "-Od")])
def test_precise_scalar_tool_debug_comparison_executes(
    tmp_path, native_comparison, flags
):
    _compare(
        tmp_path,
        PRECISE_SCALAR,
        PRECISE_OPERANDS * 32,
        [0] * 32,
        shape=(32, 1, 1),
        groups=(1, 1, 1),
        flags=flags,
        device=native_comparison,
    )


@pytest.mark.parametrize("shape,rounding", [((32, 4, 1), False), ((16, 8, 1), True)])
def test_matrix_tool_debug_comparison_executes(
    tmp_path, native_comparison, shape, rounding
):
    source = _codegen().generate_stage(
        parse(_source(width=shape[0], height=shape[1])), "compute"
    )
    words, expected = _case(rounding=rounding)
    _compare(
        tmp_path,
        source,
        words,
        expected,
        shape=shape,
        groups=(3, 1, 1),
        device=native_comparison,
    )


@pytest.mark.parametrize(
    "damage",
    [None, "truncated", "trailing", "version", "empty", "long", "nul", "encoding"],
)
def test_warp_adapter_receipt_requires_complete_identity(damage):
    name = "Explicit WARP"
    magic, length = 0x31505257, len(name)
    encoded = name.encode("utf-16-le")
    if damage == "version":
        magic = 0
    if damage == "empty":
        length = 0
    if damage == "long":
        length = 128
    if damage == "nul":
        encoded = b"\0\0" + encoded[2:]
    if damage == "encoding":
        encoded = b"\0\xd8" + encoded[2:]
    payload = (
        struct.pack("<IIIQQQIII", magic, 2, 0x1414, 0, 0, 4096, 17, 0, length) + encoded
    )
    if damage == "truncated":
        payload = payload[:12]
    if damage == "trailing":
        payload += b"\0"
    if damage:
        with pytest.raises(ValueError):
            _decode_warp_adapter(payload)
    else:
        from crosstl.project.directx_runtime import _device_identity

        adapter = _decode_warp_adapter(payload)
        assert _device_identity(adapter) == {
            "name": name,
            "isHardware": False,
            "vendorId": 0x1414,
            "dedicatedVideoMemory": 0,
            "dedicatedSystemMemory": 0,
            "sharedSystemMemory": 4096,
        }
        assert adapter.adapter_luid == [17, 0]


@pytest.mark.parametrize(
    "failure", [None, "numerical", "dispatch", "flags", "module", "request", "adapter"]
)
def test_comparison_keeps_both_modes_and_exact_module(tmp_path, monkeypatch, failure):
    from crosstl.project.directx_runtime import _device_identity

    compiled, executed = [], []
    module = sys.modules[__name__]
    device = SimpleNamespace(
        name="Explicit WARP",
        is_hardware=False,
        vendor_id=0x1414,
        dedicated_video_memory=0,
        dedicated_system_memory=0,
        shared_system_memory=4096,
        adapter_luid=[17, 0],
    )

    def compile_source(source, work, **options):
        compiled.append((source, options))
        artifact, binary = work / "shader.hlsl", work / "shader.dxil"
        artifact.write_bytes(b"source")
        binary.write_bytes(b"DXBC compiled shader")
        return artifact, binary

    def execute(work, source, words, expected, **options):
        executed.append(work.name)
        assert source == "source" and words == [1, 2, 3] and expected == [0]
        assert options["workgroup_size"] == (32, 1, 1)
        assert options["workgroup_count"] == (1, 1, 1)
        assert options["buffer_byte_offset"] == 16
        artifact, binary = options["module_compiler"](source, work, flags=("-Gis",))
        debug = work.name == "tool-debug"
        if failure == "dispatch" and not debug:
            raise RuntimeError("Default dispatch failed")
        evidence = {
            "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "moduleSha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "runtime": {
                "directxRuntime": {
                    "adapterIdentity": (
                        {"name": "Another adapter"}
                        if failure == "adapter" and debug
                        else _device_identity(device)
                    ),
                    "requestedPipelineStateFlags": int(debug),
                    "requestSHA256": (
                        "different" if failure == "request" and debug else "same"
                    ),
                    **(
                        {"workerPipelineStateFlags": 0 if failure == "flags" else 1}
                        if debug
                        else {}
                    ),
                }
            },
        }
        if failure == "module" and debug:
            evidence["moduleSha256"] = "different"
        (work / "evidence.json").write_text(json.dumps(evidence))
        return [1] if failure == "numerical" and not debug else [0]

    monkeypatch.setattr(module, "_compile", compile_source)
    monkeypatch.setattr(module, "_execute_words", execute)
    arguments = (tmp_path, "source", [1, 2, 3], [0])
    options = {
        "shape": (32, 1, 1),
        "groups": (1, 1, 1),
        "flags": ("-Gis",),
        "device": device,
    }
    if failure:
        with pytest.raises(AssertionError):
            _compare(*arguments, **options)
    else:
        _compare(*arguments, **options)
    assert compiled == [("source", {"flags": ("-Gis",)})]
    assert executed == ["default", "tool-debug"]
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["replacesRequiredGate"] is False
    assert report["adapterSelection"] == "EnumWarpAdapter"
    assert report["requestedAdapter"] == _device_identity(device)
    assert report["adapterLuid"] == [17, 0]
    assert list(report["modes"]) == executed
    if failure in {"numerical", "dispatch"}:
        assert report["modes"]["tool-debug"]["matchesReference"] is True

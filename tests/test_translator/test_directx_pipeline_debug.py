"""Compare WARP pipeline modes without recompiling the shader or changing inputs."""

import hashlib
import json
import os
import sys

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


def _compare(directory, source, words, expected, *, shape, groups, flags=()):
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
                *args, device=compushady.get_current_device(), tool_debug=enabled
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
        finally:
            (directory / "comparison.json").write_text(json.dumps(report, indent=2))
    assert artifact.read_bytes() == source_bytes and module.read_bytes() == module_bytes
    modes = list(report["modes"].values())
    assert not any("error" in mode for mode in modes), report
    assert modes[0]["requestSHA256"] == modes[1]["requestSHA256"], report
    assert all(mode["matchesReference"] for mode in modes), report


@pytest.fixture
def native_comparison(monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for WARP pipeline comparison")
    assert sys.platform == "win32", "WARP pipeline comparison requires Windows"
    monkeypatch.setenv("CROSTL_REQUIRE_DIRECTX_FLOAT_ATOMICS", "1")


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
    )


@pytest.mark.parametrize("shape,rounding", [((32, 4, 1), False), ((16, 8, 1), True)])
def test_matrix_tool_debug_comparison_executes(
    tmp_path, native_comparison, shape, rounding
):
    source = _codegen().generate_stage(
        parse(_source(width=shape[0], height=shape[1])), "compute"
    )
    words, expected = _case(rounding=rounding)
    _compare(tmp_path, source, words, expected, shape=shape, groups=(3, 1, 1))


@pytest.mark.parametrize(
    "failure", [None, "numerical", "dispatch", "flags", "module", "request"]
)
def test_comparison_keeps_both_modes_and_exact_module(tmp_path, monkeypatch, failure):
    compiled, executed = [], []
    module = sys.modules[__name__]

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
    options = {"shape": (32, 1, 1), "groups": (1, 1, 1), "flags": ("-Gis",)}
    if failure:
        with pytest.raises(AssertionError):
            _compare(*arguments, **options)
    else:
        _compare(*arguments, **options)
    assert compiled == [("source", {"flags": ("-Gis",)})]
    assert executed == ["default", "tool-debug"]
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["replacesRequiredGate"] is False
    assert list(report["modes"]) == executed
    if failure in {"numerical", "dispatch"}:
        assert report["modes"]["tool-debug"]["matchesReference"] is True

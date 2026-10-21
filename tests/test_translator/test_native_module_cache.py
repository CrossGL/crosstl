"""Session reuse of validated native runtime modules."""

from pathlib import Path

import pytest

from crosstl.project.native_module_cache import ValidatedModuleCache
from crosstl.project.runtime_verification import (
    NativeRuntimeValidationCommand,
    OpenGLRuntimeParityAdapter,
    verify_runtime_test_manifest,
)
from tests.test_translator.test_runtime_verification import (
    PASSED,
    FakeNativeRuntime,
    _artifact_report,
    _native_runtime_artifact,
    _native_runtime_manifest,
)


def _specialized_artifact(tmp_path, body="void main() {}\n"):
    artifact_path = tmp_path / "out" / "opengl" / "debug" / "add.glsl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        "#version 450\nlayout(constant_id = 0) const uint tile_size = 1u;\n" + body,
        encoding="utf-8",
    )
    return artifact_path


class RecordingRuntime(FakeNativeRuntime):
    def __init__(self):
        super().__init__()
        self.modules = []

    def load_artifact(self, adapter, state, module_path):
        self.modules.append(Path(module_path).read_bytes())
        return super().load_artifact(adapter, state, module_path)


def _adapter(calls, *, reuse, fail=False):
    def command(command, *, input_text=None):
        calls.append(command)
        if fail:
            return {"returncode": 1, "stderr": "rejected"}
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_bytes(b"\x03\x02#\x07" + bytes([len(calls)]) * 4)
        return {"returncode": 0}

    runtime = RecordingRuntime()
    adapter = OpenGLRuntimeParityAdapter(
        runtime=runtime,
        required_tools=(),
        tool_resolver=lambda tool: f"/fake/{tool}",
        command_runner=command,
        reuse_validated_modules=reuse,
    )
    return adapter, runtime


def _run(tmp_path, adapter):
    return verify_runtime_test_manifest(
        _artifact_report(tmp_path, [_native_runtime_artifact()]),
        _native_runtime_manifest(),
        executors={"opengl": adapter},
    )["results"][0]


def test_reused_module_skips_identical_validation_and_records_each_step(tmp_path):
    _specialized_artifact(tmp_path)
    calls = []
    adapter, runtime = _adapter(calls, reuse=True)
    first = _run(tmp_path, adapter)
    second = _run(tmp_path, adapter)
    assert first["status"] == second["status"] == PASSED
    assert len(calls) == 1
    assert runtime.prepared.module_path.suffix == ".spv"
    # The second execution loads its own copy of the stored module.
    assert len(runtime.modules) == 2 and runtime.modules[0] == runtime.modules[1]
    steps = {
        step["action"]: step
        for step in second["executor"]["details"]["adapterSteps"]
        if step["phase"] == "validate"
    }
    step = steps["compile-glsl-to-opengl-spirv-for-runtime"]
    assert step["status"] == PASSED
    assert len(step["details"]["reusedValidatedModule"]) == 64
    assert adapter.validated_modules.hits == 1
    assert adapter.validated_modules.misses == 1


def test_changed_artifact_bytes_are_validated_again(tmp_path):
    _specialized_artifact(tmp_path)
    calls = []
    adapter, _runtime = _adapter(calls, reuse=True)
    _run(tmp_path, adapter)
    _specialized_artifact(tmp_path, "void main() { }\n")
    _run(tmp_path, adapter)
    assert len(calls) == 2


def test_failed_validation_is_never_reused(tmp_path):
    _specialized_artifact(tmp_path)
    calls = []
    adapter, _runtime = _adapter(calls, reuse=True, fail=True)
    first, second = _run(tmp_path, adapter), _run(tmp_path, adapter)
    assert first["status"] == second["status"] != PASSED
    assert len(calls) == 2


def test_validation_runs_every_time_without_reuse(tmp_path):
    _specialized_artifact(tmp_path)
    calls = []
    adapter, _runtime = _adapter(calls, reuse=False)
    _run(tmp_path, adapter)
    _run(tmp_path, adapter)
    assert len(calls) == 2 and adapter.validated_modules is None


def test_reuse_option_must_be_boolean():
    with pytest.raises(TypeError, match="must be a bool"):
        OpenGLRuntimeParityAdapter(reuse_validated_modules=1)


def test_cache_key_ignores_per_execution_paths_only(tmp_path):
    cache = ValidatedModuleCache()
    keys = []
    for name in ("one", "two"):
        scratch = tmp_path / name
        scratch.mkdir()
        artifact = scratch / "snapshot" / "kernel.metal"
        artifact.parent.mkdir()
        artifact.write_text("kernel void k() {}\n", encoding="utf-8")
        commands = (
            NativeRuntimeValidationCommand(
                command=(
                    "xcrun",
                    "metal",
                    "-c",
                    str(artifact),
                    "-o",
                    str(scratch / "k.air"),
                ),
                action="compile",
                module_path=scratch / "k.air",
            ),
        )
        keys.append(cache.key(artifact, scratch, commands))
    assert keys[0] == keys[1]
    changed = (
        NativeRuntimeValidationCommand(
            command=("xcrun", "metal", "-O0", "-c", str(artifact)),
            action="compile",
        ),
    )
    assert cache.key(artifact, scratch, changed) != keys[1]


def test_reuse_restores_every_produced_module_at_its_new_path(tmp_path):
    # A two-step toolchain (compile to an intermediate, then link) leaves both
    # modules in scratch; a reused execution must leave the same files.
    cache = ValidatedModuleCache()
    artifact = tmp_path / "kernel.metal"
    artifact.write_text("kernel void k() {}\n", encoding="utf-8")
    first = tmp_path / "first"
    first.mkdir()
    (first / "k.air").write_bytes(b"intermediate")
    (first / "k.metallib").write_bytes(b"library")
    cache.store("key", artifact, [first / "k.air", first / "k.metallib"])
    (first / "k.air").write_bytes(b"changed after store")
    second = tmp_path / "second" / "nested"
    assert cache.reuse("key", artifact, [second / "k.air", second / "k.metallib"])
    assert (second / "k.air").read_bytes() == b"intermediate"
    assert (second / "k.metallib").read_bytes() == b"library"
    assert not cache.reuse("other", artifact, [second / "k.air"])
    assert (cache.hits, cache.misses) == (1, 1)
    with pytest.raises(ValueError, match="stored module count"):
        cache.reuse("key", artifact, [second / "k.metallib"])


def test_validation_only_sequences_restore_nothing(tmp_path):
    cache = ValidatedModuleCache()
    artifact = tmp_path / "kernel.spv"
    artifact.write_bytes(b"\x03\x02#\x07")
    cache.store("key", artifact, [artifact])
    assert cache.reuse("key", artifact, [artifact])
    assert sorted(path.name for path in tmp_path.iterdir()) == ["kernel.spv"]

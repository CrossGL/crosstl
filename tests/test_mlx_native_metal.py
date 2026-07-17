import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "demos" / "integrations" / "mlx" / "run_mlx_native_metal.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "mlx-native-metal.yml"
DOCUMENTATION_PATH = ROOT / "demos" / "integrations" / "mlx" / "NATIVE_METAL.md"
README_PATH = ROOT / "demos" / "integrations" / "mlx" / "README.md"


def _load_helper():
    spec = importlib.util.spec_from_file_location("mlx_native_metal_proof", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_checkouts(module, tmp_path, monkeypatch):
    metal_root = tmp_path / "mlx-metal"
    source_hashes = []
    for source, _ in module.EXPECTED_SOURCE_SHA256:
        source_path = metal_root / source
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(f"// pinned fake source: {source}\n", encoding="utf-8")
        source_hashes.append(
            (source, hashlib.sha256(source_path.read_bytes()).hexdigest())
        )
    monkeypatch.setattr(module, "EXPECTED_SOURCE_SHA256", tuple(source_hashes))

    test_root = tmp_path / "mlx-full"
    (test_root / "python" / "tests").mkdir(parents=True)
    (test_root / "setup.py").write_text("# fake pinned package\n", encoding="utf-8")
    output_dir = tmp_path / "evidence"
    return metal_root, test_root, output_dir


class FakeRunner:
    def __init__(
        self,
        module,
        tmp_path,
        *,
        revision=None,
        behavior=None,
        device_type="gpu",
        python_total=None,
        python_skipped=None,
        cpp_cases=None,
        cpp_assertions=None,
        architecture="arm64",
    ):
        self.module = module
        self.revision = revision or module.MLX_COMMIT
        self.behavior = behavior or {}
        self.device_type = device_type
        self.python_total = python_total or module.EXPECTED_UPSTREAM_TEST_COUNT
        self.python_skipped = (
            module.EXPECTED_UPSTREAM_SKIP_COUNT
            if python_skipped is None
            else python_skipped
        )
        self.cpp_cases = cpp_cases or module.EXPECTED_CPP_TEST_CASE_COUNT
        self.cpp_assertions = cpp_assertions or module.EXPECTED_CPP_ASSERTION_COUNT
        self.architecture = architecture
        self.sdk_path = (tmp_path / "MacOSX.sdk").resolve()
        self.metal_path = (tmp_path / "toolchain" / "metal").resolve()
        self.metal_nm_path = (tmp_path / "toolchain" / "metal-nm").resolve()
        self.calls = []

    def __call__(
        self,
        name,
        command,
        *,
        log_dir,
        timeout_seconds,
        cwd=None,
    ):
        del timeout_seconds
        command = tuple(command)
        self.calls.append((name, command, cwd))
        returncode = 0
        stdout = ""
        stderr = ""

        if name in {"checkout-revision", "python-checkout-revision"}:
            stdout = self.revision + "\n"
        elif name in {"checkout-kernel-status", "python-checkout-status"}:
            stdout = ""
        elif name == "toolchain-machine-architecture":
            stdout = self.architecture + "\n"
        elif name == "toolchain-sdk-path":
            stdout = str(self.sdk_path) + "\n"
        elif name == "toolchain-sdk-version":
            stdout = "26.2\n"
        elif name == "toolchain-sdk-build-version":
            stdout = "25C5000\n"
        elif name == "toolchain-metal-path":
            stdout = str(self.metal_path) + "\n"
        elif name == "toolchain-metal-nm-path":
            stdout = str(self.metal_nm_path) + "\n"
        elif name == "toolchain-metal-version":
            stdout = (
                "Apple metal version 32000.1 (metalfe-32000.1)\n"
                "Target: air64-apple-darwin25.0.0\n"
                "Thread model: posix\n"
                "/tmp/ignored\n"
                "InstalledDir: /fake/Metal.xctoolchain/usr/bin\n"
            )
        elif name == "toolchain-metal-nm-version":
            stdout = "Apple LLVM version 32000.1\n"
        elif name == "toolchain-xcode-version":
            stdout = "Xcode 26.2\nBuild version 17C5000\n"
        elif name == "toolchain-macos-version":
            stdout = (
                "ProductName:\t\tmacOS\n"
                "ProductVersion:\t\t26.2\n"
                "BuildVersion:\t\t25C5000\n"
            )
        elif name.startswith("compile-"):
            if self.behavior.get(name) == "nonzero":
                returncode = 1
                stderr = "synthetic Metal compile failure\n"
            elif self.behavior.get(name) != "omit-air":
                air_path = Path(command[command.index("-o") + 1])
                air_path.write_bytes(("AIR:" + name).encode("ascii"))
        elif name == "link-metallib":
            metallib_path = Path(command[command.index("-o") + 1])
            if self.behavior.get(name) == "nonzero":
                returncode = 1
                stderr = "synthetic Metal link failure\n"
            elif self.behavior.get(name) == "empty":
                metallib_path.write_bytes(b"")
            else:
                metallib_path.write_bytes(b"METALLIB")
        elif name == "inspect-metallib":
            if self.behavior.get(name) == "empty":
                stdout = ""
            else:
                stdout = "kernel_alpha\nkernel_beta\n"
        elif name == "python-version":
            stdout = "Python 3.13.5\n"
        elif name == "python-package-build":
            stdout = "Successfully installed mlx\n"
        elif name == "python-mlx-runtime-identity":
            stdout = json.dumps(
                {
                    "defaultDevice": f"Device({self.device_type}, 0)",
                    "defaultDeviceType": self.device_type,
                    "mlxVersion": "0.32.1.dev0+4367c73",
                },
                sort_keys=True,
            )
            stdout += "\n"
        elif name == "python-unittest":
            stderr = (
                "." * 10
                + f"\nRan {self.python_total} tests in 1.234s\n\n"
                + f"OK (skipped={self.python_skipped})\n"
            )
        elif name == "cpp-cmake-version":
            stdout = "cmake version 4.1.0\n"
        elif name == "cpp-ninja-version":
            stdout = "1.13.1\n"
        elif name == "cpp-configure":
            Path(command[command.index("-B") + 1]).mkdir(parents=True, exist_ok=True)
            stdout = "-- Configuring done\n"
        elif name == "cpp-build-tests":
            build_dir = Path(command[command.index("--build") + 1])
            executable = build_dir / "tests" / "tests"
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_bytes(b"fake-doctest-executable")
        elif name == "cpp-aggregate-tests":
            stdout = (
                '[doctest] run with "--help" for options\n'
                "========================================\n"
                f"[doctest] test cases:  {self.cpp_cases} |  "
                f"{self.cpp_cases} passed | 0 failed | 0 skipped\n"
                f"[doctest] assertions: {self.cpp_assertions} | "
                f"{self.cpp_assertions} passed | 0 failed |\n"
                "[doctest] Status: SUCCESS!\n"
            )
        else:
            raise AssertionError(f"unexpected command: {name}: {command}")

        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = log_dir / f"{name}.stdout.log"
        stderr_log = log_dir / f"{name}.stderr.log"
        stdout_log.write_text(stdout, encoding="utf-8")
        stderr_log.write_text(stderr, encoding="utf-8")
        return self.module.CommandResult(
            name=name,
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            cwd=cwd,
        )


def _run_fake_proof(module, tmp_path, monkeypatch, **runner_options):
    metal_root, test_root, output_dir = _prepare_checkouts(
        module, tmp_path, monkeypatch
    )
    ci_environment = runner_options.pop("ci_environment", False)
    runner = FakeRunner(module, tmp_path, **runner_options)
    payload = module.run_proof(
        metal_root,
        test_root,
        output_dir,
        python_executable="/opt/python/3.13/bin/python",
        runner=runner,
        jobs=1,
        ci_environment=ci_environment,
    )
    return payload, runner, metal_root, test_root, output_dir


def test_fake_native_baseline_emits_complete_deterministic_evidence(
    tmp_path, monkeypatch
):
    module = _load_helper()
    payload, runner, metal_root, test_root, output_dir = _run_fake_proof(
        module, tmp_path, monkeypatch
    )

    assert payload["status"] == "passed"
    assert payload["mlx"]["sourceCheckout"]["revision"] == module.MLX_COMMIT
    assert payload["mlx"]["testCheckout"]["revision"] == module.MLX_COMMIT
    source_proof = payload["sourceCompileLink"]
    assert payload["toolchain"]["architecture"] == "arm64"
    manifest = source_proof["sourceManifest"]
    assert manifest["sourceCount"] == 40
    assert manifest["expectedSha256"] == manifest["actualSha256"]
    assert manifest["profileCounts"] == {"metal3.2": 33, "metal4.0": 7}
    compile_evidence = source_proof["compile"]
    assert compile_evidence["successfulUnitCount"] == 40
    assert len(compile_evidence["units"]) == 40
    assert source_proof["link"]["airInputCount"] == 40
    assert source_proof["link"]["metallib"]["sizeBytes"] > 0
    assert source_proof["link"]["inspection"]["definedSymbolCount"] == 2

    commands = {name: command for name, command, _ in runner.calls}
    for unit in compile_evidence["units"]:
        command = commands[f"compile-{unit['index']:02d}"]
        assert f"-std={unit['profile']}" in command
        assert "-mmacosx-version-min=26.2" in command
        assert "-Wall" in command
        assert "-Wextra" in command
        assert "-fno-fast-math" in command
        assert command[command.index("-I") + 1] == str(metal_root)
    assert commands["python-package-build"][1:] == (
        "-m",
        "pip",
        "install",
        "-e",
        ".",
        "--no-build-isolation",
    )
    assert commands["python-unittest"][1:] == (
        "-m",
        "unittest",
        "discover",
        "python/tests",
    )
    assert commands["cpp-aggregate-tests"] == (
        str(output_dir / "cpp-build" / "tests" / "tests"),
    )

    python_evidence = payload["upstreamPythonTests"]
    assert python_evidence["python"]["minor"] == 13
    assert python_evidence["mlxVersion"] == "0.32.1.dev0+4367c73"
    assert python_evidence["defaultDeviceType"] == "gpu"
    assert python_evidence["unitTests"]["environment"] == "local"
    assert python_evidence["unitTests"]["expectedSkipCount"] == 44
    assert python_evidence["unitTests"]["counts"] == {
        "total": 776,
        "passed": 732,
        "skipped": 44,
        "failures": 0,
        "errors": 0,
    }
    cpp_evidence = payload["upstreamCppTests"]
    assert cpp_evidence["configuration"]["generator"] == "Ninja"
    assert set(cpp_evidence["configuration"]["definitions"]) >= {
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2",
        "-DMLX_BUILD_TESTS=ON",
        "-DMLX_BUILD_PYTHON_BINDINGS=OFF",
        "-DMLX_BUILD_EXAMPLES=OFF",
        "-DMLX_BUILD_BENCHMARKS=OFF",
    }
    assert cpp_evidence["aggregateTests"]["processCount"] == 1
    assert cpp_evidence["aggregateTests"]["counts"]["testCases"] == {
        "total": 260,
        "passed": 260,
        "failed": 0,
        "skipped": 0,
    }
    assert cpp_evidence["aggregateTests"]["counts"]["assertions"] == {
        "total": 3490,
        "passed": 3490,
        "failed": 0,
    }
    assert payload["claims"] == {
        "nativeUpstreamSourceCompilation": True,
        "nativeUpstreamPythonTests": True,
        "nativeUpstreamCppTests": True,
        "translatedTargetCorrectness": False,
        "translatedTargetRuntimeExecution": False,
        "runtimeParity": False,
        "numericalParity": False,
    }
    assert json.loads((output_dir / module.EVIDENCE_FILENAME).read_text()) == payload
    assert (
        python_evidence["packageBuild"]["invocation"]["workingDirectory"]
        == "$MLX_PYTHON_ROOT"
    )
    assert test_root.is_dir()


def test_ci_python_skip_accounting_is_explicit(tmp_path, monkeypatch):
    module = _load_helper()
    payload, _runner, _metal_root, _test_root, _output_dir = _run_fake_proof(
        module,
        tmp_path,
        monkeypatch,
        python_skipped=module.EXPECTED_UPSTREAM_CI_SKIP_COUNT,
        ci_environment=True,
    )

    unit_tests = payload["upstreamPythonTests"]["unitTests"]
    assert unit_tests["environment"] == "ci"
    assert unit_tests["expectedSkipCount"] == 46
    assert unit_tests["counts"] == {
        "total": 776,
        "passed": 730,
        "skipped": 46,
        "failures": 0,
        "errors": 0,
    }


def test_checkout_revision_mismatch_stops_before_compilation(tmp_path, monkeypatch):
    module = _load_helper()
    metal_root, test_root, output_dir = _prepare_checkouts(
        module, tmp_path, monkeypatch
    )
    runner = FakeRunner(module, tmp_path, revision="0" * 40)

    with pytest.raises(module.MlxNativeMetalProofError, match="must be pinned"):
        module.run_proof(
            metal_root,
            test_root,
            output_dir,
            python_executable="python3.13",
            runner=runner,
            jobs=1,
        )

    assert not any(name.startswith("compile-") for name, _, _ in runner.calls)


def test_source_manifest_drift_stops_before_toolchain_probe(tmp_path, monkeypatch):
    module = _load_helper()
    metal_root, test_root, output_dir = _prepare_checkouts(
        module, tmp_path, monkeypatch
    )
    extra = metal_root / module.MLX_METAL_KERNEL_ROOT / "unexpected.metal"
    extra.write_text("kernel void unexpected() {}\n", encoding="utf-8")
    runner = FakeRunner(module, tmp_path)

    with pytest.raises(module.MlxNativeMetalProofError, match="manifest changed"):
        module.run_proof(
            metal_root,
            test_root,
            output_dir,
            python_executable="python3.13",
            runner=runner,
            jobs=1,
        )

    assert not any(name.startswith("toolchain-") for name, _, _ in runner.calls)


def test_profile_accounting_change_fails_closed(tmp_path, monkeypatch):
    module = _load_helper()
    metal_root, test_root, output_dir = _prepare_checkouts(
        module, tmp_path, monkeypatch
    )
    monkeypatch.setattr(module, "EXPECTED_NAX_SOURCE_COUNT", 6)
    runner = FakeRunner(module, tmp_path)

    with pytest.raises(module.MlxNativeMetalProofError, match="profile accounting"):
        module.run_proof(
            metal_root,
            test_root,
            output_dir,
            python_executable="python3.13",
            runner=runner,
            jobs=1,
        )


@pytest.mark.parametrize("behavior", ["nonzero", "omit-air"])
def test_compile_failure_or_missing_air_fails_closed(tmp_path, monkeypatch, behavior):
    module = _load_helper()
    metal_root, test_root, output_dir = _prepare_checkouts(
        module, tmp_path, monkeypatch
    )
    runner = FakeRunner(
        module,
        tmp_path,
        behavior={"compile-00": behavior},
    )

    with pytest.raises(module.MlxNativeMetalProofError, match="compiles failed"):
        module.run_proof(
            metal_root,
            test_root,
            output_dir,
            python_executable="python3.13",
            runner=runner,
            jobs=1,
        )

    assert "link-metallib" not in {name for name, _, _ in runner.calls}


def test_empty_metallib_fails_closed(tmp_path, monkeypatch):
    module = _load_helper()
    metal_root, test_root, output_dir = _prepare_checkouts(
        module, tmp_path, monkeypatch
    )
    runner = FakeRunner(module, tmp_path, behavior={"link-metallib": "empty"})

    with pytest.raises(module.MlxNativeMetalProofError, match="output is empty"):
        module.run_proof(
            metal_root,
            test_root,
            output_dir,
            python_executable="python3.13",
            runner=runner,
            jobs=1,
        )


def test_empty_metal_nm_inspection_fails_closed(tmp_path, monkeypatch):
    module = _load_helper()
    metal_root, test_root, output_dir = _prepare_checkouts(
        module, tmp_path, monkeypatch
    )
    runner = FakeRunner(module, tmp_path, behavior={"inspect-metallib": "empty"})

    with pytest.raises(module.MlxNativeMetalProofError, match="no defined symbols"):
        module.run_proof(
            metal_root,
            test_root,
            output_dir,
            python_executable="python3.13",
            runner=runner,
            jobs=1,
        )


def test_cpu_default_device_fails_native_gpu_baseline(tmp_path, monkeypatch):
    module = _load_helper()
    metal_root, test_root, output_dir = _prepare_checkouts(
        module, tmp_path, monkeypatch
    )
    runner = FakeRunner(module, tmp_path, device_type="cpu")

    with pytest.raises(module.MlxNativeMetalProofError, match="requires a GPU"):
        module.run_proof(
            metal_root,
            test_root,
            output_dir,
            python_executable="python3.13",
            runner=runner,
            jobs=1,
        )

    names = {name for name, _, _ in runner.calls}
    assert "python-unittest" not in names
    assert "cpp-configure" not in names


def test_non_arm64_host_fails_before_metal_compilation(tmp_path, monkeypatch):
    module = _load_helper()
    metal_root, test_root, output_dir = _prepare_checkouts(
        module, tmp_path, monkeypatch
    )
    runner = FakeRunner(module, tmp_path, architecture="x86_64")

    with pytest.raises(module.MlxNativeMetalProofError, match="requires arm64"):
        module.run_proof(
            metal_root,
            test_root,
            output_dir,
            python_executable="python3.13",
            runner=runner,
            jobs=1,
        )

    assert not any(name.startswith("compile-") for name, _, _ in runner.calls)


def test_python_test_count_drift_fails_closed(tmp_path, monkeypatch):
    module = _load_helper()
    metal_root, test_root, output_dir = _prepare_checkouts(
        module, tmp_path, monkeypatch
    )
    runner = FakeRunner(module, tmp_path, python_total=775)

    with pytest.raises(module.MlxNativeMetalProofError, match="accounting changed"):
        module.run_proof(
            metal_root,
            test_root,
            output_dir,
            python_executable="python3.13",
            runner=runner,
            jobs=1,
        )


def test_ci_python_skip_count_drift_fails_closed(tmp_path, monkeypatch):
    module = _load_helper()
    metal_root, test_root, output_dir = _prepare_checkouts(
        module, tmp_path, monkeypatch
    )
    runner = FakeRunner(module, tmp_path, python_skipped=45)

    with pytest.raises(module.MlxNativeMetalProofError, match="accounting changed"):
        module.run_proof(
            metal_root,
            test_root,
            output_dir,
            python_executable="python3.13",
            runner=runner,
            jobs=1,
            ci_environment=True,
        )


def test_cpp_test_count_drift_fails_closed(tmp_path, monkeypatch):
    module = _load_helper()
    metal_root, test_root, output_dir = _prepare_checkouts(
        module, tmp_path, monkeypatch
    )
    runner = FakeRunner(module, tmp_path, cpp_assertions=3489)

    with pytest.raises(module.MlxNativeMetalProofError, match="assertion accounting"):
        module.run_proof(
            metal_root,
            test_root,
            output_dir,
            python_executable="python3.13",
            runner=runner,
            jobs=1,
        )


def test_workflow_uses_macos_26_arm64_and_uploads_only_report_and_logs():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "runs-on: macos-26" in text
    assert "runs-on: macos-26-intel" not in text
    assert "macos-latest" not in text.lower()
    assert 'python-version: "3.13"' in text
    assert "sparse-checkout set /mlx/backend/metal/kernels/" in text
    assert "--mlx-python-root mlx-upstream" in text
    assert "torch" not in text.lower()
    assert "4367c73b60541ddd5a266ce4644fd93d20223b6e" in text
    upload_block = text.split("Upload native Metal evidence and logs", 1)[1]
    assert ".crosstl-mlx-native-metal/evidence.json" in upload_block
    assert ".crosstl-mlx-native-metal/logs" in upload_block
    assert ".metallib" not in upload_block


def test_native_baseline_documentation_keeps_claims_and_results_separate():
    text = DOCUMENTATION_PATH.read_text(encoding="utf-8")
    normalized_text = " ".join(text.split())

    assert "33 standard units" in text
    assert "seven" in text and "_nax.metal" in text
    assert "exactly 776 tests" in normalized_text
    assert "44 baseline skips" in normalized_text
    assert "required count is 46" in normalized_text
    assert "Any other skip count fails closed" in normalized_text
    assert "260 of 260 cases" in text
    assert "3,490 of 3,490 assertions" in text
    assert "aggregate `tests/tests` doctest" in text
    assert "executable once" in text
    assert "not a skipped test or coverage" in text
    assert "does not establish translated-target correctness" in text
    assert "Torch is intentionally not installed" in text


def test_main_readme_discovers_native_baseline_without_overclaiming():
    text = README_PATH.read_text(encoding="utf-8")
    link = "[pinned native MLX Metal reference baseline](NATIVE_METAL.md)"

    assert link in text
    baseline_bullet = text[text.index(link) :].split("\n-", 1)[0]
    baseline_bullet = " ".join(baseline_bullet.split())
    assert (
        "does not establish translated-target correctness, runtime parity, "
        "or numerical parity" in baseline_bullet
    )

"""Complete pinned scan entries, including exact original-Metal readbacks."""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    translate_project,
)
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from crosstl.translator.dispatch_regions import plan_dispatch_regions
from demos.integrations.mlx.portable_host import bfloat_storage
from demos.integrations.mlx.tests.kernels.test_current_complex_power import MLX_COMMIT
from tests import workflow_text
from tests.runtime_helpers import compile_metal, run
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile

ROOT = Path(__file__).resolve().parents[5]
SOURCE = "mlx/backend/metal/kernels/scan.metal"
SOURCE_SHA256 = "49faeb1a36f346b2aa0f68a5f5fe2544af2109fec5357425550f97a79926b6f3"
REFERENCE = Path(__file__).resolve().parents[1] / "fixtures/scan_reference.json"
REFERENCE_SHA256 = "8302e4268f6ed3bcbed4ffed60ba9749aaec1279085a96c4e122f80e23e800b9"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_SCAN"
AXES = (1, 3, 31, 32, 33, 127, 128, 129, 257, 513)
DTYPES = ("float32", "bfloat16")
SCAN_LAUNCHES = tuple(
    pytest.param(width, marks=pytest.mark.xdist_group(f"scan-width-{width}"))
    for width in (32, 128, 256)
)
WIDE_AXES = (1, 31, 33, 129, 511, 512, 513, 1023, 1024, 1025)
WIDE_REFERENCE = REFERENCE.with_name("scan_subgroup_reference.json")
WIDE_REFERENCE_SHA256 = (
    "8e8162d7423789ef90411d3df2f1efebc6eb98695d112de363262083152e1c12"
)


def _entry(dtype, operation="sum"):
    return f"contig_scan_inclusive_{operation}_{dtype}_{dtype}"


def _guard(dtype):
    return 0x3EAB if dtype == "bfloat16" else 0xC7F12000


def _guard_words(dtype):
    return [_guard(dtype)] * (64 if dtype == "complex64" else 32)


def _input_words(dtype, axis, operation="sum"):
    if dtype == "complex64":
        assert operation == "sum"
        patterns = (
            [
                ((index % 19 - 9) * 0.125, (index % 7 - 3) * -0.25)
                for index in range(19)
            ],
            [(2.0**24, 1.0), (1.0, -(2.0**24)), (-(2.0**24), 1.0), (1.0, 2.0**24)],
            [(-0.0, 0.0), (0.0, -0.0), (0.125, -0.125), (-0.125, 0.125)],
        )
        return [
            struct.unpack("<I", struct.pack("<f", value))[0]
            for pattern in patterns
            for index in range(axis)
            for value in pattern[index % len(pattern)]
        ]
    if operation != "sum":
        assert operation in {"min", "max"} and dtype in {"float32", "int32"}
        patterns = (
            (
                [0x80000000, 0, 0x3F000000, 0xBF000000, 0x7F800000, 0xFF800000],
                [0x40000000, 0x3F800000, 0xBF800000, 0xC0000000],
                [0x3F800000, 0x7FA12345, 0xBF800000, 0x7FC01234],
            )
            if dtype == "float32"
            else (
                [0x80000000, 0x7FFFFFFF, 0, 0xFFFFFFFF],
                [7, 4, 1, 0xFFFFFFFE, 0xFFFFFFFB],
                [0, 1, 0xFFFFFFFF, 2, 0xFFFFFFFE],
            )
        )
        return [
            pattern[index % len(pattern)]
            for pattern in patterns
            for index in range(axis)
        ]
    patterns = (
        [((index * 7) % 19 - 9) * 0.125 for index in range(19)],
        [2.0**24, 1.0, -(2.0**24), 1.0],
        [-0.0, -0.0, 0.25, -0.25, 0.0],
    )
    words = [
        struct.unpack("<I", struct.pack("<f", pattern[index % len(pattern)]))[0]
        for pattern in patterns
        for index in range(axis)
    ]
    if dtype == "bfloat16":
        assert all(word & 0xFFFF == 0 for word in words)
        words = [word >> 16 for word in words]
    return words


def _references():
    data = REFERENCE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == REFERENCE_SHA256
    reference = json.loads(data)
    assert reference["upstreamCommit"] == MLX_COMMIT
    assert reference["source"] == SOURCE and reference["sourceSha256"] == SOURCE_SHA256
    assert reference["reference"] == "original-metal"
    assert reference["workgroupSize"] == [32, 1, 1]
    assert reference["workgroupCount"] == [1, 3, 1]
    assert reference["guardElements"] == 32
    cases = {(case["dtype"], case["axisSize"]): case for case in reference["cases"]}
    assert len(reference["cases"]) == len(cases) == 20
    assert set(cases) == {(dtype, axis) for dtype in DTYPES for axis in AXES}
    for (dtype, axis), case in cases.items():
        words = case["outputWords"]
        assert len(words) == 3 * axis + 32
        assert words[-32:] == [_guard(dtype)] * 32
        assert all(
            type(word) is int and 0 <= word < 2 ** (32 if dtype == "float32" else 16)
            for word in words
        )
    return cases


def _packet(dtype, target, words):
    if dtype == "int32":
        return {
            "dtype": dtype,
            "shape": [len(words)],
            "values": [word if word < 1 << 31 else word - (1 << 32) for word in words],
        }
    if dtype in {"float32", "complex64"}:
        assert dtype != "complex64" or len(words) % 2 == 0
        return {
            "dtype": "float32",
            "encoding": FLOAT32_BITS,
            "shape": [len(words)],
            "values": list(words),
        }
    encoding = bfloat_storage.encoding(target)
    return {
        "dtype": {"metal": "bfloat16", "opengl": "float32", "directx": "uint16"}[
            target
        ],
        "shape": [len(words)],
        "values": bfloat_storage.pack(words, target),
        **({"encoding": encoding} if encoding else {}),
    }


def _wide_references():
    data = WIDE_REFERENCE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == WIDE_REFERENCE_SHA256
    reference = json.loads(data)
    assert reference["upstreamCommit"] == MLX_COMMIT
    assert reference["source"] == SOURCE and reference["sourceSha256"] == SOURCE_SHA256
    assert reference["reference"] == "original-metal"
    assert reference["compileFlags"] == ["-std=metal3.2", "-fno-fast-math"]
    assert reference["workgroupCount"] == [1, 3, 1] and reference["guardElements"] == 32
    cases = {
        (case["workgroupWidth"], case["dtype"], case["axisSize"]): case
        for case in reference["cases"]
    }
    assert len(cases) == len(reference["cases"]) == 40
    assert set(cases) == {
        (width, dtype, axis)
        for width in (128, 256)
        for dtype in DTYPES
        for axis in WIDE_AXES
    }
    for (_, dtype, axis), case in cases.items():
        words = case["outputWords"]
        assert len(words) == 3 * axis + 32
        assert words[-32:] == [_guard(dtype)] * 32
        assert all(
            type(word) is int and 0 <= word < 2 ** (32 if dtype == "float32" else 16)
            for word in words
        )
    return cases


def test_multisubgroup_scan_reference_preserves_native_launch_boundaries():
    cases = _wide_references()
    assert {key[0] for key in cases} == {128, 256}
    for width in (128, 256):
        assert {width * 4 - 1, width * 4, width * 4 + 1} <= set(WIDE_AXES)


def test_scan_references_keep_exact_bytes_with_windows_checkout_settings(tmp_path):
    run(["git", "init", str(tmp_path)], tmp_path, "init")
    git = ["git", "-C", str(tmp_path)]
    run([*git, "config", "core.autocrlf", "true"], tmp_path, "line-endings")
    (tmp_path / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
    for source in (REFERENCE, WIDE_REFERENCE):
        destination = tmp_path / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    run([*git, "add", ".gitattributes", "demos"], tmp_path, "index")
    for source in (REFERENCE, WIDE_REFERENCE):
        (tmp_path / source.relative_to(ROOT)).unlink()
    run([*git, "checkout-index", "--all"], tmp_path, "checkout")
    for source, digest in (
        (REFERENCE, REFERENCE_SHA256),
        (WIDE_REFERENCE, WIDE_REFERENCE_SHA256),
    ):
        actual = (tmp_path / source.relative_to(ROOT)).read_bytes()
        assert b"\r" not in actual
        assert hashlib.sha256(actual).hexdigest() == digest


def _check_words(actual, expected):
    assert len(actual) == len(expected)
    assert all(type(word) is int for word in actual)
    differences = [
        (i, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b
    ]
    assert not differences, differences[:20]


def test_scan_reference_retains_boundaries_cancellation_and_signed_zero():
    cases = _references()
    assert sum(3 * axis for _, axis in cases) == 7524
    assert (
        sum(len(case["outputWords"]) - 3 * axis for (_, axis), case in cases.items())
        == 640
    )
    for dtype in DTYPES:
        words = _input_words(dtype, 3)
        assert len(words) == 9
        assert words[6:8] == [0x80000000 if dtype == "float32" else 0x8000] * 2
        # The rounded cancellation result is not the real-number prefix sum.
        assert cases[dtype, 3]["outputWords"][5] == 0
        assert cases[dtype, 3]["outputWords"][6:8] == [0, 0]


@pytest.mark.parametrize("damage", ("value", "guard", "zero-sign", "missing", "float"))
def test_scan_word_comparison_rejects_corruption(damage):
    expected = _references()["float32", 3]["outputWords"]
    actual = expected.copy()
    if damage == "missing":
        actual.pop()
    elif damage == "float":
        actual[5] = 0.0
    else:
        actual[{"value": 0, "guard": -1, "zero-sign": 5}[damage]] ^= (
            1 if damage != "zero-sign" else 0x80000000
        )
    with pytest.raises(AssertionError):
        _check_words(actual, expected)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("dtype", DTYPES)
def test_scan_transport_preserves_word_identity(dtype, target):
    words = _input_words(dtype, 33) + [_guard(dtype)] * 32
    packet = _packet(dtype, target, words)
    assert packet["shape"] == [131]
    actual = packet["values"]
    if dtype == "bfloat16":
        actual = bfloat_storage.unpack(actual, target)
        assert (
            packet["dtype"]
            == {"metal": "bfloat16", "directx": "uint16", "opengl": "float32"}[target]
        )
    _check_words(actual, words)


@pytest.fixture(scope="module")
def scan_package(tmp_path_factory, request):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned complete scan execution")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    target = os.environ["CROSTL_MLX_CURRENT_TARGET"]
    width = getattr(request, "param", 32)
    entries = tuple(_entry(dtype) for dtype in DTYPES)
    if isinstance(width, tuple):
        width, entries = width
    assert width in {32, 64, 128, 256}
    assert target in {"metal", "opengl", "directx"}
    assert (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
        ).strip()
        == MLX_COMMIT
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert hashlib.sha256((root / SOURCE).read_bytes()).hexdigest() == SOURCE_SHA256
    work = tmp_path_factory.mktemp("complete-scan-package")
    with tempfile.TemporaryDirectory(prefix=".crosstl-scan-", dir=root) as directory:
        region = plan_dispatch_regions((width, 3, 1), (width, 1, 1))[0].to_json()
        config = ProjectConfig(
            root=root,
            source_roots=("mlx/backend/metal/kernels",),
            include_patterns=(SOURCE,),
            include_dirs=(".",),
            targets=(target,),
            output_dir=Path(directory).name + "/out",
            entry_points={SOURCE: entries},
            workgroup_size_rules={SOURCE: (width, 1, 1)},
            metadata_preconditions=[
                dict(
                    source=SOURCE,
                    entryPoint=entry,
                    parameter="axis_size",
                    minimum=1,
                    maximum=max(AXES + WIDE_AXES),
                )
                for entry in entries
            ],
            source_options={
                "metal": {
                    "preserve_resource_origins": True,
                    "max_template_specializations": 256,
                    "max_template_materialization_work": 8192,
                    "target_options": {
                        "directx": {
                            "software_subgroup_width": 32,
                            "relative_wave_shuffle_out_of_range": "self",
                            "binary32_additive_profile": "native-precise",
                            "dispatch_region": region,
                        },
                        "opengl": {
                            "software_subgroup_width": 32,
                            "dispatch_region": region,
                        },
                    },
                }
            },
        )
        report = translate_project(config, format_output=False)
        report.write_json(work / "report.json")
        assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()[
            "diagnostics"
        ]
        assert all(
            artifact["provenance"].get("binary32AdditiveProfile")
            == ("native-precise" if target == "directx" else None)
            for artifact in report.to_json()["artifacts"]
        )
        manifest = build_runtime_artifact_manifest(work / "report.json")
        assert manifest["success"], manifest
        (work / "artifacts.json").write_text(json.dumps(manifest, indent=2))
        package = work / "package"
        assert build_runtime_package(work / "artifacts.json", package)["success"]
        loader = build_runtime_loader_manifest(package / "runtime-package.json")
        assert loader["success"] and len(loader["loadUnits"]) == len(entries), loader
        descriptors, modules = {}, {}
        for unit in loader["loadUnits"]:
            entry = unit["entryPoint"]["source"]
            descriptor = build_native_loader_abi_descriptor(
                loader, load_unit_id=unit["id"]
            )
            assert entry in entries
            descriptors[entry] = descriptor
            (work / f"{entry}.json").write_text(json.dumps(descriptor, indent=2))
            compilation = work / entry
            compilation.mkdir()
            artifact = package / descriptor["artifact"]["packagePath"]
            modules[entry] = _compile(
                artifact.read_text(),
                target,
                compilation,
                metal_compile_flags=("-std=metal3.2", "-fno-fast-math"),
            )[1]
        original = None
        if target == "metal":
            compiler = work / "original"
            compiler.mkdir()
            original_source = compiler / "scan.metal"
            original_source.write_bytes((root / SOURCE).read_bytes())
            library = compile_metal(
                original_source,
                root,
                compiler,
                flags=(
                    "-std=metal3.2",
                    "-fno-fast-math",
                    "-Wno-c++17-extensions",
                    "-Wno-c++20-extensions",
                ),
            )
            runner = compiler / "readback"
            run(
                [
                    "xcrun",
                    "swiftc",
                    ROOT
                    / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
                    "-o",
                    runner,
                ],
                compiler,
                "build-runner",
            )
            original = runner, library
        yield target, package, descriptors, modules, original, width


@pytest.mark.parametrize("scan_package", SCAN_LAUNCHES, indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
def test_complete_scan_package_retains_source_buffer_layout(scan_package, dtype):
    target, _, descriptors, _, _, workgroup_width = scan_package
    descriptor = descriptors[_entry(dtype)]
    assert descriptor["target"] == target
    bindings = {
        binding["provenance"]["sourceResource"]["parameter"]: binding
        for binding in descriptor["bindings"]
        if "sourceResource" in binding.get("provenance", {})
    }
    assert set(bindings) == {"in", "out", "axis_size"}
    # The checked dispatch region embeds launch dimensions without an extra buffer.
    assert len(descriptor["bindings"]) == 3
    assert bindings["axis_size"]["provenance"]["metadataPrecondition"] == {
        "dtype": "uint64",
        "minimum": 1,
        "maximum": max(AXES + WIDE_AXES),
        "sourceResource": bindings["axis_size"]["provenance"]["sourceResource"],
    }
    assert descriptor["provenance"].get("dispatchRegion") == (
        None
        if target == "metal"
        else plan_dispatch_regions((workgroup_width, 3, 1), (workgroup_width, 1, 1))[
            0
        ].to_json()
    )
    for parameter, slot in (("in", 0), ("out", 1), ("axis_size", 2)):
        binding = bindings[parameter]
        origin = binding["provenance"]["sourceResource"]
        assert origin["backend"] == "metal" and origin["entryPoint"] == _entry(dtype)
        assert origin["parameterIndex"] == slot
        assert binding["coordinates"]["binding"] == slot
        layout = binding["scalarLayout"]
        width = (
            8
            if parameter == "axis_size"
            else 2 if dtype == "bfloat16" and target != "opengl" else 4
        )
        assert layout["elementStrideBytes"] == layout["elementSizeBytes"] == width
        assert binding["access"] == ("read_write" if parameter == "out" else "read")
        assert binding["kind"] == (
            "constant-buffer" if parameter == "axis_size" else "buffer"
        )


def _original_readback(
    work, original, dtype, axis, expected, width=32, operation="sum"
):
    runner, library = original
    code = "H" if dtype == "bfloat16" else "I"
    values = (
        _input_words(dtype, axis, operation) + _guard_words(dtype),
        [_guard(dtype)] * len(expected),
        [axis],
    )
    payloads, paths = [], []
    for index, words in enumerate(values):
        payload = struct.pack("<" + ("Q" if index == 2 else code) * len(words), *words)
        path = work / f"original-input-{index}.bin"
        path.write_bytes(payload)
        paths.append(str(path))
        payloads.append(payload)
    request = work / "original-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": paths,
                "workgroupCount": [1, 3, 1],
                "workgroupSize": [width, 1, 1],
                "simdWidth": 32,
            },
            indent=2,
        )
    )
    output = work / "original-readback"
    run(
        [runner, library, _entry(dtype, operation), request, output],
        work,
        "original-execute",
    )
    readback = (output / "buffer-1.bin").read_bytes()
    assert len(readback) == len(payloads[1])
    _check_words(list(struct.unpack("<" + code * len(expected), readback)), expected)
    for index in (0, 2):
        assert (output / f"buffer-{index}.bin").read_bytes() == payloads[index]


@pytest.mark.parametrize("scan_package", SCAN_LAUNCHES[:1], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis", AXES)
def test_complete_scan_executes_natively(scan_package, tmp_path, dtype, axis):
    expected = _references()[dtype, axis]["outputWords"]
    _execute_scan_case(scan_package, tmp_path, dtype, axis, expected)


@pytest.mark.parametrize("scan_package", SCAN_LAUNCHES[1:], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("axis", WIDE_AXES)
def test_multisubgroup_scan_executes_natively(scan_package, tmp_path, dtype, axis):
    expected = _wide_references()[scan_package[-1], dtype, axis]["outputWords"]
    _execute_scan_case(scan_package, tmp_path, dtype, axis, expected)


def _execute_scan_case(scan_package, tmp_path, dtype, axis, expected, operation="sum"):
    target, package, descriptors, modules, original, width = scan_package
    if original:
        _original_readback(tmp_path, original, dtype, axis, expected, width, operation)
    descriptor = descriptors[_entry(dtype, operation)]
    names = {
        binding["provenance"]["sourceResource"]["parameter"]: binding["name"]
        for binding in descriptor["bindings"]
        if "sourceResource" in binding.get("provenance", {})
    }
    assert set(names) == {"in", "out", "axis_size"}, names
    inputs = {
        names["in"]: _packet(
            dtype, target, _input_words(dtype, axis, operation) + _guard_words(dtype)
        ),
        names["out"]: _packet(dtype, target, [_guard(dtype)] * len(expected)),
        names["axis_size"]: {"dtype": "uint64", "shape": [1], "values": [axis]},
    }
    outputs = {names["out"]: _packet(dtype, target, expected)}
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupSize": [width, 1, 1], "workgroupCount": [1, 3, 1]},
        expected_target=target,
        observe_inputs=(names["in"], names["axis_size"]),
    )
    assert not request.execution_plan.diagnostics, request.execution_plan.diagnostics
    (tmp_path / "descriptor.json").write_text(json.dumps(descriptor, indent=2))
    (tmp_path / "inputs.json").write_text(
        json.dumps({"inputs": inputs, "expected": outputs}, indent=2)
    )

    def validate(artifact, directory, actual_target):
        assert actual_target == target
        assert (
            artifact.read_bytes()
            == (package / descriptor["artifact"]["packagePath"]).read_bytes()
        )
        module = modules[_entry(dtype, operation)]
        copied = directory / module.name
        shutil.copyfile(module, copied)
        return copied

    _execute(
        request,
        outputs,
        tmp_path,
        validate=validate,
        metal_language_version="3.2" if target == "metal" else None,
    )


def test_ci_requires_complete_scan_execution_on_each_native_platform():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    job = workflow_text.workflow_job_text(workflow, "mlx-metal-porting")
    step = workflow_text.workflow_step_section(job, "Validate pinned complete scans")
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "CROSTL_MLX_CURRENT_ROOT:" in step
    for platform, target in (
        ("Linux", "opengl"),
        ("Windows", "directx"),
        ("macOS", "metal"),
    ):
        assert f"{platform}) export CROSTL_MLX_CURRENT_TARGET={target}" in step
    assert "test_current_scan.py" in step
    assert "--timeout-seconds 240" in step and "pytest -q -n auto" in step
    assert "--dist=loadgroup" in step
    for parameter in SCAN_LAUNCHES:
        (width,) = parameter.values
        assert [(mark.name, mark.args) for mark in parameter.marks] == [
            ("xdist_group", (f"scan-width-{width}",))
        ]
    assert "--basetemp=" in step and "--junitxml=" in step
    upload = workflow_text.workflow_step_section(job, "Upload complete scan evidence")
    assert "if: always()" in upload and "if-no-files-found: error" in upload
    assert "path: mlx-scan-results" in upload
    assert "include-hidden-files: true" in upload

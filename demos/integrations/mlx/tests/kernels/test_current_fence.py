"""Prove fence.metal's three entries at the current pin.

``fence.cpp``'s fast fences synchronize command buffers through a shared
``uint`` in ``coherent(system)`` memory: ``fence_update`` stores the fence's
count and ``fence_wait`` spins until the stored value reaches it, one thread
each, and ``input_coherent`` rewrites every word of an array in place in
``(1024, 1, 1)`` threadgroups before a cross-device update. Each kernel ends
or loops on a sequentially consistent device-memory fence at
``thread_scope_system``.

Metal translates all three (Metal to CrossGL to Metal) with the volatile
``coherent(system)`` storage and the system-scope fences intact; each is
pinned, compiled with ``-Werror`` and run natively, and must leave its
buffers exactly as the unchanged upstream kernel does, word for word.
``fence_wait`` runs only with the stored value already at or past the
awaited one, the cases in which the upstream loop ends.

Direct3D and OpenGL cannot express a fence at system scope. HLSL's
barriers (``DeviceMemoryBarrier``, ``AllMemoryBarrier`` and Shader Model
6.8's ``Barrier``, whose widest scope is ``DEVICE_SCOPE``) and GLSL's
``memoryBarrier`` functions order memory for the device only; host
visibility goes through API fences and maps. So each target refuses the
three entries with its typed ``atomic-fence-unsupported`` diagnostic, and
the gate pins that refusal.
"""

import hashlib
import json
import re
import shutil
import struct
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
    validate_project_report,
)
from demos.integrations.mlx.tests.kernels.current_gate import (
    assert_identical,
    compact_contract_text,
    parse_pin,
    require_pinned_checkout,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import _run
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_FENCE"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/fence.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
SOURCE = "mlx/backend/metal/kernels/fence.metal"
SOURCE_SHA256 = "6d3f3c27dd038a0f731054f7466f951b02134ce4124b1c1bbbdabfd9498d1913"
HOST_SOURCE = "mlx/backend/metal/fence.cpp"
HOST_SHA256 = "8eaaaf3cdaf3bca2b51b5098a5cc903898e8fcc83f17c8e38a2d663c6adcf8c4"
# fence.cpp's threadgroups and each kernel's [[buffer(i)]] parameters.
ENTRIES = {
    "input_coherent": {"workgroup": (1024, 1, 1), "parameters": ("input", "size")},
    "fence_update": {"workgroup": (1, 1, 1), "parameters": ("timestamp", "value")},
    "fence_wait": {"workgroup": (1, 1, 1), "parameters": ("timestamp", "value")},
}
OUTPUTS = {
    "input_coherent": "input",
    "fence_update": "timestamp",
    "fence_wait": "timestamp",
}
GUARD = 0x6A15BEEF
GUARD_COUNT = 32
REFUSALS = {
    "directx": "project.translate.directx-atomic-fence-unsupported",
    "opengl": "project.translate.opengl-atomic-fence-unsupported",
}
REFUSAL_DETAIL = "unsupported system thread scope"
FENCE = ("mem_device", "memory_order_seq_cst", "thread_scope_system")
# input_coherent's word counts: one partial threadgroup, one full, and a
# partial second threadgroup.
COHERENT_SIZES = (1, 1024, 1027)
# (stored, awaited or written) values; fence_wait's stored value is never
# below the awaited one.
UPDATE_CASES = ((0, 7), (7, 7), (0xFFFFFFFE, 0xFFFFFFFF))
WAIT_CASES = ((5, 5), (9, 5), (0xFFFFFFFF, 1))
# coherent(system) storage and system-scope fences need Metal 3.2, the
# language MLX builds its library with.
METAL_LANGUAGE = "3.2"


def _metal_library(source, output, root, *, upstream=False):
    """Compile and link a Metal library at METAL_LANGUAGE, warnings as
    errors for translations."""
    air = output.with_suffix(".air")
    _run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            *([] if upstream else ["-Werror"]),
            f"-std=metal{METAL_LANGUAGE}",
            "-fno-fast-math",
            "-I",
            root,
            "-c",
            source,
            "-o",
            air,
        ],
        output.parent,
        output.stem + "-compile",
    )
    _run(["xcrun", "metallib", air, "-o", output], output.parent, output.stem + "-link")
    assert output.stat().st_size > 0
    return output


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _refusal_pin(target):
    return f"refused:{REFUSALS[target]}"


def _words(count, salt):
    """Distinct words covering both halves of the 32-bit range."""
    return [(index * 0x9E3779B1 + salt) & 0xFFFFFFFF for index in range(count)]


def cases(entry):
    """Each case's storage words, scalar, grid and expected storage."""
    found = []
    if entry == "input_coherent":
        for size in COHERENT_SIZES:
            groups = (size + 1023) // 1024
            words = _words(size, size) + [GUARD] * GUARD_COUNT
            found.append(
                {
                    "label": f"size-{size}",
                    "storage": words,
                    "scalar": size,
                    "grid": [groups, 1, 1],
                    "expected": list(words),
                }
            )
        return found
    for stored, value in UPDATE_CASES if entry == "fence_update" else WAIT_CASES:
        words = [stored] + [GUARD] * GUARD_COUNT
        expected = [value if entry == "fence_update" else stored] + words[1:]
        found.append(
            {
                "label": f"{stored}-{value}",
                "storage": words,
                "scalar": value,
                "grid": [1, 1, 1],
                "expected": expected,
            }
        )
    return found


def _expected_validation_contract():
    return {
        "workgroups": {
            entry: list(info["workgroup"]) for entry, info in ENTRIES.items()
        },
        "fence": list(FENCE),
        "storage": "volatile coherent(system) device uint",
        "cases": {
            "input_coherent": list(COHERENT_SIZES),
            "fence_update": [list(item) for item in UPDATE_CASES],
            "fence_wait": [list(item) for item in WAIT_CASES],
        },
        "guards": f"{GUARD_COUNT} guards after every buffer stay untouched",
        "metalUpstreamComparison": (
            "the unchanged upstream kernel, word for word over every buffer"
        ),
        "refusals": {target: REFUSALS[target] for target in ("opengl", "directx")},
        "refusalReason": (
            "HLSL and GLSL memory barriers reach device scope at most; "
            "fence.metal fences at thread_scope_system"
        ),
    }


def expected_contract(artifacts=None):
    if artifacts is None:
        artifacts = _contract()["artifacts"]
    return {
        "schemaVersion": 1,
        "kind": "crosstl-mlx-current-tree-fence-contract",
        "provenance": {
            "repository": "https://github.com/ml-explore/mlx.git",
            "commit": MLX_COMMIT,
            "source": SOURCE,
            "sourceSha256": SOURCE_SHA256,
            "hostSource": {"path": HOST_SOURCE, "sha256": HOST_SHA256},
        },
        "corpusCensus": {
            "metalUnitCount": 49,
            "entryCount": 17832,
            "discoveryDiagnosticCount": 0,
        },
        "entries": list(ENTRIES),
        "artifacts": artifacts,
        "validationContract": _expected_validation_contract(),
        "scope": {
            "instantiatedEntryCount": len(ENTRIES),
            "coveredEntryCount": len(ENTRIES),
            "artifactCountByTarget": {"metal": 3, "opengl": 0, "directx": 0},
            "refusedCountByTarget": {"metal": 0, "opengl": 3, "directx": 3},
            "dispatchCountByTarget": {
                "metal": sum(len(cases(entry)) for entry in ENTRIES),
                "opengl": 0,
                "directx": 0,
            },
            "completeCorpusEntryCount": 17832,
            "upstreamMlxTestSuiteExecuted": False,
            "mlxHostRuntimeRedirectionImplemented": False,
        },
    }


def _config(work, root, target, entry):
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=str(Path(work).relative_to(root) / "out"),
        entry_points={SOURCE: (entry,)},
        workgroup_size=ENTRIES[entry]["workgroup"],
    )


def test_current_mlx_fence_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract()
    assert CONTRACT_PATH.read_text(encoding="utf-8") == compact_contract_text(contract)
    assert list(contract["artifacts"]) == list(ENTRIES)
    for pins in contract["artifacts"].values():
        assert list(pins) == list(TARGETS)
        record = parse_pin(pins["metal"])
        assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
        assert record["sizeBytes"] > 0
        for target in ("opengl", "directx"):
            assert pins[target] == _refusal_pin(target)
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "fence.current-tree.translation.json" in readme


def test_fence_cases_keep_guards_and_never_wait_forever():
    for entry in ENTRIES:
        for case in cases(entry):
            assert case["storage"][-GUARD_COUNT:] == [GUARD] * GUARD_COUNT
            assert case["expected"][-GUARD_COUNT:] == [GUARD] * GUARD_COUNT
            if entry == "fence_wait":
                assert case["storage"][0] >= case["scalar"]
            if entry == "input_coherent":
                assert case["grid"][0] * 1024 >= case["scalar"]
                assert case["expected"] == case["storage"]
    assert [case["expected"][0] for case in cases("fence_update")] == [
        value for _, value in UPDATE_CASES
    ]


@pytest.mark.parametrize("target", TARGETS)
def test_fence_configs_select_one_entry(tmp_path, target):
    for entry, info in ENTRIES.items():
        config = _config(tmp_path / "work", tmp_path, target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        assert tuple(config.workgroup_size) == info["workgroup"]


# ---------------------------------------------------------------------------
# Native parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(REQUIRE_ENV, "fence", TARGETS, MLX_COMMIT)
    source = (root / SOURCE).read_text(encoding="utf-8")
    assert hashlib.sha256(source.encode("utf-8")).hexdigest() == SOURCE_SHA256
    host = (root / HOST_SOURCE).read_bytes()
    assert hashlib.sha256(host).hexdigest() == HOST_SHA256
    for entry, info in ENTRIES.items():
        first, second = info["parameters"]
        assert re.search(
            rf"\[\[kernel\]\] void {entry}\(\s*volatile coherent\(system\) device uint\* "
            rf"{first} \[\[buffer\(0\)\]\],\s*(const )?constant uint& {second} "
            r"\[\[buffer\(1\)\]\]",
            source,
        ), entry
    assert source.count("metal::thread_scope_system);") == 3
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("fence-metal-reference")
    runner = work / "readback"
    _run(
        [
            "xcrun",
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            runner,
        ],
        work,
        "build-runner",
    )
    return runner, _metal_library(
        root / SOURCE, work / "upstream.metallib", root, upstream=True
    )


def _translate(root, work, entry, record):
    translation = translate_project(
        _config(work, root, "metal", entry),
        format_output=False,
        validate=True,
        run_toolchains=False,
    )
    report = translation.to_json()
    translation.write_json(work / "report.json")
    assert validate_project_report(work / "report.json")["success"] is True
    assert report["diagnostics"] == []
    (artifact,) = report["artifacts"]
    assert artifact["sourceHash"] == {"algorithm": "sha256", "value": SOURCE_SHA256}
    check_generated_artifact(
        artifact, sha256=record["sha256"], size_bytes=record["sizeBytes"]
    )
    generated = root / artifact["path"]
    text = generated.read_text(encoding="utf-8")
    # The coherence and the system-scope fence survive the round trip.
    assert "coherent(system)" in text and "volatile" in text
    assert all(part in text for part in FENCE)
    _metal_library(generated, work / "translated.metallib", root)
    return _prepare_native_package(translation, work)


def _refuse(root, work, target, entry):
    report = translate_project(
        _config(work, root, target, entry),
        format_output=False,
        validate=True,
        run_toolchains=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert report["summary"]["failedCount"] == 1
    first, follow_up = report["diagnostics"]
    assert first["code"] == REFUSALS[target], first
    assert first["severity"] == "error" and first["target"] == target, first
    assert REFUSAL_DETAIL in first["message"]
    assert all(part in first["message"] for part in FENCE), first["message"]
    assert follow_up["code"] == "project.validate.failed-artifact", follow_up
    assert not list(work.rglob("*.hlsl")) and not list(work.rglob("*.glsl"))


def _value(words):
    return {"dtype": "uint32", "shape": [len(words)], "values": list(words)}


def _run_case(work, entry, case, executor, translated, reference):
    descriptor, package = translated
    output_name = OUTPUTS[entry]
    storage = {output_name: case["storage"]}
    inputs, observed, output = {}, [], None
    for binding in descriptor["bindings"]:
        # Metal reflects each parameter under its own name.
        name = binding["name"]
        assert name in ENTRIES[entry]["parameters"], binding
        words = storage.get(name, [case["scalar"]])
        inputs[binding["name"]] = _value(words)
        if name == output_name:
            assert binding["access"] == "read_write", binding
            output = binding["name"]
        else:
            observed.append(binding["name"])
    assert output is not None and len(inputs) == 2
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output: _value(case["expected"])},
        {
            "workgroupCount": case["grid"],
            "workgroupSize": list(ENTRIES[entry]["workgroup"]),
        },
        expected_target="metal",
        observe_inputs=tuple(observed),
    )
    availability = executor.is_available(request)
    assert availability.available, availability.reason
    result = executor.run(request)
    assert result.status == "ok", result.details
    assert all(
        item["unchanged"] for item in result.details["inputObservations"].values()
    )
    words = result.outputs[output]["values"]
    assert_identical(case["label"], words, case["expected"])
    runner, library = reference
    files = []
    payloads = [struct.pack(f"<{len(case['storage'])}I", *case["storage"])]
    payloads.append(struct.pack("<I", case["scalar"]))
    for index, payload in enumerate(payloads):
        path = work / f"{case['label']}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    request_path = work / f"{case['label']}-upstream-request.json"
    request_path.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": case["grid"],
                "workgroupSize": list(ENTRIES[entry]["workgroup"]),
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    readback = work / f"{case['label']}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request_path, readback], work, case["label"])
    )
    raw = (readback / "buffer-0.bin").read_bytes()
    assert_identical(
        case["label"], words, list(struct.unpack(f"<{len(raw) // 4}I", raw))
    )
    assert (readback / "buffer-1.bin").read_bytes() == payloads[1]
    return {"case": case["label"], "device": device, "identicalToUpstream": True}


@pytest.mark.parametrize("entry", list(ENTRIES))
def test_current_mlx_fence_native_parity(current_mlx, metal_reference, tmp_path, entry):
    root, target = current_mlx
    pin = _contract()["artifacts"][entry][target]
    with tempfile.TemporaryDirectory(prefix=".current-fence-", dir=root) as path:
        work = Path(path)
        try:
            if target != "metal":
                assert pin == _refusal_pin(target)
                _refuse(root, work, target, entry)
                return
            translated = _translate(root, work, entry, parse_pin(pin))
            executor = _executor(target, metal_language_version=METAL_LANGUAGE)
            try:
                evidence = [
                    _run_case(work, entry, case, executor, translated, metal_reference)
                    for case in cases(entry)
                ]
            finally:
                close = getattr(executor.runtime_adapter.runtime, "close", None)
                if callable(close):
                    close()
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entry": entry,
                        "artifact": pin,
                        "cases": evidence,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.copytree(
                work,
                tmp_path / "evidence",
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*.air", "*.metallib", "out", "*.bin"),
            )


def test_ci_requires_native_fence_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_step(
        workflow, "Prove current MLX fence native parity", group="ordering"
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/fence",
        "--junitxml=mlx-current-results/fence.xml",
        "demos/integrations/mlx/tests/kernels/test_current_fence.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_fence.py"
    )

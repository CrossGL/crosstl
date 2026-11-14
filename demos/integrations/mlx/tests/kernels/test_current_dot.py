"""Translate and execute the pinned MLX dot-product kernels on all backends."""

import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    load_project_config,
    translate_project,
    validate_project_report,
)
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.kernels.test_current_cross_entropy import (
    _bits,
    _bits_float32,
    _element_layout,
    _float32_bits,
    _quantize,
    _runtime_value,
)
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/dot.metal"
SOURCE_SHA256 = "97bcad13d09c3d5fed87482a0bb9719d6eeff9b21d364967cd6aec5b695b3462"
HEADER = "mlx/backend/metal/kernels/dot.h"
HEADER_SHA256 = "24ca982ca5558e43100be7d6dc08722b512b06461c8460d14134eb78f52431ba"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_DOT"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/dot.current-tree.translation.json"
)
DTYPES = ("float32", "float16", "bfloat16")
ENTRIES = tuple(f"dot_product_{dtype}_it32_tg512_sg16" for dtype in DTYPES)
TARGETS = ("metal", "opengl", "directx")
# MLX's dot_product dispatch: 512-thread groups of 16 SIMD groups, each thread
# reducing 32 items, so one group produces the partial sum of 16384 elements.
WORKGROUP_SIZE = (512, 1, 1)
ITEMS_PER_THREAD = 32
SIMD_SIZE = 32
GROUP_ELEMENTS = WORKGROUP_SIZE[0] * ITEMS_PER_THREAD
# Apple GPUs reduce 32-lane SIMD groups. OpenGL and DirectX lower the
# reductions to software 32-lane groups, so neither Mesa's nor WARP's native
# subgroup width or wave packing matters. The second reduction runs under
# ``tid < SIMD_GROUPS``; both targets predicate it so every invocation reaches
# the barriers, with inactive invocations contributing the sum identity.
SOFTWARE_SUBGROUP_TARGETS = ("opengl", "directx")
SUBGROUPS = {
    "metal": "native-simdgroup-32",
    "opengl": "software-subgroup-32",
    "directx": "software-subgroup-32",
}
# name -> (element count, value kind)
CASES = {
    "one": (1, "exact"),
    "three": (3, "exact"),
    "vector": (8, "exact"),
    "vector-tail": (13, "exact"),
    "row-tail": (129, "exact"),
    "simd-block": (1024, "exact"),
    "simd-tail": (1027, "exact"),
    "group-minus-one": (GROUP_ELEMENTS - 1, "exact"),
    "group": (GROUP_ELEMENTS, "exact"),
    "groups": (GROUP_ELEMENTS + 1, "exact"),
    "many-groups": (40000, "exact"),
    "rounding": (20000, "rounding"),
    "infinite": (100, "infinite"),
}
INFINITE_INDEX = 37
# Exact cases use multiples of 1/8 in [-4, 4]: every value is exact in all
# three input types, every product is a multiple of 2^-6 below 16 in
# magnitude, and every partial sum of at most 16384 products is exact in
# binary32, so all summation orders and contractions agree bit for bit.
EXACT_NUMERATOR_LIMIT = 32
EXACT_DENOMINATOR = 8
# Any summation tree of depth d has error at most d * 2^-24 * sum(|products|)
# to first order. The kernel's deepest tree adds 8 thread steps, 3 lane folds,
# and two 31-step SIMD reductions, so 128 bounds rounded products as well.
ROUNDING_BOUND_UNITS = 128
INPUT_GUARD_COUNT = 16
OUTPUT_GUARD_COUNT = 16
OUTPUT_GUARD_BITS = 0xC0DEC0DE


def _dtype(entry):
    return entry.split("_")[2]


def _groups(count):
    threads = (count + ITEMS_PER_THREAD - 1) // ITEMS_PER_THREAD
    return (threads + WORKGROUP_SIZE[0] - 1) // WORKGROUP_SIZE[0]


def _values(name, dtype, salt):
    count, kind = CASES[name]
    state = (zlib.crc32(f"{name}:{salt}".encode()) << 1) | 1
    values = []
    for index in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        if kind == "rounding":
            value = ((state >> 11) / 2**53) * 4.0 - 2.0
        else:
            numerator = (state >> 33) % (2 * EXACT_NUMERATOR_LIMIT + 1)
            value = (numerator - EXACT_NUMERATOR_LIMIT) / EXACT_DENOMINATOR
        if kind == "infinite" and salt == "a" and index == INFINITE_INDEX:
            value = math.inf
        if kind == "infinite" and salt == "b" and index == INFINITE_INDEX:
            value = 0.5
        values.append(_quantize(dtype, value))
    return values


def _reference(name, dtype):
    """Return inputs, exact partial sums and per-partial error bounds."""
    count, kind = CASES[name]
    a, b = _values(name, dtype, "a"), _values(name, dtype, "b")
    partials, bounds = [], []
    for group in range(_groups(count)):
        lo, hi = group * GROUP_ELEMENTS, min(count, (group + 1) * GROUP_ELEMENTS)
        # Products of two binary32 values are exact in binary64.
        products = [a[index] * b[index] for index in range(lo, hi)]
        partials.append(math.fsum(products))
        bounds.append(
            ROUNDING_BOUND_UNITS * 2.0**-24 * math.fsum(abs(p) for p in products)
            if kind == "rounding"
            else 0.0
        )
    return a, b, partials, bounds


def _config(output, target, entry):
    options = ""
    if target in SOFTWARE_SUBGROUP_TARGETS:
        options += f"""
[project.source_options.metal.target_options.{target}]
software_subgroup_width = {SIMD_SIZE}
"""
    if target == "directx":
        options += 'relative_wave_shuffle_out_of_range = "self"\n'
    return f"""
[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
output_dir = "{output}"
[project.entry_points]
"{SOURCE}" = "{entry}"
[project.entry_workgroup_size_rules."{SOURCE}"]
"{entry}" = [{", ".join(str(item) for item in WORKGROUP_SIZE)}]
[project.source_options.metal]
max_template_specializations = 64
max_template_materialization_work = 4096
{options}"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _expected_validation_contract():
    return {
        "workgroupSize": list(WORKGROUP_SIZE),
        "groupElementCount": GROUP_ELEMENTS,
        "subgroups": dict(SUBGROUPS),
        "cases": {
            name: {"elementCount": count, "kind": kind, "groups": _groups(count)}
            for name, (count, kind) in CASES.items()
        },
        "exactValues": {
            "numeratorLimit": EXACT_NUMERATOR_LIMIT,
            "denominator": EXACT_DENOMINATOR,
        },
        "roundingBound": {"units": ROUNDING_BOUND_UNITS, "unitRoundoff": 2.0**-24},
        "inputGuardCount": INPUT_GUARD_COUNT,
        "outputGuardCount": OUTPUT_GUARD_COUNT,
        "observedInputCount": 3,
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def _expected_scope():
    return {
        "coveredEntryCount": len(ENTRIES),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "partialCountByEntry": {
            entry: sum(_groups(count) for count, _ in CASES.values())
            for entry in ENTRIES
        },
        "completeCorpusEntryCount": 17832,
        # MLX sums the partials with a separate all-reduce dispatch.
        "partialAllReduceCovered": False,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_dot_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-dot-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "header": HEADER,
        "headerSha256": HEADER_SHA256,
    }
    assert contract["corpusCensus"] == {
        "metalUnitCount": 49,
        "entryCount": 17832,
        "discoveryDiagnosticCount": 0,
    }
    assert tuple(contract["entries"]) == ENTRIES
    assert set(contract["artifacts"]) == set(TARGETS)
    for target in TARGETS:
        artifacts = contract["artifacts"][target]
        assert tuple(artifacts) == ENTRIES
        for record in artifacts.values():
            assert set(record) == {"sha256", "sizeBytes"}
            assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
    assert contract["validationContract"] == _expected_validation_contract()
    assert contract["scope"] == _expected_scope()


@pytest.mark.parametrize("dtype", DTYPES)
def test_dot_cases_cover_kernel_paths(dtype):
    for name, (count, kind) in CASES.items():
        a, b, partials, bounds = _reference(name, dtype)
        assert len(a) == len(b) == count
        assert len(partials) == len(bounds) == _groups(count)
        assert all(_quantize(dtype, value) == value for value in a + b)
        if kind == "exact":
            # Exactness argument: |products| < 16 in steps of 2^-6.
            assert all((value * 8).is_integer() and abs(value) <= 4 for value in a)
            assert all(bound == 0.0 for bound in bounds)
            assert all(
                struct.unpack("<f", struct.pack("<f", partial))[0] == partial
                for partial in partials
            )
        elif kind == "rounding":
            assert all(bound > 0.0 for bound in bounds)
        else:
            assert math.isinf(partials[0]) and partials[0] > 0
    vector = 16 // (4 if dtype == "float32" else 2)
    # Element counts below, at and past a vector, a SIMD block of 1024
    # elements and a 16384-element group all appear.
    counts = {count for count, _ in CASES.values()}
    assert any(count % vector for count in counts)
    assert {1024, 1027, GROUP_ELEMENTS - 1, GROUP_ELEMENTS} <= counts
    assert {_groups(count) for count in counts} == {1, 2, 3}
    # The rounding case does not happen to be exact.
    _, _, partials, _ = _reference("rounding", dtype)
    assert any(
        struct.unpack("<f", struct.pack("<f", partial))[0] != partial
        for partial in partials
    )


def test_dot_rounding_bound_rejects_a_lost_product():
    a, b, partials, bounds = _reference("rounding", "float32")
    missing = partials[0] - a[5] * b[5]
    assert abs(missing - partials[0]) > bounds[0]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_dot_configs_scope_subgroup_options(tmp_path, target, entry):
    path = tmp_path / "crosstl.toml"
    path.write_text(_config("out", target, entry), encoding="utf-8")
    config = load_project_config(tmp_path, path)
    assert list(config.targets) == [target]
    text = path.read_text(encoding="utf-8")
    software = target in SOFTWARE_SUBGROUP_TARGETS
    assert ("software_subgroup_width = 32" in text) is software
    assert ('relative_wave_shuffle_out_of_range = "self"' in text) is (
        target == "directx"
    )
    assert "[project.subgroup_width_rules]" not in text


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native dot-product gate")
    assert root_value and target in TARGETS
    for name in {
        "directx": ["dxc"],
        "opengl": ["glslangValidator", "spirv-val"],
        "metal": ["xcrun"],
    }[target]:
        assert shutil.which(name), f"Required tool is missing: {name}"
    root = Path(root_value).resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert revision == MLX_COMMIT
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
    assert hashlib.sha256((root / HEADER).read_bytes()).hexdigest() == HEADER_SHA256
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("dot-metal-reference")
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


def _padded(values):
    return list(values) + [math.nan] * INPUT_GUARD_COUNT


def _metal_payload(dtype, values):
    if dtype == "float32":
        return struct.pack(f"<{len(values)}I", *(_float32_bits(v) for v in values))
    return struct.pack(f"<{len(values)}H", *(_bits(dtype, v) for v in values))


def _upstream_metal(work, reference, entry, name, a, b):
    runner, library = reference
    dtype = _dtype(entry)
    count = CASES[name][0]
    groups = _groups(count)
    payloads = [
        _metal_payload(dtype, _padded(a)),
        _metal_payload(dtype, _padded(b)),
        struct.pack(
            f"<{groups + OUTPUT_GUARD_COUNT}I",
            *([OUTPUT_GUARD_BITS] * (groups + OUTPUT_GUARD_COUNT)),
        ),
        struct.pack("<i", count),
    ]
    files = []
    for index, payload in enumerate(payloads):
        path = work / f"{name}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    request = work / f"{name}-upstream-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": [groups, 1, 1],
                "workgroupSize": list(WORKGROUP_SIZE),
                "simdWidth": SIMD_SIZE,
            }
        ),
        encoding="utf-8",
    )
    output = work / f"{name}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, output], work, f"{name}-upstream")
    )
    for index in (0, 1, 3):
        assert (output / f"buffer-{index}.bin").read_bytes() == payloads[index]
    raw = (output / "buffer-2.bin").read_bytes()
    words = list(struct.unpack(f"<{len(raw) // 4}I", raw))
    assert words[groups:] == [OUTPUT_GUARD_BITS] * OUTPUT_GUARD_COUNT
    return device, words[:groups]


def _dispatch(descriptor, package, target, entry, name, a, b):
    dtype = _dtype(entry)
    count = CASES[name][0]
    groups = _groups(count)
    inputs, observed, output = {}, [], None
    prefix = re.sub(r"_+", "_", entry) + "_"
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            continue
        layout = binding["scalarLayout"]
        # DirectX constant members carry the sanitized entry name as a prefix.
        member = layout.get("memberName", binding["name"]).removeprefix(prefix)
        member = member.removesuffix("Buffer").rstrip("_")
        element_type, encoding = _element_layout(binding)
        if member in {"a", "b"}:
            value = _runtime_value(
                dtype, element_type, _padded(a if member == "a" else b), encoding
            )
        elif member == "n":
            value = {"dtype": "int32", "shape": [1], "values": [count]}
        else:
            assert member == "output", member
            assert element_type == "float32" and encoding is None
            value = {
                "dtype": "float32",
                "shape": [groups + OUTPUT_GUARD_COUNT],
                "encoding": FLOAT32_BITS,
                "values": [OUTPUT_GUARD_BITS] * (groups + OUTPUT_GUARD_COUNT),
            }
            output = binding["name"]
        if member in {"a", "b"} and encoding is not None:
            assert encoding["logicalElementType"] == dtype, (member, encoding)
        elif member in {"a", "b"}:
            assert element_type == value["dtype"], (member, element_type)
        inputs[binding["name"]] = value
        if member != "output":
            observed.append(binding["name"])
    assert output is not None and len(observed) == 3, (output, observed)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output: dict(inputs[output])},
        {"workgroupCount": [groups, 1, 1], "workgroupSize": list(WORKGROUP_SIZE)},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output


def _check_partials(name, words, partials, bounds):
    """Compare binary32 partial words with the reference; return the max error."""
    assert len(words) == len(partials)
    worst = 0.0
    for index, (word, expected, bound) in enumerate(zip(words, partials, bounds)):
        actual = _bits_float32(word)
        if CASES[name][1] != "rounding":
            assert word == _float32_bits(expected), (name, index, actual, expected)
            continue
        error = abs(actual - expected)
        assert error <= bound, (name, index, actual, expected, bound)
        worst = max(worst, error)
    return worst


@pytest.mark.parametrize("entry", ENTRIES)
def test_current_mlx_dot_native_parity(current_mlx, metal_reference, tmp_path, entry):
    root, target = current_mlx
    dtype = _dtype(entry)
    record = _contract()["artifacts"][target][entry]
    with tempfile.TemporaryDirectory(prefix=".current-dot-", dir=root) as directory:
        work = Path(directory)
        try:
            config = work / "crosstl.toml"
            config.write_text(
                _config(f"{work.name}/out", target, entry), encoding="utf-8"
            )
            translation = translate_project(
                load_project_config(root, config),
                format_output=False,
                validate=True,
                run_toolchains=False,
            )
            report = translation.to_json()
            translation.write_json(work / "report.json")
            validation = validate_project_report(work / "report.json")
            assert validation["success"] is True, validation
            assert report["diagnostics"] == []
            assert report["summary"]["translatedCount"] == 1
            assert report["summary"]["failedCount"] == 0
            (artifact,) = report["artifacts"]
            assert artifact["sourceHash"] == {
                "algorithm": "sha256",
                "value": SOURCE_SHA256,
            }
            check_generated_artifact(
                artifact, sha256=record["sha256"], size_bytes=record["sizeBytes"]
            )
            (execution,) = artifact["execution"]["entryPoints"]
            assert execution["workgroupSize"] == list(WORKGROUP_SIZE)
            generated = root / artifact["path"]
            text = generated.read_text(encoding="utf-8")
            assert "unsupported" not in text
            if target == "opengl":
                assert "subgroupAdd" not in text and "gl_Subgroup" not in text
                assert text.count("crossglSoftwareSubgroupSumFloat(") == 3
                assert "bool crossglSoftwareSubgroupActive = (tid < 16);" in text
                assert (
                    "float crossglSoftwareSubgroupInput = uintBitsToFloat(0x80000000u);"
                    in text
                )
            if target == "directx":
                assert "WaveActiveSum" not in text and "[WaveSize" not in text
                assert text.count("__crossgl_software_subgroup_sum_float(") == 3
                assert "bool crossglSoftwareSubgroupActive = tid < 16;" in text
                assert (
                    "float crossglSoftwareSubgroupInput = asfloat(0x80000000u);" in text
                )
            if target == "metal":
                _metal_library(generated, work / "translated.metallib", root)
            else:
                _validate_generated_artifact(generated, work, target)
            descriptor, package = _prepare_native_package(translation, work)
            executor = _executor(target)
            evidence = []
            try:
                for name in CASES:
                    a, b, partials, bounds = _reference(name, dtype)
                    request, output = _dispatch(
                        descriptor, package, target, entry, name, a, b
                    )
                    availability = executor.is_available(request)
                    assert availability.available, availability.reason
                    result = executor.run(request)
                    (work / f"{name}-result.json").write_text(
                        json.dumps(
                            {
                                "status": result.status,
                                "outputs": result.outputs,
                                "details": result.details,
                            },
                            indent=2,
                            default=str,
                        ),
                        encoding="utf-8",
                    )
                    assert result.status == "ok"
                    value = result.outputs[output]
                    assert value["dtype"] == "float32"
                    groups = len(partials)
                    words = value["values"][:groups]
                    assert all(isinstance(word, int) for word in words), words[:4]
                    assert value["values"][groups:] == (
                        [OUTPUT_GUARD_BITS] * OUTPUT_GUARD_COUNT
                    )
                    worst = _check_partials(name, words, partials, bounds)
                    observations = result.details["inputObservations"]
                    assert len(observations) == 3
                    assert all(item["unchanged"] for item in observations.values())
                    upstream = None
                    if target == "metal":
                        device, upstream_words = _upstream_metal(
                            work, metal_reference, entry, name, a, b
                        )
                        _check_partials(name, upstream_words, partials, bounds)
                        upstream = {
                            "device": device,
                            "bitIdenticalToTranslated": upstream_words == words,
                        }
                        assert upstream_words == words
                    evidence.append(
                        {
                            "case": name,
                            "elementCount": CASES[name][0],
                            "kind": CASES[name][1],
                            "partialCount": groups,
                            "maxAbsoluteError": worst,
                            "outputGuardCount": OUTPUT_GUARD_COUNT,
                            "upstream": upstream,
                        }
                    )
            finally:
                close = getattr(executor.runtime_adapter.runtime, "close", None)
                if callable(close):
                    close()
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entry": entry,
                        "target": target,
                        "dtype": dtype,
                        "subgroups": SUBGROUPS[target],
                        "artifactSha256": record["sha256"],
                        "cases": evidence,
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)


def test_ci_requires_native_dot_parity():
    from tests import workflow_text
    from tests.ci_helpers import (
        assert_native_step_prerequisites,
        assert_workflow_triggers,
    )

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = workflow_text.workflow_job_step_section(
        workflow,
        "mlx-metal-porting",
        "Prove current MLX dot native parity",
    )
    assert_native_step_prerequisites(step, job="mlx-metal-porting")
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/dot",
        "--junitxml=mlx-current-results/dot.xml",
        "demos/integrations/mlx/tests/kernels/test_current_dot.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_dot.py"
    )
    upload = workflow_text.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Upload current MLX runtime diagnostics"
    )
    assert "if: always()" in upload
    assert "path: mlx-current-results" in upload
    assert workflow.index("Prove current MLX random native parity") < (
        workflow.index("Prove current MLX dot native parity")
    )
    assert workflow.index("Prove current MLX dot native parity") < (
        workflow.index("Upload current MLX runtime diagnostics")
    )

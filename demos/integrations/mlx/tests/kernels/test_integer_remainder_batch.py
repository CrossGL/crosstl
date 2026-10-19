"""Inventory, package selection and failure retention for integer binary demos."""

import hashlib
import json

import pytest

from crosstl.project import load_project_config, translate_project
from demos.integrations.mlx.tests.kernels import test_current_integer_remainder as rem


def test_batch_preserves_single_entry_configuration_and_input_inventory(tmp_path):
    path = tmp_path / "crosstl.toml"
    path.write_text(rem._remainder_config(), encoding="utf-8")
    batch = load_project_config(tmp_path, path)
    entries = [rem._entry(dtype) for dtype in rem.TYPES]
    assert entries == [
        "vv_Remainderbool_",
        "vv_Remainderint8",
        "vv_Remainderuint8",
        "vv_Remainderint16",
        "vv_Remainderuint16",
        "vv_Remainderint32",
        "vv_Remainderint64",
    ]
    assert tuple(batch.entry_points[rem.MLX_BINARY_SOURCE]) == tuple(entries)
    assert dict(batch.entry_workgroup_size_rules[rem.MLX_BINARY_SOURCE]) == {
        "*": ("1", "1", "1")
    }
    for workload in rem.BINARY_OPENGL_WORKLOADS:
        if workload.entry_point not in entries:
            continue
        path.write_text(rem._project_config(workload), encoding="utf-8")
        single = load_project_config(tmp_path, path)
        assert single.source_options == batch.source_options
        assert single.index_range_assertions == batch.index_range_assertions
        assert single.entry_points[rem.MLX_BINARY_SOURCE] == workload.entry_point
        assert dict(single.entry_workgroup_size_rules[rem.MLX_BINARY_SOURCE]) == {
            workload.entry_point: ("1", "1", "1")
        }
    assert sum(len(rem._pairs(dtype)) for dtype in rem.TYPES) == 168730
    assert len(entries) * 8 == 56


@pytest.fixture
def small_binary_root(tmp_path):
    source = tmp_path / rem.MLX_BINARY_SOURCE
    source.parent.mkdir(parents=True)
    source.write_text(
        """#include <metal_stdlib>
        using namespace metal;
        template <typename T>
        [[kernel]] void binary(
            const device T* a [[buffer(0)]],
            const device T* b [[buffer(1)]],
            device T* c [[buffer(2)]],
            constant uint& size [[buffer(3)]],
            uint i [[thread_position_in_grid]]) {
            if (i < size) c[i] = a[i];
        }
        """
        + "\n".join(
            f'instantiate_kernel("{rem._entry(dtype)}", binary, {typename})'
            for dtype, (_, typename, _, _) in rem.TYPES.items()
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_batch_selects_each_reflected_entry(small_binary_root, target):
    root = small_binary_root
    config_path = root / "crosstl.toml"
    config_path.write_text(rem._remainder_config(), encoding="utf-8")
    report = translate_project(
        load_project_config(root, config_path), targets=(target,), format_output=False
    )
    artifacts, descriptors, package = rem._prepare_remainder_batch(report, root, target)
    assert (
        set(artifacts) == set(descriptors) == {rem._entry(dtype) for dtype in rem.TYPES}
    )
    assert len({d["artifact"]["packagePath"] for d in descriptors.values()}) == 7
    for dtype in rem.TYPES:
        entry = rem._entry(dtype)
        descriptor = descriptors[entry]
        assert descriptor["target"] == target
        source = package / descriptor["artifact"]["packagePath"]
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest()
            == artifacts[entry]["generatedHash"]["value"]
        )
        pairs = rem._pairs(dtype)[:2]
        expected = [rem._expected(a, b, dtype) for a, b in pairs] + [
            rem._guard(dtype)
        ] * 8
        request, inputs, outputs = rem._request(
            descriptor, package, dtype, target, pairs, expected
        )
        assert not request.execution_plan.diagnostics
        assert inputs["a"]["values"] == [a for a, _ in pairs]
        (output,) = outputs.values()
        assert output["values"] == expected


@pytest.mark.parametrize("failure", (None, "translation", "int16"))
def test_batch_translates_once_and_retains_failed_evidence(
    monkeypatch, small_binary_root, failure
):
    root = small_binary_root
    monkeypatch.setenv(rem.REQUIRE_ENV, "1")
    monkeypatch.setenv("CROSTL_MLX_CURRENT_ROOT", str(root))
    monkeypatch.setenv("CROSTL_MLX_CURRENT_TARGET", "opengl")
    monkeypatch.setattr(
        rem,
        "MLX_BINARY_SHA256",
        hashlib.sha256((root / rem.MLX_BINARY_SOURCE).read_bytes()).hexdigest(),
    )
    calls = []

    def revision(command, **kwargs):
        assert command == ["git", "-C", str(root), "rev-parse", "HEAD"]
        return rem.MLX_COMMIT + "\n"

    def clean(command, **kwargs):
        assert command == [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
        ]
        assert kwargs["check"]

    def translate(config, **kwargs):
        calls.append("translate")
        assert tuple(config.entry_points[rem.MLX_BINARY_SOURCE]) == tuple(
            rem._entry(dtype) for dtype in rem.TYPES
        )
        report = translate_project(config, **kwargs)
        if failure == "translation":
            report.write_json(root / kwargs["output_dir"] / "failed-report.json")
            raise RuntimeError("translation failed")
        return report

    def execute(work, dtype, target, artifact, descriptor, package, runner, library):
        calls.append(dtype)
        assert target == "opengl" and runner is None and library is None
        assert artifact["entryPoint"]["source"] == rem._entry(dtype)
        assert (package / descriptor["artifact"]["packagePath"]).is_file()
        (work / "execution-attempt.json").write_text(json.dumps({"dtype": dtype}))
        if failure == dtype:
            raise RuntimeError("execution failed")

    monkeypatch.setattr(rem.subprocess, "check_output", revision)
    monkeypatch.setattr(rem.subprocess, "run", clean)
    monkeypatch.setattr(rem, "translate_project", translate)
    monkeypatch.setattr(rem, "_run_remainder_case", execute)
    output = root / "retained"
    output.mkdir()
    source_calls = []

    def source_control(source_root, target):
        source_calls.append((source_root, target))
        return None, None

    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            rem.test_current_integer_remainder_native_parity(output, source_control)
    else:
        rem.test_current_integer_remainder_native_parity(output, source_control)
    assert calls.count("translate") == 1
    assert source_calls == [(root, "opengl")]
    evidence = output / "evidence"
    assert (evidence / "crosstl.toml").is_file()
    if failure == "translation":
        assert (evidence / "out/failed-report.json").is_file()
        assert calls == ["translate"]
    else:
        expected = list(rem.TYPES)
        if failure:
            expected = expected[: expected.index(failure) + 1]
        assert calls == ["translate", *expected]
        assert (evidence / "report.json").is_file()
        assert (evidence / "loader.json").is_file()
        for dtype in expected:
            assert json.loads(
                (evidence / dtype / "execution-attempt.json").read_text()
            ) == {"dtype": dtype}
    assert not list(root.glob(".current-integer-remainder-*"))

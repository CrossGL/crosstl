"""Manifest source reflection is reused without sharing artifact contracts."""

from collections import Counter

import pytest

from crosstl.project import pipeline


@pytest.mark.parametrize("source_available", [False, True])
def test_manifest_reuses_source_reflection_only_within_one_build(
    tmp_path, monkeypatch, source_available
):
    (tmp_path / "kernels.metal").write_text(
        """#include <metal_stdlib>
using namespace metal;
kernel void first(device uint* result [[buffer(0)]],
                  uint index [[thread_position_in_grid]]) { result[index] = 1u; }
kernel void second(device uint* result [[buffer(0)]],
                   uint index [[thread_position_in_grid]]) { result[index] = 2u; }
""",
        encoding="utf-8",
    )
    config = pipeline.ProjectConfig(
        root=tmp_path,
        targets=["metal", "directx", "opengl"],
        output_dir="out",
        entry_points={"kernels.metal": ["first", "second"]},
        variants={
            "small": {"workgroup_size": [32, 1, 1]},
            "large": {"workgroup_size": [64, 1, 1]},
        },
    )
    report = pipeline.translate_project(config, format_output=False)
    assert report.to_json()["summary"]["failedCount"] == 0
    path = tmp_path / "report.json"
    report.write_json(path)
    original_source = pipeline._runtime_manifest_source_host_interface
    original_reflected = pipeline._runtime_manifest_reflected_host_interface
    sources, generated = Counter(), Counter()

    def source(root, artifact):
        sources[(str(root), artifact["source"], artifact["target"])] += 1
        return original_source(root, artifact) if source_available else None

    def reflected(root, artifact):
        generated[artifact["path"]] += 1
        return original_reflected(root, artifact)

    monkeypatch.setattr(pipeline, "_runtime_manifest_source_host_interface", source)
    monkeypatch.setattr(
        pipeline, "_runtime_manifest_reflected_host_interface", reflected
    )
    manifest = pipeline.build_runtime_artifact_manifest(path)
    assert manifest["success"]
    assert len(manifest["artifacts"]) == 12
    assert len(sources) == 3 and set(sources.values()) == {1}
    assert len(generated) == 12 and set(generated.values()) == {1}
    for artifact, original in zip(manifest["artifacts"], report.to_json()["artifacts"]):
        uncached = pipeline._runtime_manifest_artifact(original, root_path=tmp_path)
        assert artifact["hostInterface"] == uncached["hostInterface"]
        assert artifact["entryPoint"] == original["entryPoint"]
        assert artifact["execution"] == original.get("execution")
        if artifact["target"] != "metal":
            assert artifact["execution"] is not None
    sources.clear()
    generated.clear()
    repeated = pipeline.build_runtime_artifact_manifest(path)
    assert repeated["artifacts"] == manifest["artifacts"]
    assert len(sources) == 3 and set(sources.values()) == {1}
    assert len(generated) == 12 and set(generated.values()) == {1}
    generated_path = tmp_path / report.to_json()["artifacts"][0]["path"]
    generated_path.write_text(generated_path.read_text() + "\n// changed\n")
    invalid = pipeline.build_runtime_artifact_manifest(path)
    assert not invalid["success"]
    assert not invalid["artifacts"]


def test_reused_source_interfaces_do_not_share_mutable_results(tmp_path, monkeypatch):
    interface = {"entryPoints": [{"name": "main", "parameters": [{"name": "x"}]}]}
    monkeypatch.setattr(
        pipeline, "_runtime_manifest_source_host_interface", lambda *_: interface
    )
    monkeypatch.setattr(
        pipeline, "_runtime_manifest_reflected_host_interface", lambda *_: None
    )
    cache = {}
    artifact = {"source": "kernel.metal", "sourceBackend": "metal", "target": "cgl"}
    first = pipeline._runtime_manifest_host_interface(
        tmp_path, artifact, source_interfaces=cache
    )
    first["entryPoints"][0]["parameters"][0]["name"] = "changed"
    second = pipeline._runtime_manifest_host_interface(
        tmp_path, artifact, source_interfaces=cache
    )
    assert second["entryPoints"][0]["parameters"][0]["name"] == "x"
    assert interface["entryPoints"][0]["parameters"][0]["name"] == "x"

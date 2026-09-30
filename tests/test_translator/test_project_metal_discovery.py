from contextlib import contextmanager
from dataclasses import replace

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project import ProjectConfig, translate_project
from crosstl.project.translation_checkpoint import (
    ProjectTranslationCheckpointRecorder,
    load_project_translation_checkpoint,
)


@contextmanager
def _uncached_discovery(monkeypatch):
    with monkeypatch.context() as patch:
        for name in (
            "template_functions",
            "non_template_function_definitions",
        ):
            patch.setattr(
                MetalPreprocessor,
                f"_find_{name}",
                getattr(MetalPreprocessor, f"_scan_{name}"),
            )

        def traits(self, code, namespace_spans=None):
            if namespace_spans is None:
                namespace_spans = self._find_namespace_spans(code)
            return self._scan_template_type_traits(code, namespace_spans)

        patch.setattr(MetalPreprocessor, "_find_template_type_traits", traits)
        yield


@pytest.fixture
def discovery_project(tmp_path):
    (tmp_path / "adjust.h").write_text(
        "template <typename T> T adjust(T value) {\n"
        "    return value + T(VARIANT_BIAS + 3);\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "kernels.metal").write_text(
        '#include <metal_stdlib>\n#include "adjust.h"\n'
        "using namespace metal;\n"
        "template <typename T, int Offset>\n"
        "[[kernel]] void launch(device T* out [[buffer(0)]],\n"
        "                      uint gid [[thread_position_in_grid]]) {\n"
        "    out[gid] = adjust<T>(T(gid + Offset));\n"
        "}\n"
        'instantiate_kernel("first", launch, float, 1)\n'
        'instantiate_kernel("second", launch, float, 2)\n',
        encoding="utf-8",
    )
    return ProjectConfig(
        root=tmp_path,
        targets=("metal", "directx", "opengl"),
        include_patterns=("kernels.metal",),
        include_dirs=(".",),
        entry_points={"kernels.metal": ("first", "second")},
        entry_workgroup_size_rules={"kernels.metal": {"*": [1, 1, 1]}},
        variants={"one": {"VARIANT_BIAS": "1"}, "two": {"VARIANT_BIAS": "2"}},
        source_options={
            "metal": {
                "max_template_specializations": 8,
                "max_template_materialization_work": 32,
            }
        },
        output_dir="translated",
    )


def _translate(config, **kwargs):
    payload = translate_project(config, format_output=False, **kwargs).to_json()
    sources = {
        artifact["path"]: (config.root / artifact["path"]).read_bytes()
        for artifact in payload["artifacts"]
        if artifact["status"] == "translated"
    }
    return payload, sources


def _assert_equivalent(actual, expected):
    actual_payload, actual_sources = actual
    expected_payload, expected_sources = expected
    for key in ("artifacts", "diagnostics", "artifactMatrix"):
        assert actual_payload[key] == expected_payload[key]
    assert actual_sources == expected_sources


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_discovery_reuse_preserves_entry_and_variant_artifacts(
    discovery_project, monkeypatch, target
):
    config = replace(discovery_project, targets=(target,))
    with _uncached_discovery(monkeypatch):
        uncached = _translate(config)
    cached = _translate(config)
    repeated = _translate(config)

    assert cached[0]["diagnostics"] == []
    assert cached[0]["summary"]["translatedCount"] == 4
    assert len(set(cached[1].values())) == 4
    _assert_equivalent(cached, uncached)
    _assert_equivalent(repeated, uncached)


@pytest.mark.parametrize("change", ["include", "defines"])
def test_discovery_reuse_observes_changed_includes_and_defines(
    discovery_project, monkeypatch, change
):
    config = discovery_project
    initial = _translate(config)
    if change == "include":
        header = config.root / "adjust.h"
        header.write_text(
            header.read_text(encoding="utf-8").replace("+ 3", "+ 7"), encoding="utf-8"
        )
    else:
        config = replace(config, variants={"one": {"VARIANT_BIAS": "4"}})
    with _uncached_discovery(monkeypatch):
        uncached = _translate(config)
    cached = _translate(config)

    assert cached[0]["diagnostics"] == []
    assert cached[0]["summary"]["translatedCount"] == (12 if change == "include" else 6)
    assert all(source != initial[1][path] for path, source in cached[1].items())
    _assert_equivalent(cached, uncached)


def test_discovery_reuse_preserves_work_limit_diagnostics(
    discovery_project, monkeypatch
):
    assert _translate(discovery_project)[0]["summary"]["translatedCount"] == 12
    config = replace(
        discovery_project,
        source_options={"metal": {"max_template_materialization_work": 1}},
    )
    with _uncached_discovery(monkeypatch):
        uncached = _translate(config)
    cached = _translate(config)

    assert cached[0]["summary"]["failedCount"] == 12
    assert cached[0]["diagnostics"]
    assert all(
        diagnostic["code"] == "project.translate.metal-template-specialization"
        for diagnostic in cached[0]["diagnostics"]
    )
    assert not cached[1]
    _assert_equivalent(cached, uncached)


def test_discovery_reuse_preserves_parallel_and_resumed_artifacts(
    discovery_project, monkeypatch
):
    config = discovery_project
    with _uncached_discovery(monkeypatch):
        uncached = _translate(config)
    checkpoint = config.root / "checkpoint.json"
    parallel = _translate(config, max_workers=2, checkpoint_path=checkpoint)
    complete = load_project_translation_checkpoint(checkpoint)
    recorder = ProjectTranslationCheckpointRecorder(
        checkpoint,
        complete["projectIdentity"],
        complete["plan"]["jobs"],
        started_at=complete["startedAt"],
        completed=complete["plan"]["completed"][:3],
        initial_diagnostics=complete["diagnostics"][
            : complete["initialDiagnosticCount"]
        ],
    )
    recorder.write_interrupted(None, RuntimeError("translation interrupted"))
    resumed = _translate(config, max_workers=2, checkpoint_path=checkpoint, resume=True)

    assert parallel[0]["summary"]["translatedCount"] == 12
    assert parallel[0]["diagnostics"] == []
    _assert_equivalent(parallel, uncached)
    _assert_equivalent(resumed, uncached)
    assert load_project_translation_checkpoint(checkpoint)["state"] == "complete"

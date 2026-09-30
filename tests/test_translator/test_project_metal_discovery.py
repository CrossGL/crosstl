import json
import os
import shutil
import struct
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project import ProjectConfig, translate_project
from crosstl.project.translation_checkpoint import (
    ProjectTranslationCheckpointRecorder,
    load_project_translation_checkpoint,
)
from tests.test_translator.test_fused_math import _dispatch
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_ARGUMENT_INFERENCE"


def constrained_struct_source():
    return """#include <metal_stdlib>
    using namespace metal;
    template <typename T> constexpr constant bool accepts = is_same_v<T, float>;
    template <typename T, typename = void> struct Cell { uint value; };
    template <typename T> struct Cell<T, enable_if_t<accepts<T>>> { float value; };
    kernel void select_storage(device const uint* values [[buffer(0)]],
                               device uint* results [[buffer(1)]],
                               uint tid [[thread_position_in_grid]]) {
        Cell<float> fractional;
        fractional.value = float(values[tid * 3]) + 0.5f;
        Cell<uint> integral;
        integral.value = values[tid * 3] + 2u;
        results[tid * 2] = as_type<uint>(fractional.value);
        results[tid * 2 + 1] = integral.value;
    }
    """


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_project_constrained_struct_preserves_both_storage_types(tmp_path, target):
    path = tmp_path / "kernel.metal"
    path.write_text(constrained_struct_source(), encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=(target,),
            include_patterns=(path.name,),
            entry_points={path.name: ("select_storage",)},
            workgroup_size=(1, 1, 1),
            output_dir="out",
        ),
        format_output=False,
    ).to_json()
    assert report["diagnostics"] == []
    assert report["summary"]["translatedCount"] == 1
    generated = (tmp_path / report["artifacts"][0]["path"]).read_text(encoding="utf-8")
    assert "struct Cell_float_void {\n    float value;" in generated
    assert "struct Cell_uint {\n    uint value;" in generated


@pytest.mark.parametrize("original", [False, True])
def test_constrained_structs_execute_with_selected_storage(tmp_path, original):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native argument inference checks")
    if original and sys.platform != "darwin":
        pytest.skip("The original Metal control requires macOS")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    source = constrained_struct_source()
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    if not original:
        report = translate_project(
            ProjectConfig(
                root=tmp_path,
                targets=(target,),
                include_patterns=(path.name,),
                entry_points={path.name: ("select_storage",)},
                workgroup_size=(1, 1, 1),
                output_dir="out",
            ),
            format_output=False,
        )
        report.write_json(tmp_path / "report.json")
        payload = report.to_json()
        assert payload["diagnostics"] == []
        assert payload["summary"]["translatedCount"] == 1
        source = (tmp_path / payload["artifacts"][0]["path"]).read_text(
            encoding="utf-8"
        )
    inputs = [(value, 0, 0) for value in range(97)]
    expected = [
        word
        for value in range(97)
        for word in (struct.unpack("<I", struct.pack("<f", value + 0.5))[0], value + 2)
    ]
    actual, evidence = _dispatch(
        tmp_path,
        target,
        source,
        inputs,
        len(expected),
        entry="select_storage" if target == "metal" else None,
    )
    evidence.update(original=original, expected=expected, actual=actual)
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    assert actual == expected


@pytest.mark.parametrize("original", [False, True])
@pytest.mark.parametrize(
    "predicate_kind", ["disjunction", "specialization", "namespace-alias"]
)
def test_named_boolean_constraints_execute_both_overloads(
    tmp_path, original, predicate_kind
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native argument inference checks")
    if original and sys.platform != "darwin":
        pytest.skip("The original Metal control requires macOS")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    predicates = {
        "disjunction": (
            """
    #pragma METAL internals : enable
    template <typename T> constexpr constant bool accepts =
        _disjunction<is_same<T, int>, is_same<T, uint>>::value;
    #pragma METAL internals : disable
        """
        ),
        "specialization": (
            """
    template <typename T> constexpr constant bool accepts = false;
    template <> constexpr constant bool accepts<uint> = true;
        """
        ),
        "namespace-alias": (
            """
    namespace traits {
      using Index = uint;
      template <typename T> constexpr constant bool accepts = is_same_v<T, Index>;
    }
        """
        ),
    }
    source = """#include <metal_stdlib>
    using namespace metal;
    PREDICATE
    template <typename T, enable_if_t<accepts<T>, bool> = true>
    T choose(T value) { return value + T(1); }
    template <typename T, enable_if_t<!accepts<T>, bool> = true>
    T choose(T value) { return value + T(2); }
    kernel void select_values(device const uint* values [[buffer(0)]],
                              device uint* results [[buffer(1)]],
                              uint tid [[thread_position_in_grid]]) {
        uint value = values[tid * 3];
        results[tid * 2] = choose(value);
        results[tid * 2 + 1] = uint(choose(float(value)));
    }
    """.replace("PREDICATE", predicates[predicate_kind])
    if predicate_kind == "namespace-alias":
        source = source.replace(
            "enable_if_t<accepts<T>", "enable_if_t<traits::accepts<T>"
        ).replace("enable_if_t<!accepts<T>", "enable_if_t<!traits::accepts<T>")
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    if not original:
        report = translate_project(
            ProjectConfig(
                root=tmp_path,
                targets=(target,),
                include_patterns=(path.name,),
                entry_points={path.name: ("select_values",)},
                workgroup_size=(1, 1, 1),
                output_dir="out",
            ),
            format_output=False,
        )
        report.write_json(tmp_path / "report.json")
        payload = report.to_json()
        assert payload["diagnostics"] == []
        assert payload["summary"]["translatedCount"] == 1
        source = (tmp_path / payload["artifacts"][0]["path"]).read_text(
            encoding="utf-8"
        )
    inputs = [(value, 0, 0) for value in range(97)]
    expected = [value + increment for value in range(97) for increment in (1, 2)]
    actual, evidence = _dispatch(
        tmp_path,
        target,
        source,
        inputs,
        len(expected),
        entry="select_values" if target == "metal" else None,
    )
    evidence.update(original=original, expected=expected, actual=actual)
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    assert actual == expected


@pytest.mark.parametrize("original", [False, True])
def test_inferred_pointer_offsets_execute(tmp_path, original):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native argument inference checks")
    if original and sys.platform != "darwin":
        pytest.skip("The original Metal control requires macOS")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    source = """#include <metal_stdlib>
    using namespace metal;
    template <typename T, enable_if_t<is_same_v<T, uint>, bool> = true>
    void store(device T* out, T value) { out[0] = value; }
    kernel void scatter_index(
        device const uint* values [[buffer(0)]], // host count
        device uint* results [[buffer(1)]], // destination, [batch, count, stride]
        uint3 tid [[thread_position_in_grid]]) {
        int count = int(values[0]);
        auto n = tid.x;
        auto batch = n / 24;
        auto lane = n % 24;
        const int stride = 24;
        auto base = results + batch * count * stride + lane;
        store(base, n + 1);
    }
    """
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    if not original:
        config = ProjectConfig(
            root=tmp_path,
            targets=(target,),
            include_patterns=(path.name,),
            entry_points={path.name: ("scatter_index",)},
            workgroup_size=(1, 1, 1),
            output_dir="out",
        )
        report = translate_project(config, format_output=False)
        report.write_json(tmp_path / "report.json")
        payload = report.to_json()
        assert payload["diagnostics"] == []
        assert payload["summary"]["translatedCount"] == 1
        source = (tmp_path / payload["artifacts"][0]["path"]).read_text(
            encoding="utf-8"
        )
    inputs = [(3, 0, 0)] * 97
    expected = [0] * 289
    for n in range(len(inputs)):
        expected[(n // 24) * 3 * 24 + n % 24] = n + 1
    actual, evidence = _dispatch(
        tmp_path,
        target,
        source,
        inputs,
        len(expected),
        entry="scatter_index" if target == "metal" else None,
    )
    evidence.update(original=original, expected=expected, actual=actual)
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    assert actual == expected


@pytest.mark.parametrize("filename", ["mlx-portable-host.yml", "mlx-metal-host.yml"])
def test_ci_requires_native_argument_inference(filename):
    workflow = (
        Path(__file__).resolve().parents[2] / ".github/workflows" / filename
    ).read_text()
    step = workflow.split("      - name: Validate Metal argument inference\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_project_metal_discovery.py" in step
    assert "--basetemp=" in step
    assert "--junitxml=" in step
    assert "--timeout-seconds 120" in step
    assert "pytest -q -n auto" in step


def test_integral_argument_inference_matches_metal_type_system(tmp_path):
    if sys.platform != "darwin" or not shutil.which("xcrun"):
        pytest.skip("Metal type checking requires the native macOS compiler")
    preprocessor = MetalPreprocessor()
    scalars = sorted(preprocessor._METAL_INTEGRAL_SCALAR_TYPES)
    pairs = [(left, right) for left in scalars for right in scalars]
    pairs.extend((name, name) for name in ("char2", "ushort3", "int4", "uint2"))
    assertions = []
    for left, right in pairs:
        for operator in ("+", "-", "*", "/", "%"):
            inferred = preprocessor._infer_argument_type(
                f"left {operator} right", {}, {"left": left, "right": right}
            )
            assert inferred is not None, (left, operator, right)
            assertions.append(
                f"static_assert(is_same_v<decltype({left}(1) {operator} {right}(1)), "
                f'{inferred}>, "{left} {operator} {right}");'
            )
    source = (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        + "\n".join(assertions)
        + "\nkernel void check_types(device uint* out [[buffer(0)]]) { out[0] = 1; }\n"
    )
    _, module = _compile(source, "metal", tmp_path)
    assert module.stat().st_size > 0


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_project_infers_compound_integral_pointer_offsets(tmp_path, target):
    source = """#include <metal_stdlib>
    using namespace metal;
    template <typename T, enable_if_t<is_same_v<T, uint>, bool> = true>
    void store(device T* out, T value) { out[0] = value; }
    kernel void scatter_index(
        device uint* out [[buffer(0)]], // [batch, count, stride]
        constant int& count [[buffer(1)]], // number of elements
        uint3 tid [[thread_position_in_grid]]) {
        auto n = tid.z;
        auto batch = n / 24;
        auto lane = n % 24;
        const int stride = 24;
        auto base = out + batch * count * stride + lane;
        store(base, n);
    }
    """
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    config = ProjectConfig(
        root=tmp_path,
        targets=(target,),
        include_patterns=(path.name,),
        entry_points={path.name: ("scatter_index",)},
        workgroup_size=(1, 1, 32),
        output_dir="out",
    )
    payload = translate_project(config, format_output=False).to_json()
    assert payload["diagnostics"] == []
    assert payload["summary"]["translatedCount"] == 1
    artifact = payload["artifacts"][0]
    generated = (tmp_path / artifact["path"]).read_text(encoding="utf-8")
    assert "store_uint" in generated
    assert "% 24" in generated
    _compile(generated, target, tmp_path)

    path.write_text(
        source.replace("auto base =", "device uint* base ="), encoding="utf-8"
    )
    control = translate_project(config, format_output=False).to_json()
    assert control["diagnostics"] == []
    assert control["summary"]["translatedCount"] == 1
    assert (tmp_path / artifact["path"]).read_text(encoding="utf-8") == generated


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_project_parameter_comments_preserve_constrained_calls(tmp_path, target):
    source = """#include <metal_stdlib>
    using namespace metal;
    template <typename T, enable_if_t<is_same_v<T, uint>, bool> = true>
    T identity(T value) { return value; }
    kernel void copy_index(
        device uint* out [[buffer(0)]], // output, [B, T]
        constant uint& count [[buffer(1)]], // number of elements
        uint tid [[thread_position_in_grid]]) {
        out[tid] = identity(tid) + count;
    }
    """
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    config = ProjectConfig(
        root=tmp_path,
        targets=(target,),
        include_patterns=(path.name,),
        entry_points={path.name: ("copy_index",)},
        workgroup_size=(32, 1, 1),
        output_dir="out",
    )
    payload = translate_project(config, format_output=False).to_json()
    assert payload["diagnostics"] == []
    assert payload["summary"]["translatedCount"] == 1
    artifact = payload["artifacts"][0]
    generated = (tmp_path / artifact["path"]).read_text(encoding="utf-8")
    assert "identity_uint(tid)" in generated
    _compile(generated, target, tmp_path)

    path.write_text(
        source.replace("// output, [B, T]", "").replace("// number of elements", ""),
        encoding="utf-8",
    )
    control = translate_project(config, format_output=False).to_json()
    assert control["diagnostics"] == []
    assert control["summary"]["translatedCount"] == 1
    for key in (
        "entryPoint",
        "generatedHash",
        "generatedSizeBytes",
        "requiredCapabilities",
    ):
        assert control["artifacts"][0][key] == artifact[key]
    assert (tmp_path / artifact["path"]).read_text(encoding="utf-8") == generated


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

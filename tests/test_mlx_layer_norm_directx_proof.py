import hashlib
import importlib.util
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROOF_PATH = ROOT / "demos" / "integrations" / "mlx" / "prove_layer_norm_directx.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "mlx-project-porting.yml"


def _load_proof():
    spec = importlib.util.spec_from_file_location(
        "mlx_layer_norm_directx_proof",
        PROOF_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _host_evidence_source():
    return textwrap.dedent("""
        void LayerNorm::eval_gpu() {
          int simd_size = 32;
          int n_reads = 8;
          int looped_limit = 6656;
          if (axis_size <= looped_limit) {
            size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
            size_t simds_needed =
                (threadgroup_needed + simd_size - 1) / simd_size;
            size_t threadgroup_size = simd_size * simds_needed;
          }
        }

        void LayerNormVJP::eval_gpu() {
          int simd_size = 32;
          int n_reads = 8;
          int looped_limit = 8192;
          if (axis_size <= looped_limit) {
            size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
            size_t simds_needed =
                (threadgroup_needed + simd_size - 1) / simd_size;
            size_t threadgroup_size = simd_size * simds_needed;
          }
        }
        """)


def _test_evidence_source():
    return textwrap.dedent("""
        class TestFast:
            def test_layer_norm(self):
                dims, dtype, eps = 4099, mx.float32, 1e-5
                x = mx.random.uniform(shape=(dims,)).astype(dtype)
                mx.fast.layer_norm(x, None, None, eps)

            def test_layer_norm_grad(self):
                f2 = lambda x, w, b, y: (
                    mx.fast.layer_norm(x, w, b, 1e-5) * y
                ).sum()
                D = 8192
                w = mx.random.uniform(shape=(D,))
                mx.grad(f2, argnums=(0, 1, 2))
        """)


def _reduced_layer_norm_source():
    return textwrap.dedent("""
            #include <metal_stdlib>
            using namespace metal;

            constant bool has_w [[function_constant(20)]];

            template <int N = 1>
            inline void threadgroup_sum(
                thread float* x,
                threadgroup float* scratch,
                uint simd_lane_id [[thread_index_in_simdgroup]],
                uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
              for (int i = 0; i < N; i++) {
                x[i] = simd_sum(x[i]);
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              if (simd_lane_id == 0) {
                scratch[simd_group_id] = x[0];
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              for (int i = 0; i < N; i++) {
                x[i] = scratch[simd_lane_id];
                x[i] = simd_sum(x[i]);
              }
            }

            template <typename T, int N_READS = 8>
            [[kernel]] void layer_norm_single_row(
                const device T* x,
                device T* out,
                uint index [[thread_position_in_grid]],
                uint simd_lane_id [[thread_index_in_simdgroup]],
                uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
              constexpr int SIMD_SIZE = 32;
              float value = x[index] + static_cast<T>(N_READS);
              threadgroup float scratch[SIMD_SIZE];
              threadgroup_sum(
                  &value, scratch, simd_lane_id, simd_group_id);
              threadgroup_sum(
                  &value, scratch, simd_lane_id, simd_group_id);
              out[index] = static_cast<T>(value);
            }

            template <typename T, int N_READS = 8>
            [[kernel]] void vjp_layer_norm_single_row(
                const device T* x,
                device T* out,
                uint index [[thread_position_in_grid]],
                uint simd_lane_id [[thread_index_in_simdgroup]],
                uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
              constexpr int SIMD_SIZE = 32;
              float value = has_w ? x[index] : static_cast<T>(N_READS);
              threadgroup float scratch[SIMD_SIZE];
              threadgroup_sum(
                  &value, scratch, simd_lane_id, simd_group_id);
              threadgroup_sum(
                  &value, scratch, simd_lane_id, simd_group_id);
              out[index] = static_cast<T>(value);
            }

            // clang-format off
            #define instantiate_layer_norm(name, itype)
            instantiate_layer_norm(float32, float)
            // clang-format on
            """).strip() + "\n"


def _write_reduced_layer_norm_source(module, mlx_root):
    source_path = mlx_root / module.MLX_LAYER_NORM_SOURCE
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        _reduced_layer_norm_source(),
        encoding="utf-8",
    )
    return source_path


def test_layer_norm_directx_contract_pins_real_source_and_cases():
    module = _load_proof()

    assert module.MLX_COMMIT == "4367c73b60541ddd5a266ce4644fd93d20223b6e"
    assert module.MLX_LAYER_NORM_SOURCE == (
        "mlx/backend/metal/kernels/layer_norm.metal"
    )
    assert module.MLX_LAYER_NORM_SHA256 == (
        "2d243f5abea7353929f9bc838ceb5a98e52a452dfc29609ad4d5974447ea689f"
    )
    assert [case["entryPoint"] for case in module.LAYER_NORM_DIRECTX_CASES] == [
        "layer_normfloat32",
        "vjp_layer_normfloat32",
    ]
    assert [case["workgroupSize"] for case in module.LAYER_NORM_DIRECTX_CASES] == [
        [544, 1, 1],
        [1024, 1, 1],
    ]
    assert all(
        case["axisSize"] <= case["loopedLimit"]
        for case in module.LAYER_NORM_DIRECTX_CASES
    )
    assert module.LAYER_NORM_SUBGROUP_WIDTH == 32
    assert module.LAYER_NORM_DIRECTX_PROFILE == "cs_6_6"


def test_source_subgroup_contract_requires_pinned_32_lane_reductions():
    module = _load_proof()

    evidence = module._verify_source_subgroup_contract(_reduced_layer_norm_source())

    assert evidence["status"] == "passed"
    assert evidence["subgroupWidth"] == 32
    assert evidence["reduction"] == {
        "helper": "threadgroup_sum",
        "primitive": "simd_sum",
        "simdSumCallCount": 2,
    }
    assert evidence["requiredDirectXEnforcement"] == {
        "attribute": "WaveSize(32)",
        "minimumShaderModel": "6.6",
        "profile": "cs_6_6",
    }
    assert [
        entry["threadgroupSumCallCount"] for entry in evidence["selectedEntries"]
    ] == [2, 2]


def test_source_subgroup_contract_fails_closed_on_non_32_width():
    module = _load_proof()
    source = _reduced_layer_norm_source().replace(
        "constexpr int SIMD_SIZE = 32;",
        "constexpr int SIMD_SIZE = 64;",
        1,
    )

    with pytest.raises(
        module.MlxLayerNormDirectXProofError,
        match="must declare exactly SIMD_SIZE = 32",
    ):
        module._verify_source_subgroup_contract(source)


def test_dispatch_evidence_derives_exact_test_backed_workgroups():
    module = _load_proof()

    evidence = module._verify_dispatch_evidence(
        _host_evidence_source(),
        _test_evidence_source(),
    )

    assert evidence["status"] == "passed"
    assert evidence["loopedEntriesIncluded"] is False
    assert [case["workgroupSize"] for case in evidence["cases"]] == [
        [544, 1, 1],
        [1024, 1, 1],
    ]
    assert [case["branch"] for case in evidence["cases"]] == [
        "single-row",
        "single-row",
    ]


def test_dispatch_evidence_fails_closed_when_formula_is_ambiguous():
    module = _load_proof()
    host_source = _host_evidence_source().replace(
        "size_t threadgroup_size = simd_size * simds_needed;",
        "size_t threadgroup_size = simd_size * simds_needed;\n"
        "size_t duplicate_threadgroup_size = simd_size * simds_needed;",
        1,
    )

    with pytest.raises(
        module.MlxLayerNormDirectXProofError,
        match="ambiguous dispatch formula",
    ):
        module._verify_dispatch_evidence(host_source, _test_evidence_source())


def test_git_blob_evidence_preserves_repository_bytes(tmp_path, monkeypatch):
    module = _load_proof()
    payload = b"first line\nsecond line\n"

    def run(command, *, check, capture_output, timeout):
        assert command[-1] == "revision:path/to/evidence.cpp"
        assert check is False
        assert capture_output is True
        assert timeout == 180
        return SimpleNamespace(returncode=0, stdout=payload, stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", run)
    log_dir = tmp_path / "logs"
    result = module._git_blob(
        tmp_path,
        "revision",
        "path/to/evidence.cpp",
        log_dir=log_dir,
    )

    assert result == payload
    assert (log_dir / "mlx-layernorm-git-show-evidence.cpp.stdout").read_bytes() == (
        payload
    )


@pytest.mark.parametrize("case_index", [0, 1])
def test_project_config_selects_one_materialized_entry_and_variant(
    tmp_path,
    case_index,
):
    module = _load_proof()
    project_root = tmp_path / "project"
    project_root.mkdir()
    case = module.LAYER_NORM_DIRECTX_CASES[case_index]

    config = module._project_config(project_root, case)

    assert config.targets == ["directx"]
    assert config.entry_points == {module.MLX_LAYER_NORM_SOURCE: case["entryPoint"]}
    assert config.selected_variants == [case["name"]]
    assert config.variant_workgroup_sizes == {}
    assert config.workgroup_size_rules == {
        module.MLX_LAYER_NORM_SOURCE: tuple(
            str(component) for component in case["workgroupSize"]
        )
    }
    assert config.subgroup_width_rules == {module.MLX_LAYER_NORM_SOURCE: "32"}
    assert config.variant_specialization_constants == (
        {case["name"]: {case["selector"]: case["value"]}}
        if case["usesFunctionConstant"]
        else {}
    )


@pytest.mark.parametrize("case_index", [0, 1])
def test_case_project_preserves_kernel_body_and_selects_one_instantiation(
    tmp_path,
    monkeypatch,
    case_index,
):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()
    source_path = _write_reduced_layer_norm_source(module, mlx_root)
    source = source_path.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "MLX_LAYER_NORM_SHA256", source_hash)
    case = module.LAYER_NORM_DIRECTX_CASES[case_index]

    project_root, projection = module._prepare_case_project(
        mlx_root,
        mlx_root / ".proof",
        case,
    )

    projected_path = project_root / module.MLX_LAYER_NORM_SOURCE
    projected = projected_path.read_text(encoding="utf-8")
    original_prefix = source.partition(module.LAYER_NORM_INSTANTIATION_MARKER)[0]
    projected_prefix = projected.partition("// clang-format off\n")[0]
    assert projected_prefix == original_prefix
    assert projected.count("instantiate_kernel(") == 1
    assert (
        f'instantiate_kernel("{case["entryPoint"]}", ' f'{case["templateName"]}, float)'
    ) in projected
    assert "#define instantiate_layer_norm" not in projected
    assert projection["sourceHash"] == {
        "algorithm": "sha256",
        "value": hashlib.sha256(projected_path.read_bytes()).hexdigest(),
    }
    assert projection["derivedFrom"]["sourceHash"] == {
        "algorithm": "sha256",
        "value": source_hash,
    }
    assert projection["transform"]["kind"] == "instantiation-manifest-selection"
    assert projection["transform"]["selectedEntryPoint"] == case["entryPoint"]


def test_case_project_fails_closed_on_ambiguous_instantiation_manifest(
    tmp_path,
    monkeypatch,
):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()
    source_path = _write_reduced_layer_norm_source(module, mlx_root)
    source_path.write_text(
        source_path.read_text(encoding="utf-8")
        + module.LAYER_NORM_INSTANTIATION_MARKER
        + "\n",
        encoding="utf-8",
    )
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "MLX_LAYER_NORM_SHA256", source_hash)

    with pytest.raises(
        module.MlxLayerNormDirectXProofError,
        match="must have one exact marker",
    ):
        module._prepare_case_project(
            mlx_root,
            mlx_root / ".proof",
            module.LAYER_NORM_DIRECTX_CASES[0],
        )


def test_selected_materialization_rejects_duplicate_entry_identity():
    module = _load_proof()
    case = module.LAYER_NORM_DIRECTX_CASES[0]
    selected = {
        "name": case["templateName"],
        "materializedName": case["entryPoint"],
        "parameters": {"N_READS": "8", "T": "float"},
        "parameterSources": {
            "N_READS": "source-default",
            "T": "source-instantiation",
        },
        "source": "source-instantiation",
        "hostName": case["entryPoint"],
    }
    artifact = {
        "templateMaterialization": {
            "status": "materialized",
            "specializationCount": 2,
            "specializations": [selected, dict(selected)],
            "unsupported": [],
        }
    }

    with pytest.raises(
        module.MlxLayerNormDirectXProofError,
        match="one host-named materialization",
    ):
        module._selected_materialization(artifact, case)


def test_reduced_layer_norm_project_proof_emits_exact_standalone_entries(
    tmp_path,
    monkeypatch,
):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()
    source_path = _write_reduced_layer_norm_source(module, mlx_root)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "MLX_LAYER_NORM_SHA256", source_hash)
    monkeypatch.setattr(module, "_find_dxc", lambda: None)

    dispatch_evidence = module._verify_dispatch_evidence(
        _host_evidence_source(),
        _test_evidence_source(),
    )
    subgroup_evidence = module._verify_source_subgroup_contract(
        source_path.read_text(encoding="utf-8")
    )

    def verify_checkout(root, *, log_dir):
        assert root == mlx_root.resolve()
        assert log_dir.is_dir()
        return (
            {
                "name": "pinned-source-identity",
                "status": "passed",
                "repository": module.MLX_REPOSITORY,
                "commit": module.MLX_COMMIT,
                "source": module.MLX_LAYER_NORM_SOURCE,
                "sourceHash": {"algorithm": "sha256", "value": source_hash},
            },
            dispatch_evidence,
            subgroup_evidence,
        )

    monkeypatch.setattr(module, "_verify_mlx_checkout", verify_checkout)
    summary = module.run_proof(
        SimpleNamespace(
            mlx_root=str(mlx_root),
            work_dir=".layer-norm-proof",
            no_clean=False,
            require_directx_toolchain=False,
        )
    )

    assert summary["status"] == "passed"
    assert summary["scope"] == {
        "source": module.MLX_LAYER_NORM_SOURCE,
        "sourceSha256": source_hash,
        "target": "directx",
        "projectTranslationApi": "crosstl.project.translate_project",
        "selectedEntryPoints": [
            "layer_normfloat32",
            "vjp_layer_normfloat32",
        ],
        "subgroupWidth": 32,
        "subgroupWidthEnforcement": "WaveSize(32)",
        "minimumShaderModel": "6.6",
        "dxcProfile": "cs_6_6",
        "translationClaimed": True,
        "nativeCompilationClaimed": False,
        "runtimeParityClaimed": False,
        "numericalExecutionIncluded": False,
        "fullMlxTestSuiteIncluded": False,
        "loopedEntriesIncluded": False,
    }
    assert [check["name"] for check in summary["checks"]] == [
        "pinned-source-identity",
        "pinned-host-dispatch-evidence",
        "pinned-source-subgroup-contract",
    ]
    workgroup_rule_path = (
        f'project.workgroup_size_rules["{module.MLX_LAYER_NORM_SOURCE}"]'
    )
    subgroup_rule_path = (
        f'project.subgroup_width_rules["{module.MLX_LAYER_NORM_SOURCE}"]'
    )
    for case, expected_entry, expected_workgroup in zip(
        summary["cases"],
        ("layer_normfloat32", "vjp_layer_normfloat32"),
        ([544, 1, 1], [1024, 1, 1]),
        strict=True,
    ):
        execution = case["execution"]
        assert execution["sourceEntryPoint"] == expected_entry
        assert execution["targetEntryPoint"] == "CSMain"
        assert execution["workgroupSize"] == expected_workgroup
        assert execution["workgroupSizeRule"] == {
            "components": [str(component) for component in expected_workgroup],
            "sourcePattern": module.MLX_LAYER_NORM_SOURCE,
            "path": workgroup_rule_path,
        }
        assert execution["provenance"] == {
            "kind": "materialized-template-rule",
            "path": workgroup_rule_path,
        }
        assert execution["subgroupWidth"] == 32
        assert execution["subgroupWidthRule"] == {
            "expression": "32",
            "sourcePattern": module.MLX_LAYER_NORM_SOURCE,
            "path": subgroup_rule_path,
        }
        assert execution["subgroupWidthProvenance"] == {
            "kind": "materialized-template-rule",
            "path": subgroup_rule_path,
        }
        assert execution["subgroupWidthEnforcement"] == {
            "mechanism": "hlsl-wave-size-attribute",
            "minimumShaderModel": "6.6",
            "entryProfiles": [{"entryPoint": "CSMain", "profile": "cs_6_6"}],
        }
        assert case["sourceProjection"]["derivedFrom"]["sourceHash"] == {
            "algorithm": "sha256",
            "value": source_hash,
        }
    assert summary["cases"][0]["specializationConstant"] is None
    assert (
        summary["cases"][1]["specializationConstant"]["emittedInSelectedEntry"] is True
    )
    assert all(
        case["nativeCompilation"]["status"] == "unavailable"
        for case in summary["cases"]
    )
    for case in summary["cases"]:
        artifact = mlx_root / case["artifact"]
        generated = artifact.read_text(encoding="utf-8")
        assert generated.count("void CSMain(") == 1
        assert generated.count("[WaveSize(32)]") == 1
        assert generated.count("[WaveSize(") == 1
        workgroup_size = case["execution"]["workgroupSize"]
        assert (
            f"[numthreads({workgroup_size[0]}, {workgroup_size[1]}, "
            f"{workgroup_size[2]})]\n[WaveSize(32)]\nvoid CSMain("
        ) in generated


def test_dxc_validation_compiles_selected_entry_with_warnings_as_errors(
    tmp_path,
    monkeypatch,
):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".proof"
    log_dir = work_dir / "logs"
    artifact_path = work_dir / "artifacts" / "layer_normfloat32.hlsl"
    artifact_path.parent.mkdir(parents=True)
    generated = "[numthreads(32, 1, 1)]\n" "[WaveSize(32)]\n" "void CSMain() {}\n"
    artifact_path.write_text(generated, encoding="utf-8")
    captured = {}

    def run_command(name, command, *, log_dir, timeout_seconds=180):
        captured["name"] = name
        captured["command"] = list(command)
        output_path = Path(command[command.index("-Fo") + 1])
        output_path.write_bytes(b"DXIL")
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "stdoutPath": stdout_path,
            "stderrPath": stderr_path,
        }

    monkeypatch.setattr(module, "_run_command", run_command)
    result = module._compile_directx_artifact(
        artifact_path,
        generated,
        module.LAYER_NORM_DIRECTX_CASES[0],
        dxc="dxc",
        mlx_root=mlx_root,
        work_dir=work_dir,
        log_dir=log_dir,
        required=True,
    )

    assert result["status"] == "compiled"
    assert result["required"] is True
    assert result["entryPoint"] == "CSMain"
    assert result["profile"] == "cs_6_6"
    assert result["subgroupWidth"] == 32
    assert result["subgroupWidthEnforcement"] == "WaveSize(32)"
    assert result["compiledArtifactCount"] == 1
    assert captured["command"][:4] == ["dxc", "-WX", "-T", "cs_6_6"]
    assert captured["command"][captured["command"].index("-E") + 1] == "CSMain"


def test_required_dxc_validation_fails_when_compiler_is_unavailable(tmp_path):
    module = _load_proof()
    mlx_root = tmp_path / "mlx"
    artifact_path = mlx_root / "artifact.hlsl"
    artifact_path.parent.mkdir()
    artifact_path.write_text(
        "[numthreads(32, 1, 1)]\n[WaveSize(32)]\nvoid CSMain() {}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        module.MlxLayerNormDirectXProofError,
        match="requires dxc",
    ):
        module._compile_directx_artifact(
            artifact_path,
            artifact_path.read_text(encoding="utf-8"),
            module.LAYER_NORM_DIRECTX_CASES[0],
            dxc=None,
            mlx_root=mlx_root,
            work_dir=mlx_root / ".proof",
            log_dir=mlx_root / ".proof" / "logs",
            required=True,
        )


def test_mlx_workflow_requires_layer_norm_directx_proof_on_windows():
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "name: Prove pinned MLX LayerNorm DirectX entries" in workflow
    assert "if: runner.os == 'Windows'" in workflow
    assert "python demos/integrations/mlx/prove_layer_norm_directx.py" in workflow
    assert "--work-dir .crosstl-mlx-porting/layer-norm-directx" in workflow
    assert "--require-directx-toolchain" in workflow
    assert 'MLX_COMMIT: "4367c73b60541ddd5a266ce4644fd93d20223b6e"' in workflow

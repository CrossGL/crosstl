import hashlib
import json
from pathlib import Path

import crosstl.project as project_api

MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
CURRENT_MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
MLX_REPOSITORY = "https://github.com/ml-explore/mlx"
MLX_HOST_SOURCE = "mlx/backend/metal/logsumexp.cpp"
MLX_KERNEL_SOURCE = "mlx/backend/metal/kernels/logsumexp.metal"
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "logsumexp.dispatch.json"
)
NATIVE_LOADER_FIXTURE = FIXTURE.with_name("logsumexp.native-loader.dispatch.json")
EXPECTED_MANIFEST_DIGEST = (
    "db762a188e05786e206d9aa5a340b6f9095a8a3e938b85a7c04836f300e97c95"
)
EXPECTED_NATIVE_LOADER_MANIFEST_DIGEST = (
    "3cfc400f25cf49cb16d028fdba59ebe8b56b729ade919f711de4b8b67bfa5ab4"
)
EXPECTED_VARIANTS = {
    "block-float32-axis-32": {
        "axisSize": 32,
        "workgroupSize": (32, 1, 1),
        "variant": (
            "sha256:0a9cafa696d2fddd6284c44673b9120120a70630a0a18714e27284f8a6189c59"
        ),
        "artifact": (
            "sha256:f51ab67b8a9ad3240e3fbb52f6de00cdf4a8532e58790758e59aa50fb95e2c52"
        ),
    },
    "block-float32-axis-1025": {
        "axisSize": 1025,
        "workgroupSize": (288, 1, 1),
        "variant": (
            "sha256:35754405ceaa5e50614c3fef95a66390b99580c9bd2394c0ba5279322fde7446"
        ),
        "artifact": (
            "sha256:ae512c102a88628c05a49f28a872c44ab582bacf74584e8ca7e6ae765263afe0"
        ),
    },
}


def _workgroup_formula():
    return {
        "op": "multiply",
        "args": [
            32,
            {
                "op": "ceilDiv",
                "args": [
                    {
                        "op": "ceilDiv",
                        "args": [{"input": "axisSize"}, 4],
                    },
                    32,
                ],
            },
        ],
    }


def test_logsumexp_dispatch_fixture_pins_schema_identity_and_provenance():
    manifest = project_api.load_dispatch_contract(FIXTURE)
    canonical = json.dumps(
        manifest.to_json(),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    assert manifest.schema_version == project_api.DISPATCH_CONTRACT_SCHEMA_VERSION == 1
    assert manifest.content_identity.to_json() == {
        "algorithm": "sha256",
        "value": EXPECTED_MANIFEST_DIGEST,
    }
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == (
        EXPECTED_MANIFEST_DIGEST
    )
    assert manifest.provenance["repository"] == MLX_REPOSITORY
    assert manifest.provenance["commit"] == MLX_COMMIT
    assert manifest.provenance["sourceReferences"] == {
        "hostDispatch": MLX_HOST_SOURCE,
        "kernel": MLX_KERNEL_SOURCE,
    }
    assert manifest.contracts[0].provenance["hostSource"] == MLX_HOST_SOURCE
    assert manifest.contracts[0].provenance["kernelSource"] == MLX_KERNEL_SOURCE


def test_logsumexp_native_loader_fixture_tracks_current_corpus_revision():
    historical = project_api.load_dispatch_contract(FIXTURE)
    current = project_api.load_dispatch_contract(NATIVE_LOADER_FIXTURE)

    assert current.content_identity.to_json() == {
        "algorithm": "sha256",
        "value": EXPECTED_NATIVE_LOADER_MANIFEST_DIGEST,
    }
    assert current.provenance["commit"] == CURRENT_MLX_COMMIT

    def executable_fields(manifest):
        variants = []
        for variant in manifest.evaluate():
            payload = variant.to_json()
            payload.pop("provenance")
            variants.append(payload)
        return variants

    assert executable_fields(current) == executable_fields(historical)

    expected = historical.to_json()
    expected["provenance"]["commit"] = CURRENT_MLX_COMMIT
    assert current.to_json() == expected


def test_logsumexp_dispatch_fixture_evaluates_unit_test_variants_exactly():
    variants = {
        variant.workload_id: variant
        for variant in project_api.load_dispatch_contract(FIXTURE).evaluate()
    }

    assert set(variants) == set(EXPECTED_VARIANTS)
    for workload_id, expected in EXPECTED_VARIANTS.items():
        variant = variants[workload_id]
        assert variant.inputs == {
            "axisSize": expected["axisSize"],
            "dtype": "float32",
            "nRows": 1,
        }
        assert variant.entry_point == "block_logsumexp_float32"
        assert variant.branch_id == "block-float32"
        assert variant.contract_id == "mlx-logsumexp-unit-tests"
        assert variant.device_id == "wave32-max1024"
        assert variant.source == MLX_KERNEL_SOURCE
        assert variant.workgroup_size == expected["workgroupSize"]
        assert variant.subgroup_width == 32
        assert variant.capabilities == {
            "maxThreadsPerWorkgroup": 1024,
            "simdWidth": 32,
        }
        assert variant.specialization_constants == {}
        assert variant.dispatch_field == "workgroupCount"
        assert variant.dispatch_size == (1, 1, 1)
        assert variant.variant_id == expected["variant"]
        assert variant.artifact_id == expected["artifact"]


def test_logsumexp_dispatch_fixture_is_test_backed_and_explicitly_bounded():
    manifest = project_api.load_dispatch_contract(FIXTURE)
    normalized = manifest.to_json()
    workloads = {workload["id"]: workload for workload in normalized["workloads"]}
    contract = normalized["contracts"][0]

    assert len(manifest.workloads) == 2
    assert len(manifest.contracts) == 1
    assert len(manifest.contracts[0].branches) == 1
    assert workloads["block-float32-axis-32"]["provenance"] == {
        "hostFunction": "LogSumExp::eval_gpu",
        "branch": "block",
        "testSource": "python/tests/test_ops.py::test_logsumexp",
        "shape": [2, 2, 8],
        "coveredAxisSizes": [3, 4, 6, 32],
        "additionalTestSource": "python/tests/test_autograd.py::test_logsumexp_grad",
    }
    assert workloads["block-float32-axis-1025"]["provenance"] == {
        "hostFunction": "LogSumExp::eval_gpu",
        "branch": "block",
        "testSource": "python/tests/test_ops.py::test_logsumexp",
        "shape": [1025],
    }
    assert contract["provenance"]["hostFormula"] == (
        "32 * ceilDiv(ceilDiv(axisSize, 4), 32)"
    )
    assert contract["branches"][0]["workgroupSize"][0] == _workgroup_formula()
    assert manifest.provenance["scope"] == {
        "description": "Pinned float32 LogSumExp unit-test dispatch records.",
        "blockEntriesIncluded": True,
        "loopedEntriesIncluded": False,
        "runtimeExecutionVerified": False,
        "numericalParityVerified": False,
    }

import hashlib
import json
from pathlib import Path

import crosstl.project as project_api

MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
MLX_REPOSITORY = "https://github.com/ml-explore/mlx"
MLX_HOST_SOURCE = "mlx/backend/metal/normalization.cpp"
MLX_KERNEL_SOURCE = "mlx/backend/metal/kernels/layer_norm.metal"
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "layer_norm.dispatch.json"
)
EXPECTED_MANIFEST_DIGEST = (
    "527140d8720a414ff5eaeb670ee9312571c5ca1a13460353d0c177193f5a8f98"
)
EXPECTED_IDENTITIES = {
    "forward-float32-axis-4099": {
        "variant": (
            "sha256:9f67f3aa323b6a35566bcce84c678121ba050a49843e10187bd5abf4ab3dd01c"
        ),
        "artifact": (
            "sha256:6c2a76fa651fc945ff5f320801d70b925a3b289c2927990ab4161535edc8d6bf"
        ),
    },
    "vjp-float32-axis-8192": {
        "variant": (
            "sha256:eed664cf712ef3aeb953e3539389a6f3adcd853d8374aa16da343055080edbf1"
        ),
        "artifact": (
            "sha256:7f6bf28c1536b59c8bd51d9f55dc80cb337698f15e1433f378e1692625e45d54"
        ),
    },
}


def _formula():
    return {
        "op": "multiply",
        "args": [
            32,
            {
                "op": "ceilDiv",
                "args": [
                    {
                        "op": "ceilDiv",
                        "args": [{"input": "axisSize"}, 8],
                    },
                    32,
                ],
            },
        ],
    }


def test_layer_norm_dispatch_fixture_pins_schema_identity_and_provenance():
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


def test_layer_norm_dispatch_fixture_evaluates_exact_finite_variants():
    result = project_api.load_dispatch_contract(FIXTURE).evaluate()
    variants = {variant.workload_id: variant for variant in result}

    assert set(variants) == set(EXPECTED_IDENTITIES)
    expected = {
        "forward-float32-axis-4099": {
            "entryPoint": "layer_normfloat32",
            "axisSize": 4099,
            "isVjp": False,
            "hasW": False,
            "workgroupSize": (544, 1, 1),
            "branchId": "forward-single-row-float32",
            "specializationConstants": {},
        },
        "vjp-float32-axis-8192": {
            "entryPoint": "vjp_layer_normfloat32",
            "axisSize": 8192,
            "isVjp": True,
            "hasW": True,
            "workgroupSize": (1024, 1, 1),
            "branchId": "vjp-single-row-float32",
            "specializationConstants": {"20": True},
        },
    }
    for workload_id, case in expected.items():
        variant = variants[workload_id]
        assert variant.entry_point == case["entryPoint"]
        assert variant.inputs == {
            "axisSize": case["axisSize"],
            "dtype": "float32",
            "isVjp": case["isVjp"],
            "hasW": case["hasW"],
        }
        assert variant.branch_id == case["branchId"]
        assert variant.contract_id == "mlx-layer-norm-single-row-float32"
        assert variant.device_id == "wave32-max1024"
        assert variant.source == MLX_KERNEL_SOURCE
        assert variant.workgroup_size == case["workgroupSize"]
        assert variant.subgroup_width == 32
        assert variant.capabilities == {
            "maxThreadsPerWorkgroup": 1024,
            "simdWidth": 32,
        }
        assert variant.specialization_constants == case["specializationConstants"]
        if variant.specialization_constants:
            assert int(next(iter(variant.specialization_constants))) == 20
        assert variant.dispatch_field == "workgroupCount"
        assert variant.dispatch_size == (1, 1, 1)
        assert variant.variant_id == EXPECTED_IDENTITIES[workload_id]["variant"]
        assert variant.artifact_id == EXPECTED_IDENTITIES[workload_id]["artifact"]


def test_layer_norm_dispatch_fixture_is_deterministic_and_explicitly_bounded():
    first = project_api.load_dispatch_contract(FIXTURE)
    second = project_api.load_dispatch_contract(FIXTURE)
    first_variants = {
        variant.workload_id: (variant.variant_id, variant.artifact_id)
        for variant in first.evaluate()
    }
    second_variants = {
        variant.workload_id: (variant.variant_id, variant.artifact_id)
        for variant in second.evaluate()
    }
    normalized = first.to_json()
    contract = normalized["contracts"][0]
    scope = first.provenance["scope"]

    assert first.content_identity == second.content_identity
    assert first_variants == second_variants
    assert len(first.workloads) == 2
    assert len(contract["branches"]) == 2
    assert contract["provenance"]["hostFormula"] == (
        "32 * ceilDiv(ceilDiv(axisSize, 8), 32)"
    )
    assert all(
        branch["workgroupSize"][0] == _formula() for branch in contract["branches"]
    )
    assert all(
        branch["dispatch"] == {"workgroupCount": [1, 1, 1]}
        for branch in contract["branches"]
    )
    assert "specializationConstants" not in contract["branches"][0]
    assert contract["branches"][1]["specializationConstants"] == {
        "20": {
            "input": "hasW",
            "provenance": {"symbol": "has_w"},
        }
    }
    assert first.devices[0].values == {
        "maxThreadsPerWorkgroup": 1024,
        "simdWidth": 32,
    }
    assert scope == {
        "description": "Pinned single-row float32 LayerNorm dispatch records.",
        "singleRowOnly": True,
        "loopedEntriesIncluded": False,
        "runtimeExecutionVerified": False,
        "numericalParityVerified": False,
    }
    assert all(
        "looped" not in branch.entry_point for branch in first.contracts[0].branches
    )

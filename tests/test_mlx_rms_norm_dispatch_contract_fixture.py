import hashlib
import json
from pathlib import Path

import crosstl.project as project_api

MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
MLX_REPOSITORY = "https://github.com/ml-explore/mlx"
MLX_HOST_SOURCE = "mlx/backend/metal/normalization.cpp"
MLX_KERNEL_SOURCE = "mlx/backend/metal/kernels/rms_norm.metal"
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "demos"
    / "integrations"
    / "mlx"
    / "contracts"
    / "rms_norm.dispatch.json"
)
EXPECTED_MANIFEST_DIGEST = (
    "ea238af83b140c33d43b79b5efd1814c398bbc09ac70cbef21375b1e8ff9a1eb"
)
EXPECTED_VARIANTS = {
    "forward-float32-axis-32": (
        "rmsfloat32",
        32,
        {},
        (2, 1, 1),
        "4306831dce3a9a479ef63093a7f2722358caf58b6fcf1a47ed808a1c28dc9ebb",
        "00c05fccf276cf11f3fb9b617b8fe0bb3c5f8766c0e4ca1ed990c093e700422e",
    ),
    "forward-float32-axis-256": (
        "rmsfloat32",
        64,
        {},
        (2, 1, 1),
        "f27b102d2ee4f473afef2448e42103223922ff29fd703e6d269d827f280e8bf7",
        "1ef80b00c1a7a2f7967177bc003a961f3e4448358716d2deb79778fc3cbfb68e",
    ),
    "forward-float32-axis-512": (
        "rmsfloat32",
        128,
        {},
        (2, 1, 1),
        "369be00b341be529b913101657bd6a78d8b8c689e9dc7c6d27e694809dd9d098",
        "a9b6980b1645e867b2502052d6d5f37a03447258bb4f819535901bf45a01d5da",
    ),
    "forward-float16-axis-32": (
        "rmsfloat16",
        32,
        {},
        (2, 1, 1),
        "51de914a20a4defb1d1b79ed26def94a994894228d6c64cae0e1132553809dbb",
        "b694e5240f2a87bfae8af862878251a45cfaeaf39fd810bf2df8a5e3724bdad7",
    ),
    "forward-bfloat16-axis-32": (
        "rmsbfloat16",
        32,
        {},
        (2, 1, 1),
        "635d28f468bf3e15a0ec7285aacc23d0de32f9596d511beb98adb3ecc2c2abf4",
        "13655322998b557a5143ac9b871dc898f0ae43c07cf9659aecd505156ed9318b",
    ),
    "forward-float32-axis-4099": (
        "rms_loopedfloat32",
        1024,
        {},
        (1, 1, 1),
        "392aee49734fcc2fd3ff4fd232a49f1a97a2bb79f3c22cc80eeabeb1f1ca1959",
        "b81c2043b10bde966cb6f4dbfa198d2b93a3e456f3026030b69557c4a8983729",
    ),
    "vjp-float32-axis-32-has-w-false": (
        "vjp_rmsfloat32",
        32,
        {"20": False},
        (800, 1, 1),
        "1a25dca51070c3b6fc96f162e6c152049d388833ba02b1f3d10cc1928c5661c4",
        "ef832e1ceb8c864a13aee3460d23658f4fffba18db1b800461628ba6ebe38e0a",
    ),
    "vjp-float32-axis-32-has-w-true": (
        "vjp_rmsfloat32",
        32,
        {"20": True},
        (800, 1, 1),
        "26177a77e484a56b7b2572516e8e1360714c88b668881237a9cf499129e34f35",
        "a9be06b43a6156fb9ee1f9a6955d03d6bda0940c2a8223b58f564c2d12bd0cd0",
    ),
    "vjp-float32-axis-256-has-w-false": (
        "vjp_rmsfloat32",
        64,
        {"20": False},
        (800, 1, 1),
        "f26ffb357ecfabef6216ff45661f298af4f164a0494da838ca579ea0812e73d2",
        "4455fc9204f826fc5d0d7f016bcaf970b75ea31d5f1598d13ceca6a2baa369e7",
    ),
    "vjp-float32-axis-256-has-w-true": (
        "vjp_rmsfloat32",
        64,
        {"20": True},
        (800, 1, 1),
        "a010324f59769fde9d71cc8968852ae1e6c8b0ddc213f035523c2c5f2e12d413",
        "0944044e2f050bedde1d05e1ae5648628e7144c04752dd49b2e2bb7bcd807b7b",
    ),
    "vjp-float32-axis-8192-has-w-false": (
        "vjp_rms_loopedfloat32",
        1024,
        {"20": False},
        (4, 1, 1),
        "fa4d57c473cd5799c46a982e3ea339debc06de21fb6a255adc11bbb546f9329b",
        "3bd55b546fc00ddf8412f092da4793c0272eec0ad7c130065ad7c1677f60cdce",
    ),
    "vjp-float32-axis-8192-has-w-true": (
        "vjp_rms_loopedfloat32",
        1024,
        {"20": True},
        (4, 1, 1),
        "8fc64f93a1be95f7acac67d6595c9f9c66c01b87cf952f29052835c20d4d765b",
        "345d524ffec14682b6d0325bc97b624b89d83dc257a17ed49bea5e11e24573f3",
    ),
}


def test_rms_norm_dispatch_fixture_pins_schema_identity_and_provenance():
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


def test_rms_norm_dispatch_fixture_evaluates_unit_test_workloads_exactly():
    result = project_api.load_dispatch_contract(FIXTURE).evaluate()
    variants = {variant.workload_id: variant for variant in result}

    assert set(variants) == set(EXPECTED_VARIANTS)
    assert len(variants) == 12
    for workload_id, expected in EXPECTED_VARIANTS.items():
        (
            entry_point,
            workgroup_x,
            specialization_constants,
            dispatch_size,
            variant_digest,
            artifact_digest,
        ) = expected
        variant = variants[workload_id]
        assert variant.entry_point == entry_point
        assert variant.contract_id == "mlx-rms-norm-unit-tests"
        assert variant.device_id == "wave32-max1024"
        assert variant.source == MLX_KERNEL_SOURCE
        assert variant.workgroup_size == (workgroup_x, 1, 1)
        assert variant.subgroup_width == 32
        assert variant.capabilities == {
            "maxThreadsPerWorkgroup": 1024,
            "simdWidth": 32,
        }
        assert variant.specialization_constants == specialization_constants
        assert variant.dispatch_field == "workgroupCount"
        assert variant.dispatch_size == dispatch_size
        assert variant.variant_id == f"sha256:{variant_digest}"
        assert variant.artifact_id == f"sha256:{artifact_digest}"


def test_rms_norm_dispatch_fixture_is_finite_test_backed_and_non_runtime():
    manifest = project_api.load_dispatch_contract(FIXTURE)
    normalized = manifest.to_json()
    workloads = {workload["id"]: workload for workload in normalized["workloads"]}
    scope = manifest.provenance["scope"]

    assert len(manifest.workloads) == 12
    assert len(manifest.contracts) == 1
    assert len(manifest.contracts[0].branches) == 6
    assert workloads["forward-float32-axis-32"]["provenance"] == {
        "hostFunction": "RMSNorm::eval_gpu",
        "branch": "single-row",
        "testSource": "python/tests/test_fast.py::test_rms_norm",
        "shape": [2, 32],
        "coveredAxisSizes": [31, 32, 33],
    }
    assert {
        workload["provenance"]["testSource"] for workload in workloads.values()
    } == {
        "python/tests/test_fast.py::test_rms_norm",
        "python/tests/test_fast.py::test_rms_norm_grad",
    }
    assert scope == {
        "description": "Pinned RMSNorm unit-test dispatch records.",
        "singleRowAndLoopedEntriesIncluded": True,
        "runtimeExecutionVerified": False,
        "numericalParityVerified": False,
    }

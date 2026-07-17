from dataclasses import replace

import pytest

from crosstl.project.dispatch_contracts import (
    DISPATCH_CONTRACT_KIND,
    DISPATCH_CONTRACT_SCHEMA_VERSION,
    parse_dispatch_contract,
)
from crosstl.project.dispatch_planning import (
    DISPATCH_ARTIFACT_PLAN_KIND,
    DISPATCH_ARTIFACT_PLAN_SCHEMA_VERSION,
    DispatchArtifactJob,
    DispatchArtifactPlan,
    DispatchArtifactPlanError,
    plan_dispatch_artifacts,
)


def _dispatch_evaluation(
    source,
    *,
    contract_id,
    entry_point,
    workgroup_size=(64, 1, 1),
    element_counts=(64,),
    manifest_source=None,
):
    payload = {
        "kind": DISPATCH_CONTRACT_KIND,
        "schemaVersion": DISPATCH_CONTRACT_SCHEMA_VERSION,
        "inputs": [{"name": "elementCount", "role": "shape", "type": "integer"}],
        "workloads": [
            {
                "id": f"count-{element_count}",
                "values": {"elementCount": element_count},
            }
            for element_count in element_counts
        ],
        "capabilities": [],
        "devices": [{"id": "portable", "values": {}}],
        "contracts": [
            {
                "id": contract_id,
                "source": source,
                "entryPoint": entry_point,
                "branches": [
                    {
                        "id": "default",
                        "when": True,
                        "workgroupSize": list(workgroup_size),
                        "dispatch": {
                            "workgroupCount": [
                                {
                                    "op": "ceilDiv",
                                    "args": [
                                        {"input": "elementCount"},
                                        workgroup_size[0],
                                    ],
                                },
                                1,
                                1,
                            ]
                        },
                    }
                ],
            }
        ],
    }
    return parse_dispatch_contract(payload, source=manifest_source).evaluate()


def _plan_fixture():
    copy_evaluation = _dispatch_evaluation(
        "kernels/copy.metal",
        contract_id="copy",
        entry_point="copy_float32",
        element_counts=(64, 257),
        manifest_source="contracts/copy.json",
    )
    reduce_evaluation = _dispatch_evaluation(
        "kernels/reduce.metal",
        contract_id="reduce",
        entry_point="reduce_float32",
        workgroup_size=(128, 1, 1),
        element_counts=(1024,),
        manifest_source="contracts/reduce.json",
    )
    return copy_evaluation, reduce_evaluation


def _error_code(error):
    prefix = "dispatch-"
    return error.code[len(prefix) :] if error.code.startswith(prefix) else error.code


def test_plans_multiple_sources_and_deduplicates_compile_equivalent_variants():
    copy_evaluation, reduce_evaluation = _plan_fixture()

    plan = plan_dispatch_artifacts(
        (copy_evaluation, reduce_evaluation),
        source_units=("kernels/copy.metal", "kernels/reduce.metal"),
    )

    assert isinstance(plan, DispatchArtifactPlan)
    assert plan.source_units == (
        "kernels/copy.metal",
        "kernels/reduce.metal",
    )
    assert len(plan.artifacts) == 2
    assert len(plan.dispatch_variants) == 3
    assert all(isinstance(job, DispatchArtifactJob) for job in plan.artifacts)

    copy_job = next(job for job in plan.artifacts if job.source == "kernels/copy.metal")
    assert copy_job.artifact_id == copy_evaluation[0].artifact_id
    assert copy_job.entry_point == "copy_float32"
    assert copy_job.workgroup_size == (64, 1, 1)
    assert copy_job.subgroup_width is None
    assert copy_job.specialization_constants == {}
    assert copy_job.dispatch_variant_ids == tuple(
        sorted(variant.variant_id for variant in copy_evaluation)
    )
    assert len(copy_job.dispatch_variant_ids) == 2

    assert {variant.variant_id for variant in plan.dispatch_variants} == {
        variant.variant_id
        for evaluation in (copy_evaluation, reduce_evaluation)
        for variant in evaluation
    }


def test_job_variant_name_contains_the_complete_artifact_sha256_digest():
    evaluation = _dispatch_evaluation(
        "kernels/copy.metal",
        contract_id="copy",
        entry_point="copy_float32",
    )

    plan = plan_dispatch_artifacts((evaluation,), source_units=("kernels/copy.metal",))

    artifact_digest = evaluation[0].artifact_id.split(":", 1)[1]
    assert len(artifact_digest) == 64
    assert plan.artifacts[0].variant_name == f"dispatch-{artifact_digest}"


def test_rejects_a_dispatch_variant_whose_source_is_not_a_scanned_unit():
    evaluation = _dispatch_evaluation(
        "kernels/copy.metal",
        contract_id="copy",
        entry_point="copy_float32",
    )

    with pytest.raises(DispatchArtifactPlanError) as exc_info:
        plan_dispatch_artifacts((evaluation,), source_units=("kernels/other.metal",))

    assert _error_code(exc_info.value) == "source-unmatched"
    assert exc_info.value.message
    assert exc_info.value.details["source"] == "kernels/copy.metal"


def test_rejects_duplicate_dispatch_variant_across_evaluations():
    evaluation = _dispatch_evaluation(
        "kernels/copy.metal",
        contract_id="copy",
        entry_point="copy_float32",
    )

    with pytest.raises(DispatchArtifactPlanError) as exc_info:
        plan_dispatch_artifacts(
            (evaluation, evaluation), source_units=("kernels/copy.metal",)
        )

    assert _error_code(exc_info.value) == "variant-duplicate"
    assert exc_info.value.details["variantId"] == evaluation[0].variant_id


def test_rejects_dispatch_variant_identity_collision():
    first = _dispatch_evaluation(
        "kernels/copy.metal",
        contract_id="copy",
        entry_point="copy_float32",
        element_counts=(64,),
    )
    second = _dispatch_evaluation(
        "kernels/copy.metal",
        contract_id="copy",
        entry_point="copy_float32",
        element_counts=(128,),
    )
    colliding = replace(second[0], variant_id=first[0].variant_id)
    colliding_evaluation = replace(second, variants=(colliding,))

    with pytest.raises(DispatchArtifactPlanError) as exc_info:
        plan_dispatch_artifacts(
            (first, colliding_evaluation),
            source_units=("kernels/copy.metal",),
        )

    assert _error_code(exc_info.value) == "variant-identity-collision"
    assert exc_info.value.details["variantId"] == first[0].variant_id


def test_rejects_conflicting_compile_contracts_for_one_artifact_identity():
    first = _dispatch_evaluation(
        "kernels/copy.metal",
        contract_id="copy",
        entry_point="copy_float32",
    )
    second = _dispatch_evaluation(
        "kernels/copy.metal",
        contract_id="copy-wide",
        entry_point="copy_float32",
        workgroup_size=(128, 1, 1),
    )
    conflicting = replace(second[0], artifact_id=first[0].artifact_id)
    conflicting_evaluation = replace(second, variants=(conflicting,))

    with pytest.raises(DispatchArtifactPlanError) as exc_info:
        plan_dispatch_artifacts(
            (first, conflicting_evaluation),
            source_units=("kernels/copy.metal",),
        )

    assert _error_code(exc_info.value) == "artifact-identity-collision"
    assert exc_info.value.details["artifactId"] == first[0].artifact_id


@pytest.mark.parametrize(
    "source_units",
    (
        ("",),
        ("/kernels/copy.metal",),
        ("kernels/../copy.metal",),
        ("kernels\\copy.metal",),
        ("./kernels/copy.metal",),
        ("C:/kernels/copy.metal",),
    ),
)
def test_rejects_non_normalized_repository_relative_source_units(source_units):
    evaluation = _dispatch_evaluation(
        "kernels/copy.metal",
        contract_id="copy",
        entry_point="copy_float32",
    )

    with pytest.raises(DispatchArtifactPlanError) as exc_info:
        plan_dispatch_artifacts((evaluation,), source_units=source_units)

    assert _error_code(exc_info.value) == "source-unit-invalid"
    assert exc_info.value.message
    assert "source" in exc_info.value.details


def test_plan_is_deterministic_across_evaluation_and_source_unit_order():
    copy_evaluation, reduce_evaluation = _plan_fixture()

    first = plan_dispatch_artifacts(
        (copy_evaluation, reduce_evaluation),
        source_units=("kernels/copy.metal", "kernels/reduce.metal"),
    )
    reversed_copy = replace(
        copy_evaluation, variants=tuple(reversed(copy_evaluation.variants))
    )
    second = plan_dispatch_artifacts(
        (reduce_evaluation, reversed_copy),
        source_units=("kernels/reduce.metal", "kernels/copy.metal"),
    )

    assert first == second
    assert first.to_json() == second.to_json()
    assert [job.artifact_id for job in first.artifacts] == sorted(
        job.artifact_id for job in first.artifacts
    )
    assert [variant.variant_id for variant in first.dispatch_variants] == sorted(
        variant.variant_id for variant in first.dispatch_variants
    )


def test_plan_json_has_stable_counts_and_preserves_dispatch_records():
    copy_evaluation, reduce_evaluation = _plan_fixture()
    plan = plan_dispatch_artifacts(
        (copy_evaluation, reduce_evaluation),
        source_units=("kernels/copy.metal", "kernels/reduce.metal"),
    )

    payload = plan.to_json()

    assert set(payload) == {
        "kind",
        "schemaVersion",
        "sourceUnitCount",
        "artifactCount",
        "dispatchVariantCount",
        "artifacts",
        "dispatchVariants",
    }
    assert payload["kind"] == DISPATCH_ARTIFACT_PLAN_KIND
    assert payload["schemaVersion"] == DISPATCH_ARTIFACT_PLAN_SCHEMA_VERSION
    assert payload["sourceUnitCount"] == 2
    assert payload["artifactCount"] == 2
    assert payload["dispatchVariantCount"] == 3
    assert payload["artifacts"] == [job.to_json() for job in plan.artifacts]
    assert payload["dispatchVariants"] == [
        variant.to_json() for variant in plan.dispatch_variants
    ]
    assert [record["artifactId"] for record in payload["artifacts"]] == sorted(
        record["artifactId"] for record in payload["artifacts"]
    )
    assert [record["variantId"] for record in payload["dispatchVariants"]] == sorted(
        record["variantId"] for record in payload["dispatchVariants"]
    )

"""Source-scoped artifact planning for evaluated host dispatch contracts."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .dispatch_contracts import (
    DispatchContractEvaluation,
    EvaluatedDispatchVariant,
)

DISPATCH_ARTIFACT_PLAN_KIND = "crosstl-dispatch-artifact-plan"
DISPATCH_ARTIFACT_PLAN_SCHEMA_VERSION = 1


class DispatchArtifactPlanError(ValueError):
    """A dispatch contract could not be mapped to finite project artifacts."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = copy.deepcopy(dict(details or {}))
        super().__init__(message)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            payload["details"] = copy.deepcopy(self.details)
        return payload


@dataclass(frozen=True)
class DispatchArtifactJob:
    """One compile artifact shared by equivalent dispatch variants."""

    artifact_id: str
    variant_name: str
    source: str
    entry_point: str
    workgroup_size: tuple[int, int, int]
    subgroup_width: int | None
    specialization_constants: Mapping[str, Any]
    dispatch_variant_ids: tuple[str, ...]
    manifest_content_identities: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifactId": self.artifact_id,
            "variant": self.variant_name,
            "source": self.source,
            "entryPoint": self.entry_point,
            "workgroupSize": list(self.workgroup_size),
            "specializationConstants": copy.deepcopy(
                dict(self.specialization_constants)
            ),
            "dispatchVariantIds": list(self.dispatch_variant_ids),
            "manifestContentIdentities": list(self.manifest_content_identities),
        }
        if self.subgroup_width is not None:
            payload["subgroupWidth"] = self.subgroup_width
        return payload


@dataclass(frozen=True)
class DispatchArtifactPlan:
    """A deterministic set of source-scoped compile jobs and dispatch records."""

    source_units: tuple[str, ...]
    artifacts: tuple[DispatchArtifactJob, ...]
    dispatch_variants: tuple[EvaluatedDispatchVariant, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": DISPATCH_ARTIFACT_PLAN_KIND,
            "schemaVersion": DISPATCH_ARTIFACT_PLAN_SCHEMA_VERSION,
            "sourceUnitCount": len(self.source_units),
            "artifactCount": len(self.artifacts),
            "dispatchVariantCount": len(self.dispatch_variants),
            "artifacts": [artifact.to_json() for artifact in self.artifacts],
            "dispatchVariants": [
                variant.to_json() for variant in self.dispatch_variants
            ],
        }


def plan_dispatch_artifacts(
    evaluations: Sequence[DispatchContractEvaluation],
    *,
    source_units: Iterable[str],
) -> DispatchArtifactPlan:
    """Plan compile artifacts without applying variants to unrelated sources."""

    if isinstance(evaluations, (str, bytes, bytearray)):
        raise TypeError("evaluations must be a sequence of contract evaluations")
    if isinstance(source_units, (str, bytes, bytearray)):
        raise TypeError("source_units must be an iterable of source paths")

    normalized_sources = tuple(
        sorted({_normalized_source_unit(value) for value in source_units})
    )
    available_sources = frozenset(normalized_sources)
    flattened: list[tuple[DispatchContractEvaluation, EvaluatedDispatchVariant]] = []
    for evaluation in evaluations:
        if not isinstance(evaluation, DispatchContractEvaluation):
            raise TypeError(
                "evaluations must contain DispatchContractEvaluation values"
            )
        flattened.extend((evaluation, variant) for variant in evaluation)

    flattened.sort(
        key=lambda item: (
            item[1].variant_id,
            str(item[0].manifest_content_identity),
        )
    )
    variants_by_id: dict[str, tuple[str, EvaluatedDispatchVariant]] = {}
    artifact_records: dict[str, dict[str, Any]] = {}

    for evaluation, variant in flattened:
        if variant.source not in available_sources:
            raise DispatchArtifactPlanError(
                "dispatch-source-unmatched",
                (
                    f"Dispatch variant {variant.variant_id!r} references source "
                    f"{variant.source!r}, which is not a discovered translation unit."
                ),
                details={
                    "variantId": variant.variant_id,
                    "source": variant.source,
                    "availableSources": list(normalized_sources),
                },
            )

        variant_payload = _canonical_json(_dispatch_variant_identity(variant))
        previous_variant = variants_by_id.get(variant.variant_id)
        if previous_variant is not None:
            code = (
                "dispatch-variant-duplicate"
                if previous_variant[0] == variant_payload
                else "dispatch-variant-identity-collision"
            )
            raise DispatchArtifactPlanError(
                code,
                "Dispatch variant identities must be globally unique.",
                details={
                    "variantId": variant.variant_id,
                    "first": previous_variant[1].to_json(),
                    "second": variant.to_json(),
                },
            )
        variants_by_id[variant.variant_id] = (variant_payload, variant)

        compile_contract = _compile_contract(variant)
        compile_payload = _canonical_json(compile_contract)
        manifest_identity = str(evaluation.manifest_content_identity)
        artifact_record = artifact_records.get(variant.artifact_id)
        if artifact_record is None:
            artifact_records[variant.artifact_id] = {
                "compilePayload": compile_payload,
                "contract": compile_contract,
                "variantIds": {variant.variant_id},
                "manifestContentIdentities": {manifest_identity},
            }
            continue
        if artifact_record["compilePayload"] != compile_payload:
            raise DispatchArtifactPlanError(
                "dispatch-artifact-identity-collision",
                "A dispatch artifact identity resolves to conflicting compile contracts.",
                details={
                    "artifactId": variant.artifact_id,
                    "first": copy.deepcopy(artifact_record["contract"]),
                    "second": copy.deepcopy(compile_contract),
                },
            )
        artifact_record["variantIds"].add(variant.variant_id)
        artifact_record["manifestContentIdentities"].add(manifest_identity)

    artifacts = tuple(
        _artifact_job(artifact_id, artifact_records[artifact_id])
        for artifact_id in sorted(artifact_records)
    )
    variants = tuple(
        variants_by_id[variant_id][1] for variant_id in sorted(variants_by_id)
    )
    return DispatchArtifactPlan(
        source_units=normalized_sources,
        artifacts=artifacts,
        dispatch_variants=variants,
    )


def _normalized_source_unit(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise DispatchArtifactPlanError(
            "dispatch-source-unit-invalid",
            "Source units must be normalized repository-relative POSIX paths.",
            details={"source": value},
        )
    parsed = PurePosixPath(value)
    if (
        value == "."
        or ":" in value
        or parsed.is_absolute()
        or ".." in parsed.parts
        or str(parsed) != value
    ):
        raise DispatchArtifactPlanError(
            "dispatch-source-unit-invalid",
            "Source units must be normalized repository-relative POSIX paths.",
            details={"source": value},
        )
    return value


def _compile_contract(variant: EvaluatedDispatchVariant) -> dict[str, Any]:
    return {
        "source": variant.source,
        "entryPoint": variant.entry_point,
        "workgroupSize": list(variant.workgroup_size),
        "subgroupWidth": variant.subgroup_width,
        "specializationConstants": copy.deepcopy(
            dict(variant.specialization_constants)
        ),
    }


def _dispatch_variant_identity(variant: EvaluatedDispatchVariant) -> dict[str, Any]:
    execution = _compile_contract(variant)
    execution["dispatch"] = {variant.dispatch_field: list(variant.dispatch_size)}
    return {
        "contractId": variant.contract_id,
        "inputs": copy.deepcopy(dict(variant.inputs)),
        "capabilities": copy.deepcopy(dict(variant.capabilities)),
        "execution": execution,
    }


def _artifact_job(
    artifact_id: str,
    record: Mapping[str, Any],
) -> DispatchArtifactJob:
    contract = record["contract"]
    return DispatchArtifactJob(
        artifact_id=artifact_id,
        variant_name=_artifact_variant_name(artifact_id),
        source=str(contract["source"]),
        entry_point=str(contract["entryPoint"]),
        workgroup_size=tuple(contract["workgroupSize"]),
        subgroup_width=contract["subgroupWidth"],
        specialization_constants=copy.deepcopy(
            dict(contract["specializationConstants"])
        ),
        dispatch_variant_ids=tuple(sorted(record["variantIds"])),
        manifest_content_identities=tuple(sorted(record["manifestContentIdentities"])),
    )


def _artifact_variant_name(artifact_id: str) -> str:
    algorithm, separator, digest = artifact_id.partition(":")
    if (
        separator != ":"
        or algorithm != "sha256"
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise DispatchArtifactPlanError(
            "dispatch-artifact-identity-invalid",
            "Dispatch artifact identities must use a lowercase SHA-256 digest.",
            details={"artifactId": artifact_id},
        )
    return f"dispatch-{digest}"


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise DispatchArtifactPlanError(
            "dispatch-plan-value-invalid",
            f"Dispatch planning data is not finite JSON: {exc}",
        ) from exc

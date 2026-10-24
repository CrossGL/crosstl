"""Bounded leading records shared by generated artifact contracts."""

from __future__ import annotations

FROZEN_SPECIALIZATION_PREFIX = "// crosstl-frozen-specializations: "
RESOURCE_STORAGE_PREFIX = "// crosstl-resource-storage: "
METADATA_PRECONDITION_PREFIX = "// crosstl-metadata-preconditions: "
IMMUTABLE_STORAGE_PREFIX = "// crosstl-immutable-storage: "
DISPATCH_REGION_PREFIX = "// crosstl-dispatch-region: "
COMBINED_SAMPLER_PREFIX = "// crosstl-combined-samplers: "
FIXED_SAMPLER_PREFIX = "// crosstl-fixed-samplers: "
ARTIFACT_CONTRACT_PREFIXES = (
    FROZEN_SPECIALIZATION_PREFIX,
    RESOURCE_STORAGE_PREFIX,
    METADATA_PRECONDITION_PREFIX,
    IMMUTABLE_STORAGE_PREFIX,
    DISPATCH_REGION_PREFIX,
    COMBINED_SAMPLER_PREFIX,
    FIXED_SAMPLER_PREFIX,
)
MAX_ARTIFACT_CONTRACT_HEADER_BYTES = 65536


def _split_contracts(source: str) -> tuple[dict[str, str], str]:
    headers = {}
    remaining = source
    for _ in ARTIFACT_CONTRACT_PREFIXES:
        prefix = next(
            (item for item in ARTIFACT_CONTRACT_PREFIXES if remaining.startswith(item)),
            None,
        )
        if prefix is None:
            break
        line, _, remaining = remaining.partition("\n")
        if prefix in headers:
            raise ValueError("Each artifact contract must be declared once.")
        if len(line.encode("utf-8")) > MAX_ARTIFACT_CONTRACT_HEADER_BYTES:
            raise ValueError("Artifact contract header exceeds the byte limit.")
        headers[prefix] = line
    for prefix in ARTIFACT_CONTRACT_PREFIXES:
        if source.count(prefix.rstrip()) != int(prefix in headers):
            raise ValueError(
                "Artifact contracts must be declared once in a contiguous block "
                "starting on the first line."
            )
    return headers, remaining


def artifact_contract_headers(source: str) -> dict[str, str]:
    """Locate known contracts without interpreting their individual schemas.

    Existing single-header artifacts remain valid. Multiple distinct records
    may appear in any order, but ordinary comments, whitespace and shader code
    end the block. Per-contract parsers still validate every JSON payload.
    """
    headers, _ = _split_contracts(source)
    return headers


def with_artifact_contract_header(source: str, header: str) -> str:
    """Add one record in canonical order without replacing existing contracts."""
    if not header:
        return source
    added, tail = _split_contracts(header)
    if len(added) != 1 or tail:
        raise ValueError("Expected one artifact contract header without source text.")
    headers, body = _split_contracts(source)
    if headers.keys() & added.keys():
        raise ValueError("Each artifact contract must be declared once.")
    headers.update(added)
    return (
        "".join(
            headers[prefix] + "\n"
            for prefix in ARTIFACT_CONTRACT_PREFIXES
            if prefix in headers
        )
        + body
    )

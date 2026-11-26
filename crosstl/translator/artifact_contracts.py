"""Bounded leading records shared by generated artifact contracts."""

from __future__ import annotations

import re

from .refusals import MetadataContractError

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


_GLSL_ES_VERSION = re.compile(r"#version[ \t]+\d+[ \t]+es[ \t]*(?:\r?\n|\Z)")


def _leading_version(source: str) -> tuple[str, str]:
    """Split off a first-line GLSL ES ``#version`` directive, which stays ahead of every record.

    GLSL ES admits nothing before ``#version``, not even a comment (GLSL ES 3.00 and
    3.10, section 3.4), so contract records follow that line there. Desktop GLSL allows
    comments before it (GLSL 4.60, section 3.3), so its records stay on the first line.
    """
    match = _GLSL_ES_VERSION.match(source)
    if match is None:
        return "", source
    return source[: match.end()], source[match.end() :]


def _split_contracts(source: str) -> tuple[dict[str, str], str]:
    headers = {}
    _, remaining = _leading_version(source)
    for _ in ARTIFACT_CONTRACT_PREFIXES:
        prefix = next(
            (item for item in ARTIFACT_CONTRACT_PREFIXES if remaining.startswith(item)),
            None,
        )
        if prefix is None:
            break
        line, _, remaining = remaining.partition("\n")
        if prefix in headers:
            raise MetadataContractError("Each artifact contract must be declared once.")
        if len(line.encode("utf-8")) > MAX_ARTIFACT_CONTRACT_HEADER_BYTES:
            raise MetadataContractError(
                "Artifact contract header exceeds the byte limit."
            )
        headers[prefix] = line
    for prefix in ARTIFACT_CONTRACT_PREFIXES:
        if source.count(prefix.rstrip()) != int(prefix in headers):
            raise MetadataContractError(
                "Artifact contracts must be declared once in a contiguous block "
                "starting on the first line (after a leading GLSL ES #version line)."
            )
    return headers, remaining


def artifact_contract_headers(source: str) -> dict[str, str]:
    """Locate known contracts without interpreting their individual schemas.

    Existing single-header artifacts remain valid. Multiple distinct records
    may appear in any order, but ordinary comments, whitespace and shader code
    end the block. The block starts on the first line, or on the second when the
    first is a GLSL ES ``#version`` directive. Per-contract parsers still validate every
    JSON payload.
    """
    headers, _ = _split_contracts(source)
    return headers


def with_artifact_contract_header(source: str, header: str) -> str:
    """Add one record in canonical order without replacing existing contracts."""
    if not header:
        return source
    added, tail = _split_contracts(header)
    if len(added) != 1 or tail or _leading_version(header)[0]:
        raise MetadataContractError(
            "Expected one artifact contract header without source text."
        )
    headers, body = _split_contracts(source)
    if headers.keys() & added.keys():
        raise MetadataContractError("Each artifact contract must be declared once.")
    headers.update(added)
    version, _ = _leading_version(source)
    return (
        version
        + "".join(
            headers[prefix] + "\n"
            for prefix in ARTIFACT_CONTRACT_PREFIXES
            if prefix in headers
        )
        + body
    )

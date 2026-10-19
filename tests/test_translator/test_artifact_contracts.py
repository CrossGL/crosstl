"""Leading artifact contracts compose without weakening their boundaries."""

import itertools

import pytest

from crosstl.formatter import CodeFormatter, ShaderLanguage
from crosstl.translator.artifact_contracts import (
    ARTIFACT_CONTRACT_PREFIXES,
    MAX_ARTIFACT_CONTRACT_HEADER_BYTES,
    artifact_contract_headers,
    with_artifact_contract_header,
)
from crosstl.translator.frozen_specializations import (
    frozen_specialization_header,
    parse_frozen_specializations,
)
from crosstl.translator.metadata_preconditions import (
    metadata_precondition_header,
    parse_metadata_precondition_header,
)
from crosstl.translator.resource_storage import (
    BINARY16_STORAGE,
    parse_resource_storage_header,
    resource_storage_header,
)

FROZEN = [dict(id=0, name="enabled", dtype="bool", value=True, frozen=True)]
STORAGE = {"values": BINARY16_STORAGE}
METADATA = [
    dict(
        sourceResource=dict(
            schemaVersion=1,
            backend="metal",
            entryPoint="copy_values",
            parameter="count",
            parameterIndex=0,
        ),
        dtype="uint32",
        minimum=1,
        maximum=4,
    )
]
HEADERS = (
    frozen_specialization_header(FROZEN),
    resource_storage_header(STORAGE),
    metadata_precondition_header(METADATA),
)
PARSERS = (
    parse_frozen_specializations,
    parse_resource_storage_header,
    parse_metadata_precondition_header,
)
RECORDS = (FROZEN, STORAGE, METADATA)
ORDERS = [
    order for size in (1, 2, 3) for order in itertools.permutations(range(3), size)
]
BODY = "// ordinary source comment\nvoid main() {}\n"


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_distinct_contracts_in_any_leading_order(order, newline):
    source = ("".join(HEADERS[index] for index in order) + BODY).replace("\n", newline)
    assert len(artifact_contract_headers(source)) == len(order)
    for index, parser in enumerate(PARSERS):
        assert parser(source) == (
            RECORDS[index] if index in order else type(RECORDS[index])()
        )


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_incremental_emission_uses_canonical_order(order):
    source = BODY
    for index in order:
        source = with_artifact_contract_header(source, HEADERS[index])
    assert source == "".join(HEADERS) + BODY
    assert with_artifact_contract_header(source, "") == source


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("prefix", ["\n", " ", "// comment\n", "/*\n", "void f() {}\n"])
def test_no_contract_can_be_moved_past_source(index, prefix):
    source = "".join(HEADERS[:index]) + prefix + "".join(HEADERS[index:]) + BODY
    for parser in PARSERS:
        with pytest.raises(ValueError, match="first line"):
            parser(source)


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("location", ["block", "body", "comment"])
def test_duplicates_rejected_even_outside_the_leading_block(index, location):
    extra = HEADERS[index]
    source = "".join(HEADERS)
    source += extra + BODY if location == "block" else BODY + extra
    if location == "comment":
        source = "".join(HEADERS) + BODY + "/*\n" + extra + "*/\n"
    for parser in PARSERS:
        with pytest.raises(ValueError, match="once"):
            parser(source)
    with pytest.raises(ValueError, match="once"):
        with_artifact_contract_header("".join(HEADERS) + BODY, extra)


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("fault", ["json", "version", "duplicate-key", "oversize"])
def test_combined_records_retain_individual_validation(index, fault):
    lines = list(HEADERS)
    if fault == "json":
        lines[index] = ARTIFACT_CONTRACT_PREFIXES[index] + "{\n"
    elif fault == "version":
        lines[index] = lines[index].replace(
            '"schemaVersion":1', '"schemaVersion":false', 1
        )
    elif fault == "duplicate-key":
        lines[index] = lines[index].replace(
            '"schemaVersion":1', '"schemaVersion":1,"schemaVersion":1', 1
        )
    else:
        lines[index] = (
            ARTIFACT_CONTRACT_PREFIXES[index]
            + " " * MAX_ARTIFACT_CONTRACT_HEADER_BYTES
            + "\n"
        )
    with pytest.raises(ValueError):
        PARSERS[index]("".join(lines) + BODY)


@pytest.mark.parametrize("header", [BODY, "".join(HEADERS), HEADERS[0] + BODY])
def test_composer_accepts_one_header_only(header):
    with pytest.raises(ValueError):
        with_artifact_contract_header(BODY, header)


@pytest.mark.parametrize("order", ORDERS)
def test_formatter_preserves_each_contract(order):
    source = "".join(HEADERS[index] for index in order) + BODY
    formatter = CodeFormatter()
    if not formatter.has_clang:
        pytest.skip("clang-format is not installed")
    formatted = formatter.format_code(source, ShaderLanguage.HLSL)
    assert artifact_contract_headers(formatted) == artifact_contract_headers(source)

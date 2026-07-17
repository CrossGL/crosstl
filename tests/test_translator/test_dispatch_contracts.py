import copy
import hashlib
import json

import pytest

import crosstl.project as project_api
from crosstl.project.dispatch_contracts import (
    DISPATCH_CONTRACT_KIND,
    DISPATCH_CONTRACT_SCHEMA_VERSION,
    DispatchContractError,
    evaluate_dispatch_contract,
    load_dispatch_contract,
    parse_dispatch_contract,
)


def _input(name, *, provenance=None):
    expression = {"input": name}
    if provenance is not None:
        expression["provenance"] = provenance
    return expression


def _capability(name, *, provenance=None):
    expression = {"capability": name}
    if provenance is not None:
        expression["provenance"] = provenance
    return expression


def _op(name, *args, provenance=None):
    expression = {"op": name, "args": list(args)}
    if provenance is not None:
        expression["provenance"] = provenance
    return expression


def _normalization_manifest():
    return {
        "kind": DISPATCH_CONTRACT_KIND,
        "schemaVersion": DISPATCH_CONTRACT_SCHEMA_VERSION,
        "provenance": {
            "repository": "ml-explore/mlx",
            "revision": "4367c73b60541ddd5a266ce4644fd93d20223b6e",
            "adapter": "normalization-host-contract",
        },
        "inputs": [
            {
                "name": "axisSize",
                "role": "shape",
                "type": "integer",
                "provenance": {"symbol": "axis_size"},
            },
            {"name": "batchSize", "role": "shape", "type": "integer"},
            {"name": "dtype", "role": "dtype", "type": "string"},
            {"name": "looped", "role": "feature", "type": "boolean"},
            {"name": "epsilon", "role": "scalar", "type": "number"},
        ],
        "workloads": [
            {
                "id": "layer-f16-looped",
                "values": {
                    "axisSize": 8192,
                    "batchSize": 2,
                    "dtype": "float16",
                    "looped": True,
                    "epsilon": 1e-5,
                },
                "provenance": {"testCase": "layer_norm_large_axis"},
            },
            {
                "id": "layer-f32-single-row",
                "values": {
                    "axisSize": 4099,
                    "batchSize": 4,
                    "dtype": "float32",
                    "looped": False,
                    "epsilon": 1e-5,
                },
                "provenance": {"testCase": "layer_norm_odd_axis"},
            },
        ],
        "capabilities": [
            {
                "name": "maxThreadsPerWorkgroup",
                "type": "integer",
                "provenance": {"api": "maxTotalThreadsPerThreadgroup"},
            },
            {"name": "simdWidth", "type": "integer"},
        ],
        "devices": [
            {
                "id": "wave32-device",
                "values": {
                    "maxThreadsPerWorkgroup": 1024,
                    "simdWidth": 32,
                },
                "provenance": {"source": "device-query-snapshot"},
            }
        ],
        "contracts": [
            {
                "id": "layer-normalization",
                "source": "mlx/backend/metal/kernels/layer_norm.metal",
                "provenance": {
                    "source": "mlx/backend/metal/normalization.cpp",
                    "line": 112,
                },
                "branches": [
                    {
                        "id": "f16-looped",
                        "when": _op(
                            "all",
                            _op("eq", _input("dtype"), "float16"),
                            _input("looped"),
                            provenance={"hostBranch": "large-axis"},
                        ),
                        "entryPoint": "layer_norm_looped_float16",
                        "workgroupSize": [
                            _op(
                                "min",
                                _capability(
                                    "maxThreadsPerWorkgroup",
                                    provenance={"deviceQuery": "max-threads"},
                                ),
                                _op("ceilDiv", _input("axisSize"), 8),
                            ),
                            1,
                            1,
                        ],
                        "subgroupWidth": _capability("simdWidth"),
                        "specializationConstants": {
                            "20": True,
                            "AXIS_SIZE": _input("axisSize"),
                            "EPSILON": _input("epsilon"),
                            "LOOPED": _input("looped"),
                        },
                        "dispatch": {"workgroupCount": [_input("batchSize"), 1, 1]},
                        "provenance": {"hostFunction": "layer_norm"},
                    },
                    {
                        "id": "f32-single-row",
                        "when": _op(
                            "all",
                            _op("eq", _input("dtype"), "float32"),
                            _op("not", _input("looped")),
                            provenance={"hostBranch": "single-row"},
                        ),
                        "entryPoint": "layer_norm_float32",
                        "workgroupSize": [
                            _op(
                                "min",
                                _capability("maxThreadsPerWorkgroup"),
                                _op("ceilDiv", _input("axisSize"), 8),
                            ),
                            1,
                            1,
                        ],
                        "subgroupWidth": _capability("simdWidth"),
                        "specializationConstants": {
                            "20": False,
                            "AXIS_SIZE": _input("axisSize"),
                            "EPSILON": _input("epsilon"),
                            "LOOPED": _input("looped"),
                        },
                        "dispatch": {"workgroupCount": [_input("batchSize"), 1, 1]},
                        "provenance": {"hostFunction": "layer_norm"},
                    },
                ],
            }
        ],
    }


def _image_manifest():
    rgba = _op("eq", _input("channels"), 4)
    return {
        "kind": DISPATCH_CONTRACT_KIND,
        "schemaVersion": DISPATCH_CONTRACT_SCHEMA_VERSION,
        "provenance": {"repository": "portable-image-fixture"},
        "inputs": [
            {"name": "width", "role": "shape", "type": "integer"},
            {"name": "height", "role": "shape", "type": "integer"},
            {"name": "channels", "role": "scalar", "type": "integer"},
            {"name": "dtype", "role": "dtype", "type": "string"},
            {
                "name": "boundsCheck",
                "role": "feature",
                "type": "boolean",
            },
        ],
        "workloads": [
            {
                "id": "rgba-hd",
                "values": {
                    "width": 1920,
                    "height": 1080,
                    "channels": 4,
                    "dtype": "float32",
                    "boundsCheck": True,
                },
            },
            {
                "id": "gray-vga",
                "values": {
                    "width": 640,
                    "height": 480,
                    "channels": 1,
                    "dtype": "float32",
                    "boundsCheck": False,
                },
            },
        ],
        "capabilities": [
            {"name": "maxWorkgroupX", "type": "integer"},
            {"name": "maxWorkgroupY", "type": "integer"},
        ],
        "devices": [
            {
                "id": "portable-compute-device",
                "values": {"maxWorkgroupX": 16, "maxWorkgroupY": 8},
            }
        ],
        "contracts": [
            {
                "id": "image-filter",
                "branches": [
                    {
                        "id": "float-image",
                        "when": _op(
                            "all",
                            _op("eq", _input("dtype"), "float32"),
                            _op(
                                "any",
                                rgba,
                                _op("eq", _input("channels"), 1),
                            ),
                        ),
                        "source": _op(
                            "select",
                            rgba,
                            "kernels/rgba_filter.comp",
                            "kernels/gray_filter.comp",
                        ),
                        "entryPoint": _op(
                            "select",
                            _input("boundsCheck"),
                            "filter_checked",
                            "filter_unchecked",
                        ),
                        "workgroupSize": [
                            _op("min", 16, _capability("maxWorkgroupX")),
                            _op("min", 8, _capability("maxWorkgroupY")),
                            1,
                        ],
                        "specializationConstants": {
                            "BOUNDS_CHECK": _input("boundsCheck"),
                            "CHANNELS": _input("channels"),
                        },
                        "dispatch": {
                            "workgroupCount": [
                                _op(
                                    "ceilDiv",
                                    _input("width"),
                                    _op("min", 16, _capability("maxWorkgroupX")),
                                ),
                                _op(
                                    "ceilDiv",
                                    _input("height"),
                                    _op("min", 8, _capability("maxWorkgroupY")),
                                ),
                                1,
                            ]
                        },
                    }
                ],
            }
        ],
    }


def _error_code(exc_info):
    return exc_info.value.code.removeprefix("project.dispatch-contract.")


def test_dispatch_contract_api_is_exported_from_project_package():
    assert project_api.parse_dispatch_contract is parse_dispatch_contract
    assert project_api.load_dispatch_contract is load_dispatch_contract
    assert project_api.evaluate_dispatch_contract is evaluate_dispatch_contract
    assert project_api.DispatchContractError is DispatchContractError


def test_loads_normalization_contract_with_sha256_identity_and_provenance(tmp_path):
    payload = _normalization_manifest()
    path = tmp_path / "normalization.dispatch.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    manifest = load_dispatch_contract(path)
    canonical = json.dumps(
        manifest.to_json(),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    assert manifest.source == str(path)
    assert manifest.content_identity.algorithm == "sha256"
    assert manifest.content_identity.to_json() == {
        "algorithm": "sha256",
        "value": manifest.content_identity.digest,
    }
    assert (
        manifest.content_identity.digest
        == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )
    assert manifest.provenance["revision"] == (
        "4367c73b60541ddd5a266ce4644fd93d20223b6e"
    )


def test_evaluates_mlx_like_normalization_variants_without_guessing_device_limits():
    result = evaluate_dispatch_contract(
        parse_dispatch_contract(_normalization_manifest(), source="normalization.json")
    )
    variants = {variant.workload_id: variant for variant in result}

    assert len(result) == 2
    assert variants["layer-f32-single-row"].entry_point == "layer_norm_float32"
    assert variants["layer-f32-single-row"].workgroup_size == (513, 1, 1)
    assert variants["layer-f32-single-row"].subgroup_width == 32
    assert variants["layer-f32-single-row"].dispatch_size == (4, 1, 1)
    assert variants["layer-f16-looped"].entry_point == "layer_norm_looped_float16"
    assert variants["layer-f16-looped"].workgroup_size == (1024, 1, 1)
    assert variants["layer-f16-looped"].dispatch_size == (2, 1, 1)
    assert variants["layer-f16-looped"].specialization_constants == {
        "20": True,
        "AXIS_SIZE": 8192,
        "EPSILON": 1e-5,
        "LOOPED": True,
    }
    assert all(variant.source.endswith("layer_norm.metal") for variant in result)
    assert all(variant.variant_id.startswith("sha256:") for variant in result)
    assert all(variant.artifact_id.startswith("sha256:") for variant in result)


def test_retains_manifest_record_and_expression_provenance():
    variant = next(
        variant
        for variant in parse_dispatch_contract(_normalization_manifest()).evaluate()
        if variant.workload_id == "layer-f16-looped"
    )
    provenance = variant.provenance
    traces = {record["path"]: record for record in provenance["expressions"]}
    predicate_path = "contracts[layer-normalization].branches[f16-looped].when"
    capability_path = (
        "contracts[layer-normalization].branches[f16-looped]"
        ".workgroupSize[0].args[0]"
    )

    assert provenance["manifest"]["record"]["repository"] == "ml-explore/mlx"
    assert provenance["inputDeclarations"]["axisSize"] == {"symbol": "axis_size"}
    assert provenance["workload"] == {"testCase": "layer_norm_large_axis"}
    assert provenance["device"] == {"source": "device-query-snapshot"}
    assert provenance["contract"]["line"] == 112
    assert provenance["branch"] == {"hostFunction": "layer_norm"}
    assert traces[predicate_path]["provenance"] == {"hostBranch": "large-axis"}
    assert traces[capability_path] == {
        "path": capability_path,
        "kind": "capability",
        "provenance": {"deviceQuery": "max-threads"},
        "result": 1024,
        "reference": "maxThreadsPerWorkgroup",
    }
    assert provenance["predicateResults"] == [
        {"branchId": "f16-looped", "matched": True},
        {"branchId": "f32-single-row", "matched": False},
    ]


def test_evaluates_ecosystem_neutral_image_contract_and_conditional_selection():
    result = parse_dispatch_contract(_image_manifest()).evaluate()
    variants = {variant.workload_id: variant for variant in result}

    assert variants["rgba-hd"].source == "kernels/rgba_filter.comp"
    assert variants["rgba-hd"].entry_point == "filter_checked"
    assert variants["rgba-hd"].workgroup_size == (16, 8, 1)
    assert variants["rgba-hd"].dispatch_size == (120, 135, 1)
    assert variants["rgba-hd"].specialization_constants == {
        "BOUNDS_CHECK": True,
        "CHANNELS": 4,
    }
    assert variants["gray-vga"].source == "kernels/gray_filter.comp"
    assert variants["gray-vga"].entry_point == "filter_unchecked"
    assert variants["gray-vga"].dispatch_size == (40, 60, 1)
    assert result.to_json()["variantCount"] == 2


def test_safe_expression_language_covers_declared_arithmetic_and_comparisons():
    payload = _image_manifest()
    payload["workloads"] = [payload["workloads"][0]]
    constants = payload["contracts"][0]["branches"][0]["specializationConstants"]
    constants.update(
        {
            "EXTENT_SUM": _op("add", _input("width"), _input("height")),
            "LAST_X": _op("subtract", _input("width"), 1),
            "PIXEL_COUNT": _op("multiply", _input("width"), _input("height")),
            "FULL_TILES_X": _op("floorDiv", _input("width"), 16),
            "MAX_EXTENT": _op("max", _input("width"), _input("height")),
            "NOT_RGB": _op("ne", _input("channels"), 3),
            "HEIGHT_LT_WIDTH": _op("lt", _input("height"), _input("width")),
            "HEIGHT_LE_WIDTH": _op("le", _input("height"), _input("width")),
            "WIDTH_GT_HEIGHT": _op("gt", _input("width"), _input("height")),
            "WIDTH_GE_HEIGHT": _op("ge", _input("width"), _input("height")),
            "LITERAL_TILE": {
                "literal": 16,
                "provenance": {"constant": "tile-width"},
            },
        }
    )

    variant = parse_dispatch_contract(payload).evaluate()[0]

    assert variant.specialization_constants == {
        "BOUNDS_CHECK": True,
        "CHANNELS": 4,
        "EXTENT_SUM": 3000,
        "FULL_TILES_X": 120,
        "HEIGHT_LE_WIDTH": True,
        "HEIGHT_LT_WIDTH": True,
        "LAST_X": 1919,
        "LITERAL_TILE": 16,
        "MAX_EXTENT": 1920,
        "NOT_RGB": True,
        "PIXEL_COUNT": 2073600,
        "WIDTH_GE_HEIGHT": True,
        "WIDTH_GT_HEIGHT": True,
    }


def test_boolean_operators_short_circuit_unselected_expressions():
    payload = _image_manifest()
    branch = payload["contracts"][0]["branches"][0]
    invalid_division = _op("eq", _op("ceilDiv", 1, 0), 1)
    branch["when"] = _op("all", False, invalid_division)

    with pytest.raises(DispatchContractError) as unmatched:
        parse_dispatch_contract(payload).evaluate()

    assert _error_code(unmatched) == "branch-unmatched"

    branch["when"] = _op("any", True, invalid_division)
    result = parse_dispatch_contract(payload).evaluate()
    assert len(result) == 2


def test_evaluation_rejects_manifest_mutation_after_identity_calculation():
    manifest = parse_dispatch_contract(_normalization_manifest())
    manifest.workloads[0].values["axisSize"] = 1

    with pytest.raises(DispatchContractError) as exc_info:
        manifest.evaluate()

    assert _error_code(exc_info) == "content-identity-mismatch"


def test_manifest_order_does_not_change_content_or_variant_identity():
    first_payload = _normalization_manifest()
    reordered_payload = copy.deepcopy(first_payload)
    for field in ("inputs", "workloads", "capabilities", "devices", "contracts"):
        reordered_payload[field].reverse()
    reordered_payload["contracts"][0]["branches"].reverse()

    first = parse_dispatch_contract(first_payload)
    reordered = parse_dispatch_contract(reordered_payload)
    first_result = first.evaluate()
    reordered_result = reordered.evaluate()

    assert first.content_identity == reordered.content_identity
    assert first.to_json() == reordered.to_json()
    assert [variant.variant_id for variant in first_result] == [
        variant.variant_id for variant in reordered_result
    ]
    assert [variant.to_json() for variant in first_result] == [
        variant.to_json() for variant in reordered_result
    ]


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda payload: payload.update({"unexpected": True}),
            "field-unknown",
        ),
        (
            lambda payload: payload.update({"schemaVersion": 2}),
            "schema-version-unsupported",
        ),
        (
            lambda payload: payload["inputs"].append(
                {"name": "dtype", "role": "dtype", "type": "string"}
            ),
            "declaration-duplicate",
        ),
        (
            lambda payload: payload["contracts"][0]["branches"][0].update(
                {"unknownOutput": 1}
            ),
            "field-unknown",
        ),
        (
            lambda payload: payload["contracts"][0]["branches"][0].update(
                {"source": "other.metal"}
            ),
            "field-conflict",
        ),
        (
            lambda payload: payload["contracts"][0]["branches"][0].update(
                {
                    "dispatch": {
                        "workgroupCount": [1, 1, 1],
                        "globalSize": [1, 1, 1],
                    }
                }
            ),
            "dispatch-geometry-conflict",
        ),
    ],
)
def test_rejects_malformed_or_conflicting_schema(mutate, expected_code):
    payload = _normalization_manifest()
    mutate(payload)

    with pytest.raises(DispatchContractError) as exc_info:
        parse_dispatch_contract(payload)

    assert _error_code(exc_info) == expected_code


@pytest.mark.parametrize(
    ("expression", "expected_code"),
    [
        ({"input": "undeclaredShape"}, "input-reference-unknown"),
        ({"capability": "unknownLimit"}, "capability-reference-unknown"),
        ({"op": "pow", "args": [2, 8]}, "operator-unknown"),
        ({"op": [], "args": []}, "operator-unknown"),
        ({"op": "ceilDiv", "args": [8]}, "operator-arity-invalid"),
        ({"python": "axisSize // 8"}, "expression-form-unknown"),
    ],
)
def test_rejects_unknown_references_operators_and_expression_forms(
    expression, expected_code
):
    payload = _normalization_manifest()
    payload["contracts"][0]["branches"][0]["workgroupSize"][0] = expression

    with pytest.raises(DispatchContractError) as exc_info:
        parse_dispatch_contract(payload)

    assert _error_code(exc_info) == expected_code


def test_rejects_missing_explicit_device_capability():
    payload = _normalization_manifest()
    del payload["devices"][0]["values"]["maxThreadsPerWorkgroup"]

    with pytest.raises(DispatchContractError) as exc_info:
        parse_dispatch_contract(payload)

    assert _error_code(exc_info) == "missing-capability"
    assert exc_info.value.details == {"missing": ["maxThreadsPerWorkgroup"]}


def test_rejects_unmatched_and_conflicting_branches():
    unmatched = _normalization_manifest()
    for branch in unmatched["contracts"][0]["branches"]:
        branch["when"] = False

    with pytest.raises(DispatchContractError) as unmatched_error:
        parse_dispatch_contract(unmatched).evaluate()

    assert _error_code(unmatched_error) == "branch-unmatched"

    conflicting = _normalization_manifest()
    conflicting["contracts"][0]["branches"][0]["when"] = True
    conflicting["contracts"][0]["branches"][1]["when"] = True

    with pytest.raises(DispatchContractError) as conflict_error:
        parse_dispatch_contract(conflicting).evaluate()

    assert _error_code(conflict_error) == "branch-conflict"
    assert conflict_error.value.details["matchingBranches"] == [
        "f16-looped",
        "f32-single-row",
    ]


@pytest.mark.parametrize("geometry", [0, -1, 1.5, True])
def test_rejects_non_integral_or_non_positive_geometry(geometry):
    payload = _image_manifest()
    payload["contracts"][0]["branches"][0]["workgroupSize"][0] = geometry

    with pytest.raises(DispatchContractError) as exc_info:
        parse_dispatch_contract(payload).evaluate()

    assert _error_code(exc_info) == "geometry-value-invalid"


def test_rejects_non_boolean_predicate_and_invalid_arithmetic():
    predicate = _image_manifest()
    predicate["contracts"][0]["branches"][0]["when"] = 1

    with pytest.raises(DispatchContractError) as predicate_error:
        parse_dispatch_contract(predicate).evaluate()

    assert _error_code(predicate_error) == "predicate-type-invalid"

    division = _image_manifest()
    division["contracts"][0]["branches"][0]["workgroupSize"][0] = _op(
        "ceilDiv", _input("width"), 0
    )

    with pytest.raises(DispatchContractError) as division_error:
        parse_dispatch_contract(division).evaluate()

    assert _error_code(division_error) == "division-invalid"


def test_rejects_variant_expansion_before_evaluating_expressions():
    payload = _image_manifest()
    payload["contracts"][0]["branches"][0]["workgroupSize"][0] = _op(
        "ceilDiv", _input("width"), 0
    )
    manifest = parse_dispatch_contract(payload)

    with pytest.raises(DispatchContractError) as exc_info:
        manifest.evaluate(max_variants=1)

    assert _error_code(exc_info) == "variant-expansion-limit-exceeded"
    assert exc_info.value.details == {"upperBound": 2, "limit": 1}


def test_rejects_duplicate_evaluated_variants_even_with_different_record_ids():
    payload = _image_manifest()
    duplicate = copy.deepcopy(payload["workloads"][0])
    duplicate["id"] = "rgba-hd-alias"
    payload["workloads"].append(duplicate)

    with pytest.raises(DispatchContractError) as exc_info:
        parse_dispatch_contract(payload).evaluate()

    assert _error_code(exc_info) == "variant-duplicate"
    assert {
        exc_info.value.details[side]["workloadId"] for side in ("first", "second")
    } == {
        "rgba-hd",
        "rgba-hd-alias",
    }


def test_rejects_unsafe_source_path_after_conditional_selection():
    payload = _image_manifest()
    payload["contracts"][0]["branches"][0]["source"] = _op(
        "select", _input("boundsCheck"), "../outside.comp", "safe.comp"
    )

    with pytest.raises(DispatchContractError) as exc_info:
        parse_dispatch_contract(payload).evaluate()

    assert _error_code(exc_info) == "source-path-invalid"


def test_loader_rejects_invalid_json_and_duplicate_object_keys(tmp_path):
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"schemaVersion": 1,', encoding="utf-8")

    with pytest.raises(DispatchContractError) as malformed_error:
        load_dispatch_contract(malformed)

    assert _error_code(malformed_error) == "json-invalid"

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"kind":"first","kind":"second","schemaVersion":1}',
        encoding="utf-8",
    )

    with pytest.raises(DispatchContractError) as duplicate_error:
        load_dispatch_contract(duplicate)

    assert _error_code(duplicate_error) == "duplicate-key"
    assert duplicate_error.value.details["key"] == "kind"

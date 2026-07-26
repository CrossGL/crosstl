import copy
import hashlib
import json

import pytest

from crosstl.project.native_deferred_compilation import (
    NATIVE_DEFERRED_COMPILATION_KIND,
    NATIVE_DEFERRED_COMPILATION_REQUEST_KIND,
    NATIVE_DEFERRED_COMPILATION_VERSION,
    NativeDeferredCompilationError,
    build_native_deferred_compilation_request,
    validate_native_deferred_compilation_request,
)
from crosstl.project.pipeline import encode_runtime_variant_key


def _hash(value):
    return {"algorithm": "sha256", "value": value * 64}


def _source(path="sources/copy.hlsl", source_format="HLSL source", digest="1"):
    return {
        "path": path,
        "format": source_format,
        "hash": _hash(digest),
        "sizeBytes": 128,
    }


def _target(backend="directx"):
    if backend == "directx":
        return {
            "backend": "directx",
            "profile": "cs_6_2",
            "stage": "compute",
            "entryPoint": "copy_values",
            "outputFormat": "DXIL binary",
        }
    return {
        "backend": "opengl",
        "profile": None,
        "stage": "compute",
        "entryPoint": "main",
        "outputFormat": "SPIR-V binary",
    }


def _variant(
    *,
    backend="directx",
    value=4,
    type_arguments=None,
    value_arguments=None,
    compile_defines=None,
    specialization_values=None,
    execution=None,
):
    target = _target(backend)
    type_arguments = type_arguments or {"T": "float"}
    value_arguments = value_arguments or {"N": value}
    compile_defines = compile_defines or {"USE_FAST_PATH": "1"}
    specialization_values = (
        specialization_values
        if specialization_values is not None
        else [{"id": 7, "name": "mode", "value": 1}]
    )
    execution = execution or {
        "workgroupSize": [32, 1, 1],
        "subgroupWidth": 32 if backend == "directx" else None,
    }
    key = encode_runtime_variant_key(
        "kernels/copy.metal",
        "copy_values",
        backend,
        target_profile=target["profile"],
        execution=execution,
        type_arguments=type_arguments,
        value_arguments=value_arguments,
        specialization_constants=specialization_values,
        defines=compile_defines,
    )
    return {
        "key": key,
        "typeArguments": type_arguments,
        "valueArguments": value_arguments,
        "compileDefines": compile_defines,
        "specializationValues": specialization_values,
        "execution": execution,
    }


def _descriptor():
    return {
        "path": "descriptors/copy.native-loader.json",
        "hash": _hash("d"),
        "sizeBytes": 512,
    }


def _request(
    *,
    backend="directx",
    source=None,
    includes=(),
    target=None,
    variant=None,
    descriptor=None,
):
    return build_native_deferred_compilation_request(
        source
        or _source(
            source_format="HLSL source" if backend == "directx" else "GLSL source",
            path=(
                "sources/copy.hlsl"
                if backend == "directx"
                else "sources/copy.comp.glsl"
            ),
        ),
        includes,
        target or _target(backend),
        variant or _variant(backend=backend),
        descriptor or _descriptor(),
    )


def _canonical_hash(payload):
    content = {key: value for key, value in payload.items() if key != "requestHash"}
    encoded = json.dumps(
        content,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_build_request_is_closed_versioned_and_self_verifying():
    request = _request(
        includes=[
            _source("includes/types.hlsl", digest="3"),
            _source("includes/math.hlsl", digest="2"),
        ]
    )

    assert request["schemaVersion"] == NATIVE_DEFERRED_COMPILATION_VERSION
    assert request["kind"] == NATIVE_DEFERRED_COMPILATION_KIND
    assert NATIVE_DEFERRED_COMPILATION_REQUEST_KIND == request["kind"]
    assert set(request) == {
        "schemaVersion",
        "kind",
        "source",
        "includes",
        "target",
        "variant",
        "expectedLoaderDescriptor",
        "requestHash",
    }
    assert [item["path"] for item in request["includes"]] == [
        "includes/math.hlsl",
        "includes/types.hlsl",
    ]
    assert request["requestHash"] == {
        "algorithm": "sha256",
        "value": _canonical_hash(request),
    }
    assert validate_native_deferred_compilation_request(request) == request
    assert (
        validate_native_deferred_compilation_request(json.loads(json.dumps(request)))
        == request
    )


def test_request_normalization_is_independent_of_map_and_include_order():
    first_types = {"T": "float", "Index": "uint"}
    second_types = {"Index": "uint", "T": "float"}
    first_values = {"N": 4, "Aligned": True}
    second_values = {"Aligned": True, "N": 4}
    first_defines = {"USE_FAST_PATH": "1", "TILE_COUNT": "4"}
    second_defines = {"TILE_COUNT": "4", "USE_FAST_PATH": "1"}
    first_specializations = [
        {"id": None, "name": "enabled", "value": True},
        {"id": 7, "name": "mode", "value": 1},
    ]
    second_specializations = list(reversed(first_specializations))

    first = _request(
        includes=[
            _source("includes/z.hlsl", digest="3"),
            _source("includes/a.hlsl", digest="2"),
        ],
        variant=_variant(
            type_arguments=first_types,
            value_arguments=first_values,
            compile_defines=first_defines,
            specialization_values=first_specializations,
        ),
    )
    second = _request(
        includes=[
            _source("includes/a.hlsl", digest="2"),
            _source("includes/z.hlsl", digest="3"),
        ],
        variant=_variant(
            type_arguments=second_types,
            value_arguments=second_values,
            compile_defines=second_defines,
            specialization_values=second_specializations,
        ),
    )

    assert first == second
    assert first["variant"]["specializationValues"] == [
        {"id": 7, "name": "mode", "value": 1},
        {"id": None, "name": "enabled", "value": True},
    ]


def test_distinct_bounded_values_produce_distinct_keys_and_request_hashes():
    first = _request(variant=_variant(value=4))
    second = _request(variant=_variant(value=8))

    assert first["variant"]["key"] != second["variant"]["key"]
    assert first["requestHash"] != second["requestHash"]


def test_opengl_glsl_to_spirv_request_is_supported():
    request = _request(
        backend="opengl",
        includes=[
            _source(
                "includes/types.glsl",
                source_format="GLSL source",
                digest="2",
            )
        ],
    )

    assert request["source"]["format"] == "GLSL source"
    assert request["target"] == {
        "backend": "opengl",
        "profile": None,
        "stage": "compute",
        "entryPoint": "main",
        "outputFormat": "SPIR-V binary",
    }
    assert request["variant"]["execution"]["subgroupWidth"] is None


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("root", "compilerArguments"),
        ("source", "compilerFlags"),
        ("target", "arguments"),
        ("variant", "compilerArguments"),
        ("execution", "dispatchCount"),
        ("descriptor", "compiler"),
    ],
)
def test_request_rejects_unknown_and_arbitrary_compiler_argument_fields(
    location, field
):
    request = _request()
    target = {
        "root": request,
        "source": request["source"],
        "target": request["target"],
        "variant": request["variant"],
        "execution": request["variant"]["execution"],
        "descriptor": request["expectedLoaderDescriptor"],
    }[location]
    target[field] = ["-unsafe-compiler-option"]

    with pytest.raises(NativeDeferredCompilationError) as raised:
        validate_native_deferred_compilation_request(request)

    assert raised.value.code.endswith("-invalid")
    assert field in raised.value.details["unsupportedFields"]


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/copy.hlsl",
        "../copy.hlsl",
        "sources/../copy.hlsl",
        "sources\\copy.hlsl",
        "C:/copy.hlsl",
        "sources//copy.hlsl",
        "sources/CON.hlsl",
        "sources/copy.hlsl.",
        "sources/co:py.hlsl",
    ],
)
def test_request_rejects_nonportable_paths(path):
    source = _source(path)

    with pytest.raises(
        NativeDeferredCompilationError,
        match="portable",
    ) as raised:
        _request(source=source)

    assert raised.value.code == "project.native-deferred-compilation.path-invalid"
    assert raised.value.path == "$.source.path"


@pytest.mark.parametrize(
    ("includes", "code"),
    [
        (
            [
                _source("includes/types.hlsl", digest="2"),
                _source("includes/types.hlsl", digest="3"),
            ],
            "include-path-duplicate",
        ),
        (
            [
                _source("includes/Types.hlsl", digest="2"),
                _source("includes/types.hlsl", digest="3"),
            ],
            "include-path-case-collision",
        ),
        (
            [_source("SOURCES/COPY.HLSL", digest="2")],
            "include-path-case-collision",
        ),
    ],
)
def test_request_rejects_duplicate_and_case_colliding_include_paths(includes, code):
    with pytest.raises(NativeDeferredCompilationError) as raised:
        _request(includes=includes)

    assert raised.value.code == f"project.native-deferred-compilation.{code}"


def test_request_rejects_descriptor_path_colliding_with_source_closure():
    descriptor = _descriptor()
    descriptor["path"] = "INCLUDES/TYPES.HLSL"

    with pytest.raises(NativeDeferredCompilationError) as raised:
        _request(
            includes=[_source("includes/types.hlsl", digest="2")],
            descriptor=descriptor,
        )

    assert raised.value.code.endswith("loader-descriptor-path-collision")


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda source: source.__setitem__(
                "hash", {"algorithm": "sha256", "value": "A" * 64}
            ),
            "source-hash-invalid",
        ),
        (
            lambda source: source.__setitem__(
                "hash", {"algorithm": "sha1", "value": "1" * 64}
            ),
            "source-hash-invalid",
        ),
        (
            lambda source: source.__setitem__("sizeBytes", True),
            "source-size-invalid",
        ),
        (
            lambda source: source.__setitem__("sizeBytes", 1 << 64),
            "source-size-invalid",
        ),
    ],
)
def test_request_rejects_invalid_source_identity(mutate, code):
    source = _source()
    mutate(source)

    with pytest.raises(NativeDeferredCompilationError) as raised:
        _request(source=source)

    assert raised.value.code == f"project.native-deferred-compilation.{code}"


@pytest.mark.parametrize(
    ("backend", "field", "value", "code"),
    [
        ("directx", "backend", "vulkan", "target-backend-unsupported"),
        ("directx", "profile", "vs_6_0", "target-profile-unsupported"),
        ("directx", "profile", "cs_6_99", "target-profile-unsupported"),
        ("directx", "stage", "fragment", "target-stage-unsupported"),
        (
            "directx",
            "outputFormat",
            "SPIR-V binary",
            "target-output-format-unsupported",
        ),
        ("opengl", "profile", "cs_6_0", "target-profile-unsupported"),
        ("opengl", "entryPoint", "copy_values", "target-entry-point-unsupported"),
        (
            "opengl",
            "outputFormat",
            "GLSL source",
            "target-output-format-unsupported",
        ),
    ],
)
def test_request_rejects_unsupported_target_contracts(backend, field, value, code):
    target = _target(backend)
    target[field] = value

    with pytest.raises(NativeDeferredCompilationError) as raised:
        _request(backend=backend, target=target)

    assert raised.value.code == f"project.native-deferred-compilation.{code}"


def test_request_rejects_source_format_target_mismatch():
    with pytest.raises(NativeDeferredCompilationError) as raised:
        _request(source=_source(source_format="GLSL source"))

    assert raised.value.code.endswith("source-format-target-mismatch")


def test_request_rejects_include_format_source_mismatch():
    with pytest.raises(NativeDeferredCompilationError) as raised:
        _request(
            includes=[
                _source(
                    "includes/types.hlsl",
                    source_format="GLSL source",
                    digest="2",
                )
            ]
        )

    assert raised.value.code.endswith("include-format-source-mismatch")
    assert raised.value.path == "$.includes[0].format"


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), [4]])
def test_request_rejects_unbounded_or_nonfinite_value_arguments(value):
    variant = _variant()
    variant["valueArguments"]["N"] = value

    with pytest.raises(NativeDeferredCompilationError) as raised:
        _request(variant=variant)

    assert raised.value.code.endswith("variant-value-arguments-invalid")
    assert raised.value.path == "$.variant.valueArguments.N"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("typeArguments", {"T": "@unresolved"}, "variant-type-arguments-invalid"),
        ("typeArguments", {"T": None}, "variant-type-arguments-invalid"),
        ("compileDefines", {"MODE": "@unresolved"}, "variant-compile-defines-invalid"),
        ("compileDefines", {"MODE": 1}, "variant-compile-defines-invalid"),
    ],
)
def test_request_rejects_unresolved_type_and_define_inputs(field, value, code):
    variant = _variant()
    variant[field] = value

    with pytest.raises(NativeDeferredCompilationError) as raised:
        _request(variant=variant)

    assert raised.value.code == f"project.native-deferred-compilation.{code}"


@pytest.mark.parametrize(
    "specializations",
    [
        [
            {"id": 7, "name": "first", "value": 1},
            {"id": 7, "name": "second", "value": 2},
        ],
        [
            {"id": 7, "name": "mode", "value": 1},
            {"id": 8, "name": "mode", "value": 2},
        ],
    ],
)
def test_request_rejects_duplicate_specialization_identities(specializations):
    variant = _variant()
    variant["specializationValues"] = specializations

    with pytest.raises(NativeDeferredCompilationError) as raised:
        _request(variant=variant)

    assert raised.value.code.endswith("variant-specialization-identity-duplicate")


@pytest.mark.parametrize("value", [None, float("-inf"), "@unresolved"])
def test_request_rejects_unresolved_specialization_values(value):
    variant = _variant()
    variant["specializationValues"][0]["value"] = value

    with pytest.raises(NativeDeferredCompilationError) as raised:
        _request(variant=variant)

    assert raised.value.code.endswith("variant-specialization-value-invalid")


@pytest.mark.parametrize(
    ("execution", "code"),
    [
        (
            {"workgroupSize": None, "subgroupWidth": None},
            "variant-workgroup-size-invalid",
        ),
        (
            {"workgroupSize": [32, 0, 1], "subgroupWidth": None},
            "variant-workgroup-size-invalid",
        ),
        (
            {"workgroupSize": [32, 1, 1], "subgroupWidth": 0},
            "variant-subgroup-width-invalid",
        ),
    ],
)
def test_request_rejects_unbounded_execution_identity(execution, code):
    variant = _variant()
    variant["execution"] = execution

    with pytest.raises(NativeDeferredCompilationError) as raised:
        _request(variant=variant)

    assert raised.value.code == f"project.native-deferred-compilation.{code}"


def test_request_rejects_noncanonical_and_mismatched_variant_keys():
    invalid = _variant()
    invalid["key"] = "not-a-runtime-variant-key"
    with pytest.raises(NativeDeferredCompilationError) as raised:
        _request(variant=invalid)
    assert raised.value.code.endswith("variant-key-invalid")

    mismatched = _variant()
    mismatched["valueArguments"]["N"] = 8
    with pytest.raises(NativeDeferredCompilationError) as raised:
        _request(variant=mismatched)
    assert raised.value.code.endswith("variant-key-binding-mismatch")
    assert raised.value.details["mismatchedFields"] == ["valueArguments"]


def test_request_rejects_unresolved_canonical_variant_key():
    variant = _variant()
    variant["key"] = encode_runtime_variant_key(
        "@unresolved",
        "copy_values",
        "directx",
        target_profile="cs_6_2",
        execution=variant["execution"],
        type_arguments=variant["typeArguments"],
        value_arguments=variant["valueArguments"],
        specialization_constants=variant["specializationValues"],
        defines=variant["compileDefines"],
    )

    with pytest.raises(NativeDeferredCompilationError) as raised:
        _request(variant=variant)

    assert raised.value.code.endswith("variant-key-unresolved")


def test_request_hash_detects_content_and_digest_drift():
    content_drift = _request()
    content_drift["source"]["sizeBytes"] += 1
    with pytest.raises(NativeDeferredCompilationError) as raised:
        validate_native_deferred_compilation_request(content_drift)
    assert raised.value.code.endswith("request-hash-mismatch")
    assert raised.value.path == "$.requestHash"

    digest_drift = _request()
    digest_drift["requestHash"]["value"] = "f" * 64
    with pytest.raises(NativeDeferredCompilationError) as raised:
        validate_native_deferred_compilation_request(digest_drift)
    assert raised.value.code.endswith("request-hash-mismatch")


def test_builder_snapshots_mutable_inputs():
    source = _source()
    includes = [_source("includes/types.hlsl", digest="2")]
    target = _target()
    variant = _variant()
    descriptor = _descriptor()

    request = build_native_deferred_compilation_request(
        source,
        includes,
        target,
        variant,
        descriptor,
    )
    source["path"] = "changed.hlsl"
    includes[0]["path"] = "changed-include.hlsl"
    target["entryPoint"] = "changed"
    variant["valueArguments"]["N"] = 99
    descriptor["path"] = "changed.json"

    assert request["source"]["path"] == "sources/copy.hlsl"
    assert request["includes"][0]["path"] == "includes/types.hlsl"
    assert request["target"]["entryPoint"] == "copy_values"
    assert request["variant"]["valueArguments"]["N"] == 4
    assert request["expectedLoaderDescriptor"]["path"] == (
        "descriptors/copy.native-loader.json"
    )


def test_error_diagnostic_is_stable_and_detached_from_caller_details():
    details = {"z": float("inf"), "a": {"values": [2, 1]}}
    error = NativeDeferredCompilationError(
        "contract-invalid",
        "Contract is invalid.",
        path="$.variant",
        details=details,
    )
    details["a"]["values"].append(0)

    assert error.code == ("project.native-deferred-compilation.contract-invalid")
    assert error.path == "$.variant"
    assert error.to_json() == {
        "severity": "error",
        "code": "project.native-deferred-compilation.contract-invalid",
        "message": "Contract is invalid.",
        "path": "$.variant",
        "details": {"a": {"values": [2, 1]}, "z": "inf"},
    }


def test_validation_does_not_mutate_caller_payload():
    request = _request(
        includes=[
            _source("includes/z.hlsl", digest="3"),
            _source("includes/a.hlsl", digest="2"),
        ]
    )
    reversed_payload = copy.deepcopy(request)
    reversed_payload["includes"].reverse()

    assert validate_native_deferred_compilation_request(reversed_payload) == request
    assert reversed_payload["includes"][0]["path"] == "includes/z.hlsl"

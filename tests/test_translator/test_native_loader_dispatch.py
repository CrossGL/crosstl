import copy
import hashlib

import pytest

from crosstl.project.native_loader_abi import (
    NATIVE_LOADER_ABI_KIND,
    NATIVE_LOADER_ABI_VERSION,
)
from crosstl.project.native_loader_dispatch import (
    NativeLoaderDispatchError,
    build_native_loader_dispatch_request,
)
from crosstl.project.runtime_verification import (
    RuntimeDispatchGeometry,
    RuntimeExecutionRequest,
    RuntimeValue,
    prepare_runtime_execution,
)


def _write_descriptor(tmp_path, target="directx", *, constants=None):
    extension = "hlsl" if target == "directx" else "comp"
    entry_point = "CSMain" if target == "directx" else "main"
    artifact_format = "HLSL source" if target == "directx" else "GLSL source"
    artifact_bytes = (
        b"[numthreads(64, 1, 1)] void CSMain() {}\n"
        if target == "directx"
        else b"#version 430\nlayout(local_size_x = 64) in;\nvoid main() {}\n"
    )
    package_path = f"artifacts/{target}/copy.{extension}"
    artifact_path = tmp_path / package_path
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(artifact_bytes)
    layout = {
        "elementType": "float32",
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
    }
    input_namespace = "srv" if target == "directx" else "storage-buffer"
    output_namespace = "uav" if target == "directx" else "storage-buffer"
    bindings = [
        {
            "name": "input_values",
            "kind": "buffer",
            "type": "float[]",
            "namespace": input_namespace,
            "coordinates": {"set": 0, "binding": 0},
            "access": "read",
            "scalarLayout": copy.deepcopy(layout),
            "provenance": {"parameter": 0},
        },
        {
            "name": "output_values",
            "kind": "buffer",
            "type": "float[]",
            "namespace": output_namespace,
            "coordinates": {"set": 0, "binding": 1},
            "access": "write",
            "scalarLayout": copy.deepcopy(layout),
            "provenance": {"parameter": 1},
        },
    ]
    if constants is None:
        constants = [
            {
                "id": 3,
                "name": "vector_width",
                "dtype": "uint32",
                "required": True,
                "value": 4,
                "provenance": {"source": "dispatch-contract"},
            }
        ]
    return {
        "schemaVersion": NATIVE_LOADER_ABI_VERSION,
        "kind": NATIVE_LOADER_ABI_KIND,
        "abiVersion": NATIVE_LOADER_ABI_VERSION,
        "unitId": f"copy:{target}",
        "target": target,
        "stage": "compute",
        "entryPoint": {
            "name": entry_point,
            "stage": "compute",
            "executionConfig": {"workgroupSize": [64, 1, 1]},
            "provenance": {"sourceName": "copy_float32"},
        },
        "artifact": {
            "packagePath": package_path,
            "format": artifact_format,
            "hash": {
                "algorithm": "sha256",
                "value": hashlib.sha256(artifact_bytes).hexdigest(),
            },
            "sizeBytes": len(artifact_bytes),
        },
        "source": {
            "path": "kernels/copy.metal",
            "artifactPath": f"out/{target}/copy.{extension}",
            "backend": "metal",
            "hash": None,
            "remap": None,
        },
        "bindings": bindings,
        "scalarLayout": {
            "constants": [],
            "bindings": [
                {"binding": binding["name"], "layout": copy.deepcopy(layout)}
                for binding in bindings
            ],
        },
        "specializationConstants": copy.deepcopy(constants),
        "provenance": {"pipeline": "metal-to-crossgl", "target": target},
    }


def _inputs():
    return {
        "input_values": {
            "dtype": "float32",
            "shape": [4],
            "values": [1.0, 2.0, 3.0, 4.0],
        }
    }


def _outputs():
    return {
        "output_values": {
            "dtype": "float32",
            "shape": [4],
        }
    }


def _build(tmp_path, target="directx", **overrides):
    descriptor = overrides.pop("descriptor", None) or _write_descriptor(
        tmp_path, target
    )
    return build_native_loader_dispatch_request(
        descriptor,
        tmp_path,
        overrides.pop("input_values", _inputs()),
        overrides.pop("output_values", _outputs()),
        overrides.pop("dispatch_geometry", (2, 1, 1)),
        overrides.pop("specialization_values", {}),
        expected_target=overrides.pop("expected_target", target),
        **overrides,
    )


@pytest.mark.parametrize(
    ("target", "entry_point", "namespaces"),
    [
        ("directx", "CSMain", ["srv", "uav"]),
        ("opengl", "main", ["storage-buffer", "storage-buffer"]),
    ],
)
def test_builds_preflighted_native_runtime_request(
    tmp_path, target, entry_point, namespaces
):
    request = _build(tmp_path, target)

    assert isinstance(request, RuntimeExecutionRequest)
    assert (
        request.artifact_path
        == (
            tmp_path
            / f"artifacts/{target}/copy.{'hlsl' if target == 'directx' else 'comp'}"
        ).resolve()
    )
    assert request.project_root == tmp_path.resolve()
    assert request.artifact["target"] == target
    assert request.fixture.entry_point == entry_point
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    assert prepare_runtime_execution(request).diagnostics == ()
    assert request.execution_plan.dispatch == request.adapter_contract.dispatch
    assert request.execution_plan.dispatch.entry_point == entry_point
    assert request.execution_plan.dispatch.workgroup_size == (64, 1, 1)
    assert request.execution_plan.dispatch.workgroup_count == (2, 1, 1)
    assert request.execution_plan.dispatch.global_size == (128, 1, 1)

    bindings = request.adapter_contract.resource_bindings
    assert [binding.name for binding in bindings] == [
        "input_values",
        "output_values",
    ]
    assert [binding.metadata["bindingNamespace"] for binding in bindings] == namespaces
    assert [(binding.set, binding.binding) for binding in bindings] == [(0, 0), (0, 1)]
    assert [binding.access for binding in bindings] == ["read", "write"]
    assert [binding.type_name for binding in bindings] == ["float[]", "float[]"]
    assert bindings[0].metadata["scalarLayout"]["elementType"] == "float32"
    assert [bound.source for bound in request.execution_plan.resource_bindings] == [
        "input",
        "expectedOutput",
    ]


def test_specialization_provenance_distinguishes_descriptor_default_and_explicit(
    tmp_path,
):
    descriptor = _write_descriptor(
        tmp_path,
        "opengl",
        constants=[
            {
                "id": 1,
                "name": "descriptor_value",
                "dtype": "uint32",
                "value": 2,
                "provenance": {"source": "reflection"},
            },
            {
                "id": 2,
                "name": "default_value",
                "dtype": "uint32",
                "default": 4,
                "provenance": {"source": "source-default"},
            },
            {
                "id": 3,
                "name": "explicit_value",
                "dtype": "uint32",
                "required": True,
                "default": 8,
                "provenance": {"source": "source-default"},
            },
        ],
    )

    request = _build(
        tmp_path,
        "opengl",
        descriptor=descriptor,
        specialization_values={3: 16},
    )

    constants = request.adapter_contract.specialization_constants
    assert [constant.constant_id for constant in constants] == [1, 2, 3]
    assert [constant.value_provenance["source"] for constant in constants] == [
        "descriptor",
        "default",
        "explicit",
    ]
    assert constants[0].value == 2
    assert constants[1].value is None
    assert constants[1].default == 4
    assert constants[2].value == 16
    assert constants[2].default == 8
    assert constants[2].overridden is True
    assert constants[2].value_provenance["descriptorProvenance"] == {
        "source": "source-default"
    }
    assert [bound.source for bound in request.execution_plan.constant_bindings] == [
        "value",
        "default",
        "value",
    ]


def test_wraps_native_loader_abi_validation_error(tmp_path):
    descriptor = _write_descriptor(tmp_path)
    del descriptor["kind"]

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, descriptor=descriptor)

    assert caught.value.code == "project.native-loader-dispatch.descriptor-invalid"
    assert caught.value.path == "$"
    assert caught.value.details["abiDiagnostic"]["code"].startswith(
        "project.native-loader-abi."
    )
    assert caught.value.to_json() == {
        "severity": "error",
        "code": caught.value.code,
        "message": "Native loader ABI descriptor validation failed.",
        "path": "$",
        "details": caught.value.details,
    }


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda descriptor: descriptor.update(target="vulkan"), "target-unsupported"),
        (
            lambda descriptor: descriptor["artifact"].update(format="DXIL bytecode"),
            "artifact-format-unsupported",
        ),
        (
            lambda descriptor: (
                descriptor.update(stage="fragment"),
                descriptor["entryPoint"].update(stage="fragment"),
            ),
            "stage-unsupported",
        ),
    ],
)
def test_rejects_unsupported_target_stage_and_artifact_combinations(
    tmp_path, mutation, code
):
    descriptor = _write_descriptor(tmp_path)
    mutation(descriptor)

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, descriptor=descriptor, expected_target=None)

    assert caught.value.code == f"project.native-loader-dispatch.{code}"


def test_rejects_expected_target_mismatch(tmp_path):
    descriptor = _write_descriptor(tmp_path, "directx")

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, descriptor=descriptor, expected_target="opengl")

    assert caught.value.code.endswith(".target-mismatch")
    assert caught.value.details == {
        "expectedTarget": "opengl",
        "actualTarget": "directx",
    }


@pytest.mark.parametrize(
    "package_path",
    [
        "/tmp/copy.hlsl",
        "../copy.hlsl",
        "C:/copy.hlsl",
        "artifacts//directx/copy.hlsl",
        "artifacts/./directx/copy.hlsl",
        "artifacts/directx/copy.hlsl/",
    ],
)
def test_rejects_unsafe_and_non_normalized_artifact_paths(tmp_path, package_path):
    descriptor = _write_descriptor(tmp_path)
    descriptor["artifact"]["packagePath"] = package_path

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, descriptor=descriptor)

    assert caught.value.code.endswith(".artifact-path-invalid")


def test_rejects_artifact_symlink_escape(tmp_path):
    descriptor = _write_descriptor(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.hlsl"
    outside.write_bytes(b"outside\n")
    artifact = tmp_path / descriptor["artifact"]["packagePath"]
    artifact.unlink()
    try:
        artifact.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, descriptor=descriptor)

    assert caught.value.code.endswith(".artifact-path-escape")


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("sizeBytes", 1, "artifact-size-mismatch"),
        ("hash", {"algorithm": "sha256", "value": "0" * 64}, "artifact-hash-mismatch"),
    ],
)
def test_attests_artifact_size_and_hash_before_building_request(
    tmp_path, field, value, code
):
    descriptor = _write_descriptor(tmp_path)
    descriptor["artifact"][field] = value

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, descriptor=descriptor)

    assert caught.value.code == f"project.native-loader-dispatch.{code}"


@pytest.mark.parametrize(
    ("mutation", "input_values", "code"),
    [
        (
            lambda binding: binding.update(kind="texture"),
            _inputs,
            "resource-unsupported",
        ),
        (
            lambda binding: binding.update(namespace="uav"),
            _inputs,
            "resource-namespace-unsupported",
        ),
        (
            lambda binding: binding["coordinates"].update(set=1),
            _inputs,
            "resource-set-unsupported",
        ),
        (
            lambda binding: binding.update(access=None),
            lambda: {},
            "resource-unsupported",
        ),
    ],
)
def test_rejects_unsupported_resource_contracts(tmp_path, mutation, input_values, code):
    descriptor = _write_descriptor(tmp_path)
    mutation(descriptor["bindings"][0])

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, descriptor=descriptor, input_values=input_values())

    assert caught.value.code == f"project.native-loader-dispatch.{code}"


@pytest.mark.parametrize(
    ("target", "namespace"),
    [("directx", "uav"), ("opengl", "storage-buffer")],
)
def test_accepts_read_write_bindings_as_output_only(tmp_path, target, namespace):
    descriptor = _write_descriptor(tmp_path, target)
    descriptor["bindings"][1]["access"] = "read_write"

    request = _build(tmp_path, target, descriptor=descriptor)

    binding = request.adapter_contract.resource_bindings[1]
    assert binding.name == "output_values"
    assert binding.access == "read_write"
    assert binding.metadata["bindingNamespace"] == namespace
    assert request.execution_plan.resource_bindings[1].source == "expectedOutput"


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_rejects_read_write_binding_supplied_only_as_input(tmp_path, target):
    descriptor = _write_descriptor(tmp_path, target)
    descriptor["bindings"][1]["access"] = "read_write"
    inputs = {
        **_inputs(),
        "output_values": {
            "dtype": "float32",
            "shape": [4],
            "values": [0.0, 0.0, 0.0, 0.0],
        },
    }

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(
            tmp_path,
            target,
            descriptor=descriptor,
            input_values=inputs,
            output_values={},
        )

    assert caught.value.code.endswith(".read-write-input-unsupported")
    assert caught.value.details == {"bindings": ["output_values"]}


def test_rejects_empty_resource_contract(tmp_path):
    descriptor = _write_descriptor(tmp_path, constants=[])
    descriptor["bindings"] = []
    descriptor["scalarLayout"]["bindings"] = []

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(
            tmp_path,
            descriptor=descriptor,
            input_values={},
            output_values={},
        )

    assert caught.value.code.endswith(".resource-unsupported")


def test_rejects_duplicate_runtime_value_names(tmp_path):
    duplicate = RuntimeValue(
        name="input_values",
        dtype="float32",
        shape=(4,),
        values=[1.0, 2.0, 3.0, 4.0],
    )

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, input_values=[duplicate, duplicate])

    assert caught.value.code.endswith(".value-duplicate")


@pytest.mark.parametrize(
    ("input_values", "output_values", "code"),
    [
        (_inputs, lambda: {}, "value-missing"),
        (
            lambda: {
                **_inputs(),
                "unknown": {"dtype": "float32", "shape": [1], "values": [0.0]},
            },
            _outputs,
            "value-extra",
        ),
        (
            lambda: {},
            lambda: {
                **_outputs(),
                "input_values": {"dtype": "float32", "shape": [4]},
            },
            "read-only-output",
        ),
    ],
)
def test_rejects_missing_extra_and_read_only_output_values(
    tmp_path, input_values, output_values, code
):
    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(
            tmp_path,
            input_values=input_values(),
            output_values=output_values(),
        )

    assert caught.value.code == f"project.native-loader-dispatch.{code}"


def test_rejects_input_output_name_overlap(tmp_path):
    outputs = _outputs()
    outputs["input_values"] = {"dtype": "float32", "shape": [4]}

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, output_values=outputs)

    assert caught.value.code.endswith(".value-duplicate")


def test_rejects_runtime_value_dtype_mismatched_with_scalar_layout(tmp_path):
    inputs = _inputs()
    inputs["input_values"]["dtype"] = "int32"
    inputs["input_values"]["values"] = [1, 2, 3, 4]

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, input_values=inputs)

    assert caught.value.code.endswith(".resource-layout-mismatch")
    assert caught.value.path == "$.bindings[0].scalarLayout"


def test_rejects_runtime_ambiguous_padded_scalar_layout(tmp_path):
    descriptor = _write_descriptor(tmp_path)
    descriptor["bindings"][0]["scalarLayout"]["elementStrideBytes"] = 8
    descriptor["scalarLayout"]["bindings"][0]["layout"]["elementStrideBytes"] = 8

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, descriptor=descriptor)

    assert caught.value.code.endswith(".resource-layout-unsupported")


def test_missing_scalar_layout_remains_fail_closed(tmp_path):
    descriptor = _write_descriptor(tmp_path)
    descriptor["bindings"][0]["scalarLayout"] = None
    descriptor["scalarLayout"]["bindings"] = descriptor["scalarLayout"]["bindings"][1:]

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, descriptor=descriptor)

    assert caught.value.code == (
        "project.native-loader-dispatch.resource-layout-unsupported"
    )
    assert caught.value.path == "$.bindings[0].scalarLayout"
    assert caught.value.details == {"binding": "input_values"}


@pytest.mark.parametrize(
    ("input_values", "output_values", "code"),
    [
        (
            lambda: {
                "input_values": {
                    "dtype": "float32",
                    "shape": [4],
                    "values": [1.0, 2.0],
                }
            },
            _outputs,
            "value-size-mismatch",
        ),
        (
            _inputs,
            lambda: {"output_values": {"dtype": "float32", "shape": [0]}},
            "value-shape-invalid",
        ),
    ],
)
def test_rejects_invalid_runtime_value_sizes_and_output_shapes(
    tmp_path, input_values, output_values, code
):
    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(
            tmp_path,
            input_values=input_values(),
            output_values=output_values(),
        )

    assert caught.value.code == f"project.native-loader-dispatch.{code}"


@pytest.mark.parametrize(
    ("value",),
    [
        ("not-a-number",),
        (float("inf"),),
        (float("1e40"),),
    ],
)
def test_rejects_runtime_values_incompatible_with_declared_dtype(tmp_path, value):
    inputs = {
        "input_values": {
            "dtype": "float32",
            "shape": [4],
            "values": [value, value, value, value],
        }
    }

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, input_values=inputs)

    assert caught.value.code.endswith(".value-data-invalid")


def test_rejects_unrepresentable_output_tolerance_with_structured_diagnostic(
    tmp_path,
):
    outputs = _outputs()
    outputs["output_values"]["tolerance"] = {"absolute": 10**10000}

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, output_values=outputs)

    assert caught.value.code.endswith(".value-tolerance-invalid")


def test_rejects_ambiguous_specialization_numeric_id(tmp_path):
    descriptor = _write_descriptor(tmp_path, "opengl")
    descriptor["specializationConstants"].append(
        {
            "id": 3,
            "name": "duplicate",
            "dtype": "uint32",
            "default": 4,
        }
    )

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, "opengl", descriptor=descriptor)

    assert caught.value.code.endswith(".specialization-id-ambiguous")


def test_rejects_ambiguous_specialization_runtime_name(tmp_path):
    descriptor = _write_descriptor(
        tmp_path,
        "opengl",
        constants=[
            {"id": 3, "name": "shared", "dtype": "uint32", "default": 4},
            {"id": 4, "name": "shared", "dtype": "uint32", "default": 8},
        ],
    )

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, "opengl", descriptor=descriptor)

    assert caught.value.code.endswith(".specialization-name-ambiguous")


@pytest.mark.parametrize(
    ("constants", "values", "code"),
    [
        (
            [{"id": 3, "name": "required", "dtype": "uint32", "required": True}],
            {},
            "specialization-value-missing",
        ),
        (
            [{"id": 3, "name": "known", "dtype": "uint32", "default": 4}],
            {9: 4},
            "specialization-value-extra",
        ),
        (
            [{"id": 3, "name": "known", "dtype": "uint32", "default": 4}],
            {"3": 4},
            "specialization-id-invalid",
        ),
    ],
)
def test_rejects_missing_extra_and_non_numeric_specialization_values(
    tmp_path, constants, values, code
):
    descriptor = _write_descriptor(tmp_path, "opengl", constants=constants)

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(
            tmp_path,
            "opengl",
            descriptor=descriptor,
            specialization_values=values,
        )

    assert caught.value.code == f"project.native-loader-dispatch.{code}"


def test_rejects_directx_specialization_override_not_materialized_in_source(tmp_path):
    descriptor = _write_descriptor(tmp_path, "directx")

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, descriptor=descriptor, specialization_values={3: 8})

    assert caught.value.code.endswith(".specialization-override-unsupported")


def test_rejects_unrepresentable_float_specialization_with_structured_diagnostic(
    tmp_path,
):
    descriptor = _write_descriptor(
        tmp_path,
        "opengl",
        constants=[
            {
                "id": 3,
                "name": "scale",
                "dtype": "float32",
                "value": 10**10000,
            }
        ],
    )

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, "opengl", descriptor=descriptor)

    assert caught.value.code.endswith(".specialization-value-invalid")


def test_rejects_directx_specialization_with_default_but_no_materialized_value(
    tmp_path,
):
    descriptor = _write_descriptor(
        tmp_path,
        "directx",
        constants=[
            {
                "id": 3,
                "name": "vector_width",
                "dtype": "uint32",
                "required": True,
                "default": 4,
            }
        ],
    )

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, descriptor=descriptor)

    assert caught.value.code.endswith(".specialization-unmaterialized")


@pytest.mark.parametrize(
    "field",
    ["workgroupSize", "workgroup_size", "numthreads", "localSize", "local_size"],
)
def test_accepts_reflected_vector_workgroup_size_spellings(tmp_path, field):
    descriptor = _write_descriptor(tmp_path, "directx")
    descriptor["entryPoint"]["executionConfig"] = {field: [32, 2, 1]}

    request = _build(tmp_path, descriptor=descriptor)

    assert request.execution_plan.dispatch.workgroup_size == (32, 2, 1)
    assert request.execution_plan.dispatch.global_size == (64, 2, 1)


def test_accepts_reflected_opengl_local_size_components(tmp_path):
    descriptor = _write_descriptor(tmp_path, "opengl")
    descriptor["entryPoint"]["executionConfig"] = {
        "local_size_x": 16,
        "local_size_y": 4,
        "local_size_z": 2,
    }

    request = _build(tmp_path, "opengl", descriptor=descriptor)

    assert request.execution_plan.dispatch.workgroup_size == (16, 4, 2)
    assert request.execution_plan.dispatch.global_size == (32, 4, 2)


def test_rejects_conflicting_reflected_workgroup_size_spellings(tmp_path):
    descriptor = _write_descriptor(tmp_path, "directx")
    descriptor["entryPoint"]["executionConfig"] = {
        "numthreads": [64, 1, 1],
        "local_size_x": 32,
        "local_size_y": 1,
        "local_size_z": 1,
    }

    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, descriptor=descriptor)

    assert caught.value.code.endswith(".workgroup-size-ambiguous")
    assert caught.value.path == "$.entryPoint.executionConfig"
    assert caught.value.details["firstField"] == "numthreads"
    assert caught.value.details["conflictingField"] == "local_size_x/y/z"


@pytest.mark.parametrize(
    ("dispatch", "code"),
    [
        ((), "dispatch-dimensions-invalid"),
        ((1, 0, 1), "dispatch-dimensions-invalid"),
        ((1, 1, 1, 1), "dispatch-dimensions-invalid"),
        (
            {"entryPoint": "Other", "workgroupCount": [1, 1, 1]},
            "entry-point-mismatch",
        ),
        (
            {"workgroupCount": [1, 1, 1], "workgroupSize": [32, 1, 1]},
            "workgroup-size-mismatch",
        ),
    ],
)
def test_rejects_invalid_dispatch_geometry(tmp_path, dispatch, code):
    with pytest.raises(NativeLoaderDispatchError) as caught:
        _build(tmp_path, dispatch_geometry=dispatch)

    assert caught.value.code == f"project.native-loader-dispatch.{code}"


def test_accepts_runtime_dispatch_geometry_and_uses_reflected_workgroup_size(
    tmp_path,
):
    request = _build(
        tmp_path,
        dispatch_geometry=RuntimeDispatchGeometry(
            entry_point="CSMain",
            workgroup_count=(3,),
        ),
    )

    dispatch = request.execution_plan.dispatch
    assert dispatch.workgroup_count == (3,)
    assert dispatch.workgroup_size == (64, 1, 1)
    assert dispatch.global_size == (192, 1, 1)
    assert dispatch.metadata["workgroupSizeSource"] == "reflection"


def test_rejects_nonportable_and_opengl_non_main_entry_points(tmp_path):
    descriptor = _write_descriptor(tmp_path)
    descriptor["entryPoint"]["name"] = "copy-kernel"

    with pytest.raises(NativeLoaderDispatchError) as invalid:
        _build(tmp_path, descriptor=descriptor)

    assert invalid.value.code.endswith(".entry-point-invalid")

    descriptor = _write_descriptor(tmp_path, "opengl")
    descriptor["entryPoint"]["name"] = "copy_kernel"
    with pytest.raises(NativeLoaderDispatchError) as unsupported:
        _build(tmp_path, "opengl", descriptor=descriptor)

    assert unsupported.value.code.endswith(".entry-point-unsupported")

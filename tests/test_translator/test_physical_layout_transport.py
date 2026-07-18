import copy

import pytest

import crosstl.project.pipeline as project_pipeline
import crosstl.project.runtime_verification as runtime_verification
from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.project.runtime_verification import RuntimeVerificationError

SCALAR_LAYOUT = {
    "physicalType": "uint",
    "elementType": "uint32",
    "elementSizeBytes": 4,
    "elementStrideBytes": 4,
    "storageLayout": "std430",
    "alignmentBytes": 4,
    "memberOffsetBytes": 0,
    "runtimeSized": True,
    "memberName": "values",
}


def test_runtime_manifest_binding_preserves_reflected_scalar_layout(tmp_path):
    artifact = tmp_path / "kernel.glsl"
    artifact.write_text(
        """#version 450
layout(std430, binding = 2) buffer Values { uint values[]; } outputValues;
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() { values[gl_GlobalInvocationID.x] = 1u; }
""",
        encoding="utf-8",
    )
    host_interface = reflect_target_host_interface(
        artifact, target="opengl", stage="compute"
    )
    original = copy.deepcopy(host_interface)

    bindings = project_pipeline._runtime_manifest_resource_bindings(host_interface)

    assert host_interface == original
    assert len(bindings) == 1
    assert bindings[0]["metadata"]["scalarLayout"] == SCALAR_LAYOUT
    parsed = runtime_verification._parse_runtime_resource_binding(
        bindings[0], field_name="artifact.resourceBindings[0]"
    )
    assert parsed.metadata["scalarLayout"] == SCALAR_LAYOUT

    bindings[0]["metadata"]["scalarLayout"]["elementSizeBytes"] = 8
    assert host_interface["resources"][0]["scalarLayout"]["elementSizeBytes"] == 4


def test_runtime_manifest_binding_does_not_infer_missing_scalar_layout():
    bindings = project_pipeline._runtime_manifest_resource_bindings(
        {
            "resources": [
                {
                    "name": "values",
                    "kind": "buffer",
                    "type": "Values",
                    "set": 0,
                    "binding": 2,
                    "access": "write",
                }
            ]
        }
    )

    assert "metadata" not in bindings[0]


def test_artifact_scalar_layout_survives_matching_adapter_override():
    artifact = {
        "resourceBindings": [
            {
                "id": "buffer|0|2|outputValues|0",
                "name": "outputValues",
                "kind": "buffer",
                "type": "Values",
                "set": 0,
                "binding": 2,
                "access": "write",
                "scalarLayout": SCALAR_LAYOUT,
                "metadata": {
                    "provenance": {"source": "hostInterface.resources"},
                    "status": "bound",
                },
            }
        ],
        "runtimeAdapter": {
            "resourceBindings": [
                {
                    "name": "outputValues",
                    "access": "read_write",
                    "value": "out",
                    "metadata": {"required": True},
                }
            ]
        },
    }

    contract = runtime_verification._runtime_adapter_contract_from_artifact(artifact)

    assert len(contract.resource_bindings) == 1
    binding = contract.resource_bindings[0]
    assert binding.binding_id == "buffer|0|2|outputValues|0"
    assert binding.name == "outputValues"
    assert binding.kind == "buffer"
    assert binding.type_name == "Values"
    assert binding.set == 0
    assert binding.binding == 2
    assert binding.access == "read_write"
    assert binding.value == "out"
    assert binding.metadata == {
        "scalarLayout": SCALAR_LAYOUT,
        "provenance": {"source": "hostInterface.resources"},
        "status": "bound",
        "required": True,
    }


@pytest.mark.parametrize(
    "binding",
    [
        {"name": "values", "scalarLayout": []},
        {"name": "values", "metadata": {"scalarLayout": "std430"}},
    ],
    ids=("top-level", "metadata"),
)
def test_runtime_binding_rejects_malformed_scalar_layout_before_dispatch(binding):
    with pytest.raises(
        RuntimeVerificationError, match="scalarLayout must be an object"
    ):
        runtime_verification._parse_runtime_resource_binding(
            binding, field_name="artifact.resourceBindings[0]"
        )

import json
import textwrap

from crosstl.project import (
    DISPATCH_CONTRACT_KIND,
    DISPATCH_CONTRACT_SCHEMA_VERSION,
    load_project_config,
    translate_project,
)


SECOND_ENTRY_SPECIALIZATION_SOURCE = textwrap.dedent("""
    shader EntryScopedSpecialization {
        bool second_only @function_constant(20);

        compute first {
            @numthreads(8, 1, 1)
            void main(uint index @gl_GlobalInvocationID) {
                uint first_value = index + 1u;
            }
        }

        compute second {
            @numthreads(8, 1, 1)
            void main(uint index @gl_GlobalInvocationID) {
                uint second_value = second_only ? index : 0u;
            }
        }
    }
    """).strip()


HELPER_SPECIALIZATION_SOURCE = textwrap.dedent("""
    shader HelperSpecialization {
        bool helper_enabled @function_constant(21);

        bool selected_enabled() {
            return helper_enabled;
        }

        compute first {
            @numthreads(8, 1, 1)
            void main(uint index @gl_GlobalInvocationID) {
                uint selected_value = selected_enabled() ? index : 0u;
            }
        }

        compute second {
            @numthreads(8, 1, 1)
            void main(uint index @gl_GlobalInvocationID) {
                uint second_value = index + 1u;
            }
        }
    }
    """).strip()


def _dispatch_manifest(entry_point, specialization_constants=None):
    return {
        "kind": DISPATCH_CONTRACT_KIND,
        "schemaVersion": DISPATCH_CONTRACT_SCHEMA_VERSION,
        "inputs": [{"name": "elementCount", "role": "shape", "type": "integer"}],
        "workloads": [{"id": "count-8", "values": {"elementCount": 8}}],
        "capabilities": [],
        "devices": [{"id": "portable", "values": {}}],
        "contracts": [
            {
                "id": f"{entry_point}-dispatch",
                "source": "kernels/multi.cgl",
                "entryPoint": entry_point,
                "branches": [
                    {
                        "id": "default",
                        "when": True,
                        "workgroupSize": [8, 1, 1],
                        "specializationConstants": specialization_constants or {},
                        "dispatch": {"workgroupCount": [1, 1, 1]},
                    }
                ],
            }
        ],
    }


def _translate_dispatch_project(
    tmp_path,
    *,
    source,
    entry_point,
    target,
    specialization_constants=None,
):
    root = tmp_path / "repo"
    kernels = root / "kernels"
    contracts = root / "contracts"
    kernels.mkdir(parents=True)
    contracts.mkdir()
    (kernels / "multi.cgl").write_text(source + "\n", encoding="utf-8")
    (contracts / "dispatch.json").write_text(
        json.dumps(
            _dispatch_manifest(entry_point, specialization_constants), indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "crosstl.toml").write_text(
        textwrap.dedent(f"""
            [project]
            include = ["kernels/multi.cgl"]
            targets = ["{target}"]
            output_dir = "generated"
            dispatch_contracts = ["contracts/dispatch.json"]
            """).strip()
        + "\n",
        encoding="utf-8",
    )

    report = translate_project(load_project_config(root), format_output=False)
    return root, report.to_json()


def test_directx_first_entry_omits_second_entry_required_constant(tmp_path):
    root, payload = _translate_dispatch_project(
        tmp_path,
        source=SECOND_ENTRY_SPECIALIZATION_SOURCE,
        entry_point="first",
        target="directx",
    )

    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    artifact = payload["artifacts"][0]
    assert artifact["status"] == "translated"
    assert artifact.get("specializationConstants", []) == []
    assert "specializationMaterialization" not in artifact

    generated = (root / artifact["path"]).read_text(encoding="utf-8")
    assert "second_only" not in generated
    assert "function_constant" not in generated


def test_directx_second_entry_requires_reachable_specialization_value(tmp_path):
    _root, payload = _translate_dispatch_project(
        tmp_path,
        source=SECOND_ENTRY_SPECIALIZATION_SOURCE,
        entry_point="second",
        target="directx",
    )

    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 1
    assert payload["artifacts"][0]["status"] == "failed"
    diagnostic = next(
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["code"]
        == "project.translate.specialization-value-required"
    )
    assert diagnostic["details"]["name"] == "second_only"
    assert diagnostic["details"]["id"] == 20


def test_directx_reachable_helper_materializes_required_specialization(tmp_path):
    root, payload = _translate_dispatch_project(
        tmp_path,
        source=HELPER_SPECIALIZATION_SOURCE,
        entry_point="first",
        target="directx",
        specialization_constants={"21": True},
    )

    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    artifact = payload["artifacts"][0]
    constants = artifact["specializationConstants"]
    assert len(constants) == 1
    assert constants[0]["name"] == "helper_enabled"
    assert constants[0]["id"] == 21
    assert constants[0]["required"] is True
    assert constants[0]["concreteValue"] is True
    assert constants[0]["valueProvenance"]["kind"] == "host-dispatch-contract"

    generated = (root / artifact["path"]).read_text(encoding="utf-8")
    assert "static const bool helper_enabled = true;" in generated
    assert "selected_enabled" in generated


def test_opengl_first_entry_omits_second_entry_specialization(tmp_path):
    root, payload = _translate_dispatch_project(
        tmp_path,
        source=SECOND_ENTRY_SPECIALIZATION_SOURCE,
        entry_point="first",
        target="opengl",
    )

    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    artifact = payload["artifacts"][0]
    assert artifact["status"] == "translated"
    assert artifact.get("specializationConstants", []) == []
    assert "specializationMaterialization" not in artifact

    generated = (root / artifact["path"]).read_text(encoding="utf-8")
    assert "second_only" not in generated
    assert "constant_id = 20" not in generated

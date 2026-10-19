import copy
import json

import pytest

from . import inventory
from .inventory import (
    CATEGORIES,
    LANGUAGES,
    REQUIRED_GROUPS,
    InventoryError,
    dumps,
    load,
    targets_for,
    validate,
)
from .inventory import glsl as glsl_inventory
from .inventory import spirv as spirv_inventory
from .inventory.__main__ import main as report_main


@pytest.fixture(scope="module")
def inventories():
    return inventory.load_all()


def minimal(language="glsl"):
    return {
        "schemaVersion": 1,
        "language": language,
        "title": "test",
        "sources": {
            "spec": {
                "title": "Spec",
                "url": "https://example.invalid/spec",
                "sections": {"1.2": "Things"},
            },
            "registry": {
                "title": "Registry",
                "url": "https://example.invalid/",
                "documents": {"EXT_x": "u"},
            },
        },
        "items": [
            {
                "id": f"{language}.{name}",
                "name": name,
                "category": category,
                "kind": "k",
                "cite": cite,
            }
            for name, category, cite in (
                ("builtin.f", "builtin-function", "spec §1.2"),
                ("extension.EXT_x", "extension", "registry EXT_x"),
                ("op.add", "operator", "spec §1.2"),
                ("qualifier.q", "qualifier", "spec §1.2"),
                ("resource.r", "resource", "spec §1.2"),
                ("stage.s", "stage", "spec §1.2"),
                ("type.t", "type", "spec §1.2"),
            )
        ],
        "cells": {},
    }


def test_every_language_has_a_valid_canonical_inventory(inventories):
    assert set(inventories) == set(LANGUAGES)
    for language, loaded in inventories.items():
        assert loaded.path.read_text() == dumps(
            loaded.data
        ), f"{language}.json is not canonical"
        assert len(loaded.items) > 100, language


def test_inventories_cover_every_required_group(inventories):
    for language, loaded in inventories.items():
        categories = {item["category"] for item in loaded.items}
        for group, members in REQUIRED_GROUPS.items():
            assert categories & members, f"{language} lacks {group}"


def test_every_cited_section_resolves(inventories):
    for loaded in inventories.values():
        assert validate(loaded.data) == []
        for item in loaded.items:
            source = item["cite"].split(" ", 1)[0]
            assert source in loaded.data["sources"], item["id"]


def test_cells_cover_targets_outside_the_source_family():
    assert targets_for("glsl") == ("metal", "hlsl", "spirv")
    assert targets_for("metal") == ("hlsl", "glsl", "essl", "spirv")
    assert targets_for("spirv") == ("metal", "hlsl", "glsl", "essl")


def test_counts_report_absent_cells_as_unknown(inventories):
    loaded = inventories["spirv"]
    counts = loaded.counts()
    assert set(counts) == set(targets_for("spirv"))
    for target_counts in counts.values():
        assert sum(target_counts.values()) == len(loaded.items)
    assert loaded.unknown() == sum(c.get("unknown", 0) for c in counts.values())


def test_minimal_inventory_is_valid():
    assert validate(minimal(), cases=set()) == []


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["items"].append(dict(d["items"][0])), "duplicate item id"),
        (lambda d: d["items"][0].update(cite="spec §9.9"), "section 9.9 is not listed"),
        (lambda d: d["items"][0].update(cite="nowhere §1.2"), "unknown source"),
        (
            lambda d: d["items"][1].update(cite="registry EXT_y"),
            "document EXT_y is not listed",
        ),
        (lambda d: d["items"][0].update(category="widget"), "category 'widget'"),
        (lambda d: d["items"][0].update(id="metal.builtin.f"), "must look like glsl."),
        (lambda d: d["items"].reverse(), "sorted by id"),
        (
            lambda d: d["items"][0].update(requires={"sm": "6.0"}),
            "'sm' is not defined for glsl",
        ),
        (
            lambda d: d["items"][0].update(requires={"profiles": ["glsl", "essl"]}),
            "sorted and unique",
        ),
        (
            lambda d: d["items"][0].update(requires={"glsl": "latest"}),
            "must be a version",
        ),
        (lambda d: d["items"][0].update(colour="red"), "unknown fields"),
        (
            lambda d: d.update(
                items=[i for i in d["items"] if i["category"] != "stage"]
            ),
            "'stages'",
        ),
        (lambda d: d.update(cells={"glsl.nothing": {}}), "unknown item"),
        (
            lambda d: d.update(
                cells={
                    "glsl.op.add": {
                        "essl": {"status": "preserved", "tests": ["case:x"]}
                    }
                }
            ),
            "essl is not a target of glsl",
        ),
        (
            lambda d: d.update(
                cells={
                    "glsl.op.add": {"metal": {"status": "maybe", "tests": ["case:x"]}}
                }
            ),
            "status must be one of",
        ),
        (
            lambda d: d.update(
                cells={"glsl.op.add": {"metal": {"status": "preserved", "tests": []}}}
            ),
            "needs at least one test",
        ),
        (
            lambda d: d.update(
                cells={
                    "glsl.op.add": {
                        "metal": {"status": "preserved", "tests": ["case:nope"]}
                    }
                }
            ),
            "does not exist",
        ),
        (
            lambda d: d.update(
                cells={
                    "glsl.op.add": {"metal": {"status": "refused", "tests": ["case:x"]}}
                }
            ),
            "must name its diagnostic code",
        ),
        (
            lambda d: d.update(
                cells={
                    "glsl.op.add": {
                        "metal": {
                            "status": "preserved",
                            "tests": ["case:x"],
                            "diagnostic": "E1",
                        }
                    }
                }
            ),
            "only refusals carry",
        ),
    ],
)
def test_schema_violations_are_reported(mutate, message):
    data = minimal()
    mutate(data)
    errors = validate(data, cases={"x"})
    assert any(message in error for error in errors), errors


def test_classified_cells_resolve_cases_and_pytest_functions():
    data = minimal()
    data["cells"] = {
        "glsl.op.add": {
            "metal": {"status": "preserved", "tests": ["case:x"]},
            "spirv": {
                "status": "refused",
                "diagnostic": "CGL0001",
                "tests": [
                    "pytest:tests/conformance/test_inventory.py::test_minimal_inventory_is_valid"
                ],
            },
            "hlsl": {
                "status": "polyfilled",
                "tests": ["pytest:tests/conformance/test_inventory.py::missing"],
            },
        }
    }
    errors = validate(data, cases={"x"})
    assert errors == [
        "glsl.op.add -> hlsl: test 'pytest:tests/conformance/test_inventory.py::missing' does not exist"
    ]


def test_pytest_references_cannot_escape_the_repository():
    data = minimal()
    data["cells"] = {
        "glsl.op.add": {
            "metal": {"status": "preserved", "tests": ["pytest:../outside.py::test_x"]}
        }
    }
    assert any("does not exist" in error for error in validate(data, cases=set()))


def test_load_reports_every_violation(tmp_path):
    data = minimal()
    data["items"][0]["category"] = "widget"
    data["items"][1]["cite"] = "registry EXT_y"
    path = tmp_path / "glsl.json"
    path.write_text(json.dumps(data))
    with pytest.raises(InventoryError) as raised:
        load("glsl", path)
    assert "widget" in str(raised.value) and "EXT_y" in str(raised.value)


def test_canonical_serialization_round_trips():
    data = minimal()
    data["cells"] = {
        "glsl.op.add": {"metal": {"status": "preserved", "tests": ["case:x"]}}
    }
    assert json.loads(dumps(data)) == data
    assert dumps(json.loads(dumps(data))) == dumps(data)


def test_report_cli(capsys):
    assert report_main([]) == 0
    out = capsys.readouterr().out
    assert "spirv:" in out and "-> metal:" in out
    assert report_main(["--require-complete"]) == 1


# SPIR-V derivation, on a synthetic grammar.

SYNTHETIC_VK = b"""<registry><spirvextensions><spirvextension name="SPV_KHR_ok"/></spirvextensions>
<spirvcapabilities><spirvcapability name="Shader"/><spirvcapability name="FooKHR"/></spirvcapabilities></registry>"""


def synthetic_grammar():
    core = {
        "major_version": 1,
        "minor_version": 6,
        "revision": 7,
        "instructions": [
            {
                "opname": "OpIAdd",
                "class": "Arithmetic",
                "opcode": 128,
                "operands": [],
                "version": "1.0",
            },
            {
                "opname": "OpEntryPoint",
                "class": "Mode-Setting",
                "opcode": 15,
                "operands": [{"kind": "ExecutionModel"}],
                "version": "1.0",
            },
            {
                "opname": "OpTypeVoid",
                "class": "Type-Declaration",
                "opcode": 19,
                "operands": [],
                "version": "1.0",
            },
            {
                "opname": "OpDecorate",
                "class": "Annotation",
                "opcode": 71,
                "operands": [],
                "version": "1.0",
            },
            {
                "opname": "OpVariable",
                "class": "Memory",
                "opcode": 59,
                "operands": [{"kind": "StorageClass"}],
                "version": "1.0",
            },
            {
                "opname": "OpEnqueueKernel",
                "class": "Device-Side_Enqueue",
                "opcode": 292,
                "operands": [{"kind": "KernelEnqueueFlags"}],
                "capabilities": ["DeviceEnqueue"],
                "version": "1.0",
            },
            {
                "opname": "OpFooKHR",
                "class": "@exclude",
                "opcode": 9000,
                "operands": [],
                "capabilities": ["FooKHR"],
                "version": "None",
            },
            {
                "opname": "OpBarNV",
                "class": "Reserved",
                "opcode": 9001,
                "operands": [],
                "extensions": ["SPV_NV_bar"],
                "version": "None",
            },
        ],
        "operand_kinds": [
            {
                "category": "ValueEnum",
                "kind": "Capability",
                "enumerants": [
                    {"enumerant": "Matrix", "value": 0},
                    {"enumerant": "Shader", "value": 1, "capabilities": ["Matrix"]},
                    {"enumerant": "DeviceEnqueue", "value": 2},
                    {"enumerant": "FooKHR", "value": 3, "extensions": ["SPV_KHR_ok"]},
                ],
            },
            {
                "category": "ValueEnum",
                "kind": "ExecutionModel",
                "enumerants": [
                    {"enumerant": "GLCompute", "value": 5, "capabilities": ["Shader"]},
                    {"enumerant": "Kernel", "value": 6, "capabilities": ["Kernel"]},
                ],
            },
            {
                "category": "ValueEnum",
                "kind": "StorageClass",
                "enumerants": [
                    {"enumerant": "Uniform", "value": 2, "capabilities": ["Shader"]},
                ],
            },
            {
                "category": "BitEnum",
                "kind": "KernelEnqueueFlags",
                "enumerants": [
                    {"enumerant": "None", "value": "0x0000"},
                    {"enumerant": "WaitKernel", "value": "0x0001"},
                ],
            },
            {"category": "Id", "kind": "IdRef"},
        ],
    }
    glsl450 = {
        "version": 100,
        "revision": 2,
        "instructions": [{"opname": "FMin", "opcode": 37, "operands": []}],
    }
    return core, glsl450


def synthetic_cells():
    return {
        "spirv.op.OpIAdd": {
            "metal": {"status": "preserved", "tests": ["case:s_add_wrap"]}
        }
    }


def test_spirv_derivation_scopes_items_to_the_vulkan_environment():
    core, glsl450 = synthetic_grammar()
    environment = spirv_inventory.Environment(SYNTHETIC_VK, core)
    data = spirv_inventory.derive(core, glsl450, environment, synthetic_cells())
    ids = [item["id"] for item in data["items"]]
    assert ids == sorted(ids)
    assert {
        "spirv.op.OpIAdd",
        "spirv.op.OpFooKHR",
        "spirv.Capability.Shader",
        "spirv.Capability.Matrix",  # implicitly declared by Shader
        "spirv.Capability.FooKHR",
        "spirv.ExecutionModel.GLCompute",
        "spirv.StorageClass.Uniform",
        "spirv.glsl-std-450.FMin",
        "spirv.extension.SPV_KHR_ok",
    } <= set(ids)
    for absent in (
        "OpEnqueueKernel",
        "OpBarNV",
        "Capability.DeviceEnqueue",
        "ExecutionModel.Kernel",
        "KernelEnqueueFlags.WaitKernel",
        "extension.SPV_NV_bar",
    ):
        assert not any(item_id.endswith(absent) for item_id in ids), absent
    by_id = {item["id"]: item for item in data["items"]}
    assert by_id["spirv.op.OpEntryPoint"]["category"] == "stage"
    assert by_id["spirv.op.OpVariable"]["category"] == "declaration"
    assert by_id["spirv.op.OpIAdd"]["cite"] == "spirv §3.3.13"
    assert by_id["spirv.op.OpFooKHR"]["cite"] == "spirv-registry SPV_KHR_ok"
    assert data["cells"] == synthetic_cells()
    assert data["generated"]["omittedOutsideVulkan"] >= 5
    assert validate(data, language="spirv", cases={"s_add_wrap"}) == []


def test_spirv_derivation_fails_closed_on_unmapped_grammar():
    core, glsl450 = synthetic_grammar()
    environment = spirv_inventory.Environment(SYNTHETIC_VK, core)
    broken = copy.deepcopy(core)
    broken["instructions"].append(
        {"opname": "OpNew", "class": "Brand-New", "opcode": 1, "operands": []}
    )
    with pytest.raises(InventoryError, match="Brand-New"):
        spirv_inventory.derive(broken, glsl450, environment)
    broken = copy.deepcopy(core)
    broken["operand_kinds"].append(
        {"category": "ValueEnum", "kind": "NewKind", "enumerants": []}
    )
    with pytest.raises(InventoryError, match="NewKind"):
        spirv_inventory.derive(broken, glsl450, environment)
    with pytest.raises(InventoryError, match="vanished"):
        spirv_inventory.derive(core, glsl450, environment, {"spirv.op.OpGone": {}})


def test_spirv_inputs_are_hash_checked(tmp_path):
    for name in spirv_inventory.INPUTS:
        (tmp_path / name).write_bytes(b"{}")
    with pytest.raises(InventoryError, match="not the pinned"):
        spirv_inventory.read_inputs(tmp_path)


# GLSL derivation, on synthetic AsciiDoc.


def test_glsl_conditionals_resolve_per_profile():
    text = "a\nifdef::GLSL[]\nb\nendif::GLSL[]\nifdef::ESSL[]\nc\nendif::ESSL[]\nifdef::ESSL[inline]\nifndef::GLSL[]\nd\nendif::GLSL[]"
    assert glsl_inventory.preprocess(text, "GLSL") == ["a", "b"]
    assert glsl_inventory.preprocess(text, "ESSL") == ["a", "c", "inline", "d"]
    with pytest.raises(InventoryError):
        glsl_inventory.preprocess("ifdef::GLSL[]\nx", "GLSL")


def test_glsl_sections_are_numbered_to_three_levels():
    lines = [
        "= Built-In Functions",
        "== Angle Functions",
        "=== Radians",
        "==== Detail",
        "=== Degrees",
        "----",
        "= not a heading",
        "----",
        "== Exponential Functions",
    ]
    chapter = glsl_inventory.Chapter(8, lines, "GLSL")
    assert chapter.sections == {
        "8": "Built-In Functions",
        "8.1": "Angle Functions",
        "8.1.1": "Radians",
        "8.1.2": "Degrees",
        "8.2": "Exponential Functions",
    }


def test_glsl_prototypes_keep_optional_and_array_parameters():
    lines = [
        "= Built-In Functions",
        "== Texture Functions",
        "[source,glsl]",
        "----",
        "gvec4 texture(gsampler2D sampler, vec2 P [, float bias])",
        "gvec4 textureGatherOffsets(gsampler2D sampler, vec2 P, ivec2 offsets[4] [, int comp])",
        "void umulExtended(highp genUType x,",
        "                  out highp genUType lsb)",
        "gvec4 imageLoad(readonly IMAGE_PARAMS)",
        "t = clamp(x, 0, 1);",
        "----",
    ]
    chapter = glsl_inventory.Chapter(8, lines, "ESSL")
    found = {entry[0]: entry[2] for entry in glsl_inventory.prototypes(chapter, set())}
    assert found == {
        "glsl.function.texture.gsampler2D-vec2-opt_float": (
            "gvec4 texture(gsampler2D, vec2, [float])"
        ),
        "glsl.function.textureGatherOffsets.gsampler2D-vec2-ivec2_array4-opt_int": (
            "gvec4 textureGatherOffsets(gsampler2D, vec2, ivec2[4], [int])"
        ),
        "glsl.function.umulExtended.genUType-genUType": (
            "void umulExtended(genUType, out genUType)"
        ),
        "glsl.function.imageLoad.IMAGE_PARAMS": (
            "gvec4 imageLoad(readonly IMAGE_PARAMS)"
        ),
    }


def test_glsl_inventory_spans_both_profiles(inventories):
    items = {item["id"]: item for item in inventories["glsl"].items}
    assert items["glsl.type.double"]["requires"]["profiles"] == ["glsl"]
    assert items["glsl.type.float"]["requires"]["profiles"] == ["essl", "glsl"]
    assert items["glsl.variable.gl_Position"]["requires"]["stages"] == [
        "geometry",
        "tessellation-control",
        "tessellation-evaluation",
        "vertex",
    ]
    assert items["glsl.variable.gl_GlobalInvocationID"]["requires"]["stages"] == [
        "compute"
    ]
    assert (
        "glsl.variable.gl_ModelViewMatrix" not in items
    )  # compatibility profile is out of scope
    assert (
        items["glsl.function.fma.genFType-genFType-genFType"]["category"]
        == "builtin-function"
    )
    assert items["glsl.layout.std430"]["category"] == "resource"
    assert {item["category"] for item in items.values()} <= set(CATEGORIES)

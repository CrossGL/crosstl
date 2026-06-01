"""Tests for the CrossTL translator language spec extractor."""

import json

from crosstl.translator import ast as ast_module
from crosstl.translator import language_spec
from crosstl.translator import lexer as lexer_module
from crosstl.translator import parser as parser_module
from crosstl.translator import stage_utils
from crosstl.translator import validation as validation_module


def test_language_spec_output_is_deterministic_json():
    first = language_spec.render_language_spec_json()
    second = language_spec.render_language_spec_json()

    assert first == second
    assert json.loads(first) == language_spec.extract_language_spec()


def test_language_spec_covers_frontend_basics():
    spec = language_spec.extract_language_spec()

    assert spec["kind"] == language_spec.SPEC_KIND
    assert spec["schemaVersion"] == 0
    assert spec["lexical"]["tokens"][0] == {
        "name": "COMMENT_SINGLE",
        "pattern": r"//.*",
    }
    assert {"spelling": "shader", "token": "SHADER"} in spec["lexical"]["keywords"]
    assert spec["lexical"]["skipTokens"] == [
        "COMMENT_MULTI",
        "COMMENT_SINGLE",
        "WHITESPACE",
    ]
    assert "compute" in spec["language"]["stages"]["canonical"]

    intrinsics = spec["language"]["intrinsics"]
    assert "WaveActiveSum" in intrinsics["wave"]
    assert "TraceRay" in intrinsics["rayTracing"]
    assert "DispatchMesh" in intrinsics["mesh"]
    assert "Proceed" in intrinsics["rayQueryMethods"]
    assert {
        "token": "TEXTURE2D",
        "spelling": "texture2d",
        "canonical": "texture2D",
    } in (spec["language"]["types"]["textures"])
    assert {
        "token": "SAMPLER2D",
        "canonical": "sampler2D",
        "kind": "sampler",
        "keywordSpellings": ["sampler2d"],
    } in spec["language"]["types"]["samplersAndImages"]

    ast_nodes = {node["name"]: node for node in spec["ast"]["classes"]}
    assert ast_nodes["ShaderNode"]["bases"] == ["ASTNode"]
    assert ast_nodes["ShaderStage"]["bases"] == ["Enum"]
    assert ast_nodes["ExecutionModel"]["bases"] == ["Enum"]
    assert {"name": "VERTEX", "value": "vertex"} in spec["ast"]["enums"]["ShaderStage"]

    validation_metadata = spec["validation"]["metadata"]
    assert "binding" in validation_metadata["singleValueNames"]
    assert {"spelling": "read", "canonical": "readonly"} in (
        spec["language"]["resources"]["resourceAccessMetadata"]
    )
    assert "image2d" in spec["language"]["resources"]["storageImageTypeNames"]
    assert (
        "rwstructuredbuffer"
        in spec["language"]["resources"]["uavResourceBufferTypeNames"]
    )
    assert "samplerstate" in spec["language"]["resources"]["samplerStateTypeNames"]


def test_language_spec_reports_ast_runtime_fields_and_aliases():
    spec = language_spec.extract_language_spec()
    class_fields = {node["class"]: node for node in spec["ast"]["classFields"]}

    primitive_type = class_fields["PrimitiveType"]
    primitive_parameters = {
        parameter["name"]: parameter
        for parameter in primitive_type["constructorParameters"]
    }
    assert primitive_parameters["name"] == {
        "name": "name",
        "kind": "positional-or-keyword",
        "annotation": "str",
        "required": True,
        "default": None,
        "optional": False,
    }
    assert primitive_parameters["size_bits"] == {
        "name": "size_bits",
        "kind": "positional-or-keyword",
        "annotation": "Optional[int]",
        "required": False,
        "default": "None",
        "optional": True,
    }
    assert primitive_parameters["kwargs"] == {
        "name": "kwargs",
        "kind": "var-keyword",
        "annotation": None,
        "required": False,
        "default": None,
        "optional": True,
    }
    primitive_fields = {field["name"]: field for field in primitive_type["fields"]}
    assert primitive_fields["size_bits"] == {
        "name": "size_bits",
        "source": "parameter",
        "parameter": "size_bits",
        "annotation": "Optional[int]",
        "required": False,
        "default": "None",
        "optional": True,
        "initializer": "size_bits",
    }

    variable_node = class_fields["VariableNode"]
    variable_fields = {field["name"]: field for field in variable_node["fields"]}
    assert "var_type" in {
        parameter["name"] for parameter in variable_node["constructorParameters"]
    }
    assert "vtype" in variable_fields
    assert "semantic" in variable_fields
    assert variable_fields["vtype"]["parameter"] == "var_type"

    assignment_fields = {
        field["name"]: field for field in class_fields["AssignmentNode"]["fields"]
    }
    assert {"left", "right"}.issubset(assignment_fields)
    assert assignment_fields["left"]["parameter"] == "target"
    assert assignment_fields["right"]["parameter"] == "value"

    function_call_fields = {
        field["name"]: field for field in class_fields["FunctionCallNode"]["fields"]
    }
    assert {"name", "args"}.issubset(function_call_fields)
    assert function_call_fields["name"]["parameter"] == "function"
    assert function_call_fields["args"]["parameter"] == "arguments"

    array_access_fields = {
        field["name"]: field for field in class_fields["ArrayAccessNode"]["fields"]
    }
    assert {"array", "index"}.issubset(array_access_fields)
    assert array_access_fields["array"]["parameter"] == "array_expr"
    assert array_access_fields["index"]["parameter"] == "index_expr"

    texture_resource_node = class_fields["TextureResourceNode"]
    texture_fields = {field["name"]: field for field in texture_resource_node["fields"]}
    assert "set_" in {
        parameter["name"]
        for parameter in texture_resource_node["constructorParameters"]
    }
    assert "set" in texture_fields
    assert texture_fields["set"]["parameter"] == "set_"

    array_node_fields = {
        field["name"]: field for field in class_fields["ArrayNode"]["fields"]
    }
    assert {
        "element_type",
        "size",
        "vtype",
    } == set(array_node_fields)
    assert array_node_fields["size"]["default"] == "None"
    assert array_node_fields["vtype"]["source"] == "derived"


def test_language_spec_reports_enum_classes_with_value_inventory():
    spec = language_spec.extract_language_spec()
    classes = {node["name"]: node for node in spec["ast"]["classes"]}
    class_fields = {node["class"]: node for node in spec["ast"]["classFields"]}

    for enum_name, values in spec["ast"]["enums"].items():
        assert classes[enum_name]["bases"] == ["Enum"]
        assert class_fields[enum_name] == {
            "class": enum_name,
            "constructorDefined": False,
            "constructorParameters": [],
            "fields": [],
        }
        assert values


def test_language_spec_uses_frontend_objects_as_authority(monkeypatch):
    monkeypatch.setitem(lexer_module.TOKENS, "SPEC_TEST_TOKEN", r"\bspec_test\b")
    monkeypatch.setitem(lexer_module.KEYWORDS, "spec_test", "SPEC_TEST_TOKEN")
    monkeypatch.setitem(stage_utils.STAGE_NAME_ALIASES, "pixel_spec", "fragment")
    monkeypatch.setattr(
        parser_module,
        "WAVE_INTRINSICS",
        set(parser_module.WAVE_INTRINSICS) | {"SpecWaveIntrinsic"},
    )
    monkeypatch.setattr(
        validation_module,
        "SINGLE_VALUE_METADATA_NAMES",
        frozenset(set(validation_module.SINGLE_VALUE_METADATA_NAMES) | {"spec_meta"}),
    )
    monkeypatch.setattr(
        validation_module,
        "STORAGE_IMAGE_TYPE_NAMES",
        frozenset(set(validation_module.STORAGE_IMAGE_TYPE_NAMES) | {"specimage"}),
    )

    class SpecExtractorTestNode(ast_module.ASTNode):
        def __init__(self, value, **kwargs):
            super().__init__(**kwargs)
            self.value = value

    SpecExtractorTestNode.__module__ = ast_module.__name__
    monkeypatch.setattr(
        ast_module, "SpecExtractorTestNode", SpecExtractorTestNode, raising=False
    )

    spec = language_spec.extract_language_spec()

    assert spec["lexical"]["tokens"][-1] == {
        "name": "SPEC_TEST_TOKEN",
        "pattern": r"\bspec_test\b",
    }
    assert {"spelling": "spec_test", "token": "SPEC_TEST_TOKEN"} in (
        spec["lexical"]["keywords"]
    )
    assert "SpecWaveIntrinsic" in spec["language"]["intrinsics"]["wave"]
    assert "spec_meta" in spec["validation"]["metadata"]["singleValueNames"]
    assert "specimage" in spec["language"]["resources"]["storageImageTypeNames"]
    assert "SpecExtractorTestNode" in {node["name"] for node in spec["ast"]["classes"]}


def test_language_spec_cli_prints_json(capsys):
    assert language_spec.main([]) == 0

    captured = capsys.readouterr()
    spec = json.loads(captured.out)

    assert spec["kind"] == language_spec.SPEC_KIND


def test_language_spec_cli_writes_output(tmp_path):
    output = tmp_path / "language-spec.json"

    assert language_spec.main(["--output", str(output)]) == 0

    assert json.loads(output.read_text(encoding="utf-8")) == (
        language_spec.extract_language_spec()
    )

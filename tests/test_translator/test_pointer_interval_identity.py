"""Source-owned offset facts survive target identifier renaming."""

import pytest

from crosstl.project import build_native_loader_dispatch_request, translate_project
from crosstl.translator.ast import IdentifierNode, UnaryOpNode
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from tests.test_translator.test_metadata_index_bounds import (
    CONDITIONS,
    GUARD,
    SOURCE,
    _index_config,
)
from tests.test_translator.test_pointer_index_bounds import (
    _operand_config,
    _replace_operand_body,
)


def _pointer_source(name, body):
    return (
        SOURCE.replace("result[0] = values[offset + index];", body)
        .replace(
            "kernel void",
            "uint read_at(const device uint* data, ulong at) { return data[at]; }\n"
            "kernel void",
        )
        .replace("values", name)
    )


def _pointer_request(descriptor, package, name, index, offset):
    values = [17 + position * 13 for position in range(16)]
    inputs, outputs, observed = {}, {}, []
    for binding in descriptor["bindings"]:
        origin = binding.get("provenance", {}).get("sourceResource")
        if origin is None:
            continue
        parameter = origin["parameter"]
        words = {
            name: values,
            "index": [index],
            "offset": [offset],
            "result": [GUARD] * 8,
        }[parameter]
        dtype = binding["scalarLayout"]["elementType"]
        inputs[binding["name"]] = dict(dtype=dtype, shape=[len(words)], values=words)
        if parameter == "result":
            expected = [GUARD] * 8
            expected[2] = values[index + offset]
            outputs[binding["name"]] = dict(dtype=dtype, shape=[8], values=expected)
        else:
            observed.append(binding["name"])
    return (
        build_native_loader_dispatch_request(
            descriptor,
            package,
            inputs,
            outputs,
            {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]},
            observe_inputs=observed,
        ),
        outputs,
    )


@pytest.mark.parametrize("name", ["output", "input", "output_", "destination"])
@pytest.mark.parametrize(
    "body",
    [
        "values += offset; result[0] = values[index];",
        "auto* shifted = values + offset; result[0] = shifted[index];",
        "values += offset; auto* shifted = values; result[0] = shifted[index];",
        "if (index > 0) { values += offset; } result[0] = values[index];",
        "values += offset; result[0] = read_at(values, index);",
        "result[0] = read_at(values + offset, index);",
        "values += offset; values += 1; result[0] = values[index];",
    ],
)
def test_pointer_bounds_survive_identifier_aliases(tmp_path, name, body):
    source = _pointer_source(name, body)
    report = translate_project(
        _index_config(tmp_path, source=source), format_output=False
    ).to_json()
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]


@pytest.mark.parametrize("other_value", [0, 4294967296])
@pytest.mark.parametrize("overflow", [False, True])
def test_emitted_alias_cannot_borrow_another_source_interval(
    tmp_path, other_value, overflow
):
    source = SOURCE.replace(
        "result[0] = values[offset + index];",
        f"ulong output_ = {other_value}ul; values += offset; "
        "result[0] = values[index];",
    ).replace("values", "output")
    conditions = (
        CONDITIONS[0],
        {
            **CONDITIONS[1],
            "minimum": 2147483647 if overflow else 0,
            "maximum": 2147483647 if overflow else 4,
        },
    )
    report = translate_project(
        _index_config(tmp_path, source=source, conditions=conditions),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == (0 if overflow else 1), report


@pytest.mark.parametrize(
    "body",
    [
        "values += offset; result[0] = values[index];",
        "values += offset; ++values; result[0] = values[0];",
        "values += offset; auto* local = values; result[0] = local[index];",
        "if (index > 0) { values += offset; } result[0] = values[index];",
        "values += offset; { ulong index = 4294967296ul; result[0] = values[index]; }",
    ],
)
@pytest.mark.parametrize("name", ["output", "input"])
def test_renamed_pointer_cannot_hide_overflow_or_shadowing(tmp_path, body, name):
    source = SOURCE.replace("result[0] = values[offset + index];", body).replace(
        "values", name
    )
    conditions = (
        CONDITIONS[0],
        {**CONDITIONS[1], "minimum": 2147483647, "maximum": 2147483647},
    )
    report = translate_project(
        _index_config(tmp_path, source=source, conditions=conditions),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report
    assert any(item["severity"] == "error" for item in report["diagnostics"])


def test_source_identifier_interval_is_not_an_emitted_name_lookup():
    generator = GLSLCodeGen()
    generator.current_identifier_aliases = {"output": "output_"}
    expression = IdentifierNode("output")
    assert generator.glsl_private_pointer_interval(
        expression, {"output": (8, 16), "output_": (0, 0)}, {}
    ) == (8, 16)
    assert (
        generator.glsl_private_pointer_interval(expression, {"output_": (0, 0)}, {})
        is None
    )


def test_generated_offset_bounds_belong_to_one_source_occurrence():
    generator = GLSLCodeGen()
    generator.local_variable_source_types["input_offset"] = "int"
    aliases = {
        "input": {"kind": "storage-pointer", "root": "values", "offset": "input_offset"}
    }
    expression = IdentifierNode("input")
    expression._glsl_control_flow_intervals = {
        "input": (2, 4),
        "input_offset": (99, 99),
    }
    binding = generator.glsl_storage_pointer_binding(expression, aliases)
    assert binding is not aliases["input"]
    offset = binding["offset"]
    assert isinstance(offset, IdentifierNode)
    assert offset.name == "input_offset"
    assert offset._glsl_control_flow_intervals == {"input_offset": (2, 4)}
    assert aliases["input"]["offset"] == "input_offset"
    later = IdentifierNode("input")
    later._glsl_control_flow_intervals = {"input_offset": (0, 0)}
    assert (
        generator.glsl_storage_pointer_binding(later, aliases)["offset"]
        == "input_offset"
    )
    later._glsl_control_flow_intervals = {"input": (8, 12)}
    assert generator.glsl_storage_pointer_binding(later, aliases)[
        "offset"
    ]._glsl_control_flow_intervals == {"input_offset": (8, 12)}
    assert offset._glsl_control_flow_intervals == {"input_offset": (2, 4)}


def test_addressed_pointer_view_renders_occurrence_bound_offset():
    generator = GLSLCodeGen()
    generator.local_variable_source_types["input_offset"] = "int"
    expression = IdentifierNode("input")
    expression._glsl_control_flow_intervals = {"input": (2, 4)}
    aliases = {
        "input": {"kind": "storage-pointer", "root": "values", "offset": "input_offset"}
    }
    binding = generator.glsl_storage_pointer_array_binding(
        UnaryOpNode("&", expression), aliases
    )
    assert binding["offset_expressions"] == ["input_offset"]
    assert generator.glsl_storage_pointer_offset_lvalue(
        binding["offset_expressions"][0]
    )
    assert binding["offset"]._glsl_control_flow_intervals == {"input_offset": (2, 4)}
    assert aliases["input"]["offset"] == "input_offset"


@pytest.mark.parametrize("offset", [0, "0", "offsets[0]", "base + 1"])
def test_offset_proof_does_not_reinterpret_arbitrary_generated_text(offset):
    generator = GLSLCodeGen()
    expression = IdentifierNode("input")
    expression._glsl_control_flow_intervals = {"input": (2, 4)}
    aliases = {"input": {"kind": "storage-pointer", "offset": offset}}
    assert (
        generator.glsl_storage_pointer_binding(expression, aliases)["offset"] == offset
    )


@pytest.mark.parametrize(
    "view",
    [
        {"byte_offset": "base_bytes"},
        {"pointer_reinterpretation": {"target_width": 4}},
        {"byte_offset": "base_bytes", "pointer_reinterpretation": {"target_width": 4}},
    ],
)
def test_reinterpreted_pointer_does_not_reuse_original_element_units(view):
    generator = GLSLCodeGen()
    expression = IdentifierNode("input")
    expression._glsl_control_flow_intervals = {"input": (2, 4)}
    aliases = {
        "input": {
            "kind": "storage-pointer",
            "offset": "input_offset",
            **view,
        }
    }
    assert (
        generator.glsl_storage_pointer_binding(expression, aliases)["offset"]
        == "input_offset"
    )


@pytest.mark.parametrize("bounds", [(-2147483649, 0), (0, 2147483648)])
def test_generated_offset_does_not_accept_out_of_domain_source_bounds(bounds):
    generator = GLSLCodeGen()
    expression = IdentifierNode("input")
    expression._glsl_control_flow_intervals = {"input": bounds}
    aliases = {"input": {"kind": "storage-pointer", "offset": "input_offset"}}
    assert (
        generator.glsl_storage_pointer_binding(expression, aliases)["offset"]
        == "input_offset"
    )


@pytest.mark.parametrize(
    "declarations,relative",
    [
        ("input += 2; input += 2147483647;", "1l"),
        ("input += 2; input -= 4;", "1l"),
        ("input += 2; if (input[0]) { input -= 4; }", "1l"),
        ("for (int i = 0; i < 3; ++i) input -= 2;", "1l"),
        ("input += 2; long input_offset = 0;", "2147483647l"),
        (
            "input += 2; long input_offset = 1; { long input_offset = -4; input += input_offset; }",
            "1l",
        ),
        ("input += 2; thread const device uint*& alias = input; alias -= 4;", "1l"),
    ],
)
def test_generated_base_cannot_hide_mutation_or_shadowing(
    tmp_path, declarations, relative
):
    config = _operand_config(
        tmp_path,
        "opengl",
        "uint8",
        source=_replace_operand_body(declarations, relative),
    )
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 0, payload
    assert any(item["severity"] == "error" for item in payload["diagnostics"])

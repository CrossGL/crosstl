"""Checked source metadata supplies conditional OpenGL index range proofs."""

import copy
from dataclasses import replace

import pytest

from crosstl._crosstl import translate
from crosstl.project import build_native_loader_dispatch_request, translate_project
from crosstl.project.native_loader_dispatch import NativeLoaderDispatchError
from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen, OpenGLIndexTypeError
from crosstl.translator.metadata_preconditions import (
    MetadataRangePrecondition,
    parse_metadata_precondition_header,
    resolve_metadata_preconditions,
)
from tests.test_translator.test_project_metadata_entry_scope import _config, _packages

SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void read_value(const device uint* values [[buffer(0)]],
                       constant ulong& index [[buffer(1)]],
                       constant ulong& offset [[buffer(2)]],
                       device uint* result [[buffer(3)]]) {
    result[0] = values[offset + index];
}
"""
CONDITIONS = (
    dict(entryPoint="read_value", parameter="index", minimum=0, maximum=3),
    dict(entryPoint="read_value", parameter="offset", minimum=0, maximum=4),
)
GUARD = 0xABCDEF01


def _index_config(root, *, source=SOURCE, conditions=CONDITIONS, target="opengl"):
    return replace(
        _config(
            root,
            target,
            source=source,
            conditions=conditions,
            entries=("read_value",),
        ),
        workgroup_size=(1, 1, 1),
    )


def _index_request(descriptor, package, index, offset, *, expected_words=None):
    values = [17 + position * 13 for position in range(16)]
    inputs, outputs, observed = {}, {}, []
    for binding in descriptor["bindings"]:
        origin = binding.get("provenance", {}).get("sourceResource")
        if origin is None:
            continue
        name, parameter = binding["name"], origin["parameter"]
        words = {
            "values": values,
            "index": [index],
            "offset": [offset],
            "result": [GUARD] * 8,
        }[parameter]
        dtype = binding["scalarLayout"]["elementType"]
        inputs[name] = dict(dtype=dtype, shape=[len(words)], values=words)
        if parameter == "result":
            expected = [GUARD] * 8
            expected[2] = (
                values[index + offset] if index + offset < len(values) else GUARD
            )
            if expected_words is not None:
                assert len(expected_words) == 8
                expected = list(expected_words)
            outputs[name] = dict(dtype=dtype, shape=[8], values=expected)
        else:
            observed.append(name)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]},
        observe_inputs=observed,
    )
    return request, outputs


@pytest.mark.parametrize(
    "body",
    [
        "result[0] = values[index];",
        "result[0] = values[offset + index];",
        "ulong position = offset + index; result[0] = values[position];",
        "result[0] = values[index * offset + 1];",
        "if (index > 0) { result[0] = values[index - 1]; }",
        "for (uint j = 0; j < index; ++j) { result[0] += values[offset + j]; }",
        "values += offset; result[0] = values[index];",
    ],
)
def test_checked_metadata_composes_with_source_index_proofs(tmp_path, body):
    source = SOURCE.replace("result[0] = values[offset + index];", body)
    _, package, (descriptor,) = _packages(_index_config(tmp_path, source=source))
    artifact = (package / descriptor["artifact"]["packagePath"]).read_text()
    records = parse_metadata_precondition_header(artifact)
    assert len(records) == 2
    assert {item["sourceResource"]["parameter"] for item in records} == {
        "index",
        "offset",
    }
    if "values +=" in body:
        assert "int(index)" in artifact
    else:
        assert "uint(" in artifact


@pytest.mark.parametrize(
    "source,conditions",
    [
        (SOURCE, ()),
        (SOURCE, CONDITIONS[:1]),
        (SOURCE, (CONDITIONS[0], {**CONDITIONS[1], "maximum": 4294967295})),
        (SOURCE, (CONDITIONS[0], {**CONDITIONS[1], "maximum": 18446744073709551615})),
        (SOURCE.replace("offset + index", "index - 1"), CONDITIONS),
        (
            SOURCE.replace("constant ulong& index", "constant volatile ulong& index"),
            CONDITIONS,
        ),
        (
            SOURCE.replace(
                "    result[0]", "    ulong index = 4294967296ul;\n    result[0]"
            ),
            CONDITIONS,
        ),
    ],
)
def test_metadata_does_not_authorize_unproved_or_overflowed_indices(
    tmp_path, source, conditions
):
    report = translate_project(
        _index_config(tmp_path, source=source, conditions=conditions),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report
    assert any(item["severity"] == "error" for item in report["diagnostics"])


@pytest.mark.parametrize(
    "conditions", [(), (CONDITIONS[0], {**CONDITIONS[1], "maximum": 4294967296})]
)
def test_wide_pointer_rebase_does_not_bypass_index_validation(tmp_path, conditions):
    source = SOURCE.replace(
        "result[0] = values[offset + index];",
        "values += offset; result[0] = values[0];",
    )
    report = translate_project(
        _index_config(tmp_path, source=source, conditions=conditions),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert (
        report["diagnostics"][0]["details"]["indexConversion"]["reason"]
        == "pointer-offset-range-unproven"
    )


@pytest.mark.parametrize("index_type", ["ulong", "uint"])
@pytest.mark.parametrize(
    "body",
    [
        "values += offset; result[0] = values[index];",
        "auto* shifted = values + offset; result[0] = shifted[index];",
        "if (index == 3) { values += offset; } result[0] = values[index];",
        "result[0] = read_at(values + offset, index);",
        "values += offset; result[0] = read_at(values, index);",
        "values += offset; auto* shifted = values; result[0] = shifted[index];",
        "values += offset; ++values; result[0] = values[0];",
        "values += offset; values += 1; result[0] = values[0];",
    ],
)
def test_rebased_index_checks_the_sum_not_only_each_operand(tmp_path, index_type, body):
    source = SOURCE.replace("constant ulong& index", f"constant {index_type}& index")
    source = source.replace(
        "kernel void",
        f"uint read_at(const device uint* data, {index_type} at) {{ return data[at]; }}\nkernel void",
    )
    source = source.replace("result[0] = values[offset + index];", body)
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


@pytest.mark.parametrize("operator", ["+=", "-="])
def test_unknown_wide_pointer_update_is_rejected_without_metadata(tmp_path, operator):
    source = SOURCE.replace(
        "result[0] = values[offset + index];",
        f"values {operator} offset; result[0] = values[0];",
    )
    report = translate_project(
        _index_config(tmp_path, source=source, conditions=()), format_output=False
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert (
        report["diagnostics"][0]["details"]["indexConversion"]["reason"]
        == "pointer-offset-range-unproven"
    )


@pytest.mark.parametrize("indirect", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("load", ["indices[0]", "indices[1]", "*indices"])
def test_pointer_offset_bounds_do_not_describe_loaded_values(
    tmp_path, indirect, local, load
):
    declaration = f"ulong at = {load}; " if local else ""
    index = "at" if local else load
    source = SOURCE.replace(
        "kernel void",
        f"uint read_at(const device uint* data, constant ulong* indices) {{ "
        f"{declaration}return data[{index}]; }}\nkernel void",
    ).replace("constant ulong& index", "constant ulong* indices")
    body = (
        "result[0] = read_at(values, indices);"
        if indirect
        else f"{declaration}result[0] = values[{index}];"
    )
    source = source.replace("result[0] = values[offset + index];", body)
    report = translate_project(
        _index_config(tmp_path, source=source, conditions=CONDITIONS[1:]),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report
    assert any(
        item["code"] == "project.translate.opengl-index-type-unsupported"
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize(
    "body",
    [
        "auto* shifted = values + 2147483647ul; result[0] = shifted[1];",
        "values += 2147483647ul; result[0] = values[1];",
    ],
)
def test_constant_wide_rebase_checks_complete_offset_without_metadata(tmp_path, body):
    source = SOURCE.replace("result[0] = values[offset + index];", body)
    report = translate_project(
        _index_config(tmp_path, source=source, conditions=()), format_output=False
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert any(item["severity"] == "error" for item in report["diagnostics"])


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_index_proof_keeps_enforced_upload_bounds(tmp_path, target):
    source = SOURCE.replace("result[0]", "result[2]")
    _, package, (descriptor,) = _packages(
        _index_config(tmp_path, source=source, target=target)
    )
    for index, offset in ((0, 0), (3, 0), (0, 4), (3, 4)):
        request, _ = _index_request(descriptor, package, index, offset)
        assert not request.execution_plan.diagnostics
    for index, offset in ((4, 0), (0, 5)):
        with pytest.raises(NativeLoaderDispatchError, match="outside"):
            _index_request(descriptor, package, index, offset)


def _generator(root, source=SOURCE):
    path = root / "index.metal"
    path.write_text(source)
    ast = parse(
        translate(
            path,
            format_output=False,
            source_options={"preserve_resource_origins": True},
        )
    )
    generator = GLSLCodeGen()
    entry = generator.glsl_software_subgroup_stage_entries(ast)[0][1]
    records = resolve_metadata_preconditions(
        entry,
        [
            MetadataRangePrecondition(
                entry_point=item["entryPoint"],
                parameter=item["parameter"],
                minimum=item["minimum"],
                maximum=item["maximum"],
            )
            for item in CONDITIONS
        ],
    )
    return generator, ast, entry, records


@pytest.mark.parametrize("method", ["generate", "generate_program", "generate_stage"])
def test_index_proofs_do_not_survive_removed_contracts(tmp_path, method):
    generator, ast, _, records = _generator(tmp_path)
    generator.set_metadata_preconditions(records)
    args = (ast, "compute") if method == "generate_stage" else (ast,)
    artifact = getattr(generator, method)(*args)
    assert parse_metadata_precondition_header(artifact) == records
    assert getattr(generator, method)(*args) == artifact
    generator.set_metadata_preconditions([])
    with pytest.raises(OpenGLIndexTypeError):
        getattr(generator, method)(*args)


def test_copied_helper_provenance_does_not_supply_entry_index_bounds(tmp_path):
    source = SOURCE.replace(
        "kernel void",
        "uint read_at(const device uint* data, ulong at) { return data[at]; }\nkernel void",
    )
    source = source.replace("values[offset + index]", "read_at(values, offset)")
    generator, ast, entry, records = _generator(tmp_path, source)
    function = next(func for func in ast.functions if func.name == "read_at")
    parameter = next(item for item in entry.parameters if item.name == "index")
    function.parameters[1].attributes = copy.deepcopy(parameter.attributes)
    generator.set_metadata_preconditions(records[:1])
    with pytest.raises(OpenGLIndexTypeError):
        generator.generate_program(ast)


@pytest.mark.parametrize("body", ["row += 1; return row[0];", "return row[1];"])
def test_pointer_range_assertion_does_not_hide_later_offset_overflow(body):
    from tests.test_translator.test_codegen.test_GLSL_codegen import (
        OPENGL_WIDE_STORAGE_POINTER_ASSERTION_SHADER,
    )

    generator = GLSLCodeGen().set_index_range_assertions(
        [
            dict(
                expression="values + offset",
                function="consume",
                minimum=0,
                maximum=2147483647,
            )
        ]
    )
    source = OPENGL_WIDE_STORAGE_POINTER_ASSERTION_SHADER.replace(
        "return row[0];", body
    )
    with pytest.raises(
        OpenGLIndexTypeError, match="composed pointer index range unproven"
    ):
        generator.generate(parse(source))


@pytest.mark.parametrize(
    "body",
    [
        "result[0] = pointers[0][0];",
        "pointers[0] += offset; result[0] = pointers[0][0];",
        "pointers[0] -= offset; result[0] = pointers[0][0];",
    ],
)
def test_pointer_array_initializer_never_emits_unlowered_glsl(tmp_path, body):
    source = SOURCE.replace(
        "result[0] = values[offset + index];",
        "const device uint* pointers[1] = {values}; " + body,
    )
    report = translate_project(
        _index_config(tmp_path, source=source), format_output=False
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    diagnostic = report["diagnostics"][0]
    assert diagnostic["code"] == "project.translate.opengl-storage-pointer-unsupported"
    assert (
        diagnostic["details"]["storagePointer"]["reason"]
        == "pointer-array-aggregate-initializer-unsupported"
    )

"""Target resource aliases remain distinct from source-local declarations."""

import pytest

from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from tests.test_translator.test_metadata_index_bounds import _index_config
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_pointer_interval_identity import _pointer_source
from tests.test_translator.test_project_metadata_entry_scope import _packages


def _collision_body(name, kind):
    local = f"{name}_"
    prefix = f"uint {local} = 29u; "
    if kind == "nested":
        prefix += f"{{ uint {local} = 31u; result[4] = {local}; }} "
    elif kind == "suffix":
        prefix += f"uint {name}_2 = 31u; result[4] = {name}_2; "
    read = "read_at(values, index)" if kind == "helper" else "values[index]"
    return prefix + "values += offset; " f"result[2] = {read}; result[3] = {local};"


@pytest.mark.parametrize("name", ["output", "input", "sample", "patch"])
@pytest.mark.parametrize("kind", ["direct", "helper", "nested", "suffix"])
def test_resource_alias_does_not_capture_local_declaration(tmp_path, name, kind):
    source = _pointer_source(name, _collision_body(name, kind))
    _, package, (descriptor,) = _packages(
        _index_config(tmp_path, source=source, target="opengl")
    )
    generated = (package / descriptor["artifact"]["packagePath"]).read_text()
    resource = next(
        item
        for item in descriptor["bindings"]
        if item["provenance"]["sourceResource"]["parameter"] == name
    )
    assert resource["coordinates"]["binding"] == 0
    assert resource["access"] == "read"
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize("preallocated", [False, True])
def test_local_name_allocation_keeps_another_sources_alias(preallocated):
    generator = GLSLCodeGen()
    generator.current_identifier_aliases = {"output": "output_"}
    if preallocated:
        generator.current_glsl_local_identifier_names = {"output_": "output_"}
    local = generator.glsl_local_declaration_identifier_name("output_")
    assert local != "output_"
    assert generator.current_identifier_aliases["output"] == "output_"
    assert generator.current_identifier_aliases["output_"] == local
    assert generator.glsl_local_declaration_identifier_name("output_") == local


def test_local_name_allocation_can_reuse_its_own_reservation():
    generator = GLSLCodeGen()
    generator.current_glsl_local_identifier_names = {"value": "value"}
    generator.local_variable_types = {"value": "uint"}
    assert generator.glsl_local_declaration_identifier_name("value") == "value"

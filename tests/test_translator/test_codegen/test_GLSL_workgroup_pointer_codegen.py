import re

import pytest

import crosstl.translator
from crosstl.translator.ast import (
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    UnaryOpNode,
)
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLWorkgroupPointerError,
)
from tests.test_translator.test_codegen.test_GLSL_codegen import (
    assert_glsl_compute_validates_if_available,
    glsl_expression_depends_on,
)


def test_atomic_add_through_workgroup_alias_preserves_base_and_access_offsets(
    tmp_path,
):
    shader = """
    shader WorkgroupPointerAtomicOffset {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                threadgroup int values[16];
                threadgroup int* p = values + 5u;
                atomicAdd(p[2], 1);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    backing = re.search(
        r"^shared\s+int\s+([A-Za-z_]\w*)\s*\[\s*16\s*\]\s*;",
        generated,
        re.MULTILINE,
    )
    assert backing is not None, generated
    assert re.search(r"\bint\s*\*", generated) is None, generated

    atomic_call = re.search(
        rf"\batomicAdd\s*\(\s*{re.escape(backing.group(1))}"
        r"\s*\[([^\]]+)\]\s*,\s*1\s*\)",
        generated,
    )
    assert atomic_call is not None, generated
    atomic_index = atomic_call.group(1)
    assert glsl_expression_depends_on(generated, atomic_index, "5u"), generated
    assert glsl_expression_depends_on(generated, atomic_index, "2"), generated

    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "workgroup_pointer_atomic_offset",
    )


def test_pointer_element_assignment_does_not_rebind_pointer_alias(tmp_path):
    shader = """
    shader WorkgroupPointerElementAssignment {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                threadgroup float destination[8];
                threadgroup float source[8];
                threadgroup float* outValues = destination + 2u;
                threadgroup float* inValues = source + 3u;
                inValues[0] = 4.0;
                outValues[0] = inValues[0];
                outValues[1] = 5.0;
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    destination = re.search(
        r"^shared\s+float\s+([A-Za-z_]\w*destination)\s*\[\s*8\s*\]\s*;",
        generated,
        re.MULTILINE,
    )
    source = re.search(
        r"^shared\s+float\s+([A-Za-z_]\w*source)\s*\[\s*8\s*\]\s*;",
        generated,
        re.MULTILINE,
    )
    assert destination is not None, generated
    assert source is not None, generated
    copy = re.search(
        rf"\b{re.escape(destination.group(1))}\s*\[([^\]]+)\]\s*=\s*"
        rf"{re.escape(source.group(1))}\s*\[([^\]]+)\]\s*;",
        generated,
    )
    second_write = re.search(
        rf"\b{re.escape(destination.group(1))}\s*\[([^\]]+)\]\s*=\s*5\.0\s*;",
        generated,
    )
    assert copy is not None, generated
    assert second_write is not None, generated
    assert glsl_expression_depends_on(generated, copy.group(1), "2u"), generated
    assert glsl_expression_depends_on(generated, copy.group(2), "3u"), generated
    assert glsl_expression_depends_on(generated, second_write.group(1), "2u"), generated

    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "workgroup_pointer_element_assignment",
    )


def test_ordinary_local_array_shadows_hoisted_workgroup_storage(tmp_path):
    shader = """
    shader WorkgroupStorageShadow {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                threadgroup int values[4];
                values[0] = 1;
                {
                    int values[4];
                    values[0] = 7;
                }
                values[1] = 2;
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    shared = re.search(
        r"^shared\s+int\s+([A-Za-z_]\w*)\s*\[\s*4\s*\]\s*;",
        generated,
        re.MULTILINE,
    )
    assert shared is not None, generated
    shared_name = re.escape(shared.group(1))
    assert re.search(rf"\b{shared_name}\s*\[\s*0\s*\]\s*=\s*1\s*;", generated)
    assert re.search(rf"\b{shared_name}\s*\[\s*1\s*\]\s*=\s*2\s*;", generated)
    local_block = re.search(
        r"\{\s*int\s+values\s*\[\s*4\s*\]\s*;\s*"
        r"values\s*\[\s*0\s*\]\s*=\s*7\s*;\s*\}",
        generated,
        re.DOTALL,
    )
    assert local_block is not None, generated
    assert shared.group(1) not in local_block.group(0), generated

    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "workgroup_storage_shadow",
    )


def test_value_parameter_shadows_global_workgroup_storage(tmp_path):
    shader = """
    shader GlobalWorkgroupParameterShadow {
        threadgroup int values[4];

        int increment(int values) {
            return values + 1;
        }

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                values[0] = increment(2);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert re.search(r"\bint\s+increment\s*\(\s*int\s+values\s*\)", generated)
    assert "return (values + 1);" in generated
    assert "values[0] = increment(2);" in generated

    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "global_workgroup_parameter_shadow",
    )


def test_bare_workgroup_pointer_expression_reports_structured_error():
    shader = """
    shader BareWorkgroupPointerExpression {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                threadgroup float values[8];
                threadgroup float* p = values + 1u;
                bool nonNull = p != nullptr;
            }
        }
    }
    """

    with pytest.raises(OpenGLWorkgroupPointerError) as exc_info:
        GLSLCodeGen().generate(crosstl.translator.parse(shader))

    error = exc_info.value
    assert error.parameter_name == "p"
    assert error.reason == "bare-pointer-expression"


def test_address_of_workgroup_pointer_reports_pointer_to_pointer_error():
    codegen = GLSLCodeGen()
    codegen.current_glsl_workgroup_pointer_aliases = {
        "p": {
            "kind": "workgroup-pointer",
            "root": "storage",
            "offset": "p_offset",
            "array_size": 8,
            "element_type": "float",
        }
    }

    with pytest.raises(OpenGLWorkgroupPointerError) as exc_info:
        codegen.generate_expression(UnaryOpNode("&", IdentifierNode("p")))

    assert exc_info.value.parameter_name == "p"
    assert exc_info.value.reason == "address-of-pointer"


def test_workgroup_pointer_struct_member_access_materializes_pointee():
    codegen = GLSLCodeGen()
    codegen.current_glsl_workgroup_pointer_aliases = {
        "p": {
            "kind": "workgroup-pointer",
            "root": "storage",
            "offset": "p_offset",
            "array_size": 8,
            "element_type": "Cell",
        }
    }
    codegen.local_variable_types["p"] = "Cell*"
    codegen.structs_by_name["Cell"] = object()
    codegen.struct_member_types["Cell"] = {"value": "int"}

    arrow = PointerAccessNode(IdentifierNode("p"), "value")
    dot = MemberAccessNode(IdentifierNode("p"), "value")

    assert codegen.generate_expression(arrow) == "storage[p_offset].value"
    assert codegen.generate_expression(dot) == "storage[p_offset].value"
    assert codegen.expression_result_type(arrow) == "int"
    assert codegen.expression_result_type(dot) == "int"


def test_workgroup_atomic_rejects_non_integer_element_type():
    shader = """
    shader WorkgroupPointerFloatAtomic {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                threadgroup float values[8];
                threadgroup float* p = values;
                atomicAdd(p[0], 1.0);
            }
        }
    }
    """

    with pytest.raises(OpenGLWorkgroupPointerError) as exc_info:
        GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert exc_info.value.parameter_name == "p"
    assert exc_info.value.reason == "atomic-element-type-unsupported"


def test_generic_workgroup_pointer_helper_specializes_without_pointer(tmp_path):
    shader = """
    shader GenericWorkgroupPointerHelper {
        generic<T> fn read_value(threadgroup T* values, uint index) -> T {
            return values[index];
        }

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                threadgroup float storage[8];
                threadgroup float* p = storage + 3u;
                float observed = read_value(p, 2u);
                storage[0] = observed;
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    backing = re.search(
        r"^shared\s+float\s+([A-Za-z_]\w*)\s*\[\s*8\s*\]\s*;",
        generated,
        re.MULTILINE,
    )
    assert backing is not None, generated
    assert re.search(r"\b(?:T|float)\s*\*", generated) is None, generated

    helper = re.search(
        r"\bfloat\s+(?P<name>read_value_float[A-Za-z0-9_]*)\s*"
        r"\((?P<params>[^)]*)\)\s*\{(?P<body>.*?)^\}",
        generated,
        re.MULTILINE | re.DOTALL,
    )
    assert helper is not None, generated
    assert "*" not in helper.group("params"), generated
    assert re.search(r"\buint\s+index\b", helper.group("params")), generated
    assert re.search(r"\bint\s+[A-Za-z_]\w*offset\b", helper.group("params")), generated

    helper_access = re.search(
        rf"\b{re.escape(backing.group(1))}\s*\[([^\]]+)\]",
        helper.group("body"),
    )
    assert helper_access is not None, generated
    assert "index" in helper_access.group(1), generated
    assert "offset" in helper_access.group(1), generated
    assert re.search(
        rf"\b{re.escape(helper.group('name'))}\s*\([^;]*\)\s*;",
        generated,
    ), generated

    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "generic_workgroup_pointer_helper",
    )


def test_nested_workgroup_pointer_helpers_preserve_bounded_backing_view(tmp_path):
    shader = """
    shader NestedWorkgroupPointerHelpers {
        void leaf(threadgroup float* values) {
            values[0] = values[1];
        }

        void middle(threadgroup float* values) {
            leaf(values + 4u);
        }

        compute {
            layout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in;

            void main(uint lane @ gl_LocalInvocationIndex) {
                threadgroup float sharedValues[64];
                middle(sharedValues + lane * 8u);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    backing = re.search(
        r"^shared\s+float\s+([A-Za-z_]\w*)\s*\[\s*64\s*\]\s*;",
        generated,
        re.MULTILINE,
    )
    assert backing is not None, generated
    assert re.search(r"\bfloat\s*\*", generated) is None, generated
    leaf = re.search(
        r"\bvoid\s+(?P<name>leaf[A-Za-z0-9_]*)\s*"
        r"\((?P<params>[^)]*)\)\s*\{(?P<body>.*?)^\}",
        generated,
        re.MULTILINE | re.DOTALL,
    )
    assert leaf is not None, generated
    accesses = re.findall(
        rf"\b{re.escape(backing.group(1))}\s*\[([^\]]+)\]",
        leaf.group("body"),
    )
    assert len(accesses) == 2, generated
    assert all("offset" in expression for expression in accesses), generated
    assert any("1" in expression for expression in accesses), generated
    assert "gl_LocalInvocationIndex" in generated, generated

    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "nested_workgroup_pointer_helpers",
    )


def test_nested_workgroup_pointer_helper_rejects_out_of_bounds_view():
    shader = """
    shader OutOfBoundsWorkgroupPointerHelpers {
        void leaf(threadgroup float* values) {
            values[0] = values[1];
        }

        void middle(threadgroup float* values) {
            leaf(values + 1u);
        }

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                threadgroup float sharedValues[8];
                middle(sharedValues + 7u);
            }
        }
    }
    """

    with pytest.raises(OpenGLWorkgroupPointerError) as exc_info:
        GLSLCodeGen().generate(crosstl.translator.parse(shader))

    error = exc_info.value
    assert error.function_name == "leaf"
    assert error.parameter_name == "values"
    assert error.backing_name is not None and error.backing_name.endswith(
        "sharedValues"
    )
    assert "8" in error.offset_expression
    assert error.materialization_name is not None
    assert error.reason == "view-out-of-bounds"

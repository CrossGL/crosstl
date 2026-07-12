import re

import pytest

import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLStoragePointerError,
)
from tests.test_translator.test_codegen.test_GLSL_codegen import (
    assert_glsl_compute_validates_if_available,
    glsl_expression_depends_on,
)


def test_structured_buffer_vector_component_index_is_not_pointer_offset(tmp_path):
    shader = """
    shader StorageVectorComponent {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main(
                StructuredBuffer<float4> source @buffer(0),
                RWStructuredBuffer<float> result @buffer(1)
            ) {
                result[0] = source[1][2];
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert re.search(
        r"\bresult\s*\[\s*0\s*\]\s*=\s*" r"source\s*\[\s*1\s*\]\s*\[\s*2\s*\]\s*;",
        generated,
    ), generated

    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "storage_vector_component_index",
    )


def test_storage_pointer_dereferences_preserve_mutable_offsets(tmp_path):
    shader = """
    shader StoragePointerDereference {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main(
                StructuredBuffer<float> sourceValues @buffer(0),
                RWStructuredBuffer<float> resultValues @buffer(1)
            ) {
                const device float* cursor = sourceValues + 1;
                float first = *cursor;
                cursor += 2;
                float second = *cursor;
                device float* writer = resultValues + 2;
                *writer = first + second;
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "int cursor_offset = int(1);" in generated
    assert "float first = sourceValues[cursor_offset];" in generated
    assert "cursor_offset += int(2);" in generated
    assert "float second = sourceValues[cursor_offset];" in generated
    assert "int writer_offset = int(2);" in generated
    assert "resultValues[writer_offset] = (first + second);" in generated
    assert "*cursor" not in generated
    assert "*writer" not in generated
    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "storage_pointer_dereference",
    )


def test_storage_pointer_dereference_retains_boolean_pointee_type(tmp_path):
    shader = """
    shader BooleanStoragePointerDereference {
        void consume(bool value) {
            return;
        }

        void consume(int value) {
            return;
        }

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main(
                StructuredBuffer<bool> sourceValues @buffer(0),
                RWStructuredBuffer<bool> resultValues @buffer(1)
            ) {
                const device bool* cursor = sourceValues + 1;
                consume(*cursor);
                device bool* writer = resultValues + 2;
                *writer = 1;
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert re.search(
        r"\bconsume[A-Za-z0-9_]*\s*\(\s*sourceValues\[cursor_offset\]\s*\)",
        generated,
    )
    assert "resultValues[writer_offset] = (1 != 0);" in generated
    assert "bool*" not in generated
    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "boolean_storage_pointer_dereference",
    )


def test_writable_storage_alias_rejects_readonly_structured_buffer():
    shader = """
    shader ReadonlyStorageWritableAlias {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main(
                StructuredBuffer<float> source @buffer(0),
                RWStructuredBuffer<float> result @buffer(1)
            ) {
                device float* writable = &source[1];
                writable[0] = 2.0;
                result[0] = source[0];
            }
        }
    }
    """

    with pytest.raises(OpenGLStoragePointerError) as exc_info:
        GLSLCodeGen().generate(crosstl.translator.parse(shader))

    error = exc_info.value
    assert error.parameter_name == "writable"
    assert error.reason is not None
    assert "access" in error.reason.lower()
    assert error.project_diagnostic_code == (
        "project.translate.opengl-storage-pointer-unsupported"
    )


def test_bare_local_storage_pointer_comparison_raises():
    shader = """
    shader BareLocalStoragePointer {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main(RWStructuredBuffer<float> values @buffer(0)) {
                device float* localValues = &values[1];
                bool nonNull = localValues != nullptr;
                values[0] = nonNull ? 1.0 : 0.0;
            }
        }
    }
    """

    with pytest.raises(OpenGLStoragePointerError) as exc_info:
        GLSLCodeGen().generate(crosstl.translator.parse(shader))

    error = exc_info.value
    assert error.parameter_name == "localValues"
    assert error.reason == "bare-pointer-expression"


def test_for_initializer_storage_pointer_root_is_preserved_or_rejected(tmp_path):
    shader = """
    shader StoragePointerForInitializerRoot {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main(
                StructuredBuffer<float> leftSource @buffer(0),
                StructuredBuffer<float> rightSource @buffer(1),
                RWStructuredBuffer<float> result @buffer(2)
            ) {
                constant float* values = &leftSource[1u];
                int i = 0;
                for (values = &rightSource[2u]; i < 1; i++) {
                    result[0] = values[0];
                }
                result[1] = values[0];
            }
        }
    }
    """

    try:
        generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))
    except OpenGLStoragePointerError as error:
        assert error.parameter_name == "values"
        assert error.reason
        assert error.project_diagnostic_code == (
            "project.translate.opengl-storage-pointer-unsupported"
        )
        diagnostic = f"{error.reason} {error}".lower()
        assert any(term in diagnostic for term in ("backing", "reassign", "root"))
        return

    post_loop_read = re.search(
        r"\bresult\s*\[\s*1\s*\]\s*=\s*"
        r"(?P<root>[A-Za-z_]\w*)\s*\[(?P<index>[^\]]+)\]\s*;",
        generated,
    )
    assert post_loop_read is not None, generated
    assert post_loop_read.group("root") == "rightSource", generated

    post_loop_index = post_loop_read.group("index")
    index_identifiers = re.findall(r"\b[A-Za-z_]\w*\b", post_loop_index)
    assert "2u" in post_loop_index or any(
        re.search(
            rf"\b{re.escape(identifier)}\s*=\s*int\s*\(\s*2u\s*\)",
            generated,
        )
        for identifier in index_identifiers
    ), generated

    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "storage_pointer_for_initializer_root",
    )


def test_global_structured_buffer_address_of_helper_call(tmp_path):
    shader = """
    shader GlobalStoragePointerHelper {
        StructuredBuffer<float> source @binding(0);
        RWStructuredBuffer<float> result @binding(1);

        float readValue(constant float* values, uint index) {
            return values[index];
        }

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                result[0] = readValue(&source[2u], 1u);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    helper = re.search(
        r"\bfloat\s+(?P<name>readValue[A-Za-z0-9_]*)\s*"
        r"\((?P<params>[^)]*)\)\s*\{(?P<body>.*?)^\}",
        generated,
        re.MULTILINE | re.DOTALL,
    )
    assert helper is not None, generated
    source_read = re.search(
        r"\bsource\s*\[(?P<index>[^\]]+)\]",
        helper.group("body"),
    )
    assert source_read is not None, generated
    assert "index" in source_read.group("index"), generated

    main_assignment = next(
        (
            line
            for line in generated.splitlines()
            if "result[0]" in line and helper.group("name") in line
        ),
        None,
    )
    assert main_assignment is not None, generated
    assert "1u" in main_assignment, generated
    assert glsl_expression_depends_on(generated, main_assignment, "2u"), generated
    assert re.search(r"\bconstant\s+float\s*\*|(?<![&])&(?![&])", generated) is None

    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "global_storage_pointer_helper",
    )


def test_storage_backed_struct_pointer_member_access(tmp_path):
    shader = """
    shader StorageStructPointerMember {
        struct Cell {
            float value;
            uint tag;
        };

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main(
                StructuredBuffer<Cell> source @buffer(0),
                RWStructuredBuffer<float> result @buffer(1)
            ) {
                constant Cell* cell = &source[2u];
                result[0] = cell->value;
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    member_read = re.search(
        r"\bresult\s*\[\s*0\s*\]\s*=\s*" r"source\s*\[(?P<index>[^\]]+)\]\.value\s*;",
        generated,
    )
    assert member_read is not None, generated
    assert glsl_expression_depends_on(
        generated,
        member_read.group("index"),
        "2u",
    ), generated
    assert "cell->value" not in generated
    assert re.search(r"\bconstant\s+Cell\s*\*", generated) is None

    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "storage_struct_pointer_member",
    )


def test_module_structured_buffer_vector_component_index_is_not_pointer_offset(
    tmp_path,
):
    shader = """
    shader ModuleStorageVectorComponent {
        StructuredBuffer<float4> source @binding(0);
        RWStructuredBuffer<float> result @binding(1);

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                result[0] = source[1][2];
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert re.search(
        r"\bresult\s*\[\s*0\s*\]\s*=\s*" r"source\s*\[\s*1\s*\]\s*\[\s*2\s*\]\s*;",
        generated,
    ), generated
    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "module_storage_vector_component_index",
    )


def test_stage_local_structured_buffer_vector_component_index_is_not_pointer_offset(
    tmp_path,
):
    shader = """
    shader StageLocalStorageVectorComponent {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            StructuredBuffer<float4> source @binding(0);
            RWStructuredBuffer<float> result @binding(1);

            void main() {
                result[0] = source[1][2];
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert re.search(
        r"\bresult\s*\[\s*0\s*\]\s*=\s*" r"source\s*\[\s*1\s*\]\s*\[\s*2\s*\]\s*;",
        generated,
    ), generated
    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "stage_local_storage_vector_component_index",
    )


def test_readonly_rwstructured_buffer_rejects_writable_pointer_alias():
    shader = """
    shader ReadonlyMetadataStorageAlias {
        RWStructuredBuffer<float> source @binding(0) @readonly;

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {
                device float* writable = &source[0];
                writable[0] = 2.0;
            }
        }
    }
    """

    with pytest.raises(OpenGLStoragePointerError) as exc_info:
        GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert exc_info.value.parameter_name == "writable"
    assert exc_info.value.reason == "access-mismatch"


def test_structured_buffer_helper_forwards_root_to_device_pointer_helper(tmp_path):
    shader = """
    shader NestedStorageResourceForwarding {
        float readPointer(device float* values, int index) {
            return values[index];
        }

        float readBuffer(StructuredBuffer<float> values, int base) {
            return readPointer(values + base, 1);
        }

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main(
                StructuredBuffer<float> source @buffer(0),
                RWStructuredBuffer<float> result @buffer(1)
            ) {
                result[0] = readBuffer(source, 2);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "device" not in generated
    assert re.search(r"\bfloat\s*\*", generated) is None
    assert "source[(" in generated or "source[" in generated
    assert "readPointer" in generated
    assert "readBuffer" in generated
    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "nested_storage_resource_forwarding",
    )


def test_overloaded_storage_pointer_helpers_specialize_by_source_signature(tmp_path):
    shader = """
    shader OverloadedStoragePointerHelpers {
        float combine(constant float* lhs, constant float* rhs, int index) {
            return lhs[index] + rhs[index];
        }

        float combine(
            constant float* lhs,
            constant float* rhs,
            constant float* extra,
            int index
        ) {
            return lhs[index] + rhs[index] + extra[index];
        }

        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main(
                StructuredBuffer<float> lhs @buffer(0),
                StructuredBuffer<float> rhs @buffer(1),
                StructuredBuffer<float> extra @buffer(2),
                RWStructuredBuffer<float> result @buffer(3)
            ) {
                result[0] = combine(lhs, rhs, 1);
                result[1] = combine(lhs, rhs, extra, 2);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(crosstl.translator.parse(shader))

    helpers = re.findall(
        r"\bfloat\s+(combine[A-Za-z0-9_]*)\s*\(([^)]*)\)\s*\{",
        generated,
    )
    assert len(helpers) == 2, generated
    assert sorted(parameters.count("int ") for _name, parameters in helpers) == [3, 4]
    assert re.search(r"\bfloat\s*\*", generated) is None
    assert_glsl_compute_validates_if_available(
        generated,
        tmp_path,
        "overloaded_storage_pointer_helpers",
    )

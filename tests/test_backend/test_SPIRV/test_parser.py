import pytest
from typing import List
from crosstl.backend.SPIRV.VulkanLexer import VulkanLexer
from crosstl.backend.SPIRV import VulkanParser
from crosstl.backend.SPIRV.VulkanAst import (
    AssignmentNode,
    ArrayAccessNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    ContinueNode,
    DiscardNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LayoutNode,
    MemberAccessNode,
    MethodCallNode,
    ReturnNode,
    SwitchNode,
    UnaryOpNode,
    UniformNode,
    VariableNode,
    WhileNode,
)


def parse_code(tokens: List):
    """Test the parser
    Args:
        tokens (List): The list of tokens generated from the lexer

    Returns:
        ASTNode: The abstract syntax tree generated from the code
    """
    parser = VulkanParser(tokens)
    return parser.parse()


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = VulkanLexer(code)
    return lexer.tokenize()


def test_mod_parsing():
    code = """
    
    void main() {
        int a = 10 % 3;  // Basic modulus
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Modulus operator parsing not implemented")


def test_bitwise_not_parsing():
    code = """
    void main() {
        int a = 5;
        int b = ~a;  // Bitwise NOT
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Bitwise NOT operator parsing not implemented")


def test_function_parameter_qualifiers_parse():
    code = """
    void accumulate(in vec3 normal, inout float weight, out vec4 color) {
        color = vec4(normal, weight);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = ast.functions[0]

    assert [(p.vtype, p.name, p.qualifiers) for p in function.params] == [
        ("vec3", "normal", ["in"]),
        ("float", "weight", ["inout"]),
        ("vec4", "color", ["out"]),
    ]


def test_function_parameter_array_suffixes_parse():
    code = """
    void sampleResources(sampler2D textures[4], uimage2D outputs[2]) {
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = ast.functions[0]

    assert [(p.vtype, p.name) for p in function.params] == [
        ("sampler2D", "textures[4]"),
        ("uimage2D", "outputs[2]"),
    ]


def test_struct_array_members_preserve_suffixes():
    code = """
    struct LightBlock {
        vec3 positions[4];
        float weights[4];
    };
    void main() {}
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    members = ast.structs[0].members

    assert [(member.vtype, member.name) for member in members] == [
        ("vec3", "positions[4]"),
        ("float", "weights[4]"),
    ]


def test_continue_parsing():
    code = """
    void main() {
        for (int i = 0; i < 4; i++) {
            if (i == 2) {
                continue;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        loop = ast.functions[0].body[0]
        condition = loop.body[0]

        assert isinstance(loop, ForNode)
        assert isinstance(condition, IfNode)
        assert isinstance(condition.if_body[0], ContinueNode)
    except SyntaxError:
        pytest.fail("Continue statement parsing not implemented")


def test_for_update_parses_structured_postfix_targets():
    code = """
    void main() {
        int items[4];
        for (int i = 0; i < 4; items[i]++) {
        }
        for (int i = 0; i < 4; object.field--) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    array_loop = ast.functions[0].body[1]
    member_loop = ast.functions[0].body[2]

    assert isinstance(array_loop, ForNode)
    assert isinstance(array_loop.update, UnaryOpNode)
    assert array_loop.update.op == "POST_INCREMENT"
    assert isinstance(array_loop.update.operand, ArrayAccessNode)
    assert array_loop.update.operand.array.name == "items"
    assert array_loop.update.operand.index.name == "i"

    assert isinstance(member_loop.update, UnaryOpNode)
    assert member_loop.update.op == "POST_DECREMENT"
    assert isinstance(member_loop.update.operand, MemberAccessNode)
    assert member_loop.update.operand.object.name == "object"
    assert member_loop.update.operand.member == "field"


def test_for_update_parses_structured_prefix_target():
    code = """
    void main() {
        int items[4];
        for (int i = 0; i < 4; ++items[i]) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loop = ast.functions[0].body[1]

    assert isinstance(loop.update, UnaryOpNode)
    assert loop.update.op == "PRE_INCREMENT"
    assert isinstance(loop.update.operand, ArrayAccessNode)


def test_for_update_parses_compound_assignment():
    code = """
    void main() {
        for (int i = 0; i < 4; i += 1) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loop = ast.functions[0].body[0]

    assert isinstance(loop.update, AssignmentNode)
    assert loop.update.operator == "+="
    assert loop.update.left.name == "i"
    assert loop.update.right == "1"


def test_for_update_parses_structured_assignment_targets():
    code = """
    void main() {
        int value = 1;
        int items[4];
        for (int i = 0; i < 4; items[i] += value) {
        }
        for (int i = 0; i < 4; object.field = value) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    array_loop = ast.functions[0].body[2]
    member_loop = ast.functions[0].body[3]

    assert isinstance(array_loop.update, AssignmentNode)
    assert array_loop.update.operator == "+="
    assert isinstance(array_loop.update.left, ArrayAccessNode)
    assert array_loop.update.left.array.name == "items"
    assert array_loop.update.left.index.name == "i"
    assert array_loop.update.right.name == "value"

    assert isinstance(member_loop.update, AssignmentNode)
    assert member_loop.update.operator == "="
    assert isinstance(member_loop.update.left, MemberAccessNode)
    assert member_loop.update.left.object.name == "object"
    assert member_loop.update.left.member == "field"
    assert member_loop.update.right.name == "value"


def test_for_parses_empty_clauses():
    code = """
    void main() {
        int i = 0;
        for (; i < 4; i++) {
        }
        for (int j = 0; ; j++) {
            break;
        }
        for (int k = 0; k < 4; ) {
            k++;
        }
        for (; ; ) {
            break;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loops = ast.functions[0].body[1:]

    init_empty_loop = loops[0]
    condition_empty_loop = loops[1]
    update_empty_loop = loops[2]
    all_empty_loop = loops[3]

    assert isinstance(init_empty_loop, ForNode)
    assert init_empty_loop.init is None
    assert isinstance(init_empty_loop.condition, BinaryOpNode)
    assert isinstance(init_empty_loop.update, UnaryOpNode)

    assert isinstance(condition_empty_loop.init, AssignmentNode)
    assert condition_empty_loop.condition is None
    assert isinstance(condition_empty_loop.update, UnaryOpNode)
    assert isinstance(condition_empty_loop.body[0], BreakNode)

    assert isinstance(update_empty_loop.init, AssignmentNode)
    assert isinstance(update_empty_loop.condition, BinaryOpNode)
    assert update_empty_loop.update is None

    assert all_empty_loop.init is None
    assert all_empty_loop.condition is None
    assert all_empty_loop.update is None


def test_for_clause_comma_lists_parse():
    code = """
    void main() {
        for (int i = 0, j = 1; i < 4; i++, j--) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loop = ast.functions[0].body[0]

    assert isinstance(loop.init, list)
    assert len(loop.init) == 2
    assert all(isinstance(item, AssignmentNode) for item in loop.init)
    assert loop.init[0].left.vtype == "int"
    assert loop.init[0].left.name == "i"
    assert loop.init[1].left.vtype == "int"
    assert loop.init[1].left.name == "j"

    assert isinstance(loop.update, list)
    assert [update.op for update in loop.update] == [
        "POST_INCREMENT",
        "POST_DECREMENT",
    ]


def test_for_update_rejects_method_call_target():
    code = """
    void main() {
        for (int i = 0; i < 4; object.method()++) {
        }
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Invalid update target: MethodCallNode"):
        parse_code(tokens)


def test_for_update_rejects_structured_postfix_trailing_assignment():
    code = """
    void main() {
        int items[4];
        for (int i = 0; i < 4; items[i]++ = 1) {
        }
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Expected RPAREN, got EQUALS"):
        parse_code(tokens)


def test_switch_parsing_preserves_cases_default_and_breaks():
    code = """
    void main() {
        int value = 1;
        switch (value) {
            case 0:
                value = 2;
                break;
            case 1:
                value = 3;
            default:
                value = 4;
                break;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    switch = ast.functions[0].body[1]

    assert isinstance(switch, SwitchNode)
    assert len(switch.cases) == 3
    assert all(isinstance(case, CaseNode) for case in switch.cases)
    assert switch.cases[0].value == "0"
    assert isinstance(switch.cases[0].body[-1], BreakNode)
    assert switch.cases[1].value == "1"
    assert not isinstance(switch.cases[1].body[-1], BreakNode)
    assert switch.cases[2].value is None
    assert isinstance(switch.cases[2].body[-1], BreakNode)


def test_switch_rejects_duplicate_default_labels():
    code = """
    void main() {
        int value = 0;
        switch (value) {
            default:
                value = 1;
            case 0:
                value = 2;
            default:
                value = 3;
        }
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="duplicate default"):
        parse_code(tokens)


def test_switch_node_preserves_empty_default_case():
    switch = SwitchNode("value", [], default_case=[])

    assert switch.default_case == []
    assert switch.default == []


def test_switch_parsing_preserves_return_and_discard_statements():
    code = """
    void main() {
        int value = 1;
        switch (value) {
            case 0:
                discard;
            case 1:
                return;
            default:
                value = 4;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    switch = ast.functions[0].body[1]

    assert isinstance(switch, SwitchNode)
    assert isinstance(switch.cases[0].body[0], DiscardNode)
    assert isinstance(switch.cases[1].body[0], ReturnNode)
    assert switch.cases[1].body[0].value is None


def test_switch_loop_control_parsing_keeps_nearest_nested_blocks():
    code = """
    void main() {
        int hits = 0;
        for (int i = 0; i < 4; i++) {
            switch (i) {
                case 0:
                    hits = hits + 1;
                    continue;
                case 1:
                    while (hits < 3) {
                        hits = hits + 1;
                        break;
                    }
                    break;
                default:
                    hits = hits + 100;
                    break;
            }
            hits = hits + 1000;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loop = ast.functions[0].body[1]
    switch = loop.body[0]
    case_zero, case_one, default_case = switch.cases

    assert isinstance(loop, ForNode)
    assert isinstance(switch, SwitchNode)
    assert isinstance(loop.body[1], AssignmentNode)
    assert isinstance(case_zero.body[-1], ContinueNode)
    assert isinstance(case_one.body[0], WhileNode)
    assert isinstance(case_one.body[0].body[-1], BreakNode)
    assert isinstance(case_one.body[-1], BreakNode)
    assert default_case.value is None
    assert isinstance(default_case.body[-1], BreakNode)


def test_loop_inside_switch_case_keeps_following_case_statement():
    code = """
    void main() {
        int hits = 0;
        int value = 0;
        switch (value) {
            case 0:
                for (int j = 0; j < 2; j++) {
                    hits = hits + j;
                    break;
                }
                hits = hits + 10;
                break;
            default:
                break;
        }
        hits = hits + 1000;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    switch = ast.functions[0].body[2]
    case_zero = switch.cases[0]

    assert isinstance(switch, SwitchNode)
    assert isinstance(case_zero.body[0], ForNode)
    assert isinstance(case_zero.body[0].body[-1], BreakNode)
    assert isinstance(case_zero.body[1], AssignmentNode)
    assert isinstance(case_zero.body[2], BreakNode)
    assert isinstance(ast.functions[0].body[3], AssignmentNode)


def test_break_at_function_scope_is_rejected():
    code = """
    void main() {
        break;
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="break used outside loop or switch"):
        parse_code(tokens)


def test_continue_at_function_scope_is_rejected():
    code = """
    void main() {
        continue;
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="continue used outside loop"):
        parse_code(tokens)


def test_continue_inside_switch_without_loop_is_rejected():
    code = """
    void main() {
        int value = 0;
        switch (value) {
            case 0:
                continue;
            default:
                break;
        }
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="continue used outside loop"):
        parse_code(tokens)


def test_loop_depth_does_not_leak_after_loop_inside_switch_case():
    code = """
    void main() {
        int value = 0;
        switch (value) {
            case 0:
                for (int i = 0; i < 2; i++) {
                    continue;
                }
                continue;
            default:
                break;
        }
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="continue used outside loop"):
        parse_code(tokens)


def test_break_inside_switch_without_loop_is_allowed():
    code = """
    void main() {
        int value = 0;
        switch (value) {
            case 0:
                break;
            default:
                break;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    switch = ast.functions[0].body[1]

    assert isinstance(switch, SwitchNode)
    assert isinstance(switch.cases[0].body[-1], BreakNode)
    assert isinstance(switch.cases[1].body[-1], BreakNode)


def test_switch_rejects_statement_before_first_case():
    code = """
    void main() {
        int value = 1;
        switch (value) {
            value = 4;
        }
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Expected CASE or DEFAULT"):
        parse_code(tokens)


def test_switch_rejects_unterminated_case():
    code = """
    void main() {
        int value = 1;
        switch (value) {
            case 0:
                value = 4;
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Unterminated switch"):
        parse_code(tokens)


def test_layout_array_member_parsing():
    code = """
    layout(set = 0, binding = 0) uniform Bones {
        mat4 transforms[64];
        vec4 weights[4];
    } bones;
    void main() {
        gl_Position = transforms[0] * vec4(1.0, 0.0, 0.0, 1.0);
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        assignment = ast.functions[0].body[0]

        assert ast.global_variables[0].struct_fields == [
            ("mat4", "transforms[64]"),
            ("vec4", "weights[4]"),
        ]
        assert isinstance(assignment.right.left, ArrayAccessNode)
        assert assignment.right.left.array.name == "transforms"
        assert assignment.right.left.index == "0"
    except SyntaxError:
        pytest.fail("Layout array member parsing not implemented")


def test_layout_custom_struct_member_type_parsing():
    code = """
    struct Light {
        vec3 position;
    };
    layout(set = 0, binding = 0) uniform Scene {
        Light lights[4];
        mat4 view;
    } scene;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert isinstance(layout, LayoutNode)
    assert layout.layout_type == "UNIFORM"
    assert layout.block_name == "Scene"
    assert layout.variable_name == "scene"
    assert layout.struct_fields == [("Light", "lights[4]"), ("mat4", "view")]


def test_custom_struct_uniform_type_parsing():
    code = """
    struct Light {
        vec3 position;
    };
    layout(set = 0, binding = 1) uniform Light activeLight;
    uniform Light fallbackLight;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]
    uniform = ast.global_variables[1]

    assert isinstance(layout, LayoutNode)
    assert layout.block_name is None
    assert layout.layout_type == "UNIFORM"
    assert layout.data_type == "Light"
    assert layout.variable_name == "activeLight"
    assert isinstance(uniform, UniformNode)
    assert uniform.vtype == "Light"
    assert uniform.name == "fallbackLight"


def test_layout_resource_uniform_parsing():
    code = """
    layout(set = 0, binding = 1) uniform sampler2D albedoTex;
    void main() {
        vec4 color = texture(albedoTex, vec2(0.5, 0.5));
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert isinstance(layout, LayoutNode)
    assert layout.qualifiers == [("set", "0"), ("binding", "1")]
    assert layout.layout_type == "UNIFORM"
    assert layout.data_type == "sampler2D"
    assert layout.variable_name == "albedoTex"
    assert layout.struct_fields == []


def test_layout_push_constant_block_parsing():
    code = """
    layout(push_constant) uniform PushConstants {
        mat4 model;
        vec4 tint;
    } pc;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert isinstance(layout, LayoutNode)
    assert layout.push_constant is True
    assert layout.qualifiers == []
    assert layout.layout_type == "UNIFORM"
    assert layout.block_name == "PushConstants"
    assert layout.variable_name == "pc"
    assert layout.struct_fields == [("mat4", "model"), ("vec4", "tint")]


def test_version_and_extension_directives_before_layout_parse():
    code = """
    #version 450
    #extension GL_EXT_nonuniform_qualifier : enable
    layout(set = 0, binding = 0) uniform sampler2D albedoTex;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert len(ast.global_variables) == 1
    assert len(ast.functions) == 1
    assert ast.functions[0].name == "main"
    assert isinstance(layout, LayoutNode)
    assert layout.qualifiers == [("set", "0"), ("binding", "0")]
    assert layout.layout_type == "UNIFORM"
    assert layout.data_type == "sampler2D"
    assert layout.variable_name == "albedoTex"


def test_precision_declarations_before_layout_parse():
    code = """
    precision highp float;
    precision mediump sampler2D;
    layout(set = 0, binding = 0) uniform sampler2D albedoTex;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert len(ast.global_variables) == 1
    assert len(ast.functions) == 1
    assert isinstance(layout, LayoutNode)
    assert layout.qualifiers == [("set", "0"), ("binding", "0")]
    assert layout.layout_type == "UNIFORM"
    assert layout.data_type == "sampler2D"
    assert layout.variable_name == "albedoTex"


def test_layout_precision_qualifier_after_in_parsing():
    code = """
    #version 310 es
    precision highp float;
    layout(location = 0) in highp vec2 vUV;
    layout(location = 0) out mediump vec4 fragColor;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    input_layout = ast.global_variables[0]
    output_layout = ast.global_variables[1]

    assert isinstance(input_layout, LayoutNode)
    assert input_layout.declaration_qualifiers == ["highp"]
    assert input_layout.layout_type == "IN"
    assert input_layout.data_type == "vec2"
    assert input_layout.variable_name == "vUV"
    assert output_layout.declaration_qualifiers == ["mediump"]
    assert output_layout.layout_type == "OUT"
    assert output_layout.data_type == "vec4"
    assert output_layout.variable_name == "fragColor"


def test_layout_interpolation_qualifier_parsing():
    code = """
    layout(location = 0) flat in int faceID;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert isinstance(layout, LayoutNode)
    assert layout.declaration_qualifiers == ["flat"]
    assert layout.layout_type == "IN"
    assert layout.data_type == "int"
    assert layout.variable_name == "faceID"


def test_layout_component_and_index_qualifier_parsing():
    code = """
    layout(location = 1, component = 2) noperspective in vec4 color;
    layout(location = 0, index = 1) out vec4 fragColor;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    color = ast.global_variables[0]
    frag_color = ast.global_variables[1]

    assert isinstance(color, LayoutNode)
    assert color.qualifiers == [("location", "1"), ("component", "2")]
    assert color.declaration_qualifiers == ["noperspective"]
    assert color.layout_type == "IN"
    assert color.data_type == "vec4"
    assert color.variable_name == "color"
    assert isinstance(frag_color, LayoutNode)
    assert frag_color.qualifiers == [("location", "0"), ("index", "1")]
    assert frag_color.layout_type == "OUT"
    assert frag_color.data_type == "vec4"
    assert frag_color.variable_name == "fragColor"


def test_layout_readonly_buffer_qualifier_parsing():
    code = """
    layout(set = 0, binding = 0) readonly buffer Particles {
        vec4 pos[];
    } particles;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert isinstance(layout, LayoutNode)
    assert layout.declaration_qualifiers == ["readonly"]
    assert layout.layout_type == "BUFFER"
    assert layout.block_name == "Particles"
    assert layout.variable_name == "particles"
    assert layout.struct_fields == [("vec4", "pos[]")]


def test_layout_storage_image_format_and_access_qualifier_parsing():
    code = """
    layout(set = 0, binding = 0, r32ui) coherent readonly uniform uimage2D counters;
    layout(set = 0, binding = 1, rgba32f) writeonly uniform image2D outImage;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    counters = ast.global_variables[0]
    out_image = ast.global_variables[1]

    assert isinstance(counters, LayoutNode)
    assert counters.qualifiers == [("set", "0"), ("binding", "0"), ("r32ui", None)]
    assert counters.declaration_qualifiers == ["coherent", "readonly"]
    assert counters.layout_type == "UNIFORM"
    assert counters.data_type == "uimage2D"
    assert counters.variable_name == "counters"
    assert isinstance(out_image, LayoutNode)
    assert out_image.qualifiers == [
        ("set", "0"),
        ("binding", "1"),
        ("rgba32f", None),
    ]
    assert out_image.declaration_qualifiers == ["writeonly"]
    assert out_image.layout_type == "UNIFORM"
    assert out_image.data_type == "image2D"
    assert out_image.variable_name == "outImage"


def test_layout_identifier_qualifier_values_parse():
    code = """
    layout(set = MATERIAL_SET, binding = ALBEDO_BINDING) uniform sampler2D albedoTex;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert isinstance(layout, LayoutNode)
    assert layout.qualifiers == [
        ("set", "MATERIAL_SET"),
        ("binding", "ALBEDO_BINDING"),
    ]
    assert layout.data_type == "sampler2D"
    assert layout.variable_name == "albedoTex"


def test_compute_local_size_layout_parsing():
    code = """
    layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]

    assert isinstance(layout, LayoutNode)
    assert layout.qualifiers == [
        ("local_size_x", "8"),
        ("local_size_y", "4"),
        ("local_size_z", "1"),
    ]
    assert layout.layout_type == "IN"
    assert layout.data_type is None
    assert layout.variable_name is None


def test_standalone_postfix_update_parsing():
    code = """
    void main() {
        int i = 0;
        do {
            i++;
            i--;
        } while (i < 3);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loop = ast.functions[0].body[1]

    assert isinstance(loop, DoWhileNode)
    assert isinstance(loop.body[0], UnaryOpNode)
    assert loop.body[0].op == "POST_INCREMENT"
    assert isinstance(loop.body[1], UnaryOpNode)
    assert loop.body[1].op == "POST_DECREMENT"


def test_postfix_update_rejects_trailing_assignment():
    code = """
    void main() {
        int i = 0;
        i++ = 1;
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Expected SEMICOLON, got EQUALS"):
        parse_code(tokens)


def test_plain_resource_uniform_parsing():
    code = """
    uniform sampler2D albedoTex;
    void main() {
        vec4 color = texture(albedoTex, vec2(0.5, 0.5));
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    uniform = ast.global_variables[0]

    assert isinstance(uniform, UniformNode)
    assert uniform.vtype == "sampler2D"
    assert uniform.name == "albedoTex"


def test_one_dimensional_sampler_uniform_parsing():
    code = """
    layout(set = 0, binding = 0) uniform sampler1D ramp;
    uniform sampler1DArray ramps;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]
    uniform = ast.global_variables[1]

    assert isinstance(layout, LayoutNode)
    assert layout.data_type == "sampler1D"
    assert layout.variable_name == "ramp"
    assert isinstance(uniform, UniformNode)
    assert uniform.vtype == "sampler1DArray"
    assert uniform.name == "ramps"


def test_atomic_uint_uniform_parsing():
    code = """
    layout(set = 0, binding = 0) uniform atomic_uint counter;
    uniform atomic_uint fallbackCounter;
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]
    uniform = ast.global_variables[1]

    assert isinstance(layout, LayoutNode)
    assert layout.data_type == "atomic_uint"
    assert layout.variable_name == "counter"
    assert isinstance(uniform, UniformNode)
    assert uniform.vtype == "atomic_uint"
    assert uniform.name == "fallbackCounter"


def test_standalone_sampler_uniform_parsing():
    code = """
    layout(set = 0, binding = 0) uniform sampler compareSampler;
    uniform sampler samplers[4];
    void main() {}
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    layout = ast.global_variables[0]
    uniform = ast.global_variables[1]

    assert isinstance(layout, LayoutNode)
    assert layout.data_type == "sampler"
    assert layout.variable_name == "compareSampler"
    assert isinstance(uniform, UniformNode)
    assert uniform.vtype == "sampler"
    assert uniform.name == "samplers[4]"


def test_standalone_function_call_statement_consumes_semicolon():
    code = """
    void helper() {
    }

    void main() {
        helper();
        int value = 1;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    call = ast.functions[1].body[0]
    following_statement = ast.functions[1].body[1]

    assert isinstance(call, FunctionCallNode)
    assert call.name == "helper"
    assert isinstance(following_statement, AssignmentNode)


def test_expression_statement_parses_unary_prefix_forms():
    code = """
    void main() {
        int value = 1;
        bool enabled;
        +value;
        !enabled;
        ++value;
        --value;
        (1 + 2);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    positive = ast.functions[0].body[2]
    logical_not = ast.functions[0].body[3]
    increment = ast.functions[0].body[4]
    decrement = ast.functions[0].body[5]
    parenthesized = ast.functions[0].body[6]

    assert isinstance(positive, UnaryOpNode)
    assert positive.op == "+"
    assert isinstance(logical_not, UnaryOpNode)
    assert logical_not.op == "!"
    assert isinstance(increment, UnaryOpNode)
    assert increment.op == "PRE_INCREMENT"
    assert isinstance(decrement, UnaryOpNode)
    assert decrement.op == "PRE_DECREMENT"
    assert parenthesized.op == "+"


def test_member_call_statement_is_not_parsed_as_variable():
    code = """
    void main() {
        image.store(1, value);
        int value = 1;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    call = ast.functions[0].body[0]
    following_statement = ast.functions[0].body[1]

    assert isinstance(call, MethodCallNode)
    assert call.object.name == "image"
    assert call.method == "store"
    assert call.args[0] == "1"
    assert isinstance(call.args[1], VariableNode)
    assert call.args[1].name == "value"
    assert isinstance(following_statement, AssignmentNode)


def test_member_call_expression_and_array_receiver_parse_structurally():
    code = """
    void main() {
        int value = object.method();
        objects[0].store(value);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    assignment = ast.functions[0].body[0]
    call = ast.functions[0].body[1]

    assert isinstance(assignment, AssignmentNode)
    assert isinstance(assignment.right, MethodCallNode)
    assert assignment.right.object.name == "object"
    assert assignment.right.method == "method"
    assert isinstance(call, MethodCallNode)
    assert isinstance(call.object, ArrayAccessNode)
    assert call.object.array.name == "objects"
    assert call.object.index == "0"
    assert call.method == "store"


def test_member_access_assignment_lhs_is_structured():
    code = """
    void main() {
        color.r = 1.0;
        color.g += 0.5;
        objects[0].field = value;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    scalar_assignment = ast.functions[0].body[0]
    compound_assignment = ast.functions[0].body[1]
    array_member_assignment = ast.functions[0].body[2]

    assert isinstance(scalar_assignment, AssignmentNode)
    assert isinstance(scalar_assignment.left, MemberAccessNode)
    assert scalar_assignment.left.object.name == "color"
    assert scalar_assignment.left.member == "r"

    assert isinstance(compound_assignment, BinaryOpNode)
    assert isinstance(compound_assignment.left, MemberAccessNode)
    assert compound_assignment.left.object.name == "color"
    assert compound_assignment.left.member == "g"
    assert compound_assignment.op == "+="

    assert isinstance(array_member_assignment, AssignmentNode)
    assert isinstance(array_member_assignment.left, MemberAccessNode)
    assert isinstance(array_member_assignment.left.object, ArrayAccessNode)
    assert array_member_assignment.left.object.array.name == "objects"
    assert array_member_assignment.left.object.index == "0"
    assert array_member_assignment.left.member == "field"


def test_typed_local_array_declaration_preserves_suffix():
    code = """
    void main() {
        float weights[4];
        weights[0] = 1.0;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    declaration = ast.functions[0].body[0]
    assignment = ast.functions[0].body[1]

    assert isinstance(declaration, VariableNode)
    assert declaration.vtype == "float"
    assert declaration.name == "weights[4]"
    assert isinstance(assignment, AssignmentNode)
    assert isinstance(assignment.left, ArrayAccessNode)


def test_custom_type_local_declaration_preserves_type_and_name():
    code = """
    struct VertexOutput {
        vec4 position;
    };

    void main() {
        VertexOutput output;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    declaration = ast.functions[0].body[0]

    assert isinstance(declaration, VariableNode)
    assert declaration.vtype == "VertexOutput"
    assert declaration.name == "output"


def test_const_local_declaration_parsing():
    code = """
    void main() {
        const float scale = 1.0;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    declaration = ast.functions[0].body[0]

    assert isinstance(declaration, AssignmentNode)
    assert isinstance(declaration.left, VariableNode)
    assert declaration.left.vtype == "const float"
    assert declaration.left.name == "scale"
    assert declaration.right == "1.0"


def test_const_global_declaration_parsing():
    code = """
    const int MAX_LIGHTS = 4;
    void main() {}
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    declaration = ast.global_variables[0]

    assert isinstance(declaration, AssignmentNode)
    assert isinstance(declaration.left, VariableNode)
    assert declaration.left.vtype == "const int"
    assert declaration.left.name == "MAX_LIGHTS"
    assert declaration.right == "4"


def test_bitwise_and_shift_precedence_parsing():
    code = """
    void main() {
        int a = 1;
        int b = 2;
        int c = 3;
        int value = a & b << c;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    assignment = ast.functions[0].body[3]
    expression = assignment.right

    assert isinstance(expression, BinaryOpNode)
    assert expression.op == "&"
    assert isinstance(expression.right, BinaryOpNode)
    assert expression.right.op == "<<"


def test_logical_and_keeps_equality_operands_grouped():
    code = """
    void main() {
        bool selected = result1 == 8u && result2 == 2u;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    assignment = ast.functions[0].body[0]
    expression = assignment.right

    assert isinstance(expression, BinaryOpNode)
    assert expression.op == "&&"
    assert isinstance(expression.left, BinaryOpNode)
    assert expression.left.op == "=="
    assert isinstance(expression.right, BinaryOpNode)
    assert expression.right.op == "=="


def test_unknown_identifier_statement_is_rejected_instead_of_dropped():
    code = """
    void main() {
        image.store 1;
        int value = 1;
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Unexpected token after identifier image"):
        parse_code(tokens)


@pytest.mark.parametrize(
    "statement",
    [
        "int value = 1 trailing;",
        "value += 1 trailing;",
        "value & 1 trailing;",
    ],
)
def test_malformed_identifier_statement_trailing_tokens_are_rejected(statement):
    code = f"""
    void main() {{
        int value = 1;
        {statement}
        int next = 2;
    }}
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Expected SEMICOLON, got IDENTIFIER"):
        parse_code(tokens)


def test_expression_statement_requires_semicolon():
    code = """
    void main() {
        (1 + 2)
        int value = 1;
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Expected SEMICOLON"):
        parse_code(tokens)


if __name__ == "__main__":
    pytest.main()

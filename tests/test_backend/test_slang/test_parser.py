from typing import List

import pytest

from crosstl.backend.slang import SlangCrossGLCodeGen, SlangLexer, SlangParser
from crosstl.backend.slang.SlangAst import (
    ArrayAccessNode,
    AssignmentNode,
    AssociatedTypeNode,
    BinaryOpNode,
    BreakNode,
    CallNode,
    CaseNode,
    CastNode,
    ContinueNode,
    DiscardNode,
    DoWhileNode,
    EnumNode,
    ExtensionNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    GenericConstraintNode,
    IfNode,
    InitializerListNode,
    InterfaceNode,
    MemberAccessNode,
    MethodCallNode,
    ParenthesizedCommaNode,
    ReturnNode,
    SwitchNode,
    TernaryOpNode,
    TypedefNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
    WhileNode,
)


def parse_code(tokens: List):
    """Test the parser
    Args:
        tokens (List): The list of tokens generated from the lexer

    Returns:
        ASTNode: The abstract syntax tree generated from the code
    """
    parser = SlangParser(tokens)
    return parser.parse()


def tokenize_code(code: str) -> List:
    lexer = SlangLexer(code)
    return lexer.tokenize()


def find_function(ast, name: str):
    for function in ast.functions:
        if function.name == name:
            return function
    raise AssertionError(f"Function {name} not found")


def test_struct_parsing():
    code = """
    struct AssembledVertex
    {
    float3	position : POSITION;
    };
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Struct parsing not implemented.")


def test_forward_struct_declaration_from_generated_conformance_sample():
    # Source: shader-slang/slang@52339028a2aa703271533454c6b9528a534bac31
    # docs/generated/tests/conformance/types-struct/struct-no-body-decl.slang
    code = """
    struct ForwardDeclared;

    struct ForwardDeclared
    {
        int x;
    }

    void main()
    {
        ForwardDeclared fd;
        fd.x = 55;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    forward, definition = ast.structs
    assert forward.name == "ForwardDeclared"
    assert forward.members == []
    assert forward.is_forward_declaration is True
    assert definition.name == "ForwardDeclared"
    assert definition.is_forward_declaration is False
    assert [(member.vtype, member.name) for member in definition.members] == [
        ("int", "x")
    ]


def test_compound_import_parsing():
    code = "import MyApp.Shadowing;\n"

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [node.module_name for node in ast.imports] == ["MyApp.Shadowing"]


def test_string_import_path_parsing():
    code = 'import "dir/file-name.slang";\n'

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [node.module_name for node in ast.imports] == ["dir/file-name.slang"]
    assert ast.global_vars == []


def test_module_include_declarations_do_not_parse_as_globals():
    code = """
    module scene;
    __include "scene-helpers";
    implementing scene;
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert ast.modules == ["scene"]
    assert ast.includes == ["scene-helpers"]
    assert ast.implementing_modules == ["scene"]
    assert ast.global_vars == []


def test_qualified_module_declaration_from_core_module_metadata():
    # Source: official Slang standard-library core metadata emits
    # SLANG_RAW("public module core;\n").
    code = """
    public module core;
    typedef half float16_t;
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert ast.modules == ["core"]
    assert ast.global_vars == []
    assert [(node.original_type, node.new_type) for node in ast.typedefs] == [
        ("half", "float16_t")
    ]


def test_identifier_include_path_normalizes_slang_file_lookup_semantics():
    # Slang modules docs: identifier-token file paths map dots to path
    # separators and underscores to hyphens, e.g. dir.file_name -> dir/file-name.
    code = """
    module utils;
    __include utils.tonemap_filter;
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert ast.modules == ["utils"]
    assert ast.includes == ["utils/tonemap-filter"]
    assert ast.global_vars == []


def test_string_module_declarations_from_precompiled_module_tests():
    code = """
    module "precompiled-module-imported";
    implementing "precompiled-module-imported";
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert ast.modules == ["precompiled-module-imported"]
    assert ast.implementing_modules == ["precompiled-module-imported"]
    assert ast.global_vars == []


def test_glsl_layout_qualifiers_and_uniform_blocks_from_libretro_shader():
    code = """
    layout(push_constant) uniform Push
    {
        vec4 SourceSize;
        float exc;
    } params;

    layout(std140, set = 0, binding = 0) uniform UBO
    {
        mat4 MVP;
    } global;

    layout(location = 0) in vec4 Position;
    layout(location = 0) out vec2 vTexCoord;
    layout(set = 0, binding = 2) uniform sampler2D Source;

    void main()
    {
        vTexCoord = Position.xy;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [buffer.name for buffer in ast.cbuffers] == ["Push", "UBO"]
    assert ast.cbuffers[0].qualifiers == ["layout(push_constant)", "uniform"]
    assert [(member.vtype, member.name) for member in ast.cbuffers[0].members] == [
        ("vec4", "SourceSize"),
        ("float", "exc"),
    ]
    assert [
        (instance.vtype, instance.name) for instance in ast.cbuffers[0].instances
    ] == [("Push", "params")]
    assert ast.cbuffers[1].qualifiers == ["layout(std140,set=0,binding=0)", "uniform"]
    assert [
        (variable.vtype, variable.name, variable.qualifiers)
        for variable in ast.global_vars
    ] == [
        ("vec4", "Position", ["layout(location=0)", "in"]),
        ("vec2", "vTexCoord", ["layout(location=0)", "out"]),
        ("sampler2D", "Source", ["layout(set=0,binding=2)", "uniform"]),
    ]


def test_glsl_storage_buffer_block_from_allow_glsl_parsing():
    code = """
    buffer MyBlockName {
        vec4 result;
    } outputBuffer;

    void main() {
        outputBuffer.result = vec4(1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    block = ast.cbuffers[0]
    instance = block.instances[0]
    assignment = find_function(ast, "main").body[0]

    assert block.name == "MyBlockName"
    assert block.glsl_block_kind == "buffer"
    assert block.qualifiers == ["buffer"]
    assert [(member.vtype, member.name) for member in block.members] == [
        ("vec4", "result")
    ]
    assert (instance.vtype, instance.name, instance.glsl_block_kind) == (
        "MyBlockName",
        "outputBuffer",
        "buffer",
    )
    assert isinstance(assignment.left, MemberAccessNode)
    assert assignment.left.object.name == "outputBuffer"
    assert assignment.left.member == "result"


def test_struct_array_member_declarator_parsing():
    code = """
    struct Cluster {
        float weights[4];
        float4 colors[2][3] : COLOR0;
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    members = ast.structs[0].members

    assert [
        (member.vtype, member.name, member.array_sizes, member.semantic)
        for member in members
    ] == [
        ("float", "weights", ["4"], None),
        ("float4", "colors", ["2", "3"], "COLOR0"),
    ]


def test_struct_comma_member_declarators_from_ray_tracing_example():
    code = """
    struct Uniforms {
        float screenWidth, screenHeight;
        float focalLength, frameHeight;
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    members = ast.structs[0].members

    assert [(member.vtype, member.name) for member in members] == [
        ("float", "screenWidth"),
        ("float", "screenHeight"),
        ("float", "focalLength"),
        ("float", "frameHeight"),
    ]


def test_pervertex_struct_member_qualifier_from_barycentric_tests():
    code = """
    struct Input
    {
        pervertex float4 color : COLOR;
        noperspective float3 baryNoPerspective : SV_Barycentrics;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    members = ast.structs[0].members

    assert [
        (member.qualifiers, member.vtype, member.name, member.semantic)
        for member in members
    ] == [
        (["pervertex"], "float4", "color", "COLOR"),
        (
            ["noperspective"],
            "float3",
            "baryNoPerspective",
            "SV_Barycentrics",
        ),
    ]


def test_visibility_qualified_struct_from_mlp_training_adam_sample():
    code = """
    public struct AdamState
    {
        internal NFloat mean;
        internal NFloat variance;
        internal int iteration;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    struct = ast.structs[0]

    assert struct.name == "AdamState"
    assert struct.qualifiers == ["public"]
    assert [
        (member.qualifiers, member.vtype, member.name) for member in struct.members
    ] == [
        (["internal"], "NFloat", "mean"),
        (["internal"], "NFloat", "variance"),
        (["internal"], "int", "iteration"),
    ]


def test_static_const_struct_field_initializer_from_mlp_training_adam_sample():
    # Source: shader-slang/slang@52339028a2aa703271533454c6b9528a534bac31
    # examples/mlp-training/adam.slang
    code = """
    public struct AdamOptimizer
    {
        public static const NFloat beta1 = 0.9h;
        public static const NFloat epsilon = 1e-7h;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    fields = ast.structs[0].members

    assert [
        (field.qualifiers, field.vtype, field.name, field.value) for field in fields
    ] == [
        (["public", "static", "const"], "NFloat", "beta1", "0.9h"),
        (["public", "static", "const"], "NFloat", "epsilon", "1e-7h"),
    ]


def test_struct_property_declaration_parses_as_property_member():
    code = """
    struct Box {
        public property int value;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    member = ast.structs[0].members[0]

    assert member.name == "value"
    assert member.vtype == "int"
    assert member.qualifiers == ["public"]
    assert member.is_property is True
    assert member.property_accessors == {}


def test_struct_property_getter_and_setter_preserve_accessor_bodies():
    code = """
    struct Box {
        int _v;
        property int value {
            get { return _v; }
            set { _v = newValue * 2; }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    prop = ast.structs[0].members[1]

    assert prop.name == "value"
    assert prop.vtype == "int"
    assert prop.is_property is True
    assert set(prop.property_accessors) == {"get", "set"}
    getter = prop.property_accessors["get"][0]
    setter = prop.property_accessors["set"][0]

    assert isinstance(getter, ReturnNode)
    assert getter.value.name == "_v"
    assert isinstance(setter, AssignmentNode)
    assert setter.operator == "="
    assert setter.left.name == "_v"
    assert setter.right.left.name == "newValue"


def test_name_first_struct_property_from_generated_interface_sample():
    code = """
    struct IntProperty
    {
        int _val;
        property prop : int { get { return _val; } }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    prop = ast.structs[0].members[1]

    assert prop.name == "prop"
    assert prop.vtype == "int"
    assert prop.is_property is True
    assert set(prop.property_accessors) == {"get"}
    assert prop.property_accessors["get"][0].value.name == "_val"


def test_struct_method_body_parsing_from_official_example():
    code = """
    struct Primitive {
        float4 data0;

        float3 getNormal() {
            return data0.xyz;
        }
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    struct = ast.structs[0]

    assert [(member.vtype, member.name) for member in struct.members] == [
        ("float4", "data0")
    ]
    assert len(struct.methods) == 1

    method = struct.methods[0]
    assert isinstance(method, FunctionNode)
    assert method.return_type == "float3"
    assert method.name == "getNormal"
    assert method.params == []
    assert not method.is_declaration
    assert isinstance(method.body[0], ReturnNode)
    assert isinstance(method.body[0].value, MemberAccessNode)
    assert method.body[0].value.member == "xyz"


def test_struct_init_method_parsing_from_shader_toy_sample():
    code = """
    struct mat2
    {
        float2x2 data;

        __init(float e00, float e01, float e10, float e11)
        {
            data = float2x2(e00, e01, e10, e11);
        }
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    struct = ast.structs[0]
    constructor = struct.methods[0]

    assert constructor.name == "__init"
    assert constructor.return_type == "void"
    assert not constructor.is_declaration
    assert [(param.vtype, param.name) for param in constructor.params] == [
        ("float", "e00"),
        ("float", "e01"),
        ("float", "e10"),
        ("float", "e11"),
    ]
    assert isinstance(constructor.body[0], AssignmentNode)


def test_core_meta_generic_extension_constructor_constraints_parse():
    # Reduced from shader-slang/slang@564ac9f050d6569efd773e2f74e7d067a4e54baa
    # source/slang/core.meta.slang generic vector conversion constructors.
    code = """
    extension vector<ToType,N>
    {
        __implicit_conversion(constraint)
        __intrinsic_op(BuiltinCast)
        __init<FromType>(vector<FromType,N> value)
            where ToType(FromType) implicit;

        __implicit_conversion(constraint+)
        [__unsafeForceInlineEarly]
        [__readNone]
        [TreatAsDifferentiable]
        __init<FromType>(FromType value) where ToType(FromType) implicit
        {
            this = __builtin_cast<vector<ToType,N>>(
                vector<FromType,N>(value));
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    extension = ast.extensions[0]
    signature, definition = extension.methods

    assert extension.extended_type == "vector<ToType, N>"
    assert [
        (method.name, method.generic_parameters) for method in extension.methods
    ] == [
        ("__init", "<FromType>"),
        ("__init", "<FromType>"),
    ]
    assert [(param.vtype, param.name) for param in signature.params] == [
        ("vector<FromType, N>", "value")
    ]
    assert [
        (constraint.parameter, constraint.relation, constraint.constraint_type)
        for constraint in signature.generic_constraints
    ] == [("ToType", "implicit", "FromType")]
    assert signature.is_declaration is True
    assert definition.is_declaration is False
    assert isinstance(definition.body[0], AssignmentNode)


def test_ray_payload_access_semantics_from_ray_tracing_sample():
    code = """
    [raypayload] struct RayPayload
    {
        float4 color : read(caller) : write(caller, closesthit, miss);
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    payload = ast.structs[0]
    color = payload.members[0]

    assert color.name == "color"
    assert color.semantic == "read(caller):write(caller,closesthit,miss)"


def test_if_parsing():
    code = """
    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        if (assembledVertex.color.r > 0.5) {
            output.out_position = assembledVertex.color;
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("if parsing not implemented.")


def test_string_literals_and_braceless_if_parsing():
    code = r"""
    void computeMain(uint tid) {
        if (tid > 0)
            return;
        println("hello from thread", tid);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    body = find_function(ast, "computeMain").body

    assert isinstance(body[0], IfNode)
    assert isinstance(body[0].if_body[0], ReturnNode)
    assert isinstance(body[1], FunctionCallNode)
    assert body[1].args[0] == '"hello from thread"'


def test_for_parsing():
    code = """
    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        for (int i = 0; i < 10; i=i+1) {
            output.out_position += assembledVertex.position;
        }

        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("for parsing not implemented.")


def test_for_initializer_parses_uint_and_bool_declarations():
    code = """
    void main(){
        for (uint i = 0; i < 4; i = i + 1) {
        }
        for (bool done = false; done == false; done = true) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    uint_loop = function.body[0]
    bool_loop = function.body[1]

    assert isinstance(uint_loop, ForNode)
    assert isinstance(uint_loop.init, AssignmentNode)
    assert uint_loop.init.left.vtype == "uint"
    assert uint_loop.init.left.name == "i"

    assert isinstance(bool_loop, ForNode)
    assert isinstance(bool_loop.init, AssignmentNode)
    assert bool_loop.init.left.vtype == "bool"
    assert bool_loop.init.left.name == "done"


def test_increment_decrement_expression_parsing():
    code = """
    void main(){
        for (int i = 0; i < 4; i++) {
        }
        j--;
        ++k;
        --items[index];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    body = find_function(ast, "main").body
    loop = body[0]

    assert isinstance(loop, ForNode)
    assert isinstance(loop.update, UnaryOpNode)
    assert loop.update.op == "POST_INCREMENT"
    assert loop.update.operand.name == "i"

    assert isinstance(body[1], UnaryOpNode)
    assert body[1].op == "POST_DECREMENT"
    assert body[1].operand.name == "j"

    assert isinstance(body[2], UnaryOpNode)
    assert body[2].op == "PRE_INCREMENT"
    assert body[2].operand.name == "k"

    assert isinstance(body[3], UnaryOpNode)
    assert body[3].op == "PRE_DECREMENT"
    assert isinstance(body[3].operand, ArrayAccessNode)


def test_postfix_update_rejects_call_target():
    code = """
    void main(){
        getValue()++;
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="Invalid postfix update target"):
        parse_code(tokens)


def test_for_initializer_uses_shared_declaration_parsing():
    code = """
    void main(Buffer<float> source){
        for (const uint i = 0; i < 4; i = i + 1) {
        }
        for (Counter cursor = Counter(0); cursor.value < 4; cursor.value += 1) {
        }
        for (Buffer<float> view = source; view.count < 4; view.count += 1) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loops = find_function(ast, "main").body

    qualified_loop = loops[0]
    custom_loop = loops[1]
    generic_loop = loops[2]

    assert isinstance(qualified_loop, ForNode)
    assert qualified_loop.init.left.qualifiers == ["const"]
    assert qualified_loop.init.left.vtype == "uint"

    assert isinstance(custom_loop, ForNode)
    assert custom_loop.init.left.vtype == "Counter"
    assert custom_loop.init.left.name == "cursor"

    assert isinstance(generic_loop, ForNode)
    assert generic_loop.init.left.vtype == "Buffer<float>"
    assert generic_loop.init.left.name == "view"


def test_for_optional_clauses_parse_as_empty_slots():
    code = """
    void main(){
        for (; ; ) {
            break;
        }
        for (int i = 0; ; ) {
            break;
        }
        for (; i < 4; ) {
            break;
        }
        for (; ; i = i + 1) {
            break;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    loops = find_function(ast, "main").body

    empty_loop = loops[0]
    init_only_loop = loops[1]
    condition_only_loop = loops[2]
    update_only_loop = loops[3]

    assert isinstance(empty_loop, ForNode)
    assert empty_loop.init is None
    assert empty_loop.condition is None
    assert empty_loop.update is None

    assert isinstance(init_only_loop.init, AssignmentNode)
    assert init_only_loop.condition is None
    assert init_only_loop.update is None

    assert condition_only_loop.init is None
    assert isinstance(condition_only_loop.condition, BinaryOpNode)
    assert condition_only_loop.update is None

    assert update_only_loop.init is None
    assert update_only_loop.condition is None
    assert isinstance(update_only_loop.update, AssignmentNode)


def test_break_continue_statement_parsing():
    code = """
    void main(){
        for (int i = 0; i < 4; i = i + 1) {
            break;
            continue;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    loop = function.body[0]

    assert isinstance(loop, ForNode)
    assert isinstance(loop.body[0], BreakNode)
    assert isinstance(loop.body[1], ContinueNode)


def test_discard_statement_parsing():
    code = """
    float4 main(float alpha) {
        if (alpha < 0.5) {
            discard;
        }
        return float4(1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    discard = find_function(ast, "main").body[0].if_body[0]

    assert isinstance(discard, DiscardNode)


def test_switch_statement_parsing():
    code = """
    void main() {
        int mode = 0;
        switch (mode) {
            case 0:
                mode = 1;
                break;
            default:
                mode = 2;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    switch = function.body[1]

    assert isinstance(switch, SwitchNode)
    assert switch.expression.name == "mode"
    assert len(switch.cases) == 1
    assert isinstance(switch.cases[0], CaseNode)
    assert switch.cases[0].value == "0"
    assert isinstance(switch.cases[0].body[0], AssignmentNode)
    assert isinstance(switch.cases[0].body[1], BreakNode)
    assert len(switch.default_case) == 1
    assert isinstance(switch.default_case[0], AssignmentNode)


def test_switch_default_before_case_preserves_order_metadata():
    code = """
    void main() {
        int mode = 0;
        switch (mode) {
            default:
                mode = 2;
            case 0:
                mode = 1;
                break;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    switch = find_function(ast, "main").body[1]

    assert len(switch.cases) == 1
    assert len(switch.default_case) == 1
    assert [case.value for case in switch.ordered_cases] == [None, "0"]
    assert switch.ordered_cases[0].body is switch.default_case


def test_switch_rejects_duplicate_default_labels():
    code = """
    void main() {
        int mode = 0;
        switch (mode) {
            default:
                mode = 2;
            case 0:
                mode = 1;
            default:
                mode = 3;
        }
    }
    """

    tokens = tokenize_code(code)
    with pytest.raises(SyntaxError, match="duplicate default"):
        parse_code(tokens)


def test_switch_case_braced_blocks_and_empty_statements_parse():
    code = """
    int chooseFace(int face)
    {
        switch (face)
        {
        case 0: {
            int selected = 1;
            break;
        };
        default: {
            break;
        };
        }
        return face;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "chooseFace")
    switch = function.body[0]

    assert isinstance(switch, SwitchNode)
    assert isinstance(switch.cases[0].body[0], AssignmentNode)
    assert isinstance(switch.cases[0].body[1], BreakNode)
    assert isinstance(switch.default_case[0], BreakNode)


def test_lambda_expression_parsing():
    code = """
    void main() {
        float folded = fold(values, 0, (int acc, int x) => (acc + x));
        float mapped = map(colors, (float3 color) => { return color; });
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    folded = function.body[0]
    mapped = function.body[1]

    folded_lambda = folded.right.args[2]
    assert isinstance(folded_lambda, FunctionCallNode)
    assert folded_lambda.name == "lambda"
    assert folded_lambda.args[0].vtype == "int"
    assert folded_lambda.args[0].name == "acc"
    assert folded_lambda.args[1].vtype == "int"
    assert folded_lambda.args[1].name == "x"
    assert isinstance(folded_lambda.args[2], BinaryOpNode)

    mapped_lambda = mapped.right.args[1]
    assert mapped_lambda.name == "lambda"
    assert mapped_lambda.args[0].vtype == "float3"
    assert mapped_lambda.args[0].name == "color"
    assert mapped_lambda.args[1] == "{ return color; }"


def test_generic_type_receiver_expression_parsing():
    code = """
    [TorchEntryPoint]
    export __extern_cpp int main() {
        var result = TorchTensor<float>.alloc(Shape(1));
        let count = result.numel();
        let vec = coopVecLoad<4>(input);
        let rs = f.eval<DataTrait0>(1.0);
        return 0;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = ast.exports[0].item
    result = function.body[0]
    count = function.body[1]
    vector_load = function.body[2]
    generic_method = function.body[3]

    assert isinstance(result, AssignmentNode)
    assert result.left.vtype == "var"
    assert result.left.name == "result"
    assert isinstance(result.right, MethodCallNode)
    assert result.right.object.name == "TorchTensor<float>"
    assert result.right.method == "alloc"
    assert result.right.args[0].name == "Shape"

    assert isinstance(count, AssignmentNode)
    assert count.left.vtype == "let"
    assert count.left.name == "count"
    assert isinstance(count.right, MethodCallNode)
    assert count.right.object.name == "result"
    assert count.right.method == "numel"

    assert isinstance(vector_load.right, FunctionCallNode)
    assert vector_load.right.name == "coopVecLoad<4>"
    assert vector_load.right.args[0].name == "input"

    assert isinstance(generic_method.right, MethodCallNode)
    assert generic_method.right.object.name == "f"
    assert generic_method.right.method == "eval<DataTrait0>"
    assert generic_method.right.args[0] == "1.0"


def test_modern_typed_let_var_declarations_from_official_docs():
    code = """
    let x : int = 7;
    var y : float = 9.0;

    struct Person
    {
        var age : int;
        float height;
    }

    void main()
    {
        let localCount : int = x;
        var localValue : float;
        var inferred = y;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    x = ast.global_vars[0]
    y = ast.global_vars[1]
    person = ast.structs[0]
    body = find_function(ast, "main").body

    assert isinstance(x, AssignmentNode)
    assert x.left.vtype == "int"
    assert x.left.name == "x"
    assert x.left.storage_modifier == "let"
    assert x.right == "7"

    assert isinstance(y, AssignmentNode)
    assert y.left.vtype == "float"
    assert y.left.name == "y"
    assert y.left.storage_modifier == "var"
    assert y.right == "9.0"

    assert [(member.vtype, member.name) for member in person.members] == [
        ("int", "age"),
        ("float", "height"),
    ]
    assert person.members[0].storage_modifier == "var"

    assert isinstance(body[0], AssignmentNode)
    assert body[0].left.vtype == "int"
    assert body[0].left.name == "localCount"
    assert body[0].left.storage_modifier == "let"
    assert isinstance(body[1], VariableNode)
    assert body[1].vtype == "float"
    assert body[1].name == "localValue"
    assert body[1].storage_modifier == "var"
    assert isinstance(body[2], AssignmentNode)
    assert body[2].left.vtype == "var"
    assert body[2].left.name == "inferred"
    assert body[2].left.storage_modifier == "var"


def test_generic_function_declaration_after_name_parsing():
    code = """
    float GetRayT<let RAY_QUERY_FLAGS: uint>(uint rayInlineFlags)
    {
        return 0.0;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "GetRayT")

    assert function.return_type == "float"
    assert getattr(function, "is_generic", False)
    assert function.generic_parameters == "<letRAY_QUERY_FLAGS:uint>"
    assert function.params[0].vtype == "uint"
    assert function.params[0].name == "rayInlineFlags"
    assert isinstance(function.body[0], ReturnNode)


def test_generic_struct_declaration_after_name_parsing():
    code = """
    struct GenericStruct<T, let N: int>
    {
        T value;
        float weights[N];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    struct = ast.structs[0]

    assert struct.name == "GenericStruct"
    assert struct.generic_parameters == "<T, letN:int>"
    assert [(member.vtype, member.name) for member in struct.members] == [
        ("T", "value"),
        ("float", "weights"),
    ]
    assert struct.members[1].array_sizes[0].name == "N"


def test_generic_declaration_prefix_with_associated_type_defaults():
    code = """
    interface IData {}
    interface IElement
    {
        associatedtype DataType : IData;
    }

    __generic<T : IElement, U : IData = T.DataType>
    U getDefaultAssociatedType()
    {
        return U();
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = ast.functions[0]

    assert function.name == "getDefaultAssociatedType"
    assert function.return_type == "U"
    assert function.is_generic
    assert isinstance(function.body[0], ReturnNode)


def test_interface_struct_and_extension_conformance_metadata_parsing():
    code = """
    interface IFoo {
        int foo();
    }

    struct MyType : IFoo {
        int value;
    };

    extension MyType : IBar {
        int bar() {
            return 1;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    interface = ast.interfaces[0]
    assert isinstance(interface, InterfaceNode)
    assert interface.name == "IFoo"
    assert len(interface.methods) == 1
    assert interface.methods[0].return_type == "int"
    assert interface.methods[0].name == "foo"
    assert interface.methods[0].is_declaration

    struct = ast.structs[0]
    assert struct.name == "MyType"
    assert struct.conformances == ["IFoo"]

    extension = ast.extensions[0]
    assert isinstance(extension, ExtensionNode)
    assert extension.extended_type == "MyType"
    assert extension.conformances == ["IBar"]
    assert len(extension.methods) == 1
    assert extension.methods[0].name == "bar"
    assert not extension.methods[0].is_declaration


def test_interface_associated_type_requirement_from_model_viewer_sample():
    code = """
    interface IMaterial
    {
        associatedtype BRDF : IBRDF;
        BRDF prepare(SurfaceGeometry geometry);
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    interface = ast.interfaces[0]

    assert len(interface.associated_types) == 1
    associated_type = interface.associated_types[0]
    assert isinstance(associated_type, AssociatedTypeNode)
    assert associated_type.name == "BRDF"
    assert associated_type.constraint_type == "IBRDF"
    assert associated_type.target_type is None
    assert len(interface.methods) == 1
    assert interface.methods[0].return_type == "BRDF"
    assert interface.methods[0].name == "prepare"


def test_interface_conformance_and_attributed_methods_parse():
    code = """
    interface IDifferentiable {}

    interface IParams : IDifferentiable
    {
        [Differentiable]
        float get();
    }

    interface IFoo
    {
        associatedtype Params : IParams;

        [BackwardDifferentiable]
        static Params decode<let N : uint>(no_diff float x);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    params = ast.interfaces[1]
    foo = ast.interfaces[2]
    method = foo.methods[0]

    assert params.conformances == ["IDifferentiable"]
    assert params.methods[0].attributes == [{"name": "Differentiable", "arguments": []}]
    assert foo.associated_types[0].constraint_type == "IParams"
    assert method.qualifiers == ["static"]
    assert method.generic_parameters == "<letN:uint>"
    assert method.attributes == [{"name": "BackwardDifferentiable", "arguments": []}]
    assert method.params[0].qualifiers == ["no_diff"]


def test_typealias_declarations_from_shader_toy_and_mlp_vec_samples():
    code = """
    typealias vec2 = float2;

    public struct MLVec<int N> : IDifferentiable
    {
        public typealias Differential = MLVec<N>;
        public CoopVec<NFloat, N> data;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    typedef = ast.typedefs[0]
    struct = ast.structs[0]
    struct_typedef = struct.typedefs[0]

    assert typedef.original_type == "float2"
    assert typedef.new_type == "vec2"
    assert struct_typedef.original_type == "MLVec<N>"
    assert struct_typedef.new_type == "Differential"
    assert struct_typedef.qualifiers == ["public"]
    assert [(member.vtype, member.name) for member in struct.members] == [
        ("CoopVec<NFloat, N>", "data")
    ]


def test_generic_typealias_declaration_from_slang_neural_modules():
    # Source: shader-slang/slang source/standard-modules/neural/hash-function.slang
    # and source/standard-modules/neural/WaveMatrix.slang declare generic
    # aliases before the `=` target type.
    code = """
    implementing neural;

    internal typealias uvec<int Dim> = Array<uint32_t, Dim>;
    internal typealias MatrixA<int Rows, int Cols> =
        WaveMatrix<half, linalg.CoopMatMatrixUse.MatrixA, Rows, Cols>;
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert ast.implementing_modules == ["neural"]
    assert [(node.original_type, node.new_type) for node in ast.typedefs] == [
        ("Array<uint32_t, Dim>", "uvec<intDim>"),
        (
            "WaveMatrix<half, linalg.CoopMatMatrixUse.MatrixA, Rows, Cols>",
            "MatrixA<intRows, intCols>",
        ),
    ]
    assert ast.typedefs[0].qualifiers == ["internal"]


def test_local_typealias_declaration_parses_in_function_body():
    code = """
    bool testVector(uint4 mask)
    {
        typealias GVec = vector<uint, 4>;
        GVec value = GVec(0);
        return true;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    body = find_function(ast, "testVector").body
    typedef = body[0]

    assert isinstance(typedef, TypedefNode)
    assert typedef.original_type == "vector<uint, 4>"
    assert typedef.new_type == "GVec"
    assert isinstance(body[1], AssignmentNode)
    assert body[1].left.vtype == "GVec"


def test_qualified_type_paths_parse_in_typedefs_structs_and_locals():
    code = """
    namespace example {
        struct Example {}
    }

    typedef float.Differential dfloat;

    struct Payload {
        example::Example value;
    };

    void computeMain() {
        A.Differential tangent = {0.2};
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    typedef = ast.typedefs[0]
    payload = ast.structs[0]
    local = ast.functions[0].body[0]

    assert typedef.original_type == "float.Differential"
    assert typedef.new_type == "dfloat"
    assert payload.members[0].vtype == "example::Example"
    assert payload.members[0].name == "value"
    assert isinstance(local, AssignmentNode)
    assert local.left.vtype == "A.Differential"
    assert local.left.name == "tangent"
    assert isinstance(local.right, InitializerListNode)


def test_module_level_require_capability_declaration_is_skipped():
    code = """
    __require_capability(cpp);

    void main() {
        return;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [function.name for function in ast.functions] == ["main"]


def test_module_level_using_namespace_declaration_is_skipped():
    code = """
    namespace N {
        int compute() { return 42; }
    }
    using namespace N;

    void main() {
        return;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [function.name for function in ast.functions] == ["main"]


def test_type_test_and_cast_expression_operators_parse():
    code = """
    struct MyInst {};

    MyInst get_inst() {
        return MyInst();
    }

    bool is_my_inst() {
        return get_inst() is MyInst;
    }

    MyInst cast_my_inst() {
        return get_inst() as MyInst;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    is_expr = find_function(ast, "is_my_inst").body[0].value
    as_expr = find_function(ast, "cast_my_inst").body[0].value

    assert isinstance(is_expr, BinaryOpNode)
    assert is_expr.op == "is"
    assert is_expr.right == "MyInst"
    assert isinstance(as_expr, BinaryOpNode)
    assert as_expr.op == "as"
    assert as_expr.right == "MyInst"


def test_operator_overload_declarations_from_official_samples_parse():
    code = """
    interface IArithmetic {}

    struct Vec2d
    {
        float x, y;
    };

    Vec2d operator+(Vec2d a, Vec2d b)
    {
        return {a.x + b.x, a.y + b.y};
    }

    T operator *<T>(T a, T b) where T : IArithmetic
    {
        return a;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    add = ast.functions[0]
    multiply = ast.functions[1]

    assert add.name == "operator+"
    assert [(param.vtype, param.name) for param in add.params] == [
        ("Vec2d", "a"),
        ("Vec2d", "b"),
    ]
    assert isinstance(add.body[0].value, InitializerListNode)
    assert multiply.name == "operator*"
    assert multiply.generic_parameters == "<T>"
    assert multiply.generic_constraints[0].constraint_type == "IArithmetic"


def test_struct_call_and_subscript_operator_methods_parse():
    code = """
    struct CubeContext
    {
        float sqr;

        void operator()(out float dx, float dOut)
        {
            dx = dOut * sqr;
        }

        float operator[](uint index)
        {
            return sqr;
        }
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    methods = ast.structs[0].methods

    assert [method.name for method in methods] == ["operator()", "operator[]"]
    assert methods[0].params[0].qualifiers == ["out"]
    assert methods[0].params[0].name == "dx"
    assert methods[1].params[0].vtype == "uint"
    assert methods[1].params[0].name == "index"


def test_struct_subscript_accessor_from_generated_conformance_sample_parse():
    # Source: shader-slang/slang@564ac9f050d6569efd773e2f74e7d067a4e54baa
    # docs/generated/tests/conformance/declarations/subscript-get-functional.slang
    code = """
    struct MyVec
    {
        float x, y;
        __subscript(int index) -> float
        {
            get { return index == 0 ? x : y; }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    method = ast.structs[0].methods[0]

    assert method.name == "operator[]"
    assert method.slang_name == "__subscript"
    assert method.return_type == "float"
    assert method.params[0].vtype == "int"
    assert method.params[0].name == "index"
    assert method.is_subscript is True
    assert list(method.property_accessors) == ["get"]
    assert method.body == method.property_accessors["get"]
    assert isinstance(method.body[0], ReturnNode)
    assert isinstance(method.body[0].value, TernaryOpNode)


def test_attributed_subscript_set_accessor_from_upstream_bug_sample_parse():
    # Source: shader-slang/slang@5230a81f2fe68afe5cb8d04a1b09d56476f6b960
    # tests/bugs/gh-4971.slang
    code = """
    struct Test {
        RWStructuredBuffer<int> val;
        __subscript(int x, int y)->int
        {
            get { return val[x * 3 + y]; }
            [nonmutating] set { val[x * 3 + y] = newValue; }
        }
    }
    Test test;

    [numthreads(1, 1, 1)]
    void computeMain()
    {
        test[0,0] = 1;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    method = ast.structs[0].methods[0]

    assert method.name == "operator[]"
    assert method.slang_name == "__subscript"
    assert [param.name for param in method.params] == ["x", "y"]
    assert list(method.property_accessors) == ["get", "set"]
    assert isinstance(method.property_accessors["get"][0], ReturnNode)
    assert isinstance(method.property_accessors["set"][0], AssignmentNode)
    assert method.property_accessors["set"][0].left.array.name == "val"
    assert method.property_accessors["set"][0].right.name == "newValue"


def test_multi_index_subscript_expressions_from_official_operator_sample_parse():
    code = """
    int test(S value, int x, int y)
    {
        value[] = 0;
        value[x] = 1;
        value[x, y] = value[1, 0];
        return value[x, y];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    body = find_function(ast, "test").body

    empty_subscript = body[0].left
    single_subscript = body[1].left
    multi_assignment = body[2]
    return_subscript = body[3].value

    assert isinstance(empty_subscript, ArrayAccessNode)
    assert empty_subscript.index is None

    assert isinstance(single_subscript, ArrayAccessNode)
    assert single_subscript.index.name == "x"

    assert isinstance(multi_assignment.left, ArrayAccessNode)
    assert isinstance(multi_assignment.left.index, ParenthesizedCommaNode)
    assert [expr.name for expr in multi_assignment.left.index.expressions] == [
        "x",
        "y",
    ]

    assert isinstance(multi_assignment.right, ArrayAccessNode)
    assert isinstance(multi_assignment.right.index, ParenthesizedCommaNode)
    assert multi_assignment.right.index.expressions == ["1", "0"]

    assert isinstance(return_subscript, ArrayAccessNode)
    assert isinstance(return_subscript.index, ParenthesizedCommaNode)


def test_func_extension_apply_declaration_parse():
    code = """
    float cube(float x) { return x * x * x; }

    struct CubeContext
    {
        float sqr;
        void operator()(out float dx, float dOut)
        {
            dx = dOut * 3.0 * sqr;
        }
    };

    __func_extension __apply(cube)(float x) -> Tuple<float, CubeContext>
    {
        let sqr = x * x;
        return makeTuple(sqr * x, CubeContext(sqr));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    extension = ast.functions[1]

    assert extension.name == "__apply(cube)"
    assert extension.return_type == "Tuple<float, CubeContext>"
    assert extension.qualifiers == ["__func_extension"]
    assert [(param.vtype, param.name) for param in extension.params] == [("float", "x")]


def test_generic_func_extension_member_method_declaration_parse():
    code = """
    struct MyVec<T : __BuiltinArithmeticType> : IDifferentiable
    {
        typedef MyVec<T> Differential;
        T x;
        T y;

        T lengthSquared() { return x * x + y * y; }
    };

    __func_extension<T : __BuiltinFloatingPointType>
    fwd_diff(MyVec<T>::lengthSquared)(DifferentialPair<MyVec<T>> self) -> DifferentialPair<T>
        where T.Differential == T
    {
        return diffPair(self.p.lengthSquared(), T(0));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    extension = ast.functions[0]

    assert extension.name == "fwd_diff(MyVec<T>::lengthSquared)"
    assert extension.return_type == "DifferentialPair<T>"
    assert extension.is_generic is True
    assert extension.generic_parameters == "<T:__BuiltinFloatingPointType>"
    assert extension.generic_constraints[0].parameter == "T.Differential"
    assert extension.generic_constraints[0].relation == "=="
    assert extension.generic_constraints[0].constraint_type == "T"


def test_generic_extension_typedef_and_specialized_function_value_parse():
    code = """
    struct MyVec<T : __BuiltinArithmeticType>
    {
        T x;
        T y;
        T lengthSquared() { return x * x + y * y; }
    };

    extension<T : __BuiltinFloatingPointType> MyVec<T> : IDifferentiable
        where T.Differential == T
    {
        typedef MyVec<T> Differential;

        static Differential dzero()
        {
            return MyVec<T>(T(0), T(0));
        }
    }

    void computeMain()
    {
        var dpv = diffPair(MyVec<float>(0.0, 0.0), MyVec<float>(0.0, 0.0));
        bwd_diff(MyVec<float>::lengthSquared)(dpv, 1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    extension = ast.extensions[0]
    call = ast.functions[0].body[1]

    assert extension.generic_parameters == "<T:__BuiltinFloatingPointType>"
    assert extension.generic_constraints[0].parameter == "T.Differential"
    assert extension.typedefs[0].new_type == "Differential"
    assert isinstance(call, CallNode)
    assert call.callee.name == "bwd_diff"
    assert call.callee.args[0].name == "MyVec<float>::lengthSquared"


def test_unnamed_signature_parameters_and_void_parameter_list_parse():
    code = """
    interface IFilter
    {
        float eval(float3, Texture2D<float4>);
        void reset(void);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    eval_method = ast.interfaces[0].methods[0]
    reset_method = ast.interfaces[0].methods[1]

    assert [(param.vtype, param.name) for param in eval_method.params] == [
        ("float3", ""),
        ("Texture2D<float4>", ""),
    ]
    assert reset_method.params == []


def test_func_keyword_name_first_parameters_from_current_docs_parse():
    code = """
    func add(x: int, y: float = 1.0f) -> float
    {
        return float(x) + y;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    add = ast.functions[0]

    assert add.name == "add"
    assert add.return_type == "float"
    assert [(param.vtype, param.name) for param in add.params] == [
        ("int", "x"),
        ("float", "y"),
    ]
    assert add.params[1].value == "1.0f"


def test_func_keyword_name_first_direction_qualifiers_from_current_docs_parse():
    # Source: Slang declarations docs, Parameters section, documents
    # `func modify(x: in out int)` direction qualifiers.
    code = """
    func modify(x: in out int)
    {
        x++;
    }

    func setValue(result: out float, value: inout int)
    {
        result = float(value);
        value = value + 1;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    modify, set_value = ast.functions

    assert modify.return_type == "void"
    assert [(param.vtype, param.name, param.qualifiers) for param in modify.params] == [
        ("int", "x", ["in", "out"]),
    ]
    assert [
        (param.vtype, param.name, param.qualifiers) for param in set_value.params
    ] == [
        ("float", "result", ["out"]),
        ("int", "value", ["inout"]),
    ]


def test_top_level_function_prototypes_from_slang_shaders_parse():
    code = """
    SceneResult Scene_GetDistance(vec3 vPos);
    float nnedi3_core(vec4 samples[8]);
    float map(vec3);

    float map(vec3 p)
    {
        return p.x;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [function.name for function in ast.functions] == [
        "Scene_GetDistance",
        "nnedi3_core",
        "map",
        "map",
    ]
    assert ast.functions[0].is_declaration
    assert ast.functions[1].is_declaration
    assert ast.functions[1].params[0].vtype == "vec4"
    assert ast.functions[1].params[0].array_sizes == ["8"]
    assert ast.functions[2].is_declaration
    assert ast.functions[2].params[0].name == ""
    assert not ast.functions[3].is_declaration


def test_nested_enum_class_in_generic_struct_parses():
    code = """
    struct GenericContainer<T>
    {
        [UnscopedEnum]
        enum class NestedEnum
        {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    struct = ast.structs[0]

    assert struct.generic_parameters == "<T>"
    assert len(struct.enums) == 1
    assert struct.enums[0].name == "NestedEnum"
    assert struct.enums[0].kind == "class"


def test_public_enum_qualifiers_from_slang_gfx_tool_parse():
    code = """
    public enum AccelerationStructureBuildFlags
    {
        None,
        AllowUpdate = 1
    };

    public enum class GeometryType
    {
        Triangles,
        ProcedurePrimitives
    };

    public struct GeometryFlags
    {
        public enum Enum
        {
            None,
            Opaque = 1
        };
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    build_flags = ast.enums[0]
    geometry_type = ast.enums[1]
    nested_enum = ast.structs[0].enums[0]

    assert build_flags.name == "AccelerationStructureBuildFlags"
    assert build_flags.qualifiers == ["public"]
    assert build_flags.members[1][0] == "AllowUpdate"
    assert build_flags.members[1][1] == "1"

    assert geometry_type.name == "GeometryType"
    assert geometry_type.kind == "class"
    assert geometry_type.qualifiers == ["public"]

    assert nested_enum.name == "Enum"
    assert nested_enum.qualifiers == ["public"]
    assert nested_enum.members[1][0] == "Opaque"


def test_nested_struct_declaration_in_struct_parses():
    code = """
    interface IAssoc
    {
        int getInner();
    }

    struct Impl
    {
        struct Assoc : IAssoc
        {
            int getInner() { return 1; }
        }

        Assoc getValue()
        {
            Assoc r;
            return r;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    struct = ast.structs[0]
    nested = struct.structs[0]

    assert nested.name == "Assoc"
    assert nested.conformances == ["IAssoc"]
    assert nested.methods[0].name == "getInner"
    assert struct.methods[0].return_type == "Assoc"
    assert struct.methods[0].name == "getValue"


def test_enum_declarations_from_gpu_printing_sample():
    code = """
    enum EmptyPrintingOp
    {
    };

    enum PrintingOp
    {
        PrintLine,
        PrintF = 2 + 1,
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert len(ast.enums) == 2
    empty_enum = ast.enums[0]
    valued_enum = ast.enums[1]

    assert isinstance(empty_enum, EnumNode)
    assert empty_enum.name == "EmptyPrintingOp"
    assert empty_enum.members == []
    assert valued_enum.name == "PrintingOp"
    assert valued_enum.members[0] == ("PrintLine", None)
    assert valued_enum.members[1][0] == "PrintF"
    assert isinstance(valued_enum.members[1][1], BinaryOpNode)


def test_generic_enum_class_from_upstream_bug_tests():
    # Reduced from shader-slang/slang tests/bugs/11042-generic-enum-scope-conflict.slang
    # and tests/diagnostics/generic-enum-underlying-type.slang.
    code = """
    enum class GenericEnum<T>
    {
        A,
        B,
    }

    enum class TestEnum<T : IArithmetic> : T
    {
        X = T(0),
        Y = T(1),
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generic_enum, typed_enum = ast.enums

    assert generic_enum.name == "GenericEnum"
    assert generic_enum.kind == "class"
    assert generic_enum.generic_parameters == "<T>"
    assert generic_enum.underlying_type is None
    assert generic_enum.members == [("A", None), ("B", None)]

    assert typed_enum.name == "TestEnum"
    assert typed_enum.kind == "class"
    assert typed_enum.generic_parameters == "<T:IArithmetic>"
    assert typed_enum.underlying_type == "T"
    assert typed_enum.members[0][0] == "X"
    assert isinstance(typed_enum.members[0][1], FunctionCallNode)


def test_function_generic_where_conformance_constraint_parsing():
    code = """
    int useFoo<T>(T value) where T : IFoo {
        return value.foo();
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "useFoo")

    assert function.generic_parameters == "<T>"
    assert len(function.generic_constraints) == 1
    constraint = function.generic_constraints[0]
    assert isinstance(constraint, GenericConstraintNode)
    assert constraint.parameter == "T"
    assert constraint.constraint_type == "IFoo"


def test_multiple_generic_where_clauses_from_official_generic_test_parse():
    code = """
    interface IContainer
    {
        associatedtype ElementType;
        ElementType getElement();
    }

    interface IScalable
    {
        int getScale();
    }

    int scaledElement<T, U>(T container, U scaler)
        where T : IContainer
        where U : IScalable
        where T.ElementType == int
    {
        return container.getElement() * scaler.getScale();
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "scaledElement")

    assert [
        (c.parameter, c.relation, c.constraint_type)
        for c in function.generic_constraints
    ] == [
        ("T", ":", "IContainer"),
        ("U", ":", "IScalable"),
        ("T.ElementType", "==", "int"),
    ]


def test_for_update_parses_array_and_member_assignment_targets():
    code = """
    void main(){
        int value = 1;
        for (int i = 0; i < 4; items[i] += value) {
        }
        for (int i = 0; i < 4; object.field = value) {
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    array_loop = function.body[1]
    member_loop = function.body[2]

    assert isinstance(array_loop, ForNode)
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


def test_standalone_array_and_member_assignment_targets_parse():
    code = """
    void main(){
        items[0] = 1;
        values[tid.x] += delta;
        object.field = value;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    item_assign = function.body[0]
    value_assign = function.body[1]
    member_assign = function.body[2]

    assert isinstance(item_assign, AssignmentNode)
    assert item_assign.operator == "="
    assert isinstance(item_assign.left, ArrayAccessNode)
    assert item_assign.left.array.name == "items"
    assert item_assign.left.index == "0"
    assert item_assign.right == "1"

    assert isinstance(value_assign, AssignmentNode)
    assert value_assign.operator == "+="
    assert isinstance(value_assign.left, ArrayAccessNode)
    assert value_assign.left.array.name == "values"
    assert isinstance(value_assign.left.index, MemberAccessNode)
    assert value_assign.left.index.object.name == "tid"
    assert value_assign.left.index.member == "x"
    assert value_assign.right.name == "delta"

    assert isinstance(member_assign, AssignmentNode)
    assert member_assign.operator == "="
    assert isinstance(member_assign.left, MemberAccessNode)
    assert member_assign.left.object.name == "object"
    assert member_assign.left.member == "field"
    assert member_assign.right.name == "value"


def test_logical_not_parsing():
    code = """
    bool negate(bool disabled) {
        return !disabled;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    value = find_function(ast, "negate").body[0].value

    assert isinstance(value, UnaryOpNode)
    assert value.op == "!"
    assert value.operand.name == "disabled"


def test_logical_and_keeps_equality_operands_grouped():
    code = """
    void main(bool a, bool b, bool c, bool d) {
        bool bothEqual = a == b && c == d;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    assignment = find_function(ast, "main").body[0]
    expression = assignment.right

    assert isinstance(expression, BinaryOpNode)
    assert expression.op == "&&"
    assert isinstance(expression.left, BinaryOpNode)
    assert expression.left.op == "=="
    assert isinstance(expression.right, BinaryOpNode)
    assert expression.right.op == "=="


def test_binary_bitwise_and_shift_precedence_parsing():
    code = """
    uint combine(uint a, uint b, uint c, uint d) {
        uint value = a | b ^ c & d << 1;
        return value;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    expression = find_function(ast, "combine").body[0].right

    assert expression.op == "|"
    assert expression.left.name == "a"
    assert expression.right.op == "^"
    assert expression.right.left.name == "b"
    assert expression.right.right.op == "&"
    assert expression.right.right.left.name == "c"
    assert expression.right.right.right.op == "<<"
    assert expression.right.right.right.left.name == "d"
    assert expression.right.right.right.right == "1"


def test_compound_bitwise_and_shift_assignment_parsing():
    code = """
    void update(uint value, uint mask) {
        value %= 3;
        value &= mask;
        value |= 1;
        value ^= mask;
        value <<= 1;
        value >>= 2;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    body = find_function(ast, "update").body

    assert [stmt.operator for stmt in body] == ["%=", "&=", "|=", "^=", "<<=", ">>="]
    assert all(isinstance(stmt, AssignmentNode) for stmt in body)


def test_assignment_parsing_is_right_associative():
    code = """
    void chain(uint a, uint b, uint c, bool flag) {
        a = b = c;
        a += b = c;
        a = flag ? b : c;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    body = find_function(ast, "chain").body
    simple = body[0]
    compound = body[1]
    ternary_assignment = body[2]

    assert isinstance(simple, AssignmentNode)
    assert simple.operator == "="
    assert simple.left.name == "a"
    assert isinstance(simple.right, AssignmentNode)
    assert simple.right.operator == "="
    assert simple.right.left.name == "b"
    assert simple.right.right.name == "c"

    assert isinstance(compound, AssignmentNode)
    assert compound.operator == "+="
    assert isinstance(compound.right, AssignmentNode)
    assert compound.right.operator == "="

    assert isinstance(ternary_assignment, AssignmentNode)
    assert isinstance(ternary_assignment.right, TernaryOpNode)


def test_while_parsing():
    code = """
    float countDown(float x) {
        while (x > 0.0) {
            x -= 1.0;
        }
        return x;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        function = find_function(ast, "countDown")
        loop = function.body[0]

        assert isinstance(loop, WhileNode)
        assert loop.condition.op == ">"
        assert isinstance(loop.body[0], AssignmentNode)
        assert loop.body[0].operator == "-="
    except SyntaxError:
        pytest.fail("while parsing not implemented.")


def test_do_while_parsing():
    code = """
    float countDown(float x) {
        do {
            x -= 1.0;
        } while (x > 0.0);
        return x;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        function = find_function(ast, "countDown")
        loop = function.body[0]

        assert isinstance(loop, DoWhileNode)
        assert loop.condition.op == ">"
        assert isinstance(loop.body[0], AssignmentNode)
        assert loop.body[0].operator == "-="
    except SyntaxError:
        pytest.fail("do-while parsing not implemented.")


def test_top_level_attribute_list_before_shader_function_parsing():
    code = """
    [numthreads(8, 8, 1)]
    [shader("compute")]
    void main(uint3 tid : SV_DispatchThreadID) {
        return;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        function = find_function(ast, "main")

        assert function.qualifier == "compute"
        assert function.numthreads == ("8", "8", "1")
        assert function.attributes == [
            {"name": "numthreads", "arguments": ["8", "8", "1"]}
        ]
        assert function.params[0].vtype.strip() == "uint3"
        assert function.params[0].semantic == "SV_DispatchThreadID"
        assert isinstance(function.body[0], ReturnNode)
        assert function.body[0].value is None
    except SyntaxError:
        pytest.fail("Top-level attribute-list parsing not implemented.")


def test_extended_shader_stage_attribute_parsing():
    code = """
    [shader("geometry")]
    void gs_main() {
        return;
    }

    [shader("raygeneration")]
    void ray_main() {
        return;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert find_function(ast, "gs_main").qualifier == "geometry"
    assert find_function(ast, "ray_main").qualifier == "raygeneration"
    generated = SlangCrossGLCodeGen.SlangToCrossGLConverter().generate(ast)
    assert "geometry {" in generated
    assert "ray_generation {" in generated


def test_attribute_list_after_shader_function_parsing():
    code = """
    [shader("compute")]
    [numthreads(8, 8, 1)]
    void main(uint3 tid : SV_DispatchThreadID) {
        return;
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        function = find_function(ast, "main")

        assert function.qualifier == "compute"
        assert function.numthreads == ("8", "8", "1")
        assert function.attributes == [
            {"name": "numthreads", "arguments": ["8", "8", "1"]}
        ]
        assert function.params[0].vtype.strip() == "uint3"
        assert function.params[0].semantic == "SV_DispatchThreadID"
        assert isinstance(function.body[0], ReturnNode)
        assert function.body[0].value is None
    except SyntaxError:
        pytest.fail("Post-shader attribute-list parsing not implemented.")


def test_modern_slang_shader_and_numthreads_attribute_parsing():
    code = """
    [[shader("compute")]]
    [numThreads(4, 2, 1)]
    void main(uint3 tid : SV_DispatchThreadID) {
        return;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")

    assert function.qualifier == "compute"
    assert function.numthreads == ("4", "2", "1")
    assert function.attributes == [{"name": "numThreads", "arguments": ["4", "2", "1"]}]


def test_generic_resource_global_parsing():
    code = """
    Texture2D<float4> albedo;
    SamplerState linearSampler;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [(var.vtype, var.name) for var in ast.global_vars] == [
        ("Texture2D<float4>", "albedo"),
        ("SamplerState", "linearSampler"),
    ]


def test_globallycoherent_resource_global_and_generic_method_call_parse():
    code = """
    globallycoherent RWByteAddressBuffer bab;
    RWStructuredBuffer<uint> output;
    in uint3 gid : SV_GroupID;

    void computeMain()
    {
        output[0] = bab.Load<uint>(0) + gid.x;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    bab = ast.global_vars[0]
    gid = ast.global_vars[2]
    assignment = find_function(ast, "computeMain").body[0]

    assert bab.qualifiers == ["globallycoherent"]
    assert bab.vtype == "RWByteAddressBuffer"
    assert bab.name == "bab"
    assert gid.qualifiers == ["in"]
    assert gid.semantic == "SV_GroupID"
    assert isinstance(assignment.right, BinaryOpNode)
    assert isinstance(assignment.right.left, MethodCallNode)
    assert assignment.right.left.method == "Load<uint>"


def test_nested_parameter_block_resource_wrapper_parsing():
    code = """
    struct S
    {
        ConstantBuffer<RWStructuredBuffer<int>> cb;
    }

    ParameterBlock<RWStructuredBuffer<int>> rwBuffer;
    ParameterBlock<ConstantBuffer<int>> constBuffer;
    ParameterBlock<ConstantBuffer<RWStructuredBuffer<int>>> nestedBuffer;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert ast.structs[0].members[0].vtype == "ConstantBuffer<RWStructuredBuffer<int>>"
    assert [(var.vtype, var.name) for var in ast.global_vars] == [
        ("ParameterBlock<RWStructuredBuffer<int>>", "rwBuffer"),
        ("ParameterBlock<ConstantBuffer<int>>", "constBuffer"),
        (
            "ParameterBlock<ConstantBuffer<RWStructuredBuffer<int>>>",
            "nestedBuffer",
        ),
    ]


def test_bound_generic_resource_global_parsing():
    code = """
    Texture2D<float4> albedo : register(t0);
    SamplerState linearSampler : register(s0);
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert len(ast.functions) == 0
    assert [(var.vtype, var.name, var.register) for var in ast.global_vars] == [
        ("Texture2D<float4>", "albedo", "t0"),
        ("SamplerState", "linearSampler", "s0"),
    ]


def test_vulkan_binding_attribute_global_resource_parsing():
    code = """
    [[vk::binding(0, 1)]]
    Texture2D<float4> albedo : register(t0);

    [[vk::binding(2)]]
    SamplerState linearSampler;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    albedo = ast.global_vars[0]
    linear_sampler = ast.global_vars[1]

    assert albedo.attributes == [{"name": "vk::binding", "arguments": ["0", "1"]}]
    assert albedo.register == "t0"
    assert linear_sampler.attributes == [{"name": "vk::binding", "arguments": ["2"]}]
    assert [(var.vtype, var.name) for var in ast.global_vars] == [
        ("Texture2D<float4>", "albedo"),
        ("SamplerState", "linearSampler"),
    ]


def test_bound_cbuffer_parsing():
    code = """
    cbuffer Camera : register(b0) {
        float4x4 viewProj;
        float4 tint[2];
        float weights[];
        float4x4 transforms[2][3];
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    cbuffer = ast.cbuffers[0]

    assert cbuffer.name == "Camera"
    assert cbuffer.register == "b0"
    assert [
        (member.vtype, member.name, member.array_sizes) for member in cbuffer.members
    ] == [
        ("float4x4", "viewProj", []),
        ("float4", "tint", ["2"]),
        ("float", "weights", [None]),
        ("float4x4", "transforms", ["2", "3"]),
    ]


def test_cbuffer_comma_member_declarators_parsing():
    code = """
    cbuffer Camera {
        float exposure, gamma;
        float4 offsets[2], tint;
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    members = ast.cbuffers[0].members

    assert [(member.vtype, member.name, member.array_sizes) for member in members] == [
        ("float", "exposure", []),
        ("float", "gamma", []),
        ("float4", "offsets", ["2"]),
        ("float4", "tint", []),
    ]


def test_vulkan_attributes_on_cbuffer_parsing():
    code = """
    [[vk::binding(0, 1)]]
    [[vk::push_constant]]
    cbuffer Camera : register(b0) {
        float4x4 viewProj;
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    cbuffer = ast.cbuffers[0]

    assert cbuffer.name == "Camera"
    assert cbuffer.register == "b0"
    assert cbuffer.attributes == [
        {"name": "vk::binding", "arguments": ["0", "1"]},
        {"name": "vk::push_constant", "arguments": []},
    ]
    assert cbuffer.members[0].name == "viewProj"


def test_vulkan_stage_location_attributes_parse_from_spirv_docs():
    # Source: Slang SPIR-V target docs list vk::location and vk::index
    # as Vulkan layout attributes for global stage variables.
    code = """
    [[vk::location(0)]]
    in float4 vColor;

    [[vk::location(1)]]
    [[vk::index(0)]]
    out float4 fragColor;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [
        (var.vtype, var.name, var.qualifiers, var.attributes) for var in ast.global_vars
    ] == [
        (
            "float4",
            "vColor",
            ["in"],
            [{"name": "vk::location", "arguments": ["0"]}],
        ),
        (
            "float4",
            "fragColor",
            ["out"],
            [
                {"name": "vk::location", "arguments": ["1"]},
                {"name": "vk::index", "arguments": ["0"]},
            ],
        ),
    ]


def test_official_shader_parameter_cbuffer_register_space_parsing():
    # Source: Slang shader parameter docs show vk::binding and register markup
    # together on a cbuffer declaration.
    code = """
    [[vk::binding(2, 9)]]
    cbuffer CSMUniforms : register(b0, space9)
    {
        float4 shadowCascadeDistances;
    };
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    cbuffer = ast.cbuffers[0]

    assert cbuffer.name == "CSMUniforms"
    assert cbuffer.register == "b0,space9"
    assert cbuffer.attributes == [{"name": "vk::binding", "arguments": ["2", "9"]}]
    assert [(member.vtype, member.name) for member in cbuffer.members] == [
        ("float4", "shadowCascadeDistances")
    ]


def test_global_resource_array_parsing():
    code = """
    StructuredBuffer<float> inputs[2];
    Texture2D<float4> textures[3] : register(t0);
    SamplerState samplers[];
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [(var.vtype, var.name, var.array_sizes) for var in ast.global_vars] == [
        ("StructuredBuffer<float>", "inputs", ["2"]),
        ("Texture2D<float4>", "textures", ["3"]),
        ("SamplerState", "samplers", [None]),
    ]
    assert ast.global_vars[1].register == "t0"


def test_namespace_global_resource_array_parsing_from_current_slang_spirv_test():
    # Source: shader-slang/slang tests/spirv/namespace-texture-array.slang
    # at 8c4e02e4021d73091a4f1d4eba842c0dd986997e.
    code = """
    struct ComputePush
    {
        uint image_id;
    };
    [[vk::push_constant]] ComputePush p;

    namespace test_namespace
    {
        [[vk::binding(0, 0)]] RWTexture2D<float4> textureTable[];
    }

    [shader("compute")]
    [numthreads(8, 8, 1)]
    void main(uint3 pixel_i : SV_DispatchThreadID)
    {
        test_namespace.textureTable[p.image_id][pixel_i.xy] = float4(0,1,0,0);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    namespaced_resource = ast.global_vars[1]
    assert namespaced_resource.vtype == "RWTexture2D<float4>"
    assert namespaced_resource.name == "test_namespace.textureTable"
    assert namespaced_resource.array_sizes == [None]
    assert namespaced_resource.attributes == [
        {"name": "vk::binding", "arguments": ["0", "0"]}
    ]

    assignment = find_function(ast, "main").body[0]
    assert isinstance(assignment.left.array.array, MemberAccessNode)
    assert assignment.left.array.array.object.name == "test_namespace"
    assert assignment.left.array.array.member == "textureTable"


def test_dotted_namespace_global_resource_parsing_from_namespace_import_sample():
    # Source: shader-slang/slang tests/language-feature/namespaces/namespace-import/m.slang
    # at 52339028a2aa703271533454c6b9528a534bac31.
    code = """
    namespace ns1.ns2
    {
        [[vk::binding(0, 0)]] RWTexture2D<float4> textureTable[];
    }

    [shader("compute")]
    [numthreads(8, 8, 1)]
    void main(uint3 pixel_i : SV_DispatchThreadID)
    {
        ns1.ns2.textureTable[0][pixel_i.xy] = float4(0,1,0,0);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    namespaced_resource = ast.global_vars[0]
    assert namespaced_resource.vtype == "RWTexture2D<float4>"
    assert namespaced_resource.name == "ns1.ns2.textureTable"
    assert namespaced_resource.array_sizes == [None]

    assignment = find_function(ast, "main").body[0]
    resource_access = assignment.left.array.array
    assert isinstance(resource_access, MemberAccessNode)
    assert resource_access.member == "textureTable"
    assert isinstance(resource_access.object, MemberAccessNode)
    assert resource_access.object.object.name == "ns1"
    assert resource_access.object.member == "ns2"


def test_local_and_parameter_array_declarator_parsing():
    code = """
    float bump(float values[2], int idx) {
        float local[2];
        float grid[2][3];
        return values[idx];
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "bump")

    assert [
        (param.vtype, param.name, param.array_sizes) for param in function.params
    ] == [
        ("float", "values", ["2"]),
        ("int", "idx", []),
    ]
    assert isinstance(function.body[0], VariableNode)
    assert function.body[0].name == "local"
    assert function.body[0].array_sizes == ["2"]
    assert isinstance(function.body[1], VariableNode)
    assert function.body[1].name == "grid"
    assert function.body[1].array_sizes == ["2", "3"]
    assert isinstance(function.body[2].value, ArrayAccessNode)


def test_geometry_primitive_array_parameter_parsing():
    code = """
    struct VSOutput {
        float4 Pos : POSITION0;
    };

    struct GSOutput {
        float4 Pos : SV_POSITION;
    };

    [shader("geometry")]
    [maxvertexcount(3)]
    void geometryMain(triangle VSOutput input[3],
                      inout TriangleStream<GSOutput> outStream,
                      uint PrimitiveID : SV_PrimitiveID) {
        return;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "geometryMain")

    assert function.qualifier == "geometry"
    assert function.attributes == [{"name": "maxvertexcount", "arguments": ["3"]}]
    assert [
        (param.qualifiers, param.vtype, param.name, param.array_sizes, param.semantic)
        for param in function.params
    ] == [
        (["triangle"], "VSOutput", "input", ["3"], None),
        (["inout"], "TriangleStream<GSOutput>", "outStream", [], None),
        ([], "uint", "PrimitiveID", [], "SV_PrimitiveID"),
    ]


def test_mesh_output_role_array_parameter_parsing():
    code = """
    struct VertexOutput
    {
        float4 position : SV_Position;
    };

    [shader("mesh")]
    [outputtopology("triangle")]
    [numthreads(1, 1, 1)]
    void meshMain(out indices uint3 triangles[1],
                  out vertices VertexOutput vertices[3],
                  uint3 DispatchThreadID : SV_DispatchThreadID)
    {
        return;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "meshMain")

    assert function.qualifier == "mesh"
    assert [
        (param.qualifiers, param.vtype, param.name, param.array_sizes, param.semantic)
        for param in function.params
    ] == [
        (["out", "indices"], "uint3", "triangles", ["1"], None),
        (["out", "vertices"], "VertexOutput", "vertices", ["3"], None),
        ([], "uint3", "DispatchThreadID", [], "SV_DispatchThreadID"),
    ]


def test_local_and_parameter_generic_resource_type_parsing():
    code = """
    float4 sample(Sampler2D<float4> tex, Texture2D<float4> image,
                  SamplerState state, float2 uv) {
        Sampler2D<float4> localTex;
        Texture2D<float4> localImage;
        SamplerState localState;
        return tex.Sample(uv);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "sample")

    assert [(param.vtype, param.name) for param in function.params] == [
        ("Sampler2D<float4>", "tex"),
        ("Texture2D<float4>", "image"),
        ("SamplerState", "state"),
        ("float2", "uv"),
    ]
    assert [(stmt.vtype, stmt.name) for stmt in function.body[:3]] == [
        ("Sampler2D<float4>", "localTex"),
        ("Texture2D<float4>", "localImage"),
        ("SamplerState", "localState"),
    ]


def test_parameter_attribute_parsing_from_cuda_format_docs():
    # Source: shader-slang/slang docs/cuda-target.md at
    # 11e97e77155f454c67dc66daf25c75386d13c378 documents
    # attribute metadata on variables, parameters, and fields.
    code = """
    float2 getValue([format("rg16f")] RWTexture2D<float2> t)
    {
        return t[int2(0, 0)];
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "getValue")

    assert function.params[0].vtype == "RWTexture2D<float2>"
    assert function.params[0].name == "t"
    assert function.params[0].attributes == [
        {"name": "format", "arguments": ['"rg16f"']}
    ]


def test_texture_method_call_parsing():
    code = """
    Texture2D<float4> albedo;
    SamplerState linearSampler;

    float4 main(float2 uv) {
        return albedo.Sample(linearSampler, uv);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    call = function.body[0].value

    assert isinstance(call, MethodCallNode)
    assert isinstance(call.object, VariableNode)
    assert call.object.name == "albedo"
    assert call.method == "Sample"
    assert len(call.args) == 2
    assert call.args[0].name == "linearSampler"
    assert call.args[1].name == "uv"


def test_hlsl_namespace_builtin_call_parsing():
    code = """
    void main(float4 color, float4x4 matrix, float4 vector, float value) {
        float4 clamped = hlsl::saturate(color);
        float4 transformed = hlsl::mul(matrix, vector);
        float wrapped = hlsl::frac(value);
        float wave = hlsl::sin(value);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")

    calls = [stmt.right for stmt in function.body]
    assert [call.name for call in calls] == [
        "hlsl::saturate",
        "hlsl::mul",
        "hlsl::frac",
        "hlsl::sin",
    ]


def test_texture_load_method_parsing():
    code = """
    Texture2D<float4> albedo;

    float4 main(int2 pixel, int mip) {
        return albedo.Load(int3(pixel, mip));
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    call = function.body[0].value

    assert isinstance(call, MethodCallNode)
    assert isinstance(call.object, VariableNode)
    assert call.object.name == "albedo"
    assert call.method == "Load"
    assert isinstance(call.args[0], FunctionCallNode)
    assert call.args[0].name == "int3"
    assert call.args[0].args[0].name == "pixel"
    assert call.args[0].args[1].name == "mip"


def test_standalone_postfix_call_parsing():
    code = """
    void main() {
        getCallback()(1.0);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    call = function.body[0]

    assert isinstance(call, CallNode)
    assert isinstance(call.callee, FunctionCallNode)
    assert call.callee.name == "getCallback"
    assert call.args == ["1.0"]


def test_intrinsic_asm_string_statement_from_link_time_options():
    code = """
    float getMacroDefinedForDownstream()
    {
        __intrinsic_asm "(DOWNSTREAM_VALUE)";
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "getMacroDefinedForDownstream")
    call = function.body[0]

    assert isinstance(call, FunctionCallNode)
    assert call.name == "__intrinsic_asm"
    assert call.args == ['"(DOWNSTREAM_VALUE)"']


def test_target_intrinsic_struct_modifier_from_interop_docs():
    # Source: Slang User Guide, Interoperation with Target-Specific Code,
    # "Defining Intrinsic Types" documents __target_intrinsic before a struct.
    code = """
    __target_intrinsic(cpp, "std::string")
    struct CppString
    {
        uint size()
        {
            __intrinsic_asm "static_cast<uint32_t>(($0).size())";
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    struct = ast.structs[0]

    assert struct.name == "CppString"
    assert struct.qualifiers == []
    assert [method.name for method in struct.methods] == ["size"]
    assert struct.methods[0].body[0].name == "__intrinsic_asm"


def test_scalar_and_matrix_top_level_declarations_parsing():
    code = """
    uint addOne(uint x) { return x + 1; }
    bool enabled(bool x) { return x; }
    float4x4 ViewProj;
    int frameIndex;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert [(function.return_type, function.name) for function in ast.functions] == [
        ("uint", "addOne"),
        ("bool", "enabled"),
    ]
    assert [(var.vtype, var.name) for var in ast.global_vars] == [
        ("float4x4", "ViewProj"),
        ("int", "frameIndex"),
    ]


def test_initialized_top_level_global_parsing():
    code = """
    static const float threshold = 0.5;
    float4 tint = float4(1.0, 0.5, 0.0, 1.0);
    float gain : register(c0) = 1f;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    threshold = ast.global_vars[0]
    assert isinstance(threshold, AssignmentNode)
    assert threshold.left.vtype == "float"
    assert threshold.left.name == "threshold"
    assert threshold.left.qualifiers == ["static", "const"]
    assert threshold.right == "0.5"

    tint = ast.global_vars[1]
    assert isinstance(tint, AssignmentNode)
    assert tint.left.vtype == "float4"
    assert tint.left.name == "tint"
    assert isinstance(tint.right, VectorConstructorNode)
    assert tint.right.type_name == "float4"

    gain = ast.global_vars[2]
    assert isinstance(gain, AssignmentNode)
    assert gain.left.register == "c0"
    assert gain.right == "1f"


def test_extern_static_const_globals_from_link_time_constant_tests():
    code = """
    extern static const bool turnOnFeature;
    extern static const float constValue;
    extern static const uint numthread = 0;
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    turn_on_feature = ast.global_vars[0]
    const_value = ast.global_vars[1]
    numthread = ast.global_vars[2]

    assert turn_on_feature.vtype == "bool"
    assert turn_on_feature.name == "turnOnFeature"
    assert turn_on_feature.qualifiers == ["extern", "static", "const"]
    assert const_value.vtype == "float"
    assert const_value.name == "constValue"
    assert const_value.qualifiers == ["extern", "static", "const"]
    assert isinstance(numthread, AssignmentNode)
    assert numthread.left.vtype == "uint"
    assert numthread.left.name == "numthread"
    assert numthread.left.qualifiers == ["extern", "static", "const"]
    assert numthread.right == "0"


def test_comma_separated_global_declarations_parsing():
    code = """
    static float exposure, gamma = 2.2, weights[2];
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    exposure = ast.global_vars[0]
    gamma = ast.global_vars[1]
    weights = ast.global_vars[2]

    assert isinstance(exposure, VariableNode)
    assert exposure.vtype == "float"
    assert exposure.name == "exposure"
    assert exposure.qualifiers == ["static"]

    assert isinstance(gamma, AssignmentNode)
    assert gamma.left.vtype == "float"
    assert gamma.left.name == "gamma"
    assert gamma.left.qualifiers == ["static"]
    assert gamma.right == "2.2"

    assert isinstance(weights, VariableNode)
    assert weights.name == "weights"
    assert weights.array_sizes == ["2"]


def test_comma_separated_local_declarations_from_public_corpus():
    code = """
    void main() {
        float3 a, b, c;
        float4x4 mx, my = float4x4(1.0), mz;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    body = find_function(ast, "main").body

    assert [(stmt.vtype, stmt.name) for stmt in body[:3]] == [
        ("float3", "a"),
        ("float3", "b"),
        ("float3", "c"),
    ]
    assert isinstance(body[3], VariableNode)
    assert body[3].vtype == "float4x4"
    assert body[3].name == "mx"
    assert isinstance(body[4], AssignmentNode)
    assert body[4].left.vtype == "float4x4"
    assert body[4].left.name == "my"
    assert isinstance(body[4].right, VectorConstructorNode)
    assert isinstance(body[5], VariableNode)
    assert body[5].vtype == "float4x4"
    assert body[5].name == "mz"


def test_initializer_list_declaration_parsing():
    code = """
    float weights[2] = {1.0, 2.0};

    void main() {
        float local[2] = {.5f, 1f,};
        float4 colors[1] = {float4(1.0, 0.5, 0.0, 1.0)};
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    global_weights = ast.global_vars[0]
    assert isinstance(global_weights, AssignmentNode)
    assert global_weights.left.name == "weights"
    assert global_weights.left.array_sizes == ["2"]
    assert isinstance(global_weights.right, InitializerListNode)
    assert global_weights.right.elements == ["1.0", "2.0"]

    body = find_function(ast, "main").body
    local = body[0]
    assert isinstance(local.right, InitializerListNode)
    assert local.right.elements == [".5f", "1f"]

    colors = body[1]
    assert isinstance(colors.right, InitializerListNode)
    assert isinstance(colors.right.elements[0], VectorConstructorNode)


def test_typed_brace_constructor_parsing():
    code = """
    float4 tint = float4{1.0, .5f, 0.0, 1.0};

    void main() {
        float4 local = float4{0.0, 1.0, 0.0, 1.0};
        float4 colors[1] = {float4{1.0, 0.0, 0.0, 1.0}};
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    tint = ast.global_vars[0]
    assert isinstance(tint.right, VectorConstructorNode)
    assert tint.right.type_name == "float4"
    assert tint.right.args == ["1.0", ".5f", "0.0", "1.0"]

    body = find_function(ast, "main").body
    assert isinstance(body[0].right, VectorConstructorNode)
    assert body[0].right.type_name == "float4"
    assert isinstance(body[1].right.elements[0], VectorConstructorNode)


def test_typed_array_constructor_parsing_from_libretro_shader():
    code = """
    int[] font = int[](0x00000000, 0x1F803FC0);

    struct Payload {
        float[8] data;
    };

    void main() {
        float weights[3] = float[3](1.0, 2.0, 3.0);
        float4 colors[2] = float4[2](
            float4(1.0, 0.0, 0.0, 1.0),
            float4(0.0, 1.0, 0.0, 1.0)
        );
        vec3 ap4[4] = vec3[](red, green, blue, black);
        ivec3 palette[2] = ivec3[](ivec3(1), ivec3(2));
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    font = ast.global_vars[0]
    assert isinstance(font, AssignmentNode)
    assert font.left.vtype == "int[]"
    assert font.left.name == "font"
    assert isinstance(font.right, VectorConstructorNode)
    assert font.right.type_name == "int[]"
    assert font.right.args == ["0x00000000", "0x1F803FC0"]

    payload = ast.structs[0]
    assert payload.members[0].vtype == "float[8]"
    assert payload.members[0].name == "data"

    body = find_function(ast, "main").body
    weights = body[0]
    assert isinstance(weights, AssignmentNode)
    assert weights.left.vtype == "float"
    assert weights.left.name == "weights"
    assert weights.left.array_sizes == ["3"]
    assert isinstance(weights.right, VectorConstructorNode)
    assert weights.right.type_name == "float[3]"
    assert weights.right.args == ["1.0", "2.0", "3.0"]

    colors = body[1]
    assert isinstance(colors.right, VectorConstructorNode)
    assert colors.right.type_name == "float4[2]"
    assert len(colors.right.args) == 2
    assert all(isinstance(arg, VectorConstructorNode) for arg in colors.right.args)

    ap4 = body[2]
    assert isinstance(ap4.right, VectorConstructorNode)
    assert ap4.right.type_name == "vec3[]"
    assert [arg.name for arg in ap4.right.args] == ["red", "green", "blue", "black"]

    palette = body[3]
    assert isinstance(palette.right, VectorConstructorNode)
    assert palette.right.type_name == "ivec3[]"
    assert len(palette.right.args) == 2


def test_local_matrix_declaration_parsing():
    code = """
    void main() {
        float4x4 transform;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    declaration = function.body[0]

    assert isinstance(declaration, VariableNode)
    assert declaration.vtype == "float4x4"
    assert declaration.name == "transform"


def test_local_matrix_constructor_declaration_parsing():
    code = """
    void main() {
        float4x4 model = float4x4(1.0);
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "main")
    declaration = function.body[0]

    assert isinstance(declaration, AssignmentNode)
    assert declaration.left.vtype == "float4x4"
    assert declaration.left.name == "model"
    assert isinstance(declaration.right, VectorConstructorNode)
    assert declaration.right.type_name == "float4x4"
    assert declaration.right.args == ["1.0"]


def test_qualified_function_parameters_globals_and_locals_parsing():
    code = """
    static Texture2D<float4> sourceTexture;

    static inline float helper(const float x) {
        return x;
    }

    void main() {
        static const float cached = 1.0;
        constexpr float scale = 2.0;
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert ast.global_vars[0].qualifiers == ["static"]

    helper = find_function(ast, "helper")
    assert helper.qualifiers == ["static", "inline"]
    assert helper.params[0].qualifiers == ["const"]

    body = find_function(ast, "main").body
    assert isinstance(body[0], AssignmentNode)
    assert body[0].left.qualifiers == ["static", "const"]
    assert body[0].left.vtype == "float"
    assert isinstance(body[1], AssignmentNode)
    assert body[1].left.qualifiers == ["constexpr"]


def test_else_parsing():
    code = """
    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        if (assembledVertex.color.r > 0.5) {
            output.out_position = assembledVertex.color;
        }
        else {
            output.out_position = float3(0.0, 0.0, 0.0);
        }
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("else parsing not implemented.")


def test_function_call_parsing():
    code = """
    float4 saturate(float4 color) {
        return color;
    }
    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex){
        VertexStageOutput output;
        output.out_position = assembledVertex.position;
        output.out_position = saturate(assembledVertex.color);
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("function call parsing not implemented.")


def test_standalone_function_call_statement_parsing():
    code = """
    void helper(float x) {
        return;
    }

    void main() {
        float x = 1.0;
        helper(x);
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        main = find_function(ast, "main")
        call = main.body[1]

        assert isinstance(call, FunctionCallNode)
        assert call.name == "helper"
        assert len(call.args) == 1
        assert isinstance(call.args[0], VariableNode)
        assert call.args[0].name == "x"
    except SyntaxError:
        pytest.fail("standalone function call statement parsing not implemented.")


def test_mod_parsing():
    code = """
    [shader("vertex")]
    VertexStageOutput vertexMain(AssembledVertex assembledVertex) {
        VertexStageOutput output;
        int a = 10 % 3;  // Basic modulus
        return output;
    }
    """
    try:
        tokens = tokenize_code(code)
        parse_code(tokens)
    except SyntaxError:
        pytest.fail("Modulus operator parsing not implemented")


def test_numeric_literal_parsing():
    code = """
    float f() {
        float a = 1e-3f;
        float b = .5f;
        float c = 1.;
        uint mask = 0xffu;
        uint count = 123u;
        return a + b + c;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    body = find_function(ast, "f").body

    assert [stmt.right for stmt in body[:5]] == [
        "1e-3f",
        ".5f",
        "1.",
        "0xffu",
        "123u",
    ]


def test_binary_integer_literals_from_generated_conformance_sample_parse():
    # Source: shader-slang/slang docs/generated/tests/conformance/
    # expressions-literal/bin-prefix-lowercase.slang at d25453d.
    code = """
    void main() {
        int x = 0b1010;
        int y = 0B1111;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    body = find_function(ast, "main").body

    assert [stmt.right for stmt in body] == ["0b1010", "0B1111"]


def test_integer_literal_underscores_from_generated_conformance_sample_parse():
    # Source: shader-slang/slang docs/generated/tests/conformance/
    # lexical-structure/integer-literal-underscore-ignored.slang at d25453d.
    code = """
    void main() {
        int a = 1_000_000;
        int b = 1000000;
        int c = 0x_FF_FF;
        bool eq = (a == b) && (c == 0xFFFF);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    body = find_function(ast, "main").body

    assert [stmt.right for stmt in body[:3]] == [
        "1000000",
        "1000000",
        "0xFFFF",
    ]
    assert isinstance(body[3].right, BinaryOpNode)


def test_generic_struct_member_and_uniform_parameters_from_official_sample():
    code = """
    static const uint THREADGROUP_SIZE_X = 8;
    static const uint THREADGROUP_SIZE_Y = THREADGROUP_SIZE_X;

    struct ImageProcessingOptions
    {
        float3 tintColor;
        float blurRadius;
        bool useLookupTable;
        StructuredBuffer<float4> lookupTable;
    }

    [shader("compute")]
    [numthreads(THREADGROUP_SIZE_X, THREADGROUP_SIZE_Y)]
    void processImage(
        uint3 threadID : SV_DispatchThreadID,
        uniform Texture2D inputImage,
        uniform RWTexture2D outputImage,
        uniform ImageProcessingOptions options)
    {
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    options = ast.structs[0]
    process_image = find_function(ast, "processImage")

    assert options.members[-1].vtype == "StructuredBuffer<float4>"
    assert options.members[-1].name == "lookupTable"
    assert process_image.numthreads == (
        "THREADGROUP_SIZE_X",
        "THREADGROUP_SIZE_Y",
        "1",
    )
    params = [
        (param.qualifiers, param.vtype, param.name) for param in process_image.params
    ]
    assert params == [
        ([], "uint3", "threadID"),
        (["uniform"], "Texture2D", "inputImage"),
        (["uniform"], "RWTexture2D", "outputImage"),
        (["uniform"], "ImageProcessingOptions", "options"),
    ]


def test_export_extern_cpp_function_parsing():
    code = """
    [TorchEntryPoint]
    export __extern_cpp int main()
    {
        return 0;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    exported = ast.exports[0].item

    assert exported.name == "main"
    assert exported.return_type == "int"
    assert "__extern_cpp" in exported.qualifiers
    assert exported.attributes == [{"name": "TorchEntryPoint", "arguments": []}]
    assert isinstance(exported.body[0], ReturnNode)


def test_c_style_scalar_cast_from_official_select_expr_sample():
    code = """
    int test(int input)
    {
        return input > 1 ? -input : input;
    }

    RWStructuredBuffer<int> outputBuffer;

    [numthreads(4, 1, 1)]
    void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
    {
        outputBuffer[dispatchThreadID.x] = test((int) dispatchThreadID.x);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    compute_main = find_function(ast, "computeMain")
    assignment = compute_main.body[0]
    call = assignment.right
    cast = call.args[0]

    assert isinstance(cast, CastNode)
    assert cast.target_type == "int"
    assert isinstance(cast.expression, MemberAccessNode)
    assert cast.expression.member == "x"


def test_c_style_user_type_zero_initialization_from_public_samples():
    code = """
    struct FragmentState
    {
        float4 color;
    };

    FragmentState makeState()
    {
        FragmentState state = (FragmentState)0;
        return state;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    make_state = find_function(ast, "makeState")
    assignment = make_state.body[0]
    cast = assignment.right

    assert isinstance(cast, CastNode)
    assert cast.target_type == "FragmentState"
    assert cast.expression == "0"


def test_c_style_user_type_identifier_cast_from_enum_sample():
    code = """
    enum class Mode : uint
    {
        Off,
        On,
    };

    Mode decode(uint raw)
    {
        Mode mode = (Mode)raw;
        return mode;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    decode = find_function(ast, "decode")
    assignment = decode.body[0]
    cast = assignment.right

    assert isinstance(cast, CastNode)
    assert cast.target_type == "Mode"
    assert cast.expression.name == "raw"


def test_parenthesized_relational_expression_is_not_generic_cast():
    code = """
    bool lessThanLimit(float t, float resT)
    {
        return (t < resT);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "lessThanLimit")
    returned = function.body[0].value

    assert isinstance(returned, BinaryOpNode)
    assert returned.op == "<"
    assert returned.left.name == "t"
    assert returned.right.name == "resT"


def test_parenthesized_relational_before_greater_than_if_is_not_generic_suffix():
    code = """
    bool formatDigit(float fValue, float fDigitIndex)
    {
        bool bNeg = (fValue < 0.0);
        if (fDigitIndex > 1.0)
        {
            return bNeg;
        }
        return false;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "formatDigit")
    declaration = function.body[0]
    if_stmt = function.body[1]

    assert isinstance(declaration.right, BinaryOpNode)
    assert declaration.right.op == "<"
    assert isinstance(if_stmt, IfNode)
    assert isinstance(if_stmt.condition, BinaryOpNode)
    assert if_stmt.condition.op == ">"


def test_parenthesized_comma_expression_from_slang_shaders():
    code = """
    float reduce(float3 v)
    {
        return max((v.x, v.y), v.z);
    }

    float2 choose(bool invert)
    {
        return float2(invert ? (0.75, 1.0) : (1.0, 0.75));
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    reduce = find_function(ast, "reduce")
    choose = find_function(ast, "choose")

    max_call = reduce.body[0].value
    assert isinstance(max_call.args[0], ParenthesizedCommaNode)
    assert [expr.member for expr in max_call.args[0].expressions] == ["x", "y"]

    constructor = choose.body[0].value
    ternary = constructor.args[0]
    assert isinstance(ternary, TernaryOpNode)
    assert isinstance(ternary.true_expr, ParenthesizedCommaNode)
    assert isinstance(ternary.false_expr, ParenthesizedCommaNode)

    generated = SlangCrossGLCodeGen.SlangToCrossGLConverter().generate(ast)
    assert "max((v.x, v.y), v.z)" in generated
    assert "(0.75, 1.0)" in generated


def test_return_comma_operator_from_slang_compute_test():
    # Source: shader-slang/slang tests/compute/comma-operator.slang at
    # 5230a81f2fe68afe5cb8d04a1b09d56476f6b960.
    code = """
    int test(int inVal)
    {
        int a = inVal;
        return a*=2, a+1;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    test = find_function(ast, "test")
    returned = test.body[1].value

    assert isinstance(returned, ParenthesizedCommaNode)
    assert len(returned.expressions) == 2
    assert isinstance(returned.expressions[0], AssignmentNode)
    assert returned.expressions[0].operator == "*="
    assert isinstance(returned.expressions[1], BinaryOpNode)

    generated = SlangCrossGLCodeGen.SlangToCrossGLConverter().generate(ast)
    assert "return (a *= 2, a + 1);" in generated


def test_for_update_list_from_slang_shaders():
    code = """
    void scan(float factorX)
    {
        float x = 1.0;
        for (int n = 0; n < 7; n++, x -= factorX * 0.5)
        {
            x += 1.0;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "scan")
    loop = function.body[1]

    assert isinstance(loop, ForNode)
    assert len(loop.update) == 2
    assert isinstance(loop.update[0], UnaryOpNode)
    assert loop.update[0].op == "POST_INCREMENT"
    assert isinstance(loop.update[1], AssignmentNode)
    assert loop.update[1].operator == "-="

    generated = SlangCrossGLCodeGen.SlangToCrossGLConverter().generate(ast)
    assert "for (int n = 0; n < 7; n++, x -= factorX * 0.5)" in generated


def test_static_type_member_call_from_saschawillems_hdr_sample():
    code = """
    float computeSpec(float fresnel, float geoAtt, float NdotV, float NdotL)
    {
        return (fresnel * geoAtt) / (NdotV * NdotL * float.getPi());
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    function = find_function(ast, "computeSpec")
    divisor = function.body[0].value.right
    static_call = divisor.right

    assert isinstance(static_call, MethodCallNode)
    assert static_call.method == "getPi"
    assert static_call.object.name == "float"


def test_parenthesized_expression_swizzle_from_autodiff_texture_learnmip_sample():
    code = """
    RWTexture2D dstTexture;

    void computeMain(uint2 p)
    {
        var val = float4(1.0);
        float4 color = float4((dstTexture[p] - val).xyz, 1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    compute_main = find_function(ast, "computeMain")
    color_assignment = compute_main.body[1]
    constructor = color_assignment.right
    swizzle = constructor.args[0]

    assert isinstance(swizzle, MemberAccessNode)
    assert swizzle.member == "xyz"
    assert isinstance(swizzle.object, BinaryOpNode)
    assert isinstance(swizzle.object.left, ArrayAccessNode)


def test_numeric_literal_swizzle_from_vulkan_samples_primitive_clipping():
    code = """
    struct VSOutput
    {
        float3 Normal;
    };

    float4 main(VSOutput input)
    {
        float4 outColor = float4(0.5 * input.Normal + 0.5.xxx, 1.0);
        return outColor;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    main = find_function(ast, "main")
    constructor = main.body[0].right
    expression = constructor.args[0]

    assert isinstance(expression.right, MemberAccessNode)
    assert expression.right.object == "0.5"
    assert expression.right.member == "xxx"


def test_struct_method_attributes_and_no_diff_from_autodiff_texture_train_sample():
    code = """
    Texture2D texRef;

    struct DifferentiableTexture
    {
        Texture2D texture;

        [BackwardDerivative(bwd_LoadTexel)]
        float4 LoadTexel(int3 location, constexpr int2 offset)
        {
            return texture.Load(location, offset);
        }
    }

    [BackwardDifferentiable]
    float3 loss(no_diff float2 uv, no_diff float4 screenPos)
    {
        float3 refColor = (no_diff texRef.Load(int3(int2(screenPos.xy), 0))).xyz;
        return refColor;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    texture = ast.structs[0]
    method = texture.methods[0]
    loss = find_function(ast, "loss")
    ref_color = loss.body[0]

    assert method.attributes == [
        {"name": "BackwardDerivative", "arguments": ["bwd_LoadTexel"]}
    ]
    assert method.params[1].qualifiers == ["constexpr"]
    assert [param.qualifiers for param in loss.params] == [["no_diff"], ["no_diff"]]
    assert isinstance(ref_color.right, MemberAccessNode)
    assert ref_color.right.member == "xyz"


def test_pointer_and_statement_attribute_syntax_from_mlp_training_samples():
    code = """
    public struct AdamOptimizer
    {
        public static const NFloat beta1 = 0.9h;
    }

    public struct FeedForwardLayer<int InputSize, int OutputSize>
    {
        public NFloat* weights;
        public NFloat* weightsGrad;
    }

    [numthreads(256, 1, 1)]
    void adjustParameters(
        uint32_t tid : SV_DispatchThreadID,
        uniform AdamState* states,
        uniform NFloat* params,
        uniform NFloat* gradients,
        uniform uint32_t count)
    {
        [ForceUnroll]
        for (int i = 0; i < 1; i++)
            AdamOptimizer::step(states[tid], params[tid], gradients[tid]);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    optimizer = ast.structs[0]
    layer = ast.structs[1]
    adjust_parameters = find_function(ast, "adjustParameters")
    loop = adjust_parameters.body[0]
    call = loop.body[0]

    assert optimizer.members[0].qualifiers == ["public", "static", "const"]
    assert optimizer.members[0].value == "0.9h"
    assert [member.vtype for member in layer.members] == ["NFloat*", "NFloat*"]
    assert [param.vtype for param in adjust_parameters.params[1:4]] == [
        "AdamState*",
        "NFloat*",
        "NFloat*",
    ]
    assert isinstance(loop, ForNode)
    assert isinstance(call, FunctionCallNode)
    assert call.name == "AdamOptimizer::step"


def test_mlp_coopvec_array_type_local_and_void_pointer_cast_parsing():
    code = """
    public struct FeedForwardLayer<int InputSize, int OutputSize>
    {
        internal void* biasesGrad;

        internal static NFloat[N] coopVecToArray(CoopVec<NFloat, N> v)
        {
            NFloat[N] arr;
            return arr;
        }

        public void evalBwd(MLVec<OutputSize> resultGrad)
        {
            coopVecReduceSumAccumulate(resultGrad.data, (void*)biasesGrad);
        }

        public override static Differential dzero()
        {
            return {};
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    struct = ast.structs[0]
    array_method = next(
        method for method in struct.methods if method.name == "coopVecToArray"
    )
    backward_method = next(
        method for method in struct.methods if method.name == "evalBwd"
    )
    override_method = next(
        method for method in struct.methods if method.name == "dzero"
    )
    array_decl = array_method.body[0]
    call = backward_method.body[0]
    cast = call.args[1]

    assert array_method.return_type == "NFloat[N]"
    assert isinstance(array_decl, VariableNode)
    assert array_decl.vtype == "NFloat[N]"
    assert array_decl.name == "arr"
    assert isinstance(call, FunctionCallNode)
    assert isinstance(cast, CastNode)
    assert cast.target_type == "void*"
    assert cast.expression.name == "biasesGrad"
    assert override_method.qualifiers == ["public", "override", "static"]


def test_groupshared_global_and_pointer_dereference_member_access_parse():
    code = """
    groupshared float4 sharedColor[64];

    struct PushConstants
    {
        float scale;
    };

    float readScale(PushConstants* pushConstants)
    {
        float scale = *pushConstants.scale;
        return scale;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    shared_color = ast.global_vars[0]
    read_scale = find_function(ast, "readScale")
    assignment = read_scale.body[0]
    dereference = assignment.right

    assert shared_color.qualifiers == ["groupshared"]
    assert shared_color.vtype == "float4"
    assert shared_color.name == "sharedColor"
    assert isinstance(dereference, UnaryOpNode)
    assert dereference.op == "*"
    assert isinstance(dereference.operand, MemberAccessNode)
    assert dereference.operand.member == "scale"


def test_official_pointer_address_of_and_arrow_member_access_parse():
    # Source: Slang User's Guide, Basic Convenience Features > Pointers (limited).
    code = """
    struct MyType
    {
        int a;
    };

    int test(MyType* pObj)
    {
        MyType* pNext = pObj + 1;
        MyType* pNext2 = &pNext[1];
        return pNext.a + pNext->a + (*pNext2).a + pNext2[0].a;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    test_function = find_function(ast, "test")
    address_of = test_function.body[1].right
    expression = test_function.body[2].value

    assert isinstance(address_of, UnaryOpNode)
    assert address_of.op == "&"
    assert isinstance(address_of.operand, ArrayAccessNode)

    terms = []
    while isinstance(expression, BinaryOpNode) and expression.op == "+":
        terms.append(expression.right)
        expression = expression.left
    terms.append(expression)
    terms.reverse()

    assert len(terms) == 4
    assert all(isinstance(term, MemberAccessNode) for term in terms)
    assert terms[0].object.name == "pNext"
    assert terms[0].member == "a"
    assert terms[1].object.name == "pNext"
    assert terms[1].member == "a"
    assert isinstance(terms[2].object, UnaryOpNode)
    assert terms[2].object.op == "*"
    assert terms[2].member == "a"
    assert isinstance(terms[3].object, ArrayAccessNode)
    assert terms[3].member == "a"


def test_interpolation_modifiers_from_vulkan_samples_parse_as_qualifiers():
    code = """
    struct VSOutput
    {
        float4 Pos : SV_POSITION;
        nointerpolation uint TextureIndex;
        nointerpolation float4 Color;
        linear centroid float2 Uv : TEXCOORD0;
        sample float4 CoverageColor : COLOR1;
    };

    [shader("fragment")]
    float4 main(
        VSOutput input,
        noperspective float3 baryCoordsAffine : SV_Barycentrics,
        centroid noperspective float2 baryCoordsNoPerspective : TEXCOORD1,
        linear sample float4 perSampleColor : COLOR2)
    {
        return input.Color + perSampleColor + float4(baryCoordsAffine, 1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    output = ast.structs[0]
    main = find_function(ast, "main")

    assert output.members[1].qualifiers == ["nointerpolation"]
    assert output.members[1].vtype == "uint"
    assert output.members[2].qualifiers == ["nointerpolation"]
    assert output.members[2].vtype == "float4"
    assert output.members[3].qualifiers == ["linear", "centroid"]
    assert output.members[3].vtype == "float2"
    assert output.members[4].qualifiers == ["sample"]
    assert output.members[4].vtype == "float4"
    assert main.params[1].qualifiers == ["noperspective"]
    assert main.params[1].vtype == "float3"
    assert main.params[2].qualifiers == ["centroid", "noperspective"]
    assert main.params[2].vtype == "float2"
    assert main.params[3].qualifiers == ["linear", "sample"]
    assert main.params[3].vtype == "float4"


def test_flat_interpolation_outputs_from_libretro_shaders_parse_as_qualifiers():
    code = """
    layout(location = 1) flat out float delta;
    layout(location = 2) out flat vec2 noise_div;

    struct VSOutput
    {
        flat out float4 Color : COLOR0;
    };

    [shader("fragment")]
    float4 main(flat in float deltaInput : COLOR1,
                out flat float outputWeight : COLOR2)
    {
        outputWeight = deltaInput;
        return float4(outputWeight, 0.0, 0.0, 1.0);
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    delta = ast.global_vars[0]
    noise_div = ast.global_vars[1]
    output = ast.structs[0]
    main = find_function(ast, "main")

    assert delta.qualifiers == ["layout(location=1)", "flat", "out"]
    assert delta.vtype == "float"
    assert delta.name == "delta"
    assert noise_div.qualifiers == ["layout(location=2)", "out", "flat"]
    assert noise_div.vtype == "vec2"
    assert output.members[0].qualifiers == ["flat", "out"]
    assert output.members[0].vtype == "float4"
    assert main.params[0].qualifiers == ["flat", "in"]
    assert main.params[0].vtype == "float"
    assert main.params[1].qualifiers == ["out", "flat"]
    assert main.params[1].vtype == "float"


def test_glsl_precision_qualifiers_from_corpus_parse():
    code = """
    precision highp float;
    precision mediump int;

    layout(location = 0) out mediump vec4 FragColor;

    layout(std140, set = 0, binding = 0) uniform Params
    {
        mediump int ui_zero;
        highp vec4 tint;
    } params;

    float4 shade(lowp in float delta : COLOR0, highp vec3 normal)
    {
        highp vec4 color = vec4(delta);
        return color;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    frag_color = ast.global_vars[0]
    params = ast.cbuffers[0]
    shade = find_function(ast, "shade")

    assert frag_color.qualifiers == ["layout(location=0)", "out", "mediump"]
    assert frag_color.vtype == "vec4"
    assert params.qualifiers == ["layout(std140,set=0,binding=0)", "uniform"]
    assert params.members[0].qualifiers == ["mediump"]
    assert params.members[0].vtype == "int"
    assert params.members[1].qualifiers == ["highp"]
    assert params.members[1].vtype == "vec4"
    assert shade.params[0].qualifiers == ["lowp", "in"]
    assert shade.params[0].vtype == "float"
    assert shade.params[1].qualifiers == ["highp"]
    assert shade.params[1].vtype == "vec3"
    assert shade.body[0].left.qualifiers == ["highp"]
    assert shade.body[0].left.vtype == "vec4"


def test_standalone_layout_declarations_from_first_slang_shader_docs():
    code = """
    layout(row_major) uniform;
    layout(row_major) buffer;
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

    void main()
    {
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    assert ast.global_vars == []
    assert ast.cbuffers == []
    assert find_function(ast, "main").name == "main"


def test_comma_separated_expression_statement_from_slang_shaders():
    code = """
    float4 shade(float wf1)
    {
        float4 w1 = float4(1.0);
        float4 w2 = float4(2.0);
        if (wf1 > 1.0)
        {
            wf1 = 1.0 / wf1;
            w1 *= wf1, w2 *= wf1;
        }
        return w1 + w2;
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    shade = find_function(ast, "shade")
    if_body = shade.body[2].if_body

    assert isinstance(if_body[1], AssignmentNode)
    assert if_body[1].left.name == "w1"
    assert if_body[1].operator == "*="
    assert isinstance(if_body[2], AssignmentNode)
    assert if_body[2].left.name == "w2"
    assert if_body[2].operator == "*="


def test_generic_prefix_struct_from_official_compute_tests_parse():
    code = """
    __generic<T>
    struct GenStruct
    {
        T x;
        T y;
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    struct = ast.structs[0]

    assert struct.name == "GenStruct"
    assert struct.generic_parameters == "<T>"
    assert [member.name for member in struct.members] == ["x", "y"]
    assert [member.vtype for member in struct.members] == ["T", "T"]


def test_numeric_bitfield_members_from_official_language_tests_parse():
    code = """
    struct S
    {
        int foo : 8;
        uint bar : 24;
    };
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    struct = ast.structs[0]

    assert [member.name for member in struct.members] == ["foo", "bar"]
    assert [member.vtype for member in struct.members] == ["int", "uint"]
    assert [member.semantic for member in struct.members] == [None, None]


if __name__ == "__main__":
    pytest.main()

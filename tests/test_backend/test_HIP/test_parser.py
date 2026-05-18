"""Test HIP Parser"""

from crosstl.backend.HIP.HipAst import (
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    InitializerListNode,
    SyncNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser, HipProgramNode


class TestHipParser:
    def test_fixed_arrays_and_initializer_lists_parsing(self):
        """Test fixed arrays and brace initializer lists"""
        code = """
        float weights[4] = {1.0f, 2.0f, 3.0f, 4.0f};

        struct Filter {
            float taps[3];
            float matrix[2][2];
        };

        __global__ void kernel(float input[4]) {
            float local[2] = {1.0f, 2.0f};
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        assert isinstance(ast, HipProgramNode)
        assert ast.statements[0].vtype == "float[4]"
        assert isinstance(ast.statements[0].value, InitializerListNode)
        assert ast.statements[1].members[0].vtype == "float[3]"
        assert ast.statements[1].members[1].vtype == "float[2][2]"
        assert ast.statements[2].params[0]["type"] == "float[4]"
        assert ast.statements[2].body[0].vtype == "float[2]"
        assert isinstance(ast.statements[2].body[0].value, InitializerListNode)

    def test_qualified_declarations_parsing(self):
        """Test const, unsigned, and static declarations"""
        code = """
        static float cached = 1.0f;
        unsigned int mask = 3u;

        __global__ void kernel(unsigned int* out, const float scale) {
            const int local = 1;
            unsigned int idx = 2u;
            static float tmp = 0.0f;
            out[0] = idx;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        assert ast.statements[0].vtype == "float"
        assert ast.statements[1].vtype == "unsigned int"
        assert ast.statements[2].params[0]["type"] == "unsigned int *"
        assert ast.statements[2].params[1]["type"] == "const float"
        assert ast.statements[2].body[0].vtype == "const int"
        assert ast.statements[2].body[1].vtype == "unsigned int"
        assert ast.statements[2].body[2].vtype == "float"

    def test_qualified_and_pointer_return_functions_parsing(self):
        """Test qualified scalar and pointer return types"""
        code = """
        unsigned int lane_mask() { return 3u; }
        float* get_data(float* data) { return data; }
        const float* get_const_data(const float* data) { return data; }
        static inline unsigned int helper(unsigned int x) { return x; }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        assert ast.statements[0].return_type == "unsigned int"
        assert ast.statements[1].return_type == "float *"
        assert ast.statements[1].params[0]["type"] == "float *"
        assert ast.statements[2].return_type == "const float *"
        assert ast.statements[2].params[0]["type"] == "const float *"
        assert ast.statements[3].return_type == "unsigned int"
        assert "static" in ast.statements[3].qualifiers
        assert "inline" in ast.statements[3].qualifiers

    def test_braced_for_body_and_sync_statement_parsing(self):
        """Test braced for bodies and sync calls parse as statements"""
        code = """
        void helper(int n) {
            for (int j = 0; j < n; j++) {
                sink(j);
            }
            __syncthreads();
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
        assert isinstance(body[0], ForNode)
        assert isinstance(body[0].init, VariableNode)
        assert body[0].init.name == "j"
        assert isinstance(body[0].body[0], FunctionCallNode)
        assert isinstance(body[1], SyncNode)

    def test_local_pointer_declarations_and_unary_pointer_expressions(self):
        """Test local pointer declarations, address-of, and dereference"""
        code = """
        void helper(float* data, unsigned int* ids) {
            float* p = data;
            const float* cp = data;
            unsigned int* ip = ids;
            float x = 1.0f;
            float* q = &x;
            *p = *q;
            ip[0] = 1u;
            a * b;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
        assert body[0].vtype == "float *"
        assert body[1].vtype == "const float *"
        assert body[2].vtype == "unsigned int *"
        assert isinstance(body[4].value, UnaryOpNode)
        assert body[4].value.op == "&"
        assert isinstance(body[5], AssignmentNode)
        assert isinstance(body[5].left, UnaryOpNode)
        assert body[5].left.op == "*"
        assert isinstance(body[5].right, UnaryOpNode)
        assert body[5].right.op == "*"
        assert isinstance(body[7], BinaryOpNode)

    def test_bitwise_logical_and_shift_expression_parsing(self):
        """Test bitwise, shift, logical, and compound shift expressions"""
        code = """
        unsigned int helper(unsigned int a, unsigned int b) {
            unsigned int x = (a & b) | (a ^ b);
            unsigned int y = x << 2;
            unsigned int z = y >> 1;
            if (a > 0 && b > 0 || a == b) {
                z <<= 1;
                z >>= 1;
            }
            return z;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
        assert isinstance(body[0].value, BinaryOpNode)
        assert body[0].value.op == "|"
        assert body[0].value.left.op == "&"
        assert body[0].value.right.op == "^"
        assert body[1].value.op == "<<"
        assert body[2].value.op == ">>"
        assert body[3].condition.op == "||"
        assert body[3].condition.left.op == "&&"
        assert body[3].if_body[0].operator == "<<="
        assert body[3].if_body[1].operator == ">>="

    def test_numeric_literal_parsing(self):
        """Test integer mask and float suffix literals parse intact"""
        code = """
        unsigned int helper() {
            unsigned int mask = 0xffu;
            unsigned int bits = 0b1010u;
            unsigned int oct = 0777u;
            float x = 1e-3f;
            float y = .5f;
            return mask | bits | oct;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
        assert body[0].value == "0xffu"
        assert body[1].value == "0b1010u"
        assert body[2].value == "0777u"
        assert body[3].value == "1e-3f"
        assert body[4].value == ".5f"

    def test_control_flow_and_cast_expression_parsing(self):
        """Test HIP control-flow nodes and cast expressions"""
        code = """
        int helper(float x, int n) {
            int i = 0;
            while (i < n) {
                i += 1;
            }
            do {
                i += 1;
            } while (i < n);
            switch (i) {
                case 1:
                    i += 2;
                    break;
                default:
                    i += 3;
                    break;
            }
            return i > 0 ? (int)x : n;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
        assert isinstance(body[1], WhileNode)
        assert isinstance(body[2], DoWhileNode)
        assert isinstance(body[3], SwitchNode)
        assert len(body[3].cases) == 1
        assert len(body[3].default_case) == 2
        assert isinstance(body[4].value, TernaryOpNode)
        assert isinstance(body[4].value.true_expr, CastNode)

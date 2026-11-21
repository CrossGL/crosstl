"""CUDA runtime API call statements in the CUDA to CrossGL converter: output parameters and pointer targets, status, value, wrapper and inline expressions, kernel function references and statement fragments."""

from ..CudaAst import (
    AssignmentNode,
    CastNode,
    FunctionCallNode,
    UnaryOpNode,
    VariableNode,
)
from ..pointer_math_import import sine_cosine_statements


class CudaImportRuntimeCallMixin:
    """Runtime call members of ``CudaToCrossGLConverter``."""

    def emit_cuda_output_parameter_call_statement(self, stmt):
        if not isinstance(stmt, FunctionCallNode):
            return False

        raw_name = stmt.name if isinstance(stmt.name, str) else self.visit(stmt.name)
        if self.is_user_defined_function(raw_name):
            return False

        statements = self.format_cuda_output_parameter_call_statements(
            raw_name, stmt.args
        )
        if statements is None:
            return False

        for statement in statements:
            self.emit(statement)
        return True

    def format_cuda_output_parameter_call_statements(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        return sine_cosine_statements(self, function_name, args, UnaryOpNode)

    def format_cuda_runtime_status_expression(self, value):
        if not isinstance(value, FunctionCallNode):
            return None
        if self.is_user_defined_function(value.name):
            return None

        wrapped_runtime = self.format_cuda_runtime_wrapper_expression(value)
        if wrapped_runtime is not None:
            return wrapped_runtime

        comments = self.format_cuda_runtime_call(value)
        if comments is None:
            return None

        return comments, self.format_cuda_success_literal(value.name)

    def format_cuda_success_literal(self, name):
        if isinstance(name, str) and name.startswith("cu") and len(name) > 2:
            if name[2].isupper():
                return "CUDA_SUCCESS"
        return "cudaSuccess"

    def format_cuda_runtime_value_expression(self, value):
        if not isinstance(value, FunctionCallNode):
            return None
        if self.is_user_defined_function(value.name):
            return None

        name = value.name if isinstance(value.name, str) else self.visit(value.name)
        if name == "cudaGetCurrentGraphExec" and not value.args:
            return ["// CUDA device graph get current exec"], "0"
        return None

    def format_cuda_runtime_wrapper_expression(self, value):
        if not isinstance(value, FunctionCallNode):
            return None
        if self.is_user_defined_function(value.name):
            return None
        if value.name not in self.CUDA_RUNTIME_ERROR_WRAPPER_NAMES:
            return None
        if len(getattr(value, "args", []) or []) != 1:
            return None
        return self.format_cuda_runtime_expression(value.args[0])

    def format_cuda_runtime_inline_expression(self, value):
        runtime_expression = self.format_cuda_runtime_expression(value)
        if runtime_expression is None:
            return None

        comments, fallback = runtime_expression
        if not comments:
            return fallback

        comment = comments[-1]
        if comment.startswith("// "):
            comment = comment[3:]
        return f"(/* {comment} */ {fallback})"

    def format_runtime_pointer_target(self, arg):
        if isinstance(arg, CastNode):
            return self.format_runtime_pointer_target(arg.expression)
        if isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.visit(arg.operand)
        return self.visit(arg)

    def format_cuda_kernel_function_reference(self, arg):
        if isinstance(arg, CastNode):
            return self.format_cuda_kernel_function_reference(arg.expression)
        if isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.visit(arg.operand)
        return self.visit(arg)

    def format_statement_fragment(self, stmt):
        if stmt is None:
            return ""
        if isinstance(stmt, list):
            return ", ".join(self.format_statement_fragment(item) for item in stmt)
        if isinstance(stmt, VariableNode):
            self.register_variable_type(stmt.name, stmt.vtype)
            var_type = self.convert_cuda_variable_type_to_crossgl(stmt.vtype, stmt.name)
            self.register_vector1_name(stmt.name, stmt.vtype)
            name = self.register_identifier_name(stmt.name)
            if stmt.value:
                value = self.visit(stmt.value)
                return f"var {name}: {var_type} = {value}"
            return f"var {name}: {var_type}"
        if isinstance(stmt, AssignmentNode):
            left = self.visit(stmt.left)
            right = self.visit(stmt.right)
            return f"{left} {stmt.operator} {right}"

        result = self.visit(stmt)
        return result if isinstance(result, str) else ""

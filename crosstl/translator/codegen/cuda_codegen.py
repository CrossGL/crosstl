"""CrossGL-to-CUDA code generator."""

from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    ContinueNode,
    ForInNode,
    ForNode,
    IfNode,
    LiteralPatternNode,
    LiteralNode,
    RangeNode,
    ReturnNode,
    StructNode,
    VariableNode,
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    ShaderNode,
    FunctionNode,
    FunctionCallNode,
    ExpressionStatementNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    UnaryOpNode,
    TernaryOpNode,
    BlockNode,
    WildcardPatternNode,
)
from .resource_diagnostics import ResourceDiagnosticMixin
from .resource_query import ResourceQueryMixin
from .resource_arrays import format_array_declarator
from .vector_arithmetic import VectorArithmeticMixin


class CudaCodeGen(VectorArithmeticMixin, ResourceQueryMixin, ResourceDiagnosticMixin):
    """Emit CUDA source from the shared CrossGL translator AST."""

    resource_diagnostic_backend = "CUDA"
    query_return_index_binary_ops = {"+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^"}
    synchronization_builtins = {
        "barrier",
        "groupMemoryBarrier",
        "memoryBarrier",
        "memoryBarrierShared",
        "memoryBarrierBuffer",
        "memoryBarrierImage",
        "workgroupBarrier",
    }
    sampled_resource_type_aliases = {
        "Texture1D": "sampler1D",
        "Texture1DArray": "sampler1DArray",
        "Texture2D": "sampler2D",
        "Texture2DArray": "sampler2DArray",
        "Texture2DMS": "sampler2DMS",
        "Texture2DMSArray": "sampler2DMSArray",
        "Texture3D": "sampler3D",
        "TextureCube": "samplerCube",
        "TextureCubeArray": "samplerCubeArray",
    }
    storage_resource_type_aliases = {
        "RWTexture1D": "image1D",
        "RWTexture1DArray": "image1DArray",
        "RWTexture2D": "image2D",
        "RWTexture2DArray": "image2DArray",
        "RWTexture3D": "image3D",
        "RWTextureCube": "imageCube",
        "RWTextureCubeArray": "imageCubeArray",
        "RWTexture2DMS": "image2DMS",
        "RWTexture2DMSArray": "image2DMSArray",
    }

    def __init__(self):
        """Initialize CUDA type maps and per-generation visitor state."""
        self.indent_level = 0
        self.output = []
        self.variable_types = {}
        self.image_resource_accesses = {}
        self.struct_member_types = {}
        self.struct_member_image_accesses = {}
        self.function_return_types = {}
        self.helper_functions = {}
        self.query_resource_names = set()
        self.query_metadata_function_params = {}
        self.query_functions_by_name = {}
        self.query_metadata_aliases = {}
        self.query_return_sources = {}
        self.query_local_resource_names_by_function = {}
        self.query_metadata_snapshot_locals_by_function = {}
        self.structured_buffer_length_names = set()
        self.structured_buffer_length_function_params = {}
        self.current_structured_buffer_length_parameters = {}
        self.current_function_name = None
        self.resource_query_info_required = False
        self.builtin_map = {
            "gl_LocalInvocationID.x": "threadIdx.x",
            "gl_LocalInvocationID.y": "threadIdx.y",
            "gl_LocalInvocationID.z": "threadIdx.z",
            "gl_WorkGroupID.x": "blockIdx.x",
            "gl_WorkGroupID.y": "blockIdx.y",
            "gl_WorkGroupID.z": "blockIdx.z",
            "gl_WorkGroupSize.x": "blockDim.x",
            "gl_WorkGroupSize.y": "blockDim.y",
            "gl_WorkGroupSize.z": "blockDim.z",
            "gl_NumWorkGroups.x": "gridDim.x",
            "gl_NumWorkGroups.y": "gridDim.y",
            "gl_NumWorkGroups.z": "gridDim.z",
            "gl_GlobalInvocationID.x": "(blockIdx.x * blockDim.x + threadIdx.x)",
            "gl_GlobalInvocationID.y": "(blockIdx.y * blockDim.y + threadIdx.y)",
            "gl_GlobalInvocationID.z": "(blockIdx.z * blockDim.z + threadIdx.z)",
        }

    def generate(self, ast_node):
        """Generate complete CUDA source for a CrossGL AST."""
        self.output = []
        self.indent_level = 0
        self.variable_types = {}
        self.image_resource_accesses = {}
        (
            self.struct_member_types,
            self.struct_member_image_accesses,
        ) = self.collect_struct_member_metadata(ast_node)
        self.function_return_types = self.collect_function_return_types(ast_node)
        self.helper_functions = {}
        self.query_metadata_aliases = {}
        self.query_return_sources = self.collect_simple_query_return_sources(ast_node)
        self.query_local_resource_names_by_function = {}
        self.query_metadata_snapshot_locals_by_function = {}
        self.resource_query_info_required = False
        (
            self.query_resource_names,
            self.query_metadata_function_params,
        ) = self.collect_resource_query_requirements(ast_node)
        (
            self.structured_buffer_length_names,
            self.structured_buffer_length_function_params,
        ) = self.collect_structured_buffer_length_requirements(ast_node)
        self.query_functions_by_name = {
            getattr(func, "name", None): func
            for func in self.query_collect_functions(ast_node)
        }
        self.query_functions_by_name = {
            name: func for name, func in self.query_functions_by_name.items() if name
        }
        self.visit(ast_node)
        self.insert_helper_functions()
        return "\n".join(self.output)

    def visit(self, node):
        """Dispatch an AST node to its CUDA visitor method."""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Fallback visitor for primitive values, lists, and unknown nodes."""
        if isinstance(node, str):
            return node
        elif isinstance(node, list):
            return [self.visit(item) for item in node]
        else:
            return str(node)

    def emit(self, code):
        """Append a line of CUDA output using the current indentation level."""
        if code.strip():
            self.output.append("    " * self.indent_level + code)
        else:
            self.output.append("")

    def emit_statement(self, node):
        """Render and append one statement node when it produces code."""
        if node is None:
            return

        if isinstance(node, AssignmentNode):
            self.emit_assignment_statement(node)
            return

        result = self.visit(node)
        if isinstance(result, str) and result.strip():
            self.emit(f"{result};")

    def emit_body(self, body):
        """Render a list-like or block-like function body."""
        if isinstance(body, list):
            for stmt in body:
                self.emit_statement(stmt)
        elif hasattr(body, "statements"):
            for stmt in body.statements:
                self.emit_statement(stmt)
        else:
            self.emit_statement(body)

    def statement_list(self, body):
        """Normalize block-like statement containers to a list."""
        if body is None:
            return []
        if isinstance(body, list):
            return body
        if hasattr(body, "statements"):
            return body.statements
        return [body]

    def collect_struct_member_metadata(self, root):
        """Collect struct member types and storage-image access before emission."""
        member_types_by_struct = {}
        member_accesses_by_struct = {}
        for struct in getattr(root, "structs", []) or []:
            struct_name = getattr(struct, "name", None)
            if not struct_name:
                continue
            member_types = {}
            member_accesses = {}
            for member in getattr(struct, "members", []) or []:
                member_name = getattr(member, "name", None)
                if not member_name:
                    continue
                if hasattr(member, "member_type"):
                    member_type = member.member_type
                elif hasattr(member, "vtype"):
                    member_type = member.vtype
                else:
                    member_type = "float"
                member_type = self.apply_readonly_qualifier_to_type(member_type, member)
                member_types[member_name] = member_type
                member_access = self.explicit_resource_access(member)
                if member_access is not None and self.is_storage_image_type(
                    member_type
                ):
                    member_accesses[member_name] = member_access
            member_types_by_struct[struct_name] = member_types
            member_accesses_by_struct[struct_name] = member_accesses
        return member_types_by_struct, member_accesses_by_struct

    def statement_body_terminates(self, body):
        """Return true when the body already exits the active control flow."""
        statements = self.statement_list(body)
        if not statements:
            return False
        return isinstance(statements[-1], (BreakNode, ContinueNode, ReturnNode))

    def is_supported_switch_match_arm(self, arm):
        """CUDA switch lowering supports only unguarded literal/default arms."""
        if getattr(arm, "guard", None) is not None:
            return False
        pattern = getattr(arm, "pattern", None)
        return isinstance(pattern, (LiteralPatternNode, WildcardPatternNode))

    def visit_ShaderNode(self, node):
        """Render a full shader/program AST as a CUDA translation unit."""
        self.emit("#include <cuda_runtime.h>")
        self.emit("#include <device_launch_parameters.h>")
        self.emit("")

        structs = getattr(node, "structs", [])
        for struct in structs:
            self.visit(struct)
            self.emit("")

        cbuffers = getattr(node, "cbuffers", [])
        for cbuffer in cbuffers:
            self.visit_cbuffer(cbuffer)
            self.emit("")

        global_vars = getattr(node, "global_variables", [])
        for var in global_vars:
            self.visit(var)
            self.emit("")

        functions = getattr(node, "functions", [])
        for func in functions:
            self.visit(func)
            self.emit("")

        # Handle legacy shader structure
        if hasattr(node, "stages") and node.stages:
            emitted_local_functions = set()
            for stage_type, stage in node.stages.items():
                if hasattr(stage, "entry_point"):
                    # Set the stage type context for proper qualifier handling
                    stage_name = (
                        str(stage_type).split(".")[-1].lower()
                        if hasattr(stage_type, "name")
                        else str(stage_type).lower()
                    )

                    # Temporarily set qualifier for compute stages
                    if stage_name == "compute" or "compute" in stage_name:
                        # Set the function qualifier to compute for proper __global__ generation
                        if hasattr(stage.entry_point, "qualifiers"):
                            if "compute" not in stage.entry_point.qualifiers:
                                stage.entry_point.qualifiers.append("compute")
                        else:
                            stage.entry_point.qualifiers = ["compute"]

                    for func in getattr(stage, "local_functions", []):
                        if id(func) in emitted_local_functions:
                            continue
                        self.visit(func)
                        self.emit("")
                        emitted_local_functions.add(id(func))

                    self.visit(stage.entry_point)
                    self.emit("")

    def visit_FunctionNode(self, node):
        """Render a CrossGL function or compute entry point as CUDA code."""
        saved_variable_types = self.variable_types.copy()
        saved_image_resource_accesses = self.image_resource_accesses.copy()
        saved_query_metadata_aliases = self.query_metadata_aliases
        saved_current_function_name = self.current_function_name
        saved_structured_buffer_length_parameters = (
            self.current_structured_buffer_length_parameters
        )
        self.current_function_name = node.name
        self.current_structured_buffer_length_parameters = {}
        self.query_metadata_aliases = {}
        qualifiers = []

        if hasattr(node, "qualifiers") and node.qualifiers:
            for qualifier in node.qualifiers:
                if qualifier == "compute":
                    qualifiers.append("__global__")
                elif qualifier in ["vertex", "fragment"]:
                    qualifiers.append("__device__")
                else:
                    qualifiers.append("__device__")
        elif hasattr(node, "qualifier") and node.qualifier:
            if node.qualifier == "compute":
                qualifiers.append("__global__")
            elif node.qualifier in ["vertex", "fragment"]:
                qualifiers.append("__device__")
            else:
                qualifiers.append("__device__")
        else:
            qualifiers.append("__device__")

        if hasattr(node, "return_type"):
            return_type = self.convert_crossgl_type_to_cuda(node.return_type)
        else:
            return_type = "void"

        qualifier_str = " ".join(qualifiers)

        params = []
        param_list = getattr(node, "parameters", getattr(node, "params", []))

        for param in param_list:
            if hasattr(param, "param_type"):
                param_type = param.param_type
            elif hasattr(param, "vtype"):
                param_type = param.vtype
            else:
                param_type = "void"

            self.register_variable_type(param.name, param_type, param)
            declaration_type = self.apply_readonly_qualifier_to_type(param_type, param)
            params.append(self.format_typed_declarator(declaration_type, param.name))
            metadata_param = self.query_metadata_parameter(param.name, param_type)
            if metadata_param:
                params.append(metadata_param)
            length_param = self.structured_buffer_length_parameter(
                node.name, param.name, param_type
            )
            if length_param:
                self.current_structured_buffer_length_parameters[param.name] = (
                    self.structured_buffer_length_name(param.name)
                )
                params.append(length_param)
            counter_param = self.structured_buffer_counter_parameter(
                param.name, param_type
            )
            if counter_param:
                params.append(counter_param)

        param_str = ", ".join(params)
        self.emit(f"{qualifier_str} {return_type} {node.name}({param_str}) {{")

        self.indent_level += 1

        body = getattr(node, "body", [])
        self.emit_body(body)

        self.indent_level -= 1
        self.emit("}")
        self.variable_types = saved_variable_types
        self.image_resource_accesses = saved_image_resource_accesses
        self.query_metadata_aliases = saved_query_metadata_aliases
        self.current_function_name = saved_current_function_name
        self.current_structured_buffer_length_parameters = (
            saved_structured_buffer_length_parameters
        )

    def visit_StructNode(self, node):
        self.emit(f"struct {node.name} {{")
        self.indent_level += 1

        members = getattr(node, "members", [])
        member_types = {}
        member_image_accesses = {}
        for member in members:
            if hasattr(member, "member_type"):
                member_type = member.member_type
            elif hasattr(member, "vtype"):
                member_type = member.vtype
            else:
                member_type = "float"

            member_type = self.apply_readonly_qualifier_to_type(member_type, member)
            member_types[member.name] = member_type
            member_access = self.explicit_resource_access(member)
            if member_access is not None and self.is_storage_image_type(member_type):
                member_image_accesses[member.name] = member_access
            self.emit(f"{self.format_typed_declarator(member_type, member.name)};")

        self.struct_member_types[node.name] = member_types
        self.struct_member_image_accesses[node.name] = member_image_accesses
        self.indent_level -= 1
        self.emit("};")

    def visit_EnumNode(self, node):
        self.emit(f"enum {node.name} {{")
        self.indent_level += 1

        variants = getattr(node, "variants", [])
        for index, variant in enumerate(variants):
            suffix = "," if index < len(variants) - 1 else ""
            value = getattr(variant, "value", None)
            if value is not None:
                self.emit(f"{variant.name} = {self.visit(value)}{suffix}")
            else:
                self.emit(f"{variant.name}{suffix}")

        self.indent_level -= 1
        self.emit("};")

    def format_variable_declaration(self, node):
        var_type = None
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))

        if hasattr(node, "var_type"):
            var_type = node.var_type
        elif hasattr(node, "vtype"):
            var_type = node.vtype

        if not var_type and initial_value is not None:
            inferred_type = self.expression_result_type(initial_value)
            self.register_variable_type(
                node.name,
                inferred_type or "auto",
                source_node=initial_value,
            )
            if self.is_query_metadata_snapshot_local(
                node.name,
                inferred_type or "auto",
            ):
                self.query_metadata_aliases[node.name] = self.query_metadata_name(
                    node.name
                )
            else:
                self.register_query_metadata_alias(
                    node.name,
                    inferred_type or "auto",
                    initial_value,
                )
            return f"auto {node.name} = {self.visit(initial_value)}"

        if var_type:
            self.register_variable_type(node.name, var_type, node, initial_value)
            if self.is_query_metadata_snapshot_local(node.name, var_type):
                self.query_metadata_aliases[node.name] = self.query_metadata_name(
                    node.name
                )
            else:
                self.register_query_metadata_alias(node.name, var_type, initial_value)
            # Check for special memory qualifiers
            qualifiers = []
            if hasattr(node, "qualifiers"):
                for qualifier in node.qualifiers:
                    qualifier = str(qualifier).lower()
                    if qualifier == "const":
                        qualifiers.append("const")
                    elif qualifier == "static":
                        qualifiers.append("static")
                    elif "workgroup" in qualifier or "shared" in qualifier:
                        qualifiers.append("__shared__")
                    elif "uniform" in qualifier:
                        qualifiers.append("__constant__")

            qualifier_str = " ".join(qualifiers)
            if qualifier_str:
                qualifier_str += " "

            declaration = (
                f"{qualifier_str}{self.format_typed_declarator(var_type, node.name)}"
            )
            if initial_value is not None:
                declaration += f" = {self.visit(initial_value)}"
            return declaration

        return node.name

    def format_typed_declarator(self, type_name, name, dynamic_array_as_pointer=True):
        type_name = self.type_name_string(type_name)

        if "[" not in type_name or "]" not in type_name:
            return f"{self.convert_crossgl_type_to_cuda(type_name)} {name}"

        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket]
        array_suffix = type_name[open_bracket:]
        mapped_base = self.convert_crossgl_type_to_cuda(base_type)

        return format_array_declarator(
            mapped_base,
            name,
            array_suffix,
            dynamic_array_as_pointer=dynamic_array_as_pointer,
        )

    def format_array_size(self, size):
        if size is None:
            return ""
        if isinstance(size, int):
            return str(size)
        return self.visit(size)

    def visit_VariableNode(self, node):
        var_type = self.get_variable_node_type(node)
        declaration = self.format_variable_declaration(node)
        if declaration != node.name:
            self.emit(f"{declaration};")
            initial_value = getattr(
                node,
                "initial_value",
                getattr(node, "value", None),
            )
            effective_var_type = var_type
            if effective_var_type is None and initial_value is not None:
                effective_var_type = self.expression_result_type(initial_value)
            metadata_declaration = self.query_metadata_snapshot_declaration(
                node.name,
                effective_var_type,
                initial_value,
            )
            if metadata_declaration is None:
                metadata_declaration = self.query_metadata_declaration(
                    node.name,
                    effective_var_type,
                )
            if metadata_declaration:
                self.emit(f"{metadata_declaration};")
            length_declaration = self.structured_buffer_length_declaration(
                node.name, var_type
            )
            if length_declaration:
                self.emit(f"{length_declaration};")
            counter_declaration = self.structured_buffer_counter_declaration(
                node.name, var_type
            )
            if counter_declaration:
                self.emit(f"{counter_declaration};")
            return None

        return node.name

    def visit_ExpressionStatementNode(self, node):
        if isinstance(node.expression, AssignmentNode):
            self.emit_assignment_statement(node.expression)
            return None

        expr = self.visit(node.expression)
        if expr and expr.strip():
            self.emit(f"{expr};")

    def emit_assignment_statement(self, node):
        assignment = self.format_assignment_expression(node)
        if assignment and assignment.strip():
            self.emit(f"{assignment};")

        metadata_assignment = self.format_query_metadata_assignment(node)
        if metadata_assignment:
            self.emit(f"{metadata_assignment};")

    def format_for_clause_expression(self, node):
        """Return a for-clause expression, preserving CUDA metadata sidecars."""
        if node is None:
            return ""
        if isinstance(node, VariableNode):
            return self.format_variable_declaration(node)

        expr_node = getattr(node, "expression", node)
        if isinstance(expr_node, AssignmentNode):
            assignment = self.format_assignment_expression(expr_node)
            metadata_assignment = self.format_query_metadata_assignment(expr_node)
            parts = [
                part
                for part in (assignment, metadata_assignment)
                if part and part.strip()
            ]
            return ", ".join(parts)

        return self.visit(expr_node) or ""

    def visit_IdentifierNode(self, node):
        name = getattr(node, "name", str(node))
        return self.builtin_map.get(name, name)

    def format_literal(self, value, literal_type=None):
        if isinstance(value, bool):
            return "true" if value else "false"
        if literal_type == "bool" and isinstance(value, str):
            lower_value = value.lower()
            if lower_value in {"true", "false"}:
                return lower_value
        if literal_type == "char":
            escaped = self.escape_literal(value, quote="'")
            return f"'{escaped}'"
        if (
            literal_type == "uint"
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return f"{value}u"
        if isinstance(value, str):
            escaped = self.escape_literal(value, quote='"')
            return f'"{escaped}"'
        return str(value)

    def escape_literal(self, value, quote):
        text = str(value)
        escaped = []
        for index, char in enumerate(text):
            if char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == quote and (index == 0 or text[index - 1] != "\\"):
                escaped.append("\\" + char)
            else:
                escaped.append(char)
        return "".join(escaped)

    def visit_LiteralNode(self, node):
        literal_type = getattr(getattr(node, "literal_type", None), "name", None)
        return self.format_literal(node.value, literal_type)

    def visit_AssignmentNode(self, node):
        return self.format_assignment_expression(node)

    def format_assignment_expression(self, node):
        target = self.visit(node.target)
        value = self.visit(node.value)
        operator = getattr(node, "operator", "=")
        compound_binary_ops = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
            "%=": "%",
        }
        if operator in compound_binary_ops:
            lowered_value = self.lower_vector_binary_operation(
                node.target,
                target,
                node.value,
                value,
                compound_binary_ops[operator],
            )
            if lowered_value is not None:
                return f"{target} = {lowered_value}"
            if operator == "%=":
                modulo = self.lower_scalar_modulo_operation(
                    node.target,
                    target,
                    node.value,
                    value,
                )
                if modulo is not None:
                    return f"{target} = {modulo}"
        compound_bitwise_ops = {
            "&=": "&",
            "|=": "|",
            "^=": "^",
            "<<=": "<<",
            ">>=": ">>",
        }
        if operator in compound_bitwise_ops:
            lowered_value = self.lower_vector_bitwise_operation(
                node.target,
                target,
                node.value,
                value,
                compound_bitwise_ops[operator],
            )
            if lowered_value is not None:
                return f"{target} = {lowered_value}"
        return f"{target} {operator} {value}"

    def visit_BinaryOpNode(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        operator = getattr(node, "operator", getattr(node, "op", "+"))
        logical = self.lower_vector_logical_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if logical is not None:
            return logical
        bitwise = self.lower_vector_bitwise_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if bitwise is not None:
            return bitwise
        comparison = self.lower_vector_comparison_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if comparison is not None:
            return comparison
        lowered = self.lower_vector_binary_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if lowered is not None:
            return lowered
        if operator == "%":
            modulo = self.lower_scalar_modulo_operation(
                node.left,
                left,
                node.right,
                right,
            )
            if modulo is not None:
                return modulo
        return f"({left} {operator} {right})"

    def visit_UnaryOpNode(self, node):
        operand = self.visit(node.operand)
        operator = getattr(node, "operator", getattr(node, "op", "+"))
        if operator == "not":
            operator = "!"
        if getattr(node, "is_postfix", getattr(node, "postfix", False)):
            return f"{operand}{operator}"
        lowered = self.lower_vector_unary_operation(node.operand, operand, operator)
        if lowered is not None:
            return lowered
        return f"{operator}{operand}"

    def visit_FunctionCallNode(self, node):
        """Visit function call"""
        function_expr = getattr(node, "function", getattr(node, "name", None))
        if hasattr(node, "function"):
            func_name = self.visit(function_expr)
        else:
            func_name = getattr(node, "name", "unknown")

        raw_args = []
        if hasattr(node, "arguments"):
            raw_args = node.arguments
        elif hasattr(node, "args"):
            raw_args = node.args

        args = [self.visit(arg) for arg in raw_args]

        if func_name == "lambda":
            return self.generate_lambda_expression(raw_args)

        is_user_function = self.is_user_defined_function(func_name)
        if not is_user_function:
            buffer_call = self.generate_buffer_call(
                function_expr, func_name, raw_args, args
            )
            if buffer_call is not None:
                return buffer_call

            structured_atomic_call = self.generate_structured_buffer_atomic_call(
                func_name, raw_args, args
            )
            if structured_atomic_call is not None:
                return structured_atomic_call

            plain_atomic_call = self.generate_plain_atomic_call(
                func_name, raw_args, args
            )
            if plain_atomic_call is not None:
                return plain_atomic_call

            resource_call = self.generate_resource_call(func_name, raw_args, args)
            if resource_call is not None:
                return resource_call

        args = self.cuda_user_function_call_arguments(func_name, raw_args, args)
        if is_user_function:
            return f"{func_name}({', '.join(args)})"

        if func_name in self.synchronization_builtins and raw_args:
            raise ValueError(
                f"CUDA synchronization builtin '{func_name}' requires 0 "
                f"arguments; got {len(raw_args)}"
            )

        if func_name == "abs" and len(args) == 1:
            abs_call = self.generate_abs_call(raw_args, args)
            if abs_call is not None:
                return abs_call

        if func_name == "sign" and len(args) == 1:
            sign_call = self.generate_sign_call(raw_args, args)
            if sign_call is not None:
                return sign_call

        if func_name == "mod" and len(args) == 2:
            mod_call = self.generate_mod_call(raw_args, args)
            if mod_call is not None:
                return mod_call

        if func_name in {"fract", "frac"} and len(args) == 1:
            fract_call = self.generate_fract_call(raw_args, args)
            if fract_call is not None:
                return fract_call

        if func_name == "clamp" and len(args) == 3:
            return self.generate_clamp_call(raw_args, args)

        if func_name in {"min", "max"} and len(args) == 2:
            min_max_call = self.generate_min_max_call(func_name, raw_args, args)
            if min_max_call is not None:
                return min_max_call

        if func_name == "atan2" and len(args) == 2:
            atan2_call = self.generate_atan2_call(raw_args, args)
            if atan2_call is not None:
                return atan2_call

        if func_name == "mix" and len(args) == 3:
            mix_call = self.generate_mix_call(raw_args, args)
            if mix_call is not None:
                return mix_call

        if func_name == "saturate" and len(args) == 1:
            saturate_call = self.generate_saturate_call(raw_args, args)
            if saturate_call is not None:
                return saturate_call

        if func_name in {"dot", "cross", "length", "normalize"}:
            geometric_call = self.generate_vector_geometric_call(
                func_name,
                raw_args,
                args,
            )
            if geometric_call is not None:
                return geometric_call

        vector_info = self.vector_type_info(func_name)
        if vector_info:
            splat_call = self.generate_vector_scalar_splat_call(
                vector_info, raw_args, args
            )
            if splat_call is not None:
                return splat_call
            constructor_call = self.generate_vector_constructor_single_eval_call(
                vector_info, raw_args, args
            )
            if constructor_call is not None:
                return constructor_call
            args = self.generate_vector_constructor_args(vector_info, raw_args, args)

        scalar_math_call = self.generate_scalar_math_call(func_name, raw_args, args)
        if scalar_math_call is not None:
            return scalar_math_call

        args_str = ", ".join(args)

        # Convert built-in functions
        func_name = self.convert_builtin_function(func_name)
        return f"{func_name}({args_str})"

    def generate_buffer_call(self, function_expr, func_name, raw_args, args):
        """Lower structured-buffer loads and stores to CUDA pointer indexing."""
        byte_address_call = self.generate_byte_address_buffer_call(
            function_expr, func_name, raw_args, args
        )
        if byte_address_call is not None:
            return byte_address_call

        member_call = self.structured_buffer_member_call(function_expr)
        if member_call is not None:
            buffer_expr, operation, buffer_type = member_call
            if operation == "Append" and args:
                return self.generate_structured_buffer_append(
                    buffer_expr, buffer_type, args[0], operation
                )
            if operation == "Consume":
                return self.generate_structured_buffer_consume(
                    buffer_expr, buffer_type, operation
                )
            if operation == "GetDimensions":
                return self.generate_structured_buffer_dimensions(
                    buffer_expr, buffer_type, raw_args, args, operation
                )
            access = self.format_structured_buffer_access(buffer_expr, raw_args, args)
            if access is None:
                return None
            if operation == "Load":
                return access
            if operation == "Store":
                if self.structured_buffer_is_writable(buffer_type) and len(args) >= 2:
                    return f"{access} = {args[1]}"
                return self.unsupported_structured_buffer_call(
                    "Store", buffer_type, "((void)0)"
                )
            return None

        if func_name == "buffer_load" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return f"{args[0]}[{args[1]}]"

        if func_name == "buffer_append" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return self.generate_structured_buffer_append(
                raw_args[0], buffer_type, args[1], func_name
            )

        if func_name == "buffer_consume" and args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return self.generate_structured_buffer_consume(
                raw_args[0], buffer_type, func_name
            )

        if func_name == "buffer_dimensions" and args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return self.generate_structured_buffer_dimensions(
                raw_args[0], buffer_type, raw_args[1:], args[1:], func_name
            )

        if func_name == "buffer_store" and len(args) >= 3:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            if self.structured_buffer_is_writable(buffer_type):
                return f"{args[0]}[{args[1]}] = {args[2]}"
            return self.unsupported_structured_buffer_call(
                "buffer_store", buffer_type, "((void)0)"
            )

        return None

    def generate_structured_buffer_dimensions(
        self, buffer_expr, buffer_type, raw_dimension_args, dimension_args, operation
    ):
        """Lower structured-buffer dimensions through an explicit length sidecar."""
        if self.structured_buffer_type_parts(buffer_type) is None:
            return None

        length_expr = self.structured_buffer_length_expression(buffer_expr)
        if length_expr is None:
            length_expr = (
                "0 /* CUDA structured buffer dimensions "
                "requires explicit length sidecar */"
            )

        if dimension_args:
            return f"{dimension_args[0]} = {length_expr}"
        return length_expr

    def generate_structured_buffer_append(
        self, buffer_expr, buffer_type, value, operation
    ):
        """Lower AppendStructuredBuffer writes through an explicit counter."""
        parts = self.structured_buffer_type_parts(buffer_type)
        if parts is None or parts[0] != "AppendStructuredBuffer":
            return self.unsupported_structured_buffer_call(
                operation, buffer_type, "((void)0)"
            )

        counter = self.structured_buffer_counter_expression(buffer_expr)
        if counter is None:
            return self.unsupported_structured_buffer_call(
                operation, buffer_type, "((void)0)"
            )

        helper_name = self.require_structured_buffer_append_helper()
        buffer_name = self.visit(buffer_expr)
        return f"{helper_name}({buffer_name}, {counter}, {value})"

    def generate_structured_buffer_consume(self, buffer_expr, buffer_type, operation):
        """Lower ConsumeStructuredBuffer reads through an explicit counter."""
        parts = self.structured_buffer_type_parts(buffer_type)
        if parts is None or parts[0] != "ConsumeStructuredBuffer":
            fallback = self.diagnostic_zero_value_for_type(
                parts[1] if parts is not None else None
            )
            return self.unsupported_structured_buffer_call(
                operation, buffer_type, fallback
            )

        counter = self.structured_buffer_counter_expression(buffer_expr)
        if counter is None:
            return self.unsupported_structured_buffer_call(
                operation,
                buffer_type,
                self.diagnostic_zero_value_for_type(parts[1]),
            )

        helper_name = self.require_structured_buffer_consume_helper()
        buffer_name = self.visit(buffer_expr)
        return f"{helper_name}({buffer_name}, {counter})"

    def require_structured_buffer_append_helper(self):
        """Register the CUDA helper for AppendStructuredBuffer operations."""
        helper_name = "cgl_append_structured_buffer"
        if helper_name in self.helper_functions:
            return helper_name

        helper = (
            "template <typename T>\n"
            "__device__ inline void "
            f"{helper_name}(T* buffer, uint* counter, const T& value)\n"
            "{\n"
            "    uint index = atomicAdd(counter, 1u);\n"
            "    buffer[index] = value;\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_structured_buffer_consume_helper(self):
        """Register the CUDA helper for ConsumeStructuredBuffer operations."""
        helper_name = "cgl_consume_structured_buffer"
        if helper_name in self.helper_functions:
            return helper_name

        helper = (
            "template <typename T>\n"
            "__device__ inline T "
            f"{helper_name}(const T* buffer, uint* counter)\n"
            "{\n"
            "    uint index = atomicSub(counter, 1u) - 1u;\n"
            "    return buffer[index];\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def structured_buffer_atomic_operations(self):
        """Return generic atomic mappings for RWStructuredBuffer element targets."""
        integer_kinds = {"int", "uint"}
        add_exchange_kinds = {*integer_kinds, "float"}
        return {
            "atomicAdd": ("atomicAdd", 2, add_exchange_kinds, "int/uint/float"),
            "atomicSub": ("atomicSub", 2, integer_kinds, "int/uint"),
            "atomicMin": ("atomicMin", 2, integer_kinds, "int/uint"),
            "atomicMax": ("atomicMax", 2, integer_kinds, "int/uint"),
            "atomicAnd": ("atomicAnd", 2, integer_kinds, "int/uint"),
            "atomicOr": ("atomicOr", 2, integer_kinds, "int/uint"),
            "atomicXor": ("atomicXor", 2, integer_kinds, "int/uint"),
            "atomicInc": ("atomicInc", 2, {"uint"}, "uint"),
            "atomicDec": ("atomicDec", 2, {"uint"}, "uint"),
            "atomicExchange": ("atomicExch", 2, add_exchange_kinds, "int/uint/float"),
            "atomicCompareExchange": ("atomicCAS", 3, integer_kinds, "int/uint"),
            "atomicCompSwap": ("atomicCAS", 3, integer_kinds, "int/uint"),
        }

    def generate_structured_buffer_atomic_call(self, func_name, raw_args, args):
        """Lower atomics on RWStructuredBuffer element lvalues to CUDA atomics."""
        operation = self.structured_buffer_atomic_operations().get(func_name)
        if operation is None or not raw_args:
            return None

        target = self.structured_buffer_atomic_target(raw_args[0])
        if target is None:
            return None

        intrinsic, required_arg_count, supported_kinds, supported_kinds_label = (
            operation
        )
        target_type = target["target_type"]
        fallback = self.diagnostic_zero_value_for_type(target_type)

        if len(args) != required_arg_count:
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                f"requires {required_arg_count} argument(s)",
                fallback,
            )

        buffer_base_type, _ = self.structured_buffer_type_parts(target["buffer_type"])
        if buffer_base_type != "RWStructuredBuffer":
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                "requires RWStructuredBuffer target",
                fallback,
            )

        scalar_kind = self.cuda_atomic_scalar_kind(target_type)
        if scalar_kind not in supported_kinds:
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                f"requires supported scalar {supported_kinds_label} target",
                fallback,
            )

        target_expr = self.visit(target["target_expr"])
        value_args = ", ".join(args[1:])
        return f"{intrinsic}(&{target_expr}, {value_args})"

    def generate_plain_atomic_call(self, func_name, raw_args, args):
        """Lower CUDA atomics on ordinary scalar lvalues to pointer operands."""
        operation = self.structured_buffer_atomic_operations().get(func_name)
        if operation is None:
            return None

        intrinsic, required_arg_count, supported_kinds, supported_kinds_label = (
            operation
        )
        raw_target = raw_args[0] if raw_args else None
        target_expr = (
            self.strip_address_of_expression(raw_target) if raw_target else None
        )
        address_taken = target_expr is not raw_target
        target_type = self.expression_result_type(target_expr) if target_expr else None
        indirect_info = self.cuda_indirect_type_info(target_type)
        pointee_type = (
            indirect_info["pointee_type"] if indirect_info is not None else None
        )
        atomic_type = pointee_type or target_type
        fallback = self.diagnostic_zero_value_for_type(atomic_type)

        if len(args) != required_arg_count:
            return self.unsupported_plain_atomic_call(
                func_name,
                f"requires {required_arg_count} argument(s)",
                fallback,
            )

        if target_expr is None or (
            pointee_type is None and not self.is_plain_atomic_lvalue(target_expr)
        ):
            return self.unsupported_plain_atomic_call(
                func_name,
                "requires assignable scalar target",
                fallback,
            )

        if (
            address_taken
            and indirect_info is not None
            and indirect_info["kind"] == "pointer"
        ):
            return self.unsupported_plain_atomic_call(
                func_name,
                f"on {indirect_info['type_label']} requires pointer target, "
                "not address of pointer",
                fallback,
            )

        readonly_reason = self.plain_atomic_readonly_target_reason(
            target_expr, indirect_info
        )
        if readonly_reason is not None:
            return self.unsupported_plain_atomic_call(
                func_name,
                readonly_reason,
                fallback,
            )

        scalar_kind = self.cuda_atomic_scalar_kind(atomic_type)
        if scalar_kind not in supported_kinds:
            type_label = (
                indirect_info["type_label"]
                if indirect_info is not None
                else self.type_name_string(atomic_type) or "unknown target"
            )
            return self.unsupported_plain_atomic_call(
                func_name,
                f"on {type_label} requires supported scalar "
                f"{supported_kinds_label} target",
                fallback,
            )

        target_code = self.visit(target_expr)
        if indirect_info is not None and indirect_info["kind"] == "pointer":
            target_pointer = target_code
        else:
            target_pointer = f"&{target_code}"
        value_args = ", ".join(args[1:])
        return f"{intrinsic}({target_pointer}, {value_args})"

    def is_plain_atomic_lvalue(self, expr):
        """Return whether an expression can be addressed for a CUDA atomic."""
        return isinstance(
            expr, (IdentifierNode, ArrayAccessNode, MemberAccessNode, PointerAccessNode)
        )

    def cuda_atomic_pointer_pointee_type(self, type_name):
        """Return a scalar pointee type for pointer-typed CUDA atomic operands."""
        info = self.cuda_indirect_type_info(type_name)
        return info["pointee_type"] if info is not None else None

    def cuda_indirect_type_info(self, type_name):
        """Return pointer/reference metadata for CUDA indirect type spellings."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        if type_name.startswith("ptr<") and type_name.endswith(">"):
            pointee = type_name[4:-1].strip()
            return {
                "kind": "pointer",
                "pointee_type": pointee,
                "readonly": False,
                "type_label": f"{pointee}*",
            }

        mapped_type = self.convert_crossgl_type_to_cuda(type_name)
        for candidate in (type_name, mapped_type):
            info = self.cuda_pointer_or_reference_type_info(candidate)
            if info is not None:
                return info
        return None

    def cuda_pointer_or_reference_type_info(self, type_name):
        """Parse a CUDA pointer/reference spelling into target metadata."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        stripped = type_name.strip()
        for suffix, kind in (("*", "pointer"), ("&", "reference")):
            if not stripped.endswith(suffix):
                continue
            pointee = stripped[: -len(suffix)].strip()
            readonly = pointee.startswith("const ")
            if readonly:
                pointee = pointee[len("const ") :].strip()
            return {
                "kind": kind,
                "pointee_type": pointee,
                "readonly": readonly,
                "type_label": stripped,
            }
        return None

    def plain_atomic_readonly_target_reason(self, target_expr, indirect_info):
        """Return a diagnostic reason for readonly plain atomic targets."""
        if indirect_info is not None and indirect_info["readonly"]:
            if indirect_info["kind"] == "reference":
                return (
                    f"on {indirect_info['type_label']} requires mutable "
                    "reference target"
                )
            return (
                f"on {indirect_info['type_label']} requires writable pointer " "target"
            )

        if isinstance(target_expr, ArrayAccessNode):
            array_expr = getattr(
                target_expr, "array_expr", getattr(target_expr, "array", None)
            )
            array_type = self.expression_result_type(array_expr)
            array_info = self.cuda_indirect_type_info(array_type)
            if array_info is not None and array_info["readonly"]:
                if array_info["kind"] == "reference":
                    return (
                        f"on {array_info['type_label']} requires mutable "
                        "reference target"
                    )
                return (
                    f"on {array_info['type_label']} requires writable pointer " "target"
                )
        return None

    def unsupported_plain_atomic_call(self, operation, reason, fallback):
        """Return diagnostic code for unsupported plain CUDA atomics."""
        return (
            f"/* unsupported {self.resource_backend_name()} atomic call: "
            f"{operation} {reason} */ {fallback}"
        )

    def structured_buffer_atomic_target(self, target_expr):
        """Return RWStructuredBuffer target metadata for an atomic lvalue."""
        target_expr = self.strip_address_of_expression(target_expr)
        element_access = self.structured_buffer_element_access(target_expr)
        if element_access is None:
            return None

        array_expr = getattr(
            element_access, "array_expr", getattr(element_access, "array", None)
        )
        buffer_type = self.expression_result_type(array_expr)
        parts = self.structured_buffer_type_parts(buffer_type)
        if parts is None:
            return None

        target_type = self.expression_result_type(target_expr) or parts[1]
        return {
            "buffer_type": buffer_type,
            "target_expr": target_expr,
            "target_type": target_type,
        }

    def strip_address_of_expression(self, expr):
        """Return the lvalue inside an address-of expression."""
        if isinstance(expr, UnaryOpNode):
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            if operator == "&":
                return getattr(expr, "operand", expr)
        return expr

    def structured_buffer_element_access(self, target_expr):
        """Return the structured-buffer element access inside an atomic lvalue."""
        if isinstance(target_expr, ArrayAccessNode):
            array_expr = getattr(
                target_expr, "array_expr", getattr(target_expr, "array", None)
            )
            buffer_type = self.expression_result_type(array_expr)
            if self.structured_buffer_type_parts(buffer_type) is not None:
                return target_expr

        if isinstance(target_expr, MemberAccessNode):
            object_expr = getattr(
                target_expr, "object_expr", getattr(target_expr, "object", None)
            )
            return self.structured_buffer_element_access(object_expr)

        return None

    def cuda_atomic_scalar_kind(self, type_name):
        """Return the CUDA atomic scalar kind supported for structured buffers."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        mapped_type = self.convert_crossgl_type_to_cuda(type_name)
        if type_name in {"uint", "u32"} or mapped_type in {"uint", "unsigned int"}:
            return "uint"
        if type_name in {"int", "i32"} or mapped_type == "int":
            return "int"
        if type_name in {"float", "f32"} or mapped_type == "float":
            return "float"
        return None

    def unsupported_structured_buffer_atomic_call(
        self, operation, buffer_type, reason, fallback
    ):
        """Return diagnostic code for unsupported structured-buffer atomics."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} structured buffer atomic: "
            f"{operation} on {buffer_type} {reason} */ {fallback}"
        )

    def generate_byte_address_buffer_call(
        self, function_expr, func_name, raw_args, args
    ):
        """Lower byte-address buffer methods to typed CUDA byte-pointer helpers."""
        member_call = self.byte_address_buffer_member_call(function_expr)
        if member_call is not None:
            buffer_expr, operation, buffer_type = member_call
            if operation == "GetDimensions":
                return self.generate_byte_address_buffer_dimensions(
                    buffer_expr, buffer_type, raw_args, args, operation
                )

            if operation in self.byte_address_buffer_atomic_operations():
                return self.generate_byte_address_buffer_atomic(
                    buffer_expr, buffer_type, operation, args
                )

            component_count = self.byte_address_buffer_component_count(operation)
            if component_count is None:
                return None

            buffer_name = self.visit(buffer_expr)
            if operation.startswith("Load") and len(args) >= 1:
                helper_name = self.require_byte_address_load_helper(component_count)
                return f"{helper_name}({buffer_name}, {args[0]})"

            if operation.startswith("Store") and len(args) >= 2:
                if self.byte_address_buffer_is_writable(buffer_type):
                    helper_name = self.require_byte_address_store_helper(
                        component_count
                    )
                    return f"{helper_name}({buffer_name}, {args[0]}, {args[1]})"
                return self.unsupported_byte_address_buffer_call(
                    operation, buffer_type, "((void)0)"
                )
            return None

        if func_name == "buffer_load" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            helper_name = self.require_byte_address_load_helper(1)
            return f"{helper_name}({args[0]}, {args[1]})"

        if func_name == "buffer_dimensions" and args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            return self.generate_byte_address_buffer_dimensions(
                raw_args[0], buffer_type, raw_args[1:], args[1:], func_name
            )

        if func_name == "buffer_store" and len(args) >= 3:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            if self.byte_address_buffer_is_writable(buffer_type):
                helper_name = self.require_byte_address_store_helper(1)
                return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"
            return self.unsupported_byte_address_buffer_call(
                "buffer_store", buffer_type, "((void)0)"
            )

        return None

    def generate_byte_address_buffer_dimensions(
        self, buffer_expr, buffer_type, raw_dimension_args, dimension_args, operation
    ):
        """Lower byte-address buffer dimensions through a byte-length sidecar."""
        if self.byte_address_buffer_base_type(buffer_type) is None:
            return None

        length_expr = self.structured_buffer_length_expression(buffer_expr)
        if length_expr is None:
            length_expr = (
                "0 /* CUDA byte-address buffer dimensions "
                "requires explicit byte-length sidecar */"
            )

        if dimension_args:
            return f"{dimension_args[0]} = {length_expr}"
        return length_expr

    def byte_address_buffer_atomic_operations(self):
        """Return supported RWByteAddressBuffer atomic method mappings."""
        return {
            "InterlockedAdd": ("add", "atomicAdd", 2),
            "InterlockedMin": ("min", "atomicMin", 2),
            "InterlockedMax": ("max", "atomicMax", 2),
            "InterlockedAnd": ("and", "atomicAnd", 2),
            "InterlockedOr": ("or", "atomicOr", 2),
            "InterlockedXor": ("xor", "atomicXor", 2),
            "InterlockedExchange": ("exchange", "atomicExch", 2),
            "InterlockedCompareExchange": ("compare_exchange", "atomicCAS", 3),
        }

    def generate_byte_address_buffer_atomic(
        self, buffer_expr, buffer_type, operation, args
    ):
        """Lower RWByteAddressBuffer Interlocked* methods to CUDA atomics."""
        operation_info = self.byte_address_buffer_atomic_operations().get(operation)
        if operation_info is None:
            return None

        operation_name, intrinsic, required_args = operation_info
        has_out_arg = len(args) == required_args + 1
        fallback = "((void)0)" if has_out_arg else "0u"
        if len(args) < required_args or len(args) > required_args + 1:
            return self.unsupported_byte_address_buffer_call(
                operation, buffer_type, fallback
            )
        if not self.byte_address_buffer_is_writable(buffer_type):
            return self.unsupported_byte_address_buffer_call(
                operation, buffer_type, fallback
            )

        helper_name = self.require_byte_address_atomic_helper(operation_name, intrinsic)
        buffer_name = self.visit(buffer_expr)
        helper_args = [buffer_name, *args[:required_args]]
        call = f"{helper_name}({', '.join(helper_args)})"
        if has_out_arg:
            return f"{args[required_args]} = {call}"
        return call

    def byte_address_atomic_helper_name(self, operation):
        """Return a stable CUDA helper name for a byte-address atomic operation."""
        return f"cgl_byte_address_atomic_{operation}_uint"

    def require_byte_address_atomic_helper(self, operation, intrinsic):
        """Register a CUDA byte-address atomic helper and return its name."""
        helper_name = self.byte_address_atomic_helper_name(operation)
        if helper_name in self.helper_functions:
            return helper_name

        pointer_expr = "reinterpret_cast<unsigned int*>(buffer + offset)"
        if operation == "compare_exchange":
            helper = (
                "__device__ inline uint "
                f"{helper_name}(unsigned char* buffer, uint offset, "
                "uint compare_value, uint value)\n"
                "{\n"
                f"    return {intrinsic}({pointer_expr}, compare_value, value);\n"
                "}"
            )
            self.helper_functions[helper_name] = helper
            return helper_name

        helper = (
            "__device__ inline uint "
            f"{helper_name}(unsigned char* buffer, uint offset, uint value)\n"
            "{\n"
            f"    return {intrinsic}({pointer_expr}, value);\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def byte_address_buffer_member_call(self, function_expr):
        """Return byte-address buffer member call pieces, if applicable."""
        if not isinstance(function_expr, MemberAccessNode):
            return None

        buffer_expr = getattr(
            function_expr, "object_expr", getattr(function_expr, "object", None)
        )
        buffer_type = self.expression_result_type(buffer_expr)
        if self.byte_address_buffer_base_type(buffer_type) is None:
            return None

        return buffer_expr, getattr(function_expr, "member", ""), buffer_type

    def byte_address_buffer_component_count(self, operation):
        """Return the uint lane count for Load/Store byte-address operations."""
        operation_counts = {
            "Load": 1,
            "Load2": 2,
            "Load3": 3,
            "Load4": 4,
            "Store": 1,
            "Store2": 2,
            "Store3": 3,
            "Store4": 4,
        }
        return operation_counts.get(operation)

    def byte_address_buffer_value_type(self, component_count):
        """Return the CUDA uint vector type for a byte-address operation."""
        if component_count == 1:
            return "uint"
        return f"uint{component_count}"

    def byte_address_helper_suffix(self, component_count):
        """Return a stable helper-name suffix for byte-address operations."""
        if component_count == 1:
            return "uint"
        return f"uint{component_count}"

    def require_byte_address_load_helper(self, component_count):
        """Register a CUDA byte-address load helper and return its name."""
        helper_name = (
            f"cgl_byte_address_load_{self.byte_address_helper_suffix(component_count)}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        if component_count == 1:
            helper = (
                "__device__ inline uint "
                f"{helper_name}(const unsigned char* buffer, uint offset)\n"
                "{\n"
                "    return *reinterpret_cast<const uint*>(buffer + offset);\n"
                "}"
            )
            self.helper_functions[helper_name] = helper
            return helper_name

        scalar_helper = self.require_byte_address_load_helper(1)
        components = ("x", "y", "z", "w")[:component_count]
        args = [
            f"{scalar_helper}(buffer, offset + {index * 4}u)"
            for index, _ in enumerate(components)
        ]
        value_type = self.byte_address_buffer_value_type(component_count)
        constructor = self.convert_builtin_function(value_type)
        helper = (
            f"__device__ inline {value_type} "
            f"{helper_name}(const unsigned char* buffer, uint offset)\n"
            "{\n"
            f"    return {constructor}({', '.join(args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_byte_address_store_helper(self, component_count):
        """Register a CUDA byte-address store helper and return its name."""
        helper_name = (
            f"cgl_byte_address_store_{self.byte_address_helper_suffix(component_count)}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        value_type = self.byte_address_buffer_value_type(component_count)
        if component_count == 1:
            helper = (
                "__device__ inline void "
                f"{helper_name}(unsigned char* buffer, uint offset, uint value)\n"
                "{\n"
                "    *reinterpret_cast<uint*>(buffer + offset) = value;\n"
                "}"
            )
            self.helper_functions[helper_name] = helper
            return helper_name

        scalar_helper = self.require_byte_address_store_helper(1)
        components = ("x", "y", "z", "w")[:component_count]
        lines = [
            f"    {scalar_helper}(buffer, offset + {index * 4}u, value.{component});"
            for index, component in enumerate(components)
        ]
        helper = (
            f"__device__ inline void {helper_name}"
            f"(unsigned char* buffer, uint offset, {value_type} value)\n"
            "{\n" + "\n".join(lines) + "\n}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def unsupported_byte_address_buffer_call(self, operation, buffer_type, fallback):
        """Return diagnostic code for unsupported byte-address buffer operations."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} byte-address buffer call: "
            f"{operation} on {buffer_type} */ {fallback}"
        )

    def structured_buffer_member_call(self, function_expr):
        """Return structured-buffer member call pieces, if applicable."""
        if not isinstance(function_expr, MemberAccessNode):
            return None

        buffer_expr = getattr(
            function_expr, "object_expr", getattr(function_expr, "object", None)
        )
        buffer_type = self.expression_result_type(buffer_expr)
        if self.structured_buffer_type_parts(buffer_type) is None:
            return None

        return buffer_expr, getattr(function_expr, "member", ""), buffer_type

    def format_structured_buffer_access(self, buffer_expr, raw_args, args):
        """Format one CUDA pointer access for a structured-buffer operation."""
        if not raw_args or not args:
            return None
        buffer_name = self.visit(buffer_expr)
        return f"{buffer_name}[{args[0]}]"

    def unsupported_structured_buffer_call(self, operation, buffer_type, fallback):
        """Return diagnostic code for unsupported structured-buffer operations."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} structured buffer call: "
            f"{operation} on {buffer_type} */ {fallback}"
        )

    def generate_lambda_expression(self, args):
        """Render CrossGL's pseudo-lambda as a CUDA device lambda."""
        if not args:
            return "[&] __device__ () {}"

        params = ", ".join(self.generate_lambda_parameter(arg) for arg in args[:-1])
        body = self.generate_lambda_body(args[-1])
        return f"[&] __device__ ({params}) {body}"

    def generate_lambda_parameter(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        typed_param = self.split_lambda_typed_parameter(raw)
        if typed_param is None:
            param_name = self.lambda_fallback_parameter_name(raw)
            return f"auto {param_name}" if param_name else "auto"

        type_name, param_name = typed_param
        mapped_type = self.lambda_parameter_type(type_name)
        return f"{mapped_type} {param_name}"

    def generate_lambda_body(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        if raw.startswith("{") and raw.endswith("}"):
            return raw
        if raw:
            return f"{{ return {raw}; }}"
        return "{}"

    def lambda_raw_argument_text(self, arg):
        if isinstance(arg, IdentifierNode):
            return arg.name
        if isinstance(arg, str):
            return arg
        return self.visit(arg)

    def split_lambda_typed_parameter(self, raw):
        if not raw:
            return None
        if any(char in raw for char in "{}()"):
            return None
        parts = raw.rsplit(None, 1)
        if len(parts) != 2:
            return None
        type_name, param_name = parts
        if not param_name.isidentifier():
            return None
        return type_name, param_name

    def lambda_parameter_type(self, type_name):
        if "<" in type_name or ">" in type_name:
            return "auto"
        return self.convert_crossgl_type_to_cuda(type_name)

    def lambda_fallback_parameter_name(self, raw):
        if not raw:
            return ""
        candidate = raw.rsplit(None, 1)[-1]
        if candidate.isidentifier():
            return candidate
        return raw

    def generate_fract_call(self, raw_args, args):
        arg_type = self.expression_result_type(raw_args[0])
        vector_info = self.vector_type_info(arg_type)
        if vector_info is not None:
            helper_name = self.require_vector_fract_helper(vector_info)
            if helper_name is not None:
                return f"{helper_name}({args[0]})"
            return None

        scalar_type = self.fract_scalar_type(arg_type)
        if scalar_type is None:
            return None
        helper_name = self.require_scalar_fract_helper(scalar_type)
        return f"{helper_name}({args[0]})"

    def fract_scalar_type(self, type_name):
        if type_name is not None and not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        mapped_type = (
            self.convert_crossgl_type_to_cuda(type_name)
            if type_name is not None
            else None
        )
        if mapped_type == "double" or type_name in {"double", "f64"}:
            return "double"
        if mapped_type == "float" or type_name is None:
            return "float"
        return None

    def require_scalar_fract_helper(self, scalar_type):
        helper_name = f"cgl_fract_{scalar_type}"
        if helper_name in self.helper_functions:
            return helper_name

        floor_name = "floor" if scalar_type == "double" else "floorf"
        helper = (
            f"__device__ inline {scalar_type} {helper_name}({scalar_type} value)\n"
            "{\n"
            f"    return value - {floor_name}(value);\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_vector_fract_helper(self, vector_info):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double"}:
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_fract"
        if helper_name in self.helper_functions:
            return helper_name

        scalar_helper_name = self.require_scalar_fract_helper(component_type)
        components = [
            f"{scalar_helper_name}(value.{component})"
            for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {vector_type} {helper_name}({vector_type} value)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def generate_vector_constructor_args(self, vector_info, raw_args, args):
        """Flatten vector arguments passed to CUDA make_* constructors."""
        if len(args) == 1:
            arg_type = self.expression_result_type(raw_args[0])
            if arg_type is not None and not self.vector_type_info(arg_type):
                return args * len(vector_info["components"])

        generated_args = []
        for raw_arg, arg_expr in zip(raw_args, args):
            arg_info = self.vector_type_info(self.expression_result_type(raw_arg))
            if arg_info is None:
                generated_args.append(arg_expr)
            else:
                generated_args.extend(
                    self.vector_argument_lane_expressions(
                        raw_arg,
                        arg_expr,
                        arg_info,
                    )
                )

            if len(generated_args) >= len(vector_info["components"]):
                return generated_args[: len(vector_info["components"])]

        return generated_args

    def vector_argument_lane_expressions(self, raw_arg, arg_expr, arg_info):
        swizzle_components = self.member_swizzle_components(raw_arg)
        if swizzle_components is not None:
            object_node = getattr(
                raw_arg,
                "object_expr",
                getattr(raw_arg, "object", None),
            )
            object_expr = self.visit(object_node)
            return [f"{object_expr}.{component}" for component in swizzle_components]

        return [f"{arg_expr}.{component}" for component in arg_info["components"]]

    def member_swizzle_components(self, node):
        if not isinstance(node, MemberAccessNode):
            return None

        object_node = getattr(node, "object_expr", getattr(node, "object", None))
        vector_info = self.vector_type_info(self.expression_result_type(object_node))
        if vector_info is None:
            return None

        component_aliases = {
            "x": "x",
            "y": "y",
            "z": "z",
            "w": "w",
            "r": "x",
            "g": "y",
            "b": "z",
            "a": "w",
        }
        member = getattr(node, "member", "")
        components = [component_aliases.get(component) for component in member]
        if not components or any(component is None for component in components):
            return None

        available_components = vector_info["components"]
        if any(component not in available_components for component in components):
            return None

        return components

    def generate_clamp_call(self, raw_args, args):
        value_type = self.expression_result_type(raw_args[0])
        value_info = self.vector_type_info(value_type)
        if not value_info:
            scalar_type = self.clamp_scalar_type(value_type)
            scalar_call = self.generate_scalar_clamp_single_eval_call(
                scalar_type,
                raw_args,
                args,
            )
            if scalar_call is not None:
                return scalar_call
            return self.format_clamp_component(
                scalar_type,
                args[0],
                args[1],
                args[2],
            )

        min_info = self.vector_type_info(self.expression_result_type(raw_args[1]))
        max_info = self.vector_type_info(self.expression_result_type(raw_args[2]))
        if min_info and len(min_info["components"]) != len(value_info["components"]):
            return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"
        if max_info and len(max_info["components"]) != len(value_info["components"]):
            return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"

        helper_name = self.require_vector_clamp_helper(
            value_info,
            min_is_vector=min_info is not None,
            max_is_vector=max_info is not None,
        )
        if helper_name is None:
            return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"
        return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"

    def generate_saturate_call(self, raw_args, args):
        value_type = self.expression_result_type(raw_args[0])
        value_info = self.vector_type_info(value_type)
        component_type = (
            value_info["component_type"]
            if value_info is not None
            else self.scalar_component_type(value_type)
        )
        if component_type not in {"float", "double"}:
            return None
        return self.generate_clamp_call(
            [raw_args[0], None, None],
            [args[0], "0.0", "1.0"],
        )

    def generate_mix_call(self, raw_args, args):
        left_type = self.expression_result_type(raw_args[0])
        right_type = self.expression_result_type(raw_args[1])
        factor_type = self.expression_result_type(raw_args[2])
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        factor_info = self.vector_type_info(factor_type)

        if not left_info and not right_info and not factor_info:
            scalar_bool_mix = self.lower_bool_scalar_mix_operation(
                raw_args[0],
                args[0],
                raw_args[1],
                args[1],
                raw_args[2],
                args[2],
            )
            if scalar_bool_mix is not None:
                return scalar_bool_mix
            return None

        if left_info is None or right_info is None:
            return None
        if (
            len(left_info["components"]) != len(right_info["components"])
            or left_info["component_type"] != right_info["component_type"]
        ):
            return None

        bool_mix = self.lower_bool_vector_mix_operation(
            raw_args[0],
            args[0],
            raw_args[1],
            args[1],
            raw_args[2],
            args[2],
            left_info,
            right_info,
            factor_info,
        )
        if bool_mix is not None:
            return bool_mix

        if factor_info is not None and (
            len(factor_info["components"]) != len(left_info["components"])
            or factor_info["component_type"] != left_info["component_type"]
        ):
            return None

        helper_name = self.require_vector_mix_helper(
            left_info,
            factor_is_vector=factor_info is not None,
        )
        if helper_name is None:
            return None
        return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"

    def require_vector_mix_helper(self, vector_info, factor_is_vector):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double"}:
            return None

        vector_type = vector_info["type"]
        factor_shape = "vector" if factor_is_vector else "scalar"
        helper_name = f"cgl_{vector_type}_mix_{factor_shape}"
        if helper_name in self.helper_functions:
            return helper_name

        factor_type = (
            vector_type
            if factor_is_vector
            else self.vector_scalar_parameter_type(vector_info)
        )
        components = []
        for component in vector_info["components"]:
            factor_component = f"a.{component}" if factor_is_vector else "a"
            components.append(
                self.format_mix_component(
                    f"x.{component}",
                    f"y.{component}",
                    factor_component,
                )
            )

        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} x, {vector_type} y, {factor_type} a)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_mix_component(self, left, right, factor):
        return f"({left} + (({right} - {left}) * {factor}))"

    def generate_atan2_call(self, raw_args, args):
        y_type = self.expression_result_type(raw_args[0])
        x_type = self.expression_result_type(raw_args[1])
        y_info = self.vector_type_info(y_type)
        x_info = self.vector_type_info(x_type)

        if y_info is None and x_info is None:
            return None
        if y_info is None or x_info is None:
            return None
        if (
            len(y_info["components"]) != len(x_info["components"])
            or y_info["component_type"] != x_info["component_type"]
        ):
            return None

        helper_name = self.require_vector_atan2_helper(y_info)
        if helper_name is None:
            return None
        return f"{helper_name}({args[0]}, {args[1]})"

    def require_vector_atan2_helper(self, vector_info):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double"}:
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_atan2"
        if helper_name in self.helper_functions:
            return helper_name

        scalar_func = "atan2" if component_type == "double" else "atan2f"
        components = [
            f"{scalar_func}(y.{component}, x.{component})"
            for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} y, {vector_type} x)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def clamp_scalar_type(self, type_name):
        if type_name is None:
            return "float"
        mapped_type = self.convert_crossgl_type_to_cuda(type_name)
        if mapped_type in {"float", "double"}:
            return mapped_type
        if mapped_type in {
            "bool",
            "char",
            "unsigned char",
            "short",
            "unsigned short",
            "int",
            "uint",
            "unsigned int",
            "long long",
            "unsigned long long",
        }:
            return mapped_type
        return "float"

    def require_vector_clamp_helper(self, vector_info, min_is_vector, max_is_vector):
        if vector_info["component_type"] == "bool":
            return None

        vector_type = vector_info["type"]
        scalar_type = self.vector_scalar_parameter_type(vector_info)
        min_shape = "vector" if min_is_vector else "scalar"
        max_shape = "vector" if max_is_vector else "scalar"
        helper_name = f"cgl_{vector_type}_clamp"
        if min_shape != "vector" or max_shape != "vector":
            helper_name += f"_{min_shape}_min_{max_shape}_max"
        if helper_name in self.helper_functions:
            return helper_name

        min_type = vector_type if min_is_vector else scalar_type
        max_type = vector_type if max_is_vector else scalar_type
        components = vector_info["components"]
        constructor = vector_info["constructor"]
        args = []
        for component in components:
            value_component = f"value.{component}"
            min_component = f"min_value.{component}" if min_is_vector else "min_value"
            max_component = f"max_value.{component}" if max_is_vector else "max_value"
            args.append(
                self.format_clamp_component(
                    vector_info["component_type"],
                    value_component,
                    min_component,
                    max_component,
                )
            )

        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} value, {min_type} min_value, {max_type} max_value)\n"
            "{\n"
            f"    return {constructor}({', '.join(args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_clamp_component(self, component_type, value, min_value, max_value):
        if component_type == "float":
            return f"fmaxf({min_value}, fminf({max_value}, {value}))"
        if component_type == "double":
            return f"fmax({min_value}, fmin({max_value}, {value}))"
        return (
            f"(({value}) < ({min_value}) ? ({min_value}) : "
            f"(({value}) > ({max_value}) ? ({max_value}) : ({value})))"
        )

    def visit_MemberAccessNode(self, node):
        """Visit member access"""
        if hasattr(node, "object_expr"):
            obj = self.visit(node.object_expr)
        else:
            obj = self.visit(node.object)
        member_access = f"{obj}.{node.member}"
        if member_access in self.builtin_map:
            return self.builtin_map[member_access]

        swizzle = self.generate_vector_swizzle(node, obj)
        if swizzle is not None:
            return swizzle

        return member_access

    def visit_PointerAccessNode(self, node):
        """Visit C-style pointer member access."""
        pointer_expr = self.visit(getattr(node, "pointer_expr", None))
        return f"{pointer_expr}->{node.member}"

    def generate_vector_swizzle(self, node, object_expr):
        object_node = getattr(node, "object_expr", getattr(node, "object", None))
        vector_info = self.vector_type_info(self.expression_result_type(object_node))
        if vector_info is None:
            return None

        components = self.member_swizzle_components(node)
        if components is None:
            return None

        if len(components) == 1:
            return f"{object_expr}.{components[0]}"

        result_type = self.vector_type_for_components(
            vector_info["component_type"], len(components)
        )
        result_info = self.vector_type_info(result_type)
        if result_info is None:
            return None

        swizzle_call = self.generate_vector_swizzle_single_eval_call(
            result_info,
            vector_info,
            object_node,
            object_expr,
            components,
        )
        if swizzle_call is not None:
            return swizzle_call

        args = [f"{object_expr}.{component}" for component in components]
        return f"{result_info['constructor']}({', '.join(args)})"

    def visit_ArrayAccessNode(self, node):
        """Visit array access"""
        if hasattr(node, "array_expr"):
            array = self.visit(node.array_expr)
        else:
            array = self.visit(node.array)

        if hasattr(node, "index_expr"):
            index = self.visit(node.index_expr)
        else:
            index = self.visit(node.index)

        return f"{array}[{index}]"

    def visit_ArrayLiteralNode(self, node):
        elements = ", ".join(self.visit(element) for element in node.elements)
        return f"{{{elements}}}"

    def visit_IfNode(self, node):
        """Visit if statement"""
        condition = self.visit(node.condition)
        self.emit(f"if ({condition}) {{")

        self.indent_level += 1

        # Handle then branch
        if hasattr(node, "then_branch"):
            self.emit_body(node.then_branch)
        elif hasattr(node, "if_body"):
            self.emit_body(node.if_body)

        self.indent_level -= 1

        # Handle else branch
        if hasattr(node, "else_branch") and node.else_branch:
            self.emit("} else {")
            self.indent_level += 1

            self.emit_body(node.else_branch)

            self.indent_level -= 1
        elif hasattr(node, "else_body") and node.else_body:
            self.emit("} else {")
            self.indent_level += 1
            self.emit_body(node.else_body)
            self.indent_level -= 1

        self.emit("}")

    def visit_ForNode(self, node):
        """Visit for loop"""
        init_str = ""
        if node.init:
            init_str = self.format_for_clause_expression(node.init)

        condition_str = ""
        if node.condition:
            condition_str = self.visit(node.condition)

        update_str = ""
        if node.update:
            update_str = self.format_for_clause_expression(node.update)

        self.emit(f"for ({init_str}; {condition_str}; {update_str}) {{")

        self.indent_level += 1

        # Handle body
        if hasattr(node, "body"):
            self.emit_body(node.body)

        self.indent_level -= 1
        self.emit("}")

    def visit_ForInNode(self, node):
        """Lower CrossGL for-in loops to counted CUDA loops."""
        pattern = getattr(node, "pattern", "item")
        iterable = getattr(node, "iterable", None)

        if isinstance(iterable, RangeNode):
            start = self.visit(iterable.start)
            end = self.visit(iterable.end)
            comparator = "<=" if iterable.inclusive else "<"
        else:
            start = "0"
            end = self.visit(iterable)
            comparator = "<"

        self.emit(
            f"for (int {pattern} = {start}; {pattern} {comparator} {end}; ++{pattern}) {{"
        )
        self.indent_level += 1
        self.emit_body(getattr(node, "body", []))
        self.indent_level -= 1
        self.emit("}")

    def visit_WhileNode(self, node):
        """Visit while loop"""
        condition = self.visit(node.condition) if node.condition else ""
        self.emit(f"while ({condition}) {{")

        self.indent_level += 1

        if hasattr(node, "body"):
            self.emit_body(node.body)

        self.indent_level -= 1
        self.emit("}")

    def visit_DoWhileNode(self, node):
        """Visit do-while loop."""
        self.emit("do {")

        self.indent_level += 1
        if hasattr(node, "body"):
            self.emit_body(node.body)
        self.indent_level -= 1

        condition = self.visit(node.condition) if node.condition else ""
        self.emit(f"}} while ({condition});")

    def visit_SwitchNode(self, node):
        """Visit switch statement"""
        expression = self.visit(node.expression)
        self.emit(f"switch ({expression}) {{")

        self.indent_level += 1
        for case in getattr(node, "cases", []):
            self.visit(case)
        self.indent_level -= 1

        self.emit("}")

    def visit_MatchNode(self, node):
        """Lower simple CrossGL match statements to CUDA switch statements."""
        expression = self.visit(node.expression)
        self.emit(f"switch ({expression}) {{")

        self.indent_level += 1
        for arm in getattr(node, "arms", []):
            if not self.is_supported_switch_match_arm(arm):
                raise ValueError(
                    "Unsupported match arm for CUDA codegen; only unguarded "
                    "literal and wildcard patterns can be lowered to switch"
                )

            pattern = arm.pattern
            if isinstance(pattern, WildcardPatternNode):
                self.emit("default:")
            else:
                self.emit(f"case {self.visit(pattern.literal)}:")

            self.indent_level += 1
            self.emit_body(arm.body)
            if not self.statement_body_terminates(arm.body):
                self.emit("break;")
            self.indent_level -= 1
        self.indent_level -= 1

        self.emit("}")

    def visit_CaseNode(self, node):
        """Visit switch case/default label"""
        if getattr(node, "value", None) is None:
            self.emit("default:")
        else:
            value = self.visit(node.value)
            self.emit(f"case {value}:")

        self.indent_level += 1
        for stmt in getattr(node, "statements", []):
            self.emit_statement(stmt)
        self.indent_level -= 1

    def visit_ReturnNode(self, node):
        """Visit return statement"""
        if node.value:
            value = self.visit(node.value)
            self.emit(f"return {value};")
        else:
            self.emit("return;")

    def visit_BreakNode(self, node):
        """Visit break statement"""
        self.emit("break;")

    def visit_ContinueNode(self, node):
        """Visit continue statement"""
        self.emit("continue;")

    def visit_BlockNode(self, node):
        """Visit block statement"""
        self.emit_body(node.statements)

    def convert_crossgl_type_to_cuda(self, crossgl_type):
        """Convert CrossGL types to CUDA equivalents"""
        crossgl_type = self.type_name_string(crossgl_type)
        if crossgl_type is None:
            return "void"

        structured_buffer_type = self.cuda_structured_buffer_type(crossgl_type)
        if structured_buffer_type is not None:
            return structured_buffer_type

        byte_address_buffer_type = self.cuda_byte_address_buffer_type(crossgl_type)
        if byte_address_buffer_type is not None:
            return byte_address_buffer_type

        type_mapping = {
            # Basic types
            "void": "void",
            "bool": "bool",
            "i8": "char",
            "u8": "unsigned char",
            "i16": "short",
            "u16": "unsigned short",
            "i32": "int",
            "u32": "unsigned int",
            "i64": "long long",
            "u64": "unsigned long long",
            "f32": "float",
            "f64": "double",
            "int": "int",
            "float": "float",
            "double": "double",
            # Vector types (with generics)
            "vec2<f32>": "float2",
            "vec3<f32>": "float3",
            "vec4<f32>": "float4",
            "vec2<f64>": "double2",
            "vec3<f64>": "double3",
            "vec4<f64>": "double4",
            "vec2<i32>": "int2",
            "vec3<i32>": "int3",
            "vec4<i32>": "int4",
            "vec2<u32>": "uint2",
            "vec3<u32>": "uint3",
            "vec4<u32>": "uint4",
            # Vector types (without generics - for compatibility)
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "dvec2": "double2",
            "dvec3": "double3",
            "dvec4": "double4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "bvec2": "uchar2",
            "bvec3": "uchar3",
            "bvec4": "uchar4",
            "vec2<bool>": "uchar2",
            "vec3<bool>": "uchar3",
            "vec4<bool>": "uchar4",
            "bool2": "uchar2",
            "bool3": "uchar3",
            "bool4": "uchar4",
            # Matrix types
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float2x3",
            "mat2x4": "float2x4",
            "mat3x2": "float3x2",
            "mat3x3": "float3x3",
            "mat3x4": "float3x4",
            "mat4x2": "float4x2",
            "mat4x3": "float4x3",
            "mat4x4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double2x3",
            "dmat2x4": "double2x4",
            "dmat3x2": "double3x2",
            "dmat3x3": "double3x3",
            "dmat3x4": "double3x4",
            "dmat4x2": "double4x2",
            "dmat4x3": "double4x3",
            "dmat4x4": "double4x4",
            # Texture/resource types
            "sampler": "cudaTextureObject_t",
            "sampler1D": "texture<float4, 1>",
            "sampler1DArray": "cudaTextureObject_t",
            "sampler2D": "texture<float4, 2>",
            "sampler3D": "texture<float4, 3>",
            "samplerCube": "textureCube<float4>",
            "sampler2DArray": "cudaTextureObject_t",
            "sampler2DShadow": "cudaTextureObject_t",
            "sampler2DArrayShadow": "cudaTextureObject_t",
            "samplerCubeShadow": "cudaTextureObject_t",
            "samplerCubeArray": "cudaTextureObject_t",
            "samplerCubeArrayShadow": "cudaTextureObject_t",
            "sampler2DMS": "cudaTextureObject_t",
            "sampler2DMSArray": "cudaTextureObject_t",
            "image1D": "cudaSurfaceObject_t",
            "image1DArray": "cudaSurfaceObject_t",
            "image2D": "cudaSurfaceObject_t",
            "image3D": "cudaSurfaceObject_t",
            "imageCube": "cudaSurfaceObject_t",
            "imageCubeArray": "cudaSurfaceObject_t",
            "image2DArray": "cudaSurfaceObject_t",
            "image2DMS": "cudaSurfaceObject_t",
            "image2DMSArray": "cudaSurfaceObject_t",
            "iimage1D": "cudaSurfaceObject_t",
            "iimage1DArray": "cudaSurfaceObject_t",
            "iimage2D": "cudaSurfaceObject_t",
            "iimage3D": "cudaSurfaceObject_t",
            "iimageCube": "cudaSurfaceObject_t",
            "iimageCubeArray": "cudaSurfaceObject_t",
            "iimage2DArray": "cudaSurfaceObject_t",
            "iimage2DMS": "cudaSurfaceObject_t",
            "iimage2DMSArray": "cudaSurfaceObject_t",
            "uimage1D": "cudaSurfaceObject_t",
            "uimage1DArray": "cudaSurfaceObject_t",
            "uimage2D": "cudaSurfaceObject_t",
            "uimage3D": "cudaSurfaceObject_t",
            "uimageCube": "cudaSurfaceObject_t",
            "uimageCubeArray": "cudaSurfaceObject_t",
            "uimage2DArray": "cudaSurfaceObject_t",
            "uimage2DMS": "cudaSurfaceObject_t",
            "uimage2DMSArray": "cudaSurfaceObject_t",
            "buffer": "CUdeviceptr",
        }

        sampled_resource_type = self.canonical_sampled_resource_type(crossgl_type)
        if sampled_resource_type:
            return type_mapping.get(sampled_resource_type, sampled_resource_type)

        storage_resource_type = self.canonical_storage_resource_type(crossgl_type)
        if storage_resource_type:
            return type_mapping.get(storage_resource_type, storage_resource_type)

        # Handle arrays
        if crossgl_type.startswith("array<") and crossgl_type.endswith(">"):
            # Extract element type and size
            inner = crossgl_type[6:-1]  # Remove "array<" and ">"
            if "," in inner:
                parts = inner.split(",")
                element_type = parts[0].strip()
                size = parts[1].strip()
                cuda_element_type = self.convert_crossgl_type_to_cuda(element_type)
                return f"{cuda_element_type}[{size}]"
            else:
                cuda_element_type = self.convert_crossgl_type_to_cuda(inner)
                return f"{cuda_element_type}*"

        # Handle pointers
        if crossgl_type.startswith("ptr<") and crossgl_type.endswith(">"):
            element_type = crossgl_type[4:-1]  # Remove "ptr<" and ">"
            cuda_element_type = self.convert_crossgl_type_to_cuda(element_type)
            return f"{cuda_element_type}*"

        return type_mapping.get(crossgl_type, crossgl_type)

    def type_name_string(self, type_name):
        """Return a stable string spelling for TypeNode or legacy type values."""
        if type_name is None:
            return None
        if (
            hasattr(type_name, "name")
            or hasattr(type_name, "element_type")
            or hasattr(type_name, "pointee_type")
            or hasattr(type_name, "referenced_type")
        ):
            return self.convert_type_node_to_string(type_name)
        return str(type_name)

    def collect_structured_buffer_length_requirements(self, root):
        """Collect buffers that need explicit length sidecar parameters."""
        functions = self.query_collect_functions(root)
        functions_by_name = {getattr(func, "name", None): func for func in functions}
        functions_by_name = {
            name: func for name, func in functions_by_name.items() if name
        }
        param_names = {
            func_name: {
                getattr(param, "name", None)
                for param in getattr(func, "parameters", getattr(func, "params", []))
            }
            for func_name, func in functions_by_name.items()
        }
        param_names = {
            func_name: {name for name in names if name}
            for func_name, names in param_names.items()
        }

        global_length_names = set()
        function_param_length_names = {
            func_name: set() for func_name in functions_by_name
        }

        def mark_resource_name(func_name, resource_name):
            if not resource_name:
                return False
            if resource_name in param_names.get(func_name, set()):
                before = len(function_param_length_names[func_name])
                function_param_length_names[func_name].add(resource_name)
                return len(function_param_length_names[func_name]) != before
            before = len(global_length_names)
            global_length_names.add(resource_name)
            return len(global_length_names) != before

        for func_name, func in functions_by_name.items():
            for call in self.query_walk_nodes(getattr(func, "body", [])):
                if not isinstance(call, FunctionCallNode):
                    continue
                buffer_expr = self.structured_buffer_dimensions_target(call)
                if buffer_expr is None:
                    continue
                mark_resource_name(func_name, self.get_expression_name(buffer_expr))

        changed = True
        while changed:
            changed = False
            for caller_name, caller in functions_by_name.items():
                caller_params = param_names.get(caller_name, set())
                for call in self.query_walk_nodes(getattr(caller, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.raw_function_call_name(call)
                    callee = functions_by_name.get(callee_name)
                    if callee is None:
                        continue

                    callee_required = function_param_length_names.get(
                        callee_name, set()
                    )
                    if not callee_required:
                        continue

                    callee_params = getattr(
                        callee, "parameters", getattr(callee, "params", [])
                    )
                    raw_args = getattr(call, "arguments", getattr(call, "args", []))
                    for index, param in enumerate(callee_params):
                        if index >= len(raw_args):
                            continue
                        param_name = getattr(param, "name", None)
                        if param_name not in callee_required:
                            continue

                        arg_name = self.get_expression_name(raw_args[index])
                        if not arg_name:
                            continue
                        if arg_name in caller_params:
                            before = len(function_param_length_names[caller_name])
                            function_param_length_names[caller_name].add(arg_name)
                            changed = (
                                changed
                                or len(function_param_length_names[caller_name])
                                != before
                            )
                        else:
                            before = len(global_length_names)
                            global_length_names.add(arg_name)
                            changed = changed or len(global_length_names) != before

        return (
            global_length_names,
            {
                func_name: names
                for func_name, names in function_param_length_names.items()
                if names
            },
        )

    def structured_buffer_dimensions_target(self, call):
        """Return the structured-buffer expression queried by a dimensions call."""
        func_name = self.raw_function_call_name(call)
        raw_args = getattr(call, "arguments", getattr(call, "args", []))
        if func_name == "buffer_dimensions" and raw_args:
            return raw_args[0]

        function_expr = getattr(call, "function", getattr(call, "name", None))
        if isinstance(function_expr, MemberAccessNode):
            member = getattr(function_expr, "member", None)
            if member == "GetDimensions":
                return getattr(
                    function_expr,
                    "object_expr",
                    getattr(function_expr, "object", None),
                )
        return None

    def generic_type_parts(self, type_name):
        """Split a generic type name into base name and top-level arguments."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        base_type = type_name.split("[", 1)[0].strip()
        generic_start = base_type.find("<")
        generic_end = base_type.rfind(">")
        if generic_start == -1 or generic_end < generic_start:
            return None

        base_name = base_type[:generic_start].strip()
        args_text = base_type[generic_start + 1 : generic_end].strip()
        args = []
        depth = 0
        current = []
        for char in args_text:
            if char == "<":
                depth += 1
                current.append(char)
            elif char == ">":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(char)

        trailing_arg = "".join(current).strip()
        if trailing_arg:
            args.append(trailing_arg)
        return base_name, args

    def structured_buffer_type_parts(self, type_name):
        """Return structured-buffer base and element type, if applicable."""
        parts = self.generic_type_parts(type_name)
        if parts is None:
            return None

        base_name, args = parts
        if (
            base_name
            not in {
                "StructuredBuffer",
                "RWStructuredBuffer",
                "AppendStructuredBuffer",
                "ConsumeStructuredBuffer",
            }
            or not args
        ):
            return None
        return base_name, args[0]

    def structured_buffer_is_writable(self, type_name):
        """Return whether a structured-buffer type permits writes."""
        parts = self.structured_buffer_type_parts(type_name)
        return parts is not None and parts[0] == "RWStructuredBuffer"

    def structured_buffer_requires_counter(self, type_name):
        """Return whether a structured-buffer type needs an explicit counter."""
        parts = self.structured_buffer_type_parts(type_name)
        return parts is not None and parts[0] in {
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def cuda_structured_buffer_type(self, type_name):
        """Map structured-buffer resources to CUDA pointer types."""
        parts = self.structured_buffer_type_parts(type_name)
        if parts is None:
            return None

        base_name, element_type = parts
        cuda_element_type = self.convert_crossgl_type_to_cuda(element_type)
        if base_name in {"StructuredBuffer", "ConsumeStructuredBuffer"}:
            return f"const {cuda_element_type}*"
        return f"{cuda_element_type}*"

    def structured_buffer_length_name(self, name):
        """Return the sidecar length parameter/declaration name for a buffer."""
        return f"{name}_length"

    def structured_buffer_requires_length(self, name):
        """Return whether a global structured buffer needs a length sidecar."""
        return bool(name and name in self.structured_buffer_length_names)

    def structured_buffer_parameter_requires_length(
        self, func_name, name, type_name=None
    ):
        """Return whether a function parameter needs a length sidecar."""
        if not name:
            return False
        if name not in self.structured_buffer_length_function_params.get(
            func_name, set()
        ):
            return False
        return type_name is None or self.buffer_type_supports_length(type_name)

    def buffer_type_supports_length(self, type_name):
        """Return whether a resource type can use a CUDA length sidecar."""
        return (
            self.structured_buffer_type_parts(type_name) is not None
            or self.byte_address_buffer_base_type(type_name) is not None
        )

    def structured_buffer_length_declaration(self, name, type_name):
        """Format a global/local sidecar length declaration when required."""
        if not self.structured_buffer_requires_length(name):
            return None
        if not self.buffer_type_supports_length(type_name):
            return None
        return self.format_structured_buffer_length_declarator(type_name, name)

    def structured_buffer_length_parameter(self, func_name, name, type_name):
        """Format a sidecar length parameter when required."""
        if not self.structured_buffer_parameter_requires_length(
            func_name, name, type_name
        ):
            return None
        return self.format_structured_buffer_length_declarator(type_name, name)

    def format_structured_buffer_length_declarator(self, type_name, name):
        """Format the CUDA uint pointer sidecar matching a buffer declarator."""
        type_name = self.type_name_string(type_name)
        length_name = self.structured_buffer_length_name(name)
        if "[" not in type_name or "]" not in type_name:
            return f"const uint* {length_name}"

        array_suffix = type_name[type_name.find("[") :]
        return format_array_declarator("const uint*", length_name, array_suffix)

    def structured_buffer_length_data_expression(self, buffer_expr):
        """Return the sidecar length pointer paired with a buffer expression."""
        if isinstance(buffer_expr, ArrayAccessNode):
            array_expr = getattr(
                buffer_expr, "array_expr", getattr(buffer_expr, "array", None)
            )
            index_expr = getattr(
                buffer_expr, "index_expr", getattr(buffer_expr, "index", None)
            )
            base_length = self.structured_buffer_length_data_expression(array_expr)
            if base_length is None:
                return None
            return f"{base_length}[{self.visit(index_expr)}]"

        name = self.get_expression_name(buffer_expr)
        if not name:
            return None

        length_parameter = self.current_structured_buffer_length_parameters.get(name)
        if length_parameter is not None:
            return length_parameter

        if self.structured_buffer_requires_length(name):
            return self.structured_buffer_length_name(name)
        return None

    def structured_buffer_length_expression(self, buffer_expr):
        """Return the scalar length expression for a structured-buffer resource."""
        length_data = self.structured_buffer_length_data_expression(buffer_expr)
        if length_data is None:
            return None
        return f"{length_data}[0]"

    def structured_buffer_counter_name(self, name):
        """Return the sidecar counter parameter/declaration name for a buffer."""
        return f"{name}_counter"

    def structured_buffer_counter_declaration(self, name, type_name):
        """Format a global/local sidecar counter declaration when required."""
        if not self.structured_buffer_requires_counter(type_name):
            return None
        return self.format_structured_buffer_counter_declarator(type_name, name)

    def structured_buffer_counter_parameter(self, name, type_name):
        """Format a sidecar counter parameter when required."""
        if not self.structured_buffer_requires_counter(type_name):
            return None
        return self.format_structured_buffer_counter_declarator(type_name, name)

    def format_structured_buffer_counter_declarator(self, type_name, name):
        """Format the CUDA uint pointer sidecar matching a buffer declarator."""
        type_name = self.type_name_string(type_name)
        counter_name = self.structured_buffer_counter_name(name)
        if "[" not in type_name or "]" not in type_name:
            return f"uint* {counter_name}"

        array_suffix = type_name[type_name.find("[") :]
        return format_array_declarator("uint*", counter_name, array_suffix)

    def structured_buffer_counter_expression(self, buffer_expr):
        """Return the sidecar counter expression paired with a buffer expression."""
        if isinstance(buffer_expr, ArrayAccessNode):
            array_expr = getattr(
                buffer_expr, "array_expr", getattr(buffer_expr, "array", None)
            )
            index_expr = getattr(
                buffer_expr, "index_expr", getattr(buffer_expr, "index", None)
            )
            base_counter = self.structured_buffer_counter_expression(array_expr)
            if base_counter is None:
                return None
            return f"{base_counter}[{self.visit(index_expr)}]"

        name = self.get_expression_name(buffer_expr)
        if not name:
            return None
        return self.structured_buffer_counter_name(name)

    def cuda_user_function_call_arguments(self, func_name, raw_args, args):
        """Expand user calls with CUDA sidecar resource arguments."""
        callee = self.query_functions_by_name.get(func_name)
        if callee is None:
            return args

        params = getattr(callee, "parameters", getattr(callee, "params", []))
        query_params = self.query_metadata_function_params.get(func_name, set())
        expanded_args = []
        for index, arg in enumerate(args):
            expanded_args.append(arg)
            if index >= len(params) or index >= len(raw_args):
                continue

            param = params[index]
            param_name = getattr(param, "name", None)
            if param_name in query_params:
                metadata_arg = self.query_metadata_expression(raw_args[index])
                if metadata_arg is None:
                    metadata_arg = self.unavailable_query_metadata_argument(
                        param_name,
                        self.get_parameter_type(param),
                    )
                expanded_args.append(metadata_arg)

            param_type = self.get_parameter_type(param)
            if self.structured_buffer_parameter_requires_length(
                func_name, param_name, param_type
            ):
                length_arg = self.structured_buffer_length_data_expression(
                    raw_args[index]
                )
                if length_arg:
                    expanded_args.append(length_arg)

            if self.structured_buffer_requires_counter(param_type):
                counter_arg = self.structured_buffer_counter_expression(raw_args[index])
                if counter_arg:
                    expanded_args.append(counter_arg)
        return expanded_args

    def unavailable_query_metadata_argument(self, param_name, param_type):
        """Return a deterministic zero metadata sidecar for untraceable resources."""
        self.resource_query_info_required = True
        type_name = self.query_type_name(param_type)
        resource_type = self.resource_base_type(type_name) or "unknown resource"
        fallback = "nullptr" if "[" in type_name else "CglResourceQueryInfo{}"
        param_label = f" argument {param_name}" if param_name else " argument"
        return (
            f"/* unsupported {self.resource_backend_name()} resource query: "
            f"metadata unavailable for {resource_type}{param_label} */ {fallback}"
        )

    def diagnostic_zero_value_for_type(self, type_name):
        """Return a CUDA fallback expression for unsupported value-producing calls."""
        if type_name is None:
            return "0"
        mapped_type = self.convert_crossgl_type_to_cuda(type_name)
        vector_info = self.vector_type_info(mapped_type) or self.vector_type_info(
            self.type_name_string(type_name)
        )
        if vector_info is not None:
            component_type = vector_info["component_type"]
            if component_type == "bool":
                zero = "false"
            elif component_type == "uint":
                zero = "0u"
            elif component_type == "double":
                zero = "0.0"
            elif component_type == "float":
                zero = "0.0f"
            else:
                zero = "0"
            return f"{vector_info['constructor']}({', '.join([zero] * len(vector_info['components']))})"

        if mapped_type == "bool":
            return "false"
        if mapped_type in {"float", "half"}:
            return "0.0f"
        if mapped_type == "double":
            return "0.0"
        if mapped_type in {"uint", "unsigned int", "u32"}:
            return "0u"
        if mapped_type in {
            "int",
            "short",
            "char",
            "long long",
            "unsigned char",
            "unsigned short",
            "unsigned long long",
        }:
            return "0"
        return f"{mapped_type}{{}}"

    def byte_address_buffer_base_type(self, type_name):
        """Return the byte-address buffer base type, if applicable."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None
        base_type = type_name.split("[", 1)[0].strip()
        if base_type in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            return base_type
        return None

    def byte_address_buffer_is_writable(self, type_name):
        """Return whether a byte-address buffer type permits writes."""
        return self.byte_address_buffer_base_type(type_name) == "RWByteAddressBuffer"

    def cuda_byte_address_buffer_type(self, type_name):
        """Map ByteAddressBuffer and RWByteAddressBuffer to CUDA byte pointers."""
        base_type = self.byte_address_buffer_base_type(type_name)
        if base_type == "ByteAddressBuffer":
            return "const unsigned char*"
        if base_type == "RWByteAddressBuffer":
            return "unsigned char*"
        return None

    def array_access_element_type(self, type_name):
        """Return the element type for CUDA arrays and structured buffers."""
        array_element_type = super().array_access_element_type(type_name)
        if array_element_type is not None:
            return array_element_type

        indirect_info = self.cuda_indirect_type_info(type_name)
        if indirect_info is not None and indirect_info["kind"] == "pointer":
            return indirect_info["pointee_type"]

        parts = self.structured_buffer_type_parts(type_name)
        if parts is not None:
            return parts[1]
        return None

    def expression_result_type(self, node):
        """Infer expression result types with CUDA structured-buffer operations."""
        if isinstance(node, PointerAccessNode):
            member_type = self.pointer_access_member_type(node)
            if member_type is not None:
                return member_type
        if isinstance(node, FunctionCallNode):
            buffer_result_type = self.buffer_call_result_type(node)
            if buffer_result_type is not None:
                return buffer_result_type
        return super().expression_result_type(node)

    def pointer_access_member_type(self, node):
        """Return the member type for ``ptr->field`` expressions."""
        pointer_expr = getattr(node, "pointer_expr", None)
        pointer_type = self.expression_result_type(pointer_expr)
        pointer_info = self.cuda_indirect_type_info(pointer_type)
        if pointer_info is None:
            return None

        struct_type = pointer_info["pointee_type"]
        if struct_type.startswith("const "):
            struct_type = struct_type[len("const ") :].strip()
        return self.struct_member_types.get(struct_type, {}).get(
            getattr(node, "member", "")
        )

    def resource_call_result_type(self, func_name, raw_args):
        """Infer CUDA-specific resource diagnostic result types."""
        result_type = super().resource_call_result_type(func_name, raw_args)
        if result_type is not None:
            return result_type

        if func_name in self.cuda_shadow_scalar_diagnostic_calls() and raw_args:
            resource_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if self.is_shadow_resource_type(resource_type):
                return "float"

        if func_name in self.cuda_sampled_diagnostic_float4_calls() and raw_args:
            resource_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if resource_type is None:
                return None
            if self.is_shadow_resource_type(resource_type):
                return "float"
            return "float4"

        if func_name in self.cuda_image_atomic_calls() and raw_args:
            resource_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            return self.cuda_image_atomic_result_type(resource_type)

        return None

    def cuda_shadow_scalar_diagnostic_calls(self):
        """Return shadow compare calls that CUDA lowers to scalar diagnostics."""
        return {
            "textureCompareProj",
            "textureCompareProjOffset",
            "textureCompareProjLod",
            "textureCompareProjLodOffset",
            "textureCompareProjGrad",
            "textureCompareProjGradOffset",
        }

    def cuda_sampled_diagnostic_float4_calls(self):
        """Return sampled-resource calls that CUDA lowers to float4 diagnostics."""
        return {
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
            "texelFetchOffset",
        }

    def cuda_image_atomic_calls(self):
        """Return image atomic calls that CUDA lowers to scalar diagnostics."""
        return {
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        }

    def cuda_image_atomic_result_type(self, resource_type):
        """Return the scalar result type for CUDA image atomic diagnostics."""
        base_type = self.resource_base_type(resource_type)
        if not isinstance(base_type, str):
            return None
        if base_type.startswith("uimage"):
            return "uint"
        return "int"

    def buffer_call_result_type(self, node):
        """Infer result type for structured and byte-address buffer read calls."""
        function_expr = getattr(node, "function", getattr(node, "name", None))
        raw_args = getattr(node, "arguments", getattr(node, "args", []))

        byte_member_call = self.byte_address_buffer_member_call(function_expr)
        if byte_member_call is not None:
            _, operation, _ = byte_member_call
            if operation == "GetDimensions":
                return "uint"
            if operation in self.byte_address_buffer_atomic_operations():
                return "uint"
            if operation.startswith("Load") and raw_args:
                component_count = self.byte_address_buffer_component_count(operation)
                if component_count is not None:
                    return self.byte_address_buffer_value_type(component_count)
            return None

        member_call = self.structured_buffer_member_call(function_expr)
        if member_call is not None:
            _, operation, buffer_type = member_call
            if operation == "Load" and raw_args:
                return self.structured_buffer_type_parts(buffer_type)[1]
            if operation == "Consume":
                return self.structured_buffer_type_parts(buffer_type)[1]
            if operation == "GetDimensions" and not raw_args:
                return "uint"
            return None

        func_name = getattr(function_expr, "name", function_expr)
        if func_name == "buffer_load" and raw_args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is not None:
                return "uint"
            parts = self.structured_buffer_type_parts(buffer_type)
            if parts is not None:
                return parts[1]
        if func_name == "buffer_consume" and raw_args:
            buffer_type = self.expression_result_type(raw_args[0])
            parts = self.structured_buffer_type_parts(buffer_type)
            if parts is not None:
                return parts[1]
        if func_name == "buffer_dimensions" and raw_args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.buffer_type_supports_length(buffer_type):
                return "uint"
        if func_name in self.structured_buffer_atomic_operations() and raw_args:
            target = self.structured_buffer_atomic_target(raw_args[0])
            if target is not None:
                return target["target_type"]
        return None

    def canonical_sampled_resource_type(self, type_name):
        """Return the sampler spelling for HLSL-style sampled resources."""
        if not isinstance(type_name, str):
            return None
        base_type = type_name.split("[", 1)[0].split("<", 1)[0].strip()
        return self.sampled_resource_type_aliases.get(base_type)

    def canonical_storage_resource_type(self, type_name):
        """Return the image spelling for HLSL-style writable resources."""
        if not isinstance(type_name, str):
            return None
        base_type = type_name.split("[", 1)[0].strip()
        base_name = base_type.split("<", 1)[0].strip()
        image_type = self.storage_resource_type_aliases.get(base_name)
        if image_type is None:
            return None

        if "<" not in base_type or ">" not in base_type:
            return image_type

        value_type = base_type.split("<", 1)[1].rsplit(">", 1)[0].strip()
        value_type = value_type.split(",", 1)[0].strip().lower()
        if value_type in {"int", "i32"}:
            return f"i{image_type}"
        if value_type in {"uint", "u32"}:
            return f"u{image_type}"
        return image_type

    def resource_base_type(self, type_name):
        """Normalize resource aliases before resource dispatch decisions."""
        base_type = ResourceDiagnosticMixin.resource_base_type(self, type_name)
        return (
            self.canonical_sampled_resource_type(base_type)
            or self.canonical_storage_resource_type(base_type)
            or base_type
        )

    def dimension_query_spec(self, type_name):
        """Return CUDA resource query metadata for supported image shapes."""
        type_name = self.resource_base_type(type_name)
        spec = ResourceQueryMixin.dimension_query_spec(self, type_name)
        if spec is not None:
            return spec

        specs = {
            "image1D": (("width",), False, False),
            "iimage1D": (("width",), False, False),
            "uimage1D": (("width",), False, False),
            "image1DArray": (("width", "elements"), False, False),
            "iimage1DArray": (("width", "elements"), False, False),
            "uimage1DArray": (("width", "elements"), False, False),
            "iimageCube": (("width", "height"), False, False),
            "uimageCube": (("width", "height"), False, False),
            "imageCubeArray": (("width", "height", "elements"), False, False),
            "iimageCubeArray": (("width", "height", "elements"), False, False),
            "uimageCubeArray": (("width", "height", "elements"), False, False),
        }
        spec = specs.get(type_name)
        if spec is None:
            return None
        dimensions, mip, samples = spec
        return {"dimensions": dimensions, "mip": mip, "samples": samples}

    def is_queryable_resource_type(self, type_name):
        """Return whether a resource type can use CUDA query metadata."""
        if type_name is None:
            return False
        resource_type = self.resource_base_type(self.query_type_name(type_name))
        return self.dimension_query_spec(resource_type) is not None

    def collect_function_resource_aliases(self, func):
        """Collect local queryable resource declarations and initializers."""
        local_names = set()
        alias_sources = {}
        body = getattr(func, "body", [])

        for node in self.query_walk_nodes(body):
            if not isinstance(node, VariableNode):
                continue
            name = getattr(node, "name", None)
            var_type = self.get_variable_node_type(node)
            if not name or not self.is_queryable_resource_type(var_type):
                continue
            local_names.add(name)
            initial_value = getattr(node, "initial_value", getattr(node, "value", None))
            if initial_value is not None:
                alias_sources[name] = initial_value

        return local_names, alias_sources

    def collect_function_resource_assignments(self, func, local_resource_names):
        """Collect simple assignments into local queryable resource variables."""
        assignment_sources = {}
        body = getattr(func, "body", [])

        for node in self.query_walk_nodes(body):
            if not isinstance(node, AssignmentNode):
                continue
            if getattr(node, "operator", "=") != "=":
                continue
            target = getattr(node, "target", None)
            if not isinstance(target, (IdentifierNode, VariableNode, str)):
                continue
            target_name = self.get_expression_name(target)
            if target_name not in local_resource_names:
                continue
            value = getattr(node, "value", None)
            if value is not None:
                assignment_sources.setdefault(target_name, []).append(value)

        return assignment_sources

    def collect_simple_query_return_sources(self, root):
        """Collect direct resource returns that can reuse caller-side metadata."""
        return_sources = {}
        functions = self.query_collect_functions(root)
        global_resource_names = {
            getattr(var, "name", None)
            for var in getattr(root, "global_variables", [])
            if self.is_queryable_resource_type(self.get_variable_node_type(var))
        }
        global_resource_names = {name for name in global_resource_names if name}
        global_variable_types = {
            getattr(var, "name", None): self.type_name_string(
                self.get_variable_node_type(var)
            )
            for var in getattr(root, "global_variables", [])
        }
        global_variable_types = {
            name: type_name
            for name, type_name in global_variable_types.items()
            if name and type_name
        }

        function_infos = []
        for func in functions:
            func_name = getattr(func, "name", None)
            return_type = getattr(func, "return_type", None)
            if not func_name or not self.is_queryable_resource_type(return_type):
                continue

            statements = self.statement_list(getattr(func, "body", []))
            if not statements or not isinstance(statements[-1], ReturnNode):
                continue

            local_alias_sources = {}
            supported_return_body = True
            for statement in statements[:-1]:
                if not isinstance(statement, VariableNode):
                    supported_return_body = False
                    break
                var_type = self.get_variable_node_type(statement)
                if not self.is_queryable_resource_type(var_type):
                    supported_return_body = False
                    break
                name = getattr(statement, "name", None)
                initial_value = getattr(
                    statement,
                    "initial_value",
                    getattr(statement, "value", None),
                )
                if not name or initial_value is None:
                    supported_return_body = False
                    break
                local_alias_sources[name] = initial_value
            if not supported_return_body:
                continue

            return_expr = getattr(statements[-1], "value", None)
            params = getattr(func, "parameters", getattr(func, "params", []))
            resource_param_indices = {
                getattr(param, "name", None): index
                for index, param in enumerate(params)
                if self.is_queryable_resource_type(self.get_parameter_type(param))
            }
            resource_param_indices = {
                name: index for name, index in resource_param_indices.items() if name
            }
            all_param_indices = {
                getattr(param, "name", None): index
                for index, param in enumerate(params)
            }
            all_param_indices = {
                name: index for name, index in all_param_indices.items() if name
            }
            param_types = {
                getattr(param, "name", None): self.type_name_string(
                    self.get_parameter_type(param)
                )
                for param in params
            }
            param_types = {
                name: type_name
                for name, type_name in param_types.items()
                if name and type_name
            }
            function_infos.append(
                (
                    func_name,
                    return_expr,
                    resource_param_indices,
                    all_param_indices,
                    param_types,
                    local_alias_sources,
                )
            )

        changed = True
        while changed:
            changed = False
            for (
                func_name,
                return_expr,
                resource_param_indices,
                all_param_indices,
                param_types,
                local_alias_sources,
            ) in function_infos:
                if func_name in return_sources:
                    continue
                return_source = self.query_return_source_descriptor(
                    return_expr,
                    resource_param_indices,
                    global_resource_names,
                    all_param_indices,
                    return_sources,
                    local_alias_sources,
                    param_types=param_types,
                    global_variable_types=global_variable_types,
                )
                if return_source is not None:
                    return_sources[func_name] = return_source
                    changed = True

        return return_sources

    def query_return_source_descriptor(
        self,
        return_expr,
        resource_param_indices,
        global_resource_names,
        all_param_indices,
        known_return_sources=None,
        local_alias_sources=None,
        local_visited=None,
        param_types=None,
        global_variable_types=None,
    ):
        """Describe a traceable resource-return expression for caller metadata."""
        if known_return_sources is None:
            known_return_sources = {}
        if local_alias_sources is None:
            local_alias_sources = {}
        if local_visited is None:
            local_visited = set()
        if param_types is None:
            param_types = {}
        if global_variable_types is None:
            global_variable_types = {}

        if isinstance(return_expr, TernaryOpNode):
            true_source = self.query_return_source_descriptor(
                return_expr.true_expr,
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                local_visited.copy(),
                param_types=param_types,
                global_variable_types=global_variable_types,
            )
            false_source = self.query_return_source_descriptor(
                return_expr.false_expr,
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                local_visited.copy(),
                param_types=param_types,
                global_variable_types=global_variable_types,
            )
            if true_source is None or false_source is None:
                return None
            return {
                "kind": "ternary",
                "condition": return_expr.condition,
                "true_source": true_source,
                "false_source": false_source,
                "param_indices": all_param_indices,
            }

        if isinstance(return_expr, FunctionCallNode):
            callee_name = self.raw_function_call_name(return_expr)
            return_source = known_return_sources.get(callee_name)
            if return_source is None:
                return None
            raw_args = getattr(
                return_expr,
                "arguments",
                getattr(return_expr, "args", []),
            )
            return self.inline_query_return_source_descriptor(
                return_source,
                raw_args,
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                param_types,
                global_variable_types,
            )

        return_base_expr = return_expr
        return_indices = []
        if isinstance(return_expr, ArrayAccessNode):
            return_base_expr, return_indices = self.query_array_access_parts(
                return_expr
            )
        elif isinstance(return_expr, MemberAccessNode):
            return_base_expr = return_expr
        elif not isinstance(return_expr, (IdentifierNode, VariableNode, str)):
            return None

        if isinstance(return_base_expr, MemberAccessNode):
            return self.query_return_member_source_descriptor(
                return_base_expr,
                return_indices,
                all_param_indices,
                param_types,
                global_variable_types,
            )

        return_name = self.get_expression_name(return_base_expr)
        if not return_name:
            return None
        if any(
            not self.is_safe_query_return_index(index, all_param_indices)
            for index in return_indices
        ):
            return None
        if return_name in local_alias_sources:
            if return_name in local_visited:
                return None
            source_descriptor = self.query_return_source_descriptor(
                local_alias_sources[return_name],
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                local_visited | {return_name},
                param_types=param_types,
                global_variable_types=global_variable_types,
            )
            if source_descriptor is None:
                return None
            return self.append_query_return_indices(
                source_descriptor,
                return_indices,
                all_param_indices,
            )
        if return_name in resource_param_indices:
            return {
                "kind": "parameter",
                "index": resource_param_indices[return_name],
                "indices": return_indices,
                "param_indices": all_param_indices,
            }
        if return_name in global_resource_names:
            return {
                "kind": "global",
                "name": return_name,
                "indices": return_indices,
                "param_indices": all_param_indices,
            }
        return None

    def query_return_member_source_descriptor(
        self,
        member_expr,
        return_indices,
        all_param_indices,
        param_types,
        global_variable_types,
    ):
        """Describe a storage-image struct member returned by a helper."""
        object_node = getattr(
            member_expr,
            "object_expr",
            getattr(member_expr, "object", None),
        )
        member = getattr(member_expr, "member", None)
        member_name = getattr(member, "name", member)
        if object_node is None or not isinstance(member_name, str):
            return None

        object_name = self.get_expression_name(object_node)
        if not object_name:
            return None

        object_kind = None
        object_type = None
        descriptor = {
            "kind": "member",
            "member": member_name,
            "indices": return_indices,
            "param_indices": all_param_indices,
        }
        if object_name in param_types:
            object_kind = "parameter"
            object_type = param_types[object_name]
            descriptor["object_index"] = all_param_indices.get(object_name)
        elif object_name in global_variable_types:
            object_kind = "global"
            object_type = global_variable_types[object_name]
            descriptor["object_name"] = object_name
        else:
            return None

        object_type = self.type_name_string(object_type)
        member_type = self.struct_member_types.get(object_type, {}).get(member_name)
        if not self.is_storage_image_type(member_type):
            return None
        if any(
            not self.is_safe_query_return_index(index, all_param_indices)
            for index in return_indices
        ):
            return None

        descriptor["object_kind"] = object_kind
        descriptor["object_type"] = object_type
        return descriptor

    def inline_query_return_source_descriptor(
        self,
        return_source,
        raw_args,
        resource_param_indices,
        global_resource_names,
        all_param_indices,
        known_return_sources,
        local_alias_sources=None,
        param_types=None,
        global_variable_types=None,
    ):
        """Inline a traceable callee resource return into the caller context."""
        if local_alias_sources is None:
            local_alias_sources = {}
        if param_types is None:
            param_types = {}
        if global_variable_types is None:
            global_variable_types = {}
        source_param_indices = return_source.get("param_indices", {})
        kind = return_source["kind"]

        if kind == "global":
            indices = self.substitute_query_return_indices(
                return_source.get("indices", []),
                source_param_indices,
                raw_args,
                all_param_indices,
            )
            if indices is None:
                return None
            return {
                "kind": "global",
                "name": return_source["name"],
                "indices": indices,
                "param_indices": all_param_indices,
            }

        if kind == "parameter":
            index = return_source["index"]
            if index >= len(raw_args):
                return None
            base_source = self.query_return_source_descriptor(
                raw_args[index],
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                param_types=param_types,
                global_variable_types=global_variable_types,
            )
            if base_source is None:
                return None
            indices = self.substitute_query_return_indices(
                return_source.get("indices", []),
                source_param_indices,
                raw_args,
                all_param_indices,
            )
            if indices is None:
                return None
            return self.append_query_return_indices(
                base_source,
                indices,
                all_param_indices,
            )

        if kind == "ternary":
            condition = self.substitute_query_return_expr(
                return_source["condition"],
                source_param_indices,
                raw_args,
            )
            if condition is None:
                return None
            true_source = self.inline_query_return_source_descriptor(
                return_source["true_source"],
                raw_args,
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                param_types,
                global_variable_types,
            )
            false_source = self.inline_query_return_source_descriptor(
                return_source["false_source"],
                raw_args,
                resource_param_indices,
                global_resource_names,
                all_param_indices,
                known_return_sources,
                local_alias_sources,
                param_types,
                global_variable_types,
            )
            if true_source is None or false_source is None:
                return None
            return {
                "kind": "ternary",
                "condition": condition,
                "true_source": true_source,
                "false_source": false_source,
                "param_indices": all_param_indices,
            }

        if kind == "member":
            indices = self.substitute_query_return_indices(
                return_source.get("indices", []),
                source_param_indices,
                raw_args,
                all_param_indices,
            )
            if indices is None:
                return None
            inlined = dict(return_source)
            inlined["indices"] = indices
            inlined["param_indices"] = all_param_indices
            return inlined

        return None

    def append_query_return_indices(self, return_source, indices, param_indices):
        """Append array indices to a flattened resource-return descriptor."""
        if not indices:
            return return_source
        if return_source["kind"] not in {"global", "parameter", "member"}:
            return None
        if any(
            not self.is_safe_query_return_index(index, param_indices)
            for index in indices
        ):
            return None
        combined = dict(return_source)
        combined["indices"] = list(return_source.get("indices", [])) + list(indices)
        combined["param_indices"] = param_indices
        return combined

    def substitute_query_return_indices(
        self,
        indices,
        param_indices,
        raw_args,
        target_param_indices,
    ):
        """Substitute callee index parameters with caller-context expressions."""
        substituted_indices = []
        for index_expr in indices:
            substituted = self.substitute_query_return_expr(
                index_expr,
                param_indices,
                raw_args,
            )
            if substituted is None or not self.is_safe_query_return_index(
                substituted,
                target_param_indices,
            ):
                return None
            substituted_indices.append(substituted)
        return substituted_indices

    def substitute_query_return_expr(self, expr, param_indices, raw_args):
        """Replace callee parameters in a return-source expression."""
        expr_name = self.get_expression_name(expr)
        if isinstance(expr, (IdentifierNode, VariableNode, str)):
            if expr_name in param_indices:
                param_index = param_indices[expr_name]
                if param_index >= len(raw_args):
                    return None
                return raw_args[param_index]
            return expr

        if isinstance(expr, BinaryOpNode):
            left = self.substitute_query_return_expr(
                expr.left,
                param_indices,
                raw_args,
            )
            right = self.substitute_query_return_expr(
                expr.right,
                param_indices,
                raw_args,
            )
            if left is None or right is None:
                return None
            return BinaryOpNode(left, expr.operator, right)

        if isinstance(expr, UnaryOpNode):
            operand = self.substitute_query_return_expr(
                expr.operand,
                param_indices,
                raw_args,
            )
            if operand is None:
                return None
            return UnaryOpNode(expr.operator, operand, expr.is_postfix)

        if isinstance(expr, TernaryOpNode):
            condition = self.substitute_query_return_expr(
                expr.condition,
                param_indices,
                raw_args,
            )
            true_expr = self.substitute_query_return_expr(
                expr.true_expr,
                param_indices,
                raw_args,
            )
            false_expr = self.substitute_query_return_expr(
                expr.false_expr,
                param_indices,
                raw_args,
            )
            if condition is None or true_expr is None or false_expr is None:
                return None
            return TernaryOpNode(condition, true_expr, false_expr)

        if isinstance(expr, ArrayAccessNode):
            array_expr = self.substitute_query_return_expr(
                expr.array_expr,
                param_indices,
                raw_args,
            )
            index_expr = self.substitute_query_return_expr(
                expr.index_expr,
                param_indices,
                raw_args,
            )
            if array_expr is None or index_expr is None:
                return None
            return ArrayAccessNode(array_expr, index_expr)

        if isinstance(expr, MemberAccessNode):
            object_expr = self.substitute_query_return_expr(
                expr.object_expr,
                param_indices,
                raw_args,
            )
            if object_expr is None:
                return None
            return MemberAccessNode(object_expr, expr.member)

        if isinstance(expr, FunctionCallNode):
            arguments = []
            for argument in getattr(expr, "arguments", getattr(expr, "args", [])):
                substituted = self.substitute_query_return_expr(
                    argument,
                    param_indices,
                    raw_args,
                )
                if substituted is None:
                    return None
                arguments.append(substituted)
            return FunctionCallNode(
                expr.function,
                arguments,
                getattr(expr, "generic_args", []),
            )

        return expr

    def query_array_access_parts(self, expr):
        """Return the base expression and ordered indices for nested array access."""
        indices = []
        current = expr
        while isinstance(current, ArrayAccessNode):
            index_node = getattr(
                current,
                "index_expr",
                getattr(current, "index", None),
            )
            array_node = getattr(
                current,
                "array_expr",
                getattr(current, "array", None),
            )
            if index_node is None or array_node is None:
                return None, []
            indices.insert(0, index_node)
            current = array_node
        return current, indices

    def is_safe_query_return_index(self, index_expr, param_indices):
        """Return whether a returned-array index can be rendered in the caller."""
        if isinstance(index_expr, LiteralNode) or isinstance(index_expr, int):
            return True
        if isinstance(index_expr, BinaryOpNode):
            operator = getattr(index_expr, "operator", getattr(index_expr, "op", None))
            return (
                operator in self.query_return_index_binary_ops
                and self.is_safe_query_return_index(index_expr.left, param_indices)
                and self.is_safe_query_return_index(index_expr.right, param_indices)
            )
        index_name = self.get_expression_name(index_expr)
        if index_name:
            return index_name in param_indices
        if isinstance(index_expr, str):
            return index_expr.lstrip("-").isdigit()
        return False

    def collect_resource_query_requirements(self, node):
        """Collect query metadata needs, resolving local CUDA resource aliases."""
        global_names, function_params = (
            ResourceQueryMixin.collect_resource_query_requirements(self, node)
        )

        functions = self.query_collect_functions(node)
        functions_by_name = {getattr(func, "name", None): func for func in functions}
        functions_by_name = {
            name: func for name, func in functions_by_name.items() if name
        }
        function_parameter_names = {
            func_name: {
                getattr(param, "name", None)
                for param in getattr(func, "parameters", getattr(func, "params", []))
            }
            for func_name, func in functions_by_name.items()
        }
        function_parameter_names = {
            func_name: {name for name in names if name}
            for func_name, names in function_parameter_names.items()
        }

        local_resource_names = {}
        local_alias_sources = {}
        local_assignment_sources = {}
        for func_name, func in functions_by_name.items():
            names, aliases = self.collect_function_resource_aliases(func)
            local_resource_names[func_name] = names
            local_alias_sources[func_name] = aliases
            local_assignment_sources[func_name] = (
                self.collect_function_resource_assignments(func, names)
            )
        local_metadata_snapshots = {func_name: set() for func_name in functions_by_name}

        global_resource_names = {
            getattr(var, "name", None)
            for var in getattr(node, "global_variables", [])
            if self.is_queryable_resource_type(self.get_variable_node_type(var))
        }
        global_resource_names = {name for name in global_resource_names if name}

        def mark_resolved_resource(func_name, resource_expr, visited=None):
            if visited is None:
                visited = set()
            if isinstance(resource_expr, TernaryOpNode):
                if not self.is_safe_query_return_actual_index(resource_expr.condition):
                    return False
                true_changed = mark_resolved_resource(
                    func_name,
                    resource_expr.true_expr,
                    visited.copy(),
                )
                false_changed = mark_resolved_resource(
                    func_name,
                    resource_expr.false_expr,
                    visited.copy(),
                )
                return true_changed or false_changed

            if isinstance(resource_expr, FunctionCallNode):
                callee_name = self.raw_function_call_name(resource_expr)
                return_source = self.query_return_sources.get(callee_name)
                if return_source is None:
                    return False
                raw_args = getattr(
                    resource_expr,
                    "arguments",
                    getattr(resource_expr, "args", []),
                )
                return mark_return_source_resources(
                    func_name,
                    return_source,
                    raw_args,
                    visited,
                )

            resource_name = self.get_expression_name(resource_expr)
            if not resource_name:
                return False
            if resource_name in visited:
                return False
            visited.add(resource_name)

            aliases = local_alias_sources.get(func_name, {})
            assignments = local_assignment_sources.get(func_name, {})
            if resource_name in local_resource_names.get(func_name, set()):
                if resource_name in assignments:
                    snapshot_names = local_metadata_snapshots.setdefault(
                        func_name,
                        set(),
                    )
                    before = len(snapshot_names)
                    snapshot_names.add(resource_name)
                    changed = len(snapshot_names) != before
                    sources = []
                    if resource_name in aliases:
                        sources.append(aliases[resource_name])
                    sources.extend(assignments.get(resource_name, []))
                    for source in sources:
                        changed = (
                            mark_resolved_resource(
                                func_name,
                                source,
                                visited.copy(),
                            )
                            or changed
                        )
                    return changed
                if resource_name in aliases:
                    return mark_resolved_resource(
                        func_name,
                        aliases[resource_name],
                        visited,
                    )
                return False

            if resource_name in function_parameter_names.get(func_name, set()):
                before = len(function_params.setdefault(func_name, set()))
                function_params[func_name].add(resource_name)
                return len(function_params[func_name]) != before

            before = len(global_names)
            global_names.add(resource_name)
            return len(global_names) != before

        def mark_return_source_resources(func_name, return_source, raw_args, visited):
            kind = return_source["kind"]
            if kind == "global":
                before = len(global_names)
                global_names.add(return_source["name"])
                return len(global_names) != before
            if kind == "parameter":
                index = return_source["index"]
                if index >= len(raw_args):
                    return False
                return mark_resolved_resource(func_name, raw_args[index], visited)
            if kind == "ternary":
                true_changed = mark_return_source_resources(
                    func_name,
                    return_source["true_source"],
                    raw_args,
                    visited.copy(),
                )
                false_changed = mark_return_source_resources(
                    func_name,
                    return_source["false_source"],
                    raw_args,
                    visited.copy(),
                )
                return true_changed or false_changed
            return False

        for func_name, func in functions_by_name.items():
            for call in self.query_walk_nodes(getattr(func, "body", [])):
                if not isinstance(call, FunctionCallNode):
                    continue
                func_call_name = self.raw_function_call_name(call)
                raw_args = getattr(call, "arguments", getattr(call, "args", []))
                if func_call_name in self.query_function_names and raw_args:
                    mark_resolved_resource(func_name, raw_args[0])

        changed = True
        while changed:
            changed = False
            for caller_name, caller in functions_by_name.items():
                for call in self.query_walk_nodes(getattr(caller, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.raw_function_call_name(call)
                    callee = functions_by_name.get(callee_name)
                    if callee is None:
                        continue

                    callee_required = function_params.get(callee_name, set())
                    if not callee_required:
                        continue

                    callee_params = getattr(
                        callee, "parameters", getattr(callee, "params", [])
                    )
                    raw_args = getattr(call, "arguments", getattr(call, "args", []))
                    for index, param in enumerate(callee_params):
                        if index >= len(raw_args):
                            continue
                        param_name = getattr(param, "name", None)
                        if param_name not in callee_required:
                            continue
                        changed = (
                            mark_resolved_resource(caller_name, raw_args[index])
                            or changed
                        )

        for names in local_resource_names.values():
            global_names.difference_update(names - global_resource_names)

        self.query_local_resource_names_by_function = {
            func_name: names
            for func_name, names in local_resource_names.items()
            if names
        }
        self.query_metadata_snapshot_locals_by_function = {
            func_name: names
            for func_name, names in local_metadata_snapshots.items()
            if names
        }
        return (
            global_names,
            {func_name: names for func_name, names in function_params.items() if names},
        )

    def register_query_metadata_alias(self, name, type_name, source_node=None):
        """Track local resource aliases that can reuse an existing metadata sidecar."""
        if not self.current_function_name or not name:
            return
        if not self.is_queryable_resource_type(type_name) or source_node is None:
            self.query_metadata_aliases.pop(name, None)
            return

        metadata_expr = self.query_metadata_expression(source_node)
        if metadata_expr is None:
            self.query_metadata_aliases.pop(name, None)
            return
        self.query_metadata_aliases[name] = metadata_expr

    def is_query_metadata_snapshot_local(self, name, type_name=None):
        """Return whether a local resource needs its own mutable metadata sidecar."""
        if not self.current_function_name or not name:
            return False
        snapshot_names = self.query_metadata_snapshot_locals_by_function.get(
            self.current_function_name,
            set(),
        )
        if name not in snapshot_names:
            return False
        if type_name is not None and not self.is_queryable_resource_type(type_name):
            return False
        return True

    def query_metadata_snapshot_declaration(self, name, type_name, initial_value=None):
        """Return a local metadata sidecar declaration for reassigned resources."""
        if not self.is_query_metadata_snapshot_local(name, type_name):
            return None

        declarator = self.format_typed_declarator(
            self.query_metadata_type(type_name),
            self.query_metadata_name(name),
            dynamic_array_as_pointer=False,
        )
        metadata_expr = None
        if initial_value is not None:
            metadata_expr = self.query_metadata_expression(initial_value)
        if metadata_expr is None:
            metadata_expr = "{}"
        return f"{declarator} = {metadata_expr}"

    def format_query_metadata_assignment(self, node):
        """Return a metadata sidecar update for a reassigned local resource."""
        if getattr(node, "operator", "=") != "=":
            return None

        target_node = getattr(node, "target", None)
        if not isinstance(target_node, (IdentifierNode, VariableNode, str)):
            return None
        target_name = self.get_expression_name(target_node)
        target_type = self.variable_types.get(target_name)
        if not self.is_query_metadata_snapshot_local(target_name, target_type):
            return None

        metadata_expr = self.query_metadata_expression(getattr(node, "value", None))
        if metadata_expr is None:
            resource_type = self.resource_base_type(target_type) or target_type
            resource_type = resource_type or "resource"
            metadata_expr = (
                "/* unsupported CUDA resource query: metadata unavailable for "
                f"{resource_type} assignment */ CglResourceQueryInfo{{}}"
            )
        return f"{self.query_metadata_name(target_name)} = {metadata_expr}"

    def query_metadata_expression(self, resource_expr):
        """Return CUDA query metadata paired with a resource expression."""
        if isinstance(resource_expr, TernaryOpNode):
            if not self.is_safe_query_return_actual_index(resource_expr.condition):
                return None
            true_type = self.resource_base_type(
                self.resource_expression_type(resource_expr.true_expr)
            )
            false_type = self.resource_base_type(
                self.resource_expression_type(resource_expr.false_expr)
            )
            if true_type is None or false_type is None or true_type != false_type:
                return None
            true_metadata = self.query_metadata_expression(resource_expr.true_expr)
            false_metadata = self.query_metadata_expression(resource_expr.false_expr)
            if true_metadata is None or false_metadata is None:
                return None
            condition = self.visit(resource_expr.condition)
            return f"({condition} ? {true_metadata} : {false_metadata})"

        if isinstance(resource_expr, FunctionCallNode):
            callee_name = self.raw_function_call_name(resource_expr)
            return_source = self.query_return_sources.get(callee_name)
            if return_source is None:
                return None
            raw_args = getattr(
                resource_expr,
                "arguments",
                getattr(resource_expr, "args", []),
            )
            return self.query_return_source_metadata_expression(
                return_source,
                raw_args,
            )

        if isinstance(resource_expr, ArrayAccessNode):
            array_node = getattr(
                resource_expr,
                "array_expr",
                getattr(resource_expr, "array", None),
            )
            index_node = getattr(
                resource_expr,
                "index_expr",
                getattr(resource_expr, "index", None),
            )
            if index_node is None or not self.is_safe_query_return_actual_index(
                index_node
            ):
                return None
            base_expr = self.query_metadata_expression(array_node)
            if base_expr is None:
                return None
            return f"{base_expr}[{self.visit(index_node)}]"

        resource_name = self.get_expression_name(resource_expr)
        if not resource_name:
            return None

        alias_expr = self.query_metadata_aliases.get(resource_name)
        if alias_expr is not None:
            return alias_expr

        current_function = self.current_function_name
        local_names = self.query_local_resource_names_by_function.get(
            current_function,
            set(),
        )
        if resource_name in local_names:
            return None

        if current_function:
            query_params = self.query_metadata_function_params.get(
                current_function,
                set(),
            )
            if resource_name in query_params:
                return self.query_metadata_name(resource_name)

        if resource_name in self.query_resource_names:
            return self.query_metadata_name(resource_name)
        return None

    def query_return_source_metadata_expression(self, return_source, raw_args):
        """Render query metadata for a resource returned by a user function."""
        if return_source["kind"] == "ternary":
            condition = self.format_query_return_safe_expression(
                return_source["condition"],
                return_source.get("param_indices", {}),
                raw_args,
            )
            if condition is None:
                return None
            true_metadata = self.query_return_source_metadata_expression(
                return_source["true_source"],
                raw_args,
            )
            false_metadata = self.query_return_source_metadata_expression(
                return_source["false_source"],
                raw_args,
            )
            if true_metadata is None or false_metadata is None:
                return None
            return f"({condition} ? {true_metadata} : {false_metadata})"

        if return_source["kind"] == "global":
            metadata_expr = self.query_metadata_name(return_source["name"])
        elif return_source["kind"] == "parameter":
            index = return_source["index"]
            if index >= len(raw_args):
                return None
            metadata_expr = self.query_metadata_expression(raw_args[index])
        else:
            return None
        if metadata_expr is None:
            return None

        param_indices = return_source.get("param_indices", {})
        for index_expr in return_source.get("indices", []):
            rendered_index = self.format_query_return_index(
                index_expr,
                param_indices,
                raw_args,
            )
            if rendered_index is None:
                return None
            metadata_expr = f"{metadata_expr}[{rendered_index}]"
        return metadata_expr

    def format_query_return_safe_expression(
        self,
        expr,
        param_indices,
        raw_args,
        allow_free_identifiers=False,
    ):
        """Render a side-effect-safe returned-resource selector expression."""
        if isinstance(expr, BinaryOpNode):
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            if operator not in (
                self.query_return_index_binary_ops
                | {"&&", "||", "==", "!=", "<", "<=", ">", ">="}
            ):
                return None
            left = self.format_query_return_safe_expression(
                expr.left,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            right = self.format_query_return_safe_expression(
                expr.right,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if left is None or right is None:
                return None
            return f"({left} {operator} {right})"

        if isinstance(expr, UnaryOpNode):
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            if getattr(expr, "is_postfix", False) or operator not in {
                "!",
                "+",
                "-",
                "~",
            }:
                return None
            operand = self.format_query_return_safe_expression(
                expr.operand,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if operand is None:
                return None
            return f"{operator}{operand}"

        if isinstance(expr, TernaryOpNode):
            condition = self.format_query_return_safe_expression(
                expr.condition,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            true_expr = self.format_query_return_safe_expression(
                expr.true_expr,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            false_expr = self.format_query_return_safe_expression(
                expr.false_expr,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if condition is None or true_expr is None or false_expr is None:
                return None
            return f"({condition} ? {true_expr} : {false_expr})"

        if isinstance(expr, ArrayAccessNode):
            array_node = getattr(
                expr,
                "array_expr",
                getattr(expr, "array", None),
            )
            index_node = getattr(
                expr,
                "index_expr",
                getattr(expr, "index", None),
            )
            if array_node is None or index_node is None:
                return None
            array_expr = self.format_query_return_safe_expression(
                array_node,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            index_expr = self.format_query_return_safe_expression(
                index_node,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if array_expr is None or index_expr is None:
                return None
            return f"{array_expr}[{index_expr}]"

        if isinstance(expr, MemberAccessNode):
            object_node = getattr(
                expr,
                "object_expr",
                getattr(expr, "object", None),
            )
            member = getattr(expr, "member", None)
            member_name = getattr(member, "name", member)
            if object_node is None or not isinstance(member_name, str):
                return None
            object_expr = self.format_query_return_safe_expression(
                object_node,
                param_indices,
                raw_args,
                allow_free_identifiers=allow_free_identifiers,
            )
            if object_expr is None:
                return None
            return f"{object_expr}.{member_name}"

        expr_name = self.get_expression_name(expr)
        if expr_name in param_indices:
            param_index = param_indices[expr_name]
            if param_index >= len(raw_args):
                return None
            return self.format_query_return_safe_expression(
                raw_args[param_index],
                {},
                [],
                allow_free_identifiers=True,
            )
        if allow_free_identifiers and expr_name:
            return self.visit(expr)
        if isinstance(expr, LiteralNode):
            return self.visit(expr)
        if isinstance(expr, bool):
            return "true" if expr else "false"
        if isinstance(expr, (int, float)):
            return str(expr)
        if isinstance(expr, str):
            stripped = expr.strip()
            if stripped in {"true", "false"} or stripped.lstrip("-").isdigit():
                return stripped
            if allow_free_identifiers and stripped.isidentifier():
                return stripped
        return None

    def format_query_return_index(self, index_expr, param_indices, raw_args):
        """Render a returned-array index in the caller's argument context."""
        if isinstance(index_expr, BinaryOpNode):
            operator = getattr(index_expr, "operator", getattr(index_expr, "op", None))
            if operator not in self.query_return_index_binary_ops:
                return None
            left = self.format_query_return_index(
                index_expr.left,
                param_indices,
                raw_args,
            )
            right = self.format_query_return_index(
                index_expr.right,
                param_indices,
                raw_args,
            )
            if left is None or right is None:
                return None
            return f"({left} {operator} {right})"
        index_name = self.get_expression_name(index_expr)
        if index_name in param_indices:
            param_index = param_indices[index_name]
            if param_index >= len(raw_args):
                return None
            actual_expr = raw_args[param_index]
            if not self.is_safe_query_return_actual_index(actual_expr):
                return None
            return self.visit(actual_expr)
        if isinstance(index_expr, LiteralNode):
            return self.visit(index_expr)
        if isinstance(index_expr, int):
            return str(index_expr)
        if isinstance(index_expr, str) and index_expr.lstrip("-").isdigit():
            return index_expr
        return None

    def is_safe_query_return_actual_index(self, actual_expr):
        """Return whether a caller argument can be reused as a metadata index."""
        if isinstance(actual_expr, LiteralNode) or isinstance(actual_expr, int):
            return True
        if isinstance(actual_expr, (IdentifierNode, VariableNode)):
            return True
        if isinstance(actual_expr, str):
            actual_expr = actual_expr.strip()
            return actual_expr.lstrip("-").isdigit() or actual_expr.isidentifier()
        if isinstance(actual_expr, BinaryOpNode):
            operator = getattr(
                actual_expr, "operator", getattr(actual_expr, "op", None)
            )
            return (
                operator in self.query_return_index_binary_ops
                and self.is_safe_query_return_actual_index(actual_expr.left)
                and self.is_safe_query_return_actual_index(actual_expr.right)
            )
        if isinstance(actual_expr, UnaryOpNode):
            operator = getattr(
                actual_expr, "operator", getattr(actual_expr, "op", None)
            )
            return (
                not getattr(actual_expr, "is_postfix", False)
                and operator in {"+", "-", "~"}
                and self.is_safe_query_return_actual_index(actual_expr.operand)
            )
        if isinstance(actual_expr, TernaryOpNode):
            return (
                self.is_safe_query_return_actual_index(actual_expr.condition)
                and self.is_safe_query_return_actual_index(actual_expr.true_expr)
                and self.is_safe_query_return_actual_index(actual_expr.false_expr)
            )
        if isinstance(actual_expr, ArrayAccessNode):
            array_node = getattr(
                actual_expr,
                "array_expr",
                getattr(actual_expr, "array", None),
            )
            index_node = getattr(
                actual_expr,
                "index_expr",
                getattr(actual_expr, "index", None),
            )
            return (
                array_node is not None
                and index_node is not None
                and self.is_safe_query_return_actual_index(array_node)
                and self.is_safe_query_return_actual_index(index_node)
            )
        if isinstance(actual_expr, MemberAccessNode):
            object_expr = getattr(
                actual_expr,
                "object_expr",
                getattr(actual_expr, "object", None),
            )
            return object_expr is not None and self.is_safe_query_return_actual_index(
                object_expr
            )
        return False

    def convert_builtin_function(self, func_name):
        """Convert CrossGL built-in functions to CUDA equivalents"""
        function_mapping = {
            # Math functions
            "sqrt": "sqrtf",
            "pow": "powf",
            "sin": "sinf",
            "cos": "cosf",
            "tan": "tanf",
            "asin": "asinf",
            "acos": "acosf",
            "atan": "atanf",
            "atan2": "atan2f",
            "sinh": "sinhf",
            "cosh": "coshf",
            "tanh": "tanhf",
            "log": "logf",
            "log2": "log2f",
            "exp": "expf",
            "exp2": "exp2f",
            "inversesqrt": "rsqrtf",
            "abs": "fabsf",
            "round": "roundf",
            "trunc": "truncf",
            "mod": "fmodf",
            "mix": "lerp",
            "min": "fminf",
            "max": "fmaxf",
            "floor": "floorf",
            "ceil": "ceilf",
            # Vector constructors
            "vec2": "make_float2",
            "vec3": "make_float3",
            "vec4": "make_float4",
            "float2": "make_float2",
            "float3": "make_float3",
            "float4": "make_float4",
            "vec2<f32>": "make_float2",
            "vec3<f32>": "make_float3",
            "vec4<f32>": "make_float4",
            "dvec2": "make_double2",
            "dvec3": "make_double3",
            "dvec4": "make_double4",
            "double2": "make_double2",
            "double3": "make_double3",
            "double4": "make_double4",
            "vec2<f64>": "make_double2",
            "vec3<f64>": "make_double3",
            "vec4<f64>": "make_double4",
            "ivec2": "make_int2",
            "ivec3": "make_int3",
            "ivec4": "make_int4",
            "int2": "make_int2",
            "int3": "make_int3",
            "int4": "make_int4",
            "vec2<i32>": "make_int2",
            "vec3<i32>": "make_int3",
            "vec4<i32>": "make_int4",
            "uvec2": "make_uint2",
            "uvec3": "make_uint3",
            "uvec4": "make_uint4",
            "uint2": "make_uint2",
            "uint3": "make_uint3",
            "uint4": "make_uint4",
            "vec2<u32>": "make_uint2",
            "vec3<u32>": "make_uint3",
            "vec4<u32>": "make_uint4",
            "bvec2": "make_uchar2",
            "bvec3": "make_uchar3",
            "bvec4": "make_uchar4",
            "uchar2": "make_uchar2",
            "uchar3": "make_uchar3",
            "uchar4": "make_uchar4",
            "vec2<bool>": "make_uchar2",
            "vec3<bool>": "make_uchar3",
            "vec4<bool>": "make_uchar4",
            "bool2": "make_uchar2",
            "bool3": "make_uchar3",
            "bool4": "make_uchar4",
            # Matrix constructors
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float2x3",
            "mat2x4": "float2x4",
            "mat3x2": "float3x2",
            "mat3x3": "float3x3",
            "mat3x4": "float3x4",
            "mat4x2": "float4x2",
            "mat4x3": "float4x3",
            "mat4x4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double2x3",
            "dmat2x4": "double2x4",
            "dmat3x2": "double3x2",
            "dmat3x3": "double3x3",
            "dmat3x4": "double3x4",
            "dmat4x2": "double4x2",
            "dmat4x3": "double4x3",
            "dmat4x4": "double4x4",
            # Atomic operations
            "atomicAdd": "atomicAdd",
            "atomicSub": "atomicSub",
            "atomicMax": "atomicMax",
            "atomicMin": "atomicMin",
            "atomicExchange": "atomicExch",
            "atomicCompareExchange": "atomicCAS",
            "atomicCompSwap": "atomicCAS",
            "atomicInc": "atomicInc",
            "atomicDec": "atomicDec",
            # Synchronization
            "barrier": "__syncthreads",
            "groupMemoryBarrier": "__threadfence_block",
            "memoryBarrier": "__threadfence",
            "memoryBarrierShared": "__threadfence_block",
            "memoryBarrierBuffer": "__threadfence",
            "memoryBarrierImage": "__threadfence",
            "workgroupBarrier": "__syncthreads",
            # Texture functions
            "texture": "tex2D",
            "textureLod": "tex2DLod",
            "textureGrad": "tex2DGrad",
        }

        return function_mapping.get(func_name, func_name)

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "name") and value.name is not None:
            return str(value.name)
        if hasattr(value, "value") and value.value is not None:
            return str(value.value).strip('"')
        return str(value)

    def explicit_resource_access(self, node):
        if node is None:
            return None

        access_names = {
            "read": "readonly",
            "readonly": "readonly",
            "write": "writeonly",
            "writeonly": "writeonly",
            "read_write": "readwrite",
            "readwrite": "readwrite",
            "access::read": "readonly",
            "access::write": "writeonly",
            "access::read_write": "readwrite",
        }
        for qualifier in getattr(node, "qualifiers", []) or []:
            access = access_names.get(str(qualifier).lower())
            if access is not None:
                return access

        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name == "access":
                arguments = getattr(attr, "arguments", []) or []
                if not arguments:
                    continue
                raw_access = self.attribute_value_to_string(arguments[0])
                access = access_names.get(str(raw_access).lower())
            else:
                access = access_names.get(attr_name)
            if access is not None:
                return access
        return None

    def declaration_qualifier_names(self, node):
        """Return normalized qualifier and attribute names on declarations."""
        names = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name:
                names.add(attr_name)
        return names

    def apply_readonly_qualifier_to_type(self, type_name, node):
        """Apply readonly declaration qualifiers to pointer/reference types."""
        type_string = self.type_name_string(type_name)
        if not type_string:
            return type_name

        if self.cuda_pointer_or_reference_type_info(type_string) is None:
            return type_string

        if not (
            self.declaration_qualifier_names(node) & {"const", "readonly", "constant"}
        ):
            return type_string

        if type_string.startswith("const "):
            return type_string
        return f"const {type_string}"

    def is_storage_image_type(self, type_name):
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        base_type = self.resource_base_type(type_name)
        if not isinstance(base_type, str):
            return False
        image_shapes = {
            "image1D",
            "image1DArray",
            "image2D",
            "image2DArray",
            "image3D",
            "imageCube",
            "imageCubeArray",
            "image2DMS",
            "image2DMSArray",
        }
        return base_type in image_shapes or any(
            base_type == f"{prefix}{shape}"
            for prefix in ("i", "u")
            for shape in image_shapes
        )

    def register_variable_type(self, name, type_name, node=None, source_node=None):
        if not name or type_name is None:
            return
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        type_name = self.apply_readonly_qualifier_to_type(type_name, node)
        self.variable_types[name] = type_name
        if not self.is_storage_image_type(type_name):
            self.image_resource_accesses.pop(name, None)
            return

        access = self.explicit_resource_access(node)
        if access is None and source_node is not None:
            access = self.image_resource_access(source_node)
        if access is None:
            self.image_resource_accesses.pop(name, None)
        else:
            self.image_resource_accesses[name] = access

    def get_expression_name(self, node):
        if isinstance(node, IdentifierNode):
            return node.name
        if isinstance(node, VariableNode):
            return node.name
        if isinstance(node, str):
            return node
        if isinstance(node, ArrayAccessNode):
            array_node = getattr(node, "array", getattr(node, "array_expr", None))
            return self.get_expression_name(array_node)
        return None

    def get_expression_type(self, node):
        name = self.get_expression_name(node)
        if name is None:
            return None
        return self.variable_types.get(name)

    def resource_expression_type(self, node):
        type_name = self.get_expression_type(node)
        if type_name is None:
            type_name = self.expression_result_type(node)
        if type_name is not None and not isinstance(type_name, str):
            return self.convert_type_node_to_string(type_name)
        return type_name

    def resource_query_metadata_unavailable_call(
        self, func_name, resource_type, fallback
    ):
        resource_type = resource_type or "unknown resource"
        return (
            f"/* unsupported {self.resource_backend_name()} resource query: "
            f"{func_name} metadata unavailable on {resource_type} */ {fallback}"
        )

    def unsupported_dimension_metadata_query_call(self, func_name, resource_type):
        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return None
        return_type = self.query_return_type(spec["dimensions"])
        fallback = self.query_constructor(
            return_type,
            ["0"] * len(spec["dimensions"]),
        )
        return self.resource_query_metadata_unavailable_call(
            func_name, resource_type, fallback
        )

    def unsupported_scalar_metadata_query_call(self, func_name, resource_type):
        return self.resource_query_metadata_unavailable_call(
            func_name, resource_type, "0"
        )

    def generate_dimension_query(self, func_name, raw_args, args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return None

        metadata_expr = self.query_metadata_expression(raw_args[0])
        if metadata_expr is None:
            return self.unsupported_dimension_metadata_query_call(
                func_name, resource_type
            )

        self.resource_query_info_required = True
        helper_name = f"cgl_{func_name}_{resource_type}"
        if spec["mip"]:
            self.ensure_query_prefix_helper()
        self.require_helper_function(
            helper_name, self.build_dimension_query_helper(helper_name, spec)
        )

        if spec["mip"]:
            lod = args[1] if len(args) > 1 else "0"
            return f"{helper_name}({metadata_expr}, {lod})"
        return f"{helper_name}({metadata_expr})"

    def generate_sample_count_query(self, func_name, raw_args, args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        spec = self.dimension_query_spec(resource_type)
        if spec is None or not spec["samples"]:
            return None

        metadata_expr = self.query_metadata_expression(raw_args[0])
        if metadata_expr is None:
            return self.unsupported_scalar_metadata_query_call(func_name, resource_type)

        self.resource_query_info_required = True
        helper_name = f"cgl_{func_name}_{resource_type}"
        self.require_helper_function(
            helper_name, self.build_sample_count_query_helper(helper_name)
        )
        return f"{helper_name}({metadata_expr})"

    def generate_texture_query_levels(self, raw_args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        if not self.is_sampled_resource_type(resource_type):
            return None
        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return None

        metadata_expr = self.query_metadata_expression(raw_args[0])
        if metadata_expr is None:
            return self.unsupported_scalar_metadata_query_call(
                "textureQueryLevels", resource_type
            )

        self.resource_query_info_required = True
        helper_name = f"cgl_textureQueryLevels_{resource_type}"
        self.require_helper_function(
            helper_name, self.build_texture_query_levels_helper(helper_name, spec)
        )
        return f"{helper_name}({metadata_expr})"

    def map_vector_arithmetic_type(self, type_name):
        return self.convert_crossgl_type_to_cuda(type_name)

    def insert_helper_functions(self):
        if not self.helper_functions:
            return

        helper_lines = []
        if self.resource_query_info_required:
            helper_lines.extend(
                [
                    "struct CglResourceQueryInfo {",
                    "    int width;",
                    "    int height;",
                    "    int depth;",
                    "    int elements;",
                    "    int levels;",
                    "    int samples;",
                    "};",
                    "",
                ]
            )
        for helper in self.helper_functions.values():
            helper_lines.extend(helper.splitlines())
            helper_lines.append("")

        self.output[3:3] = helper_lines

    def vector_component_count_for_type(self, type_name):
        if not isinstance(type_name, str):
            return None

        base_type = type_name.split("[", 1)[0].strip()
        scalar_types = {
            "bool",
            "double",
            "f16",
            "f32",
            "f64",
            "float",
            "half",
            "i32",
            "int",
            "u32",
            "uint",
        }
        if base_type in scalar_types:
            return 1

        vector_prefixes = (
            "bvec",
            "dvec",
            "ivec",
            "uvec",
            "vec",
            "bool",
            "double",
            "float",
            "half",
            "int",
            "uint",
        )
        for prefix in vector_prefixes:
            if not base_type.startswith(prefix):
                continue
            suffix = base_type[len(prefix) :]
            if suffix and suffix[0] in {"2", "3", "4"}:
                return int(suffix[0])
        return None

    def texel_fetch_coordinate_count(self, raw_coord):
        coord_type = self.get_expression_type(raw_coord)
        if coord_type is None:
            coord_type = self.expression_result_type(raw_coord)
        return self.vector_component_count_for_type(coord_type)

    def expected_texel_fetch_coordinate_count(self, texture_type):
        return {
            "sampler1D": 1,
            "sampler1DArray": 2,
            "sampler2D": 2,
            "sampler2DArray": 3,
            "sampler3D": 3,
        }.get(texture_type)

    def expected_texture_coordinate_count(self, texture_type):
        return {
            "sampler1D": 1,
            "sampler1DArray": 2,
            "sampler2D": 2,
            "sampler2DArray": 3,
            "sampler3D": 3,
            "samplerCube": 3,
            "samplerCubeArray": 4,
        }.get(texture_type)

    def texture_coordinate_rank_diagnostic(self, func_name, texture_type, raw_coord):
        expected_coord_count = self.expected_texture_coordinate_count(texture_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_sampled_resource_call(
                f"{func_name} coordinate rank",
                texture_type,
                [],
            )
        return None

    def texture_offset_argument_index(self, func_name):
        return {
            "textureOffset": 2,
            "textureLodOffset": 3,
            "textureGradOffset": 4,
            "textureProjOffset": 2,
            "textureProjLodOffset": 3,
            "textureProjGradOffset": 4,
            "texelFetchOffset": 3,
        }.get(func_name)

    def expected_texture_offset_coordinate_count(self, texture_type):
        return {
            "sampler1D": 1,
            "sampler1DArray": 1,
            "sampler2D": 2,
            "sampler2DArray": 2,
            "sampler3D": 3,
        }.get(texture_type)

    def texture_offset_rank_diagnostic(self, func_name, texture_type, raw_offset):
        expected_offset_count = self.expected_texture_offset_coordinate_count(
            texture_type
        )
        actual_offset_count = self.texel_fetch_coordinate_count(raw_offset)
        if (
            expected_offset_count is not None
            and actual_offset_count is not None
            and actual_offset_count != expected_offset_count
        ):
            return self.unsupported_sampled_resource_call(
                f"{func_name} offset rank",
                texture_type,
                [],
            )
        return None

    def expected_texture_gradient_coordinate_counts(self, texture_type):
        return {
            "sampler1D": {1},
            "sampler1DArray": {1},
            "sampler2D": {2},
            "sampler2DArray": {2},
            "sampler3D": {3, 4},
            "samplerCube": {3, 4},
            "samplerCubeArray": {3, 4},
        }.get(texture_type)

    def texture_gradient_rank_diagnostic(self, texture_type, *raw_gradients):
        expected_gradient_counts = self.expected_texture_gradient_coordinate_counts(
            texture_type
        )
        if expected_gradient_counts is None:
            return None

        for raw_gradient in raw_gradients:
            actual_gradient_count = self.texel_fetch_coordinate_count(raw_gradient)
            if (
                actual_gradient_count is not None
                and actual_gradient_count not in expected_gradient_counts
            ):
                return self.unsupported_sampled_resource_call(
                    "textureGrad derivative rank",
                    texture_type,
                    [],
                )
        return None

    def cuda_texture_gradient_argument(self, texture_type, raw_gradient, gradient):
        if texture_type not in {"sampler3D", "samplerCube", "samplerCubeArray"}:
            return gradient

        gradient_count = self.texel_fetch_coordinate_count(raw_gradient)
        if gradient_count == 3:
            return (
                f"make_float4({self.coord_component(gradient, 'x')}, "
                f"{self.coord_component(gradient, 'y')}, "
                f"{self.coord_component(gradient, 'z')}, 0.0f)"
            )
        return gradient

    def expected_image_coordinate_count(self, image_type):
        image_type = self.resource_base_type(image_type)
        if not isinstance(image_type, str):
            return None
        if "1DArray" in image_type:
            return 2
        if "1D" in image_type:
            return 1
        if "3D" in image_type:
            return 3
        if "Cube" in image_type:
            return 3
        if "Array" in image_type:
            return 3
        if "2D" in image_type:
            return 2
        return None

    def unsupported_image_coordinate_rank_call(self, func_name, image_type):
        fallback = "((void)0)"
        if func_name == "imageLoad":
            fallback = self.zero_value_for_type(self.image_value_type(image_type))
        return (
            f"/* unsupported CUDA image resource call: "
            f"{func_name} coordinate rank on {image_type} */ {fallback}"
        )

    def image_resource_access(self, image_arg):
        if isinstance(image_arg, FunctionCallNode):
            callee_name = self.raw_function_call_name(image_arg)
            return_source = self.query_return_sources.get(callee_name)
            if return_source is None:
                return None
            raw_args = getattr(
                image_arg,
                "arguments",
                getattr(image_arg, "args", []),
            )
            return self.query_return_source_image_access(return_source, raw_args)

        if isinstance(image_arg, ArrayAccessNode):
            array_node = getattr(
                image_arg, "array", getattr(image_arg, "array_expr", None)
            )
            return self.image_resource_access(array_node)

        if isinstance(image_arg, MemberAccessNode):
            object_node = getattr(
                image_arg,
                "object_expr",
                getattr(image_arg, "object", None),
            )
            object_type = self.type_name_string(
                self.expression_result_type(object_node)
            )
            member = getattr(image_arg, "member", None)
            access = self.struct_member_image_accesses.get(object_type, {}).get(member)
            if access is not None:
                return access

        image_name = self.get_expression_name(image_arg)
        if not image_name:
            return None
        return self.image_resource_accesses.get(image_name)

    def query_return_source_image_access(self, return_source, raw_args):
        """Return storage-image access for a traceable returned resource."""
        kind = return_source.get("kind")
        if kind == "ternary":
            true_access = self.query_return_source_image_access(
                return_source["true_source"],
                raw_args,
            )
            false_access = self.query_return_source_image_access(
                return_source["false_source"],
                raw_args,
            )
            if true_access is None or true_access != false_access:
                return None
            return true_access

        if kind == "global":
            return self.image_resource_access(return_source.get("name"))

        if kind == "parameter":
            index = return_source.get("index")
            if index is None or index >= len(raw_args):
                return None
            return self.image_resource_access(raw_args[index])

        if kind == "member":
            object_type = return_source.get("object_type")
            member = return_source.get("member")
            return self.struct_member_image_accesses.get(object_type, {}).get(member)

        return None

    def unsupported_image_access_call(self, func_name, image_type, reason):
        image_type = image_type or "unknown resource"
        fallback = "((void)0)"
        if func_name == "imageLoad":
            fallback = self.zero_value_for_type(self.image_value_type(image_type))
        return (
            f"/* unsupported CUDA image access: "
            f"{func_name} {reason} on {image_type} */ {fallback}"
        )

    def image_access_diagnostic(self, func_name, image_type, raw_image):
        access = self.image_resource_access(raw_image)
        if func_name == "imageLoad" and access == "writeonly":
            return self.unsupported_image_access_call(
                func_name,
                image_type,
                "requires readable image resource",
            )
        if func_name == "imageStore" and access == "readonly":
            return self.unsupported_image_access_call(
                func_name,
                image_type,
                "requires writable image resource",
            )
        return None

    def image_coordinate_rank_diagnostic(self, func_name, image_type, raw_coord):
        expected_coord_count = self.expected_image_coordinate_count(image_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_image_coordinate_rank_call(func_name, image_type)
        return None

    def unsupported_image_atomic_coordinate_rank_call(self, func_name, image_type):
        image_type = image_type or "unknown resource"
        return (
            f"/* unsupported CUDA image atomic resource call: "
            f"{func_name} coordinate rank on {image_type} */ "
            f"{self.image_atomic_zero_value(image_type)}"
        )

    def unsupported_image_atomic_access_call(self, func_name, image_type):
        image_type = image_type or "unknown resource"
        return (
            f"/* unsupported CUDA image atomic resource call: "
            f"{func_name} requires readwrite image resource on {image_type} */ "
            f"{self.image_atomic_zero_value(image_type)}"
        )

    def image_atomic_access_diagnostic(self, func_name, image_type, raw_image):
        if self.image_resource_access(raw_image) in {"readonly", "writeonly"}:
            return self.unsupported_image_atomic_access_call(func_name, image_type)
        return None

    def image_atomic_coordinate_rank_diagnostic(self, func_name, image_type, raw_coord):
        expected_coord_count = self.expected_image_coordinate_count(image_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_image_atomic_coordinate_rank_call(
                func_name, image_type
            )
        return None

    def unsupported_scalar_resource_query_call(self, func_name, resource_type):
        resource_type = resource_type or "unknown resource"
        return (
            f"/* unsupported {self.resource_backend_name()} resource query: "
            f"{func_name} on {resource_type} */ 0"
        )

    def generate_resource_call(self, func_name, raw_args, args):
        if func_name in {"textureSize", "imageSize"}:
            return self.generate_dimension_query(func_name, raw_args, args)

        if func_name in {"textureSamples", "imageSamples"}:
            sample_count_query = self.generate_sample_count_query(
                func_name, raw_args, args
            )
            if sample_count_query is not None:
                return sample_count_query
            if raw_args:
                resource_type = self.resource_base_type(
                    self.resource_expression_type(raw_args[0])
                )
                if resource_type is not None:
                    return self.unsupported_scalar_resource_query_call(
                        func_name, resource_type
                    )
            return None

        if func_name == "textureQueryLevels":
            return self.generate_texture_query_levels(raw_args)

        if func_name == "textureQueryLod" and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_resource_query_call(
                    func_name, texture_type, args
                )

        if (
            func_name
            in {
                "texture",
                "textureLod",
                "textureGrad",
                "textureGather",
                "textureCompare",
                "textureCompareLod",
                "textureCompareGrad",
                "textureCompareOffset",
                "textureCompareLodOffset",
                "textureCompareGradOffset",
                "textureCompareProj",
                "textureCompareProjOffset",
                "textureCompareProjLod",
                "textureCompareProjLodOffset",
                "textureCompareProjGrad",
                "textureCompareProjGradOffset",
                "textureGatherCompare",
                "textureGatherCompareOffset",
            }
            and len(args) >= 2
        ):
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if self.is_shadow_resource_type(texture_type):
                return self.unsupported_shadow_resource_call(
                    func_name, texture_type, args
                )

        if func_name == "textureGather" and len(args) >= 2:
            texture_gather = self.generate_texture_gather_call(raw_args, args)
            if texture_gather is not None:
                return texture_gather

        if (
            func_name
            in {
                "textureGather",
                "textureGatherOffset",
                "textureGatherOffsets",
            }
            and len(args) >= 2
        ):
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

        if (
            func_name
            in {
                "textureOffset",
                "textureLodOffset",
                "textureGradOffset",
                "textureProj",
                "textureProjOffset",
                "textureProjLod",
                "textureProjLodOffset",
                "textureProjGrad",
                "textureProjGradOffset",
                "texelFetchOffset",
            }
            and raw_args
        ):
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if texture_type is not None:
                if self.is_shadow_resource_type(texture_type):
                    return self.unsupported_shadow_resource_call(
                        func_name, texture_type, args
                    )
                if self.is_multisample_resource_type(texture_type):
                    return self.unsupported_multisample_resource_call(
                        func_name, texture_type, args
                    )
                offset_arg_index = self.texture_offset_argument_index(func_name)
                if offset_arg_index is not None and len(raw_args) > offset_arg_index:
                    offset_diagnostic = self.texture_offset_rank_diagnostic(
                        func_name,
                        texture_type,
                        raw_args[offset_arg_index],
                    )
                    if offset_diagnostic is not None:
                        return offset_diagnostic
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

        if func_name in {
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        }:
            image_type = None
            if raw_args:
                image_type = self.resource_base_type(
                    self.resource_expression_type(raw_args[0])
                )
                access_diagnostic = self.image_atomic_access_diagnostic(
                    func_name, image_type, raw_args[0]
                )
                if access_diagnostic is not None:
                    return access_diagnostic
            if len(raw_args) >= 2:
                coordinate_diagnostic = self.image_atomic_coordinate_rank_diagnostic(
                    func_name, image_type, raw_args[1]
                )
                if coordinate_diagnostic is not None:
                    return coordinate_diagnostic
            return self.unsupported_image_atomic_resource_call(
                func_name, image_type, args
            )

        if func_name in {"texture", "textureLod", "textureGrad"} and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if self.is_multisample_resource_type(texture_type):
                return self.unsupported_multisample_resource_call(
                    func_name, texture_type, args
                )
            coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
                func_name, texture_type, raw_args[1]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            grad_x = None
            grad_y = None
            if func_name == "textureGrad" and len(args) >= 4:
                gradient_diagnostic = self.texture_gradient_rank_diagnostic(
                    texture_type,
                    raw_args[2],
                    raw_args[3],
                )
                if gradient_diagnostic is not None:
                    return gradient_diagnostic
                grad_x = self.cuda_texture_gradient_argument(
                    texture_type,
                    raw_args[2],
                    args[2],
                )
                grad_y = self.cuda_texture_gradient_argument(
                    texture_type,
                    raw_args[3],
                    args[3],
                )

            texture_name = args[0]
            coord = args[1]
            if texture_type == "sampler1D":
                if func_name == "texture":
                    return f"tex1D({texture_name}, {coord})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex1DLod({texture_name}, {coord}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return f"tex1DGrad({texture_name}, {coord}, {grad_x}, {grad_y})"

            if texture_type == "sampler1DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex1DLayered<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex1DLayeredLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"tex1DLayeredGrad<float4>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_type == "sampler2D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex2D({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex2DLod({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return f"tex2DGrad({coord_args}, {grad_x}, {grad_y})"

            if texture_type == "sampler2DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex2DLayered<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex2DLayeredLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"tex2DLayeredGrad<float4>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_type == "sampler3D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex3D({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex3DLod({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return f"tex3DGrad({coord_args}, {grad_x}, {grad_y})"

            if texture_type == "samplerCube":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"texCubemap({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"texCubemapLod({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return f"texCubemapGrad({coord_args}, {grad_x}, {grad_y})"

            if texture_type == "samplerCubeArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}, "
                    f"{self.coord_component(coord, 'w')}"
                )
                if func_name == "texture":
                    return f"texCubemapLayered<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"texCubemapLayeredLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"texCubemapLayeredGrad<float4>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

        if func_name == "texelFetch" and len(args) >= 3:
            texture_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if self.is_multisample_resource_type(texture_type):
                return self.unsupported_multisample_resource_call(
                    func_name, texture_type, args
                )
            if texture_type in {"samplerCube", "samplerCubeArray"}:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

            texture_name = args[0]
            coord = args[1]
            expected_coord_count = self.expected_texel_fetch_coordinate_count(
                texture_type
            )
            actual_coord_count = self.texel_fetch_coordinate_count(raw_args[1])
            if (
                expected_coord_count is not None
                and actual_coord_count is not None
                and actual_coord_count != expected_coord_count
            ):
                return self.unsupported_sampled_resource_call(
                    "texelFetch coordinate rank",
                    texture_type,
                    args,
                )
            if texture_type == "sampler1D":
                return f"tex1Dfetch({texture_name}, {coord})"
            if texture_type == "sampler1DArray":
                return (
                    f"tex1DLayered<float4>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')})"
                )
            if texture_type == "sampler2D":
                return (
                    f"tex2D({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')})"
                )
            if texture_type == "sampler2DArray":
                return (
                    f"tex2DLayered<float4>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')})"
                )
            if texture_type == "sampler3D":
                return (
                    f"tex3D({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')})"
                )

        if func_name == "imageLoad" and len(args) >= 2:
            image_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if image_type is None:
                return None
            if self.is_multisample_resource_type(image_type):
                return self.unsupported_multisample_resource_call(
                    func_name, image_type, args
                )
            access_diagnostic = self.image_access_diagnostic(
                func_name, image_type, raw_args[0]
            )
            if access_diagnostic is not None:
                return access_diagnostic

            image_name = args[0]
            coord = args[1]
            coordinate_diagnostic = self.image_coordinate_rank_diagnostic(
                func_name, image_type, raw_args[1]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            value_type = self.image_value_type(image_type)
            x = self.surface_x_offset(coord, value_type)
            y = self.coord_component(coord, "y")

            if "CubeArray" in image_type:
                layer_face = self.coord_component(coord, "z")
                return (
                    f"surfCubemapLayeredread<{value_type}>"
                    f"({image_name}, {x}, {y}, {layer_face})"
                )
            if "Cube" in image_type:
                face = self.coord_component(coord, "z")
                return f"surfCubemapread<{value_type}>({image_name}, {x}, {y}, {face})"
            if "3D" in image_type:
                z = self.coord_component(coord, "z")
                return f"surf3Dread<{value_type}>({image_name}, {x}, {y}, {z})"
            if "1DArray" in image_type:
                layer = self.coord_component(coord, "y")
                return f"surf1DLayeredread<{value_type}>({image_name}, {x}, {layer})"
            if "Array" in image_type:
                layer = self.coord_component(coord, "z")
                return (
                    f"surf2DLayeredread<{value_type}>"
                    f"({image_name}, {x}, {y}, {layer})"
                )
            if "1D" in image_type:
                x = f"{coord} * sizeof({value_type})"
                return f"surf1Dread<{value_type}>({image_name}, {x})"
            if "2D" in image_type:
                return f"surf2Dread<{value_type}>({image_name}, {x}, {y})"

        if func_name == "imageStore" and len(args) >= 3:
            image_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if image_type is None:
                return None
            if self.is_multisample_resource_type(image_type):
                return self.unsupported_multisample_resource_call(
                    func_name, image_type, args
                )
            access_diagnostic = self.image_access_diagnostic(
                func_name, image_type, raw_args[0]
            )
            if access_diagnostic is not None:
                return access_diagnostic

            image_name = args[0]
            coord = args[1]
            value = args[2]
            coordinate_diagnostic = self.image_coordinate_rank_diagnostic(
                func_name, image_type, raw_args[1]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            value_type = self.image_value_type(image_type)
            x = self.surface_x_offset(coord, value_type)
            y = self.coord_component(coord, "y")

            if "CubeArray" in image_type:
                layer_face = self.coord_component(coord, "z")
                return (
                    f"surfCubemapLayeredwrite"
                    f"({value}, {image_name}, {x}, {y}, {layer_face})"
                )
            if "Cube" in image_type:
                face = self.coord_component(coord, "z")
                return f"surfCubemapwrite({value}, {image_name}, {x}, {y}, {face})"
            if "3D" in image_type:
                z = self.coord_component(coord, "z")
                return f"surf3Dwrite({value}, {image_name}, {x}, {y}, {z})"
            if "1DArray" in image_type:
                layer = self.coord_component(coord, "y")
                return f"surf1DLayeredwrite({value}, {image_name}, {x}, {layer})"
            if "Array" in image_type:
                layer = self.coord_component(coord, "z")
                return f"surf2DLayeredwrite({value}, {image_name}, {x}, {y}, {layer})"
            if "1D" in image_type:
                x = f"{coord} * sizeof({value_type})"
                return f"surf1Dwrite({value}, {image_name}, {x})"
            if "2D" in image_type:
                return f"surf2Dwrite({value}, {image_name}, {x}, {y})"

        return None

    def generate_texture_gather_call(self, raw_args, args):
        texture_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        if texture_type != "sampler2D":
            return None

        coordinate_index = 1
        component = None
        component_arg = None
        if len(args) == 2:
            pass
        elif len(args) == 3:
            if self.is_sampler_state_argument(raw_args[1]):
                coordinate_index = 2
            else:
                component = args[2]
                component_arg = raw_args[2]
        elif len(args) == 4 and self.is_sampler_state_argument(raw_args[1]):
            coordinate_index = 2
            component = args[3]
            component_arg = raw_args[3]
        else:
            return None

        coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
            "textureGather",
            texture_type,
            raw_args[coordinate_index],
        )
        if coordinate_diagnostic is not None:
            return coordinate_diagnostic

        component_diagnostic = self.texture_gather_component_diagnostic(
            texture_type, component_arg
        )
        if component_diagnostic is not None:
            return component_diagnostic

        texture_name = args[0]
        coord = args[coordinate_index]
        gather_args = (
            f"{texture_name}, "
            f"{self.coord_component(coord, 'x')}, "
            f"{self.coord_component(coord, 'y')}"
        )
        if component is not None:
            return f"tex2Dgather<float4>({gather_args}, {component})"
        return f"tex2Dgather<float4>({gather_args})"

    def texture_gather_component_diagnostic(self, texture_type, component_arg):
        if not self.is_texture_gather_literal_component(component_arg):
            return None
        component_value = self.literal_int_value(component_arg)
        if component_value in {0, 1, 2, 3}:
            return None
        return self.unsupported_sampled_resource_call(
            "textureGather component literal must be 0, 1, 2, or 3",
            texture_type,
            [],
        )

    def is_texture_gather_literal_component(self, node):
        if isinstance(node, LiteralNode):
            return True
        if isinstance(node, UnaryOpNode) and not getattr(node, "is_postfix", False):
            operator = getattr(node, "operator", getattr(node, "op", None))
            if operator in {"+", "-"}:
                return self.is_texture_gather_literal_component(node.operand)
        return False

    def literal_int_value(self, node):
        if isinstance(node, LiteralNode):
            value = node.value
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                literal_type = getattr(
                    getattr(node, "literal_type", None), "name", None
                )
                if literal_type in {"char", "string"}:
                    return None
                try:
                    return int(value, 0)
                except ValueError:
                    return None
            return None

        if isinstance(node, UnaryOpNode) and not getattr(node, "is_postfix", False):
            operator = getattr(node, "operator", getattr(node, "op", None))
            if operator not in {"+", "-"}:
                return None
            value = self.literal_int_value(node.operand)
            if value is None:
                return None
            if operator == "-":
                return -value
            return value

        return None

    def is_sampler_state_argument(self, raw_arg):
        return self.resource_base_type(self.get_expression_type(raw_arg)) == "sampler"

    def visit_cbuffer(self, cbuffer):
        """Visit constant buffer (convert to CUDA constant memory)"""
        self.emit(f"// Constant buffer: {cbuffer.name}")
        for member in cbuffer.members:
            if hasattr(member, "member_type"):
                member_type = member.member_type
            else:
                member_type = member.vtype
            declaration = self.format_typed_declarator(member_type, member.name)
            self.emit(f"__constant__ {declaration};")

    def visit_ArrayNode(self, node):
        """Visit array declaration"""
        if hasattr(node, "element_type"):
            element_type = self.convert_crossgl_type_to_cuda(node.element_type)
        else:
            element_type = self.convert_crossgl_type_to_cuda(node.vtype)

        if node.size:
            self.emit(
                f"{element_type} {node.name}[{self.format_array_size(node.size)}];"
            )
        else:
            # Dynamic array - use pointer in CUDA
            self.emit(f"{element_type}* {node.name};")

    def visit_TernaryOpNode(self, node):
        """Visit ternary conditional operator"""
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        lowered = self.lower_vector_ternary_operation(
            node.condition,
            condition,
            node.true_expr,
            true_expr,
            node.false_expr,
            false_expr,
        )
        if lowered is not None:
            return lowered
        return f"({condition} ? {true_expr} : {false_expr})"

    def visit_list(self, node_list):
        """Visit a list of nodes"""
        results = []
        for node in node_list:
            result = self.visit(node)
            if result:
                results.append(result)
        return results

    def visit_str(self, node):
        """Visit string literals"""
        return node

    def visit_int(self, node):
        """Visit integer literals"""
        return str(node)

    def visit_float(self, node):
        """Visit float literals"""
        return str(node)

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "pointee_type"):
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            prefix = "" if getattr(type_node, "is_mutable", True) else "const "
            return f"{prefix}{pointee_type}*"
        if hasattr(type_node, "referenced_type"):
            referenced_type = self.convert_type_node_to_string(
                type_node.referenced_type
            )
            prefix = "" if getattr(type_node, "is_mutable", False) else "const "
            return f"{prefix}{referenced_type}&"
        if hasattr(type_node, "name"):
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                args = ", ".join(
                    self.convert_type_node_to_string(arg) for arg in generic_args
                )
                return f"{type_node.name}<{args}>"
            return type_node.name
        elif hasattr(type_node, "element_type"):
            if hasattr(type_node, "rows"):
                element_type = self.convert_type_node_to_string(type_node.element_type)
                prefix = "dmat" if element_type == "double" else "mat"
                return f"{prefix}{type_node.rows}x{type_node.cols}"
            elif not hasattr(type_node, "size"):
                return str(type_node)
            elif str(type(type_node)).find("ArrayType") != -1:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    return f"{element_type}[{self.format_array_size(type_node.size)}]"
                else:
                    return f"{element_type}[]"
            else:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
                if element_type == "float":
                    return f"vec{size}"
                elif element_type == "int":
                    return f"ivec{size}"
                elif element_type == "uint":
                    return f"uvec{size}"
                else:
                    return f"{element_type}{size}"
        else:
            return str(type_node)

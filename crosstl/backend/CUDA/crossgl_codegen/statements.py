"""Lambdas, unique pointers, new and delete, type aliases, atomics, synchronization, inline assembly, built-ins, member and array access, initializers and control-flow statements in the CUDA to CrossGL converter."""

import re

from ...assignment_operands import grouped_operand
from ..CudaAst import InitializerListNode, MemberAccessNode, UnaryOpNode, VariableNode


class CudaImportStatementMixin:
    """Statement members of ``CudaToCrossGLConverter``."""

    def format_vector_constructor(self, vector_name, args, element_type="f32"):
        if vector_name == "vec1":
            return args[0]
        return f"{vector_name}<{element_type}>({', '.join(args)})"

    def format_lambda_call(self, args):
        if not args:
            return "lambda()"

        rendered_args = [self.format_lambda_parameter(arg) for arg in args[:-1]] + [
            self.format_lambda_body(args[-1])
        ]
        return f"lambda({', '.join(rendered_args)})"

    def format_lambda_parameter(self, arg):
        if isinstance(arg, VariableNode):
            if arg.vtype:
                param_type = self.convert_cuda_variable_type_to_crossgl(
                    arg.vtype, arg.name
                )
                return f"{param_type} {arg.name}"
            return arg.name
        return self.format_lambda_body(arg)

    def format_lambda_body(self, arg):
        if isinstance(arg, str):
            return arg
        return self.visit(arg)

    def format_cuda_fold_expression(self, args):
        expression = args[0] if args else ""
        return self.refuse_import(
            f"CUDA fold expression {expression}: CrossGL has no parameter packs", "0"
        )

    def is_get_method_call(self, node):
        return (
            isinstance(node.name, MemberAccessNode)
            and node.name.member == "get"
            and not node.args
            and self.is_unique_ptr_expression(node.name.object)
        )

    def format_make_unique_call(self, function_name, args):
        base_name, template_args = self.parse_cpp_template(function_name)
        if not self.is_std_make_unique_base_name(base_name) or not template_args:
            return None

        target_type, is_array = self.unwrap_array_template_type(template_args[0])
        target_type = self.convert_cuda_type_to_crossgl(target_type)
        args_str = ", ".join(args)
        if is_array:
            return f"new_array<{target_type}>({args_str})"
        return f"new<{target_type}>({args_str})"

    def format_unique_ptr_constructor_call(self, function_name, args):
        base_name, _ = self.parse_cpp_template(function_name)
        if len(args) != 1:
            return None
        if not self.is_std_unique_ptr_base_name(
            base_name
        ) and not self.is_unique_ptr_type_name(function_name):
            return None

        return args[0]

    def visit_NewNode(self, node):
        target_type = self.convert_cuda_type_to_crossgl(node.target_type)
        if node.is_array:
            size = self.visit(node.size) if node.size is not None else ""
            return f"new_array<{target_type}>({size})"

        args = ", ".join(self.visit(arg) for arg in node.args)
        return f"new<{target_type}>({args})"

    def visit_DeleteNode(self, node):
        target = self.visit(node.expression)
        form = "delete[]" if node.is_array else "delete"
        self.refuse_import(f"CUDA {form} {target}: CrossGL has no device heap")

    def visit_TypeAliasNode(self, node):
        """Write ``typedef``/``using`` as a CrossGL ``typedef`` (#3309).

        CrossGL has no ``decltype``, so such an alias stays a comment and its
        uses are refused downstream as undeclared types.
        """
        self.register_type_alias(node.name, node.alias_type)
        # An alias named ``let`` is declared as ``let_``, the name its uses get.
        alias_name = self.sanitize_identifier_name(node.name)
        elaborated = re.fullmatch(
            r"(?:struct|union)\s+([A-Za-z_]\w*)", str(node.alias_type).strip()
        )
        if elaborated is not None:
            # ``typedef struct Pair PairT;`` names the record ``Pair``; the
            # self-alias ``typedef struct Q {...} Q;`` declares nothing new.
            alias_type = elaborated.group(1)
            if alias_type == node.name:
                return
            alias_type = self.rename_keyword_record_names(alias_type)
        elif str(node.alias_type).strip().startswith(("struct ", "union ")):
            alias_type = node.alias_type
        elif self.is_decltype_type_name(node.alias_type):
            self.emit(f"// typedef {node.alias_type} {node.name};")
            return
        else:
            alias_type = self.convert_cuda_type_to_crossgl(node.alias_type)
            layouts = getattr(self, "emitted_buffer_layouts", None)
            if (
                layouts is not None
                and not getattr(node, "array_sizes", None)
                and not getattr(node, "declarator_type_suffix", "")
            ):
                layouts.alias(alias_name, alias_type)
        self.emit(f"typedef {alias_type} {alias_name};")

    def format_atomic_argument(self, arg, index):
        if index == 0 and isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.visit(arg.operand)
        return self.visit(arg)

    def visit_AtomicOperationNode(self, node):
        args = [self.format_atomic_argument(arg, i) for i, arg in enumerate(node.args)]
        args_str = ", ".join(args)
        operation = node.operation
        scope = None
        for suffix, scope_name in (("_block", "block"), ("_system", "system")):
            if operation.endswith(suffix):
                operation = operation[: -len(suffix)]
                scope = scope_name
                break

        atomic_map = {
            "atomicAdd": "atomicAdd",
            "atomicSub": "atomicSub",
            "atomicMax": "atomicMax",
            "atomicMin": "atomicMin",
            "atomicExch": "atomicExchange",
            "atomicCAS": "atomicCompareExchange",
            "atomicAnd": "atomicAnd",
            "atomicOr": "atomicOr",
            "atomicXor": "atomicXor",
            "atomicInc": "atomicInc",
            "atomicDec": "atomicDec",
        }

        # The parser builds this node only for ATOMIC_FUNCTION_NAMES, whose base names atomic_map covers.
        crossgl_func = atomic_map[operation]
        expression = f"{crossgl_func}({args_str})"
        if scope == "system":
            # A system-scoped atomic is also atomic for the host and peer devices. CrossGL keeps its CUDA name, which
            # CUDA and HIP output write back and the other generators refuse with a typed diagnostic.
            return f"{node.operation}({args_str})"
        # A block-scoped atomic is atomic for its block; the device-scoped CrossGL atomic is atomic for every
        # thread on the device, which includes the block.
        return expression

    def visit_CudaAsmNode(self, node):
        volatility = " volatile" if node.is_volatile else ""
        self.refuse_import(
            f"CUDA inline PTX asm{volatility}({node.template}): CrossGL has no inline assembly"
        )

    def format_cuda_asm_operands(self, operands):
        formatted = []
        for operand in operands:
            prefix = (
                f"[{operand.symbolic_name}] "
                if operand.symbolic_name is not None
                else ""
            )
            expression = (
                self.visit(operand.expression)
                if operand.expression is not None
                else None
            )
            if expression is None:
                formatted.append(f"{prefix}{operand.constraint}")
            else:
                formatted.append(f"{prefix}{operand.constraint}({expression})")
        return ", ".join(formatted)

    def visit_CudaBuiltinNode(self, node):
        builtin_map = {
            "threadIdx": "gl_LocalInvocationID",
            "blockIdx": "gl_WorkGroupID",
            "gridDim": "gl_NumWorkGroups",
            "blockDim": "gl_WorkGroupSize",
            "warpSize": "32",  # Constant in CrossGL
        }

        # The parser builds this node only for the five built-in tokens builtin_map covers.
        base_name = builtin_map[node.builtin_name]
        if node.component:
            return f"{base_name}.{node.component}"
        else:
            return base_name

    def visit_MemberAccessNode(self, node):
        obj = self.visit(node.object)
        operator = "->" if getattr(node, "is_pointer", False) else "."
        if node.member == "x" and operator == "." and self.is_vector1_name(node.object):
            return obj
        member = self.sanitize_identifier_name(node.member)
        return f"{obj}{operator}{member}"

    def visit_ArrayAccessNode(self, node):
        array = self.visit(node.array)
        index = self.visit(node.index)
        return f"{array}[{index}]"

    def visit_InitializerListNode(self, node):
        elements = ", ".join(self.visit(element) for element in node.elements)
        return f"{{{elements}}}"

    def visit_DesignatedInitializerNode(self, node):
        designators = []
        for kind, target in node.designators:
            if kind == "index":
                designators.append(f"[{self.visit(target)}]")
            else:
                designators.append(f".{target}")

        value = self.visit(node.value)
        return f"{''.join(designators)} = {value}"

    def visit_BlockNode(self, node):
        """A nested ``{ ... }`` stays a CrossGL block, with its own variable and name scopes."""
        self.emit("{")
        self.indent_level += 1
        self.push_variable_type_scope()
        self.push_identifier_name_scope()
        try:
            for stmt in node.statements:
                self.emit_statement(stmt)
        finally:
            self.pop_identifier_name_scope()
            self.pop_variable_type_scope()
            self.indent_level -= 1
        self.emit("}")
        return ""

    def visit_IfNode(self, node):
        condition = self.visit(node.condition)
        self.emit(f"if ({condition}) {{")

        self.indent_level += 1
        if isinstance(node.if_body, list):
            for stmt in node.if_body:
                self.emit_statement(stmt)
        else:
            self.emit_statement(node.if_body)
        self.indent_level -= 1

        if node.else_body:
            self.emit("} else {")
            self.indent_level += 1
            if isinstance(node.else_body, list):
                for stmt in node.else_body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(node.else_body)
            self.indent_level -= 1

        self.emit("}")

    def visit_ForNode(self, node):
        scoped_init = isinstance(node.init, list)
        if scoped_init:
            self.emit("{")
            self.indent_level += 1
            for stmt in node.init:
                self.emit_statement(stmt)
            init = ""
        else:
            init = self.format_statement_fragment(node.init)
        condition = self.visit(node.condition) if node.condition else ""
        update = self.format_statement_fragment(node.update)

        self.emit(f"for ({init}; {condition}; {update}) {{")

        self.indent_level += 1
        if isinstance(node.body, list):
            for stmt in node.body:
                self.emit_statement(stmt)
        else:
            self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit("}")
        if scoped_init:
            self.indent_level -= 1
            self.emit("}")

    def visit_RangeForNode(self, node):
        iterable = self.visit(node.iterable)
        self.emit(f"for {node.name} in {iterable} {{")

        self.indent_level += 1
        if isinstance(node.body, list):
            for stmt in node.body:
                self.emit_statement(stmt)
        else:
            self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit("}")

    def visit_WhileNode(self, node):
        condition = self.visit(node.condition)
        self.emit(f"while ({condition}) {{")

        self.indent_level += 1
        if isinstance(node.body, list):
            for stmt in node.body:
                self.emit_statement(stmt)
        else:
            self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit("}")

    def visit_DoWhileNode(self, node):
        condition = self.visit(node.condition)
        self.emit("do {")

        self.indent_level += 1
        if isinstance(node.body, list):
            for stmt in node.body:
                self.emit_statement(stmt)
        else:
            self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit(f"}} while ({condition});")

    def visit_SwitchNode(self, node):
        expression = self.visit(node.expression)
        self.emit(f"switch ({expression}) {{")

        self.indent_level += 1
        ordered_cases = getattr(node, "ordered_cases", None)
        if ordered_cases is not None:
            for case in ordered_cases:
                if case.value is None:
                    self.emit("default:")
                    self.indent_level += 1
                    for stmt in case.body:
                        self.emit_statement(stmt)
                    self.indent_level -= 1
                else:
                    self.visit(case)
        else:
            for case in node.cases:
                self.visit(case)

            if node.default_case is not None:
                self.emit("default:")
                self.indent_level += 1
                for stmt in node.default_case:
                    self.emit_statement(stmt)
                self.indent_level -= 1

        self.indent_level -= 1
        self.emit("}")

    def visit_CaseNode(self, node):
        value = self.visit(node.value)
        self.emit(f"case {value}:")

        self.indent_level += 1
        for stmt in node.body:
            self.emit_statement(stmt)
        self.indent_level -= 1

    def visit_ReturnNode(self, node):
        if node.value:
            runtime_expression = self.format_cuda_runtime_expression(node.value)
            if runtime_expression is not None:
                comments, value = runtime_expression
                for comment in comments:
                    self.emit(comment)
                self.emit(f"return {value};")
                return

            if isinstance(node.value, InitializerListNode):
                elements = getattr(node.value, "elements", None) or []
                fallback = self.visit(elements[0]) if len(elements) == 1 else "0"
                self.emit(f"return {fallback} /* CUDA initializer list return */;")
                return

            value = self.visit(node.value)
            self.emit(f"return {value};")
        else:
            self.emit("return;")

    def visit_BreakNode(self, node):
        self.emit("break;")

    def visit_ContinueNode(self, node):
        self.emit("continue;")

    def visit_TernaryOpNode(self, node):
        condition = grouped_operand(self, node.condition)
        true_expr = grouped_operand(self, node.true_expr)
        false_expr = grouped_operand(self, node.false_expr)
        return f"({condition} ? {true_expr} : {false_expr})"

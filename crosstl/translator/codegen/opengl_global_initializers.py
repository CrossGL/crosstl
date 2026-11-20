"""Compile-time globals in the GLSL generator: global initializers, stage IO layouts, declarations, interface variables and type names."""

from ...glsl_builtins import GLSL_BUILTIN_INT_LIMITS
from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    ArrayType,
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    MeshOpNode,
    PointerType,
    RayQueryOpNode,
    RayTracingOpNode,
    ReferenceType,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)
from ..float_limit_constants import render_float_limit_constant
from ..refusals import OpenGLRefusal
from ..standard_constants import render_standard_math_constant
from .array_utils import (
    evaluate_literal_int_expression,
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .fragment_output_targets import frag_data_attachment_index
from .glsl_exact_lookups import glsl_unmapped_identifier
from .glsl_invariant_builtins import (
    has_invariant_attribute,
    invariant_builtin_redeclaration,
)
from .glsl_tessellation_input_arrays import implicit_tessellation_input_type
from .opengl_values import (
    OpenGLCompileTimeGlobalError,
    OpenGLGlobalInitializerError,
)
from .stage_utils import (
    collect_stage_entry_records,
    normalize_stage_name,
)


class OpenGLGlobalInitializerMixin:
    """Global initializer members of ``GLSLCodeGen``."""

    def prepare_glsl_compile_time_globals(self, ast, global_vars):
        specialization_dependent_names = set(self.glsl_specialization_constant_names)
        known_constant_names = {
            self.glsl_specialization_constant_name(node)
            for node in getattr(ast, "constants", []) or []
            if self.glsl_specialization_constant_name(node)
        }
        known_constant_names.update(self.glsl_specialization_constant_names)
        known_constant_names.update(getattr(self, "enum_variant_constants", {}))
        known_constant_names.update(GLSL_BUILTIN_INT_LIMITS)
        known_constant_names.add("gl_WorkGroupSize")
        self.collect_glsl_specialization_dependent_constants(
            getattr(ast, "constants", []) or [], specialization_dependent_names
        )
        candidate_names = {
            self.resource_node_name(node)
            for node in global_vars or []
            if self.is_glsl_compile_time_global_candidate(node)
        }
        candidate_names.discard(None)
        literal_int_constants = dict(self.literal_int_constants)

        for node in global_vars or []:
            if not self.is_glsl_compile_time_global_candidate(node):
                continue
            name = self.resource_node_name(node, "<unnamed>")
            if not self.is_glsl_compile_time_value_type(
                self.resource_node_type(node), int_constants=literal_int_constants
            ):
                type_name = self.type_name_string(self.resource_node_type(node))
                raise OpenGLCompileTimeGlobalError(
                    f"OpenGL compile-time global '{name}' cannot use value type "
                    f"'{type_name}'",
                    variable_name=name,
                    expression_kind=self.resource_node_type(node).__class__.__name__,
                    detail=type_name,
                    reason="unsupported-value-type",
                    source_location=getattr(node, "source_location", None),
                )
            specialization_dependent, bitcast_dependent = (
                self.validate_glsl_compile_time_global_expression(
                    node,
                    getattr(node, "initial_value", None),
                    known_constant_names,
                    candidate_names,
                    specialization_dependent_names,
                )
            )
            if specialization_dependent and bitcast_dependent:
                self.opengl_compile_time_global_error(
                    node,
                    getattr(node, "initial_value", None),
                    reason="specialization-bitcast-expression",
                    detail="bitcast depends on an unresolved specialization value",
                )
            self.glsl_compile_time_global_node_ids.add(id(node))
            known_constant_names.add(name)
            if specialization_dependent:
                specialization_dependent_names.add(name)
            if bitcast_dependent:
                self.glsl_compile_time_bitcast_substitution_node_ids.add(id(node))
                self.glsl_compile_time_bitcast_substitutions[name] = (
                    getattr(node, "initial_value", None),
                    self.resource_node_type(node),
                )

            literal_int_value = self.glsl_compile_time_global_integer_value(
                node,
                literal_int_constants,
            )
            if literal_int_value is not None:
                literal_int_constants[name] = literal_int_value
                self.literal_int_constants[name] = literal_int_value
                self.current_compile_time_int_constants[name] = literal_int_value

    def collect_glsl_specialization_dependent_constants(self, constants, dependent):
        pending = list(constants or [])
        changed = True
        while changed:
            changed = False
            remaining = []
            for node in pending:
                name = self.glsl_specialization_constant_name(node)
                if not name or name in dependent:
                    continue
                value = self.glsl_specialization_constant_value(node)
                references = {
                    expression.name
                    for expression in self.walk_ast(value)
                    if isinstance(expression, IdentifierNode)
                }
                if references & dependent:
                    dependent.add(name)
                    changed = True
                else:
                    remaining.append(node)
            pending = remaining

    def glsl_compile_time_global_integer_value(self, node, constants):
        mapped_type = self.map_type(self.resource_node_type(node))
        if mapped_type not in {"int", "uint", "int64_t", "uint64_t"}:
            return None
        return evaluate_literal_int_expression(
            getattr(node, "initial_value", None), constants
        )

    def is_glsl_compile_time_global_candidate(self, node):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if "constant" not in qualifiers:
            return False
        if getattr(node, "initial_value", None) is None:
            return False
        if getattr(node, "resource_qualifiers", None):
            return False
        if qualifiers & {
            "buffer",
            "device",
            "function",
            "global",
            "groupshared",
            "in",
            "inout",
            "input",
            "local",
            "out",
            "output",
            "private",
            "readonly",
            "readwrite",
            "restrict",
            "shared",
            "storage",
            "thread",
            "threadgroup",
            "threadgroup_imageblock",
            "uniform",
            "volatile",
            "workgroup",
            "writeonly",
        }:
            return False
        if any(
            self.is_resource_binding_attribute(attr)
            or self.is_resource_memory_attribute(attr)
            or str(getattr(attr, "name", "")).lower() in {"access", "uav"}
            for attr in getattr(node, "attributes", []) or []
        ):
            return False
        if self.glsl_variable_layout_prefix(node).strip():
            return False
        return not self.is_glsl_compile_time_resource_type(
            self.resource_node_type(node)
        )

    def is_glsl_compile_time_resource_type(self, vtype, seen=None):
        if isinstance(vtype, (PointerType, ReferenceType)):
            return True
        if isinstance(vtype, ArrayType):
            return self.is_glsl_compile_time_resource_type(vtype.element_type, seen)

        type_name = self.type_name_string(vtype)
        if not type_name:
            return False
        base_type, _array_suffix = split_array_type_suffix(type_name)
        resolved_type = self.resolve_glsl_type_alias(base_type.strip())
        if resolved_type.rstrip().endswith(("*", "&")):
            return True
        if self.is_structured_buffer_type(
            resolved_type
        ) or self.is_constant_buffer_type(resolved_type):
            return True

        mapped_type = self.map_type(resolved_type)
        if self.is_opaque_resource_type(mapped_type) or self.is_ray_query_type_name(
            mapped_type
        ):
            return True

        seen = set(seen or ())
        source_mapped_type = self.glsl_source_type_identifier_name(mapped_type)
        if source_mapped_type in seen:
            return False
        struct = self.structs_by_name.get(source_mapped_type)
        if struct is None:
            return False
        seen.add(source_mapped_type)
        return any(
            self.is_glsl_compile_time_resource_type(member.member_type, seen)
            for member in getattr(struct, "members", []) or []
        )

    def is_glsl_compile_time_value_type(self, vtype, seen=None, *, int_constants=None):
        if isinstance(vtype, (PointerType, ReferenceType)):
            return False
        if isinstance(vtype, ArrayType):
            size = evaluate_literal_int_expression(
                getattr(vtype, "size", None),
                self.literal_int_constants if int_constants is None else int_constants,
            )
            return (
                size is not None
                and size > 0
                and self.is_glsl_compile_time_value_type(
                    vtype.element_type,
                    seen,
                    int_constants=int_constants,
                )
            )

        type_name = self.type_name_string(vtype)
        if not type_name or "[]" in type_name.replace(" ", ""):
            return False
        base_type = self.resource_base_type(vtype)
        if self.is_structured_buffer_type(base_type) or self.is_constant_buffer_type(
            base_type
        ):
            return False
        mapped_resource_type = self.map_resource_type_with_format(base_type)
        if self.is_opaque_resource_type(mapped_resource_type):
            return False
        if self.type_contains_opaque_resource(vtype):
            return False

        mapped_type = self.map_type(base_type)
        if self.is_opaque_resource_type(mapped_type) or self.is_ray_query_type_name(
            mapped_type
        ):
            return False
        seen = set(seen or ())
        source_mapped_type = self.glsl_source_type_identifier_name(mapped_type)
        if source_mapped_type in seen:
            return True
        struct = self.structs_by_name.get(source_mapped_type)
        if struct is None:
            return (
                self.is_scalar_value_type(mapped_type)
                or self.is_vector_value_type(mapped_type)
                or self.glsl_matrix_dimensions(mapped_type) is not None
                or mapped_type == "int"
            )
        seen.add(source_mapped_type)
        return all(
            self.is_glsl_compile_time_value_type(
                member.member_type,
                seen,
                int_constants=int_constants,
            )
            for member in getattr(struct, "members", []) or []
        )

    def validate_glsl_compile_time_global_expression(
        self,
        node,
        expression,
        known_constant_names,
        candidate_names,
        specialization_dependent_names,
    ):
        def reject(reason, current, detail=None):
            self.opengl_compile_time_global_error(
                node,
                current,
                reason=reason,
                detail=detail,
            )

        def validate(current):
            if isinstance(current, LiteralNode):
                return False, False
            if isinstance(current, (IdentifierNode, VariableNode)):
                name = getattr(current, "name", None)
                if name in known_constant_names:
                    return (
                        name in specialization_dependent_names,
                        name in self.glsl_compile_time_bitcast_substitutions,
                    )
                if name and render_standard_math_constant(name, "opengl") is not None:
                    return False, False
                if name and render_float_limit_constant(name, "opengl") is not None:
                    return False, False
                reason = (
                    "forward-reference" if name in candidate_names else "runtime-value"
                )
                reject(reason, current, name)
            if isinstance(current, ArrayLiteralNode):
                dependencies = [
                    validate(element)
                    for element in getattr(current, "elements", []) or []
                ]
                return (
                    any(item[0] for item in dependencies),
                    any(item[1] for item in dependencies),
                )
            if isinstance(current, MemberAccessNode):
                return validate(current.object)
            if isinstance(current, SwizzleNode):
                return validate(current.vector_expr)
            if isinstance(current, ArrayAccessNode):
                array_dependency = validate(current.array)
                index_dependency = validate(current.index)
                if index_dependency[0]:
                    reject("specialization-array-index", current)
                return (
                    array_dependency[0],
                    array_dependency[1] or index_dependency[1],
                )
            if isinstance(current, BinaryOpNode):
                left = validate(current.left)
                right = validate(current.right)
                return left[0] or right[0], left[1] or right[1]
            if isinstance(current, UnaryOpNode):
                if self.map_operator(current.op) in {"++", "--"}:
                    reject("mutation", current)
                return validate(current.operand)
            if isinstance(current, TernaryOpNode):
                dependencies = (
                    validate(current.condition),
                    validate(current.true_expr),
                    validate(current.false_expr),
                )
                return (
                    any(item[0] for item in dependencies),
                    any(item[1] for item in dependencies),
                )
            if isinstance(current, CastNode):
                return validate(current.expression)
            if isinstance(current, FunctionCallNode):
                function_name = self.function_call_name(current)
                if function_name in self.function_return_types:
                    reject("function-call", current, function_name)
                bitcast = self.glsl_compile_time_bitcast_contract(node, current)
                if bitcast is not None:
                    try:
                        specialization_dependent, _bitcast_dependent = validate(
                            bitcast["operand"]
                        )
                    except OpenGLCompileTimeGlobalError as error:
                        if error.reason not in {
                            "forward-reference",
                            "function-call",
                            "runtime-value",
                        }:
                            raise
                        self.opengl_compile_time_global_error(
                            node,
                            current,
                            reason=f"bitcast-operand-{error.reason}",
                            detail=error.detail,
                            operation=bitcast["operation"],
                            operand_type=bitcast["operand_type"],
                            result_type=bitcast["result_type"],
                        )
                    if specialization_dependent:
                        self.opengl_compile_time_global_error(
                            node,
                            current,
                            reason="specialization-bitcast",
                            detail=function_name,
                            operation=function_name,
                            operand_type=bitcast["operand_type"],
                            result_type=bitcast["result_type"],
                        )
                    return False, True
                arguments = (
                    getattr(current, "arguments", getattr(current, "args", [])) or []
                )
                dependencies = [validate(argument) for argument in arguments]
                specialization_dependent = any(item[0] for item in dependencies)
                bitcast_dependent = any(item[1] for item in dependencies)
                if self.glsl_constructor_type(function_name) is not None:
                    return specialization_dependent, bitcast_dependent
                if function_name not in {
                    "abs",
                    "acos",
                    "asin",
                    "ceil",
                    "clamp",
                    "cos",
                    "degrees",
                    "dot",
                    "exp",
                    "exp2",
                    "floor",
                    "inversesqrt",
                    "length",
                    "log",
                    "log2",
                    "max",
                    "min",
                    "mod",
                    "normalize",
                    "pow",
                    "radians",
                    "round",
                    "sign",
                    "sin",
                    "sqrt",
                    "trunc",
                }:
                    reject("function-call", current, function_name)
                if specialization_dependent:
                    reject("specialization-function-call", current, function_name)
                return False, bitcast_dependent
            if isinstance(current, ConstructorNode):
                dependencies = [
                    validate(argument)
                    for argument in list(
                        getattr(current, "arguments", []) or []
                    ) + list((getattr(current, "named_arguments", {}) or {}).values())
                ]
                return (
                    any(item[0] for item in dependencies),
                    any(item[1] for item in dependencies),
                )
            reject("unsupported-expression", current)

        return validate(expression)

    def glsl_empty_struct_type_name(self, vtype):
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        base_type, _array_suffix = split_array_type_suffix(type_name)
        resolved_type = self.resolve_glsl_type_alias(base_type.strip())
        mapped_type = self.map_type(resolved_type)
        candidates = (
            mapped_type,
            resolved_type,
            base_type.strip(),
            mapped_type.rsplit("::", 1)[-1],
            resolved_type.rsplit("::", 1)[-1],
        )
        for candidate in candidates:
            struct_node = self.structs_by_name.get(candidate)
            if struct_node is not None and not list(
                getattr(struct_node, "members", []) or []
            ):
                return candidate
        return None

    def collect_glsl_empty_struct_global_ids(self, global_vars):
        global_names = {
            self.resource_node_name(node)
            for node in global_vars or []
            if self.resource_node_name(node)
        }
        omitted = set()
        for node in global_vars or []:
            if self.glsl_empty_struct_type_name(self.resource_node_type(node)) is None:
                continue
            initializer = getattr(node, "initial_value", None)
            if initializer is not None and not self.glsl_side_effect_free_expression(
                initializer
            ):
                name = self.resource_node_name(node, "<unnamed>")
                dependencies = self.glsl_global_initializer_dependencies(
                    initializer, global_names
                )
                raise OpenGLGlobalInitializerError(
                    "OpenGL cannot lower empty-structure global "
                    f"'{name}' because discarding its initializer arguments may "
                    "lose side effects",
                    variable_name=name,
                    dependencies=dependencies,
                    reason="empty-structure-global-initializer-side-effects",
                    source_location=getattr(initializer, "source_location", None)
                    or getattr(node, "source_location", None),
                )
            omitted.add(id(node))
        return omitted

    def generate_glsl_empty_struct_global_declaration(
        self, node, declaration, expected_type
    ):
        initializer = getattr(node, "initial_value", None)
        if initializer is None:
            return f"{declaration};\n"
        rendered_initializer = self.generate_expression_with_expected(
            initializer, expected_type
        )
        return f"{declaration} = {rendered_initializer};\n"

    def prepare_glsl_runtime_global_initializers(
        self, ast, global_vars, functions, target_stage
    ):
        initialized_nodes = {
            self.resource_node_name(node): node
            for node in global_vars or []
            if self.resource_node_name(node)
            and getattr(node, "initial_value", None) is not None
        }
        if not initialized_nodes:
            return

        global_nodes = {
            self.resource_node_name(node): node
            for node in global_vars or []
            if self.resource_node_name(node)
        }
        global_names = set(global_nodes)
        dependencies = {
            name: self.glsl_global_initializer_dependencies(
                getattr(node, "initial_value", None), global_names
            )
            for name, node in initialized_nodes.items()
        }
        ordered_names = self.order_glsl_global_initializers(
            initialized_nodes, dependencies
        )
        runtime_names = {
            name
            for name, node in global_nodes.items()
            if self.global_variable_qualifier(node).strip() != "const"
        }
        runtime_initializers = {
            name: initialized_nodes[name]
            for name in ordered_names
            if self.global_variable_qualifier(initialized_nodes[name]).strip()
            == "uniform"
            and dependencies[name] & runtime_names
        }
        if not runtime_initializers:
            return

        for name, node in runtime_initializers.items():
            self.validate_glsl_runtime_global_initializer(
                name,
                node,
                dependencies[name],
            )

        self.glsl_runtime_global_initializer_node_ids = {
            id(node) for node in runtime_initializers.values()
        }
        runtime_initializer_names = set(runtime_initializers)
        functions_by_name = {}
        for function in functions or []:
            name = getattr(function, "name", None)
            if name:
                functions_by_name.setdefault(name, []).append(function)

        direct_references = {
            id(function): (
                self.glsl_function_free_identifier_names(function)
                & runtime_initializer_names
            )
            for function in functions or []
        }
        direct_calls = {
            id(function): self.glsl_called_function_names(function)
            for function in functions or []
        }

        def reachable_initializers(function, visited=None):
            visited = set(visited or ())
            function_id = id(function)
            if function_id in visited:
                return set()
            visited.add(function_id)
            required = set(direct_references.get(function_id, set()))
            for called_name in direct_calls.get(function_id, set()):
                for called_function in functions_by_name.get(called_name, []):
                    required.update(
                        reachable_initializers(called_function, set(visited))
                    )
            return required

        stage_entries = collect_stage_entry_records(
            ast,
            target_stage,
            self.combined_stage_entry_types(),
        )
        for _entry_id, _stage_name, entry_function in stage_entries:
            required = reachable_initializers(entry_function)
            pending = list(required)
            while pending:
                name = pending.pop()
                for dependency in dependencies.get(name, set()):
                    if (
                        dependency in runtime_initializer_names
                        and dependency not in required
                    ):
                        required.add(dependency)
                        pending.append(dependency)
            self.glsl_stage_runtime_global_initializers[id(entry_function)] = [
                runtime_initializers[name] for name in ordered_names if name in required
            ]

    def glsl_global_initializer_dependencies(self, expression, global_names):
        return {
            node.name
            for node in self.walk_ast(expression)
            if isinstance(node, IdentifierNode) and node.name in global_names
        }

    def order_glsl_global_initializers(self, initialized_nodes, dependencies):
        state = {}
        stack = []
        ordered = []
        source_order = {name: index for index, name in enumerate(initialized_nodes)}

        def visit(name):
            current_state = state.get(name, 0)
            if current_state == 2:
                return
            if current_state == 1:
                start = stack.index(name)
                cycle = stack[start:] + [name]
                raise OpenGLGlobalInitializerError(
                    "OpenGL global initializer dependency cycle: " + " -> ".join(cycle),
                    variable_name=name,
                    dependencies=dependencies.get(name, set()),
                    cycle=cycle,
                    reason="dependency-cycle",
                    source_location=getattr(
                        initialized_nodes.get(name), "source_location", None
                    ),
                )

            state[name] = 1
            stack.append(name)
            for dependency in sorted(
                dependencies.get(name, set()),
                key=lambda item: (source_order.get(item, len(source_order)), item),
            ):
                if dependency in initialized_nodes:
                    visit(dependency)
            stack.pop()
            state[name] = 2
            ordered.append(name)

        for name in initialized_nodes:
            visit(name)
        return ordered

    def validate_glsl_runtime_global_initializer(self, name, node, dependencies):
        initializer = getattr(node, "initial_value", None)
        for expression in self.walk_ast(initializer):
            reason = None
            detail = None
            if isinstance(expression, FunctionCallNode):
                reason = "function-call"
                detail = self.function_call_name(expression) or "<expression>"
            elif isinstance(expression, AssignmentNode):
                reason = "assignment"
            elif isinstance(expression, UnaryOpNode) and self.map_operator(
                expression.op
            ) in {"++", "--"}:
                reason = "mutation"
            elif isinstance(
                expression,
                (WaveOpNode, MeshOpNode, RayQueryOpNode, RayTracingOpNode),
            ):
                reason = "stage-operation"
                detail = expression.__class__.__name__

            if reason is None:
                continue
            suffix = f" '{detail}'" if detail else ""
            raise OpenGLGlobalInitializerError(
                f"OpenGL runtime initializer for '{name}' cannot contain "
                f"{reason.replace('-', ' ')}{suffix}",
                variable_name=name,
                dependencies=dependencies,
                reason=reason,
                source_location=getattr(expression, "source_location", None)
                or getattr(node, "source_location", None),
            )

    def glsl_function_free_identifier_names(self, function):
        body = getattr(function, "body", None)
        bound_names = {
            getattr(parameter, "name", None)
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
        }
        callee_identifier_ids = set()
        for node in self.walk_ast(body):
            if isinstance(node, (VariableNode, ArrayNode)):
                bound_names.add(getattr(node, "name", None))
            elif isinstance(node, FunctionCallNode):
                function_expression = getattr(
                    node, "function", getattr(node, "name", None)
                )
                if isinstance(function_expression, IdentifierNode):
                    callee_identifier_ids.add(id(function_expression))
        bound_names.discard(None)
        return {
            node.name
            for node in self.walk_ast(body)
            if isinstance(node, IdentifierNode)
            and id(node) not in callee_identifier_ids
            and node.name not in bound_names
        }

    def glsl_called_function_names(self, function):
        return {
            name
            for node in self.walk_ast(getattr(function, "body", None))
            if isinstance(node, FunctionCallNode)
            for name in [self.function_call_name(node)]
            if name
        }

    def generate_glsl_runtime_global_initializers(self, function, indent):
        code = ""
        indent_string = "    " * indent
        for node in self.glsl_stage_runtime_global_initializers.get(id(function), []):
            source_name = self.resource_node_name(node, "")
            emitted_name = self.current_identifier_aliases.get(source_name)
            if emitted_name is None:
                emitted_name = glsl_unmapped_identifier(
                    self, source_name, "global variable"
                )
            expected_type = self.global_variable_types.get(source_name, "float")
            initializer = self.generate_expression_with_expected(
                getattr(node, "initial_value", None), expected_type
            )
            code += f"{indent_string}{emitted_name} = {initializer};\n"
        return code

    def global_stage_io_layout_direction(self, qualifier):
        parts = qualifier.split()
        if "in" in parts:
            return "input"
        if "out" in parts:
            return "output"
        return None

    def reserve_global_stage_io_layout(self, node, qualifier, layout, mapped_type):
        direction = self.global_stage_io_layout_direction(qualifier)
        if direction is None:
            return

        stage_name = (
            self.current_global_stage_io_stage or self.current_target_stage or "global"
        )
        if self.is_per_vertex_stage_io(node, qualifier, direction):
            mapped_type = self.stage_io_per_vertex_element_type(mapped_type)
        self.reserve_stage_io_layout(
            self.stage_io_used_locations,
            stage_name,
            direction,
            layout,
            mapped_type,
            self.resource_node_name(node, "<unnamed>"),
        )

    @staticmethod
    def is_pervertex_stage_io(node, qualifier):
        names = {
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(node, "attributes", None) or []
        }
        return "pervertex" in names or bool(
            set(str(qualifier).split()) & {"pervertexEXT", "pervertexNV"}
        )

    def is_per_vertex_stage_io(self, node, qualifier, direction):
        if "patch" in str(qualifier).split():
            return False
        if self.is_pervertex_stage_io(node, qualifier):
            return True
        stage = str(self.current_global_stage_io_stage or "").lower()
        return (stage, direction) in self.PER_VERTEX_STAGE_IO

    @staticmethod
    def stage_io_per_vertex_element_type(mapped_type):
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        close = array_suffix.find("]")
        return base_type + array_suffix[close + 1 :] if close >= 0 else mapped_type

    def stage_io_layout_has(self, layout, names):
        if not layout.startswith("layout("):
            return False
        start = layout.find("(")
        end = layout.rfind(")")
        if start == -1 or end == -1 or end <= start:
            return False

        for item in layout[start + 1 : end].split(","):
            key = item.partition("=")[0].strip()
            if key in names:
                return True
        return False

    def generate_global_variable_declaration(
        self, node, declaration, vtype, mapped_type_for_layout=None
    ):
        name = self.resource_node_name(node, "")
        builtin_output = self.fragment_output_variable_builtin_target(node)
        if builtin_output is not None:
            self.current_identifier_aliases[name] = builtin_output
            return ""
        if id(node) in self.glsl_compile_time_bitcast_substitution_node_ids:
            return ""

        runtime_initializer = id(node) in self.glsl_runtime_global_initializer_node_ids
        qualifier = "" if runtime_initializer else self.global_variable_qualifier(node)
        if qualifier.strip() == "const" and self.glsl_value_contains_call(
            getattr(node, "initial_value", None), "asdouble"
        ):
            # packDouble2x32 is not a constant expression in GLSL, so the
            # constant is substituted where it is used.
            self.glsl_compile_time_bitcast_substitutions[name] = (
                node.initial_value,
                vtype,
            )
            return ""
        emitted_name = self.glsl_global_declaration_identifier_name(node, name)
        if emitted_name != name:
            declaration = format_c_style_array_declaration(
                mapped_type_for_layout if mapped_type_for_layout is not None else vtype,
                emitted_name,
            )
        initializer = ""
        initial_value = getattr(node, "initial_value", None)
        if initial_value is not None and not runtime_initializer:
            initializer_type = (
                mapped_type_for_layout if mapped_type_for_layout is not None else vtype
            )
            if self.glsl_narrow_float_width(vtype) is not None:
                initializer_type = vtype
            self.glsl_generating_global_initializer = True
            try:
                initializer = " = " + self.generate_expression_with_expected(
                    initial_value, initializer_type
                )
            finally:
                self.glsl_generating_global_initializer = False
        elif (
            initial_value is None
            and qualifier == ""
            and "static" in {
                str(q).lower() for q in getattr(node, "qualifiers", []) or []
            }
        ):
            zero_value = self.glsl_static_global_zero_value(
                mapped_type_for_layout if mapped_type_for_layout is not None else vtype
            )
            if zero_value:
                initializer = f" = {zero_value}"
        layout = self.glsl_variable_layout_prefix(node)
        mapped_type = (
            mapped_type_for_layout if mapped_type_for_layout is not None else vtype
        )
        stage_name = getattr(
            self, "current_global_stage_io_stage", self.current_target_stage
        )
        if self.global_stage_io_layout_direction(
            qualifier
        ) == "input" and self.requires_flat_stage_input(stage_name, mapped_type):
            qualifier = f"{self.with_flat_stage_input_qualifier(qualifier.strip())} "
        implicit_type = implicit_tessellation_input_type(
            self, node, qualifier, mapped_type, stage_name
        )
        if implicit_type != mapped_type:
            mapped_type = implicit_type
            declaration = format_c_style_array_declaration(mapped_type, emitted_name)
        self.reserve_global_stage_io_layout(
            node,
            qualifier,
            layout,
            mapped_type,
        )
        return f"{layout}{qualifier}{declaration}{initializer};\n"

    def glsl_compile_time_bitcast_substitution_expression(self, name):
        substitution = self.glsl_compile_time_bitcast_substitutions.get(name)
        if substitution is None or name in self.local_variable_types:
            return None
        if name in self.glsl_active_compile_time_bitcast_substitutions:
            raise OpenGLCompileTimeGlobalError(
                f"OpenGL compile-time bitcast substitution for '{name}' is cyclic",
                variable_name=name,
                expression_kind="IdentifierNode",
                detail=name,
                reason="bitcast-substitution-cycle",
            )

        expression, expected_type = substitution
        self.glsl_active_compile_time_bitcast_substitutions.add(name)
        try:
            rendered = self.generate_expression_with_expected(expression, expected_type)
        finally:
            self.glsl_active_compile_time_bitcast_substitutions.remove(name)
        return f"({rendered})"

    def generate_stage_local_interface_variable_declaration(self, node):
        builtin_name = self.glsl_builtin_stage_interface_name(node)
        if builtin_name is not None:
            source_name = self.resource_node_name(node, "")
            if source_name:
                if source_name != builtin_name:
                    self.current_identifier_aliases[source_name] = builtin_name
                else:
                    self.current_identifier_aliases.pop(source_name, None)
            if has_invariant_attribute(node):
                return invariant_builtin_redeclaration(
                    self,
                    builtin_name,
                    getattr(
                        self, "current_global_stage_io_stage", self.current_target_stage
                    ),
                    source_name or builtin_name,
                )
            return ""

        vtype, _, array_suffix, _ = self.resource_declaration_shape(node)
        mapped_type = f"{self.map_type(vtype)}{array_suffix}"
        source_name = self.resource_node_name(node, "")
        emitted_name = self.glsl_global_identifier_name(source_name)
        direction = self.stage_local_interface_direction(node)
        stage_name = getattr(
            self, "current_global_stage_io_stage", self.current_target_stage
        )
        linked_name = (
            self.stage_local_interface_linked_name(node, direction, stage_name)
            if direction is not None
            else None
        )
        if linked_name is not None and linked_name != emitted_name:
            emitted_name = linked_name
            self.current_identifier_aliases[source_name] = linked_name
        if direction is not None:
            prefix = self.stage_io_declaration_prefix(
                node, direction, stage_name=stage_name
            )
            mapped_type = implicit_tessellation_input_type(
                self, node, prefix, mapped_type, stage_name
            )
        declaration = format_c_style_array_declaration(mapped_type, emitted_name)
        if direction is not None:
            if direction == "in" and self.requires_flat_stage_input(
                stage_name, mapped_type
            ):
                prefix = self.with_flat_stage_io_declaration_prefix(prefix)
            elif (
                direction == "out"
                and normalize_stage_name(stage_name) == "vertex"
                and self.requires_flat_stage_io(mapped_type)
            ):
                # An integer cannot be interpolated; GLSL ES 3.00 section 4.3.6 requires a vertex output that is or
                # contains one to be flat, as the fragment input it feeds is, and stage output structs emit so.
                prefix = self.with_flat_stage_io_declaration_prefix(prefix)
            layout = self.glsl_variable_layout_prefix(node)
            self.reserve_global_stage_io_layout(node, f"{prefix} ", layout, mapped_type)
            return f"{prefix} {declaration};\n"

        return self.generate_global_variable_declaration(
            node, declaration, vtype, mapped_type_for_layout=mapped_type
        )

    def stage_local_interface_linked_name(self, node, direction, stage_name):
        """The name a stage-level ``in`` or ``out`` must carry to link with its neighbour stage, or ``None``."""
        return None

    def stage_local_interface_direction(self, node):
        names = {
            str(qualifier).lower().replace("-", "_")
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name:
                names.add(str(attr_name).lower().replace("-", "_"))

        if names & {"input", "in"}:
            return "in"
        if names & {"output", "out"}:
            return "out"
        if "inout" in names:
            return "inout"
        return None

    def with_flat_stage_io_declaration_prefix(self, prefix):
        parts = prefix.split()
        if "flat" in parts:
            return prefix
        for direction in ("in", "out", "inout"):
            marker = f" {direction}"
            if prefix.endswith(marker):
                return f"{prefix[: -len(marker)]} flat{marker}"
            marker = f" {direction} "
            if marker in prefix:
                return prefix.replace(marker, f" flat{marker}", 1)
        return f"flat {prefix}" if prefix else "flat"

    def glsl_builtin_stage_interface_name(self, node):
        mapped_semantic = self.map_semantic(self.semantic_from_node(node))
        if not mapped_semantic or mapped_semantic.startswith("layout("):
            return None
        if not mapped_semantic.startswith("gl_"):
            return None
        if self.glsl_variable_layout_prefix(node).strip():
            return None
        return mapped_semantic

    def fragment_output_variable_builtin_target(self, node):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if "out" not in qualifiers:
            return None
        if self.map_semantic(self.semantic_from_node(node)) == "gl_SampleMask":
            return "gl_SampleMask[0]"
        return None

    def glsl_variable_qualifier_prefix(self, node):
        qualifiers = [
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        ]
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name:
                qualifiers.append(str(attr_name).lower())

        emitted = []

        def add(qualifier):
            if qualifier not in emitted:
                emitted.append(qualifier)

        for qualifier in qualifiers:
            normalized = (
                qualifier[len("glsl_") :]
                if qualifier.startswith("glsl_")
                else qualifier
            )
            normalized = normalized.replace("-", "_")
            if normalized in {"perprimitive", "perprimitiveext"}:
                add("perprimitiveEXT")
            elif normalized in {"pervertex", "pervertexext"}:
                add("pervertexEXT")
            elif normalized in {
                "task_payload_shared",
                "taskpayloadshared",
                "taskpayloadsharedext",
            }:
                add("taskPayloadSharedEXT")
            elif normalized in {"raypayload", "raypayloadext"}:
                add("rayPayloadEXT")
            elif normalized in {"raypayloadin", "raypayloadinext"}:
                add("rayPayloadInEXT")
            elif normalized in {"hitattribute", "hitattributeext"}:
                add("hitAttributeEXT")
            elif normalized in {"callabledata", "callabledataext"}:
                add("callableDataEXT")
            elif normalized in {"callabledatain", "callabledatainext"}:
                add("callableDataInEXT")
            elif normalized in {"tileimage", "tileimageext"}:
                add("tileImageEXT")
            elif normalized in {"shared", "groupshared", "workgroup", "threadgroup"}:
                add("shared")
            elif normalized in {
                "invariant",
                "precise",
                "patch",
                "flat",
                "smooth",
                "noperspective",
                "centroid",
                "sample",
                "in",
                "out",
                "lowp",
                "mediump",
                "highp",
            }:
                add(normalized)

        if not emitted:
            return ""

        order = {
            "invariant": 0,
            "precise": 1,
            "patch": 2,
            "perprimitiveEXT": 3,
            "pervertexEXT": 3,
            "flat": 4,
            "smooth": 5,
            "noperspective": 6,
            "centroid": 7,
            "sample": 8,
            "in": 9,
            "out": 10,
            "tileImageEXT": 11,
            "lowp": 12,
            "mediump": 13,
            "highp": 14,
            "shared": 15,
            "taskPayloadSharedEXT": 16,
            "rayPayloadEXT": 17,
            "rayPayloadInEXT": 18,
            "hitAttributeEXT": 19,
            "callableDataEXT": 20,
            "callableDataInEXT": 21,
        }
        emitted.sort(key=lambda qualifier: order.get(qualifier, len(order)))
        return " ".join(emitted)

    def glsl_variable_layout_prefix(self, node):
        layout_parts = []
        seen_layout_parts = {}
        node_name = self.resource_node_name(node, "<unnamed>")
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).lower()
            if normalized.startswith("glsl_"):
                normalized = normalized[len("glsl_") :]
            normalized = normalized.replace("-", "_")
            if normalized in self.GLSL_BLEND_SUPPORT_LAYOUT_NAMES:
                continue
            if normalized in self.GLSL_BARE_LAYOUT_ATTRIBUTE_NAMES:
                arguments = getattr(attr, "arguments", []) or []
                if arguments:
                    continue
                previous = seen_layout_parts.get(normalized)
                if previous is not None:
                    raise OpenGLRefusal(
                        "Duplicate OpenGL layout metadata for "
                        f"'{node_name}': {normalized}"
                    )
                seen_layout_parts[normalized] = normalized
                layout_parts.append(normalized)
                continue
            if normalized not in self.GLSL_LAYOUT_ATTRIBUTE_NAMES:
                continue
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) != 1:
                continue
            layout_part = (
                f"{normalized} = {self.attribute_value_to_string(arguments[0])}"
            )
            previous = seen_layout_parts.get(normalized)
            if previous == layout_part:
                raise OpenGLRefusal(
                    f"Duplicate OpenGL layout metadata for '{node_name}': {layout_part}"
                )
            if previous is not None:
                raise OpenGLRefusal(
                    "Conflicting OpenGL layout metadata for "
                    f"'{node_name}': {previous} differs from {layout_part}"
                )
            seen_layout_parts[normalized] = layout_part
            layout_parts.append(layout_part)
        if not layout_parts:
            return ""
        return f"layout({', '.join(layout_parts)}) "

    def is_glsl_stage_io_metadata_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False

        normalized = str(attr_name).lower()
        if normalized.startswith("glsl_"):
            normalized = normalized[len("glsl_") :]
        normalized = normalized.replace("-", "_")

        return (
            normalized in self.GLSL_LAYOUT_ATTRIBUTE_NAMES
            or normalized in self.GLSL_BARE_LAYOUT_ATTRIBUTE_NAMES
            or normalized in self.GLSL_VARIABLE_QUALIFIER_ATTRIBUTE_NAMES
            or normalized in self.GLSL_STORAGE_QUALIFIER_ATTRIBUTE_NAMES
            or normalized in {
                "input",
                "output",
                "interface_block",
                "interface_instance",
                "interface_array",
            }
        )

    def stage_io_layout_for_node(self, node):
        explicit_layout = self.glsl_variable_layout_prefix(node).strip()
        if explicit_layout:
            return explicit_layout
        attachment = frag_data_attachment_index(node)
        if attachment is not None:
            return f"layout(location = {attachment})"
        return self.map_semantic(self.semantic_from_node(node))

    def stage_io_declaration_prefix(
        self, node, direction, layout=None, stage_name=None
    ):
        parts = []
        if layout is None:
            layout = self.stage_io_layout_for_node(node)
        if layout.startswith("layout(") and self.should_emit_stage_io_layout(
            stage_name, direction
        ):
            parts.append(layout)

        qualifiers = [
            qualifier
            for qualifier in self.glsl_variable_qualifier_prefix(node).split()
            if qualifier not in {"in", "out", "inout"}
        ]
        precision = [
            qualifier
            for qualifier in qualifiers
            if qualifier in self.GLSL_PRECISION_QUALIFIERS
        ]
        other_qualifiers = [
            qualifier
            for qualifier in qualifiers
            if qualifier not in self.GLSL_PRECISION_QUALIFIERS
        ]
        parts.extend(other_qualifiers)
        parts.append(direction)
        parts.extend(precision)
        return " ".join(parts)

    def should_emit_stage_io_layout(self, stage_name, direction):
        return True

    def requires_flat_stage_input(self, stage_name, mapped_type):
        if stage_name != "fragment":
            return False
        base_type, _ = split_array_type_suffix(str(mapped_type))
        return base_type in {
            "int",
            "uint",
            "bool",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "bvec2",
            "bvec3",
            "bvec4",
        }

    def with_flat_stage_input_qualifier(self, prefix):
        parts = prefix.split()
        if "flat" in parts:
            return prefix
        try:
            direction_index = parts.index("in")
        except ValueError:
            return f"flat {prefix}" if prefix else "flat"
        parts.insert(direction_index, "flat")
        return " ".join(parts)

    def type_name_string(self, vtype):
        if vtype is None:
            return None
        if (
            isinstance(vtype, (PointerType, ReferenceType))
            or hasattr(vtype, "name")
            or hasattr(vtype, "element_type")
        ):
            return self.convert_type_node_to_string(vtype)
        return str(vtype)

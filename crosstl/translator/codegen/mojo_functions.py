"""Function and entry-point emission in the Mojo generator: entry names, GPU built-in parameters and mutated-parameter analysis."""

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    IdentifierNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
)
from .mojo_tables import (
    MOJO_GPU_BUILTIN_ALIASES,
    MOJO_GPU_BUILTIN_PARAMETER_ORDER,
    MOJO_MESH_STAGE_TYPES,
    MOJO_RAY_STAGE_TYPES,
    MOJO_SUPPORTED_STAGE_TYPES,
)
from .stage_utils import is_fragment_output_parameter


class MojoFunctionMixin:
    """Function members of ``MojoCodeGen``."""

    def generate_function(self, func, indent=0, shader_type=None, stage_node=None):
        """Render one CrossGL function or shader entry point as Mojo code."""
        code = ""
        "    " * indent
        previous_variable_types = self.variable_types.copy()
        previous_resource_access_qualifiers = self.resource_access_qualifiers.copy()
        previous_resource_access_qualifier_aliases = {
            name: list(candidates)
            for name, candidates in self.resource_access_qualifier_aliases.items()
        }
        previous_resource_access_path_qualifier_aliases = {
            path: list(candidates)
            for path, candidates in self.resource_access_path_qualifier_aliases.items()
        }
        previous_return_type = self.current_return_type
        previous_shader_type = self.current_shader_type
        previous_generic_function_substitutions = (
            self.current_generic_function_substitutions
        )
        self.current_shader_type = shader_type
        self.current_generic_function_substitutions = (
            getattr(func, "_generic_substitutions", {}) or {}
        )

        param_list = list(getattr(func, "parameters", getattr(func, "params", [])))
        param_infos = []
        for p in param_list:
            param_type = self.function_parameter_type_name(p)
            param_infos.append((p, param_type))
            self.register_variable_type(getattr(p, "name", None), param_type)

        extra_param_infos = [
            (name, type_name)
            for name, type_name in self.stage_local_function_extra_params.get(
                getattr(func, "name", None), []
            )
            if name not in {getattr(param, "name", None) for param, _ in param_infos}
        ]
        for name, type_name in extra_param_infos:
            self.register_variable_type(name, type_name)

        if shader_type is not None:
            # An entry point is the Mojo kernel: it takes every buffer and uniform value as an argument.
            known_names = {getattr(param, "name", None) for param, _ in param_infos}
            known_names.update(name for name, _ in extra_param_infos)
            extra_param_infos += [
                (name, type_name)
                for name, type_name in self.mojo_entry_kernel_arguments()
                if name not in known_names
            ]
            for name, type_name in extra_param_infos:
                self.register_variable_type(name, type_name)
        # Mojo reads the GPU indices anywhere in device code, so they are locals, not kernel parameters.
        synthetic_compute_builtin_params = []

        if hasattr(func, "return_type"):
            return_type = self.convert_type_node_to_string(func.return_type)
        else:
            return_type = "void"
        return_semantic = self.function_return_semantic(func)
        fragment_output_info = None
        if self.type_name(return_type) == "void" and return_semantic is None:
            for p, param_type in param_infos:
                semantic = self.node_semantic(p)
                if is_fragment_output_parameter(shader_type, p, semantic):
                    fragment_output_info = (p, param_type, semantic)
                    return_type = param_type
                    return_semantic = semantic
                    break
        if fragment_output_info is not None:
            output_param = fragment_output_info[0]
            param_infos = [
                param_info
                for param_info in param_infos
                if param_info[0] is not output_param
            ]

        if shader_type == "geometry":
            self.validate_geometry_stage(func, param_infos)
        if shader_type in {"tessellation_control", "tessellation_evaluation"}:
            self.validate_tessellation_stage(func, shader_type, param_infos)
        if shader_type in MOJO_MESH_STAGE_TYPES:
            self.validate_mesh_stage(func, shader_type, param_infos)

        param_names = {p.name for p, _ in param_infos if hasattr(p, "name")}
        param_names.update(name for name, _ in extra_param_infos)
        param_names.update(name for name, _ in synthetic_compute_builtin_params)
        mutated_params = self.collect_mutated_parameters(
            getattr(func, "body", []), param_names
        )
        params = []
        local_alias_declarations = []
        emitted_param_names = set()
        parameter_semantic_comments = []
        used_parameter_semantics = {}
        for p, param_type in param_infos:
            semantic = self.node_semantic(p)
            self.validate_parameter_semantic(
                shader_type,
                p.name,
                param_type,
                semantic,
                used_parameter_semantics,
            )

            dispatch_thread_id_bridge = (
                self.mojo_compute_global_invocation_id_parameter_bridge(
                    p,
                    param_type,
                    semantic,
                    emitted_param_names,
                    param_names,
                )
                if shader_type == "compute"
                else None
            )
            if dispatch_thread_id_bridge is not None:
                param_name, native_type, local_alias = dispatch_thread_id_bridge
                self.register_variable_type(param_name, native_type)
                self.register_variable_type(p.name, param_type)
                if semantic:
                    parameter_semantic_comments.append(
                        self.generate_parameter_semantic_comment(
                            shader_type, p.name, semantic
                        )
                    )
                local_alias_declarations.append(
                    self.mojo_gpu_index_local(param_name, native_type, "global_idx")
                )
                local_alias_declarations.append(local_alias)
                emitted_param_names.add(param_name)
                continue
            index_local = (
                self.mojo_compute_entry_parameter_local(p.name, param_type, semantic)
                if shader_type == "compute"
                else None
            )
            if index_local is not None:
                self.register_variable_type(p.name, param_type)
                if semantic:
                    parameter_semantic_comments.append(
                        self.generate_parameter_semantic_comment(
                            shader_type, p.name, semantic
                        )
                    )
                local_alias_declarations.append(index_local)
                emitted_param_names.add(p.name)
                continue

            self.register_variable_type(p.name, param_type)
            self.register_resource_access_metadata(p, param_type)
            if semantic:
                parameter_semantic_comments.append(
                    self.generate_parameter_semantic_comment(
                        shader_type, p.name, semantic
                    )
                )
            ownership = "var " if p.name in mutated_params else ""
            params.append(
                f"{ownership}{self.mojo_identifier(p.name)}: {self.map_type(param_type)}"
            )
        for name, param_type in extra_param_infos:
            # Storing through a pointer argument does not change the argument itself.
            ownership = (
                "var "
                if name in mutated_params
                and self.parse_pointer_type_name(str(param_type)) is None
                else ""
            )
            params.append(
                f"{ownership}{self.mojo_identifier(name)}: {self.map_type(param_type)}"
            )
        for name, param_type in synthetic_compute_builtin_params:
            params.append(f"{self.mojo_identifier(name)}: {self.map_type(param_type)}")

        params_str = ", ".join(params) if params else ""

        function_name = self.shader_entry_function_name(
            func,
            shader_type,
            [*param_infos, *extra_param_infos, *synthetic_compute_builtin_params],
            return_type,
        )
        self.function_return_types[func.name] = return_type
        if function_name != func.name:
            self.function_return_types[function_name] = return_type
        self.current_return_type = return_type
        if return_semantic:
            self.validate_return_semantic(shader_type, return_type, return_semantic)
        self.validate_struct_return_semantics(shader_type, return_type)

        if shader_type in MOJO_SUPPORTED_STAGE_TYPES:
            code += f"# CrossGL shader stage: {shader_type}\n"
        if shader_type == "geometry":
            code += self.generate_geometry_stage_comments(func, param_infos)
        if shader_type in {"tessellation_control", "tessellation_evaluation"}:
            code += self.generate_tessellation_stage_comments(
                func, shader_type, param_infos
            )
        if shader_type in MOJO_MESH_STAGE_TYPES:
            code += self.generate_mesh_stage_comments(
                func, shader_type, param_infos, stage_node
            )
        for comment in parameter_semantic_comments:
            code += comment
        if return_semantic:
            code += self.generate_return_semantic_comment(shader_type, return_semantic)

        code += (
            f"def {self.mojo_identifier(function_name)}({params_str}) -> "
            f"{self.map_type(return_type)}:\n"
        )
        if stage_node is not None:
            code += self.generate_function_local_shared_declarations(
                stage_node, indent + 1
            )
        if fragment_output_info is not None:
            output_param, output_type, _semantic = fragment_output_info
            output_name = getattr(output_param, "name", None)
            self.register_variable_type(output_name, output_type)
            code += (
                f"{'    ' * (indent + 1)}var {self.mojo_identifier(output_name)}: "
                f"{self.map_type(output_type)}\n"
            )
        for declaration in local_alias_declarations:
            code += "    " * (indent + 1) + declaration + "\n"
        code += self.mojo_function_local_builtin_declarations(
            getattr(func, "body", []), indent + 1
        )

        body = getattr(func, "body", [])
        statements = None
        if hasattr(body, "statements"):
            statements = body.statements
            for stmt in body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(body, list):
            statements = body
            for stmt in body:
                code += self.generate_statement(stmt, indent + 1)
        else:
            code += "    pass\n"

        if statements is not None and not statements:
            code += "    pass\n"
        if (
            fragment_output_info is not None
            and statements is not None
            and not self.statement_body_terminates_inner_loop(statements)
        ):
            output_param = fragment_output_info[0]
            code += (
                f"{'    ' * (indent + 1)}return "
                f"{self.mojo_identifier(getattr(output_param, 'name', None))}\n"
            )

        code += "\n"
        self.variable_types = previous_variable_types
        self.resource_access_qualifiers = previous_resource_access_qualifiers
        self.resource_access_qualifier_aliases = (
            previous_resource_access_qualifier_aliases
        )
        self.resource_access_path_qualifier_aliases = (
            previous_resource_access_path_qualifier_aliases
        )
        self.current_return_type = previous_return_type
        self.current_shader_type = previous_shader_type
        self.current_generic_function_substitutions = (
            previous_generic_function_substitutions
        )
        return code

    def shader_entry_function_name(
        self, func, shader_type, param_infos=None, return_type=None
    ):
        name = getattr(func, "name", "main")
        if name != "main":
            return name
        if shader_type in {"vertex", "fragment"}:
            return f"{shader_type}_main"
        if shader_type == "compute" and not self.compute_entry_can_remain_main(
            param_infos, return_type
        ):
            return "compute_main"
        if shader_type in MOJO_RAY_STAGE_TYPES:
            return f"{shader_type}_main"
        if shader_type == "geometry":
            return "geometry_main"
        if shader_type in {"tessellation_control", "tessellation_evaluation"}:
            return f"{shader_type}_main"
        if shader_type in MOJO_MESH_STAGE_TYPES:
            return f"{shader_type}_main"
        return name

    def compute_entry_can_remain_main(self, param_infos, return_type):
        # Mojo runs a module's main on the host as the program's entry point, so a GPU kernel never takes the name.
        return False

    def compute_gpu_builtin_parameter_infos(self, body, existing_names):
        required_builtins = self.collect_compute_gpu_builtin_aliases(body)
        if not required_builtins:
            return []

        existing_names = {name for name in existing_names if isinstance(name, str)}
        ordered = []
        seen_aliases = set()
        for alias_name in MOJO_GPU_BUILTIN_PARAMETER_ORDER:
            for candidate_alias, type_name in sorted(required_builtins):
                if candidate_alias != alias_name or candidate_alias in seen_aliases:
                    continue
                seen_aliases.add(candidate_alias)
                if candidate_alias not in existing_names:
                    ordered.append((candidate_alias, type_name))
        for alias_name, type_name in sorted(required_builtins):
            if alias_name in seen_aliases:
                continue
            seen_aliases.add(alias_name)
            if alias_name not in existing_names:
                ordered.append((alias_name, type_name))
        return ordered

    def collect_compute_gpu_builtin_aliases(self, node, required=None, visited=None):
        if required is None:
            required = set()
        if visited is None:
            visited = set()
        if node is None or isinstance(node, (int, float, bool)):
            return required
        if isinstance(node, str):
            builtin = MOJO_GPU_BUILTIN_ALIASES.get(node)
            if builtin is not None:
                required.add(builtin)
            return required
        if isinstance(node, (list, tuple, set)):
            for item in node:
                self.collect_compute_gpu_builtin_aliases(item, required, visited)
            return required

        node_id = id(node)
        if node_id in visited:
            return required
        visited.add(node_id)

        builtin_name = getattr(node, "builtin_name", None)
        if builtin_name is None and isinstance(node, IdentifierNode):
            builtin_name = getattr(node, "name", None)
        builtin = MOJO_GPU_BUILTIN_ALIASES.get(builtin_name)
        if builtin is not None:
            required.add(builtin)

        for attr_name in (
            "statements",
            "body",
            "expression",
            "initial_value",
            "object",
            "object_expr",
            "vector_expr",
            "array",
            "index",
            "left",
            "right",
            "condition",
            "true_expr",
            "false_expr",
            "function",
            "args",
            "arguments",
            "then_branch",
            "else_branch",
            "if_body",
            "else_body",
        ):
            if not hasattr(node, attr_name):
                continue
            self.collect_compute_gpu_builtin_aliases(
                getattr(node, attr_name), required, visited
            )
        return required

    def mojo_compute_global_invocation_id_parameter_bridge(
        self, parameter, param_type, semantic, emitted_param_names, parameter_names
    ):
        if semantic is None:
            return None
        if self.normalized_semantic_key(semantic) != "global_invocation_id":
            return None

        mapped_type = self.map_type(param_type)
        if mapped_type not in {"UInt", "UInt32", "SIMD[DType.uint32, 2]"}:
            return None

        source_name = getattr(parameter, "name", None)
        if not isinstance(source_name, str) or not source_name:
            return None

        local_name = self.mojo_identifier(source_name)
        used_names = set(emitted_param_names)
        used_names.update(parameter_names)
        used_names.update(self.variable_types)
        native_name = self.mojo_unique_local_identifier(
            f"{local_name}_globalInvocationID", used_names
        )
        if mapped_type in {"UInt", "UInt32"}:
            alias_value = f"{native_name}[0]"
        else:
            alias_value = f"SIMD[DType.uint32, 2]({native_name}[0], {native_name}[1])"
        return native_name, "uint3", f"var {local_name}: {mapped_type} = {alias_value}"

    def mojo_unique_local_identifier(self, candidate, used_names):
        candidate = self.mojo_identifier(candidate)
        while candidate in used_names or candidate in self.mojo_reserved_identifiers():
            candidate = f"{candidate}_"
        return candidate

    def collect_mutated_parameters(self, body, param_names):
        mutated = set()
        for stmt in self.body_statements(body):
            self.collect_mutated_parameters_from_node(stmt, param_names, mutated)
        return mutated

    def body_statements(self, body):
        if hasattr(body, "statements"):
            return body.statements
        if isinstance(body, list):
            return body
        if body is None:
            return []
        return [body]

    def collect_mutated_parameters_from_node(self, node, param_names, mutated):
        if node is None:
            return

        if isinstance(node, AssignmentNode):
            root_name = self.assignment_target_root(node.left)
            if root_name in param_names:
                mutated.add(root_name)
            self.collect_mutated_parameters_from_node(node.right, param_names, mutated)
            return

        if isinstance(node, UnaryOpNode) and self.map_operator(node.op) in ["++", "--"]:
            root_name = self.assignment_target_root(node.operand)
            if root_name in param_names:
                mutated.add(root_name)
            return

        dimensions_call = self.get_expression_statement_call(node)
        if dimensions_call is not None:
            func_expr = getattr(dimensions_call, "function", None)
            if func_expr is None:
                func_expr = getattr(dimensions_call, "name", None)
            if (
                isinstance(func_expr, MemberAccessNode)
                and getattr(func_expr, "member", None) == "GetDimensions"
            ):
                for arg in self.get_get_dimensions_out_arguments(dimensions_call):
                    root_name = self.assignment_target_root(arg)
                    if root_name in param_names:
                        mutated.add(root_name)
                return

        for child in self.node_children(node):
            self.collect_mutated_parameters_from_node(child, param_names, mutated)

    def node_children(self, node):
        children = []
        for attr in (
            "init",
            "condition",
            "update",
            "body",
            "then_branch",
            "if_body",
            "else_branch",
            "else_body",
            "value",
            "expression",
            "left",
            "right",
            "object",
            "object_expr",
            "array",
            "array_expr",
            "index",
            "index_expr",
            "operand",
            "vector_expr",
        ):
            if hasattr(node, attr):
                children.append(getattr(node, attr))

        for attr in ("statements", "args", "arguments"):
            if hasattr(node, attr):
                children.extend(getattr(node, attr))
        if isinstance(node, ArrayLiteralNode):
            children.extend(node.elements)

        return children

    def assignment_target_root(self, target):
        if isinstance(target, str):
            return target
        if isinstance(target, VariableNode) and hasattr(target, "name"):
            return target.name
        if isinstance(target, ArrayAccessNode):
            return self.assignment_target_root(target.array)
        if isinstance(target, MemberAccessNode):
            return self.assignment_target_root(target.object)
        if hasattr(target, "__class__") and "Identifier" in str(target.__class__):
            return getattr(target, "name", None)
        if hasattr(target, "__class__") and "Swizzle" in str(target.__class__):
            return self.assignment_target_root(getattr(target, "vector_expr", None))
        return None

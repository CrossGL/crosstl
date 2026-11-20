"""Stage functions in the GLSL generator: mesh output parameter declarations, tessellation control, ray queries, statement sequences, barriers and compile-time type text."""

import re
from copy import deepcopy

from ..ast import (
    BlockNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    RayQueryOpNode,
    ReferenceType,
    ResourceMemoryQualifierNode,
    ReturnNode,
    SwitchNode,
    VariableNode,
    WhileNode,
)
from ..refusals import OpenGLRefusal
from .array_utils import format_c_style_array_declaration
from .glsl_exact_lookups import glsl_ray_tracing_intrinsic_call
from .glsl_mesh_locations import mesh_output_layouts
from .glsl_ray_queries import glsl_ray_query_initialize_flags
from .resource_arrays import is_private_pointer_parameter
from .stage_utils import (
    collect_stage_entry_records,
    collect_stage_local_cbuffers,
    collect_stage_local_variables,
    function_stage_name,
    normalize_stage_name,
    should_emit_qualified_function,
    stage_matches,
)


class OpenGLStageFunctionMixin:
    """Stage function members of ``GLSLCodeGen``."""

    def generate_mesh_output_parameter_declarations(self, mesh_outputs):
        code = ""
        emitted = set()
        layouts = mesh_output_layouts(self, mesh_outputs)
        for info in mesh_outputs.values():
            role = info["role"]
            if role == "indices":
                continue
            for member_name, member_info in info["members"].items():
                semantic = member_info["semantic"]
                if role == "vertices" and self.is_vertex_builtin_output(semantic):
                    continue
                if role == "primitives" and self.is_mesh_primitive_builtin(semantic):
                    continue

                member = member_info["node"]
                output_name = member_info["output_name"]
                declaration_key = (role, output_name)
                if declaration_key in emitted:
                    continue
                emitted.add(declaration_key)

                member_type = member_info["type"]
                count = info["count"]
                array_type = (
                    f"{member_type}[{count}]"
                    if count is not None
                    else f"{member_type}[]"
                )
                declaration = format_c_style_array_declaration(array_type, output_name)
                prefix = self.mesh_output_declaration_prefix(
                    member, role, layouts.get((role, output_name))
                )
                code += f"{prefix} {declaration};\n"
        return code

    def mesh_output_declaration_prefix(self, member, role, layout=None):
        prefix = self.stage_io_declaration_prefix(member, "out", layout=layout)
        if role != "primitives":
            return prefix

        parts = prefix.split()
        if "perprimitiveEXT" in parts:
            return prefix
        if "out" in parts:
            parts.insert(parts.index("out"), "perprimitiveEXT")
            return " ".join(parts)
        return f"{prefix} perprimitiveEXT".strip()

    def is_mesh_primitive_builtin(self, semantic):
        return semantic in {
            "gl_PrimitiveID",
            "gl_Layer",
            "gl_ViewportIndex",
            "gl_CullPrimitiveEXT",
        }

    def map_ray_tracing_intrinsic(self, operation, args):
        return glsl_ray_tracing_intrinsic_call(self, operation, args)

    def map_ray_query_intrinsic(self, operation, query, args):
        mapping = self.RAY_QUERY_METHOD_MAP.get(operation)
        query_expr = self.generate_expression(query)
        if mapping is None:
            generated_args = [self.generate_expression(arg) for arg in args]
            return f"{query_expr}.{operation}({', '.join(generated_args)})"
        if not self.can_lower_ray_query_receiver(operation, query):
            generated_args = [self.generate_expression(arg) for arg in args]
            return f"{query_expr}.{operation}({', '.join(generated_args)})"

        self.validate_ray_query_intrinsic_arguments(operation, args)
        function_name, committed = mapping
        call_args = [query_expr]
        if committed is not None:
            call_args.append(committed)
        args = self.ray_query_initialize_arguments(operation, args)
        args = [self.generate_expression(arg) for arg in args]
        if operation == "TraceRayInline" and len(args) > 1:
            args[1] = glsl_ray_query_initialize_flags(self, query, args[1])
        call_args.extend(args)
        call = f"{function_name}({', '.join(call_args)})"
        if operation == "CandidateProceduralPrimitiveNonOpaque":
            return f"(!{call})"
        return call

    def is_bool_ray_query_call(self, expr):
        if isinstance(expr, RayQueryOpNode):
            return getattr(expr, "operation", None) in self.RAY_QUERY_BOOL_METHODS
        if not isinstance(expr, FunctionCallNode):
            return False
        callee = getattr(expr, "function", getattr(expr, "name", None))
        if not isinstance(callee, MemberAccessNode):
            return False
        if callee.member not in self.RAY_QUERY_BOOL_METHODS:
            return False
        return self.is_ray_query_type(self.expression_result_type(callee.object))

    def can_lower_ray_query_receiver(self, operation, query):
        receiver_type = self.expression_result_type(query)
        if self.is_ray_query_type(receiver_type):
            return True
        if receiver_type is None:
            return False
        raise OpenGLRefusal(
            f"GLSL ray query {operation} requires a RayQuery receiver, "
            f"got {self.type_name_string(receiver_type)}"
        )

    def validate_ray_query_intrinsic_arguments(self, operation, args):
        allowed_arities = self.RAY_QUERY_METHOD_ARITIES.get(operation)
        if allowed_arities is None:
            return
        actual_arity = len(args)
        if actual_arity in allowed_arities:
            return
        if operation in self.RAY_QUERY_POSITION_FETCH_METHODS:
            raise OpenGLRefusal(
                f"GLSL ray query {operation} requires exactly one "
                f"output-array argument, got {actual_arity}"
            )
        raise OpenGLRefusal(
            f"GLSL ray query {operation} expects "
            f"{self.ray_query_arity_label(allowed_arities)}, got {actual_arity}"
        )

    def ray_query_arity_label(self, allowed_arities):
        if len(allowed_arities) == 1:
            arity = allowed_arities[0]
            plural = "" if arity == 1 else "s"
            return f"{arity} argument{plural}"
        return f"{' or '.join(str(arity) for arity in allowed_arities)} arguments"

    def ray_query_initialize_arguments(self, operation, args):
        if operation != "TraceRayInline" or len(args) != 4:
            return args

        acceleration, ray_flags, cull_mask, ray_desc = args
        return [
            acceleration,
            ray_flags,
            cull_mask,
            *[
                MemberAccessNode(
                    ray_desc,
                    self.ray_desc_field_name(ray_desc, field_names),
                )
                for field_names in self.RAY_QUERY_RAY_DESC_FIELDS
            ],
        ]

    def ray_desc_field_name(self, ray_desc, field_names):
        ray_desc_type = self.type_name_string(self.expression_result_type(ray_desc))
        members = self.struct_member_types.get(ray_desc_type, {})
        for field_name in field_names:
            if field_name in members:
                return field_name
        return field_names[0]

    def glsl_tessellation_domain(self, domain):
        domain_name = str(domain).strip().strip('"').lower()
        domain_map = {
            "tri": "triangles",
            "triangle": "triangles",
            "triangles": "triangles",
            "quad": "quads",
            "quads": "quads",
            "isoline": "isolines",
            "isolines": "isolines",
        }
        mapped = domain_map.get(domain_name)
        if mapped is None:
            raise OpenGLRefusal(f"GLSL tessellation domain cannot be lowered: {domain}")
        return mapped

    def glsl_tessellation_partitioning(self, partitioning):
        partition_name = str(partitioning).strip().strip('"').lower()
        partition_map = {
            # HLSL "integer" partitioning rounds levels up to integers, as equal_spacing does.
            "integer": "equal_spacing",
            "equal": "equal_spacing",
            "equal_spacing": "equal_spacing",
            "fractional_even": "fractional_even_spacing",
            "fractional_even_spacing": "fractional_even_spacing",
            "fractional_odd": "fractional_odd_spacing",
            "fractional_odd_spacing": "fractional_odd_spacing",
        }
        mapped = partition_map.get(partition_name)
        if mapped is None:
            raise OpenGLRefusal(
                f"GLSL tessellation partitioning cannot be lowered: {partitioning}"
            )
        return mapped

    def glsl_tessellation_output_topology_layout(self, topology, domain):
        if topology is None:
            return None, False

        topology_name = str(topology).strip().strip('"').lower()
        if topology_name in {"triangle_cw", "triangles_cw"}:
            if domain != "triangles":
                raise OpenGLRefusal(
                    "GLSL tessellation outputtopology triangle_cw requires "
                    "triangles domain"
                )
            return "cw", False
        if topology_name in {"triangle_ccw", "triangles_ccw"}:
            if domain != "triangles":
                raise OpenGLRefusal(
                    "GLSL tessellation outputtopology triangle_ccw requires "
                    "triangles domain"
                )
            return "ccw", False
        if topology_name in {"line", "lines"}:
            if domain != "isolines":
                raise OpenGLRefusal(
                    "GLSL tessellation outputtopology line requires isolines domain"
                )
            return None, False
        if topology_name in {"point", "points"}:
            return None, True

        raise OpenGLRefusal(
            f"GLSL tessellation outputtopology cannot be lowered: {topology}"
        )

    def structured_buffer_operation_access_requirement(self, func_name):
        if func_name == "buffer_load":
            return "read"
        if func_name in {"buffer_store", "buffer_append"}:
            return "write"
        if func_name == "buffer_consume":
            return "read_write"
        return None

    def merge_access_requirement(self, current, incoming):
        if incoming is None:
            return current
        if current is None or current == incoming:
            return incoming
        return "read_write"

    def collect_function_structured_buffer_access_requirements(self, functions):
        functions = list(functions)
        requirements = {
            getattr(func, "name", None): {}
            for func in functions
            if getattr(func, "name", None)
        }
        parameter_names = {
            func_name: list(names)
            for func_name, names in self.function_parameter_names.items()
        }
        parameter_sets = {
            func_name: set(names) for func_name, names in parameter_names.items()
        }

        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            parameter_set = parameter_sets.get(func_name, set())
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                operation = self.function_call_name(node)
                required_access = self.structured_buffer_operation_access_requirement(
                    operation
                )
                if required_access is None:
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if not args:
                    continue
                target_name = self.expression_name(args[0])
                if target_name not in parameter_set:
                    continue
                current = requirements[func_name].get(target_name)
                requirements[func_name][target_name] = self.merge_access_requirement(
                    current, required_access
                )

        changed = True
        while changed:
            changed = False
            for func in functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue
                parameter_set = parameter_sets.get(func_name, set())
                if not parameter_set:
                    continue
                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    callee_name = self.function_call_name(node)
                    callee_requirements = requirements.get(callee_name)
                    if not callee_requirements:
                        continue
                    callee_parameters = parameter_names.get(callee_name, [])
                    args = getattr(node, "arguments", getattr(node, "args", []))
                    for callee_param, required_access in callee_requirements.items():
                        try:
                            index = callee_parameters.index(callee_param)
                        except ValueError:
                            continue
                        if index >= len(args):
                            continue
                        target_name = self.expression_name(args[index])
                        if target_name not in parameter_set:
                            continue
                        current = requirements[func_name].get(target_name)
                        merged = self.merge_access_requirement(current, required_access)
                        if merged != current:
                            requirements[func_name][target_name] = merged
                            changed = True

        return {name: reqs for name, reqs in requirements.items() if reqs}

    def validate_glsl_tessellation_control_barriers(
        self,
        ast,
        target_stage,
        functions,
    ):
        """Reject tessellation-control barriers GLSL cannot legally place."""
        if target_stage is not None and target_stage != "tessellation_control":
            return

        entry_records = collect_stage_entry_records(
            ast,
            target_stage,
            self.combined_stage_entry_types(),
        )
        entry_functions = [
            func
            for _func_id, stage_name, func in entry_records
            if stage_name == "tessellation_control"
        ]
        if not entry_functions:
            return

        entry_ids = {id(func) for func in entry_functions}
        for func in entry_functions:
            self.validate_glsl_tessellation_control_entry_barriers(func)

        for func in self.glsl_tessellation_control_helper_functions(
            ast,
            target_stage,
            functions,
            entry_ids,
        ):
            barrier_name = self.glsl_tessellation_control_first_native_barrier(
                getattr(func, "body", [])
            )
            if barrier_name is None:
                continue
            raise OpenGLRefusal(
                "GLSL tessellation_control barrier() may only be emitted "
                "directly in main(); helper function "
                f"'{func.name}' reaches '{barrier_name}'"
            )

    def glsl_tessellation_control_helper_functions(
        self,
        ast,
        target_stage,
        functions,
        entry_ids,
    ):
        helpers = []
        seen = set()

        def add_helper(func):
            func_id = id(func)
            if func_id in entry_ids or func_id in seen:
                return
            seen.add(func_id)
            helpers.append(func)

        stage_entry_types = self.combined_stage_entry_types()
        top_level_function_ids = {
            id(func) for func in getattr(ast, "functions", []) or []
        }

        for func in functions:
            if id(func) not in top_level_function_ids:
                continue
            qualifier_name = function_stage_name(func)
            if qualifier_name in stage_entry_types:
                continue
            if should_emit_qualified_function(target_stage, qualifier_name):
                add_helper(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name != "tessellation_control" or not stage_matches(
                target_stage,
                stage_name,
            ):
                continue
            for func in getattr(stage, "local_functions", []) or []:
                add_helper(func)

        return helpers

    def validate_glsl_tessellation_control_entry_barriers(self, func):
        self.validate_glsl_tessellation_control_barrier_statement_sequence(
            self.glsl_statement_sequence(getattr(func, "body", [])),
            return_seen=False,
            control_depth=0,
        )

    def glsl_statement_sequence(self, body):
        if body is None:
            return []
        if hasattr(body, "statements"):
            return list(body.statements)
        if isinstance(body, list):
            return body
        return [body]

    def validate_glsl_tessellation_control_barrier_statement_sequence(
        self,
        statements,
        return_seen,
        control_depth,
    ):
        for stmt in statements:
            return_seen = self.validate_glsl_tessellation_control_barrier_statement(
                stmt,
                return_seen,
                control_depth,
            )
        return return_seen

    def validate_glsl_tessellation_control_barrier_statement(
        self,
        stmt,
        return_seen,
        control_depth,
    ):
        barrier_name = self.glsl_tessellation_control_standalone_barrier_statement(stmt)
        if barrier_name is not None:
            if control_depth:
                raise OpenGLRefusal(
                    "GLSL tessellation_control barrier() must not be emitted "
                    f"inside control flow; found '{barrier_name}'"
                )
            if return_seen:
                raise OpenGLRefusal(
                    "GLSL tessellation_control barrier() cannot be emitted "
                    f"after a return statement in main(); found '{barrier_name}'"
                )
            return return_seen

        if isinstance(stmt, ReturnNode):
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "value", None)
            )
            return True

        if isinstance(stmt, BlockNode):
            return self.validate_glsl_tessellation_control_barrier_statement_sequence(
                self.glsl_statement_sequence(stmt),
                return_seen,
                control_depth,
            )

        if isinstance(stmt, IfNode):
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "condition", getattr(stmt, "if_condition", None))
            )
            return_seen = (
                self.validate_glsl_tessellation_control_barrier_statement_sequence(
                    self.glsl_statement_sequence(
                        getattr(stmt, "if_body", getattr(stmt, "then_branch", []))
                    ),
                    return_seen,
                    control_depth + 1,
                )
            )
            for condition, body in zip(
                getattr(stmt, "else_if_conditions", []) or [],
                getattr(stmt, "else_if_bodies", []) or [],
            ):
                self.validate_glsl_tessellation_control_nonstandalone_barriers(
                    condition
                )
                return_seen = (
                    self.validate_glsl_tessellation_control_barrier_statement_sequence(
                        self.glsl_statement_sequence(body),
                        return_seen,
                        control_depth + 1,
                    )
                )
            else_body = getattr(stmt, "else_body", getattr(stmt, "else_branch", None))
            if else_body is not None:
                return_seen = (
                    self.validate_glsl_tessellation_control_barrier_statement_sequence(
                        self.glsl_statement_sequence(else_body),
                        return_seen,
                        control_depth + 1,
                    )
                )
            return return_seen

        if isinstance(stmt, ForNode):
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "init", None)
            )
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "condition", None)
            )
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "update", None)
            )
            return self.validate_glsl_tessellation_control_barrier_statement_sequence(
                self.glsl_statement_sequence(getattr(stmt, "body", [])),
                return_seen,
                control_depth + 1,
            )

        if isinstance(stmt, ForInNode):
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "iterable", None)
            )
            return self.validate_glsl_tessellation_control_barrier_statement_sequence(
                self.glsl_statement_sequence(getattr(stmt, "body", [])),
                return_seen,
                control_depth + 1,
            )

        if isinstance(stmt, WhileNode):
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "condition", None)
            )
            return self.validate_glsl_tessellation_control_barrier_statement_sequence(
                self.glsl_statement_sequence(getattr(stmt, "body", [])),
                return_seen,
                control_depth + 1,
            )

        if isinstance(stmt, DoWhileNode):
            return_seen = (
                self.validate_glsl_tessellation_control_barrier_statement_sequence(
                    self.glsl_statement_sequence(getattr(stmt, "body", [])),
                    return_seen,
                    control_depth + 1,
                )
            )
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "condition", None)
            )
            return return_seen

        if isinstance(stmt, LoopNode):
            return self.validate_glsl_tessellation_control_barrier_statement_sequence(
                self.glsl_statement_sequence(getattr(stmt, "body", [])),
                return_seen,
                control_depth + 1,
            )

        if isinstance(stmt, SwitchNode):
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "expression", None)
            )
            for case in getattr(stmt, "cases", []) or []:
                return_seen = (
                    self.validate_glsl_tessellation_control_barrier_statement_sequence(
                        self.glsl_statement_sequence(getattr(case, "statements", [])),
                        return_seen,
                        control_depth + 1,
                    )
                )
            default_case = getattr(stmt, "default_case", None)
            if default_case is not None:
                return_seen = (
                    self.validate_glsl_tessellation_control_barrier_statement_sequence(
                        self.glsl_statement_sequence(
                            getattr(default_case, "statements", default_case)
                        ),
                        return_seen,
                        control_depth + 1,
                    )
                )
            return return_seen

        if isinstance(stmt, MatchNode):
            self.validate_glsl_tessellation_control_nonstandalone_barriers(
                getattr(stmt, "expression", None)
            )
            for arm in getattr(stmt, "arms", []) or []:
                self.validate_glsl_tessellation_control_nonstandalone_barriers(
                    getattr(arm, "guard", None)
                )
                return_seen = (
                    self.validate_glsl_tessellation_control_barrier_statement_sequence(
                        self.glsl_statement_sequence(getattr(arm, "body", [])),
                        return_seen,
                        control_depth + 1,
                    )
                )
            return return_seen

        self.validate_glsl_tessellation_control_nonstandalone_barriers(stmt)
        return return_seen

    def validate_glsl_tessellation_control_nonstandalone_barriers(self, node):
        barrier_name = self.glsl_tessellation_control_first_native_barrier(node)
        if barrier_name is None:
            return
        raise OpenGLRefusal(
            "GLSL tessellation_control barrier() must be emitted as a "
            f"standalone statement in main(); found '{barrier_name}'"
        )

    def glsl_tessellation_control_standalone_barrier_statement(self, stmt):
        expression = (
            stmt.expression if isinstance(stmt, ExpressionStatementNode) else stmt
        )
        if not isinstance(expression, FunctionCallNode):
            return None
        return self.glsl_tessellation_control_native_barrier_call_name(expression)

    def glsl_tessellation_control_first_native_barrier(self, node):
        for child in self.walk_ast(node):
            if not isinstance(child, FunctionCallNode):
                continue
            barrier_name = self.glsl_tessellation_control_native_barrier_call_name(
                child
            )
            if barrier_name is not None:
                return barrier_name
        return None

    def glsl_tessellation_control_native_barrier_call_name(self, call):
        func_name = self.function_call_name(call)
        if func_name not in self.GLSL_TESSELLATION_CONTROL_BARRIER_BUILTINS:
            return None
        if func_name in self.function_return_types:
            return None
        args = getattr(call, "arguments", getattr(call, "args", [])) or []
        if args:
            return None
        return func_name

    def glsl_parameter_qualifiers(self, parameter, array_view_writeback=True):
        qualifiers = []
        seen = set()
        source_qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        for qualifier in getattr(parameter, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in self.GLSL_PARAMETER_QUALIFIERS and normalized not in seen:
                qualifiers.append(normalized)
                seen.add(normalized)
        raw_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        if (
            array_view_writeback
            and getattr(parameter, "_glsl_scalar_array_view_inout", False)
            and not seen.intersection({"out", "inout"})
            and not source_qualifiers.intersection(
                {"const", "constant", "readonly", "in"}
            )
        ):
            qualifiers.append("inout")
            seen.add("inout")
        if (
            is_private_pointer_parameter(parameter)
            and getattr(raw_type, "is_mutable", True)
            and not seen.intersection({"out", "inout"})
            and not source_qualifiers.intersection(
                {"const", "constant", "readonly", "in"}
            )
        ):
            qualifiers.append("inout")
            seen.add("inout")
        if (
            isinstance(raw_type, ReferenceType)
            and not seen.intersection({"out", "inout"})
            and not source_qualifiers.intersection(
                {"const", "constant", "readonly", "in"}
            )
        ):
            qualifiers.append("inout")
        return qualifiers

    def stage_functions(self, ast, stage_name):
        functions = []

        for func in getattr(ast, "functions", []) or []:
            if function_stage_name(func) == stage_name:
                functions.append(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            current_stage = normalize_stage_name(stage_type)
            if current_stage == stage_name and hasattr(stage, "entry_point"):
                functions.append(stage.entry_point)

        return functions

    def stage_entry_function_ids(self, ast):
        stage_names = {
            "vertex",
            "fragment",
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
            "task",
            "amplification",
            "object",
            "ray_generation",
            "ray_intersection",
            "ray_closest_hit",
            "ray_any_hit",
            "ray_miss",
            "ray_callable",
        }
        entry_ids = set()
        for stage_name in stage_names:
            entry_ids.update(id(func) for func in self.stage_functions(ast, stage_name))
        return entry_ids

    def type_node_references_name(self, type_node, struct_name):
        if type_node is None:
            return False
        if self.type_node_name(type_node) == struct_name:
            return True
        referenced_type = getattr(type_node, "referenced_type", None)
        if referenced_type is not None and self.type_node_references_name(
            referenced_type, struct_name
        ):
            return True
        pointee_type = getattr(type_node, "pointee_type", None)
        if pointee_type is not None and self.type_node_references_name(
            pointee_type, struct_name
        ):
            return True
        element_type = getattr(type_node, "element_type", None)
        if element_type is not None and self.type_node_references_name(
            element_type, struct_name
        ):
            return True
        return any(
            self.type_node_references_name(arg, struct_name)
            for arg in getattr(type_node, "generic_args", []) or []
        )

    def statement_declares_type(self, statement, struct_name):
        if isinstance(statement, VariableNode) and self.type_node_references_name(
            getattr(statement, "var_type", None), struct_name
        ):
            return True
        for child_name in ("body", "else_body", "cases", "default", "statements"):
            child = getattr(statement, child_name, None)
            if child is None:
                continue
            if isinstance(child, list):
                if any(
                    self.statement_declares_type(item, struct_name) for item in child
                ):
                    return True
            elif self.statement_declares_type(child, struct_name):
                return True
        return False

    def function_signature_references_type(self, func, struct_name):
        if self.type_node_references_name(
            getattr(func, "return_type", None), struct_name
        ):
            return True
        return any(
            self.type_node_references_name(
                getattr(param, "param_type", None), struct_name
            )
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []
        )

    def function_body_declares_type(self, func, struct_name):
        body = getattr(func, "body", [])
        statements = getattr(body, "statements", body if isinstance(body, list) else [])
        return any(
            self.statement_declares_type(stmt, struct_name) for stmt in statements
        )

    def struct_member_references_type(self, struct, struct_name):
        return any(
            self.type_node_references_name(
                getattr(member, "member_type", None), struct_name
            )
            for member in getattr(struct, "members", []) or []
        )

    def stage_io_struct_referenced_outside_stage_entries(self, ast, struct_name):
        stage_entry_ids = self.stage_entry_function_ids(ast)

        def entry_declares_parameter_struct(func):
            parameter_struct_names = {
                self.type_node_name(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
                for param in (
                    getattr(func, "parameters", getattr(func, "params", [])) or []
                )
            }
            return struct_name in parameter_struct_names and (
                self.function_body_declares_type(func, struct_name)
            )

        seen_entries = set()
        for stage_type in getattr(ast, "stages", {}):
            stage_name = normalize_stage_name(stage_type)
            for func in self.stage_functions(ast, stage_name):
                if id(func) not in stage_entry_ids or id(func) in seen_entries:
                    continue
                seen_entries.add(id(func))
                if entry_declares_parameter_struct(func):
                    return True
                # Only vertex and fragment entry points flatten their interface structure locals into stage
                # variables; any other stage's local of that type is an ordinary structure value.
                if stage_name not in {
                    "vertex",
                    "fragment",
                } and self.function_body_declares_type(func, struct_name):
                    return True

        for func in getattr(ast, "functions", []) or []:
            if id(func) in stage_entry_ids:
                if id(func) not in seen_entries and entry_declares_parameter_struct(
                    func
                ):
                    return True
                continue
            if self.function_signature_references_type(
                func, struct_name
            ) or self.function_body_declares_type(func, struct_name):
                return True

        for stage in getattr(ast, "stages", {}).values():
            for func in getattr(stage, "functions", []) or []:
                if id(func) in stage_entry_ids:
                    if id(func) not in seen_entries and entry_declares_parameter_struct(
                        func
                    ):
                        return True
                    continue
                if self.function_signature_references_type(
                    func, struct_name
                ) or self.function_body_declares_type(func, struct_name):
                    return True

        for struct in getattr(ast, "structs", []) or []:
            if getattr(struct, "name", None) == struct_name:
                continue
            if self.struct_member_references_type(struct, struct_name):
                return True
        return False

    def should_emit_flattened_stage_io_struct(
        self, ast, struct_name, emitted_io, emit_graphics_io
    ):
        if not emit_graphics_io:
            return True
        if not emitted_io:
            return True
        if struct_name in self.fragment_output_struct_names:
            return True
        return self.stage_io_struct_referenced_outside_stage_entries(ast, struct_name)

    def stage_parameter_struct_names(self, ast, stage_name):
        struct_names = set()
        for func in self.stage_functions(ast, stage_name):
            parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
            for param in parameters:
                if self.is_stage_entry_resource_parameter(param):
                    continue
                type_name = self.type_node_name(getattr(param, "param_type", None))
                if type_name in self.structs_by_name:
                    struct_names.add(type_name)
        return struct_names

    def stage_return_struct_names(self, ast, stage_name):
        struct_names = set()
        for func in self.stage_functions(ast, stage_name):
            type_name = self.type_node_name(getattr(func, "return_type", None))
            if type_name in self.structs_by_name:
                struct_names.add(type_name)
        return struct_names

    def collect_stage_entry_resource_parameters(
        self, ast, target_stage=None, base_declarations=None
    ):
        stage_entry_types = {
            "vertex",
            "fragment",
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
            "task",
            "amplification",
            "object",
            "ray_generation",
            "ray_intersection",
            "ray_closest_hit",
            "ray_any_hit",
            "ray_miss",
            "ray_callable",
        }
        parameters = []
        seen = set()
        declarations_by_name = {}
        used_names = set()
        for node in base_declarations or []:
            record = self.glsl_resource_declaration_record(node)
            if record is None:
                continue
            source_name, declaration = record
            emitted_name = self.glsl_module_identifier_name(source_name)
            declarations_by_name.setdefault(emitted_name, declaration)
            used_names.add(emitted_name)

        def add_parameters(func):
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
                if id(param) in seen:
                    continue
                if not self.is_stage_entry_resource_parameter(param):
                    continue
                declaration_node = self.glsl_stage_entry_resource_parameter_declaration(
                    param
                )
                record = self.glsl_resource_declaration_record(declaration_node)
                if record is None:
                    continue
                original_name, declaration = record
                emitted_name = self.glsl_module_identifier_name(original_name)
                existing = declarations_by_name.get(emitted_name)
                if existing is not None and self.glsl_resource_declarations_conflict(
                    existing,
                    declaration,
                ):
                    emitted_name = self.glsl_scoped_entry_resource_name(
                        func,
                        original_name,
                        used_names,
                    )
                    declaration_node.name = emitted_name
                    record = self.glsl_resource_declaration_record(declaration_node)
                    if record is not None:
                        emitted_name, declaration = record
                declaration_node.name = emitted_name
                scalar_reference_type = (
                    self.stage_entry_metal_scalar_reference_element_type(param)
                )
                if scalar_reference_type is not None:
                    self.stage_entry_resource_parameter_aliases.setdefault(
                        id(func), {}
                    )[original_name] = f"{emitted_name}[0]"
                elif emitted_name != original_name:
                    self.stage_entry_resource_parameter_aliases.setdefault(
                        id(func), {}
                    )[original_name] = emitted_name
                if emitted_name:
                    existing = declarations_by_name.get(emitted_name)
                    if existing is None:
                        declarations_by_name[emitted_name] = declaration
                    elif existing.get("binding") is None:
                        existing["binding"] = declaration.get("binding")
                    used_names.add(emitted_name)
                    self.stage_entry_resource_declaration_names.add(emitted_name)
                    pointer_binding = self.glsl_stage_entry_storage_pointer_binding(
                        declaration_node, emitted_name
                    )
                    if pointer_binding is not None:
                        bindings = self.stage_entry_storage_pointer_bindings.setdefault(
                            id(func), {}
                        )
                        bindings[original_name] = dict(pointer_binding)
                        bindings[emitted_name] = dict(pointer_binding)
                parameters.append(declaration_node)
                seen.add(id(param))

        for func in getattr(ast, "functions", []) or []:
            qualifier_name = function_stage_name(func)
            if qualifier_name not in stage_entry_types:
                continue
            if should_emit_qualified_function(target_stage, qualifier_name):
                add_parameters(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                add_parameters(entry_point)

        return parameters

    def glsl_stage_entry_resource_parameter_declaration(self, parameter):
        constant_struct_type = self.stage_entry_constant_struct_parameter_type(
            parameter
        )
        if constant_struct_type is not None:
            return VariableNode(
                name=getattr(parameter, "name", None),
                var_type=constant_struct_type,
                attributes=list(getattr(parameter, "attributes", []) or []),
                qualifiers=["uniform"],
                source_location=getattr(parameter, "source_location", None),
                annotations=deepcopy(getattr(parameter, "annotations", {})),
            )

        resource_type = self.stage_entry_resource_parameter_type(parameter)
        resource_qualifiers = deepcopy(
            list(getattr(parameter, "resource_qualifiers", []) or [])
        )
        existing_contracts = {
            (qualifier.kind, qualifier.scope)
            for qualifier in resource_qualifiers
            if isinstance(qualifier, ResourceMemoryQualifierNode)
        }
        for qualifier in self.typed_resource_memory_qualifiers(
            parameter,
            raw_type=self.resource_node_type(parameter),
        ):
            contract = (qualifier.kind, qualifier.scope)
            if contract in existing_contracts:
                continue
            resource_qualifiers.append(deepcopy(qualifier))
            existing_contracts.add(contract)
        node = VariableNode(
            name=getattr(parameter, "name", None),
            var_type=(
                resource_type
                if resource_type is not None
                else self.resource_node_type(parameter)
            ),
            attributes=list(getattr(parameter, "attributes", []) or []),
            qualifiers=list(getattr(parameter, "qualifiers", []) or []),
            resource_qualifiers=resource_qualifiers,
            source_location=getattr(parameter, "source_location", None),
            annotations=deepcopy(getattr(parameter, "annotations", {})),
        )
        for attr_name in (
            "array_sizes",
            "declarator_type_suffix",
            "declarator_type_suffix_grouped",
        ):
            if hasattr(parameter, attr_name):
                setattr(node, attr_name, getattr(parameter, attr_name))
        return node

    def glsl_resource_declaration_record(self, node):
        name = self.resource_node_name(node)
        identity = self.resource_declaration_identity(node)
        if not name or identity is None:
            return None
        return (
            name,
            {
                "identity": identity,
                "binding": self.explicit_resource_binding_index(node),
            },
        )

    def glsl_resource_declarations_conflict(self, existing, candidate):
        if existing["identity"] != candidate["identity"]:
            return True
        existing_binding = existing.get("binding")
        candidate_binding = candidate.get("binding")
        return (
            existing_binding is not None
            and candidate_binding is not None
            and existing_binding != candidate_binding
        )

    def glsl_scoped_entry_resource_name(self, func, parameter_name, used_names):
        func_name = self.glsl_sanitized_identifier_base(
            getattr(func, "name", None),
            fallback="entry",
        )
        parameter_name = self.glsl_sanitized_identifier_base(
            parameter_name,
            fallback="resource",
        )
        base_name = "_".join(part for part in (func_name, parameter_name) if part)
        if not base_name:
            base_name = "entry_resource"
        return self.glsl_unique_identifier(base_name, used_names)

    def collect_global_variable_types(self, nodes):
        variable_types = {}
        for node in nodes or []:
            name = self.resource_node_name(node)
            if not name:
                continue
            vtype, _array_size, array_suffix, _resource_count = (
                self.resource_declaration_shape(node)
            )
            variable_types[name] = f"{self.type_name_string(vtype)}{array_suffix}"
        return variable_types

    def collect_glsl_global_identifier_names(self, ast, target_stage=None):
        declarations = (
            list(getattr(ast, "global_variables", []) or [])
            + list(getattr(ast, "constants", []) or [])
            + collect_stage_local_variables(ast, target_stage)
        )
        names = set()
        for node in declarations:
            name = self.resource_node_name(node)
            if name:
                names.add(name)
        for import_node in getattr(ast, "imports", []) or []:
            alias = getattr(import_node, "alias", None)
            if alias:
                names.add(alias)
            names.update(
                item for item in getattr(import_node, "items", []) or [] if item
            )
        cbuffers = list(getattr(ast, "cbuffers", []) or [])
        cbuffers += collect_stage_local_cbuffers(ast, target_stage)
        for cbuffer in cbuffers:
            name = getattr(cbuffer, "name", None)
            if name:
                names.add(name)
            names.update(
                member_name
                for member in getattr(cbuffer, "members", []) or []
                if (member_name := getattr(member, "name", None))
            )
        return names

    def glsl_preprocessor_identifier_names(self, ast):
        names = set()
        for directive in getattr(ast, "preprocessors", []) or []:
            if getattr(directive, "directive", None) != "define":
                continue
            content = str(getattr(directive, "content", "") or "")
            match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", content)
            if match is not None:
                names.add(match.group(1))
        return names

    def prepare_glsl_program_identifier_names(
        self,
        ast,
        structs,
        functions,
        target_stage=None,
    ):
        type_names = {
            getattr(node, "name", None)
            for node in structs or ()
            if getattr(node, "name", None)
        }
        type_names.update(
            specialization.get("struct_name")
            for specialization in self.generic_struct_specializations.values()
            if specialization.get("struct_name")
        )
        type_names.update(
            specialization.get("struct_name")
            for specialization in self.generic_enum_specializations.values()
            if specialization.get("struct_name")
        )

        names = set(self.glsl_global_identifier_names)
        names.update(type_names)
        names.update(
            getattr(function, "name", None)
            for function in functions or ()
            if getattr(function, "name", None)
        )
        for _entry_id, _stage_name, function in collect_stage_entry_records(
            ast,
            target_stage,
            self.combined_stage_entry_types(),
        ):
            names.update(
                getattr(parameter, "name", None)
                for parameter in (
                    getattr(function, "parameters", getattr(function, "params", []))
                    or ()
                )
                if getattr(parameter, "name", None)
            )
        for struct in structs or ():
            names.update(
                getattr(member, "name", None)
                for member in getattr(struct, "members", []) or []
                if getattr(member, "name", None)
            )

        self.prepare_glsl_module_identifier_names(
            names,
            type_names=type_names,
            reserved_names=self.glsl_preprocessor_identifier_names(ast),
        )
        for name in sorted(self.glsl_builtin_redeclaration_names(ast)):
            self.glsl_module_identifier_names[name] = name
            self.glsl_module_used_identifier_names.add(name)

    def prepare_glsl_trailing_zero_helper_names(self, functions):
        reserved_names = set()
        for function in functions or ():
            for name in self.collect_glsl_function_local_identifier_names(function):
                reserved_names.add(name)
                reserved_names.add(self.glsl_sanitized_identifier_base(name))
        self.glsl_trailing_zero_reserved_names = reserved_names
        self.glsl_trailing_zero_helper_names = {}

    def glsl_trailing_zero_helper_name(self, operand_type):
        helper_name = self.glsl_trailing_zero_helper_names.get(operand_type)
        if helper_name is not None:
            return helper_name

        used_names = set(self.glsl_module_used_identifier_names)
        used_names.update(self.glsl_trailing_zero_reserved_names)
        used_names.update(self.glsl_trailing_zero_helper_names.values())
        helper_name = self.glsl_unique_identifier(
            f"crossgl_ctz_{operand_type}",
            used_names,
        )
        self.glsl_trailing_zero_helper_names[operand_type] = helper_name
        self.glsl_module_used_identifier_names.add(helper_name)
        self.glsl_module_generated_identifier_names[("trailing-zero", operand_type)] = (
            helper_name
        )
        return helper_name

    def prepare_glsl_enum_identifier_names(self):
        self.enum_variant_constants = {
            path: self.glsl_generated_module_identifier(
                ("enum-constant", path),
                source_name,
            )
            for path, source_name in sorted(self.enum_variant_constants.items())
        }
        self.enum_variant_constructors = {
            path: self.glsl_generated_module_identifier(
                ("enum-constructor", path),
                source_name,
            )
            for path, source_name in sorted(self.enum_variant_constructors.items())
        }

        constructor_names = {}
        for type_text, specialization in sorted(
            self.generic_enum_specializations.items()
        ):
            definition = specialization["definition"]
            for variant in sorted(
                getattr(definition["enum"], "variants", ()) or (),
                key=lambda item: item.name,
            ):
                source_name = f"{specialization['struct_name']}_{variant.name}_make"
                constructor_names[(type_text, variant.name)] = (
                    self.glsl_generated_module_identifier(
                        ("generic-enum-constructor", type_text, variant.name),
                        source_name,
                    )
                )
        self.glsl_generic_enum_constructor_names = constructor_names

    def is_stage_local_resource_variable(self, node):
        vtype = self.type_name_string(
            getattr(node, "var_type", getattr(node, "vtype", "float"))
        )
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        base_type = self.resource_base_type(vtype)
        mapped_type = self.map_resource_type_with_format(base_type, node)
        return (
            bool(qualifiers & {"uniform", "buffer"})
            or mapped_type == "sampler"
            or self.is_opaque_resource_type(mapped_type)
            or self.is_structured_buffer_type(vtype)
            or self.is_constant_buffer_type(vtype)
            or self.is_glsl_buffer_block_variable(node, vtype)
        )

    def is_stage_local_interface_variable(self, node):
        if self.is_stage_local_resource_variable(node):
            return False
        return bool(
            self.stage_local_interface_direction(node)
            or self.glsl_variable_layout_prefix(node)
            or self.glsl_variable_qualifier_prefix(node)
        )

    def canonical_glsl_compile_time_type_text(self, type_text):
        type_text = str(type_text)
        for name, value in sorted(self.literal_int_constants.items()):
            type_text = type_text.replace(f"[{name}]", f"[{value}]")
        return type_text

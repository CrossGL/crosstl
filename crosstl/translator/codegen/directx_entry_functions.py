"""Entry functions in the HLSL generator: built-in and fragment parameters, required stage parameters, vertex entries, program dependencies, compute outputs, hull signatures and mapped types."""

import re

from ..ast import (
    ArrayAccessNode,
    AttributeNode,
    BinaryOpNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .array_utils import (
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .directx_values import DirectXMappedOverloadError
from .enum_utils import sanitize_type_name
from .hlsl_host_workgroup_size import hlsl_entry_local_size
from .stage_utils import (
    function_stage_name,
    normalize_stage_name,
    stage_matches,
)


class DirectXEntryFunctionMixin:
    """Entry function members of ``HLSLCodeGen``."""

    def hlsl_stage_io_variable_semantic(self, variable):
        for attr in getattr(variable, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            normalized = str(attr_name).lower()
            if normalized in {
                "input",
                "output",
                "in",
                "out",
                "highp",
                "mediump",
                "lowp",
            }:
                continue
            if attr_name in self.semantic_map or normalized == "location":
                return self.semantic_attribute_name(attr)
        return None

    def hlsl_stage_io_member_attributes(self, variable):
        retained = []
        for attr in getattr(variable, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            normalized = str(attr_name).lower()
            if normalized in {
                "input",
                "output",
                "in",
                "out",
                "highp",
                "mediump",
                "lowp",
            }:
                continue
            if attr_name in self.semantic_map or normalized in {
                "location",
                "centroid",
                "flat",
                "noperspective",
                "sample",
                "smooth",
            }:
                retained.append(
                    AttributeNode(attr_name, list(getattr(attr, "arguments", []) or []))
                )
        return retained

    def collect_hlsl_vertex_entry_output_struct_names(self, ast, target_stage=None):
        if not stage_matches(target_stage, "vertex"):
            return set()

        struct_names = set()
        for func in self.hlsl_stage_entry_functions(ast, "vertex"):
            return_type = self.type_name_string(
                getattr(func, "return_type", getattr(func, "vtype", None))
            )
            if not return_type:
                continue
            base_type = return_type.split("<", 1)[0].split("[", 1)[0].strip()
            if base_type in self.structs_by_name:
                struct_names.add(base_type)
        return struct_names

    def collect_hlsl_vertex_entry_input_struct_names(self, ast, target_stage=None):
        if not stage_matches(target_stage, "vertex"):
            return set()

        struct_names = set()
        for func in self.hlsl_stage_entry_functions(ast, "vertex"):
            for parameter in (
                getattr(func, "parameters", getattr(func, "params", [])) or []
            ):
                struct_name = self.hlsl_parameter_user_struct_type(parameter)
                if struct_name is not None:
                    struct_names.add(struct_name)
        return struct_names

    def collect_hlsl_fragment_entry_input_struct_names(self, ast, target_stage=None):
        if not stage_matches(target_stage, "fragment"):
            return set()

        struct_names = set()
        for func in self.hlsl_stage_entry_functions(ast, "fragment"):
            for parameter in (
                getattr(func, "parameters", getattr(func, "params", [])) or []
            ):
                struct_name = self.hlsl_parameter_user_struct_type(parameter)
                if struct_name is not None:
                    struct_names.add(struct_name)
        return struct_names

    def collect_hlsl_fragment_entry_output_struct_names(self, ast, target_stage=None):
        if not stage_matches(target_stage, "fragment"):
            return set()

        struct_names = set()
        for func in self.hlsl_stage_entry_functions(ast, "fragment"):
            return_type = self.type_name_string(
                getattr(func, "return_type", getattr(func, "vtype", None))
            )
            if not return_type:
                continue
            base_type = return_type.split("<", 1)[0].split("[", 1)[0].strip()
            if base_type in self.structs_by_name:
                struct_names.add(base_type)
        return struct_names

    def collect_hlsl_geometry_entry_vertex_struct_names(self, ast, target_stage=None):
        """Structures a geometry entry receives its primitive's vertices in or appends to a stream."""
        if not stage_matches(target_stage, "geometry"):
            return set()

        struct_names = set()
        stream_types = {"PointStream", "LineStream", "TriangleStream"}
        primitives = {"point", "line", "triangle", "lineadj", "triangleadj"}
        for func in self.hlsl_stage_entry_functions(ast, "geometry"):
            for parameter in (
                getattr(func, "parameters", getattr(func, "params", [])) or []
            ):
                if self.hlsl_parameter_type_base(parameter) in stream_types:
                    type_name = self.hlsl_parameter_type_generic_argument(parameter, 0)
                elif set(self.hlsl_parameter_qualifiers(parameter)) & primitives:
                    type_name = self.type_name_string(
                        getattr(
                            parameter, "param_type", getattr(parameter, "vtype", None)
                        )
                    )
                else:
                    continue
                base_type = (
                    str(type_name or "").split("<", 1)[0].split("[", 1)[0].strip()
                )
                if base_type in self.structs_by_name:
                    struct_names.add(base_type)
        return struct_names

    def hlsl_program_hull_output_control_points(self, ast):
        counts = {
            self.hlsl_stage_attribute_int_argument(func, "outputcontrolpoints")
            for func in self.hlsl_stage_entry_functions(ast, "tessellation_control")
        }
        counts.discard(None)
        if len(counts) == 1:
            return next(iter(counts))
        return None

    def hlsl_program_hull_output_element_type(self, ast):
        output_types = set()
        for func in self.hlsl_stage_entry_functions(ast, "tessellation_control"):
            return_type = self.type_name_string(
                getattr(func, "return_type", getattr(func, "vtype", None))
            )
            if not return_type or self.map_type(return_type) == "void":
                continue
            output_types.add(self.map_type(return_type))
        if len(output_types) == 1:
            return next(iter(output_types))
        return None

    def hlsl_program_hull_domain(self, ast):
        domains = {
            self.canonical_hlsl_tessellation_domain(
                self.normalized_hlsl_stage_attribute_argument(func, "domain")
            )
            for func in self.hlsl_stage_entry_functions(ast, "tessellation_control")
        }
        domains.discard(None)
        if len(domains) == 1:
            return next(iter(domains))
        return None

    def hlsl_program_mesh_payload_types(self, ast):
        payload_types = {
            self.hlsl_mesh_payload_type_from_parameters(
                getattr(func, "parameters", getattr(func, "params", [])) or []
            )
            for func in self.hlsl_stage_entry_functions(ast, "mesh")
        }
        payload_types.discard(None)
        return payload_types

    def hlsl_program_dispatch_mesh_payload_types(self, ast):
        payload_types = set()
        global_functions_by_name = {
            func.name: func
            for func in getattr(ast, "functions", []) or []
            if getattr(func, "name", None)
        }
        stages = getattr(ast, "stages", {}) or {}
        previous_available_functions = self.current_hlsl_available_functions
        for stage_name in ("task", "amplification", "object"):
            for func in self.hlsl_stage_entry_functions(ast, stage_name):
                stage_functions_by_name = dict(global_functions_by_name)
                for stage_type, stage in stages.items():
                    if normalize_stage_name(stage_type) != stage_name:
                        continue
                    if getattr(stage, "entry_point", None) is not func:
                        continue
                    stage_functions_by_name.update(
                        {
                            helper.name: helper
                            for helper in getattr(stage, "local_functions", []) or []
                            if getattr(helper, "name", None)
                        }
                    )

                self.current_hlsl_available_functions = {
                    name: helper
                    for name, helper in stage_functions_by_name.items()
                    if helper is not func
                }
                try:
                    payload_types.update(
                        self.hlsl_dispatch_mesh_payload_types_for_function(func)
                    )
                finally:
                    self.current_hlsl_available_functions = previous_available_functions
        return payload_types

    def hlsl_program_has_amplification_stage(self, ast):
        for stage_name in ("task", "amplification", "object"):
            if self.hlsl_stage_entry_functions(ast, stage_name):
                return True
        return False

    def collect_functions(self, root):
        kept = getattr(self, "hlsl_program_functions", None)
        if kept is not None and (functions := kept.lookup(root)) is not None:
            return functions
        functions = []
        for node in self.walk_ast(root):
            if hasattr(node, "body") and hasattr(node, "parameters"):
                functions.append(node)
        return functions

    def collect_hlsl_function_name_aliases(self, functions):
        names = {
            getattr(func, "name", None)
            for func in functions or []
            if getattr(func, "name", None)
        }
        used_names = set(names)
        used_names.update(self.global_variable_types)
        used_names.update(self.structs_by_name)
        used_names.update(getattr(self, "enum_type_names", set()))
        used_names.update(getattr(self, "enum_struct_type_names", set()))
        aliases = {}
        # DXC resolves select as an intrinsic even with a same-named declaration.
        reserved_functions = set(self.HLSL_RESERVED_LOCAL_IDENTIFIER_NAMES)
        if any(
            getattr(function, "name", None) == "select"
            and len(getattr(function, "parameters", ())) >= 3
            for function in functions
        ):
            reserved_functions.add("select")
            for function in functions:
                used_names.update(self.collect_hlsl_function_identifier_names(function))
        for name in sorted(names & reserved_functions):
            alias = f"{name}_"
            while (
                alias in used_names
                or alias in self.HLSL_RESERVED_LOCAL_IDENTIFIER_NAMES
            ):
                alias += "_"
            aliases[name] = alias
            used_names.add(alias)
        return aliases

    def prepare_hlsl_mapped_function_overloads(self, functions, reserved_names=()):
        """Assign target names when distinct source overloads collapse in HLSL."""
        groups = {}
        for function in functions or []:
            function_name = getattr(function, "name", None)
            if not function_name or function_stage_name(function):
                continue
            groups.setdefault(function_name, []).append(function)

        self.hlsl_function_overloads_by_name = groups
        self.hlsl_function_target_names = {}
        self.hlsl_function_target_names_by_signature = {}
        self.hlsl_mapped_overload_names = set()
        used_names = set(groups)
        used_names.update(self.hlsl_function_name_aliases.values())
        used_names.update(reserved_names)

        def assign_target_names(function_name, source_groups):
            self.hlsl_mapped_overload_names.add(function_name)
            for source_key, declarations in sorted(
                source_groups.items(), key=lambda item: repr(item[0])
            ):
                signature_key = (function_name, source_key)
                target_name = self.hlsl_function_target_names_by_signature.get(
                    signature_key
                )
                if target_name is None:
                    suffix = self.hlsl_function_overload_suffix(source_key)
                    base_name = self.hlsl_sanitized_identifier(
                        f"{function_name}_{suffix}"
                    )
                    target_name = base_name
                    suffix_index = 2
                    while (
                        target_name in used_names
                        or target_name in self.HLSL_RESERVED_LOCAL_IDENTIFIER_NAMES
                    ):
                        target_name = f"{base_name}_{suffix_index}"
                        suffix_index += 1
                    used_names.add(target_name)
                    self.hlsl_function_target_names_by_signature[signature_key] = (
                        target_name
                    )
                for declaration in declarations:
                    self.hlsl_function_target_names[id(declaration)] = target_name
                representative = next(
                    (
                        declaration
                        for declaration in declarations
                        if getattr(declaration, "body", None) is not None
                    ),
                    declarations[0],
                )
                self.register_hlsl_function_target_metadata(target_name, representative)

        for function_name, overloads in groups.items():
            all_source_groups = {}
            for function in overloads:
                source_key = self.hlsl_source_function_signature(function)
                all_source_groups.setdefault(source_key, []).append(function)
            mapped_groups = {}
            for function in overloads:
                target_key = self.hlsl_mapped_function_signature_key(function)
                mapped_groups.setdefault(target_key, []).append(function)

            for mapped_overloads in mapped_groups.values():
                source_groups = {}
                for function in mapped_overloads:
                    source_key = self.hlsl_source_function_signature(function)
                    source_groups.setdefault(source_key, []).append(function)
                if len(source_groups) < 2:
                    continue

                if any(
                    self.hlsl_function_has_resource_contract(function)
                    for function in mapped_overloads
                ):
                    candidates = [
                        self.hlsl_function_candidate_signature(declarations[0])
                        for _source_key, declarations in sorted(
                            source_groups.items(), key=lambda item: repr(item[0])
                        )
                    ]
                    raise DirectXMappedOverloadError(
                        "DirectX codegen cannot preserve mapped overloads with "
                        f"resource parameters for '{function_name}'; candidates "
                        f"are {', '.join(candidates)}",
                        function_name=function_name,
                        candidates=candidates,
                        reason="resource-parameter-collision",
                    )

                assign_target_names(function_name, source_groups)

            has_pointer_overload = len(all_source_groups) > 1 and any(
                self.hlsl_resource_pointer_parameter(parameter)
                or self.hlsl_resource_pointer_array_parameter_type_node(parameter)
                is not None
                for function in overloads
                for parameter in (
                    getattr(function, "parameters", getattr(function, "params", []))
                    or []
                )
            )
            if has_pointer_overload:
                assign_target_names(function_name, all_source_groups)

    def hlsl_source_function_signature(self, function):
        return tuple(
            self.hlsl_source_type_signature(
                getattr(parameter, "param_type", getattr(parameter, "vtype", None))
            )
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
        )

    def hlsl_source_type_signature(self, value_type):
        type_name = self.type_name_string(value_type) or "void"
        return re.sub(r"\s+", " ", str(type_name).strip())

    def hlsl_mapped_function_signature_key(self, function):
        function_name = getattr(function, "name", None)
        parameters = []
        for parameter in (
            getattr(function, "parameters", getattr(function, "params", [])) or []
        ):
            raw_type = self.type_name_string(
                getattr(parameter, "param_type", getattr(parameter, "vtype", None))
            )
            mapped_type = self.map_resource_parameter_type_with_hint(
                raw_type, parameter, function_name
            )
            parameters.append(re.sub(r"\s+", " ", str(mapped_type).strip()))
        return tuple(parameters)

    def hlsl_function_has_resource_contract(self, function):
        function_name = getattr(function, "name", None)
        for parameter in (
            getattr(function, "parameters", getattr(function, "params", [])) or []
        ):
            raw_type = self.type_name_string(
                getattr(parameter, "param_type", getattr(parameter, "vtype", None))
            )
            mapped_type = self.map_resource_parameter_type_with_hint(
                raw_type, parameter, function_name
            )
            if self.is_resource_parameter_type(raw_type) or (
                self.is_hlsl_global_resource_type(mapped_type)
            ):
                return True
        return False

    def hlsl_function_overload_suffix(self, source_signature):
        parts = []
        for type_name in source_signature:
            part = re.sub(r"[^0-9A-Za-z]+", "_", type_name).strip("_")
            parts.append(part or "value")
        return "_".join(parts) or "void"

    def hlsl_mapped_overload_reserved_names(self, ast):
        names = set(self.global_variable_types)
        names.update(self.structs_by_name)
        for directive in getattr(ast, "preprocessors", []) or []:
            if getattr(directive, "directive", None) != "define":
                continue
            content = str(getattr(directive, "content", "") or "")
            match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", content)
            if match is not None:
                names.add(match.group(1))
        for node in self.walk_ast(ast):
            name = getattr(node, "name", None)
            if isinstance(name, str) and name:
                names.add(name)
        return names

    def register_hlsl_function_target_metadata(self, target_name, function):
        parameters = (
            getattr(function, "parameters", getattr(function, "params", [])) or []
        )
        self.function_return_types[target_name] = self.hlsl_function_return_type_name(
            self.type_name_string(getattr(function, "return_type", "void"))
        )
        self.function_parameter_names[target_name] = [
            getattr(parameter, "name", None) for parameter in parameters
        ]
        self.function_parameter_types[target_name] = [
            self.function_parameter_type_name(parameter) for parameter in parameters
        ]

    def hlsl_function_candidate_signature(self, function):
        parameters = ", ".join(self.hlsl_source_function_signature(function))
        return_type = self.hlsl_source_type_signature(
            getattr(function, "return_type", None)
        )
        return f"{getattr(function, 'name', '')}({parameters}) -> {return_type}"

    def hlsl_function_declaration_name(self, func, entry_name=None):
        if entry_name:
            return entry_name
        name = getattr(func, "name", None)
        target_name = self.hlsl_function_target_names.get(id(func))
        if target_name is None and name:
            source_key = self.hlsl_source_function_signature(func)
            target_name = self.hlsl_function_target_names_by_signature.get(
                (name, source_key)
            )
        if target_name is not None:
            return target_name
        return self.hlsl_function_spelling(name)

    def hlsl_decoded_materialized_name(self, name):
        if not isinstance(name, str) or "_u" not in name:
            return name
        decoded = name
        for encoded, character in (
            ("_u3c", "<"),
            ("_u2c", ","),
            ("_u3e", ">"),
            ("_u20", " "),
        ):
            decoded = decoded.replace(encoded, character)
        return decoded

    def hlsl_materialized_function_name(self, name):
        decoded = self.hlsl_decoded_materialized_name(name)
        if decoded == name:
            return name
        candidate = sanitize_type_name(decoded)
        available_names = set(self.function_parameter_names)
        available_names.update(self.function_private_pointer_parameters)
        return candidate if candidate in available_names else name

    def hlsl_function_call_name(self, name):
        if name in self.HLSL_NONUNIFORM_RESOURCE_INDEX_NAMES:
            return "NonUniformResourceIndex"
        name = self.hlsl_materialized_function_name(name)
        return self.hlsl_function_spelling(name)

    def hlsl_function_spelling(self, name):
        """The HLSL name of function ``name``: the name the generator gave it, or its own.

        ``hlsl_function_name_aliases`` holds only the user functions the generator renamed (an HLSL keyword, a
        clash with an intrinsic or an overload). Every other callee is written as spelled: it is an HLSL intrinsic
        or a function the program declares, and ``refuse_undeclared_functions`` and ``written_function_names``
        refuse any other name before a program is returned.
        """
        if name in self.hlsl_function_name_aliases:
            return self.hlsl_function_name_aliases[name]
        return name

    def hlsl_dependency_function_call_name(self, call):
        return self.hlsl_function_call_name(self.function_call_name(call))

    def resolve_hlsl_function_overload(
        self, function_name, arguments, *, call_node=None
    ):
        candidates = [
            function
            for function in self.hlsl_function_overloads_by_name.get(function_name, ())
            if len(
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
            == len(arguments)
        ]
        if not candidates:
            return None

        unique_candidates = {}
        for function in candidates:
            source_key = self.hlsl_source_function_signature(function)
            previous = unique_candidates.get(source_key)
            if previous is None or (
                getattr(previous, "body", None) is None
                and getattr(function, "body", None) is not None
            ):
                unique_candidates[source_key] = function
        candidates = list(unique_candidates.values())
        if len(candidates) == 1:
            return candidates[0]

        argument_types = [
            self.hlsl_source_expression_type(argument) for argument in arguments
        ]
        scored = []
        for function in candidates:
            parameters = (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
            score = 0
            compatible = True
            for argument_type, parameter in zip(argument_types, parameters):
                if argument_type is None:
                    continue
                parameter_type = getattr(
                    parameter, "param_type", getattr(parameter, "vtype", None)
                )
                match_score = self.hlsl_function_type_match_score(
                    argument_type, parameter_type
                )
                if match_score is None:
                    compatible = False
                    break
                score += match_score
            if compatible:
                scored.append((score, function))

        if scored:
            highest_score = max(score for score, _function in scored)
            winners = [function for score, function in scored if score == highest_score]
            if len(winners) == 1:
                return winners[0]
        else:
            winners = candidates

        argument_names = [
            (
                self.hlsl_source_type_signature(argument_type)
                if argument_type is not None
                else "unknown"
            )
            for argument_type in argument_types
        ]
        candidate_names = [
            self.hlsl_function_candidate_signature(function) for function in winners
        ]
        raise DirectXMappedOverloadError(
            "DirectX codegen cannot preserve source overload binding for "
            f"'{function_name}({', '.join(argument_names)})' after HLSL type "
            f"mapping; candidates are {', '.join(candidate_names)}",
            function_name=function_name,
            argument_types=argument_names,
            candidates=candidate_names,
            reason="call-binding-ambiguous",
            source_location=getattr(call_node, "source_location", None),
        )

    def hlsl_function_type_match_score(self, actual_type, parameter_type):
        if actual_type is None or parameter_type is None:
            return None
        actual_signature = self.hlsl_source_type_signature(actual_type)
        parameter_signature = self.hlsl_source_type_signature(parameter_type)
        if actual_signature == parameter_signature:
            return 8
        if self.map_type(actual_type) == self.map_type(parameter_type):
            return 2
        if self.is_scalar_value_type(actual_type) and self.is_scalar_value_type(
            parameter_type
        ):
            return 1
        return None

    def hlsl_source_expression_type(self, expression):
        if expression is None:
            return None
        if isinstance(expression, (VariableNode, IdentifierNode)) or (
            hasattr(expression, "__class__")
            and "IdentifierNode" in str(expression.__class__)
        ):
            name = getattr(expression, "name", None)
            return (
                self.local_variable_source_types.get(name)
                or self.local_variable_types.get(name)
                or self.global_variable_types.get(name)
                or self.expression_result_type(expression)
            )
        if isinstance(expression, ArrayAccessNode):
            array_type = self.hlsl_source_expression_type(expression.array)
            type_name = self.type_name_string(array_type)
            if not type_name:
                return None
            base_type, array_suffix = split_array_type_suffix(type_name)
            if array_suffix:
                remaining_suffix = re.sub(r"^\s*\[[^\]]*\]", "", array_suffix)
                return f"{base_type}{remaining_suffix}"
            if type_name.rstrip().endswith(("*", "&")):
                return type_name.rstrip()[:-1].strip()
            return self.expression_result_type(expression)
        if isinstance(expression, BinaryOpNode):
            operator = self.map_operator(
                getattr(expression, "op", getattr(expression, "operator", None))
            )
            if operator in {"==", "!=", "<", "<=", ">", ">=", "&&", "||"}:
                return self.expression_result_type(expression)
            left_type = self.hlsl_source_expression_type(expression.left)
            right_type = self.hlsl_source_expression_type(expression.right)
            if left_type is None:
                return right_type
            if right_type is None:
                return left_type
            if self.hlsl_source_type_signature(
                left_type
            ) == self.hlsl_source_type_signature(right_type):
                return left_type
            return self.expression_result_type(expression)
        if isinstance(expression, UnaryOpNode):
            operand_type = self.hlsl_source_expression_type(
                getattr(expression, "operand", getattr(expression, "expr", None))
            )
            if self.map_operator(expression.op) == "!":
                return self.hlsl_boolean_expression_result_type(operand_type)
            if expression.op == "*" and not getattr(expression, "is_postfix", False):
                pointee_type = self.hlsl_pointer_pointee_type_once(operand_type)
                if pointee_type is not None:
                    return pointee_type
            return operand_type
        if isinstance(expression, TernaryOpNode):
            true_type = self.hlsl_source_expression_type(expression.true_expr)
            false_type = self.hlsl_source_expression_type(expression.false_expr)
            if true_type is None:
                return false_type
            if false_type is None:
                return true_type
            if self.hlsl_source_type_signature(
                true_type
            ) == self.hlsl_source_type_signature(false_type):
                return true_type
            return self.expression_result_type(expression)
        if isinstance(expression, ConstructorNode):
            return self.type_name_string(getattr(expression, "constructor_type", None))
        if isinstance(expression, FunctionCallNode):
            function_name = self.function_call_name(expression)
            if self.hlsl_type_constructor_name(function_name) is not None:
                return function_name
            if (
                function_name not in getattr(self, "function_return_types", {})
                and self.hlsl_bfloat16_vector_lanes(function_name) is not None
            ):
                return function_name
            resolved = self.resolve_hlsl_function_overload(
                function_name,
                expression.arguments,
                call_node=expression,
            )
            if resolved is not None:
                return self.type_name_string(getattr(resolved, "return_type", None))
        return self.expression_result_type(expression)

    def collect_function_parameters(self, functions):
        parameters = []
        for func in functions or []:
            parameters.extend(getattr(func, "parameters", getattr(func, "params", [])))
        return parameters

    def function_parameter_type_name(self, parameter):
        if hasattr(parameter, "param_type"):
            return self.type_name_string(parameter.param_type)
        if hasattr(parameter, "vtype"):
            return self.type_name_string(parameter.vtype)
        if isinstance(parameter, (list, tuple)) and parameter:
            return self.type_name_string(parameter[0])
        return None

    def hlsl_parameter_raw_type(self, parameter):
        if hasattr(parameter, "param_type"):
            return parameter.param_type
        if hasattr(parameter, "vtype"):
            return parameter.vtype
        if isinstance(parameter, (list, tuple)) and parameter:
            return parameter[0]
        return "float"

    def collect_function_parameter_types(self, functions):
        parameter_types = {}
        for func in functions or []:
            function_name = getattr(func, "name", None)
            if not function_name:
                continue
            types = [
                self.function_parameter_type_name(parameter)
                for parameter in getattr(
                    func, "parameters", getattr(func, "params", [])
                )
            ]
            if (
                function_name in parameter_types
                and parameter_types[function_name] != types
            ):
                parameter_types[function_name] = []
            else:
                parameter_types[function_name] = types
        return parameter_types

    def collect_function_overloads(self, functions):
        """Map each function name to its list of (param_types, return_type).

        Unlike ``function_return_types`` (one entry per name), this preserves
        every overload so a call can be resolved to the matching signature. MLX
        headers overload names such as ``log1p`` for float / bfloat16 / complex64,
        and collapsing them to a single return type mis-typed float arithmetic as
        complex (producing invalid ``__crossgl_complex64_mul(complex, float)``).
        """
        overloads = {}
        for func in functions or []:
            function_name = getattr(func, "name", None)
            if not function_name:
                continue
            param_types = [
                self.function_parameter_type_name(parameter)
                for parameter in getattr(
                    func, "parameters", getattr(func, "params", [])
                )
            ]
            return_type = self.hlsl_function_return_type_name(
                self.type_name_string(getattr(func, "return_type", "void"))
            )
            overloads.setdefault(function_name, []).append((param_types, return_type))
        return overloads

    def hlsl_resolve_overloaded_return_type(self, func_name, args):
        overloads = getattr(self, "function_overloads", {}).get(func_name)
        if not overloads or len(overloads) < 2:
            return None
        distinct_returns = {return_type for _params, return_type in overloads}
        if len(distinct_returns) == 1:
            return next(iter(distinct_returns))

        argument_types = [
            self.map_type(self.type_name_string(self.expression_result_type(argument)))
            for argument in args or []
        ]
        if any(not argument_type for argument_type in argument_types):
            return None
        for param_types, return_type in overloads:
            if len(param_types) != len(argument_types):
                continue
            mapped_params = [
                self.map_type(self.type_name_string(param_type))
                for param_type in param_types
            ]
            if mapped_params == argument_types:
                return return_type
        return None

    def hlsl_called_function_names(self, func):
        called_names = set()
        current_name = getattr(func, "name", None)
        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            func_name = self.function_call_name(node)
            if (
                func_name
                and func_name != current_name
                and func_name in self.function_return_types
            ):
                called_names.add(func_name)
        return called_names

    def direct_hlsl_stage_parameter_dependencies(self, func, stage_parameters):
        local_names = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", getattr(func, "params", []))
            if getattr(param, "name", None)
        }
        for node in self.walk_ast(getattr(func, "body", [])):
            if isinstance(node, VariableNode) and getattr(node, "name", None):
                local_names.add(node.name)

        dependencies = set()
        for node in self.walk_ast(getattr(func, "body", [])):
            if not (hasattr(node, "__class__") and "Identifier" in str(node.__class__)):
                continue
            name = getattr(node, "name", None)
            if name and name in stage_parameters and name not in local_names:
                dependencies.add(name)
        return dependencies

    def collect_hlsl_function_stage_parameter_dependencies(
        self, ast, target_stage=None
    ):
        direct_dependencies = {}
        function_calls = {}
        parameter_nodes = {}

        for func in self.collect_functions(ast):
            func_name = getattr(func, "name", None)
            if func_name:
                direct_dependencies.setdefault(func_name, set())
                function_calls.setdefault(
                    func_name, self.hlsl_called_function_names(func)
                )

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue
            entry_point = getattr(stage, "entry_point", None)
            stage_parameters = {
                parameter.name: parameter
                for parameter in (
                    getattr(
                        entry_point, "parameters", getattr(entry_point, "params", [])
                    )
                    or []
                )
                if getattr(parameter, "name", None)
            }
            stage_parameters.update(
                self.collect_hlsl_stage_entry_local_dependencies(entry_point)
            )
            parameter_nodes.update(stage_parameters)

            stage_functions = list(getattr(stage, "local_functions", []) or [])
            if entry_point is not None:
                stage_functions.append(entry_point)
            for func in stage_functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue
                direct_dependencies[func_name] = (
                    self.direct_hlsl_stage_parameter_dependencies(
                        func, stage_parameters
                    )
                )
                function_calls[func_name] = self.hlsl_called_function_names(func)

        dependencies = {name: set(deps) for name, deps in direct_dependencies.items()}
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                before = set(dependencies.get(func_name, set()))
                for called_name in calls:
                    dependencies.setdefault(func_name, set()).update(
                        dependencies.get(called_name, set())
                    )
                if dependencies.get(func_name, set()) != before:
                    changed = True

        return {
            func_name: [
                parameter_nodes[name]
                for name in sorted(dependency_names)
                if name in parameter_nodes
            ]
            for func_name, dependency_names in dependencies.items()
            if dependency_names
        }

    def collect_hlsl_stage_entry_local_dependencies(self, entry_point):
        dependencies = {}
        if entry_point is None:
            return dependencies

        for node in self.hlsl_statement_body_items(getattr(entry_point, "body", [])):
            if not isinstance(node, VariableNode):
                continue
            name = getattr(node, "name", None)
            if name:
                dependencies[name] = node
        return dependencies

    def required_hlsl_stage_parameters(self, func_name):
        return self.function_stage_parameter_dependencies.get(func_name, [])

    def append_hlsl_parameter_declaration(self, params_str, declaration):
        if not params_str:
            return declaration
        return f"{params_str}, {declaration}"

    def hlsl_stage_parameter_declaration(self, parameter):
        raw_type = self.hlsl_parameter_raw_type(parameter)
        param_type = self.map_resource_parameter_type_with_hint(
            raw_type, parameter, None
        )
        param_type = self.directx_resource_declaration_type(param_type)
        param_type = self.apply_hlsl_parameter_qualifiers(param_type, parameter)
        if isinstance(parameter, VariableNode) and "const" not in getattr(
            parameter, "qualifiers", []
        ):
            param_type = f"inout {param_type}"
        return format_c_style_array_declaration(
            param_type, self.hlsl_declaration_identifier_name(parameter.name)
        )

    def append_required_hlsl_stage_parameter_parameters(
        self, params_str, func_name, existing_param_names=None
    ):
        existing_param_names = set(existing_param_names or set())
        for parameter in self.required_hlsl_stage_parameters(func_name):
            name = getattr(parameter, "name", None)
            if not name or name in existing_param_names:
                continue
            params_str = self.append_hlsl_parameter_declaration(
                params_str, self.hlsl_stage_parameter_declaration(parameter)
            )
        return params_str

    def used_hlsl_vertex_builtin_names(self, body):
        builtin_names = {"gl_VertexIndex", "gl_InstanceIndex", "gl_ViewIndex"}
        used_names = set()
        for node in self.walk_ast(body):
            class_name = node.__class__.__name__
            if "Identifier" not in class_name and class_name != "VariableNode":
                continue
            name = getattr(node, "name", "")
            base_name = name.split(".", 1)[0]
            if base_name in builtin_names:
                used_names.add(base_name)
        return used_names

    def required_hlsl_vertex_builtin_parameters(self, func):
        used_names = self.used_hlsl_vertex_builtin_names(getattr(func, "body", []))
        # Like SV_VertexID for gl_VertexIndex, SV_InstanceID does not include the draw's base instance, which
        # gl_InstanceIndex does (Vulkan) and SV_StartInstanceLocation (Shader Model 6.8) carries.
        builtin_parameters = [
            ("gl_VertexIndex", "uint", "SV_VertexID"),
            ("gl_InstanceIndex", "uint", "SV_InstanceID"),
            ("gl_ViewIndex", "uint", "SV_ViewID"),
        ]
        return [
            parameter for parameter in builtin_parameters if parameter[0] in used_names
        ]

    def explicit_hlsl_vertex_builtin_parameters(self, parameters):
        """Lower-case system value -> the vertex input that already carries it."""
        stage_parameters = {}
        system_values = ("SV_VertexID", "SV_InstanceID", "SV_ViewID")
        for parameter in parameters or []:
            name = getattr(parameter, "name", None)
            semantic = self.hlsl_canonical_semantic(self.semantic_from_node(parameter))
            if name and semantic:
                stage_parameters.setdefault(str(semantic).lower(), name)
                continue
            for system_value in system_values:
                member = self.hlsl_struct_member_with_semantic(parameter, system_value)
                if name and member is not None:
                    stage_parameters.setdefault(
                        system_value.lower(), f"{name}.{member}"
                    )
        return stage_parameters

    def explicit_hlsl_fragment_builtin_parameters(self, parameters):
        stage_parameters = {}
        for parameter in parameters or []:
            semantic = self.hlsl_canonical_semantic(self.semantic_from_node(parameter))
            if str(semantic or "").lower() == "sv_position":
                stage_parameters["gl_FragCoord"] = getattr(parameter, "name", None)
                continue
            # A stage input structure with an SV_Position member already carries the fragment
            # coordinate; a second SV_Position parameter would overlap it ("overlapping semantic index").
            member = self.hlsl_struct_member_with_semantic(parameter, "SV_Position")
            if member is not None and "gl_FragCoord" not in stage_parameters:
                stage_parameters["gl_FragCoord"] = f"{parameter.name}.{member}"
        return stage_parameters

    def hlsl_struct_member_with_semantic(self, parameter, semantic):
        raw_type = self.hlsl_parameter_raw_type(parameter)
        struct = self.structs_by_name.get(
            str(getattr(raw_type, "name", raw_type) or "")
        )
        if struct is None or not getattr(parameter, "name", None):
            return None
        for member in getattr(struct, "members", None) or []:
            member_semantic = self.hlsl_canonical_semantic(
                self.semantic_from_node(member)
            )
            if str(member_semantic or "").lower() == semantic.lower():
                return getattr(member, "name", None)
        return None

    def used_hlsl_fragment_builtin_names(self, body):
        builtin_names = {"gl_FragCoord", "gl_FragSizeEXT"}
        used_names = set()
        for node in self.walk_ast(body):
            class_name = node.__class__.__name__
            if "Identifier" not in class_name and class_name != "VariableNode":
                continue
            name = getattr(node, "name", "")
            base_name = name.split(".", 1)[0]
            if base_name in builtin_names:
                used_names.add(base_name)
        return used_names

    def hlsl_fragment_builtin_parameter_specs(self):
        return {
            "gl_FragCoord": ("_crossglFragCoord", "float4"),
            "gl_FragSizeEXT": ("_crossglFragSize", "uint2"),
        }

    def hlsl_fragment_builtin_parameter_type(self, builtin_name):
        spec = self.hlsl_fragment_builtin_parameter_specs().get(builtin_name)
        return spec[1] if spec is not None else None

    def direct_hlsl_fragment_builtin_dependencies(self, func):
        return set(
            self.used_hlsl_fragment_builtin_names(getattr(func, "body", []))
        ) & set(self.hlsl_fragment_builtin_parameter_specs())

    def collect_hlsl_function_fragment_builtin_dependencies(
        self, ast, target_stage=None
    ):
        direct_dependencies = {}
        function_calls = {}

        for func in self.collect_functions(ast):
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            direct_dependencies.setdefault(
                func_name,
                self.direct_hlsl_fragment_builtin_dependencies(func),
            )
            function_calls.setdefault(func_name, self.hlsl_called_function_names(func))

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name != "fragment" or not stage_matches(target_stage, stage_name):
                continue
            stage_functions = list(getattr(stage, "local_functions", []) or [])
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                stage_functions.append(entry_point)
            for func in stage_functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue
                direct_dependencies[func_name] = (
                    self.direct_hlsl_fragment_builtin_dependencies(func)
                )
                function_calls[func_name] = self.hlsl_called_function_names(func)

        dependencies = {name: set(deps) for name, deps in direct_dependencies.items()}
        changed = True
        while changed:
            changed = False
            for func_name, calls in function_calls.items():
                before = set(dependencies.get(func_name, set()))
                for called_name in calls:
                    dependencies.setdefault(func_name, set()).update(
                        dependencies.get(called_name, set())
                    )
                if dependencies.get(func_name, set()) != before:
                    changed = True

        return {
            func_name: sorted(dependency_names)
            for func_name, dependency_names in dependencies.items()
            if dependency_names
        }

    def collect_hlsl_fragment_builtin_parameter_names(self, ast, dependencies):
        functions_by_name = {
            getattr(func, "name", None): func
            for func in self.collect_functions(ast)
            if getattr(func, "name", None)
        }
        parameter_names = {}
        specs = self.hlsl_fragment_builtin_parameter_specs()
        for func_name, builtin_names in (dependencies or {}).items():
            func = functions_by_name.get(func_name)
            if func is None:
                continue
            reserved_names = self.collect_hlsl_function_identifier_names(func)
            reserved_names.update(self.HLSL_RESERVED_LOCAL_IDENTIFIER_NAMES)
            names = {}
            for builtin_name in builtin_names:
                spec = specs.get(builtin_name)
                if spec is None:
                    continue
                base_name, _param_type = spec
                name = self.hlsl_unique_local_identifier(base_name, reserved_names)
                reserved_names.add(name)
                names[builtin_name] = name
            if names:
                parameter_names[func_name] = names
        return parameter_names

    def required_hlsl_fragment_builtin_names(self, func_name):
        return list(
            (self.function_hlsl_fragment_builtin_dependencies or {}).get(
                func_name,
                [],
            )
        )

    def hlsl_fragment_builtin_parameter_name(self, func_name, builtin_name):
        return (
            (self.function_hlsl_fragment_builtin_parameter_names or {})
            .get(func_name, {})
            .get(builtin_name)
        )

    def append_required_hlsl_fragment_builtin_parameters(
        self, params_str, func_name, existing_param_names=None
    ):
        existing_param_names = set(existing_param_names or set())
        for builtin_name in self.required_hlsl_fragment_builtin_names(func_name):
            name = self.hlsl_fragment_builtin_parameter_name(func_name, builtin_name)
            param_type = self.hlsl_fragment_builtin_parameter_type(builtin_name)
            if not name or not param_type or name in existing_param_names:
                continue
            params_str = self.append_hlsl_parameter_declaration(
                params_str,
                f"{param_type} {name}",
            )
            existing_param_names.add(name)
        return params_str

    def required_hlsl_fragment_builtin_parameters(
        self, func, reserved_names=None, explicit_stage_builtins=None
    ):
        used_names = self.used_hlsl_fragment_builtin_names(getattr(func, "body", []))
        used_names.update(
            self.required_hlsl_fragment_builtin_names(getattr(func, "name", None))
        )
        reserved_names = set(reserved_names or ())
        explicit_stage_builtins = explicit_stage_builtins or {}
        builtin_parameters = [
            ("gl_FragCoord", "_crossglFragCoord", "float4", "SV_Position"),
            (
                "gl_FragSizeEXT",
                "_crossglShadingRate",
                "uint",
                "SV_ShadingRate",
            ),
        ]
        required_parameters = []
        for builtin_name, base_name, param_type, semantic in builtin_parameters:
            if (
                builtin_name not in used_names
                or builtin_name in explicit_stage_builtins
            ):
                continue
            name = self.hlsl_unique_local_identifier(base_name, reserved_names)
            reserved_names.add(name)
            prologue = None
            if builtin_name == "gl_FragSizeEXT":
                alias_name = self.hlsl_unique_local_identifier(
                    "_crossglFragSize", reserved_names
                )
                reserved_names.add(alias_name)
                self.required_hlsl_fragment_shading_rate_helper = True
                prologue = (
                    alias_name,
                    "uint2",
                    f"uint2 {alias_name} = "
                    f"CrossGLFragmentSizeFromShadingRate({name});",
                )
            required_parameters.append(
                (builtin_name, name, param_type, semantic, prologue)
            )
        return required_parameters

    def used_hlsl_compute_builtin_names(self, body):
        builtin_names = {
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_LocalInvocationIndex",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
            "threads_per_grid",
            "gl_SubgroupID",
            *self.HLSL_WAVE_BUILTIN_INTRINSICS,
        }
        used_names = set()
        for node in self.walk_ast(body):
            if not (hasattr(node, "__class__") and "Identifier" in str(node.__class__)):
                continue
            name = getattr(node, "name", "")
            base_name = name.split(".", 1)[0]
            if base_name in builtin_names:
                used_names.add(base_name)
        return used_names

    def required_hlsl_compute_builtin_parameters(self, func, execution_config=None):
        used_names = self.used_hlsl_compute_builtin_names(getattr(func, "body", []))
        subgroup_invocation_expression = self.HLSL_WAVE_BUILTIN_INTRINSICS[
            "gl_SubgroupInvocationID"
        ]
        subgroup_size_expression = self.HLSL_WAVE_BUILTIN_INTRINSICS["gl_SubgroupSize"]
        subgroup_count_expression = self.HLSL_WAVE_BUILTIN_INTRINSICS[
            "gl_NumSubgroups"
        ].replace(
            self.HLSL_WAVE_GROUP_SIZE_PLACEHOLDER,
            self.hlsl_compute_group_size_constant(execution_config),
        )
        if self.software_subgroup_width is not None:
            if "gl_SubgroupInvocationID" in used_names:
                subgroup_invocation_expression = (
                    self.hlsl_software_subgroup_builtin_expression(
                        "gl_SubgroupInvocationID", execution_config
                    )
                )
            subgroup_size_expression = self.hlsl_software_subgroup_builtin_expression(
                "gl_SubgroupSize", execution_config
            )
            subgroup_count_expression = self.hlsl_software_subgroup_builtin_expression(
                "gl_NumSubgroups", execution_config
            )
        builtin_parameters = [
            (
                "gl_GlobalInvocationID",
                "dispatchThreadID",
                "uint3",
                "SV_DispatchThreadID",
            ),
            ("gl_LocalInvocationID", "groupThreadID", "uint3", "SV_GroupThreadID"),
            ("gl_WorkGroupID", "groupID", "uint3", "SV_GroupID"),
            ("gl_LocalInvocationIndex", "groupIndex", "uint", "SV_GroupIndex"),
            (
                "gl_WorkGroupSize",
                None,
                None,
                None,
                self.hlsl_compute_workgroup_size_expression(execution_config),
            ),
            (
                "gl_NumWorkGroups",
                None,
                None,
                None,
                self.hlsl_compute_num_workgroups_expression(),
            ),
            (
                # Raw Metal [[threads_per_grid]] kept as-is by the frontend; HLSL has
                # no system value, so a body reference lowers to
                # num_workgroups * uint3(numthreads) when the uniform is available.
                "threads_per_grid",
                None,
                None,
                None,
                self.hlsl_compute_value_builtin_expression(
                    "threads_per_grid", execution_config
                ),
            ),
            (
                "gl_SubgroupInvocationID",
                None,
                None,
                None,
                subgroup_invocation_expression,
            ),
            (
                "gl_SubgroupSize",
                None,
                None,
                None,
                subgroup_size_expression,
            ),
            (
                "gl_NumSubgroups",
                None,
                None,
                None,
                subgroup_count_expression,
            ),
        ]
        reserved_names = set(self.local_variable_types)
        reserved_names.update(self.global_variable_types)
        reserved_names.update(self.current_identifier_reserved_names)
        reserved_names.update(self.current_identifier_aliases.values())

        # gl_SubgroupID has no direct HLSL system-value semantic. Explicit
        # software-subgroup mode derives a deterministic logical ID from
        # SV_GroupIndex; physical-wave mode cannot divide by WaveGetLaneCount()
        # because Direct3D does not promise contiguous flattened group indices in
        # each wave, so it allocates one collision-safe ID in a uniform prologue.
        subgroup_id_used = "gl_SubgroupID" in used_names
        group_index_name = None

        required_parameters = []
        for parameter in builtin_parameters:
            builtin_name, base_name, param_type, semantic, *alias = parameter
            needed = builtin_name in used_names
            if builtin_name == "gl_LocalInvocationIndex" and subgroup_id_used:
                needed = True
            if not needed:
                continue
            alias_expression = alias[0] if alias else None
            if alias_expression is not None:
                required_parameters.append(
                    (builtin_name, None, None, None, alias_expression)
                )
                continue
            name = self.hlsl_unique_local_identifier(base_name, reserved_names)
            reserved_names.add(name)
            if builtin_name == "gl_LocalInvocationIndex":
                group_index_name = name
            required_parameters.append((builtin_name, name, param_type, semantic, None))

        if subgroup_id_used:
            if group_index_name is None:
                group_index_name = self.hlsl_unique_local_identifier(
                    "groupIndex", reserved_names
                )
                reserved_names.add(group_index_name)
                required_parameters.append(
                    (
                        "_crossgl_subgroup_group_index",
                        group_index_name,
                        "uint",
                        "SV_GroupIndex",
                        None,
                    )
                )
            subgroup_id_alias = self.hlsl_compute_subgroup_id_expression(
                func,
                execution_config,
                group_index_name=group_index_name,
            )
            required_parameters.append(
                ("gl_SubgroupID", None, None, None, subgroup_id_alias)
            )
        return required_parameters

    def hlsl_compute_workgroup_size_expression(self, execution_config=None):
        x, y, z = hlsl_entry_local_size(execution_config)
        return f"uint3({x}, {y}, {z})"

    def hlsl_compute_group_size_constant(self, execution_config=None):
        """Return the flattened thread-group size (product of the numthreads dims)
        as an HLSL unsigned expression. When every dimension is an integer literal
        the product is folded to a single ``<n>u`` constant; otherwise the
        dimensions are multiplied at runtime."""
        dims = hlsl_entry_local_size(execution_config)
        try:
            product = 1
            for dim in dims:
                product *= int(str(dim))
            return f"{product}u"
        except (TypeError, ValueError):
            return "(" + " * ".join(f"uint({dim})" for dim in dims) + ")"

    def hlsl_compute_num_workgroups_expression(self):
        uniform = self.hlsl_compute_num_workgroups_uniform or {}
        return uniform.get("member_name")

"""Program-wide collection passes of the Slang generator: user functions, symbols, structs, enums, stage-parameter dependencies, lowered struct resources and explicit samplers."""

import re

from ..ast import (
    ArrayAccessNode,
    EnumNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    MemberAccessNode,
    StructNode,
    VariableNode,
)
from ..refusals import SlangRefusal
from .array_utils import format_c_style_array_declaration
from .enum_utils import sanitize_type_name
from .generic_function_utils import (
    collect_unresolved_generic_function_call_names,
    generic_function_parameters,
    iter_function_nodes,
    raise_unresolved_generic_function_call,
)
from .slang_projected_compare import SLANG_PROJECTED_COMPARE_FORMS
from .stage_utils import normalize_stage_name


class SlangProgramSymbolsMixin:
    """Program-collection members of ``SlangCodeGen``."""

    def finish_generation(self, result, outermost):
        if not outermost:
            return result

        struct_helpers = self.emit_struct_dependent_helper_functions()
        marker = self.slang_struct_dependent_helper_marker()
        if marker in result:
            result = result.replace(marker, struct_helpers, 1)
            struct_helpers = ""
        helpers = self.emit_helper_functions()
        self._generating = False
        constants = self.slang_workgroup_size_constants().declarations()
        generated = constants + helpers + struct_helpers + result
        self.reject_unresolved_generic_output(generated)
        return generated

    def reject_unsupported_generic_functions(self, ast_node):
        """Reject generic functions before emitting non-compilable Slang code."""
        functions = list(iter_function_nodes(ast_node)) + list(
            (self.generic_function_specializations or {}).values()
        )
        unresolved = collect_unresolved_generic_function_call_names(self, functions)
        if unresolved:
            raise_unresolved_generic_function_call(self, unresolved[0], "Slang")

    def reject_unresolved_generic_output(self, result):
        specialized_source_names = {
            key[0]
            for key in self.generic_function_specializations or {}
            if isinstance(key, tuple) and key
        }
        generic_names = {
            param
            for funcs in (self.generic_function_definitions or {}).values()
            for func in funcs
            if getattr(func, "name", None) not in specialized_source_names
            for param in generic_function_parameters(func)
        }
        for name in sorted(generic_names):
            if re.search(rf"\b{re.escape(name)}\b", result):
                raise SlangRefusal(
                    "Slang codegen cannot emit unresolved generic parameter "
                    f"'{name}'; specialize generic declarations before Slang "
                    "generation"
                )

    def emit_helper_functions(self):
        helpers = self.hlsl_matrix_orientation.helper_code()
        helpers += self.slang_buffer_length.code()
        if self.helper_functions:
            helpers += "\n\n".join(self.helper_functions.values()) + "\n\n"
        return helpers

    def slang_struct_dependent_helper_marker(self):
        return "/* __crossgl_slang_struct_dependent_helpers__ */\n"

    def emit_struct_dependent_helper_functions(self):
        helpers = self.generate_slang_glsl_buffer_aggregate_load_helpers()
        helpers += self.generate_slang_byteaddress_atomic_helpers()
        array_values = self.slang_array_values.declarations()
        # Generated last: the helpers above call the three-component accessors too.
        return array_values + self.slang_three_component_access.declarations() + helpers

    def collect_user_function_names(self, node):
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                names.add(current.name)
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        names.discard(None)
        return names

    def collect_user_functions_by_name(self, node):
        functions = {}
        ambiguous_names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                name = getattr(current, "name", None)
                if name:
                    if name in functions and functions[name] is not current:
                        ambiguous_names.add(name)
                    else:
                        functions[name] = current
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        for name in ambiguous_names:
            functions.pop(name, None)
        functions.pop(None, None)
        return functions

    def collect_user_symbol_names(self, node):
        names = set()

        def add_name(current):
            name = getattr(current, "name", None)
            if name:
                names.add(name)

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, (FunctionNode, StructNode)):
                add_name(current)
            for attr in ("global_variables", "cbuffers"):
                for declaration in getattr(current, attr, []) or []:
                    add_name(declaration)
            for function in getattr(current, "functions", []) or []:
                collect(function)
            for function in getattr(current, "local_functions", []) or []:
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    for declaration in getattr(stage, "local_variables", []) or []:
                        add_name(declaration)
                    collect(stage)

        collect(node)
        names.discard(None)
        return names

    def collect_user_function_return_types(self, node):
        return_types = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                return_type = getattr(current, "return_type", None)
                return_types[current.name] = (
                    self.convert_type_node_to_string(return_type)
                    if return_type is not None
                    else "void"
                )
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return_types.pop(None, None)
        return return_types

    def collect_user_function_parameter_types(self, node):
        parameter_types = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                type_names = self.function_parameter_type_names(current)
                if (
                    current.name in parameter_types
                    and parameter_types[current.name] != type_names
                ):
                    parameter_types[current.name] = []
                else:
                    parameter_types[current.name] = type_names
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        parameter_types.pop(None, None)
        return parameter_types

    def function_parameter_type_names(self, function):
        names = []
        for parameter in getattr(
            function, "parameters", getattr(function, "params", [])
        ):
            if hasattr(parameter, "param_type"):
                names.append(self.convert_type_node_to_string(parameter.param_type))
            elif hasattr(parameter, "vtype"):
                names.append(str(parameter.vtype))
            elif isinstance(parameter, (list, tuple)) and parameter:
                names.append(self.type_name_string(parameter[0]))
            else:
                names.append(None)
        return names

    def slang_called_function_names(self, func):
        called_names = set()
        current_name = getattr(func, "name", None)
        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, FunctionCallNode):
                continue
            func_name = self.function_call_simple_callee_name(node)
            if (
                func_name
                and func_name != current_name
                and func_name in self.user_function_names
            ):
                called_names.add(func_name)
        return called_names

    def direct_slang_stage_parameter_dependencies(self, func, stage_parameters):
        local_names = {
            getattr(param, "name", None)
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []
            if getattr(param, "name", None)
        }
        for node in self.walk_ast(getattr(func, "body", [])):
            if isinstance(node, VariableNode) and getattr(node, "name", None):
                local_names.add(node.name)

        dependencies = set()
        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, IdentifierNode):
                continue
            name = getattr(node, "name", None)
            if name and name in stage_parameters and name not in local_names:
                dependencies.add(name)
        return dependencies

    def collect_slang_function_stage_parameter_dependencies(self, ast):
        direct_dependencies = {}
        function_calls = {}
        parameter_nodes = {}

        for func in getattr(ast, "functions", []) or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            direct_dependencies.setdefault(func_name, set())
            function_calls.setdefault(func_name, self.slang_called_function_names(func))

        stages = getattr(ast, "stages", {}) or {}
        if isinstance(stages, dict):
            for _stage_type, stage in stages.items():
                entry_point = getattr(stage, "entry_point", None)
                stage_parameters = {
                    parameter.name: parameter
                    for parameter in (
                        getattr(
                            entry_point,
                            "parameters",
                            getattr(entry_point, "params", []),
                        )
                        or []
                    )
                    if getattr(parameter, "name", None)
                }
                parameter_nodes.update(stage_parameters)

                stage_functions = list(getattr(stage, "local_functions", []) or [])
                if entry_point is not None:
                    stage_functions.append(entry_point)
                for func in stage_functions:
                    func_name = getattr(func, "name", None)
                    if not func_name:
                        continue
                    direct_dependencies[func_name] = (
                        self.direct_slang_stage_parameter_dependencies(
                            func, stage_parameters
                        )
                    )
                    function_calls[func_name] = self.slang_called_function_names(func)

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

    def required_slang_stage_parameters(self, func_name):
        return self.function_stage_parameter_dependencies.get(func_name, [])

    def slang_stage_parameter_dependency_declaration(self, parameter):
        param_type_name = self.slang_parameter_type_name(parameter) or "float"
        param_type = self.map_resource_type_with_format(param_type_name, parameter)
        declaration = format_c_style_array_declaration(param_type, parameter.name)
        return self.slang_parameter_qualifier_prefix(parameter) + declaration

    def collect_user_structs(self, node):
        structs = []
        seen = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, StructNode):
                node_id = id(current)
                if node_id not in seen:
                    seen.add(node_id)
                    structs.append(current)
            for struct in getattr(current, "structs", []) or []:
                collect(struct)
            for struct in getattr(current, "local_structs", []) or []:
                collect(struct)
            for function in getattr(current, "functions", []) or []:
                collect(function)
            for function in getattr(current, "local_functions", []) or []:
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return structs

    def slang_stage_entry_functions(self, ast, expected_stage_name):
        functions = []
        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if getattr(func, "qualifiers", None)
                else getattr(func, "qualifier", None)
            )
            if normalize_stage_name(qualifier) == expected_stage_name:
                functions.append(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            if normalize_stage_name(stage_type) != expected_stage_name:
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                functions.append(entry_point)
        return functions

    def collect_slang_vertex_entry_output_struct_names(self, ast):
        struct_names = set()
        for func in self.slang_stage_entry_functions(ast, "vertex"):
            return_type = self.type_name_string(
                getattr(func, "return_type", getattr(func, "vtype", None))
            )
            if not return_type:
                continue
            base_type = return_type.split("<", 1)[0].split("[", 1)[0].strip()
            if base_type in self.structs_by_name:
                struct_names.add(base_type)
        return struct_names

    def collect_slang_vertex_entry_input_struct_names(self, ast):
        struct_names = set()
        for func in self.slang_stage_entry_functions(ast, "vertex"):
            for parameter in (
                getattr(func, "parameters", getattr(func, "params", [])) or []
            ):
                struct_name = self.slang_parameter_user_struct_type(parameter)
                if struct_name is not None:
                    struct_names.add(struct_name)
        return struct_names

    def collect_slang_fragment_entry_input_struct_names(self, ast):
        struct_names = set()
        for func in self.slang_stage_entry_functions(ast, "fragment"):
            for parameter in (
                getattr(func, "parameters", getattr(func, "params", [])) or []
            ):
                struct_name = self.slang_parameter_user_struct_type(parameter)
                if struct_name is not None:
                    struct_names.add(struct_name)
        return struct_names

    def collect_slang_fragment_entry_output_struct_names(self, ast):
        struct_names = set()
        for func in self.slang_stage_entry_functions(ast, "fragment"):
            return_type_name = self.type_name_string(
                getattr(func, "return_type", getattr(func, "vtype", None))
            )
            if not return_type_name:
                continue
            base_type = return_type_name.split("<", 1)[0].split("[", 1)[0].strip()
            if base_type in self.user_structs_by_name:
                struct_names.add(base_type)
        return struct_names

    def collect_slang_lowered_struct_resource_members(self, structs):
        lowered_members = {}
        used_global_names = set(self.user_symbol_names)
        for struct in structs or []:
            struct_name = getattr(struct, "name", None)
            if not struct_name:
                continue
            for member in getattr(struct, "members", []) or []:
                raw_type = self.slang_struct_member_type_name(member)
                if not self.slang_struct_member_is_resource(raw_type):
                    continue
                member_name = getattr(member, "name", None)
                if not member_name:
                    continue
                lowered_members.setdefault(struct_name, {})[member_name] = {
                    "member": member,
                    "type": raw_type,
                    "mapped_type": self.map_resource_type_with_format(raw_type, member),
                    "global_name": self.slang_lowered_struct_resource_global_name(
                        struct_name, member_name, used_global_names
                    ),
                }
        return lowered_members

    def slang_lowered_struct_resource_global_name(
        self, struct_name, member_name, used_global_names
    ):
        base_name = sanitize_type_name(f"{struct_name}_{member_name}") or member_name
        global_name = base_name
        suffix = 1
        while global_name in used_global_names:
            global_name = f"{base_name}_{suffix}"
            suffix += 1
        used_global_names.add(global_name)
        return global_name

    def slang_struct_member_is_resource(self, raw_type):
        if not raw_type:
            return False
        mapped_type = self.map_resource_type_with_format(raw_type)
        return (
            self.is_sampled_texture_resource_type(raw_type)
            or self.is_storage_image_type(raw_type)
            or self.is_sampler_state_type(raw_type)
            or self.is_buffer_resource_type(mapped_type)
        )

    def slang_lowered_struct_resource_member_info(self, struct_name, member_name):
        if not struct_name or not member_name:
            return None
        return self.slang_lowered_struct_resource_members.get(struct_name, {}).get(
            member_name
        )

    def slang_should_lower_struct_resource_member(self, struct_name, member_name):
        return (
            self.slang_lowered_struct_resource_member_info(struct_name, member_name)
            is not None
        )

    def generate_slang_lowered_struct_resource_member_globals(self):
        declarations = []
        seen = {}
        for members in self.slang_lowered_struct_resource_members.values():
            for member_info in members.values():
                global_name = member_info["global_name"]
                raw_type = member_info["type"]
                mapped_type = member_info["mapped_type"]
                previous_type = seen.get(global_name)
                if previous_type is not None:
                    if previous_type != mapped_type:
                        raise SlangRefusal(
                            "Conflicting Slang lowered struct resource member "
                            f"'{global_name}' has both {previous_type} and "
                            f"{mapped_type}"
                        )
                    continue
                seen[global_name] = mapped_type
                self.register_variable_type(
                    global_name, raw_type, member_info["member"]
                )
                declaration = self.format_declaration(
                    raw_type, global_name, member_info["member"]
                )
                declaration = self.apply_slang_resource_binding_decorations(
                    declaration, member_info["member"], raw_type, auto_assign=True
                )
                declarations.append(f"{declaration};\n")
        return "".join(declarations)

    def collect_user_enums(self, node):
        enums = []
        seen = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, EnumNode):
                node_id = id(current)
                if node_id not in seen:
                    seen.add(node_id)
                    enums.append(current)
            for struct in getattr(current, "structs", []) or []:
                collect(struct)
            for function in getattr(current, "functions", []) or []:
                collect(function)
            for function in getattr(current, "local_functions", []) or []:
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return enums

    def collect_enum_struct_member_types(self, enums):
        member_types = {}
        for enum in enums or []:
            fields = {"variant": "int"}
            for variants in self.enum_struct_variant_fields.get(enum.name, {}).values():
                for field_name, field_type in variants.items():
                    fields[field_name] = field_type
            member_types[enum.name] = fields
        return member_types

    def collect_user_struct_names(self, node):
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, StructNode):
                names.add(current.name)
            for struct in getattr(current, "structs", []) or []:
                collect(struct)
            for function in getattr(current, "functions", []) or []:
                collect(function)
            for function in getattr(current, "local_functions", []) or []:
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        names.discard(None)
        return names

    def collect_slang_resource_declaration_nodes(self, node):
        declarations = []

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return

            declarations.extend(getattr(current, "global_variables", []) or [])

            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage_type, stage in stages.items():
                    stage_name = self.get_stage_name(stage_type)
                    declarations.extend(
                        self.slang_stage_global_local_variables(
                            stage_name, getattr(stage, "local_variables", [])
                        )
                    )

        collect(node)
        return declarations

    def expression_root_identifier_name(self, node):
        if isinstance(node, IdentifierNode):
            return node.name
        if isinstance(node, ArrayAccessNode):
            return self.expression_root_identifier_name(
                getattr(node, "array", getattr(node, "array_expr", None))
            )
        if isinstance(node, MemberAccessNode):
            return self.expression_root_identifier_name(node.object)
        return None

    def collect_explicit_sampler_resource_names(self, node):
        declared_types = self.collect_declared_type_names(node)
        explicit_texture_names = set()
        implicit_texture_names = set()
        explicit_comparison_pairs = []
        sampled_calls = {
            "texture",
            "textureLod",
            "textureGrad",
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
            "textureQueryLod",
        }
        compare_calls = {
            "textureCompare",
            "textureCompareLod",
            "textureCompareLodOffset",
            "textureCompareGrad",
            "textureCompareOffset",
            "textureGatherCompare",
            "textureGatherCompareOffset",
            *SLANG_PROJECTED_COMPARE_FORMS,
        }

        for current in self.walk_ast(node):
            if not isinstance(current, FunctionCallNode):
                continue
            func_expr = getattr(current, "function", None)
            if func_expr is None:
                func_expr = getattr(current, "name", None)
            if hasattr(func_expr, "name") and getattr(func_expr, "name", None):
                callee = func_expr.name
            else:
                callee = func_expr if isinstance(func_expr, str) else None
            if callee not in sampled_calls and callee not in compare_calls:
                continue

            args = getattr(current, "args", [])
            if len(args) < 2:
                continue
            texture_name = self.expression_root_identifier_name(args[0])
            sampler_name = self.expression_root_identifier_name(args[1])
            has_explicit_sampler = self.is_sampler_state_type(
                declared_types.get(sampler_name)
            )
            if not has_explicit_sampler:
                if texture_name:
                    implicit_texture_names.add(texture_name)
                continue
            if len(args) < 3:
                continue
            if texture_name:
                explicit_texture_names.add(texture_name)
            if callee in compare_calls and sampler_name:
                explicit_comparison_pairs.append((texture_name, sampler_name))

        separated_texture_names = explicit_texture_names - implicit_texture_names
        comparison_sampler_names = {
            sampler_name
            for texture_name, sampler_name in explicit_comparison_pairs
            if texture_name in separated_texture_names
        }
        return separated_texture_names, comparison_sampler_names

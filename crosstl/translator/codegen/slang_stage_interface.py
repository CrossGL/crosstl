"""Stage generation of the Slang generator: stage interfaces, merged parameters, patch-constant functions and tessellation factors."""

from ..ast import (
    EnumNode,
    ParameterNode,
)
from ..refusals import SlangRefusal
from ..semantics.builtin_values import glsl_builtin_name
from .array_utils import split_array_type_suffix
from .generic_function_utils import (
    generic_function_emission_list,
    generic_function_parameters,
)


class SlangStageInterfaceMixin:
    """Stage-interface members of ``SlangCodeGen``."""

    def generate_stage(self, stage_type, stage):
        """Render one staged entry point and its local functions."""
        stage_name = self.get_stage_name(stage_type)
        result = f"// {stage_name.title()} Shader\n"

        # Structures declared in the stage block, before the variables and functions that use them.
        for struct in getattr(stage, "local_structs", None) or []:
            if isinstance(struct, EnumNode):
                continue
            struct_code = self.generate_struct(struct)
            if struct_code:
                result += struct_code + "\n\n"

        local_variables = getattr(stage, "local_variables", [])
        for local_var in self.slang_stage_global_local_variables(
            stage_name, local_variables
        ):
            result += self.generate_global_variable(local_var)

        entry_point = getattr(stage, "entry_point", None)
        local_functions = getattr(stage, "local_functions", [])
        self.validate_slang_stage_patch_constant_function_shapes(
            entry_point, local_functions, stage_name
        )
        patch_constant_function_names = self.slang_stage_patch_constant_function_names(
            entry_point, stage_name
        )
        forward_declaration_ids = self.slang_forward_declaration_function_ids(
            local_functions
        )
        for func in local_functions:
            if id(func) in forward_declaration_ids:
                continue
            if generic_function_parameters(func):
                for specialized_func in generic_function_emission_list(self, func):
                    result += self.generate_function(specialized_func) + "\n\n"
                continue
            if getattr(func, "name", None) in patch_constant_function_names:
                result += (
                    self.generate_function(
                        func,
                        shader_type=stage_name,
                        emit_stage_decorations=False,
                        stage_role="patch_constant",
                    )
                    + "\n\n"
                )
            else:
                result += self.generate_function(func) + "\n\n"

        if entry_point is not None:
            # The stage's layout qualifiers, such as layout(early_fragment_tests) in;, decorate its entry point.
            self.current_slang_stage_node = stage
            try:
                result += self.generate_function(
                    entry_point,
                    shader_type=stage_name,
                    execution_config=getattr(stage, "execution_config", None),
                    entry_name=self.stage_entry_name_overrides.get(id(entry_point)),
                    extra_parameters=self.slang_stage_interface_parameters(
                        stage_name, local_variables
                    ),
                )
            finally:
                self.current_slang_stage_node = None
            result += "\n\n"

        return result

    def slang_forward_declaration_function_ids(self, functions):
        """Return prototype-only stage functions that have a real definition."""
        body_function_names = {
            getattr(func, "name", None)
            for func in functions or []
            if getattr(func, "name", None)
            and self.get_statements(getattr(func, "body", []))
        }
        if not body_function_names:
            return set()

        return {
            id(func)
            for func in functions or []
            if getattr(func, "name", None) in body_function_names
            and not self.get_statements(getattr(func, "body", []))
        }

    def slang_stage_interface_local_variables(self, stage_name, local_variables):
        """Return stage-local declarations that lower to Slang entry parameters."""
        shader_stage = self.slang_shader_stage_name(stage_name)
        if shader_stage not in self.slang_ray_stage_types():
            return []

        return [
            local_var
            for local_var in local_variables or []
            if self.slang_ray_semantic_role(local_var, stage_name) is not None
        ]

    def slang_stage_bound_builtin_input_ids(self, stage_name, local_variables):
        """The stage-level ``@input`` built-ins that the entry point binds itself, by the GLSL name it reads.

        The SPIR-V importer declares a compute stage's built-in inputs (``uint3 gl_GlobalInvocationID @input
        @gl_GlobalInvocationID;``), and the entry point reads them by that name, which Slang binds as an entry
        parameter (``SV_DispatchThreadID``) or an intrinsic (``WorkgroupCount()``). Declared as well, the variable
        became a global without ``static``, which Slang makes an implicit uniform shader parameter: the artifact
        declared a constants block the host does not provide.
        """
        bound = set(self.slang_implicit_stage_parameter_candidates(stage_name))
        bound.update(
            self.slang_stage_intrinsic_builtin_candidates(
                self.slang_shader_stage_name(stage_name)
            )
        )
        ids = set()
        for local_var in local_variables or []:
            attributes = getattr(local_var, "attributes", None) or []
            if getattr(local_var, "initial_value", None) is not None or not any(
                str(getattr(attribute, "name", "")).lower() == "input"
                for attribute in attributes
            ):
                continue
            name = getattr(local_var, "name", None)
            if name in bound and any(
                glsl_builtin_name(attribute) == name for attribute in attributes
            ):
                ids.add(id(local_var))
        return ids

    def slang_stage_global_local_variables(self, stage_name, local_variables):
        interface_ids = {
            id(local_var)
            for local_var in self.slang_stage_interface_local_variables(
                stage_name, local_variables
            )
        }
        interface_ids |= self.slang_stage_bound_builtin_input_ids(
            stage_name, local_variables
        )
        return [
            local_var
            for local_var in local_variables or []
            if id(local_var) not in interface_ids
        ]

    def slang_stage_interface_parameters(self, stage_name, local_variables):
        parameters = []
        for local_var in self.slang_stage_interface_local_variables(
            stage_name, local_variables
        ):
            if getattr(local_var, "initial_value", None) is not None:
                raise SlangRefusal(
                    "Slang ray stage interface variable "
                    f"'{local_var.name}' cannot have an initializer"
                )

            parameter = ParameterNode(
                name=local_var.name,
                param_type=getattr(local_var, "var_type", getattr(local_var, "vtype")),
                attributes=list(getattr(local_var, "attributes", []) or []),
                qualifiers=list(getattr(local_var, "qualifiers", []) or []),
                source_location=getattr(local_var, "source_location", None),
            )
            parameter.semantic = getattr(local_var, "semantic", None)
            parameter.add_annotation("slang_stage_local_interface", True)
            parameters.append(parameter)
        return parameters

    def slang_merge_function_parameters(
        self, param_list, extra_parameters, shader_type=None
    ):
        if not extra_parameters:
            return param_list

        merged_parameters = list(param_list or [])
        existing_names = {
            getattr(param, "name", None)
            for param in merged_parameters
            if getattr(param, "name", None)
        }
        for parameter in extra_parameters:
            if parameter.name in existing_names:
                stage_name = shader_type or "function"
                raise SlangRefusal(
                    f"Slang {stage_name} stage interface variable "
                    f"'{parameter.name}' duplicates an entry parameter"
                )
            existing_names.add(parameter.name)
            merged_parameters.append(parameter)
        return merged_parameters

    def slang_stage_patch_constant_function_names(self, entry_point, stage_name):
        if entry_point is None or self.slang_shader_stage_name(stage_name) != "hull":
            return set()

        arguments = self.slang_stage_attribute_arguments(
            entry_point, "patchconstantfunc"
        )
        if len(arguments) != 1:
            return set()

        function_name = self.slang_stage_attribute_value_to_string(arguments[0])
        return {function_name} if function_name else set()

    def validate_slang_stage_patch_constant_function_shapes(
        self, entry_point, local_functions, stage_name
    ):
        if entry_point is None or self.slang_shader_stage_name(stage_name) != "hull":
            return

        patch_constant_function_names = self.slang_stage_patch_constant_function_names(
            entry_point, stage_name
        )
        if not patch_constant_function_names:
            return

        local_functions_by_name = {
            getattr(func, "name", None): func for func in local_functions or []
        }
        hull_input_patches = self.slang_patch_parameters(
            getattr(entry_point, "parameters", getattr(entry_point, "params", [])),
            "InputPatch",
        )

        for function_name in patch_constant_function_names:
            patch_constant_function = local_functions_by_name.get(function_name)
            if patch_constant_function is None:
                continue

            return_type_name = self.convert_type_node_to_string(
                getattr(patch_constant_function, "return_type", "void")
            )
            if self.convert_type(return_type_name) == "void":
                raise SlangRefusal(
                    "Slang tessellation_control stage patchconstantfunc "
                    f"'{function_name}' requires a non-void return type"
                )

            patch_constant_input_patches = self.slang_patch_parameters(
                getattr(
                    patch_constant_function,
                    "parameters",
                    getattr(patch_constant_function, "params", []),
                ),
                "InputPatch",
            )
            if len(patch_constant_input_patches) > 1:
                raise SlangRefusal(
                    "Slang tessellation_control stage patchconstantfunc "
                    f"'{function_name}' requires at most one "
                    "InputPatch<..., N> parameter"
                )

            if patch_constant_input_patches and hull_input_patches:
                _patch_param, patch_shape = patch_constant_input_patches[0]
                _hull_param, hull_shape = hull_input_patches[0]
                if patch_shape != hull_shape:
                    patch_type = self.format_slang_patch_shape(
                        "InputPatch", patch_shape
                    )
                    hull_type = self.format_slang_patch_shape("InputPatch", hull_shape)
                    raise SlangRefusal(
                        "Slang tessellation_control stage patchconstantfunc "
                        f"'{function_name}' {patch_type} must match hull entry "
                        f"{hull_type}"
                    )

            self.validate_slang_patch_constant_tess_factor_semantics(
                entry_point, patch_constant_function, function_name
            )

    def validate_slang_patch_constant_tess_factor_semantics(
        self, hull_entry, patch_function, function_name
    ):
        return_struct = self.slang_return_struct(patch_function)
        if return_struct is None:
            return

        semantics = self.slang_struct_semantics(return_struct)
        if "sv_tessfactor" not in semantics:
            raise SlangRefusal(
                "Slang tessellation_control stage patchconstantfunc "
                f"'{function_name}' must return a struct containing "
                "SV_TessFactor"
            )

        domain = self.normalized_slang_stage_attribute_argument(hull_entry, "domain")
        domain = self.canonical_slang_tessellation_domain(domain)
        if domain in {"tri", "quad"} and "sv_insidetessfactor" not in semantics:
            raise SlangRefusal(
                "Slang tessellation_control stage patchconstantfunc "
                f"'{function_name}' must return a struct containing "
                f"SV_InsideTessFactor for {domain} domains"
            )

        self.validate_slang_tess_factor_member_types(return_struct, function_name)

        factor_counts = self.slang_struct_semantic_member_counts(return_struct)
        expected_outer_counts = {
            "tri": 3,
            "quad": 4,
            "isoline": 2,
        }
        expected_inner_counts = {
            "tri": 1,
            "quad": 2,
        }
        expected_outer_count = expected_outer_counts.get(domain)
        expected_inner_count = expected_inner_counts.get(domain)
        outer_count = factor_counts.get("sv_tessfactor")
        inner_count = factor_counts.get("sv_insidetessfactor")

        if (
            expected_outer_count is not None
            and outer_count is not None
            and outer_count != expected_outer_count
        ):
            raise SlangRefusal(
                "Slang tessellation_control stage patchconstantfunc "
                f"'{function_name}' must return {expected_outer_count} "
                f"SV_TessFactor value(s) for {domain} domains"
            )
        if (
            expected_inner_count is not None
            and inner_count is not None
            and inner_count != expected_inner_count
        ):
            raise SlangRefusal(
                "Slang tessellation_control stage patchconstantfunc "
                f"'{function_name}' must return {expected_inner_count} "
                f"SV_InsideTessFactor value(s) for {domain} domains"
            )
        if expected_inner_count is None and inner_count is not None:
            raise SlangRefusal(
                "Slang tessellation_control stage patchconstantfunc "
                f"'{function_name}' must not return SV_InsideTessFactor "
                f"for {domain} domains"
            )

    def validate_slang_tess_factor_member_types(self, struct_node, function_name):
        tess_factor_semantics = {
            "sv_tessfactor": "SV_TessFactor",
            "sv_insidetessfactor": "SV_InsideTessFactor",
        }
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            if semantic is None:
                continue

            semantic_key = str(self.map_semantic(semantic)).lower()
            semantic_name = tess_factor_semantics.get(semantic_key)
            if semantic_name is None:
                continue

            if self.slang_tess_factor_member_count(member) is not None:
                continue

            member_name = getattr(member, "name", "<anonymous>")
            member_type = self.slang_tess_factor_member_type_name(member)
            mapped_type = self.convert_type(member_type) or member_type
            raise SlangRefusal(
                "Slang tessellation_control stage patchconstantfunc "
                f"'{function_name}' {semantic_name} member '{member_name}' "
                f"uses invalid type {mapped_type}; tessellation factors "
                "must use a floating scalar, vector, or scalar array type"
            )

    def slang_return_struct(self, func):
        return_type_name = self.type_name_string(
            getattr(func, "return_type", getattr(func, "vtype", None))
        )
        if not return_type_name:
            return None
        base_type = return_type_name.split("<", 1)[0].split("[", 1)[0].strip()
        return self.user_structs_by_name.get(base_type)

    def slang_struct_semantics(self, struct_node):
        semantics = set()
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            if semantic is None:
                continue
            semantics.add(str(self.map_semantic(semantic)).lower())
        return semantics

    def slang_struct_semantic_member_counts(self, struct_node):
        counts = {}
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            if semantic is None:
                continue
            semantic_key = str(self.map_semantic(semantic)).lower()
            count = self.slang_tess_factor_member_count(member)
            if count is not None:
                counts[semantic_key] = counts.get(semantic_key, 0) + count
        return counts

    def slang_tess_factor_member_count(self, member):
        type_name = self.slang_tess_factor_member_type_name(member)
        mapped_type = self.convert_type(type_name)
        if mapped_type is None:
            return None

        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix:
            if self.slang_tess_factor_scalar_count(base_type) != 1:
                return None

            first_dimension = array_suffix[1:].split("]", 1)[0]
            remaining_dimensions = array_suffix.split("]", 1)[1]
            if remaining_dimensions:
                return None

            try:
                return int(first_dimension)
            except ValueError:
                return None

        return self.slang_tess_factor_scalar_count(mapped_type)

    def slang_tess_factor_member_type_name(self, member):
        member_type = getattr(member, "member_type", getattr(member, "vtype", None))
        if member_type is None and hasattr(member, "element_type"):
            member_type = member.element_type
        return self.type_name_string(member_type)

    def slang_tess_factor_scalar_count(self, mapped_type):
        if mapped_type is None:
            return None

        for scalar_base in ("min16float", "float", "half", "double"):
            if mapped_type.startswith(scalar_base):
                suffix = mapped_type[len(scalar_base) :]
                if suffix in {"2", "3", "4"}:
                    return int(suffix)
                if suffix == "":
                    return 1
        return None

    def format_slang_patch_shape(self, patch_type, shape):
        element_type, patch_size = shape
        if patch_size is None:
            return f"{patch_type}<{element_type}>"
        return f"{patch_type}<{element_type}, {patch_size}>"

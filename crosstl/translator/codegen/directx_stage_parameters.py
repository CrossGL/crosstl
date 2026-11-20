"""Stage parameters and attributes in the HLSL generator: implicit samplers and textures, patch and output control counts, comparison arguments, domains, global bases and signatures."""

from ..ast import (
    FunctionCallNode,
    ReferenceType,
)
from ..refusals import DirectXRefusal
from .array_utils import split_array_type_suffix
from .directx_values import DirectXWaveSizeProfileError
from .image_access_contracts import (
    TEXTURE_COMPARE_INTRINSIC_NAMES,
    TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES,
    TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES,
    TEXTURE_QUERY_LOD_INTRINSIC_NAMES,
    is_texture_query_lod_operation,
)
from .stage_utils import stage_layout_entry_value


class DirectXStageParameterMixin:
    """Stage parameter members of ``HLSLCodeGen``."""

    def collect_global_implicit_sampler_texture_names(
        self, root, global_texture_names, sampler_names, implicit_params
    ):
        texture_names = set()
        global_texture_types = self.collect_global_texture_types(root)
        functions = self.collect_functions(root)
        texture_funcs = TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                func_name = self.expression_name(func_expr)
                if func_name not in texture_funcs:
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if len(args) < 2:
                    continue
                texture_name = self.expression_name(args[0])
                if texture_name not in global_texture_names:
                    continue
                if self.has_explicit_sampler_argument(func_name, args, sampler_names):
                    continue
                if self.projected_texture_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.projected_texture_compare_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_gather_compare_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_gather_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_gather_offset_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_sample_offset_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_compare_offset_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.multisample_texture_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                texture_names.add(texture_name)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                callee_implicit = implicit_params.get(callee_name, {})
                if not callee_implicit:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                callee_params = self.function_parameter_names.get(callee_name, [])
                for texture_param in callee_implicit:
                    try:
                        texture_index = callee_params.index(texture_param)
                    except ValueError:
                        continue
                    if texture_index >= len(args):
                        continue
                    texture_name = self.expression_name(args[texture_index])
                    if texture_name in global_texture_names:
                        texture_names.add(texture_name)

        return texture_names

    def collect_global_implicit_regular_sampler_texture_names(
        self, root, global_texture_names, sampler_names, implicit_params
    ):
        texture_names = set()
        global_texture_types = self.collect_global_texture_types(root)
        functions = self.collect_functions(root)
        texture_funcs = (
            self.regular_texture_function_names() - TEXTURE_QUERY_LOD_INTRINSIC_NAMES
        )

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                func_name = self.expression_name(func_expr)
                if func_name not in texture_funcs:
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if len(args) < 2:
                    continue
                texture_name = self.expression_name(args[0])
                if texture_name not in global_texture_names:
                    continue
                if self.has_explicit_sampler_argument(func_name, args, sampler_names):
                    continue
                if self.projected_texture_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_gather_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_gather_offset_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.texture_sample_offset_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                if self.multisample_texture_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                texture_names.add(texture_name)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                callee_implicit = implicit_params.get(callee_name, {})
                if not callee_implicit:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                callee_params = self.function_parameter_names.get(callee_name, [])
                for texture_param, sampler_info in callee_implicit.items():
                    if not sampler_info.get("regular"):
                        continue
                    try:
                        texture_index = callee_params.index(texture_param)
                    except ValueError:
                        continue
                    if texture_index >= len(args):
                        continue
                    texture_name = self.expression_name(args[texture_index])
                    if texture_name in global_texture_names:
                        texture_names.add(texture_name)

        return texture_names

    def collect_global_implicit_query_lod_texture_names(
        self, root, global_texture_names, sampler_names, implicit_params
    ):
        texture_names = set()
        global_texture_types = self.collect_global_texture_types(root)
        functions = self.collect_functions(root)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                func_name = self.expression_name(func_expr)
                if not is_texture_query_lod_operation(func_name):
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if len(args) < 2:
                    continue
                texture_name = self.expression_name(args[0])
                if texture_name not in global_texture_names:
                    continue
                if self.has_explicit_sampler_argument(func_name, args, sampler_names):
                    continue
                if self.multisample_texture_call_is_diagnostic_only(
                    func_name, global_texture_types.get(texture_name)
                ):
                    continue
                texture_names.add(texture_name)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                callee_implicit = implicit_params.get(callee_name, {})
                if not callee_implicit:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                callee_params = self.function_parameter_names.get(callee_name, [])
                for texture_param, sampler_info in callee_implicit.items():
                    if not sampler_info.get("query_lod"):
                        continue
                    try:
                        texture_index = callee_params.index(texture_param)
                    except ValueError:
                        continue
                    if texture_index >= len(args):
                        continue
                    texture_name = self.expression_name(args[texture_index])
                    if texture_name in global_texture_names:
                        texture_names.add(texture_name)

        return texture_names

    def collect_sampler_parameter_names(self, root):
        sampler_names = set()
        visited = set()

        def visit(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
                return
            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)

            for param in getattr(value, "parameters", getattr(value, "params", [])):
                param_type = getattr(param, "param_type", getattr(param, "vtype", None))
                if self.is_sampler_type(param_type):
                    sampler_names.add(param.name)

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    visit(child)

        visit(root)
        return sampler_names

    def collect_comparison_sampler_parameters(self, root):
        comparison_params = {}
        diagnostic_compare_params = {}
        functions = self.collect_functions(root)
        functions_by_name = {
            getattr(func, "name", None): func
            for func in functions
            if getattr(func, "name", None)
        }
        global_resource_types = self.collect_global_texture_types(root)

        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            resource_param_types = {
                param.name: getattr(param, "param_type", getattr(param, "vtype", None))
                for param in getattr(func, "parameters", getattr(func, "params", []))
                if self.is_texture_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
                or self.is_image_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
            }
            sampler_params = {
                param.name
                for param in getattr(func, "parameters", getattr(func, "params", []))
                if self.is_sampler_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
            }
            if not sampler_params:
                continue

            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                if self.expression_name(func_expr) not in {
                    *TEXTURE_COMPARE_INTRINSIC_NAMES,
                    *TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES,
                }:
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if len(args) < 4:
                    continue
                texture_name = self.expression_name(args[0])
                texture_type = resource_param_types.get(texture_name)
                if texture_type is not None:
                    texture_type = self.map_resource_type_with_format(texture_type)
                else:
                    texture_type = self.texture_argument_analysis_type(
                        args[0], global_resource_types
                    )
                texture_func = self.expression_name(func_expr)
                sampler_name = self.expression_name(args[1])
                if sampler_name in sampler_params:
                    diagnostic_compare_params.setdefault(func_name, set()).add(
                        sampler_name
                    )
                if self.texture_call_is_diagnostic_only(
                    texture_func, texture_type
                ) and not self.diagnostic_texture_compare_sampler_parameter_is_comparison(
                    texture_func, texture_type
                ):
                    continue
                if sampler_name in sampler_params:
                    comparison_params.setdefault(func_name, set()).add(sampler_name)

        changed = True
        while changed:
            changed = False
            for func in functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue

                params = getattr(func, "parameters", getattr(func, "params", []))
                sampler_param_names = {
                    param.name
                    for param in params
                    if self.is_sampler_type(
                        getattr(param, "param_type", getattr(param, "vtype", None))
                    )
                }
                if not sampler_param_names:
                    continue

                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    callee_name = self.expression_name(func_expr)
                    required_indices = self.comparison_sampler_parameter_indices(
                        functions, comparison_params, callee_name
                    )
                    if not required_indices:
                        continue

                    args = getattr(node, "arguments", getattr(node, "args", []))
                    for index in required_indices:
                        if index >= len(args):
                            continue
                        arg_name = self.expression_name(args[index])
                        if arg_name in sampler_param_names:
                            current = comparison_params.setdefault(func_name, set())
                            if arg_name not in current:
                                current.add(arg_name)
                                changed = True

        direct_comparison_sampler_names = self.collect_explicit_sampler_resource_names(
            root, self.comparison_texture_function_names()
        )

        changed = True
        while changed:
            changed = False
            comparison_sampler_names = (
                direct_comparison_sampler_names
                | self.collect_sampler_arguments(root, comparison_params)
            )

            for func in functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue

                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    callee_name = self.expression_name(func_expr)
                    candidate_params = diagnostic_compare_params.get(callee_name)
                    if not candidate_params:
                        continue

                    args = getattr(node, "arguments", getattr(node, "args", []))
                    callee = functions_by_name.get(callee_name)
                    if callee is None:
                        continue
                    params = getattr(
                        callee, "parameters", getattr(callee, "params", [])
                    )
                    for index, param in enumerate(params):
                        if param.name not in candidate_params or index >= len(args):
                            continue
                        arg_name = self.expression_name(args[index])
                        if arg_name not in comparison_sampler_names:
                            continue
                        current = comparison_params.setdefault(callee_name, set())
                        if param.name not in current:
                            current.add(param.name)
                            changed = True

        return comparison_params

    def collect_comparison_sampler_arguments(self, root, comparison_params):
        comparison_sampler_names = set()
        functions = self.collect_functions(root)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                required_indices = self.comparison_sampler_parameter_indices(
                    functions, comparison_params, callee_name
                )
                if not required_indices:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                for index in required_indices:
                    if index >= len(args):
                        continue
                    arg_name = self.expression_name(args[index])
                    if arg_name:
                        comparison_sampler_names.add(arg_name)

        return comparison_sampler_names

    def comparison_sampler_parameter_indices(
        self, functions, comparison_params, function_name
    ):
        if not function_name or function_name not in comparison_params:
            return set()

        for func in functions:
            if getattr(func, "name", None) != function_name:
                continue
            indices = set()
            params = getattr(func, "parameters", getattr(func, "params", []))
            for index, param in enumerate(params):
                if param.name in comparison_params[function_name]:
                    indices.add(index)
            return indices

        return set()

    def collect_implicit_texture_sampler_parameters(self, root):
        implicit_params = {}
        functions = self.collect_functions(root)
        texture_funcs = TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES

        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue

            params = getattr(func, "parameters", getattr(func, "params", []))
            param_names = {param.name for param in params}
            texture_param_types = {
                param.name: getattr(param, "param_type", getattr(param, "vtype", None))
                for param in params
                if self.is_texture_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
            }
            sampler_params = {
                param.name
                for param in params
                if self.is_sampler_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
            }
            if not texture_param_types:
                continue

            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                texture_func = self.expression_name(func_expr)
                if texture_func not in texture_funcs:
                    continue
                args = getattr(node, "arguments", getattr(node, "args", []))
                if len(args) < 2:
                    continue

                texture_name = self.expression_name(args[0])
                if texture_name not in texture_param_types:
                    continue
                if self.has_explicit_sampler_argument(
                    texture_func, args, sampler_params
                ):
                    continue
                if self.projected_texture_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue
                if self.projected_texture_compare_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue
                if self.texture_gather_compare_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue
                if self.texture_gather_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue
                if self.texture_gather_offset_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue
                if self.texture_sample_offset_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue
                if self.texture_compare_offset_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue
                if self.multisample_texture_call_is_diagnostic_only(
                    texture_func, texture_param_types[texture_name]
                ):
                    continue

                query_lod = is_texture_query_lod_operation(texture_func)
                regular = (
                    not query_lod
                    and texture_func in self.regular_texture_function_names()
                )
                comparison = (
                    not query_lod
                    and texture_func in self.comparison_texture_function_names()
                )
                self.add_implicit_texture_sampler_parameter(
                    implicit_params,
                    func_name,
                    texture_name,
                    comparison,
                    param_names,
                    query_lod=query_lod,
                    regular=regular,
                )

        changed = True
        while changed:
            changed = False
            for func in functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue

                params = getattr(func, "parameters", getattr(func, "params", []))
                param_names = {param.name for param in params}
                texture_param_names = {
                    param.name
                    for param in params
                    if self.is_texture_type(
                        getattr(param, "param_type", getattr(param, "vtype", None))
                    )
                }
                if not texture_param_names:
                    continue

                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    callee_name = self.expression_name(func_expr)
                    callee_implicit = implicit_params.get(callee_name, {})
                    if not callee_implicit:
                        continue

                    args = getattr(node, "arguments", getattr(node, "args", []))
                    callee_params = self.function_parameter_names.get(callee_name, [])
                    for texture_param, sampler_info in callee_implicit.items():
                        try:
                            texture_index = callee_params.index(texture_param)
                        except ValueError:
                            continue
                        if texture_index >= len(args):
                            continue
                        arg_name = self.expression_name(args[texture_index])
                        if arg_name not in texture_param_names:
                            continue
                        changed |= self.add_implicit_texture_sampler_parameter(
                            implicit_params,
                            func_name,
                            arg_name,
                            sampler_info["comparison"],
                            param_names,
                            query_lod=sampler_info.get("query_lod", False),
                            regular=sampler_info.get("regular", False),
                        )

        return implicit_params

    def add_implicit_texture_sampler_parameter(
        self,
        implicit_params,
        func_name,
        texture_name,
        comparison,
        param_names,
        query_lod=False,
        regular=False,
    ):
        sampler_name = f"{texture_name}Sampler"
        new_info = {
            "sampler_name": sampler_name,
            "comparison": comparison,
            "synthetic": sampler_name not in param_names,
            "query_lod": query_lod,
            "regular": regular,
        }
        current = implicit_params.setdefault(func_name, {}).get(texture_name)
        if current is None:
            implicit_params[func_name][texture_name] = new_info
            return True
        changed = False
        if comparison and not current["comparison"]:
            current["comparison"] = True
            changed = True
        if query_lod and not current.get("query_lod"):
            current["query_lod"] = True
            changed = True
        if regular and not current.get("regular"):
            current["regular"] = True
            changed = True
        return changed

    def has_explicit_sampler_argument(self, func_name, args, sampler_names):
        if func_name in {
            *TEXTURE_COMPARE_INTRINSIC_NAMES,
            *TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES,
        }:
            if len(args) < 4:
                return False
        elif len(args) < 3:
            return False

        sampler_name = self.expression_name(args[1])
        if sampler_name in self.local_variable_types:
            sampler_type = self.local_variable_types.get(sampler_name)
            return sampler_type is not None and self.is_sampler_type(sampler_type)
        if sampler_name in sampler_names:
            return True
        sampler_type = self.expression_result_type(args[1])
        return sampler_type is not None and self.is_sampler_type(sampler_type)

    def collect_implicit_comparison_texture_arguments(self, root, implicit_params):
        texture_names = set()
        functions = self.collect_functions(root)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                callee_implicit = implicit_params.get(callee_name, {})
                if not callee_implicit:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                callee_params = self.function_parameter_names.get(callee_name, [])
                for texture_param, sampler_info in callee_implicit.items():
                    if not sampler_info["comparison"]:
                        continue
                    try:
                        texture_index = callee_params.index(texture_param)
                    except ValueError:
                        continue
                    if texture_index >= len(args):
                        continue
                    texture_name = self.expression_name(args[texture_index])
                    if texture_name:
                        texture_names.add(texture_name)

        return texture_names

    def stage_entry_types(self):
        return {
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
            "intersection",
            "anyhit",
            "closesthit",
            "miss",
            "callable",
        }

    def stage_entry_base_name(self, stage_name, func):
        shader_map = {
            "vertex": "VSMain",
            "fragment": "PSMain",
            "compute": "CSMain",
            "geometry": "GSMain",
            "tessellation_control": "HSMain",
            "tessellation_evaluation": "DSMain",
            "mesh": "MSMain",
            "amplification": "ASMain",
            "task": "ASMain",
            "object": "ASMain",
            "ray_generation": "RayGenMain",
            "ray_intersection": "IntersectionMain",
            "ray_closest_hit": "ClosestHitMain",
            "ray_any_hit": "AnyHitMain",
            "ray_miss": "MissMain",
            "ray_callable": "CallableMain",
            "intersection": "IntersectionMain",
            "closesthit": "ClosestHitMain",
            "anyhit": "AnyHitMain",
            "miss": "MissMain",
            "callable": "CallableMain",
        }
        return shader_map.get(stage_name, getattr(func, "name", None) or "main")

    def hlsl_stage_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None

        normalized = str(attr_name).lower()
        if normalized.startswith("hlsl_"):
            normalized = normalized[len("hlsl_") :]

        valid_names = {
            "compute",
            "domain",
            "fragment",
            "instance",
            "max_total_threads_per_threadgroup",
            "max_total_threadgroups_per_mesh_grid",
            "maxvertexcount",
            "maxtessfactor",
            "numthreads",
            "outputcontrolpoints",
            "outputtopology",
            "partitioning",
            "patchconstantfunc",
            "stage_entry",
            "vertex",
            "workgroup_size",
        }
        if normalized in valid_names:
            return normalized
        return None

    def hlsl_root_signature_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False

        normalized = str(attr_name).lower()
        if normalized.startswith("hlsl_"):
            normalized = normalized[len("hlsl_") :]
        return normalized in {"rootsignature", "root_signature"}

    def generate_hlsl_root_signature_attribute(self, func):
        for attr in getattr(func, "attributes", []) or []:
            if not self.hlsl_root_signature_attribute(attr):
                continue
            arguments = getattr(attr, "arguments", []) or []
            if not arguments:
                return "[RootSignature()]\n"
            argument = arguments[0]
            value = self.hlsl_stage_attribute_value_to_string(argument)
            literal_type = getattr(argument, "literal_type", None)
            literal_type_name = str(getattr(literal_type, "name", literal_type))
            if literal_type_name == "string":
                escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
                value = f'"{escaped}"'
            return f"[RootSignature({value})]\n"
        return ""

    def hlsl_stage_attribute_names(self, func):
        names = set()
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.hlsl_stage_attribute_name(attr)
            if attr_name and getattr(attr, "arguments", []):
                names.add(attr_name)
        return names

    def hlsl_stage_attribute_argument(self, func, expected_name):
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.hlsl_stage_attribute_name(attr)
            if attr_name != expected_name:
                continue

            arguments = getattr(attr, "arguments", []) or []
            if arguments:
                return self.hlsl_stage_attribute_value_to_string(arguments[0])
        return None

    def hlsl_stage_attribute_arguments(self, func, expected_name):
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.hlsl_stage_attribute_name(attr)
            if attr_name != expected_name:
                continue
            return getattr(attr, "arguments", []) or []
        return []

    def hlsl_waveops_include_helper_lanes_attribute(self, attr):
        attr_name = str(getattr(attr, "name", ""))
        normalized = attr_name.lower()
        if normalized.startswith("hlsl_"):
            normalized = normalized[len("hlsl_") :]
        return normalized == "waveopsincludehelperlanes"

    def hlsl_wave_size_attribute(self, attr):
        attr_name = str(getattr(attr, "name", ""))
        normalized = attr_name.lower()
        if normalized.startswith("hlsl_"):
            normalized = normalized[len("hlsl_") :]
        return normalized == "wavesize"

    def hlsl_wave_size_lane_value(self, argument, label):
        lane_count = self.hlsl_int_literal_value(argument)
        if lane_count is None:
            raise DirectXRefusal(
                f"DirectX WaveSize attribute {label} requires an immediate "
                "integer argument"
            )
        if lane_count not in self.HLSL_WAVE_SIZE_LANE_COUNTS:
            raise DirectXRefusal(
                f"DirectX WaveSize attribute {label} lane count must be one of "
                "4, 8, 16, 32, 64, or 128"
            )
        return lane_count

    def generate_hlsl_waveops_include_helper_lanes_attribute(self, func, shader_type):
        found = False
        for attr in getattr(func, "attributes", []) or []:
            if not self.hlsl_waveops_include_helper_lanes_attribute(attr):
                continue
            found = True
            arguments = getattr(attr, "arguments", []) or []
            if arguments:
                raise DirectXRefusal(
                    "DirectX WaveOpsIncludeHelperLanes attribute does not "
                    "accept argument(s)"
                )

        if not found:
            return ""
        if shader_type != "fragment":
            raise DirectXRefusal(
                "DirectX WaveOpsIncludeHelperLanes attribute is only valid on "
                "fragment/pixel shader entry points"
            )
        return "[WaveOpsIncludeHelperLanes]\n"

    def generate_hlsl_wave_size_attribute(self, func, shader_type):
        found = False
        wave_size_values = None
        for attr in getattr(func, "attributes", []) or []:
            if not self.hlsl_wave_size_attribute(attr):
                continue
            found = True
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) not in {1, 2, 3}:
                raise DirectXRefusal(
                    "DirectX WaveSize attribute requires 1, 2, or 3 arguments"
                )

            labels = (
                ("lane count",)
                if len(arguments) == 1
                else ("minimum", "maximum", "preferred")
            )
            wave_size_values = [
                self.hlsl_wave_size_lane_value(argument, labels[index])
                for index, argument in enumerate(arguments)
            ]
            if len(wave_size_values) >= 2 and wave_size_values[0] > wave_size_values[1]:
                raise DirectXRefusal(
                    "DirectX WaveSize attribute minimum lane count must be "
                    "less than or equal to maximum lane count"
                )
            if len(wave_size_values) == 3 and not (
                wave_size_values[0] <= wave_size_values[2] <= wave_size_values[1]
            ):
                raise DirectXRefusal(
                    "DirectX WaveSize attribute preferred lane count must be "
                    "between minimum and maximum lane counts"
                )

        if not found:
            return ""
        if shader_type != "compute":
            raise DirectXRefusal(
                "DirectX WaveSize attribute is only valid on compute shader "
                "entry points"
            )
        arguments_text = ", ".join(str(value) for value in wave_size_values)
        if self.target_profile in {"dx10", "dx11"}:
            shader_model = "6.6" if len(wave_size_values) == 1 else "6.8"
            raise DirectXWaveSizeProfileError(
                f"DirectX profile {self.target_profile} does not support the "
                f"WaveSize attribute; [WaveSize({arguments_text})] requires "
                f"DirectX 12 / Shader Model {shader_model}",
                target_profile=self.target_profile,
                wave_size=tuple(wave_size_values),
                shader_model=shader_model,
            )
        return f"[WaveSize({arguments_text})]\n"

    def normalized_hlsl_stage_attribute_argument(self, func, expected_name):
        value = self.hlsl_stage_attribute_argument(func, expected_name)
        if value is None:
            return None
        return str(value).strip('"').lower()

    def canonical_hlsl_tessellation_domain(self, domain):
        if domain is None:
            return None
        normalized = str(domain).strip('"').lower()
        if normalized == "triangle":
            return "tri"
        return normalized

    def hlsl_int_literal_value(self, value):
        if value is None:
            return None
        if hasattr(value, "value"):
            value = value.value
        elif hasattr(value, "name"):
            value = value.name
        if isinstance(value, int) and not isinstance(value, bool):
            return value

        text = str(value).strip().strip('"').replace("_", "")
        if not text:
            return None
        while text and text[-1] in {"u", "U", "l", "L"}:
            text = text[:-1]
        try:
            return int(text, 0)
        except ValueError:
            return None

    def hlsl_constant_int_value(self, value, constants=None):
        if constants is None:
            constants = self.literal_int_constants
        parsed = self.literal_int_value(value, constants)
        if parsed is not None:
            return parsed
        return self.hlsl_int_literal_value(value)

    def hlsl_float_literal_value(self, value):
        if value is None:
            return None
        if hasattr(value, "value"):
            value = value.value
        elif hasattr(value, "name"):
            value = value.name
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

        text = str(value).strip().strip('"').replace("_", "")
        if not text:
            return None
        while text and text[-1] in {"f", "F"}:
            text = text[:-1]
        try:
            return float(text)
        except ValueError:
            return None

    def hlsl_stage_attribute_int_argument(self, func, expected_name):
        return self.hlsl_int_literal_value(
            self.hlsl_stage_attribute_argument(func, expected_name)
        )

    def hlsl_stage_attribute_float_argument(self, func, expected_name):
        return self.hlsl_float_literal_value(
            self.hlsl_stage_attribute_argument(func, expected_name)
        )

    def hlsl_geometry_maxvertexcount_value(self, func, stage_node=None):
        value = self.hlsl_stage_attribute_argument(func, "maxvertexcount")
        if value is not None:
            return value
        if stage_node is None:
            return None
        return stage_layout_entry_value(
            stage_node, "max_vertices", "out"
        ) or stage_layout_entry_value(stage_node, "maxvertexcount", "out")

    def hlsl_geometry_maxvertexcount_int_value(self, func, stage_node=None):
        return self.hlsl_int_literal_value(
            self.hlsl_geometry_maxvertexcount_value(func, stage_node)
        )

    def hlsl_parameter_type_base(self, parameter):
        param_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        type_name = self.type_name_string(param_type)
        if not type_name:
            return None
        return type_name.split("<", 1)[0].strip()

    def hlsl_parameter_array_count(self, parameter):
        size_expr = self.hlsl_parameter_array_size_expression(parameter)
        if size_expr is None:
            return None
        return self.hlsl_constant_int_value(size_expr)

    def hlsl_parameter_mapped_base_and_array_suffix(self, parameter):
        param_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        type_name = self.type_name_string(param_type)
        if not type_name:
            return None, None

        mapped_type = self.map_type(type_name)
        return split_array_type_suffix(str(mapped_type))

    def hlsl_parameter_is_array(self, parameter):
        param_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        if str(type(param_type)).find("ArrayType") != -1:
            return True

        type_name = self.type_name_string(param_type)
        return bool(type_name and split_array_type_suffix(str(type_name))[1])

    def hlsl_parameter_component_count(self, parameter):
        param_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        type_name = self.type_name_string(param_type)
        if not type_name:
            return None

        base_type, _array_suffix = self.hlsl_parameter_mapped_base_and_array_suffix(
            parameter
        )
        if base_type is None:
            return None
        scalar_bases = (
            "min16float",
            "min10float",
            "min16int",
            "min12int",
            "min16uint",
            "int16_t",
            "uint16_t",
            "float",
            "half",
            "double",
            "int",
            "uint",
            "bool",
        )
        for scalar_base in scalar_bases:
            if not base_type.startswith(scalar_base):
                continue
            suffix = base_type[len(scalar_base) :]
            if suffix in {"2", "3", "4"}:
                return int(suffix)
            if suffix == "":
                return 1
        return None

    def split_hlsl_generic_argument_string(self, arguments):
        split_arguments = []
        start = 0
        depth = 0
        for index, char in enumerate(arguments):
            if char == "<":
                depth += 1
            elif char == ">":
                depth = max(depth - 1, 0)
            elif char == "," and depth == 0:
                split_arguments.append(arguments[start:index].strip())
                start = index + 1
        split_arguments.append(arguments[start:].strip())
        return [argument for argument in split_arguments if argument]

    def hlsl_type_generic_arguments(self, vtype):
        generic_args = getattr(vtype, "generic_args", None)
        if generic_args:
            return generic_args

        type_name = self.type_name_string(vtype)
        if not type_name or "<" not in type_name or not type_name.endswith(">"):
            return []
        arguments = type_name.split("<", 1)[1][:-1].strip()
        return self.split_hlsl_generic_argument_string(arguments)

    def hlsl_parameter_type_generic_argument(self, parameter, index):
        param_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        generic_args = self.hlsl_type_generic_arguments(param_type)
        if len(generic_args) <= index:
            return None
        return self.type_name_string(generic_args[index])

    def hlsl_patch_parameter(self, parameters, patch_type):
        for parameter in parameters:
            if self.hlsl_parameter_type_base(parameter) == patch_type:
                return parameter
        return None

    def hlsl_patch_element_type(self, parameters, patch_type):
        parameter = self.hlsl_patch_parameter(parameters, patch_type)
        if parameter is None:
            return None
        element_type = self.hlsl_parameter_type_generic_argument(parameter, 0)
        if element_type is None:
            return None
        return self.map_type(element_type)

    def hlsl_patch_control_point_count(self, parameters, patch_type):
        for parameter in parameters:
            if self.hlsl_parameter_type_base(parameter) != patch_type:
                continue
            param_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            generic_args = self.hlsl_type_generic_arguments(param_type)
            if len(generic_args) < 2:
                return None
            return self.hlsl_int_literal_value(generic_args[1])
        return None

    def validate_hlsl_patch_parameter_signature(self, parameter, patch_type, context):
        param_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        type_name = self.type_name_string(param_type) or patch_type
        generic_args = self.hlsl_type_generic_arguments(param_type)
        parameter_name = getattr(parameter, "name", None)
        name_clause = (
            f" parameter '{parameter_name}'" if parameter_name else " parameter"
        )

        if len(generic_args) != 2:
            raise DirectXRefusal(
                f"DirectX {context} {patch_type}{name_clause} must use "
                f"{patch_type}<T, N>; found '{type_name}'"
            )

        element_type = self.type_name_string(generic_args[0])
        if not element_type:
            raise DirectXRefusal(
                f"DirectX {context} {patch_type}{name_clause} must include "
                "an element type"
            )

        control_point_count = self.hlsl_int_literal_value(generic_args[1])
        control_point_text = self.type_name_string(generic_args[1])
        if control_point_count is None:
            raise DirectXRefusal(
                f"DirectX {context} {patch_type}{name_clause} control point "
                f"count '{control_point_text}' must be an integer literal"
            )
        if control_point_count <= 0:
            raise DirectXRefusal(
                f"DirectX {context} {patch_type}{name_clause} control point "
                f"count ({control_point_count}) must be positive"
            )
        if control_point_count > self.HLSL_PATCH_CONTROL_POINT_LIMIT:
            raise DirectXRefusal(
                f"DirectX {context} {patch_type}{name_clause} control point "
                f"count ({control_point_count}) must be at most "
                f"{self.HLSL_PATCH_CONTROL_POINT_LIMIT}"
            )

    def validate_hlsl_patch_parameter_signatures(self, parameters, patch_type, context):
        for parameter in parameters:
            if self.hlsl_parameter_type_base(parameter) != patch_type:
                continue
            self.validate_hlsl_patch_parameter_signature(parameter, patch_type, context)

    def hlsl_input_patch_control_point_count(self, parameters):
        return self.hlsl_patch_control_point_count(parameters, "InputPatch")

    def hlsl_output_patch_control_point_count(self, parameters):
        return self.hlsl_patch_control_point_count(parameters, "OutputPatch")

    def validate_hlsl_output_control_points(self, func, parameters):
        output_control_points = self.hlsl_stage_attribute_int_argument(
            func, "outputcontrolpoints"
        )
        input_patch_control_points = self.hlsl_input_patch_control_point_count(
            parameters
        )
        if output_control_points is None or input_patch_control_points is None:
            return
        if output_control_points != input_patch_control_points:
            raise DirectXRefusal(
                "DirectX tessellation_control stage outputcontrolpoints "
                f"({output_control_points}) must match the InputPatch control "
                f"point count ({input_patch_control_points})"
            )

    def validate_hlsl_output_control_points_value(self, func, shader_type):
        if shader_type != "tessellation_control":
            return

        output_control_points = self.hlsl_stage_attribute_int_argument(
            func, "outputcontrolpoints"
        )
        if output_control_points is None:
            return
        if output_control_points <= 0:
            raise DirectXRefusal(
                "DirectX tessellation_control stage outputcontrolpoints "
                f"({output_control_points}) must be positive"
            )
        if output_control_points > self.HLSL_PATCH_CONTROL_POINT_LIMIT:
            raise DirectXRefusal(
                "DirectX tessellation_control stage outputcontrolpoints "
                f"({output_control_points}) must be at most "
                f"{self.HLSL_PATCH_CONTROL_POINT_LIMIT}"
            )

    def validate_hlsl_max_tess_factor_value(self, func, shader_type):
        if "maxtessfactor" not in self.hlsl_stage_attribute_names(func):
            return

        if shader_type != "tessellation_control":
            raise DirectXRefusal(
                f"DirectX {shader_type or 'helper'} stage maxtessfactor "
                "attribute is only valid on tessellation_control stages"
            )

        max_tess_factor = self.hlsl_stage_attribute_float_argument(
            func, "maxtessfactor"
        )
        max_tess_factor_text = self.hlsl_stage_attribute_argument(func, "maxtessfactor")
        if max_tess_factor is None:
            raise DirectXRefusal(
                "DirectX tessellation_control stage maxtessfactor "
                f"'{max_tess_factor_text}' must be a numeric literal"
            )
        if not (1.0 <= max_tess_factor <= 64.0):
            raise DirectXRefusal(
                "DirectX tessellation_control stage maxtessfactor "
                f"({max_tess_factor_text}) must be in the range 1.0..64.0"
            )

    def validate_hlsl_max_vertex_count_value(self, func, shader_type, stage_node=None):
        if shader_type != "geometry":
            return

        max_vertex_count = self.hlsl_geometry_maxvertexcount_int_value(
            func,
            stage_node=stage_node,
        )
        if max_vertex_count is None:
            return
        if max_vertex_count <= 0:
            raise DirectXRefusal(
                "DirectX geometry stage maxvertexcount "
                f"({max_vertex_count}) must be positive"
            )

    def validate_hlsl_domain_output_patch_control_points(self, parameters):
        hull_output_control_points = self.current_hlsl_hull_output_control_points
        output_patch_control_points = self.hlsl_output_patch_control_point_count(
            parameters
        )
        if hull_output_control_points is None or output_patch_control_points is None:
            return
        if output_patch_control_points != hull_output_control_points:
            raise DirectXRefusal(
                "DirectX tessellation_evaluation stage OutputPatch control "
                f"point count ({output_patch_control_points}) must match the "
                "tessellation_control outputcontrolpoints "
                f"({hull_output_control_points})"
            )

    def validate_hlsl_domain_output_patch_element_type(self, parameters):
        hull_output_type = self.current_hlsl_hull_output_element_type
        output_patch_type = self.hlsl_patch_element_type(parameters, "OutputPatch")
        if hull_output_type is None or output_patch_type is None:
            return
        if output_patch_type != hull_output_type:
            raise DirectXRefusal(
                "DirectX tessellation_evaluation stage OutputPatch element type "
                f"'{output_patch_type}' must match tessellation_control output "
                f"type '{hull_output_type}'"
            )

    def validate_hlsl_domain_matches_hull(self, func, shader_type):
        if shader_type != "tessellation_evaluation":
            return

        hull_domain = self.current_hlsl_hull_domain
        domain = self.canonical_hlsl_tessellation_domain(
            self.normalized_hlsl_stage_attribute_argument(func, "domain")
        )
        if hull_domain is None or domain is None:
            return
        if domain != hull_domain:
            raise DirectXRefusal(
                "DirectX tessellation_evaluation stage domain "
                f"'{domain}' must match tessellation_control domain "
                f"'{hull_domain}'"
            )

    def hlsl_parameter_has_semantic(self, parameter, expected_semantic):
        semantic = self.semantic_from_node(parameter)
        if semantic is None:
            return False
        mapped_semantic = self.hlsl_semantic_name(semantic)
        return str(mapped_semantic).lower() == expected_semantic.lower()

    def hlsl_parameter_qualifiers(self, parameter):
        qualifiers = []
        allowed_qualifiers = {
            "const",
            "in",
            "out",
            "inout",
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in getattr(parameter, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)

        parameter_type = getattr(
            parameter, "param_type", getattr(parameter, "vtype", None)
        )
        source_qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        private_word_view = self.hlsl_private_pointer_word_view_for_variant(
            self.current_hlsl_private_pointer_variant.get(
                getattr(parameter, "name", None)
            )
        )
        if (
            self.hlsl_private_pointer_parameter(parameter)
            and getattr(parameter_type, "is_mutable", True)
            and private_word_view is None
            and not set(qualifiers).intersection({"const", "in", "out", "inout"})
        ):
            qualifiers.append("inout")
        if (
            self.hlsl_private_pointer_parameter(parameter)
            and private_word_view is not None
            and not set(qualifiers).intersection({"const", "in", "out", "inout"})
        ):
            qualifiers.append("in")
        if (
            self.hlsl_workgroup_pointer_declaration(parameter)
            and getattr(parameter_type, "is_mutable", True)
            and not set(qualifiers).intersection({"const", "in", "out", "inout"})
            and not source_qualifiers.intersection(
                {"const", "constant", "readonly", "in"}
            )
        ):
            qualifiers.append("inout")
        if (
            isinstance(parameter_type, ReferenceType)
            and getattr(parameter_type, "is_mutable", False)
            and not set(qualifiers).intersection({"const", "in", "out", "inout"})
            and not source_qualifiers.intersection(
                {"const", "constant", "readonly", "in"}
            )
        ):
            qualifiers.append("inout")

        for role in self.hlsl_mesh_parameter_roles(parameter):
            if role not in qualifiers:
                qualifiers.append(role)

        for attr in getattr(parameter, "attributes", []) or []:
            if getattr(attr, "name", None) != "primitive":
                continue
            arguments = getattr(attr, "arguments", []) or []
            if not arguments:
                continue
            primitive = self.hlsl_stage_attribute_value_to_string(arguments[0])
            normalized = str(primitive).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)

        return qualifiers

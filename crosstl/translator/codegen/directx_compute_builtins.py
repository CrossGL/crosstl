"""Compute and Metal built-ins in the HLSL generator: bitcasts, SIMD shuffles, explicit identifiers, wave and subgroup ids, signbit, copysign and inverse functions."""

import re

from ..float_limit_constants import render_float_limit_constant
from ..refusals import DirectXRefusal
from ..standard_constants import render_standard_math_constant
from .array_utils import (
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .directx_values import (
    DirectXCopySignError,
    DirectXInverseHyperbolicError,
    DirectXSignBitError,
    DirectXTrailingZeroBuiltinError,
)
from .hlsl_function_body_scans import HlslFunctionBodyScans
from .hlsl_host_workgroup_size import hlsl_entry_local_size
from .hlsl_narrow_system_values import HLSL_NARROW_UNSIGNED_SYSTEM_VALUE_TYPES
from .software_quad_reductions import (
    QUAD_BUILTIN_OPERATORS,
    quad_builtin_expression,
)
from .stage_utils import normalize_stage_name


class DirectXComputeBuiltinMixin:
    """Compute builtin members of ``HLSLCodeGen``."""

    def hlsl_trailing_zero_operand_type_info(self, operand_type):
        source_type = self.type_name_string(operand_type)
        if not source_type:
            return None
        source_type = source_type.strip()
        if source_type in self.HLSL_NARROW_INTEGER_TYPE_NAMES:
            return None

        mapped_type = self.map_type(source_type)
        if mapped_type == "int":
            return {
                "source": source_type,
                "mapped": mapped_type,
                "unsigned": "uint",
                "width": 32,
            }
        if mapped_type == "uint":
            return {
                "source": source_type,
                "mapped": mapped_type,
                "unsigned": "uint",
                "width": 32,
            }
        if mapped_type == "int64_t":
            return {
                "source": source_type,
                "mapped": mapped_type,
                "unsigned": "uint64_t",
                "width": 64,
            }
        if mapped_type == "uint64_t":
            return {
                "source": source_type,
                "mapped": mapped_type,
                "unsigned": "uint64_t",
                "width": 64,
            }
        return None

    def directx_trailing_zero_builtin_error(
        self,
        func_name,
        *,
        reason,
        operand_type=None,
        argument_count=None,
        source_location=None,
    ):
        if reason == "invalid-arity":
            message = (
                f"DirectX trailing-zero builtin '{func_name}' requires exactly "
                f"1 scalar integer operand, got {argument_count}"
            )
        elif reason == "unresolved-operand-type":
            message = (
                f"DirectX translation could not resolve source type metadata for "
                f"the operand of trailing-zero builtin '{func_name}' in function "
                f"'{self.current_function_name or '<global>'}'"
            )
        elif reason == "unsupported-operand-shape":
            message = (
                f"DirectX trailing-zero builtin '{func_name}' requires a scalar "
                f"integer operand; got '{operand_type}'"
            )
        elif reason == "unsupported-integer-width":
            message = (
                f"DirectX trailing-zero builtin '{func_name}' supports only 32-bit "
                f"or 64-bit integer operands; got '{operand_type}'"
            )
        elif reason == "operand-narrowing-required":
            message = (
                f"DirectX trailing-zero builtin '{func_name}' has a 32-bit operand "
                f"contract and cannot consume '{operand_type}' without narrowing"
            )
        else:
            message = (
                f"DirectX trailing-zero builtin '{func_name}' requires a 32-bit "
                f"or 64-bit integer operand; got '{operand_type}'"
            )
        return DirectXTrailingZeroBuiltinError(
            message,
            builtin_name=func_name,
            operand_type=operand_type,
            reason=reason,
            source_location=source_location,
        )

    def generate_hlsl_trailing_zero_call(
        self, func_name, args, *, source_location=None
    ):
        if (
            func_name not in self.HLSL_CLANG_TRAILING_ZERO_BUILTINS
            or self.hlsl_trailing_zero_builtin_is_shadowed(func_name)
        ):
            return None
        if len(args) != 1:
            raise self.directx_trailing_zero_builtin_error(
                func_name,
                reason="invalid-arity",
                argument_count=len(args),
                source_location=source_location,
            )

        argument = args[0]
        operand_type = self.expression_result_type(argument)
        if operand_type is None:
            raise self.directx_trailing_zero_builtin_error(
                func_name,
                reason="unresolved-operand-type",
                source_location=source_location,
            )

        source_type = self.type_name_string(operand_type)
        mapped_type = self.map_type(source_type)
        type_info = self.hlsl_trailing_zero_operand_type_info(source_type)
        if type_info is None:
            if self.is_vector_value_type(mapped_type) or self.hlsl_matrix_shape(
                mapped_type
            ):
                reason = "unsupported-operand-shape"
            elif (
                source_type in self.HLSL_NARROW_INTEGER_TYPE_NAMES
                or self.hlsl_scalar_or_vector_component_type(mapped_type)
                in {"int16_t", "min12int", "min16int", "min16uint", "uint16_t"}
            ):
                reason = "unsupported-integer-width"
            else:
                reason = "unsupported-operand-type"
            raise self.directx_trailing_zero_builtin_error(
                func_name,
                reason=reason,
                operand_type=source_type,
                source_location=source_location,
            )

        required_width = self.HLSL_CLANG_TRAILING_ZERO_WIDTHS[func_name]
        if type_info["width"] > required_width:
            raise self.directx_trailing_zero_builtin_error(
                func_name,
                reason="operand-narrowing-required",
                operand_type=source_type,
                source_location=source_location,
            )

        unsigned_type = "uint64_t" if required_width == 64 else "uint"
        helper_name = self.hlsl_trailing_zero_helper_name(unsigned_type)
        self.required_hlsl_trailing_zero_helpers.add(unsigned_type)
        argument_code = self.generate_expression_with_expected(argument, None)
        if type_info["width"] == required_width and type_info["mapped"] in {
            "int",
            "int64_t",
        }:
            argument_code = f"{unsigned_type}({argument_code})"
        return f"{helper_name}({argument_code})"

    def hlsl_identifier_name(self, name):
        if name in self.current_identifier_aliases:
            return self.current_identifier_aliases[name]
        return name

    def hlsl_standard_math_constant_expression(self, name):
        # HLSL defines none of the C/Metal float limit macros (NAN, INFINITY,
        # FLT_MAX, ...). Render them from their exact binary32 encodings.
        literal = render_float_limit_constant(
            name, "directx"
        ) or render_standard_math_constant(name, "directx")
        if literal is None:
            return None

        if (
            name in self.local_variable_types
            or name in self.global_variable_types
            or name in self.standard_math_constant_shadow_names
            or name in self.current_identifier_aliases
            or name in self.current_hlsl_resource_pointer_aliases
        ):
            return None
        return literal

    def hlsl_type_constructor_name(self, name):
        if not isinstance(name, str):
            return None
        if name in getattr(self, "hlsl_type_aliases", {}):
            return name
        if name in getattr(self, "function_return_types", {}):
            return None
        if name in self.METAL_TYPE_ALIAS_GLOBALS:
            return self.map_type(name)
        if name in self.HLSL_CANONICAL_NARROW_FLOAT_CONSTRUCTORS:
            return self.map_type(name)
        if name in self.HLSL_CANONICAL_64_BIT_VECTOR_CONSTRUCTORS:
            return self.map_type(name)
        generic_vector = self.hlsl_generic_vector_type_name(name)
        if generic_vector is not None:
            return generic_vector
        decoded = self.hlsl_decoded_materialized_name(name)
        if decoded != name:
            return self.hlsl_generic_vector_type_name(decoded)
        return None

    def hlsl_generic_vector_type_name(self, type_name):
        if not isinstance(type_name, str):
            return None
        match = re.fullmatch(
            r"(?:metal::)?(?:vec|vector)\s*<\s*([^,>]+)\s*,\s*([234])\s*>",
            type_name.strip(),
        )
        if match is None:
            return None
        element_type = self.map_type(match.group(1).strip())
        return f"{element_type}{match.group(2)}"

    def hlsl_graphics_builtin_result_type(self, name):
        if name == "gl_FragCoord" and name in self.current_identifier_aliases:
            return "float4"
        if name == "gl_FragSizeEXT" and name in self.current_identifier_aliases:
            return "uint2"
        return None

    def hlsl_function_body_scan(self, func):
        """``func``'s body scan for this render (``hlsl_function_body_scans``)."""
        scans = getattr(self, "hlsl_function_body_scans", None)
        if scans is None:
            scans = self.hlsl_function_body_scans = HlslFunctionBodyScans(self)
        return scans.scan(func)

    def collect_hlsl_function_identifier_names(self, func):
        return set(self.hlsl_function_body_scan(func).identifier_names)

    def hlsl_declaration_identifier_name(self, name):
        if not isinstance(name, str):
            return name
        if name in self.current_identifier_aliases:
            return self.current_identifier_aliases[name]
        if (
            name not in self.HLSL_RESERVED_LOCAL_IDENTIFIER_NAMES
            and name not in self.current_identifier_aliases.values()
        ):
            return name

        used_names = set(self.local_variable_types)
        used_names.update(self.global_variable_types)
        used_names.update(self.current_identifier_reserved_names)
        used_names.update(self.current_identifier_aliases.values())
        alias = f"{name}_"
        while alias in used_names or alias in self.HLSL_RESERVED_LOCAL_IDENTIFIER_NAMES:
            alias += "_"
        self.current_identifier_aliases[name] = alias
        return alias

    def hlsl_unique_local_identifier(self, base_name, used_names):
        candidate = base_name
        while (
            candidate in used_names
            or candidate in self.HLSL_RESERVED_LOCAL_IDENTIFIER_NAMES
            or candidate in self.current_identifier_reserved_names
        ):
            candidate += "_"
        return candidate

    def hlsl_compute_dispatch_thread_id_parameter_bridge(
        self, parameter, param_type, semantic, emitted_param_names
    ):
        if self.hlsl_canonical_semantic(semantic) != "SV_DispatchThreadID":
            return None

        base_type, array_suffix = split_array_type_suffix(str(param_type))
        if array_suffix or base_type not in {"uint", "uint2"}:
            return None

        parameter_name = getattr(parameter, "name", None)
        if not isinstance(parameter_name, str) or not parameter_name:
            return None

        local_name = self.hlsl_declaration_identifier_name(parameter_name)
        used_names = set(emitted_param_names)
        used_names.update(self.local_variable_types)
        used_names.update(self.global_variable_types)
        used_names.update(self.current_identifier_aliases.values())
        hlsl_param_name = self.hlsl_unique_local_identifier(
            f"{local_name}_dispatchThreadID", used_names
        )
        emitted_param_names.add(hlsl_param_name)
        self.local_variable_types[hlsl_param_name] = "uint3"

        declaration = format_c_style_array_declaration("uint3", hlsl_param_name)
        if base_type == "uint":
            local_alias = f"uint {local_name} = {hlsl_param_name}.x;"
        else:
            local_alias = f"uint2 {local_name} = {hlsl_param_name}.xy;"
        return declaration, local_alias

    def hlsl_wave_builtin_intrinsic_expression(self, semantic):
        if not isinstance(semantic, str):
            return None
        return self.HLSL_WAVE_BUILTIN_INTRINSICS.get(semantic.strip())

    def hlsl_non_entry_compute_builtin_semantic(self, effective_shader_type, semantic):
        """True when ``semantic`` is a GLSL compute/subgroup builtin appearing on a
        parameter of a non-stage-entry helper function.

        Such builtins (gl_GlobalInvocationID, gl_SubgroupID, gl_WorkGroupSize, ...)
        only have an HLSL system-value/wave-intrinsic form on a true stage entry
        point. On an ordinary helper they have no HLSL semantic and would leak as an
        invalid ``: gl_*`` parameter semantic, so the caller drops the semantic and
        emits a plain parameter (matching GLSL/SPIR-V, which receive these as plain
        function arguments)."""
        if not isinstance(semantic, str):
            return False
        if normalize_stage_name(effective_shader_type) in self.stage_entry_types():
            return False
        return semantic.strip() in self.HLSL_GLSL_COMPUTE_BUILTIN_SEMANTICS

    def hlsl_software_subgroup_builtin_expression(
        self,
        semantic,
        execution_config=None,
        *,
        param_list=None,
        group_index_name=None,
    ):
        if self.software_subgroup_width is None:
            return None
        canonical = semantic.strip() if isinstance(semantic, str) else semantic
        if canonical == "gl_SubgroupSize":
            return f"{self.software_subgroup_width}u"
        if canonical == "gl_NumSubgroups":
            group_size = self.hlsl_compute_group_size_constant(execution_config)
            if (
                self.hlsl_software_subgroup_invocation_count
                % self.software_subgroup_width
            ):
                return f"(({group_size} + {self.software_subgroup_width - 1}u) / {self.software_subgroup_width}u)"
            return f"({group_size} / {self.software_subgroup_width}u)"
        if canonical not in {"gl_SubgroupID", "gl_SubgroupInvocationID"}:
            return None
        if group_index_name is None:
            group_index_name = self.hlsl_compute_group_index_dependency_name(param_list)
        operator = "/" if canonical == "gl_SubgroupID" else "%"
        return f"({group_index_name} {operator} " f"{self.software_subgroup_width}u)"

    def hlsl_compute_wave_builtin_parameter_prologue(
        self,
        parameter,
        param_type,
        semantic,
        func,
        execution_config=None,
        param_list=None,
    ):
        """Bridge an explicit compute parameter that carries a subgroup/wave
        builtin semantic into HLSL.

        Metal's ``[[thread_index_in_simdgroup]]`` / ``[[threads_per_simdgroup]]`` /
        ``[[simdgroup_index_in_threadgroup]]`` / ``[[simdgroups_per_threadgroup]]``
        (canonicalised to ``gl_SubgroupInvocationID`` / ``gl_SubgroupSize`` /
        ``gl_SubgroupID`` / ``gl_NumSubgroups``) have no HLSL system-value
        semantic, so they cannot be emitted as entry-point parameters. Instead the
        parameter becomes a body-local initialised from the matching wave intrinsic
        (``WaveGetLaneIndex()`` / ``WaveGetLaneCount()``), a proven single-wave zero
        ID or synchronized unique physical-wave ID for ``gl_SubgroupID``, or the
        compile-time thread-group size derivation for ``gl_NumSubgroups``. Returns
        the prologue statement, or ``None`` when the parameter does not carry a wave
        builtin semantic.
        """
        canonical_semantic = semantic.strip() if isinstance(semantic, str) else semantic
        intrinsic = None
        if canonical_semantic in QUAD_BUILTIN_OPERATORS:
            # Quads are four consecutive linear invocations of the group.
            intrinsic = quad_builtin_expression(
                canonical_semantic,
                self.hlsl_compute_group_index_dependency_name(param_list),
            )
        elif self.software_subgroup_width is not None:
            intrinsic = self.hlsl_software_subgroup_builtin_expression(
                canonical_semantic,
                execution_config,
                param_list=param_list,
            )
        if intrinsic is None and canonical_semantic == "gl_SubgroupID":
            intrinsic = self.hlsl_compute_subgroup_id_expression(
                func,
                execution_config,
                param_list=param_list,
            )
        elif intrinsic is None:
            intrinsic = self.hlsl_wave_builtin_intrinsic_expression(semantic)
        if intrinsic is None:
            return None

        parameter_name = getattr(parameter, "name", None)
        if not isinstance(parameter_name, str) or not parameter_name:
            return None

        base_type, array_suffix = split_array_type_suffix(str(param_type))
        if array_suffix:
            return None

        if self.HLSL_WAVE_GROUP_SIZE_PLACEHOLDER in intrinsic:
            intrinsic = intrinsic.replace(
                self.HLSL_WAVE_GROUP_SIZE_PLACEHOLDER,
                self.hlsl_compute_group_size_constant(execution_config),
            )
        if base_type in HLSL_NARROW_UNSIGNED_SYSTEM_VALUE_TYPES:
            # A Metal `ushort` lane index or count: narrow explicitly.
            intrinsic = f"({base_type})({intrinsic})"
        local_name = self.hlsl_declaration_identifier_name(parameter_name)
        self.local_variable_types[local_name] = base_type
        return f"{base_type} {local_name} = {intrinsic};"

    # Sentinel returned by hlsl_compute_value_builtin_parameter_prologue to signal
    # that a value-builtin parameter has no HLSL expression form and is unused, so it
    # must be dropped from the signature entirely (rather than emitted or lowered).
    HLSL_COMPUTE_DROP_PARAMETER = object()

    def hlsl_compute_value_builtin_semantic(self, semantic):
        """True when ``semantic`` is a compute builtin that has no valid HLSL
        parameter system value (it is a compile-time constant or uniform/derived
        expression): gl_WorkGroupSize / gl_NumWorkGroups / threads_per_grid."""
        if not isinstance(semantic, str):
            return False
        return semantic.strip() in self.HLSL_COMPUTE_VALUE_BUILTIN_SEMANTICS

    def hlsl_compute_value_builtin_expression(self, semantic, execution_config=None):
        """Return the HLSL expression a value-builtin lowers to, or ``None`` when no
        expression is available (e.g. a num-workgroups uniform was not materialised).

        These builtins have no HLSL system value, so a body reference / parameter is
        replaced by:
          gl_WorkGroupSize -> uint3(<numthreads x>, <y>, <z>)  (compile-time constant)
          gl_NumWorkGroups -> the num-workgroups uniform member  (or None if absent)
          threads_per_grid -> num_workgroups * uint3(numthreads) (or None if absent)
        """
        key = semantic.strip() if isinstance(semantic, str) else semantic
        if key == "gl_WorkGroupSize":
            return self.hlsl_compute_workgroup_size_expression(execution_config)
        if key == "gl_NumWorkGroups":
            return self.hlsl_compute_num_workgroups_expression()
        if key == "threads_per_grid":
            num_workgroups = self.hlsl_compute_num_workgroups_expression()
            if not num_workgroups:
                # HLSL has no system value for the total grid extent and we have no
                # num-workgroups uniform to multiply, so it cannot be expressed.
                return None
            workgroup_size = self.hlsl_compute_workgroup_size_expression(
                execution_config
            )
            return f"({num_workgroups} * {workgroup_size})"
        return None

    def hlsl_project_compute_vector3_expression(self, expression, target_type):
        normalized_type = str(target_type).strip()
        match = re.fullmatch(
            r"(?P<scalar>u?int|min16uint|uint16_t)(?P<width>[123]?)", normalized_type
        )
        if match is None:
            return expression
        width = int(match.group("width") or "1")
        if width == 1:
            return f"{normalized_type}(({expression}).x)"
        if width == 2:
            return f"{normalized_type}(({expression}).xy)"
        if normalized_type == "uint3":
            return expression
        return f"{normalized_type}({expression})"

    def hlsl_compute_value_builtin_parameter_prologue(
        self, parameter, param_type, semantic, func, execution_config=None
    ):
        """Bridge a compute *entry* parameter that carries a value-builtin semantic
        (gl_WorkGroupSize / gl_NumWorkGroups / threads_per_grid) into HLSL.

        Metal kernels routinely declare ``[[threads_per_threadgroup]]`` /
        ``[[threads_per_grid]]`` / ``[[threadgroups_per_grid]]`` entry parameters.
        These canonicalise to builtins with no HLSL parameter system value, so they
        cannot be emitted as ``: <builtin>`` semantics (and gl_WorkGroupSize would
        additionally trip the unresolved-construct guardrail). Instead:

        * when an HLSL expression exists, the parameter becomes a body-local
          initialised from it (so any body references resolve naturally), and
        * when no expression is available *and the parameter is unused* in the body,
          the parameter is dropped.

        Returns the prologue statement (a ``str``), ``HLSL_COMPUTE_DROP_PARAMETER``
        when the parameter should be dropped, or ``None`` when the parameter does not
        carry a value-builtin semantic (caller handles it normally).
        """
        if not self.hlsl_compute_value_builtin_semantic(semantic):
            return None

        parameter_name = getattr(parameter, "name", None)
        if not isinstance(parameter_name, str) or not parameter_name:
            return None

        base_type, array_suffix = split_array_type_suffix(str(param_type))
        if array_suffix:
            return None

        expression = self.hlsl_compute_value_builtin_expression(
            semantic, execution_config
        )
        if expression is None:
            # No HLSL expression form (e.g. threads_per_grid with no num-workgroups
            # uniform). If the body never references the parameter, drop it.
            # Otherwise it is genuinely needed but unrepresentable as a system value,
            # so fall through to normal emission, which strips the invalid semantic
            # and keeps it as a plain parameter rather than leaving a dangling
            # identifier (matching how GLSL/SPIR-V pass it as an ordinary argument).
            if self.hlsl_compute_parameter_unused(func, parameter_name):
                return self.HLSL_COMPUTE_DROP_PARAMETER
            return None

        expression = self.hlsl_project_compute_vector3_expression(expression, base_type)

        local_name = self.hlsl_declaration_identifier_name(parameter_name)
        self.local_variable_types[local_name] = base_type
        return f"{base_type} {local_name} = {expression};"

    def hlsl_compute_parameter_unused(self, func, parameter_name):
        """True when ``parameter_name`` is never referenced in ``func``'s body."""
        target = self.hlsl_declaration_identifier_name(parameter_name)
        for node in self.walk_ast(getattr(func, "body", [])):
            if not (hasattr(node, "__class__") and "Identifier" in str(node.__class__)):
                continue
            name = getattr(node, "name", "")
            if not isinstance(name, str):
                continue
            base_name = name.split(".", 1)[0]
            if base_name in (parameter_name, target):
                return False
        return True

    def hlsl_compute_group_index_dependency_name(self, param_list=None):
        """Return the name of the SV_GroupIndex parameter that gl_SubgroupID
        derivations depend on, allocating and recording one for injection by the
        post-loop compute-builtin parameter pass when it is not already in scope.

        Reuses an SV_GroupIndex parameter already present (an explicit
        @gl_LocalInvocationIndex parameter, or the one emitted for a
        gl_LocalInvocationIndex body reference) so only a single such parameter is
        added to the entry point."""
        existing = self.current_identifier_aliases.get("gl_LocalInvocationIndex")
        if existing:
            return existing
        for sibling in param_list or []:
            sibling_semantic = self.hlsl_canonical_semantic(
                self.semantic_from_node(sibling)
            )
            if sibling_semantic == "SV_GroupIndex":
                return self.hlsl_declaration_identifier_name(
                    getattr(sibling, "name", None)
                )
        if self.current_hlsl_compute_group_index_param:
            return self.current_hlsl_compute_group_index_param
        reserved_names = set(self.local_variable_types)
        reserved_names.update(self.global_variable_types)
        reserved_names.update(self.current_identifier_reserved_names)
        reserved_names.update(self.current_identifier_aliases.values())
        name = self.hlsl_unique_local_identifier("groupIndex", reserved_names)
        self.current_hlsl_compute_group_index_param = name
        self.local_variable_types[name] = "uint"
        return name

    def hlsl_compute_minimum_wave_size(self, func):
        wave_size_values = None
        for attr in getattr(func, "attributes", []) or []:
            if not self.hlsl_wave_size_attribute(attr):
                continue
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) not in {1, 2, 3}:
                return None
            labels = (
                ("lane count",)
                if len(arguments) == 1
                else ("minimum", "maximum", "preferred")
            )
            wave_size_values = [
                self.hlsl_wave_size_lane_value(argument, labels[index])
                for index, argument in enumerate(arguments)
            ]
        return wave_size_values[0] if wave_size_values else None

    def hlsl_compute_single_wave_proven(self, func, execution_config=None):
        minimum_wave_size = self.hlsl_compute_minimum_wave_size(func)
        if minimum_wave_size is None:
            return False
        try:
            workgroup_size = 1
            for dimension in hlsl_entry_local_size(execution_config):
                workgroup_size *= int(str(dimension).strip())
        except (TypeError, ValueError):
            return False
        return 0 < workgroup_size <= minimum_wave_size

    def hlsl_compute_subgroup_id_expression(
        self,
        func,
        execution_config=None,
        *,
        param_list=None,
        group_index_name=None,
    ):
        if self.software_subgroup_width is not None:
            return self.hlsl_software_subgroup_builtin_expression(
                "gl_SubgroupID",
                execution_config,
                param_list=param_list,
                group_index_name=group_index_name,
            )
        if self.hlsl_compute_single_wave_proven(func, execution_config):
            if group_index_name is None:
                group_index_name = self.hlsl_compute_group_index_dependency_name(
                    param_list
                )
            return f"({group_index_name} / WaveGetLaneCount())"
        return self.hlsl_compute_physical_subgroup_id_dependency_name(
            param_list,
            group_index_name=group_index_name,
        )

    def hlsl_compute_physical_subgroup_id_dependency_name(
        self, param_list=None, *, group_index_name=None
    ):
        if self.current_hlsl_compute_subgroup_id_local is not None:
            return self.current_hlsl_compute_subgroup_id_local

        if group_index_name is None:
            group_index_name = self.hlsl_compute_group_index_dependency_name(param_list)
        reserved_names = set(self.local_variable_types)
        reserved_names.update(self.global_variable_types)
        reserved_names.update(self.current_identifier_reserved_names)
        reserved_names.update(self.current_identifier_aliases.values())
        local_name = self.hlsl_unique_local_identifier(
            "crossglPhysicalSubgroupID", reserved_names
        )
        self.current_hlsl_compute_subgroup_id_local = local_name
        self.current_hlsl_compute_subgroup_id_group_index = group_index_name
        self.local_variable_types[local_name] = "uint"
        self.required_hlsl_physical_subgroup_id_helper = True
        return local_name

    def hlsl_metal_simd_shuffle_name(self, func_name):
        if not func_name:
            return None
        normalized = str(func_name).strip().replace("_u3a_u3a", "::")
        if normalized.startswith("metal::"):
            normalized = normalized.split("::", 1)[1]
        if normalized in {
            "simd_broadcast",
            "simd_shuffle",
            "simd_shuffle_down",
            "simd_shuffle_up",
            "simd_shuffle_and_fill_up",
        }:
            return normalized
        return None

    def hlsl_metal_simd_shuffle_result_type(self, func_name, args):
        simd_name = self.hlsl_metal_simd_shuffle_name(func_name)
        if simd_name is None or not args:
            return None
        return self.expression_result_type(args[0])

    def hlsl_metal_simd_shuffle_source_overload(self, func_name, args):
        candidates = [
            function
            for function in self.hlsl_function_overloads_by_name.get(func_name, ())
            if len(
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
            == len(args)
        ]
        if not candidates:
            return None
        argument_types = [
            self.hlsl_source_expression_type(argument) for argument in args
        ]
        if any(argument_type is None for argument_type in argument_types):
            return None
        compatible = []
        for function in candidates:
            parameters = list(
                getattr(function, "parameters", getattr(function, "params", [])) or []
            )
            if all(
                self.hlsl_function_type_match_score(
                    argument_type,
                    getattr(parameter, "param_type", getattr(parameter, "vtype", None)),
                )
                is not None
                for argument_type, parameter in zip(argument_types, parameters)
            ):
                compatible.append(function)
        if not compatible:
            return None
        resolved = self.resolve_hlsl_function_overload(func_name, args)
        return resolved if resolved in compatible else None

    def generate_hlsl_metal_simd_shuffle_call(self, func_name, args):
        simd_name = self.hlsl_metal_simd_shuffle_name(func_name)
        if simd_name is None:
            return None
        if self.hlsl_metal_simd_shuffle_source_overload(func_name, args) is not None:
            # Plain Metal simd calls with a compatible source definition represent
            # explicit overload dispatch. Canonical builtin calls arrive as
            # WaveOpNode and continue through the intrinsic lowering paths.
            return None

        value_type = self.expression_result_type(args[0]) if args else None
        complex_value = self.is_hlsl_complex64_type(value_type)

        def complex_wave_read(data, lane):
            self.required_hlsl_complex64_helpers.add(
                "__crossgl_complex64_wave_read_lane_at"
            )
            return f"__crossgl_complex64_wave_read_lane_at({data}, {lane})"

        if simd_name in {"simd_broadcast", "simd_shuffle"} and len(args) == 2:
            data = self.generate_expression_with_expected(args[0], None)
            lane = self.generate_expression_with_expected(args[1], "uint")
            if self.software_subgroup_width is not None:
                self.validate_hlsl_wave_scalar_int_uint_argument(
                    "WaveReadLaneAt", args[1], "lane index"
                )
                return self.hlsl_cast_metal_simd_shuffle_result(
                    self.hlsl_software_subgroup_call(
                        "WaveReadLaneAt", value_type, data, lane
                    )
                )
            if complex_value:
                return complex_wave_read(data, lane)
            return self.hlsl_cast_metal_simd_shuffle_result(
                f"WaveReadLaneAt({data}, {lane})"
            )

        if simd_name == "simd_shuffle_down" and len(args) == 2:
            data = self.generate_expression_with_expected(args[0], None)
            delta = self.generate_expression_with_expected(args[1], "uint")
            if self.software_subgroup_width is not None:
                return self.hlsl_cast_metal_simd_shuffle_result(
                    self.hlsl_software_subgroup_call(
                        "WaveShuffleDown", value_type, data, delta
                    )
                )
            if self.relative_wave_shuffle_out_of_range == "self":
                return self.hlsl_cast_metal_simd_shuffle_result(
                    self.hlsl_defined_relative_wave_shuffle_call(
                        "WaveShuffleDown", value_type, data, delta
                    )
                )
            if complex_value:
                return complex_wave_read(data, f"(WaveGetLaneIndex() + uint({delta}))")
            return self.hlsl_cast_metal_simd_shuffle_result(
                f"WaveReadLaneAt({data}, (WaveGetLaneIndex() + uint({delta})))"
            )

        if simd_name == "simd_shuffle_up" and len(args) == 2:
            data = self.generate_expression_with_expected(args[0], None)
            delta = self.generate_expression_with_expected(args[1], "uint")
            if self.software_subgroup_width is not None:
                return self.hlsl_cast_metal_simd_shuffle_result(
                    self.hlsl_software_subgroup_call(
                        "WaveShuffleUp", value_type, data, delta
                    )
                )
            if self.relative_wave_shuffle_out_of_range == "self":
                return self.hlsl_cast_metal_simd_shuffle_result(
                    self.hlsl_defined_relative_wave_shuffle_call(
                        "WaveShuffleUp", value_type, data, delta
                    )
                )
            if complex_value:
                return complex_wave_read(data, f"(WaveGetLaneIndex() - uint({delta}))")
            return self.hlsl_cast_metal_simd_shuffle_result(
                f"WaveReadLaneAt({data}, (WaveGetLaneIndex() - uint({delta})))"
            )

        if simd_name == "simd_shuffle_and_fill_up" and len(args) in {3, 4}:
            if (
                self.software_subgroup_width is not None
                or len(args) == 4
                or not complex_value
            ):
                return self.generate_hlsl_wave_intrinsic_call(
                    "WaveShuffleAndFillUp", args
                )
            data = self.generate_expression_with_expected(args[0], None)
            filling = self.generate_expression_with_expected(args[1], None)
            delta = self.generate_expression_with_expected(args[2], "uint")
            if complex_value:
                self.required_hlsl_complex64_helpers.update(
                    {
                        "__crossgl_complex64_wave_read_lane_at",
                        "__crossgl_complex64_wave_shuffle_and_fill_up",
                    }
                )
                return (
                    "__crossgl_complex64_wave_shuffle_and_fill_up("
                    f"{data}, {filling}, uint({delta}))"
                )

        return None

    def hlsl_cast_metal_simd_shuffle_result(self, expression):
        expected_type = self.map_type(
            self.type_name_string(self.current_expression_expected_type)
        )
        if expected_type == "bool":
            return f"(({expression}) != 0)"
        return expression

    def hlsl_signbit_type_info(self, type_name):
        if type_name is None:
            return None
        mapped_type = self.map_type(type_name)
        match = re.fullmatch(r"float([234]?)", mapped_type or "")
        if match is None:
            return None
        lanes = int(match.group(1)) if match.group(1) else 1
        result_type = "bool" if lanes == 1 else f"bool{lanes}"
        return {"type": mapped_type, "lanes": lanes, "result_type": result_type}

    def hlsl_signbit_result_type(self, args):
        if len(args) != 1:
            return None
        info = self.hlsl_signbit_type_info(self.hlsl_source_expression_type(args[0]))
        return info["result_type"] if info is not None else None

    def hlsl_signbit_error(
        self,
        message,
        *,
        operand_type=None,
        result_type=None,
        reason,
        source_location=None,
    ):
        return DirectXSignBitError(
            message,
            operand_type=operand_type,
            result_type=result_type,
            target_profile=self.target_profile or "default",
            reason=reason,
            source_location=source_location,
        )

    def generate_hlsl_signbit_call(self, func_name, args, *, call_node=None):
        if func_name not in self.HLSL_SIGNBIT_NAMES:
            return None

        resolved_overload = self.resolve_hlsl_function_overload(
            func_name,
            args,
            call_node=call_node,
        )
        if resolved_overload is not None:
            return None

        source_location = getattr(call_node, "source_location", None)
        if len(args) != 1:
            raise self.hlsl_signbit_error(
                "DirectX signbit requires exactly one operand",
                reason="invalid-arity",
                source_location=source_location,
            )

        source_type = self.hlsl_source_expression_type(args[0])
        display_type = (
            self.type_name_string(source_type) if source_type is not None else None
        )
        if source_type is None:
            raise self.hlsl_signbit_error(
                "DirectX signbit requires a statically known operand type",
                operand_type=display_type,
                reason="unresolved-operand-type",
                source_location=source_location,
            )

        info = self.hlsl_signbit_type_info(source_type)
        if info is None:
            raise self.hlsl_signbit_error(
                "DirectX signbit supports only 32-bit float scalar and vector "
                f"operands, got {display_type}",
                operand_type=display_type,
                reason="unsupported-operand-type",
                source_location=source_location,
            )

        value = self.generate_expression_with_expected(args[0], info["type"])
        if info["lanes"] == 1:
            return f"((asuint({value}) & 0x80000000u) != 0u)"
        lanes = info["lanes"]
        mask = f"uint{lanes}({', '.join(['0x80000000u'] * lanes)})"
        zero = f"uint{lanes}({', '.join(['0u'] * lanes)})"
        return f"((asuint({value}) & {mask}) != {zero})"

    def hlsl_copysign_type_info(self, type_name):
        if type_name is None:
            return None
        mapped_type = self.map_type(type_name)
        match = re.fullmatch(r"float([234]?)", mapped_type or "")
        if match is None:
            return None
        lanes = int(match.group(1)) if match.group(1) else 1
        return {"type": mapped_type, "lanes": lanes}

    def hlsl_copysign_error(
        self,
        message,
        *,
        operand_types,
        result_type=None,
        reason,
        source_location=None,
    ):
        return DirectXCopySignError(
            message,
            operand_types=operand_types,
            result_type=result_type,
            target_profile=self.target_profile or "default",
            reason=reason,
            source_location=source_location,
        )

    def generate_hlsl_copysign_call(
        self,
        func_name,
        args,
        *,
        call_node=None,
    ):
        if func_name not in self.HLSL_COPYSIGN_NAMES:
            return None

        resolved_overload = self.resolve_hlsl_function_overload(
            func_name,
            args,
            call_node=call_node,
        )
        if resolved_overload is not None:
            return None

        source_location = getattr(call_node, "source_location", None)
        if len(args) != 2:
            raise self.hlsl_copysign_error(
                "DirectX copysign requires exactly two operands",
                operand_types=(),
                reason="invalid-arity",
                source_location=source_location,
            )

        source_types = tuple(self.hlsl_source_expression_type(arg) for arg in args)
        display_types = tuple(
            self.type_name_string(source_type) if source_type is not None else None
            for source_type in source_types
        )
        if any(source_type is None for source_type in source_types):
            raise self.hlsl_copysign_error(
                "DirectX copysign requires statically known operand types",
                operand_types=display_types,
                reason="unresolved-operand-type",
                source_location=source_location,
            )

        operand_info = tuple(
            self.hlsl_copysign_type_info(source_type) for source_type in source_types
        )
        if any(info is None for info in operand_info):
            raise self.hlsl_copysign_error(
                "DirectX copysign supports only 32-bit float scalar and vector "
                f"operands, got {display_types[0]} and {display_types[1]}",
                operand_types=display_types,
                reason="unsupported-operand-type",
                source_location=source_location,
            )

        magnitude_info, sign_info = operand_info
        if magnitude_info["type"] != sign_info["type"]:
            raise self.hlsl_copysign_error(
                "DirectX copysign operands must have identical scalar or vector "
                f"shapes, got {display_types[0]} and {display_types[1]}",
                operand_types=display_types,
                result_type=magnitude_info["type"],
                reason="operand-shape-mismatch",
                source_location=source_location,
            )

        value_type = magnitude_info["type"]
        magnitude = self.generate_expression_with_expected(args[0], value_type)
        sign_source = self.generate_expression_with_expected(args[1], value_type)
        lanes = magnitude_info["lanes"]
        mask_type = "uint" if lanes == 1 else f"uint{lanes}"
        magnitude_mask = "0x7fffffffu"
        sign_mask = "0x80000000u"
        if lanes != 1:
            magnitude_mask = f"{mask_type}({', '.join(['0x7fffffffu'] * lanes)})"
            sign_mask = f"{mask_type}({', '.join(['0x80000000u'] * lanes)})"
        return (
            "asfloat(((asuint({magnitude}) & {magnitude_mask}) | "
            "(asuint({sign_source}) & {sign_mask})))"
        ).format(
            magnitude=magnitude,
            magnitude_mask=magnitude_mask,
            sign_source=sign_source,
            sign_mask=sign_mask,
        )

    def hlsl_inverse_hyperbolic_type_info(self, type_name):
        if type_name is None:
            return None
        mapped_type = self.map_type(type_name)
        match = re.fullmatch(
            r"(float16_t|float|half|min16float|min10float|double)([234]?)",
            mapped_type or "",
        )
        if match is None:
            return {"type": mapped_type, "precision": None, "lanes": None}
        precision, lane_text = match.groups()
        return {
            "type": mapped_type,
            "precision": precision,
            "lanes": int(lane_text) if lane_text else 1,
        }

    def hlsl_inverse_hyperbolic_error(
        self,
        message,
        *,
        operation,
        operand_type=None,
        result_type=None,
        reason,
        source_location=None,
    ):
        return DirectXInverseHyperbolicError(
            message,
            operation=operation,
            operand_type=operand_type,
            result_type=result_type,
            target_profile=self.target_profile or "default",
            reason=reason,
            source_location=source_location,
        )

    def generate_hlsl_inverse_hyperbolic_call(
        self,
        func_name,
        args,
        *,
        call_node=None,
    ):
        operation = self.HLSL_INVERSE_HYPERBOLIC_ALIASES.get(func_name)
        if operation is None:
            return None

        resolved_overload = self.resolve_hlsl_function_overload(
            func_name,
            args,
            call_node=call_node,
        )
        if resolved_overload is not None:
            return None

        source_location = getattr(call_node, "source_location", None)
        if len(args) != 1:
            raise self.hlsl_inverse_hyperbolic_error(
                f"DirectX {operation} requires exactly one operand",
                operation=operation,
                reason="invalid-arity",
                source_location=source_location,
            )

        source_type = self.hlsl_source_expression_type(args[0])
        display_type = (
            self.type_name_string(source_type) if source_type is not None else None
        )
        if source_type is None:
            raise self.hlsl_inverse_hyperbolic_error(
                f"DirectX {operation} requires a statically known operand type",
                operation=operation,
                reason="unresolved-operand-type",
                source_location=source_location,
            )

        if self.is_hlsl_bfloat16_type(source_type):
            self.required_hlsl_inverse_hyperbolic_helpers.add((operation, "float"))
            helper_name = self.hlsl_inverse_hyperbolic_helper_name(operation, "float")
            argument = self.generate_expression(args[0])
            decoded = self.hlsl_bfloat16_to_float_expression(argument)
            return self.hlsl_float_to_bfloat16_expression(f"{helper_name}({decoded})")

        type_info = self.hlsl_inverse_hyperbolic_type_info(source_type)
        mapped_type = type_info["type"]
        precision = type_info["precision"]
        if precision == "double":
            if self.target_profile == "dx10":
                reason = "double-precision-profile-unsupported"
                detail = "DirectX 10 cannot represent the required double precision"
            else:
                reason = "double-transcendental-contract-unavailable"
                detail = (
                    "the selected DirectX profile has no portable double-precision "
                    "transcendental contract"
                )
            raise self.hlsl_inverse_hyperbolic_error(
                f"DirectX {operation} cannot lower {display_type}: {detail}",
                operation=operation,
                operand_type=display_type,
                result_type=mapped_type,
                reason=reason,
                source_location=source_location,
            )

        if precision == "min10float":
            raise self.hlsl_inverse_hyperbolic_error(
                f"DirectX {operation} cannot preserve exceptional-value behavior "
                f"for {display_type}",
                operation=operation,
                operand_type=display_type,
                result_type=mapped_type,
                reason="minimum-precision-contract-unsupported",
                source_location=source_location,
            )

        if mapped_type in self.HLSL_INVERSE_HYPERBOLIC_NARROW_TYPES:
            if precision == "min16float" and self.target_profile == "dx10":
                raise self.hlsl_inverse_hyperbolic_error(
                    f"DirectX {operation} cannot use {display_type} with the "
                    "DirectX 10 profile",
                    operation=operation,
                    operand_type=display_type,
                    result_type=mapped_type,
                    reason="minimum-precision-profile-unsupported",
                    source_location=source_location,
                )
        elif mapped_type not in self.HLSL_INVERSE_HYPERBOLIC_FLOAT_TYPES:
            raise self.hlsl_inverse_hyperbolic_error(
                f"DirectX {operation} supports only scalar and 2/3/4-component "
                f"floating operands, got {display_type}",
                operation=operation,
                operand_type=display_type,
                result_type=mapped_type,
                reason="unsupported-operand-type",
                source_location=source_location,
            )

        self.required_hlsl_inverse_hyperbolic_helpers.add((operation, mapped_type))
        helper_name = self.hlsl_inverse_hyperbolic_helper_name(operation, mapped_type)
        argument = self.generate_expression_with_expected(args[0], mapped_type)
        return f"{helper_name}({argument})"

    def hlsl_bitcast_result_type(self, func_name, args):
        if (
            not func_name
            or func_name in getattr(self, "function_return_types", {})
            or not args
        ):
            return None
        target_component = self.HLSL_BITCAST_FUNCTION_TARGETS.get(func_name)
        if target_component is None:
            return None

        source_type = self.expression_result_type(args[0])
        mapped_source_type = self.map_type(source_type or target_component)
        for source_component in ("float", "int", "uint"):
            if mapped_source_type == source_component:
                return target_component
            if mapped_source_type.startswith(source_component):
                suffix = mapped_source_type[len(source_component) :]
                if suffix in {"2", "3", "4"}:
                    return f"{target_component}{suffix}"
        return target_component

    def hlsl_metal_as_type_source_target_type(self, func_name):
        if not func_name or func_name in getattr(self, "function_return_types", {}):
            return None
        match = re.fullmatch(r"(?:metal::)?as_type<(.+)>", str(func_name).strip())
        if match is None:
            return None
        return match.group(1).strip()

    def hlsl_metal_as_type_target_type(self, func_name):
        target_type = self.hlsl_metal_as_type_source_target_type(func_name)
        return self.map_type(target_type) if target_type is not None else None

    def hlsl_explicit_bitcast_type_info(self, type_name):
        if not type_name:
            return None
        source_name = str(self.type_name_string(type_name) or "").strip()
        widened_float16 = re.fullmatch(
            r"(?:float16_t|float16|half|f16)(?P<shape>[234]?)",
            source_name,
        )
        if self.widen_native_float16 and widened_float16 is not None:
            mapped_type = "float16_t" + (widened_float16.group("shape") or "")
        else:
            mapped_type = self.map_type(type_name)
        component_widths = (
            ("uint64_t", 64),
            ("int64_t", 64),
            ("double", 64),
            ("float", 32),
            ("uint", 32),
            ("int", 32),
            ("float16_t", 16),
            ("uint16_t", 16),
            ("int16_t", 16),
        )
        for component_type, component_width in component_widths:
            if mapped_type == component_type:
                return {
                    "type": mapped_type,
                    "component": component_type,
                    "component_width": component_width,
                    "lanes": 1,
                    "width": component_width,
                }
            if mapped_type.startswith(component_type):
                suffix = mapped_type[len(component_type) :]
                if suffix in {"2", "3", "4"}:
                    lanes = int(suffix)
                    return {
                        "type": mapped_type,
                        "component": component_type,
                        "component_width": component_width,
                        "lanes": lanes,
                        "width": component_width * lanes,
                    }
        return None

    def require_hlsl_explicit_bitcast_helper(self, helper_name):
        self.required_hlsl_explicit_bitcast_helpers.add(helper_name)

    def hlsl_explicit_bitcast_uint2_payload(self, source_info, argument_code):
        if source_info["lanes"] == 2 and source_info["component_width"] == 32:
            if source_info["component"] == "uint":
                return argument_code
            return f"asuint({argument_code})"

        if source_info["lanes"] != 1 or source_info["component_width"] != 64:
            return None
        if source_info["component"] == "double":
            self.require_hlsl_explicit_bitcast_helper("double_to_uint2")
            return f"__crossgl_bitcast_double_to_uint2({argument_code})"
        if source_info["component"] in {"int64_t", "uint64_t"}:
            self.require_hlsl_explicit_bitcast_helper("uint64_to_uint2")
            return "__crossgl_bitcast_uint64_to_uint2" f"(uint64_t({argument_code}))"
        return None

    def hlsl_explicit_bitcast_from_uint2(self, target_info, payload_code):
        if target_info["lanes"] == 2 and target_info["component_width"] == 32:
            if target_info["component"] == "uint":
                return payload_code
            intrinsic = "asint" if target_info["component"] == "int" else "asfloat"
            return f"{intrinsic}({payload_code})"

        if target_info["lanes"] != 1 or target_info["component_width"] != 64:
            return None
        if target_info["component"] == "double":
            self.require_hlsl_explicit_bitcast_helper("uint2_to_double")
            return f"__crossgl_bitcast_uint2_to_double({payload_code})"
        if target_info["component"] in {"int64_t", "uint64_t"}:
            self.require_hlsl_explicit_bitcast_helper("uint2_to_uint64")
            wide_value = f"__crossgl_bitcast_uint2_to_uint64({payload_code})"
            if target_info["component"] == "int64_t":
                return f"int64_t({wide_value})"
            return wide_value
        return None

    def hlsl_explicit_bitcast_expression(self, target_type, source_type, argument_code):
        target_info = self.hlsl_explicit_bitcast_type_info(target_type)
        source_info = self.hlsl_explicit_bitcast_type_info(source_type)
        if target_info is None or source_info is None:
            return None
        if target_info["width"] != source_info["width"]:
            raise DirectXRefusal(
                f"DirectX cannot bitcast {source_info['type']} "
                f"({source_info['width']} bits) to {target_info['type']} "
                f"({target_info['width']} bits); total bit widths must match"
            )
        if target_info["type"] == source_info["type"]:
            return argument_code

        if target_info["width"] == 32 and {
            source_info["component_width"],
            target_info["component_width"],
        } == {16, 32}:
            unpack = target_info["component_width"] == 16
            narrow = target_info if unpack else source_info
            helper = "uint_to_uint16x2" if unpack else "uint16x2_to_uint"
            name = f"__crossgl_bitcast_{helper}"
            if name in self.function_return_types or name in self.global_variable_types:
                raise DirectXRefusal(
                    f"DirectX packed 16-bit bitcasts require unshadowed {name}"
                )
            if (
                self.widen_native_float16
                and narrow["component"] == "float16_t"
                and not unpack
            ):
                raise DirectXRefusal(
                    "DirectX cannot repack widened float16 values without binary16 rounding"
                )
            self.require_hlsl_explicit_bitcast_helper(helper)
            self.requires_hlsl_native_16_bit_types = True
            if unpack:
                payload = argument_code
                if source_info["component"] != "uint":
                    payload = f"asuint({payload})"
                payload = f"{name}({payload})"
                if narrow["component"] == "uint16_t":
                    return payload
                if self.widen_native_float16 and narrow["component"] == "float16_t":
                    self.require_hlsl_explicit_bitcast_helper("binary16_to_float")
                    return f"__crossgl_binary16_to_float(uint2({payload}))"
                intrinsic = (
                    "asfloat16" if narrow["component"] == "float16_t" else "asint16"
                )
                return f"{intrinsic}({payload})"
            payload = (
                argument_code
                if narrow["component"] == "uint16_t"
                else f"asuint16({argument_code})"
            )
            packed = f"{name}({payload})"
            if target_info["component"] == "uint":
                return packed
            intrinsic = "asfloat" if target_info["component"] == "float" else "asint"
            return f"{intrinsic}({packed})"

        if target_info["width"] == 64:
            payload = self.hlsl_explicit_bitcast_uint2_payload(
                source_info, argument_code
            )
            if payload is not None:
                result = self.hlsl_explicit_bitcast_from_uint2(target_info, payload)
                if result is not None:
                    return result

        if (
            target_info["lanes"] == source_info["lanes"]
            and target_info["component_width"] == 16
            and source_info["component_width"] == 16
        ):
            if (
                self.widen_native_float16
                and source_info["component"] == "float16_t"
                and target_info["component"] != "float16_t"
            ):
                raise DirectXRefusal(
                    "DirectX widen_native_float16 cannot bitcast a widened "
                    "float32 value back to a native 16-bit payload without "
                    "reintroducing binary16 rounding"
                )
            if self.widen_native_float16 and target_info["component"] == "float16_t":
                width = target_info["lanes"]
                uint_type = "uint" if width == 1 else f"uint{width}"
                payload = argument_code
                if source_info["component"] != "uint16_t":
                    payload = f"asuint16({payload})"
                self.require_hlsl_explicit_bitcast_helper("binary16_to_float")
                return "__crossgl_binary16_to_float(" f"{uint_type}({payload}))"
            intrinsic = {
                "float16_t": "asfloat16",
                "int16_t": "asint16",
                "uint16_t": "asuint16",
            }[target_info["component"]]
            return f"{intrinsic}({argument_code})"

        if (
            target_info["lanes"] == source_info["lanes"]
            and target_info["component_width"] == 32
            and source_info["component_width"] == 32
        ):
            intrinsic = {
                "float": "asfloat",
                "int": "asint",
                "uint": "asuint",
            }[target_info["component"]]
            return f"{intrinsic}({argument_code})"

        raise DirectXRefusal(
            "DirectX explicit bitcast lowering does not support "
            f"{source_info['type']} to {target_info['type']}; use a matching "
            "32-bit scalar/vector shape or a 64-bit scalar and 2-lane payload"
        )

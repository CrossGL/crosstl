"""Stage semantics in the HLSL generator: member and struct default semantics, mesh output counts, patch inputs, semantic ranges and slots, and vertex and tessellation interfaces."""

import re

from ..ast import (
    ArrayNode,
    IdentifierNode,
    StructMemberNode,
    StructNode,
    VariableNode,
)
from ..refusals import DirectXRefusal
from .array_utils import (
    parse_array_type,
    split_array_type_suffix,
)
from .directx_values import (
    DirectXMeshStageProfileError,
    DirectXRayTracingStageProfileError,
    DirectXSemanticArraySizeError,
)
from .hlsl_glsl_tessellation_interface import refuse_glsl_style_tessellation_interface
from .hlsl_refusals import DirectXStageAttributesMissingError
from .hlsl_render_targets import is_render_target_value_type
from .hlsl_struct_varyings import assign_struct_varying_semantic
from .hlsl_vk_locations import hlsl_location_replaces_semantic
from .interface_conflicts import InterfaceLocationConflictError
from .stage_utils import (
    assign_stage_entry_names,
    collect_stage_entry_records,
    collect_stage_entry_reserved_function_names,
    normalize_stage_name,
    requests_early_fragment_tests,
    stage_matches,
)


class DirectXStageSemanticMixin:
    """Stage semantic members of ``HLSLCodeGen``."""

    def validate_hlsl_dispatch_mesh_payload_storage(self, func):
        declarations = self.hlsl_function_visible_variable_declarations(func)
        for args in self.hlsl_dispatch_mesh_calls(func):
            if len(args) < 4:
                continue

            payload_expr = args[3]
            payload_name = self.expression_name(payload_expr)
            declaration = declarations.get(payload_name)
            if (
                payload_name is None
                or declaration is None
                or not self.hlsl_declaration_has_groupshared_qualifier(declaration)
            ):
                raise DirectXRefusal(
                    "DirectX amplification DispatchMesh payload argument must be "
                    "a groupshared user-defined struct variable"
                )

    def validate_hlsl_dispatch_mesh_payloads(self, func, shader_type):
        if shader_type not in {"task", "amplification", "object"}:
            return

        payload_types = self.hlsl_dispatch_mesh_payload_types_for_function(func)
        if not payload_types:
            return
        for reachable_func in self.hlsl_function_and_reachable_helpers(func):
            self.validate_hlsl_dispatch_mesh_payload_storage(reachable_func)

        mesh_payload_types = self.current_hlsl_mesh_payload_types
        for payload_type in payload_types:
            if payload_type is None:
                if mesh_payload_types:
                    raise DirectXRefusal(
                        "DirectX amplification DispatchMesh call must pass a "
                        "payload argument when the mesh stage declares a "
                        "payload parameter"
                    )
                continue

            if payload_type not in self.structs_by_name:
                raise DirectXRefusal(
                    "DirectX amplification DispatchMesh payload argument must "
                    "be a user-defined struct value"
                )

            if mesh_payload_types and payload_type not in mesh_payload_types:
                expected = ", ".join(sorted(mesh_payload_types))
                raise DirectXRefusal(
                    "DirectX amplification DispatchMesh payload type "
                    f"'{payload_type}' must match mesh payload type(s): "
                    f"{expected}"
                )

    def validate_hlsl_patch_constant_function(self, func, shader_type):
        if shader_type != "tessellation_control":
            return

        patch_function_name = self.hlsl_stage_attribute_argument(
            func, "patchconstantfunc"
        )
        if not patch_function_name:
            return

        parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
        available_functions = self.current_hlsl_available_functions or {}
        patch_function = available_functions.get(patch_function_name)
        if patch_function is None or patch_function is func:
            if not parameters:
                return
            raise DirectXRefusal(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' must reference an emitted helper "
                "function in the global or tessellation_control stage scope"
            )

        return_type = self.type_name_string(
            getattr(
                patch_function, "return_type", getattr(patch_function, "vtype", None)
            )
        )
        if return_type is None or self.map_type(return_type) == "void":
            raise DirectXRefusal(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' must return patch-constant data"
            )

        patch_parameters = (
            getattr(patch_function, "parameters", getattr(patch_function, "params", []))
            or []
        )
        if patch_parameters and "InputPatch" not in {
            self.hlsl_parameter_type_base(parameter) for parameter in patch_parameters
        }:
            raise DirectXRefusal(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' parameters must include "
                "InputPatch<T, N>"
            )
        self.validate_hlsl_patch_parameter_signatures(
            patch_parameters,
            "InputPatch",
            f"tessellation_control stage patchconstantfunc '{patch_function_name}'",
        )
        self.validate_hlsl_patch_constant_input_patch_signature(
            parameters, patch_parameters, patch_function_name
        )
        self.validate_hlsl_scalar_integer_semantic_types(
            patch_parameters,
            "tessellation_control patchconstantfunc",
            ("SV_PrimitiveID",),
        )
        self.validate_hlsl_patch_constant_tess_factor_semantics(
            func, patch_function, patch_function_name
        )

    def validate_hlsl_patch_constant_input_patch_signature(
        self, hull_parameters, patch_parameters, patch_function_name
    ):
        if not patch_parameters:
            return

        hull_element_type = self.hlsl_patch_element_type(hull_parameters, "InputPatch")
        patch_element_type = self.hlsl_patch_element_type(
            patch_parameters, "InputPatch"
        )
        if (
            hull_element_type is not None
            and patch_element_type is not None
            and patch_element_type != hull_element_type
        ):
            raise DirectXRefusal(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' InputPatch element type "
                f"'{patch_element_type}' must match tessellation_control "
                f"InputPatch element type '{hull_element_type}'"
            )

        hull_control_points = self.hlsl_input_patch_control_point_count(hull_parameters)
        patch_control_points = self.hlsl_input_patch_control_point_count(
            patch_parameters
        )
        if (
            hull_control_points is not None
            and patch_control_points is not None
            and patch_control_points != hull_control_points
        ):
            raise DirectXRefusal(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' InputPatch control point count "
                f"({patch_control_points}) must match tessellation_control "
                f"InputPatch control point count ({hull_control_points})"
            )

    def hlsl_semantic_key(self, semantic):
        mapped_semantic = self.hlsl_semantic_name(semantic)
        return str(mapped_semantic).lower()

    def hlsl_struct_member_declared_semantic(self, member):
        return self.semantic_from_struct_member(member)

    def hlsl_semantic_range_start(self, semantic):
        mapped_semantic = self.hlsl_semantic_name(semantic)
        semantic_text = str(mapped_semantic)
        suffix_start = len(semantic_text)
        while suffix_start > 0 and semantic_text[suffix_start - 1].isdigit():
            suffix_start -= 1

        base = semantic_text[:suffix_start] or semantic_text
        suffix = semantic_text[suffix_start:]
        return base.lower(), int(suffix) if suffix else 0

    def hlsl_struct_member_semantic_slot_count(self, struct_name, member, semantic):
        count = self.hlsl_struct_member_array_slot_count(struct_name, member, semantic)
        return count * self.hlsl_matrix_semantic_slot_count(member)

    def hlsl_matrix_semantic_slot_count(self, member):
        """The semantic indices one element of a matrix member takes: a column-major matrix (HLSL's default)
        takes one per column and a row-major one one per row; anything else takes one.
        """
        raw_type = self.hlsl_struct_member_raw_type(member)
        element = (
            getattr(raw_type, "element_type", None)
            if hasattr(raw_type, "size")
            else None
        )
        type_name = self.map_type(element if element is not None else raw_type)
        base_type, _suffix = split_array_type_suffix(str(type_name or ""))
        shape = re.fullmatch(
            r"(?:float|half|double|int|uint|bool|min16float|float16_t)([1-4])x([1-4])",
            base_type,
        )
        if shape is None:
            return 1
        rows, columns = int(shape.group(1)), int(shape.group(2))
        return (
            rows
            if self.hlsl_matrix_orientation_prefix(member).strip() == "row_major"
            else columns
        )

    def hlsl_struct_member_array_slot_count(self, struct_name, member, semantic):
        raw_type = self.hlsl_struct_member_raw_type(member)
        size_expr = None
        if isinstance(member, ArrayNode):
            size_expr = getattr(member, "size", None)
        elif raw_type is not None and str(type(raw_type)).find("ArrayType") != -1:
            size_expr = getattr(raw_type, "size", None)
        else:
            type_name = self.type_name_string(raw_type)
            _base_type, array_suffix = split_array_type_suffix(str(type_name or ""))
            if not array_suffix:
                return 1
            first_dimension = array_suffix[1:].split("]", 1)[0]
            size_expr = first_dimension if first_dimension else None

        count = self.literal_int_value(size_expr, self.literal_int_constants)
        if count is not None and count > 0:
            return count

        member_name = getattr(member, "name", "<anonymous>")
        semantic_name = self.hlsl_semantic_name(semantic)
        size_label = (
            self.expression_to_string(size_expr) if size_expr is not None else ""
        )
        size_suffix = f" from '{size_label}'" if size_label else ""
        raise DirectXSemanticArraySizeError(
            f"DirectX struct '{struct_name}' member '{member_name}' uses "
            f"array-valued semantic '{semantic_name}' but its array size"
            f"{size_suffix} cannot be resolved to a positive integer. "
            "Use a literal or compile-time integer constant array size before "
            "translating stage input/output semantics to DirectX."
        )

    def hlsl_parameter_semantic_slot_count(self, parameter, semantic):
        size_expr = self.hlsl_parameter_array_size_expression(parameter)
        if size_expr is None:
            return 1

        count = self.literal_int_value(size_expr, self.literal_int_constants)
        if count is not None and count > 0:
            return count

        parameter_name = getattr(parameter, "name", "<anonymous>")
        semantic_name = self.hlsl_semantic_name(semantic)
        size_label = (
            self.expression_to_string(size_expr) if size_expr is not None else ""
        )
        size_suffix = f" from '{size_label}'" if size_label else ""
        raise DirectXSemanticArraySizeError(
            f"DirectX entry parameter '{parameter_name}' uses array-valued "
            f"semantic '{semantic_name}' but its array size{size_suffix} "
            "cannot be resolved to a positive integer. Use a literal or "
            "compile-time integer constant array size before translating "
            "stage input/output semantics to DirectX."
        )

    def hlsl_reserve_semantic_range(
        self, used_ranges, struct_name, member_name, semantic, count
    ):
        semantic_base, start = self.hlsl_semantic_range_start(semantic)
        count = max(count or 1, 1)
        end = start + count - 1
        ranges = used_ranges.setdefault(semantic_base, [])
        for used_start, used_end, used_member in ranges:
            if start <= used_end and used_start <= end:
                range_label = self.hlsl_semantic_range_label(semantic, start, end)
                used_label = self.hlsl_semantic_range_label(
                    semantic, used_start, used_end
                )
                raise InterfaceLocationConflictError(
                    "directx",
                    f"Conflicting DirectX semantic range in struct '{struct_name}' "
                    f"for member '{member_name}': {range_label} overlaps "
                    f"member '{used_member}' {used_label}",
                )
        ranges.append((start, end, member_name))

    def hlsl_semantic_range_label(self, semantic, start, end):
        semantic_base, _ = self.hlsl_semantic_range_start(semantic)
        if start == end:
            return f"{semantic_base.upper()}{start}"
        return f"{semantic_base.upper()}{start}-{semantic_base.upper()}{end}"

    def hlsl_semantic_range_is_used(self, used_ranges, semantic, count=1):
        semantic_base, start = self.hlsl_semantic_range_start(semantic)
        end = start + max(count or 1, 1) - 1
        for used_start, used_end, _used_member in used_ranges.get(semantic_base, []):
            if start <= used_end and used_start <= end:
                return True
        return False

    def hlsl_next_available_semantic_index(
        self, used_ranges, semantic_base, start, count
    ):
        semantic_key, _ = self.hlsl_semantic_range_start(semantic_base)
        count = max(count or 1, 1)
        index = max(start, 0)
        ranges = used_ranges.get(semantic_key, [])
        while True:
            end = index + count - 1
            conflict_end = None
            for used_start, used_end, _used_member in ranges:
                if index <= used_end and used_start <= end:
                    conflict_end = (
                        used_end
                        if conflict_end is None
                        else max(conflict_end, used_end)
                    )
            if conflict_end is None:
                return index
            index = conflict_end + 1

    def hlsl_declared_struct_semantic_ranges(self, struct_node):
        used_ranges = {}
        struct_name = getattr(struct_node, "name", "<anonymous>")
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.hlsl_struct_member_declared_semantic(member)
            if semantic is None or hlsl_location_replaces_semantic(self, member):
                continue
            count = self.hlsl_struct_member_semantic_slot_count(
                struct_name, member, semantic
            )
            self.hlsl_reserve_semantic_range(
                used_ranges,
                struct_name,
                getattr(member, "name", "<anonymous>"),
                semantic,
                count,
            )
        return used_ranges

    def hlsl_can_default_fragment_input_semantic(self, member_type):
        type_name = self.type_name_string(member_type)
        if not type_name:
            return False

        base_type, _array_size = parse_array_type(str(type_name))
        mapped_type = self.map_type(base_type)
        if mapped_type in self.structs_by_name:
            return False

        return not (
            self.is_resource_parameter_type(mapped_type)
            or self.is_hlsl_readonly_buffer_type(mapped_type)
            or self.is_hlsl_uav_buffer_type(mapped_type)
        )

    def hlsl_default_vertex_input_member_semantics(self, struct_node):
        if (
            getattr(struct_node, "name", None)
            not in self.vertex_entry_input_struct_names
        ):
            return {}

        used_ranges = self.hlsl_declared_struct_semantic_ranges(struct_node)

        defaults = {}
        position_names = {"position", "vertexposition", "vertex_position", "pos"}
        if not self.hlsl_semantic_range_is_used(used_ranges, "POSITION"):
            for member in getattr(struct_node, "members", []) or []:
                member_name = getattr(member, "name", None)
                if not member_name:
                    continue
                if member_name.lower() not in position_names:
                    continue
                if self.hlsl_struct_member_declared_semantic(member) is not None:
                    continue
                if not self.hlsl_can_default_fragment_input_semantic(
                    self.hlsl_struct_member_type_name(member)
                ):
                    continue
                defaults[member_name] = "POSITION"
                self.hlsl_reserve_semantic_range(
                    used_ranges,
                    struct_node.name,
                    member_name,
                    "POSITION",
                    1,
                )
                break

        next_texcoord = 0
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name or member_name in defaults:
                continue

            declared_semantic = self.hlsl_struct_member_declared_semantic(member)
            location_index = self.hlsl_location_attribute_index(member)
            if declared_semantic is not None and not hlsl_location_replaces_semantic(
                self, member
            ):
                continue
            if not self.hlsl_can_default_fragment_input_semantic(
                self.hlsl_struct_member_type_name(member)
            ):
                continue

            if location_index is not None:
                next_texcoord = location_index
            count = self.hlsl_struct_member_semantic_slot_count(
                struct_node.name, member, f"TEXCOORD{next_texcoord}"
            )
            next_texcoord = self.hlsl_next_available_semantic_index(
                used_ranges, "TEXCOORD", next_texcoord, count
            )
            semantic = f"TEXCOORD{next_texcoord}"
            defaults[member_name] = semantic
            self.hlsl_reserve_semantic_range(
                used_ranges,
                struct_node.name,
                member_name,
                semantic,
                count,
            )
            next_texcoord += count
        return defaults

    def hlsl_parameter_signature_directions(self, parameter):
        """The entry signatures (``in``, ``out``) a parameter's semantic belongs to.

        The input and output signatures of an HLSL entry point are separate, so
        ``float2 uv : TEXCOORD0`` and ``out float2 o : TEXCOORD0`` do not overlap.
        """
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", None) or []
        }
        if "inout" in qualifiers:
            return ("in", "out")
        if "out" in qualifiers:
            return ("out",)
        return ("in",)

    def hlsl_default_vertex_parameter_semantics(self, parameters, func):
        used_ranges = {"in": {}, "out": {}}
        for parameter in parameters or []:
            semantic = self.semantic_from_node(parameter)
            if semantic is None:
                continue
            count = self.hlsl_parameter_semantic_slot_count(parameter, semantic)
            for direction in self.hlsl_parameter_signature_directions(parameter):
                self.hlsl_reserve_semantic_range(
                    used_ranges[direction],
                    getattr(func, "name", "<entry>"),
                    getattr(parameter, "name", "<anonymous>"),
                    semantic,
                    count,
                )

        defaults = {}
        next_texcoord = {"in": 0, "out": 0}
        for parameter in parameters or []:
            if self.semantic_from_node(parameter) is not None:
                continue
            if self.hlsl_entry_resource_parameter_global_type(parameter, func):
                continue
            if self.hlsl_parameter_user_struct_type(parameter) is not None:
                continue
            if not self.hlsl_can_default_fragment_input_semantic(
                self.hlsl_parameter_raw_type(parameter)
            ):
                continue

            directions = self.hlsl_parameter_signature_directions(parameter)
            start = max(next_texcoord[direction] for direction in directions)
            count = self.hlsl_parameter_semantic_slot_count(
                parameter, f"TEXCOORD{start}"
            )
            index = start
            while True:
                candidates = {
                    self.hlsl_next_available_semantic_index(
                        used_ranges[direction], "TEXCOORD", index, count
                    )
                    for direction in directions
                }
                if len(candidates) == 1:
                    index = candidates.pop()
                    break
                index = max(candidates)
            semantic = f"TEXCOORD{index}"
            defaults[id(parameter)] = semantic
            for direction in directions:
                self.hlsl_reserve_semantic_range(
                    used_ranges[direction],
                    getattr(func, "name", "<entry>"),
                    getattr(parameter, "name", "<anonymous>"),
                    semantic,
                    count,
                )
                next_texcoord[direction] = index + count
        return defaults

    def hlsl_default_vertex_output_member_semantics(self, struct_node):
        if (
            getattr(struct_node, "name", None)
            not in self.vertex_entry_output_struct_names
        ):
            return {}

        used_ranges = self.hlsl_declared_struct_semantic_ranges(struct_node)

        defaults = {}
        position_names = {
            "position",
            "clipposition",
            "clip_position",
            "glposition",
            "gl_position",
        }
        if not self.hlsl_semantic_range_is_used(used_ranges, "SV_Position"):
            for member in getattr(struct_node, "members", []) or []:
                member_name = getattr(member, "name", None)
                if not member_name:
                    continue
                if member_name.lower() not in position_names:
                    continue
                if self.hlsl_struct_member_declared_semantic(member) is not None:
                    continue
                if self.map_type(self.hlsl_struct_member_type_name(member)) != "float4":
                    continue
                defaults[member_name] = "SV_Position"
                self.hlsl_reserve_semantic_range(
                    used_ranges,
                    struct_node.name,
                    member_name,
                    "SV_Position",
                    1,
                )
                break

        next_texcoord = 0
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name:
                continue
            if member_name in defaults:
                continue
            struct_next = assign_struct_varying_semantic(
                self, struct_node, member, used_ranges, defaults
            )
            if struct_next is not None:
                next_texcoord = struct_next
                continue
            if not self.hlsl_can_default_fragment_input_semantic(
                self.hlsl_struct_member_type_name(member)
            ):
                continue

            declared_semantic = self.hlsl_struct_member_declared_semantic(member)
            location_index = self.hlsl_location_attribute_index(member)
            if declared_semantic is not None and not hlsl_location_replaces_semantic(
                self, member
            ):
                continue

            if location_index is not None:
                next_texcoord = location_index
            count = self.hlsl_struct_member_semantic_slot_count(
                struct_node.name, member, f"TEXCOORD{next_texcoord}"
            )
            next_texcoord = self.hlsl_next_available_semantic_index(
                used_ranges, "TEXCOORD", next_texcoord, count
            )
            semantic = f"TEXCOORD{next_texcoord}"
            defaults[member_name] = semantic
            self.hlsl_reserve_semantic_range(
                used_ranges,
                struct_node.name,
                member_name,
                semantic,
                count,
            )
            next_texcoord += count
        return defaults

    def hlsl_default_fragment_input_member_semantics(self, struct_node):
        if (
            getattr(struct_node, "name", None)
            not in self.fragment_entry_input_struct_names
        ):
            return {}
        return self.hlsl_default_location_member_semantics(struct_node)

    def hlsl_default_geometry_vertex_member_semantics(self, struct_node):
        # A geometry stage's input and output vertices link with the neighbouring stages' varyings, which
        # take TEXCOORD<location> (vertex outputs and fragment inputs above).
        if (
            getattr(struct_node, "name", None)
            not in self.geometry_entry_vertex_struct_names
        ):
            return {}
        return self.hlsl_default_location_member_semantics(struct_node)

    def hlsl_default_location_member_semantics(self, struct_node, first_texcoord=0):
        used_ranges = self.hlsl_declared_struct_semantic_ranges(struct_node)

        defaults = {}
        next_texcoord = first_texcoord
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name:
                continue
            struct_next = assign_struct_varying_semantic(
                self, struct_node, member, used_ranges, defaults
            )
            if struct_next is not None:
                next_texcoord = struct_next
                continue
            if not self.hlsl_can_default_fragment_input_semantic(
                self.hlsl_struct_member_type_name(member)
            ):
                continue

            declared_semantic = self.hlsl_struct_member_declared_semantic(member)
            location_index = self.hlsl_location_attribute_index(member)
            if declared_semantic is not None and not hlsl_location_replaces_semantic(
                self, member
            ):
                continue

            if location_index is not None:
                next_texcoord = location_index
            count = self.hlsl_struct_member_semantic_slot_count(
                struct_node.name, member, f"TEXCOORD{next_texcoord}"
            )
            next_texcoord = self.hlsl_next_available_semantic_index(
                used_ranges, "TEXCOORD", next_texcoord, count
            )
            semantic = f"TEXCOORD{next_texcoord}"
            defaults[member_name] = semantic
            self.hlsl_reserve_semantic_range(
                used_ranges,
                struct_node.name,
                member_name,
                semantic,
                count,
            )
            next_texcoord += count
        return defaults

    def hlsl_default_fragment_output_member_semantics(self, struct_node):
        if (
            getattr(struct_node, "name", None)
            not in self.fragment_entry_output_struct_names
        ):
            return {}

        used_ranges = self.hlsl_declared_struct_semantic_ranges(struct_node)

        defaults = {}
        # A fragment output's location is its render target index.
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            location_index = self.hlsl_location_attribute_index(member)
            if not member_name or location_index is None:
                continue
            semantic = (
                "SV_TARGET" if location_index == 0 else f"SV_TARGET{location_index}"
            )
            self.hlsl_reserve_semantic_range(
                used_ranges, struct_node.name, member_name, semantic, 1
            )
            defaults[member_name] = semantic

        next_target = 0
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name or member_name in defaults:
                continue
            if self.hlsl_struct_member_declared_semantic(member) is not None:
                continue

            mapped_type = self.map_type(self.hlsl_struct_member_type_name(member))
            normalized_name = str(member_name).replace("_", "").lower()
            if normalized_name in {"depth", "fragdepth", "glfragdepth"}:
                if mapped_type == "float" and not self.hlsl_semantic_range_is_used(
                    used_ranges, "SV_DEPTH"
                ):
                    defaults[member_name] = "SV_DEPTH"
                    self.hlsl_reserve_semantic_range(
                        used_ranges,
                        struct_node.name,
                        member_name,
                        "SV_DEPTH",
                        1,
                    )
                continue

            if mapped_type != "float4":
                continue
            while self.hlsl_semantic_range_is_used(
                used_ranges, f"SV_TARGET{next_target}"
            ) or (
                next_target == 0
                and self.hlsl_semantic_range_is_used(used_ranges, "SV_TARGET")
            ):
                next_target += 1
            semantic = "SV_TARGET" if next_target == 0 else f"SV_TARGET{next_target}"
            defaults[member_name] = semantic
            self.hlsl_reserve_semantic_range(
                used_ranges,
                struct_node.name,
                member_name,
                semantic,
                1,
            )
            next_target += 1
        return defaults

    def hlsl_default_struct_member_semantics(self, struct_node):
        defaults = self.hlsl_default_vertex_input_member_semantics(struct_node)
        defaults.update(self.hlsl_default_vertex_output_member_semantics(struct_node))
        defaults.update(self.hlsl_default_fragment_input_member_semantics(struct_node))
        defaults.update(self.hlsl_default_fragment_output_member_semantics(struct_node))
        defaults.update(self.hlsl_default_geometry_vertex_member_semantics(struct_node))
        defaults.update(self.hlsl_default_mesh_output_member_semantics(struct_node))
        return defaults

    def hlsl_default_mesh_output_member_semantics(self, struct_node):
        # Mesh vertex and primitive outputs link with the pixel inputs, which default to TEXCOORD<location>;
        # hlsl_mesh_semantics.py starts the primitive members after the vertex members.
        first_texcoord = self.hlsl_mesh_output_struct_offsets.get(
            getattr(struct_node, "name", None)
        )
        if first_texcoord is None:
            return {}
        return self.hlsl_default_location_member_semantics(struct_node, first_texcoord)

    def default_hlsl_stage_return_semantic(self, shader_type, return_type):
        if shader_type == "vertex" and return_type == "float4":
            return "gl_Position"
        if shader_type == "fragment" and is_render_target_value_type(return_type):
            # A fragment function's value return is colour attachment 0 (MSL 5.2.3.4, GLSL 4.60 section 4.4.2),
            # whatever its numeric scalar or vector type.
            return "gl_FragColor"
        return None

    def hlsl_return_struct(self, func):
        return_type = self.type_name_string(
            getattr(func, "return_type", getattr(func, "vtype", None))
        )
        if not return_type:
            return None
        base_type = return_type.split("<", 1)[0].split("[", 1)[0].strip()
        return self.structs_by_name.get(base_type)

    def hlsl_struct_semantics(self, struct_node):
        semantics = set()
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            if semantic is None:
                continue
            mapped_semantic = self.hlsl_semantic_name(semantic)
            semantics.add(str(mapped_semantic).lower())
        return semantics

    def hlsl_struct_semantic_member_counts(self, struct_node):
        counts = {}
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            if semantic is None:
                continue
            mapped_semantic = self.hlsl_semantic_name(semantic)
            semantic_key = str(mapped_semantic).lower()
            member_type = getattr(member, "member_type", getattr(member, "vtype", None))
            if member_type is None and hasattr(member, "element_type"):
                member_type = member.element_type
            self.validate_hlsl_tess_factor_member_semantic_type(
                getattr(struct_node, "name", "<anonymous>"),
                getattr(member, "name", "<anonymous>"),
                member_type,
                semantic,
                mapped_semantic,
            )
            count = self.hlsl_tess_factor_member_count(member)
            if count is not None:
                counts[semantic_key] = counts.get(semantic_key, 0) + count
        return counts

    def hlsl_tess_factor_member_count(self, member):
        member_type = getattr(member, "member_type", getattr(member, "vtype", None))
        if member_type is None and hasattr(member, "element_type"):
            member_type = member.element_type

        type_name = self.type_name_string(member_type)
        mapped_type = self.map_type(type_name)
        _base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix:
            first_dimension = array_suffix[1:].split("]", 1)[0]
            try:
                return int(first_dimension)
            except ValueError:
                return None

        for scalar_base in ("min16float", "float", "half", "double"):
            if mapped_type.startswith(scalar_base):
                suffix = mapped_type[len(scalar_base) :]
                if suffix in {"2", "3", "4"}:
                    return int(suffix)
                if suffix == "":
                    return 1
        return None

    def validate_hlsl_patch_constant_tess_factor_semantics(
        self, hull_func, patch_function, patch_function_name
    ):
        return_struct = self.hlsl_return_struct(patch_function)
        if return_struct is None:
            return_type = self.type_name_string(
                getattr(
                    patch_function,
                    "return_type",
                    getattr(patch_function, "vtype", None),
                )
            )
            raise DirectXRefusal(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' must return a struct containing "
                "SV_TessFactor member data; direct returns such as "
                f"'{return_type or '<unknown>'}' cannot represent HLSL "
                "tessellation factor arrays"
            )

        semantics = self.hlsl_struct_semantics(return_struct)
        if "sv_tessfactor" not in semantics:
            raise DirectXRefusal(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' must return a struct containing "
                "SV_TessFactor"
            )

        normalized_domain = self.normalized_hlsl_stage_attribute_argument(
            hull_func, "domain"
        )
        if normalized_domain in {"tri", "triangle", "quad"} and (
            "sv_insidetessfactor" not in semantics
        ):
            raise DirectXRefusal(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' must return a struct containing "
                f"SV_InsideTessFactor for {normalized_domain} domains"
            )
        factor_counts = self.hlsl_struct_semantic_member_counts(return_struct)
        expected_outer_counts = {
            "tri": 3,
            "triangle": 3,
            "quad": 4,
            "isoline": 2,
        }
        expected_inner_counts = {
            "tri": 1,
            "triangle": 1,
            "quad": 2,
        }
        expected_outer_count = expected_outer_counts.get(normalized_domain)
        expected_inner_count = expected_inner_counts.get(normalized_domain)
        outer_count = factor_counts.get("sv_tessfactor")
        inner_count = factor_counts.get("sv_insidetessfactor")

        if (
            expected_outer_count is not None
            and outer_count is not None
            and outer_count != expected_outer_count
        ):
            raise DirectXRefusal(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' must return {expected_outer_count} "
                f"SV_TessFactor value(s) for {normalized_domain} domains"
            )
        if (
            expected_inner_count is not None
            and inner_count is not None
            and inner_count != expected_inner_count
        ):
            raise DirectXRefusal(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' must return {expected_inner_count} "
                f"SV_InsideTessFactor value(s) for {normalized_domain} domains"
            )
        if expected_inner_count is None and inner_count is not None:
            raise DirectXRefusal(
                "DirectX tessellation_control stage patchconstantfunc "
                f"'{patch_function_name}' must not return SV_InsideTessFactor "
                f"for {normalized_domain} domains"
            )

    def validate_hlsl_tessellation_domain_topology(self, func, shader_type):
        if shader_type != "tessellation_control":
            return

        domain = self.normalized_hlsl_stage_attribute_argument(func, "domain")
        topology = self.normalized_hlsl_stage_attribute_argument(func, "outputtopology")
        if not domain or not topology:
            return

        if domain in {"tri", "triangle", "quad"} and topology == "line":
            raise DirectXRefusal(
                "DirectX tessellation_control stage domain "
                f"'{domain}' requires outputtopology triangle_cw or "
                "triangle_ccw"
            )
        if domain == "isoline" and topology in {"triangle_cw", "triangle_ccw"}:
            raise DirectXRefusal(
                "DirectX tessellation_control stage domain 'isoline' "
                "requires outputtopology line"
            )

    def validate_hlsl_tessellation_domain(self, func, shader_type):
        if shader_type not in {"tessellation_control", "tessellation_evaluation"}:
            return

        domain = self.normalized_hlsl_stage_attribute_argument(func, "domain")
        if not domain:
            return

        canonical_domain = self.canonical_hlsl_tessellation_domain(domain)
        valid_domains = {"tri", "quad", "isoline"}
        if canonical_domain not in valid_domains:
            valid_values = ", ".join(sorted(valid_domains))
            raise DirectXRefusal(
                f"DirectX {shader_type} stage domain '{domain}' must be "
                f"one of: {valid_values}"
            )

    def validate_hlsl_tessellation_output_topology(self, func, shader_type):
        if shader_type != "tessellation_control":
            return

        topology = self.normalized_hlsl_stage_attribute_argument(func, "outputtopology")
        if not topology:
            return

        valid_topologies = {
            "point",
            "line",
            "triangle_cw",
            "triangle_ccw",
        }
        if topology not in valid_topologies:
            valid_values = ", ".join(sorted(valid_topologies))
            raise DirectXRefusal(
                "DirectX tessellation_control stage outputtopology "
                f"'{topology}' must be one of: {valid_values}"
            )

    def validate_hlsl_mesh_output_topology(self, func, shader_type):
        if shader_type != "mesh":
            return

        topology = self.normalized_hlsl_stage_attribute_argument(func, "outputtopology")
        if not topology:
            raise DirectXRefusal(
                "DirectX mesh stage requires HLSL attribute(s): outputtopology"
            )

        valid_topologies = {"line", "triangle"}
        if topology not in valid_topologies:
            valid_values = ", ".join(sorted(valid_topologies))
            raise DirectXRefusal(
                "DirectX mesh stage outputtopology "
                f"'{topology}' must be one of: {valid_values}"
            )

    def validate_hlsl_mesh_thread_group_size(self, shader_type, values):
        if shader_type not in {"mesh", "task", "amplification", "object"}:
            return
        if len(values) != 3 or any(value is None for value in values):
            return

        values = [int(value) for value in values]
        product = values[0] * values[1] * values[2]
        if product > 128:
            raise DirectXRefusal(
                f"DirectX {shader_type} stage numthreads product must not exceed 128"
            )

    def validate_hlsl_tessellation_partitioning(self, func, shader_type):
        if shader_type != "tessellation_control":
            return

        partitioning = self.normalized_hlsl_stage_attribute_argument(
            func, "partitioning"
        )
        if not partitioning:
            return

        valid_partitioning = {
            "integer",
            "fractional_even",
            "fractional_odd",
            "pow2",
        }
        if partitioning not in valid_partitioning:
            valid_values = ", ".join(sorted(valid_partitioning))
            raise DirectXRefusal(
                "DirectX tessellation_control stage partitioning "
                f"'{partitioning}' must be one of: {valid_values}"
            )

    HLSL_RAY_TRACING_SHADER_ATTRIBUTES = {
        "ray_generation": "raygeneration",
        "ray_intersection": "intersection",
        "ray_closest_hit": "closesthit",
        "ray_any_hit": "anyhit",
        "ray_miss": "miss",
        "ray_callable": "callable",
    }

    def validate_directx_ray_tracing_stage_profile(self, shader_type):
        shader_attribute = self.HLSL_RAY_TRACING_SHADER_ATTRIBUTES.get(shader_type)
        if shader_attribute is None or self.target_profile not in {"dx10", "dx11"}:
            return

        raise DirectXRayTracingStageProfileError(
            f"DirectX profile {self.target_profile} does not support the "
            f"'{shader_attribute}' ray tracing stage; DirectX Raytracing requires "
            "DirectX 12 / Shader Model 6.3",
            target_profile=self.target_profile,
            shader_type=shader_type,
            shader_attribute=shader_attribute,
        )

    HLSL_MESH_SHADING_SHADER_ATTRIBUTES = {
        "mesh": "mesh",
        "task": "amplification",
        "amplification": "amplification",
        "object": "amplification",
    }

    def validate_directx_mesh_stage_profile(self, shader_type):
        shader_attribute = self.HLSL_MESH_SHADING_SHADER_ATTRIBUTES.get(shader_type)
        if shader_attribute is None or self.target_profile not in {"dx10", "dx11"}:
            return

        raise DirectXMeshStageProfileError(
            f"DirectX profile {self.target_profile} does not support the "
            f"'{shader_attribute}' stage; mesh and amplification shaders require "
            "DirectX 12 / Shader Model 6.5",
            target_profile=self.target_profile,
            shader_type=shader_type,
            shader_attribute=shader_attribute,
        )

    def validate_hlsl_stage_requirements(self, func, shader_type, stage_node=None):
        self.validate_directx_ray_tracing_stage_profile(shader_type)
        self.validate_directx_mesh_stage_profile(shader_type)
        self.validate_hlsl_max_tess_factor_value(func, shader_type)
        self.validate_hlsl_mesh_output_topology(func, shader_type)

        required_attributes = {
            "geometry": {"maxvertexcount"},
            "tessellation_control": {
                "domain",
                "outputcontrolpoints",
                "outputtopology",
                "partitioning",
                "patchconstantfunc",
            },
            "tessellation_evaluation": {"domain"},
        }.get(shader_type)
        if not required_attributes:
            self.validate_hlsl_stage_parameter_requirements(func, shader_type)
            return

        if shader_type == "mesh":
            self.validate_hlsl_stage_parameter_requirements(func, shader_type)
            return

        present_attributes = self.hlsl_stage_attribute_names(func)
        if (
            shader_type == "geometry"
            and self.hlsl_geometry_maxvertexcount_value(
                func,
                stage_node=stage_node,
            )
            is not None
        ):
            present_attributes.add("maxvertexcount")
        missing_attributes = sorted(required_attributes - present_attributes)
        if missing_attributes:
            raise DirectXStageAttributesMissingError(shader_type, missing_attributes)
        refuse_glsl_style_tessellation_interface(self, func, shader_type, stage_node)
        self.validate_hlsl_tessellation_domain(func, shader_type)
        self.validate_hlsl_tessellation_output_topology(func, shader_type)
        self.validate_hlsl_tessellation_domain_topology(func, shader_type)
        self.validate_hlsl_tessellation_partitioning(func, shader_type)
        self.validate_hlsl_output_control_points_value(func, shader_type)
        self.validate_hlsl_max_vertex_count_value(
            func,
            shader_type,
            stage_node=stage_node,
        )
        self.validate_hlsl_domain_matches_hull(func, shader_type)
        self.validate_hlsl_stage_parameter_requirements(func, shader_type)
        self.validate_hlsl_patch_constant_function(func, shader_type)

    def hlsl_stage_attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "value"):
            return str(value.value).strip('"')
        if hasattr(value, "name"):
            return str(value.name)
        return str(value)

    def generate_hlsl_stage_attributes(self, func, shader_type, stage_node=None):
        if shader_type == "fragment":
            if requests_early_fragment_tests(func, stage_node):
                return "[earlydepthstencil]\n"
            return ""
        if shader_type not in {
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
        }:
            return ""

        quoted_argument_attributes = {
            "domain",
            "outputtopology",
            "partitioning",
            "patchconstantfunc",
        }
        code = ""
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.hlsl_stage_attribute_name(attr)
            if not attr_name:
                continue
            # Metal's max_total_threads_per_threadgroup only bounds the size that numthreads fixes in HLSL, and
            # max_total_threadgroups_per_mesh_grid only bounds the DispatchMesh grid.
            if attr_name in {
                "numthreads",
                "max_total_threads_per_threadgroup",
                "max_total_threadgroups_per_mesh_grid",
            }:
                continue

            arguments = getattr(attr, "arguments", []) or []
            if not arguments:
                continue

            argument_values = [
                self.hlsl_stage_attribute_value_to_string(argument)
                for argument in arguments
            ]
            if attr_name == "domain":
                argument_values = [
                    self.canonical_hlsl_tessellation_domain(value) or value
                    for value in argument_values
                ]
            if attr_name in quoted_argument_attributes:
                argument_values = [f'"{value}"' for value in argument_values]
            code += f"[{attr_name}({', '.join(argument_values)})]\n"

        if (
            shader_type == "geometry"
            and "maxvertexcount" not in self.hlsl_stage_attribute_names(func)
        ):
            maxvertexcount = self.hlsl_geometry_maxvertexcount_value(
                func,
                stage_node=stage_node,
            )
            if maxvertexcount is not None:
                code += f"[maxvertexcount({maxvertexcount})]\n"

        return code

    def stage_entry_names(self, ast, target_stage=None):
        stage_entry_types = self.stage_entry_types()
        entries = collect_stage_entry_records(ast, target_stage, stage_entry_types)
        used_names = collect_stage_entry_reserved_function_names(
            ast, target_stage, stage_entry_types
        )
        return assign_stage_entry_names(entries, used_names, self.stage_entry_base_name)

    def hlsl_stage_entry_functions(self, ast, expected_stage_name):
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

    def collect_hlsl_stage_output_lowerings(self, ast, target_stage, existing_structs):
        if not stage_matches(target_stage, "vertex"):
            return {}

        used_struct_names = {
            getattr(struct, "name", "")
            for struct in existing_structs
            if getattr(struct, "name", "")
        }
        lowerings = {}
        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name != "vertex" or not stage_matches(target_stage, stage_name):
                continue

            entry_point = getattr(stage, "entry_point", None)
            if entry_point is None:
                continue

            referenced_names = self.hlsl_referenced_identifier_names(entry_point)
            stage_inputs = []
            stage_outputs = []
            for variable in getattr(stage, "local_variables", []) or []:
                name = getattr(variable, "name", None)
                if not name or name not in referenced_names:
                    continue
                if self.hlsl_stage_local_variable_direction(variable) == "input":
                    stage_inputs.append(variable)
                elif self.hlsl_stage_local_variable_direction(variable) == "output":
                    stage_outputs.append(variable)

            if not stage_inputs and not stage_outputs:
                continue

            raw_return_type = self.type_name_string(
                getattr(entry_point, "return_type", getattr(entry_point, "vtype", None))
            )
            can_lower_outputs = (
                not raw_return_type or self.map_type(raw_return_type) == "void"
            )
            if not can_lower_outputs:
                stage_outputs = []

            input_struct = None
            input_struct_name = None
            input_local_name = None
            input_aliases = {}
            if stage_inputs:
                input_struct_name = self.hlsl_unique_stage_io_struct_name(
                    "VertexInput", "SPIRVVertexInput", used_struct_names
                )
                used_struct_names.add(input_struct_name)
                input_local_name = self.hlsl_unique_stage_io_local_name(
                    entry_point, "input", "spirvInput"
                )
                input_members, input_aliases = self.hlsl_stage_io_struct_members(
                    stage_inputs, input_local_name
                )
                input_struct = StructNode(input_struct_name, input_members)

            output_struct = None
            output_struct_name = None
            output_local_name = None
            output_aliases = {}
            if stage_outputs:
                output_struct_name = self.hlsl_unique_stage_io_struct_name(
                    "VertexOutput", "SPIRVVertexOutput", used_struct_names
                )
                used_struct_names.add(output_struct_name)
                output_local_name = self.hlsl_unique_stage_io_local_name(
                    entry_point, "output", "spirvOutput"
                )
                output_members, output_aliases = self.hlsl_stage_io_struct_members(
                    stage_outputs, output_local_name
                )
                output_struct = StructNode(output_struct_name, output_members)

            if input_struct is None and output_struct is None:
                continue

            lowerings[id(stage)] = {
                "stage_name": stage_name,
                "input_struct_name": input_struct_name,
                "input_local_name": input_local_name,
                "struct_name": output_struct_name,
                "local_name": output_local_name,
                "structs": [
                    struct
                    for struct in (input_struct, output_struct)
                    if struct is not None
                ],
                "aliases": {**input_aliases, **output_aliases},
            }

        return lowerings

    def hlsl_should_omit_lowered_spirv_pervertex_struct(self, struct_node):
        if getattr(struct_node, "name", None) != "gl_PerVertex":
            return False

        builtin_members = {
            "gl_Position",
            "gl_PointSize",
            "gl_ClipDistance",
            "gl_CullDistance",
        }
        member_names = {
            getattr(member, "name", None)
            for member in getattr(struct_node, "members", []) or []
            if getattr(member, "name", None)
        }
        return bool(member_names) and member_names <= builtin_members

    def hlsl_referenced_identifier_names(self, func):
        names = set()
        for node in self.walk_ast(getattr(func, "body", None)):
            if isinstance(node, IdentifierNode):
                name = getattr(node, "name", None)
            elif isinstance(node, VariableNode):
                name = getattr(node, "name", None)
            else:
                continue
            if name:
                names.add(name)
        return names

    def hlsl_stage_local_variable_direction(self, variable):
        for attr in getattr(variable, "attributes", []) or []:
            name = str(getattr(attr, "name", "")).lower()
            if name == "input":
                return "input"
            if name == "output":
                return "output"
        qualifiers = {
            str(qualifier).lower() for qualifier in getattr(variable, "qualifiers", [])
        }
        if "in" in qualifiers:
            return "input"
        if "out" in qualifiers:
            return "output"
        return None

    def hlsl_unique_stage_io_struct_name(self, preferred, fallback, used_names):
        if preferred not in used_names:
            return preferred
        if fallback not in used_names:
            return fallback
        index = 2
        while f"{fallback}{index}" in used_names:
            index += 1
        return f"{fallback}{index}"

    def hlsl_unique_stage_io_local_name(self, entry_point, preferred, fallback):
        used_names = self.collect_hlsl_function_identifier_names(entry_point)
        used_names.update(self.HLSL_RESERVED_LOCAL_IDENTIFIER_NAMES)
        if preferred not in used_names:
            return preferred
        if fallback not in used_names:
            return fallback
        index = 2
        while f"{fallback}{index}" in used_names:
            index += 1
        return f"{fallback}{index}"

    def hlsl_stage_io_struct_members(self, variables, local_name):
        members = []
        aliases = {}
        used_member_names = set()
        for variable in variables:
            field_name = self.hlsl_stage_io_field_name(variable, used_member_names)
            used_member_names.add(field_name)
            member = StructMemberNode(
                field_name,
                getattr(variable, "var_type", getattr(variable, "vtype", None)),
                attributes=self.hlsl_stage_io_member_attributes(variable),
            )
            members.append(member)
            aliases[variable.name] = f"{local_name}.{field_name}"
        return members, aliases

    def hlsl_stage_io_field_name(self, variable, used_names):
        semantic = self.hlsl_stage_io_variable_semantic(variable)
        builtin_names = {
            "gl_Position": "position",
            "gl_PointSize": "pointSize",
            "gl_ClipDistance": "clipDistance",
            "gl_CullDistance": "cullDistance",
        }
        base_name = builtin_names.get(semantic)
        if base_name is None:
            base_name = str(getattr(variable, "name", "value") or "value")
            base_name = base_name.lstrip("_") or "value"
        candidate = self.hlsl_sanitized_identifier(base_name)
        while (
            candidate in used_names
            or candidate in self.HLSL_RESERVED_LOCAL_IDENTIFIER_NAMES
        ):
            candidate += "_"
        return candidate

    def hlsl_sanitized_identifier(self, name):
        text = "".join(
            char if (char.isalnum() or char == "_") else "_"
            for char in str(name or "value")
        )
        text = text or "value"
        if text[0].isdigit():
            text = f"value_{text}"
        return text

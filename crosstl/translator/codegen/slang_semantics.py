"""Semantics in the Slang generator: default stage I/O semantics, resource attribute predicates, semantic mapping and semantic validation."""

import re

from ..refusals import SlangRefusal
from ..renames import renamed
from ..semantics.builtin_names import is_crossgl_gl_builtin
from .array_utils import split_array_type_suffix
from .hlsl_image_formats import slang_image_format_prefix
from .slang_conservative_depth import SLANG_DEPTH_SEMANTICS
from .slang_exact_lookups import slang_semantic_spelling_is_valid
from .slang_resource_memory import slang_resource_memory_prefix
from .slang_view_shading_rate import (
    SLANG_PRIMITIVE_SHADING_RATE_STAGES,
    validate_slang_view_shading_rate_inputs,
)


class SlangSemanticsMixin:
    """Semantic members of ``SlangCodeGen``."""

    def slang_parameter_direction_qualifiers(self, parameter):
        return {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
            if str(qualifier).lower() in {"const", "in", "out", "inout"}
        }

    def slang_parameter_user_struct_type(self, parameter):
        type_name = self.type_name_string(
            getattr(
                parameter,
                "param_type",
                getattr(parameter, "var_type", getattr(parameter, "vtype", None)),
            )
        )
        if not type_name:
            return None

        base_type, array_suffix = split_array_type_suffix(type_name)
        if array_suffix:
            return None

        mapped_type = self.convert_type(base_type)
        if mapped_type in self.user_struct_names:
            return mapped_type
        return None

    def slang_semantic_key(self, semantic):
        if semantic is None:
            return None
        return str(self.map_semantic(semantic)).lower()

    def slang_struct_member_declared_semantic(self, member):
        return self.semantic_from_node(member)

    def slang_struct_member_type_name(self, member):
        member_type = getattr(member, "member_type", getattr(member, "vtype", None))
        if member_type is None and hasattr(member, "element_type"):
            member_type = member.element_type
        return self.type_name_string(member_type)

    def slang_can_default_io_semantic(self, member_type):
        type_name = self.type_name_string(member_type)
        if not type_name:
            return False

        base_type, _array_suffix = split_array_type_suffix(str(type_name))
        mapped_type = self.map_resource_type_with_format(base_type)
        if mapped_type in self.user_struct_names:
            return False
        if mapped_type == "RaytracingAccelerationStructure":
            return False
        return not (
            self.is_storage_image_type(mapped_type)
            or self.is_sampled_texture_resource_type(mapped_type)
            or self.is_buffer_resource_type(mapped_type)
        )

    def slang_default_vertex_input_member_semantics(self, struct_node):
        if (
            getattr(struct_node, "name", None)
            not in self.vertex_entry_input_struct_names
        ):
            return {}

        used_semantics = set()
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.slang_struct_member_declared_semantic(member)
            if semantic is not None:
                used_semantics.add(self.slang_semantic_key(semantic))

        defaults = {}
        position_names = {"position", "vertexposition", "vertex_position", "pos"}
        if self.slang_semantic_key("POSITION") not in used_semantics:
            for member in getattr(struct_node, "members", []) or []:
                member_name = getattr(member, "name", None)
                if not member_name or member_name.lower() not in position_names:
                    continue
                if self.slang_struct_member_declared_semantic(member) is not None:
                    continue
                if not self.slang_can_default_io_semantic(
                    self.slang_struct_member_type_name(member)
                ):
                    continue
                defaults[member_name] = "POSITION"
                used_semantics.add(self.slang_semantic_key("POSITION"))
                break

        next_texcoord = 0
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name or member_name in defaults:
                continue
            if self.slang_struct_member_declared_semantic(member) is not None:
                continue
            if not self.slang_can_default_io_semantic(
                self.slang_struct_member_type_name(member)
            ):
                continue

            while self.slang_semantic_key(f"TEXCOORD{next_texcoord}") in used_semantics:
                next_texcoord += 1
            semantic = f"TEXCOORD{next_texcoord}"
            defaults[member_name] = semantic
            used_semantics.add(self.slang_semantic_key(semantic))
            next_texcoord += 1
        return defaults

    def slang_default_vertex_output_member_semantics(self, struct_node):
        if (
            getattr(struct_node, "name", None)
            not in self.vertex_entry_output_struct_names
        ):
            return {}

        used_semantics = set()
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.slang_struct_member_declared_semantic(member)
            if semantic is not None:
                used_semantics.add(self.slang_semantic_key(semantic))

        defaults = {}
        position_names = {"position", "clipposition", "clip_position"}
        if self.slang_semantic_key("SV_Position") not in used_semantics:
            for member in getattr(struct_node, "members", []) or []:
                member_name = getattr(member, "name", None)
                if not member_name or member_name.lower() not in position_names:
                    continue
                if self.slang_struct_member_declared_semantic(member) is not None:
                    continue
                if (
                    self.convert_type(self.slang_struct_member_type_name(member))
                    != "float4"
                ):
                    continue
                defaults[member_name] = "SV_Position"
                used_semantics.add(self.slang_semantic_key("SV_Position"))
                break

        next_texcoord = 0
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name or member_name in defaults:
                continue
            if self.slang_struct_member_declared_semantic(member) is not None:
                continue
            if not self.slang_can_default_io_semantic(
                self.slang_struct_member_type_name(member)
            ):
                continue

            while self.slang_semantic_key(f"TEXCOORD{next_texcoord}") in used_semantics:
                next_texcoord += 1
            semantic = f"TEXCOORD{next_texcoord}"
            defaults[member_name] = semantic
            used_semantics.add(self.slang_semantic_key(semantic))
            next_texcoord += 1
        return defaults

    def slang_default_fragment_input_member_semantics(self, struct_node, defaults):
        if (
            getattr(struct_node, "name", None)
            not in self.fragment_entry_input_struct_names
        ):
            return defaults

        used_semantics = {
            self.slang_semantic_key(semantic)
            for semantic in defaults.values()
            if semantic is not None
        }
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.slang_struct_member_declared_semantic(member)
            if semantic is not None:
                used_semantics.add(self.slang_semantic_key(semantic))

        next_texcoord = 0
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name or member_name in defaults:
                continue
            if self.slang_struct_member_declared_semantic(member) is not None:
                continue
            if not self.slang_can_default_io_semantic(
                self.slang_struct_member_type_name(member)
            ):
                continue

            while self.slang_semantic_key(f"TEXCOORD{next_texcoord}") in used_semantics:
                next_texcoord += 1
            semantic = f"TEXCOORD{next_texcoord}"
            defaults[member_name] = semantic
            used_semantics.add(self.slang_semantic_key(semantic))
            next_texcoord += 1
        return defaults

    def slang_default_fragment_output_member_semantics(self, struct_node, defaults):
        if (
            getattr(struct_node, "name", None)
            not in self.fragment_entry_output_struct_names
        ):
            return defaults

        used_semantics = {
            self.slang_semantic_key(semantic)
            for semantic in defaults.values()
            if semantic is not None
        }
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.slang_struct_member_declared_semantic(member)
            if semantic is not None:
                used_semantics.add(self.slang_semantic_key(semantic))

        next_target = 0
        for member in getattr(struct_node, "members", []) or []:
            member_name = getattr(member, "name", None)
            if not member_name or member_name in defaults:
                continue
            if self.slang_struct_member_declared_semantic(member) is not None:
                continue

            mapped_type = self.convert_type(self.slang_struct_member_type_name(member))
            normalized_name = str(member_name).replace("_", "").lower()
            if normalized_name in {"depth", "fragdepth", "glfragdepth"}:
                if (
                    mapped_type == "float"
                    and self.slang_semantic_key("SV_Depth") not in used_semantics
                ):
                    defaults[member_name] = "SV_Depth"
                    used_semantics.add(self.slang_semantic_key("SV_Depth"))
                continue

            if mapped_type != "float4":
                continue
            while self.slang_semantic_key(
                f"SV_Target{next_target}"
            ) in used_semantics or (
                next_target == 0
                and self.slang_semantic_key("SV_Target") in used_semantics
            ):
                next_target += 1
            semantic = "SV_Target" if next_target == 0 else f"SV_Target{next_target}"
            defaults[member_name] = semantic
            used_semantics.add(self.slang_semantic_key(semantic))
            next_target += 1
        return defaults

    def slang_default_struct_member_semantics(self, struct_node):
        defaults = self.slang_default_vertex_input_member_semantics(struct_node)
        defaults.update(self.slang_default_vertex_output_member_semantics(struct_node))
        defaults = self.slang_default_fragment_input_member_semantics(
            struct_node, defaults
        )
        return self.slang_default_fragment_output_member_semantics(
            struct_node, defaults
        )

    def default_slang_stage_return_semantic(self, shader_type, return_type):
        if shader_type == "vertex" and return_type == "float4":
            return "gl_Position"
        if shader_type == "fragment" and return_type == "float4":
            return "gl_FragColor"
        return None

    def slang_shader_stage_marker_name(self, name):
        if name is None:
            return None

        normalized = str(name).lower()
        if normalized.startswith("slang_"):
            normalized = normalized[len("slang_") :]
        elif normalized.startswith("hlsl_"):
            normalized = normalized[len("hlsl_") :]

        valid_names = {
            "amplification",
            "anyhit",
            "callable",
            "closesthit",
            "compute",
            "domain",
            "fragment",
            "geometry",
            "hull",
            "intersection",
            "local_size",
            "local_size_x",
            "local_size_y",
            "local_size_z",
            "mesh",
            "miss",
            "numthreads",
            "object",
            "pixel",
            "ray_any_hit",
            "ray_callable",
            "ray_closest_hit",
            "ray_generation",
            "ray_intersection",
            "ray_miss",
            "raygen",
            "raygeneration",
            "stage_entry",
            "task",
            "tesscontrol",
            "tesseval",
            "tessellation_control",
            "tessellation_evaluation",
            "vertex",
            "workgroup_size",
        }
        if normalized in valid_names:
            return normalized
        return None

    def is_resource_format_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        attr_name = str(attr_name).lower()
        return attr_name == "format" or attr_name in self.supported_image_formats()

    def is_resource_binding_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "binding",
            "buffer",
            "group",
            "register",
            "sampler",
            "set",
            "space",
            "texture",
        }

    def is_resource_memory_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "coherent",
            "globallycoherent",
            "readonly",
            "readwrite",
            "restrict",
            "volatile",
            "writeonly",
        }

    def slang_storage_image_format_prefix(self, mapped_type, node):
        if node is None:
            return ""
        base_type = self.resource_base_type(mapped_type)
        if not isinstance(base_type, str) or not base_type.startswith(
            ("RWTexture", "RasterizerOrderedTexture")
        ):
            return ""
        return slang_image_format_prefix(self.explicit_image_format(node))

    def slang_resource_memory_qualifier_prefix(self, mapped_type, node):
        return slang_resource_memory_prefix(self, mapped_type, node)

    def stage_semantic_map(self, shader_type):
        shader_stage = self.slang_shader_stage_name(shader_type)
        if shader_stage == "geometry":
            return {
                "gl_PrimitiveIDIn": "SV_PrimitiveID",
                "gl_InvocationID": "SV_GSInstanceID",
            }
        if shader_stage == "hull":
            return {
                "gl_InvocationID": "SV_OutputControlPointID",
                "gl_PrimitiveID": "SV_PrimitiveID",
            }
        if shader_stage == "domain":
            return {
                "gl_TessCoord": "SV_DomainLocation",
                "gl_PrimitiveID": "SV_PrimitiveID",
            }
        return {}

    def map_semantic(self, semantic, shader_type=None):
        if semantic is None:
            return None
        semantic_name = str(semantic)
        stage_semantic = self.stage_semantic_map(shader_type).get(semantic_name)
        if stage_semantic:
            return stage_semantic
        if semantic_name.startswith("gl_FragColor"):
            target_index = semantic_name[len("gl_FragColor") :]
            if target_index.isdigit():
                return f"SV_Target{target_index}"
        frag_data = re.fullmatch(r"gl_FragData\[(\d+)\]", semantic_name)
        if frag_data:
            return f"SV_Target{frag_data.group(1)}"
        # ``semantic_map`` renames CrossGL built-ins to Slang system values; a user semantic keeps its spelling, and
        # ``semantic_suffix`` refuses an unrenamed built-in where a declaration would carry it.
        return renamed(self.semantic_map, semantic_name)

    def slang_semantic_output_kind(self, semantic):
        if semantic is None:
            return None

        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        input_only_sources = {
            "gl_baseinstance",
            "gl_basevertex",
            "gl_drawid",
            "gl_fragcoord",
            "gl_frontfacing",
            "gl_globalinvocationid",
            "gl_instanceid",
            "gl_invocationid",
            "gl_localinvocationid",
            "gl_localinvocationindex",
            "gl_pointcoord",
            "gl_barycoordext",
            "gl_barycoordnoperspext",
            "gl_sampleid",
            "gl_samplemaskin",
            "gl_shadingrateext",
            "gl_tesscoord",
            "gl_vertexid",
            "gl_viewindex",
            "gl_workgroupid",
        }
        if lower_name in input_only_sources:
            return "input_only"

        mapped_semantic = str(self.map_semantic(semantic))
        mapped_upper = mapped_semantic.upper()
        if lower_name == "gl_position" or mapped_upper == "SV_POSITION":
            return "position"
        if lower_name == "gl_fragdepth" or mapped_upper in SLANG_DEPTH_SEMANTICS:
            return "depth"
        if lower_name == "gl_fragstencilrefext" or mapped_upper == "SV_STENCILREF":
            return "stencil_ref"
        if lower_name == "gl_samplemask" or mapped_upper == "SV_COVERAGE":
            return "coverage"
        if lower_name == "gl_primitiveshadingrateext":
            return "primitive_shading_rate"
        if lower_name == "gl_tesslevelouter" or mapped_upper == "SV_TESSFACTOR":
            return "tess_factor"
        if lower_name == "gl_tesslevelinner" or mapped_upper == "SV_INSIDETESSFACTOR":
            return "inside_tess_factor"
        if lower_name.startswith("gl_fragcolor"):
            suffix = lower_name[len("gl_fragcolor") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        if mapped_upper.startswith("SV_TARGET"):
            suffix = mapped_upper[len("SV_TARGET") :]
            if suffix == "" or suffix.isdigit():
                return "color"

        if mapped_upper in {
            "SV_DISPATCHTHREADID",
            "SV_DOMAINLOCATION",
            "SV_GROUPID",
            "SV_GROUPINDEX",
            "SV_GROUPTHREADID",
            "SV_GSINSTANCEID",
            "SV_INSTANCEID",
            "SV_ISFRONTFACE",
            "SV_OUTPUTCONTROLPOINTID",
            "SV_POINTCOORD",
            "SV_SAMPLEINDEX",
            "SV_STARTINSTANCELOCATION",
            "SV_STARTVERTEXLOCATION",
            "SV_VERTEXID",
        }:
            return "input_only"
        return None

    def is_slang_float_scalar_type(self, type_name):
        mapped_type = self.convert_type(type_name)
        if mapped_type is None:
            return False
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        return not array_suffix and base_type == "float"

    def is_slang_int_scalar_type(self, type_name):
        mapped_type = self.convert_type(type_name)
        if mapped_type is None:
            return False
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        return not array_suffix and base_type == "int"

    def is_slang_uint_scalar_type(self, type_name):
        mapped_type = self.convert_type(type_name)
        if mapped_type is None:
            return False
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        return not array_suffix and base_type == "uint"

    def is_slang_render_target_type(self, type_name):
        mapped_type = self.convert_type(type_name)
        if mapped_type is None:
            return False
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        return not array_suffix and bool(
            re.fullmatch(r"(?:float|half|int|uint)[1-4]?", base_type)
        )

    def is_slang_float_vector_width(self, type_name, width):
        mapped_type = self.convert_type(type_name)
        if mapped_type is None:
            return False
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        return not array_suffix and base_type == f"float{width}"

    def validate_slang_builtin_semantic_type(self, semantic, type_name, context):
        kind = self.slang_semantic_output_kind(semantic)
        if kind is None or kind == "input_only":
            return
        if kind in {"tess_factor", "inside_tess_factor"}:
            return

        if kind == "color" and str(semantic).lower() != "gl_fragcolor":
            # A render target SV_Target<N> takes a float, half, int or uint scalar or vector, as in HLSL; GLSL
            # declares an integer colour output for an integer attachment.
            if self.is_slang_render_target_type(type_name):
                return
            raise SlangRefusal(
                f"Unsupported {semantic} {context} for Slang codegen; "
                "expected a scalar or vector of float, half, int or uint"
            )
        if kind in {"position", "color"}:
            if self.is_slang_float_vector_width(type_name, 4):
                return
            raise SlangRefusal(
                f"Unsupported {semantic} {context} for Slang codegen; "
                "expected vec4-compatible type"
            )

        if kind == "depth" and not self.is_slang_float_scalar_type(type_name):
            raise SlangRefusal(
                f"Unsupported {semantic} {context} for Slang codegen; "
                "expected float type"
            )

        if kind == "primitive_shading_rate":
            if self.is_slang_int_scalar_type(
                type_name
            ) or self.is_slang_uint_scalar_type(type_name):
                return
            raise SlangRefusal(
                f"Unsupported {semantic} {context} for Slang codegen; "
                "expected int or uint type"
            )
        if kind == "stencil_ref" and self.is_slang_int_scalar_type(type_name):
            # GLSL declares gl_FragStencilRefARB an int; slangc writes an int SV_StencilRef as the SPIR-V
            # FragStencilRefEXT output, which takes a 32-bit integer of either signedness.
            return
        if kind in {"coverage", "stencil_ref"} and not self.is_slang_uint_scalar_type(
            type_name
        ):
            raise SlangRefusal(
                f"Unsupported {semantic} {context} for Slang codegen; "
                "expected uint type"
            )

    def validate_slang_output_semantic_stage(
        self, shader_type, semantic, context, stage_role=None
    ):
        kind = self.slang_semantic_output_kind(semantic)
        if kind is None:
            return
        if kind == "input_only":
            raise SlangRefusal(
                f"Unsupported {semantic} {context} for Slang codegen; "
                "input-only builtin semantics cannot be used as outputs"
            )
        if shader_type is None:
            return

        shader_stage = self.slang_shader_stage_name(shader_type)
        if kind in {"tess_factor", "inside_tess_factor"}:
            if shader_stage == "hull" and stage_role == "patch_constant":
                return
            raise SlangRefusal(
                f"Unsupported {semantic} {context} for Slang {shader_type} stage; "
                "valid stage is tessellation_control patch constant"
            )

        allowed_stages = {
            "position": {"domain", "geometry", "hull", "mesh", "vertex"},
            "color": {"fragment"},
            "coverage": {"fragment"},
            "depth": {"fragment"},
            "stencil_ref": {"fragment"},
            "primitive_shading_rate": SLANG_PRIMITIVE_SHADING_RATE_STAGES,
        }[kind]
        if shader_stage not in allowed_stages:
            allowed = ", ".join(sorted(allowed_stages))
            raise SlangRefusal(
                f"Unsupported {semantic} {context} for Slang {shader_type} stage; "
                f"valid stage is {allowed}"
            )

    def slang_stage_input_semantic_rules(self):
        return {
            "vertex": {
                "SV_VERTEXID": "uint",
                "SV_INSTANCEID": "uint",
                "SV_STARTVERTEXLOCATION": "int",
                "SV_STARTINSTANCELOCATION": "uint",
                "SV_DRAWID": "uint",
            },
            "fragment": {
                "SV_POSITION": "float4",
                "SV_POINTCOORD": "float2",
                "SV_ISFRONTFACE": "bool",
                "SV_PRIMITIVEID": "uint",
                "SV_COVERAGE": "uint",
                "SV_SAMPLEINDEX": "uint",
                "SV_BARYCENTRICS": "float3",
                "SV_RENDERTARGETARRAYINDEX": "uint",
                "SV_VIEWPORTARRAYINDEX": "uint",
            },
            "compute": {
                "SV_GROUPID": "uint3",
                "SV_GROUPTHREADID": "uint3",
                "SV_DISPATCHTHREADID": "uint3",
                "SV_GROUPINDEX": "uint",
            },
            "mesh": {
                "SV_GROUPID": "uint3",
                "SV_GROUPTHREADID": "uint3",
                "SV_DISPATCHTHREADID": "uint3",
                "SV_GROUPINDEX": "uint",
            },
            "amplification": {
                "SV_GROUPID": "uint3",
                "SV_GROUPTHREADID": "uint3",
                "SV_DISPATCHTHREADID": "uint3",
                "SV_GROUPINDEX": "uint",
            },
            "geometry": {
                "SV_PRIMITIVEID": "uint",
                "SV_GSINSTANCEID": "uint",
            },
            "hull": {
                "SV_OUTPUTCONTROLPOINTID": "uint",
                "SV_PRIMITIVEID": "uint",
            },
            "domain": {
                "SV_DOMAINLOCATION": "float3",
                "SV_PRIMITIVEID": "uint",
            },
        }

    def slang_stage_input_semantic_stages(self, mapped_semantic: str) -> set:
        mapped_upper = str(mapped_semantic).upper()
        return {
            stage
            for stage, rules in self.slang_stage_input_semantic_rules().items()
            if mapped_upper in rules
        }

    def slang_parameter_semantic_type_matches(self, type_name, expected_type) -> bool:
        mapped_type = self.convert_type(type_name)
        if mapped_type is None:
            return False
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        return not array_suffix and base_type == expected_type

    def validate_slang_stage_parameter_semantic_type(
        self, parameter, semantic, expected_type
    ):
        type_name = self.slang_parameter_type_name(parameter)
        if self.slang_parameter_semantic_type_matches(type_name, expected_type):
            return
        mapped_semantic = self.map_semantic(semantic, "compute")
        mapped_type = self.convert_type(type_name)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if (
            expected_type == "uint3"
            and mapped_semantic in {
                "SV_DispatchThreadID",
                "SV_GroupThreadID",
                "SV_GroupID",
            }
            and not array_suffix
            and base_type in {"uint", "uint2"}
        ):
            return

        raise SlangRefusal(
            f"Unsupported {semantic} stage parameter semantic for Slang codegen; "
            f"expected {expected_type} type"
        )

    def slang_domain_location_type(self, function):
        """SV_DomainLocation's type for a domain entry: float3 on a tri domain, float2 on quad and isoline."""
        if function is None:
            return "float3"
        domain = self.normalized_slang_stage_attribute_argument(function, "domain")
        domain = self.canonical_slang_tessellation_domain(domain) if domain else None
        return "float2" if domain in {"quad", "isoline"} else "float3"

    def validate_slang_stage_parameter_semantics(
        self, shader_type, parameters, stage_role=None, function=None
    ):
        if shader_type is None:
            return

        shader_stage = self.slang_shader_stage_name(shader_type)
        rules = self.slang_stage_input_semantic_rules().get(shader_stage, {})
        seen_system_semantics = {}
        validate_slang_view_shading_rate_inputs(self, shader_type, parameters)

        for parameter in parameters or []:
            semantic = self.semantic_from_node(parameter)
            if semantic is None:
                continue
            # An ``out`` parameter carries a stage output, as in HLSL.
            if "out" in {
                str(qualifier).lower()
                for qualifier in getattr(parameter, "qualifiers", None) or []
            }:
                continue
            if (
                self.slang_ray_semantic_role(parameter, shader_type)
                or self.is_slang_mesh_payload_parameter(parameter)
                or self.slang_mesh_output_role_from_parameter(parameter)
            ):
                continue

            if str(semantic).lower() == "gl_samplemask":
                raise SlangRefusal(
                    f"Unsupported {semantic} stage parameter semantic for Slang "
                    f"{shader_type} stage; output-only builtin semantics cannot "
                    "be used as inputs"
                )

            mapped_semantic = self.map_semantic(semantic, shader_type)
            mapped_upper = str(mapped_semantic).upper()
            expected_type = rules.get(mapped_upper)
            if expected_type is not None and mapped_upper == "SV_DOMAINLOCATION":
                expected_type = self.slang_domain_location_type(function)
            if expected_type is not None:
                previous_name = seen_system_semantics.get(mapped_upper)
                if previous_name is not None:
                    raise SlangRefusal(
                        f"Duplicate Slang stage parameter semantic "
                        f"{mapped_semantic} on '{previous_name}' and "
                        f"'{parameter.name}'"
                    )
                seen_system_semantics[mapped_upper] = parameter.name
                self.validate_slang_stage_parameter_semantic_type(
                    parameter, semantic, expected_type
                )
                continue

            valid_stages = self.slang_stage_input_semantic_stages(mapped_semantic)
            if valid_stages:
                valid = ", ".join(sorted(valid_stages))
                raise SlangRefusal(
                    f"Unsupported {semantic} stage parameter semantic for Slang "
                    f"{shader_type} stage; valid stage is {valid}"
                )

            kind = self.slang_semantic_output_kind(semantic)
            if kind in {
                "color",
                "coverage",
                "depth",
                "inside_tess_factor",
                "tess_factor",
            }:
                raise SlangRefusal(
                    f"Unsupported {semantic} stage parameter semantic for Slang "
                    f"{shader_type} stage; output-only builtin semantics cannot "
                    "be used as inputs"
                )

    def semantic_suffix(self, semantic, shader_type=None):
        """The ``: semantic`` a declaration takes; a built-in with no Slang system value is refused.

        ``map_semantic`` returns such a ``gl_`` name unchanged, and slangc would read it as a user semantic, an
        ordinary varying, so the declaration would not carry the built-in.
        """
        mapped_semantic = self.map_semantic(semantic, shader_type)
        name = str(mapped_semantic or "")
        if name and not slang_semantic_spelling_is_valid(name):
            self.refuse_lowering(
                f"'{name}' is not a Slang semantic; a semantic is an identifier "
                "with an optional numeric index"
            )
        elif (
            name.startswith("gl_")
            and is_crossgl_gl_builtin(name)
            and not name.lower().startswith(("gl_fragcolor", "gl_fragdata"))
        ):
            self.refuse_lowering(
                f"Slang has no system value for the built-in @{name}; "
                "it would become a user varying, not the built-in"
            )
        return f" : {mapped_semantic}" if mapped_semantic else ""

"""Stage input and output semantics in the Mojo generator: semantic kinds, expected types and their validation."""

import re

from ..refusals import MojoRefusal
from .mojo_tables import (
    MOJO_INPUT_ONLY_RETURN_SEMANTICS,
    MOJO_INTEGER_DTYPES,
    MOJO_NON_SEMANTIC_ATTRIBUTES,
    MOJO_RAY_STAGE_TYPES,
    MOJO_SCALAR_DTYPES,
)


class MojoSemanticMixin:
    """Semantic members of ``MojoCodeGen``."""

    def extract_semantic_from_attributes(self, attributes):
        """Extract shader semantic information from AST attributes."""
        for attr in attributes:
            name = getattr(attr, "name", None)
            if self.is_semantic_attribute_name(name):
                return str(name)
        return None

    def node_semantic(self, node):
        semantic = getattr(node, "semantic", None)
        if semantic:
            return semantic
        return self.extract_semantic_from_attributes(
            getattr(node, "attributes", []) or []
        )

    def function_return_semantic(self, func):
        return self.extract_semantic_from_attributes(
            getattr(func, "attributes", []) or []
        )

    def is_semantic_attribute_name(self, name):
        if name is None:
            return False

        semantic = str(name)
        lower = semantic.lower()
        upper = semantic.upper()
        if lower in MOJO_NON_SEMANTIC_ATTRIBUTES:
            return False
        if semantic in self.semantic_map:
            return True
        if upper in self.semantic_map:
            return True
        if re.fullmatch(r"(COLOR|TEXCOORD|NORMAL|TANGENT|BINORMAL)\d*", upper):
            return True
        if re.fullmatch(r"SV_TARGET\d*", upper):
            return True
        if upper in {
            "SV_DEPTH",
            "SV_DISPATCHTHREADID",
            "SV_GROUPID",
            "SV_GROUPINDEX",
            "SV_GROUPTHREADID",
            "SV_INSTANCEID",
            "SV_ISFRONTFACE",
            "SV_POSITION",
            "SV_VERTEXID",
        }:
            return True
        return lower.startswith("gl_")

    def semantic_output_kind(self, semantic):
        lower = str(semantic).lower()
        upper = str(semantic).upper()
        if lower == "gl_position" or upper == "SV_POSITION":
            return "position"
        if re.fullmatch(r"gl_fragcolor\d*", lower) or re.fullmatch(
            r"SV_TARGET\d*", upper
        ):
            return "color"
        if lower == "gl_fragdepth" or upper == "SV_DEPTH":
            return "depth"
        if lower in MOJO_INPUT_ONLY_RETURN_SEMANTICS:
            return "input_only"
        return None

    def validate_builtin_semantic_type(self, semantic, type_name, context):
        kind = self.semantic_output_kind(semantic)
        if kind is None or kind == "input_only":
            return

        if kind in {"position", "color"}:
            if self.is_float_vector_width(type_name, 4):
                return
            raise MojoRefusal(
                f"Unsupported {semantic} {context} for Mojo codegen; "
                "expected vec4-compatible return type"
            )

        if kind == "depth" and not self.is_float_scalar_type(type_name):
            raise MojoRefusal(
                f"Unsupported {semantic} {context} for Mojo codegen; "
                "expected float return type"
            )

    def validate_return_semantic(self, shader_type, return_type, semantic):
        if self.type_name(return_type) == "void":
            raise MojoRefusal(
                f"Unsupported {shader_type or 'function'} {semantic} "
                "return semantic for Mojo codegen; void return type"
            )

        kind = self.semantic_output_kind(semantic)
        if kind == "input_only":
            raise MojoRefusal(
                f"Unsupported {semantic} return semantic for Mojo codegen; "
                "input-only builtin semantics cannot be used as return semantics"
            )

        self.validate_output_semantic_stage(shader_type, semantic, "return semantic")
        self.validate_builtin_semantic_type(semantic, return_type, "return semantic")

    def validate_struct_return_semantics(self, shader_type, return_type):
        if shader_type is None:
            return

        for member_name, semantic in self.struct_member_semantics.get(
            self.type_name(return_type), {}
        ).items():
            self.validate_output_semantic_stage(
                shader_type,
                semantic,
                f"struct return semantic '{return_type}.{member_name}'",
            )

    def validate_output_semantic_stage(self, shader_type, semantic, context):
        kind = self.semantic_output_kind(semantic)
        if kind is None or kind == "input_only" or shader_type is None:
            return

        allowed_stages = {
            "position": {"vertex"},
            "color": {"fragment"},
            "depth": {"fragment"},
        }[kind]
        if shader_type not in allowed_stages:
            allowed = ", ".join(sorted(allowed_stages))
            raise MojoRefusal(
                f"Unsupported {semantic} {context} for Mojo {shader_type} stage; "
                f"valid stage is {allowed}"
            )

    def is_float_vector_width(self, type_name, width):
        vector_info = self.vector_type_info(type_name)
        return (
            vector_info is not None
            and vector_info[0] == "DType.float32"
            and vector_info[1] == width
        )

    def is_float_scalar_type(self, type_name):
        return MOJO_SCALAR_DTYPES.get(self.type_name(type_name)) == "DType.float32"

    def generate_return_semantic_comment(self, shader_type, semantic):
        mapped_semantic = self.map_semantic(semantic)
        return (
            "# CrossGL return semantic: "
            f"stage={shader_type or 'function'} "
            f"semantic={mapped_semantic} source={semantic}\n"
        )

    def generate_parameter_semantic_comment(self, shader_type, param_name, semantic):
        mapped_semantic = self.map_semantic(semantic)
        return (
            "# CrossGL parameter semantic: "
            f"stage={shader_type or 'function'} "
            f"name={param_name} semantic={mapped_semantic} source={semantic}\n"
        )

    def semantic_parameter_kind(self, semantic):
        lower = str(semantic).lower()
        upper = str(semantic).upper()

        if (
            lower == "gl_position"
            or lower == "gl_fragdepth"
            or re.fullmatch(r"gl_fragcolor\d*", lower)
            or upper == "SV_DEPTH"
            or re.fullmatch(r"SV_TARGET\d*", upper)
        ):
            return "output_only"

        if lower == "gl_fragcoord" or upper == "SV_POSITION":
            return ("fragment", "float_vec4")
        if lower == "gl_frontfacing" or upper == "SV_ISFRONTFACE":
            return ("fragment", "bool_scalar")
        if lower == "gl_pointcoord":
            return ("fragment", "float_vec2")

        if lower in {"gl_vertexid"} or upper == "SV_VERTEXID":
            return ("vertex", "integer_scalar")
        if lower in {"gl_instanceid"} or upper == "SV_INSTANCEID":
            return ("vertex", "integer_scalar")

        if lower in {"gl_globalinvocationid"} or upper == "SV_DISPATCHTHREADID":
            return ("compute", "integer_vec3")
        if lower in {"gl_localinvocationid"} or upper == "SV_GROUPTHREADID":
            return ("compute", "integer_vec3")
        if lower in {"gl_workgroupid"} or upper == "SV_GROUPID":
            return ("compute", "integer_vec3")
        if upper == "SV_GROUPINDEX":
            return ("compute", "integer_scalar")

        if lower == "gl_launchidext" or upper == "SV_DISPATCHRAYSINDEX":
            return ("ray", "integer_vec3")
        if lower == "gl_launchsizeext" or upper == "SV_DISPATCHRAYSDIMENSIONS":
            return ("ray", "integer_vec3")
        if lower == "gl_hittext":
            return ("ray_hit", "float_scalar")
        if lower == "gl_hitkindext":
            return ("ray_any_hit", "integer_scalar")

        if re.fullmatch(r"(COLOR|TEXCOORD|NORMAL|TANGENT|BINORMAL)\d*", upper):
            return ("vertex_fragment", None)
        if upper == "POSITION":
            return ("vertex_fragment", None)
        return None

    def validate_parameter_semantic(
        self, shader_type, param_name, param_type, semantic, used_semantics
    ):
        if semantic is None:
            return

        semantic_key = self.normalized_semantic_key(semantic)
        previous_param = used_semantics.get(semantic_key)
        if previous_param is not None:
            raise MojoRefusal(
                f"Conflicting Mojo {shader_type or 'function'} parameter semantic "
                f"for '{param_name}': {self.map_semantic(semantic)} overlaps "
                f"'{previous_param}'"
            )
        used_semantics[semantic_key] = param_name

        kind = self.semantic_parameter_kind(semantic)
        if kind == "output_only" and shader_type == "fragment":
            semantic_name = str(semantic).lower()
            if semantic_name == "gl_position":
                kind = ("fragment", "float_vec4")
        if kind is None:
            return
        if kind == "output_only":
            raise MojoRefusal(
                f"Unsupported {semantic} parameter semantic for Mojo codegen; "
                "output-only builtin semantics cannot be used as parameters"
            )

        stage_kind, expected_type = kind
        if stage_kind == "vertex_fragment":
            allowed_stages = {"vertex", "fragment"}
        elif stage_kind == "ray":
            allowed_stages = MOJO_RAY_STAGE_TYPES
        elif stage_kind == "ray_hit":
            allowed_stages = {"ray_any_hit", "ray_closest_hit", "ray_intersection"}
        else:
            allowed_stages = {stage_kind}
        if shader_type is not None and shader_type not in allowed_stages:
            allowed = ", ".join(sorted(allowed_stages))
            raise MojoRefusal(
                f"Unsupported {semantic} parameter semantic for Mojo {shader_type} "
                f"stage; valid stage is {allowed}"
            )

        if not self.parameter_semantic_type_matches(expected_type, param_type):
            normalized_semantic = self.normalized_semantic_key(semantic)
            if (
                expected_type == "integer_vec3"
                and normalized_semantic == "global_invocation_id"
                and (
                    self.is_scalar_integer_type(param_type)
                    or self.is_integer_vector_width(param_type, 2)
                )
            ):
                return
            expected = self.parameter_semantic_expected_type(expected_type)
            raise MojoRefusal(
                f"Unsupported {semantic} parameter semantic for Mojo codegen; "
                f"expected {expected}"
            )

    def normalized_semantic_key(self, semantic):
        return self.map_semantic(semantic).lower()

    def parameter_semantic_type_matches(self, expected_type, param_type):
        if expected_type is None:
            return True
        if expected_type == "bool_scalar":
            return self.is_bool_scalar_type(param_type)
        if expected_type == "float_vec2":
            return self.is_float_vector_width(param_type, 2)
        if expected_type == "float_vec4":
            return self.is_float_vector_width(param_type, 4)
        if expected_type == "float_scalar":
            return self.is_float_scalar_type(param_type)
        if expected_type == "integer_scalar":
            return self.is_scalar_integer_type(param_type)
        if expected_type == "integer_vec3":
            return self.is_integer_vector_width(param_type, 3)
        return True

    def parameter_semantic_expected_type(self, expected_type):
        return {
            "bool_scalar": "bool",
            "float_vec2": "vec2-compatible type",
            "float_vec4": "vec4-compatible type",
            "float_scalar": "float scalar type",
            "integer_scalar": "integer scalar type",
            "integer_vec3": "integer vec3-compatible type",
        }.get(expected_type, "compatible type")

    def is_bool_scalar_type(self, type_name):
        return MOJO_SCALAR_DTYPES.get(self.type_name(type_name)) == "DType.bool"

    def is_integer_vector_width(self, type_name, width):
        vector_info = self.vector_type_info(type_name)
        return (
            vector_info is not None
            and vector_info[0] in MOJO_INTEGER_DTYPES
            and vector_info[1] == width
        )

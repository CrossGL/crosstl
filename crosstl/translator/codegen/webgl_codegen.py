"""CrossGL-to-WebGL GLSL ES code generator."""

from copy import copy

from ..ast import FunctionCallNode, StageMap
from .GLSL_codegen import GLSLCodeGen
from .stage_utils import STAGE_QUALIFIER_NAMES, normalize_stage_name


class WebGLCodeGen(GLSLCodeGen):
    """Generate WebGL 2.0 compatible GLSL ES output from CrossGL ASTs."""

    UNSUPPORTED_STAGE_NAMES = (
        {
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
        }
        | GLSLCodeGen.MESH_STAGE_NAMES
        | GLSLCodeGen.RAY_STAGE_NAMES
    )
    DEFAULT_PRECISION_LINES = (
        ("float", "precision highp float;"),
        ("int", "precision highp int;"),
    )
    SUPPORTED_STAGE_NAMES = {"fragment", "vertex"}

    def default_glsl_version_line(self, ast, target_stage=None):
        return "#version 300 es"

    def map_image_base_type_with_format(self, vtype, node=None):
        mapped_type = super().map_image_base_type_with_format(vtype, node)
        return self.sampled_image_type(mapped_type)

    def glsl_resource_binding_layouts_supported(self, version_line):
        return False

    def glsl_dynamic_resource_call_dispatch_info(self, expr):
        dispatch = super().glsl_dynamic_resource_call_dispatch_info(expr)
        if dispatch is not None:
            return dispatch
        if not isinstance(expr, FunctionCallNode):
            return None

        func_name = self.function_call_name(expr)
        if not func_name:
            return None
        args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        dynamic_info = self.webgl_dynamic_sampler_array_call_info(func_name, args)
        if dynamic_info is None:
            return None

        cases = []
        for index in range(dynamic_info["array_size"]):
            static_args = list(args)
            static_args[dynamic_info["arg_index"]] = (
                self.glsl_static_array_access_argument(dynamic_info, index)
            )
            rendered_args = ", ".join(
                self.generate_function_call_arguments(func_name, static_args)
            )
            cases.append((index, f"{func_name}({rendered_args})"))

        return {
            "index_expr": dynamic_info["index_expr"],
            "cases": cases,
            "return_type": self.expression_result_type(expr),
        }

    def webgl_dynamic_sampler_array_call_info(self, func_name, args):
        callee = self.function_definitions.get(func_name)
        if callee is None:
            return None

        params = list(getattr(callee, "parameters", getattr(callee, "params", [])))
        dynamic_arg = None
        for index, (param, arg) in enumerate(zip(params, args or [])):
            param_type = self.type_name_string(
                getattr(param, "param_type", getattr(param, "vtype", None))
            )
            if not self.is_sampled_texture_type(param_type):
                continue
            dynamic_info = self.glsl_dynamic_resource_array_access_info(
                arg,
                self.current_resource_aliases,
            )
            if dynamic_info is None:
                continue
            if dynamic_arg is not None:
                return None
            dynamic_arg = {
                "arg_index": index,
                **dynamic_info,
            }

        return dynamic_arg

    def should_emit_stage_io_layout(self, stage_name, direction):
        if normalize_stage_name(stage_name) == "fragment" and direction == "in":
            return False
        return super().should_emit_stage_io_layout(stage_name, direction)

    def generate_program(self, ast, target_stage=None):
        target_stage = normalize_stage_name(target_stage)
        self.validate_webgl_stage_support(ast, target_stage)
        code = super().generate_program(
            self.webgl_supported_stage_ast(ast, target_stage),
            target_stage=target_stage,
        )
        return self._with_default_precision(code)

    def validate_webgl_stage_support(self, ast, target_stage=None):
        normalized_target_stage = normalize_stage_name(target_stage)
        if normalized_target_stage:
            stages = {normalized_target_stage}
        else:
            stages = self.webgl_stage_names(ast)

        unsupported = sorted(stages & self.UNSUPPORTED_STAGE_NAMES)
        if unsupported:
            supported = stages & self.SUPPORTED_STAGE_NAMES
            if supported:
                return
            raise ValueError(
                "WebGL target does not support shader stage(s): "
                + ", ".join(unsupported)
            )

    def webgl_stage_names(self, ast):
        stages = set()
        for func in getattr(ast, "functions", []) or []:
            stage_name = self.webgl_function_stage_name(func)
            if stage_name:
                stages.add(stage_name)
        for stage_type in getattr(ast, "stages", {}) or {}:
            stage_name = normalize_stage_name(stage_type)
            if stage_name:
                stages.add(stage_name)
        return stages

    def webgl_supported_stage_ast(self, ast, target_stage=None):
        target_stage = normalize_stage_name(target_stage)
        filtered = copy(ast)
        filtered.functions = [
            func
            for func in getattr(ast, "functions", []) or []
            if self.should_emit_webgl_function(func, target_stage)
        ]
        filtered.stages = StageMap()
        for stage_type, stage in (getattr(ast, "stages", {}) or {}).items():
            if self.should_emit_webgl_stage(stage_type, target_stage):
                filtered.stages.append(stage_type, stage)
        return filtered

    def should_emit_webgl_function(self, func, target_stage=None):
        stage_name = self.webgl_function_stage_name(func)
        if not stage_name:
            return True
        if stage_name in self.UNSUPPORTED_STAGE_NAMES:
            return False
        if target_stage is not None:
            return stage_name == target_stage
        return stage_name in self.SUPPORTED_STAGE_NAMES

    def should_emit_webgl_stage(self, stage_type, target_stage=None):
        stage_name = normalize_stage_name(stage_type)
        if stage_name in self.UNSUPPORTED_STAGE_NAMES:
            return False
        if target_stage is not None:
            return stage_name == target_stage
        return stage_name in self.SUPPORTED_STAGE_NAMES

    def webgl_function_stage_name(self, func):
        qualifiers = list(getattr(func, "qualifiers", []) or [])
        qualifier = getattr(func, "qualifier", None)
        if qualifier:
            qualifiers.append(qualifier)
        for entry in qualifiers:
            stage_name = normalize_stage_name(entry)
            if stage_name in STAGE_QUALIFIER_NAMES:
                return stage_name
        for attr in getattr(func, "attributes", []) or []:
            stage_name = normalize_stage_name(getattr(attr, "name", ""))
            if stage_name in STAGE_QUALIFIER_NAMES:
                return stage_name
        return None

    def _with_default_precision(self, code):
        lines = code.splitlines()
        if not lines:
            return code

        existing_precision = {
            "float": any(
                line.strip().startswith("precision ")
                and line.strip().endswith(" float;")
                for line in lines
            ),
            "int": any(
                line.strip().startswith("precision ") and line.strip().endswith(" int;")
                for line in lines
            ),
        }
        precision_lines = [
            line
            for scalar_kind, line in self.DEFAULT_PRECISION_LINES
            if not existing_precision[scalar_kind]
        ]
        if not precision_lines:
            return code

        insert_at = next(
            (
                index + 1
                for index, line in enumerate(lines)
                if line.startswith("#version")
            ),
            0,
        )
        while insert_at < len(lines) and lines[insert_at].startswith("#extension"):
            insert_at += 1
        lines[insert_at:insert_at] = precision_lines
        return "\n".join(lines) + ("\n" if code.endswith("\n") else "")

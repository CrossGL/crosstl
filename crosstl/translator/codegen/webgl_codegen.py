"""CrossGL-to-WebGL GLSL ES code generator."""

from copy import copy

from ..ast import StageMap
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

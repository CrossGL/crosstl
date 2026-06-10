"""CrossGL-to-WebGL GLSL ES code generator."""

from .GLSL_codegen import GLSLCodeGen
from .stage_utils import normalize_stage_name


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

    def default_glsl_version_line(self, ast, target_stage=None):
        return "#version 300 es"

    def generate_program(self, ast, target_stage=None):
        self.validate_webgl_stage_support(ast, target_stage)
        code = super().generate_program(ast, target_stage=target_stage)
        return self._with_default_precision(code)

    def validate_webgl_stage_support(self, ast, target_stage=None):
        stages = set()
        normalized_target_stage = normalize_stage_name(target_stage)
        if normalized_target_stage:
            stages.add(normalized_target_stage)

        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if getattr(func, "qualifiers", None)
                else getattr(func, "qualifier", None)
            )
            stage_name = normalize_stage_name(qualifier)
            if stage_name:
                stages.add(stage_name)
        for stage_type in getattr(ast, "stages", {}) or {}:
            stage_name = normalize_stage_name(stage_type)
            if stage_name:
                stages.add(stage_name)

        unsupported = sorted(stages & self.UNSUPPORTED_STAGE_NAMES)
        if unsupported:
            raise ValueError(
                "WebGL target does not support shader stage(s): "
                + ", ".join(unsupported)
            )

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

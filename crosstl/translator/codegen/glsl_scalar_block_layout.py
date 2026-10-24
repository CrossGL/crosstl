"""The ``GL_EXT_scalar_block_layout`` directive of GLSL output that declares a ``scalar`` buffer block.

A ``layout(scalar)`` block qualifier needs ``GL_EXT_scalar_block_layout`` (GLSL_EXT_scalar_block_layout, "Modify
Section 4.4.5, Uniform and Shader Storage Block Layout Qualifiers"). A GLSL source that declares one carries the
directive along with its other preprocessor lines, but a CrossGL program from another language does not: an HLSL
shader record imported with the scalar layout (:mod:`crosstl.backend.DirectX.shader_record_import`) then emitted a
block glslang rejects. :func:`scalar_block_layout_extension_line` adds the directive when a block needs it and the
program does not already enable it.
"""

from __future__ import annotations

__all__ = ["SCALAR_BLOCK_LAYOUT_EXTENSION", "scalar_block_layout_extension_line"]

SCALAR_BLOCK_LAYOUT_EXTENSION = "GL_EXT_scalar_block_layout"


def scalar_block_layout_extension_line(codegen, ast):
    """``#extension GL_EXT_scalar_block_layout : require`` when a buffer block needs it, else ``None``."""
    if SCALAR_BLOCK_LAYOUT_EXTENSION in codegen.glsl_preprocessor_extensions(ast):
        return None
    stages = getattr(ast, "stages", None) or {}
    stage_variables = [
        variable
        for stage in (stages.values() if hasattr(stages, "values") else stages)
        for variable in getattr(stage, "local_variables", None) or []
    ]
    for node in list(getattr(ast, "global_variables", None) or []) + stage_variables:
        attribute = codegen.glsl_buffer_block_attribute(node)
        if attribute is None:
            continue
        for argument in getattr(attribute, "arguments", None) or []:
            text = codegen.attribute_value_to_string(argument)
            if text is not None and str(text).strip().lower() == "scalar":
                return f"#extension {SCALAR_BLOCK_LAYOUT_EXTENSION} : require"
    return None

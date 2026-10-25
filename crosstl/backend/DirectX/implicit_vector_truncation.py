"""CrossGL spelling of HLSL's implicit vector truncation on assignment and initialization.

HLSL converts a vector to a narrower vector or to a scalar implicitly by keeping its leading components; DXC
accepts it with warning X3206 ("implicit truncation of vector type"). CrossGL, like GLSL (GLSL 4.60 section
4.1.10), has no implicit width-changing conversion, so the importer spells the truncation as a swizzle of the
leading components.
"""

_COMPONENTS = "xyzw"


class HLSLImplicitVectorTruncationMixin:
    """Implicit vector truncation members of ``HLSLToCrossGLConverter``."""

    def implicitly_truncated_value(self, value, rendered, target_width):
        """``rendered`` narrowed to ``target_width`` components when ``value`` is a wider vector."""
        if not target_width or not isinstance(rendered, str):
            return rendered
        if self.matrix_expression_shape(value) is not None:
            return rendered
        width = self.hlsl_component_width(value)
        if width is None or width <= 1 or width <= target_width:
            return rendered
        return self.swizzle_rendered_expression(rendered, _COMPONENTS[:target_width])

    def assignment_target_width(self, target):
        """Component count of a scalar or vector assignment target, otherwise ``None``."""
        if self.matrix_expression_shape(target) is not None:
            return None
        return self.hlsl_component_width(target)

    def truncated_assignment_value(self, node, rendered):
        """The right-hand side of ``node`` narrowed to its target's width."""
        return self.implicitly_truncated_value(
            node.right, rendered, self.assignment_target_width(node.left)
        )

    def truncated_initializer_value(self, declaration, rendered):
        """A local initializer narrowed to its declared scalar or vector type."""
        if getattr(declaration, "array_size", None) or getattr(
            declaration, "array_sizes", None
        ):
            return rendered
        width = self.raw_type_component_width(getattr(declaration, "vtype", None))
        return self.implicitly_truncated_value(declaration.value, rendered, width)

"""Texture gathers and level/sample-count queries in the WGSL generator.

``textureGather(gsampler, P [, comp])`` and ``textureGatherOffset(gsampler, P, offset [, comp])`` read one component
of the four texels a bilinear lookup would use (GLSL 4.60 section 8.9.4). WGSL writes them as
``textureGather(component, t, s, coords [, array_index] [, offset])`` (WGSL section 17.7.6), whose component must be
a const-expression from 0 to 3. GLSL also requires a constant integral component, so a component that is not an
integer constant from 0 to 3 is refused rather than written.

A gather on a shadow sampler with a ``refZ`` operand, and CrossGL's ``textureGatherCompare`` and
``textureGatherCompareOffset``, compare each texel with the reference: WGSL writes them as
``textureGatherCompare(t, s_cmp, coords [, array_index], depth_ref [, offset])`` (WGSL section 17.7.7). A gather on a
depth texture without a reference is refused, because the texture's split sampler is a ``sampler_comparison`` and
WGSL's ``textureGather`` on a depth texture takes a filtering ``sampler``.

WGSL takes a gather offset only for 2D and 2D array textures, as GLSL does, so an offset gather on a cube texture is
refused. ``textureGatherOffsets`` has no WGSL counterpart and stays refused by the texture call dispatcher.

``textureQueryLevels`` (GLSL 4.60 section 8.9.1) is ``textureNumLevels`` and ``textureSamples`` is
``textureNumSamples`` (WGSL section 17.7.11 and 17.7.12). Both return ``u32`` in WGSL and ``int`` in CrossGL, so the
result is converted to the type its context expects.
"""

from __future__ import annotations

import re

from ..ast import IdentifierNode, LiteralNode, UnaryOpNode
from ..refusals import WGSLRefusal

__all__ = ["WGSLTextureGatherMixin"]

# The CrossGL gather operations the dispatcher sends here, by normalized name.
GATHER_OPERATIONS = frozenset(
    {
        "texturegather",
        "texturegatheroffset",
        "texturegathercompare",
        "texturegathercompareoffset",
    }
)

# WGSL texture types textureGather and textureGatherCompare accept (WGSL sections 17.7.6 and 17.7.7).
_GATHER_TEXTURES = frozenset(
    {
        "texture_2d",
        "texture_2d_array",
        "texture_cube",
        "texture_cube_array",
    }
)
_GATHER_COMPARE_TEXTURES = frozenset(
    {
        "texture_depth_2d",
        "texture_depth_2d_array",
        "texture_depth_cube",
        "texture_depth_cube_array",
    }
)
# The texture types whose gathers take an offset.
_GATHER_OFFSET_TEXTURES = frozenset(
    {"texture_2d", "texture_2d_array", "texture_depth_2d", "texture_depth_2d_array"}
)
# WGSL texture types textureNumLevels accepts (WGSL section 17.7.11): every sampled, non-multisampled type.
_LEVEL_QUERY_TEXTURES = frozenset(
    {
        "texture_1d",
        "texture_2d",
        "texture_2d_array",
        "texture_3d",
        "texture_cube",
        "texture_cube_array",
        "texture_depth_2d",
        "texture_depth_2d_array",
        "texture_depth_cube",
        "texture_depth_cube_array",
    }
)


def _base_texture_type(texture_type):
    return re.sub(r"<[^>]*>$", "", texture_type or "")


def _constant_component(expr):
    """The integer value of a literal gather component, or None."""
    if isinstance(expr, UnaryOpNode) and getattr(expr, "operator", None) == "-":
        value = _constant_component(getattr(expr, "operand", None))
        return None if value is None else -value
    if not isinstance(expr, LiteralNode):
        return None
    value = expr.value
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


class WGSLTextureGatherMixin:
    """Gather and texture query members of ``WGSLCodeGen``."""

    def generate_texture_gather_call(self, function_name, args):
        normalized = self.semantic_key(function_name)
        if not args:
            raise WGSLRefusal(
                f"WGSL target requires {function_name}() to name a texture"
            )
        texture = args[0]
        texture_type = _base_texture_type(
            self.sampled_texture_type(self.expression_type(texture))
        )
        explicit = len(args) > 1 and self.is_sampler_type(self.expression_type(args[1]))
        operands = list(args[2:] if explicit else args[1:])
        if not operands:
            raise WGSLRefusal(
                f"WGSL target requires {function_name}() to take a coordinate"
            )
        coords, *rest = operands
        offset_form = normalized.endswith("offset")
        compare = normalized.startswith("texturegathercompare") or (
            texture_type in _GATHER_COMPARE_TEXTURES
        )
        if compare:
            return self._texture_gather_compare_call(
                function_name, texture, texture_type, args, explicit, coords, rest
            )
        if texture_type not in _GATHER_TEXTURES:
            raise WGSLRefusal(
                f"WGSL target cannot lower {function_name}() on "
                f"{self.generate_expression(texture)}: WGSL's textureGather takes a "
                "2D, 2D array, cube or cube array texture (WGSL section 17.7.6)"
            )
        if explicit:
            self.refuse_comparison_sampler(
                args[1], function_name, texture, "textureGather"
            )
        offset = None
        if offset_form:
            if not rest:
                raise WGSLRefusal(
                    f"WGSL target requires {function_name}() to take an offset"
                )
            offset, *rest = rest
            self._require_gather_offset_texture(function_name, texture, texture_type)
        if len(rest) > 1:
            raise WGSLRefusal(
                f"WGSL target supports {function_name}() with one optional "
                f"component operand; got {len(rest)} trailing operands"
            )
        component = self._gather_component(function_name, rest[0] if rest else None)
        call_args = [
            str(component),
            self.generate_expression(texture),
            (
                self.generate_expression(args[1])
                if explicit
                else self.texture_sampler_expression(texture)
            ),
            self.sampled_texture_coordinates(texture, coords, function_name),
        ]
        if offset is not None:
            call_args.append(self.generate_expression(offset))
        return "textureGather(" + ", ".join(call_args) + ")"

    def _texture_gather_compare_call(
        self, function_name, texture, texture_type, args, explicit, coords, rest
    ):
        if texture_type not in _GATHER_COMPARE_TEXTURES:
            raise WGSLRefusal(
                f"WGSL target cannot lower {function_name}() on "
                f"{self.generate_expression(texture)}: WGSL's textureGatherCompare "
                "takes a depth texture (WGSL section 17.7.7)"
            )
        if not rest:
            raise WGSLRefusal(
                f"WGSL target cannot lower {function_name}() on the depth texture "
                f"{self.generate_expression(texture)} without a depth reference: its "
                "split sampler is a sampler_comparison, which WGSL's textureGather "
                "does not take"
            )
        offset_form = self.semantic_key(function_name).endswith("offset")
        expected = 2 if offset_form else 1
        if len(rest) != expected:
            raise WGSLRefusal(
                f"WGSL target supports {function_name}() on a depth texture with a "
                f"depth reference{' and an offset' if offset_form else ''}; got "
                f"{len(rest)} operand(s) after the coordinate"
            )
        if explicit:
            if not self.is_comparison_sampler_type(self.expression_type(args[1])):
                raise WGSLRefusal(
                    f"WGSL target requires {function_name}() explicit sampler "
                    "operand to be a comparison_sampler"
                )
            sampler = self.generate_expression(args[1])
        else:
            sampler = self.texture_sampler_expression(texture)
        call_args = [
            self.generate_expression(texture),
            sampler,
            self.sampled_texture_coordinates(texture, coords, function_name),
            self.generate_expression(rest[0]),
        ]
        if offset_form:
            self._require_gather_offset_texture(function_name, texture, texture_type)
            call_args.append(self.generate_expression(rest[1]))
        return "textureGatherCompare(" + ", ".join(call_args) + ")"

    def _require_gather_offset_texture(self, function_name, texture, texture_type):
        if texture_type in _GATHER_OFFSET_TEXTURES:
            return
        raise WGSLRefusal(
            f"WGSL target cannot lower {function_name}() on "
            f"{self.generate_expression(texture)}: WGSL takes a gather offset only "
            "for 2D and 2D array textures (WGSL sections 17.7.6 and 17.7.7)"
        )

    def _gather_component(self, function_name, expr):
        if expr is None:
            return 0
        value = _constant_component(expr)
        if value is None and isinstance(expr, IdentifierNode):
            # A named integer constant, such as ``const int GREEN = 1;``.
            value = self.resolve_integer_expression(expr)
            if isinstance(value, bool) or not isinstance(value, int):
                value = None
        if value is None or not 0 <= value <= 3:
            raise WGSLRefusal(
                f"WGSL target requires the {function_name}() component to be an "
                "integer constant from 0 to 3, because WGSL's textureGather takes "
                "it as a const-expression (WGSL section 17.7.6)"
            )
        return value

    def generate_texture_query_levels_call(self, function_name, args):
        if len(args) != 1:
            raise WGSLRefusal(
                f"WGSL target supports {function_name}() with exactly 1 argument; "
                f"got {len(args)}"
            )
        texture = args[0]
        texture_type = _base_texture_type(
            self.sampled_texture_type(self.expression_type(texture))
        )
        if texture_type not in _LEVEL_QUERY_TEXTURES:
            raise WGSLRefusal(
                f"WGSL target cannot lower {function_name}() on "
                f"{self.generate_expression(texture)}: WGSL's textureNumLevels takes "
                "a sampled, non-multisampled texture (WGSL section 17.7.11)"
            )
        return self._texture_count_result(
            f"textureNumLevels({self.generate_expression(texture)})"
        )

    def generate_texture_samples_call(self, function_name, args):
        if len(args) != 1:
            raise WGSLRefusal(
                f"WGSL target supports {function_name}() with exactly 1 argument; "
                f"got {len(args)}"
            )
        texture = args[0]
        if self.multisampled_texture_type(self.expression_type(texture)) is None:
            raise WGSLRefusal(
                f"WGSL target cannot lower {function_name}() on "
                f"{self.generate_expression(texture)}: WGSL's textureNumSamples takes "
                "a multisampled texture (WGSL section 17.7.12)"
            )
        return self._texture_count_result(
            f"textureNumSamples({self.generate_expression(texture)})"
        )

    def _texture_count_result(self, call):
        """``call`` (a u32 count) as the type the expression's context expects; CrossGL's result is int."""
        expected = self._current_expression_expected_type
        try:
            expected_name = (
                self.type_name_string(expected) if expected is not None else None
            )
        except ValueError:
            expected_name = None
        if expected_name == "u32":
            return call
        if expected_name == "f32":
            return f"f32({call})"
        return f"i32({call})"

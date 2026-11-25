"""Boolean group votes in the GLSL generator: GL_ARB_shader_group_vote for a stage whose only subgroup operations are
votes on bools (#3009)."""

from ..ast import (
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    ParameterNode,
    StructMemberNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)
from .glsl_builtin_extensions import BUILTIN_EXTENSIONS
from .glsl_version_floor import glsl_version_reaches

__all__ = ["OpenGLGroupVoteMixin"]


class OpenGLGroupVoteMixin:
    # GLSL 4.60 (section 8.18) took these votes from GL_ARB_shader_group_vote, which OpenGL drivers without the
    # GL_KHR_shader_subgroup extensions implement (Mesa's llvmpipe rejects GL_KHR_shader_subgroup_basic). They take
    # a bool; SPIR-V gives the invocation group subgroup scope, as for subgroupAll and friends.
    GLSL_GROUP_VOTE_FUNCTIONS = {
        "WaveActiveAllTrue": "allInvocations",
        "WaveActiveAnyTrue": "anyInvocation",
        "WaveActiveAllEqual": "allInvocationsEqual",
    }
    GLSL_GROUP_VOTE_EXTENSION = "GL_ARB_shader_group_vote"

    def glsl_group_vote_suffix(self, ast, target_stage=None):
        """The suffix the stage's votes take as GL_ARB_shader_group_vote functions: ``"ARB"`` from GLSL 4.30 to 4.50,
        ``""`` from 4.60, where they are core. ``None`` keeps them GL_KHR_shader_subgroup_vote calls: the version is
        below 4.30, the stage uses another subgroup operation or built-in, or it votes on a value this check cannot
        show is a bool (#3009).
        """
        version_line = getattr(self, "current_glsl_version_line", None)
        version_number = self.glsl_version_number(version_line)
        if (
            version_number is None
            or self.software_subgroup_width is not None
            or self.glsl_version_is_es(version_line)
            or self.GLSL_GROUP_VOTE_EXTENSION in getattr(
                self, "current_glsl_disabled_extensions", set()
            )
        ):
            return None
        operations = self.glsl_wave_operations(ast, target_stage)
        if not operations or not operations <= set(self.GLSL_GROUP_VOTE_FUNCTIONS):
            return None
        if (
            self.glsl_stage_requires_exact_subgroup_width(ast, target_stage)
            or self.uses_glsl_subgroup_basic_builtin(ast, target_stage)
            or self.glsl_software_cooperative_matrix_uses_multiply_accumulate(
                ast, target_stage
            )
        ):
            return None
        roots = self.ray_query_search_roots(ast, target_stage)
        if self.uses_glsl_subgroup_extension_builtin(roots):
            return None
        # Every declaration in the program counts, uniform-block and struct members included: a name declared with
        # any other type, anywhere, is not taken for a bool.
        declared_types = {}
        for node in self.walk_ast(ast):
            if isinstance(node, VariableNode):
                declared = getattr(node, "vtype", None)
            elif isinstance(node, ParameterNode):
                declared = node.param_type
            elif isinstance(node, StructMemberNode):
                declared = node.member_type
            elif isinstance(node, ConstantNode):
                declared = node.const_type
            else:
                continue
            declared_types.setdefault(node.name, set()).add(
                self.type_name_string(declared)
            )
        bool_names = {
            name for name, types in declared_types.items() if types == {"bool"}
        }
        for root in roots:
            for node in self.walk_ast(root):
                if isinstance(node, WaveOpNode):
                    operation = node.operation
                elif isinstance(node, FunctionCallNode):
                    operation = self.glsl_wave_operation_name(
                        self.function_call_name(node)
                    )
                else:
                    continue
                if operation == "WaveActiveAllEqual" and not (
                    len(node.arguments) == 1
                    and self.glsl_is_bool_vote_operand(node.arguments[0], bool_names)
                ):
                    return None
        # GL_ARB_shader_group_vote requires OpenGL 4.3, and glslang declares its functions only from #version 430.
        # A program without a #version line of its own rises to 4.30 for them rather than needing
        # GL_KHR_shader_subgroup_vote, which drivers such as Mesa's llvmpipe do not implement.
        if version_number < 430 and not glsl_version_reaches(
            self, "boolean group votes with GL_ARB_shader_group_vote", 430, None
        ):
            return None
        return "" if version_number >= 460 else "ARB"

    def set_glsl_stage_group_vote_suffix(self, ast, target_stage=None):
        """Record :meth:`glsl_group_vote_suffix` for the stage being generated."""
        self.current_glsl_group_vote_suffix = self.glsl_group_vote_suffix(
            ast, target_stage
        )

    def uses_glsl_subgroup_extension_builtin(self, roots):
        """Whether ``roots`` read a built-in a GL_KHR_shader_subgroup extension declares, such as
        ``gl_SubgroupEqMask``, by name or through a semantic."""
        for root in roots:
            for node in self.walk_ast(root):
                names = []
                if isinstance(node, IdentifierNode):
                    names.append(node.name)
                if isinstance(getattr(node, "name", None), str):
                    semantic = self.semantic_from_node(node)
                    if semantic is not None:
                        names.append(self.map_semantic(semantic))
                for name in names:
                    if isinstance(name, str) and BUILTIN_EXTENSIONS.get(
                        name, ""
                    ).startswith("GL_KHR_shader_subgroup_"):
                        return True
        return False

    def glsl_is_bool_vote_operand(self, node, bool_names):
        """Whether ``node`` is a scalar bool: a bool literal, a name declared only as ``bool``, or ``!``, ``&&`` and
        ``||`` of such operands."""
        if isinstance(node, LiteralNode):
            return isinstance(node.value, bool)
        if isinstance(node, (IdentifierNode, VariableNode)):
            return node.name in bool_names
        if isinstance(node, UnaryOpNode):
            return node.operator == "!" and self.glsl_is_bool_vote_operand(
                node.operand, bool_names
            )
        if isinstance(node, BinaryOpNode):
            operator = self.map_operator(
                getattr(node, "op", getattr(node, "operator", None))
            )
            return (
                operator in {"&&", "||"}
                and self.glsl_is_bool_vote_operand(node.left, bool_names)
                and self.glsl_is_bool_vote_operand(node.right, bool_names)
            )
        return False

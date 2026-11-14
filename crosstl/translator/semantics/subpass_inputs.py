"""Vulkan subpass inputs as textures read at the fragment's own pixel, for targets without input attachments.

A Vulkan input attachment is declared in GLSL as ``subpassInput`` (GL_KHR_vulkan_glsl) and in SPIR-V as an
``OpTypeImage`` with ``Dim SubpassData``. It is read with ``subpassLoad(x)``, which returns the attachment's value
at the fragment's own framebuffer coordinates; the multisampled form ``subpassLoad(x, sample)`` reads one sample
there. OpenGL, GLSL ES, Direct3D and Metal shaders have no input attachments. The attachment written by an earlier
pass is bound as a texture there instead, and the value comes from a texel fetch at the fragment's integer pixel
coordinates. SPIRV-Cross ports subpass inputs to GLSL and HLSL the same way.

For those targets :func:`lower_subpass_inputs` rewrites the program in place:

* ``subpassInput``, ``isubpassInput`` and ``usubpassInput`` declarations and parameters become ``sampler2D``,
  ``isampler2D`` and ``usampler2D``, and the ``MS`` forms become ``sampler2DMS`` and its integer variants. The
  declarations keep their ``@set`` and ``@binding``, which is where the host binds the attachment's image, and
  drop ``@input_attachment_index``, which no texture has;
* ``subpassLoad(x)`` becomes ``texelFetch(x, ivec2(gl_FragCoord.xy), 0)``, and ``subpassLoad(x, sample)`` becomes
  ``texelFetch(x, ivec2(gl_FragCoord.xy), sample)``. Each generator spells the fragment coordinate as its
  built-in (HLSL ``SV_Position``, MSL ``[[position]]``).

Under multiview, ``subpassLoad`` reads the layer of the current view. A program that reads ``gl_ViewIndex`` is
therefore refused with :class:`SubpassInputUnsupportedError`, because a ``sampler2D`` read has no layer. The Metal
generator binds the fragment position (``[[position]]``) only as an entry-point parameter, so for Metal a subpass
input read in a helper function is refused as well.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name, iter_functions
from .rewrite import call, child_slots, get_child, set_child

__all__ = ["SubpassInputUnsupportedError", "lower_subpass_inputs"]

TARGETS = frozenset(
    {
        "opengl",
        "glsl",
        "webgl",
        "metal",
        "directx",
        "hlsl",
        "dx11",
        "dx12",
        "d3d11",
        "d3d12",
    }
)
_TEXTURE_TYPES = {
    f"{prefix}subpassInput{ms}": f"{prefix}sampler2D{ms}"
    for prefix in ("", "i", "u")
    for ms in ("", "MS")
}


class SubpassInputUnsupportedError(TranslationDiagnostic):
    """A subpass input read that a texture fetch at the fragment's pixel cannot reproduce."""

    def __init__(self, target, detail):
        self.missing_capabilities = (f"{target}.input-attachment",)
        super().__init__(
            f"subpass input {detail}",
            target=target,
            code=f"{CODE_PREFIX}{target}-subpass-input-unsupported",
        )


def _texture_type(type_node):
    """The texture type that replaces a subpass input type, or ``None``."""
    if isinstance(type_node, A.ArrayType):
        element = _texture_type(type_node.element_type)
        return None if element is None else A.ArrayType(element, type_node.size)
    name = getattr(type_node, "name", type_node)
    if not isinstance(name, str) or getattr(type_node, "generic_args", None):
        return None
    replacement = _TEXTURE_TYPES.get(name)
    return None if replacement is None else A.NamedType(replacement)


def _lower_declaration(node, field):
    declared = getattr(node, field, None)
    lowered = _texture_type(declared)
    if lowered is None:
        return False
    for name, value in list(vars(node).items()):
        # Alias attributes (``vtype``) hold the same type object.
        if value is declared:
            setattr(node, name, lowered)
    node.attributes = [
        attribute
        for attribute in getattr(node, "attributes", None) or []
        if getattr(attribute, "name", None) != "input_attachment_index"
    ]
    return True


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _declarations(program):
    yield from getattr(program, "global_variables", None) or []
    for stage in _stages(program):
        yield from getattr(stage, "local_variables", None) or []


def _fragment_coordinate():
    return call("ivec2", [A.MemberAccessNode(A.IdentifierNode("gl_FragCoord"), "xy")])


def _rewrite(node, replace):
    for field, key in list(child_slots(node)):
        child = get_child(node, field, key)
        _rewrite(child, replace)
        replacement = replace(child)
        if replacement is not child:
            set_child(node, field, key, replacement)


def _reads_view_index(functions):
    found = []

    def note(node):
        if isinstance(node, A.IdentifierNode) and node.name == "gl_ViewIndex":
            found.append(node)
        return node

    for function in functions:
        _rewrite(function, note)
    return bool(found)


def lower_subpass_inputs(program, target) -> int:
    """Rewrite the subpass inputs of ``program`` as textures for ``target``, in place; returns the reads lowered."""
    target = str(target or "").lower()
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    lowered_names = {
        node.name
        for node in _declarations(program)
        if _lower_declaration(node, "var_type")
    }
    functions = list(iter_functions(program))
    for function in functions:
        for parameter in getattr(function, "parameters", None) or []:
            if _lower_declaration(parameter, "param_type"):
                lowered_names.add(parameter.name)
    if not lowered_names:
        return 0
    if _reads_view_index(functions):
        raise SubpassInputUnsupportedError(
            target,
            "read under multiview: subpassLoad reads the current view's layer of the attachment, which a "
            f"{target} texture read at the fragment's pixel does not select",
        )
    defined = {getattr(function, "name", None) for function in functions}
    lowered = 0

    def replace(node):
        nonlocal lowered
        if not isinstance(node, A.FunctionCallNode) or "subpassLoad" in defined:
            return node
        if call_name(node) != "subpassLoad":
            return node
        arguments = list(getattr(node, "arguments", None) or [])
        if len(arguments) not in (1, 2):
            return node
        sample = (
            arguments[1]
            if len(arguments) == 2
            else A.LiteralNode(0, A.PrimitiveType("int"))
        )
        lowered += 1
        return call("texelFetch", [arguments[0], _fragment_coordinate(), sample])

    entry_points = {
        id(getattr(stage, "entry_point", None))
        for stage in _stages(program)
        if getattr(stage, "entry_point", None) is not None
    }
    for function in functions:
        before = lowered
        _rewrite(function, replace)
        if target == "metal" and lowered > before and id(function) not in entry_points:
            raise SubpassInputUnsupportedError(
                target,
                f"read in helper function '{function.name}': the Metal generator binds the fragment position "
                "only as an entry-point parameter",
            )
    return lowered

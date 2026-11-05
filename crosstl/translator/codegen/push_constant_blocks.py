"""Push-constant blocks in GLSL and Slang output.

CrossGL marks a constant block that the host pushes through the command buffer (``vkCmdPushConstants``) instead of
binding it through a descriptor with ``@push_constant``: ``cbuffer Constants @push_constant { ... }``, or a
``ConstantBuffer<T> name @push_constant`` (``uniform T name @push_constant``) whose members are read as
``name.member``. Every importer writes it (GLSL ``layout(push_constant)``, HLSL and Slang ``[[vk::push_constant]]``,
SPIR-V ``PushConstant`` storage), and SPIR-V output declares the block in the ``PushConstant`` storage class
(:mod:`.spirv_push_constants`). The block is laid out with the std430 rules unless it names ``std140`` or
``scalar`` through ``@ buffer_layout``, and a member ``@ offset(N)`` places that member at byte ``N``, as in a
push-constant range that several stages share (Vulkan 1.3 section 15.6.4, GL_KHR_vulkan_glsl).

GLSL and Slang output declared the block as an ordinary uniform buffer at a descriptor binding, which the host never
binds for push constants, so the program read constants from an unbound descriptor and nothing reported it.

* GLSL has push constants only when it is compiled for Vulkan (GL_KHR_vulkan_glsl, which defines ``VULKAN``;
  GL_ARB_gl_spirv removes ``push_constant``). The block is declared twice: ``layout(push_constant) uniform`` with
  its members' ``layout(offset = N)`` qualifiers under ``#ifdef VULKAN``, and the uniform block OpenGL output has
  always declared (with its binding, and with the register loader of :mod:`.glsl_hlsl_cbuffer` when std140 cannot
  place the members) in the ``#else`` branch. A register loader the stage entries call is defined as an empty
  function in the Vulkan branch, where the block's members are read directly. A ``scalar`` block enables
  ``GL_EXT_scalar_block_layout`` in the Vulkan branch.
* Slang declares ``[[vk::push_constant]] cbuffer Name { [[vk::offset(N)]] T member; }`` or
  ``[[vk::push_constant]] ConstantBuffer<T> name``, which ``slangc`` lays out with the std430 rules and places in
  the ``PushConstant`` storage class; the block takes no ``vk::binding``, and ``slangc`` gives it a Direct3D
  register. Slang has no attribute that lays a push-constant block out with the std140 or scalar rules, so such a
  block is refused.

A stage has at most one push-constant block (GL_KHR_vulkan_glsl, VUID-StandaloneSpirv-OpEntryPoint-06674); a second
one is refused, as is an array of push-constant blocks and a member array without a size. Every refusal is a
:class:`PushConstantBlockError`.
"""

from __future__ import annotations

from ..ast import ArrayNode, FunctionNode
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .array_utils import format_c_style_array_declaration
from .buffer_member_layout import member_explicit_offset
from .glsl_block_layout import declared_buffer_layout
from .hlsl_buffer_layout import HlslLayoutError

__all__ = [
    "PUSH_CONSTANT_GUARD",
    "PushConstantBlockError",
    "glsl_push_constant_cbuffer",
    "glsl_push_constant_struct_block",
    "is_push_constant",
    "push_constant_block_layout",
    "slang_push_constant_cbuffer",
    "slang_push_constant_declaration",
]

PUSH_CONSTANT_ATTRIBUTE = "push_constant"
# GL_KHR_vulkan_glsl defines VULKAN when a shader is compiled for Vulkan; GL_ARB_gl_spirv has no push constants.
PUSH_CONSTANT_GUARD = "#ifdef VULKAN\n"
PUSH_CONSTANT_LAYOUTS = ("std430", "std140", "scalar")
SCALAR_BLOCK_LAYOUT_EXTENSION = "#extension GL_EXT_scalar_block_layout : require\n"


class PushConstantBlockError(TranslationDiagnostic):
    """A push-constant block the target cannot declare as one."""

    def __init__(self, target, block, reason):
        self.block = block
        self.missing_capabilities = (f"{target}.push-constant-block",)
        super().__init__(
            f"push-constant block '{block}': {reason}",
            target=target,
            code=f"{CODE_PREFIX}{target}-push-constant-block-unsupported",
        )


def is_push_constant(node) -> bool:
    """Whether a CrossGL cbuffer or constant-buffer variable carries ``@push_constant``."""
    return any(
        str(getattr(attribute, "name", "")).lower() == PUSH_CONSTANT_ATTRIBUTE
        for attribute in getattr(node, "attributes", None) or []
    )


def push_constant_block_layout(node, target, name) -> str:
    """The block layout of a push-constant block: the one it declares, else ``std430``."""
    layout = declared_buffer_layout(getattr(node, "attributes", None))
    if layout is None:
        return "std430"
    if layout not in PUSH_CONSTANT_LAYOUTS:
        raise PushConstantBlockError(
            target,
            name,
            f"it declares the {layout} layout; push constants are laid out with the std430, std140 or scalar "
            "rules",
        )
    return layout


def _claim(codegen, target, name) -> None:
    """Record a push-constant block of the program being generated; a second, different one is refused."""
    names = getattr(codegen, "push_constant_block_names", None)
    if names is None:
        names = []
        codegen.push_constant_block_names = names
    if names and name not in names:
        raise PushConstantBlockError(
            target,
            name,
            f"the program already declares push-constant block '{names[0]}', and a stage has at most one "
            "(GL_KHR_vulkan_glsl, VUID-StandaloneSpirv-OpEntryPoint-06674)",
        )
    if name not in names:
        names.append(name)


def _members(node):
    return [
        member
        for member in getattr(node, "members", None) or []
        if getattr(member, "name", None) and not isinstance(member, FunctionNode)
    ]


def _member_offset(member, target, block):
    try:
        return member_explicit_offset(member)
    except HlslLayoutError as exc:
        raise PushConstantBlockError(target, block, str(exc)) from exc


# GLSL ---------------------------------------------------------------------------------------------------------------


def _glsl_member_type(codegen, member, target, block, name):
    """``(type, suffix)`` of a GLSL block member: the mapped type and the array suffix written after its name."""
    if isinstance(member, ArrayNode):
        element = getattr(member, "element_type", getattr(member, "vtype", "float"))
        if not member.size:
            raise PushConstantBlockError(
                target,
                block,
                f"member '{name}' is an array without a size, which a uniform block cannot hold",
            )
        return codegen.map_type(element), f"[{member.size}]"
    if hasattr(member, "member_type"):
        member_type = member.member_type
        if not isinstance(member_type, str):
            member_type = codegen.convert_type_node_to_string(member_type)
        mapped = codegen.map_type(member_type)
    else:
        mapped = codegen.map_type(getattr(member, "vtype", "float"))
    if str(mapped).endswith("[]"):
        raise PushConstantBlockError(
            target,
            block,
            f"member '{name}' is an array without a size, which a uniform block cannot hold",
        )
    return mapped, ""


def _glsl_members(codegen, members, name_of, block, offsets):
    """The member declarations of a GLSL block; ``offsets`` writes each ``@ offset(N)`` as ``layout(offset = N)``."""
    code = ""
    for member in members:
        name = name_of(member)
        qualifiers = []
        if codegen.glsl_uniform_member_orientation(member):
            qualifiers.append("row_major")
        offset = _member_offset(member, "opengl", block) if offsets else None
        if offset is not None:
            qualifiers.append(f"offset = {offset}")
        member_type, suffix = _glsl_member_type(codegen, member, "opengl", block, name)
        layout = f"layout({', '.join(qualifiers)}) " if qualifiers else ""
        declaration = format_c_style_array_declaration(member_type, f"{name}{suffix}")
        code += f"    {layout}{declaration};\n"
    return code


def _glsl_vulkan_block(codegen, node, layout, block_name, body, instance):
    qualifiers = "push_constant" if layout == "std430" else f"push_constant, {layout}"
    extension = SCALAR_BLOCK_LAYOUT_EXTENSION if layout == "scalar" else ""
    return f"{extension}layout({qualifiers}) uniform {block_name} {{\n{body}}}{instance};\n"


def _glsl_guarded(vulkan, opengl, loaders):
    # A register loader of the OpenGL branch is called by every stage entry; in the Vulkan branch it does nothing.
    stubs = "".join(f"void {name}() {{}}\n" for name in loaders)
    return f"{PUSH_CONSTANT_GUARD}{vulkan}{stubs}#else\n{opengl}#endif\n"


def glsl_push_constant_cbuffer(codegen, node, layout, block_name):
    """The declaration of a ``cbuffer`` marked ``@push_constant`` in GLSL output, or ``None`` for another cbuffer.

    ``layout`` is the ``layout(std140, binding = N)`` prefix of the OpenGL uniform block that stands for it when the
    shader is not compiled for Vulkan.
    """
    if not is_push_constant(node):
        return None
    block_layout = push_constant_block_layout(node, "opengl", node.name)
    _claim(codegen, "opengl", node.name)
    members = _members(node)

    def name_of(member):
        return codegen.glsl_global_identifier_name(member.name)

    vulkan = _glsl_vulkan_block(
        codegen,
        node,
        block_layout,
        block_name,
        _glsl_members(codegen, members, name_of, node.name, offsets=True),
        "",
    )
    if not hasattr(codegen, "glsl_hlsl_cbuffer_loaders"):
        codegen.reset_glsl_hlsl_cbuffers()
    loaders_before = list(codegen.glsl_hlsl_cbuffer_loaders)
    opengl = codegen.glsl_hlsl_cbuffer_block(node, layout, block_name)
    if opengl is None:
        # std140 places every member where the block's layout does, so the uniform block needs no offsets.
        body = _glsl_members(codegen, members, name_of, node.name, offsets=False)
        opengl = f"{layout} uniform {block_name} {{\n{body}}};\n"
    loaders = [
        name for name in codegen.glsl_hlsl_cbuffer_loaders if name not in loaders_before
    ]
    return _glsl_guarded(vulkan, opengl, loaders)


def glsl_push_constant_struct_block(
    codegen, node, struct, block_name, var_name, array_suffix, opengl, marker=""
):
    """``opengl``, the uniform block of a ``ConstantBuffer<T>`` or ``uniform T`` declaration, guarded by its
    push-constant block when the declaration is marked ``@push_constant``; ``opengl`` itself otherwise.

    ``marker`` is the resource identity line that starts ``opengl``; it stays in front of both branches.
    """
    if node is None or not is_push_constant(node):
        return opengl
    if array_suffix:
        raise PushConstantBlockError(
            "opengl",
            var_name,
            "it is an array; a stage has one push-constant block",
        )
    block_layout = push_constant_block_layout(node, "opengl", var_name)
    _claim(codegen, "opengl", var_name)
    struct_name = getattr(struct, "name", None)
    for member in _members(struct):
        if _member_offset(member, "opengl", var_name) is not None:
            # The uniform block of the OpenGL branch, like the other generators, does not place the members of a
            # struct; a cbuffer carries member offsets.
            raise PushConstantBlockError(
                "opengl",
                var_name,
                f"member '{member.name}' of struct '{struct_name}' has an explicit offset; declare the block as "
                "a cbuffer to place its members",
            )

    def name_of(member):
        return codegen.glsl_struct_member_name(struct_name, member.name)

    vulkan = _glsl_vulkan_block(
        codegen,
        node,
        block_layout,
        block_name,
        _glsl_members(codegen, _members(struct), name_of, var_name, offsets=False),
        f" {var_name}",
    )
    if marker and opengl.startswith(marker):
        opengl = opengl[len(marker) :]
    else:
        marker = ""
    return marker + _glsl_guarded(vulkan, opengl, [])


# Slang --------------------------------------------------------------------------------------------------------------


def _slang_layout(node, name):
    layout = push_constant_block_layout(node, "slang", name)
    if layout != "std430":
        raise PushConstantBlockError(
            "slang",
            name,
            f"it declares the {layout} layout, and Slang lays push constants out with the std430 rules only",
        )


def slang_push_constant_declaration(node, declaration):
    """``[[vk::push_constant]] declaration`` for a constant-buffer variable marked ``@push_constant``, else
    ``None``. The declaration takes no binding: the host pushes it, and ``slangc`` gives it a Direct3D register.
    """
    if not is_push_constant(node):
        return None
    name = getattr(node, "name", None) or "?"
    if isinstance(node, ArrayNode) or "[" in str(declaration):
        raise PushConstantBlockError(
            "slang", name, "it is an array; a stage has one push-constant block"
        )
    _slang_layout(node, name)
    return f"[[vk::push_constant]] {declaration}"


def slang_push_constant_cbuffer(codegen, node):
    """The ``[[vk::push_constant]] cbuffer`` declaration of a ``cbuffer`` marked ``@push_constant``, else ``None``.

    Each member's ``@ offset(N)`` becomes ``[[vk::offset(N)]]``.
    """
    if not is_push_constant(node):
        return None
    _slang_layout(node, node.name)
    result = f"[[vk::push_constant]] cbuffer {node.name} {{\n"
    for member in _members(node):
        if hasattr(member, "member_type"):
            member_type_name = codegen.convert_type_node_to_string(member.member_type)
            # A block's members are module-scope names; their types pick mul() for matrix products.
            codegen.variable_types[member.name] = member_type_name
            member_type = codegen.convert_type(member_type_name)
        else:
            member_type = codegen.convert_type(getattr(member, "vtype", "float"))
        offset = _member_offset(member, "slang", node.name)
        placement = f"[[vk::offset({offset})]] " if offset is not None else ""
        result += (
            f"    {placement}{codegen.format_declaration(member_type, member.name)};\n"
        )
    return result + "};"

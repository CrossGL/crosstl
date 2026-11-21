"""Separate samplers and runtime-sized texture arrays in GLSL output that is compiled for Vulkan.

Desktop OpenGL GLSL has only combined image samplers: a ``sampler2D`` uniform names a texture unit, and the host
binds a texture and a sampler object to that unit (OpenGL 4.6 section 8.2). GLSL for Vulkan (GL_KHR_vulkan_glsl,
whose compilations define ``VULKAN``) also has separate ``texture2D`` and ``sampler`` descriptors, which a shader
combines where it samples with the ``sampler2D(t, s)`` constructor. With GL_EXT_nonuniform_qualifier it also has
arrays of descriptors whose length the descriptor set decides (``uniform sampler2D tex[]``, the
``RuntimeDescriptorArray`` capability). OpenGL GLSL has neither, so GLSL output combines every separate texture
with each sampler it is read through (:mod:`.sampler_pairs`) and sizes every unsized texture array from its indices.
A Vulkan compilation of that output asked the host for combined image samplers and for a fixed number of textures,
which the source never declared.

Such a resource is now declared once per API, as push-constant blocks are (:mod:`.push_constant_blocks`), and a
sampling call picks its operand through a macro that the output defines after its ``#version`` and ``#extension``
lines:

.. code-block:: glsl

    #ifdef VULKAN
    #define CROSSGL_SAMPLED(vulkan, opengl) vulkan
    #else
    #define CROSSGL_SAMPLED(vulkan, opengl) opengl
    #endif
    #ifdef VULKAN
    layout(binding = 0) uniform texture2D albedo;
    #else
    layout(binding = 0) uniform sampler2D albedo;
    #endif
    #ifndef VULKAN
    layout(binding = 1) uniform sampler2D albedo_linear;
    #endif
    #ifdef VULKAN
    layout(binding = 2) uniform sampler point;
    layout(binding = 3) uniform sampler linear;
    #endif
    ... texture(CROSSGL_SAMPLED(sampler2D(albedo, point), albedo), uv)
    ... texture(CROSSGL_SAMPLED(sampler2D(albedo, linear), albedo_linear), uv)

* A separate texture (``texture2D`` and the other non-depth, single-sample shapes) declared at module or stage scope
  keeps its type in the Vulkan branch when every use of it is a sampling call (``texture``, ``textureLod``,
  ``textureGrad``, their offset and projected forms) through a sampler that is declared at module or stage scope and
  is not an array. The combined uniforms of its other sampler pairs exist only in the OpenGL branch. Each of those
  samplers is declared in the Vulkan branch, at the set and binding it names, or else at the lowest binding that no
  other resource of the program takes.
* An unsized array of textures or images declared at module or stage scope stays unsized in the Vulkan branch when
  every use of it indexes it, and that branch requires GL_EXT_nonuniform_qualifier. The OpenGL branch keeps the
  length that its indices need.

Every other form keeps the combined declaration in both compilations, as before: a depth comparison, a gather, a
fetch, a size query, a texture or sampler passed to a helper, a sampler array, a name that another declaration
reuses. So does WebGL and GLSL ES output, and a version without ``layout(binding)``, for which a Vulkan compilation
cannot give a descriptor its binding. After rendering, :meth:`VulkanResourceForms.unwrapped` names each texture that
the program still reads outside its declarations and ``CROSSGL_SAMPLED`` operands, and each runtime array that it
names without an index, and the generator renders the program again without those forms.
"""

from __future__ import annotations

import re

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    ExpressionNode,
    FunctionCallNode,
    IdentifierNode,
)
from .array_utils import format_c_style_array_declaration
from .glsl_spirv_varying_locations import spirv_location_macro_definition
from .glsl_vulkan_descriptor_sets import descriptor_binding_macro_definition
from .image_access_contracts import TEXTURE_SAMPLING_INTRINSIC_NAMES

__all__ = [
    "SAMPLED_OPERAND_MACRO",
    "VulkanResourceForms",
]

SAMPLED_OPERAND_MACRO = "CROSSGL_SAMPLED"
NONUNIFORM_EXTENSION = "GL_EXT_nonuniform_qualifier"

_SHAPES = ("1D", "1DArray", "2D", "2DArray", "3D", "Cube", "CubeArray")
# The separate texture types that a sampler combines into a filtered sampled image, with that combined type.
_SAMPLED_TYPES = {
    f"{prefix}texture{shape}": f"{prefix}sampler{shape}"
    for prefix in ("", "i", "u")
    for shape in _SHAPES
}
_SAMPLER_TYPES = frozenset({"sampler", "SamplerState"})
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_SINGLE_EXTENT = re.compile(r"^\[\s*\d+\s*\]$")
_GUARD = "#ifdef VULKAN\n"


def _type_of(node):
    return getattr(node, "var_type", None) or getattr(node, "vtype", None)


def _base_name(type_node):
    if type_node is None or isinstance(type_node, ArrayType):
        return ""
    text = getattr(type_node, "name", None) or str(type_node)
    return str(text).strip()


def _roots(ast):
    """The module and stage-scope variable declarations of ``ast``, in declaration order."""
    nodes = list(getattr(ast, "global_variables", None) or [])
    for stage in (getattr(ast, "stages", None) or {}).values():
        nodes.extend(getattr(stage, "local_variables", None) or [])
    return [node for node in nodes if isinstance(getattr(node, "name", None), str)]


def _declared_names(codegen, ast):
    """How many declarations (variables, parameters, members) of ``ast`` take each name."""
    counts = {}
    for node in codegen.walk_ast(ast):
        name = getattr(node, "name", None)
        if (
            isinstance(name, str)
            and not isinstance(node, ExpressionNode)
            and any(
                hasattr(node, attribute)
                for attribute in ("var_type", "param_type", "member_type")
            )
        ):
            counts[name] = counts.get(name, 0) + 1
    return counts


class VulkanResourceForms:
    """The separate textures, their samplers and the runtime texture arrays of one rendering, and what it emitted."""

    def __init__(self, textures=None, samplers=None, arrays=None):
        # texture name -> {"type": separate type, "sampled": combined type, "uniforms": {uniform: sampler}}
        self.textures = textures or {}
        # sampler name -> its declaration, in declaration order
        self.samplers = samplers or {}
        self.arrays = arrays or set()
        self.active = False
        self.macro_used = False
        self.emitted_textures = {}
        self.emitted_primaries = {}
        self.emitted_arrays = {}
        self.fragments = []

    @classmethod
    def plan(cls, codegen, source_ast, ast):
        """The forms of ``ast``, the program after :func:`.sampler_pairs.lower_combined_sampler_pairs`.

        ``source_ast`` is the program before that lowering, which still spells each texture's separate type.
        ``codegen.combined_sampler_pairs`` is the lowering's list of (uniform, texture, sampler) pairs, and the
        resources named in ``codegen.glsl_vulkan_resource_exclusions`` keep the combined form.
        """
        if not getattr(codegen, "GLSL_VULKAN_RESOURCE_FORMS", False):
            return cls()
        pairs = codegen.combined_sampler_pairs
        exclusions = codegen.glsl_vulkan_resource_exclusions
        roots = _roots(ast)
        separate = {
            node.name: _base_name(_type_of(node))
            for node in _roots(source_ast)
            if _base_name(_type_of(node)) in _SAMPLED_TYPES
        }
        arrays = {
            node.name
            for node in roots
            if node.name not in exclusions
            and cls._is_runtime_array(codegen, _type_of(node))
        }
        if not separate and not arrays:
            return cls()
        declared = _declared_names(codegen, ast)
        unique = {name for name, count in declared.items() if count == 1}
        samplers = {
            node.name: node
            for node in roots
            if node.name in unique and _base_name(_type_of(node)) in _SAMPLER_TYPES
        }
        textures = cls._textures(separate, pairs, samplers, unique, exclusions)
        arrays = {name for name in arrays if name in unique}
        textures, arrays = cls._check_uses(codegen, ast, textures, arrays)
        used = {
            sampler
            for texture in textures.values()
            for sampler in texture["uniforms"].values()
        }
        return cls(
            textures,
            {name: node for name, node in samplers.items() if name in used},
            arrays,
        )

    @staticmethod
    def _textures(separate, pairs, samplers, unique, exclusions):
        textures = {}
        rejected = set(exclusions)
        for pair in pairs or []:
            texture = pair.get("texture")
            if texture not in separate:
                continue
            simple = (
                pair.get("sampler") in samplers
                and not pair.get("shadow")
                and not any(
                    key in pair
                    for key in ("elementwise", "samplerArray", "samplerIndex")
                )
                and pair.get("uniform") in unique
                and texture in unique
            )
            if not simple:
                rejected.add(texture)
                continue
            entry = textures.setdefault(
                texture,
                {
                    "type": separate[texture],
                    "sampled": _SAMPLED_TYPES[separate[texture]],
                    "uniforms": {},
                },
            )
            entry["uniforms"][pair["uniform"]] = pair["sampler"]
        return {
            name: entry
            for name, entry in textures.items()
            if name not in rejected and name in entry["uniforms"]
        }

    @staticmethod
    def _is_runtime_array(codegen, type_node):
        if not isinstance(type_node, ArrayType) or type_node.size is not None:
            return False
        element = type_node.element_type
        if isinstance(element, ArrayType):
            return False
        return codegen.is_inferable_resource_array_type(
            codegen.convert_type_node_to_string(element)
        )

    @staticmethod
    def _check_uses(codegen, ast, textures, arrays):
        """Keep the textures read only by sampling calls through their pair's sampler, and the indexed arrays."""
        uniforms = {
            uniform: (texture, sampler)
            for texture, entry in textures.items()
            for uniform, sampler in entry["uniforms"].items()
        }
        allowed = set()
        nodes = list(codegen.walk_ast(ast))
        for node in nodes:
            if isinstance(node, ArrayAccessNode) and isinstance(
                node.array_expr, IdentifierNode
            ):
                if node.array_expr.name in arrays:
                    allowed.add(id(node.array_expr))
            if not isinstance(node, FunctionCallNode):
                continue
            name = codegen.function_call_name(node)
            args = list(getattr(node, "arguments", None) or [])
            if name not in TEXTURE_SAMPLING_INTRINSIC_NAMES or len(args) < 3:
                continue
            texture, sampler = args[0], args[1]
            if (
                isinstance(texture, IdentifierNode)
                and texture.name in uniforms
                and isinstance(sampler, IdentifierNode)
                and sampler.name == uniforms[texture.name][1]
            ):
                allowed.add(id(texture))
                allowed.add(id(sampler))
        rejected = set()
        for node in nodes:
            if not isinstance(node, IdentifierNode) or id(node) in allowed:
                continue
            if node.name in uniforms:
                rejected.add(uniforms[node.name][0])
            elif node.name in arrays:
                rejected.add(node.name)
        return (
            {name: entry for name, entry in textures.items() if name not in rejected},
            {name for name in arrays if name not in rejected},
        )

    # Emission ---------------------------------------------------------------------------------------------------

    def activate(self, codegen, version_line):
        """Whether this rendering writes the Vulkan forms: desktop GLSL with ``layout(binding)``."""
        self.active = bool(
            (self.textures or self.arrays)
            and getattr(
                codegen, "current_glsl_resource_binding_layouts_supported", False
            )
            and not codegen.glsl_version_is_es(version_line)
        )
        self.macro_used = False
        self.emitted_textures = {}
        self.emitted_primaries = {}
        self.emitted_arrays = {}
        self.fragments = []
        return self.active

    def declaration(
        self, source_name, glsl_name, qualifiers, declaration, mapped_type, suffix
    ):
        """The declaration of an opaque resource uniform, in each API's form when it has two."""
        line = f"{qualifiers} {declaration};\n"
        if not self.active:
            return line
        texture = self._texture_of(source_name)
        if texture is not None:
            uniforms = self.emitted_textures.setdefault(texture, set())
            uniforms.add(glsl_name)
            if source_name != texture:
                text = f"#ifndef VULKAN\n{line}#endif\n"
            else:
                self.emitted_primaries[texture] = glsl_name
                vulkan = format_c_style_array_declaration(
                    f"{self.textures[texture]['type']}{suffix}", glsl_name
                )
                text = f"{_GUARD}{qualifiers} {vulkan};\n#else\n{line}#endif\n"
            self.fragments.append(text)
            return text
        if source_name in self.arrays and _SINGLE_EXTENT.match(suffix or ""):
            self.emitted_arrays[source_name] = glsl_name
            vulkan = format_c_style_array_declaration(f"{mapped_type}[]", glsl_name)
            text = f"{_GUARD}{qualifiers} {vulkan};\n#else\n{line}#endif\n"
            self.fragments.append(text)
            return text
        return line

    def _texture_of(self, uniform):
        for texture, entry in self.textures.items():
            if uniform in entry["uniforms"]:
                return texture
        return None

    def sampler_declarations(self, codegen):
        """The ``#ifdef VULKAN`` declarations of the samplers that the separate textures are read through."""
        if not self.active or not self.emitted_primaries:
            return ""
        used = set()
        for record in getattr(codegen, "glsl_resource_binding_records", None) or []:
            count = max(int(record.get("count") or 1), 1)
            used.update(range(int(record["binding"]), int(record["binding"]) + count))
        explicit = {}
        for name, node in self.samplers.items():
            choices = codegen.explicit_resource_binding_choices(
                node, flatten_sets=False
            )
            bindings = [
                binding
                for attribute, _source, binding in choices
                if attribute in {"binding", "sampler"}
            ]
            if bindings:
                descriptor_set = codegen.resource_descriptor_set_index(node)
                explicit[name] = (descriptor_set, bindings[0])
                if not descriptor_set:
                    used.add(bindings[0])
        emitted = {
            sampler
            for texture in self.emitted_primaries
            for sampler in self.textures[texture]["uniforms"].values()
        }
        lines = []
        for name in self.samplers:
            if name not in emitted:
                continue
            if name in explicit:
                descriptor_set, binding = explicit[name]
            else:
                descriptor_set, binding = 0, 0
                while binding in used:
                    binding += 1
                used.add(binding)
            placement = f"set = {descriptor_set}, " if descriptor_set else ""
            glsl_name = codegen.glsl_global_identifier_name(name)
            lines.append(
                f"layout({placement}binding = {binding}) uniform sampler {glsl_name};\n"
            )
        return f"{_GUARD}{''.join(lines)}#endif\n"

    def sampled(self, args, texture_operand, sampler_operand):
        """The texture operand of a sampling call through an explicit sampler, in each API's form when it has two."""
        if not self.active or not args or not isinstance(args[0], IdentifierNode):
            return texture_operand
        texture = self._texture_of(args[0].name)
        texture_name = self.emitted_primaries.get(texture)
        if texture_name is None:
            return texture_operand
        sampled = (
            f"{SAMPLED_OPERAND_MACRO}({self.textures[texture]['sampled']}"
            f"({texture_name}, {sampler_operand}), {texture_operand})"
        )
        self.macro_used = True
        self.fragments.append(sampled)
        return sampled

    def with_prelude(
        self, code, index, descriptor_binding_macro=False, location_macro=False
    ):
        """``code`` with :meth:`prelude` inserted at ``index``, the end of its ``#extension`` lines.

        The varying location macro of :mod:`.glsl_spirv_varying_locations` precedes it when ``location_macro`` is
        true, and the descriptor binding macro of :mod:`.glsl_vulkan_descriptor_sets` follows it when
        ``descriptor_binding_macro`` is true, since all three are definitions of the SPIR-V branch.
        """
        text = spirv_location_macro_definition() if location_macro else ""
        text += self.prelude(code)
        if descriptor_binding_macro:
            text += descriptor_binding_macro_definition()
        return code[:index] + text + code[index:] if text else code

    def prelude(self, code):
        """The macro definitions and guarded extension the rendered forms need, for after the ``#extension`` lines."""
        if not self.active:
            return ""
        text = ""
        if self.macro_used:
            macro = f"{SAMPLED_OPERAND_MACRO}(vulkan, opengl)"
            text += f"{_GUARD}#define {macro} vulkan\n#else\n#define {macro} opengl\n#endif\n"
        if self.emitted_arrays and not re.search(
            rf"^\s*#\s*extension\s+{NONUNIFORM_EXTENSION}\b", code, re.M
        ):
            text += f"{_GUARD}#extension {NONUNIFORM_EXTENSION} : require\n#endif\n"
        return text

    def unwrapped(self, code):
        """The textures and arrays the rendered ``code`` uses outside the forms this module wrote for them."""
        if not self.active or not (self.emitted_textures or self.emitted_arrays):
            return set()
        text = _COMMENT.sub(" ", code)
        for fragment in sorted(set(self.fragments), key=len, reverse=True):
            text = text.replace(fragment, " ")
        failed = set()
        for texture, names in sorted(self.emitted_textures.items()):
            pattern = "|".join(re.escape(name) for name in sorted(names))
            if re.search(rf"\b(?:{pattern})\b", text):
                failed.add(texture)
        for array, name in sorted(self.emitted_arrays.items()):
            if re.search(rf"\b{re.escape(name)}\b(?!\s*\[)", text):
                failed.add(array)
        return failed

"""One descriptor type per set 0 binding in GLSL output that is compiled for Vulkan.

OpenGL keeps a separate array of binding points for each resource class: uniform blocks, storage blocks, texture
image units and image units (OpenGL 4.6 sections 7.6.3, 7.8, 8.1 and 8.26). GLSL output numbers each class from 0,
so a uniform block, a storage block and a texture can all have ``binding = 0``. A Vulkan compilation of the same
source (glslang ``-V`` defines ``VULKAN``) puts all of them in one descriptor set, where a binding number names one
``VkDescriptorSetLayoutBinding`` with one descriptor type (Vulkan 1.3 section 14.2.1; GL_KHR_vulkan_glsl,
"Descriptor Sets"). glslang and spirv-val accept such a module, but no pipeline layout can bind both resources: a
host binds one of them and the shader reads it through the other.

After rendering, :func:`with_distinct_vulkan_bindings` reads the set 0 binding of every declaration that a Vulkan
compilation sees and moves each binding that several descriptor types share. The declarations of one type keep the
number, chosen in this order: a type whose binding the source wrote (a ``binding``, ``register`` or ``@binding``
the generator kept), then storage blocks, so that a compute kernel's buffers stay where the OpenGL host binds them,
then the first declaration. Every other type takes the lowest binding that no set 0 declaration uses, and
declarations of one type that shared a number move together, since OpenGL reads them as views of one resource. A
declaration that both APIs compile takes ``CROSSGL_BINDING(0, vulkan, opengl)`` (:mod:`.glsl_vulkan_descriptor_sets`),
so the OpenGL compilation keeps its binding point; one that only the Vulkan compilation sees, inside
``#ifdef VULKAN``, takes the new binding directly.

.. code-block:: glsl

    layout(std430, binding = 0) buffer finBuffer { float fin[]; };
    layout(std430, binding = 1) buffer outpBuffer { float outp[]; };
    layout(std140, CROSSGL_BINDING(0, 2, 0)) uniform Params { float scale; };

A binding that only one descriptor type uses is unchanged, so a Vulkan source whose bindings were already distinct
keeps them in both compilations.
"""

from __future__ import annotations

import re

from .glsl_vulkan_descriptor_sets import (
    DESCRIPTOR_BINDING_MACRO,
    descriptor_binding_macro_definition,
    descriptor_binding_qualifier,
)

__all__ = ["vulkan_descriptor_type", "with_distinct_vulkan_bindings"]

STORAGE_BUFFER = "storage-buffer"
UNIFORM_BUFFER = "uniform-buffer"

# The OpenGL binding table of each descriptor type, as resource binding records name it
# (``GLSL_BINDING_RECORD_NAMESPACES`` in :mod:`.glsl_hlsl_resources`).
_OPENGL_TABLE = {
    STORAGE_BUFFER: "shader-storage-buffer",
    UNIFORM_BUFFER: "uniform-buffer",
    "combined-image-sampler": "texture",
    "sampled-image": "texture",
    "uniform-texel-buffer": "texture",
    "storage-image": "image",
    "storage-texel-buffer": "image",
    "sampler": "sampler",
}

# Opaque uniform types by descriptor type (GL_KHR_vulkan_glsl; GLSL 4.60 section 4.1.7). Order matters: the
# buffer forms precede the general sampler, texture and image patterns.
_OPAQUE_TYPES = (
    (re.compile(r"^sampler(?:Shadow)?$"), "sampler"),
    (re.compile(r"^[iu]?samplerBuffer$"), "uniform-texel-buffer"),
    (re.compile(r"^[iu]?textureBuffer$"), "uniform-texel-buffer"),
    (re.compile(r"^(?:[iu]|[iu]64)?imageBuffer$"), "storage-texel-buffer"),
    (re.compile(r"^[iu]?sampler\w+$"), "combined-image-sampler"),
    (re.compile(r"^[iu]?texture\w+$"), "sampled-image"),
    (re.compile(r"^(?:[iu]|[iu]64)?image\w+$"), "storage-image"),
    (re.compile(r"^accelerationStructure(?:EXT|NV)$"), "acceleration-structure"),
    (re.compile(r"^[iu]?subpassInput(?:MS)?$"), "input-attachment"),
)

# Words that may stand between ``uniform`` and the type of an opaque uniform or the name of a block.
_QUALIFIERS = frozenset(
    {
        "coherent",
        "volatile",
        "restrict",
        "readonly",
        "writeonly",
        "highp",
        "mediump",
        "lowp",
        "nonuniformEXT",
        "devicecoherent",
        "queuefamilycoherent",
        "workgroupcoherent",
        "subgroupcoherent",
        "nonprivate",
    }
)

_DIRECTIVE = re.compile(r"^\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)$")
_LAYOUT_START = re.compile(r"^\s*layout\s*\(")
_LAYOUT = re.compile(r"\s*layout\s*\(")
_TOKEN = re.compile(r"[A-Za-z_]\w*|[{;\[]")
_BINDING_ITEM = re.compile(r"^binding\s*=\s*(\d+)$")
_SET_ITEM = re.compile(r"^set\s*=\s*(\d+)$")
_MACRO_ITEM = re.compile(
    rf"^{DESCRIPTOR_BINDING_MACRO}\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$"
)
_MACRO_DEFINED = re.compile(
    rf"^\s*#\s*define\s+{DESCRIPTOR_BINDING_MACRO}\b", re.MULTILINE
)
_VULKAN, _OPENGL, _EITHER = "vulkan", "opengl", "either"


def vulkan_descriptor_type(declaration):
    """The Vulkan descriptor type of the resource that ``declaration`` declares after its layout qualifiers.

    ``declaration`` is the text after ``layout(...)``, as in ``uniform sampler2D tex;``. The result is ``None``
    for a declaration that declares no descriptor, such as an atomic counter or a stage input.
    """
    tokens = _TOKEN.findall(declaration)
    if "buffer" in tokens:
        return STORAGE_BUFFER
    if "uniform" not in tokens:
        return None
    rest = [
        token
        for token in tokens[tokens.index("uniform") + 1 :]
        if token not in _QUALIFIERS
    ]
    if not rest or rest[0] == "{":
        return None
    following = rest[1] if len(rest) > 1 else "{"
    if following in {"{", ";", "["}:
        # ``uniform Params {`` or ``uniform Params`` with its brace on the next line.
        return UNIFORM_BUFFER if following != ";" else None
    for pattern, descriptor_type in _OPAQUE_TYPES:
        if pattern.match(rest[0]):
            return descriptor_type
    return None


def _condition(keyword, argument):
    argument = argument.strip()
    if keyword in {"ifdef", "ifndef"} and argument == "VULKAN":
        return _VULKAN if keyword == "ifdef" else _OPENGL
    if keyword == "if":
        if re.fullmatch(r"defined\s*\(?\s*VULKAN\s*\)?", argument):
            return _VULKAN
        if re.fullmatch(r"!\s*defined\s*\(?\s*VULKAN\s*\)?", argument):
            return _OPENGL
    return _EITHER


def _closing_parenthesis(line, start):
    depth = 0
    for index in range(start, len(line)):
        if line[index] == "(":
            depth += 1
        elif line[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _layout_items(line):
    """``(items, declaration)``: each top-level layout item with its span in ``line``, and the text after them."""
    items, position = [], 0
    while True:
        match = _LAYOUT.match(line, position)
        if match is None:
            return items, line[position:]
        close = _closing_parenthesis(line, match.end() - 1)
        if close is None:
            return None, ""
        depth, start = 0, match.end()
        for index in range(match.end(), close + 1):
            char = line[index]
            if char == "(":
                depth += 1
            elif char == ")" and depth:
                depth -= 1
            elif (char == "," and not depth) or index == close:
                text = line[start:index]
                stripped = text.strip()
                if stripped:
                    offset = start + text.index(stripped)
                    items.append((stripped, offset, offset + len(stripped)))
                start = index + 1
        position = close + 1


def _declaration(lines, index, visibility, explicit):
    """The entry of the resource declaration that starts at ``lines[index]``, or ``None``."""
    items, declaration = _layout_items(lines[index])
    if not items:
        return None
    # A formatter may break a declaration after ``uniform``; its type or block brace is on a later line.
    for following in lines[index + 1 : index + 4]:
        if re.search(r"[{;]", declaration):
            break
        declaration += following
    descriptor_set, binding, item, opengl_binding = 0, None, None, None
    for text, start, end in items:
        if text == "push_constant":
            return None
        if match := _SET_ITEM.match(text):
            descriptor_set = int(match.group(1))
        elif match := _BINDING_ITEM.match(text):
            binding = opengl_binding = int(match.group(1))
            item = (start, end)
        elif match := _MACRO_ITEM.match(text):
            descriptor_set, binding = int(match.group(1)), int(match.group(2))
            opengl_binding = int(match.group(3))
            item = (start, end)
    if binding is None:
        return None
    descriptor_type = vulkan_descriptor_type(declaration)
    table = _OPENGL_TABLE.get(descriptor_type)
    return {
        "line": index,
        "set": descriptor_set,
        "binding": binding,
        "type": descriptor_type,
        "item": item,
        "opengl": opengl_binding,
        "vulkan_only": visibility == _VULKAN,
        "explicit": (table, opengl_binding) in explicit,
    }


def _entries(lines, explicit):
    stack, entries = [], []
    for index, line in enumerate(lines):
        directive = _DIRECTIVE.match(line)
        if directive:
            keyword, argument = directive.groups()
            if keyword == "endif":
                if stack:
                    stack.pop()
            elif keyword == "else":
                if stack:
                    stack[-1] = {_VULKAN: _OPENGL, _OPENGL: _VULKAN}.get(
                        stack[-1], _EITHER
                    )
            elif keyword == "elif":
                if stack:
                    stack[-1] = _EITHER
            else:
                stack.append(_condition(keyword, argument))
            continue
        if _OPENGL in stack or not _LAYOUT_START.match(line):
            continue
        visibility = _VULKAN if _VULKAN in stack else _EITHER
        entry = _declaration(lines, index, visibility, explicit)
        if entry is not None and entry["set"] == 0:
            entries.append(entry)
    return entries


def _moves(entries):
    """``{(binding, descriptor type): new binding}`` for each descriptor type that leaves a shared binding."""
    by_binding = {}
    for entry in entries:
        if entry["type"] is not None:
            by_binding.setdefault(entry["binding"], []).append(entry)
    used = {entry["binding"] for entry in entries}
    leaving = []
    for binding in sorted(by_binding):
        group = by_binding[binding]
        if len({entry["type"] for entry in group}) < 2:
            continue
        keeper = min(
            group,
            key=lambda entry: (
                not entry["explicit"],
                entry["type"] != STORAGE_BUFFER,
                entry["line"],
            ),
        )["type"]
        for entry in group:
            key = (binding, entry["type"])
            if entry["type"] != keeper and key not in leaving:
                leaving.append(key)
    leaving.sort(
        key=lambda key: min(
            entry["line"] for entry in by_binding[key[0]] if entry["type"] == key[1]
        )
    )
    moves, candidate = {}, 0
    for key in leaving:
        while candidate in used:
            candidate += 1
        moves[key] = candidate
        used.add(candidate)
    return moves


def _macro_insertion_index(lines):
    """The line before which a definition reaches every declaration: the first one outside the preprocessor."""
    depth, block_start = 0, 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        directive = _DIRECTIVE.match(line)
        if directive:
            keyword = directive.group(1)
            if keyword in {"ifdef", "ifndef", "if"}:
                if depth == 0:
                    block_start = index
                depth += 1
            elif keyword == "endif" and depth:
                depth -= 1
            continue
        if not stripped or stripped.startswith(("#", "//")):
            continue
        return index if depth == 0 else block_start
    return len(lines)


def with_distinct_vulkan_bindings(code, explicit=frozenset()):
    """``code`` with each set 0 binding that several descriptor types share in a Vulkan compilation moved.

    ``explicit`` holds the ``(OpenGL binding table, binding)`` pairs whose binding the source wrote, with the
    table names of resource binding records.
    """
    lines = code.splitlines(keepends=True)
    entries = _entries(lines, explicit)
    moves = _moves(entries)
    if not moves:
        return code
    macro_used = False
    for entry in entries:
        new = moves.get((entry["binding"], entry["type"]))
        if new is None:
            continue
        if entry["vulkan_only"]:
            replacement = f"binding = {new}"
        else:
            replacement = descriptor_binding_qualifier(0, new, entry["opengl"])
            macro_used = True
        start, end = entry["item"]
        line = lines[entry["line"]]
        lines[entry["line"]] = line[:start] + replacement + line[end:]
    if macro_used and not _MACRO_DEFINED.search(code):
        index = _macro_insertion_index(lines)
        lines.insert(index, descriptor_binding_macro_definition())
    return "".join(lines)

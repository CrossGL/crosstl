"""GL_KHR_memory_scope_semantics memory qualifiers on CrossGL resources.

GL_KHR_memory_scope_semantics adds coherence at a named scope to GLSL storage buffers and images:
``devicecoherent``, ``queuefamilycoherent``, ``workgroupcoherent``, ``subgroupcoherent`` and
``shadercallcoherent``, and ``nonprivate``, which lets barriers make an access available and visible at their
own scope. CrossGL carries them as bare attributes (``RWStructuredBuffer<uint> b @workgroupcoherent;``) or, on a
GLSL-style storage block, as block qualifiers (``layout(std430) workgroupcoherent buffer B { ... } b;``).
``nontemporal`` (GL_EXT_nontemporal_keyword) travels with them; it is a cache hint and changes no result.

Only GLSL under ``#pragma use_vulkan_memory_model`` can state these scopes; glslang ignores the qualifiers
without that pragma. Every other target gets coherence across the device. Device scope contains each scope the
extension names (a queue family, a workgroup, a subgroup and a shader-call tree all run on one device), and a
coherent access is also non-private, so device coherence keeps every guarantee these qualifiers give:

- HLSL and Slang: ``globallycoherent``.
- SPIR-V under the GLSL450 memory model: the ``Coherent`` decoration.
- GLSL without the Vulkan memory model: ``coherent`` (GLSL 4.60 section 4.10).
- MSL 3.2 buffers: ``coherent(threadgroup)`` for workgroup and subgroup scope, ``coherent(device)`` otherwise.
"""

from __future__ import annotations

__all__ = [
    "MEMORY_SCOPE_EXTENSION",
    "NONTEMPORAL_QUALIFIER",
    "SCOPED_COHERENCE_QUALIFIERS",
    "VULKAN_MEMORY_SCOPE_QUALIFIERS",
    "declares_vulkan_memory_model",
    "glsl_memory_scope_qualifiers",
    "metal_coherence_scope",
    "needs_device_coherence",
    "vulkan_memory_scope_names",
]

MEMORY_SCOPE_EXTENSION = "GL_KHR_memory_scope_semantics"

# Each scoped coherence qualifier and the SPIR-V scope it names.
SCOPED_COHERENCE_QUALIFIERS = {
    "devicecoherent": "device",
    "queuefamilycoherent": "queuefamily",
    "workgroupcoherent": "workgroup",
    "subgroupcoherent": "subgroup",
    "shadercallcoherent": "shadercall",
}
NONPRIVATE_QUALIFIER = "nonprivate"
NONTEMPORAL_QUALIFIER = "nontemporal"

# The qualifiers that ask for coherence, in the order GLSL output writes them.
_COHERENCE_ORDER = (*SCOPED_COHERENCE_QUALIFIERS, NONPRIVATE_QUALIFIER)
VULKAN_MEMORY_SCOPE_QUALIFIERS = frozenset((*_COHERENCE_ORDER, NONTEMPORAL_QUALIFIER))

# Scopes an MSL threadgroup contains; the rest need device coherence.
_THREADGROUP_SCOPES = frozenset({"workgroup", "subgroup"})


def vulkan_memory_scope_names(node):
    """The coherence-bearing GL_KHR_memory_scope_semantics qualifiers of ``node``, in GLSL order.

    Block qualifiers and bare attributes both count. ``nontemporal`` is left out: it asks for no coherence.
    """
    names = {str(qualifier).strip().lower() for qualifier in _qualifiers(node)}
    for attribute in getattr(node, "attributes", None) or []:
        name = getattr(attribute, "name", None)
        if name and not (getattr(attribute, "arguments", None) or []):
            names.add(str(name).lower())
    return tuple(name for name in _COHERENCE_ORDER if name in names)


def _qualifiers(node):
    values = list(getattr(node, "qualifiers", None) or [])
    values.extend(getattr(node, "resource_qualifiers", None) or [])
    return [value for value in values if isinstance(value, str)]


def needs_device_coherence(node):
    """Whether a target without these qualifiers has to make ``node`` coherent across the device."""
    return bool(vulkan_memory_scope_names(node))


def metal_coherence_scope(node):
    """The MSL 3.2 ``coherent(...)`` scope ``node``'s qualifiers need, or ``None`` when they need none."""
    names = vulkan_memory_scope_names(node)
    if not names:
        return None
    if all(
        SCOPED_COHERENCE_QUALIFIERS.get(name) in _THREADGROUP_SCOPES for name in names
    ):
        return "threadgroup"
    return "device"


def declares_vulkan_memory_model(directives, extensions):
    """Whether GLSL output keeps ``#pragma use_vulkan_memory_model`` and enables the memory-scope extension.

    ``directives`` are the program's ``PreprocessorNode`` objects and ``extensions`` the names its
    ``#extension`` lines enable. Only then do the scoped qualifiers mean in GLSL what they mean in CrossGL.
    """
    if MEMORY_SCOPE_EXTENSION not in extensions:
        return False
    for directive in directives or []:
        if str(getattr(directive, "directive", "")).lstrip("#") != "pragma":
            continue
        content = str(getattr(directive, "content", "") or "").split()
        if content == ["use_vulkan_memory_model"]:
            return True
    return False


def glsl_memory_scope_qualifiers(node, vulkan_memory_model):
    """The GLSL qualifiers that keep ``node``'s coherence: the scoped names under the Vulkan memory model.

    Without ``#pragma use_vulkan_memory_model`` glslang accepts the scoped qualifiers but gives them no meaning,
    so the output asks for ``coherent`` instead.
    """
    names = vulkan_memory_scope_names(node)
    if not names:
        return ()
    return names if vulkan_memory_model else ("coherent",)

"""Vulkan descriptor slots for resources bound by HLSL ``register`` and ``vk::binding``.

The HLSL importer keeps a resource's bindings as ``@ register(t3, space1)`` and ``@ vk_binding(5, 2)``:

* ``vk_binding(binding[, set])`` is DXC's ``[[vk::binding(X, Y)]]``: it names the exact descriptor slot. A missing
  set is the register space, or 0 (the DXC SPIR-V code generation documentation, "Descriptor sets and bindings").
* ``register(Xn[, spaceM])`` names register ``n`` of class ``X`` in space ``M``. The four classes (``b`` constant
  buffers, ``t`` shader resources, ``u`` unordered-access views, ``s`` samplers) are separate namespaces in Direct3D,
  while a Vulkan descriptor set has one, so ``b0`` and ``t0`` cannot both keep binding 0. Space ``M`` becomes
  descriptor set ``M``. Each resource keeps binding ``n`` when that binding of its set is free and otherwise takes
  the next free one. Resources are visited in class order (b, t, u, s), then register number, then declaration
  order, so the mapping does not depend on declaration order; the GLSL and Metal emitters use the same rule for
  their binding tables. An array of ``k`` resources holds registers ``n`` to ``n + k - 1`` and one descriptor
  binding with ``k`` descriptors.

Resources indexed by an MSL ``@ texture(n)`` or ``@ sampler(n)`` are placed after the registers, by the same rule
(:mod:`.spirv_argument_tables`): textures, then samplers, each keeping binding ``n`` when it is free.

Exact bindings (``@ binding``, ``@ buffer``, ``vk_binding``) are reserved before any register is placed. Two
resources of one class and space that claim the same register are rejected, as Direct3D rejects them. Two
``vk_binding`` resources may share one slot only as the halves of a combined image sampler: one sampled image and
one sampler. Vulkan lets separate ``OpTypeImage`` (Sampled 1) and ``OpTypeSampler`` variables with the same set and
binding consume one ``VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER`` descriptor (the Vulkan specification, "Shader
Resource and Descriptor Type Correspondence"), and DXC emits that layout for this source.
"""

from __future__ import annotations

from .glsl_hlsl_resources import HlslRegisterOverlapError
from .spirv_argument_tables import argument_table_index, place_argument_table_slots

__all__ = [
    "HLSL_REGISTER_CLASS_ORDER",
    "reserve_exact_binding",
    "reserve_hlsl_descriptor_slots",
]

HLSL_REGISTER_CLASS_ORDER = {"b": 0, "t": 1, "u": 2, "s": 3}
EXACT_BINDING_ATTRIBUTES = {"binding", "buffer", "texture", "sampler"}
GLOBAL_ARGUMENT_TABLES = {"texture": 1, "sampler": 2}


def _arguments(attr):
    arguments = getattr(attr, "arguments", None)
    if arguments is None:
        arguments = getattr(attr, "args", None)
    return list(arguments or [])


def _text(codegen, value):
    text = codegen.attribute_value_to_string(value)
    return None if text is None else str(text).strip().lower()


def _space(codegen, value):
    text = _text(codegen, value)
    if text is None or not text.startswith("space") or not text[5:].isdigit():
        raise ValueError(f"Invalid HLSL register space '{text}'")
    return int(text[5:])


def _register(codegen, node):
    """``(class, index, space)`` of a node's ``register`` attribute, or ``None``."""
    selected = None
    for attr in getattr(node, "attributes", None) or []:
        if str(getattr(attr, "name", "") or "").lower() != "register":
            continue
        arguments = _arguments(attr)
        if not arguments or len(arguments) > 2:
            continue
        text = _text(codegen, arguments[0])
        if not text:
            continue
        register_class, digits = text[:1], text[1:]
        if register_class not in HLSL_REGISTER_CLASS_ORDER or not digits.isdigit():
            continue
        space = _space(codegen, arguments[1]) if len(arguments) == 2 else 0
        selected = (register_class, int(digits), space)
    return selected


def _vk_binding(codegen, node, default_set):
    """``(set, binding)`` of a node's ``vk_binding`` attribute, or ``None``."""
    for attr in getattr(node, "attributes", None) or []:
        if str(getattr(attr, "name", "") or "").lower() != "vk_binding":
            continue
        arguments = _arguments(attr)
        if not arguments or len(arguments) > 2:
            raise ValueError("vk::binding takes a binding and an optional set")
        values = [codegen.literal_int_argument(argument) for argument in arguments]
        if any(value is None or value < 0 for value in values):
            raise ValueError("vk::binding needs non-negative integer arguments")
        binding = values[0]
        descriptor_set = values[1] if len(values) == 2 else default_set
        return descriptor_set, binding
    return None


def _bound_exactly(codegen, node):
    return any(
        str(getattr(attr, "name", "") or "").lower() in {"binding", "buffer"}
        and _arguments(attr)
        for attr in getattr(node, "attributes", None) or []
    )


def _has_exact_binding(codegen, node):
    return any(
        str(getattr(attr, "name", "") or "").lower() in EXACT_BINDING_ATTRIBUTES
        and _arguments(attr)
        for attr in getattr(node, "attributes", None) or []
    ) or any(
        codegen.explicit_interface_integer_attribute(node, name) is not None
        for name in ("set", "group", "space")
    )


def _register_count(codegen, node):
    """The number of registers a resource holds: its array size, or 1."""
    size = getattr(node, "array_size", None)
    if size is None:
        var_type = getattr(node, "var_type", getattr(node, "vtype", None))
        size = getattr(var_type, "size", None)
    if size is None:
        return 1
    count = codegen.literal_int_argument(size)
    return count if count is not None and count > 0 else 1


SEPARATE_SAMPLER_TYPES = {"sampler", "samplershadow"}
NON_COMBINABLE_IMAGE_SUFFIXES = ("buffer", "ms", "msarray")


def _combined_sampler_half(node):
    """``"sampler"`` or ``"image"`` when ``node`` can be one half of a combined image sampler, else ``None``."""
    var_type = getattr(node, "var_type", getattr(node, "vtype", None))
    while getattr(var_type, "element_type", None) is not None:
        var_type = var_type.element_type
    name = str(getattr(var_type, "name", var_type) or "").strip().lower()
    if name in SEPARATE_SAMPLER_TYPES:
        return "sampler"
    if name.startswith(("sampler", "texture")) and not name.endswith(
        NON_COMBINABLE_IMAGE_SUFFIXES
    ):
        return "image"
    return None


def reserve_exact_binding(codegen, key, node, name):
    """Reserve the exact ``(set, binding)`` slot ``key`` for ``node``.

    A second claimant is accepted only as the other half of a combined image sampler; both claimants are then
    recorded in ``codegen.shared_binding_nodes`` so the slot can be handed out to each of them.
    """
    occupants = codegen.binding_occupants
    shared = codegen.shared_binding_nodes
    half = _combined_sampler_half(node)
    if key in codegen.reserved_resource_bindings:
        claim = occupants.get(key)
        halves = [] if claim is None else [occupant_half for occupant_half, _ in claim]
        if half is None or not halves or None in halves or half in halves:
            raise ValueError(
                f"Duplicate SPIR-V resource binding set {key[0]} binding {key[1]} ({name})"
            )
        claim.append((half, id(node)))
        shared.update(node_id for _, node_id in claim)
        return
    codegen.reserved_resource_bindings.add(key)
    occupants[key] = [(half, id(node))]


def _next_free_binding(codegen, descriptor_set, start):
    binding = start
    while (descriptor_set, binding) in codegen.reserved_resource_bindings:
        binding += 1
    return binding


def _claim_registers(claimed, register, count, name):
    register_class, index, space = register
    for number in range(index, index + count):
        key = (register_class, number, space)
        if key in claimed:
            raise HlslRegisterOverlapError(
                f"HLSL resources '{claimed[key]}' and '{name}' both claim register "
                f"{register_class}{number} in space{space}"
            )
        claimed[key] = name


def reserve_hlsl_descriptor_slots(
    codegen, nodes, skip=lambda node: False, scope=lambda node: None
):
    """Assign the descriptor slots of resources bound by ``vk_binding``, ``register``, ``texture`` or ``sampler``.

    ``nodes`` are the program's descriptor-bound declarations in declaration order; ``skip(node)`` excludes those
    that take no descriptor (push-constant blocks), and ``scope(node)`` names the stage that declares a node (each
    stage has its own MSL argument tables). Returns ``{id(node): (set, binding)}``; the bindings are added to
    ``codegen.reserved_resource_bindings``.
    """
    slots = {}
    deferred = []
    indexed = []
    seen = set()
    for order, node in enumerate(nodes):
        if id(node) in seen or skip(node):
            continue
        table_index = argument_table_index(codegen, node, GLOBAL_ARGUMENT_TABLES)
        if table_index is not None and not _bound_exactly(codegen, node):
            seen.add(id(node))
            indexed.append((*table_index, order, scope(node), node))
            continue
        if _has_exact_binding(codegen, node):
            continue
        seen.add(id(node))
        register = _register(codegen, node)
        exact = _vk_binding(codegen, node, register[2] if register else 0)
        name = getattr(node, "name", "resource")
        if exact is not None:
            reserve_exact_binding(codegen, exact, node, name)
            slots[id(node)] = exact
        elif register is not None:
            deferred.append((register, order, node, name))
    claimed = {}
    for register, _order, node, name in sorted(
        deferred,
        key=lambda entry: (
            HLSL_REGISTER_CLASS_ORDER[entry[0][0]],
            entry[0][1],
            entry[0][2],
            entry[1],
        ),
    ):
        _claim_registers(claimed, register, _register_count(codegen, node), name)
        descriptor_set = register[2]
        binding = _next_free_binding(codegen, descriptor_set, register[1])
        codegen.reserved_resource_bindings.add((descriptor_set, binding))
        slots[id(node)] = (descriptor_set, binding)
    slots.update(
        place_argument_table_slots(
            codegen,
            indexed,
            lambda slot: slot in codegen.reserved_resource_bindings,
        )
    )
    return slots

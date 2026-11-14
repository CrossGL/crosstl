"""Vulkan descriptor slots for resources bound by MSL argument-table indices.

MSL binds a function's resources through three separate argument tables: ``[[buffer(n)]]``, ``[[texture(n)]]``
and ``[[sampler(n)]]`` (the Metal Shading Language Specification, "Attributes to Locate Buffer, Texture, and
Sampler"), and each entry point has its own tables. The Metal importer keeps the indices as ``@ buffer(n)``,
``@ texture(n)`` and ``@ sampler(n)``. A Vulkan descriptor set has one binding namespace, so ``buffer(0)`` and
``texture(0)`` cannot both keep binding 0.

The slots follow the rule :mod:`.spirv_hlsl_registers` applies to HLSL register classes. Resources are visited in
table order (buffers, textures, samplers), then index, then declaration order, and each keeps binding ``n`` of its
set when that binding is free and otherwise takes the next free one, so the mapping does not depend on declaration
order. An array of ``k`` resources holds indices ``n`` to ``n + k - 1`` and one descriptor binding.

Two resources of one table and scope that claim the same index are refused with
:class:`ArgumentTableOverlapError`, as the Metal compiler refuses them ("cannot reserve 'texture' resource location
at index 0"). Resources bound exactly (``@ binding``, ``@ vk_binding``) or by an HLSL ``register`` keep their own
allocation.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "ARGUMENT_TABLE_ORDER",
    "ArgumentTableOverlapError",
    "argument_table_index",
    "plan_entry_descriptor_slots",
    "place_argument_table_slots",
]

ARGUMENT_TABLE_ORDER = {"buffer": 0, "texture": 1, "sampler": 2}
_OWN_ALLOCATION_ATTRIBUTES = {"binding", "vk_binding", "register"}


class ArgumentTableOverlapError(TranslationDiagnostic):
    """Two resources of one MSL argument table claim the same index."""

    code = CODE_PREFIX + "argument-table-overlap"
    missing_capabilities = ("spirv.argument-table-allocation",)

    def __init__(self, reason):
        super().__init__(reason, target="spirv")


def _arguments(attr):
    arguments = getattr(attr, "arguments", None)
    if arguments is None:
        arguments = getattr(attr, "args", None)
    return list(arguments or [])


def _node_type(node):
    return getattr(
        node, "param_type", getattr(node, "var_type", getattr(node, "vtype", None))
    )


def _index_count(codegen, node):
    """The number of indices a resource holds: its array size, or 1."""
    size = getattr(node, "array_size", None)
    if size is None:
        size = getattr(_node_type(node), "size", None)
    if size is None:
        return 1
    count = codegen.literal_int_argument(size)
    return count if count is not None and count > 0 else 1


def argument_table_index(codegen, node, tables=ARGUMENT_TABLE_ORDER):
    """``(table, index)`` of a node's ``buffer``, ``texture`` or ``sampler`` index, or ``None``.

    ``None`` also when the node is bound exactly or by an HLSL ``register``, or when the index is not an integer
    literal (the binding-constant pass has already folded every index it can resolve).
    """
    selected = None
    for attr in getattr(node, "attributes", None) or []:
        name = str(getattr(attr, "name", "") or "").lower()
        if name in _OWN_ALLOCATION_ATTRIBUTES:
            return None
        if name not in tables:
            continue
        arguments = _arguments(attr)
        if len(arguments) != 1:
            continue
        index = codegen.literal_int_argument(arguments[0])
        if index is None or index < 0:
            continue
        selected = (name, index)
    return selected


def _claim(claimed, key, count, name):
    table, index, descriptor_set, scope = key
    for number in range(index, index + count):
        slot = (table, number, descriptor_set, scope)
        if slot in claimed:
            raise ArgumentTableOverlapError(
                f"Resources '{claimed[slot]}' and '{name}' both claim {table} index {number}"
            )
        claimed[slot] = name


def place_argument_table_slots(codegen, claims, taken):
    """Place ``claims`` of ``(table, index, order, scope, node)``; return ``{id(node): (set, binding)}``.

    ``taken(slot)`` tells whether a ``(set, binding)`` slot is already held. Each placed slot is added to
    ``codegen.reserved_resource_bindings``.
    """
    claimed = {}
    slots = {}
    for table, index, _order, scope, node in sorted(
        claims, key=lambda claim: (ARGUMENT_TABLE_ORDER[claim[0]], claim[1], claim[2])
    ):
        name = getattr(node, "name", "resource")
        descriptor_set = codegen.resource_descriptor_set(node)
        _claim(
            claimed,
            (table, index, descriptor_set, scope),
            _index_count(codegen, node),
            name,
        )
        binding = index
        while taken((descriptor_set, binding)):
            binding += 1
        codegen.reserved_resource_bindings.add((descriptor_set, binding))
        slots[id(node)] = (descriptor_set, binding)
    return slots


def plan_entry_descriptor_slots(codegen, params, takes_descriptor):
    """Descriptor slots of an entry point's ``buffer``/``texture``/``sampler`` parameters, by parameter name.

    ``params`` are the entry point's parameters in declaration order and ``takes_descriptor(param)`` tells which of
    them the entry point declares as descriptors. The slots are reserved before any parameter is declared, so
    parameters without an index take the bindings that remain.
    """
    claims = []
    for order, param in enumerate(params):
        if not takes_descriptor(param):
            continue
        index = argument_table_index(codegen, param)
        if index is not None:
            claims.append((index[0], index[1], order, None, param))

    def taken(slot):
        return (
            slot in codegen.reserved_resource_bindings
            or slot in codegen.used_resource_bindings
        )

    slots = place_argument_table_slots(codegen, claims, taken)
    return {
        getattr(param, "name", None): slots[id(param)]
        for *_rest, param in claims
        if getattr(param, "name", None)
    }

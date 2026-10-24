"""Fixed sampler state on targets whose host fixes it: a bound sampler plus a ``crosstl-fixed-samplers`` record.

MSL declares sampler state in shader source (``constexpr sampler s(filter::linear)``, MSL 3.2 section 2.10) and
CrossGL keeps it as a ``sampler(...)`` constructor (:mod:`.sampler_states`). Direct3D and Vulkan fix sampler state
outside the shader instead, in objects the shader only names through a binding:

* Direct3D 12 through a static sampler of the root signature (``D3D12_STATIC_SAMPLER_DESC``) or a sampler
  descriptor, and Direct3D 11 through a sampler state object (``ID3D11Device::CreateSamplerState``), both bound to
  the shader's ``SamplerState`` register;
* Vulkan through an immutable sampler of the descriptor set layout (``VkDescriptorSetLayoutBinding::
  pImmutableSamplers``) or a sampler descriptor at the shader's ``sampler`` binding.

:func:`bind_fixed_samplers` turns every fixed sampler of a program into a sampler resource the host binds, so the
shader samples exactly as it would through a bound sampler, and records each one's state in
``program.annotations["fixed_samplers"]``. The generators write those records, with the register or binding they
gave each sampler, in a ``// crosstl-fixed-samplers:`` artifact header (HLSL) or an ``OpSourceExtension
"crosstl-fixed-samplers: ..."`` debug instruction (SPIR-V); the host creates the sampler from it. The state keeps the CrossGL (MSL) vocabulary of
:class:`.sampler_states.SamplerState`. The host maps it as follows:

* ``clamp_to_zero`` addressing is border addressing with a transparent black border;
* ``mip_filter::none`` is point mip filtering with the maximum LOD at 0 (Direct3D) or 0.25 (Vulkan section 13.1.1);
* ``max_anisotropy(n)`` above 1 selects anisotropic filtering with that maximum;
* ``lod_clamp(min, max)`` sets the minimum and maximum LOD;
* ``compare_func`` is the comparison function of a comparison sampler.

Each fixed sampler becomes one module-scope sampler. A sampler declared at module scope keeps its declaration; a
local one moves to module scope, and an inline ``sampler(...)`` operand becomes a reference to a new one. Each takes
the next sampler slot after every slot the program already names, so no existing binding moves, and keeps its name
unless another declaration uses it. Pixel (unnormalized) coordinates are refused: Direct3D samplers have none, and
Vulkan restricts unnormalized samplers to single-level images sampled without LOD, offsets or comparison, which this
pass does not check.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from .. import ast as A
from ..artifact_contracts import FIXED_SAMPLER_PREFIX
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import iter_functions
from .rewrite import child_slots, get_child, set_child
from .sampler_states import SamplerState, decode_sampler_state, is_sampler_constructor

__all__ = [
    "ANNOTATION",
    "FixedSamplerCoordinatesUnsupportedError",
    "bind_fixed_samplers",
    "fixed_sampler_header",
]

ANNOTATION = "fixed_samplers"
FIXED_SAMPLER_SCHEMA_VERSION = 1
# Every spelling of a target whose host binds the sampler state.
BINDING_TARGETS = frozenset(
    {"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12", "vulkan", "spirv"}
)
_FIXED_QUALIFIERS = frozenset({"const", "constexpr"})
_DROPPED_QUALIFIERS = _FIXED_QUALIFIERS | {"static"}
_INLINE_NAME = "crosstl_fixed_sampler"
_VULKAN_TARGETS = frozenset({"vulkan", "spirv"})
_SET_ATTRIBUTES = frozenset({"set", "group", "space"})


class FixedSamplerCoordinatesUnsupportedError(TranslationDiagnostic):
    """A fixed sampler with pixel coordinates on a target whose host samplers cannot carry them."""

    def __init__(self, target, name):
        self.missing_capabilities = (f"{target}.unnormalized-sampler-coordinates",)
        super().__init__(
            f"fixed sampler {name} uses pixel coordinates (coord::pixel), which {target} cannot bind: Direct3D "
            "samplers have no unnormalized coordinates, and Vulkan allows them only for single-level images "
            "sampled without LOD, offsets or comparison",
            target=target,
            code=f"{CODE_PREFIX}{target}-unnormalized-sampler-coordinates-unsupported",
        )


def _is_sampler_type(var_type) -> bool:
    return getattr(var_type, "name", None) == "sampler" or str(var_type) == "sampler"


def _fixed_declaration_state(node):
    """The state a fixed sampler declaration fixes, or ``None`` for any other node."""
    if not isinstance(node, A.VariableNode) or not _is_sampler_type(
        getattr(node, "var_type", None)
    ):
        return None
    initial = getattr(node, "initial_value", None)
    if initial is not None:
        return (
            decode_sampler_state(initial) if is_sampler_constructor(initial) else None
        )
    qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", None) or []}
    return SamplerState() if qualifiers & _FIXED_QUALIFIERS else None


def _ordered_nodes(node, owner=None, field=None, key=None):
    """``(node, owner, field, key)`` for every node under ``node`` in source order."""
    if isinstance(node, list):
        for index, item in enumerate(node):
            yield from _ordered_nodes(item, owner, field, index)
        return
    if not isinstance(node, A.ASTNode):
        return
    yield node, owner, field, key
    for child_field, child_key in child_slots(node):
        yield from _ordered_nodes(
            get_child(node, child_field, child_key), node, child_field, child_key
        )


def _declared_names(program):
    counts = {}

    def count(name):
        if name:
            counts[name] = counts.get(name, 0) + 1

    for node, *_ in _ordered_nodes(program):
        if isinstance(node, (A.VariableNode, A.FunctionNode, A.StructNode)):
            count(getattr(node, "name", None))
        for parameter in getattr(node, "parameters", None) or []:
            count(getattr(parameter, "name", None))
    return counts


def _slot_number(attribute):
    name = str(getattr(attribute, "name", "")).lower()
    arguments = getattr(attribute, "arguments", None) or []
    if not arguments:
        return None
    value = getattr(arguments[0], "value", getattr(arguments[0], "name", arguments[0]))
    text = str(value).strip()
    if name == "sampler" and text.isdigit():
        return int(text)
    if name == "register" and text[:1].lower() == "s" and text[1:].isdigit():
        return int(text[1:])
    return None


def _attributes(program):
    for node, *_ in _ordered_nodes(program):
        for holder in [node, *(getattr(node, "parameters", None) or [])]:
            yield from getattr(holder, "attributes", None) or []


def _first_free_slot(program) -> int:
    used = [
        slot
        for attribute in _attributes(program)
        if (slot := _slot_number(attribute)) is not None
    ]
    return max(used, default=-1) + 1


def _first_free_set(program) -> int:
    used = []
    for attribute in _attributes(program):
        arguments = getattr(attribute, "arguments", None) or []
        if str(getattr(attribute, "name", "")).lower() in _SET_ATTRIBUTES and arguments:
            value = str(getattr(arguments[0], "value", "")).strip()
            if value.isdigit():
                used.append(int(value))
    # Resources without a set attribute are in set 0.
    return max(used, default=0) + 1


def _integer_attribute(name, value):
    return A.AttributeNode(name, [A.LiteralNode(value, A.PrimitiveType("int"))])


def _globals(program):
    yield program.global_variables
    stages = getattr(program, "stages", None) or {}
    for stage in stages.values() if isinstance(stages, dict) else stages:
        yield stage.local_variables


def _rename(body, old, new):
    for node, *_ in _ordered_nodes(body):
        if isinstance(node, A.IdentifierNode) and node.name == old:
            node.name = new
            node.identifier = new


class _Binder:
    def __init__(self, program, target):
        self.program = program
        self.target = target
        self.counts = _declared_names(program)
        self.vulkan = target in _VULKAN_TARGETS
        self.slot = (
            _first_free_set(program) if self.vulkan else _first_free_slot(program)
        )
        self.index = 0
        self.records = []

    def name_for(self, preferred):
        if preferred and self.counts.get(preferred, 0) <= 1:
            return preferred
        stem = preferred or _INLINE_NAME
        index = 0
        while f"{stem}_{index}" in self.counts:
            index += 1
        name = f"{stem}_{index}"
        self.counts[name] = 1
        return name

    def bind(self, state, name, label):
        """The slot attributes of the next fixed sampler: its own Vulkan set, or the next HLSL sampler register."""
        if state.coord == "pixel":
            raise FixedSamplerCoordinatesUnsupportedError(self.target, label)
        record = {"name": name, "state": _state_json(state)}
        if self.vulkan:
            record.update(set=self.slot, binding=self.index)
            attributes = [
                _integer_attribute("set", self.slot),
                _integer_attribute("binding", self.index),
            ]
            self.index += 1
        else:
            record.update(register=f"s{self.slot}", space=0)
            attributes = [_integer_attribute("sampler", self.slot)]
            self.slot += 1
        self.records.append(record)
        return attributes

    def run(self):
        for declarations in _globals(self.program):
            for node in declarations:
                state = _fixed_declaration_state(node)
                if state is not None:
                    node.attributes = [
                        *node.attributes,
                        *self.bind(state, node.name, f"'{node.name}'"),
                    ]
                    node.initial_value = None
                    node.qualifiers = [
                        q
                        for q in node.qualifiers
                        if str(q).lower() not in _DROPPED_QUALIFIERS
                    ]
        constants = getattr(self.program, "constants", None) or []
        for node in list(constants):
            value = getattr(node, "value", None)
            if _is_sampler_type(getattr(node, "const_type", None)) and (
                is_sampler_constructor(value)
            ):
                constants.remove(node)
                self.declare(decode_sampler_state(value), node.name, f"'{node.name}'")
        for function in list(iter_functions(self.program)):
            self.bind_function(function)
        return self.records

    def bind_function(self, function):
        removals = []
        for node, owner, field, key in list(
            _ordered_nodes(function.body, function, "body")
        ):
            state = _fixed_declaration_state(node)
            if state is not None and owner is not None and key is not None:
                name = self.name_for(node.name)
                self.declare(state, name, f"'{node.name}'")
                if name != node.name:
                    _rename(function.body, node.name, name)
                removals.append((getattr(owner, field), key))
            elif is_sampler_constructor(node) and not isinstance(owner, A.VariableNode):
                name = self.name_for(None)
                self.declare(decode_sampler_state(node), name, "sampler(...)")
                set_child(owner, field, key, A.IdentifierNode(name))
        for statements, index in sorted(removals, key=lambda item: -item[1]):
            statements.pop(index)

    def declare(self, state, name, label):
        attributes = self.bind(state, name, label)
        self.program.global_variables.append(
            A.VariableNode(name, A.NamedType("sampler"), attributes=attributes)
        )


def _state_json(state):
    record = asdict(state)
    if record["lod_clamp"] is not None:
        record["lod_clamp"] = list(record["lod_clamp"])
    return record


def bind_fixed_samplers(program, target):
    """Turn every fixed sampler of ``program`` into a bound sampler for ``target``; returns the state records."""
    target = str(target or "").lower()
    if target not in BINDING_TARGETS or not isinstance(program, A.ShaderNode):
        return []
    records = _Binder(program, target).run()
    if records:
        program.annotations[ANNOTATION] = records
    return records


def fixed_sampler_header(records):
    """The ``// crosstl-fixed-samplers:`` header line for ``records``, or ``""`` when there are none."""
    if not records:
        return ""
    payload = {"samplers": records, "schemaVersion": FIXED_SAMPLER_SCHEMA_VERSION}
    return FIXED_SAMPLER_PREFIX + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )

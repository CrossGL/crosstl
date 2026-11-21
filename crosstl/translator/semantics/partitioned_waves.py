"""Partitioned wave operations as loops over the lanes of the subgroup.

:data:`~crosstl.translator.wave_operations.WAVE_PARTITIONED_OPERATIONS` lists the CrossGL reductions and scans over
the partition a ``uint4`` mask names (``WaveMultiSum``, ``WaveMultiPrefixInclusiveMin``, ...). No generator has
them, so this pass replaces each call with a call of a CrossGL function, one per operation and operand type, that
computes it from operations every subgroup target has:

* ``WaveActiveBallot(true)`` masks the lanes active at the call, and ``WaveGetLaneCount()`` and
  ``WaveGetLaneIndex()`` give the subgroup size and the calling lane;
* a loop over every lane index reads that lane's value with ``WaveReadLaneAt``, one component at a time. The bound
  and the index are the same in every active lane, so every active lane executes each read, and the value of a
  lane that is inactive or outside the partition is read but not used;
* the values of the lanes whose bit is set in both the partition mask and the active mask, and whose index is in
  the operation's scope (any lane for a reduction, a lower or equal index for an inclusive scan, a lower index for
  an exclusive one), are combined in lane order, starting from the operator's identity as the SPIR-V partitioned
  group operations define it: 0 for a sum, 1 for a product, all bits set for ``and``, +INF or the largest integer
  for ``min``, and -INF or the smallest integer for ``max``.

Every lane of a partition combines the same values in the same order, so a reduction gives each of them the same
result. Operands are 32-bit integers, 16-, 32- or 64-bit floats and their vectors; the bitwise operators also take a
``bool``, where they are the logical operators. Any other operand, or a call whose operand type cannot be inferred,
is refused with :class:`PartitionedWaveUnsupportedError`.

A target that emits the operations itself registers them with :func:`register_native_partitioned_wave_operations`,
and its calls are left alone. CrossGL output keeps every call.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..wave_operations import WAVE_PARTITIONED_OPERATIONS
from .hub_types import HubType
from .lane_reads import (
    lane_read,
    parse_helper_function,
    type_spelling,
    wave_operand_type,
)
from .rewrite import call, child_slots, get_child, set_child
from .scoped_walk import ScopedWalker

__all__ = [
    "PartitionedWaveUnsupportedError",
    "lower_partitioned_wave_operations",
    "native_partitioned_wave_operations",
    "register_native_partitioned_wave_operations",
]

_NATIVE: dict[str, frozenset[str]] = {"cgl": frozenset(WAVE_PARTITIONED_OPERATIONS)}

_INFINITY = "uintBitsToFloat(0x7f800000u)"
# The identity of each operator for each operand kind (SPIR-V 3.42.24, "Non-Uniform Instructions").
_IDENTITIES = {
    ("sum", "f"): "0.0",
    ("sum", "i"): "0",
    ("sum", "u"): "0u",
    ("product", "f"): "1.0",
    ("product", "i"): "1",
    ("product", "u"): "1u",
    ("min", "f"): _INFINITY,
    ("min", "i"): "2147483647",
    ("min", "u"): "0xffffffffu",
    ("max", "f"): f"-{_INFINITY}",
    ("max", "i"): "(-2147483647 - 1)",
    ("max", "u"): "0u",
    ("and", "i"): "-1",
    ("and", "u"): "0xffffffffu",
    ("and", "b"): "true",
    ("or", "i"): "0",
    ("or", "u"): "0u",
    ("or", "b"): "false",
    ("xor", "i"): "0",
    ("xor", "u"): "0u",
    ("xor", "b"): "false",
}
_COMBINE = {
    "sum": "result + other",
    "product": "result * other",
    "min": "min(result, other)",
    "max": "max(result, other)",
    "and": "result & other",
    "or": "result | other",
    "xor": "result ^ other",
}
_COMBINE_BOOL = {
    "and": "result && other",
    "or": "result || other",
    "xor": "result != other",
}
_SCOPES = {
    "reduce": "",
    "inclusive": "lane <= self_lane && ",
    "exclusive": "lane < self_lane && ",
}


class PartitionedWaveUnsupportedError(TranslationDiagnostic):
    """A partitioned wave operation whose operand type the lane-loop lowering cannot compute."""

    def __init__(self, target, what, *, reason_code, source_location=None):
        self.reason_code = reason_code
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.partitioned-wave-{reason_code}",)
        super().__init__(
            what,
            target=target,
            code=f"{CODE_PREFIX}{target}-partitioned-wave-unsupported",
        )


def register_native_partitioned_wave_operations(target, operations):
    """Declare that ``target``'s generator emits ``operations`` itself, so their calls are not lowered."""
    unknown = sorted(set(operations) - set(WAVE_PARTITIONED_OPERATIONS))
    if unknown:
        raise ValueError(f"not partitioned wave operations: {', '.join(unknown)}")
    _NATIVE[str(target)] = _NATIVE.get(str(target), frozenset()) | frozenset(operations)


def native_partitioned_wave_operations(target) -> frozenset[str]:
    return _NATIVE.get(str(target or ""), frozenset())


def _helper_source(name, operator, scope, value: HubType, spelling):
    scalar = type_spelling(value.with_lanes(1)) if value.lanes > 1 else spelling
    identity = _IDENTITIES[(operator, value.kind)]
    if value.kind == "f" and value.bits != 32:
        identity = f"{scalar}({identity})"
    if value.lanes > 1:
        identity = f"{spelling}({identity})"
    read = lane_read(value, spelling)
    combine = (_COMBINE_BOOL if value.kind == "b" else _COMBINE)[operator]
    return (
        f"{spelling} {name}({spelling} value, uvec4 mask) {{\n"
        f"    {spelling} result = {identity};\n"
        "    uvec4 active = WaveActiveBallot(true);\n"
        "    uint count = WaveGetLaneCount();\n"
        "    uint self_lane = WaveGetLaneIndex();\n"
        "    for (uint lane = 0u; lane < count; lane++) {\n"
        f"        {spelling} other = {read};\n"
        "        uint word = lane < 32u ? mask.x & active.x : (lane < 64u ? mask.y & active.y : "
        "(lane < 96u ? mask.z & active.z : mask.w & active.w));\n"
        f"        if ({_SCOPES[scope]}((word >> (lane & 31u)) & 1u) != 0u) {{\n"
        f"            result = {combine};\n"
        "        }\n"
        "    }\n"
        "    return result;\n"
        "}\n"
    )


class _PartitionedWaveLowering(ScopedWalker):
    def __init__(self, program, target):
        super().__init__(program)
        self.target = target
        self.native = native_partitioned_wave_operations(target)
        self.replacements: dict[int, A.FunctionCallNode] = {}
        # The replaced calls, kept alive so that their ids are not reused before :meth:`finish`.
        self.originals: list[A.WaveOpNode] = []
        self.helpers: dict[tuple[str, HubType], str] = {}
        self.functions: list[A.FunctionNode] = []
        self.taken = {
            getattr(node, "name", None)
            for node in program.walk()
            if isinstance(node, (A.FunctionNode, A.VariableNode, A.IdentifierNode))
        }

    def lowered(self, node) -> bool:
        return (
            isinstance(node, A.WaveOpNode)
            and node.operation in WAVE_PARTITIONED_OPERATIONS
            and node.operation not in self.native
        )

    def refuse(self, node, what, reason_code):
        raise PartitionedWaveUnsupportedError(
            self.target,
            f"{node.operation}: {what}",
            reason_code=reason_code,
            source_location=getattr(node, "source_location", None),
        )

    def operand_type(self, expr) -> HubType | None:
        return wave_operand_type(self.scope, expr)

    def visit(self, node):
        if not self.lowered(node):
            return
        if len(node.arguments or []) != 2:
            self.refuse(node, "takes a value and a uint4 partition mask", "arity")
        value = self.operand_type(node.arguments[0])
        if value is None:
            self.refuse(node, "the operand type cannot be inferred", "operand-type")
        operator, scope = WAVE_PARTITIONED_OPERATIONS[node.operation]
        spelling = type_spelling(value)
        if (
            spelling is None
            or (operator, value.kind) not in _IDENTITIES
            or (value.kind == "b" and value.lanes != 1)
        ):
            self.refuse(
                node,
                f"has no lane-loop lowering for a {value.kind}{value.bits}x{value.lanes} operand",
                "operand-type",
            )
        name = self.helper(node.operation, operator, scope, value, spelling)
        replacement = call(name, list(node.arguments))
        replacement.source_location = getattr(node, "source_location", None)
        self.replacements[id(node)] = replacement
        self.originals.append(node)

    def helper(self, operation, operator, scope, value, spelling):
        key = (operation, value)
        if key in self.helpers:
            return self.helpers[key]
        stem = f"crossgl_{operation}_{spelling}"
        name, suffix = stem, 1
        while name in self.taken:
            suffix += 1
            name = f"{stem}_{suffix}"
        self.taken.add(name)
        self.helpers[key] = name
        self.functions.append(
            parse_helper_function(
                _helper_source(name, operator, scope, value, spelling)
            )
        )
        return name

    def finish(self):
        # A replacement's arguments may hold another lowered call, which the next walk reaches.
        changed = True
        while changed:
            changed = False
            for node in list(self.program.walk()):
                for field, key in list(child_slots(node)):
                    replacement = self.replacements.get(id(get_child(node, field, key)))
                    if replacement is not None:
                        set_child(node, field, key, replacement)
                        changed = True
        for node in self.program.walk():
            if self.lowered(node):
                self.refuse(node, "is outside every function body", "placement")
        functions = getattr(self.program, "functions", None)
        if functions is None:
            functions = []
            self.program.functions = functions
        functions[:0] = self.functions


def lower_partitioned_wave_operations(program, target) -> int:
    """Lower ``program``'s partitioned wave operations for ``target``; return the number of calls lowered."""
    if not isinstance(program, A.ShaderNode):
        return 0
    native = native_partitioned_wave_operations(target)
    if not any(
        isinstance(node, A.WaveOpNode)
        and node.operation in WAVE_PARTITIONED_OPERATIONS
        and node.operation not in native
        for node in program.walk()
    ):
        return 0
    lowering = _PartitionedWaveLowering(program, str(target or ""))
    lowering.run_program()
    lowering.finish()
    return len(lowering.replacements)

"""Subgroup rotations as reads of the rotated lane.

:data:`~crosstl.translator.wave_operations.WAVE_ROTATE_OPERATIONS` lists the CrossGL rotations, ``WaveRotate(value,
delta)`` and ``WaveClusteredRotate(value, delta, clusterSize)``. Their semantics are those of SPIR-V
``OpGroupNonUniformRotateKHR`` (SPV_KHR_subgroup_rotate): the result is the value of lane

* ``(WaveGetLaneIndex() + delta) & (WaveGetLaneCount() - 1)`` for ``WaveRotate``, and
* ``((WaveGetLaneIndex() + delta) & (clusterSize - 1)) + (WaveGetLaneIndex() & ~(clusterSize - 1))`` for
  ``WaveClusteredRotate``.

Vulkan, Direct3D 12 and Metal give every subgroup a power-of-two size, and ``clusterSize`` is a constant power of two,
so the masks compute the lane modulo the subgroup or the cluster. No generator has the rotations, so this pass
replaces each call with a call of a CrossGL function, one per operation and operand type, that computes that lane
and reads it with ``WaveReadLaneAt`` (see :mod:`.lane_reads`). ``delta`` and ``clusterSize`` are converted to
``uint`` at the call. Operands are 32-bit integers, 16-, 32- or 64-bit floats, their vectors and ``bool``; any other
operand, a call whose operand type cannot be inferred and a call with the wrong number of arguments are refused with
:class:`WaveRotationUnsupportedError`.

A target that emits the rotations itself registers them with :func:`register_native_wave_rotations`, and its calls
are left alone. CrossGL output keeps every call.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..refusals import RegistryRefusal
from ..wave_operations import WAVE_ROTATE_OPERATIONS
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
    "WaveRotationUnsupportedError",
    "lower_wave_rotations",
    "native_wave_rotations",
    "register_native_wave_rotations",
]

_NATIVE: dict[str, frozenset[str]] = {"cgl": frozenset(WAVE_ROTATE_OPERATIONS)}

_SOURCE_LANES = {
    "WaveRotate": (
        "uint delta",
        "(WaveGetLaneIndex() + delta) & (WaveGetLaneCount() - 1u)",
    ),
    "WaveClusteredRotate": (
        "uint delta, uint clusterSize",
        "((WaveGetLaneIndex() + delta) & (clusterSize - 1u))"
        " + (WaveGetLaneIndex() & ~(clusterSize - 1u))",
    ),
}
_UINT = HubType("u", 32)


class WaveRotationUnsupportedError(TranslationDiagnostic):
    """A subgroup rotation whose operand the lane-read lowering cannot take."""

    def __init__(self, target, what, *, reason_code, source_location=None):
        self.reason_code = reason_code
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.wave-rotation-{reason_code}",)
        super().__init__(
            what,
            target=target,
            code=f"{CODE_PREFIX}{target}-wave-rotation-unsupported",
        )


def register_native_wave_rotations(target, operations):
    """Declare that ``target``'s generator emits ``operations`` itself, so their calls are not lowered."""
    unknown = sorted(set(operations) - set(WAVE_ROTATE_OPERATIONS))
    if unknown:
        raise RegistryRefusal(f"not wave rotations: {', '.join(unknown)}")
    _NATIVE[str(target)] = _NATIVE.get(str(target), frozenset()) | frozenset(operations)


def native_wave_rotations(target) -> frozenset[str]:
    return _NATIVE.get(str(target or ""), frozenset())


def _helper_source(name, operation, value: HubType, spelling):
    parameters, source_lane = _SOURCE_LANES[operation]
    return (
        f"{spelling} {name}({spelling} value, {parameters}) {{\n"
        f"    uint lane = {source_lane};\n"
        f"    return {lane_read(value, spelling)};\n"
        "}\n"
    )


class _WaveRotationLowering(ScopedWalker):
    def __init__(self, program, target):
        super().__init__(program)
        self.target = target
        self.native = native_wave_rotations(target)
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
            and node.operation in WAVE_ROTATE_OPERATIONS
            and node.operation not in self.native
        )

    def refuse(self, node, what, reason_code):
        raise WaveRotationUnsupportedError(
            self.target,
            f"{node.operation}: {what}",
            reason_code=reason_code,
            source_location=getattr(node, "source_location", None),
        )

    def visit(self, node):
        if not self.lowered(node):
            return
        arguments = list(node.arguments or [])
        if len(arguments) != WAVE_ROTATE_OPERATIONS[node.operation]:
            what = "takes a value and a uint delta"
            if node.operation == "WaveClusteredRotate":
                what += " and a constant cluster size"
            self.refuse(node, what, "arity")
        value = wave_operand_type(self.scope, arguments[0])
        if value is None:
            self.refuse(node, "the operand type cannot be inferred", "operand-type")
        spelling = type_spelling(value)
        if spelling is None:
            self.refuse(
                node,
                f"has no lane-read lowering for a {value.kind}{value.bits}x{value.lanes} operand",
                "operand-type",
            )
        for index in range(1, len(arguments)):
            if wave_operand_type(self.scope, arguments[index]) != _UINT:
                arguments[index] = call("uint", [arguments[index]])
        replacement = call(self.helper(node.operation, value, spelling), arguments)
        replacement.source_location = getattr(node, "source_location", None)
        self.replacements[id(node)] = replacement
        self.originals.append(node)

    def helper(self, operation, value, spelling):
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
            parse_helper_function(_helper_source(name, operation, value, spelling))
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


def lower_wave_rotations(program, target) -> int:
    """Lower ``program``'s subgroup rotations for ``target``; return the number of calls lowered."""
    if not isinstance(program, A.ShaderNode):
        return 0
    native = native_wave_rotations(target)
    if not any(
        isinstance(node, A.WaveOpNode)
        and node.operation in WAVE_ROTATE_OPERATIONS
        and node.operation not in native
        for node in program.walk()
    ):
        return 0
    lowering = _WaveRotationLowering(program, str(target or ""))
    lowering.run_program()
    lowering.finish()
    return len(lowering.replacements)

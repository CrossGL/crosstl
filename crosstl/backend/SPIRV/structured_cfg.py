"""Structured control flow reconstruction for imported SPIR-V functions.

A SPIR-V shader function is a control flow graph whose structure is declared by merge instructions (SPIR-V 1.6
section 2.11): ``OpSelectionMerge`` heads an ``if`` or ``switch`` construct and ``OpLoopMerge`` a loop, naming its
merge block and continue target. CrossGL has only structured statements, so the importer rebuilds the statement
tree from those declarations:

* A loop becomes ``while (true) { ... }``. Its header's own statements run on every iteration, so they open the
  body. A branch to the loop's merge block is ``break``; the back edge to its header ends the iteration. A branch to
  its continue target runs the continue construct and then ``continue``s: the construct is emitted where control
  reaches it, which duplicates it at every early ``continue`` but keeps its effects on every path.
* A selection becomes ``if``/``else`` and a switch ``switch`` with fall-through cases ordered after the cases that
  fall into them.
* A conditional branch without a merge instruction is allowed only when one target leaves the construct; it becomes
  ``if (condition) { break; }`` or the like.
* Every ``OpPhi`` that is not a plain value or a two-way selection becomes a function-scope variable assigned on each
  incoming edge. The copies of one edge form a parallel assignment, so a copy that reads a variable another copy of
  the same edge writes reads it through a temporary.

Anything the statement tree cannot express, such as a branch out of two constructs at once, raises
:class:`SpirvUnstructuredControlFlowError` rather than being flattened into straight-line code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

from .VulkanAst import (
    AssignmentNode,
    BreakNode,
    CaseNode,
    ContinueNode,
    IfNode,
    SwitchNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)

__all__ = [
    "FunctionCfg",
    "SpirvUnstructuredControlFlowError",
    "StructuredEmitter",
    "build_function_cfg",
    "classify_phis",
    "referenced_spirv_ids",
]

BRANCH_TERMINATORS = frozenset({"OpBranch", "OpBranchConditional", "OpSwitch"})


class SpirvUnstructuredControlFlowError(TranslationDiagnostic):
    """A SPIR-V control flow shape that CrossGL's structured statements cannot express."""

    def __init__(self, detail):
        self.missing_capabilities = ("spirv.import.unstructured_control_flow",)
        super().__init__(
            f"SPIR-V control flow cannot be expressed as structured CrossGL statements: {detail}",
            target="cgl",
            code=f"{CODE_PREFIX}spirv-unstructured-control-flow",
        )


@dataclass
class Block:
    label: str
    index: int
    instructions: list = field(default_factory=list)
    merge: tuple | None = None
    terminator: tuple = ("OpUnreachable", [])
    successors: list = field(default_factory=list)
    phis: list = field(default_factory=list)


@dataclass
class FunctionCfg:
    blocks: dict
    order: list
    predecessors: dict

    @property
    def entry(self):
        return self.order[0] if self.order else None


def _terminator_successors(opcode, operands):
    if opcode == "OpBranch":
        return list(operands[:1])
    if opcode == "OpBranchConditional":
        return list(dict.fromkeys(operands[1:3]))
    if opcode == "OpSwitch":
        targets = [operands[1]] if len(operands) >= 2 else []
        targets.extend(operands[3::2])
        return list(dict.fromkeys(targets))
    return []


def build_function_cfg(raw_instructions):
    """The function's basic blocks in layout order, each with its merge declaration, terminator and phis."""
    blocks = {}
    order = []
    current = None
    for result_id, opcode, operands, _line in raw_instructions:
        if opcode == "OpLabel" and result_id:
            current = Block(result_id, len(order))
            blocks[result_id] = current
            order.append(result_id)
            continue
        if current is None or opcode == "OpFunctionEnd":
            continue
        current.instructions.append((result_id, opcode, operands))
        if opcode == "OpLoopMerge" and len(operands) >= 2:
            current.merge = ("loop", operands[0], operands[1])
        elif opcode == "OpSelectionMerge" and operands:
            current.merge = ("selection", operands[0])
        elif opcode == "OpPhi" and result_id and operands:
            incoming = [
                (operands[index], operands[index + 1])
                for index in range(1, len(operands) - 1, 2)
            ]
            current.phis.append((result_id, operands[0], incoming))
        if opcode in BRANCH_TERMINATORS:
            current.terminator = (opcode, list(operands))
            current.successors = _terminator_successors(opcode, operands)
    predecessors = {label: [] for label in order}
    for label in order:
        for successor in blocks[label].successors:
            if successor not in blocks:
                raise SpirvUnstructuredControlFlowError(
                    f"block {label} branches to {successor}, which is not a block of the function"
                )
            if label not in predecessors[successor]:
                predecessors[successor].append(label)
    return FunctionCfg(blocks, order, predecessors)


def classify_phis(cfg):
    """How each ``OpPhi`` is imported: ``("same", value)``, ``("select", condition, true, false)`` or ``("variable",)``.

    A phi whose incoming values are all one value is that value. A phi at the merge of a two-way selection whose
    incoming edges come straight from the two branch targets is a conditional expression on the selection's
    condition. Every other phi is a variable assigned on its incoming edges.
    """
    selections = {}
    for label in cfg.order:
        block = cfg.blocks[label]
        opcode, operands = block.terminator
        if (
            block.merge
            and block.merge[0] == "selection"
            and opcode == "OpBranchConditional"
            and len(operands) >= 3
            and operands[1] != operands[2]
        ):
            selections[block.merge[1]] = (operands[0], operands[1], operands[2])
    kinds = {}
    for label in cfg.order:
        for phi_id, _type_id, incoming in cfg.blocks[label].phis:
            values = [value for value, _pred in incoming]
            if values and all(value == values[0] for value in values):
                kinds[phi_id] = ("same", values[0])
                continue
            selection = selections.get(label)
            if selection is not None and len(incoming) == 2:
                by_label = {pred: value for value, pred in incoming}
                condition, true_label, false_label = selection
                if (
                    true_label in by_label
                    and false_label in by_label
                    and label not in (true_label, false_label)
                ):
                    kinds[phi_id] = (
                        "select",
                        condition,
                        by_label[true_label],
                        by_label[false_label],
                    )
                    continue
            kinds[phi_id] = ("variable",)
    return kinds


def _child_nodes(node):
    if isinstance(node, (list, tuple)):
        yield from node
        return
    attributes = getattr(node, "__dict__", None)
    if not attributes:
        return
    for value in attributes.values():
        if isinstance(value, (list, tuple)):
            yield from value
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            yield value


def referenced_spirv_ids(node, candidates):
    """The ``candidates`` ids that ``VariableNode`` references within ``node`` carry."""
    found = set()
    stack = [node]
    seen = set()
    while stack:
        current = stack.pop()
        if current is None or isinstance(current, (str, int, float, bool)):
            continue
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        if isinstance(current, VariableNode):
            spirv_id = getattr(current, "spirv_id", None)
            if spirv_id in candidates:
                found.add(spirv_id)
        stack.extend(_child_nodes(current))
    return found


def negate(condition):
    if isinstance(condition, UnaryOpNode) and condition.op == "!":
        return condition.operand
    return UnaryOpNode("!", condition)


@dataclass
class _Frame:
    kind: str
    merge: str | None = None
    header: str | None = None
    continue_target: str | None = None
    next_case: str | None = None
    case_targets: frozenset = frozenset()


class StructuredEmitter:
    """Builds a function's CrossGL statement tree from its CFG and per-block statements.

    ``hooks`` provides ``statements(label)``, the statements the block's instructions produced;
    ``expression(value_id)``; ``phi_variable(phi_id)``, the reference to a variable phi; and
    ``phi_temporary(phi_id)``, a ``(declaration, reference)`` pair for a copy temporary.
    """

    def __init__(self, cfg, phi_kinds, hooks):
        self.cfg = cfg
        self.hooks = hooks
        self.variable_phis = {}
        for label in cfg.order:
            entries = [
                (phi_id, {pred: value for value, pred in incoming})
                for phi_id, _type_id, incoming in cfg.blocks[label].phis
                if phi_kinds.get(phi_id, ("variable",))[0] == "variable"
            ]
            if entries:
                self.variable_phis[label] = entries
        self._emitted = set()
        self._active = set()

    def emit_function(self):
        body = []
        if self.cfg.entry is not None:
            self._emit_region(self._enter(self.cfg.entry, []), [], body)
        return body

    def _emit_region(self, label, frames, out):
        while label is not None:
            label = self._emit_one(label, frames, out)

    def _enter(self, label, frames):
        duplicating = any(frame.kind == "continue" for frame in frames)
        if label in self._active:
            raise SpirvUnstructuredControlFlowError(
                f"block {label} is re-entered without a loop header"
            )
        if not duplicating:
            if label in self._emitted:
                raise SpirvUnstructuredControlFlowError(
                    f"block {label} is reached from more than one construct"
                )
            self._emitted.add(label)
        return label

    def _emit_one(self, label, frames, out):
        block = self.cfg.blocks[label]
        if block.merge and block.merge[0] == "loop":
            return self._emit_loop(block, frames, out)
        self._active.add(label)
        try:
            out.extend(self.hooks.statements(label))
            return self._emit_terminator(block, frames, out)
        finally:
            self._active.discard(label)

    def _emit_terminator(self, block, frames, out):
        opcode, operands = block.terminator
        if opcode == "OpBranch":
            return self._follow(block.label, operands[0], frames, out)
        if opcode == "OpBranchConditional":
            if operands[1] == operands[2]:
                return self._follow(block.label, operands[1], frames, out)
            if block.merge and block.merge[0] == "selection":
                return self._emit_selection(block, frames, out)
            return self._emit_conditional_exit(block, frames, out)
        if opcode == "OpSwitch":
            if not (block.merge and block.merge[0] == "selection"):
                raise SpirvUnstructuredControlFlowError(
                    f"OpSwitch in block {block.label} has no OpSelectionMerge"
                )
            return self._emit_switch(block, frames, out)
        return None

    def _classify(self, target, frames):
        crossed_switch = False
        for depth, frame in enumerate(reversed(frames)):
            innermost = depth == 0
            if frame.kind == "selection":
                if innermost and target == frame.merge:
                    return "stop"
                continue
            if frame.kind == "continue":
                if target == frame.continue_target:
                    raise SpirvUnstructuredControlFlowError(
                        f"continue target {target} is reached from its own continue construct"
                    )
                continue
            if frame.kind == "switch":
                if target == frame.merge:
                    return "break"
                if target == frame.next_case and innermost:
                    return "fallthrough"
                if target in frame.case_targets:
                    raise SpirvUnstructuredControlFlowError(
                        f"branch to switch case {target} is not a fall-through from the preceding case"
                    )
                crossed_switch = True
                continue
            if target == frame.header:
                return "back"
            if target == frame.continue_target:
                return "continue"
            if target == frame.merge:
                if crossed_switch:
                    raise SpirvUnstructuredControlFlowError(
                        f"branch to loop merge {target} from inside a switch breaks out of two constructs"
                    )
                return "break"
            break
        self._refuse_outer_exit(target, frames)
        return "inline"

    def _refuse_outer_exit(self, target, frames):
        """Refuse a branch to a construct boundary that no single ``break``, ``continue`` or region end reaches."""
        for frame in frames:
            if target in (frame.merge, frame.header, frame.continue_target):
                raise SpirvUnstructuredControlFlowError(
                    f"branch to {target} leaves more than one construct"
                )

    def _loop_frame(self, frames):
        for frame in reversed(frames):
            if frame.kind == "loop":
                return frame
        return None

    def _follow(self, pred, target, frames, out):
        """Emit the edge ``pred -> target``; returns ``target`` when emission continues there."""
        if pred is not None:
            out.extend(self.edge_copies(pred, target))
        kind = self._classify(target, frames)
        if kind == "inline":
            return self._enter(target, frames)
        if kind == "break":
            out.append(BreakNode())
        elif kind == "back":
            out.append(ContinueNode())
        elif kind == "continue":
            loop = self._loop_frame(frames)
            continue_frames = [
                *frames,
                _Frame("continue", continue_target=loop.continue_target),
            ]
            self._emit_region(
                self._enter(loop.continue_target, continue_frames),
                continue_frames,
                out,
            )
        return None

    def _is_jump(self, kind):
        return kind in ("break", "back", "continue")

    def _arm(self, pred, target, frames):
        arm = []
        self._emit_region(self._follow(pred, target, frames, arm), frames, arm)
        return arm

    def _emit_selection(self, block, frames, out):
        condition_id, true_label, false_label = block.terminator[1][:3]
        merge = block.merge[1]
        inner = [*frames, _Frame("selection", merge=merge)]
        true_arm = self._arm(block.label, true_label, inner)
        false_arm = self._arm(block.label, false_label, inner)
        self._append_if(self.hooks.expression(condition_id), true_arm, false_arm, out)
        return self._follow(None, merge, frames, out)

    def _append_if(self, condition, true_arm, false_arm, out):
        if not true_arm and not false_arm:
            return
        if not true_arm:
            out.append(IfNode(negate(condition), false_arm))
            return
        out.append(IfNode(condition, true_arm, false_arm or None))

    def _emit_conditional_exit(self, block, frames, out):
        condition_id, true_label, false_label = block.terminator[1][:3]
        condition = self.hooks.expression(condition_id)
        true_kind = self._classify(true_label, frames)
        false_kind = self._classify(false_label, frames)
        if true_kind == "inline" and false_kind == "inline":
            raise SpirvUnstructuredControlFlowError(
                f"conditional branch in block {block.label} has no merge instruction and neither target leaves "
                "the construct"
            )
        if true_kind != "inline" and false_kind != "inline":
            true_arm = self._arm(block.label, true_label, frames)
            false_arm = self._arm(block.label, false_label, frames)
            if false_kind == "break" or (
                self._is_jump(false_kind) and not self._is_jump(true_kind)
            ):
                condition, true_arm, false_arm = negate(condition), false_arm, true_arm
                true_kind, false_kind = false_kind, true_kind
            if self._is_jump(true_kind):
                # The guarded arm leaves the region, so the other one can follow it unguarded.
                self._append_if(condition, true_arm, [], out)
                out.extend(false_arm)
            else:
                self._append_if(condition, true_arm, false_arm, out)
            return None
        if true_kind == "inline":
            exit_label, exit_kind, stay_label = false_label, false_kind, true_label
            exit_condition = negate(condition)
        else:
            exit_label, exit_kind, stay_label = true_label, true_kind, false_label
            exit_condition = condition
        exit_arm = self._arm(block.label, exit_label, frames)
        if self._is_jump(exit_kind):
            self._append_if(exit_condition, exit_arm, [], out)
            return self._follow(block.label, stay_label, frames, out)
        stay_arm = self._arm(block.label, stay_label, frames)
        self._append_if(exit_condition, exit_arm, stay_arm, out)
        return None

    def _emit_loop(self, block, frames, out):
        _kind, merge, continue_target = block.merge
        loop = _Frame(
            "loop", merge=merge, header=block.label, continue_target=continue_target
        )
        inner = [*frames, loop]
        body = []
        self._active.add(block.label)
        try:
            body.extend(self.hooks.statements(block.label))
            following = self._emit_terminator(block, inner, body)
        finally:
            self._active.discard(block.label)
        self._emit_region(following, inner, body)
        out.append(_finish_loop(body))
        return self._follow(None, merge, frames, out)

    def _fallthrough_targets(self, block, targets, frames):
        merge = block.merge[1]
        stops = {merge, block.label}
        for frame in frames:
            stops.update(
                label
                for label in (frame.merge, frame.header, frame.continue_target)
                if label
            )
        falls = {}
        for target in targets:
            seen = {target}
            stack = [target]
            reached = set()
            while stack:
                label = stack.pop()
                for successor in self.cfg.blocks[label].successors:
                    if successor in stops or successor in seen:
                        continue
                    if successor in targets and successor != target:
                        reached.add(successor)
                        continue
                    seen.add(successor)
                    stack.append(successor)
            if len(reached) > 1:
                raise SpirvUnstructuredControlFlowError(
                    f"switch case {target} falls through to more than one case"
                )
            if reached:
                falls[target] = reached.pop()
        return falls

    def _case_order(self, targets, falls):
        """``targets`` in ``OpSwitch`` operand order, each case that falls through followed by the case it enters."""
        fallen_into = {}
        for source, destination in falls.items():
            if destination in fallen_into:
                raise SpirvUnstructuredControlFlowError(
                    f"two switch cases fall through to case {destination}"
                )
            fallen_into[destination] = source
        order = []
        for target in targets:
            if target in fallen_into:
                continue
            current = target
            while current is not None and current not in order:
                order.append(current)
                current = falls.get(current)
        if len(order) != len(targets):
            raise SpirvUnstructuredControlFlowError(
                "switch cases fall through to each other in a cycle"
            )
        return order

    def _emit_switch(self, block, frames, out):
        operands = block.terminator[1]
        selector_id, default_label = operands[0], operands[1]
        pairs = list(zip(operands[2::2], operands[3::2]))
        merge = block.merge[1]
        targets = list(
            dict.fromkeys(
                [label for _literal, label in pairs if label != merge]
                + ([default_label] if default_label != merge else [])
            )
        )
        falls = self._fallthrough_targets(block, set(targets), frames)
        order = self._case_order(targets, falls)
        fallen_into = set(falls.values())
        for target in order:
            if target in fallen_into:
                # Copies at the case's start would also run on fall-through, which assigns these phis itself, so
                # the switch edge assigns them before the switch: only this case reads them.
                out.extend(self.edge_copies(block.label, target))
        cases = []
        for target in order:
            frame = _Frame(
                "switch",
                merge=merge,
                next_case=falls.get(target),
                case_targets=frozenset(targets),
            )
            case_frames = [*frames, frame]
            body = (
                [] if target in fallen_into else self.edge_copies(block.label, target)
            )
            self._emit_region(self._enter(target, case_frames), case_frames, body)
            self._append_case_group(
                cases,
                [literal for literal, label in pairs if label == target],
                default_label == target,
                body,
            )
        merge_frame = _Frame("switch", merge=merge, case_targets=frozenset(targets))
        merge_body = self._arm(block.label, merge, [*frames, merge_frame])
        merge_literals = [literal for literal, label in pairs if label == merge]
        default_is_merge = default_label == merge
        if not (default_is_merge and len(merge_body) == 1):
            self._append_case_group(cases, merge_literals, default_is_merge, merge_body)
        out.append(SwitchNode(self.hooks.expression(selector_id), cases))
        return self._follow(None, merge, frames, out)

    def _append_case_group(self, cases, literals, is_default, body):
        labels = [*literals, None] if is_default else list(literals)
        if not labels:
            return
        for literal in labels[:-1]:
            cases.append(CaseNode(literal, []))
        cases.append(CaseNode(labels[-1], body))

    def edge_copies(self, pred, target):
        """The assignments to ``target``'s variable phis on its incoming edge from ``pred``."""
        group = [
            (phi_id, incoming[pred])
            for phi_id, incoming in self.variable_phis.get(target, [])
            if pred in incoming and incoming[pred] != phi_id
        ]
        if not group:
            return []
        values = [self.hooks.expression(value) for _phi_id, value in group]
        copies = []
        sources = []
        assigned = set()
        for (phi_id, _value), value in zip(group, values):
            # The copies run in order, so one that reads a phi an earlier copy assigns reads it from a temporary
            # taken before any of them.
            if referenced_spirv_ids(value, assigned):
                declaration, reference = self.hooks.phi_temporary(phi_id)
                copies.append(AssignmentNode(declaration, value))
                sources.append(reference)
            else:
                sources.append(value)
            assigned.add(phi_id)
        for (phi_id, _value), source in zip(group, sources):
            copies.append(AssignmentNode(self.hooks.phi_variable(phi_id), source))
        return copies


def _strip_trailing_continue(body):
    if not body:
        return body
    last = body[-1]
    if isinstance(last, ContinueNode):
        return body[:-1]
    if isinstance(last, IfNode) and not getattr(last, "else_if_chain", None):
        if_body = _strip_trailing_continue(list(last.if_body or []))
        else_body = _strip_trailing_continue(list(last.else_body or []))
        if not if_body and not else_body:
            return body[:-1]
        if not if_body:
            replacement = IfNode(negate(last.condition), else_body)
        else:
            replacement = IfNode(last.condition, if_body, else_body or None)
        return [*body[:-1], replacement]
    return body


def _finish_loop(body):
    body = _strip_trailing_continue(body)
    if body:
        first = body[0]
        if (
            isinstance(first, IfNode)
            and not first.else_body
            and not getattr(first, "else_if_chain", None)
            and len(first.if_body or []) == 1
            and isinstance(first.if_body[0], BreakNode)
        ):
            return WhileNode(negate(first.condition), body[1:])
    return WhileNode("true", body)

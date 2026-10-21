"""Which SPIR-V values must be evaluated where they are defined rather than where they are used.

The importer turns each SPIR-V result into an expression and writes it out inline at every use, which is only
sound when evaluating it later gives the same value and the same effects:

* A value with effects, such as an atomic read-modify-write or a non-void function call, would run once per use, or
  not at all, and after effects that should follow it.
* A memory read (``OpLoad``, a storage image read) would observe any write to memory it may alias that runs
  between the read and the use.
* A derivative, implicitly sampled texture or subgroup operation depends on the set of invocations active where it
  executes, so it must not move into another block.

:func:`values_to_materialize` finds those values; the importer assigns each to a local where it is defined.
A use is an *emission point*: the instruction that writes the value out, or, through a value that is itself
inlined, every point where that value is written out. A phi variable's incoming value is written out on its
incoming edge, at the end of the predecessor block. The memory a read may observe is identified by its *root*: the
local, global or parameter its pointer is derived from, the device memory of storage buffers and images, or an
unknown root that aliases everything.
"""

from __future__ import annotations

__all__ = ["values_to_materialize"]

_POINTER_DERIVATIONS = frozenset(
    {
        "OpAccessChain",
        "OpInBoundsAccessChain",
        "OpPtrAccessChain",
        "OpInBoundsPtrAccessChain",
        "OpCopyObject",
    }
)
# Instructions that take pointer operands without writing through them.
_POINTER_READERS = _POINTER_DERIVATIONS | {
    "OpLoad",
    "OpArrayLength",
    "OpPtrEqual",
    "OpPtrNotEqual",
    "OpPtrDiff",
    "OpImageTexelPointer",
    "OpSelect",
    "OpPhi",
    "OpVariable",
    "OpAtomicLoad",
    "OpCooperativeMatrixLoadKHR",
}
_READ_OPCODES = frozenset(
    {"OpLoad", "OpImageRead", "OpImageSparseRead", "OpCooperativeMatrixLoadKHR"}
)
_EFFECT_OPCODES = frozenset(
    {"OpFunctionCall", "OpReportIntersectionKHR", "OpRayQueryProceedKHR"}
)
_BARRIERS = frozenset({"OpControlBarrier", "OpMemoryBarrier", "OpMemoryNamedBarrier"})
_EMITS = frozenset(
    {"OpEmitVertex", "OpEndPrimitive", "OpEmitStreamVertex", "OpEndStreamPrimitive"}
)
_READONLY_STORAGE = frozenset({"Input", "UniformConstant", "PushConstant"})
_DEVICE_STORAGE = frozenset({"StorageBuffer", "PhysicalStorageBuffer"})

READONLY = ("readonly",)
DEVICE = ("device",)
IMAGE = ("image",)
UNKNOWN = ("unknown",)


def _is_convergent(opcode):
    return (
        opcode.startswith(("OpGroup", "OpSubgroup", "OpDPd", "OpFwidth"))
        or "ImplicitLod" in opcode
        or opcode == "OpImageQueryLod"
    )


class _Module:
    """Pointer roots, value types and value kinds of one function."""

    def __init__(self, cfg, types, decorations, global_variables, parameters):
        self.cfg = cfg
        self.types = types
        self.decorations = decorations
        self.globals = {
            variable["id"]: variable.get("storage_class")
            for variable in global_variables
        }
        self.value_type_ids = dict(parameters)
        self.parameters = set(self.value_type_ids)
        self.definitions = {}
        self.locals = set()
        for label in cfg.order:
            for index, (result_id, opcode, operands) in enumerate(
                cfg.blocks[label].instructions
            ):
                if not result_id:
                    continue
                self.definitions[result_id] = (label, index, opcode, operands)
                if opcode == "OpVariable" and len(operands) >= 2:
                    if operands[1] == "Function":
                        self.locals.add(result_id)
                    self.value_type_ids[result_id] = operands[0]
                elif operands:
                    self.value_type_ids.setdefault(result_id, operands[0])
        self._roots = {}

    def is_pointer(self, value_id):
        type_id = self.value_type_ids.get(value_id)
        return self.types.get(type_id, {}).get("kind") == "pointer"

    def _global_root(self, value_id, storage_class):
        if storage_class in _READONLY_STORAGE:
            return READONLY
        if storage_class in _DEVICE_STORAGE:
            return DEVICE
        if storage_class == "Image":
            return IMAGE
        if storage_class == "Uniform":
            pointer_type = self.types.get(self.value_type_ids.get(value_id), {})
            pointee = pointer_type.get("type_id")
            entries = self.decorations.get(pointee, []) or []
            if any(entry and entry[0] == "BufferBlock" for entry in entries):
                return DEVICE
            return READONLY
        return ("global", value_id, storage_class)

    def root(self, pointer_id):
        if pointer_id in self._roots:
            return self._roots[pointer_id]
        current = pointer_id
        for _step in range(256):
            definition = self.definitions.get(current)
            if definition is None:
                break
            _label, _index, opcode, operands = definition
            if opcode == "OpImageTexelPointer":
                self._roots[pointer_id] = IMAGE
                return IMAGE
            if opcode not in _POINTER_DERIVATIONS or len(operands) < 2:
                break
            current = operands[1]
        if current in self.locals:
            root = ("local", current)
        elif current in self.globals:
            root = self._global_root(current, self.globals[current])
        elif current in self.parameters:
            root = ("param", current)
        else:
            storage = self.types.get(self.value_type_ids.get(current), {}).get(
                "storage_class"
            )
            root = DEVICE if storage == "PhysicalStorageBuffer" else UNKNOWN
        self._roots[pointer_id] = root
        return root

    def writes(self, opcode, operands, result_id):
        """The memory effects of one instruction, as ``("store", root)``, ``("call", roots)`` and the like."""
        if opcode in _BARRIERS:
            return [("barrier",)]
        if opcode in _EMITS:
            return [("emit",)]
        if opcode == "OpImageWrite":
            return [("store", IMAGE)]
        value_operands = operands[1:] if result_id else operands
        if opcode == "OpFunctionCall":
            pointer_roots = frozenset(
                self.root(operand)
                for operand in operands[2:]
                if self.is_pointer(operand)
            )
            return [("call", pointer_roots)]
        if opcode in _POINTER_READERS:
            return []
        if opcode in {"OpStore", "OpCopyMemory", "OpCopyMemorySized"} and operands:
            return [("store", self.root(operands[0]))]
        return [
            ("store", self.root(operand))
            for operand in value_operands
            if isinstance(operand, str) and self.is_pointer(operand)
        ]

    def read_root(self, opcode, operands):
        if opcode in {"OpImageRead", "OpImageSparseRead"}:
            return IMAGE
        if opcode.startswith("OpRayQueryGet") or opcode in _READ_OPCODES:
            return self.root(operands[1]) if len(operands) >= 2 else UNKNOWN
        return None


def _aliases(write, root):
    if root == READONLY:
        return False
    kind = write[0]
    if kind == "barrier":
        return root[0] in {"device", "image", "unknown", "param"} or (
            root[0] == "global" and root[2] == "Workgroup"
        )
    if kind == "emit":
        return root[0] in {"unknown", "param"} or (
            root[0] == "global" and root[2] == "Output"
        )
    if kind == "call":
        return root[0] != "local" or root in write[1]
    target = write[1]
    if target == READONLY:
        return False
    if target == root or UNKNOWN in (target, root):
        return True
    device_like = {"device", "image"}
    if target[0] in device_like and root[0] in device_like:
        return True
    if target[0] == "param":
        return root[0] != "local"
    if root[0] == "param":
        return target[0] != "local"
    return False


class _Paths:
    """Writes that may run on some path from one point to another without passing the first point again."""

    def __init__(self, cfg, block_writes):
        self.cfg = cfg
        self.block_writes = block_writes
        self._forward = {}
        self._between = {}

    def _reach(self, start_labels, avoid, edges):
        seen = set()
        stack = [label for label in start_labels if label != avoid]
        while stack:
            label = stack.pop()
            if label in seen:
                continue
            seen.add(label)
            stack.extend(
                successor
                for successor in edges(label)
                if successor != avoid and successor not in seen
            )
        return seen

    def _middle(self, source, target):
        key = (source, target)
        if key not in self._between:
            if source not in self._forward:
                self._forward[source] = self._reach(
                    self.cfg.blocks[source].successors,
                    source,
                    lambda label: self.cfg.blocks[label].successors,
                )
            forward = self._forward[source]
            backward = self._reach(
                self.cfg.predecessors.get(target, []),
                source,
                lambda label: self.cfg.predecessors.get(label, []),
            )
            middle = forward & backward
            writes = [
                write
                for label in sorted(middle - {target}, key=self._index)
                for _index, write in self.block_writes.get(label, [])
            ]
            self._between[key] = (writes, target in middle)
        return self._between[key]

    def _index(self, label):
        return self.cfg.blocks[label].index

    def writes_between(self, start, end):
        (source, start_index), (target, end_index) = start, end
        if source == target and start_index < end_index:
            return [
                write
                for index, write in self.block_writes.get(source, [])
                if start_index < index < end_index
            ]
        middle, target_in_cycle = self._middle(source, target)
        after_start = [
            write
            for index, write in self.block_writes.get(source, [])
            if index > start_index
        ]
        before_end = [
            write
            for index, write in self.block_writes.get(target, [])
            if target_in_cycle or index < end_index
        ]
        if source == target:
            before_end = []
        return [*after_start, *middle, *before_end]


def _value_kind(opcode, result_type_id, types):
    if opcode == "OpPhi":
        return "phi"
    if opcode in _EFFECT_OPCODES or opcode.startswith("OpAtomic"):
        if types.get(result_type_id, {}).get("name") == "void":
            return "statement"
        return "effect"
    if _is_convergent(opcode):
        return "convergent"
    if opcode in _READ_OPCODES or opcode.startswith("OpRayQueryGet"):
        return "read"
    return "pure"


class _Analysis:
    def __init__(self, module, phi_kinds, materializable):
        self.module = module
        self.cfg = module.cfg
        self.phi_kinds = phi_kinds
        self.materializable = materializable
        self.users = {}
        self.block_writes = {}
        self.kinds = {}
        self._collect()
        self.paths = _Paths(self.cfg, self.block_writes)
        self.points = {}
        self.forced = set()

    def _collect(self):
        module = self.module
        for label in self.cfg.order:
            block = self.cfg.blocks[label]
            writes = []
            for index, (result_id, opcode, operands) in enumerate(block.instructions):
                if result_id:
                    kind = _value_kind(
                        opcode, operands[0] if operands else None, module.types
                    )
                    if opcode == "OpExtInst" and any(
                        isinstance(operand, str) and module.is_pointer(operand)
                        for operand in operands[3:]
                    ):
                        kind = "effect"
                    self.kinds[result_id] = kind
                for write in module.writes(opcode, operands, result_id):
                    writes.append((index, write))
                self._record_uses(label, index, result_id, opcode, operands)
            if writes:
                self.block_writes[label] = writes

    def _record_uses(self, label, index, result_id, opcode, operands):
        definitions = self.module.definitions
        if opcode == "OpPhi" and result_id:
            kind = self.phi_kinds.get(result_id, ("variable",))
            if kind[0] == "select" and kind[1] in definitions:
                # The selection's condition is evaluated again where the conditional expression is written out.
                self.users.setdefault(kind[1], []).append(("value", result_id))
            for operand_index in range(1, len(operands) - 1, 2):
                value, pred = operands[operand_index], operands[operand_index + 1]
                if value in definitions:
                    self.users.setdefault(value, []).append(("phi", result_id, pred))
            return
        user = ("value", result_id) if result_id else ("point", (label, index))
        for operand in operands[1:] if result_id else operands:
            if isinstance(operand, str) and operand in definitions:
                self.users.setdefault(operand, []).append(user)

    def _definition_point(self, value_id):
        label, index, _opcode, _operands = self.module.definitions[value_id]
        return (label, index)

    def _user_points(self, user):
        kind = user[0]
        if kind == "point":
            return {user[1]}
        if kind == "phi":
            _kind, phi_id, pred = user
            if self.phi_kinds.get(phi_id, ("variable",))[0] == "variable":
                pred_block = self.cfg.blocks.get(pred)
                return {(pred, len(pred_block.instructions))} if pred_block else set()
            return self.emission_points(phi_id)
        return self.emission_points(user[1])

    def emission_points(self, value_id):
        """Where ``value_id``'s expression is written out, deciding first whether it is materialized."""
        if value_id in self.points:
            return self.points[value_id]
        self.points[value_id] = {self._definition_point(value_id)}
        points = set()
        for user in self.users.get(value_id, []):
            points |= self._user_points(user)
        if self._must_materialize(value_id, points):
            self.forced.add(value_id)
            points = {self._definition_point(value_id)}
        elif self.kinds.get(value_id) in {"effect", "statement"}:
            points = {self._definition_point(value_id)}
        self.points[value_id] = points
        return points

    def _must_materialize(self, value_id, points):
        kind = self.kinds.get(value_id)
        if not points or kind in {"pure", "phi", "statement", None}:
            return False
        label, index, opcode, operands = self.module.definitions[value_id]
        if not self.materializable(self.module.value_type_ids.get(value_id)):
            return False
        if kind == "effect":
            return not self._used_once_next(value_id)
        if kind == "convergent":
            return any(point[0] != label for point in points)
        root = self.module.read_root(opcode, operands)
        if root is None or root == READONLY:
            return False
        start = (label, index)
        return any(
            _aliases(write, root)
            for point in sorted(
                points, key=lambda point: (self.cfg.blocks[point[0]].index, point[1])
            )
            for write in self.paths.writes_between(start, point)
        )

    def _used_once_next(self, value_id):
        """Whether a scalar or vector effect is written out once, later in its block, with only pure values between.

        Such an effect can be evaluated where it is written out: nothing that runs in between observes or changes
        memory, so moving it there keeps its order relative to every other effect and read.
        """
        type_id = self.module.value_type_ids.get(value_id)
        if self.module.types.get(type_id, {}).get("kind") not in {"scalar", "vector"}:
            return False
        current = value_id
        while True:
            users = self.users.get(current, [])
            if len(users) != 1:
                return False
            user = users[0]
            if user[0] == "point":
                point = user[1]
                break
            if user[0] != "value" or self.kinds.get(user[1]) != "pure":
                return False
            current = user[1]
        label, index, _opcode, _operands = self.module.definitions[value_id]
        if point[0] != label or point[1] <= index:
            return False
        if any(
            index < write_index < point[1]
            for write_index, _write in self.block_writes.get(label, [])
        ):
            return False
        instructions = self.cfg.blocks[label].instructions
        return all(
            not result_id or self.kinds.get(result_id) == "pure"
            for result_id, _opcode, _operands in instructions[index + 1 : point[1]]
        )

    def run(self):
        for label in reversed(self.cfg.order):
            for result_id, _opcode, _operands in reversed(
                self.cfg.blocks[label].instructions
            ):
                if result_id and result_id in self.kinds:
                    self.emission_points(result_id)
        return self.forced


def values_to_materialize(
    cfg,
    phi_kinds,
    types,
    decorations,
    global_variables,
    materializable,
    parameters=(),
):
    """The ids of the function's values that must be assigned to a local where they are defined.

    ``materializable(type_id)`` says whether a value of that type can be held in a CrossGL local; ``parameters`` are the
    function's ``(parameter id, type id)`` pairs.
    """
    module = _Module(cfg, types, decorations, global_variables, parameters)
    return _Analysis(module, phi_kinds, materializable).run()

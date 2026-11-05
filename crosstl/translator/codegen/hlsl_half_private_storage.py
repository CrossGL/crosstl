"""Keep binary16 values in DirectX workgroup and private arrays as raw bits.

The D3D11.3 functional specification lets every operation except a raw move
change the bit pattern of a NaN (section 3.1.3.2), and WARP does change it for
typed 16-bit floating loads, stores and selections. HLSL output already keeps
binary16 buffer elements as ``uint16_t`` (``hlsl_half_storage``). This pass does
the same for arrays in groupshared and function memory and for the threadgroup
and thread pointers into them:

- each such array, pointer or array parameter is declared with ``uint16``
  elements;
- an element read becomes ``as_type<float16>(element)`` and an element write
  ``element = as_type<uint16>(value)``, so a copy is an integer move;
- a selection between stored elements and binary16 values selects the bits;
- an element passed to an ``inout`` or ``out`` binary16 parameter goes through a
  typed temporary that is copied back as bits.

Arithmetic and comparisons keep their binary16 types.

Storage slots that are copied, passed or pointed through one another form one
group with one representation. If any slot of a group is used in a way this
pass does not rewrite, the whole group keeps its typed representation, which is
what the generator wrote before.
"""

import re
from copy import copy

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    AssignmentNode,
    AtomicOpNode,
    BinaryOpNode,
    BlockNode,
    BufferOpNode,
    CastNode,
    CooperativeMatrixOpNode,
    ExpressionStatementNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LambdaNode,
    LiteralNode,
    MemberAccessNode,
    MeshOpNode,
    NamedType,
    ParameterNode,
    PointerType,
    PrimitiveType,
    RayQueryOpNode,
    RayTracingOpNode,
    StructNode,
    SwizzleNode,
    TernaryOpNode,
    TextureOpNode,
    TypeNode,
    UnaryOpNode,
    VariableNode,
    VectorType,
)
from ..semantics.rewrite import child_slots, clone, get_child, set_child

__all__ = ["lower_half_private_storage"]

_POINTER_SPACES = frozenset(
    {
        "threadgroup",
        "workgroup",
        "groupshared",
        "shared",
        "thread",
        "private",
        "function",
    }
)
_WORKGROUP_QUALIFIERS = frozenset({"threadgroup", "workgroup", "groupshared", "shared"})
_EXCLUDED_QUALIFIERS = frozenset(
    {
        "const",
        "constant",
        "static",
        "uniform",
        "buffer",
        "in",
        "out",
        "inout",
        "push_constant",
        "constexpr",
    }
)
_OUT_QUALIFIERS = frozenset({"inout", "out"})
_COMPOUND = {"+=": "+", "-=": "-", "*=": "*", "/=": "/"}
_COMPARISONS = frozenset({"==", "!=", "<", "<=", ">", ">="})
_PURE_BINARY = frozenset(
    {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "&&", "||"} | _COMPARISONS
)
_PURE_UNARY = frozenset({"+", "-", "~", "!"})
_INTEGER_CONVERSIONS = frozenset(
    {
        "int",
        "uint",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "short",
        "ushort",
        "long",
        "ulong",
        "bool",
    }
)
# Built-ins that write through an argument.
_OUT_ARGUMENT_BUILTINS = frozenset(
    {
        "modf",
        "frexp",
        "sincos",
        "umulExtended",
        "imulExtended",
        "uaddCarry",
        "usubBorrow",
    }
)
# Operation nodes that may write or take the address of their operands.
_OPAQUE_OPERATIONS = (
    AtomicOpNode,
    BufferOpNode,
    TextureOpNode,
    CooperativeMatrixOpNode,
    RayTracingOpNode,
    RayQueryOpNode,
    MeshOpNode,
    LambdaNode,
)
_HALF = re.compile(r"float16_t([234]?)")


def _value_type_name(width):
    return "float16" if width == 1 else f"f16vec{width}"


def _bits_type_name(width):
    return "uint16" if width == 1 else f"u16vec{width}"


def _call_name(node):
    function = getattr(node, "function", None)
    if isinstance(function, IdentifierNode):
        return function.name
    return function if isinstance(function, str) else None


class _Slot:
    """An array, pointer or array parameter that may hold binary16 elements."""

    __slots__ = ("node", "kind", "width", "depth")

    def __init__(self, node, kind, width, depth):
        self.node = node
        self.kind = kind
        self.width = width
        self.depth = depth


class _Scope:
    """Flat name resolution for one function: its declarations, then the stage's, then the program's."""

    def __init__(self, local, outer):
        self.local = local
        self.outer = outer

    def get(self, name):
        if name in self.local:
            return self.local[name]
        return self.outer.get(name)


class _Pass:
    def __init__(self, program, map_type):
        self.program = program
        self.map_type = map_type
        self.slots = {}
        self.parent = {}
        self.poisoned = set()
        self.functions = {}
        self.entries = set()
        self.names = set()
        self.decoded = {}
        self.lowered_roots = None
        self.scope = None
        self.temp_index = 0

    # -- types -------------------------------------------------------------

    def width(self, vtype):
        """1-4 when ``vtype`` is a native binary16 scalar or vector, else None."""
        if isinstance(vtype, VectorType):
            if not isinstance(vtype.size, int) or self.width(vtype.element_type) != 1:
                return None
            return vtype.size if 2 <= vtype.size <= 4 else None
        if not isinstance(vtype, (NamedType, PrimitiveType, str)):
            return None
        if isinstance(vtype, NamedType) and vtype.generic_args:
            return None
        try:
            mapped = str(self.map_type(vtype))
        except (ValueError, TypeError, AttributeError, KeyError):
            return None
        match = _HALF.fullmatch(mapped)
        if match is None:
            return None
        return int(match.group(1) or 1)

    def slot_for(self, node, *, parameter, workgroup_only=False):
        vtype = node.param_type if parameter else getattr(node, "var_type", None)
        if vtype is None or getattr(node, "attributes", None):
            return None
        qualifiers = set(getattr(node, "qualifiers", None) or [])
        if isinstance(vtype, PointerType):
            width = self.width(vtype.pointee_type)
            if width is None or vtype.address_space not in _POINTER_SPACES:
                return None
            if not parameter and qualifiers & (_EXCLUDED_QUALIFIERS - {"const"}):
                return None
            return _Slot(node, "pointer", width, 1)
        depth = 0
        element = vtype
        while isinstance(element, ArrayType):
            if element.size is None:
                return None
            element = element.element_type
            depth += 1
        width = self.width(element)
        if depth == 0 or width is None:
            return None
        if not parameter:
            if getattr(node, "initial_value", None) is not None:
                return None
            if qualifiers & _EXCLUDED_QUALIFIERS:
                return None
            if workgroup_only and not qualifiers & _WORKGROUP_QUALIFIERS:
                return None
        return _Slot(node, "array", width, depth)

    # -- groups ------------------------------------------------------------

    def add_slot(self, slot):
        if slot is not None:
            self.slots[id(slot.node)] = slot
            self.parent[id(slot.node)] = id(slot.node)

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, slots):
        slots = list(slots)
        for slot in slots[1:]:
            if slot.width != slots[0].width:
                self.poison(slot)
                self.poison(slots[0])
            self.parent[self.find(id(slot.node))] = self.find(id(slots[0].node))

    def poison(self, slot):
        self.poisoned.add(id(slot.node))

    def lowered(self, slot):
        return self.find(id(slot.node)) in self.lowered_roots

    # -- program structure ---------------------------------------------------

    def function_scopes(self):
        """Yield ``(function, outer)`` for every function body with its enclosing declarations."""
        program_scope = {}
        for variable in getattr(self.program, "global_variables", None) or []:
            if getattr(variable, "name", None):
                program_scope[variable.name] = variable
        for function in getattr(self.program, "functions", None) or []:
            if isinstance(function, FunctionNode):
                yield function, program_scope
        stages = getattr(self.program, "stages", None) or {}
        stage_list = list(stages.values()) if isinstance(stages, dict) else list(stages)
        for stage in stage_list:
            stage_scope = dict(program_scope)
            for variable in getattr(stage, "local_variables", None) or []:
                if getattr(variable, "name", None):
                    stage_scope[variable.name] = variable
            for function in getattr(stage, "local_functions", None) or []:
                if isinstance(function, FunctionNode):
                    yield function, stage_scope
            entry = getattr(stage, "entry_point", None)
            if isinstance(entry, FunctionNode):
                self.entries.add(id(entry))
                yield entry, stage_scope

    def declarations(self, node, found):
        if node is None or isinstance(node, (FunctionNode, StructNode, TypeNode)):
            return
        if isinstance(node, VariableNode) and node.name:
            found.append(node)
        for field, key in child_slots(node):
            self.declarations(get_child(node, field, key), found)

    def collect(self):
        self.bodies = []
        for function, outer in self.function_scopes():
            if function.name:
                self.functions.setdefault(function.name, []).append(function)
        global_slots = {}
        for variable in getattr(self.program, "global_variables", None) or []:
            slot = self.slot_for(variable, parameter=False, workgroup_only=True)
            self.add_slot(slot)
            if slot is not None:
                global_slots[variable.name] = slot
        stages = getattr(self.program, "stages", None) or {}
        for stage in list(stages.values()) if isinstance(stages, dict) else stages:
            for variable in getattr(stage, "local_variables", None) or []:
                slot = self.slot_for(variable, parameter=False, workgroup_only=True)
                self.add_slot(slot)
                if slot is not None:
                    global_slots[variable.name] = slot
        for function, outer in self.function_scopes():
            if getattr(function, "body", None) is None:
                continue
            entry = id(function) in self.entries or any(
                getattr(attribute, "name", None) == "stage_entry"
                for attribute in getattr(function, "attributes", None) or []
            )
            local, counts = {}, {}
            for parameter in function.parameters or []:
                local[parameter.name] = parameter
                counts[parameter.name] = counts.get(parameter.name, 0) + 1
                if not entry and not getattr(function, "generic_params", None):
                    self.add_slot(self.slot_for(parameter, parameter=True))
            found = []
            self.declarations(function.body, found)
            for variable in found:
                local[variable.name] = variable
                counts[variable.name] = counts.get(variable.name, 0) + 1
                self.add_slot(self.slot_for(variable, parameter=False))
            for name, count in counts.items():
                # One name declared twice in a body is resolved by position, which this pass does not track.
                if count > 1 or name in global_slots:
                    for node in [*(function.parameters or []), *found]:
                        if node.name == name and id(node) in self.slots:
                            self.poison(self.slots[id(node)])
                    if name in global_slots:
                        self.poison(global_slots[name])
            self.bodies.append((function, _Scope(local, outer)))
        for node in _walk_all(self.program):
            name = getattr(node, "name", None)
            if isinstance(name, str):
                self.names.add(name)

    # -- resolution --------------------------------------------------------

    def lookup(self, node):
        if not isinstance(node, IdentifierNode):
            return None
        declaration = self.scope.get(node.name)
        return self.slots.get(id(declaration)) if declaration is not None else None

    def declared_width(self, node):
        """Width of a binary16 scalar or vector named by ``node``, if it is one."""
        if isinstance(node, IdentifierNode):
            declaration = self.scope.get(node.name)
            if declaration is None or id(declaration) in self.slots:
                return None
            vtype = (
                declaration.param_type
                if isinstance(declaration, ParameterNode)
                else getattr(declaration, "var_type", None)
            )
            return self.width(vtype)
        return None

    def pointer_targets(self, node):
        """Slots a pointer-valued expression points into, or None."""
        if isinstance(node, IdentifierNode):
            slot = self.lookup(node)
            if slot is not None and (slot.kind == "pointer" or slot.depth == 1):
                return [slot]
            return None
        if isinstance(node, BinaryOpNode) and node.operator in {"+", "-"}:
            left = self.pointer_targets(node.left)
            right = self.pointer_targets(node.right)
            if left is not None and right is None:
                return left
            if node.operator == "+" and right is not None and left is None:
                return right
            return None
        if isinstance(node, UnaryOpNode) and node.operator == "&":
            return self.element(node.operand)
        if isinstance(node, TernaryOpNode):
            yes = self.pointer_targets(node.true_expr)
            no = self.pointer_targets(node.false_expr)
            if yes is not None and no is not None:
                return yes + no
        return None

    def element(self, node):
        """Slots whose binary16 element ``node`` designates, or None."""
        if isinstance(node, ArrayAccessNode):
            base, depth = node, 0
            while isinstance(base, ArrayAccessNode):
                base = base.array_expr
                depth += 1
            slot = self.lookup(base)
            if slot is not None and slot.kind == "array":
                return [slot] if slot.depth == depth else None
            return self.pointer_targets(base) if depth == 1 else None
        if (
            isinstance(node, UnaryOpNode)
            and node.operator == "*"
            and not node.is_postfix
        ):
            return self.pointer_targets(node.operand)
        return None

    def pure(self, node, local_only):
        """True if evaluating ``node`` twice gives the same value and has no effect."""
        if isinstance(node, LiteralNode):
            return True
        if isinstance(node, IdentifierNode):
            if not local_only:
                return True
            return node.name in self.scope.local
        if isinstance(node, ArrayAccessNode):
            return self.pure(node.array_expr, local_only) and self.pure(
                node.index_expr, local_only
            )
        if isinstance(node, (MemberAccessNode, SwizzleNode)):
            return self.pure(
                getattr(node, "object_expr", getattr(node, "vector_expr", None)),
                local_only,
            )
        if isinstance(node, UnaryOpNode):
            return (
                node.operator in _PURE_UNARY | {"*", "&"}
                and not node.is_postfix
                and self.pure(node.operand, local_only)
            )
        if isinstance(node, BinaryOpNode):
            return node.operator in _PURE_BINARY and all(
                self.pure(side, local_only) for side in (node.left, node.right)
            )
        if isinstance(node, TernaryOpNode):
            return all(
                self.pure(part, local_only)
                for part in (node.condition, node.true_expr, node.false_expr)
            )
        if isinstance(node, CastNode):
            return self.pure(node.expression, local_only)
        if isinstance(node, FunctionCallNode):
            return (
                _call_name(node) in _INTEGER_CONVERSIONS
                and _call_name(node) not in self.functions
                and all(self.pure(argument, local_only) for argument in node.arguments)
            )
        return False

    def callee(self, node):
        name = _call_name(node)
        if name is None:
            return None, None
        candidates = self.functions.get(name)
        if not candidates:
            return name, None
        if len(candidates) == 1 and len(candidates[0].parameters or []) == len(
            node.arguments
        ):
            return name, candidates[0]
        return name, False


class _Analysis(_Pass):
    """Group the slots and poison every group with a use the rewrite does not cover."""

    def run(self):
        self.collect()
        for function, scope in self.bodies:
            self.scope = scope
            self.visit(function.body, "value")
        poisoned_roots = {self.find(key) for key in self.poisoned}
        self.lowered_roots = {self.find(key) for key in self.slots} - poisoned_roots

    def visit(self, node, ctx):
        if node is None or isinstance(node, (FunctionNode, StructNode, TypeNode)):
            return
        if isinstance(node, list):
            for item in node:
                self.visit(item, ctx)
            return
        targets = self.element(node)
        if targets is not None:
            self.use_element(node, targets, ctx)
            return
        targets = self.pointer_targets(node)
        if targets is not None:
            self.use_pointer(node, targets, ctx)
            return
        if isinstance(node, IdentifierNode):
            slot = self.lookup(node)
            if slot is not None:
                self.poison(slot)
            return
        handler = getattr(self, "visit_" + type(node).__name__, None)
        if handler is not None:
            handler(node, ctx)
            return
        child_ctx = "lvalue" if isinstance(node, _OPAQUE_OPERATIONS) else "value"
        for field, key in child_slots(node):
            self.visit(get_child(node, field, key), child_ctx)

    def use_element(self, node, targets, ctx):
        self.union(targets)
        if ctx == "compound":
            accepted = self.pure(node, local_only=False)
        elif ctx == "out-argument":
            accepted = self.pure(node, local_only=True)
        else:
            accepted = ctx in {"value", "assign"}
        if not accepted:
            for slot in targets:
                self.poison(slot)
        self.visit_parts(node)

    def use_pointer(self, node, targets, ctx):
        if isinstance(ctx, tuple):
            self.union([*targets, ctx[1]])
        elif ctx != "compare":
            for slot in targets:
                self.poison(slot)
        self.visit_parts(node)

    def visit_parts(self, node):
        """Visit the index and offset operands of an element or pointer expression."""
        if isinstance(node, ArrayAccessNode):
            self.visit(node.index_expr, "value")
            if not isinstance(node.array_expr, IdentifierNode):
                self.visit_parts(node.array_expr)
        elif isinstance(node, UnaryOpNode):
            self.visit_parts(node.operand)
        elif isinstance(node, BinaryOpNode):
            for side in (node.left, node.right):
                if self.pointer_targets(side) is not None:
                    self.visit_parts(side)
                else:
                    self.visit(side, "value")
        elif isinstance(node, TernaryOpNode):
            self.visit(node.condition, "value")
            self.visit_parts(node.true_expr)
            self.visit_parts(node.false_expr)

    def into(self, value, slot):
        targets = self.pointer_targets(value)
        if targets is None:
            self.poison(slot)
            self.visit(value, "value")
            return
        self.use_pointer(value, targets, ("into", slot))

    def visit_VariableNode(self, node, ctx):
        slot = self.slots.get(id(node))
        if node.initial_value is None:
            return
        if slot is not None and slot.kind == "pointer":
            self.into(node.initial_value, slot)
        else:
            self.visit(node.initial_value, "value")

    def visit_ExpressionStatementNode(self, node, ctx):
        self.visit(node.expression, "statement")

    def visit_ForNode(self, node, ctx):
        for part in (node.init, node.update):
            parts = part if isinstance(part, list) else [part]
            for item in parts:
                self.visit(
                    item, "statement" if isinstance(item, AssignmentNode) else "value"
                )
        self.visit(node.condition, "value")
        self.visit(node.body, "value")

    def visit_AssignmentNode(self, node, ctx):
        statement = ctx == "statement"
        target, value, operator = node.target, node.value, node.operator
        targets = self.element(target)
        if targets is not None:
            if operator == "=" and statement:
                use = "assign"
            elif operator in _COMPOUND and statement:
                use = "compound"
            else:
                use = "lvalue"
            self.use_element(target, targets, use)
            self.visit(value, "value")
            return
        slot = self.lookup(target)
        if slot is None:
            self.visit_lvalue(target)
            self.visit(value, "value")
            return
        if slot.kind == "pointer" and operator == "=":
            self.into(value, slot)
            return
        if slot.kind == "pointer" and operator in {"+=", "-="}:
            self.visit(value, "value")
            return
        other = self.lookup(value)
        if (
            slot.kind == "array"
            and operator == "="
            and statement
            and other is not None
            and other.kind == "array"
            and other.depth == slot.depth
        ):
            self.union([slot, other])
            return
        self.poison(slot)
        self.visit(value, "value")

    def visit_lvalue(self, node):
        """Visit the target of a write that is not an element of a slot."""
        if isinstance(node, (MemberAccessNode, SwizzleNode)):
            base = getattr(node, "object_expr", getattr(node, "vector_expr", None))
            self.visit_lvalue(base)
        elif isinstance(node, ArrayAccessNode):
            self.visit(node.index_expr, "value")
            self.visit_lvalue(node.array_expr)
        else:
            self.visit(node, "lvalue")

    def visit_MemberAccessNode(self, node, ctx):
        self.visit(node.object_expr, "lvalue" if ctx == "lvalue" else "value")

    def visit_SwizzleNode(self, node, ctx):
        self.visit(node.vector_expr, "lvalue" if ctx == "lvalue" else "value")

    def visit_UnaryOpNode(self, node, ctx):
        if node.operator in {"++", "--"}:
            slot = self.lookup(node.operand)
            if slot is not None and slot.kind == "pointer":
                return
            self.visit(node.operand, "lvalue")
            return
        if node.operator == "&":
            self.visit(node.operand, "lvalue")
            return
        self.visit(node.operand, "value")

    def visit_BinaryOpNode(self, node, ctx):
        for side in (node.left, node.right):
            targets = (
                self.pointer_targets(side) if node.operator in _COMPARISONS else None
            )
            if targets is not None:
                self.use_pointer(side, targets, "compare")
            else:
                self.visit(side, "value")

    def visit_TernaryOpNode(self, node, ctx):
        self.visit(node.condition, "value")
        arm_ctx = "value" if ctx in {"value", "statement"} else "lvalue"
        self.visit(node.true_expr, arm_ctx)
        self.visit(node.false_expr, arm_ctx)

    def visit_FunctionCallNode(self, node, ctx):
        name, function = self.callee(node)
        arguments = node.arguments
        if function is None and name is not None:
            argument_ctx = "lvalue" if name in _OUT_ARGUMENT_BUILTINS else "value"
            if name.startswith(("atomic", "Interlocked")):
                argument_ctx = "lvalue"
            for argument in arguments:
                self.visit(argument, argument_ctx)
            return
        if not function:
            for argument in arguments:
                self.visit(argument, "lvalue")
            return
        generic = bool(getattr(function, "generic_params", None))
        written = set()
        out_elements = []
        for argument, parameter in zip(arguments, function.parameters):
            if set(parameter.qualifiers or []) & _OUT_QUALIFIERS and isinstance(
                argument, IdentifierNode
            ):
                written.add(argument.name)
        for argument, parameter in zip(arguments, function.parameters):
            slot = None if generic else self.slots.get(id(parameter))
            out = bool(set(parameter.qualifiers or []) & _OUT_QUALIFIERS)
            if slot is not None:
                self.pass_to_slot(argument, slot)
                continue
            targets = self.element(argument)
            if targets is not None and out:
                use = "out-argument" if ctx == "statement" else "lvalue"
                if self.width(parameter.param_type) is None and not generic:
                    use = "lvalue"
                self.use_element(argument, targets, use)
                out_elements.append((argument, targets))
                continue
            self.visit(argument, "lvalue" if out else "value")
        for argument, targets in out_elements:
            used = {
                item.name
                for item in _walk_all(argument)
                if isinstance(item, IdentifierNode)
            }
            if used & written:
                for slot in targets:
                    self.poison(slot)

    def pass_to_slot(self, argument, slot):
        if slot.kind == "pointer":
            self.into(argument, slot)
            return
        other = self.lookup(argument)
        if other is None or other.kind != "array" or other.depth != slot.depth:
            self.poison(slot)
            self.visit(argument, "value")
            return
        self.union([slot, other])


class _Rewrite(_Pass):
    """Rewrite the declarations and uses of the lowered slots."""

    def __init__(self, analysis):
        self.__dict__.update(analysis.__dict__)

    def run(self):
        for function, scope in self.bodies:
            self.scope = scope
            function.body = self.rewrite(function.body, "value")
        for slot in self.slots.values():
            if self.lowered(slot):
                self.retype(slot)
        return self.program

    def retype(self, slot):
        node = slot.node
        bits = NamedType(_bits_type_name(slot.width))

        def replace(vtype):
            if isinstance(vtype, ArrayType):
                result = copy(vtype)
                result.element_type = replace(vtype.element_type)
                return result
            if isinstance(vtype, PointerType):
                result = copy(vtype)
                result.pointee_type = bits
                return result
            return bits

        if isinstance(node, ParameterNode):
            node.param_type = replace(node.param_type)
        else:
            node.var_type = node.vtype = replace(node.var_type)

    def lowered_element(self, node):
        targets = self.element(node)
        if targets is None or not self.lowered(targets[0]):
            return None
        return targets[0]

    def decode(self, node, width):
        call = FunctionCallNode(
            IdentifierNode(f"as_type<{_value_type_name(width)}>"),
            [node],
            source_location=getattr(node, "source_location", None),
        )
        self.decoded[id(call)] = width
        return call

    def value_width(self, node):
        if id(node) in self.decoded:
            return self.decoded[id(node)]
        if isinstance(node, FunctionCallNode):
            name = _call_name(node)
            if name and name not in self.functions:
                return self.width(name)
            return None
        if isinstance(node, TernaryOpNode):
            yes = self.value_width(node.true_expr)
            return (
                yes
                if yes is not None and yes == self.value_width(node.false_expr)
                else None
            )
        return self.declared_width(node)

    def encode(self, node, width):
        if id(node) in self.decoded:
            return node.arguments[0]
        if self.value_width(node) != width:
            node = FunctionCallNode(IdentifierNode(_value_type_name(width)), [node])
        return FunctionCallNode(
            IdentifierNode(f"as_type<{_bits_type_name(width)}>"),
            [node],
            source_location=getattr(node, "source_location", None),
        )

    def rewrite(self, node, ctx):
        if node is None or isinstance(node, (FunctionNode, StructNode, TypeNode)):
            return node
        if isinstance(node, list):
            return [self.rewrite(item, ctx) for item in node]
        slot = self.lowered_element(node)
        if slot is not None:
            self.rewrite_parts(node)
            return self.decode(node, slot.width) if ctx == "value" else node
        if self.pointer_targets(node) is not None:
            self.rewrite_parts(node)
            return node
        if isinstance(node, IdentifierNode):
            return node
        handler = getattr(self, "rewrite_" + type(node).__name__, None)
        if handler is not None:
            return handler(node, ctx)
        self.rewrite_children(node)
        return node

    def rewrite_children(self, node, ctx="value"):
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            new = self.rewrite(child, ctx)
            if new is not child:
                set_child(node, field, key, new)

    def rewrite_parts(self, node):
        if isinstance(node, ArrayAccessNode):
            index = self.rewrite(node.index_expr, "value")
            if index is not node.index_expr:
                set_child(node, "index_expr", None, index)
            if not isinstance(node.array_expr, IdentifierNode):
                self.rewrite_parts(node.array_expr)
        elif isinstance(node, UnaryOpNode):
            self.rewrite_parts(node.operand)
        elif isinstance(node, BinaryOpNode):
            for field in ("left", "right"):
                side = getattr(node, field)
                if self.pointer_targets(side) is not None:
                    self.rewrite_parts(side)
                else:
                    new = self.rewrite(side, "value")
                    if new is not side:
                        set_child(node, field, None, new)
        elif isinstance(node, TernaryOpNode):
            condition = self.rewrite(node.condition, "value")
            if condition is not node.condition:
                set_child(node, "condition", None, condition)
            self.rewrite_parts(node.true_expr)
            self.rewrite_parts(node.false_expr)

    def rewrite_ExpressionStatementNode(self, node, ctx):
        expression = node.expression
        if isinstance(expression, FunctionCallNode):
            expanded = self.expand_out_arguments(node, expression)
            if expanded is not None:
                return expanded
        new = self.rewrite(expression, "statement")
        if new is not expression:
            set_child(node, "expression", None, new)
        return node

    def rewrite_ForNode(self, node, ctx):
        for field in ("init", "update"):
            part = getattr(node, field)
            if isinstance(part, list):
                setattr(
                    node,
                    field,
                    [
                        self.rewrite(
                            item,
                            (
                                "statement"
                                if isinstance(item, AssignmentNode)
                                else "value"
                            ),
                        )
                        for item in part
                    ],
                )
            elif part is not None:
                new = self.rewrite(
                    part, "statement" if isinstance(part, AssignmentNode) else "value"
                )
                if new is not part:
                    set_child(node, field, None, new)
        for field in ("condition", "body"):
            part = getattr(node, field)
            new = self.rewrite(part, "value")
            if new is not part:
                set_child(node, field, None, new)
        return node

    def rewrite_AssignmentNode(self, node, ctx):
        slot = self.lowered_element(node.target)
        if slot is None:
            self.rewrite_children(node)
            return node
        self.rewrite_parts(node.target)
        value = self.rewrite(node.value, "value")
        if node.operator in _COMPOUND:
            value = BinaryOpNode(
                self.decode(clone(node.target), slot.width),
                _COMPOUND[node.operator],
                value,
            )
            node.operator = "="
        set_child(node, "value", None, self.encode(value, slot.width))
        return node

    def rewrite_TernaryOpNode(self, node, ctx):
        self.rewrite_children(node)
        arms = (node.true_expr, node.false_expr)
        if ctx != "value" or not any(id(arm) in self.decoded for arm in arms):
            return node
        widths = {self.value_width(arm) for arm in arms}
        if len(widths) != 1 or None in widths:
            return node
        width = widths.pop()
        set_child(node, "true_expr", None, self.encode(node.true_expr, width))
        set_child(node, "false_expr", None, self.encode(node.false_expr, width))
        return self.decode(node, width)

    def expand_out_arguments(self, statement, call):
        _, function = self.callee(call)
        if not function:
            return None
        generic = bool(getattr(function, "generic_params", None))
        before, after = [], []
        for index, (argument, parameter) in enumerate(
            zip(call.arguments, function.parameters)
        ):
            qualifiers = set(parameter.qualifiers or [])
            slot = self.lowered_element(argument)
            if (
                slot is None
                or not qualifiers & _OUT_QUALIFIERS
                or (not generic and id(parameter) in self.slots)
            ):
                continue
            self.rewrite_parts(argument)
            name = self.fresh_name()
            value_type = NamedType(_value_type_name(slot.width))
            initial = (
                self.decode(clone(argument), slot.width)
                if "inout" in qualifiers
                else None
            )
            before.append(VariableNode(name, value_type, initial_value=initial))
            call.arguments[index] = IdentifierNode(name)
            after.append(
                ExpressionStatementNode(
                    AssignmentNode(
                        argument,
                        FunctionCallNode(
                            IdentifierNode(f"as_type<{_bits_type_name(slot.width)}>"),
                            [IdentifierNode(name)],
                        ),
                    )
                )
            )
        if not before:
            return None
        call.args = call.arguments
        self.rewrite_children(call)
        return BlockNode(
            [*before, statement, *after],
            source_location=getattr(statement, "source_location", None),
        )

    def fresh_name(self):
        while True:
            self.temp_index += 1
            name = f"__crossgl_half_bits{self.temp_index}"
            if name not in self.names:
                self.names.add(name)
                return name


def _walk_all(node, seen=None):
    if seen is None:
        seen = set()
    if node is None or id(node) in seen:
        return
    seen.add(id(node))
    yield node
    if isinstance(node, list):
        for item in node:
            yield from _walk_all(item, seen)
        return
    if not hasattr(node, "__dict__"):
        return
    for field, key in child_slots(node):
        yield from _walk_all(get_child(node, field, key), seen)


def lower_half_private_storage(ast, map_type):
    """Rewrite binary16 workgroup and private storage of ``ast`` to raw bits in place.

    ``ast`` must be the generator's private copy. ``map_type`` maps a CrossGL
    type to its HLSL spelling; storage is lowered only for types it maps to a
    native ``float16_t`` scalar or vector.
    """
    analysis = _Analysis(ast, map_type)
    analysis.run()
    if not analysis.lowered_roots:
        return ast
    return _Rewrite(analysis).run()

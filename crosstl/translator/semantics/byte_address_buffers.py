"""HLSL byte-address buffers as arrays of 32-bit words, for targets that have no byte-addressed buffer view.

A ``ByteAddressBuffer`` or ``RWByteAddressBuffer`` is addressed in bytes, and its ``Load``/``Store`` methods move
32-bit words: ``Load(a)`` reads the word at byte address ``a``, ``Load2`` to ``Load4`` read two to four consecutive
words, and the address must be a multiple of four (the HLSL reference pages for ``ByteAddressBuffer::Load`` and
``RWByteAddressBuffer::Store``; DXIL ``rawBufferLoad`` and SPIR-V lowerings divide it by four). CrossGL spells them
``buffer_load(b, a)`` to ``buffer_load4``, ``buffer_store(b, a, v)`` to ``buffer_store4``, the byte size query
``buffer_dimensions(b, bytes)``, and the ``Interlocked*`` methods ``atomicAdd(b, a, v)`` and
``atomicCompareExchange(b, a, compare, v)``.

GLSL and Metal have no byte-addressed view, but a storage buffer of ``uint`` words binds the same memory: an
``std430`` runtime array of ``uint`` in GLSL, a ``device uint*`` in Metal. :func:`lower_byte_address_buffers`
rewrites every byte-address buffer declaration and parameter into the matching ``StructuredBuffer<uint>`` or
``RWStructuredBuffer<uint>`` and each access into word indexing at ``uint(a) >> 2u``:

- ``buffer_load(b, a)`` becomes ``b[w]`` and ``buffer_load2(b, a)`` becomes ``uvec2(b[w], b[w + 1u])``;
- ``buffer_store(b, a, v)`` becomes ``b[w] = uint(v)``, and ``buffer_store2(b, a, v)`` a block that evaluates the
  address and value once and stores each component;
- ``buffer_dimensions(b, bytes)`` becomes ``bytes = buffer_dimensions(b) * 4u``, the word count times four;
- an atomic on ``(b, a)`` becomes the same atomic on ``b[w]`` with ``uint`` operands, and its ``uint`` result
  converts into the variable that receives it, as HLSL converts the ``original_value`` output.

``InterlockedMin`` and ``InterlockedMax`` compare as signed integers when the value is an ``int`` (DXIL selects
``IMin``/``IMax`` from the value's type), which a ``uint`` word's atomics cannot do, so with an ``int`` value they
become a compare-exchange loop on the word (:mod:`.byte_address_signed_atomics`). Everything the pass cannot lower
faithfully is reported through ``refuse(reason)`` and left in place: a rasterizer-ordered buffer, a byte-address
buffer in a structure or a return type, a use other than the accesses above or an argument of a program function, an
address, buffer index or value with side effects that a multi-word access would evaluate twice, an atomic whose
operand count does not match a byte-address form, a signed ``atomicMin``/``atomicMax`` inside a larger expression,
and one whose value is neither ``int`` nor ``uint``.
"""

from __future__ import annotations

import copy

from .. import ast as A
from .byte_address_signed_atomics import signed_ordered_atomic_replacement
from .hub_types import TypeScope, type_from_node
from .rewrite import child_slots, clone, get_child, is_pure, set_child

__all__ = ["BYTE_ADDRESS_BUFFER_KINDS", "lower_byte_address_buffers"]

# Byte-address resource -> the uint storage buffer that binds the same memory.
BYTE_ADDRESS_BUFFER_KINDS = {
    "ByteAddressBuffer": "StructuredBuffer",
    "RWByteAddressBuffer": "RWStructuredBuffer",
    "RasterizerOrderedByteAddressBuffer": None,
}
_LOADS = {"buffer_load": 1, "buffer_load2": 2, "buffer_load3": 3, "buffer_load4": 4}
_STORES = {
    "buffer_store": 1,
    "buffer_store2": 2,
    "buffer_store3": 3,
    "buffer_store4": 4,
}
_ATOMICS = frozenset(
    {
        "atomicAdd",
        "atomicAnd",
        "atomicOr",
        "atomicXor",
        "atomicMin",
        "atomicMax",
        "atomicExchange",
    }
)
_COMPARE_EXCHANGE = "atomicCompareExchange"
_ORDERED_ATOMICS = frozenset({"atomicMin", "atomicMax"})
_SCALAR_NAMES = {"i": "int", "f": "float"}
_COMPONENTS = "xyzw"
_SCOPES = (A.BlockNode, A.ForNode, A.ForInNode, A.FunctionNode)


def _uint(value):
    return A.LiteralNode(value, A.PrimitiveType("uint"))


def _call(name, arguments):
    return A.FunctionCallNode(A.IdentifierNode(name), list(arguments))


def _or_refused(lowered):
    """A lowered node, or ``False`` for an access the pass refused (and leaves in place)."""
    return False if lowered is None else lowered


def _call_name(node):
    function = getattr(node, "function", None)
    if isinstance(function, A.IdentifierNode):
        return function.name
    return function if isinstance(function, str) else None


def _vector_name(kind, count):
    """CrossGL spelling of a ``kind`` (``uint``, ``int``, ``float``, ``bool``) vector of ``count`` lanes."""
    if count == 1:
        return kind
    prefix = {"uint": "u", "int": "i", "float": "", "bool": "b"}[kind]
    return f"{prefix}vec{count}"


def _byte_address_kind(type_node):
    """The byte-address resource a (possibly arrayed) type names, or ``None``."""
    while isinstance(type_node, A.ArrayType):
        type_node = type_node.element_type
    if isinstance(type_node, A.NamedType) and not type_node.generic_args:
        if type_node.name in BYTE_ADDRESS_BUFFER_KINDS:
            return type_node.name
    return None


def _word_buffer_type(type_node):
    """``type_node`` with its byte-address resource replaced by the uint storage buffer that binds the same memory."""
    if isinstance(type_node, A.ArrayType):
        arrayed = copy.copy(type_node)
        arrayed.element_type = _word_buffer_type(type_node.element_type)
        return arrayed
    return A.NamedType(
        BYTE_ADDRESS_BUFFER_KINDS[type_node.name], [A.PrimitiveType("uint")]
    )


def _declarations(program):
    """Every variable, parameter and structure member of the program, with the attributes that hold its type."""
    stack, seen = [program], set()
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
            continue
        if isinstance(node, A.StageMap):
            stack.extend(node.values())
            continue
        if not isinstance(node, A.ASTNode) or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, (A.VariableNode, A.ParameterNode, A.StructMemberNode)):
            yield node
        for name, value in vars(node).items():
            if name in ("parent", "annotations", "source_location"):
                continue
            if isinstance(value, (A.ASTNode, list, A.StageMap)):
                stack.append(value)


def _declared_type(node):
    for attribute in ("var_type", "param_type", "member_type", "vtype"):
        value = getattr(node, attribute, None)
        if value is not None:
            return value
    return None


def _has_byte_address_buffers(program):
    """Whether a global, stage variable, parameter or structure member is a byte-address buffer.

    A resource-typed local can only alias one of these, so function bodies need not be searched.
    """
    candidates = list(getattr(program, "global_variables", None) or [])
    for struct in getattr(program, "structs", None) or []:
        candidates.extend(getattr(struct, "members", None) or [])
    for stage in (getattr(program, "stages", None) or {}).values():
        candidates.extend(getattr(stage, "local_variables", None) or [])
    for function in _program_functions(program):
        candidates.extend(getattr(function, "parameters", None) or [])
    return any(
        _byte_address_kind(_declared_type(node)) is not None for node in candidates
    )


class _Lowerer:
    """One walk over the program with lexical scopes, rewriting each byte-address access in place."""

    def __init__(self, program, refuse):
        self.program = program
        self.refuse = refuse
        self.scope = TypeScope(program)
        self.functions = {
            getattr(function, "name", None) for function in _program_functions(program)
        }
        self.used_names = {
            node.name
            for node in _declarations(program)
            if isinstance(getattr(node, "name", None), str)
        }
        self.generated = set()
        # Accesses already refused, whose buffer arguments are not reported a second time.
        self.refused = set()

    # Names and types --------------------------------------------------------------------------------------------
    def fresh_name(self, stem):
        name, suffix = stem, 0
        while name in self.used_names:
            suffix += 1
            name = f"{stem}_{suffix}"
        self.used_names.add(name)
        return name

    def buffer_kind(self, expr):
        """The byte-address resource ``expr`` (a name or an indexed name) denotes in the current scope, or ``None``."""
        if not isinstance(expr, (A.IdentifierNode, A.ArrayAccessNode)):
            return None
        declared = self.scope.node_type(expr)
        if declared is None or isinstance(declared, A.ArrayType):
            return None
        return _byte_address_kind(declared)

    def unsigned_scalar(self, expr, lanes=1):
        inferred = self.scope.infer(expr)
        return (
            inferred is not None
            and inferred.kind == "u"
            and inferred.bits == 32
            and inferred.lanes == lanes
        )

    def scalar_kind(self, expr):
        """``"i"``, ``"u"`` or ``"f"`` for a 32-bit scalar ``expr`` (a value or a ``TypeNode``), else ``None``."""
        inferred = (
            type_from_node(expr)
            if isinstance(expr, A.TypeNode)
            else self.scope.infer(expr)
        )
        if inferred is None or inferred.lanes != 1 or inferred.bits != 32:
            return None
        return inferred.kind if inferred.kind in ("i", "u", "f") else None

    def as_uint(self, expr, lanes=1):
        """``expr`` converted to ``uint`` (or ``uvecN``) as HLSL converts an argument of a ``uint`` parameter."""
        if self.unsigned_scalar(expr, lanes):
            return expr
        return _call(_vector_name("uint", lanes), [expr])

    def word_index(self, address):
        """``uint(address) >> 2u``, the word that holds byte ``address``; folded for a literal address."""
        if (
            isinstance(address, A.LiteralNode)
            and isinstance(address.value, int)
            and not isinstance(address.value, bool)
            and address.value >= 0
        ):
            return _uint(address.value >> 2)
        return A.BinaryOpNode(self.as_uint(address), ">>", _uint(2))

    @staticmethod
    def offset_word(word, offset):
        if offset == 0:
            return word
        if isinstance(word, A.LiteralNode):
            return _uint(word.value + offset)
        return A.BinaryOpNode(clone(word), "+", _uint(offset))

    def element(self, buffer, word):
        node = A.ArrayAccessNode(clone(buffer), word)
        self.generated.add(id(node))
        return node

    # Accesses ---------------------------------------------------------------------------------------------------
    def multi_use_refusal(self, name, buffer, parts):
        """A refusal reason when ``name`` would evaluate its buffer index or ``parts`` more than once."""
        if not is_pure(buffer):
            return f"{name} on a byte-address buffer whose index has side effects"
        for label, part in parts:
            if not is_pure(part):
                return (
                    f"{name} on a byte-address buffer with a {label} that has side effects: "
                    "each word would evaluate it again"
                )
        return None

    def load(self, call, name, args):
        count = _LOADS[name]
        if len(args) != 2:
            self.refuse(
                f"{name} on a byte-address buffer takes a buffer and a byte address"
            )
            return None
        buffer, address = args
        if count == 1:
            return self.element(buffer, self.word_index(address))
        reason = self.multi_use_refusal(name, buffer, [("byte address", address)])
        if reason is not None:
            self.refuse(reason)
            return None
        word = self.word_index(address)
        words = [self.element(buffer, self.offset_word(word, i)) for i in range(count)]
        return _call(_vector_name("uint", count), words)

    def atomic(self, call, name, args, receiver=None):
        """The atomic on the word at ``(buffer, address)``; ``receiver`` is the variable its result is assigned to."""
        expected = 4 if name == _COMPARE_EXCHANGE else 3
        if len(args) != expected:
            self.refuse(
                f"{name} on a byte-address buffer with {len(args)} operands: the byte-address form takes the "
                f"buffer, a byte address and {expected - 2} value operand(s)"
            )
            return None
        buffer, address, *values = args
        if name in _ORDERED_ATOMICS and not self.unsigned_scalar(values[-1]):
            if self.scalar_kind(values[-1]) == "i":
                self.refuse(
                    f"{name} on a byte-address buffer with an int value inside an expression: the uint word the "
                    "target binds compares as unsigned, and the signed form is a compare-exchange loop that needs "
                    "a statement, an assignment or a declaration in a block of its own"
                )
            else:
                self.refuse(
                    f"{name} on a byte-address buffer with a value that is neither int nor uint: HLSL selects "
                    "the signed or the unsigned comparison from the value's type"
                )
            return None
        target = self.element(buffer, self.word_index(address))
        # The Interlocked methods take uint operands (a float converts numerically) and return the original
        # uint word, which converts as HLSL converts it into the receiving variable. Without a receiver, a
        # signed value keeps its bits as an int.
        result = _call(name, [target, *(self.as_uint(value) for value in values)])
        if receiver is not None:
            kind = self.scalar_kind(receiver)
        else:
            kind = "i" if self.scalar_kind(values[-1]) == "i" else None
        if kind in _SCALAR_NAMES:
            return _call(_SCALAR_NAMES[kind], [result])
        return result

    def dimension(self, call, args):
        buffer = args[0]
        words = _call("buffer_dimensions", [clone(buffer)])
        self.generated.add(id(words))
        bytes_expr = A.BinaryOpNode(words, "*", _uint(4))
        if len(args) == 1:
            return bytes_expr
        if len(args) != 2:
            self.refuse(
                "buffer_dimensions on a byte-address buffer takes the buffer and one byte-count output"
            )
            return None
        target = args[1]
        inferred = self.scope.infer(target)
        if inferred is None or inferred.lanes != 1 or inferred.kind == "b":
            self.refuse(
                "buffer_dimensions on a byte-address buffer into an output whose type is not a known scalar"
            )
            return None
        if not (inferred.kind == "u" and inferred.bits == 32):
            scalar = {"i": "int", "u": "uint", "f": "float"}[inferred.kind]
            if inferred.bits != 32:
                self.refuse(
                    "buffer_dimensions on a byte-address buffer into an output that is not 32 bits wide"
                )
                return None
            bytes_expr = _call(scalar, [bytes_expr])
        return A.AssignmentNode(target, bytes_expr)

    def store(self, name, args):
        """The statement that performs ``name`` (a ``buffer_store*`` call), or ``None`` after a refusal."""
        count = _STORES[name]
        if len(args) != 3:
            self.refuse(
                f"{name} on a byte-address buffer takes a buffer, a byte address and a value"
            )
            return None
        buffer, address, value = args
        if count == 1:
            assignment = A.AssignmentNode(
                self.element(buffer, self.word_index(address)), self.as_uint(value)
            )
            return A.ExpressionStatementNode(assignment)
        if not is_pure(buffer):
            self.refuse(f"{name} on a byte-address buffer whose index has side effects")
            return None
        word_name = self.fresh_name("crosstl_byte_address_word")
        value_name = self.fresh_name("crosstl_byte_address_value")
        _vector_name("uint", count)
        statements = [
            A.VariableNode(
                word_name, A.PrimitiveType("uint"), self.word_index(address)
            ),
            A.VariableNode(
                value_name,
                A.VectorType(A.PrimitiveType("uint"), count),
                self.as_uint(value, count),
            ),
        ]
        word = A.IdentifierNode(word_name)
        for index in range(count):
            target = self.element(buffer, self.offset_word(word, index))
            component = A.MemberAccessNode(
                A.IdentifierNode(value_name), _COMPONENTS[index]
            )
            statements.append(
                A.ExpressionStatementNode(A.AssignmentNode(target, component))
            )
        return A.BlockNode(statements)

    # Traversal --------------------------------------------------------------------------------------------------
    def statement_replacement(self, node, in_block=False):
        """The lowered statement for a byte-address store or dimension query used as a statement, else ``None``.

        A signed ``atomicMin``/``atomicMax`` statement, assignment or declaration becomes a compare-exchange loop:
        a block, or for a declaration directly inside a block (``in_block``) a list of statements to splice there.
        """
        signed = signed_ordered_atomic_replacement(self, node, in_block)
        if signed is not None:
            return signed
        if not isinstance(node, A.ExpressionStatementNode):
            return None
        call = node.expression
        name = _call_name(call) if isinstance(call, A.FunctionCallNode) else None
        args = list(getattr(call, "arguments", None) or [])
        if name not in _STORES and name != "buffer_dimensions":
            return None
        if not args or self.buffer_kind(args[0]) is None:
            return None
        if name in _STORES:
            return _or_refused(self.store(name, args))
        lowered = self.dimension(call, args)
        if lowered is None:
            return False
        return A.ExpressionStatementNode(lowered)

    def expression_replacement(self, node, receiver=None):
        """The lowered expression for a byte-address load, atomic or size query, else ``None``.

        ``receiver`` is the variable or the type of the declaration the expression is assigned to.
        """
        if not isinstance(node, A.FunctionCallNode) or id(node) in self.generated:
            return None
        name = _call_name(node)
        args = list(node.arguments or [])
        if not args or self.buffer_kind(args[0]) is None:
            return None
        if name in _LOADS:
            return _or_refused(self.load(node, name, args))
        if name in _ATOMICS or name == _COMPARE_EXCHANGE:
            return _or_refused(self.atomic(node, name, args, receiver))
        if name == "buffer_dimensions" and len(args) == 1:
            return _or_refused(self.dimension(node, args))
        if name in _STORES or name == "buffer_dimensions":
            self.refuse(f"{name} on a byte-address buffer inside an expression")
            return False
        return None

    def check_call_arguments(self, node):
        """Byte-address buffers may be passed to program functions; refuse them as arguments of anything else."""
        name = _call_name(node)
        for argument in node.arguments or []:
            if self.buffer_kind(argument) is not None and name not in self.functions:
                self.refuse(
                    f"byte-address buffer passed to '{name}', which has no byte-address form on this target"
                )

    def visit(self, node):
        if isinstance(node, A.ArrayAccessNode) and id(node) in self.generated:
            # A word access the pass wrote: its buffer is the lowered one, its index may hold more accesses.
            self.visit_buffer_reference(node.array_expr)
            self.visit_children(node, fields=("index_expr",))
            return
        scoped = isinstance(node, _SCOPES)
        if scoped:
            self.scope.push()
        if isinstance(node, A.FunctionNode):
            for parameter in node.parameters or []:
                self.scope.declare(parameter.name, parameter.param_type)
        if isinstance(node, A.FunctionCallNode) and id(node) not in (
            self.generated | self.refused
        ):
            self.check_call_arguments(node)
        if isinstance(node, A.IdentifierNode) and self.buffer_kind(node) is not None:
            self.refuse(
                f"byte-address buffer '{node.name}' used other than through a load, store, atomic, size query "
                "or a program function argument"
            )
        self.visit_children(node)
        if isinstance(node, A.VariableNode):
            self.scope.declare(node.name, node.var_type)
        if scoped:
            self.scope.pop()

    def visit_children(self, node, fields=None):
        call_buffers = set()
        if isinstance(node, A.FunctionCallNode):
            # A buffer argument is checked by its call, not as a bare name; the callee is not a value.
            call_buffers = {
                id(argument)
                for argument in node.arguments or []
                if self.buffer_kind(argument) is not None
            }
        splices = []
        for field, key in list(child_slots(node)):
            if fields is not None and field not in fields:
                continue
            if field in ("parameters", "return_type") and isinstance(
                node, A.FunctionNode
            ):
                continue
            if isinstance(node, A.FunctionCallNode) and field in ("function", "name"):
                continue
            child = get_child(node, field, key)
            in_block = isinstance(node, A.BlockNode) and field == "statements"
            replacement = self.statement_replacement(child, in_block)
            if isinstance(replacement, list):
                # Statements that replace a declaration in its block, spliced in once the block is visited.
                splices.append((key, replacement))
                for statement in replacement:
                    self.visit(statement)
                continue
            if replacement is None:
                replacement = self.expression_replacement(
                    child, _receiver(node, field, key)
                )
            if replacement is False:
                self.refused.add(id(child))
            elif replacement is not None:
                set_child(node, field, key, replacement)
                child = replacement
            if id(child) in call_buffers:
                self.visit_buffer_reference(child)
                continue
            self.visit(child)
        for key, statements in reversed(splices):
            node.statements[key : key + 1] = statements

    def visit_buffer_reference(self, expr):
        """Visit the indices of a buffer reference such as ``buffers[i]`` without treating the name as a use."""
        if isinstance(expr, A.ArrayAccessNode):
            self.visit_buffer_reference(expr.array_expr)
            self.visit_children(expr, fields=("index_expr",))

    # Program ----------------------------------------------------------------------------------------------------
    def run(self):
        program = self.program
        for constant in getattr(program, "constants", None) or []:
            self.scope.declare(constant.name, constant.const_type)
        for cbuffer in getattr(program, "cbuffers", None) or []:
            self.scope.declare_cbuffer(cbuffer)
        for variable in getattr(program, "global_variables", None) or []:
            self.scope.declare(variable.name, variable.var_type)
        for function in getattr(program, "functions", None) or []:
            self.visit(function)
        for stage in (getattr(program, "stages", None) or {}).values():
            self.scope.push()
            for cbuffer in getattr(stage, "local_cbuffers", None) or []:
                self.scope.declare_cbuffer(cbuffer)
            for variable in getattr(stage, "local_variables", None) or []:
                self.scope.declare(variable.name, variable.var_type)
            for function in getattr(stage, "local_functions", None) or []:
                self.visit(function)
            if getattr(stage, "entry_point", None) is not None:
                self.visit(stage.entry_point)
            self.scope.pop()
        self.rewrite_declarations()
        return program

    def rewrite_declarations(self):
        for node in list(_declarations(self.program)):
            declared = _declared_type(node)
            kind = _byte_address_kind(declared)
            if kind is None:
                continue
            name = getattr(node, "name", None)
            if isinstance(node, A.StructMemberNode):
                self.refuse(f"byte-address buffer '{name}' as a structure member")
                continue
            if BYTE_ADDRESS_BUFFER_KINDS[kind] is None:
                self.refuse(
                    f"{kind} '{name}': rasterizer-ordered access has no lowering on this target"
                )
                continue
            replacement = _word_buffer_type(declared)
            for attribute in ("var_type", "param_type", "vtype"):
                if getattr(node, attribute, None) is declared:
                    setattr(node, attribute, replacement)
        for function in _program_functions(self.program):
            if _byte_address_kind(getattr(function, "return_type", None)) is not None:
                self.refuse(
                    f"function '{function.name}' returns a byte-address buffer, which this target cannot return"
                )


def _receiver(node, field, key):
    """The assigned variable, or the declared type, that receives ``node``'s child at ``(field, key)``."""
    if key is not None:
        return None
    if isinstance(node, A.AssignmentNode) and field == "value" and node.operator == "=":
        return node.target
    if isinstance(node, A.VariableNode) and field == "initial_value":
        return node.var_type
    return None


def _program_functions(program):
    functions = list(getattr(program, "functions", None) or [])
    for stage in (getattr(program, "stages", None) or {}).values():
        functions.extend(getattr(stage, "local_functions", None) or [])
        if getattr(stage, "entry_point", None) is not None:
            functions.append(stage.entry_point)
    return functions


def lower_byte_address_buffers(program, refuse):
    """A copy of ``program`` whose byte-address buffers are uint word buffers; ``program`` when it has none.

    ``refuse(reason)`` receives one reason per construct that cannot be lowered faithfully.
    """
    if not isinstance(program, A.ShaderNode) or not _has_byte_address_buffers(program):
        return program
    lowered = copy.deepcopy(program)
    return _Lowerer(lowered, refuse or (lambda _reason: None)).run()

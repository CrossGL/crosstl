"""Pack GLSL shared arrays that a pointer of another scalar width views into 32-bit words.

GLSL cannot alias a ``shared`` array as another element type. A Metal or CrossGL kernel can view
``threadgroup half tile[128]`` through ``threadgroup float* words = (threadgroup float*)tile``, so before the
generator runs this pass declares such an array as ``shared uint`` words and rewrites every element access of
the array and of its views:

- a 32-bit element reads ``uintBitsToFloat(tile[i])`` or ``int(tile[i])`` and writes
  ``tile[i] = floatBitsToUint(value)`` or ``tile[i] = uint(int(value))``;
- an 8- or 16-bit element at index ``i`` is lane ``i % k`` of word ``i / k``, where ``k = 32 / width``. A read
  extracts its lane with ``unpackHalf2x16`` for ``half`` or ``bitfieldExtract`` for integers. A write clears the
  lane with ``atomicAnd`` and then sets it with ``atomicOr``, so two threads that write different lanes of one
  word do not race, just as their separate elements did not.

The pass rewrites a function-local array only when

- it is a one-dimensional array of 8-, 16- or 32-bit scalars with a constant length and no initializer;
- at least one view has a scalar width other than the element's, or another scalar at 16 or 8 bits, so the
  generator cannot alias the two;
- every use of the array and of its views is an element read, a statement that assigns, compound-assigns,
  increments or decrements one element, or the cast that initializes a view local. A view is cast from the whole
  array, from ``&tile[c]`` or from ``tile + c`` with an integer constant ``c`` at a view element boundary, and is
  never reassigned. A cast may wrap the source in further casts, such as a ``const`` view cast around
  ``(threadgroup float*)&tile[c]``.

Every other array is left unchanged, so the generator's pointer-reinterpretation diagnostics stand for it. In
particular a view that keeps the 32-bit width but changes the type, such as ``(threadgroup float*)tile`` over a
``threadgroup uint`` array, is still refused with ``workgroup-view-type-change``: the DirectX half of #4823 packs
only width-changing views too, and its same-width views lower by offset, which GLSL does not do yet (#4823).
"""

from collections import Counter

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    ExpressionStatementNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    NamedType,
    ParameterNode,
    PointerReinterpretNode,
    PointerType,
    PrimitiveType,
    SwizzleNode,
    UnaryOpNode,
    VariableNode,
)
from ..semantics.rewrite import child_slots, clone, get_child, is_pure, set_child

__all__ = ["lower_glsl_groupshared_word_views"]

_WORKGROUP_SPACES = frozenset({"threadgroup", "workgroup", "groupshared", "shared"})

# CrossGL scalar spelling -> (canonical spelling, kind, bits).
_SCALARS = {
    "float": ("float", "float", 32),
    "float32": ("float", "float", 32),
    "int": ("int", "int", 32),
    "int32": ("int", "int", 32),
    "uint": ("uint", "uint", 32),
    "uint32": ("uint", "uint", 32),
    "float16": ("float16", "float", 16),
    "half": ("float16", "float", 16),
    "int16": ("int16", "int", 16),
    "short": ("int16", "int", 16),
    "uint16": ("uint16", "uint", 16),
    "ushort": ("uint16", "uint", 16),
    "int8": ("int8", "int", 8),
    "char": ("int8", "int", 8),
    "uint8": ("uint8", "uint", 8),
    "uchar": ("uint8", "uint", 8),
}
_COMPOUND_OPERATORS = {
    "+=": "+",
    "-=": "-",
    "*=": "*",
    "/=": "/",
    "%=": "%",
    "&=": "&",
    "|=": "|",
    "^=": "^",
    "<<=": "<<",
    ">>=": ">>",
}
_STEP_OPERATORS = {"++": "+", "--": "-"}


class _Unsupported(Exception):
    """An array use this pass does not rewrite; the array keeps its declaration."""


class _Site:
    """One node of the function, the slot that holds it and its depth below the function."""

    __slots__ = ("node", "parent", "field", "key", "depth")

    def __init__(self, node, parent, field, key, depth):
        self.node = node
        self.parent = parent
        self.field = field
        self.key = key
        self.depth = depth


def _sites(root):
    """Every structural descendant of ``root`` once, with the slot that holds it."""
    found = []
    seen = {id(root)}
    stack = [(root, 0)]
    while stack:
        parent, depth = stack.pop()
        for field, key in child_slots(parent):
            child = get_child(parent, field, key)
            if id(child) in seen:
                continue
            seen.add(id(child))
            found.append(_Site(child, parent, field, key, depth + 1))
            stack.append((child, depth + 1))
    return found


def _scalar(type_node):
    if isinstance(type_node, NamedType) and getattr(type_node, "generic_args", None):
        return None
    if isinstance(type_node, (PrimitiveType, NamedType)):
        return _SCALARS.get(getattr(type_node, "name", None))
    return None


def _is_workgroup(variable):
    qualifiers = set(getattr(variable, "qualifiers", None) or [])
    space = getattr(getattr(variable, "var_type", None), "address_space", None)
    return bool(qualifiers & _WORKGROUP_SPACES) or space in _WORKGROUP_SPACES


def _constant_int(node):
    if isinstance(node, int) and not isinstance(node, bool):
        return node
    if isinstance(node, LiteralNode) and isinstance(node.value, int):
        if not isinstance(node.value, bool):
            return node.value
    return None


def _operator(node):
    return getattr(node, "operator", None) or getattr(node, "op", None)


def _uint(value):
    return LiteralNode(value, PrimitiveType("uint"))


def _int(value):
    return LiteralNode(value, PrimitiveType("int"))


def _call(name, *arguments):
    return FunctionCallNode(IdentifierNode(name), list(arguments))


class _Array:
    """A candidate array: its declaration, scalar and views (name -> (scalar, element offset))."""

    def __init__(self, declaration, scalar, length):
        self.declaration = declaration
        self.scalar = scalar
        self.length = length
        self.views = {}
        self.view_sites = []

    def words(self):
        return (self.length * self.scalar[2] + 31) // 32


class _Function:
    """The arrays of one function that this pass can pack, and their rewrite."""

    def __init__(self, function, taken, writing_functions):
        self.function = function
        self.taken = taken
        self.writing_functions = writing_functions
        self.sites = _sites(function)
        self.by_node = {id(site.node): site for site in self.sites}
        self.declared = Counter(
            site.node.name
            for site in self.sites
            if isinstance(site.node, (VariableNode, ParameterNode))
        )

    def candidates(self):
        arrays = {}
        for site in self.sites:
            node = site.node
            if not isinstance(node, VariableNode) or not _is_workgroup(node):
                continue
            var_type = node.var_type
            if not isinstance(var_type, ArrayType) or node.initial_value is not None:
                continue
            scalar = _scalar(var_type.element_type)
            length = _constant_int(var_type.size)
            if scalar is None or not length or length < 0:
                continue
            if self.declared[node.name] != 1 or node.name in self.taken:
                continue
            arrays[node.name] = _Array(node, scalar, length)
        for site in self.sites:
            self._add_view(site, arrays)
        return {
            name: array
            for name, array in arrays.items()
            # A view of another width needs packing, as does a view of another
            # scalar at 16 or 8 bits: the two types need different lane
            # decodes, so the generator cannot alias them either. A view that
            # keeps the 32-bit width but changes the type stays with the
            # generator, which refuses it with workgroup-view-type-change.
            if any(
                view[0] != array.scalar
                and 32 not in (view[0][2], array.scalar[2])
                or view[0][2] != array.scalar[2]
                for view in array.views.values()
            )
        }

    def _add_view(self, site, arrays):
        node = site.node
        if not isinstance(node, VariableNode) or not isinstance(
            node.initial_value, PointerReinterpretNode
        ):
            return
        # A cast may wrap the source in further casts, such as a ``const`` view
        # cast around ``(threadgroup float*)&tile[c]``; only the outermost
        # target type decides how the view reads the array.
        expression = node.initial_value.expression
        while isinstance(expression, PointerReinterpretNode):
            expression = expression.expression
        source, offset = self._view_source(expression)
        array = arrays.get(source)
        if array is None:
            return
        target = node.initial_value.target_type
        scalar = _scalar(getattr(target, "pointee_type", None))
        if (
            scalar is None
            or not isinstance(node.var_type, PointerType)
            or not _is_workgroup(node)
            or self.declared[node.name] != 1
            or not isinstance(site.key, int)
            or (offset * array.scalar[2]) % scalar[2]
        ):
            arrays.pop(source)
            return
        array.views[node.name] = (scalar, offset * array.scalar[2] // scalar[2])
        array.view_sites.append(site)

    @staticmethod
    def _view_source(expression):
        if isinstance(expression, IdentifierNode):
            return expression.name, 0
        if isinstance(expression, UnaryOpNode) and _operator(expression) == "&":
            operand = expression.operand
            if isinstance(operand, ArrayAccessNode) and isinstance(
                operand.array_expr, IdentifierNode
            ):
                offset = _constant_int(operand.index_expr)
                if offset is not None and offset >= 0:
                    return operand.array_expr.name, offset
        if isinstance(expression, BinaryOpNode) and _operator(expression) == "+":
            if isinstance(expression.left, IdentifierNode):
                offset = _constant_int(expression.right)
                if offset is not None and offset >= 0:
                    return expression.left.name, offset
        return None, 0

    def parent_of(self, node):
        site = self.by_node.get(id(node))
        return site.parent if site is not None else None

    def plan(self, array):
        """The element accesses of ``array`` and its views, or raise :class:`_Unsupported`."""
        names = {array.declaration.name: (array.scalar, 0), **array.views}
        initializers = {id(site.node.initial_value) for site in array.view_sites}
        accesses = []
        for site in self.sites:
            node = site.node
            if not isinstance(node, IdentifierNode) or node.name not in names:
                continue
            if self._is_view_base(node, initializers):
                continue
            parent = site.parent
            if not isinstance(parent, ArrayAccessNode) or parent.array_expr is not node:
                raise _Unsupported(node.name)
            accesses.append(self._access(parent, names[node.name]))
        return accesses

    def _is_view_base(self, node, initializers):
        """True for the array named in a view's initializer: ``tile``, ``&tile[c]`` or ``tile + c``.

        A cast may wrap the source in further ``PointerReinterpretNode`` nodes, such as a ``const`` view cast
        around ``(threadgroup float*)&tile[c]``; those hops are skipped on the way to the initializer.
        """
        chain = []
        current = node
        for _ in range(5):
            current = self.parent_of(current)
            if current is None:
                return False
            if id(current) in initializers:
                break
            chain.append(current)
        else:
            return False
        while chain and isinstance(chain[-1], PointerReinterpretNode):
            chain.pop()
        if not chain:
            return True
        if len(chain) == 1:
            return isinstance(chain[0], BinaryOpNode) and chain[0].left is node
        if len(chain) != 2:
            return False
        access, address = chain
        return (
            isinstance(access, ArrayAccessNode)
            and access.array_expr is node
            and isinstance(address, UnaryOpNode)
            and _operator(address) == "&"
        )

    def _access(self, access, view):
        site = self.by_node[id(access)]
        parent = site.parent
        scalar, offset = view
        if isinstance(parent, AssignmentNode) and parent.target is access:
            statement = self.parent_of(parent)
            if not self._is_statement_of(statement, parent):
                raise _Unsupported("assignment value")
            return ("write", statement, access, scalar, offset, _operator(parent))
        if isinstance(parent, UnaryOpNode) and _operator(parent) in _STEP_OPERATORS:
            statement = self.parent_of(parent)
            if not self._is_statement_of(statement, parent):
                raise _Unsupported("step value")
            return ("step", statement, access, scalar, offset, _operator(parent))
        if isinstance(parent, (SwizzleNode, MemberAccessNode)):
            raise _Unsupported("component access")
        if isinstance(parent, UnaryOpNode) and _operator(parent) == "&":
            raise _Unsupported("address")
        if isinstance(parent, FunctionCallNode) and self._writes_argument(
            parent, access
        ):
            raise _Unsupported("call argument")
        if scalar[2] != 32 and not is_pure(access.index_expr):
            raise _Unsupported("index")
        return ("read", site, access, scalar, offset, None)

    @staticmethod
    def _is_statement_of(statement, expression):
        return (
            isinstance(statement, ExpressionStatementNode)
            and statement.expression is expression
        )

    def _writes_argument(self, call, argument):
        name = getattr(call.function, "name", call.function)
        if not isinstance(name, str):
            return True
        if name.startswith("atomic"):
            return True
        return name.partition("<")[0] in self.writing_functions


class _Emitter:
    """Word-array expressions for one packed array."""

    def __init__(self, array, fresh):
        self.name = array.declaration.name
        self.fresh = fresh

    def _word(self, position, bits):
        if bits == 32:
            return ArrayAccessNode(IdentifierNode(self.name), position)
        per_word = 32 // bits
        return ArrayAccessNode(
            IdentifierNode(self.name),
            BinaryOpNode(position, ">>", _uint(per_word.bit_length() - 1)),
        )

    @staticmethod
    def _shift(position, bits):
        per_word = 32 // bits
        return BinaryOpNode(
            BinaryOpNode(position, "&", _uint(per_word - 1)), "*", _uint(bits)
        )

    @staticmethod
    def _position(index, offset):
        position = _call("uint", index)
        if offset:
            position = BinaryOpNode(position, "+", _uint(offset))
        return position

    def load(self, index, scalar, offset):
        position = self._position(index, offset)
        bits = scalar[2]
        word = self._word(position, bits)
        return self._decode(word, clone(position), scalar)

    def _decode(self, word, position, scalar):
        """The element ``scalar`` at ``position`` of the 32-bit ``word`` that holds it.

        GLSL has no bit cast between a ``uint`` word and a 32-bit scalar other than ``uintBitsToFloat`` and
        ``int(word)``, and no 16- or 8-bit scalar type in a ``shared`` array, so a narrow lane is extracted with
        ``unpackHalf2x16`` for ``half`` and ``bitfieldExtract`` for integers. ``bitfieldExtract`` sign-extends a
        signed lane, so no manual sign extension is needed.
        """
        _, kind, bits = scalar
        if bits == 32:
            if kind == "uint":
                return word
            if kind == "int":
                return _call("int", word)
            return _call("uintBitsToFloat", word)
        shift = self._shift(position, bits)
        if kind == "float":
            # MemberAccessNode, not SwizzleNode: the parser spells ``.x`` on a
            # call result as a member access, and the GLSL expression lowering
            # only handles that shape on a call.
            return MemberAccessNode(
                _call("unpackHalf2x16", BinaryOpNode(word, ">>", shift)),
                "x",
            )
        if kind == "int":
            return _call(
                "bitfieldExtract",
                _call("int", word),
                _call("int", shift),
                _int(bits),
            )
        return _call(
            "bitfieldExtract",
            word,
            _call("int", shift),
            _int(bits),
        )

    @staticmethod
    def encode(value, scalar):
        """The 32-bit word bits of ``value`` as a ``scalar`` element.

        ``floatBitsToUint`` is the bit cast for a 32-bit float; ``uint(int(value))`` keeps the bits of a signed
        integer because the conversion is modulo 2**32. A ``half`` packs into the low 16 bits with
        ``packHalf2x16(vec2(value, 0.0))``; a narrow integer keeps only its lane bits.
        """
        _, kind, bits = scalar
        typed = _call(kind, value)
        if bits == 32:
            if kind == "float":
                return _call("floatBitsToUint", typed)
            if kind == "int":
                return _call("uint", typed)
            return typed
        if kind == "float":
            return _call(
                "packHalf2x16",
                _call("vec2", typed, LiteralNode(0.0, PrimitiveType("float"))),
            )
        if kind == "int":
            typed = _call("uint", typed)
        return BinaryOpNode(typed, "&", _uint((1 << bits) - 1))

    def store(self, statement_site, access, scalar, offset, value_of, reads_old):
        """Replace the statement that writes ``access`` with the word writes.

        ``value_of(old)`` builds the new element value; ``old()`` builds the element's previous value. A narrow
        write that reads its previous value takes it from the word ``atomicAnd`` returns as it clears the lane, so
        the lane is never read by a plain load that the atomics could be reordered with.
        """
        bits = scalar[2]
        index_name = self.fresh("__crossgl_word_index")

        def index():
            return IdentifierNode(index_name)

        statements = [
            VariableNode(
                index_name,
                PrimitiveType("uint"),
                self._position(access.index_expr, offset),
            )
        ]
        if bits == 32:
            value = value_of(lambda: self.load(index(), scalar, 0))
            statements.append(
                ExpressionStatementNode(
                    AssignmentNode(
                        self._word(index(), bits), self.encode(value, scalar)
                    )
                )
            )
            set_child(
                statement_site.parent,
                statement_site.field,
                statement_site.key,
                BlockNode(statements),
            )
            return
        bits_name = self.fresh("__crossgl_word_bits")
        clear = _call(
            "atomicAnd",
            self._word(index(), bits),
            UnaryOpNode(
                "~",
                BinaryOpNode(_uint((1 << bits) - 1), "<<", self._shift(index(), bits)),
            ),
        )
        if reads_old:
            original_name = self.fresh("__crossgl_word_original")
            statements.append(VariableNode(original_name, PrimitiveType("uint"), clear))
            value = value_of(
                lambda: self._decode(IdentifierNode(original_name), index(), scalar)
            )
            statements.append(
                VariableNode(
                    bits_name, PrimitiveType("uint"), self.encode(value, scalar)
                )
            )
        else:
            statements.append(
                VariableNode(
                    bits_name,
                    PrimitiveType("uint"),
                    self.encode(value_of(None), scalar),
                )
            )
            statements.append(ExpressionStatementNode(clear))
        statements.append(
            ExpressionStatementNode(
                _call(
                    "atomicOr",
                    self._word(index(), bits),
                    BinaryOpNode(
                        IdentifierNode(bits_name), "<<", self._shift(index(), bits)
                    ),
                )
            )
        )
        set_child(
            statement_site.parent,
            statement_site.field,
            statement_site.key,
            BlockNode(statements),
        )


def _names(program):
    names = set()
    for site in _sites(program):
        name = getattr(site.node, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


def _program_names(program):
    """Names declared at program scope, which a function-local array must not shadow, and user functions."""
    names = set()
    for field in ("global_variables", "constants", "functions", "structs", "cbuffers"):
        for node in getattr(program, field, None) or []:
            name = getattr(node, "name", None)
            if isinstance(name, str):
                names.add(name)
    return names


def _rewrite(function, array, accesses, fresh):
    emitter = _Emitter(array, fresh)

    def replaced_depth(item):
        kind, site = item[0], item[1]
        return -(site.depth if kind == "read" else function.by_node[id(site)].depth)

    # Deeper replacements first, so a statement that is rewritten later already holds rewritten reads.
    for kind, site, access, scalar, offset, operator in sorted(
        accesses, key=replaced_depth
    ):
        if kind == "read":
            set_child(
                site.parent,
                site.field,
                site.key,
                emitter.load(access.index_expr, scalar, offset),
            )
            continue
        statement_site = function.by_node[id(site)]
        assignment = site.expression
        if kind == "step":
            step = _STEP_OPERATORS[operator]

            def value_of(load, step=step, scalar=scalar):
                return BinaryOpNode(
                    load(), step, _call(scalar[1], LiteralNode(1, PrimitiveType("int")))
                )

        elif operator == "=":

            def value_of(load, value=assignment.value):
                return value

        elif operator in _COMPOUND_OPERATORS:

            def value_of(
                load, value=assignment.value, op=_COMPOUND_OPERATORS[operator]
            ):
                return BinaryOpNode(load(), op, value)

        else:
            raise _Unsupported(operator)
        emitter.store(statement_site, access, scalar, offset, value_of, operator != "=")
    declaration = array.declaration
    declaration.var_type = ArrayType(
        PrimitiveType("uint"), LiteralNode(array.words(), PrimitiveType("int"))
    )
    declaration.vtype = declaration.var_type
    for site in sorted(array.view_sites, key=lambda item: -item.key):
        del getattr(site.parent, site.field)[site.key]


def lower_glsl_groupshared_word_views(ast):
    """Pack the function-local workgroup arrays of ``ast`` that views of another width alias, in place.

    ``ast`` must be the generator's private copy; it is returned for chaining.
    """
    taken = _program_names(ast)
    used = _names(ast)
    counter = Counter()

    def fresh(stem):
        while True:
            counter[stem] += 1
            name = f"{stem}{counter[stem]}"
            if name not in used:
                used.add(name)
                return name

    functions = [
        site.node for site in _sites(ast) if isinstance(site.node, FunctionNode)
    ]
    writing_functions = {
        function.name
        for function in functions
        if any(
            {"out", "inout"} & set(getattr(parameter, "qualifiers", None) or [])
            for parameter in getattr(function, "parameters", None) or []
        )
    }
    for node in functions:
        tried = set()
        while True:
            function = _Function(node, taken, writing_functions)
            candidates = function.candidates()
            pending = sorted(set(candidates) - tried)
            if not pending:
                break
            name = pending[0]
            tried.add(name)
            array = candidates[name]
            try:
                accesses = function.plan(array)
            except _Unsupported:
                continue
            _rewrite(function, array, accesses, fresh)
    return ast

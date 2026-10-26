"""Language-neutral compute programs for grammar-based fuzzing.

A :class:`Program` is one compute kernel with an ``int`` input buffer ``a``, an ``int`` output buffer ``o`` and the
global invocation index ``i``, plus helper functions. Every expression is explicitly typed: operands of a binary
operator have the same type, conversions are explicit and comparisons take scalars, so one program prints as valid
MSL, HLSL, GLSL, CrossGL and SPIR-V without any implicit-conversion rule of its own. Integer divisors are forced
odd and shift counts are masked, so no constant folding can divide by zero or shift past the width. A float literal
converted to ``int`` or ``uint`` is clamped into the target range, so no conversion folds to an undefined value.

``programs(config)`` is the hypothesis strategy. Hypothesis shrinks a failing program structurally (fewer
statements, smaller expressions), which is what makes crash reproducers short.

Control flow is kept structured so the SPIR-V printer needs no phi nodes and no unreachable blocks: ``return`` only
ends a helper's top-level body, ``break`` and ``continue`` only end the then-block of an ``if`` inside a loop, and
every ``switch`` case ends in ``break``.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

KINDS = ("int", "uint", "float", "bool")
COMPONENTS = "xyzw"


@dataclass(frozen=True)
class Type:
    kind: str
    size: int = 1

    @property
    def is_vector(self) -> bool:
        return self.size > 1

    def scalar(self) -> Type:
        return Type(self.kind)

    def __str__(self) -> str:
        return self.kind if self.size == 1 else f"{self.kind}{self.size}"


INT = Type("int")
UINT = Type("uint")
FLOAT = Type("float")
BOOL = Type("bool")


# Expressions. Every node carries its result type.
@dataclass(frozen=True)
class Literal:
    type: Type
    value: object


@dataclass(frozen=True)
class Var:
    name: str
    type: Type


@dataclass(frozen=True)
class Load:
    """``a[i]``: the input element of this invocation."""

    type: Type = INT


@dataclass(frozen=True)
class Unary:
    op: str
    operand: object
    type: Type


@dataclass(frozen=True)
class Binary:
    op: str
    left: object
    right: object
    type: Type


@dataclass(frozen=True)
class Select:
    condition: object
    then: object
    otherwise: object
    type: Type


@dataclass(frozen=True)
class Swizzle:
    base: object
    components: str
    type: Type


@dataclass(frozen=True)
class Construct:
    type: Type
    args: tuple[object, ...]


@dataclass(frozen=True)
class Convert:
    type: Type
    operand: object


@dataclass(frozen=True)
class Builtin:
    name: str
    args: tuple[object, ...]
    type: Type


@dataclass(frozen=True)
class Call:
    function: str
    args: tuple[object, ...]
    type: Type


# Statements.
@dataclass(frozen=True)
class Declare:
    name: str
    type: Type
    value: object


@dataclass(frozen=True)
class Assign:
    name: str
    value: object
    op: str | None = None  # "+" for "+=", and so on


@dataclass(frozen=True)
class If:
    condition: object
    then: tuple[object, ...]
    otherwise: tuple[object, ...] | None = None


@dataclass(frozen=True)
class For:
    name: str
    count: int
    body: tuple[object, ...]


@dataclass(frozen=True)
class Switch:
    selector: object
    cases: tuple[tuple[int, tuple[object, ...]], ...]
    default: tuple[object, ...]


@dataclass(frozen=True)
class Break:
    pass


@dataclass(frozen=True)
class Continue:
    pass


@dataclass(frozen=True)
class Return:
    value: object


@dataclass(frozen=True)
class Function:
    name: str
    params: tuple[tuple[str, Type], ...]
    result: Type
    body: tuple[object, ...]  # ends with Return


@dataclass(frozen=True)
class Program:
    functions: tuple[Function, ...]
    body: tuple[object, ...]
    output: object  # int expression stored to o[i]


# name: (kinds, arity, vector arguments allowed, result is scalar)
BUILTINS = {
    "abs": (("int", "float"), 1, True, False),
    "min": (("int", "uint", "float"), 2, True, False),
    "max": (("int", "uint", "float"), 2, True, False),
    "clamp": (("int", "uint", "float"), 3, True, False),
    "floor": (("float",), 1, True, False),
    "ceil": (("float",), 1, True, False),
    "fract": (("float",), 1, True, False),
    "sqrt": (("float",), 1, True, False),
    "sin": (("float",), 1, True, False),
    "cos": (("float",), 1, True, False),
    "exp": (("float",), 1, True, False),
    "mix": (("float",), 3, True, False),
    "dot": (("float",), 2, True, True),
    "length": (("float",), 1, True, True),
}

ARITHMETIC = ("+", "-", "*", "/")
BITWISE = ("&", "|", "^", "<<", ">>")
COMPARISON = ("<", "<=", ">", ">=", "==", "!=")
COMPOUND = ("+", "-", "*", "&", "|", "^")

INT_EDGES = (0, 1, -1, 2, 7, 31, 32, 255, 65535, 2147483647, -2147483648)
UINT_EDGES = (0, 1, 2, 31, 32, 255, 65535, 2147483647, 2147483648, 4294967295)
FLOAT_EDGES = (0.0, -0.0, 1.0, -1.0, 0.5, 2.5, 1e-07, 1e10, 3.4028234663852886e38)
# The float32 values nearest each integer type's bounds that still truncate to a value of that type. Converting a
# float outside them is undefined, and compilers such as DXC fold such a constant conversion to an undefined value.
FLOAT_TO_INTEGER_RANGE = {
    "int": (-2147483648.0, 2147483520.0),
    "uint": (0.0, 4294967040.0),
}


@dataclass(frozen=True)
class Config:
    """What the target language can express. Scalar-only languages set ``vectors=False``."""

    vectors: bool = True
    builtins: tuple[str, ...] = tuple(BUILTINS)
    max_depth: int = 3
    max_statements: int = 24
    max_functions: int = 2
    switch: bool = True


def _clamp_float_literals(node, low, high):
    """Clamp a float literal, or the literals of a vector built from them, into ``[low, high]``."""
    if isinstance(node, Literal) and node.type.kind == "float":
        return Literal(node.type, min(max(node.value, low), high))
    if isinstance(node, Construct):
        return Construct(
            node.type,
            tuple(_clamp_float_literals(arg, low, high) for arg in node.args),
        )
    return node


class _Builder:
    def __init__(self, draw, config: Config):
        self.draw = draw
        self.config = config
        self.scopes = [[]]  # lists of (name, type, mutable)
        self.functions = []
        self.counter = 0
        self.statements = 0
        self.in_kernel = False  # a[i] and i exist only in the kernel body

    # -- helpers -------------------------------------------------------------------------------------------------
    def integer(self, low, high):
        return self.draw(st.integers(low, high))

    def choice(self, options):
        return options[self.integer(0, len(options) - 1)]

    def fresh(self, prefix):
        self.counter += 1
        return f"{prefix}{self.counter}"

    def visible(self, type_=None, mutable=False):
        found = []
        for scope in self.scopes:
            for name, var_type, is_mutable in scope:
                if (type_ is None or var_type == type_) and (is_mutable or not mutable):
                    found.append((name, var_type))
        return found

    def any_type(self):
        kind = self.choice(KINDS)
        size = (
            self.integer(2, 4) if self.config.vectors and self.integer(0, 2) == 0 else 1
        )
        return Type(kind, size)

    # -- expressions ---------------------------------------------------------------------------------------------
    def literal(self, type_):
        if type_.kind == "bool":
            return Literal(BOOL, self.draw(st.booleans()))
        if type_.kind == "int":
            value = self.draw(
                st.one_of(st.sampled_from(INT_EDGES), st.integers(-100, 100))
            )
        elif type_.kind == "uint":
            value = self.draw(
                st.one_of(st.sampled_from(UINT_EDGES), st.integers(0, 100))
            )
        else:
            value = self.draw(
                st.one_of(
                    st.sampled_from(FLOAT_EDGES),
                    st.floats(
                        -1e6, 1e6, width=32, allow_nan=False, allow_infinity=False
                    ),
                )
            )
        return Literal(type_.scalar(), value)

    def leaf(self, type_):
        options = ["literal"]
        if self.visible(type_):
            options += ["var", "var"]
        if type_ == INT and self.in_kernel:
            options.append("load")
        if type_ == UINT and self.in_kernel:
            options.append("index")
        option = self.choice(options)
        if option == "var":
            name, var_type = self.choice(self.visible(type_))
            return Var(name, var_type)
        if option == "load":
            return Load()
        if option == "index":
            return Var("i", UINT)
        if type_.is_vector:
            return Construct(type_, (self.literal(type_.scalar()),))
        return self.literal(type_)

    def expr(self, type_, depth=None):
        depth = self.config.max_depth if depth is None else depth
        if depth <= 0 or self.integer(0, 3) == 0:
            return self.leaf(type_)
        options = self.operators(type_)
        return getattr(self, "op_" + self.choice(options))(type_, depth - 1)

    def operators(self, type_):
        kind = type_.kind
        options = ["select"]
        if kind in ("int", "uint", "float"):
            options += ["arith", "arith", "unary_minus"]
        if kind in ("int", "uint"):
            options += ["bitwise", "complement"]
        if kind == "bool" and not type_.is_vector:
            options += ["compare", "compare", "logical", "not"]
        if self.config.vectors:
            options.append("swizzle")
        if type_.is_vector:
            options += ["construct", "construct"]
        options.append("convert")
        if self.builtins_for(type_):
            options += ["builtin", "builtin"]
        if any(f.result == type_ for f in self.functions):
            options.append("call")
        return options

    def op_select(self, type_, depth):
        return Select(
            self.expr(BOOL, depth),
            self.expr(type_, depth),
            self.expr(type_, depth),
            type_,
        )

    def op_arith(self, type_, depth):
        op = self.choice(ARITHMETIC + (("%",) if type_.kind != "float" else ()))
        left, right = self.expr(type_, depth), self.expr(type_, depth)
        if op in ("/", "%") and type_.kind != "float":
            right = Binary("|", right, self.splat(type_, 1), type_)  # never zero
        return Binary(op, left, right, type_)

    def op_unary_minus(self, type_, depth):
        return Unary("-", self.expr(type_, depth), type_)

    def op_bitwise(self, type_, depth):
        op = self.choice(BITWISE)
        left, right = self.expr(type_, depth), self.expr(type_, depth)
        if op in ("<<", ">>"):
            right = Binary("&", right, self.splat(type_, 31), type_)
        return Binary(op, left, right, type_)

    def op_complement(self, type_, depth):
        return Unary("~", self.expr(type_, depth), type_)

    def op_compare(self, type_, depth):
        operand = Type(self.choice(KINDS))
        ops = COMPARISON if operand.kind != "bool" else ("==", "!=")
        return Binary(
            self.choice(ops), self.expr(operand, depth), self.expr(operand, depth), BOOL
        )

    def op_logical(self, type_, depth):
        return Binary(
            self.choice(("&&", "||")),
            self.expr(BOOL, depth),
            self.expr(BOOL, depth),
            BOOL,
        )

    def op_not(self, type_, depth):
        return Unary("!", self.expr(BOOL, depth), BOOL)

    def op_swizzle(self, type_, depth):
        size = self.integer(max(2, type_.size), 4)
        base = self.expr(Type(type_.kind, size), depth)
        components = "".join(
            COMPONENTS[self.integer(0, size - 1)] for _ in range(type_.size)
        )
        return Swizzle(base, components, type_)

    def op_construct(self, type_, depth):
        if self.integer(0, 3) == 0:
            return Construct(type_, (self.expr(type_.scalar(), depth),))
        args, remaining = [], type_.size
        while remaining:
            size = self.integer(1, remaining) if self.config.vectors else 1
            if size == type_.size:
                size = 1
            args.append(self.expr(Type(type_.kind, size), depth))
            remaining -= size
        return Construct(type_, tuple(args))

    def op_convert(self, type_, depth):
        kind = self.choice(tuple(k for k in KINDS if k != type_.kind))
        operand = self.expr(Type(kind, type_.size), depth)
        if kind == "float" and type_.kind in FLOAT_TO_INTEGER_RANGE:
            operand = _clamp_float_literals(
                operand, *FLOAT_TO_INTEGER_RANGE[type_.kind]
            )
        return Convert(type_, operand)

    def builtins_for(self, type_):
        names = []
        for name in self.config.builtins:
            kinds, _, vectors, scalar_result = BUILTINS[name]
            if type_.kind not in kinds:
                continue
            if scalar_result:
                if not type_.is_vector and self.config.vectors:
                    names.append(name)
            elif vectors or not type_.is_vector:
                names.append(name)
        return names

    def op_builtin(self, type_, depth):
        name = self.choice(self.builtins_for(type_))
        _, arity, _, scalar_result = BUILTINS[name]
        operand = type_
        if scalar_result:
            operand = Type(type_.kind, self.integer(2, 4))
        return Builtin(
            name, tuple(self.expr(operand, depth) for _ in range(arity)), type_
        )

    def op_call(self, type_, depth):
        function = self.choice([f for f in self.functions if f.result == type_])
        args = tuple(self.expr(param_type, depth) for _, param_type in function.params)
        return Call(function.name, args, type_)

    def splat(self, type_, value):
        literal = Literal(type_.scalar(), value)
        return Construct(type_, (literal,)) if type_.is_vector else literal

    # -- statements ----------------------------------------------------------------------------------------------
    def block(self, depth, in_loop=False, max_len=4, keep_scope=False):
        """A statement list in a new scope; ``keep_scope`` leaves its declarations visible afterwards."""
        self.scopes.append([])
        statements = []
        for _ in range(self.integer(0, max_len)):
            if self.statements >= self.config.max_statements:
                break
            self.statements += 1
            statements.append(self.statement(depth, in_loop))
        if not keep_scope:
            self.scopes.pop()
        return tuple(statements)

    def declare(self, depth):
        type_ = self.any_type()
        name = self.fresh("v")
        statement = Declare(name, type_, self.expr(type_, depth))
        self.scopes[-1].append((name, type_, True))
        return statement

    def statement(self, depth, in_loop):
        options = ["declare", "declare", "assign"]
        if depth > 0:
            options += ["if", "for"] + (["switch"] if self.config.switch else [])
        option = self.choice(options)
        if option == "assign" and self.visible(mutable=True):
            name, type_ = self.choice(self.visible(mutable=True))
            ops = [None]
            if type_.kind in ("int", "uint", "float"):
                ops += [
                    op for op in COMPOUND if type_.kind != "float" or op in ARITHMETIC
                ]
            return Assign(name, self.expr(type_, depth), self.choice(ops))
        if option == "if":
            condition = self.expr(BOOL, depth)
            then = self.block(depth - 1, in_loop)
            if in_loop and self.integer(0, 2) == 0:
                then = then + (self.choice((Break(), Continue())),)
            otherwise = self.block(depth - 1, in_loop) if self.integer(0, 1) else None
            return If(condition, then, otherwise)
        if option == "for":
            name = self.fresh("k")
            self.scopes.append([(name, INT, False)])
            body = self.block(depth - 1, in_loop=True)
            self.scopes.pop()
            return For(name, self.integer(1, 4), body)
        if option == "switch":
            labels = sorted(set(self.draw(st.lists(st.integers(-3, 8), max_size=3))))
            cases = tuple((label, self.block(depth - 1)) for label in labels)
            return Switch(self.expr(INT, depth), cases, self.block(depth - 1))
        return self.declare(depth)

    def function(self, index):
        params = []
        self.scopes = [[]]
        for _ in range(self.integer(0, 3)):
            name, type_ = self.fresh("p"), self.any_type()
            params.append((name, type_))
            self.scopes[0].append((name, type_, True))
        result = self.any_type()
        body = self.block(self.config.max_depth - 1, keep_scope=True)
        value = self.expr(result)
        self.scopes = [[]]
        return Function(f"f{index}", tuple(params), result, body + (Return(value),))

    def program(self):
        for index in range(self.integer(0, self.config.max_functions)):
            self.functions.append(self.function(index))
        self.scopes = [[]]
        self.in_kernel = True
        body = self.block(self.config.max_depth, max_len=6, keep_scope=True)
        output = self.expr(INT)
        return Program(tuple(self.functions), body, output)


def programs(config: Config = Config()):
    """The hypothesis strategy of well-typed programs for ``config``."""

    @st.composite
    def build(draw):
        return _Builder(draw, config).program()

    return build()

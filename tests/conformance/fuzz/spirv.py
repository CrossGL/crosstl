"""Print a :class:`~tests.conformance.fuzz.program.Program` as SPIR-V assembly for a Vulkan 1.1 environment.

Every local lives in a ``Function`` variable, so expressions need no phi nodes. Control flow follows the
structured rules directly: ``if`` is a selection construct, ``for`` a loop construct whose continue target
increments the counter, and ``switch`` an ``OpSwitch`` whose cases all branch to the merge block. ``break`` and
``continue`` branch to the innermost loop's merge block or continue target. The program model never places a
statement after a branch, so no block is unreachable.
"""

from __future__ import annotations

from .printers import float32
from .program import (
    BOOL,
    INT,
    UINT,
    Assign,
    Binary,
    Break,
    Builtin,
    Call,
    Construct,
    Continue,
    Convert,
    Declare,
    For,
    If,
    Literal,
    Load,
    Return,
    Select,
    Switch,
    Swizzle,
    Type,
    Unary,
    Var,
)

_ARITHMETIC = {
    "+": ("OpIAdd", "OpIAdd", "OpFAdd"),
    "-": ("OpISub", "OpISub", "OpFSub"),
    "*": ("OpIMul", "OpIMul", "OpFMul"),
    "/": ("OpSDiv", "OpUDiv", "OpFDiv"),
    "%": ("OpSRem", "OpUMod", None),
    "&": ("OpBitwiseAnd", "OpBitwiseAnd", None),
    "|": ("OpBitwiseOr", "OpBitwiseOr", None),
    "^": ("OpBitwiseXor", "OpBitwiseXor", None),
    "<<": ("OpShiftLeftLogical", "OpShiftLeftLogical", None),
    ">>": ("OpShiftRightArithmetic", "OpShiftRightLogical", None),
}
_COMPARE = {
    "<": ("OpSLessThan", "OpULessThan", "OpFOrdLessThan", None),
    "<=": ("OpSLessThanEqual", "OpULessThanEqual", "OpFOrdLessThanEqual", None),
    ">": ("OpSGreaterThan", "OpUGreaterThan", "OpFOrdGreaterThan", None),
    ">=": (
        "OpSGreaterThanEqual",
        "OpUGreaterThanEqual",
        "OpFOrdGreaterThanEqual",
        None,
    ),
    "==": ("OpIEqual", "OpIEqual", "OpFOrdEqual", "OpLogicalEqual"),
    "!=": ("OpINotEqual", "OpINotEqual", "OpFUnordNotEqual", "OpLogicalNotEqual"),
}
_EXT = {
    "abs": ("SAbs", None, "FAbs"),
    "min": ("SMin", "UMin", "FMin"),
    "max": ("SMax", "UMax", "FMax"),
    "clamp": ("SClamp", "UClamp", "FClamp"),
}
_FLOAT_EXT = {
    "floor": "Floor",
    "ceil": "Ceil",
    "fract": "Fract",
    "sqrt": "Sqrt",
    "sin": "Sin",
    "cos": "Cos",
    "exp": "Exp",
    "mix": "FMix",
    "length": "Length",
}
_KIND_INDEX = {"int": 0, "uint": 1, "float": 2, "bool": 3}


class SpirvPrinter:
    language = "vulkan"

    def __init__(self):
        self.globals = []
        self.known = {}
        self.next_id = 0

    # -- ids, types and constants --------------------------------------------------------------------------------
    def fresh(self, prefix="_"):
        self.next_id += 1
        return f"%{prefix}{self.next_id}"

    def global_id(self, key, name, instruction):
        """Declare ``name = instruction`` once in the global section; ``instruction`` may be a thunk."""
        if key not in self.known:
            text = instruction() if callable(instruction) else instruction
            self.globals.append(f"{name} = {text}")
            self.known[key] = name
        return self.known[key]

    def type(self, type_):
        if type_.is_vector:
            scalar = self.type(type_.scalar())
            return self.global_id(
                ("type", type_),
                f"%v{type_.size}{type_.kind}",
                f"OpTypeVector {scalar} {type_.size}",
            )
        instruction = {
            "int": "OpTypeInt 32 1",
            "uint": "OpTypeInt 32 0",
            "float": "OpTypeFloat 32",
            "bool": "OpTypeBool",
        }[type_.kind]
        return self.global_id(("type", type_), f"%{type_.kind}", instruction)

    def pointer(self, storage, type_id):
        name = f"%ptr_{storage}_{type_id[1:]}"
        return self.global_id(
            ("ptr", storage, type_id), name, f"OpTypePointer {storage} {type_id}"
        )

    def constant(self, type_, value):
        if type_.is_vector:
            scalar = self.constant(type_.scalar(), value)
            parts = " ".join([scalar] * type_.size)
            return self.global_id(
                ("const", type_, value),
                self.fresh("c"),
                lambda: f"OpConstantComposite {self.type(type_)} {parts}",
            )
        type_id = self.type(type_)
        if type_.kind == "bool":
            text = "OpConstantTrue" if value else "OpConstantFalse"
            return self.global_id(
                ("const", type_, value), self.fresh("c"), f"{text} {type_id}"
            )
        if type_.kind == "float":
            value = float32(float(value))
            # spirv-as reads decimal literals with strtof, which rejects subnormals; spirv-dis prints them in hex.
            text = value.hex() if 0 < abs(value) < 2.0**-126 else repr(value)
            key = ("const", type_, value.hex())
            return self.global_id(key, self.fresh("c"), f"OpConstant {type_id} {text}")
        return self.global_id(
            ("const", type_, value), self.fresh("c"), f"OpConstant {type_id} {value}"
        )

    def null(self, type_):
        return self.global_id(
            ("null", type_),
            self.fresh("c"),
            lambda: f"OpConstantNull {self.type(type_)}",
        )

    # -- emission ------------------------------------------------------------------------------------------------
    def emit(self, text):
        self.code.append(text)

    def value(self, type_, instruction):
        result = self.fresh()
        self.emit(f"{result} = {instruction.format(t=self.type(type_))}")
        return result

    def label(self, name):
        self.emit(f"{name} = OpLabel")
        self.terminated = False

    def branch(self, target):
        if not self.terminated:
            self.emit(f"OpBranch {target}")
        self.terminated = True

    def local(self, name, type_):
        pointer = self.fresh(f"var_{name}_")
        self.locals.append(
            f"{pointer} = OpVariable {self.pointer('Function', self.type(type_))} Function"
        )
        self.variables[name] = (pointer, type_)
        return pointer

    # -- expressions ---------------------------------------------------------------------------------------------
    def expr(self, node):
        if isinstance(node, Literal):
            return self.constant(node.type, node.value)
        if isinstance(node, Var):
            pointer, type_ = self.variables[node.name]
            return self.value(type_, f"OpLoad {{t}} {pointer}")
        if isinstance(node, Load):
            return self.value(INT, f"OpLoad {{t}} {self.element('a')}")
        if isinstance(node, Unary):
            operand = self.expr(node.operand)
            op = {"~": "OpNot", "!": "OpLogicalNot"}.get(node.op)
            op = op or ("OpFNegate" if node.type.kind == "float" else "OpSNegate")
            return self.value(node.type, f"{op} {{t}} {operand}")
        if isinstance(node, Binary):
            return self.binary(node)
        if isinstance(node, Select):
            condition = self.expr(node.condition)
            then, otherwise = self.expr(node.then), self.expr(node.otherwise)
            if node.type.is_vector:
                parts = " ".join([condition] * node.type.size)
                condition = self.value(
                    Type("bool", node.type.size), f"OpCompositeConstruct {{t}} {parts}"
                )
            return self.value(
                node.type, f"OpSelect {{t}} {condition} {then} {otherwise}"
            )
        if isinstance(node, Swizzle):
            base = self.expr(node.base)
            indices = ["xyzw".index(c) for c in node.components]
            if len(indices) == 1:
                return self.value(
                    node.type, f"OpCompositeExtract {{t}} {base} {indices[0]}"
                )
            listed = " ".join(map(str, indices))
            return self.value(
                node.type, f"OpVectorShuffle {{t}} {base} {base} {listed}"
            )
        if isinstance(node, Construct):
            args = [self.expr(arg) for arg in node.args]
            if len(args) == 1 and not node.args[0].type.is_vector:
                args = args * node.type.size
            return self.value(node.type, f"OpCompositeConstruct {{t}} {' '.join(args)}")
        if isinstance(node, Convert):
            return self.convert(node)
        if isinstance(node, Builtin):
            return self.builtin(node)
        if isinstance(node, Call):
            args = " ".join(self.expr(arg) for arg in node.args)
            return self.value(
                node.type, f"OpFunctionCall {{t}} %{node.function} {args}".rstrip()
            )
        raise TypeError(f"unknown expression {node!r}")

    def binary(self, node):
        left, right = self.expr(node.left), self.expr(node.right)
        if node.op in ("&&", "||"):
            op = "OpLogicalAnd" if node.op == "&&" else "OpLogicalOr"
        elif node.op in _COMPARE:
            op = _COMPARE[node.op][_KIND_INDEX[node.left.type.kind]]
        else:
            op = _ARITHMETIC[node.op][_KIND_INDEX[node.type.kind]]
        return self.value(node.type, f"{op} {{t}} {left} {right}")

    def convert(self, node):
        source, target = node.operand.type.kind, node.type.kind
        operand = self.expr(node.operand)
        if target == "bool":
            op = "OpFUnordNotEqual" if source == "float" else "OpINotEqual"
            return self.value(
                node.type, f"{op} {{t}} {operand} {self.null(node.operand.type)}"
            )
        if source == "bool":
            one = self.constant(node.type, 1.0 if target == "float" else 1)
            return self.value(
                node.type, f"OpSelect {{t}} {operand} {one} {self.null(node.type)}"
            )
        op = {
            ("int", "uint"): "OpBitcast",
            ("uint", "int"): "OpBitcast",
            ("int", "float"): "OpConvertSToF",
            ("uint", "float"): "OpConvertUToF",
            ("float", "int"): "OpConvertFToS",
            ("float", "uint"): "OpConvertFToU",
        }[(source, target)]
        return self.value(node.type, f"{op} {{t}} {operand}")

    def builtin(self, node):
        args = " ".join(self.expr(arg) for arg in node.args)
        if node.name == "dot":
            return self.value(node.type, f"OpDot {{t}} {args}")
        if node.name in _EXT:
            name = _EXT[node.name][_KIND_INDEX[node.type.kind]]
        else:
            name = _FLOAT_EXT[node.name]
        return self.value(node.type, f"OpExtInst {{t}} %glsl {name} {args}")

    def element(self, buffer):
        index = self.expr(Var("i", UINT))
        pointer = self.pointer("StorageBuffer", self.type(INT))
        zero = self.constant(INT, 0)
        return self.value(INT, f"OpAccessChain {pointer} %{buffer} {zero} {index}")

    # -- statements ----------------------------------------------------------------------------------------------
    def block(self, statements):
        for statement in statements:
            self.statement(statement)

    def statement(self, node):
        if isinstance(node, Declare):
            pointer = self.local(node.name, node.type)
            self.emit(f"OpStore {pointer} {self.expr(node.value)}")
        elif isinstance(node, Assign):
            pointer, type_ = self.variables[node.name]
            value = node.value
            if node.op:
                value = Binary(node.op, Var(node.name, type_), value, type_)
            self.emit(f"OpStore {pointer} {self.expr(value)}")
        elif isinstance(node, If):
            self.statement_if(node)
        elif isinstance(node, For):
            self.statement_for(node)
        elif isinstance(node, Switch):
            self.statement_switch(node)
        elif isinstance(node, Break):
            self.branch(self.loops[-1][0])
        elif isinstance(node, Continue):
            self.branch(self.loops[-1][1])
        elif isinstance(node, Return):
            self.emit(f"OpReturnValue {self.expr(node.value)}")
            self.terminated = True
        else:
            raise TypeError(f"unknown statement {node!r}")

    def statement_if(self, node):
        condition = self.expr(node.condition)
        then, merge = self.fresh("L"), self.fresh("L")
        otherwise = self.fresh("L") if node.otherwise is not None else merge
        self.emit(f"OpSelectionMerge {merge} None")
        self.emit(f"OpBranchConditional {condition} {then} {otherwise}")
        self.terminated = True
        self.label(then)
        self.block(node.then)
        self.branch(merge)
        if node.otherwise is not None:
            self.label(otherwise)
            self.block(node.otherwise)
            self.branch(merge)
        self.label(merge)

    def statement_for(self, node):
        counter = self.local(node.name, INT)
        self.emit(f"OpStore {counter} {self.constant(INT, 0)}")
        header, check, body = self.fresh("L"), self.fresh("L"), self.fresh("L")
        step, merge = self.fresh("L"), self.fresh("L")
        self.branch(header)
        self.label(header)
        self.emit(f"OpLoopMerge {merge} {step} None")
        self.branch(check)
        self.label(check)
        current = self.value(INT, f"OpLoad {{t}} {counter}")
        bound = self.constant(INT, node.count)
        condition = self.value(BOOL, f"OpSLessThan {{t}} {current} {bound}")
        self.emit(f"OpBranchConditional {condition} {body} {merge}")
        self.terminated = True
        self.label(body)
        self.loops.append((merge, step))
        self.block(node.body)
        self.loops.pop()
        self.branch(step)
        self.label(step)
        current = self.value(INT, f"OpLoad {{t}} {counter}")
        one = self.constant(INT, 1)
        self.emit(
            f"OpStore {counter} {self.value(INT, f'OpIAdd {{t}} {current} {one}')}"
        )
        self.branch(header)
        self.label(merge)

    def statement_switch(self, node):
        selector = self.expr(node.selector)
        merge, default = self.fresh("L"), self.fresh("L")
        labels = [(value, self.fresh("L"), body) for value, body in node.cases]
        targets = " ".join(f"{value} {label}" for value, label, _ in labels)
        self.emit(f"OpSelectionMerge {merge} None")
        self.emit(f"OpSwitch {selector} {default} {targets}".rstrip())
        self.terminated = True
        for _, label, body in labels:
            self.label(label)
            self.block(body)
            self.branch(merge)
        self.label(default)
        self.block(node.default)
        self.branch(merge)
        self.label(merge)

    # -- functions -----------------------------------------------------------------------------------------------
    def function(self, name, result, params, body, kernel=None):
        self.code, self.locals, self.variables, self.loops = [], [], {}, []
        result_id = (
            self.type(result)
            if result
            else self.global_id(("void",), "%void", "OpTypeVoid")
        )
        param_types = [self.type(t) for _, t in params]
        signature = " ".join([result_id, *param_types])
        function_type = self.global_id(
            ("fn", signature), self.fresh("fn"), f"OpTypeFunction {signature}"
        )
        header = [f"%{name} = OpFunction {result_id} None {function_type}"]
        values = []
        for (param, _), type_id in zip(params, param_types):
            values.append(self.fresh(f"arg_{param}_"))
            header.append(f"{values[-1]} = OpFunctionParameter {type_id}")
        self.terminated = False
        for (param, type_), value in zip(params, values):
            self.emit(f"OpStore {self.local(param, type_)} {value}")
        if kernel is not None:
            invocation = self.value(Type("uint", 3), "OpLoad {t} %gid")
            index = self.value(UINT, f"OpCompositeExtract {{t}} {invocation} 0")
            self.emit(f"OpStore {self.local('i', UINT)} {index}")
        self.block(body)
        if kernel is not None:
            output = self.expr(kernel.output)
            self.emit(f"OpStore {self.element('o')} {output}")
            self.emit("OpReturn")
        entry = self.fresh("L")
        return (
            header
            + [f"{entry} = OpLabel"]
            + self.locals
            + self.code
            + ["OpFunctionEnd"]
        )

    def print(self, program) -> str:
        int_id = self.type(INT)
        array = self.global_id(("rta",), "%rta_int", f"OpTypeRuntimeArray {int_id}")
        for buffer in ("Inp", "Outp"):
            self.global_id(("struct", buffer), f"%{buffer}", f"OpTypeStruct {array}")
        invocation = self.pointer("Input", self.type(Type("uint", 3)))
        self.global_id(("var", "gid"), "%gid", f"OpVariable {invocation} Input")
        for buffer, struct in (("a", "%Inp"), ("o", "%Outp")):
            pointer = self.pointer("StorageBuffer", struct)
            self.global_id(
                ("var", buffer), f"%{buffer}", f"OpVariable {pointer} StorageBuffer"
            )
        functions = []
        for function in program.functions:
            body = function.body
            functions += self.function(
                function.name, function.result, function.params, body
            )
        functions += self.function("main", None, (), program.body, kernel=program)
        header = [
            "; SPIR-V",
            "; Version: 1.3",
            "; Generator: CrossTL conformance fuzzer; 0",
            "OpCapability Shader",
            '%glsl = OpExtInstImport "GLSL.std.450"',
            "OpMemoryModel Logical GLSL450",
            'OpEntryPoint GLCompute %main "main" %gid',
            "OpExecutionMode %main LocalSize 64 1 1",
            "OpDecorate %gid BuiltIn GlobalInvocationId",
            "OpDecorate %rta_int ArrayStride 4",
            "OpDecorate %Inp Block",
            "OpMemberDecorate %Inp 0 Offset 0",
            "OpMemberDecorate %Inp 0 NonWritable",
            "OpDecorate %Outp Block",
            "OpMemberDecorate %Outp 0 Offset 0",
            "OpDecorate %a DescriptorSet 0",
            "OpDecorate %a Binding 0",
            "OpDecorate %o DescriptorSet 0",
            "OpDecorate %o Binding 1",
        ]
        return "\n".join(header + self.globals + functions) + "\n"

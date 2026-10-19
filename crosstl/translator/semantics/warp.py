"""CUDA warp shuffles in CrossGL programs: an exact workgroup-memory emulation for targets without CUDA warps.

CrossGL programs may call the CUDA warp shuffles ``__shfl_sync``, ``__shfl_up_sync``, ``__shfl_down_sync`` and
``__shfl_xor_sync`` directly. CUDA defines them over a warp: 32 threads of consecutive linear thread index, the
first warp starting at thread 0 (CUDA C++ Programming Guide, "SIMT Architecture" and "Warp Shuffle Functions"; the
source-lane rules are those of PTX ``shfl.sync``). No other target has that unit. Vulkan, Direct3D, Metal, OpenGL
and WebGPU subgroups have an implementation-defined size and no specified mapping from invocations to subgroups,
so a shuffle cannot be lowered to subgroup operations without changing which lanes exchange values. The emitters
used to print the CUDA names verbatim.

For a compute entry point whose declared workgroup holds at most one warp (``W = x * y * z <= 32``), the warp is
the workgroup and a lane's index is ``gl_LocalInvocationIndex``. CUDA requires every lane named in the mask to
execute the same shuffle, so with a mask that names all ``W`` lanes every invocation of the workgroup reaches each
shuffle, and a workgroup barrier there is valid. This pass then rewrites each shuffle into::

    T value = <value>;                    uint source = <PTX source lane, or own lane>;
    exchange[gl_LocalInvocationIndex] = value;   barrier();
    T result = exchange[source];          barrier();

which returns exactly the value CUDA returns: lane ``source``'s value when PTX's range check passes and the lane
exists, and the caller's own value otherwise (a lane at or past ``W`` is inactive; CUDA leaves that result
undefined). The second barrier keeps the next exchange from overwriting a slot still being read.

Every other use is refused with :class:`CudaWarpIntrinsicUnsupportedError`: a workgroup of more than one warp
(warps may diverge, so a workgroup barrier is not valid where a warp shuffle is), an undeclared workgroup size, a
mask that is not a literal naming every lane, a width that is not a literal power of two up to 32, a shuffle
outside a compute entry point, in a conditionally evaluated operand, or in a statement whose other operands have
side effects, and the other warp intrinsics (votes, reductions, matches, ``__syncwarp``). WGSL rejects a barrier
outside uniform control flow (WGSL section 15.2, uniformity analysis), so for WGSL the shuffle must also be at a
point a conservative uniformity check proves uniform.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import (
    TypeScope,
    call_name,
    is_pure_callee,
    iter_functions,
    type_from_name,
)
from .rewrite import call, child_slots, clone, get_child, is_pure, set_child

__all__ = [
    "SHUFFLES",
    "WARP_TARGETS",
    "CudaWarpIntrinsicUnsupportedError",
    "legalize_cuda_warp_intrinsics",
]

WARP_SIZE = 32
SHUFFLES = {
    "__shfl_sync": "idx",
    "__shfl_up_sync": "up",
    "__shfl_down_sync": "down",
    "__shfl_xor_sync": "bfly",
}
OTHER_WARP_INTRINSICS = frozenset(
    {
        "__shfl",
        "__shfl_up",
        "__shfl_down",
        "__shfl_xor",
        "__ballot",
        "__ballot_sync",
        "__any",
        "__any_sync",
        "__all",
        "__all_sync",
        "__uni_sync",
        "__activemask",
        "__syncwarp",
        "__match_any_sync",
        "__match_all_sync",
        "__reduce_add_sync",
        "__reduce_min_sync",
        "__reduce_max_sync",
        "__reduce_and_sync",
        "__reduce_or_sync",
        "__reduce_xor_sync",
    }
)
# Targets without CUDA warps, and why a warp cannot be mapped onto their subgroups.
WARP_TARGETS = {
    "metal": (
        "the SIMD-group width is a property of the device (MSL threads_per_simdgroup)"
    ),
    "directx": (
        "the wave size is driver-defined between 4 and 128 lanes (HLSL WaveGetLaneCount)"
    ),
    "opengl": "gl_SubgroupSize is implementation-chosen (GL_KHR_shader_subgroup)",
    "webgl": "WebGL 2 has neither compute workgroups nor subgroups",
    "vulkan": "subgroupSize is implementation-defined (Vulkan limits, subgroupSize)",
    "wgsl": (
        "subgroup sizes are implementation-chosen powers of two in [4, 128] (WGSL subgroups)"
    ),
    "slang": "the wave size follows the compilation target (WaveGetLaneCount)",
}
_UNIFORM_BUILTINS = frozenset(
    {"gl_WorkGroupID", "gl_NumWorkGroups", "gl_WorkGroupSize"}
)
_UNIFORM_QUALIFIERS = frozenset({"uniform", "const", "constant"})
_ESCAPES = (A.BreakNode, A.ContinueNode, A.ReturnNode)
_LANE = "gl_LocalInvocationIndex"


class CudaWarpIntrinsicUnsupportedError(TranslationDiagnostic):
    """A CUDA warp intrinsic the target cannot compute with CUDA's lane semantics."""

    def __init__(self, target, what, *, reason_code, source_location=None):
        self.reason_code = reason_code
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.cuda-warp-{reason_code}",)
        super().__init__(
            f"{what}; a CUDA warp is 32 threads of consecutive index, and on {target} "
            f"{WARP_TARGETS[target]}",
            target=target,
            code=f"{CODE_PREFIX}{target}-cuda-warp-intrinsic-unsupported",
        )


def _uint(value):
    return A.LiteralNode(int(value), A.PrimitiveType("uint"))


def _binary(left, op, right):
    return A.BinaryOpNode(left, op, right)


def _ident(name):
    return A.IdentifierNode(name)


def _statement(expression):
    return A.ExpressionStatementNode(expression)


def _literal_int(node):
    """The integer value of a literal (``0xFFFFFFFF``, ``32u``), or ``None``."""
    if not isinstance(node, A.LiteralNode) or isinstance(node.value, bool):
        return None
    if isinstance(node.value, int):
        return node.value
    if isinstance(node.value, str):
        text = node.value.strip().rstrip("uUlL")
        try:
            return int(text, 0)
        except ValueError:
            return None
    return None


def _descendants(node):
    yield node
    for field, key in child_slots(node):
        yield from _descendants(get_child(node, field, key))


def _contains(node, ids):
    return any(id(item) in ids for item in _descendants(node))


def _off_path_pure(node, ids):
    """True when every operand that is not an ancestor of a replaced shuffle is free of side effects."""
    if id(node) in ids:
        return True
    if not _contains(node, ids):
        return is_pure(node)
    return all(
        _off_path_pure(get_child(node, field, key), ids)
        for field, key in child_slots(node)
    )


def _value_type(node):
    """A scalar or vector numeric ``TypeNode`` the exchange array can hold, or ``None``."""
    element = node.element_type if isinstance(node, A.VectorType) else node
    if not isinstance(element, A.PrimitiveType):
        return None
    hub = type_from_name(element.name)
    if hub is None or hub.kind not in ("i", "u", "f"):
        return None
    return node


class WarpLegalizer:
    def __init__(self, program, target):
        self.program = program
        self.target = target
        self.scope = TypeScope(program)
        self.user_functions = {
            f.name for f in iter_functions(program) if getattr(f, "body", None)
        }
        self.constants = {
            c.name: _literal_int(getattr(c, "value", None))
            for c in getattr(program, "constants", None) or []
        }
        self.names = {
            value
            for node in program.walk()
            for value in (
                getattr(node, "name", None),
                getattr(node, "identifier", None),
            )
            if isinstance(value, str)
        }
        self.lanes = 0
        self.exchanges = {}
        self.declarations = []
        self.uniform_here = True
        self.tainted = False
        self.locals = frozenset()
        self.nonuniform = set()
        self.uniform_globals = set()

    # Program structure ------------------------------------------------------------------------------------------
    def is_warp_call(self, node):
        if not isinstance(node, A.FunctionCallNode):
            return False
        name = call_name(node)
        return (
            name in SHUFFLES or name in OTHER_WARP_INTRINSICS
        ) and name not in self.user_functions

    def mentions(self, node):
        return node is not None and any(self.is_warp_call(item) for item in node.walk())

    def refuse(self, what, reason_code, node=None):
        raise CudaWarpIntrinsicUnsupportedError(
            self.target,
            what,
            reason_code=reason_code,
            source_location=getattr(node, "source_location", None),
        )

    def fresh(self, stem):
        index = 0
        while f"{stem}_{index}" in self.names:
            index += 1
        name = f"{stem}_{index}"
        self.names.add(name)
        return name

    def run(self):
        if not self.mentions(self.program):
            return False
        for constant in getattr(self.program, "constants", None) or []:
            self.scope.declare(constant.name, constant.const_type)
            self.uniform_globals.add(constant.name)
        self._declare_globals(getattr(self.program, "global_variables", None) or [])
        for function in getattr(self.program, "functions", None) or []:
            self._refuse_inside(function, "a function other than a compute entry point")
        stages = getattr(self.program, "stages", None) or {}
        for stage in stages.values() if isinstance(stages, dict) else stages:
            for function in getattr(stage, "local_functions", None) or []:
                self._refuse_inside(function, "a function other than the entry point")
            entry = getattr(stage, "entry_point", None)
            if not isinstance(entry, A.FunctionNode) or not self.mentions(entry):
                continue
            kind = getattr(getattr(stage, "stage", None), "value", stage.stage)
            if str(kind) != "compute":
                self.refuse(f"a CUDA warp intrinsic in a {kind} stage", "stage", entry)
            self.entry(stage, entry)
        return True

    def _declare_globals(self, variables):
        for variable in variables:
            self.scope.declare(variable.name, variable.var_type)
            if set(variable.qualifiers or []) & _UNIFORM_QUALIFIERS:
                self.uniform_globals.add(variable.name)

    def _refuse_inside(self, function, where):
        if isinstance(function, A.FunctionNode) and self.mentions(function):
            self.refuse(
                f"a CUDA warp intrinsic in {where}", "helper-function", function
            )

    def workgroup_lanes(self, stage, entry):
        config = getattr(stage, "execution_config", None) or {}
        keys = ("local_size_x", "local_size_y", "local_size_z")
        if not any(key in config for key in keys):
            self.refuse(
                "a CUDA warp shuffle in an entry point with no declared workgroup size",
                "workgroup-size",
                entry,
            )
        lanes = 1
        for key in keys:
            text = str(config.get(key, "1")).strip().rstrip("uU")
            value = int(text) if text.isdigit() else self.constants.get(text)
            if value is None or value < 1:
                self.refuse(
                    f"a CUDA warp shuffle with a non-literal workgroup size ({key} = {text})",
                    "workgroup-size",
                    entry,
                )
            lanes *= value
        if lanes > WARP_SIZE:
            self.refuse(
                f"a CUDA warp shuffle in a workgroup of {lanes} invocations: its warps may "
                "diverge, so a workgroup barrier cannot emulate one warp's exchange",
                "multiple-warps",
                entry,
            )
        return lanes

    def entry(self, stage, entry):
        self.lanes = self.workgroup_lanes(stage, entry)
        self.exchanges, self.declarations = {}, []
        self.scope.push()
        self._declare_globals(getattr(stage, "local_variables", None) or [])
        for parameter in entry.parameters or []:
            self.scope.declare(parameter.name, getattr(parameter, "param_type", None))
        if self.target == "wgsl":
            self.nonuniform = self.nonuniform_locals(entry.body)
        self.uniform_here, self.tainted = True, False
        if isinstance(entry.body, list):
            entry.body[:] = self.statements(entry.body)
        else:
            entry.body = self.branch(entry.body)
        self.scope.pop()
        stage.local_variables = list(stage.local_variables or []) + self.declarations

    # Statements -------------------------------------------------------------------------------------------------
    def statements(self, items):
        result = []
        for item in items:
            result.extend(self.statement(item))
        return result

    def branch(self, node, uniform=True):
        """Rewrite a nested statement; ``uniform`` says whether control reaching it stays uniform."""
        if node is None:
            return None
        outer = self.uniform_here
        self.uniform_here = outer and uniform
        try:
            if isinstance(node, list):
                return self.statements(node)
            rewritten = self.statement(node)
            return rewritten[0] if len(rewritten) == 1 else A.BlockNode(rewritten)
        finally:
            self.uniform_here = outer

    def statement(self, node):
        """Rewrite one statement; returns it preceded by the statements its shuffles expand to."""
        if isinstance(node, A.BlockNode):
            self.scope.push()
            node.statements = self.statements(node.statements or [])
            self.scope.pop()
            return [node]
        if isinstance(node, A.VariableNode):
            prelude = self.eager(node, "initial_value", node)
            self.scope.declare(node.name, node.var_type)
            return prelude + [node]
        if isinstance(node, A.ExpressionStatementNode):
            return self.eager(node, "expression", node) + [node]
        if isinstance(node, A.AssignmentNode):
            return self.eager_all(node, ("target", "value")) + [node]
        if isinstance(node, A.ReturnNode):
            prelude = self.eager(node, "value", node)
            self.escape(node)
            return prelude + [node]
        if isinstance(node, (A.BreakNode, A.ContinueNode)):
            self.escape(node)
            return [node]
        if isinstance(node, A.IfNode):
            return self.if_statement(node)
        if isinstance(node, (A.ForNode, A.WhileNode, A.DoWhileNode)):
            return self.loop(node)
        if isinstance(node, A.SwitchNode):
            return self.switch(node)
        if self.mentions(node):
            self.refuse(
                f"a CUDA warp shuffle inside a {type(node).__name__}", "statement", node
            )
        return [node]

    def escape(self, node):
        """A jump at a non-uniform point makes everything after it non-uniform (conservatively, for WGSL)."""
        if not self.uniform_here:
            self.tainted = True

    def if_statement(self, node):
        prelude = self.eager(node, "condition", node)
        uniform = self.is_uniform(node.condition)
        for condition in node.else_if_conditions or []:
            self.forbid(condition, "an else-if condition")
            uniform = uniform and self.is_uniform(condition)
        set_child(node, "then_branch", None, self.branch(node.then_branch, uniform))
        bodies = node.else_if_bodies or []
        bodies[:] = [self.branch(body, uniform) for body in bodies]
        if node.else_branch is not None:
            set_child(node, "else_branch", None, self.branch(node.else_branch, uniform))
        return prelude + [node]

    def loop(self, node):
        prelude = []
        uniform = not self.divergent_exit(node.body)
        self.scope.push()
        if isinstance(node, A.ForNode) and node.init is not None:
            init = self.statement(node.init)
            prelude, node.init = init[:-1], init[-1]
            uniform = uniform and self.is_uniform_statement(node.init)
        for field in ("condition", "update"):
            part = getattr(node, field, None)
            if part is not None:
                self.forbid(part, f"a loop {field}")
                uniform = uniform and self.is_uniform_statement(part)
        set_child(node, "body", None, self.branch(node.body, uniform))
        self.scope.pop()
        if not uniform:
            self.tainted = True
        return prelude + [node]

    def switch(self, node):
        prelude = self.eager(node, "expression", node)
        uniform = self.is_uniform(node.expression)
        for case in node.cases or []:
            self.forbid(getattr(case, "value", None), "a case label")
            outer = self.uniform_here
            self.uniform_here = outer and uniform
            self.scope.push()
            case.statements = self.statements(case.statements or [])
            self.scope.pop()
            self.uniform_here = outer
        if node.default_case is not None:
            node.default_case = self.branch(node.default_case, uniform)
        return prelude + [node]

    def forbid(self, node, where):
        if self.mentions(node):
            self.refuse(
                f"a CUDA warp shuffle in {where}, which is not evaluated exactly once",
                "conditional-evaluation",
                node,
            )

    # Expressions ------------------------------------------------------------------------------------------------
    def eager_all(self, node, fields):
        prelude, results = [], set()
        for field in fields:
            child = getattr(node, field, None)
            if child is not None:
                replacement = self.expression(child, prelude, results)
                if replacement is not child:
                    set_child(node, field, None, replacement)
        if results and not _off_path_pure(node, results):
            self.refuse(
                "a CUDA warp shuffle in a statement whose other operands have side effects",
                "evaluation-order",
                node,
            )
        return prelude

    def eager(self, node, field, owner):
        return (
            self.eager_all(owner, (field,))
            if getattr(node, field, None) is not None
            else []
        )

    def expression(self, node, prelude, results, conditional=False):
        if isinstance(node, (A.FunctionNode, A.LambdaNode)):
            return node
        if self.is_warp_call(node):
            if conditional:
                self.forbid(node, "a conditionally evaluated operand")
            arguments = [
                self.expression(argument, prelude, results)
                for argument in node.arguments or []
            ]
            result = self.shuffle(node, arguments, prelude)
            results.add(id(result))
            return result
        operator = getattr(node, "op", None)
        lazy = set()
        if isinstance(node, A.BinaryOpNode) and operator in ("&&", "||"):
            lazy = {"right"}
        elif isinstance(node, A.TernaryOpNode):
            lazy = {"true_expr", "false_expr"}
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            replacement = self.expression(
                child, prelude, results, conditional or field in lazy
            )
            if replacement is not child:
                set_child(node, field, key, replacement)
        return node

    def shuffle(self, node, arguments, prelude):
        name = call_name(node)
        if name not in SHUFFLES:
            self.refuse(f"CUDA warp intrinsic {name}", "intrinsic", node)
        if len(arguments) not in (3, 4):
            self.refuse(f"{name} with {len(arguments)} arguments", "arguments", node)
        if self.target == "wgsl" and (self.tainted or not self.uniform_here):
            self.refuse(
                f"{name} outside uniform control flow, where WGSL forbids the workgroup "
                "barrier its emulation needs (WGSL section 15.2)",
                "non-uniform-control-flow",
                node,
            )
        mask, value, operand = arguments[:3]
        self.check_mask(name, mask)
        width = self.width(name, arguments[3] if len(arguments) == 4 else None)
        value_type = self.exchange_type(name, value)
        value_name = self.fresh("cgl_warp_value")
        prelude.append(A.VariableNode(value_name, clone_type(value_type), value))
        source_name = self.source_lane(SHUFFLES[name], operand, width, prelude)
        exchange = self.exchange(value_type)
        result_name = self.fresh("cgl_warp_result")
        prelude.extend(
            [
                _statement(
                    A.AssignmentNode(
                        A.ArrayAccessNode(_ident(exchange), _ident(_LANE)),
                        _ident(value_name),
                    )
                ),
                _statement(call("barrier", [])),
                A.VariableNode(
                    result_name,
                    clone_type(value_type),
                    A.ArrayAccessNode(_ident(exchange), _ident(source_name)),
                ),
                _statement(call("barrier", [])),
            ]
        )
        self.scope.declare(value_name, value_type)
        self.scope.declare(result_name, value_type)
        return _ident(result_name)

    def check_mask(self, name, mask):
        value = _literal_int(mask)
        if mask is not None and isinstance(mask, A.IdentifierNode):
            value = self.constants.get(mask.name)
        lanes = (1 << self.lanes) - 1
        if value is None or value & lanes != lanes:
            self.refuse(
                f"{name} with a mask that is not a literal naming all {self.lanes} lanes of "
                "the workgroup, so not every invocation reaches the emulation's barrier",
                "mask",
                mask,
            )

    def width(self, name, node):
        if node is None:
            return WARP_SIZE
        value = _literal_int(node)
        if isinstance(node, A.IdentifierNode):
            value = (
                WARP_SIZE if node.name == "warpSize" else self.constants.get(node.name)
            )
        if value is None or value < 1 or value > WARP_SIZE or value & (value - 1):
            self.refuse(
                f"{name} with a width that is not a literal power of two up to 32",
                "width",
                node,
            )
        return value

    def exchange_type(self, name, value):
        declared = self.scope.node_type(value)
        if declared is None and isinstance(value, A.LiteralNode):
            declared = value.literal_type
        if declared is None:
            inferred = self.scope.infer(value)
            if inferred is not None and inferred.lanes == 1 and inferred.bits == 32:
                declared = A.PrimitiveType(
                    {"i": "int", "u": "uint", "f": "float"}.get(inferred.kind, "bool")
                )
        checked = _value_type(declared) if declared is not None else None
        if checked is None:
            self.refuse(
                f"{name} of a value whose numeric type is unknown", "value-type", value
            )
        return checked

    def exchange(self, value_type):
        key = repr(value_type)
        if key not in self.exchanges:
            name = self.fresh("cgl_warp_exchange")
            self.exchanges[key] = name
            self.declarations.append(
                A.VariableNode(
                    name,
                    A.ArrayType(
                        clone_type(value_type),
                        A.LiteralNode(self.lanes, A.PrimitiveType("int")),
                    ),
                    qualifiers=["shared"],
                )
            )
        return self.exchanges[key]

    def source_lane(self, mode, operand, width, prelude):
        """Declare the lane to read: PTX's source lane when its range check passes and the lane exists, else own."""
        segment = (WARP_SIZE - width) & (WARP_SIZE - 1)
        low = width - 1
        lane = _ident(_LANE)
        if mode == "up":
            delta = self.temporary(
                _binary(call("uint", [operand]), "&", _uint(31)), prelude
            )
            keep = _binary(_binary(_ident(_LANE), "&", _uint(low)), ">=", _ident(delta))
            chosen = A.TernaryOpNode(
                keep, _binary(lane, "-", _ident(delta)), _ident(_LANE)
            )
            return self.temporary(chosen, prelude)
        if mode == "idx":
            bits = _binary(call("uint", [operand]), "&", _uint(low))
            candidate = (
                _binary(_binary(lane, "&", _uint(segment)), "|", bits)
                if segment
                else bits
            )
        else:
            offset = _binary(call("uint", [operand]), "&", _uint(31))
            candidate = _binary(lane, "+" if mode == "down" else "^", offset)
        source = self.temporary(candidate, prelude)
        checks = []
        if mode != "idx" and segment:
            limit = _binary(
                _binary(_ident(_LANE), "&", _uint(segment)), "|", _uint(low)
            )
            checks.append(_binary(_ident(source), "<=", limit))
        if mode == "down" or self.lanes < WARP_SIZE or segment:
            checks.append(_binary(_ident(source), "<", _uint(self.lanes)))
        if not checks:
            return source
        condition = (
            checks[0] if len(checks) == 1 else _binary(checks[0], "&&", checks[1])
        )
        return self.temporary(
            A.TernaryOpNode(condition, _ident(source), _ident(_LANE)), prelude
        )

    def temporary(self, value, prelude):
        name = self.fresh("cgl_warp_lane")
        prelude.append(A.VariableNode(name, A.PrimitiveType("uint"), value))
        self.scope.declare(name, A.PrimitiveType("uint"))
        return name

    # Uniformity (WGSL) ------------------------------------------------------------------------------------------
    def is_uniform(self, node):
        return self.target != "wgsl" or self._uniform_expression(node)

    def is_uniform_statement(self, node):
        if self.target != "wgsl":
            return True
        if isinstance(node, A.VariableNode):
            return node.initial_value is None or self.is_uniform(node.initial_value)
        if isinstance(node, A.ExpressionStatementNode):
            node = node.expression
        if isinstance(node, A.AssignmentNode):
            root = _root_name(node.target)
            return root not in self.nonuniform and self.is_uniform(node.value)
        if isinstance(node, A.UnaryOpNode):
            return _root_name(node.operand) not in self.nonuniform
        return self.is_uniform(node)

    def _uniform_expression(self, node):
        if node is None or isinstance(node, A.LiteralNode):
            return True
        if isinstance(node, A.IdentifierNode):
            if node.name in self.nonuniform:
                return False
            return (
                node.name in self.uniform_globals
                or node.name in _UNIFORM_BUILTINS
                or node.name in self.locals
            )
        if isinstance(node, A.FunctionCallNode):
            if not is_pure_callee(call_name(node)):
                return False
            return all(self._uniform_expression(a) for a in node.arguments or [])
        if isinstance(node, A.UnaryOpNode) and node.op in ("++", "--"):
            return False
        if isinstance(
            node,
            (
                A.MemberAccessNode,
                A.SwizzleNode,
                A.ArrayAccessNode,
                A.BinaryOpNode,
                A.UnaryOpNode,
                A.TernaryOpNode,
                A.CastNode,
            ),
        ):
            return all(
                self._uniform_expression(get_child(node, field, key))
                for field, key in child_slots(node)
            )
        return False

    def nonuniform_locals(self, body):
        """Locals of the entry point that may hold different values in different invocations (a fixed point)."""
        declared = {
            node.name for node in _walk(body) if isinstance(node, A.VariableNode)
        }
        self.locals = frozenset(declared)
        self.nonuniform = set()
        while True:
            before = len(self.nonuniform)
            self._writes(body, False)
            if len(self.nonuniform) == before:
                return self.nonuniform

    def _writes(self, node, divergent):
        if node is None:
            return
        if isinstance(node, list):
            for item in node:
                self._writes(item, divergent)
            return
        if isinstance(node, A.VariableNode):
            if divergent or (
                node.initial_value is not None
                and not self._uniform_expression(node.initial_value)
            ):
                self.nonuniform.add(node.name)
        elif isinstance(node, A.AssignmentNode):
            root = _root_name(node.target)
            if root and (
                divergent
                or not self._uniform_expression(node.value)
                or not self._uniform_expression(node.target)
            ):
                self.nonuniform.add(root)
        elif isinstance(node, A.UnaryOpNode) and node.op in ("++", "--"):
            root = _root_name(node.operand)
            if root and divergent:
                self.nonuniform.add(root)
        nested = divergent
        if isinstance(node, A.IfNode):
            nested = divergent or not all(
                self._uniform_expression(c)
                for c in [node.condition, *(node.else_if_conditions or [])]
            )
        elif isinstance(node, (A.ForNode, A.WhileNode, A.DoWhileNode)):
            nested = (
                divergent
                or not self._uniform_expression(node.condition)
                or self.divergent_exit(node.body)
            )
        elif isinstance(node, A.SwitchNode):
            nested = divergent or not self._uniform_expression(node.expression)
        for field, key in child_slots(node):
            child = get_child(node, field, key)
            self._writes(child, nested if field != "condition" else divergent)

    def divergent_exit(self, body):
        """True when a break, continue or return in ``body`` sits under a non-uniform condition (WGSL only)."""
        if self.target != "wgsl":
            return False

        def search(node, divergent):
            if isinstance(node, _ESCAPES) and divergent:
                return True
            if isinstance(node, A.IfNode):
                divergent = divergent or not all(
                    self._uniform_expression(c)
                    for c in [node.condition, *(node.else_if_conditions or [])]
                )
            elif isinstance(node, (A.ForNode, A.WhileNode, A.DoWhileNode)):
                divergent = divergent or not self._uniform_expression(node.condition)
            elif isinstance(node, A.SwitchNode):
                divergent = divergent or not self._uniform_expression(node.expression)
            return any(
                search(get_child(node, field, key), divergent)
                for field, key in child_slots(node)
            )

        return body is not None and search(body, False)


def _walk(node):
    if isinstance(node, list):
        for item in node:
            yield from _walk(item)
        return
    if isinstance(node, A.ASTNode):
        yield from _descendants(node)


def _root_name(node):
    while isinstance(node, (A.ArrayAccessNode, A.MemberAccessNode, A.SwizzleNode)):
        node = (
            getattr(node, "array_expr", None)
            or getattr(node, "object_expr", None)
            or getattr(node, "vector_expr", None)
        )
    return node.name if isinstance(node, A.IdentifierNode) else None


def clone_type(node):
    return clone(node) if isinstance(node, A.ASTNode) else node


def legalize_cuda_warp_intrinsics(program, target) -> bool:
    """Emulate or refuse ``program``'s CUDA warp intrinsics for ``target``; returns whether any were present."""
    if target not in WARP_TARGETS:
        return False
    return WarpLegalizer(program, target).run()

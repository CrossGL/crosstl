"""CrossGL wave operations in WGSL: the subgroup and quad built-in functions.

WGSL's subgroup built-ins (WGSL, "Subgroup Built-in Functions" and "Quad Operations") compute over the active
invocations of the subgroup, as the CrossGL wave operations do (GL_KHR_shader_subgroup; SPIR-V
``OpGroupNonUniform*``), and naga lowers each to the SPIR-V instruction that the CrossGL operation names:

* reductions ``WaveActive{Sum,Product,Min,Max,BitAnd,BitOr,BitXor}`` are ``subgroup{Add,Mul,Min,Max,And,Or,Xor}``,
  and the scans ``WavePrefix[Inclusive]{Sum,Product}`` are ``subgroup{Exclusive,Inclusive}{Add,Mul}``;
* votes and ballots ``WaveActiveAllTrue``, ``WaveActiveAnyTrue`` and ``WaveActiveBallot`` are ``subgroupAll``,
  ``subgroupAny`` and ``subgroupBallot``;
* lane reads ``WaveReadLaneFirst`` and ``WaveShuffle{Xor,Up,Down}`` are ``subgroupBroadcastFirst`` and
  ``subgroupShuffle{Xor,Up,Down}``. ``WaveReadLaneAt`` is ``subgroupBroadcast`` for a literal lane, which WGSL
  requires to be a const-expression, and ``subgroupShuffle`` otherwise;
* quad reads ``QuadReadAcross{X,Y,Diagonal}`` are ``quadSwap{X,Y,Diagonal}``, and ``QuadReadLaneAt`` with a literal
  lane is ``quadBroadcast``. Vulkan numbers the invocations of a quad by the two low bits of their subgroup
  invocation id, so a lane chosen at run time is the shuffle from ``(subgroup_invocation_id & ~3u) | (lane & 3u)``.

The others are written with those: ``WaveActiveCountBits(b)`` counts the bits of ``subgroupBallot(b)``,
``WavePrefixCountBits(b)`` is the exclusive sum of ``select(0u, 1u, b)``, ``WaveGetLaneIndex()`` and
``WaveGetLaneCount()`` are the ``subgroup_invocation_id`` and ``subgroup_size`` built-in inputs, and
``WaveIsFirstLane()`` compares the invocation id with the first active invocation's, because naga 30 has no
``subgroupElect``.

WGSL's lane reads and quad operations take numbers only, and ``subgroup{And,Or,Xor}`` integers only, so a ``bool``
operand travels as the ``u32`` 0 or 1 and comes back as ``!= 0u``, which is exact for every one of them. The built-in
inputs reach entry points only (:mod:`.wgsl_functions` refuses them in helpers). Like the subgroup built-in inputs,
the functions need no ``enable subgroups;`` directive in naga 30, which rejects it as not yet implemented. Every other
wave operation (the min, max and bitwise scans, clustered reductions, ballot arithmetic, partitions and rotations)
has no WGSL built-in and is refused.
"""

from __future__ import annotations

import re

from ..ast import LiteralNode, WaveOpNode
from ..refusals import WGSLRefusal

__all__ = [
    "WGSL_WAVE_BUILTIN_INPUTS",
    "wgsl_wave_builtin_inputs",
    "wgsl_wave_operation",
    "wgsl_wave_result_type",
]

# Operations that read every operand as a value: name -> (WGSL function, operand kinds it accepts).
_NUMERIC = "numeric"
_INTEGER = "integer"
_WAVE_FUNCTIONS = {
    "WaveActiveSum": ("subgroupAdd", _NUMERIC),
    "WaveActiveProduct": ("subgroupMul", _NUMERIC),
    "WaveActiveMin": ("subgroupMin", _NUMERIC),
    "WaveActiveMax": ("subgroupMax", _NUMERIC),
    "WavePrefixSum": ("subgroupExclusiveAdd", _NUMERIC),
    "WavePrefixProduct": ("subgroupExclusiveMul", _NUMERIC),
    "WavePrefixInclusiveSum": ("subgroupInclusiveAdd", _NUMERIC),
    "WavePrefixInclusiveProduct": ("subgroupInclusiveMul", _NUMERIC),
    "WaveActiveBitAnd": ("subgroupAnd", _INTEGER),
    "WaveActiveBitOr": ("subgroupOr", _INTEGER),
    "WaveActiveBitXor": ("subgroupXor", _INTEGER),
}
# Lane reads: name -> (WGSL function, number of operands). The operands after the value are u32 lane numbers.
_LANE_READS = {
    "WaveReadLaneFirst": ("subgroupBroadcastFirst", 1),
    "WaveShuffleXor": ("subgroupShuffleXor", 2),
    "WaveShuffleUp": ("subgroupShuffleUp", 2),
    "WaveShuffleDown": ("subgroupShuffleDown", 2),
    "QuadReadAcrossX": ("quadSwapX", 1),
    "QuadReadAcrossY": ("quadSwapY", 1),
    "QuadReadAcrossDiagonal": ("quadSwapDiagonal", 1),
}
_BOOLEAN_VOTES = {
    "WaveActiveAllTrue": "subgroupAll",
    "WaveActiveAnyTrue": "subgroupAny",
}

# The built-in inputs that an operation reads: name -> WGSL built-in.
WGSL_WAVE_BUILTIN_INPUTS = {
    "WaveGetLaneIndex": "subgroup_invocation_id",
    "WaveGetLaneCount": "subgroup_size",
    "WaveIsFirstLane": "subgroup_invocation_id",
}

_SCALARS = "bool|i32|u32|f32|f16"
_TYPE = re.compile(
    rf"(?P<scalar>{_SCALARS})|vec(?P<count>[234])<(?P<component>{_SCALARS})>"
)


def _shape(codegen, expression):
    """``(component, count)`` of a scalar or vector operand's WGSL type, ``count`` None for a scalar; else None."""
    value_type = codegen.expression_type(expression)
    if value_type is None:
        return None
    if not isinstance(value_type, str):
        try:
            value_type = codegen.type_name_string(value_type)
        except ValueError:
            return None
    match = _TYPE.fullmatch(str(value_type).replace(" ", ""))
    if match is None:
        return None
    if match.group("scalar"):
        return match.group("scalar"), None
    return match.group("component"), match.group("count")


def _type_text(component, count):
    return component if count is None else f"vec{count}<{component}>"


def _is_literal_lane(expression):
    return (
        isinstance(expression, LiteralNode)
        and isinstance(expression.value, int)
        and not isinstance(expression.value, bool)
        and 0 <= expression.value < 128
    )


def _refuse(operation, reason):
    raise WGSLRefusal(f"WGSL target cannot write wave operation {operation}: {reason}")


def wgsl_wave_builtin_inputs(node):
    """The WGSL built-in input ``node`` reads, if it is a wave operation that reads one, else None."""
    if not isinstance(node, WaveOpNode):
        return None
    if node.operation == "QuadReadLaneAt" and len(node.arguments) == 2:
        return None if _is_literal_lane(node.arguments[1]) else "subgroup_invocation_id"
    return WGSL_WAVE_BUILTIN_INPUTS.get(node.operation)


def wgsl_wave_result_type(codegen, node):
    """The WGSL type name of a wave operation's value, or None when it is not known."""
    operation = node.operation
    if operation in {"WaveActiveAllTrue", "WaveActiveAnyTrue", "WaveIsFirstLane"}:
        return "bool"
    if operation == "WaveActiveBallot":
        return "vec4<u32>"
    if operation in {
        "WaveActiveCountBits",
        "WavePrefixCountBits",
        "WaveGetLaneIndex",
        "WaveGetLaneCount",
    }:
        return "u32"
    if node.arguments:
        shape = _shape(codegen, node.arguments[0])
        if shape is not None:
            return _type_text(*shape)
    return None


def _lane(codegen, expression):
    """A lane number operand as ``u32``."""
    if _is_literal_lane(expression):
        return f"{expression.value}u"
    text = codegen.generate_expression(expression)
    shape = _shape(codegen, expression)
    if shape == ("u32", None):
        return text
    if shape is None or shape[1] is not None or shape[0] not in {"i32", "u32"}:
        _refuse("lane operand", "a lane number must be an integer scalar")
    return f"u32({text})"


def _value(codegen, operation, expression, kinds):
    """``(text, back)``: the operand as WGSL accepts it, and how to turn the result back into its CrossGL type."""
    shape = _shape(codegen, expression)
    if shape is None:
        _refuse(operation, "its operand type is not a WGSL scalar or vector")
    component, count = shape
    text = codegen.generate_expression(expression)
    if component != "bool":
        if kinds == _INTEGER and component not in {"i32", "u32"}:
            _refuse(operation, f"WGSL computes it for integers only, not {component}")
        return text, lambda result: result
    if kinds == _NUMERIC:
        _refuse(operation, "WGSL computes it for numbers only, not bool")
    zero, one = (
        ("0u", "1u")
        if count is None
        else (f"vec{count}<u32>(0u)", f"vec{count}<u32>(1u)")
    )
    return f"select({zero}, {one}, {text})", lambda result: f"({result} != {zero})"


def wgsl_wave_operation(codegen, node):
    """The WGSL expression of the wave operation ``node``; refuses one WGSL cannot express."""
    operation = node.operation
    args = list(node.arguments)
    builtin = WGSL_WAVE_BUILTIN_INPUTS.get(operation)
    if builtin is not None and args:
        _refuse(operation, "it takes no operands")
    if operation in {"WaveGetLaneIndex", "WaveGetLaneCount"}:
        return builtin
    if operation == "WaveIsFirstLane":
        return f"(subgroupBroadcastFirst({builtin}) == {builtin})"
    if operation in _WAVE_FUNCTIONS:
        function, kinds = _WAVE_FUNCTIONS[operation]
        if len(args) != 1:
            _refuse(operation, f"it takes one operand, not {len(args)}")
        text, back = _value(codegen, operation, args[0], kinds)
        return back(f"{function}({text})")
    if operation in _BOOLEAN_VOTES or operation in {
        "WaveActiveBallot",
        "WaveActiveCountBits",
        "WavePrefixCountBits",
    }:
        if len(args) != 1 or _shape(codegen, args[0]) != ("bool", None):
            _refuse(operation, "it takes one bool operand")
        predicate = codegen.generate_expression(args[0])
        if operation in _BOOLEAN_VOTES:
            return f"{_BOOLEAN_VOTES[operation]}({predicate})"
        if operation == "WaveActiveBallot":
            return f"subgroupBallot({predicate})"
        if operation == "WaveActiveCountBits":
            return f"dot(countOneBits(subgroupBallot({predicate})), vec4<u32>(1u))"
        return f"subgroupExclusiveAdd(select(0u, 1u, {predicate}))"
    if operation in _LANE_READS or operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
        function, arity = _LANE_READS.get(operation, (None, 2))
        if len(args) != arity:
            _refuse(operation, f"it takes {arity} operands, not {len(args)}")
        text, back = _value(codegen, operation, args[0], None)
        lanes = [_lane(codegen, lane) for lane in args[1:]]
        if operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            if _is_literal_lane(args[1]):
                function = (
                    "subgroupBroadcast"
                    if operation == "WaveReadLaneAt"
                    else "quadBroadcast"
                )
            elif operation == "WaveReadLaneAt":
                function = "subgroupShuffle"
            else:
                function = "subgroupShuffle"
                lanes = [
                    "((subgroup_invocation_id & 4294967292u) | " f"({lanes[0]} & 3u))"
                ]
        return back(f"{function}({', '.join([text, *lanes])})")
    _refuse(operation, "WGSL has no subgroup built-in function for it")

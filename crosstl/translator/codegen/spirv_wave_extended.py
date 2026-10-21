"""SPIR-V lowering of the CrossGL wave operations that map to one ``OpGroupNonUniform*`` instruction each.

Covers the extended operations of ``crosstl.translator.wave_operations`` (scans with min, max and bitwise
operators, clustered reductions and ballot arithmetic) and the Shader Model 6 operations the arithmetic and vote
paths of the generator did not reach: ``WaveActiveAllEqual``, ``WaveActiveCountBits``, ``WavePrefixCountBits``,
``QuadActiveSum`` and the bitwise reductions of ``bool`` values, which are the logical group operations. Each lowers
to the instruction glslang emits for the matching GL_KHR_shader_subgroup function.
"""

from __future__ import annotations

from ..wave_operations import (
    WAVE_BALLOT_BIT_OPERATIONS,
    WAVE_BITWISE_OPERATORS,
    cluster_size_value,
    wave_operation_operator,
)
from .array_utils import evaluate_literal_int_expression

__all__ = ["SPIRV_SUBGROUP_BUILTINS", "lower_spirv_wave_extension"]

# GLSL subgroup built-in variables beyond gl_SubgroupSize and gl_SubgroupInvocationID: name -> (type, BuiltIn,
# storage class). The masks need GroupNonUniformBallot; gl_SubgroupID and gl_NumSubgroups exist in compute-like
# stages only.
SPIRV_SUBGROUP_BUILTINS = {
    "gl_SubgroupEqMask": ("uvec4", "SubgroupEqMask", "Input"),
    "gl_SubgroupGeMask": ("uvec4", "SubgroupGeMask", "Input"),
    "gl_SubgroupGtMask": ("uvec4", "SubgroupGtMask", "Input"),
    "gl_SubgroupLeMask": ("uvec4", "SubgroupLeMask", "Input"),
    "gl_SubgroupLtMask": ("uvec4", "SubgroupLtMask", "Input"),
    "gl_SubgroupID": ("uint", "SubgroupId", "Input"),
    "gl_NumSubgroups": ("uint", "NumSubgroups", "Input"),
}

_FLOAT = {"float", "double"}
_INTEGER = {"int", "uint"}

# operator -> {component kind -> opcode}
_ARITHMETIC_OPCODES = {
    "sum": {"float": "FAdd", "int": "IAdd", "uint": "IAdd"},
    "product": {"float": "FMul", "int": "IMul", "uint": "IMul"},
    "min": {"float": "FMin", "int": "SMin", "uint": "UMin"},
    "max": {"float": "FMax", "int": "SMax", "uint": "UMax"},
    "and": {"int": "BitwiseAnd", "uint": "BitwiseAnd", "bool": "LogicalAnd"},
    "or": {"int": "BitwiseOr", "uint": "BitwiseOr", "bool": "LogicalOr"},
    "xor": {"int": "BitwiseXor", "uint": "BitwiseXor", "bool": "LogicalXor"},
}

_GROUP_OPERATIONS = {
    "inclusive": "InclusiveScan",
    "exclusive": "ExclusiveScan",
    "cluster": "ClusteredReduce",
    "reduce": "Reduce",
}

_ACTIVE_BOOLEAN_REDUCTIONS = {
    "WaveActiveBitAnd": "and",
    "WaveActiveBitOr": "or",
    "WaveActiveBitXor": "xor",
}

_BALLOT_COUNT_SCANS = {
    "WaveBallotBitCount": "Reduce",
    "WaveBallotInclusiveBitCount": "InclusiveScan",
    "WaveBallotExclusiveBitCount": "ExclusiveScan",
}

_PREDICATE_COUNTS = {
    "WaveActiveCountBits": "Reduce",
    "WavePrefixCountBits": "ExclusiveScan",
}


def _result(codegen, id_value, type_id):
    from .SPIRV_codegen import SpirvId

    result = SpirvId(id_value, type_id.type)
    codegen.value_types[id_value] = type_id
    return result


def _component_kind(codegen, value):
    component = codegen.scalar_or_vector_component_type(value.type)
    component = codegen.normalize_primitive_name(component)
    if component in _FLOAT:
        return "float"
    if component in _INTEGER or component == "bool":
        return component
    return None


def _refuse(codegen, operation, args, reason):
    codegen.refuse_lowering(f"{operation} {reason}")
    return codegen.wave_result_default(operation, args)


def _is_ballot(codegen, value):
    info = codegen.vector_component_type_and_count(value.type.base_type)
    return info is not None and info[1] == 4 and info[0] == "uint"


def _emit_group_arithmetic(codegen, operation, args, operator, group_operation, extra):
    value = args[0]
    kind = _component_kind(codegen, value)
    opcode = _ARITHMETIC_OPCODES[operator].get(kind)
    if opcode is None:
        return _refuse(
            codegen,
            operation,
            args,
            f"has no {operator} operator for {value.type.base_type} values",
        )
    if group_operation == "ClusteredReduce":
        codegen.require_group_non_uniform("GroupNonUniformClustered")
    else:
        codegen.require_group_non_uniform("GroupNonUniformArithmetic")
    result_type = codegen.ensure_registered_type(value.type)
    scope = codegen.subgroup_scope_id()
    id_value = codegen.get_id()
    codegen.emit(
        f"%{id_value} = OpGroupNonUniform{opcode} %{result_type.id} %{scope.id} "
        f"{group_operation} %{value.id}{extra}"
    )
    return _result(codegen, id_value, result_type)


def _lower_scan_or_cluster(codegen, operation, argument_exprs, args):
    operator, scope_name = wave_operation_operator(operation)
    expected = 2 if scope_name == "cluster" else 1
    if len(args) != expected:
        return _refuse(codegen, operation, args, f"requires {expected} argument(s)")
    extra = ""
    if scope_name == "cluster":
        size = cluster_size_value(evaluate_literal_int_expression(argument_exprs[1]))
        if size is None:
            return _refuse(
                codegen,
                operation,
                args,
                "requires a constant power-of-two cluster size",
            )
        cluster = codegen.register_constant(
            size, codegen.register_primitive_type("uint")
        )
        extra = f" %{cluster.id}"
    return _emit_group_arithmetic(
        codegen, operation, args, operator, _GROUP_OPERATIONS[scope_name], extra
    )


def _lower_ballot_operation(codegen, operation, args):
    expected = WAVE_BALLOT_BIT_OPERATIONS[operation]
    if len(args) != expected:
        return _refuse(codegen, operation, args, f"requires {expected} argument(s)")
    if not _is_ballot(codegen, args[0]):
        return _refuse(codegen, operation, args, "requires a uint4 ballot mask")
    codegen.require_group_non_uniform("GroupNonUniformBallot")
    scope = codegen.subgroup_scope_id()
    uint_type = codegen.register_primitive_type("uint")
    bool_type = codegen.register_primitive_type("bool")
    id_value = codegen.get_id()
    mask = args[0]
    scan = _BALLOT_COUNT_SCANS.get(operation)
    if scan is not None:
        codegen.emit(
            f"%{id_value} = OpGroupNonUniformBallotBitCount %{uint_type.id} "
            f"%{scope.id} {scan} %{mask.id}"
        )
        return _result(codegen, id_value, uint_type)
    if operation in {"WaveBallotFindLSB", "WaveBallotFindMSB"}:
        opcode = "FindLSB" if operation == "WaveBallotFindLSB" else "FindMSB"
        codegen.emit(
            f"%{id_value} = OpGroupNonUniformBallot{opcode} %{uint_type.id} "
            f"%{scope.id} %{mask.id}"
        )
        return _result(codegen, id_value, uint_type)
    if operation == "WaveInverseBallot":
        codegen.emit(
            f"%{id_value} = OpGroupNonUniformInverseBallot %{bool_type.id} "
            f"%{scope.id} %{mask.id}"
        )
        return _result(codegen, id_value, bool_type)
    lane = codegen.convert_wave_lane_operand(operation, args[1])
    if lane is None:
        return codegen.wave_result_default(operation, args)
    codegen.emit(
        f"%{id_value} = OpGroupNonUniformBallotBitExtract %{bool_type.id} "
        f"%{scope.id} %{mask.id} %{lane.id}"
    )
    return _result(codegen, id_value, bool_type)


def _lower_predicate_count(codegen, operation, args):
    if len(args) != 1 or args[0].type.base_type != "bool":
        return _refuse(codegen, operation, args, "requires one scalar bool argument")
    codegen.require_group_non_uniform("GroupNonUniformBallot")
    scope = codegen.subgroup_scope_id()
    uint_type = codegen.register_primitive_type("uint")
    ballot_type = codegen.register_vector_type(uint_type, 4)
    ballot = codegen.get_id()
    codegen.emit(
        f"%{ballot} = OpGroupNonUniformBallot %{ballot_type.id} %{scope.id} "
        f"%{args[0].id}"
    )
    id_value = codegen.get_id()
    codegen.emit(
        f"%{id_value} = OpGroupNonUniformBallotBitCount %{uint_type.id} %{scope.id} "
        f"{_PREDICATE_COUNTS[operation]} %{ballot}"
    )
    return _result(codegen, id_value, uint_type)


def _lower_all_equal(codegen, operation, args):
    if len(args) != 1:
        return _refuse(codegen, operation, args, "requires exactly one argument")
    if _component_kind(codegen, args[0]) is None:
        return _refuse(
            codegen, operation, args, "requires a scalar or vector numeric or bool"
        )
    codegen.require_group_non_uniform("GroupNonUniformVote")
    bool_type = codegen.register_primitive_type("bool")
    scope = codegen.subgroup_scope_id()
    id_value = codegen.get_id()
    codegen.emit(
        f"%{id_value} = OpGroupNonUniformAllEqual %{bool_type.id} %{scope.id} "
        f"%{args[0].id}"
    )
    return _result(codegen, id_value, bool_type)


def lower_spirv_wave_extension(codegen, operation, argument_exprs, args):
    """Lower ``operation`` and return its value, or ``None`` when another path of the generator owns it."""
    if wave_operation_operator(operation) is not None:
        return _lower_scan_or_cluster(codegen, operation, argument_exprs, args)
    if operation in WAVE_BALLOT_BIT_OPERATIONS:
        return _lower_ballot_operation(codegen, operation, args)
    if operation in _PREDICATE_COUNTS:
        return _lower_predicate_count(codegen, operation, args)
    if operation == "WaveActiveAllEqual":
        return _lower_all_equal(codegen, operation, args)
    if operation == "QuadActiveSum":
        if len(args) != 1:
            return _refuse(codegen, operation, args, "requires exactly one argument")
        cluster = codegen.register_constant(4, codegen.register_primitive_type("uint"))
        return _emit_group_arithmetic(
            codegen, operation, args, "sum", "ClusteredReduce", f" %{cluster.id}"
        )
    boolean_operator = _ACTIVE_BOOLEAN_REDUCTIONS.get(operation)
    if (
        boolean_operator in WAVE_BITWISE_OPERATORS
        and len(args) == 1
        and _component_kind(codegen, args[0]) == "bool"
    ):
        return _emit_group_arithmetic(
            codegen, operation, args, boolean_operator, "Reduce", ""
        )
    return None

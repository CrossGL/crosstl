"""OpenCL C built-in functions written as the CrossGL built-ins that compute the same values.

Most are renames: the integer functions of OpenCL C 3.0 section 6.15.3 (``mul_hi``, ``mad_hi``, ``abs_diff``,
``add_sat``, ``sub_sat``, ``mad_sat``), the ``native_`` and ``half_`` math functions of section 6.15.2, whose
accuracy is implementation-defined, as the full-accuracy functions, the ``fast_`` geometric functions of section
6.15.5, the ``cl_khr_extended_bit_ops`` functions, and the sub-group functions of the ``cl_khr_subgroup*``
extensions as the GLSL ``KHR_shader_subgroup`` functions CrossGL has.

The rest need a conversion:

- ``as_<type>`` (section 6.4.4.2) reinterprets the bits of a value of the same size, which CrossGL writes as
  ``intBitsToFloat``, ``floatBitsToUint``, ``longBitsToDouble``, ``asfloat16``, ``asint16`` or an integer conversion,
  which keeps the bits of an integer of the same width.
- The scalar relational functions (section 6.15.6) return the ``int`` 1 or 0 of a comparison; ``isnotequal`` is
  true for unordered operands, as ``!=`` is.
- The sub-group predicates take and return ``int``, the CrossGL ones ``bool``.
- ``bitselect`` takes each bit from its second operand where the third has a one, from its first elsewhere; a float
  operand is read and written through its bits.

A call this module does not recognize, such as a vector relational function, is left to the generic path, which
refuses an OpenCL C built-in that has no CrossGL counterpart.
"""

import re

__all__ = ["format_opencl_library_call"]

_RENAMES = {
    "mul_hi": "mulhi",
    "mad_hi": "madhi",
    "abs_diff": "absdiff",
    "add_sat": "addsat",
    "sub_sat": "subsat",
    "mad_sat": "madsat",
    "bit_reverse": "bitfieldReverse",
    "fast_length": "length",
    "fast_distance": "distance",
    "fast_normalize": "normalize",
}
for _prefix in ("native_", "half_"):
    for _name in (
        "cos",
        "exp",
        "exp2",
        "exp10",
        "log",
        "log2",
        "log10",
        "powr",
        "sin",
        "sqrt",
        "tan",
    ):
        _RENAMES[_prefix + _name] = _name
    _RENAMES[_prefix + "rsqrt"] = "inversesqrt"
    _RENAMES[_prefix + "recip"] = "rcp"

_GROUP_OPERATIONS = {
    "add": "Add",
    "mul": "Mul",
    "min": "Min",
    "max": "Max",
    "and": "And",
    "or": "Or",
    "xor": "Xor",
}
_LOGICAL_OPERATIONS = {"logical_and": "And", "logical_or": "Or", "logical_xor": "Xor"}

# Sub-group functions with the arguments and result of their CrossGL counterpart. A broadcast whose lane index is
# not a constant is a shuffle (GLSL's subgroupBroadcast needs a constant).
_SUB_GROUP_RENAMES = {
    "sub_group_broadcast": "subgroupShuffle",
    "sub_group_non_uniform_broadcast": "subgroupShuffle",
    "sub_group_broadcast_first": "subgroupBroadcastFirst",
    "sub_group_shuffle": "subgroupShuffle",
    "sub_group_shuffle_xor": "subgroupShuffleXor",
    "sub_group_shuffle_up": "subgroupShuffleUp",
    "sub_group_shuffle_down": "subgroupShuffleDown",
    "sub_group_rotate": "subgroupRotate",
    "sub_group_clustered_rotate": "subgroupClusteredRotate",
    "sub_group_ballot_bit_count": "subgroupBallotBitCount",
    "sub_group_ballot_inclusive_scan": "subgroupBallotInclusiveBitCount",
    "sub_group_ballot_exclusive_scan": "subgroupBallotExclusiveBitCount",
    "sub_group_ballot_find_lsb": "subgroupBallotFindLSB",
    "sub_group_ballot_find_msb": "subgroupBallotFindMSB",
}
# DXC declares HLSL's WaveActiveBitAnd, WaveActiveBitOr and WaveActiveBitXor for unsigned integers only, and the HLSL
# output does not yet convert a signed operand, so the bitwise sub-group reductions of cl_khr_subgroup_non_uniform_arithmetic
# stay refused until it does; their scans and clustered reductions are lowered otherwise and keep their mapping.
_UNCONVERTED_REDUCTIONS = frozenset({"and", "or", "xor"})
for _op, _crossgl in _GROUP_OPERATIONS.items():
    for _kind in ("", "non_uniform_"):
        if _op not in _UNCONVERTED_REDUCTIONS:
            _SUB_GROUP_RENAMES[f"sub_group_{_kind}reduce_{_op}"] = f"subgroup{_crossgl}"
        _SUB_GROUP_RENAMES[f"sub_group_{_kind}scan_inclusive_{_op}"] = (
            f"subgroupInclusive{_crossgl}"
        )
        _SUB_GROUP_RENAMES[f"sub_group_{_kind}scan_exclusive_{_op}"] = (
            f"subgroupExclusive{_crossgl}"
        )
    _SUB_GROUP_RENAMES[f"sub_group_clustered_reduce_{_op}"] = (
        f"subgroupClustered{_crossgl}"
    )

# Sub-group functions whose operand is an int predicate and whose result is an int.
_SUB_GROUP_PREDICATES = {
    "sub_group_all": "subgroupAll",
    "sub_group_any": "subgroupAny",
    "sub_group_non_uniform_all": "subgroupAll",
    "sub_group_non_uniform_any": "subgroupAny",
}
for _op, _crossgl in _LOGICAL_OPERATIONS.items():
    for _kind in ("", "non_uniform_"):
        _SUB_GROUP_PREDICATES[f"sub_group_{_kind}reduce_{_op}"] = f"subgroup{_crossgl}"
        _SUB_GROUP_PREDICATES[f"sub_group_{_kind}scan_inclusive_{_op}"] = (
            f"subgroupInclusive{_crossgl}"
        )
        _SUB_GROUP_PREDICATES[f"sub_group_{_kind}scan_exclusive_{_op}"] = (
            f"subgroupExclusive{_crossgl}"
        )
    _SUB_GROUP_PREDICATES[f"sub_group_clustered_reduce_{_op}"] = (
        f"subgroupClustered{_crossgl}"
    )

# OpenCL C 3.0 section 6.15.15 and cl_khr_work_group_uniform_arithmetic: the work-group collectives are CrossGL's
# workgroup functions (crosstl/translator/workgroup_operations.py), which combine in local linear id order as well.
_WORK_GROUP_RENAMES = {}
_WORK_GROUP_PREDICATES = {
    "work_group_all": "workgroupAll",
    "work_group_any": "workgroupAny",
}
for _op, _crossgl in _GROUP_OPERATIONS.items():
    _WORK_GROUP_RENAMES[f"work_group_reduce_{_op}"] = f"workgroup{_crossgl}"
    _WORK_GROUP_RENAMES[f"work_group_scan_inclusive_{_op}"] = (
        f"workgroupInclusive{_crossgl}"
    )
    _WORK_GROUP_RENAMES[f"work_group_scan_exclusive_{_op}"] = (
        f"workgroupExclusive{_crossgl}"
    )
for _op, _crossgl in _LOGICAL_OPERATIONS.items():
    _WORK_GROUP_PREDICATES[f"work_group_reduce_{_op}"] = f"workgroup{_crossgl}"
    _WORK_GROUP_PREDICATES[f"work_group_scan_inclusive_{_op}"] = (
        f"workgroupInclusive{_crossgl}"
    )
    _WORK_GROUP_PREDICATES[f"work_group_scan_exclusive_{_op}"] = (
        f"workgroupExclusive{_crossgl}"
    )

_SUB_GROUP_VALUES = {
    "get_max_sub_group_size": "gl_SubgroupSize",
    "get_num_sub_groups": "gl_NumSubgroups",
    "get_enqueued_num_sub_groups": "gl_NumSubgroups",
    "get_sub_group_id": "gl_SubgroupID",
    "get_sub_group_local_id": "gl_SubgroupInvocationID",
    "get_sub_group_eq_mask": "gl_SubgroupEqMask",
    "get_sub_group_ge_mask": "gl_SubgroupGeMask",
    "get_sub_group_gt_mask": "gl_SubgroupGtMask",
    "get_sub_group_le_mask": "gl_SubgroupLeMask",
    "get_sub_group_lt_mask": "gl_SubgroupLtMask",
}

_SCALAR_COMPARISONS = {
    "isequal": "==",
    "isnotequal": "!=",
    "isgreater": ">",
    "isgreaterequal": ">=",
    "isless": "<",
    "islessequal": "<=",
}

_SIZES = {
    "i8": 8,
    "u8": 8,
    "i16": 16,
    "u16": 16,
    "f16": 16,
    "i32": 32,
    "u32": 32,
    "f32": 32,
    "i64": 64,
    "u64": 64,
    "f64": 64,
}
# Text that reads no memory twice in a way a second read could change: no assignment, increment, decrement or call
# (a CrossGL conversion such as ``u32(x)`` is written as a call and is excluded as well).
_SIDE_EFFECT = re.compile(r"\+\+|--|(?<![=!<>])=(?!=)|[A-Za-z_]\w*\s*\(")
# OpenCL C 3.0 section 6.15.17: the number of work-items in the sub-group, fewer than the maximum in a work-group's
# last sub-group when its size is not a multiple. CrossGL sub-groups take the work-group's invocations in order.
_SUB_GROUP_SIZE = (
    "min(gl_SubgroupSize, gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z"
    " - gl_SubgroupID * gl_SubgroupSize)"
)


def _shape(type_name):
    """``(lanes, element)`` of a CrossGL scalar or ``vecN<element>`` type, or ``None``."""
    match = re.fullmatch(r"vec([234])<(\w+)>", type_name or "")
    if match:
        return int(match.group(1)), match.group(2)
    if type_name in _SIZES:
        return 1, type_name
    return None


def _typed(lanes, element):
    return element if lanes == 1 else f"vec{lanes}<{element}>"


def _reinterpret(converter, func_name, arg):
    """``as_<type>(arg)``, or ``None`` when ``func_name`` is not one or the sizes differ."""
    target = converter.convert_hip_type_to_crossgl(func_name[3:])
    target_shape = _shape(target)
    source_shape = _shape(converter.opencl_expression_vector_type(arg))
    if target_shape is None or source_shape is None:
        return None
    (lanes, element), (source_lanes, source_element) = target_shape, source_shape
    if lanes != source_lanes or _SIZES[element] != _SIZES[source_element]:
        return None
    value = converter.visit(arg)
    if element == source_element:
        return value
    if element[0] != "f" and source_element[0] != "f":
        return f"{target}({value})"
    signed = _typed(lanes, f"i{_SIZES[element]}")
    if element[0] == "f":
        if element == "f16":
            return f"asfloat16({value})"
        if element == "f32":
            function = (
                "uintBitsToFloat" if source_element == "u32" else "intBitsToFloat"
            )
            return f"{function}({value})"
        if source_element == "u64":
            value = f"{signed}({value})"
        return f"longBitsToDouble({value})"
    if source_element == "f16":
        return f"{'asuint16' if element == 'u16' else 'asint16'}({value})"
    if source_element == "f32":
        function = "floatBitsToUint" if element == "u32" else "floatBitsToInt"
        return f"{function}({value})"
    bits = f"doubleBitsToLong({value})"
    return bits if element == "i64" else f"{target}({bits})"


def _scalar_operands(converter, args, repeated=True):
    """The text of ``args`` when every one is a scalar, and, when ``repeated``, one that can be read twice; else
    ``None``."""
    texts = []
    for arg in args:
        shape = _shape(converter.opencl_expression_vector_type(arg))
        text = converter.visit(arg)
        if shape is None or shape[0] != 1 or (repeated and _SIDE_EFFECT.search(text)):
            return None
        texts.append(text)
    return texts


def _relational(converter, func_name, args):
    if len(args) != 2:
        return None
    operands = _scalar_operands(converter, args, repeated=func_name == "islessgreater")
    if operands is None:
        return None
    left, right = operands
    if func_name == "islessgreater":
        return f"i32(({left} < {right}) || ({left} > {right}))"
    return f"i32({left} {_SCALAR_COMPARISONS[func_name]} {right})"


def _bitselect(converter, args):
    if len(args) != 3:
        return None
    operands = _scalar_operands(converter, args)
    if operands is None:
        return None
    element = _shape(converter.opencl_expression_vector_type(args[0]))[1]
    if element == "f32":
        a, b, c = (f"floatBitsToUint({text})" for text in operands)
        return f"uintBitsToFloat(({a} & ~{c}) | ({b} & {c}))"
    if element[0] in "iu":
        a, b, c = operands
        return f"(({a} & ~{c}) | ({b} & {c}))"
    return None


def _bitfield(converter, func_name, args):
    """``cl_khr_extended_bit_ops`` bitfield functions, whose offsets and counts are ``uint``; CrossGL's are ``int``."""
    values = [converter.visit(arg) for arg in args]
    if func_name == "bitfield_insert" and len(values) == 4:
        base, insert, offset, count = values
        return f"bitfieldInsert({base}, {insert}, i32({offset}), i32({count}))"
    if len(values) != 3:
        return None
    shape = _shape(converter.opencl_expression_vector_type(args[0]))
    if shape is None:
        return None
    lanes, element = shape
    if element not in ("i32", "u32"):
        return None
    signed = func_name.endswith("_signed")
    result = _typed(lanes, ("i" if signed else "u") + element[1:])
    base, offset, count = values
    if result != _typed(lanes, element):
        base = f"{result}({base})"
    return f"bitfieldExtract({base}, i32({offset}), i32({count}))"


def _work_group_broadcast(converter, args):
    """``work_group_broadcast(a, x[, y[, z]])``: ``a`` of the work-item with local id ``(x, y, z)``.

    CrossGL's ``workgroupBroadcast`` takes the local invocation index, which is OpenCL C's local linear id
    ``(z * size_y + y) * size_x + x`` (section 6.15.1).
    """
    value = converter.visit(args[0])
    ids = [f"u32({converter.visit(arg)})" for arg in args[1:]]
    index = ids[0]
    if len(ids) > 1:
        index = f"{index} + {ids[1]} * gl_WorkGroupSize.x"
    if len(ids) > 2:
        index = f"{index} + {ids[2]} * gl_WorkGroupSize.x * gl_WorkGroupSize.y"
    return f"workgroupBroadcast({value}, {index})"


def format_opencl_library_call(converter, func_name, args):
    """The CrossGL expression of the OpenCL C built-in call ``func_name(args)``, or ``None``."""
    if converter.is_user_defined_function(func_name):
        return None
    if func_name in _RENAMES:
        values = ", ".join(converter.visit(arg) for arg in args)
        return f"{_RENAMES[func_name]}({values})"
    if func_name in ("native_divide", "half_divide") and len(args) == 2:
        left, right = (converter.visit(arg) for arg in args)
        return f"(({left}) / ({right}))"
    if func_name in _SUB_GROUP_RENAMES:
        values = ", ".join(converter.visit(arg) for arg in args)
        return f"{_SUB_GROUP_RENAMES[func_name]}({values})"
    if func_name in _SUB_GROUP_PREDICATES and args:
        values = [f"({converter.visit(args[0])}) != 0"]
        values += [converter.visit(arg) for arg in args[1:]]
        return f"i32({_SUB_GROUP_PREDICATES[func_name]}({', '.join(values)}))"
    if func_name in _WORK_GROUP_RENAMES and len(args) == 1:
        return f"{_WORK_GROUP_RENAMES[func_name]}({converter.visit(args[0])})"
    if func_name in _WORK_GROUP_PREDICATES and len(args) == 1:
        predicate = f"({converter.visit(args[0])}) != 0"
        return f"i32({_WORK_GROUP_PREDICATES[func_name]}({predicate}))"
    if func_name == "work_group_broadcast" and 2 <= len(args) <= 4:
        return _work_group_broadcast(converter, args)
    if not args and func_name in _SUB_GROUP_VALUES:
        return _SUB_GROUP_VALUES[func_name]
    if not args and func_name == "get_sub_group_size":
        return _SUB_GROUP_SIZE
    if not args and func_name == "sub_group_elect":
        return "i32(subgroupElect())"
    if func_name == "sub_group_non_uniform_all_equal" and len(args) == 1:
        return f"i32(subgroupAllEqual({converter.visit(args[0])}))"
    if func_name == "sub_group_ballot" and len(args) == 1:
        return f"subgroupBallot(({converter.visit(args[0])}) != 0)"
    if func_name == "sub_group_inverse_ballot" and len(args) == 1:
        return f"i32(subgroupInverseBallot({converter.visit(args[0])}))"
    if func_name == "sub_group_ballot_bit_extract" and len(args) == 2:
        value, index = (converter.visit(arg) for arg in args)
        return f"i32(subgroupBallotBitExtract({value}, {index}))"
    if func_name.startswith("as_") and len(args) == 1:
        return _reinterpret(converter, func_name, args[0])
    if func_name in _SCALAR_COMPARISONS or func_name == "islessgreater":
        return _relational(converter, func_name, args)
    if func_name == "bitselect":
        return _bitselect(converter, args)
    if func_name in (
        "bitfield_insert",
        "bitfield_extract_signed",
        "bitfield_extract_unsigned",
    ):
        return _bitfield(converter, func_name, args)
    return None

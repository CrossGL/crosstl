"""Parse and check the elementwise kernels MLX generates for a fused tape.

``Compiled::eval_gpu`` (``mlx/backend/metal/compiled.cpp``) builds one kernel
per layout variant from the tape: every non-constant input, the inputs'
strides for a strided variant, every output, then the output shape or the
contiguous element count and, for the dynamic variant, the rank. The host's
``compiled.inc`` ports that ``build_kernel`` verbatim and passes the source of
the variant a launch selects, named by the FNV-1a hash of MLX's library name.
Only the 32-bit index variants run; the large ones index from the data
pointer with signed 64-bit offsets, which the host refuses before launch.
"""

import re

# The headers metal::utils(), unary_ops(), binary_ops() and ternary_ops()
# embed in every compiled library, in that order.
PREAMBLE = "".join(
    f'#include "mlx/backend/metal/kernels/{name}.h"\n'
    for name in ("utils", "unary_ops", "binary_ops", "ternary_ops")
)
ENTRY_PREFIX = "compiled_"
ENTRY = re.compile(
    r"compiled_[0-9a-f]{16}_(?P<variant>contiguous(?:_n)?|strided_(?:[1-7]|dynamic))"
)
# get_type_string's spellings of the dtypes the host carries.
TYPES = {
    "bool": "bool_",
    "uint8_t": "uint8",
    "uint16_t": "uint16",
    "uint32_t": "uint32",
    "uint64_t": "uint64",
    "int8_t": "int8",
    "int16_t": "int16",
    "int32_t": "int32",
    "int64_t": "int64",
    "float16_t": "float16",
    "float": "float32",
    "bfloat16_t": "bfloat16",
    "complex64_t": "complex64",
}
_TYPE = "|".join(re.escape(name) for name in sorted(TYPES, key=len, reverse=True))
_TYPE_GROUP = f"(?:{_TYPE})"
_NAME = r"[A-Z]+"
_INPUT = re.compile(
    rf"    device const (?P<type>{_TYPE})\* (?P<name>{_NAME}) \[\[buffer\((?P<slot>\d+)\)\]\],"
)
_OUTPUT = re.compile(
    rf"    device (?P<type>{_TYPE})\* (?P<name>{_NAME}) \[\[buffer\((?P<slot>\d+)\)\]\],"
)
_SCALARS = {
    "in_strides": (
        re.compile(r"    constant const int64_t\* in_strides \[\[buffer\((\d+)\)\]\],"),
        "int64",
    ),
    "output_shape": (
        re.compile(r"    constant const int\* output_shape \[\[buffer\((\d+)\)\]\],"),
        "int32",
    ),
    "size": (
        re.compile(r"    constant const uint& size \[\[buffer\((\d+)\)\]\],"),
        "uint32",
    ),
    "ndim": (
        re.compile(r"    constant const int& ndim \[\[buffer\((\d+)\)\]\],"),
        "int32",
    ),
}
_TAIL = (
    "    uint3 pos [[thread_position_in_grid]],",
    "    uint3 grid [[threads_per_grid]]) {",
)
# The tape's operations: ``Op()(tmp_A, ...)`` and ``cast_to<T>(tmp_A)``.
_OPERATION = re.compile(
    rf"^  (?P<type>{_TYPE}) tmp_(?P<name>{_NAME}) = "
    rf"(?:(?P<op>[A-Z][A-Za-z0-9]*)\(\)\((?P<args>tmp_{_NAME}(?:, tmp_{_NAME})*)\)"
    rf"|cast_to<(?P<cast>{_TYPE})>\(tmp_(?P<operand>{_NAME})\));$",
    re.M,
)
_DECLARATION = re.compile(
    rf"^  (?:(?P<type>{_TYPE}) tmp_(?P<name>{_NAME}) = "
    rf"|auto tmp_(?P<constant>{_NAME}) = static_cast<(?P<ctype>{_TYPE})>\()",
    re.M,
)
# A host launch covers at most 65535 one-thread groups along an axis.
MAX_GROUPS = 65535
MAX_ELEMENTS = 2**31 - 1


def variant(entry):
    """The layout variant of a compiled entry, refusing any other name."""
    match = ENTRY.fullmatch(entry) if isinstance(entry, str) else None
    if match is None:
        raise ValueError("Unknown compiled entry")
    return match.group("variant")


def parse(entry, source):
    """The parameters of ``entry``'s kernel in ``source``, in buffer order.

    Each parameter is ``(name, dtype, direction)`` with direction 0 for an
    input and 1 for an output; the source must be exactly what the host's
    ``build_kernel`` port emits for that variant.
    """
    kind = variant(entry)
    if not isinstance(source, str) or not source.startswith(PREAMBLE):
        raise ValueError("Compiled source does not start with MLX's kernel headers")
    lines = source[len(PREAMBLE) :].split("\n")
    header = [f'[[host_name("{entry}")]]', f"[[kernel]] void {entry}("]
    if lines[:2] != header or source.count("[[kernel]]") != 1:
        raise ValueError("Compiled source does not declare exactly its entry")
    parameters, index = [], 2
    while index < len(lines) and (match := _INPUT.fullmatch(lines[index])):
        parameters.append((match["name"], TYPES[match["type"]], 0, int(match["slot"])))
        index += 1
    strided = kind.startswith("strided")
    names = {"in_strides"} if strided and parameters else set()
    if names:
        match = _SCALARS["in_strides"][0].fullmatch(lines[index])
        if match is None:
            raise ValueError("Compiled source does not bind its input strides")
        parameters.append(("in_strides", "int64", 0, int(match[1])))
        index += 1
    first_output = len(parameters)
    while index < len(lines) and (match := _OUTPUT.fullmatch(lines[index])):
        parameters.append((match["name"], TYPES[match["type"]], 1, int(match["slot"])))
        index += 1
    if len(parameters) == first_output:
        raise ValueError("Compiled source writes no output")
    expected = ["output_shape"] if strided else ["size"]
    if kind == "strided_dynamic":
        expected.append("ndim")
    for name in expected:
        pattern, dtype = _SCALARS[name]
        match = pattern.fullmatch(lines[index]) if index < len(lines) else None
        if match is None:
            raise ValueError(f"Compiled source does not bind its {name}")
        parameters.append((name, dtype, 0, int(match[1])))
        index += 1
    if tuple(lines[index : index + 2]) != _TAIL:
        raise ValueError("Compiled source parameters differ from build_kernel's")
    slots = [slot for *_, slot in parameters]
    if slots != list(range(len(parameters))) or len(slots) > 31:
        raise ValueError("Compiled source buffer slots differ from build_kernel's")
    names = [name for name, *_ in parameters]
    if len(set(names)) != len(names):
        raise ValueError("Compiled source parameter names repeat")
    return tuple((name, dtype, direction) for name, dtype, direction, _ in parameters)


def operations(source):
    """``(kind, operation, operand dtype)`` of every tape step, in order.

    ``kind`` is ``v_``, ``vv_`` or ``ternary_v`` by operand count, as the
    elementwise families name their kernels; casts are ``("cast", target)``.
    """
    types = {}
    for match in _DECLARATION.finditer(source):
        if match["constant"]:
            types[match["constant"]] = TYPES[match["ctype"]]
        else:
            types[match["name"]] = TYPES[match["type"]]
    steps = []
    for match in _OPERATION.finditer(source):
        if match["cast"]:
            steps.append(("cast", TYPES[match["cast"]], types[match["operand"]]))
            continue
        operands = [name.removeprefix("tmp_") for name in match["args"].split(", ")]
        if any(name not in types for name in operands) or len(operands) > 3:
            raise ValueError("Compiled tape reads an undeclared value")
        kind = {1: "v_", 2: "vv_", 3: "ternary_v"}[len(operands)]
        # Select's first operand is its Boolean condition.
        steps.append((kind, match["op"], types[operands[-1]]))
    if not steps:
        raise ValueError("Compiled source computes no tape step")
    return tuple(steps)


# Operations whose integer results no target defines for some operands: the
# vector families refuse those operands before dispatch, but a fused tape
# computes them from intermediate values the host never sees.
UNCHECKED_INTEGER_OPERATIONS = frozenset(
    {"Divide", "Remainder", "LeftShift", "RightShift"}
)
INEXACT = frozenset({"float16", "float32", "bfloat16", "complex64"})


def require_defined(steps):
    """Refuse tapes with integer divisions, remainders or shifts."""
    for kind, operation, dtype in steps:
        if (
            kind != "cast"
            and operation in UNCHECKED_INTEGER_OPERATIONS
            and dtype not in INEXACT
        ):
            raise ValueError(
                "CrossTL compiled kernels refuse integer divisions, remainders and "
                "shifts, whose operands the host cannot check"
            )
    return steps


def work_per_thread(source):
    """The ``N_`` elements each thread of the kernel covers."""
    found = re.findall(r"^  constexpr int N_ = (\d+);$", source, re.M)
    if len(found) != 1 or int(found[0]) not in {1, 2, 4, 8}:
        raise ValueError("Compiled source does not fix its work per thread")
    return int(found[0])


def strided_inputs(source):
    """The inputs a strided kernel reads at ``index_<name>``, in stride order."""
    return tuple(re.findall(rf"^  uint index_({_NAME}) = ", source, re.M))


def _words(buffer, ctype, count):
    import ctypes

    return list(ctypes.cast(buffer.data, ctypes.POINTER(ctype * count)).contents)


def validate(entry, source, parameters, supplied, threads, execution):
    """Check the native buffers and launch against the kernel's parameters.

    Returns the launch metadata the trace records: the variant, the work per
    thread, the element count or output shape and strides, and the groups.
    """
    import ctypes

    kind = variant(entry)
    if set(supplied) != {name for name, *_ in parameters}:
        raise ValueError("Compiled buffers do not match their kernel")
    for name, dtype, direction in parameters:
        buffer = supplied[name]
        if (
            buffer.dtype != dtype.encode("ascii")
            or not buffer.data
            or buffer.output != direction
            or not 0 < buffer.count <= MAX_ELEMENTS
        ):
            raise ValueError("Compiled buffer type, extent or direction differs")
    groups, size = execution["workgroupCount"], execution["workgroupSize"]
    if (
        size != [1, 1, 1]
        or any(not 0 < extent <= MAX_GROUPS for extent in groups)
        or groups[0] * groups[1] * groups[2] != threads
        or "threadGridSize" in execution
    ):
        raise ValueError("Compiled launch differs from its one-thread groups")
    work = work_per_thread(source)
    if (kind == "contiguous_n") != (work > 1) or (
        kind.startswith("strided")
        and work != (2 if kind == "strided_dynamic" or int(kind[-1]) > 3 else 1)
    ):
        raise ValueError("Compiled work per thread differs from its variant")
    vectors = [name for name, _, direction in parameters if direction == 0]
    outputs = [name for name, _, direction in parameters if direction == 1]
    metadata = {"variant": kind, "workPerThread": work, "groups": list(groups)}
    if kind.startswith("contiguous"):
        elements = _words(supplied["size"], ctypes.c_uint32, 1)[0]
        scalars = set(
            re.findall(rf"^  {_TYPE_GROUP} tmp_({_NAME}) = \1\[0\];$", source, re.M)
        )
        if (
            supplied["size"].count != 1
            or groups[1:] != [1, 1]
            or groups[0] != (elements + work - 1) // work
            or any(
                supplied[name].count != (1 if name in scalars else elements)
                for name in vectors + outputs
                if name != "size"
            )
        ):
            raise ValueError("Contiguous compiled extents differ from their launch")
        metadata["size"] = elements
        return metadata
    shape_buffer = supplied["output_shape"]
    rank = shape_buffer.count
    if kind == "strided_dynamic":
        rank_value = _words(supplied["ndim"], ctypes.c_int32, 1)[0]
        if supplied["ndim"].count != 1 or rank_value != rank or rank < 8:
            raise ValueError("Dynamic compiled rank differs from its shape")
    elif rank != int(kind[-1]):
        raise ValueError("Strided compiled rank differs from its variant")
    shape = _words(shape_buffer, ctypes.c_int32, rank)
    if any(extent < 1 for extent in shape):
        raise ValueError("Strided compiled shape must be positive")
    total = 1
    for extent in shape:
        total *= extent
    dim0, dim1 = shape[-1], shape[-2] if rank > 1 else 1
    expected = [(dim0 + work - 1) // work, dim1, total // (dim0 * dim1)]
    strided = strided_inputs(source)
    names = [
        name for name in vectors if name not in {"in_strides", "output_shape", "ndim"}
    ]
    if (
        groups != expected
        or any(supplied[name].count != total for name in outputs)
        or not set(strided) <= set(names)
    ):
        raise ValueError("Strided compiled extents differ from their launch")
    strides = []
    if strided:
        buffer = supplied["in_strides"]
        if buffer.count != rank * len(strided):
            raise ValueError("Strided compiled strides differ from their inputs")
        flat = _words(buffer, ctypes.c_int64, buffer.count)
        strides = [flat[i * rank : (i + 1) * rank] for i in range(len(strided))]
        for name, steps in zip(strided, strides):
            if any(step < 0 for step in steps):
                raise ValueError("Strided compiled inputs require non-negative strides")
            span = 1 + sum((extent - 1) * step for extent, step in zip(shape, steps))
            if supplied[name].count != span:
                raise ValueError("Strided compiled input span differs from its view")
    if any(supplied[name].count != 1 for name in names if name not in strided):
        raise ValueError("Strided compiled scalars must hold one element")
    metadata.update(shape=shape, strides=dict(zip(strided, strides)))
    return metadata

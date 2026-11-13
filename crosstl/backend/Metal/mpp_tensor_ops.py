"""Metal 4 MetalPerformancePrimitives tensor-operations importer.

Models ``mpp::tensor_ops::matmul2d`` and its cooperative-tensor operands for
the Metal preprocessor so ``auto`` variables bound to cooperative-tensor factory
methods receive a concrete CrossGL cooperative-matrix spelling.

The rewrite runs as a text-level pass immediately after macro expansion and
before template materialisation.  For the Metal round-trip target the original
``mpp::`` source is preserved verbatim by writing it back through the
``PRESERVED_INCLUDE_SENTINEL`` mechanism; the CrossGL translator later routes
that target to the Metal-to-Metal path which accepts the original.

Supported patterns (Metal 4 SDK, Xcode 27)::

    // Descriptor (constexpr or inline literal)
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        M, N, K, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Operator object
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    // Cooperative-tensor factory methods
    auto ct_a = op.template get_left_input_cooperative_tensor<A, B, C>();
    auto ct_b = op.template get_right_input_cooperative_tensor<A, B, C>();
    auto ct_c = op.template get_destination_cooperative_tensor<typeA, typeB, C>();

    // Capacity and index accessors
    for (ushort i = 0; i < ct_a.get_capacity(); i++) { … }
    auto ix = ct_a.get_multidimensional_index(i);  // {column, row}
    ct_a[i] = …;
    op.run(ct_a, ct_b, ct_c);

See also :mod:`.opaque_matrix_contracts` for the ``simdgroup_matrix`` path.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "MPP_MATMUL2D_RECEIVER_RE",
    "rewrite_mpp_tensor_ops_auto_types",
]

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# Matches ``mpp::tensor_ops::matmul2d_descriptor(M, N, K, …)`` in a
# constexpr-auto declaration.  Groups: (name, arg-list).
_DESCRIPTOR_DECL_RE = re.compile(
    r"constexpr\s+auto\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
    r"mpp\s*::\s*tensor_ops\s*::\s*matmul2d_descriptor\s*\("
    r"(?P<args>[^)]+)\)",
    re.DOTALL,
)

# Matches ``mpp::tensor_ops::matmul2d<desc, …> name`` variable declarations.
# Groups: (desc, name).
_MATMUL2D_DECL_RE = re.compile(
    r"mpp\s*::\s*tensor_ops\s*::\s*matmul2d\s*<\s*"
    r"(?P<desc>[A-Za-z_]\w*)"
    r"[^>]*>\s*"
    r"(?P<name>[A-Za-z_]\w*)\s*;",
)

# Matches the cooperative-tensor factory calls:
#   op.template get_{left_input,right_input,destination}_cooperative_tensor<A, B, C>()
_FACTORY_CALL_RE = re.compile(
    r"(?P<op>[A-Za-z_]\w*)"
    r"\s*\.\s*(?:template\s+)?"
    r"(?P<method>get_(?:left_input|right_input|destination)_cooperative_tensor)"
    r"\s*<\s*(?P<types>[^>]+)>\s*\(\s*\)",
)

# Matches the auto declarations produced by factory calls:
#   auto ct_a = op.template get_left_input_cooperative_tensor<half, half, float>();
_AUTO_FACTORY_DECL_RE = re.compile(
    r"(?:const\s+)?auto\s+(?P<varname>[A-Za-z_]\w*)\s*=\s*"
    r"(?P<op>[A-Za-z_]\w*)"
    r"\s*\.\s*(?:template\s+)?"
    r"(?P<method>get_(?:left_input|right_input|destination)_cooperative_tensor)"
    r"\s*<\s*(?P<types>[^>]+)>\s*\(\s*\)\s*;",
    re.DOTALL,
)

# Matches ``remove_addrspace_t<decltype(var)>`` for cooperative-tensor vars.
_REMOVE_ADDRSPACE_DECLTYPE_RE = re.compile(
    r"(?:(?:::)?metal\s*::\s*)?remove_addrspace_t\s*<\s*decltype\s*\(\s*"
    r"(?P<var>[A-Za-z_]\w*)"
    r"\s*\)\s*>",
)

# Matches the opening ``<`` of a template, used for the include guard.
_METAL_PERF_PRIMITIVES_INCLUDE_RE = re.compile(
    r"#\s*include\s*<MetalPerformancePrimitives/MetalPerformancePrimitives\.h>"
)

#: Regex for detecting ``mpp::tensor_ops::matmul2d<`` type references.
#: Used by the preprocessor to add ``matmul2d`` to builtin receiver templates.
MPP_MATMUL2D_RECEIVER_RE = re.compile(
    r"(?<![A-Za-z0-9_:])mpp\s*::\s*tensor_ops\s*::\s*matmul2d\s*<"
)


# ---------------------------------------------------------------------------
# Descriptor parsing
# ---------------------------------------------------------------------------

def _parse_int_literal(text: str) -> int | None:
    """Parse a simple positive integer literal, returning ``None`` on failure."""
    text = text.strip()
    try:
        value = int(text, 10)
        return value if value > 0 else None
    except (ValueError, TypeError):
        return None


def _parse_descriptor_args(arg_text: str) -> tuple[int, int, int, bool, bool, str] | None:
    """Parse ``matmul2d_descriptor(M, N, K, transL, transR, relax, mode)`` args.

    Returns ``(M, N, K, transpose_left, transpose_right, mode_name)`` or
    ``None`` if the args cannot be resolved.
    """
    # Split on top-level commas (no nested angle/paren tracking needed here –
    # descriptor args are simple literals and enum values)
    parts = [p.strip() for p in arg_text.split(",")]
    if len(parts) < 7:
        return None

    m = _parse_int_literal(parts[0])
    n = _parse_int_literal(parts[1])
    k = _parse_int_literal(parts[2])
    if m is None or n is None or k is None:
        return None
    trans_left = parts[3].strip().lower() in ("true", "1")
    trans_right = parts[4].strip().lower() in ("true", "1")
    # Mode: last arg, e.g.
    #   mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
    mode_text = parts[6].strip()
    if "multiply_accumulate" in mode_text:
        mode = "multiply_accumulate"
    elif "multiply" in mode_text:
        mode = "multiply"
    else:
        mode = "multiply_accumulate"
    return m, n, k, trans_left, trans_right, mode


# ---------------------------------------------------------------------------
# CrossGL cooperative-matrix type spelling
# ---------------------------------------------------------------------------

def _cooperative_matrix_type(
    element_type: str, rows: int, cols: int
) -> str:
    """Return the CrossGL uncontracted cooperative-matrix type spelling."""
    return (
        f"CooperativeMatrix<{element_type},{rows},{cols},"
        "subgroup,unspecified,unspecified>"
    )


def _operand_dimensions(
    m: int, n: int, k: int, method: str, trans_left: bool, trans_right: bool
) -> tuple[int, int]:
    """Return (rows, cols) for the named operand factory method."""
    if method == "get_left_input_cooperative_tensor":
        # Left operand: M × K (or K × M if transposed)
        return (k, m) if trans_left else (m, k)
    if method == "get_right_input_cooperative_tensor":
        # Right operand: K × N (or N × K if transposed)
        return (n, k) if trans_right else (k, n)
    # get_destination_cooperative_tensor: M × N
    return (m, n)


def _result_element_type(method: str, type_args: list[str]) -> str:
    """Pick the element type from the factory method's template arguments.

    The convention for all three factory methods in the MPP API is:
    ``<InputTypeA, InputTypeB, AccumulatorType>``
    where the accumulator/output type is the last argument.
    ``remove_addrspace_t<decltype(x)>`` follows the same position rule.
    """
    if not type_args:
        return "float"
    # For destination the accumulator type (3rd arg) is the element type;
    # for the input operands use the corresponding input type (1st or 2nd arg).
    if method == "get_left_input_cooperative_tensor":
        return type_args[0].strip() if type_args else "float"
    if method == "get_right_input_cooperative_tensor":
        return (type_args[1].strip() if len(type_args) > 1 else type_args[0].strip())
    # destination
    return type_args[-1].strip() if type_args else "float"


# ---------------------------------------------------------------------------
# Main rewrite pass
# ---------------------------------------------------------------------------

def rewrite_mpp_tensor_ops_auto_types(
    preprocessor: Any,
    code: str,
) -> str:
    """Rewrite ``mpp::tensor_ops`` auto variables to CrossGL cooperative-matrix types.

    This pass is called from :class:`MetalPreprocessor.preprocess` **after**
    macro expansion and **before** template materialisation.  It performs a
    text-level scan and applies three kinds of rewrites:

    1. Records constexpr ``matmul2d_descriptor`` declarations → (M, N, K, …).
    2. Records ``matmul2d<desc, …> op`` variable declarations → (op → desc-name).
    3. Rewrites ``auto ct = op.get_…_cooperative_tensor<A,B,C>()`` to a
       concrete CrossGL cooperative-matrix type declaration.
    4. Rewrites ``remove_addrspace_t<decltype(ct)>`` to the concrete type.
    5. Skips the whole pass if no ``mpp::`` appears in the source.
    """
    if "mpp" not in code or "tensor_ops" not in code:
        return code

    ignored_spans = preprocessor._find_comment_and_literal_spans(code)

    # ---- Step 1: collect constexpr descriptor declarations ----
    descriptors: dict[str, tuple[int, int, int, bool, bool, str]] = {}
    for match in _DESCRIPTOR_DECL_RE.finditer(code):
        if preprocessor._containing_span(match.start(), ignored_spans) is not None:
            continue
        name = match.group("name")
        parsed = _parse_descriptor_args(match.group("args"))
        if parsed is not None:
            descriptors[name] = parsed
    # ---- Step 2: collect matmul2d variable declarations ----
    # Maps op_name → descriptor_name
    matmul2d_ops: dict[str, str] = {}
    for match in _MATMUL2D_DECL_RE.finditer(code):
        if preprocessor._containing_span(match.start(), ignored_spans) is not None:
            continue
        desc_name = match.group("desc")
        op_name = match.group("name")
        matmul2d_ops[op_name] = desc_name
    # If no ops were found, nothing to rewrite.
    if not matmul2d_ops and not descriptors:
        return code
    # ---- Step 3: rewrite auto cooperative-tensor declarations ----
    # Maps variable_name → CrossGL type spelling (for step 4).
    coop_tensor_types: dict[str, str] = {}
    replacements: list[tuple[int, int, str]] = []
    for match in _AUTO_FACTORY_DECL_RE.finditer(code):
        if preprocessor._containing_span(match.start(), ignored_spans) is not None:
            continue
        op_name = match.group("op")
        if op_name not in matmul2d_ops:
            continue
        desc_name = matmul2d_ops[op_name]
        if desc_name not in descriptors:
            continue
        m, n, k, trans_left, trans_right, mode = descriptors[desc_name]
        method = match.group("method")
        type_args = [t.strip() for t in match.group("types").split(",")]
        rows, cols = _operand_dimensions(m, n, k, method, trans_left, trans_right)
        elem_type = _result_element_type(method, type_args)
        coop_type = _cooperative_matrix_type(elem_type, rows, cols)
        var_name = match.group("varname")
        coop_tensor_types[var_name] = coop_type
        # Replace the auto declaration with a typed one; keep the semicolon.
        # The initializer is replaced with a default-constructed cooperative matrix.
        new_decl = f"{coop_type} {var_name};"
        replacements.append((match.start(), match.end(), new_decl))
    if replacements:
        code = preprocessor._apply_text_replacements(code, replacements)
        # Refresh ignored spans after replacements.
        ignored_spans = preprocessor._find_comment_and_literal_spans(code)
        replacements = []
    # ---- Step 4: rewrite remove_addrspace_t<decltype(var)> ----
    if coop_tensor_types:
        for match in _REMOVE_ADDRSPACE_DECLTYPE_RE.finditer(code):
            if preprocessor._containing_span(match.start(), ignored_spans) is not None:
                continue
            var = match.group("var")
            if var in coop_tensor_types:
                replacements.append((match.start(), match.end(), coop_tensor_types[var]))
        if replacements:
            code = preprocessor._apply_text_replacements(code, replacements)
    return code


# ---------------------------------------------------------------------------
# Type-inference hook for the auto-local collection pass
# ---------------------------------------------------------------------------

def infer_mpp_cooperative_tensor_call_type(
    op_name: str,
    method: str,
    type_args: list[str],
    matmul2d_ops: dict[str, str],
    descriptors: dict[str, tuple[int, int, int, bool, bool, str]],
) -> str | None:
    """Return the CrossGL type for an MPP factory call, or ``None`` if unknown.

    This hook is used by the preprocessor's argument-type inference when it
    encounters ``op.get_*_cooperative_tensor<A, B, C>()`` as an initializer.
    """
    desc_name = matmul2d_ops.get(op_name)
    if desc_name is None:
        return None
    parsed = descriptors.get(desc_name)
    if parsed is None:
        return None
    m, n, k, trans_left, trans_right, mode = parsed
    rows, cols = _operand_dimensions(m, n, k, method, trans_left, trans_right)
    elem_type = _result_element_type(method, type_args)
    return _cooperative_matrix_type(elem_type, rows, cols)


# ---------------------------------------------------------------------------
# Scan helpers exposed for the preprocessor
# ---------------------------------------------------------------------------

def scan_mpp_matmul2d_declarations(
    preprocessor: Any,
    code: str,
) -> tuple[dict[str, tuple[int, int, int, bool, bool, str]], dict[str, str]]:
    """Return (descriptors, matmul2d_ops) collected from *code*.

    ``descriptors`` maps constexpr-descriptor-name → (M, N, K, transL, transR, mode).
    ``matmul2d_ops`` maps operator-variable-name → descriptor-name.
    """
    if "mpp" not in code or "tensor_ops" not in code:
        return {}, {}
    ignored_spans = preprocessor._find_comment_and_literal_spans(code)
    descriptors: dict[str, tuple[int, int, int, bool, bool, str]] = {}
    for match in _DESCRIPTOR_DECL_RE.finditer(code):
        if preprocessor._containing_span(match.start(), ignored_spans) is not None:
            continue
        parsed = _parse_descriptor_args(match.group("args"))
        if parsed is not None:
            descriptors[match.group("name")] = parsed
    matmul2d_ops: dict[str, str] = {}
    for match in _MATMUL2D_DECL_RE.finditer(code):
        if preprocessor._containing_span(match.start(), ignored_spans) is not None:
            continue
        matmul2d_ops[match.group("name")] = match.group("desc")
    return descriptors, matmul2d_ops

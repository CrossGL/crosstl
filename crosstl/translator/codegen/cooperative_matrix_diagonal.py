"""Software lowering of the cooperative-matrix diagonal constructor.

``cooperative_matrix_diagonal(value)`` (and the equivalent one-value
``CooperativeMatrix<...>(value)`` constructor) builds a matrix whose diagonal
elements are ``value`` and whose other elements are zero. It is Metal's
one-value ``simdgroup_matrix<T, Cols, Rows>(T)`` constructor.

The OpenGL and DirectX software lowerings keep each lane's fragment elements
in that lane, so a diagonal needs no cross-lane exchange: lane ``l`` sets its
element ``e`` to ``value`` exactly when the registered fragment mapping places
``(l, e)`` at a row equal to its column. Like transfers, this needs a fragment
mapping whose closed-form lane coordinates are registered and verified;
anything else is refused with a typed diagnostic.
"""

from __future__ import annotations

from typing import Callable

DIAGONAL_REQUIREMENT = "the tile_4x4_row_pair 8x8/32x2 fragment contract"


def registered_mapping_closed_form(
    mapping_name: str | None,
    shape: tuple[int, int],
    distribution: tuple[int, int],
    coordinate: Callable[[int, int], tuple[int, int]] | None,
) -> tuple[str, str] | None:
    """The ``(row, column)`` closed form of a verified fragment mapping.

    ``shape`` is ``(rows, columns)``, ``distribution`` is ``(subgroup_size,
    elements_per_lane)`` and ``coordinate`` is the contract's own
    ``(lane, element) -> (row, column)`` table. The closed form is returned
    only when it reproduces that table for every lane element.
    """
    from .GLSL_codegen import GLSLCodeGen

    closed_form = GLSLCodeGen.GLSL_COOPERATIVE_MATRIX_TRANSFER_MAPPINGS.get(
        mapping_name
    )
    if (
        closed_form is None
        or coordinate is None
        or tuple(shape) != (8, 8)
        or tuple(distribution) != (32, 2)
    ):
        return None
    subgroup_size, elements_per_lane = distribution
    if any(
        coordinate(lane, element)
        != GLSLCodeGen.glsl_cooperative_matrix_transfer_coordinate(lane, element)
        for lane in range(subgroup_size)
        for element in range(elements_per_lane)
    ):
        return None
    return closed_form


def diagonal_element_loop(
    closed_form: tuple[str, str],
    elements_per_lane: int,
    assignment: str,
    zero: str,
    *,
    unroll: str = "",
) -> list:
    """The per-element loop that writes ``value`` on the diagonal only."""
    row, column = closed_form
    lines = []
    if unroll:
        lines.append(f"    {unroll}")
    lines.extend(
        [
            f"    for (uint element = 0u; element < {elements_per_lane}u; ++element) {{",
            "        // The registered lane coordinates of this fragment element.",
            f"        uint row = {row};",
            f"        uint column = {column};",
            f"        {assignment} = row == column ? value : {zero};",
            "    }",
        ]
    )
    return lines


# --- OpenGL -------------------------------------------------------------------


def generate_glsl_cooperative_matrix_diagonal(codegen, node) -> str:
    """Lower a diagonal to the lane-local helper of its result contract."""
    if len(node.arguments) != 1:
        codegen.glsl_cooperative_matrix_operation_error(
            node,
            "OpenGL cooperative-matrix diagonal expects 1 argument, got "
            f"{len(node.arguments)}",
            reason="invalid-argument-count",
        )
    contract = codegen.glsl_cooperative_matrix_result_contract(node, [])
    if contract is None:
        codegen.glsl_cooperative_matrix_operation_error(
            node,
            "OpenGL cooperative-matrix diagonal requires a declared matrix result",
            reason="missing-result-contract",
        )
    mapping = contract["fragment_mapping_contract"]
    closed_form = registered_mapping_closed_form(
        contract["fragment_mapping"],
        (contract["rows"], contract["columns"]),
        (contract["subgroup_size"], contract["elements_per_lane"]),
        getattr(mapping, "coordinate", None),
    )
    if closed_form is None:
        codegen.glsl_cooperative_matrix_operation_error(
            node,
            "OpenGL cooperative-matrix diagonal only supports "
            f"{DIAGONAL_REQUIREMENT}",
            reason="incompatible-fragment-mapping",
            matrix_type=contract["matrix_type"],
        )
    if codegen.software_subgroup_width != 32:
        codegen.glsl_cooperative_matrix_operation_error(
            node,
            "OpenGL cooperative-matrix diagonal requires 32-lane software subgroups",
            reason="exact-subgroup-width-required",
            matrix_type=contract["matrix_type"],
        )
    helper = codegen.glsl_cooperative_matrix_helper_name(contract, "diagonal")
    component = contract["mapped_component_type"]
    value = codegen.generate_expression_with_expected(node.arguments[0], component)
    # The helper parameter evaluates the value once for every element.
    return f"{helper}({component}({value}))"


def generate_glsl_cooperative_matrix_diagonal_helper(
    type_name: str, helper_name: str, contract: dict
) -> str:
    """``T helper(component value)``: ``value`` on the diagonal, zero elsewhere."""
    from .GLSL_codegen import GLSLCodeGen

    component = contract["mapped_component_type"]
    closed_form = GLSLCodeGen.GLSL_COOPERATIVE_MATRIX_TRANSFER_MAPPINGS[
        contract["fragment_mapping"]
    ]
    lines = [
        f"{type_name} {helper_name}({component} value) {{",
        f"    {type_name} result;",
        "    uint lane = gl_LocalInvocationIndex % 32u;",
        *diagonal_element_loop(
            closed_form,
            contract["elements_per_lane"],
            "result.elements[element]",
            f"{component}(0)",
        ),
        "    return result;",
        "}",
    ]
    return "\n".join(lines) + "\n"


# --- DirectX ------------------------------------------------------------------


def generate_hlsl_cooperative_matrix_diagonal(codegen, expr) -> str:
    """Lower a diagonal to a per-type helper called with the logical lane."""
    if len(expr.arguments) != 1:
        raise codegen.directx_cooperative_matrix_operation_error(
            expr,
            "DirectX cooperative matrix diagonal requires exactly one value",
            reason="invalid-operation-arity",
        )
    lowering = codegen.directx_cooperative_matrix_result_lowering(expr)
    if lowering is None:
        raise codegen.directx_cooperative_matrix_operation_error(
            expr,
            "DirectX cooperative matrix diagonal requires a declared matrix result",
            reason="missing-result-contract",
        )
    mapping = lowering["fragment_mapping_contract"]
    closed_form = registered_mapping_closed_form(
        lowering["fragment_mapping"],
        (lowering["rows"], lowering["columns"]),
        (lowering["subgroup_size"], lowering["elements_per_lane"]),
        getattr(mapping, "coordinate", None),
    )
    if closed_form is None:
        raise codegen.directx_cooperative_matrix_operation_error(
            expr,
            f"DirectX cooperative matrix diagonal only supports {DIAGONAL_REQUIREMENT}",
            reason="incompatible-fragment-mapping",
        )
    if codegen.software_subgroup_width != 32:
        raise codegen.directx_cooperative_matrix_operation_error(
            expr,
            "DirectX cooperative matrix diagonal requires 32-lane software subgroups",
            reason="exact-subgroup-width-required",
        )
    invocation = codegen.hlsl_software_subgroup_invocation_expressions.get(
        codegen.current_function_name
    )
    if invocation is None:
        raise codegen.directx_cooperative_matrix_operation_error(
            expr,
            "DirectX cooperative matrix diagonal requires validated software "
            "subgroups",
            reason="software-subgroup-contract-required",
        )
    hlsl_type = lowering["hlsl_type"]
    component = lowering["component_type"]
    helper = codegen.hlsl_software_subgroup_identifier(
        ("matrix", "diagonal", hlsl_type),
        f"__crossgl_software_matrix_diagonal_{hlsl_type}",
    )
    codegen.required_hlsl_cooperative_matrix_diagonals[hlsl_type] = (
        helper,
        component,
        lowering["elements_per_lane"],
        closed_form,
    )
    value = codegen.generate_expression_with_expected(expr.arguments[0], component)
    return f"{helper}(({component})({value}), uint({invocation}) % 32u)"


def generate_hlsl_cooperative_matrix_diagonal_helpers(codegen) -> str:
    """One helper per lowered fragment type, in deterministic order."""
    code = ""
    for hlsl_type, (helper, component, elements, closed_form) in sorted(
        codegen.required_hlsl_cooperative_matrix_diagonals.items()
    ):
        lines = [
            f"{hlsl_type} {helper}({component} value, uint lane) {{",
            f"    {hlsl_type} result;",
            *diagonal_element_loop(
                closed_form,
                elements,
                "result[element]",
                f"({component})0",
                unroll="[unroll]",
            ),
            "    return result;",
            "}",
        ]
        code += "\n".join(lines) + "\n\n"
    return code

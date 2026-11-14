"""Lane-fragment contracts for SIMD-group matrices a module never views lane by lane.

The Metal importer learns a ``simdgroup_matrix`` type's lane-fragment contract from a
``thread_elements()`` view: indexing it or reinterpreting it as a whole-lane vector. A
kernel that only loads, fills, multiplies and stores its matrices never views them, so
their CrossGL types kept an unspecified contract and every target that lowers
cooperative matrices lane by lane (OpenGL and DirectX software lowering) refused them,
even when the caller configured the lane-coordinate mapping those lowerings need.

Without a view no result depends on which lane owns which element: loads, stores,
fills and products agree for every complete mapping. When a mapping is configured, the
matrices the module never views therefore take the configured mapping's contract on the
32-lane SIMD-group the views use. Types whose dimensions are not literal integers, or
for which the mapping has no registered contract, keep the unspecified contract, and
the targets that need one keep refusing them with their typed diagnostics.
"""

from __future__ import annotations

import re

from ...translator.cooperative_matrix import get_cooperative_matrix_fragment_mapping

#: Lane width of a Metal SIMD-group, the same width ``thread_elements()`` views assume.
METAL_SIMD_GROUP_WIDTH = 32
#: Fragment layout of a Metal SIMD-group matrix's lane-local elements.
FRAGMENT_LAYOUT = "metal_thread_elements"
#: Provenance of a contract that no lane-level view in the source constrains.
FRAGMENT_PROVENANCE = "metal_opaque_simdgroup_matrix"

# A CrossGL matrix type exactly as the importer writes it before any contract is known.
_UNCONTRACTED_TYPE = re.compile(
    r"CooperativeMatrix<([A-Za-z_][A-Za-z0-9_]*),([1-9][0-9]*),([1-9][0-9]*),"
    r"subgroup,unspecified,unspecified>"
)


def _contracted_type(
    element_type: str,
    rows: int,
    columns: int,
    mapping: str,
    mapping_provenance: str | None,
) -> str | None:
    """Return the matrix type with the configured mapping's contract, if one exists."""
    if (rows * columns) % METAL_SIMD_GROUP_WIDTH:
        return None
    elements_per_lane = rows * columns // METAL_SIMD_GROUP_WIDTH
    if (
        get_cooperative_matrix_fragment_mapping(
            mapping, rows, columns, METAL_SIMD_GROUP_WIDTH, elements_per_lane
        )
        is None
    ):
        return None
    arguments = [
        element_type,
        str(rows),
        str(columns),
        "subgroup",
        "unspecified",
        "unspecified",
        FRAGMENT_LAYOUT,
        str(METAL_SIMD_GROUP_WIDTH),
        str(elements_per_lane),
        FRAGMENT_PROVENANCE,
        mapping,
    ]
    if mapping_provenance:
        arguments.append(mapping_provenance)
    return f"CooperativeMatrix<{','.join(arguments)}>"


def complete_opaque_matrix_contracts(
    code: str,
    replacements: dict[str, str],
    mapping: str | None,
    mapping_provenance: str | None,
) -> None:
    """Add contracts for the uncontracted matrix types ``code`` uses to ``replacements``.

    ``replacements`` maps an uncontracted CrossGL matrix type to its contracted type;
    entries a lane-level view already registered are left unchanged.
    """
    if mapping is None:
        return
    for match in _UNCONTRACTED_TYPE.finditer(code):
        uncontracted = match.group(0)
        if uncontracted in replacements:
            continue
        contracted = _contracted_type(
            match.group(1),
            int(match.group(2)),
            int(match.group(3)),
            mapping,
            mapping_provenance,
        )
        if contracted is not None:
            replacements[uncontracted] = contracted

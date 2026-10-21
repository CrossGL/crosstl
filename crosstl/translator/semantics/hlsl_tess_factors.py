"""Tessellation factor arrays sized to the patch domain for HLSL.

GLSL declares ``gl_TessLevelOuter[4]`` and ``gl_TessLevelInner[2]`` for every domain, and HLSL that DXC compiles to
SPIR-V may do the same; DXIL requires ``SV_TessFactor`` and ``SV_InsideTessFactor`` to hold exactly the factors of
the domain (D3D11 functional specification section 10.3: 3 and 1 for ``tri``, 4 and 2 for ``quad``, 2 and none for
``isoline``; DXC's validator rejects other sizes). The tessellator reads only those factors, so a patch-constant
struct whose factor arrays are longer than the domain's is narrowed to the domain's count when no expression can
observe the difference: every use indexes the array with a constant below that count. Other uses are refused with
:class:`TessFactorUnsupportedError`.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hlsl_ray_interfaces import _functions, _walk
from .stage_varyings import literal_int, member_type, semantic, stage_name, stages

__all__ = ["TessFactorUnsupportedError", "fit_tess_factors_to_domain"]

HLSL_TARGETS = frozenset({"directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
# Factor semantic -> index into the domain's (outer, inner) counts.
_ROLES = {
    "gl_tesslevelouter": 0,
    "sv_tessfactor": 0,
    "gl_tesslevelinner": 1,
    "sv_insidetessfactor": 1,
}
_COUNTS = {"tri": (3, 1), "triangle": (3, 1), "quad": (4, 2), "isoline": (2, 0)}
_TESSELLATION_STAGES = ("tessellation_control", "tessellation_evaluation")


class TessFactorUnsupportedError(TranslationDiagnostic):
    """A tessellation factor array longer than its domain's that is used other than through constant indices."""

    def __init__(self, detail):
        self.missing_capabilities = ("directx.tess-factor-array",)
        super().__init__(
            f"{detail}; DXIL requires SV_TessFactor and SV_InsideTessFactor to hold exactly the domain's factors",
            target="directx",
            code=f"{CODE_PREFIX}directx-tess-factor-unsupported",
        )


def _domain(program):
    found = set()
    for stage in stages(program):
        entry = getattr(stage, "entry_point", None)
        if stage_name(stage) not in _TESSELLATION_STAGES or entry is None:
            continue
        for attribute in entry.attributes or []:
            if str(getattr(attribute, "name", "")).lower() != "domain":
                continue
            arguments = attribute.arguments or []
            if len(arguments) == 1:
                value = getattr(arguments[0], "value", arguments[0])
                found.add(str(value).strip().strip("\"'").lower())
    return found.pop() if len(found) == 1 else None


def _structs(program):
    yield from getattr(program, "structs", None) or []
    for stage in stages(program):
        yield from getattr(stage, "local_structs", None) or []


def _oversized_factors(program, counts):
    """Member name -> (member, domain count) for each factor array longer than the domain's count."""
    found = {}
    for struct in _structs(program):
        for member in getattr(struct, "members", None) or []:
            role = _ROLES.get(str(semantic(member) or "").lower())
            kind = member_type(member)
            if role is None or not isinstance(kind, A.ArrayType):
                continue
            size, count = literal_int(kind.size), counts[role]
            if size is not None and count and size > count:
                found[member.name] = (member, count)
    return found


def fit_tess_factors_to_domain(program, target) -> int:
    """Narrow ``program``'s oversized tessellation factor arrays for HLSL; returns how many members changed."""
    if str(target or "").lower() not in HLSL_TARGETS or not isinstance(
        program, A.ShaderNode
    ):
        return 0
    counts = _COUNTS.get(_domain(program))
    if counts is None:
        return 0
    oversized = _oversized_factors(program, counts)
    if not oversized:
        return 0
    indexed = set()
    nodes = [node for function in _functions(program) for node in _walk(function)]
    for node in nodes:
        array = getattr(node, "array_expr", None)
        if not (
            isinstance(node, A.ArrayAccessNode)
            and isinstance(array, A.MemberAccessNode)
            and array.member in oversized
        ):
            continue
        index, count = literal_int(node.index_expr), oversized[array.member][1]
        if index is None or not 0 <= index < count:
            raise TessFactorUnsupportedError(
                f"tessellation factor '{array.member}' is indexed with a value that may reach its "
                f"elements past the domain's {count}"
            )
        indexed.add(id(array))
    for node in nodes:
        if (
            isinstance(node, A.MemberAccessNode)
            and node.member in oversized
            and id(node) not in indexed
        ):
            raise TessFactorUnsupportedError(
                f"tessellation factor array '{node.member}' is used as a whole, so narrowing it to the "
                "domain's count would change the value"
            )
    for member, count in oversized.values():
        member.member_type = A.ArrayType(member_type(member).element_type, count)
    return len(oversized)

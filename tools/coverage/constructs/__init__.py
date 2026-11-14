"""Language-level features: control flow, operators, types, compute and graphics built-ins, resources and stages.

The built-in function lists in :mod:`..features` say nothing about what a language can declare, how it branches, what
its operators compute or which stages it has. The features here fill that in. Each one is anchored to the SPIR-V core
grammar items it needs (an instruction, a capability, a storage class, a built-in, an image dimension or format, or an
execution model), and the anchors are checked against the pinned grammar. The stages are the grammar's shader
execution models and the scalar types its width capabilities, so a new stage or width upstream shows up as untested
until it has a probe. The curated features are listed by category in :mod:`.language`, :mod:`.compute`,
:mod:`.resources` and :mod:`.graphics`.

Probe templates are written per language: CrossGL's in :mod:`..crossgl`, the others in :mod:`..templates`. A language
with no spelling for a feature, such as WGSL for a geometry stage, leaves it out of its feature list; one that has the
feature but no template yet reports it as untested.
"""

from __future__ import annotations

import json
import re
from string import Template as _Text

from .. import crossgl, templates
from ..dialect import ProbeError
from ..model import Feature, Type
from ..templates import ABSENT, Template
from ..validators import GLSL_STAGES
from . import compute, graphics, language, resources
from .base import Construct

CATEGORIES = (
    "control-flow",
    "operator",
    "type",
    "compute",
    "graphics",
    "resource",
    "stage",
)
# The capabilities that make an execution model a shader stage; Kernel and the NV-only mesh models are left out.
_STAGE_CAPABILITIES = frozenset(
    {"Shader", "Geometry", "Tessellation", "MeshShadingEXT", "RayTracingKHR"}
)
# The scalar types each width capability adds.
_WIDTHS = {
    "Float16": ("half",),
    "Float64": ("double",),
    "Int8": ("int8", "uint8"),
    "Int16": ("int16", "uint16"),
    "Int64": ("int64", "uint64"),
}
# The buffers a kernel probe may declare, bound in this order. A probe declares the input buffers its template reads
# and always the output buffer, so one that reads a single input binds it at 0 and the output at 1, as the
# conformance harness that runs probes binds them.
_BUFFERS = (("float", "fin"), ("int", "iin"), ("float", "outp"))
_INPUTS = re.compile(r"\b(fin|iin)\b")
# The languages whose probes are compiled to the SPIR-V input probes (:mod:`..spirv_input`).
_SPIRV_SOURCES = ("glsl", "hlsl")

# The curated features, by category. The stages and scalar widths are added from the grammar.
CURATED: tuple[Construct, ...] = (
    language.CONTROL_FLOW
    + language.OPERATORS
    + language.TYPES
    + compute.COMPUTE
    + resources.RESOURCES
    + graphics.GRAPHICS
)


def _kinds(grammar: dict) -> dict[str, list[dict]]:
    return {k["kind"]: k.get("enumerants", []) for k in grammar["operand_kinds"]}


def _stage_name(enumerant: str) -> str:
    """``TessellationControl`` -> ``tessellation-control``, ``GLCompute`` -> ``compute``, ``AnyHitKHR`` -> ``any-hit``."""
    bare = re.sub(r"(KHR|EXT|NV)$", "", enumerant).removeprefix("GL")
    return re.sub(r"(?<!^)(?=[A-Z])", "-", bare).lower()


def universe(grammar_text: str) -> list[Construct]:
    """Every language-level feature, in report order; raises ``ValueError`` if an anchor is not in the grammar."""
    grammar = json.loads(grammar_text)
    kinds = _kinds(grammar)
    capabilities = {e["enumerant"] for e in kinds["Capability"]}
    found = {i["opname"] for i in grammar["instructions"]}
    for kind in (
        "StorageClass",
        "BuiltIn",
        "Decoration",
        "ExecutionModel",
        "ExecutionMode",
        "Dim",
        "ImageFormat",
    ):
        found |= {f"{kind}.{e['enumerant']}" for e in kinds[kind]}
    found |= {f"Capability.{name}" for name in capabilities}
    constructs = list(CURATED)
    for capability, names in _WIDTHS.items():
        anchor = f"Capability.{capability}"
        constructs += [Construct("type", name, (anchor,)) for name in names]
    for model in kinds["ExecutionModel"]:
        if _STAGE_CAPABILITIES & set(model.get("capabilities", ())):
            name = _stage_name(model["enumerant"])
            anchor = f"ExecutionModel.{model['enumerant']}"
            constructs.append(Construct("stage", name, (anchor,), name))
    missing = sorted({a for c in constructs for a in c.anchors} - found)
    if missing:
        raise ValueError(
            f"anchors missing from the SPIR-V grammar: {', '.join(missing)}"
        )
    return constructs


def _template(name: str, language: str) -> Template | None:
    table = (
        crossgl.CONSTRUCTS if language == "crossgl" else templates.TEMPLATES[language]
    )
    return table.get(name)


def features(language: str, grammar_text: str) -> list[Feature]:
    """The language-level features ``language`` has.

    SPIR-V has every one that GLSL or HLSL spells, since they are SPIR-V's own items and its input probes are compiled
    from those two languages (:mod:`..spirv_input`). A construct that neither spells, such as OpenCL C's ``.hi``, is
    another language's name for instructions that other constructs already measure.
    """
    result = []
    for construct in universe(grammar_text):
        if language == "spirv":
            if not any(spells(construct.name, source) for source in _SPIRV_SOURCES):
                continue
            template = None
        else:
            template = _template(construct.name, language)
        if template is ABSENT:
            continue
        result.append(
            Feature(
                language,
                construct.category,
                construct.name,
                None,
                group=" ".join(construct.anchors),
                stage=construct.stage,
                target=template.target if template else "",
            )
        )
    return result


def is_construct(feature: Feature) -> bool:
    return feature.category in CATEGORIES


def spells(name: str, language: str) -> bool:
    """Whether ``language`` has a spelling for the construct ``name``: a probe template, or none yet but not
    :data:`ABSENT` either."""
    return _template(name, language) is not ABSENT


def semantic_id(feature: Feature) -> str:
    """Constructs are defined independently of any language, so the construct itself is the semantic feature."""
    return f"construct:{feature.category}/{feature.name}"


def _spellings(dialect) -> dict[str, str]:
    """Type placeholders a template may use: ``$vec4`` is ``float4`` in HLSL and Metal."""
    names = {"float": "float", "int": "int", "uint": "uint"}
    spelled = {key: dialect.type(Type(value)) for key, value in names.items()}
    for size in (2, 3, 4):
        spelled[f"vec{size}"] = dialect.type(Type("float", (size,)))
        spelled[f"ivec{size}"] = dialect.type(Type("int", (size,)))
        spelled[f"uvec{size}"] = dialect.type(Type("uint", (size,)))
    try:
        spelled["mat2"] = dialect.type(Type("float", (2, 2)))
        spelled["mat2x3"] = dialect.type(Type("float", (2, 3)))
        spelled["mat3"] = dialect.type(Type("float", (3, 3)))
        spelled["mat4"] = dialect.type(Type("float", (4, 4)))
    except ProbeError:
        pass  # the language lacks these matrices, and its templates do not use them
    try:
        spelled["bvec2"] = dialect.type(Type("bool", (2,)))
        spelled["bvec4"] = dialect.type(Type("bool", (4,)))
    except ProbeError:
        pass  # the language has no bool vectors, and its templates use neither $bvec2 nor $bvec4
    return spelled


def _lines(text: str, spelled: dict[str, str]) -> list[str]:
    return _Text(text).substitute(spelled).splitlines() if text else []


# GLSL tools tell a program's stage from its file suffix.
_PROGRAM_SUFFIX = {
    language: {stage: f".{suffix}" for stage, suffix in GLSL_STAGES.items()}
    for language in ("glsl", "webgl")
}


def build(feature: Feature, dialect):
    """The probe of a language-level feature in ``dialect``'s language, as ``(source, entry, suffix)``."""
    language = dialect.language
    template = _template(feature.name, language)
    if template is ABSENT:
        raise ProbeError(f"{language} has no spelling for {feature.name}")
    if template is None:
        raise ProbeError(f"no {language} probe template for {feature.name} yet")
    spelled = _spellings(dialect)
    if template.program:
        source = _Text(template.program).substitute(spelled)
        suffix = _PROGRAM_SUFFIX.get(language, {}).get(feature.stage, dialect.suffix)
        return source, template.entry or dialect.entry, suffix
    parts = (template.body, template.prelude, template.declarations, template.params)
    read = set(_INPUTS.findall("\n".join(parts)))
    buffers = [
        (dialect.type(Type(scalar)), name)
        for scalar, name in _BUFFERS
        if name == "outp" or name in read
    ]
    if template.output:
        buffers[-1] = (_Text(template.output).substitute(spelled), "outp")
    source = dialect.kernel(
        buffers,
        _lines(template.body, spelled),
        prelude=_lines(template.prelude, spelled),
        declarations=_lines(template.declarations, spelled),
        params=_lines(template.params, spelled),
    )
    return source, dialect.entry, dialect.suffix

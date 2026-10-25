"""Language-level features: control flow, operators, types, compute, resources and shader stages.

The built-in function lists in :mod:`.features` say nothing about what a language can declare, how it branches, what
its operators compute or which stages it has. The features here fill that in. Each one is anchored to the SPIR-V core grammar items it needs (an
instruction, a capability, a storage class, a built-in or an execution model), and the anchors are checked against the
pinned grammar. The stages are the grammar's shader execution models and the scalar types its width capabilities, so
a new stage or width upstream shows up as untested until it has a probe.

Probe templates are written per language: CrossGL's in :mod:`.crossgl`, the others in :mod:`.templates`. A language
with no spelling for a feature, such as WGSL for a geometry stage, leaves it out of its feature list; one that has the
feature but no template yet reports it as untested.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from string import Template as _Text

from . import crossgl, templates
from .dialect import ProbeError
from .model import Feature, Type
from .templates import ABSENT, Template
from .validators import GLSL_STAGES

CATEGORIES = ("control-flow", "operator", "type", "compute", "resource", "stage")
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


@dataclass(frozen=True)
class Construct:
    category: str
    name: str
    anchors: tuple[str, ...]
    stage: str = "compute"


def _curated() -> list[Construct]:
    def c(category: str, name: str, *anchors: str, stage: str = "compute"):
        return Construct(category, name, anchors, stage)

    return [
        c("control-flow", "if-else", "OpSelectionMerge", "OpBranchConditional"),
        c("control-flow", "switch", "OpSwitch"),
        c("control-flow", "for-loop", "OpLoopMerge"),
        c("control-flow", "while-loop", "OpLoopMerge"),
        c("control-flow", "do-while-loop", "OpLoopMerge"),
        c("control-flow", "break", "OpLoopMerge", "OpBranch"),
        c("control-flow", "continue", "OpLoopMerge", "OpBranch"),
        c("control-flow", "early-return", "OpReturn"),
        c("control-flow", "conditional-expression", "OpSelect"),
        c("control-flow", "short-circuit-logic", "OpLogicalAnd", "OpLogicalOr"),
        c("control-flow", "function-call", "OpFunctionCall"),
        c("control-flow", "out-parameter", "OpFunctionCall", "StorageClass.Function"),
        c("control-flow", "inout-parameter", "OpFunctionCall", "StorageClass.Function"),
        c(
            "control-flow",
            "discard",
            "OpKill",
            "OpDemoteToHelperInvocation",
            stage="fragment",
        ),
        c("operator", "integer-arithmetic", "OpIAdd", "OpISub", "OpIMul"),
        c("operator", "float-arithmetic", "OpFAdd", "OpFSub", "OpFMul", "OpFDiv"),
        c("operator", "integer-division", "OpSDiv", "OpSMod"),
        c("operator", "unsigned-division", "OpUDiv", "OpUMod"),
        c("operator", "signed-shift", "OpShiftLeftLogical", "OpShiftRightArithmetic"),
        c("operator", "unsigned-shift", "OpShiftLeftLogical", "OpShiftRightLogical"),
        c(
            "operator",
            "bitwise-logic",
            "OpBitwiseAnd",
            "OpBitwiseOr",
            "OpBitwiseXor",
            "OpNot",
        ),
        c("operator", "negation", "OpSNegate", "OpFNegate"),
        c(
            "operator",
            "signed-comparison",
            "OpSLessThan",
            "OpSGreaterThanEqual",
            "OpIEqual",
            "OpINotEqual",
        ),
        c(
            "operator",
            "unsigned-comparison",
            "OpULessThan",
            "OpUGreaterThanEqual",
            "OpUGreaterThan",
            "OpULessThanEqual",
        ),
        c(
            "operator",
            "float-comparison",
            "OpFOrdLessThan",
            "OpFOrdGreaterThanEqual",
            "OpFOrdEqual",
            "OpFUnordNotEqual",
        ),
        c(
            "operator",
            "boolean-logic",
            "OpLogicalNot",
            "OpLogicalEqual",
            "OpLogicalNotEqual",
        ),
        c("operator", "bool-conversion", "OpSelect", "OpINotEqual"),
        c("operator", "increment-decrement", "OpIAdd", "OpISub"),
        c("operator", "float-to-integer", "OpConvertFToS", "OpConvertFToU"),
        c("operator", "integer-to-float", "OpConvertSToF", "OpConvertUToF"),
        c("operator", "signed-to-unsigned", "OpBitcast"),
        c(
            "operator",
            "compound-assignment",
            "OpIAdd",
            "OpIMul",
            "OpSDiv",
            "OpSMod",
            "OpShiftLeftLogical",
            "OpBitwiseAnd",
        ),
        c("operator", "vector-scalar-product", "OpVectorTimesScalar"),
        c("operator", "matrix-vector-product", "OpMatrixTimesVector"),
        c("operator", "matrix-product", "OpMatrixTimesMatrix"),
        c("operator", "vector-matrix-product", "OpVectorTimesMatrix"),
        c("type", "array", "OpTypeArray"),
        c("type", "struct", "OpTypeStruct"),
        c("type", "array-of-structs", "OpTypeArray", "OpTypeStruct"),
        c("type", "array-initializer", "OpTypeArray", "OpConstantComposite"),
        c("type", "nested-struct", "OpTypeStruct", "OpAccessChain"),
        c("type", "swizzle", "OpVectorShuffle"),
        c("type", "swizzle-assignment", "OpVectorShuffle", "OpAccessChain"),
        c("type", "matrix-indexing", "OpTypeMatrix", "OpAccessChain"),
        c("type", "global-constant", "OpConstant"),
        c("type", "runtime-array-length", "OpTypeRuntimeArray", "OpArrayLength"),
        c("compute", "shared-memory", "StorageClass.Workgroup", "OpControlBarrier"),
        c("compute", "local-invocation-id", "BuiltIn.LocalInvocationId"),
        c("compute", "workgroup-id", "BuiltIn.WorkgroupId"),
        c("compute", "local-invocation-index", "BuiltIn.LocalInvocationIndex"),
        c("compute", "num-workgroups", "BuiltIn.NumWorkgroups"),
        c("resource", "uniform-buffer", "StorageClass.Uniform"),
        c(
            "resource",
            "read-only-buffer",
            "StorageClass.StorageBuffer",
            "Decoration.NonWritable",
        ),
        c(
            "resource",
            "sampled-texture",
            "OpTypeSampledImage",
            "OpImageSampleExplicitLod",
        ),
        c("resource", "storage-image", "OpTypeImage", "OpImageWrite"),
    ]


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
    for kind in ("StorageClass", "BuiltIn", "Decoration", "ExecutionModel"):
        found |= {f"{kind}.{e['enumerant']}" for e in kinds[kind]}
    found |= {f"Capability.{name}" for name in capabilities}
    constructs = _curated()
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
    """The language-level features ``language`` has. SPIR-V has all of them: they are SPIR-V's own items."""
    result = []
    for construct in universe(grammar_text):
        if language != "spirv" and _template(construct.name, language) is ABSENT:
            continue
        result.append(
            Feature(
                language,
                construct.category,
                construct.name,
                None,
                group=" ".join(construct.anchors),
                stage=construct.stage,
            )
        )
    return result


def is_construct(feature: Feature) -> bool:
    return feature.category in CATEGORIES


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
    except ProbeError:
        pass  # the language has no matrices, and its templates do not use $mat2
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
    source = dialect.kernel(
        buffers,
        _lines(template.body, spelled),
        prelude=_lines(template.prelude, spelled),
        declarations=_lines(template.declarations, spelled),
        params=_lines(template.params, spelled),
    )
    return source, dialect.entry, dialect.suffix

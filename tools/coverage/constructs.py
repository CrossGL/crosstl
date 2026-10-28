"""Language-level features: control flow, operators, types, compute and graphics built-ins, resources and stages.

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
        c("control-flow", "array-parameter", "OpFunctionCall", "OpTypeArray"),
        c(
            "control-flow",
            "struct-return",
            "OpFunctionCall",
            "OpReturnValue",
            "OpTypeStruct",
        ),
        c("control-flow", "function-overloading", "OpFunction", "OpFunctionCall"),
        c("control-flow", "switch-fallthrough", "OpSwitch"),
        c("control-flow", "block-scope", "OpVariable", "StorageClass.Function"),
        c("control-flow", "vector-conditional", "OpSelect", "OpTypeVector"),
        c("control-flow", "loop-return", "OpLoopMerge", "OpReturnValue"),
        c("control-flow", "nested-call", "OpFunctionCall"),
        c("control-flow", "struct-parameter", "OpFunctionParameter", "OpTypeStruct"),
        c("control-flow", "nested-loop", "OpLoopMerge", "OpBranch"),
        c("control-flow", "switch-in-loop", "OpSwitch", "OpLoopMerge"),
        c("control-flow", "do-while-continue", "OpLoopMerge", "OpBranchConditional"),
        c("control-flow", "chained-conditional", "OpSelect"),
        c("control-flow", "nested-switch", "OpSwitch", "OpSelectionMerge"),
        c("control-flow", "unsigned-switch", "OpSwitch", "OpTypeInt"),
        c("control-flow", "multi-variable-loop", "OpLoopMerge", "OpISub"),
        c("control-flow", "forward-declaration", "OpFunction", "OpFunctionCall"),
        c(
            "control-flow",
            "conditional-side-effect",
            "OpSelectionMerge",
            "OpFunctionCall",
        ),
        c("control-flow", "switch-return", "OpSwitch", "OpReturnValue"),
        c("control-flow", "loop-in-switch", "OpSwitch", "OpLoopMerge"),
        c("control-flow", "infinite-loop", "OpLoopMerge", "OpBranch"),
        c("control-flow", "loop-condition-call", "OpLoopMerge", "OpFunctionCall"),
        c("control-flow", "switch-default-position", "OpSwitch", "OpSelectionMerge"),
        c("control-flow", "for-loop-variants", "OpLoopMerge", "OpBranchConditional"),
        c("control-flow", "void-return", "OpReturn", "OpFunctionCall"),
        c("control-flow", "inout-argument-lvalues", "OpFunctionCall", "OpAccessChain"),
        c("control-flow", "switch-case-scope", "OpSwitch", "OpVariable"),
        c(
            "control-flow",
            "aggregate-inout-parameter",
            "OpFunctionCall",
            "OpTypeStruct",
        ),
        c("control-flow", "out-parameter-forwarding", "OpFunctionCall", "OpStore"),
        c("control-flow", "value-parameter-mutation", "OpFunctionParameter", "OpStore"),
        c("control-flow", "matrix-parameter", "OpFunctionCall", "OpTypeMatrix"),
        c("control-flow", "struct-conditional", "OpSelect", "OpTypeStruct"),
        c("control-flow", "loop-counter-write", "OpLoopMerge", "OpStore"),
        c("control-flow", "unreachable-statement", "OpReturnValue", "OpBranch"),
        c("control-flow", "float-loop-counter", "OpLoopMerge", "OpFOrdLessThan"),
        c("control-flow", "empty-statement", "OpBranch", "OpLoopMerge"),
        c("control-flow", "default-only-switch", "OpSwitch"),
        c("control-flow", "integer-condition", "OpINotEqual", "OpBranchConditional"),
        c("control-flow", "global-shadowing", "OpFunctionParameter", "OpVariable"),
        c("control-flow", "default-argument", "OpFunctionCall", "OpFunctionParameter"),
        c("control-flow", "generic-function", "OpFunctionCall", "OpIAdd", "OpFAdd"),
        c("control-flow", "const-parameter", "OpFunctionParameter", "OpFunctionCall"),
        c("control-flow", "condition-assignment", "OpLoopMerge", "OpIAdd", "OpISub"),
        c("control-flow", "intrinsic-name", "OpFunctionCall"),
        c("control-flow", "control-attribute", "OpLoopMerge", "OpSelectionMerge"),
        c(
            "control-flow",
            "struct-array-parameter",
            "OpFunctionParameter",
            "OpTypeArray",
        ),
        c("control-flow", "unsigned-countdown", "OpLoopMerge", "OpULessThan", "OpISub"),
        c(
            "control-flow",
            "sibling-redeclaration",
            "OpVariable",
            "StorageClass.Function",
        ),
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
        c(
            "operator",
            "vector-comparison",
            "OpFOrdLessThan",
            "OpFOrdEqual",
            "OpAny",
            "OpAll",
        ),
        c(
            "operator",
            "integer-vector-arithmetic",
            "OpTypeVector",
            "OpIMul",
            "OpShiftRightArithmetic",
            "OpSDiv",
        ),
        c("operator", "unsigned-wraparound", "OpIAdd", "OpIMul", "OpISub"),
        c("operator", "operator-precedence", "OpShiftLeftLogical", "OpBitwiseXor"),
        c("operator", "matrix-arithmetic", "OpMatrixTimesScalar", "OpFAdd"),
        c("operator", "implicit-conversion", "OpConvertSToF", "OpConvertUToF"),
        c(
            "operator",
            "matrix-compound-assignment",
            "OpMatrixTimesMatrix",
            "OpVectorTimesMatrix",
        ),
        c(
            "operator",
            "indexed-compound-assignment",
            "OpAccessChain",
            "OpFunctionCall",
        ),
        c("operator", "vector-compound-assignment", "OpVectorTimesScalar", "OpFDiv"),
        c("operator", "unsigned-float-conversion", "OpConvertFToU", "OpConvertUToF"),
        c("operator", "chained-assignment", "OpStore", "OpLoad"),
        c("operator", "unsigned-vector-arithmetic", "OpUDiv", "OpShiftRightLogical"),
        c("operator", "lvalue-increment", "OpIAdd", "OpAccessChain"),
        c("operator", "vector-unary-operators", "OpFNegate", "OpNot"),
        c("operator", "vector-scalar-arithmetic", "OpFSub", "OpFDiv"),
        c("operator", "signed-division", "OpSDiv", "OpShiftRightArithmetic"),
        c("operator", "matrix-negation", "OpFNegate", "OpTypeMatrix"),
        c("operator", "unsigned-negation", "OpSNegate", "OpNot"),
        c("operator", "float-remainder", "OpFRem"),
        c("operator", "logical-xor", "OpLogicalNotEqual"),
        c("operator", "increment-in-index", "OpIAdd", "OpAccessChain"),
        c("operator", "comma-operator", "OpIAdd", "OpIMul"),
        c("operator", "aggregate-equality", "OpFOrdEqual", "OpAll", "OpLogicalAnd"),
        c("operator", "bool-bitwise", "OpBitwiseAnd", "OpBitwiseOr", "OpBitwiseXor"),
        c(
            "operator",
            "nested-unary",
            "OpFNegate",
            "OpSNegate",
            "OpNot",
            "OpLogicalNot",
        ),
        c(
            "operator",
            "vector-shift",
            "OpShiftLeftLogical",
            "OpShiftRightArithmetic",
            "OpShiftRightLogical",
        ),
        c(
            "operator",
            "vector-bitwise",
            "OpBitwiseAnd",
            "OpBitwiseOr",
            "OpBitwiseXor",
            "OpNot",
        ),
        c(
            "operator",
            "integer-vector-comparison",
            "OpSLessThan",
            "OpUGreaterThan",
            "OpIEqual",
            "OpAny",
        ),
        c(
            "operator",
            "bool-vector-logic",
            "OpLogicalNot",
            "OpLogicalEqual",
            "OpLogicalNotEqual",
        ),
        c("type", "array", "OpTypeArray"),
        c("type", "struct", "OpTypeStruct"),
        c("type", "array-of-structs", "OpTypeArray", "OpTypeStruct"),
        c("type", "array-initializer", "OpTypeArray", "OpConstantComposite"),
        c("type", "array-copy", "OpTypeArray", "OpStore"),
        c("type", "nested-struct", "OpTypeStruct", "OpAccessChain"),
        c("type", "swizzle", "OpVectorShuffle"),
        c("type", "swizzle-assignment", "OpVectorShuffle", "OpAccessChain"),
        c("type", "matrix-indexing", "OpTypeMatrix", "OpAccessChain"),
        c("type", "vector-constructor", "OpCompositeConstruct"),
        c("type", "matrix-from-vectors", "OpTypeMatrix", "OpCompositeConstruct"),
        c("type", "struct-constructor", "OpTypeStruct", "OpCompositeConstruct"),
        c("type", "struct-copy", "OpTypeStruct", "OpStore"),
        c("type", "array-of-arrays", "OpTypeArray", "OpAccessChain"),
        c("type", "vector-dynamic-index", "OpTypeVector", "OpAccessChain"),
        c("type", "global-variable", "StorageClass.Private"),
        c("type", "numeric-literals", "OpConstant"),
        c(
            "type",
            "struct-array-member",
            "OpTypeStruct",
            "OpTypeArray",
            "OpAccessChain",
        ),
        c("type", "multiple-declarators", "OpVariable"),
        c("type", "const-local", "OpConstant"),
        c("type", "constant-array", "OpConstantComposite", "OpTypeArray"),
        c("type", "matrix-column-assignment", "OpTypeMatrix", "OpAccessChain"),
        c("type", "vector-array", "OpTypeArray", "OpTypeVector"),
        c(
            "type",
            "non-square-matrix",
            "OpTypeMatrix",
            "OpMatrixTimesVector",
            "OpVectorTimesMatrix",
        ),
        c("type", "struct-vector-member", "OpTypeStruct", "OpVectorShuffle"),
        c("type", "diagonal-matrix", "OpCompositeConstruct", "OpTypeMatrix"),
        c("type", "nested-member-assignment", "OpTypeStruct", "OpAccessChain"),
        c("type", "bool-array", "OpTypeBool", "OpTypeArray"),
        c("type", "expression-swizzle", "OpVectorShuffle"),
        c("type", "deferred-initialization", "OpVariable", "OpStore"),
        c("type", "integer-literal-limits", "OpConstant", "OpTypeInt"),
        c("type", "matrix-array", "OpTypeArray", "OpTypeMatrix"),
        c("type", "vector-conversion", "OpConvertSToF", "OpConvertFToS"),
        c("type", "struct-matrix-member", "OpTypeStruct", "OpTypeMatrix"),
        c("type", "global-array", "StorageClass.Private", "OpTypeArray"),
        c("type", "color-swizzle", "OpVectorShuffle", "OpCompositeExtract"),
        c("type", "struct-array-copy", "OpTypeArray", "OpTypeStruct"),
        c("type", "global-constant", "OpConstant"),
        c("type", "octal-literal", "OpConstant"),
        c("type", "literal-suffixes", "OpConstant"),
        c("type", "constant-array-size", "OpTypeArray", "OpConstant"),
        c("type", "local-struct", "OpTypeStruct"),
        c("type", "type-alias", "OpTypeFloat", "OpTypeVector", "OpTypeStruct"),
        c("type", "enum-constant", "OpSwitch", "OpIEqual"),
        c("type", "struct-method", "OpFSub", "OpFAdd"),
        c("type", "constant-expression", "OpConstant", "OpConstantComposite"),
        c("type", "array-return", "OpTypeArray", "OpFunctionCall", "OpReturnValue"),
        c("type", "constant-struct", "OpConstantComposite", "OpTypeStruct"),
        c("type", "constant-vector-array", "OpConstantComposite", "OpTypeArray"),
        c("type", "constant-matrix", "OpConstantComposite", "OpTypeMatrix"),
        c(
            "type",
            "zero-initializer",
            "OpConstantComposite",
            "OpTypeStruct",
            "OpTypeArray",
        ),
        c("type", "call-result-access", "OpFunctionCall", "OpCompositeExtract"),
        c("type", "reserved-identifier", "OpName"),
        c("type", "runtime-array-length", "OpTypeRuntimeArray", "OpArrayLength"),
        c("compute", "shared-memory", "StorageClass.Workgroup", "OpControlBarrier"),
        c("compute", "local-invocation-id", "BuiltIn.LocalInvocationId"),
        c("compute", "workgroup-id", "BuiltIn.WorkgroupId"),
        c("compute", "local-invocation-index", "BuiltIn.LocalInvocationIndex"),
        c("compute", "num-workgroups", "BuiltIn.NumWorkgroups"),
        c(
            "compute",
            "invocation-id-in-helper",
            "OpFunctionCall",
            "BuiltIn.GlobalInvocationId",
        ),
        c(
            "compute",
            "shared-memory-in-helper",
            "OpFunctionCall",
            "StorageClass.Workgroup",
            "OpControlBarrier",
        ),
        c("compute", "barrier-in-loop", "OpLoopMerge", "OpControlBarrier"),
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
        c(
            "resource",
            "buffer-in-helper",
            "OpFunctionCall",
            "StorageClass.StorageBuffer",
        ),
        c(
            "resource",
            "buffer-element-argument",
            "OpFunctionCall",
            "OpAccessChain",
            "StorageClass.StorageBuffer",
        ),
        c("resource", "uniform-in-helper", "OpFunctionCall", "StorageClass.Uniform"),
        c(
            "resource",
            "vector-buffer",
            "StorageClass.StorageBuffer",
            "OpTypeVector",
            "Decoration.ArrayStride",
        ),
        c(
            "resource",
            "struct-buffer",
            "StorageClass.StorageBuffer",
            "OpTypeStruct",
            "Decoration.Offset",
        ),
        c(
            "resource",
            "struct-buffer-copy",
            "StorageClass.StorageBuffer",
            "OpTypeStruct",
            "OpCompositeExtract",
        ),
        c("resource", "uniform-members", "StorageClass.Uniform", "Decoration.Offset"),
        c("resource", "uniform-vector", "StorageClass.Uniform", "OpTypeVector"),
        c(
            "resource",
            "uniform-vector-packing",
            "StorageClass.Uniform",
            "Decoration.Offset",
        ),
        c(
            "resource",
            "matrix-buffer",
            "StorageClass.StorageBuffer",
            "OpTypeMatrix",
            "Decoration.MatrixStride",
        ),
        c(
            "resource",
            "struct-buffer-packing",
            "StorageClass.StorageBuffer",
            "OpTypeStruct",
            "Decoration.Offset",
        ),
        c(
            "resource",
            "struct-array-buffer",
            "StorageClass.StorageBuffer",
            "OpTypeArray",
            "Decoration.ArrayStride",
        ),
        c(
            "resource",
            "nested-struct-buffer",
            "StorageClass.StorageBuffer",
            "OpTypeStruct",
            "Decoration.Offset",
        ),
        c("resource", "integer-buffer", "StorageClass.StorageBuffer", "OpIMul"),
        c(
            "resource",
            "unsigned-buffer",
            "StorageClass.StorageBuffer",
            "OpShiftRightLogical",
        ),
        c(
            "resource",
            "uniform-array",
            "StorageClass.Uniform",
            "OpTypeArray",
            "Decoration.ArrayStride",
        ),
        c(
            "resource",
            "uniform-vector-array",
            "StorageClass.Uniform",
            "OpTypeArray",
            "OpTypeVector",
        ),
        c(
            "resource",
            "uniform-matrix",
            "StorageClass.Uniform",
            "OpTypeMatrix",
            "Decoration.ColMajor",
        ),
        c(
            "compute",
            "shared-vector-array",
            "StorageClass.Workgroup",
            "OpTypeVector",
            "OpControlBarrier",
        ),
        c(
            "compute",
            "shared-struct-array",
            "StorageClass.Workgroup",
            "OpTypeStruct",
            "OpControlBarrier",
        ),
        c(
            "compute",
            "shared-2d-array",
            "StorageClass.Workgroup",
            "OpTypeArray",
            "OpControlBarrier",
        ),
        c("graphics", "fragment-coordinate", "BuiltIn.FragCoord", stage="fragment"),
        c("graphics", "front-facing", "BuiltIn.FrontFacing", stage="fragment"),
        c("graphics", "fragment-depth", "BuiltIn.FragDepth", stage="fragment"),
        c("graphics", "flat-interpolation", "Decoration.Flat", stage="fragment"),
        c(
            "graphics",
            "noperspective-interpolation",
            "Decoration.NoPerspective",
            stage="fragment",
        ),
        c(
            "graphics",
            "centroid-interpolation",
            "Decoration.Centroid",
            stage="fragment",
        ),
        c(
            "graphics",
            "sample-interpolation",
            "Decoration.Sample",
            "Capability.SampleRateShading",
            stage="fragment",
        ),
        c(
            "graphics",
            "multiple-render-targets",
            "Decoration.Location",
            stage="fragment",
        ),
        c(
            "resource",
            "uniform-struct",
            "StorageClass.Uniform",
            "OpTypeStruct",
            "Decoration.Offset",
        ),
        c(
            "resource",
            "uniform-struct-array",
            "StorageClass.Uniform",
            "OpTypeStruct",
            "Decoration.ArrayStride",
        ),
        c(
            "resource",
            "uniform-struct-copy",
            "StorageClass.Uniform",
            "OpTypeStruct",
            "OpCompositeExtract",
        ),
        c(
            "resource",
            "uniform-integer",
            "StorageClass.Uniform",
            "OpShiftRightArithmetic",
        ),
        c(
            "resource",
            "uniform-unsigned",
            "StorageClass.Uniform",
            "OpShiftRightLogical",
        ),
        c(
            "resource",
            "uniform-integer-vector",
            "StorageClass.Uniform",
            "OpTypeVector",
            "OpShiftRightArithmetic",
        ),
        c("type", "reserved-member-name", "OpMemberName"),
        c("type", "reserved-type-name", "OpName", "OpTypeStruct"),
        c("type", "reserved-global-name", "OpName", "StorageClass.Private"),
        c("type", "vector-truncation", "OpCompositeExtract", "OpConvertSToF"),
        c("type", "matrix-resize", "OpTypeMatrix", "OpCompositeConstruct"),
        c("type", "bool-vector-conversion", "OpINotEqual", "OpSelect"),
        c(
            "resource",
            "uniform-matrix2",
            "StorageClass.Uniform",
            "OpTypeMatrix",
            "Decoration.MatrixStride",
        ),
        c(
            "resource",
            "uniform-matrix3",
            "StorageClass.Uniform",
            "OpTypeMatrix",
            "Decoration.MatrixStride",
        ),
        c(
            "resource",
            "uniform-non-square-matrix",
            "StorageClass.Uniform",
            "OpTypeMatrix",
            "Decoration.ColMajor",
        ),
        c("resource", "uniform-bool", "StorageClass.Uniform", "OpINotEqual"),
        c(
            "resource",
            "uniform-bool-vector",
            "StorageClass.Uniform",
            "OpINotEqual",
            "OpTypeBool",
        ),
        c(
            "resource",
            "uniform-std140-vector",
            "StorageClass.Uniform",
            "Decoration.Offset",
        ),
        c(
            "resource",
            "uniform-cbuffer-vector",
            "StorageClass.Uniform",
            "Decoration.Offset",
        ),
        c(
            "resource",
            "uniform-std140-array",
            "StorageClass.Uniform",
            "Decoration.ArrayStride",
        ),
        c(
            "resource",
            "uniform-row-major-matrix",
            "StorageClass.Uniform",
            "Decoration.RowMajor",
        ),
        c(
            "resource",
            "uniform-row-major-non-square",
            "StorageClass.Uniform",
            "Decoration.RowMajor",
            "Decoration.Offset",
        ),
        c(
            "resource",
            "uniform-row-major-array",
            "StorageClass.Uniform",
            "Decoration.RowMajor",
            "Decoration.ArrayStride",
        ),
        c(
            "resource",
            "buffer-row-major-matrix",
            "StorageClass.StorageBuffer",
            "Decoration.RowMajor",
        ),
        c(
            "resource",
            "buffer-row-major-non-square",
            "StorageClass.StorageBuffer",
            "Decoration.RowMajor",
            "Decoration.Offset",
        ),
        c(
            "resource",
            "buffer-row-major-array",
            "StorageClass.StorageBuffer",
            "Decoration.RowMajor",
            "Decoration.ArrayStride",
        ),
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

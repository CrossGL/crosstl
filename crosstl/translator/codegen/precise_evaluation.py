"""Precise floating-point evaluation on the CUDA, HIP, WGSL, Rust and Mojo targets.

A CrossGL declaration marked ``precise`` (``@precise`` or the ``precise`` qualifier on a function, its result, a
parameter or a local) requires its floating-point arithmetic to be evaluated as written: a multiply and an add are not
contracted into one fused operation and operations are not reassociated (GLSL 4.60 section 4.9 "The Precise
Qualifier", DXIL "Precise qualifier", SPIR-V ``NoContraction``). The GLSL, HLSL, Metal and SPIR-V generators lower
it themselves. The generators below used to drop the marker and return arithmetic the target compiler may contract.

- CUDA. nvcc contracts a multiply and an add into a multiply-add by default (``--fmad=true``). ``__fmul_rn`` and
  ``__dmul_rn`` "will never be merged into a single multiply-add instruction" (CUDA Math API), and a contraction
  needs a multiply, so every floating multiply of a precisely evaluated function, and of the vector helpers it calls,
  is emitted as one of them. nvcc does not reassociate without ``--use_fast_math``.
- HIP. Clang honours ``#pragma clang fp contract(off)`` at the start of a compound statement for the arithmetic
  written in it (Clang "Extensions: Controlling Floating Point Behavior"). HIP's ``__fmul_rn`` is a plain multiply
  that Clang may still contract, so the precisely evaluated function, and each vector helper it calls, starts with
  the pragma instead.
- Arithmetic whose lowering lives in helpers that keep neither guarantee (matrices and ``half`` values) is refused.
- WGSL lets an implementation reassociate operations and fuse them, and has no qualifier that forbids either (WGSL
  section 15.7.5 "Reassociation and Fusion"). Rust shaders compile through rust-gpu, whose SPIR-V arithmetic carries
  no ``NoContraction`` decoration, so Vulkan assumes ``AllowContract`` and ``AllowReassoc`` for it. Mojo documents no
  control over floating-point contraction. These targets refuse a program that evaluates precisely.

A function evaluates precisely when it, or any declaration in it, is marked ``precise``. Evaluating a function's
other arithmetic as written as well only removes contractions the source allowed, which never yields a result the
source forbids.
"""

from __future__ import annotations

import re

from ..ast import AssignmentNode, BinaryOpNode, FunctionNode
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..semantics.rewrite import is_pure

__all__ = [
    "PRECISE_EVALUATION_REFUSALS",
    "PreciseEvaluationMixin",
    "PreciseEvaluationUnsupportedError",
    "function_evaluates_precisely",
    "precisely_evaluated_functions",
    "refuse_precise_evaluation",
]

# Why each target that has no lowering refuses precise evaluation.
PRECISE_EVALUATION_REFUSALS = {
    "wgsl": (
        "WGSL lets an implementation reassociate operations and fuse them, and has no qualifier that forbids "
        "either (WGSL section 15.7.5 Reassociation and Fusion)"
    ),
    "rust": (
        "Rust shaders compile through rust-gpu, whose SPIR-V arithmetic carries no NoContraction decoration, so "
        "Vulkan may contract and reassociate it"
    ),
    "mojo": (
        "Mojo documents no control over floating-point contraction or reassociation"
    ),
}

# Floating types whose arithmetic the C-family generators lower through helpers without either guarantee.
_UNGUARDED_FLOAT_TYPE = re.compile(
    r"(?:^|[^a-z])(?:d?mat[2-4](?:x[2-4])?|(?:float|double|half)[2-4]x[2-4]|half|float16|f16|bf16|bfloat16)",
    re.IGNORECASE,
)

_HALF_SCALAR = re.compile(r"half|f16|float16(?:_t)?|__half", re.IGNORECASE)

_ARITHMETIC_OPERATORS = frozenset({"+", "-", "*", "/"})


def _modifier_names(node):
    names = [
        str(getattr(value, "name", value) or "").lower()
        for value in getattr(node, "qualifiers", None) or ()
    ]
    names.extend(
        str(getattr(value, "name", value) or "").lower()
        for value in getattr(node, "attributes", None) or ()
    )
    return names


def _marked_precise(node) -> bool:
    return "precise" in _modifier_names(node)


def _contained_nodes(root):
    """Every object reachable from ``root`` once, without following ``parent`` or ``annotations`` links.

    Some AST values refer to themselves, so values already produced are skipped; the callers only test membership.
    """
    seen = set()
    stack = [root]
    while stack:
        value = stack.pop()
        if value is None or isinstance(value, (str, int, float, bool)):
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)
            continue
        if isinstance(value, dict):
            stack.extend(value.values())
            continue
        if id(value) in seen or not hasattr(value, "__dict__"):
            continue
        seen.add(id(value))
        yield value
        stack.extend(
            child
            for key, child in vars(value).items()
            if key not in {"parent", "annotations"}
        )


def function_evaluates_precisely(function) -> bool:
    """Whether ``function``, a parameter, or a declaration in its body is marked ``precise``."""
    if function is None:
        return False
    if _marked_precise(function):
        return True
    roots = (
        getattr(function, "parameters", None) or getattr(function, "params", None),
        getattr(function, "body", None),
    )
    for root in roots:
        if root is None:
            continue
        for value in _contained_nodes(root):
            if _marked_precise(value):
                return True
    return False


def precisely_evaluated_functions(ast) -> list[str]:
    """Names of the functions of ``ast`` that evaluate precisely, sorted and without repeats."""
    names = set()
    for value in _contained_nodes(ast):
        if isinstance(value, FunctionNode) and function_evaluates_precisely(value):
            names.add(str(value.name))
    return sorted(names)


class PreciseEvaluationUnsupportedError(TranslationDiagnostic):
    """The program evaluates floating-point arithmetic precisely, and the target cannot keep that."""

    code = f"{CODE_PREFIX}precise-evaluation-unsupported"

    def __init__(self, label: str, target: str, constructs, reason: str) -> None:
        self.constructs = tuple(constructs)
        self.missing_capabilities = (f"{target}.precise-evaluation",)
        super().__init__(
            f"{label} cannot evaluate the precise floating-point arithmetic of "
            f"{', '.join(self.constructs)} as written: {reason}",
            target=target,
        )


def refuse_precise_evaluation(ast, *, label: str, target: str) -> None:
    """Raise :class:`PreciseEvaluationUnsupportedError` when a function of ``ast`` evaluates precisely."""
    functions = precisely_evaluated_functions(ast)
    if functions:
        raise PreciseEvaluationUnsupportedError(
            label,
            target,
            [f"function {name}" for name in functions],
            PRECISE_EVALUATION_REFUSALS[target],
        )


class PreciseEvaluationMixin:
    """Precise evaluation members of the CUDA and HIP generators.

    A generator sets ``PRECISE_MULTIPLY_INTRINSICS`` (component type to a multiply that is never contracted) or
    ``PRECISE_BODY_PRAGMA`` (a pragma that disables contraction for the compound statement it starts), calls
    :meth:`enter_precise_evaluation` and :meth:`leave_precise_evaluation` around each function body, emits
    :meth:`precise_body_prologue` first in it, and asks :meth:`precise_binary_operation` and
    :meth:`precise_multiply_assignment` before lowering an operator. ``require_vector_binary_helper`` asks
    :meth:`precise_vector_helper` for the helpers a precise function calls.
    """

    PRECISE_MULTIPLY_INTRINSICS: dict[str, str] = {}
    PRECISE_BODY_PRAGMA: str | None = None
    PRECISE_TARGET_LABEL = "The target"

    current_precise_evaluation = False

    def enter_precise_evaluation(self, function):
        """Start emitting ``function``'s body; return the state :meth:`leave_precise_evaluation` restores."""
        previous = self.current_precise_evaluation
        self.current_precise_evaluation = function_evaluates_precisely(function)
        return previous

    def leave_precise_evaluation(self, previous):
        self.current_precise_evaluation = previous

    def precise_body_prologue(self) -> list[str]:
        """Lines that start the body of a precisely evaluated function."""
        if self.current_precise_evaluation and self.PRECISE_BODY_PRAGMA:
            return [self.PRECISE_BODY_PRAGMA]
        return []

    def precise_type_kind(self, value_type):
        """Classify ``value_type`` for precise arithmetic.

        ``"float"`` and ``"double"`` scalars, ``"half"`` scalars, ``"vector"`` for vectors of ``float`` or
        ``double``, ``"unguarded"`` for every other floating type (matrices, ``half`` vectors), and ``None`` for
        types that are not floating.
        """
        if value_type is None:
            return None
        vector_info = self.vector_type_info(value_type)
        if vector_info is not None:
            component = vector_info.get("component_type")
            if component in {"float", "double"}:
                return "vector"
            return "unguarded" if component == "half" else None
        scalar = self.scalar_component_type(value_type)
        if scalar in {"float", "double"}:
            return scalar
        spelled = str(
            (
                value_type
                if isinstance(value_type, str)
                else self.convert_type_node_to_string(value_type)
            )
            or ""
        ).strip()
        if _HALF_SCALAR.fullmatch(spelled):
            return "half"
        if _UNGUARDED_FLOAT_TYPE.search(spelled):
            return "unguarded"
        return None

    def precise_binary_operation(self, node, left, right, operator):
        """The precise form of ``left operator right``, or ``None`` when the usual lowering keeps it.

        Arithmetic on a type the target has no uncontracted multiply for (matrices, ``half`` vectors, and ``half``
        scalars on targets without ``PRECISE_MULTIPLY_INTRINSICS["half"]``) is refused.
        """
        if not self.current_precise_evaluation or operator not in _ARITHMETIC_OPERATORS:
            return None
        kinds = {
            self.precise_type_kind(self.expression_result_type(operand))
            for operand in (node.left, node.right)
        }
        kinds.add(self.precise_type_kind(self.expression_result_type(node)))
        unguarded = "unguarded" in kinds or (
            "half" in kinds and "half" not in self.PRECISE_MULTIPLY_INTRINSICS
        )
        # A multiply of a half by another type has no single uncontracted intrinsic.
        unguarded = unguarded or (
            operator == "*" and "half" in kinds and kinds != {"half"}
        )
        if unguarded:
            return self.refuse_lowering(
                f"precise '{operator}' on {self.precise_operand_spelling(node)}: "
                f"{self.PRECISE_TARGET_LABEL} has no uncontracted form of that arithmetic",
                f"({left} {operator} {right})",
            )
        if operator != "*" or "vector" in kinds:
            return None
        intrinsic = None
        for kind in ("double", "float", "half"):
            if kind in kinds:
                intrinsic = self.PRECISE_MULTIPLY_INTRINSICS.get(kind)
                break
        if intrinsic is None:
            return None
        return f"{intrinsic}({left}, {right})"

    def precise_operand_spelling(self, node):
        spellings = []
        for operand in (node.left, node.right):
            value_type = self.expression_result_type(operand)
            if value_type is not None and not isinstance(value_type, str):
                value_type = self.convert_type_node_to_string(value_type)
            spellings.append(str(value_type or "an operand of unknown type"))
        return " and ".join(spellings)

    def precise_multiply_assignment(self, node, target, value, operator):
        """``target = target * value`` for a floating ``*=`` that a multiply intrinsic must carry, or ``None``."""
        if (
            not self.current_precise_evaluation
            or operator != "*="
            or not self.PRECISE_MULTIPLY_INTRINSICS
        ):
            return None
        kinds = {
            self.precise_type_kind(self.expression_result_type(operand))
            for operand in (target, value)
        }
        if not kinds & {"float", "double", "half", "vector", "unguarded"}:
            return None
        if not is_pure(target):
            self.refuse_lowering(
                "precise '*=' whose target has side effects: the multiply is written as "
                "'x = x * y' to keep it uncontracted, which would evaluate the target twice"
            )
            return None
        product = BinaryOpNode(target, "*", value)
        location = getattr(value, "source_location", None)
        if location is not None:
            product.source_location = location
        rewritten = AssignmentNode(target, product, "=")
        if getattr(node, "source_location", None) is not None:
            rewritten.source_location = node.source_location
        return rewritten

    def precise_vector_helper(self, vector_info, operator) -> bool:
        """Whether a vector helper the current function calls needs its precise variant."""
        if not self.current_precise_evaluation:
            return False
        if vector_info.get("component_type") not in {"float", "double"}:
            return False
        if self.PRECISE_BODY_PRAGMA:
            return operator in _ARITHMETIC_OPERATORS
        return operator == "*" and bool(self.PRECISE_MULTIPLY_INTRINSICS)

    def format_precise_vector_component(self, component_type, operator, left, right):
        """One component of a precise vector helper."""
        intrinsic = self.PRECISE_MULTIPLY_INTRINSICS.get(component_type)
        if operator == "*" and intrinsic:
            return f"{intrinsic}({left}, {right})"
        return self.format_vector_binary_component(
            component_type, operator, left, right
        )

    def precise_helper_prologue(self) -> str:
        """Text that starts the body of a precise vector helper."""
        if self.PRECISE_BODY_PRAGMA:
            return f"    {self.PRECISE_BODY_PRAGMA}\n"
        return ""

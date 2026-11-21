"""Lane-symmetry rules for OpenGL software subgroups.

The facts themselves live in :mod:`lane_symmetry`; this module supplies the
OpenGL generator's view of collectives, calls, parameters and storage.
"""

from contextlib import contextmanager

from ..ast import (
    ASTNode,
    BinaryOpNode,
    FunctionCallNode,
    IfNode,
    PointerType,
    ReferenceType,
    TernaryOpNode,
    WaveOpNode,
)
from .array_utils import evaluate_literal_int_expression
from .lane_symmetry import SYMMETRIC, UNIFORM, LaneSymmetryAnalysis, SymmetryRules

GLSL_LANE_SYMMETRY_REDUCTIONS = frozenset(
    {
        "WaveActiveSum",
        "WaveActiveProduct",
        "WaveActiveMin",
        "WaveActiveMax",
        "WaveActiveAllTrue",
        "WaveActiveAnyTrue",
    }
)


def glsl_collective_operation(codegen, node):
    """The software subgroup value operation ``node`` performs, or None."""
    if isinstance(node, FunctionCallNode):
        if codegen.function_call_name(node) in codegen.function_definitions:
            return None
    elif not isinstance(node, WaveOpNode):
        return None
    direct = codegen.glsl_software_subgroup_direct_operation(node)
    if direct is None:
        return None
    operation = direct[0]
    if operation not in codegen.GLSL_SOFTWARE_SUBGROUP_OPERATIONS:
        return None
    return operation


def glsl_lane_symmetry_analysis(codegen, functions_by_id, invocation_count):
    """Lane-symmetry analysis for the software subgroups ``codegen`` lowers."""
    analysis = codegen.glsl_software_subgroup_uniform_returns
    call_mutations = codegen.glsl_software_subgroup_call_mutations
    resolved = codegen.glsl_software_subgroup_resolved_call_targets
    constants = dict(codegen.glsl_active_literal_int_constants())
    width = codegen.software_subgroup_width

    def qualifiers(parameter):
        return set(codegen.glsl_parameter_qualifiers(parameter))

    def call_writes(call):
        if id(call) in call_mutations:
            return call_mutations[id(call)]
        return codegen.glsl_software_subgroup_expression_identifier_names(
            list(call.arguments or [])
        )

    def storage_parameter(parameter):
        parameter_type = getattr(parameter, "param_type", None)
        if isinstance(parameter_type, PointerType):
            return True
        if isinstance(parameter_type, ReferenceType):
            return parameter_type.address_space not in {None, "", "thread"}
        return bool(analysis.resource_type(parameter_type, parameter, read_only=False))

    def private_reference(parameter):
        parameter_type = getattr(parameter, "param_type", None)
        if isinstance(parameter_type, ReferenceType):
            return not storage_parameter(parameter)
        return bool(qualifiers(parameter) & {"out", "inout"}) and bool(
            analysis.value_type(parameter_type)
        )

    def seeded_parameter(parameter):
        parameter_type = getattr(parameter, "param_type", None)
        return (
            not isinstance(parameter_type, ReferenceType)
            and not qualifiers(parameter) & {"out", "inout"}
            and bool(
                analysis.value_type(parameter_type)
                or analysis.resource_type(parameter_type, parameter)
            )
        )

    return LaneSymmetryAnalysis(
        SymmetryRules(
            walk=codegen.walk_ast,
            map_operator=codegen.map_operator,
            collective=lambda node: glsl_collective_operation(codegen, node),
            call_target=lambda call: functions_by_id.get(resolved.get(id(call))),
            pure_call=lambda call: id(call)
            in codegen.glsl_software_subgroup_uniform_call_ids,
            call_dependencies=analysis.dependencies,
            volatile=lambda declaration: any(
                analysis._qualified(node, {"volatile"})
                for node in (declaration, getattr(declaration, "var_type", None))
            ),
            storage_parameter=storage_parameter,
            private_reference=private_reference,
            storage_load=analysis.storage_load,
            constant_int=lambda expression: evaluate_literal_int_expression(
                expression, constants
            ),
            call_writes=call_writes,
            seeded_parameter=seeded_parameter,
            width=width,
            full_subgroups=bool(invocation_count) and invocation_count % width == 0,
            reductions=GLSL_LANE_SYMMETRY_REDUCTIONS,
            broadcasts=frozenset({"WaveReadLaneAt"}),
            lane_preserving=frozenset(
                {
                    *codegen.GLSL_SOFTWARE_SUBGROUP_SCANS,
                    *codegen.GLSL_SOFTWARE_SUBGROUP_RELATIVE_SHUFFLES,
                }
            ),
        )
    )


class GLSLHelperSymmetry:
    """Lane-symmetry facts of a compute entry point and the helpers it reaches.

    Facts are kept per function in ``codegen.glsl_software_subgroup_symmetries``.
    A helper parameter is lane-symmetric when every reachable call passes a
    lane-symmetric argument, and a helper is entered symmetrically when every
    reachable call is.
    """

    def __init__(self, codegen, functions_by_id, invocation_count):
        self.codegen = codegen
        self.analysis = glsl_lane_symmetry_analysis(
            codegen, functions_by_id, invocation_count
        )
        self.facts = codegen.glsl_software_subgroup_symmetries
        self.parameters = {}
        self.entries = {}

    def analyze_entry(self, function, uniform_names):
        codegen = self.codegen
        seeds = {name: UNIFORM for name in uniform_names}
        seeds.setdefault("gl_SubgroupInvocationID", SYMMETRIC)
        for parameter in getattr(function, "parameters", None) or []:
            semantic = codegen.semantic_from_node(parameter)
            mapped = codegen.map_semantic(semantic) if semantic else None
            if parameter.name not in seeds and "gl_SubgroupInvocationID" in {
                semantic,
                mapped,
            }:
                seeds[parameter.name] = SYMMETRIC
        self.facts[id(function)] = self.analysis.analyze(function, seeds, True)

    def analyze_helper(self, function_id, function, uniform_names):
        seeds = {name: UNIFORM for name in uniform_names}
        symmetric = self.parameters.get(function_id, set())
        for index, parameter in enumerate(function.parameters or []):
            if parameter.name not in seeds and index in symmetric:
                seeds[parameter.name] = SYMMETRIC
        self.facts[function_id] = self.analysis.analyze(
            function, seeds, self.entries.get(function_id, False)
        )

    def record_call(self, function_id, call, target_id):
        """Intersect what ``call`` in ``function_id`` passes to ``target_id``."""
        symmetry = self.facts.get(function_id)
        symmetric = (
            {
                index
                for index, argument in enumerate(call.arguments or [])
                if symmetry.level(argument) >= SYMMETRIC
            }
            if symmetry is not None
            else set()
        )
        if target_id not in self.parameters:
            self.parameters[target_id] = symmetric
        else:
            self.parameters[target_id].intersection_update(symmetric)
        self.entries[target_id] = self.entries.get(target_id, True) and (
            symmetry is not None and symmetry.entered_symmetrically(call)
        )

    def validating(self, function):
        return validating_function(self.codegen, function)


@contextmanager
def validating_function(codegen, function):
    """Expose ``function``'s facts to the uniformity proofs of its body."""
    facts = codegen.glsl_software_subgroup_symmetries.get(id(function))
    codegen.glsl_software_subgroup_symmetry = facts
    try:
        yield
    finally:
        codegen.glsl_software_subgroup_symmetry = None


class OpenGLLaneSymmetryExitMixin:
    """Generator queries that read the lane-symmetry facts of one function."""

    def glsl_software_subgroup_collective_uniform(self, node):
        """Whether ``node`` is a collective result every holder agrees on.

        That covers a collective of lane-symmetric values, and a name or
        helper result built from one. Values without a collective keep the
        generator's other proofs.
        """
        symmetry = getattr(self, "glsl_software_subgroup_symmetry", None)
        return (
            symmetry is not None
            and symmetry.collective_derived(node)
            and symmetry.level(node) == UNIFORM
        )

    def glsl_software_subgroup_condition_calls(self, statement, uniform=None):
        """Calls an ``if`` statement's first condition makes uniformly.

        Only the first condition runs whenever the statement does. A call
        under a conditional operator, or in the right operand of
        ``&&``/``||``, runs only where the operator's condition selects it,
        so it counts only when ``uniform(condition)`` proves that every
        invocation selects it alike. Without ``uniform`` such calls are
        left out.
        """
        if not isinstance(statement, IfNode):
            return []
        calls = []
        pending = [getattr(statement, "condition", None)]
        while pending:
            node = pending.pop()
            if not isinstance(node, ASTNode):
                continue
            if isinstance(node, TernaryOpNode):
                pending.append(node.condition)
                if uniform is not None and uniform(node.condition):
                    pending.extend((node.false_expr, node.true_expr))
                continue
            if isinstance(node, BinaryOpNode) and self.map_operator(
                getattr(node, "op", getattr(node, "operator", None))
            ) in {"&&", "||"}:
                pending.append(node.left)
                if uniform is not None and uniform(node.left):
                    pending.append(node.right)
                continue
            if isinstance(node, FunctionCallNode):
                calls.append(node)
            pending.extend(reversed(list(node.child_nodes())))
        return calls

    def glsl_software_subgroup_uniform_exit(self, node):
        """Whether every invocation takes the exit ``node`` together.

        The lane-symmetry facts of the function under validation prove it
        when every condition guarding the exit, and every exit before it, is
        workgroup-uniform.
        """
        symmetry = getattr(self, "glsl_software_subgroup_symmetry", None)
        return symmetry is not None and symmetry.reach(node) == UNIFORM

"""Static element intervals of the workgroup pointer accesses in a GLSL function, for ``GLSLCodeGen.glsl_workgroup_pointer_direct_access_intervals``."""

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    IdentifierNode,
    IfNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    PointerAccessNode,
    RangeNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from .opengl_workgroup_pointer_branches import OpenGLWorkgroupPointerBranchMixin
from .opengl_workgroup_pointer_expressions import OpenGLWorkgroupPointerExpressionMixin
from .opengl_workgroup_pointer_loops import OpenGLWorkgroupPointerLoopMixin


class GLSLWorkgroupPointerAccessAnalysis(
    OpenGLWorkgroupPointerExpressionMixin,
    OpenGLWorkgroupPointerBranchMixin,
    OpenGLWorkgroupPointerLoopMixin,
):
    """Element intervals of the workgroup pointer accesses of one GLSL function, propagated through its statements."""

    def __init__(
        self,
        *,
        generator,
        element_types,
        constants,
        access_intervals,
        mutated_interval_names,
    ):
        self.generator = generator
        self.element_types = element_types
        self.constants = constants
        self.access_intervals = access_intervals
        self.mutated_interval_names = mutated_interval_names

    def retain_refinable_offset(self, binding):
        if binding is None:
            return None
        binding = dict(binding)
        dependencies = self.generator.glsl_interval_dependency_names(
            binding.get("offset", 0)
        )
        binding["_glsl_workgroup_proof_offset_refinable"] = not (
            dependencies & self.mutated_interval_names
        )
        return binding

    def merge_intervals(self, left, right):
        return {
            name: (
                min(left[name][0], right[name][0]),
                max(left[name][1], right[name][1]),
            )
            for name in left.keys() & right.keys()
        }

    def clear_interval_names(self, active_intervals, names):
        for name in names:
            active_intervals.pop(name, None)
            self.generator.glsl_clear_index_component_intervals(active_intervals, name)

    def alias_offset_interval(self, binding, active_intervals):
        if binding is None or binding.get("_glsl_workgroup_proof_offset_unprovable"):
            return None
        return self.generator.glsl_workgroup_pointer_offset_interval(
            binding,
            active_intervals,
        )

    def poison_alias(self, binding):
        poisoned = dict(binding)
        poisoned["offset"] = IdentifierNode("__crosstl_unprovable_workgroup_offset")
        poisoned["offset_interval"] = None
        poisoned["_glsl_workgroup_proof_offset_refinable"] = False
        poisoned["_glsl_workgroup_proof_offset_unprovable"] = True
        return poisoned

    def alias_state_key(self, binding, active_intervals):
        if binding is None:
            return None
        return (
            binding.get("kind"),
            binding.get("root"),
            binding.get("element_type"),
            self.alias_offset_interval(binding, active_intervals),
            bool(binding.get("_glsl_workgroup_proof_offset_unprovable")),
        )

    def changed_alias_names(
        self,
        before_aliases,
        before_intervals,
        after_aliases,
        after_intervals,
    ):
        return {
            name
            for name, binding in before_aliases.items()
            if self.alias_state_key(binding, before_intervals)
            != self.alias_state_key(after_aliases.get(name), after_intervals)
        }

    def merge_alias_states(self, target_aliases, branch_states):
        for name, original in list(target_aliases.items()):
            bindings = [aliases.get(name) for aliases, _ in branch_states]
            if any(binding is None for binding in bindings):
                target_aliases[name] = self.poison_alias(original)
                continue
            if all(binding == original for binding in bindings):
                continue
            signatures = {
                (
                    binding.get("kind"),
                    binding.get("root"),
                    binding.get("element_type"),
                )
                for binding in bindings
            }
            offset_intervals = [
                self.alias_offset_interval(binding, state_intervals)
                for binding, (_, state_intervals) in zip(
                    bindings,
                    branch_states,
                )
            ]
            if len(signatures) != 1 or any(
                interval is None for interval in offset_intervals
            ):
                target_aliases[name] = self.poison_alias(original)
                continue
            merged = dict(bindings[0])
            merged["offset_interval"] = (
                min(interval[0] for interval in offset_intervals),
                max(interval[1] for interval in offset_intervals),
            )
            merged["_glsl_workgroup_proof_offset_refinable"] = False
            merged.pop("_glsl_workgroup_proof_offset_unprovable", None)
            target_aliases[name] = merged

    def propagate_outer_state(
        self,
        active_aliases,
        active_intervals,
        original_aliases,
        original_intervals,
        local_aliases,
        local_intervals,
        *,
        shadow_names=(),
    ):
        shadow_names = set(shadow_names)
        for name, original in original_aliases.items():
            if name in shadow_names:
                active_aliases[name] = dict(original)
            elif name in local_aliases:
                active_aliases[name] = dict(local_aliases[name])
            else:
                active_aliases[name] = self.poison_alias(original)
        for name in list(active_aliases):
            if name not in original_aliases:
                active_aliases.pop(name, None)

        for key, original in original_intervals.items():
            shadowed = any(
                key == name or key.startswith(f"{name}.") for name in shadow_names
            )
            if shadowed:
                active_intervals[key] = original
            elif key in local_intervals:
                active_intervals[key] = local_intervals[key]
            else:
                active_intervals.pop(key, None)
        for key in list(active_intervals):
            if key not in original_intervals:
                active_intervals.pop(key, None)

    def exact_classic_for_iteration_values(self, node):
        values = self.generator.glsl_storage_pointer_array_loop_iteration_values(node)
        if values is None:
            return None
        condition = getattr(node, "condition", None)
        init = getattr(node, "init", None)
        if not isinstance(condition, BinaryOpNode) or init is None:
            return None
        loop_name = getattr(init, "name", None)
        if self.generator.expression_name(condition.left) == loop_name:
            bound = condition.right
        elif self.generator.expression_name(condition.right) == loop_name:
            bound = condition.left
        else:
            return None
        if self.generator.literal_int_value(bound, self.constants) is None:
            return None
        return values

    def exact_for_in_iteration_values(self, node):
        iterable = getattr(node, "iterable", None)
        if isinstance(iterable, RangeNode):
            start = self.generator.literal_int_value(iterable.start, self.constants)
            end = self.generator.literal_int_value(iterable.end, self.constants)
            if start is None or end is None:
                return None
            stop = end + (1 if getattr(iterable, "inclusive", False) else 0)
            values = tuple(range(start, stop))
        else:
            count = self.generator.literal_int_value(iterable, self.constants)
            if count is None or count < 0:
                return None
            values = tuple(range(count))
        return values if len(values) <= 4096 else None

    def record_access(self, pointer, index, active_aliases, active_intervals):
        binding = self.generator.glsl_workgroup_pointer_binding(
            pointer,
            active_aliases,
            active_intervals,
        )
        if binding is None or binding.get("root") not in self.element_types:
            return
        root = binding["root"]
        base_interval = self.alias_offset_interval(
            binding,
            active_intervals,
        )
        if binding.get("_glsl_workgroup_proof_offset_refinable"):
            refined_interval = self.generator.glsl_private_pointer_interval(
                binding.get("offset", 0),
                active_intervals,
                self.constants,
            )
            if (
                refined_interval is not None
                and base_interval is not None
                and refined_interval[0] >= base_interval[0]
                and refined_interval[1] <= base_interval[1]
            ):
                base_interval = refined_interval
        index_interval = self.generator.glsl_private_pointer_interval(
            index,
            active_intervals,
            self.constants,
        )
        if base_interval is None or index_interval is None:
            self.access_intervals[root] = None
            return
        candidate = (
            base_interval[0] + index_interval[0],
            base_interval[1] + index_interval[1],
        )
        current = self.access_intervals.get(root)
        if current is None and root in self.access_intervals:
            return
        if current is None:
            self.access_intervals[root] = candidate
        else:
            self.access_intervals[root] = (
                min(current[0], candidate[0]),
                max(current[1], candidate[1]),
            )

    def visit_sequence(self, value, active_aliases, active_intervals):
        statements = getattr(value, "statements", value)
        if not isinstance(statements, (list, tuple)):
            statements = [statements]
        for statement in statements:
            self.visit(statement, active_aliases, active_intervals)

    def visit_scoped_sequence(self, value, active_aliases, active_intervals):
        statements = getattr(value, "statements", value)
        if not isinstance(statements, (list, tuple)):
            statements = [statements]
        outer_alias_names = set(active_aliases)
        outer_interval_keys = set(active_intervals)
        scoped_aliases = dict(active_aliases)
        scoped_intervals = dict(active_intervals)
        shadowed_aliases = {}
        shadowed_intervals = {}
        for statement in statements:
            if isinstance(statement, (VariableNode, ArrayNode)):
                name = getattr(statement, "name", None)
                if name:
                    if name in outer_alias_names and name not in shadowed_aliases:
                        shadowed_aliases[name] = dict(scoped_aliases[name])
                    for key in outer_interval_keys:
                        if (
                            key == name or key.startswith(f"{name}.")
                        ) and key not in shadowed_intervals:
                            shadowed_intervals[key] = scoped_intervals.get(key)
            self.visit(statement, scoped_aliases, scoped_intervals)
        for name in outer_alias_names:
            if name in shadowed_aliases:
                active_aliases[name] = shadowed_aliases[name]
            elif name in scoped_aliases:
                active_aliases[name] = scoped_aliases[name]
            else:
                active_aliases[name] = self.poison_alias(active_aliases[name])
        for key in outer_interval_keys:
            if key in shadowed_intervals:
                value_after_scope = shadowed_intervals[key]
            else:
                value_after_scope = scoped_intervals.get(key)
            if value_after_scope is None:
                active_intervals.pop(key, None)
            else:
                active_intervals[key] = value_after_scope

    def visit(self, value, active_aliases, active_intervals):
        if value is None or isinstance(value, (str, int, float, bool)):
            return
        if isinstance(value, dict):
            for child in value.values():
                self.visit(child, active_aliases, active_intervals)
            return
        if isinstance(value, (list, tuple, set)):
            self.visit_sequence(list(value), active_aliases, active_intervals)
            return
        if isinstance(value, (VariableNode, ArrayNode)):
            return self.visit_declaration(value, active_aliases, active_intervals)
        if isinstance(value, AssignmentNode):
            return self.visit_assignment(value, active_aliases, active_intervals)
        if isinstance(value, ArrayAccessNode):
            self.record_access(
                value.array,
                value.index,
                active_aliases,
                active_intervals,
            )
            self.visit(value.index, active_aliases, active_intervals)
            return
        if isinstance(value, PointerAccessNode):
            self.record_access(
                value.pointer_expr,
                0,
                active_aliases,
                active_intervals,
            )
            return
        if isinstance(value, MemberAccessNode):
            self.record_access(
                value.object,
                0,
                active_aliases,
                active_intervals,
            )
        if (
            isinstance(value, UnaryOpNode)
            and self.generator.map_operator(value.op) == "&"
        ):
            for name in self.generator.glsl_mutation_target_names(value.operand):
                active_intervals.pop(name, None)
                self.generator.glsl_clear_index_component_intervals(
                    active_intervals,
                    name,
                )
            return
        if (
            isinstance(value, UnaryOpNode)
            and self.generator.map_operator(value.op) == "*"
        ):
            return self.visit_dereference(value, active_aliases, active_intervals)
        if isinstance(value, UnaryOpNode) and (
            self.generator.map_operator(value.op) in {"++", "--"}
            or value.op in {
                "PRE_INCREMENT",
                "PRE_DECREMENT",
                "POST_INCREMENT",
                "POST_DECREMENT",
            }
        ):
            return self.visit_increment(value, active_aliases, active_intervals)
        if isinstance(value, TernaryOpNode):
            return self.visit_ternary(value, active_aliases, active_intervals)
        if isinstance(value, BinaryOpNode) and self.generator.map_operator(
            value.op
        ) in {
            "&&",
            "||",
        }:
            return self.visit_short_circuit(value, active_aliases, active_intervals)
        if isinstance(value, BlockNode):
            self.visit_scoped_sequence(value, active_aliases, active_intervals)
            return
        if isinstance(value, IfNode):
            return self.visit_if(value, active_aliases, active_intervals)
        if isinstance(value, (SwitchNode, MatchNode)):
            return self.visit_switch(value, active_aliases, active_intervals)
        if isinstance(value, ForNode):
            return self.visit_for(value, active_aliases, active_intervals)
        if isinstance(value, ForInNode):
            return self.visit_for_in(value, active_aliases, active_intervals)
        if isinstance(value, (WhileNode, DoWhileNode, LoopNode)):
            return self.visit_while(value, active_aliases, active_intervals)
        if hasattr(value, "child_nodes"):
            for child in value.child_nodes():
                self.visit(child, active_aliases, active_intervals)


def glsl_workgroup_pointer_direct_access_intervals(
    generator,
    function,
    parameter_intervals,
):
    parameters = list(
        getattr(function, "parameters", getattr(function, "params", [])) or []
    )
    element_types = {
        parameter.name: generator.glsl_workgroup_pointer_element_type(parameter)
        for parameter in parameters
        if generator.glsl_workgroup_pointer_element_type(parameter) is not None
        and getattr(parameter, "name", None)
    }
    if not element_types:
        return {}

    constants = generator.initial_literal_int_constants(function)
    intervals = dict(parameter_intervals)
    for name, value in constants.items():
        literal = generator.literal_int_value(value, constants)
        if literal is not None:
            intervals[name] = (literal, literal)
    aliases = {
        name: {
            "kind": "workgroup-pointer",
            "root": name,
            "offset": 0,
            "offset_interval": (0, 0),
            "element_type": element_type,
        }
        for name, element_type in element_types.items()
    }
    access_intervals = {}
    mutated_interval_names = generator.glsl_function_mutated_interval_names(function)

    analysis = GLSLWorkgroupPointerAccessAnalysis(
        generator=generator,
        element_types=element_types,
        constants=constants,
        access_intervals=access_intervals,
        mutated_interval_names=mutated_interval_names,
    )
    visit_sequence = analysis.visit_sequence
    # The body is analyzed outside its own generation scope, so its parameters
    # must shadow same-named caller locals when helper calls are summarized.
    parameter_types = {
        parameter.name: getattr(
            parameter, "param_type", getattr(parameter, "vtype", None)
        )
        for parameter in parameters
        if getattr(parameter, "name", None)
    }
    with generator.glsl_lexical_source_type_scope(parameter_types):
        visit_sequence(getattr(function, "body", []), aliases, intervals)
    return access_intervals

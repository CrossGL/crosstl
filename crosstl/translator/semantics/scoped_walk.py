"""A post-order walk over every function body with the declarations in scope, for rule passes that need types."""

from __future__ import annotations

from .. import ast as A
from .hub_types import TypeScope
from .rewrite import child_slots, get_child

__all__ = ["ScopedWalker", "parameter_type", "program_stages"]


def parameter_type(parameter):
    declared = getattr(parameter, "param_type", None)
    return declared if declared is not None else getattr(parameter, "var_type", None)


def program_stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if isinstance(stages, dict) else list(stages)


class ScopedWalker:
    """Visits every node of every function body after its children, tracking declarations in a :class:`TypeScope`.

    Program constants, globals and constant buffers are in scope everywhere; a stage's variables and constant
    buffers within the functions of that stage; parameters and block-local declarations as they appear. Subclasses
    override :meth:`visit`.
    """

    def __init__(self, program):
        self.program = program
        self.scope = TypeScope(program)
        for constant in getattr(program, "constants", None) or []:
            self.scope.declare(constant.name, constant.const_type)
        for variable in getattr(program, "global_variables", None) or []:
            # Layout declarations share the list and declare no value.
            if getattr(variable, "name", None) is None:
                continue
            self.scope.declare(variable.name, variable.var_type)
            self.scope.declare_block_members(variable)
        for cbuffer in getattr(program, "cbuffers", None) or []:
            self.scope.declare_cbuffer(cbuffer)

    def run_program(self):
        for function in getattr(self.program, "functions", None) or []:
            if isinstance(function, A.FunctionNode):
                self.run(function, None)
        for stage in program_stages(self.program):
            for function in getattr(stage, "local_functions", None) or []:
                if isinstance(function, A.FunctionNode):
                    self.run(function, stage)
            entry = getattr(stage, "entry_point", None)
            if isinstance(entry, A.FunctionNode):
                self.run(entry, stage)

    def run(self, function, stage):
        if getattr(function, "body", None) is None:
            return
        scope = self.scope
        scope.push()
        if stage is not None:
            for variable in getattr(stage, "local_variables", None) or []:
                if getattr(variable, "name", None) is None:
                    continue
                scope.declare(variable.name, variable.var_type)
                scope.declare_block_members(variable)
            for cbuffer in getattr(stage, "local_cbuffers", None) or []:
                scope.declare_cbuffer(cbuffer)
        scope.push()
        for parameter in function.parameters or []:
            scope.declare(getattr(parameter, "name", None), parameter_type(parameter))
        self._walk(function.body)
        scope.pop()
        scope.pop()

    def _walk(self, node):
        if node is None or isinstance(node, (A.FunctionNode, A.StructNode)):
            return
        scoped = isinstance(node, (A.BlockNode, A.ForNode))
        if scoped:
            self.scope.push()
        for field, key in list(child_slots(node)):
            self._walk(get_child(node, field, key))
        if isinstance(node, A.VariableNode):
            self.scope.declare(node.name, node.var_type)
        else:
            self.visit(node)
        if scoped:
            self.scope.pop()

    def visit(self, node):
        """Called once per non-declaration node, after its children."""

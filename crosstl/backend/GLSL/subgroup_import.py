"""Import GL_KHR_shader_subgroup built-in functions as canonical CrossGL wave operations.

CrossGL spells subgroup operations with the wave names every generator lowers (see
``crosstl.translator.wave_operations``). Passing GLSL's ``subgroupAdd`` through as an ordinary call left it to the
generators, which emitted it verbatim for HLSL and Metal and refused it for SPIR-V.
"""

from __future__ import annotations

from crosstl.translator.wave_operations import GLSL_SUBGROUP_FUNCTIONS

from .OpenglAst import FunctionCallNode, VariableNode

# Counting the bits of a fresh ballot is a single wave operation of the predicate.
_BALLOT_COUNT_FOLDS = {
    "subgroupBallotBitCount": "WaveActiveCountBits",
    "subgroupBallotExclusiveBitCount": "WavePrefixCountBits",
}


def _call_name(node):
    name = getattr(node, "name", None)
    if isinstance(name, VariableNode):
        name = name.name
    return name if isinstance(name, str) else None


def _fresh_ballot_predicate(argument):
    if not isinstance(argument, FunctionCallNode):
        return None
    if _call_name(argument) != "subgroupBallot" or len(argument.args) != 1:
        return None
    return argument.args[0]


def import_subgroup_call(codegen, name, args):
    """The CrossGL text of a GLSL subgroup built-in call, or ``None`` when ``name`` is not one.

    A user function of the same name and arity shadows the built-in.
    """
    canonical = GLSL_SUBGROUP_FUNCTIONS.get(name)
    if canonical is None:
        return None
    if (name, len(args)) in getattr(codegen, "user_function_arities", ()):
        return None
    folded = _BALLOT_COUNT_FOLDS.get(name)
    if folded is not None and len(args) == 1:
        predicate = _fresh_ballot_predicate(args[0])
        if predicate is not None:
            return f"{folded}({codegen.generate_expression(predicate)})"
    arguments = ", ".join(codegen.generate_expression(arg) for arg in args)
    return f"{canonical}({arguments})"

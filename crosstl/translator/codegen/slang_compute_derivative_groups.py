"""Refuse derivative operations in a Slang compute stage whose thread group is not a derivative group.

slangc accepts compute derivatives only when the thread group forms a derivative group (slangc 2026.20,
issue #4657):

- a group with ``y == 1`` forms linear quads of four consecutive threads, so ``x`` must be a multiple of four;
- a group with ``y > 1`` forms 2x2 quads of the ``x`` and ``y`` dimensions, so ``x`` and ``y`` must both be even.

Otherwise slangc rejects the program with error E31211 (linear thread count) or E31210 (quad thread count).
The ``z`` dimension is free in both modes: linear quads are consecutive ``SV_GroupIndex`` values inside one
``z`` slice when ``y == 1``, and 2x2 quads form in ``x``/``y`` inside each ``z`` slice when ``y > 1``. That
``z`` reading follows the issue's rule; slangc was not available here to check a ``z > 1`` group directly.

A derivative operation is a ``dFdx``/``dFdy``/``fwidth`` builtin (including the coarse and fine forms, in
either the CrossGL spelling ``dFdx`` or the Slang spelling ``ddx`` the GLSL and HLSL frontends already use)
or a biased sample, which Slang spells ``SampleBias``: ``texture``/``textureProj`` with a bias argument, and
``textureOffset``/``textureProjOffset`` with an offset and a bias argument. Explicit gradients
(``textureGrad``) and explicit levels (``textureLod``) need no quads and are left alone, as is an unbiassed
sample. The check follows calls through helper functions, and a user function that shadows a builtin name is
not treated as the builtin.

A thread group whose size is not statically known (a specialization constant or a host-chosen size) is left
to slangc: it checks the size the pipeline specializes to.
"""

from __future__ import annotations

from .. import ast as A
from ..host_workgroup_size import HostDefinedWorkgroupSizeError
from .ast_walk import walk_ast_values
from .specialized_workgroup_size import has_specialized_workgroup_size
from .stage_utils import compute_local_size

__all__ = [
    "SLANG_DERIVATIVE_BUILTINS",
    "find_slang_compute_derivative_use",
    "slang_compute_derivative_group",
    "validate_slang_compute_derivative_group",
]

SLANG_DERIVATIVE_BUILTINS = frozenset(
    {
        # CrossGL spellings, as written in a CrossGL program.
        "dFdx",
        "dFdy",
        "dFdxCoarse",
        "dFdxFine",
        "dFdyCoarse",
        "dFdyFine",
        "fwidth",
        "fwidthCoarse",
        "fwidthFine",
        # Slang spellings, which the GLSL and HLSL frontends already use: a GLSL ``dFdx`` arrives as ``ddx``.
        "ddx",
        "ddy",
        "ddx_coarse",
        "ddx_fine",
        "ddy_coarse",
        "ddy_fine",
    }
)

# A biased sample, by the extra arguments the Slang emitter turns into ``SampleBias``: the number of extra
# arguments after the coordinate (and an explicit sampler, when one is given) that means a bias is present.
_BIAS_EXTRA_COUNTS = {
    "texture": 1,
    "textureProj": 1,
    "textureOffset": 2,
    "textureProjOffset": 2,
}


def slang_compute_derivative_group(size):
    """Whether a ``(x, y, z)`` thread group forms the derivative group slangc needs."""
    x, y, _z = (int(value) for value in size)
    if y == 1:
        return x % 4 == 0
    return x % 2 == 0 and y % 2 == 0


def _call_name(call):
    func = getattr(call, "function", None) or getattr(call, "name", None)
    if isinstance(func, A.IdentifierNode):
        return func.name
    if isinstance(func, str):
        return func
    if isinstance(func, A.MemberAccessNode):
        return func.member
    return None


def _call_arguments(call):
    return list(getattr(call, "arguments", None) or getattr(call, "args", None) or [])


def _texture_bias_operation(generator, call):
    """``"SampleBias"`` when ``call`` is a texture call the Slang emitter spells ``SampleBias``, else None."""
    name = _call_name(call)
    wanted = _BIAS_EXTRA_COUNTS.get(name)
    if wanted is None:
        return None
    args = _call_arguments(call)
    coord_index = 2 if generator.is_explicit_sampler_argument(args) else 1
    if len(args) - (coord_index + 1) != wanted:
        return None
    bias = args[-1]
    if generator.scalar_texture_bias_unsupported_reason(bias) is not None:
        # Not a scalar bias: the emitter refuses the call itself with its own reason.
        return None
    return "SampleBias"


def _derivative_use_in_function(generator, func, user_function_names):
    """The first derivative operation ``func`` calls directly: ``(operation, source)`` or None."""
    for node in walk_ast_values(getattr(func, "body", [])):
        if not isinstance(node, A.FunctionCallNode):
            continue
        name = _call_name(node)
        if name is None:
            continue
        if name == "SampleBias":
            return "SampleBias", node
        if name in SLANG_DERIVATIVE_BUILTINS and name not in user_function_names:
            return name, node
        if _texture_bias_operation(generator, node) is not None:
            return "SampleBias", node
    return None


def find_slang_compute_derivative_use(generator, entry):
    """The first derivative operation reachable from ``entry``: ``(operation, via)`` or None.

    ``operation`` is the CrossGL builtin name (``dFdx``) or ``"SampleBias"``; ``via`` is the helper the
    operation is reached through, or None when the entry calls it directly.
    """
    functions_by_name = getattr(generator, "user_functions_by_name", None) or {}
    user_function_names = getattr(generator, "user_function_names", None) or set()
    seen = set()
    queue = [entry]
    while queue:
        func = queue.pop(0)
        name = getattr(func, "name", None)
        if name in seen:
            continue
        seen.add(name)
        found = _derivative_use_in_function(generator, func, user_function_names)
        if found is not None:
            operation, _node = found
            return operation, None if func is entry else name
        for node in walk_ast_values(getattr(func, "body", [])):
            if not isinstance(node, A.FunctionCallNode):
                continue
            target = functions_by_name.get(_call_name(node))
            if target is not None and getattr(target, "name", None) not in seen:
                queue.append(target)
    return None


def _static_workgroup_size(execution_config):
    """The ``(x, y, z)`` thread group as ints, or None when its size is not statically known."""
    if has_specialized_workgroup_size(execution_config):
        return None
    try:
        size = compute_local_size(execution_config)
    except HostDefinedWorkgroupSizeError:
        return None
    try:
        return tuple(int(value) for value in size)
    except (TypeError, ValueError):
        return None


def validate_slang_compute_derivative_group(
    generator, entry, shader_type, execution_config
):
    """Refuse derivatives in a Slang compute stage whose thread group is not a derivative group.

    A stage with a valid group, a stage that takes no derivatives, and a stage whose size is not statically
    known are left alone.
    """
    if shader_type != "compute":
        return
    size = _static_workgroup_size(execution_config)
    if size is None or slang_compute_derivative_group(size):
        return
    found = find_slang_compute_derivative_use(generator, entry)
    if found is None:
        return
    operation, via = found
    entry_name = getattr(entry, "name", None) or "<entry>"
    x, y, z = size
    if via is None:
        called = f"calls {operation}"
    else:
        called = f"calls {operation} through '{via}'"
    if y == 1:
        needed = "a group with y == 1 needs x to be a multiple of four (linear quads)"
    else:
        needed = "a group with y > 1 needs even x and y (2x2 quads)"
    generator.refuse_lowering(
        f"Slang compute entry '{entry_name}' {called} but its {x}x{y}x{z} thread group is not a "
        f"derivative group: {needed}; slangc rejects the derivatives with error E31211 or E31210"
    )

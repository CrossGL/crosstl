"""Names of the WGSL built-in functions, which a function the shader declares must not take.

WGSL lets a module-scope declaration shadow a predeclared built-in function (WGSL, "Declaration and Scope"), so a
shader function named ``clamp`` would hide the built-in for the whole module, and a call of the built-in elsewhere in
the shader would reach the shader's function instead. The generator gives such a function another name, the way it
does for reserved words, and calls of it follow the new name.
"""

from __future__ import annotations

from .wgsl_must_use import WGSL_MUST_USE_BUILTINS

__all__ = ["WGSL_BUILTIN_FUNCTIONS"]

# Built-in functions that are not ``@must_use``: atomics, barriers and textureStore (WGSL, "Built-in Functions").
_EFFECT_BUILTINS = frozenset(
    {
        "atomicAdd",
        "atomicAnd",
        "atomicCompareExchangeWeak",
        "atomicExchange",
        "atomicLoad",
        "atomicMax",
        "atomicMin",
        "atomicOr",
        "atomicStore",
        "atomicSub",
        "atomicXor",
        "storageBarrier",
        "textureBarrier",
        "textureStore",
        "workgroupBarrier",
    }
)

WGSL_BUILTIN_FUNCTIONS = WGSL_MUST_USE_BUILTINS | _EFFECT_BUILTINS

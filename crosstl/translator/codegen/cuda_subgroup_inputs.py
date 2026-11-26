"""CUDA and HIP output: subgroup built-in inputs declared as stage variables.

The SPIR-V importer declares a compute shader's ``BuiltIn SubgroupSize`` (``SubgroupLocalInvocationId``,
``SubgroupId``, ``NumSubgroups``) input as the stage-local ``uint gl_SubgroupSize @input @gl_SubgroupSize;``. It is
the built-in itself, which the generators write as an expression of ``warpSize`` and the thread index
(:mod:`.gpu_subgroup_builtins`); declared at file scope it would be an uninitialized host variable, which a kernel
cannot read ("reference to __host__ variable in __global__ function").
"""

from .gpu_subgroup_builtins import subgroup_builtin_expression


def is_subgroup_builtin_input(var):
    """Whether the stage variable ``var`` is a subgroup built-in input rather than a variable to declare."""
    if subgroup_builtin_expression(getattr(var, "name", None)) is None:
        return False
    return any(
        str(getattr(attribute, "name", "")).lower() == "input"
        for attribute in getattr(var, "attributes", []) or []
    )

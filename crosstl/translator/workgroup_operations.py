"""CrossGL's workgroup collective operations: reductions, scans, votes and a broadcast over a whole workgroup.

The subgroup functions (GL_KHR_shader_subgroup_arithmetic, :data:`.wave_operations.GLSL_SUBGROUP_FUNCTIONS`) combine
the values of one subgroup. These combine the values of every invocation of the workgroup, as OpenCL C's
``work_group_reduce_<op>``, ``work_group_scan_inclusive_<op>``, ``work_group_scan_exclusive_<op>``,
``work_group_broadcast``, ``work_group_all`` and ``work_group_any`` do (OpenCL C 3.0, section 6.15.15, with the
operators of ``cl_khr_work_group_uniform_arithmetic``), and MAX's ``block.sum``, ``block.min``, ``block.max``,
``block.prefix_sum`` and ``block.broadcast`` (``max.gpu.primitives.block``). CrossGL names them as the subgroup
functions with ``workgroup`` for ``subgroup``:

* ``workgroup<Op>(x)`` returns the combination of ``x`` over every invocation of the workgroup to each of them;
* ``workgroupInclusive<Op>(x)`` and ``workgroupExclusive<Op>(x)`` return the combination over the invocations whose
  local invocation index is at most, or below, the caller's; an exclusive scan of the first invocation returns the
  operator's identity;
* ``workgroupAll(b)`` and ``workgroupAny(b)`` return whether ``b`` holds in every, or in some, invocation;
* ``workgroupBroadcast(x, id)`` returns ``x`` as the invocation whose local invocation index is ``id`` (a ``uint``,
  the same in every invocation) holds it.

``<Op>`` is ``Add``, ``Mul``, ``Min``, ``Max``, ``And``, ``Or`` or ``Xor``; the bitwise operators take integers, or
a ``bool``, where they are the logical operators. Every invocation of the workgroup must reach each call, in
workgroup-uniform control flow, as for ``workgroupBarrier``. The values are combined in local invocation index
order, so every invocation receives the same result.
"""

from __future__ import annotations

__all__ = [
    "WORKGROUP_BROADCAST",
    "WORKGROUP_COLLECTIVES",
    "WORKGROUP_OPERATORS",
    "WORKGROUP_VOTES",
]

# The CrossGL operator suffix and the operator name of :mod:`.semantics.partitioned_waves`'s tables.
WORKGROUP_OPERATORS = {
    "Add": "sum",
    "Mul": "product",
    "Min": "min",
    "Max": "max",
    "And": "and",
    "Or": "or",
    "Xor": "xor",
}
WORKGROUP_BROADCAST = "workgroupBroadcast"
WORKGROUP_VOTES = {"workgroupAll": "and", "workgroupAny": "or"}
# Each collective's ``(operator, scope)``: the scope is ``reduce``, ``inclusive``, ``exclusive`` or ``broadcast``.
WORKGROUP_COLLECTIVES = {
    **{
        f"workgroup{prefix}{suffix}": (operator, scope)
        for prefix, scope in (
            ("", "reduce"),
            ("Inclusive", "inclusive"),
            ("Exclusive", "exclusive"),
        )
        for suffix, operator in WORKGROUP_OPERATORS.items()
    },
    **{name: (operator, "reduce") for name, operator in WORKGROUP_VOTES.items()},
    WORKGROUP_BROADCAST: ("", "broadcast"),
}

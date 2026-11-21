"""MAX's memory barriers, named barriers and scheduling groups: how they are read from max.gpu.sync and probed."""

from dataclasses import replace

import pytest

from tools.coverage import probes, validators
from tools.coverage.execution import THREADS
from tools.coverage.features import mojo
from tools.coverage.model import VOID, Param, Signature, Type

SYNC_INIT = """
from .sync import (
    AMDScheduleBarrierMask,
    async_copy_arrive,
    cp_async_bulk_wait_group,
    mbarrier_arrive,
    mbarrier_init,
    named_barrier,
    schedule_group_barrier,
)
"""

SYNC = """
def named_barrier[
    num_threads: Int32,
](id: Int32 = 0):
    pass


def schedule_group_barrier(
    mask: AMDScheduleBarrierMask, size: Int32, sync_id: Int32
):
    pass


def async_copy_arrive[
    type: AnyType, address_space: AddressSpace, *, noinc: Bool = False
](address: Pointer[mut=True, type, _, address_space=address_space]):
    pass


def mbarrier_init[
    type: AnyType
](
    shared_mem: Pointer[mut=True, type, _, address_space=.SHARED],
    num_threads: Int32,
):
    pass


def mbarrier_arrive[
    type: AnyType
](shared_mem: Pointer[mut=True, type, _, address_space=.SHARED]) -> Int:
    return 0


def cp_async_bulk_wait_group[n: Int32, read: Bool = True]():
    pass
"""


def sync_features():
    features = mojo.parse("", "", "", "", "", sync_init=SYNC_INIT, sync=SYNC)
    return {f.name: f for f in features if f.group == "sync"}


def test_a_barrier_object_is_a_pointer_into_shared_memory():
    features = sync_features()
    shared = Param(Type("float"), "workgroup address")
    assert features["mbarrier_init"].signature == Signature(
        VOID, (shared, Param(Type("int")))
    )
    assert features["mbarrier_arrive"].signature == Signature(Type("int64"), (shared,))
    # An address space the call infers takes shared memory too, where MAX's memory barriers live.
    assert features["async_copy_arrive"].signature == Signature(VOID, (shared,))


def test_compile_time_counts_are_given_values():
    features = sync_features()
    # A named barrier waits for a whole number of warps; the id has a default.
    assert features["named_barrier"].spelling == f"named_barrier[num_threads={THREADS}]"
    assert features["named_barrier"].signature == Signature(VOID, ())
    assert features["cp_async_bulk_wait_group"].spelling == (
        "cp_async_bulk_wait_group[n=1]"
    )


def lines(feature, language):
    source = probes.build(feature, language).source
    return [line.strip() for line in source.splitlines()]


def test_mojo_allocates_the_barrier_in_shared_memory_and_crossgl_declares_it():
    init = sync_features()["mbarrier_init"]
    source = lines(init, "mojo")
    assert "from std.memory import stack_allocation" in source
    assert "from max.gpu.memory import AddressSpace" in source
    assert (
        "var a0 = stack_allocation[1, Float32, address_space=AddressSpace.SHARED]()"
        in source
    )
    assert "a0[0] = fin[i + 0]" in source
    assert "mbarrier_init(a0, a1)" in source
    source = lines(init, "crossgl")
    assert "shared float a0;" in source
    assert "a0 = fin[i + 0u];" in source
    assert "mbarrier_init(&a0, a1);" in source
    # A language without pointers says so rather than spelling one.
    with pytest.raises(probes.ProbeError, match="no probe passes glsl a pointer"):
        probes.build(replace(init, language="glsl"), "glsl")


def test_a_scheduling_group_takes_every_alu_instruction_and_crossgl_has_none():
    group = sync_features()["schedule_group_barrier"]
    source = lines(group, "mojo")
    assert "from max.gpu.sync import AMDScheduleBarrierMask" in source
    assert "schedule_group_barrier(AMDScheduleBarrierMask.ALL_ALU, a1, a2)" in source
    with pytest.raises(probes.NoSpelling, match="no AMD instruction scheduling"):
        probes.build(group, "crossgl")


@pytest.mark.skipif(
    not validators.available("mojo"), reason="the Mojo compiler is not installed"
)
@pytest.mark.parametrize("name", sorted(sync_features()))
@pytest.mark.parametrize("baseline", [False, True], ids=["probe", "baseline"])
def test_barrier_probes_are_valid_mojo(name, baseline):
    probe = probes.build(sync_features()[name], "mojo", baseline=baseline)
    verdict = validators.validate("mojo", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log

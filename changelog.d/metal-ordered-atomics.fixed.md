Metal atomics translate in both directions. Imported Metal atomic functions
without `_explicit` (`atomic_fetch_add`, `atomic_load`, `atomic_store`,
`atomic_exchange`, `atomic_compare_exchange_weak`, the other `atomic_fetch_*`
and the 64-bit `atomic_max` and `atomic_min`) used to reach CrossGL verbatim.
They are sequentially consistent, so the importer now writes the relaxed CrossGL
atomic between two sequentially consistent `atomicThreadFence` calls when the
call is a whole statement, and refuses it elsewhere. An `_explicit`
read-modify-write with an order other than `memory_order_relaxed` is refused
instead of losing its order, and `atomic_max_explicit` and `atomic_min_explicit`
import as `atomicMax` and `atomicMin`. Metal output writes a 64-bit unsigned
`atomicMax` or `atomicMin` whose result is unused as `atomic_max_explicit` or
`atomic_min_explicit` on an `atomic_ulong` view, and refuses one whose result is
used or whose target is threadgroup memory. The coverage probe for Metal's
`atomic_fetch_sub` now spells it in CrossGL as `atomicAdd` of the negated
operand.

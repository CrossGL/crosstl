<!-- Written from coverage.json by tools/coverage/docs.py when the Coverage workflow publishes. Do not edit. -->

# Mojo

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z (CrossTL 3.9.0) by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). The Coverage workflow writes this page from `coverage.json` each time it publishes; [COVERAGE.md](../COVERAGE.md) defines each outcome. Score = supported / (supported + refused + wrong).

The [backend guide](https://github.com/CrossGL/crosstl/blob/main/docs/source/backends/mojo.rst) describes how CrossTL translates Mojo.

| Direction | Score | Supported | Refused | Wrong | Untested | Executed | Reference tool |
|---|---:|---:|---:|---:|---:|---|---|
| output | 92.5 % | 298 | 24 | 0 | 0 | compile-only | Mojo 1.2.0 (2016d0cd) |
| input | 60.9 % | 196 | 126 | 0 | 0 | compile-only | Mojo 1.2.0 (2016d0cd) |

## Feature lists

- [Mojo 1.2.0 std.atomic functions and Atomic methods](https://github.com/modular/modular/blob/527d069ee11a41ac5320af1f9402db0a66b2b868/Mojo/stdlib/std/atomic/atomic.mojo), pinned at `527d069ee11a`
- [Mojo 1.2.0 std.atomic public API](https://github.com/modular/modular/blob/527d069ee11a41ac5320af1f9402db0a66b2b868/Mojo/stdlib/std/atomic/__init__.mojo), pinned at `527d069ee11a`
- [Mojo 1.2.0 std.bit functions](https://github.com/modular/modular/blob/527d069ee11a41ac5320af1f9402db0a66b2b868/Mojo/stdlib/std/bit/bit.mojo), pinned at `527d069ee11a`
- [Mojo 1.2.0 std.bit public API](https://github.com/modular/modular/blob/527d069ee11a41ac5320af1f9402db0a66b2b868/Mojo/stdlib/std/bit/__init__.mojo), pinned at `527d069ee11a`
- [MAX 26.7.0 max.gpu.primitives.block functions](https://github.com/modular/modular/blob/527d069ee11a41ac5320af1f9402db0a66b2b868/max/mojo/max/gpu/primitives/block.mojo), pinned at `527d069ee11a`
- [MAX 26.7.0 max.gpu.primitives.id functions](https://github.com/modular/modular/blob/527d069ee11a41ac5320af1f9402db0a66b2b868/max/mojo/max/gpu/primitives/id.mojo), pinned at `527d069ee11a`
- [Mojo 1.2.0 GPU index functions max.gpu re-exports](https://github.com/modular/modular/blob/527d069ee11a41ac5320af1f9402db0a66b2b868/Mojo/stdlib/std/_gpu/primitives/id.mojo), pinned at `527d069ee11a`
- [MAX 26.7.0 max.gpu synchronization functions](https://github.com/modular/modular/blob/527d069ee11a41ac5320af1f9402db0a66b2b868/max/mojo/max/gpu/sync/sync.mojo), pinned at `527d069ee11a`
- [MAX 26.7.0 max.gpu.sync public API](https://github.com/modular/modular/blob/527d069ee11a41ac5320af1f9402db0a66b2b868/max/mojo/max/gpu/sync/__init__.mojo), pinned at `527d069ee11a`
- [MAX 26.7.0 max.gpu.primitives.warp functions](https://github.com/modular/modular/blob/527d069ee11a41ac5320af1f9402db0a66b2b868/max/mojo/max/gpu/primitives/warp.mojo), pinned at `527d069ee11a`
- [Mojo 1.2.0 std.math functions](https://github.com/modular/modular/blob/527d069ee11a41ac5320af1f9402db0a66b2b868/Mojo/stdlib/std/math/math.mojo), pinned at `527d069ee11a`
- [Mojo 1.2.0 std.math public API](https://github.com/modular/modular/blob/527d069ee11a41ac5320af1f9402db0a66b2b868/Mojo/stdlib/std/math/__init__.mojo), pinned at `527d069ee11a`
- [Mojo 1.2.0 floating-point predicates std.math re-exports](https://github.com/modular/modular/blob/527d069ee11a41ac5320af1f9402db0a66b2b868/Mojo/stdlib/std/utils/numerics.mojo), pinned at `527d069ee11a`

## By category

Each cell counts the supported features of a category over its supported, refused and wrong ones.

| Category | output | input |
|---|---:|---:|
| compute | 16/16 | 10/16 |
| control-flow | 47/47 | 42/47 |
| function | 111/135 | 58/135 |
| operator | 42/42 | 34/42 |
| resource | 29/29 | 26/29 |
| stage | 1/1 | 1/1 |
| type | 52/52 | 25/52 |

## Translation pairs

The cells of the translation matrix that translate Mojo straight to another language or another language straight to it, measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z. Each cell measures the features of the source's input row that the target has too, and its page lists the gaps.

| Source | Target | Score | Supported | Refused | Wrong | Untested | Executed |
|---|---|---:|---:|---:|---:|---:|---|
| Mojo | CUDA | [78.8 %](../pairs/mojo-to-cuda.md) | 175 | 46 | 1 | 0 | compile-only |
| Mojo | GLSL ES | [63.3 %](../pairs/mojo-to-gles.md) | 145 | 70 | 14 | 2 | compile-only |
| Mojo | GLSL | [65.9 %](../pairs/mojo-to-glsl.md) | 153 | 64 | 15 | 0 | compile-only |
| Mojo | HIP | [78.1 %](../pairs/mojo-to-hip.md) | 175 | 47 | 2 | 0 | compile-only |
| Mojo | HLSL | [73.3 %](../pairs/mojo-to-hlsl.md) | 159 | 57 | 1 | 0 | compile-only |
| Mojo | Metal | [70.4 %](../pairs/mojo-to-metal.md) | 174 | 72 | 1 | 0 | compile-only |
| Mojo | Rust | [57.8 %](../pairs/mojo-to-rust.md) | 134 | 93 | 5 | 1 | compile-only |
| Mojo | Slang | [73.8 %](../pairs/mojo-to-slang.md) | 169 | 59 | 1 | 0 | compile-only |
| Mojo | SPIR-V | [71.4 %](../pairs/mojo-to-spirv.md) | 170 | 67 | 1 | 0 | compile-only |
| Mojo | WGSL | [64.8 %](../pairs/mojo-to-wgsl.md) | 142 | 75 | 2 | 0 | compile-only |
| CUDA | Mojo | [88.8 %](../pairs/cuda-to-mojo.md) | 239 | 24 | 6 | 0 | compile-only |
| GLSL ES | Mojo | [94.7 %](../pairs/gles-to-mojo.md) | 213 | 11 | 1 | 2 | compile-only |
| GLSL | Mojo | [94.7 %](../pairs/glsl-to-mojo.md) | 216 | 11 | 1 | 0 | compile-only |
| HIP | Mojo | [87.1 %](../pairs/hip-to-mojo.md) | 237 | 26 | 9 | 0 | compile-only |
| HLSL | Mojo | [92.2 %](../pairs/hlsl-to-mojo.md) | 200 | 13 | 4 | 0 | compile-only |
| Metal | Mojo | [71.1 %](../pairs/metal-to-mojo.md) | 182 | 70 | 4 | 0 | compile-only |
| OpenCL C | Mojo | [89.4 %](../pairs/opencl-to-mojo.md) | 194 | 20 | 3 | 0 | compile-only |
| Rust | Mojo | [68.9 %](../pairs/rust-to-mojo.md) | 173 | 34 | 44 | 1 | compile-only |
| Slang | Mojo | [93.9 %](../pairs/slang-to-mojo.md) | 216 | 11 | 3 | 0 | compile-only |
| SPIR-V | Mojo | [75.0 %](../pairs/spirv-to-mojo.md) | 174 | 12 | 46 | 2 | compile-only |
| WebGL | Mojo | [97.1 %](../pairs/webgl-to-mojo.md) | 165 | 5 | 0 | 0 | compile-only |
| WGSL | Mojo | [92.6 %](../pairs/wgsl-to-mojo.md) | 200 | 13 | 3 | 2 | compile-only |

## Gaps

A diagnostic drops its line numbers and writes the probe's own variables as `<probe>`, and one that several features share writes each feature's name as `<feature>`.

### Output

Refused, by diagnostic:

- `project.translate.undeclared-function: UndeclaredFunctionError: function '<feature>' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target mojo) [project.translate.undeclared-function]` (18): `async_copy_arrive`, `cp_async_bulk_commit_group`, `cp_async_bulk_wait_group`, `mbarrier_arrive`, `mbarrier_arrive_expect_tx_relaxed`, `mbarrier_arrive_expect_tx_shared`, `mbarrier_init`, `mbarrier_test_wait`, `mbarrier_try_wait_parity_shared`, `named_barrier`, `named_barrier_arrive`, `s_waitcnt`, `s_waitcnt_barrier`, `scalb`, `schedule_barrier`, `sm_id`, `umma_arrive_leader_cta`, `umma_arrive_peer_cta`
- `CrossGL has no spelling for this operation: no CrossGL warp reduction applies a function the kernel passes` (2): `warp.lane_group_reduce`, `warp.reduce`
- `CrossGL has no AMD instruction scheduling groups`: `schedule_group_barrier`
- `CrossGL has no spelling for this operation: CrossGL's compare-and-swap returns the value it found, and compare_exchange returns whether it stored and writes the value it found to its expected argument`: `Atomic.compare_exchange`
- `CrossGL has no spelling for this operation: CrossGL's lane match over the whole warp is CUDA's __match_all_sync, which also returns whether the lanes agree through a pointer`: `warp.match_all`
- `CrossGL has no spelling for this operation: no CrossGL function returns a quotient and a remainder together`: `divmod`

### Input

Refused, by diagnostic:

- `project.translate.undeclared-function: UndeclaredFunctionError: function '<feature>' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target mojo) [project.translate.undeclared-function]` (39): `align_down`, `align_up`, `async_copy_arrive`, `bit_not`, `bit_reverse`, `bit_width`, `byte_swap`, `ceildiv`, `count_leading_zeros`, `count_trailing_zeros`, `cp_async_bulk_commit_group`, `divmod`, `gamma`, `isclose`, `lane_id`, `log2_ceil`, `log2_floor`, `mbarrier_arrive`, `mbarrier_arrive_expect_tx_relaxed`, `mbarrier_arrive_expect_tx_shared`, `mbarrier_init`, `mbarrier_test_wait`, `mbarrier_try_wait_parity_shared`, `mulwide`, `next_power_of_two`, `pop_count`, `prev_power_of_two`, `recip`, `s_waitcnt`, `s_waitcnt_barrier`, `scalb`, `schedule_barrier`, `schedule_group_barrier`, `sm_id`, `syncwarp`, `ulp`, `umma_arrive_leader_cta`, `umma_arrive_peer_cta`, `warp_id`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call '<feature>[N](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]` (10): `block.broadcast`, `block.max`, `block.min`, `block.prefix_sum`, `block.sum`, `cp_async_bulk_wait_group`, `named_barrier`, `named_barrier_arrive`, `rotate_bits_left`, `rotate_bits_right`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the Mojo method call '<feature>(...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]` (9): `warp.broadcast`, `warp.max`, `warp.min`, `warp.prefix_sum`, `warp.shuffle_down`, `warp.shuffle_idx`, `warp.shuffle_up`, `warp.shuffle_xor`, `warp.sum`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call 'float[N](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]` (8): `aggregate-inout-parameter`, `array`, `array-copy`, `array-parameter`, `indexed-compound-assignment`, `inout-argument-lvalues`, `struct-array-member`, `zero-initializer`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call '<feature>[Ordering.RELAXED](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]` (5): `Atomic.fetch_add`, `Atomic.load`, `Atomic.max`, `Atomic.min`, `Atomic.store`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call '<feature>[DType.uint32](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]` (3): `warp.match_all`, `warp.match_any`, `warp.vote`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call '<feature>[warp.WARP_SIZE](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]` (3): `warp.lane_group_max`, `warp.lane_group_min`, `warp.lane_group_sum`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call 'Pair[N](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]` (3): `array-of-structs`, `struct-array-copy`, `struct-array-parameter`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call '<feature>[dtype=DType.float32](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]` (2): `inf`, `nan`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier 'WARP_SIZE' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target mojo) [project.translate.undeclared-identifier]` (2): `num-subgroups`, `subgroup-size`
- `project.translate.crossgl-syntax-error: CrossGLSyntaxError: Expected function name, got LBRACE`: `reserved-type-name`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the Mojo method call 'h.join(...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `vector-constructor`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the empty subscript 'p[]' has no CrossGL form: the importer cannot tell a pointer dereference from another type's __getitem__() (target cgl) [project.translate.mojo-import-unsupported]`: `pointer-local`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the empty subscript 'x[]' has no CrossGL form: the importer cannot tell a pointer dereference from another type's __getitem__() (target cgl) [project.translate.mojo-import-unsupported]`: `buffer-element-argument`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call 'iota[dtype=DType.float32, width=N](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `iota`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call 'warp.lane_group_reduce[shuffle = warp.shuffle_xor, num_lanes = warp.WARP_SIZE](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `warp.lane_group_reduce`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call 'Atomic.compare_exchange[success_ordering = Ordering.RELAXED, failure_ordering = Ordering.RELAXED](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `Atomic.compare_exchange`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call 'warp.reduce[warp.shuffle_xor](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `warp.reduce`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call 'bool[N](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `bool-array`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call 'materialize[ORIGIN](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `constant-struct`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call 'materialize[WEIGHTS](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `constant-array`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call 's.cast[DType.uint32](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `vector-shift`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call 'v.cast[DType.int32](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `integer-promotion`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo construct: the parameterized Mojo call 'vec2[N](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `vector-array`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the Mojo method call 'Layout.row_major(...)' has no CrossGL form in the importer; the Mojo method call 'LayoutTensor[DType.float32, N, MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation(...)' has no CrossGL form in the importer (target cgl`: `shared-2d-array`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the Mojo method call 'a.lt(...)' has no CrossGL form in the importer; the Mojo method call 'a.eq(...)' has no CrossGL form in the importer; the Mojo method call 'lt.reduce_or(...)' has no CrossGL form in the importer; the Mojo method call 'lt.reduce_and(..`: `vector-comparison`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the Mojo method call 'a.lt(...)' has no CrossGL form in the importer; the Mojo method call 'a.ge(...)' has no CrossGL form in the importer; the Mojo method call 'u.gt(...)' has no CrossGL form in the importer; the Mojo method call 'u.le(...)' has no Cross`: `integer-vector-comparison`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the Mojo method call 'b.eq(...)' has no CrossGL form in the importer; the Mojo method call 'b.ne(...)' has no CrossGL form in the importer; the Mojo method call 'e.reduce_and(...)' has no CrossGL form in the importer; the Mojo method call 'd.reduce_or(...)`: `bool-vector-logic`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the Mojo method call 's.width(...)' has no CrossGL form in the importer; the Mojo method call 's.widen(...)' has no CrossGL form in the importer; the Mojo method call 's.at(...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-impo`: `struct-method`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the empty subscript 'p[]' has no CrossGL form: the importer cannot tell a pointer dereference from another type's __getitem__(); the empty subscript '(q + i)[]' has no CrossGL form: the importer cannot tell a pointer dereference from another type's __getit`: `pointer-arithmetic`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the parameterized Mojo call '(outp + i).bitcast[Atomic[Int32]](...)' has no CrossGL form in the importer; the empty subscript 'N[]' has no CrossGL form: the importer cannot tell a pointer dereference from another type's __getitem__(); the parameterized Moj`: `Atomic.fetch_sub`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the parameterized Mojo call '(v * N.N).shuffle[N, N](...)' has no CrossGL form in the importer; the parameterized Mojo call 'v.shuffle[N, N, N, N](...)' has no CrossGL form in the importer; the parameterized Mojo call '(v + N).shuffle[N, N](...)' has no Cr`: `expression-swizzle`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the parameterized Mojo call 'Array[Float32, N](...)' has no CrossGL form in the importer; the parameterized Mojo call 'Array[Float32, N*N](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `constant-array-size`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the parameterized Mojo call 'a.cast[DType.float32](...)' has no CrossGL form in the importer; the parameterized Mojo call 'g.cast[DType.int32](...)' has no CrossGL form in the importer; the parameterized Mojo call 'a.cast[DType.uint32](...)' has no CrossGL`: `vector-conversion`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the parameterized Mojo call 'float[N](...)' has no CrossGL form in the importer; the Mojo method call 'a.eq(...)' has no CrossGL form in the importer; the Mojo method call 'N.reduce_and(...)' has no CrossGL form in the importer; the Mojo method call 'a.ne(`: `aggregate-equality`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the parameterized Mojo call 'float[N](...)' has no CrossGL form in the importer; the parameterized Mojo call 'Item[N](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `nested-member-assignment`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the parameterized Mojo call 'float[N](...)' has no CrossGL form in the importer; the parameterized Mojo call 'float[N][N](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `array-of-arrays`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the parameterized Mojo call 'float[N](...)' has no CrossGL form in the importer; the parameterized Mojo call 'spread4(x).shuffle[N, N](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `call-result-access`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the parameterized Mojo call 'int[N](...)' has no CrossGL form in the importer; the parameterized Mojo call 'float[N](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `lvalue-increment`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the parameterized Mojo call 'k.cast[DType.bool_](...)' has no CrossGL form in the importer; the parameterized Mojo call 'f.cast[DType.bool_](...)' has no CrossGL form in the importer; the parameterized Mojo call 'c.cast[DType.int32](...)' has no CrossGL fo`: `bool-vector-conversion`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the parameterized Mojo call 'materialize[OFFSETS](...)' has no CrossGL form in the importer; the parameterized Mojo call 'table[(N - k)].shuffle[N, N](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `constant-vector-array`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the parameterized Mojo call 'params.f.cast[DType.bool_](...)' has no CrossGL form in the importer; the parameterized Mojo call 'params.g.cast[DType.bool_](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `uniform-bool-vector`
- `project.translate.mojo-import-unsupported: MojoImportRefused: CrossGL cannot represent N Mojo constructs: the parameterized Mojo call 'v.shuffle[N, N, N, N](...)' has no CrossGL form in the importer; the parameterized Mojo call 'v.slice[N](...)' has no CrossGL form in the importer (target cgl) [project.translate.mojo-import-unsupported]`: `swizzle`
- `project.translate.mojo-lowering-unsupported: MojoLoweringRefused: Mojo cannot lower N construct: function: call to 'fence', a built-in function of another shading language that Mojo does not define and the program does not declare (target mojo) [project.translate.mojo-lowering-unsupported]`: `fence`
- `project.translate.mojo-lowering-unsupported: MojoLoweringRefused: Mojo cannot lower N construct: type: the built-in type 'SIMD' has no Mojo spelling (target mojo) [project.translate.mojo-lowering-unsupported]`: `generic-function`
- `project.translate.undeclared-function: UndeclaredFunctionError: function 'lane_id' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target mojo) [project.translate.undeclared-function]`: `subgroup-invocation-id`
- `project.translate.undeclared-function: UndeclaredFunctionError: function 'var' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target mojo) [project.translate.undeclared-function]`: `multiple-declarators`
- `project.translate.undeclared-function: UndeclaredFunctionError: function 'warp_id' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target mojo) [project.translate.undeclared-function]`: `subgroup-id`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier 'V' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target mojo) [project.translate.undeclared-identifier]`: `constant-expression`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'MutAnyOrigin' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target mojo) [project.translate.undeclared-type]`: `buffer-in-helper`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'MutUntrackedOrigin' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target mojo) [project.translate.undeclared-type]`: `shared-memory-in-helper`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'Real' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target mojo) [project.translate.undeclared-type]`: `type-alias`

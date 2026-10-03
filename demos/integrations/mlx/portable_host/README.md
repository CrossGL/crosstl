# Portable MLX Host Execution

This harness builds pinned MLX with its original Metal and CUDA backends disabled
and connects its C++ GPU evaluation path to translated native runtime packages.
Python calls such as
`mlx.core.arange` reach an MLX primitive, dispatch the translated upstream kernel,
and receive the device result in the original MLX array. Python operations and
upstream tests are not replaced.

## Scope

The current adapter implements `Arange` for `float32`, `int32`, `uint32`, `int64`
and `uint64`, plus 30 float32 unary operations from the pinned `unary.metal`:
absolute value, negation, sign, square, square root, reciprocal square root,
floor, ceil, round, exponential, expm1, logarithms, log1p, sigmoid, erf, inverse
erf, trigonometric functions and their hyperbolic and inverse forms. The exact
entry list is in `packages.py`. Shared-buffer views reuse upstream MLX's
shape, stride and ownership logic: strided views, broadcasts, copy aliases,
dimension insertion/removal, transpose, slicing, split, dependency/custom-transform
outputs and stop-gradient. Bounded integer and array indexing use the Gather
integration described below. A separate required native CI proof executes
unchanged `gather_front<float, int, int, N>` specializations for `N = 1, 4, 8`,
including negative/duplicate indices, partial chunks and exact storage words.
That separate kernel proof does not itself establish Python indexing coverage.
General gather now resolves template arguments from concrete array-member
elements, including its nested index-buffer access, and preserves address-space
qualifiers when passing addresses through pointer members. The unchanged wrapper
compiles and executes on Metal with its bfloat conversion helpers retained.
The required macOS gate covers 24 general-gather workloads, including zero-index
specializations. DirectX now lowers private constant/device pointer aggregates
to resource identities and element offsets. A required Windows gate covers the
same 20 indexed workloads; the four zero-index cases remain unsupported there.
OpenGL uses the same private resource identities and specializes helpers against
concrete storage-buffer bindings. Its required Linux gate covers the 20 indexed
workloads with explicit index-range assertions derived from their allocations.
Offsets remain signed 64-bit values until a proven final subscript conversion;
unbounded accesses still fail translation. These assertions describe the tested
workloads, not arbitrary MLX inputs. Both targets preserve the wrapper's returned
void helper call instead of omitting its writes. Separate controls check calls
through helpers, conditional returns and single argument evaluation on each OS.
Host Gather dispatch validates the actual source and index allocations before
building its specialization and supplying those range assertions.
Concrete float, half and integer vector
constructors and declared vector conversions have a separate required native
gate on all three platforms. It covers aliases, splats, mixed constructors,
receiver qualification, mutations and single evaluation, with original Metal
controls. A separate required macOS gate verifies native bfloat vector aliases,
declared conversions, rounding, raw payload copies, indexing, swizzles and vector
sizes against the original source. It does not establish DirectX/OpenGL bfloat
parity or complete arithmetic coverage. Aggregate lowering and host dispatch
remain bounded; the `gather_front` proof alone does not establish general
gather support.
Contiguous conversion, reshape, flatten and unflatten
dispatch translated copies when sharing storage is insufficient. Copies support
matching float32, int32, uint32 and bool arrays, including negative and zero strides.
The optional integer64 packages extend copies to int64 and uint64.
`Full`, including `zeros`, `ones` and `full_like`, uses those copies to materialize
broadcast values in these types, including int64 and uint64 with the optional
packages. Other storage widths are not yet
supported by this copy path.
Binary addition, subtraction, multiplication, minimum and maximum support
float32, int32 and uint32; division supports float32. Broadcasts and non-row-contiguous binary
inputs are materialized by translated copies before the unchanged vector-vector
binary entry executes. Casts between float32, int32 and uint32 use six unchanged
`v_copy` entries, including automatic promotion before mixed-type arithmetic.
Six additional casts convert between bool and those three numeric types.
Strided cast inputs are materialized through translated copies. Optional integer64
packages extend these casts, basic arithmetic and comparisons to int64 and uint64.
Casts involving other types still fail explicitly. Equal, not-equal and ordered comparisons
support float32, int32, uint32 and bool inputs with bool outputs. Logical and,
or and not use bool inputs, including upstream casts from numeric inputs.
The NaN-equality entry supports float32; the maintained host workloads exercise
scalar `array_equal(equal_nan=True)`. General array equality additionally needs
the reduction packages described below. The optional bitwise package family supports
Boolean operator overloads and 32-bit integer AND, OR, XOR, shifts and inversion.
Concatenate supports float32, int32, uint32 and bool inputs through destination-strided copies.
Optional selection packages support `where` with Boolean conditions and
float32, int32, uint32 or bool values. Optional absolute-value packages extend
`abs` to int32, uint32 and bool. Other primitives retain MLX's explicit
unsupported-GPU errors. Other unary inputs except Abs, LogicalNot and BitwiseInvert
must be float32. Stored-contiguous broadcasts and column-major views retain
their metadata; noncontiguous inputs use translated copies before unary dispatch.
Dispatch is synchronous and uses host staging buffers. Individual copy inputs and
most elementwise operations remain bounded to 65,535 stored elements; concatenation
can produce larger outputs as described below. Elementwise operations use one thread per workgroup; reductions
preserve upstream launch widths and multipass planning. Empty elementwise arrays
do not dispatch. This is a
host integration proof, not a complete MLX backend or a performance benchmark.

The pin is `9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8`. CI builds and runs the same
proof on Linux/OpenGL, Windows/Direct3D 12 and macOS/generated Metal. Mesa software
rendering and pinned WARP make Windows/Linux execution reproducible without
dedicated GPU runners. The macOS path uses the same callback adapter with generated
Metal packages, not MLX's original Metal backend. DirectX 10/11, Vulkan,
asynchronous queues, persistent GPU allocations, automatic operation
selection, and the complete MLX suite are not covered here.

## General Gather

The adapter implements MLX's `Gather::eval_gpu` using the unchanged pinned JIT
wrapper and `indexing/gather.h`. `HostRuntime(..., mlx_root=...)` enables on-demand
translation and packaging for the selected target. Cache identity includes the
upstream revision, translator implementation, packaging recipe, entry point and
validated index bound. Changed kernels or JIT definitions are rejected, including
when loading an existing cached package.

Supported source storage is float32, int32, uint32, int64, uint64 and bool;
indices may be signed or unsigned 32-bit or 64-bit integers. The current contract
allows one through ten index arrays, rank at most 64, and at most 65,535 elements
in each input storage span and output. Scalar indices, multiple indices,
negative index values, transposed and strided sources, and broadcast views retain
their MLX semantics. Negative storage strides are materialized by translated GPU
copies; source and output values are never computed or corrected on the CPU.
Metadata and index values are checked on the host before submission to establish
allocation safety. Empty results do not dispatch.

Each invocation retains its source artifact identity, submitted module, compiler
validation steps, available compiled binaries, buffer
uploads, launch geometry, result storage words and 32 trailing guard elements.
Floating-point values travel as binary32 words, preserving signed zero and NaN
payloads. Reflection must match every supplied binding and physical storage type;
invalid metadata or readbacks are rejected before writing into the MLX output.

```sh
python -m demos.integrations.mlx.portable_host.verify_gather \
  --mlx-root mlx-upstream --packages host-packages \
  --integer64 integer64-packages --reductions equality-reductions \
  --output-dir gather-evidence
```

The equality companion is `all_reduce_andbool_` at width 32. The verifier runs
42 workloads in separate CPU and native processes: six source storage types
times dense, transposed, strided, broadcast, reversed, scalar-index and
multiple-index layouts. It also runs unchanged upstream
`test_ops.TestOps.test_take` on each path, without skips. Retained uploads are
independently reconstructed into reference views and compared to native
readbacks and the MLX result. Missing dispatches, changed modules, corrupt guards
and incomplete upstream results fail verification. Three-OS CI requires this
check after building MLX with its original GPU backends disabled.

`GatherAxis`, used by `take_along_axis`, dispatches the unchanged
`indexing/gather_axis.h` specialization. Its source and index contiguity flags,
removed-axis shape and strides, axis extent and three-dimensional grid follow
the pinned upstream host implementation. It uses the same six source storage
types, four index types and allocation bounds as general Gather. Negative
source or index storage strides are materialized through translated copies;
positive and zero strides retain their views. Index values, allocation spans
and contiguity claims are checked before native submission.

Add `--axis` to the command above to run 60 axis-gather workloads and unchanged
upstream `test_take_along_axis` in separate CPU and native processes. Coverage
includes each axis, flattened input, noncontiguous source/index combinations,
broadcasts, negative indices, reversed views, exact floating-point payloads and
64-bit values beyond binary64's exact integer range. The required three-OS gate
retains the same compiled-module, upload, readback and guard evidence as general
Gather. No extra OpenGL index-range assertion is needed for this kernel.

Zero-index general-gather specializations, additional storage types, larger
allocations and full indexing/autodiff coverage are not claimed.
These workloads and the additional upstream test do not establish full-suite
or complete-backend parity.

## General Scatter

General-scatter translation preserves pointer address spaces when offsets use
scalar structure members ([#2053](https://github.com/CrossGL/crosstl/issues/2053))
and resolves constrained integer minimum/maximum helpers without selecting
unrelated float specializations
([#2052](https://github.com/CrossGL/crosstl/issues/2052)). Unsupported pointer
calls fail translation instead of replacing their computation with zero.
Compiler acceptance alone does not establish numerical correctness: the required
verifier retains every operation and fails on unsupported or incorrect results.

The adapter implements indexed `Scatter::eval_gpu` for int32 and uint32
replacement, addition, minimum and maximum. It translates the exact pinned JIT
wrapper and `indexing/scatter.h`; the upstream kernels and Python tests remain
unchanged. Indexed assignment and `array.at` updates reach this path through
MLX's normal primitive selection.

The source is copied by a translated GPU kernel before updates, preserving
source views and updates that alias them. Index arrays may use signed or unsigned
32-bit or 64-bit storage. One through ten index arrays, scalar indices,
noncontiguous and broadcast updates, negative indices and repeated destinations
are supported within the existing 65,535-element storage and rank-64 bounds.
Negative storage strides are materialized by translated copies. Output slice
bounds, index values, buffer spans and contiguity are checked before translation
or native submission. The host supplies the unchanged fifteen metadata/data
bindings plus each index buffer, and preserves upstream work-per-thread selection
of 1, 4, 8, 16 or 32 and its partial final chunk. Independent invocations use
one-thread workgroups; atomic updates still synchronize accesses to shared output
storage. This synchronous schedule is not a performance claim.

```sh
python -m demos.integrations.mlx.portable_host.verify_scatter \
  --mlx-root mlx-upstream --packages host-packages \
  --integer64 integer64-packages --output-dir scatter-evidence
```

The verifier runs 112 workloads and unchanged upstream
`test_array.TestArray.test_setitem_with_list` in separate CPU and generated-backend
processes. Retained evidence includes initial output storage, update and index
uploads, metadata, launch geometry, native compilation, readbacks and 32 trailing
guard elements. The audit reconstructs each indexed update from those uploads
and reconciles it with the public MLX result. Duplicate replacements use identical
values so the expected result does not assume a particular thread ordering.
A separate three-OS CI job requires this proof without extending the existing
indexing job's execution budget.

Host-level product updates, float and packed storage, zero-index specializations, larger
allocations and the complete upstream indexing/autodiff suite remain unsupported.

Integer atomic loads have a separate required numerical gate on all three native
targets. It covers signed and unsigned boundaries, buffer members, pointer and
resource-reference helpers, conditional calls, address side effects and a
32-thread shared counter. Two additional controls execute the unchanged signed
and unsigned `mlx_atomic_load_explicit` helpers from the pinned `atomic.h`,
preserving the input buffers and output guards. The OpenGL fixture explicitly
bounds the helper's 64-bit offset to the three elements it dispatches; this is
not an arbitrary-allocation guarantee. Metal retains `atomic_load_explicit`; DirectX and
OpenGL use an atomic OR with zero and return the observed value without changing
the stored bits. Those foreign operations require writable storage; read-only
inputs are diagnosed instead of silently changing their access contract. Only
explicit relaxed ordering is supported. Unsupported orders, element types and
untracked storage fail translation.

Integer compare-exchange has a separate required numerical gate on the same three
targets. It checks Boolean success, expected-value writeback on failure, one-time
address and argument evaluation, resource and workgroup storage, and contending
updates. Six additional controls execute the unchanged signed and unsigned MLX
compare-exchange and multiplication helpers, including eight concurrent
multiply-by-two updates. These use bounded, representable integer results and
explicit relaxed ordering. Native weak compare-exchange may fail spuriously;
success controls retry rather than assuming every matching comparison succeeds.
OpenGL's helper offsets use explicit workload bounds, not general allocation
inference. Unsupported orders, mismatched types, read-only destinations and
untracked expected-value pointers produce diagnostics.

The multiplication controls also exercise transitive constrained helper
specialization and compile-time branch selection. Separate native cases check
namespace ownership, explicit specializations, reused helpers, mixed runtime
and compile-time branches, and discarded type-dependent expressions. Dependent
calls with proven class arguments also retain their associated source namespaces
after specialization, so a later helper declaration can be found without making
unrelated namespaces or later primitive-only overloads visible. The scatter gate
includes unchanged `Prod<int>` policies with duplicate indices, negative and zero
factors, strided updates and partial chunks. These
kernel-level contracts do not yet redirect MLX host product updates or implement
packed storage. They do not establish full upstream-suite parity.

In particular, the full `test_array_at` method also needs random generation,
floating-point atomics and product updates; passing the list-index assignment
test does not establish that broader coverage. No output is computed or corrected
on the CPU by this adapter.

## Axis Scatter

`ScatterAxis::eval_gpu` dispatches the unchanged pinned
`indexing/scatter_axis.h` for int32 and uint32 replacement and addition. Public
`put_along_axis` operations use replacement; vector-Jacobian products of
`take_along_axis` exercise additive scatter through MLX's own differentiation
rules. Indices may be int32, uint32, int64 or uint64. Other output types and
reduction operations fail explicitly.

The adapter first copies the source with a translated copy kernel, preserving
the source even when updates alias it. Positive and zero strides retain their
views; negative update or index strides are materialized by translated copies.
The removed-axis shape, strides, contiguity flags and three-dimensional launch
follow upstream MLX. Update and output element counts are checked separately:
the number of updates need not equal the destination size. Each input storage
span, update count and output count is bounded to 65,535 elements; rank is at
most 64. Every reachable index and allocation span is validated before dispatch.
Empty updates retain the copied source and do not submit a scatter kernel.

```sh
python -m demos.integrations.mlx.portable_host.verify_scatter_axis \
  --mlx-root mlx-upstream --packages host-packages \
  --integer64 integer64-packages --output-dir scatter-axis-evidence
```

The required three-OS CI step runs 56 workloads in separate CPU and native
processes. They cover both operations and storage types, all four index types,
dense and transposed sources, strided and broadcast views, reversed views,
negative indices, flattened inputs, aliasing updates and empty updates. Duplicate
replacement indices carry identical values so the expected result does not
depend on thread ordering. Additive cases retain duplicate destinations to
exercise atomic accumulation.

The verifier reconstructs each result from retained uploads and checks the
native readback against the final MLX result. It also checks source preservation,
exact atomic storage layout, compiled modules, artifact identity, dispatch
accounting and 32 trailing output guards. Missing execution or corrupt evidence
fails verification; output values are never computed or corrected on the CPU.
This command does not claim the complete upstream `test_put_along_axis`:
that test includes floating-point scatter, still tracked by
[#1986](https://github.com/CrossGL/crosstl/issues/1986). Its evidence explicitly
records no upstream test-suite run and no complete-backend parity.

Five additional native controls check concrete function-object member forwarding,
nested call operators, receiver state and single argument evaluation. They are
required on all three operating systems, with original Metal comparisons on macOS.
Resource-reference lowering preserves integer atomic destinations as storage
lvalues, including buffer identity, element offsets and scalar aggregate fields.
Twenty-eight required native controls cover signed and unsigned updates, returned
old values, stores, aliases, resource selection, argument evaluation and contention
across workgroups. Ordinary functions with atomic-like names retain their own
semantics. Unsupported storage types and member-array destinations fail explicitly.
Floating-point atomics and Metal compare-exchange remain separate contracts;
this does not establish the full atomic scope of
[#2051](https://github.com/CrossGL/crosstl/issues/2051).

Two required native general-scatter cases use the unchanged pinned JIT wrapper and
`indexing/scatter.h`: one strided index/update input with `NWORK=1`, and two
contiguous index inputs with `NWORK=4`, including a partial final chunk. Both use
integer additive updates, duplicate and negative indices, and guarded output
storage. These are kernel-execution checks. General `Scatter::eval_gpu` host
integration, floating-point scatter and full indexing-suite parity remain outside
the current proof.

## Upstream Adaptations

Configure `core.autocrlf=false` before checking out MLX, including on Windows.
The native proofs compare upstream test files against their exact Git blob bytes;
automatic line-ending conversion is treated as a source modification. Every CI
job that checks out MLX applies this setting before checkout.

`prepare.py` requires a clean, exact-pinned checkout. It changes only the selected
no-GPU backend build definition and adds four explicitly named backend files:

- `crosstl_backend.cpp` registers a versioned synchronous dispatch callback,
  implements device/stream hooks, `Arange::eval_gpu` and 29 unary primitive hooks,
  and stages output storage. MLX's Log and Sqrt primitives select their log-base
  and reciprocal variants, yielding 30 float32 entries plus Boolean LogicalNot. View hooks reuse
  upstream shared implementations and perform no CPU elementwise computation.
  Slicing uses the upstream shared-buffer helper, including signed strides,
  offsets, nested slices and empty outputs. Noncontiguous unary inputs are
  materialized through the translated copy entry; contiguous input spans are
  checked against the allocation before upload.
  Copying layouts use the unchanged `ggn2_dynamic_copyuint32uint32` specialization
  on storage words, preserving every 32-bit payload without floating-point
  conversion. Source spans are checked against the actual allocation before
  upload; the logical origin is rebased for negative strides. Shape and stride
  metadata are independently validated at the callback boundary. Contiguous
  destination strides and three-dimensional dispatch follow the source kernel.
  Six binary primitive hooks select 16 unchanged `vv_` specializations. They
  validate input shapes and types, allocate dense output and stage strided inputs
  through translated copies. Neither layout conversion nor arithmetic falls
  back to CPU computation. Binary buffer donation is not implemented.
  Eight comparison/logical primitive hooks select 27 additional entries;
  Equal selects NaNEqual when requested by upstream. Non-floating NaN-equality
  uses the corresponding ordinary equality entry.
  BitwiseBinary selects 13 optional Boolean/int32/uint32 entries, reusing the
  binary layout checks and translated copies. Shift counts outside 0 through 31
  are rejected before the bitwise dispatch. There is no CPU bitwise fallback.
  BitwiseInvert selects two optional int32/uint32 unary entries, preserving
  upstream unary allocation, buffer donation and stored-contiguous layout metadata.
  AsType selects a source/destination-specific copy entry, validates array shapes
  and allocation bounds, and allocates a dense destination. Casts perform no CPU
  elementwise conversion.
  Reduce selects the unchanged whole-array entry and retains the upstream
  one-pass or two-pass plan for the supported dtypes and sizes.
  Full receives the upstream broadcast/cast value and materializes it with the
  same general-copy entry. Scalar fills, row/column broadcasts and strided values
  use the source strides, with fresh output allocation and no CPU fill loop.
  Concatenate allocates its output once and dispatches each nonempty input into
  a strided destination slice. Later copies upload the initialized output so
  earlier slices remain intact. There is no CPU concatenation fallback.
- `crosstl_dispatch.h` defines the typed C buffer/callback ABI.
- `crosstl_primitives.cpp` copies upstream unsupported primitive definitions,
  removing only the implemented primitive stubs.
- `crosstl_event.cpp` copies upstream events and adds synchronous GPU wait/signal
  handling. CPU scheduling behavior remains unchanged.

No upstream Metal kernel or Python test is edited. Preparation records hashes of
every adapted file. Integer argument conversion explicitly implements truncation
and modular wrapping; it does not depend on undefined out-of-range C++ casts.
The pinned Python `full` binding retains the dtype of an array-valued fill even
when a different `dtype` is requested. The harness verifies this unchanged
behavior; its mixed-type fill explicitly casts the broadcast value before `full`.
The host adapter is repository-specific; package generation, reflection, artifact
verification and native execution use the public CrossTL project interfaces.
Unary output allocation uses MLX's existing `set_unary_output_data` helper,
including its contiguous strides and buffer-donation behavior. It performs no
CPU calculation. Host inputs are copied before output readback, including when
MLX donates the input allocation. Translation explicitly selects the
`rne-flush` binary32 FMA profile required by the pinned Erf and Expm1 bodies.
All selected copy and cast entries genuinely use the same `[1, 1, 1]` workgroup
size and share one matching configuration rule. Distinct per-entry rules remain
subject to the project validation defect tracked in
[#1970](https://github.com/CrossGL/crosstl/issues/1970).

`runtime.py` retains the callback for the process lifetime and permits one
registration. It verifies argument names, direction, counts and physical layouts
against the selected descriptor. Native failures propagate as MLX errors; there
is no CPU computation fallback. The element cap is deliberate pending general
[native dispatch-limit validation](https://github.com/CrossGL/crosstl/issues/1959).
MLX stores Boolean arrays as one-byte values. Generated Metal preserves that
physical layout; generated HLSL and GLSL use four-byte unsigned storage. The
callback packs and unpacks canonical zero/one values according to reflection,
without evaluating logical operations on the CPU. Noncanonical input or output
values fail before destination storage is changed. Boolean copies use the
unchanged `ggn2_dynamic_copybool_bool_` entry and allocation checks use the actual
MLX item size. DirectX constant names follow the generator's sanitized entry
prefix, including removal of a trailing underscore from Boolean entry names.

Callback ABI version 3 carries explicit three-dimensional workgroup counts,
sizes and an optional exact thread grid separately from the logical element count. The C++ host selects each
launch; the Python boundary validates it against the operation and reflected
package before execution and records both dimensions in the trace. Copy kernels
retain their two-elements-per-invocation grid, including odd final rows. The
currently integrated elementwise and copy entries still use one-thread
workgroups. Reduction dispatch carries the exact width and row grid for each
pass. Older callback versions are rejected; rebuild the adapted MLX
wheel when updating the ABI.

Copy destinations additionally accept direction value 2 for initialized
input/output storage. Values 0 and 1 retain their input-only and output-only
meanings, and the buffer layout and callback signature are unchanged. Older
callbacks reject direction 2 rather than silently discarding destination data.
The first concatenation copy uses output-only storage initialized by the runtime;
later copies preserve it. No uninitialized host output is uploaded.

## Whole-Array Reductions

The optional reduction packages connect MLX's `Reduce::eval_gpu` to unchanged
`all_reduce` entries for float32, int32 and uint32 sum, product, minimum and
maximum, plus Boolean all/any. Boolean minimum/maximum use upstream's all/any
mapping. Nonempty inputs and outputs must have matching storage types; numeric all/any,
Boolean sum/product and narrow or complex types are not implemented yet.

The host uses upstream's reduction planner. Noncontiguous whole-array inputs
are materialized with translated copies when required. Arrays up to 4,096
elements use the exact source launch width, rounded to a multiple of 32. Larger
bounded arrays retain the two source passes: 128 partial rows followed by a
32-thread final reduction. Intermediate values are staged synchronously, not
kept in persistent GPU allocations. Unsupported column plans still produce
explicit errors. No reduction arithmetic is
performed on the CPU by the adapter.

### Empty Reductions

The `init` package family translates upstream `init_reduce` for float32, int32
and uint32 sum/product and Boolean all/any. Empty Boolean sum/product uses MLX's
int32 output promotion; empty numeric all/any produces Boolean outputs without
reading input storage. Outputs with no elements require no dispatch, including
valid empty minimum/maximum outputs. Reducing an empty axis with minimum or
maximum retains MLX's upstream error.

Unsigned sum/product initialization is instantiated by MLX's JIT rather than
listed in `reduce.metal`. The package builder retains an additional translation
unit that includes the unchanged source and adds those two template declarations
using upstream's instantiation macro. Kernel bodies are not modified. The host
uses one-thread workgroups because this kernel observes only the global ID,
not subgroup or workgroup geometry. Nonempty outputs are bounded to 65,535
elements; unsupported output types and missing packages are rejected explicitly.

```sh
python -m demos.integrations.mlx.portable_host.reduction_packages --mlx-root mlx-upstream --target metal --family init --output-dir init-packages
python -m demos.integrations.mlx.portable_host.verify_empty_reductions --mlx-root mlx-upstream --packages packages --reductions init-packages --output-dir empty-evidence
```

Use `opengl` or `directx` on the corresponding native platform. The verifier
compares 68 cases against separate CPU MLX and NumPy references, checks 50 native
dispatches and their output guards, and runs unchanged upstream
`test_reduce.TestReduce.test_zero_size`. It retains readbacks, package hashes,
source adaptation records and three separate rejection controls. These checks
cover empty reductions, not the complete upstream test suite.
Output buffers start with values different from the required identity, so a
missing write cannot pass merely because the allocation was already zeroed.

The pinned CPU reduction planner can hang on empty arrays imported from NumPy
with zero strides. Its reference worker constructs equivalent contiguous empty
MLX arrays instead; the native worker still evaluates the imported NumPy arrays.
The retained records identify this input-construction difference. No CPU source,
test method, expected result or comparison tolerance is patched. The unchanged
upstream zero-size test is run in both workers with its original constructors.

`reduction_packages` generates every multiple-of-32 width from 32 through 1,024
by default: 448 entry/width combinations per target. HLSL and GLSL use explicit
32-lane software subgroups; they do not assume the hardware wave width. Metal
uses native subgroups and receives the launch dimensions through the callback.
The callback validates Metal widths independently while configured Metal variant
dimensions remain absent from package execution metadata, tracked in
[#1388](https://github.com/CrossGL/crosstl/issues/1388).
`--width` and `--entry` can restrict a diagnostic build, but such a subset does
not establish coverage of the complete bounded host plan.

```sh
python -m demos.integrations.mlx.portable_host.reduction_packages --mlx-root mlx-upstream --target opengl --jobs 2 --output-dir reduction-packages
python -m demos.integrations.mlx.portable_host.verify --mlx-root mlx-upstream --packages host-packages --reductions reduction-packages --output-dir reduction-evidence
```

The required three-platform workflow builds all widths and verifies actual MLX
calls against independent references and the CPU backend. Cases cover launch
boundaries, multi-pass reductions, reversed/transposed/broadcast inputs, integer
overflow, NaNs, infinities, signed zero and Boolean identities. Native traces
retain intermediate readbacks, size metadata, dispatch dimensions and output
guards. The verifier requires every source pass and retains readbacks before
comparison. These are additional host workloads, not replacements for upstream
tests or evidence that the full upstream reduction suite passes.

## Row Reductions

The row package family implements the same numeric and Boolean operations for
rows larger than 64 elements. It follows MLX's selection between the four-row
`row_reduce_simple` kernel and `row_reduce_looped` specializations with one, two
or five reduction dimensions. Simple launches require at least 32 input rows;
the final workgroup retains upstream's overlapping output tile. Looped launches
retain output strides, non-row reduction strides and source allocation offsets.
General plans use translated copies only when upstream's planner requires them.

The source width is 32 for rows through 512 elements, 128 through 1,024, and
then a multiple of 32 capped at 1,024. The default row build includes all 26
reachable widths and all 56 entries, producing 1,456 packages per target.
Runtime checks reject invalid ranks, mismatched metadata arrays, out-of-range
source spans and incorrect launch dimensions before device execution. The GLSL
simple-row index assertion covers 0 through 131,071; every dispatch checks the
source span plus four values per lane against that bound. This assertion does
not enlarge an allocation or permit an out-of-bounds memory access.

```sh
python -m demos.integrations.mlx.portable_host.reduction_packages --mlx-root mlx-upstream --target opengl --family row --jobs 2 --output-dir row-packages
python -m demos.integrations.mlx.portable_host.verify_rows --mlx-root mlx-upstream --packages host-packages --reductions row-packages --output-dir row-evidence
```

Row translation is split into eight Linux jobs per target, each containing all
56 entries and three or four disjoint launch widths. Translation does not require
the target driver. Each successful shard records the MLX pin, translator Git
revision, source hash and package-index hash. Native jobs download the eight
shards from the same workflow run and reject missing or duplicate shards, mixed
revisions, mismatched targets and changed artifact bytes before using them.
The collection retains each artifact's original package directory and descriptor.

The native matrix compiles all 1,456 variants on the corresponding operating
system: Metal with warnings as errors and fast math disabled, DXC shader model
6.6 with warnings as errors, or glslang for OpenGL followed by SPIR-V validation.
Compilation failures stop verification and retain compiler output. Module hashes
and per-variant results are uploaded alongside the translation checkpoints and
native numerical evidence.

For an equivalent local collection, build shards 0 through 7 into
`row-collection/shards/shard-N/packages`, then run:

```sh
python -m demos.integrations.mlx.portable_host.reduction_shards build --mlx-root mlx-upstream --target opengl --shard 0 --jobs 2 --output-dir row-collection/shards/shard-0/packages
# Repeat the build for shards 1 through 7, using the same translator revision.
python -m demos.integrations.mlx.portable_host.reduction_shards collect --directory row-collection --target opengl
python -m demos.integrations.mlx.portable_host.reduction_shards compile --directory row-collection --target opengl --output-dir row-compiled
python -m demos.integrations.mlx.portable_host.verify_rows --mlx-root mlx-upstream --packages host-packages --reductions row-collection --require-all-widths --output-dir row-evidence
```

The separate row CI matrix executes on all three operating systems. Workloads cover
every integrated entry, reachable launch boundaries, multidimensional axes,
slices, transposes, early/late NaNs, infinities and signed zeros. CPU and
generated-native results are compared with NumPy references. The verifier
requires the exact source entry, launch geometry,
one native reduction per case and intact output guards. Not every packaged
entry/width pair is executable within the current 65,535-element host limit.
The maintained 1,870 standalone cases execute 896 distinct entry/width pairs;
the other 560 receive compilation coverage, not numerical coverage. The retained
case list identifies precisely which combinations execute. `--require-all-widths`
rejects diagnostic subsets in CI; single-directory packages remain supported
for local checks. Compilation success is not evidence of numerical parity.
These workloads supplement, rather than replace, unchanged upstream tests.

One `HostRuntime` can load multiple reduction directories with
`reductions=[all_packages, row_packages]`. Each entry/width pair retains its
own package directory; duplicate variants and mismatched targets fail before
runtime initialization. Passing a single directory remains supported.

Use `verify_rows --all-reductions reduction-packages` with the row command above
to also verify mixed graphs in the same runtime. The additional 28 cases cover
row-to-scalar reduction and scalar-to-broadcast-copy-to-row-to-scalar reduction
for every integrated operation and storage type. Their 84 native dispatches
must preserve dependency order, intermediate values, launch geometry and output
guards. Artifact hashes are checked against the base, row or whole-array package
that supplied each dispatch. CI requires these cases in each row job, using
whole-array companion packages at width 32; the separate whole-array jobs
continue to build and execute all 32 widths.

Small-row launches use the unchanged `row_reduce_small` entries for rows of
1 to 64 elements. Pass the pinned checkout as `mlx_root` to `HostRuntime` to
enable on-demand translation. The host retains upstream's scalar/cooperative
choice: scalar when `(non_row_reductions < 32 && row_size <= 8)` or
`non_row_reductions <= 8`, otherwise cooperative with 32 threads per row.
Metal receives the exact thread grid. DirectX and OpenGL execute the complete
region plan with shared allocations, preserving incomplete workgroups without
rounding up the source grid or changing the upstream kernels.

Packages live under `small-rows` in the base package directory. Cache identity
includes the pinned revision, target, source entry, translation implementation,
recipe and complete region geometry. Source cleanliness and package integrity
are checked on reuse. The cache stores translated sources, not native binaries
or computed results; each dispatch uses the configured native compiler and
runtime. ABI version 2 wheels must be rebuilt.

```sh
python -m demos.integrations.mlx.portable_host.verify_small_rows --mlx-root mlx-upstream --packages packages --output-dir small-row-evidence
```

Use `--jobs 2` to run two native worker processes concurrently, as CI does.
Each worker retains its own results, dispatch trace, logs and exit status. The
verifier checks each partition and then the combined set of 34 workloads before
publishing evidence. Worker failures and deadlines remain fatal; concurrency
does not remove cases or relax numerical checks. The default is one worker.

The dedicated three-OS CI gate compares 34 host workloads against MLX CPU and
NumPy, checks unchanged host inputs and output guards, and retains package identity
and DirectX/OpenGL region module evidence. These are additional integration
workloads, not 34 additional upstream unit tests. The existing upstream-test
gate remains required. Every operation has distinguishable row results to reject
constant-output and wrong-row errors. Negative-stride row views, additional storage types,
the 65,535-element bound and full upstream-suite parity remain open work;
partial-workgroup coverage is tracked in
[#2011](https://github.com/CrossGL/crosstl/issues/2011).

### Column Reductions

The column family connects upstream looped and two-pass column plans to generated
kernels. The default build includes all 84 entries: fourteen operation/type
combinations, three template ranks, and two dispatch modes, each at 256 threads.
Selection follows upstream's reduction count and contiguous output span. Small
columns and the separate long-column algorithm still report explicit errors.

The adapter preserves reduction strides, broadcast strides, allocation offsets
and output geometry. Ranks, metadata array lengths, source spans and logical
sizes are checked before dispatch. Two-pass plans write 32 intermediate rows;
the second native dispatch consumes those actual results without host reduction
arithmetic. The existing 65,535-element logical and physical bounds still apply.

```sh
python -m demos.integrations.mlx.portable_host.reduction_packages --mlx-root mlx-upstream --target opengl --family column --jobs 2 --output-dir column-packages
python -m demos.integrations.mlx.portable_host.verify_columns --mlx-root mlx-upstream --packages host-packages --reductions column-packages --output-dir column-evidence
```

The column CI matrix requires native execution on Windows, Linux and macOS with
the same full entry set. Workloads use actual MLX reductions and a separate CPU
process, covering tile tails, interleaved reduction axes, sliced and broadcast
views, NaNs, infinities and signed zero. Independent references check every
intermediate and final native readback; retained traces include launch geometry,
artifact hashes and output guards. These are bounded integration workloads, not
evidence that the complete upstream reduction suite or MLX backend is supported.
Separate native processes also verify that invalid allocation spans, small and
long column plans, and oversized inputs fail before any shader dispatch.

Small-column integration also requires a native-barrier convergence check,
tracked in [#2021](https://github.com/CrossGL/crosstl/issues/2021). For example,
upstream reduces a `(31, 129)` array with two X workgroups of `[32, 8, 1]`.
In the final group, eight invocations reach the shared-memory barrier while
248 return before it. The generated HLSL and GLSL retain this control flow.
Compiler acceptance and passing local readbacks do not establish portable
barrier participation. The host continues to reject this plan until the source
semantics are established and the target lowering or checked launch contract
preserves them. The existing looped and two-pass gates do not cover this kernel.

## Random Generation Readiness

`RandomBits` is not integrated into the host adapter. Successful compilation of
the two DirectX random entries does not establish numerical parity. The
`random_audit` command translates unchanged `random.metal`, creates public runtime
packages and compares native results against an independent integer Threefry
reference. It retains translation reports, descriptors, input values, native
readbacks, compiler/dispatch identity and per-case failures. Incorrect output or
an unsupported runtime contract produces a nonzero exit status.

Run against a clean checkout of the pinned revision, using the matching native
platform and a new output directory:

```sh
python -m demos.integrations.mlx.random_audit --mlx-root mlx-upstream --target opengl --output-dir random-opengl-audit
```

Use `directx` on Windows or `metal` on macOS. The audit covers both contiguous
and strided key entries, one or three keys, odd/even word counts and 17 trailing
guard values, for 20 cases per target. GLSL index assertions are backed by these
bounded key spans and output extents. Partial-byte outputs are not covered;
their source allocation and tail-write contract still needs separate validation.

At CrossTL `046b8d15`, the unchanged original Metal kernels match all 20 reference
cases. Generated OpenGL passes glslang and SPIR-V validation but returns zeros
in all 20 native cases, with intact guards. Generated Metal has independent
compilation and byte-layout blockers. DirectX compiles both entries, but its
random numerical execution has not been established. The current blockers are:

| Contract | Issue |
| --- | --- |
| Reflected byte buffers and native byte transport | [#2022](https://github.com/CrossGL/crosstl/issues/2022) |
| Signed/unsigned byte conversion semantics | [#2023](https://github.com/CrossGL/crosstl/issues/2023) |
| Shared union storage in OpenGL | [#2024](https://github.com/CrossGL/crosstl/issues/2024) |
| Partial vector initialization in Metal | [#2025](https://github.com/CrossGL/crosstl/issues/2025) |
| Fixed-array range iteration in Metal | [#2026](https://github.com/CrossGL/crosstl/issues/2026) |
| Aggregate type lookup under name shadowing | [#2027](https://github.com/CrossGL/crosstl/issues/2027) |
| Narrow vector layout in Metal aggregates | [#2028](https://github.com/CrossGL/crosstl/issues/2028) |

CI checks the audit's reference, binding and evidence-validation contracts on all
three operating systems. The failing native random audit is not a passing host
gate and does not count toward upstream-suite coverage. After the blockers are
resolved, require this native audit in CI before enabling `RandomBits` dispatch.
No kernel edits, generated-source repairs or readback corrections are applied.

## Run

Use Python 3.12, a C++20 toolchain, CMake, Ninja and the relevant runtime.
Linux additionally needs OpenBLAS, LAPACK/LAPACKE and Mesa EGL development/runtime
packages. Windows needs Visual Studio C++ build tools and DXC; the workflow pins
DXC and WARP archives with SHA-256 checks. macOS requires Xcode's Metal and Swift
toolchains and an available Metal device. Run from the CrossTL repository root:

Python 3.13 also works for the Linux build, but the Windows graphics dependency
currently requires the published Python 3.12 wheel. CI uses 3.12 on all three
platforms and retains dependency-install output even when setup fails.

```sh
python -m pip install -e . moderngl PyOpenGL setuptools wheel cmake ninja nanobind numpy packaging
git clone https://github.com/ml-explore/mlx.git mlx-upstream
git -C mlx-upstream checkout 9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8
python -m demos.integrations.mlx.portable_host.prepare --mlx-root mlx-upstream --output adaptation.json
CMAKE_ARGS='-DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_CROSTL_HOST=ON -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON' python -m pip install -e ./mlx-upstream --no-build-isolation
python -m demos.integrations.mlx.portable_host.packages --mlx-root mlx-upstream --target opengl --output-dir host-packages
LIBGL_ALWAYS_SOFTWARE=1 python -m demos.integrations.mlx.portable_host.verify --mlx-root mlx-upstream --packages host-packages --output-dir host-evidence
```

For Windows use `directx-runtime` and `--target directx`. For macOS omit the
OpenGL dependencies and use `--target metal`. Keep the same build options in
`CMAKE_ARGS`, including `MLX_BUILD_METAL=OFF`: the generated Metal runtime is
external to MLX's original backend. It compiles each checked source with
warnings fatal and fast math disabled, then executes the reflected entry and
returns native buffer readbacks. Every target explicitly dispatches the
one-thread-per-workgroup geometry used by these packages.
Copy dispatch supports at most 64 axes, 65,535 logical elements and 65,535 uploaded
source words. Other sizes and storage widths remain explicit errors. The runtime
checks an additional 128-byte destination guard before copying device results
back into the MLX array.
The complete platform setup is in
[`mlx-portable-host.yml`](../../../../.github/workflows/mlx-portable-host.yml).
Output directories must be new so evidence from different runs cannot mix.

The host workflow first runs required native scalar-math and fused-arithmetic
checks, including precise arcsine, Windows signed-zero angle selection and
explicit binary32 FMA profiles. After checking out MLX, it checks
complex power, binary dispatch shapes, buffer layouts and native dispatch limits
before building the adapted library. These bounded checks retain their generated
artifacts, readbacks and test reports even if a later build or host test fails.
They also remain in the project-porting workflow; the focused host checks do not
replace corpus-wide or upstream-suite validation.

The integer-arithmetic gate also checks indexed mixed-width compound assignments.
HLSL captures effectful indices before its helper's copy-in/copy-out operation,
while retaining conditional and loop-local evaluation. Twelve original/generated
Metal and required Windows cases cover eight operators, private and nested arrays,
structure members, assignment results, unchanged neighbors and guards. Original
HLSL controls use the DirectX adapter on Windows. OpenGL rejects these unsupported
effectful compound destinations; this gate does not claim their OpenGL execution.

The Windows gate also executes generated floating-point atomic add and exchange
on scalar and structured buffers, using Shader Model 6.0 bitwise compare/exchange
for addition. Readbacks check contention, returned values, single operand
evaluation, pointer offsets, conditional execution and unchanged neighboring
fields. Special-value cases cover signed zero, infinities, NaNs and flushed
denormals; NaN payload preservation is not an arithmetic guarantee. The same
oracle is checked against the original Metal source in the macOS workflow.

Required arctangent and comparison-arithmetic gates cover two failures found
by the host workload on Windows: insufficient precision in the target `atan`
intrinsic, and loss of negative values when subtracting comparison results.
The gates execute reduced kernels and the unchanged pinned ArcTan/Sign entries
through public translation and native-loader interfaces. The host verifier
continues to check the original inputs and tolerances independently.
Arctangent controls construct signed lane inputs by bit reinterpretation so
the check measures the function on the intended binary32 values. Loss of
subnormal payloads in HLSL input negation is tracked separately in
[#1997](https://github.com/CrossGL/crosstl/issues/1997); it is not accepted as a
passing arctangent result or a completed expression-lowering contract.

The copy-layout gate executes eight unchanged float32 copy entries on eleven layouts:
vector/scalar copies, strided and broadcast one-dimensional inputs, transposes,
row broadcasts, three-dimensional permutations, and strided/broadcast
four-dimensional inputs with odd row lengths. Dynamic copies also check negative
source/destination strides, nonzero offsets and untouched destination gaps.
For the one-dimensional dynamic OpenGL entry, project index assertions bound the
final source/destination addresses to the independently enumerated workload.
They do not claim support for arbitrary 64-bit OpenGL resource addresses.
Readbacks must match independent
coordinate references bit for bit, preserve negative zero and leave 128-byte
trailing guards unchanged. The macOS gate also executes the original Metal
entries and checks that all input buffers remain unchanged. The host adapter
separately uses the unchanged uint32 general-copy specialization for bit-exact
32-bit storage copies, with its own layout and source-preservation workloads.

A separate required compiler gate translates the pinned
`seq_gated_delta_vjp_float_128_128_24_24_1` entry and compiles it with DXC,
Shader Model 6.2 and native 16-bit types. That gate retains the source identity,
translation report and compiled module. It does not establish numerical
correctness of the full backward kernel or add that operation to the MLX host
adapter. OpenGL floating-point atomic lowering remains outstanding.

## Required Evidence

The verifier checks the pin, reconstructs the five adapted files from the pinned
originals and current templates, and compares their exact bytes before and after
execution. Missing, changed or symlinked adapter files and unrelated tracked
source changes are rejected. It also verifies the unchanged upstream test source,
then runs isolated CPU-reference and translated-GPU processes. Both must pass
these upstream tests without skips:

- `test_arange_overload_dispatch`
- `test_arange_inferred_dtype`
- `test_arange_corner_cases_cast`
- `test_abs`, `test_negative`, `test_floor`, `test_ceil`
- `test_square`, `test_sqrt`, `test_rsqrt`
- `test_exp`, `test_expm1`, `test_erf`, `test_sin`, `test_cos`
- `test_transpose_noargs`, `test_transpose_axis`, `test_broadcast`, `test_split`
- `test_subtract`, `test_multiply`
- `test_diff`, `test_flip`
- `test_array.TestArray.test_array_type_cast`
- `test_bartlett_general`, `test_blackman_general`, `test_hamming_general`, `test_hanning_general`
- `test_shape_overflow_error`
- `test_comparisons`, `test_logical_not`, `test_logical_xor`
- `test_isclose`, `test_allclose`

The CPU reference uses the unchanged CPU backend in the same adapted MLX build;
it is not a separately rebuilt pristine binary or a Metal comparison.

It also checks 20 array-creation cases spanning all five types and lengths 0, 1,
7 and 257. The 130 unary records cover the same lengths for all 30 operations,
six special-value cases, two large-angle cases, 8,193 consecutive float32 inputs
at and above one for inverse hyperbolic cosine, and an
arange/abs/negative/square chain, totaling 16,202 unary output values. Independent Python math references
check both CPU and native results, not just agreement between them. Readbacks
must have complete values and unchanged case identities; finite comparisons
use `rtol=2e-5, atol=1e-6`, with exact zero values/signs and nonfinite
classification. The near-one inverse hyperbolic cosine case uses the upstream
`rtol=1e-5, atol=1e-6` limits. Upstream tests and their tolerances are unchanged.

The pinned generic CPU Erf approximation returns negative zero for both zero
input signs. The Apple Silicon Accelerate build uses eight-lane blocks with
the same behavior, followed by scalar tails that return positive zero for both
signs. The CPU reference follows those block boundaries and records its profile
in the summary. Generated GPU Erf must preserve the input sign as the source
Metal implementation does. Both references check zero bits exactly; no readback
is corrected to make the two paths agree.

Another 59 records check 461 view and chained-unary values against independently
constructed NumPy arrays, including offsets, reversed/gapped strides, zero-stride
broadcasts, split outputs, empty shapes, source preservation and exact 64-bit
integers. Positive, reversed, nested, transposed and broadcast slices feed
translated square kernels, including row slices and empty slices. Twenty-five
square dispatches, eight materialization copies and one dependency negation
check execution through views. Broadcast dispatch sizes use stored elements
rather than the larger logical shape. Views that only change metadata do not
count as kernel execution.

Another 33 records check copying layouts and source preservation across float32,
int32 and uint32. Independent NumPy strided references check all 1,716 storage
words exactly, including NaN payloads, signaling NaN words, subnormals and signed
zeros. Transposes, gapped/reversed strides, broadcasts, odd four-dimensional
rows, copying reshapes, flattening and empty outputs are covered. The 27 native
copy dispatches retain their three-dimensional geometry and checked guard words.

Binary workloads add 134 records with 5,274 outputs: empty/scalar/vector inputs,
257-element tails, matrices, transposes, broadcasts, reversed/gapped strides,
unsigned subtraction wraparound, NaNs, infinities and signed zeros. Operand words
must remain unchanged. Independent references use `rtol=2e-6, atol=1e-6` for finite
float32 results, exact zero signs and nonfinite classification, and exact integer
results. Minimum and maximum follow MLX's explicit second-operand tie rule;
NumPy's minimum/maximum zero-sign behavior is not used as that reference.
Every binary dispatch checks and retains a 128-byte output guard before host
readback. The native conditional-selection gate also checks scalar/vector payloads
and exactly-once condition and selected-branch evaluation, tracking
[#1998](https://github.com/CrossGL/crosstl/issues/1998).

Cast workloads add 50 records with 1,976 outputs across all six conversions and
eight layouts, plus two automatically promoted additions. Exact output words
check truncation toward zero, signed/unsigned integer wrapping, integer-to-float
rounding at binary32 precision boundaries, and source preservation. Floating
inputs for integer destinations are finite and within the representable range;
nonfinite and out-of-range float-to-integer conversions are not covered by this
portability proof. Each cast checks a 128-byte device output guard before host
readback. Empty conversions do not dispatch.

Full workloads add 62 records with 2,421 storage words, including scalar fills,
257-element tails, row/column and three-dimensional broadcasts, transposed and
reversed inputs, negative four-dimensional strides, `full_like`, zeros, ones,
empty outputs and source preservation. Exact words include signed zero,
subnormals and NaN payloads. A mixed-type fill also requires translated broadcast
materialization and explicit casting before the final copy. All 52 added dispatches retain
their native geometry and output guards.

Boolean workloads add 290 records with 11,497 outputs across comparisons, logical
operations, six casts, copies and fills. They cover scalar/empty arrays,
257-element tails, matrices, transposes, broadcasts, reversed strides, NaNs,
infinities and signed zeros. Source and output bytes must match independent
references exactly. Each Boolean output retains 32 alternating guard values,
encoded in the target's physical storage type; numeric cast outputs retain the
existing 128-byte guard. Empty arrays and already-contiguous aliases do not
count as kernel execution.

Subnormal comparison parity is not established by these workloads. An additional
eight-value probe found that original Metal and generated Metal compare binary32
subnormals as zero on the tested Apple device, while generated OpenGL on Mesa
retains their nonzero comparison behavior. The CPU reference agrees with OpenGL
for that probe, not original Metal. All six comparison operators are affected;
float-to-bool casts retain nonzero subnormals on both tested targets. An explicit
source comparison profile and cross-target parity coverage are tracked in
[#2000](https://github.com/CrossGL/crosstl/issues/2000). No host readback is
normalized to conceal this difference. The host adapter rejects subnormal float
comparison inputs on DirectX and OpenGL before dispatch until source parity is
established. This check does not reject Boolean casts or ordinary comparison
inputs, and does not claim to fix the underlying translator contract.

Native traces must start with the exact 919 nonempty workload dispatches, cover
all 93 entries, and retain artifact identities and runtime/device details.
Separate negative processes reject an unsupported primitive, oversized arange,
a missing artifact, unary inputs with unsupported dtype or size, contiguous and
strided unary inputs that exceed their allocations, and
copies with unsupported dtype, excessive size or an invalid source allocation.
Binary inputs with unsupported dtype or excessive size are also rejected, as
are casts with unsupported types, excessive size or an invalid source span.
Full also rejects unsupported int16 storage, excessive output size and a source
view extending before its allocation.
The native worker has a hard 300-second process-tree deadline; the CPU reference
and eighteen negative processes each retain a 180-second deadline. All are
attempted so a failure does not discard the other diagnostic results. CI allows
2,400 seconds for package translation and 4,000 seconds for verification within
a bounded 330-minute platform job. Small-row host workloads run in separate
300-minute platform jobs after the base host proof. These jobs also require
translated empty-reduction, bitwise, concatenation, selection and integer
absolute-value execution and retain their packages and readbacks.

The workflow preserves an active native run when a branch changes and keeps only
the newest pending run for that ref. This lets long reduction jobs finish without
building a queue of superseded revisions. Retained evidence identifies its source
revision; results from an earlier commit do not establish that a newer one passes.

The evidence directory retains before/after adaptation hashes, command status,
stdout/stderr, upstream test logs, dispatch traces and numerical results. The
schema-version-2 summary records both upstream test-file hashes and is written
only after source and result checks pass,
including explicit rejection messages from all eighteen negative cases. Source
identity checks do not attest to a separately supplied binary; CI builds MLX from
the verified sources and retains its build log. `fullUpstreamSuite` and
`fullTranslatedBackend` remain `false`: extending primitive coverage and then
running the entire suite is the next stage, not an implicit property of this proof.

## Bitwise Host Operations

The optional family contains 15 unchanged specializations: 13 from `binary.metal`,
int32/uint32 AND, OR, XOR, left shift and right shift, plus Boolean AND, OR and
XOR; and two int32/uint32 inversion entries from `unary.metal`.
Existing base packages remain at 93 entries. Rebuild older 13-entry optional
packages before using this adapter. Build the additional packages
and run the host proof with a prepared MLX checkout and its installed host build:

```bash
python -m demos.integrations.mlx.portable_host.packages \
  --mlx-root mlx-upstream --target opengl --family bitwise --output-dir bitwise-packages
python -m demos.integrations.mlx.portable_host.verify_bitwise \
  --mlx-root mlx-upstream --packages host-packages --bitwise bitwise-packages \
  --output-dir bitwise-evidence
```

Use `directx` on Windows or `metal` on macOS when building the corresponding
packages. The three-platform CI requires this proof after the base host test.
It compares 120 workloads against separate MLX CPU execution, NumPy and an exact
Python integer reference. Each native run requires 172 dispatches, including
67 translated layout copies. Binary operations materialize transpose, broadcast
and negative-stride inputs. Inversion preserves stored-contiguous transpose and
broadcast layouts; negative-stride inputs still require a translated copy. Empty
inputs require no dispatch. The proof retains numerical readbacks before any
comparison failure, validates compiler and native-dispatch identities, checks
artifact hashes and output guards, and requires ten rejection controls.
Inversion controls reject missing packages, int64 and uint8 storage, and inputs
above the stored-element limit.

Unsigned shifts exercise all 32 bits and counts through 31; signed right shifts
include negative operands. Signed left-shift cases use nonnegative values and
representable results, without making a parity claim for undefined source
overflow. Inputs and outputs remain bounded to 65,535 elements. Other integer
widths and Boolean shifts are not implemented by this family. Upstream maps
Boolean inversion to the existing LogicalNot path.
The complete upstream `test_bitwise_ops` also requires random generation and
those missing widths and operations; the 120 workloads are not a substitute for
passing that unchanged test. No upstream kernel or test is patched.

The workload layout helper reinterprets a NumPy base allocation using the view's
dtype before constructing the MLX view. This preserves signedness when the view
and its base have different dtypes; it does not perform arithmetic on the CPU.

## Concatenation

Concatenation reuses the unchanged word-copy and Boolean-copy entries in the
93-entry base package. Source inputs retain the 65,535-element logical and physical
span bounds. Destinations use checked signed 32-bit indices and can exceed that
input bound. The maintained large cases and unchanged upstream test each produce
65,536-element outputs. This does not establish arbitrary-size backend support.

Destination offsets, strides, allocation bounds and nonoverlapping element
addresses are checked before submission. Overlapping source/destination spans are
rejected. After execution, every untouched destination element and the trailing
guard must match the uploaded values before any host output is modified.
Both the destination contents and copy metadata are retained in the trace.

The proof runs 72 raw-payload cases across float32, int32, uint32 and bool.
It covers both matrix axes, flattening, transposes, negative strides, broadcasts,
empty inputs, three-input sequences and larger outputs. Float32 inputs include
signed zero, subnormal values, infinities and NaN payloads; these are copied as
storage words, not recomputed as floating-point values. Each native run requires
128 workload dispatches plus the unchanged upstream
`test_ops.TestOps.test_concatenate`, which exercises all axes and permutations.
Four rejection controls cover missing packages, unsupported int64/uint8 storage
and an input exceeding the per-copy bound.

The upstream test's array-equality assertion requires one Boolean reduction:

```bash
python -m demos.integrations.mlx.portable_host.reduction_packages \
  --mlx-root mlx-upstream --target opengl --entry all_reduce_andbool_ --width 32 \
  --output-dir concatenate-reductions
python -m demos.integrations.mlx.portable_host.verify_concatenate \
  --mlx-root mlx-upstream --packages host-packages --reductions concatenate-reductions \
  --output-dir concatenate-evidence
```

Use the matching target on Windows/DirectX or macOS/Metal. The existing
three-platform host workflow requires this proof and retains all failure output.
No upstream tests, kernels or tolerances are changed. The additional upstream
test is separate from the 34-test base selection; the full MLX suite remains
incomplete.

## Selection And Integer Absolute Value

The optional `selection` family contains four unchanged `v_Select` entries from
`ternary.metal`. MLX performs its normal dtype promotion and broadcasting, then
the adapter materializes strided operands through translated copies and dispatches
the selected kernel. Values may be float32, int32, uint32 or bool; the condition
must reach the adapter as bool. Outputs are limited to 65,535 elements. Empty
results do not dispatch.

The optional `absolute` family contains three unchanged `v_Abs` entries from
`unary.metal` for int32, uint32 and bool. These entries are also needed by MLX's
unchanged `where` test, whose comparison helper computes integer absolute values.
Stored-contiguous transposes and broadcasts retain their physical layout;
reversed inputs use translated copies. The bound is 65,535 stored elements.
The signed minimum follows the original Metal result, remaining `INT32_MIN`.
Neither family changes the base 93-entry package contract.

```bash
python -m demos.integrations.mlx.portable_host.packages \
  --mlx-root mlx-upstream --target opengl --family selection --output-dir selection-packages
python -m demos.integrations.mlx.portable_host.packages \
  --mlx-root mlx-upstream --target opengl --family absolute --output-dir absolute-packages
python -m demos.integrations.mlx.portable_host.verify_selection \
  --mlx-root mlx-upstream --packages host-packages --selection selection-packages \
  --absolute absolute-packages --reductions concatenate-reductions \
  --output-dir selection-evidence
```

Use the width-32 Boolean reduction package built above and the matching target
on Windows/DirectX or macOS/Metal. The three-platform workflow requires both
package families and native verification; compiler or numerical failure fails
the job. Evidence is retained even when verification fails.

The proof runs 45 selection and 24 absolute-value workloads separately on MLX
CPU and the translated backend. These cover empty/scalar/vector arrays, tails,
matrices, transposes, broadcasts, reversed strides, Boolean masks, integer
boundaries, numeric conditions and mixed-type promotion. Selection checks exact
finite float32 bits, signed zeros, subnormals and infinities, but only NaN
classification: the native JSON transport does not preserve NaN payloads.
Integer and Boolean values are exact. Every workload checks input preservation,
and native outputs include checked guard values and launch geometry.

The same workers run `test_ops.TestOps.test_where` from the pinned checkout
without changing its assertions. The native proof requires all workload
dispatches, upstream execution, compiler/dispatch identity and artifact hashes.
Eight separate processes check missing packages, unsupported types and oversized
inputs without dispatch. Before/after source checks require byte-identical
upstream tests and unchanged prepared sources. These 69 workloads and one
upstream test extend coverage; they do not establish full-suite parity.

## 64-Bit Integer Operations

The optional `integer64` package family selects 44 unchanged upstream entries:
two general copies, 18 casts, two absolute-value kernels, ten arithmetic kernels
and twelve comparisons. The host adapter retains MLX's dtype promotion and layout
handling. Casts connect int64 and uint64 with float32, int32, uint32 and bool;
arithmetic covers addition, subtraction, multiplication, minimum and maximum.
The base 93-entry package and callback ABI are unchanged.

The required three-OS CI job runs 360 workloads: each entry across empty,
scalar, vector, tail, matrix, transposed, reversed and broadcast layouts,
plus eight scalar/broadcast `full`, `zeros` and `ones` cases.
Already-contiguous copies and empty outputs need no dispatch. All 44 entries
must execute in the remaining cases. Checks include exact high-bit values,
input preservation, output guards, native compilation, artifact identity and
launch geometry. References use NumPy with exact result bytes, including casts;
float-to-integer cases remain inside the representable destination range.
The original Metal absolute-value kernel preserves `INT64_MIN`, and the
translated path must do the same.

Both workers also run the unchanged upstream `test_clip` and `test_meshgrid`
methods. These exercise integer promotion and composite comparisons through
the host adapter. The required companion packages provide 32-bit absolute
value, width-32 Boolean whole reductions and width-1 Boolean initialization.
Source checks reject altered upstream tests or adapter files.

```bash
python -m demos.integrations.mlx.portable_host.packages \
  --mlx-root mlx-upstream --target opengl --family integer64 --output-dir integer64-packages
python -m demos.integrations.mlx.portable_host.reduction_packages \
  --mlx-root mlx-upstream --target opengl --family init --entry init_reduce_andbool_ \
  --width 1 --output-dir integer64-init
python -m demos.integrations.mlx.portable_host.verify_integer64 \
  --mlx-root mlx-upstream --packages host-packages --integer64 integer64-packages \
  --absolute absolute-packages --reductions concatenate-reductions \
  --reductions integer64-init --output-dir integer64-evidence
```

Use the matching Windows/DirectX or macOS/Metal target. Build the base,
absolute-value and Boolean whole-reduction packages as described above.
The verifier also requires five rejection controls for missing packages,
unsupported int16 absolute value, oversized inputs, and unsupported 64-bit
reductions and bitwise operations.

This does not add 64-bit division, selection, concatenation or general unary
operations. The 65,535-element bounds and synchronous host-staging model remain.
The two upstream tests are additional coverage, not full-suite parity.

### Padding and Slice Updates

`Pad` fills the destination through an unchanged translated copy kernel, then
copies the input into its padded region. `SliceUpdate` copies the base array and
replaces the requested slice through the same kernel. No array computation is
performed by a CPU fallback. Destination strides may be negative; source and
destination bounds, preserved regions and nonoverlapping writes are checked
before native submission. Aliased base/update views remain alive while a distinct
output allocation is populated.

These hooks support float32, int32, uint32, bool, int64 and uint64, with at most
65,535 output elements. The wider types require the optional `integer64` packages.
MLX's unchanged edge, reflect and symmetric padding implementations compose these
hooks with shared-buffer slices. Slice-update reductions require the optional
packages described below; replacement support does not imply scatter or general
indexing support.

`verify_padding` runs 102 workloads on CPU and generated GPU paths, checking exact
storage bytes, intermediate copy readbacks, output guards, signed strides,
broadcast inputs, aliased updates, empty regions and the maximum supported output
size. Float copies include NaN payloads, subnormals and signed zero. Seven isolated
negative workers reject missing packages, unsupported types, oversized outputs
and reductions without the optional slice-update packages. The unchanged upstream `test_pad`,
`test_pad_reflect_symmetric` and `test_slice_update_reversed` methods must also pass
without skips; `test_pad` retains its gradient check. Their Boolean reductions
require widths 32 and 64.

```bash
python -m demos.integrations.mlx.portable_host.reduction_packages \
  --mlx-root mlx-upstream --target opengl --entry all_reduce_andbool_ --width 64 \
  --output-dir padding-reductions
python -m demos.integrations.mlx.portable_host.verify_padding \
  --mlx-root mlx-upstream --packages host-packages --integer64 integer64-packages \
  --reductions concatenate-reductions --reductions padding-reductions \
  --output-dir padding-evidence
```

The three-OS integer64 CI job requires this additional proof using its existing
host build and packages. It retains source hashes, compiler and dispatch identity,
raw readbacks and failed worker logs. The three additional upstream methods do
not establish full MLX test-suite or backend parity.

### Slice Reductions

The optional `slice-update` family instantiates the pinned upstream
`indexing/scatter.h::slice_update_op_impl` for Sum, Prod, Min and Max across
float32, int32, uint32, bool, int64 and uint64. Its 24-entry wrapper adds includes
and template instantiations only; it does not replace kernel bodies. The generated
wrapper and translation report are retained with the packages.

The host adapter copies the base into a distinct output, materializes non-dense
updates through translated copy kernels, then dispatches the unchanged reduction
kernel with shape, stride and offset metadata. No CPU arithmetic or readback
correction is used. Bounds checks use MLX's normalized start and update shape:
upstream intentionally retains unnormalized stop indices. Negative bounds,
clipped stops, reversed destinations and aliased inputs therefore retain upstream
slice semantics.

`crosstl_mlx_register_runtime` registers both dispatch and package-availability
callbacks, retaining dispatch ABI version 3. This lets the adapter reject a
missing reduction package before copying the base. The earlier dispatch-only
registration entry point remains available, but cannot enable slice reductions.
The Python adapter and prepared MLX build must be regenerated together.

`verify_slice_updates` requires 304 CPU/native workloads, the unchanged
`test_array_at_slice_update_extensive` method and eight isolated rejection
checks. It checks exact result storage, source preservation, native copy
and reduction readbacks, destination metadata, guards and artifact hashes.
The separate three-OS slice-update job requires the same verifier after building
its matching host adapter and downloading the same-run integer64 packages.

```bash
python -m demos.integrations.mlx.portable_host.packages \
  --mlx-root mlx-upstream --target opengl --family slice-update --output-dir slice-update-packages
python -m demos.integrations.mlx.portable_host.verify_slice_updates \
  --mlx-root mlx-upstream --packages host-packages --integer64 integer64-packages \
  --slice-updates slice-update-packages --reductions concatenate-reductions \
  --reductions padding-reductions --output-dir slice-update-evidence
```

Float slice updates use the public `ieee754-binary32` storage encoding while
retaining float32 shader arithmetic. Untouched NaN sign/payload bits, signed
zeros, infinities and subnormals are checked byte-for-byte; traces retain actual
native storage words alongside display values. Special-value storage cases do
not impose payload-preservation rules on arithmetic NaN results.

This bounded integration does not establish full upstream suite parity.
General scatter, other storage widths, outputs above 65,535
elements and asynchronous device-resident execution remain separate work.

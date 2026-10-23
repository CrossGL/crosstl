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
The optional integer64 packages extend copies to int64 and uint64. The optional
half packages provide float16 copies and float16/float32 casts as described below.
`Full`, including `zeros`, `ones` and `full_like`, uses those copies to materialize
broadcast values in these types, including int64 and uint64 with the optional
packages. Other storage widths are not yet
supported by this copy path.
Binary addition, subtraction, multiplication, minimum, maximum and MLX's floored
remainder support float32, int32 and uint32; division and LogAddExp support float32.
Integer remainders reject zero divisors and signed `INT_MIN % -1` before dispatch,
because all three targets leave them undefined. Broadcasts and non-row-contiguous binary
inputs are materialized by translated copies before the unchanged vector-vector
binary entry executes. `DivMod` (see below) instead dispatches the strided
`binary_two` kernels directly, for every real integer and floating type. Casts between float32, int32 and uint32 use six unchanged
`v_copy` entries, including automatic promotion before mixed-type arithmetic.
Six additional casts convert between bool and those three numeric types.
Strided cast inputs are materialized through translated copies. Optional integer64
packages extend these casts, basic arithmetic and comparisons to int64 and uint64.
Dense casts dispatch independent batches of at most 65,535 elements, advancing
source and destination pointers by their respective storage widths. The array
itself is not limited to one batch. Noncontiguous inputs still inherit the copy
path's size restrictions. Identity casts share their input without dispatch.
Casts involving other types still fail explicitly. Equal, not-equal and ordered comparisons
support float32, int32, uint32 and bool inputs with bool outputs. Logical and,
or and not use bool inputs, including upstream casts from numeric inputs.
The NaN-equality entry supports float32; the maintained host workloads exercise
scalar `array_equal(equal_nan=True)`. General array equality additionally needs
the reduction packages described below. The optional bitwise package family supports
Boolean operator overloads and 32-bit integer AND, OR, XOR, shifts and inversion.
Binary arithmetic, comparisons, logical AND/OR and bitwise binary operations use
independent batches of at most 65,535 elements. Both source pointers and the
destination pointer advance by their own storage widths, including Boolean
comparison outputs. Shift counts are checked across the complete input before
the first binary dispatch. Input materialization still inherits the copy limits.
Concatenate supports float32, int32, uint32 and bool inputs through destination-strided copies.
Optional selection packages support `where` with Boolean conditions and
float32, int32, uint32 or bool values. Optional absolute-value packages extend
`abs` to int32, uint32 and bool. Other primitives retain MLX's explicit
unsupported-GPU errors. Other unary inputs except Abs, LogicalNot and BitwiseInvert
must be float32. Stored-contiguous broadcasts and column-major views retain
their metadata; noncontiguous inputs use translated copies before unary dispatch.
Dispatch is synchronous and uses host staging buffers. Layout copies accept signed
32-bit element counts and storage spans, with up to 64 axes and 65,535 workgroups
per launch axis. Unary operations also batch stored elements without changing
their layout. Arange, selection and shaped reductions retain their current size limits;
binary operations, casts and concatenation can produce larger outputs as described below.
Elementwise operations use one thread per workgroup; reductions
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

## Cumulative Operations

`Scan::eval_gpu` now reaches the translated-package callback. Its source entry,
scalar types, axis extent, storage stride and workgroup geometry are checked
before native execution. Packages retain the unchanged pinned `scan.metal` and
its headers. The cache identity binds the complete launch, checked scalar values,
source revision and translation implementation. Scalar bounds remain obligations
on uploaded metadata, not substitutions into the source. DirectX and OpenGL
packages also retain an artifact-owned dispatch-region contract. Cache reloads
verify these contracts against the package, and native preparation checks them
again before submission. No prefix arithmetic is performed on the host, and
translation failures remain errors.

The required `--scans` host verification runs in the existing Metal, OpenGL and
DirectX jobs, within the existing 4,000-second host-verification deadline. Its
54 API cases cover inclusive sum, minimum and maximum for float32 and int32,
nonzero input offsets, row- and column-major contiguous views, and axis lengths
1, 33, 129 and 257. Those lengths select 32-, 64- and 96-thread workgroups.
Thirty cases use the actual strided entries with 33 columns, 256-thread
workgroups, a partial second column block, and axis lengths 1, 31, 32, 33 and 65.
Each case checks an independent exact reference, CPU/native agreement, unchanged
host input, device input and metadata observations, 32 input and output guards,
and the identities of the packaged source and compiled native modules. The
metadata audit matches each source parameter and ordinal, exact bound, checked
value and upload digest to the corresponding device observation. The scan
subprocess retains its 900-second deadline; no additional runner is allocated.

The separate complete-kernel scan tests retain their original sum, extrema and
complex references. Their packages declare the exercised axis interval 1..1025
and three-row launch geometry as checked contracts; the original inputs,
workgroup widths, guards and 240-second per-batch deadlines remain unchanged.

```bash
python -m demos.integrations.mlx.portable_host.verify_scans \
  --mlx-root mlx-upstream --packages host-packages --output-dir scan-evidence
```

This route remains bounded to 65,535 logical and stored elements. It uses the
pinned source planner with an explicit 1,024-thread source pipeline limit;
larger native widths require their own qualification. Empty arrays do not
dispatch. Stored-contiguous layouts and eligible donation are retained;
other layouts require the existing translated-copy path. Unsupported complex
and narrow-integer host storage is rejected explicitly.

This is not complete cumulative-operation coverage. Reverse scans currently
encounter unproved wide pointer offsets on OpenGL. Forward strided entries use
checked scalar and dispatch bounds for the software-subgroup path; their
qualification is limited to the cases above.
Product, log-add-exp, additional scalar types and full unchanged upstream scan
tests require separate numerical qualification. The independently captured
original bfloat16 scan-versus-reduction failure in
[`METAL_HOST.md`](../METAL_HOST.md#captured-scan-replay) remains unresolved; this
host proof does not change its input, assertion or required outcome.

## Batched Casts

The required batched-cast verifier also runs in each native host CI job:

```bash
python -m demos.integrations.mlx.portable_host.verify_cast_batches \
  --mlx-root mlx-upstream --packages host-packages --output-dir cast-batches
```

Its eleven API cases compare CPU and translated results bit for bit across all
six float32/int32/uint32 conversions, Boolean-to-float and float-to-Boolean
conversion, and an identity cast. Offset views, the 65,535-element boundary,
65,536 elements and a 131,075-element matrix cover pointer advancement and tails.
The verifier retains complete binary input/output arrays and checks each native
batch's uploaded values, output, guards, launch and compiled artifact identity.
This is cast coverage, not a claim that large composed operations or the full
upstream suite pass. Half and 64-bit batches are not part of these eleven cases.

## Larger Layout Copies

The required native host jobs also run the large-copy verifier:

```bash
python -m demos.integrations.mlx.portable_host.verify_large_copies \
  --mlx-root mlx-upstream --packages host-packages --output-dir large-copies
```

Its 35 API cases compare CPU and generated results bit for bit for float32, int32,
uint32 and Boolean storage. They cover negative strides, transposes, permutations,
broadcasts, nonzero source offsets, source spans larger than the logical array and
each exact launch-axis limit. Saved evidence includes source/result arrays,
uploaded buffers, layout metadata, destination guards and compiled native artifacts.
Original source arrays must remain unchanged. Half and 64-bit large copies are not
part of these cases.

Copies use the unchanged two-elements-per-thread dynamic copy kernel and its
three-dimensional launch. A flat copy above 131,070 elements still exceeds the
first launch-axis limit; larger multidimensional arrays can fit. Invalid backing
views, overlapping destinations and out-of-range signed indices are rejected
before native submission. The adapter stages whole source/destination spans, so
large sparse layouts can require substantial host and device memory. This removes
the former 65,535-element copy limit, not the remaining shaped-reduction size limits.

## Batched Binary Operations

The required native host jobs run this verifier inside their existing bounded
verification step, without adding runner jobs:

```bash
python -m demos.integrations.mlx.portable_host.verify_binary_batches \
  --mlx-root mlx-upstream --packages host-packages --output-dir binary-batches
```

Its 52 API cases cover every base arithmetic, comparison and logical binary entry
except NaN-equality, whose large-array API additionally needs the optional reduction packages.
They include 65,535, 65,536 and 131,075 elements, distinct input offsets, a large
transpose, a reversed input and row/column broadcasts. Remainder divisors are
nonzero signed powers of two (uint32: 1 through 997), so every floored result is
exact on each target. The 106 binary batches and four materializing copies require
exact CPU/reference/native bytes (LogAddExp: within 1e-6 + 2e-6 |reference|, the
base workload's float32 bound, because exp and log1p are not correctly rounded, and
each native batch must carry the retained readback's exact bits), unchanged
source storage, physical Boolean widths, uploaded inputs, destination guards,
launch geometry and retained compiled artifacts. Float operands are finite and
exactly representable; the existing nonfinite and small-array tests remain in
place. Half and 64-bit batch execution are not covered by these cases.

These checks establish the batched operations, not full upstream-suite parity.

## DivMod

`DivMod` is MLX's two-output floored quotient and remainder. Its host port
follows `binary_op_gpu` in `mlx/backend/metal/binary.cpp` for row-contiguous
outputs: scalar and dense operands use the unchanged `ss_`, `sv_`, `vs_` and
`vv_` `binary_two.metal` entries, and every other layout collapses its
contiguous axes and uses `g1_`, `g2_`, `g3_` or `gn2_` with the operands'
strides, so broadcasts need no copy kernel. Inputs with negative strides are
first copied densely. The single-element `N = 1` entries are used throughout,
so no work-per-thread variant is needed. All 88 kernel and dtype pairs (float16,
bfloat16, float32 and the 8-, 16-, 32- and 64-bit integers) are translated from
the pinned source on first use and cached under an identity that covers the
translator, the packaging recipe and the source hash. The floating entries pin
the same binary32 division, comparison, remainder and additive profiles, and
the binary16 remainder profile, as the `binary_two` kernel parity tests.
OpenGL packages assert their element indices in `[0, 2**31 - 1]`; the host
bounds outputs and operand spans at 2**20 elements and every grid axis at
65,535.

Integer operands are refused before dispatch when a divisor is zero or a signed
`MIN / -1` would overflow, because no target defines those results. The
callback checks every buffer against its kernel's contract (types, extents,
spans, directions, overlap and launch geometry), uploads narrow integers as
32-bit words where a target lacks 8- or 16-bit storage, and decodes each
readback exactly, with a guard after each output. `divmod_evidence` recomputes
both outputs of a dispatch from its retained operands: integers exactly, and
floating values with correctly rounded division, MLX's floored `fmod` and
Metal's binary32-quotient binary16 remainder. The evidence check accepts only
zero and normal finite values, so a dispatch that reaches subnormals,
infinities or NaNs fails it rather than passing on an unmodelled flush.

## Batched Unary Operations

The same required native verification step runs:

```bash
python -m demos.integrations.mlx.portable_host.verify_unary_batches \
  --mlx-root mlx-upstream --packages host-packages --output-dir unary-batches
```

Unary dispatch batches physical stored elements, not the logical array size.
It preserves MLX's contiguous strides and donation rules, checks the complete
backing allocation, and requires the translated entry before the first unary
submission. A large broadcast may therefore need fewer dispatches than a dense
array with the same shape. Noncontiguous materialization retains the copy limits.

The 18 cases exercise ten float32/Boolean operations at 65,535, 65,536 and
131,075 elements, nonzero source offsets, column-major transposes, and row/scalar
broadcasts. Each target requires 36 native dispatches, exact CPU/reference/native
bytes, unchanged source storage, preserved output strides, complete uploaded
buffers, output guards, and compiled-artifact receipts. These finite,
exactly-representable inputs complement the existing small-array and nonfinite
unary checks. They do not establish large half/64-bit execution or full-suite parity.

## Affine Quantization

Affine host verification runs in the existing three-platform half-host jobs,
using the same-run base packages and a rebuilt host at the same upstream pin.
Its 1,800-second command deadline is unchanged. Moving it out of the base host
job keeps both jobs within their declared budgets without adding runners or
removing native cases; affine logs and artifacts are retained with half-host evidence.

`HostRuntime(..., mlx_root=...)` routes MLX's `Quantize::eval_gpu` to the pinned
`affine_quantize` and `affine_dequantize` entries in `quantized.metal`. The source
and included kernel headers are unchanged. Group sizes 32, 64 and 128 and bit
widths 2, 3, 4, 5, 6 and 8 have explicit layout contracts for float32, float16 and
bfloat16. The block-scaled route is described below; its remaining integration
requirements are tracked in
[#2140](https://github.com/CrossGL/crosstl/issues/2140). The translator supports
declaration-initializer and resolved by-value function-argument loads from
single-scalar storage wrappers when the source and member layouts match
([#2141](https://github.com/CrossGL/crosstl/issues/2141)).
Indexed, offset and addressed loads retain the original scalar storage, including
targets that widen byte elements. Native regressions check 8-, 16- and 32-bit
unsigned payloads, conversion operators, constructor behavior, overload selection,
single evaluation of indices, unchanged source values and output guards
in the existing three-platform scalar-alias gate. This does not establish
support for general aggregate reinterpretation, reference copies,
volatile accesses or address-space changes. Argument loads require a resolved
matching value parameter; reference parameters and explicit copy constructors
are not replaced with field-wise copies. DirectX helper pointer parameters now
take the byte layout of their callers
([#2145](https://github.com/CrossGL/crosstl/issues/2145)). Helpers reached only
from widened root buffers, including through local aliases, pointer arithmetic,
nested calls and storage-pointer arrays, index one 32-bit carrier per element.
Helpers reached only through byte views cast from 32-bit words keep packed lanes.
A helper that receives both layouts is rejected with a project diagnostic
instead of guessing one layout; per-layout specialization is not implemented.
Writes through reinterpreted views remain unsupported. The scalar-alias gate
executes widened signed and unsigned bytes and packed unsigned bytes through
nested helpers on all three platforms. Concrete scale conversion
is covered by source-level regressions
([#1744](https://github.com/CrossGL/crosstl/issues/1744)); explicit logarithm policies
address the all-zero MX scale boundary
([#2143](https://github.com/CrossGL/crosstl/issues/2143)). A combined
quantize/dequantize kernel does not exercise the packed buffers and encoded scale
storage exposed by the two public operations.

Separate dequantization probes cover all 12 declared float32, float16 and bfloat16
entries, including NVFP4 with a global scale. Generated Metal and OpenGL outputs
match original Metal for these bounded inputs with the explicitly selected
`rne-flush` binary32 multiplication profile. All 12 entries also compile with
DXC; that compiler check is not Windows numerical execution. Default-profile
subnormal differences remain tracked in
[#2114](https://github.com/CrossGL/crosstl/issues/2114). These probes do not establish
a passing upstream quantization suite.

Separate quantizer probes use `binary32_log2_operand_profile = "flush-subnormals"`
and `binary32_log2_accuracy_profile = "portable-finite"`, alongside the explicit
multiplication profile. All 12 declared entries match original Metal on generated
Metal and OpenGL across 60 cases per target: signed values, halfway values, zero
blocks and scale boundaries. Packed payloads and encoded scales are checked
separately, with 32,304 computed values and 2,040 guards per target. These public
source options replace experimental generated-shader edits; source kernels,
input storage and comparisons are unchanged. The required generic logarithm
regressions run in the existing native arithmetic CI step. These bounded probes
are not complete-family execution or Windows numerical evidence. Default
logarithm behavior is unchanged.

Packed outputs are not universally bit-identical between the pinned upstream CPU
and Metal kernels, including FP4 halfway cases, signed-zero encodings and the
scale chosen for all-zero MX blocks. These upstream differences are not
translator defects. Port validation must retain original Metal controls and
check the upstream operation's numerical expectations; CPU byte identity alone
is not a sufficient acceptance contract.

The adaptation registers this multi-output primitive in the synchronous backend,
allocates MLX's packed uint32, scale and bias arrays, and presents the packed data
as the byte buffers expected by the upstream kernels. Complete quantization groups
are dispatched in batches of at most 65,535 workgroups. This is a hardware launch
bound, not a 65,535-element array limit. Partial groups and overlapping output
allocations are rejected before dispatch. All native outputs and guard elements
are validated before any result is copied back to MLX; packing and quantization
arithmetic are performed by the translated kernels, not host substitutes.

Packages are built on demand from the selected original entry. Cache identities
include the pinned revision, source hash, translation implementation, layout and
packaging recipes. OpenGL index assertions cover every permitted batch, including
offsets computed by lanes that do not write. Float16 and bfloat16 carriers preserve
their storage bits; malformed or inexact returned carriers are rejected rather
than rounded by the host.

The existing three-platform host jobs run `verify_quantization.py` through the
actual MLX APIs, retaining native modules, dispatch receipts and guarded readbacks.
Cases include 128-by-512 and 256-by-512 inputs, all supported affine bit widths,
a strided input and a launch crossing the 65,535-workgroup batch boundary. These
deterministic cases do not establish arbitrary numerical parity or a passing full
upstream quantization suite. Noncontiguous inputs still use the existing translated
copy path, with its size and dtype restrictions; in particular, strided bfloat16
copies are not implemented. Other upstream-test prerequisites such as large random
arrays and quantized matrix multiplication remain separate work.

## Affine Quantized Matrix-Vector Products

`QuantizedMatmul::eval_gpu` routes affine products with few input vectors to the
translated kernels of the unchanged `quantized.metal`, as quantized.cpp selects
them. Transposed weights with fewer vectors than `get_qmv_batch_limit` go to the
`dispatch_qmv` kernels: `affine_qmv_quad` when `K` is 64 or 128 and the width is
a power of two, `affine_qmv_fast` when `N` and `K` are aligned, otherwise
`affine_qmv`. Untransposed weights with fewer than four vectors and `K` below
1024 go to `affine_qvm`. Each kernel has its `batch_0` and `batch_1` variants.
The host reports no Apple GPU generation, so it uses the lowest generation's
batch limits and never selects `affine_qmv_wide`. Untransposed vector products
with `K` of 1024 or more go to `affine_qvm_split_k` in 8 partitions (32 above
`K` of 8192), whose partial products a translated column sum adds:
`col_reduce_small` below 32 partitions, `col_reduce_looped` otherwise.

Products with at least the vector limit's rows go to `qmm_layout.py`'s
matrix-matrix kernels. Single matrices with transposed weights take
`qmm_splitk`: `affine_qmm_t_splitk` writes partial products over whole
`BK`-wide, whole-group partitions of `K`, chosen to target about 512
threadgroups, and the same column sums add them; when `K` admits no split they
take `affine_qmm_t`, as batched transposed products do. Untransposed products
take `affine_qmm_n`. The host has no NAX unit, so it never selects `qmm_nax`.
Split-K products of float16 and bfloat16 write partial products in the element
type, and MLX sums them in it: the host routes them to the unchanged
`col_reduce_small_1_reduce_sum{float16,bfloat16}` entries below 32 partitions,
`col_reduce_looped_1_32_32_reduce_sum...` from 32, and above 256 partitions of
fewer than 32,768 outputs to `col_reduce_2pass_1_32_32_reduce_sum...` followed
by the looped pass over its 32 partials, as reduce.cpp does. Each of them
rounds every addition. Column packages hold these 16-bit entries only when
asked for them by name (`reduction_packages.PARTITION_SUM_ENTRIES`). Partition
sums bind up to 2^28 - 1 partial products, the bound of the products themselves.

`matmul_layout.py` rejects a launch unless its entry, grid and workgroup are the
ones MLX's routing would choose for the bound shapes, strides and buffer extents.
Every batch offset must name an element of its buffer and every buffer holds at
most 2^28 - 1 elements, which the OpenGL translation asserts for its 32-bit batch
offsets. `qmv_quad` reads one scale and bias per row of its last tile, past `N`;
the host pads those buffers with poison elements and the output with guards, and
copies the output back only after validating them. `batch_0` kernels reflect the
batch parameters MLX does not bind; the host supplies inert unbatched values.

The three-platform `MLX quantized host` jobs translate every routed entry ahead
of execution with `python -m demos.integrations.mlx.portable_host.matmul_packages
--verification --jobs 3`, then run `verify_quantized_matmul.py` through
`mx.quantized_matmul`. Its twenty-three cases cover every route and batch variant,
all three element types and every width. The transposed cases include a partial
`qmv_quad` tile, broadcast weights, a strided activation batch and two batch
axes; the untransposed ones a `K` tail, a `K` shorter than one 32-row block, a
half-empty last tile, broadcast and shared weights and two batch axes. Inputs
lie on dyadic grids that keep every float32 partial sum exact, so each output
must match an exact float64 reference bit for bit, and float32 results must also
match MLX's CPU backend. Four more `qvm_split_k` cases sum float16 and bfloat16
partitions in the small and the looped pass. The same jobs translate the matrix-matrix entries with
`qmm_packages --verification` and run `verify_quantized_qmm.py`: twenty-two cases
over `affine_qmm_t`, `affine_qmm_t_splitk` with small, looped and two-pass
partition sums,
and `affine_qmm_n`, in all three element types, with stacked, broadcast and
shared weights, folded rows, partial row and column tiles and `K` tails. Split-K
cases cover float16 and bfloat16 partition sums, sixteen partitions that each
threadgroup row adds two of, and 65,536 and 131,072 partial products (`test_qmm`'s
float16 shape), and the looped and two-pass sums of 64 to 512 partitions in
every element type. A 16-bit product rounds each partition and each addition.
Below 32 partitions its reference repeats those roundings in the small column
sum's order, and they change hundreds of outputs against a single rounding of
the exact product. The looped and two-pass sums add through simd reductions,
whose order Metal leaves unspecified, so their 16-bit cases use unit scales and
at most one nonzero activation per partition: every partial product and every
sum of them is an integer the element type holds, and each output must equal
the exact product bit for bit in any order. The unchanged upstream `test_qmv` and `test_qvm` also
need the floating-point `Matmul` route below for their reference products;
`test_qmv`'s unbatched references are matrix-matrix products, which take the
steel GEMM or split-K route.

`verify_upstream_qvm.py`, `verify_upstream_qmv.py`,
`verify_upstream_qvm_splitk.py` (`test_qvm_splitk` and
`test_qvm_splitk_multi_row`), `verify_upstream_qmm_shapes.py`
(`test_qmm_shapes`, `test_small_matrix`, `test_non_multiples` and
`test_throw`, whose one unchecked product takes `affine_qmv_fast`) and
`verify_upstream_qmm.py` (`test_qmm`) run those
tests unchanged, first on MLX's CPU and then through the translated host, with
the shared steps in `upstream_runner.py`. Each runner translates every quantized and
floating-point product entry the test reaches before the native worker starts,
and packages the whole-array maximum of its error bound on demand. The native
trace must hold exactly the affine, `gemv` and steel dispatches that
quantized.cpp and matmul.cpp would make for every subtest: `test_qmv` reaches
60 `affine_qmv_quad`, `affine_qmv_fast` and `affine_qmv` entries, both `gemv`
tiles, two fused GEMM entries and the split-K product with its accumulation;
the quantized-shape tests reach `affine_qmv`, `affine_qvm`, `affine_qmm_n` and
`affine_qmm_t_splitk` with its small partition sum. `test_qmm` computes in
float16 on the GPU device, so its native run is float16 throughout while the
CPU run is float32: 288 products over three group sizes and three widths reach
`affine_qmv`, `affine_qmv_fast`, `affine_qmv_quad`, `affine_qmm_n`,
`affine_qmm_t` and `affine_qmm_t_splitk` with the float16 small partition sum,
against float16 steel GEMM and split-K references. Its runner also translates
the `half` and `half-arithmetic` families for the casts and the error bound's
difference, absolute value and comparison, and packages the float16 maximum in
its own reduction directory. `test_qmm_large_dims` (outputs over 2^15 rows or
columns) is not run yet. The `MLX upstream` jobs run each test on Metal,
Direct3D 12 (WARP) and OpenGL (Mesa) in the heavy tier.

The block-scaled runners follow the same steps. `verify_upstream_fp_qmv.py`
and `verify_upstream_fp_qvm.py` run `test_fp_qmv` and `test_fp_qvm` in
float32. `verify_upstream_fp_qmv_large_output.py` runs
`test_fp_qmv_large_output` in float16, bfloat16 and float32, as the GPU device
does. Each type reaches `nvfp4_qmv_fast` of that type, alone and batched, for
4,096 outputs, against a `gemv` reference in the same type. Its `allclose` is a
translated chain of 16-bit and float32 element-wise operations ending in an
`and` reduction.

`verify_upstream_qmv_small_non_multiples.py` runs
`test_qmv_small_non_multiples`, which mixes both families. One to four rows of
32 values multiply 3, 5, 7 or 10 output rows quantized as 4-bit and 6-bit
affine groups and as nvfp4. No width is a multiple of 8, so every affine product
reaches `affine_qmv`, one nvfp4 row `nvfp4_qmv` and two or four rows
`nvfp4_qmv_wide`. The references take `gemv` for one row and
`steel_gemm_fused` otherwise.

`verify_upstream_qmm_vjp.py` runs `test_qmm_vjp` and `test_qmm_jvp`. Each
differentiates an 8-bit, group-64 product of a `(2, 64, K)` input with respect
to that input, for transposed and untransposed weights, and compares the
derivative with the same product written out through `allclose`. The
derivatives are quantized products of the cotangent or tangent with the
weights: 128 folded rows that reach `affine_qmm_n` or `affine_qmm_t_splitk`
with its small float32 partition sum. Neither test evaluates the primal
output, so the trace must hold exactly two launches of each product.

```bash
python -m demos.integrations.mlx.portable_host.verify_upstream_qmv \
  --mlx-root mlx-upstream --packages host-packages --output-dir upstream-qmv
```

## Floating-Point Matrix-Vector Products

`Matmul::eval_gpu` routes float32, float16 and bfloat16 products with one
output row or column to the translated `gemv` and `gemv_t` kernels of the
unchanged `gemv.metal`, as matmul.cpp selects them: `check_transpose` on both
operands, `collapse_batches`, the fold of a contiguous batch into `M`, then
`gemv_axbpy`'s kernel and tile shape. The host reports no Apple GPU generation,
so it never selects `gemv_wide`, and `Matmul` never computes `axpby`. The host
copies the operands MLX would copy and, in addition, views with negative
strides. Empty inputs produce zeros through the translated copy. Single dot
products, which MLX sends to `dot_product` and an all-reduce, fail with an
explicit error. Matrix-matrix products take the steel GEMM route below.

`gemv_layout.py` rejects a launch unless its entry, grid and workgroup are the
ones MLX's routing would choose for the bound sizes, leading dimension and batch
strides, and every batch's matrix and vector lie inside their buffers of at most
2^28 - 1 elements, which bounds the offsets the OpenGL translation asserts. The
kernels reflect the `axpby` bindings `gemv_axbpy` leaves unbound; the host binds
inert values for them. The `MLX quantized host` jobs translate every routed
entry with `python -m demos.integrations.mlx.portable_host.gemv_packages
--verification --jobs 3`, then run `verify_matmul.py` through `mx.matmul`. Its
cases reach ten of the twelve instantiated tiles in all three element types,
both batch variants, transposed and strided operands, broadcast and
`as_strided` batches, and the batch fold. Each output must match an exact
float64 reference bit for bit, float32 outputs must also match MLX's CPU
backend, and a single dot product must fail with the host's error without
dispatching.

## Floating-Point Matrix-Matrix Products

Products with more than one output row and column follow matmul.cpp's
`steel_matmul` after the same operand preparation. The host reports no Apple
GPU architecture and no NAX unit, so `steel_matmul_axpby` compares the product
against the split-K case with its 1,024-tile threshold and otherwise selects
`steel_gemm_fused` with `GEMM_TPARAM_MACRO`'s medium-device tile, 64x64x16
with 2x2 simdgroups. The host names each entry by matmul.cpp's hash name, so
the translation freezes the kernel's `has_batch` and `align_M`/`N`/`K`
function constants; `Matmul` never reads `C`, so `use_out_source` and
`do_axpby` stay false. Split-K products take the route in the next section.

`gemm_layout.py` rejects a launch unless its entry, constants, `GEMMParams`,
grid and workgroup are what matmul.cpp would bind for the sizes and strides,
and every batch's `A` and `B` tiles lie inside their buffers. `gemm_packages`
translates entries with the options and range assertions of the current-tree
GEMM gate (`test_current_gemm_family.py`), and the host packs `GEMMParams`
into the reflected record layout. The `MLX quantized host` jobs translate the
routed entries with `python -m demos.integrations.mlx.portable_host.gemm_packages
--verification --jobs 3` before `verify_matmul.py`, whose GEMM cases cover
every transposition and element type, partial and whole tiles, several tiles
per axis, batches with and without `has_batch`, a batch folded into `M` and a
copied operand, with the same exact references.

A single product whose `M x N` output has at most 1,024 tiles of 16, with
`K / 16 >= 8` and `K >= max(M, N)`, follows `steel_gemm_splitk_axpby`: the
host allocates the float32 `C_split` temporary of matmul.cpp's partition
count, runs the `gemm_splitk` entry for its tile (16 or 32 per axis) and
alignment, then `gemm_splitk_accum` over the partitions. matmul.cpp launches
the accumulation with `dispatch_threads` over exactly `N x M` threads; the
kernel reads only `thread_position_in_grid` and has no bounds check, so the
host covers the same threads with one-thread workgroups. `gemm_splitk_layout.py`
checks `GEMMSpiltKParams`, the grid and each buffer extent against
matmul.cpp's plan, and `gemm_splitk_packages` translates both kernels with the
options and range assertions of `test_current_gemm_splitk.py`. The split-K
cases of `verify_matmul.py` cover every transposition, element type and tile,
aligned and unaligned `M x N` and `K`, 4 to 32 partitions, partitions without
a whole `K` tile, a batch folded into `M` and a copied operand; the trace must
show each accumulation reading exactly the partitions its product wrote.

## Block Quantization

The host adapter routes MXFP4, MXFP8 and NVFP4 quantization and dequantization
to unchanged `fp_quantized.metal` specializations. Payloads remain packed uint32
MLX arrays; scales remain encoded uint8 arrays. No affine bias is introduced.
NVFP4 accepts an optional scalar float32 global scale. Specializations without
a global scale retain an inert binding for the source template's unused argument.
The adapter never substitutes that binding for an active scale.

Float32, float16 and bfloat16 storage have explicit contracts. Each dispatch
checks complete groups, exact byte counts, buffer directions, alignment, output
overlaps and geometry before native execution. Quantization uses 32 lanes for MX
and 16 for NVFP4; dequantization uses one lane per packed byte. Large operations
are batched at 65,535 groups with independently advanced payload and scale pointers.
The packaging recipe selects the explicit multiplication and logarithm profiles
described above, and records the unchanged source and translation identity.

```sh
python -m demos.integrations.mlx.portable_host.verify_block_quantization \
  --mlx-root mlx-upstream --packages host-packages --output-dir block-evidence
```

This verifier calls the two public MLX APIs separately, with explicit native
streams. It covers all 24 declared entries through 25 workloads: three dtypes,
four format/global-scale combinations, contiguous and offset views, plus a
65,536-group NVFP4 case. Independent exact-format references check packed bytes,
encoded scales and reconstructed values, including adjacent blocks with different
magnitudes and an all-zero block. Every output has 17 trailing guards; the host
commits outputs only after all readbacks validate. Full inputs, readbacks,
module identities and dispatch traces are retained.
Each distinct packaged entry is also checked by the platform toolchain before
execution: Metal with warnings as errors and fast math disabled, DXC shader model
6.6 with warnings as errors, or glslang followed by SPIR-V validation. Compiler
commands, output and compiled-module hashes are retained separately from native
numerical results.

The same verifier runs in the existing Metal, OpenGL and DirectX half-host CI
jobs, without adding operating-system runners. This bounded API proof does not
establish a passing upstream suite. General-gather batching now permits the large
indexed inputs in these tests. With large slice assignments enabled, all four
unchanged block-format tests can execute bfloat16 casts, arithmetic and copies
with the optional bfloat family. Local generated Metal and OpenGL execution now
passes all four unchanged tests: MXFP4 and MXFP8 quantize/dequantize, MXFP8 scale
error, and NVFP4 quantize/dequantize with its global-scale graph. These runs load
the bfloat extrema family and Boolean/float32 reduction variants at widths 32,
128 and 256; separate CPU controls pass the same tests. Windows execution and
full-suite parity remain required under
[#2147](https://github.com/CrossGL/crosstl/issues/2147). Larger copy
launches remain separate work under [#2148](https://github.com/CrossGL/crosstl/issues/2148). Those
tests are not rewritten, skipped or given wider tolerances. Strided bfloat16 and
encoded-scale copies remain subject to the existing copy-path restrictions.

The unchanged upstream tests also have a separate required CI step in those same
three jobs. It reuses the existing host build, base, random and bfloat packages;
only the six Boolean/float32 assertion-reduction variants are added. Each test
runs once in a CPU control process and once on the generated backend. The runner
rejects skips and expected failures, checks per-test dispatch accounting, and
audits retained modules against the translated package identities. Source hashes
are checked before and after execution. No upstream assertions, tolerances or
test bodies are modified.

```sh
python -m demos.integrations.mlx.portable_host.reduction_packages \
  --mlx-root mlx-upstream --target opengl --width 32 --width 128 --width 256 \
  --entry all_reduce_andbool_ --entry all_reduce_maxfloat32 \
  --output-dir block-assertion-reductions
python -m demos.integrations.mlx.portable_host.verify_upstream_blocks \
  --mlx-root mlx-upstream --packages host-packages --random random-packages \
  --bfloat bfloat-packages --reductions bfloat-reductions \
  --reductions block-assertion-reductions --output-dir upstream-blocks
```

The same commands accept packages generated for Metal or DirectX. This proves
the four named block-format tests, not the complete upstream MLX suite.
Dense and quantized matrix multiplication still reject before native dispatch;
their host integration is tracked separately in
[#2151](https://github.com/CrossGL/crosstl/issues/2151).
The tiled GEMM translation work uses the unchanged source kernel, not repeated
vector operations as a substitute. Recursive template-member lowering now
preserves the surrounding struct aliases and overload context. Its required
native regression covers explicit and inferred template calls, call operators,
repeated mutations, integer wraparound and output guards; Metal also runs the
original source as a control. These helper checks do not establish matrix
execution. The selected `steel_gemm_fused` entry still fails on explicit
`auto*` pointer deduction during source overload resolution, tracked in
[#2152](https://github.com/CrossGL/crosstl/issues/2152).
Project materialization also retains non-type argument grouping on every target.
Its native regression uses dependent aliases and default arguments with
conditional, additive and shift expressions. Automatically
discovered and explicitly selected entries must preserve both specializations,
integer wraparound and output guards; the Metal job compares the original
template kernel. This addresses the arithmetic corruption in
[#2155](https://github.com/CrossGL/crosstl/issues/2155), not full GEMM execution.
Template argument scanning distinguishes left shifts from nested template
delimiters, including expressions inside parentheses, brackets and braces
([#2156](https://github.com/CrossGL/crosstl/issues/2156)).

## General Gather

The adapter implements MLX's `Gather::eval_gpu` using the unchanged pinned JIT
wrapper and `indexing/gather.h`. `HostRuntime(..., mlx_root=...)` enables on-demand
translation and packaging for the selected target. Cache identity includes the
upstream revision, translator implementation, packaging recipe, entry point and
validated index bound. General-gather packages use the storage upper bound of 268,435,422,
so workloads with different array sizes reuse one translation of an entry.
Every dispatch still validates its actual allocations, indices and launch before
loading the package; a cache hit does not bypass those checks. OpenGL packages
carry the covering range assertion where required, while Metal and DirectX do
not need that narrowing assertion. Changed kernels or JIT definitions are
rejected, including when loading an existing cached package.

Supported source storage is float32, int32, uint32, int64, uint64 and bool;
indices may be signed or unsigned 32-bit or 64-bit integers. The current contract
allows one through ten index arrays, rank at most 64, and at most 268,435,423 elements
in each input storage span and output. This conservative element bound leaves room
for guards even with eight-byte storage. Each dispatch axis remains limited to
65,535 workgroups. Larger index grids are split into exact rectangular slabs;
index pointers and output views advance while source coordinates, index strides
and logical output order are preserved. The source slice per index is still
limited to 65,535 values. Scalar indices, multiple indices,
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
python -m demos.integrations.mlx.portable_host.verify_gather --capacity \
  --mlx-root mlx-upstream --packages host-packages \
  --integer64 integer64-packages --reductions equality-reductions \
  --output-dir gather-evidence
```

The equality companion is `all_reduce_andbool_` at width 32. The verifier runs
42 base workloads in separate CPU and native processes: six source storage types
times dense, transposed, strided, broadcast, reversed, scalar-index and
multiple-index layouts. `--capacity`, required by CI, adds 13 workloads covering
large offset allocations, strided and broadcast index arrays, multidimensional
batch boundaries, and small source tables with 65,534, 65,535 and 131,071 indices.
The audit checks each batch's uploads and native readback against the complete
logical workload, including output order, input storage and trailing guards.
It also runs unchanged upstream
`test_ops.TestOps.test_take` on each path, without skips. Retained uploads are
independently reconstructed into reference views and compared to native
readbacks and the MLX result. Missing dispatches, changed modules, corrupt guards
and incomplete upstream results fail verification. Three-OS CI requires this
check after building MLX with its original GPU backends disabled.

`GatherAxis`, used by `take_along_axis`, dispatches the unchanged
`indexing/gather_axis.h` specialization. Its source and index contiguity flags,
removed-axis shape and strides, axis extent and three-dimensional grid follow
the pinned upstream host implementation. It uses the same six source storage
types and four index types as general Gather, but retains its 65,535-element
allocation bound. Negative
source or index storage strides are materialized through translated copies;
positive and zero strides retain their views. Index values, allocation spans
and contiguity claims are checked before native submission.

Replace `--capacity` with `--axis` to run 60 axis-gather workloads and unchanged
upstream `test_take_along_axis` in separate CPU and native processes. Coverage
includes each axis, flattened input, noncontiguous source/index combinations,
broadcasts, negative indices, reversed views, exact floating-point payloads and
64-bit values beyond binary64's exact integer range. The required three-OS gate
retains the same compiled-module, upload, readback and guard evidence as general
Gather. No extra OpenGL index-range assertion is needed for this kernel.

Zero-index general-gather specializations, additional storage types, source slices
larger than one launch axis, allocations above the documented bound and full
indexing/autodiff coverage are not claimed. Negative-stride materialization remains
subject to the separate translated copy limits.
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
replacement, addition, multiplication, minimum and maximum. It translates the exact pinned JIT
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

The verifier runs 146 workloads and unchanged upstream
`test_array.TestArray.test_setitem_with_list` in separate CPU and generated-backend
processes. Retained evidence includes initial output storage, update and index
uploads, metadata, launch geometry, native compilation, readbacks and 32 trailing
guard elements. The audit reconstructs each indexed update from those uploads
and reconciles it with the public MLX result. Duplicate replacements use identical
values so the expected result does not assume a particular thread ordering.
The original 112 workloads are retained. An additional 32 signed and unsigned
product cases cover the same views and all five work-per-thread settings, with
zero and negative factors, repeated destinations, source/update aliases and
partial chunks. Factors keep every intermediate product representable regardless
of update order. The independent audit hashes raw storage words so negative
signed results retain their exact bit patterns.
Two more cases exercise the 65,535-element storage limit, including a write to
index 65,534 and its negative-index equivalent. They check every unchanged
interior element as well as the endpoints and guard storage.
A separate three-OS CI job requires this proof without extending the existing
indexing job's execution budget.

Metal, DirectX and OpenGL also route float32 indexed assignment and `array.at`
updates through the same unchanged scatter kernels. Initial output storage,
updates and readbacks preserve the original binary32 words. Metal and DirectX
retain float32 bindings with explicit binary32 encoding. OpenGL uses a uint32
atomic output allocation without float encoding; its non-atomic update buffer
remains float32 with binary32 encoding. The adapter validates both roles and the
reflected atomic member before dispatch, and copies raw readback words into the
MLX result without numeric conversion.

```sh
python -m demos.integrations.mlx.portable_host.verify_scatter \
  --mlx-root mlx-upstream --packages host-packages \
  --integer64 integer64-packages --float32 --output-dir float-scatter-evidence
```

This separate profile runs 81 CPU/native workloads across the five policies,
all existing view layouts and work-per-thread settings. Fractional arithmetic
inputs have exactly representable results regardless of update order; a
replacement case preserves signed zero, subnormals, infinities and NaN payloads.
It also runs the unchanged list-index assignment test above, which is integer
coverage, not an additional upstream float test. Required Linux, Windows and
Xcode 27.1 jobs retain separate artifacts; the original three-OS integer profile remains
unchanged. These bounded workloads do not establish arbitrary floating reduction
order, nonfinite arithmetic or full upstream indexing-suite parity.

Packed storage, zero-index specializations, larger
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

Float compare-exchange has a required Metal/DirectX gate covering eight storage
and call shapes. It checks Boolean success, expected-value writeback, single
evaluation, signed zero, subnormals, infinities and NaN payloads through raw
binary32 words. DirectX uses bitwise compare-exchange and integer equality of
the observed and expected payloads. Five additional controls execute the pinned
MLX compare, multiplication, minimum and maximum helpers, including eight
contending multiply-by-two updates. Multiplication inputs are finite values,
signed zeros and infinities; these controls do not establish arbitrary NaN
arithmetic semantics. The upstream header and tests remain unchanged.

The strict Metal compare-exchange gate uses Xcode 27.1. Compiler 32023.883 from
the macOS 26 image lowers the float success result to `fcmp ueq`, reporting
success for some bitwise mismatches involving signed zero, subnormals or NaNs.
Both original and translated sources reproduce this compiler defect. Compiler
32023.918 uses integer equality for the success result and passes the unchanged
controls. CI retains the compiler versions and all numerical assertions; the
macOS 26 job still exercises float loads/stores and integer atomic operations.
This does not establish correct float compare-exchange on older Metal compilers.

The float scatter gate executes all five policies from the pinned JIT wrapper:
replacement, sum, product, minimum and maximum. Sixteen cases cover fractional
values, strided updates and indices, duplicate and negative indices, partial
four-update chunks, and 65 concurrent updates to one destination. Arithmetic
inputs have order-independent, exactly representable results; replacement also
checks signed zero, subnormals, infinities and NaN payloads as raw binary32 words.
Every case retains 32 output guard words. Required macOS and Windows checks run
the generated kernels; macOS also executes the original pinned wrapper. This is
kernel execution coverage; the separate host profile above verifies MLX dispatch.

OpenGL lowers binary32 atomic allocations to unsigned-word storage and keeps
logical float arithmetic behind bit-preserving loads and stores. Struct-backed
allocations use separate physical types; private struct values retain their
original types. Atomic load, store, exchange, addition and compare-exchange use
core integer atomics, without vendor float-atomic extensions. Compare-exchange
uses bitwise equality and preserves expected-value writeback.

The Linux gate runs the reduced memory/compare cases, ordinary storage updates,
unchanged MLX atomic helpers and the same sixteen scatter cases. Reflection
describes the physical uint32 storage. Test uploads and readbacks carry those
words unchanged; non-atomic float input buffers remain float32. Configured index
and shared-access bounds are tied to each fixture's actual dispatch geometry.
Mixed interface blocks, narrow-float or matrix-backed atomic allocations,
whole-array transfers, mixed-precision updates and escaping mutable logical
references retain structured diagnostics.

OpenGL float scatter routing through the MLX host adapter is not enabled yet.
These package-level numerical controls do not establish complete upstream-suite
parity or arbitrary floating-point reduction-order equivalence.

Generated Metal shared float loads, stores and compare-exchange use atomic uint
storage with bitcasts, so they also compile under Metal 4.0. Native shared float
overloads require Metal 4.1. On the older CI toolchain, the owned reduced fixtures
use reference-only uint-atomic overloads for their original-source control;
the evidence retains that header and the compile flags. All operations and raw
word expectations remain enabled. No compatibility header is injected into
translated input, generated output or the pinned upstream MLX helpers.

The multiplication controls also exercise transitive constrained helper
specialization and compile-time branch selection. Separate native cases check
namespace ownership, explicit specializations, reused helpers, mixed runtime
and compile-time branches, and discarded type-dependent expressions. Dependent
calls with proven class arguments also retain their associated source namespaces
after specialization, so a later helper declaration can be found without making
unrelated namespaces or later primitive-only overloads visible. The scatter gate
includes unchanged `Prod<int>` policies with duplicate indices, negative and zero
factors, strided updates and partial chunks. The host integration above uses
these same policies for integer product updates. Packed storage and full
upstream-suite parity remain unimplemented.

In particular, the full `test_array_at` method also needs random generation,
floating-point atomics and floating-point product updates; passing the list-index assignment
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

The required three-OS CI step runs 84 workloads in separate CPU and native
processes. They cover both operations and storage types, all four index types,
dense and transposed sources, strided and broadcast views, reversed views,
negative indices, flattened inputs, aliasing updates and empty updates. Duplicate
replacement indices carry identical values so the expected result does not
depend on thread ordering. Additive cases retain duplicate destinations to
exercise atomic accumulation.

The original 56 workloads are retained unchanged. Another 28 int32 workloads
exercise negative source values, updates and results across both operations and
every layout, including aliases and empty updates. Native readbacks retain signed
numeric values; the verifier checks their declared storage range before deriving
bit patterns for comparison. Unsigned substitutions, floating-point values,
Booleans and incomplete readbacks are rejected.

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
Floating-point atomic load and compare-exchange remain separate contracts;
this does not establish the full atomic scope of
[#2051](https://github.com/CrossGL/crosstl/issues/2051).

Two required native general-scatter cases use the unchanged pinned JIT wrapper and
`indexing/scatter.h`: one strided index/update input with `NWORK=1`, and two
contiguous index inputs with `NWORK=4`, including a partial final chunk. Both use
integer additive updates, duplicate and negative indices, and guarded output
storage. These are kernel-execution checks, separate from the integer
`Scatter::eval_gpu` host integration described above. Floating-point scatter and
full indexing-suite parity remain outside the current proof.

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
  Eight binary primitive hooks select 20 unchanged `vv_` specializations. They
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
It also selects `flush-subnormals` for binary32 square root and reciprocal
square root. These source-device policies are retained in translation and
package provenance; they do not change CrossTL's default profiles for other
projects. Subnormal operands retain their sign when flushed, so square root
produces signed zero and reciprocal square root produces signed infinity.
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
arrays retain the two source passes: logical inputs up to 64 MiB use 128 partial
rows followed by a 32-thread final reduction; larger inputs use 4,096 partial
rows followed by a 1,024-thread final reduction. The threshold uses MLX's logical
item size, including one byte for Boolean values, independently of widened target
storage. Whole-array inputs must fit their backing allocation and the signed
32-bit buffer-index range; native device buffer limits also apply. Row
reductions retain their 65,535-element bound; column reductions are bounded as
described under Column Reductions. Intermediate values are staged synchronously, not
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

The default workload includes 28 additional cases at 65,536 and 131,075 elements
across all 14 entries. These use nonzero allocation offsets and a distinct final
element to exercise full rows and tails. Both passes' numerical readbacks, source
size metadata and output guards are checked, not only the final scalar. Invalid
large backing views must fail before any native dispatch. Launch-plan tests cover
both sides of the 64 MiB threshold for every logical dtype; those tests do not by
themselves establish native execution of such large allocations. No floating-point
tolerance or upstream test is changed by the adapter.

The same three whole-array CI jobs require `--upstream-quantization`. This runs
the pinned `test_quantized.TestQuantized.test_quantize_dequantize` unchanged in
separate CPU and native processes. The native trace must contain all 36 affine
specializations, both random and zero-valued input families, the upstream strided
slice comparison and all 36 large Boolean error assertions. The test's explicit
CPU comparison remains intact; the adapter does not redirect unsupported work to
a CPU fallback. Random packages are translated from the same pin, and source
hashes, worker logs, commands and native traces are retained even on failure.
No test is skipped or assigned a more permissive tolerance. This is one upstream
test, not evidence that the entire quantization or MLX suite passes.

```sh
python -m demos.integrations.mlx.portable_host.verify_upstream_quantization --mlx-root mlx-upstream --packages host-packages --reductions reduction-packages --output-dir upstream-quantization-evidence
```

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

For checked per-dispatch translation, build the catalog with `--on-demand`:

```sh
python -m demos.integrations.mlx.portable_host.reduction_packages --mlx-root mlx-upstream --target opengl --family row --on-demand --output-dir checked-row-packages
python -m demos.integrations.mlx.portable_host.verify_rows --mlx-root mlx-upstream --packages host-packages --reductions checked-row-packages --output-dir checked-row-evidence
```

This path validates the actual uploaded ranks, shapes, strides, source allocation
span and launch geometry before translation. Each scalar and array element becomes
an exact metadata requirement in the package. No generated-index assertion is
used. The cache identity includes those values, source and translator hashes, and
the layout validator; changing a stride cannot reuse another view's package.
Verification loads the existing package without rebuilding it and checks the
recorded uploads and artifact identity. `HostRuntime` requires `mlx_root` when
loading these catalogs; `verify_rows` supplies it from `--mlx-root`.

The existing native checked-reductions CI batch exercises this cache for the
simple-row kernel, including original/generated Metal comparison and guarded
native execution on each target. The separate all-width row collection below
still uses static packages. Its compilation coverage must not be attributed to
on-demand packages, nor does either path establish full MLX backend coverage.

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
columns take `col_reduce_small`; the separate long-column algorithm still reports
an explicit error.

The adapter preserves reduction strides, broadcast strides, allocation offsets
and output geometry. Ranks, metadata array lengths, source spans and logical
sizes are checked before dispatch. Two-pass plans write 32 intermediate rows;
the second native dispatch consumes those actual results without host reduction
arithmetic. Inputs, outputs and source spans hold at most 2^28 - 1 elements, far
below the `INT32_MAX` at which MLX selects 64-bit index kernels; every shape axis
and stride stays below 65,536 and every launch axis at most 65,535 workgroups.
Row plans, unlike column plans, retain the 65,535-element bound. The workloads
include small, looped and two-pass plans over 76,800 to 131,072 elements.

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

## Half-Precision Storage

The `half` package family contains the unchanged pinned
`ggn2_dynamic_copyfloat16float16`, `v_copyfloat16float32` and
`v_copyfloat32float16` entries. `HostRuntime(..., half=...)` enables their
dispatch through MLX's existing contiguous-copy and cast primitives. Positive,
negative and broadcast strides use the same allocation validation as other
copy types, with two-byte logical element sizes.

Metal and DirectX transfer raw binary16 words. OpenGL uses exact binary32
carriers for the same logical words, preserving signed zero, subnormals,
infinities and NaN payloads during copies. Transfer code only encodes and
decodes storage; numeric casts run in the translated kernel. An OpenGL
readback that is not an exact binary16 representation is rejected rather than
rounded by the host.

Generated HLSL half buffers use unsigned 16-bit physical storage with an explicit
binary16 codec in the package descriptor. The callback and evidence audit validate
that contract through the translator's shared storage validator while retaining the logical
`float16` payload, two-byte stride and exact readback checks. Missing or
incompatible codecs are rejected; the demo does not reinterpret arbitrary integer
buffers as half values or repair results after execution. Windows native checks
remain required to establish correctness of the generated storage operations.
The evidence audit also checks the physical element size, encoded storage
alignment and offset; a logical type annotation alone cannot authorize a different
buffer layout.

```sh
python -m demos.integrations.mlx.portable_host.packages \
  --mlx-root mlx-upstream --target metal --family half --output-dir half-packages
python -m demos.integrations.mlx.portable_host.verify_half \
  --mlx-root mlx-upstream --packages host-packages --half half-packages \
  --output-dir half-evidence
```

The required three-platform job runs 28 public-API workloads in separate CPU
and translated-backend processes. Four reversed-copy batches cover all 65,536
binary16 words; other cases exercise casts, empty arrays, scalars, tails,
matrices, transposes, negative strides and broadcasts. Exact logical results,
native readbacks, trailing guards, dispatch identities and retained compiler
artifacts are checked. A separate process rejects missing half packages before
dispatch. No upstream kernel or test is modified. This family covers storage
and casts; arithmetic uses the separate family below. Neither family establishes
bfloat16 operations or full upstream-suite parity.

### Half Arithmetic

The `half-arithmetic` family adds the pinned half-precision Add, Subtract,
Multiply, Divide, Minimum, Maximum, Abs and comparison entries, including
NaN-aware equality. `HostRuntime(..., half=..., half_arithmetic=...)` uses
translated copies to materialize noncontiguous operands and sends the logical
half values to these kernels. Results must already be representable as
binary16 when read back; host code does not round or repair arithmetic results.

```sh
python -m demos.integrations.mlx.portable_host.packages \
  --mlx-root mlx-upstream --target metal --family half-arithmetic \
  --output-dir half-arithmetic-packages
python -m demos.integrations.mlx.portable_host.random_packages \
  --mlx-root mlx-upstream --target metal --output-dir random-packages
python -m demos.integrations.mlx.portable_host.verify_half_arithmetic \
  --mlx-root mlx-upstream --packages host-packages --half half-packages \
  --arithmetic half-arithmetic-packages --random random-packages \
  --output-dir half-arithmetic-evidence
```

The three-platform gate requires 120 public-API workloads and the unchanged
upstream `test_random.TestRandom.test_broadcastable_scale_loc` in separate CPU
and native processes. Workloads cover all 14 entries, dense and strided layouts,
empty inputs, rounding boundaries, subnormals, overflow and comparisons involving
NaNs and infinities. It checks exact workload results, dispatch accounting,
native guards, source identity and retained compiler artifacts. Missing
arithmetic packages must fail before dispatch. These checks do not establish
all half transcendental operations, arbitrary allocation sizes or the complete
random test module.

## Bfloat16 Host Operations

The optional `bfloat` package family contains 17 pinned source entries:
float32/bfloat16 casts, strided copies, absolute value, six arithmetic operations
and seven comparisons. `HostRuntime(..., bfloat=...)` preserves bfloat16 output
allocations; it does not promote the computation graph or use CPU fallback.
Full-array fills and replacement slice assignments use the same translated copy
path. Bfloat minimum and maximum reductions have a separate optional package
family described below. Reduction updates, sums and products remain unsupported.
Typed storage views remain separate work under
[#2150](https://github.com/CrossGL/crosstl/issues/2150); they must not be
substituted with numeric casts.

Metal uses two-byte bfloat storage. DirectX uses two-byte unsigned integer
carriers, while OpenGL carries each exact bfloat word in the upper 16 bits of a
float32 storage word. Transport performs no numerical rounding. Mismatched
reflection, inexact carriers, invalid readbacks and damaged guards are rejected
before committing outputs. Non-Metal subnormal comparisons retain the explicit
diagnostic tracked in [#2000](https://github.com/CrossGL/crosstl/issues/2000).

```sh
python -m demos.integrations.mlx.portable_host.packages \
  --mlx-root mlx-upstream --target metal --family bfloat --output-dir bfloat-packages
python -m demos.integrations.mlx.portable_host.verify_bfloat \
  --mlx-root mlx-upstream --packages host-packages --bfloat bfloat-packages \
  --output-dir bfloat-evidence
```

The required three-platform proof checks 145 public-API workloads against
separate CPU execution and an independent bit-level rounding reference, with
204 native dispatches per target. It covers empty/scalar arrays, strided,
transposed and broadcast inputs, nearest-even conversion ties and 65,537-element
batch boundaries. Every entry must compile strictly; every dispatch retains
artifact identity, native module hashes and output guards. Missing packages
must fail before dispatch. Unit tests additionally check lossless transport of
all 65,536 bfloat storage words; that transport check is not an exhaustive
native arithmetic proof.

Local Metal and OpenGL runs pass this proof. All 17 HLSL entries compile with
DXC; Windows execution remains a separate required CI result. These checks do
not establish exceptional-value arithmetic, all bfloat operators, arbitrary
copy grids or a passing complete upstream suite. The existing half-host jobs
run the native checks without additional runners; shared Python contract tests
run on Linux only. The job deadline accounts for the bounded commands and setup.

### 16-bit Float Extrema

The reduction builder's `bfloat` and `half` families package the unchanged
`reduce.metal` minimum and maximum entries for bfloat16 and float16. An
extremum selects an input element, so these results need no rounding. Host
routing preserves two-byte logical allocation sizes, upstream launch selection
and intermediate 16-bit storage in two-pass whole-array reductions. Small rows
use the existing exact-grid package cache, including partial workgroups and
cooperative reductions. Larger row and column plans, and 16-bit sums and
products, remain explicit unsupported operations; they are not promoted to
float32.

```sh
python -m demos.integrations.mlx.portable_host.reduction_packages \
  --mlx-root mlx-upstream --target metal --family half \
  --width 32 --width 128 --width 256 --output-dir half-reductions
python -m demos.integrations.mlx.portable_host.verify_extrema_reductions \
  --mlx-root mlx-upstream --packages host-packages --dtype float16 \
  --elementwise half-packages --reductions half-reductions \
  --output-dir half-reductions-evidence
```

`--family bfloat` with `--dtype bfloat16 --elementwise bfloat-packages` runs the
bfloat proof. Each proof checks the public-API workloads against separate CPU
execution and independent storage-word references: 36 for bfloat16 and 38 for
float16, which adds signed binary16 subnormals. They cover negative maxima,
positive and negative infinity, NaNs, signed zero, 65,536-element inputs, both
reduction passes, scalar/cooperative rows and a 1,031-row partial workgroup.
Every partial result, output guard, source artifact and native module is checked.
Bfloat subnormal inputs stay refused off Metal under
[#2000](https://github.com/CrossGL/crosstl/issues/2000); the half arithmetic
proof already establishes binary16 subnormal comparisons on every target.
Local Metal and OpenGL runs each pass all 36 bfloat workloads with 40 native
dispatches; a local Metal run passes all 38 float16 workloads with 42. All six
HLSL and GLSL variants of each family compile with DXC and glslang, as do the
float16 small-row packages the proof launches. Windows and Linux native
execution are separate required CI results, not inferred from those compiler
checks.
The default reduction package inventory is unchanged. Existing platform jobs
run this optional family without adding runners.

Software subgroup lowering retains narrow conversions and their rounding when
converging pure vote-guarded reductions. It does not treat subgroup votes as
workgroup-uniform or speculate memory accesses and effectful calls. The shared
native regression gate also compares generated and original Metal execution.
Non-Metal subnormal bfloat reductions remain rejected under
[#2000](https://github.com/CrossGL/crosstl/issues/2000). These checks do not establish
all narrow reduction operations or a complete upstream test-suite pass.

## Random Generation

The optional random package family connects `RandomBits::eval_gpu` to the
unchanged `rbitsc` and `rbits` kernels. The
`HostRuntime(..., random=..., mlx_root=...)` constructor loads and verifies both
packages against the source pin, entry identities and artifact
contents. Key arrays must contain uint32 pairs. Contiguous, transposed,
broadcast and positive-stride key views use their original layout; negative
strides are materialized by translated copy kernels. Empty outputs do not
dispatch. The host validates input allocation spans, metadata, launch geometry
and disjoint output storage before native execution.

Native output is transferred as signed byte carriers and copied into the MLX
array without computing random values on the CPU. Per-key slots shorter than
four bytes use the padded layout described below. Output size is independent
of the 65,535-element key-storage bound. The allocation and 17 guard bytes
must fit signed 32-bit shader indices, and each launch dimension must fit
65,535 workgroups. With the existing one-thread workgroups this permits up to
524,280 bytes per key. The full grid is retained: splitting it naively would
change Threefry counters and therefore the random stream. Larger grids still
fail explicitly before dispatch. Device allocation limits also apply.

The initial package verifies both entries. Dispatch then selects a cached variant
bound to the validated key count, native bytes per key and exact workgroup grid.
Its scalar precondition is embedded in the artifact, and non-Metal artifacts
also retain their dispatch-region contract. Cache identities include the pinned
source, translation implementation and packaging recipe. Evidence verification
loads the existing variant and checks the dispatched artifact; it never creates
a replacement package. OpenGL output indexing uses these checked bounds, while
the strided-key assertions remain subject to the host's allocation-span checks.

Random package indexes record the output bound and dispatch contract; regenerate
older packages before using this host route. This remains synchronous integration,
not a complete random API or backend. Other elementwise and copy size limits
still prevent large composed distributions and the full upstream suite.

Whole-array assertion reductions use an explicit `--on-demand` catalog. The
catalog declares permitted entries and widths; it is not a translated artifact
or evidence of execution. After validating the host buffers and upstream launch
plan, dispatch creates a package bound to the exact input count, row size and
grid. The cache supports the existing float32, int32, uint32, Boolean and bfloat16
whole-array entries, with separate packages for different counts even when their
launch widths match. Initialization, row and column reductions retain their
existing package paths.

Each cached package carries checked scalar metadata and, for HLSL and GLSL, a
dispatch-region contract. Source, implementation, recipe and artifact identities
are verified when loading it. Trace verification requires the pinned source and
loads the recorded package without translating a replacement. Native dispatch
also observes the input and metadata buffers to detect device-side modification.
Static reduction packages remain supported for existing callers.

```bash
python -m demos.integrations.mlx.portable_host.random_packages \
  --mlx-root mlx-upstream --target metal --output-dir random-packages
python -m demos.integrations.mlx.portable_host.reduction_packages \
  --mlx-root mlx-upstream --target metal \
  --entry all_reduce_andbool_ --entry all_reduce_sumfloat32 \
  --width 32 --width 256 --on-demand --output-dir random-reductions
python -m demos.integrations.mlx.portable_host.verify_random \
  --mlx-root mlx-upstream --packages host-packages --random random-packages \
  --reductions random-reductions --output-dir random-evidence
```

The verifier runs 48 public-API workloads: key splitting with six key layouts,
including empty outputs, and float32 uniform generation with three explicit
seeds. Six larger split cases cover 65,536 through 786,432 output bytes,
contiguous, strided and broadcast keys, and the 65,535-workgroup boundary.
It also runs the unchanged upstream `test_global_rng`, `test_key`,
`test_key_split`, `test_uniform` and `test_gumbel` in separate CPU and
translated-backend processes. Results are
checked against an integer Threefry reference, with exact float32 output words
for the maintained uniform cases. Every random dispatch retains its uploads,
complete native bytes, guards, logical output hash, generated source and native
compiler evidence. Two separate processes check missing-package and oversized
output rejection before dispatch. CI requires this verification on all three
platforms. The upstream uniform test checks bfloat16 metadata without evaluating
that array; passing it does not establish bfloat16 execution support. The complete
random suite remains outside the demonstrated coverage.

A full-module probe of the pinned `test_random.py` completes 14 tests on CPU
and initially reported eight errors through generated Metal. Supplying the
Boolean and sum reduction variants resolves the uniform and Gumbel failures
without changing either test. With the half storage and arithmetic families,
the same module reports five errors: the broadcast scale/location test now
passes, while normal/Laplace still encounters an unsupported dtype. Remaining
work includes further dtype support, arg-reductions, sorting, and larger random
and reduction allocations. Tests that inspect
only lazy array metadata do not establish native execution. These failures are
not skipped or reclassified by the required five-test profile above.

The same job then runs 22 more unchanged `test_ops` tests on the families it
has already translated:

```bash
python -m demos.integrations.mlx.portable_host.verify_upstream_ops \
  --mlx-root mlx-upstream --packages host-packages --random random-packages \
  --bitwise bitwise-packages --selection selection-packages \
  --absolute absolute-packages --reductions init-packages \
  --reductions random-reductions --output-dir upstream-op-evidence
```

They are `test_trunc`, `test_erfinv`, `test_logical_and`, `test_logical_or`,
`test_full_ones_zeros`, `test_atleast_1d`/`2d`/`3d`, `test_tile`,
`test_repeat`, `test_roll`, `test_roll_errors`, `test_tri`, `test_tril`,
`test_triu`, `test_as_strided`, `test_view`, `test_depends`, `test_eye`,
`test_diag`, `test_diagonal` and `test_divmod`. Each runs once on the MLX CPU and once
through the translated host, in separate processes, and every native run must
dispatch. Every dispatch must belong to exactly one of the base, random,
bitwise, selection, absolute-value or reduction indexes and pass that
family's artifact and native-identity checks, or be a gather, scatter or
DivMod kernel specialized from the pinned sources, whose result is recomputed
from its retained inputs and whose native modules are checked. A test that
reaches scans or the 64-bit and 16-bit package families therefore fails
instead of running an unchecked package. A separate process
checks that a missing comparison package is refused before dispatch. On Metal
the native run dispatches 1,818 translated kernels, 12 of them indexing
kernels and 37 DivMod kernels (25 kernel and dtype pairs, translated on first
use; the cold native run takes about four minutes).

`View` follows `mlx/backend/gpu/primitives.cpp`: views that keep the element
size, narrow a contiguous last axis or start from a row-contiguous array share
the buffer. The remaining views copy densely before the reinterpretation. When
narrow elements pack into whole 32-bit words (or whole output elements) along
a contiguous last axis, the copy moves those words, so the bytes match
upstream's element copy without loading 8- or 16-bit values. Boolean views of
integer data therefore round-trip even when their bytes are not 0 or 1.

### Kernel Evidence

Successful compilation alone does not establish numerical parity. The
`random_audit` command translates unchanged `random.metal`, creates public runtime
packages and compares native results against an independent integer Threefry
reference. It retains translation reports, descriptors, input values, native
readbacks, compiled Metal libraries, compiler/dispatch identity and per-case failures. Incorrect output or
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
random numerical execution has not been established. The failures identified by
that baseline are tracked here:

| Contract | Issue |
| --- | --- |
| Reflected byte buffers and native byte transport | [#2022](https://github.com/CrossGL/crosstl/issues/2022) |
| Signed/unsigned byte conversion semantics | [#2023](https://github.com/CrossGL/crosstl/issues/2023) |
| Shared union storage in OpenGL | [#2024](https://github.com/CrossGL/crosstl/issues/2024) |
| Partial vector initialization in Metal | [#2025](https://github.com/CrossGL/crosstl/issues/2025) |
| Fixed-array range iteration in Metal | [#2026](https://github.com/CrossGL/crosstl/issues/2026) |
| Aggregate type lookup under name shadowing | [#2027](https://github.com/CrossGL/crosstl/issues/2027) |
| Narrow vector layout in Metal aggregates | [#2028](https://github.com/CrossGL/crosstl/issues/2028) |

The generated Metal, OpenGL and DirectX paths now pass all 20 cases. Windows
CI verifies native Direct3D execution, not only DXC compilation. OpenGL union
values now share one word allocation, with explicit byte packing and bitcasts
instead of independent fields. Signed byte results retain their exact values in
the portable 32-bit carrier. The native controls cover 4-, 8- and 16-byte unions,
copies, helper returns, indexed members and single-evaluation selectors.
Unsupported layouts, buffer union reflection and reference arguments through
union members produce structured diagnostics.

Byte conversion now retains the source's signed or unsigned eight-bit range
before widening into Metal/HLSL arithmetic carriers. Required native controls
on all three operating systems cover 38 workloads with runtime int32 boundary inputs,
casts, aliases, initialization, assignment, value arguments and returns, arrays,
aggregate members, direct and rebased buffer stores, vector constructors and single-evaluation
calls. Local byte increment and compound assignment preserve wraparound;
unsupported Metal/HLSL indexed updates fail explicitly rather than repeating
their selectors. Byte storage transport remains a separate contract.

CI checks the audit's reference, binding and evidence-validation contracts on all
three operating systems. Each gather workflow additionally requires all 20
native random cases. Metal retains each dispatched library and
checks its hash against the execution identity; OpenGL retains the exact GLSL
submitted to its native compiler, and DirectX retains the compiled DXIL module.
These are kernel gates, not host integration or upstream-suite coverage.
The host integration above adds MLX dispatch and upstream-test evidence to these
separate kernel checks.
No kernel edits, generated-source repairs or readback corrections are applied.

Separate original/generated Metal probes with tightly packed three-byte outputs
write into the first trailing guard; three-key probes also overwrite bytes in
neighboring key outputs. Both paths exhibit the behavior, so it is not a
translation discrepancy. Six- and eleven-byte probes preserve the tested guards.
The upstream kernel writes its first four output bytes unconditionally. Host
integration must account for this source allocation requirement and per-key
spacing without concealing overwritten values or relaxing guard checks.

The separate `--byte-tails` audit profile verifies a per-key staging layout for
this requirement. Outputs shorter than four bytes receive a four-byte native
slot; longer outputs retain their requested byte count. The number of counter
words and the launch geometry remain unchanged. `RandomOutputLayout` copies the
requested bytes from each slot without calculating or correcting random values.
The audit checks the complete native allocation, including each padding byte,
against the integer reference before checking the logical output. All 17 trailing
guards must remain unchanged.

This profile has 48 cases: contiguous and strided keys, one and three keys, and
1, 2, 3, 5, 6, 7, 9, 10, 11, 15, 17 and 33 output bytes per key. Native Metal
and OpenGL pass this profile locally; the three-platform gather workflow requires
the same profile in CI, including native Direct3D execution on Windows. It keeps
the existing 20 word-aligned cases as a separate required step. This establishes
bounded byte transport, not `RandomBits` host routing or upstream random-suite
parity.

The audit writes its active case, phase, elapsed time and completed-case count
to `evidence.json` before package preparation and native execution, and emits
the same progress to the CI log. Evidence remains unsuccessful until every case,
source verification and runtime cleanup finishes. JSON updates use atomic file
replacement so an interrupted run retains the last complete progress record.
The Windows byte-tail step has a 1,200-second overall budget for all 48 cases;
Metal and OpenGL retain their 450-second budgets. Native operation deadlines,
numerical comparisons, input observations and guard checks are unchanged.

```bash
python -m demos.integrations.mlx.random_audit --byte-tails \
  --mlx-root /path/to/pinned/mlx --target metal \
  --output-dir /path/to/random-byte-evidence
```

The required macOS gather workflow separately verifies the byte-storage and
narrow-aggregate prerequisites. Signed and unsigned byte buffers retain one-byte
elements through reflection, packaging, native upload and readback. Its 32
original/generated controls cover scalar and constant buffers, references,
two- and four-lane vectors, aliases, homogeneous structures and nonzero allocation
offsets. Invalid values and layouts fail before dispatch. DirectX/OpenGL widened
storage cannot accept these Metal byte payloads without a supported physical
representation.

Twenty additional original/generated controls verify 8- and 16-bit aggregate
members, overlapping unions, vector arrays, nested and generic structures,
dynamic lane reads and native sizes, with poisoned result guards. Narrow
aggregate storage remains distinct from widened arithmetic values. The public
loader also preserves explicit allocation views instead of silently replacing
them with offset-zero buffers
([#2063](https://github.com/CrossGL/crosstl/issues/2063)).

Vector list initialization now remains distinct from scalar-splat construction
through the shared representation. Metal retains braces; DirectX and OpenGL
explicitly initialize omitted components to zero. Required native checks cover
126 cases per target, including empty, partial and full lists, Boolean and numeric
lanes, aliases, returned vectors, aggregate members, assignments and ordered
side effects. The 48 signed/unsigned 64-bit cases use exact values beyond binary64
integer precision and cover two-, three- and four-lane vectors without narrowing
through 32-bit or floating-point intermediates. Excess components, including
explicit 64-bit scalar constructors, fail project translation. The Metal checks also
execute the original source; each target retains compiled artifacts and guarded
readbacks. Byte and vector storage have a separate bounded macOS step so these
checks do not consume the general-gather execution budget.

Range bindings retain their declared types and reference qualifiers through the
shared representation. Generated Metal uses native array iteration; DirectX and
OpenGL capture subarray selectors once and read each element from the original
array. References retain their storage identity rather than using a copy with
delayed writeback. Eighteen required native controls per target cover conversion,
copy/reference distinctions, changes through aliases, nested arrays, global
tables, member arrays, early exits and binding scopes. Metal also executes the
original sources. Unsupported reference shadowing, addressable binding views and
reference transport through helper parameters remain explicit diagnostics on the portable targets;
ordinary by-value helper arguments and copied loop bindings are covered separately
([#2064](https://github.com/CrossGL/crosstl/issues/2064)).

Local aggregate declarations use elaborated structure/union tags when a function
or local value hides the type name. Eight original/generated Metal controls
cover entry, helper, parameter and local names while preserving reflected entry
identities. The unchanged Metal random audit now passes all 20 contiguous
`rbitsc` and strided `rbits` workloads, including exact output bytes and neighboring
guards. The OpenGL audit also passes these 20 workloads with shared union storage.
DirectX also passes the same 20 cases on Windows. The separate partial-byte
profile and host verification establish their own allocation and execution
contracts; neither implies complete upstream-suite coverage.

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
[`demo-project-testing.yml`](../../../../.github/workflows/demo-project-testing.yml).
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
adapter. The OpenGL word-backed float atomic controls above do not establish
numerical parity for the complete gated-delta backward kernel.

Float atomic load/store checks require native execution on Metal and DirectX,
including the unchanged pinned `mlx_atomic_load_explicit<float>` and
`mlx_atomic_store_explicit<float>` helpers. Twenty-four reduced cases cover
buffer fields, pointer offsets, resource aggregates, shared memory, nested calls,
conditional operands and returned stores. Raw binary32 words preserve signed
zeros, subnormals, infinities and NaN payloads; readbacks also check address/value
evaluation counts, untouched inputs and buffer guards. Original Metal wrappers
execute alongside generated Metal. DirectX loads use a non-modifying bitwise
compare/exchange and stores use exchange, not floating-point addition by zero.
Required Windows execution is separate from DXC compilation. OpenGL rejects
these float-storage operations explicitly until their physical storage and
toolchain contracts are implemented. These tests do not establish float
compare-exchange support or complete MLX atomic/runtime parity.
Two additional native controls check fractional float addition and exchange
through resource aggregates, preserving old values and both resource bindings.

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
7 and 257. The 132 unary records cover the same lengths for all 30 operations,
six special-value cases, two large-angle cases, 8,193 consecutive float32 inputs
at and above one for inverse hyperbolic cosine, 1,670 boundary inputs each for
square root and reciprocal square root, and an arange/abs/negative/square chain,
totaling 19,542 unary output values. Independent Python math references
check both CPU and native results, not just agreement between them. Readbacks
must have complete values and unchanged case identities; finite comparisons
use `rtol=2e-5, atol=1e-6`, with exact zero values/signs and nonfinite
classification. The near-one inverse hyperbolic cosine case uses the upstream
`rtol=1e-5, atol=1e-6` limits. Upstream tests and their tolerances are unchanged.
The generated root boundary cases require exact binary32 values and zero signs,
with NaNs compared by classification. Their 120-digit Decimal references select
the nearest binary32 result directly. Inputs span every finite exponent,
neighbors of one, signed subnormals, zeros, negative values and infinities.
The unchanged Metal kernels are the source-device control, not the CPU backend.
The CPU reference preserves subnormal operands. In the pinned Apple Silicon
Accelerate vector path, the two smallest positive binary32 inputs in this
inventory instead produce infinity for reciprocal square root; the CPU-only
reference records these observed overflows explicitly. This does not relax the
generated-output checks or alter any readback. Other CPU reciprocal-root
rounding differences remain subject to the existing CPU tolerance.

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
a missing artifact, unary inputs with unsupported dtype, contiguous and
strided unary inputs that exceed their allocations (including a large view), and
copies with unsupported dtype, excessive size or an invalid source allocation.
Binary inputs with unsupported dtype or insufficient backing storage are also rejected, as
are casts with unsupported types or invalid source spans, including a view
larger than one dispatch whose backing allocation is too short.
Full also rejects unsupported int16 storage, excessive output size and a source
view extending before its allocation.
The native worker has a hard 900-second process-tree deadline; the CPU reference
and eighteen negative processes each retain a 180-second deadline. All are
attempted so a failure does not discard the other diagnostic results. The former
300-second native limit expired on hosted macOS after 801 completed dispatches,
before all workloads and upstream tests finished. A timed-out worker remains a
failure even if it has already written result files. CI allows
2,400 seconds for package translation and 4,000 seconds for verification within
a bounded 330-minute platform job. Small-row host workloads run in separate
300-minute platform jobs after the base host proof. These jobs also require
translated empty-reduction, bitwise, concatenation, selection and integer
absolute-value execution and retain their packages and readbacks.

Each job and matrix leg preserves its active execution when a branch changes
and keeps only one pending execution for that ref. The workflow has no parent
concurrency rule that could cancel active jobs or serialize unrelated matrix
legs. This lets long reduction jobs finish without building a queue of
superseded revisions. Retained evidence identifies its source revision;
results from an earlier commit do not establish that a newer one passes.

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
It compares 135 workloads against separate MLX CPU execution, NumPy and an exact
Python integer reference. The original 120 cases are retained, followed by one
65,536-element case for each of the 13 binary and two inversion entries. Each native run requires
202 dispatches, including
67 translated layout copies. Binary operations materialize transpose, broadcast
and negative-stride inputs. Inversion preserves stored-contiguous transpose and
broadcast layouts; negative-stride inputs still require a translated copy. Empty
inputs require no dispatch. The proof retains numerical readbacks before any
comparison failure, validates compiler and native-dispatch identities, checks
artifact hashes and output guards, and requires ten rejection controls.
Inversion controls reject missing packages, int64 and uint8 storage, and inputs
whose large stored view exceeds its backing allocation.

Unsigned shifts exercise all 32 bits and counts through 31; signed right shifts
include negative operands. Signed left-shift cases use nonnegative values and
representable results, without making a parity claim for undefined source
overflow. Binary operations and inversion may span multiple 65,535-element
batches. A bad shift count at the final element of a
65,536-element input must fail before any binary dispatch. Other integer
widths and Boolean shifts are not implemented by this family. Upstream maps
Boolean inversion to the existing LogicalNot path.
The complete upstream `test_bitwise_ops` also requires random generation and
those missing widths and operations; the 135 workloads are not a substitute for
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
reversed inputs use translated copies. Stored values use the unary batch path.
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
unsupported int16 absolute value, invalid large backing views, and unsupported 64-bit
reductions and bitwise operations.

This does not add 64-bit division, selection, concatenation or general unary
operations. Other primitive limits and synchronous host staging remain; large
64-bit absolute-value execution is not established by this proof.
The two upstream tests are additional coverage, not full-suite parity.

### Padding and Slice Updates

`Pad` fills the destination through an unchanged translated copy kernel, then
copies the input into its padded region. `SliceUpdate` copies the base array and
replaces the requested slice through the same kernel. No array computation is
performed by a CPU fallback. Destination strides may be negative; source and
destination bounds, preserved regions and nonoverlapping writes are checked
before native submission. Aliased base/update views remain alive while a distinct
output allocation is populated.

These hooks support float32, int32, uint32, bool, int64 and uint64. Padding retains
its 65,535-output-element limit; slice updates use the separate allocation and
dispatch bounds described below. The wider types require the optional `integer64` packages.
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

Destination allocations are bounded independently from the 65,535-invocation
limit of an individual reduction dispatch. The adapter accepts up to
268,435,423 destination elements, subject to the translated copy path's separate
per-axis limits. Larger updates are divided into dense rectangular slabs. Each
slab retains the original destination strides, rebases its offset, and advances
only the update pointer. Full-destination readbacks check untouched values and
guards before committing the result. The source kernel is unchanged.

`crosstl_mlx_register_runtime` registers both dispatch and package-availability
callbacks, retaining dispatch ABI version 3. This lets the adapter reject a
missing reduction package before copying the base. The earlier dispatch-only
registration entry point remains available, but cannot enable slice reductions.
The Python adapter and prepared MLX build must be regenerated together.

`verify_slice_updates` requires 304 existing and 16 allocation-capacity CPU/native
workloads, the unchanged
`test_array_at_slice_update_extensive` method and eight isolated rejection
checks. It checks exact result storage, source preservation, native copy
and reduction readbacks, destination metadata, guards and artifact hashes.
Capacity cases cover all six storage types, every reduction operation, slice
replacement, offset inputs, negative destination strides, strided updates,
broadcasts, aliasing, rank-three arrays and the old allocation boundary. Each
batch's uploaded values and complete intermediate destination are checked
against an independent flattened selection, including ordered exact coverage.
The verifier checks retained compiled modules, source identities, entry points
and actual dispatch geometry. Boolean full-region arithmetic can be simplified
into binary operations before reaching the slice hook; that separate host gap
is tracked in [#2149](https://github.com/CrossGL/crosstl/issues/2149).
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
General scatter, other storage widths, copy launches exceeding 65,535 groups
on any axis and asynchronous device-resident execution remain separate work.

Project Porting Implementation Notes
====================================

These notes describe individual generated helpers, the source and target
constructs each one covers, and the bounded proofs that exercise them. The
:doc:`porting-guide` covers the workflow, configuration and report.

Entry-Scoped Compute Artifacts
------------------------------

Repositories can request standalone artifacts for one or more materialized
source entries by adding a repository-relative selector table to
``crosstl.toml``:

.. code-block:: toml

   [project]
   include = ["kernels/arange.metal"]
   include_dirs = ["."]
   targets = ["directx", "metal", "opengl"]
   output_dir = "crosstl-out"

   [project.entry_points]
   "kernels/arange.metal" = ["arangeuint32", "arangec64"]

Each value may be one entry name or an ordered array of entry names. An array
creates one independently checkpointed artifact per entry in the declared
order. Empty arrays and duplicate names are rejected.

For DirectX compute output, each selected source entry is emitted as target
entry ``CSMain``. A source such as ``kernels/arange.metal`` produces
``crosstl-out/directx/kernels/arange/arangeuint32.hlsl`` and
``crosstl-out/directx/kernels/arange/arangec64.hlsl``. Each standalone HLSL
artifact retains only that entry's reachable helpers, resources, constants,
and execution contract. Explicit registers and spaces remain unchanged, while
runtime-loader metadata records the selected ``cs_6_0`` entry profile.

For OpenGL compute output, the same selection produces one ``.glsl`` artifact
per entry with target entry ``main``. For Metal compute output, it produces one
``.metal`` artifact per entry and retains only the selected kernel, reachable
helpers, referenced declarations, and recursively referenced struct families.
The emitted Metal entry name is recorded exactly rather than normalized to a
fixed ``main``. Runtime manifests reflect the standalone Metal source and keep
``buffer``, ``texture``, and ``sampler`` index spaces independent.

For all three targets, the portability report records every source entry,
target entry, and reflected stage; embedded validation records carry the same
identities. Runtime artifact manifests then reflect only the selected stage
interface from each standalone output.

Selection is exact after source materialization. Missing or ambiguous entries
fail with structured diagnostics and no target file. Targets that do not yet
implement standalone entry generation also fail explicitly instead of pruning
an aggregate artifact. When ``project.entry_points`` is absent, project
translation keeps the existing aggregate output path and behavior.

Entry selection scopes shader or kernel translation; it does not infer host
dispatch dimensions, runtime bindings, or backend integration. Record those
requirements through the corresponding dispatch and runtime contracts.

Repeated Metal project translations retain up to eight successful template
materializations in memory. Includes and macros are expanded afresh before
lookup. Reuse requires the same unit, source text, expanded sources, macro
tables, entry, target, variant, defines, include paths and every source option
that materialization reads, compared by exact type and order. Dispatch regions,
subgroup widths and other target-lowering options are withheld from
materialization, so one materialization can serve several dispatch contracts.
Returned records are copied for each caller, and failed specializations are not
retained. Target generation, dispatch contracts, reflection and package
verification still run independently for each artifact. This cache is not a
persistent translation checkpoint.

The next MLX kernel-tree increment is pinned independently at
``9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8``. Entry discovery must report
exactly 49 Metal units, 17,832 entries, and zero diagnostics. The compact
``arg_reduce.current-tree.translation.json`` contract pins all 24 discovered
``arg_reduce.metal`` entries and 72 deterministic Metal, OpenGL, and DirectX
artifacts. Required macOS, Linux, and Windows CI compiles all 24 entries with
the native Metal compiler, ``glslangValidator``/``spirv-val``, and DXC,
respectively. Numerical runtime parity includes a representative
float32 subset: ``argmin_float32`` and ``argmax_float32`` must execute on Metal,
Mesa EGL, and Direct3D 12 WARP over two rows, axis sizes 32 and 129, strides 1
and 2, ordinary values, and NaN/Infinity values; the Metal path also executes
the exact upstream metallib for parity. Twelve signed/unsigned 16/32/64-bit
entries add 96 integer cases, covering extrema, lowest-index ties, axis sizes
31, 32, 33 and 129, and both strides. All three targets require these 14 entries
and 112 cases. This is a 24/17,832 deterministic translation and native-compiler
increment with 14/17,832 numerical runtime coverage, not full-tree coverage.
The contract explicitly records that the
upstream MLX test suite and MLX host-runtime redirection have not yet been
implemented. Strict JSON runtime requests encode non-finite float32 inputs with
the exact strings ``nan``, ``+infinity``, and ``-infinity`` before
deterministic IEEE-754 packing.

``searchsorted.current-tree.translation.json`` pins all 26 discovered
``searchsorted.metal`` entries (13 element types, left and right) and their 78
Metal, OpenGL and DirectX artifacts at the same revision. Every entry is
compiled by the platform validator and executed on Metal, Mesa EGL and
Direct3D 12 WARP: three sequences (length 1, length 33, and length 40 with
stride 2) against 48 queries on 1D and 3D grids, with 16 output guards and four
unchanged inputs. Queries cover duplicates, absent extremes, sign-bit
boundaries and NaN ordering for floating and complex values. Metal also runs
the unchanged upstream kernel. OpenGL relies on two host index-range
preconditions (``int64(mid) * a_stride`` and ``offset`` in signed 32-bit
range). This is a 26/17,832 increment; it does not run the upstream MLX test
suite or redirect the MLX host runtime.

Two units are new at that revision. ``gather_mm_offsets.current-tree.translation.json``
pins the single ``gather_mm_offsets`` entry; it is not a template, so a
project-level ``workgroup_size = [1, 1, 1]`` reproduces MLX's one-thread-per-group
grid. Six cases (no rows, one row, grouped and empty experts, 1,000 sorted rows,
unsigned sign-bit indices and an unsorted sequence) run on all three backends
with decoy padding, 16 output guards and two unchanged inputs.
``cross_entropy.current-tree.translation.json`` pins all six forward and VJP
entries for float32, float16 and bfloat16. A 128-thread workgroup exercises the
forward kernel's four-group reduction, which OpenGL and DirectX lower to
software 32-lane groups. Rows of 7, 512 and 1,300 classes cover idle threads,
an exact pass and three passes with a tail, plus negative targets, masked
``-inf`` classes, large logits and a ``+inf`` logit. Float32 results must be
within 5e-5 absolute plus 5e-5 relative of a float64 reference; 16-bit VJP
results must be within one unit in the last place, with at least 99% correctly
rounded. OpenGL relies on host preconditions that row offsets and element
indices fit in signed 32-bit range. The Metal round trip keeps ``fast::exp``,
so translated Metal must match the upstream kernel bit for bit; OpenGL and
DirectX call their default ``exp``. Metal also runs both unchanged upstream
kernels. Together these cover
7/17,832 entries.

``arange.current-tree.translation.json`` pins all 11 ``arange.metal`` entries
and their 33 artifacts. Every entry executes exactly on all three backends:
integer cases wrap modulo the type width and form 64-bit products above 2^32,
and floating cases cover the fused ``start + index * step`` rounding, binary32
subnormal flushing, overflow, binary16 subnormals and index rounding up to
65,534. OpenGL and DirectX select ``floating_contraction_profile = "rne-flush"``
to reproduce Metal's contraction; separate rounding would differ in 1,269, 906
and 227 of 4,096 float32, float16 and bfloat16 results of one case. Each
dispatch checks 16 output guards and two unchanged constants, and Metal also
runs the unchanged upstream kernel. This is an 11/17,832 increment.

``random.current-tree.translation.json`` pins both ``random.metal`` entries,
``rbitsc`` and ``rbits``, and their six artifacts. One-thread workgroups make
``threads_per_grid`` equal the dispatched workgroup count, the ``rbits`` union
lowers to a packed word pair, and the ``char`` output widens to sign-extended
32-bit slots on OpenGL and DirectX. All three backends must match every byte of
24 cases (21,403 key bytes). 15 cases reproduce the Threefry 2x32 known answers
pinned by MLX's ``tests/random_tests.cpp``, including the vmapped row and column
key layouts; the rest cover wrapping key schedules, every word-count parity and
edge-byte combination, 2,193 workgroups, broadcast and permuted key views and
unread decoy words. Each dispatch checks 16 output guards and its unchanged
inputs, and Metal also runs the unchanged upstream kernel. OpenGL relies on host
preconditions that the per-key byte offset and composed byte index fit in
signed 32-bit range. This is a 2/17,832 increment.

``dot.current-tree.translation.json`` pins the three ``dot.metal`` entries and
their nine artifacts. MLX's ``metal::vec<T, 4>`` storage views stay native
Metal vector views. OpenGL and DirectX keep one physical element per 16-bit
value, so they gather the lanes element by element; DirectX decodes bfloat16
lanes with an exact shift. Both reduce in software 32-lane groups and predicate
the reduction under ``tid < SIMD_GROUPS``. Inactive invocations contribute the
sum identity while every invocation reaches the barriers. All three backends
execute 13 cases of up to 40,000 elements. The exact cases must match bit for
bit, the rounding case must stay within its summation bound and the infinite
case must produce +inf. Metal's partial sums must equal the unchanged upstream
kernel's bit for bit. This is a 3/17,832 increment.

``logsumexp.current-tree.translation.json`` pins the six ``logsumexp.metal``
entries and their 36 artifacts: the block kernel at the 32-, 288- and
1,024-thread groups that MLX derives from the row length, and the looped kernel
at 1,024 threads. OpenGL and DirectX reduce in software 32-lane groups and
predicate the block kernel's second reductions under ``simd_group_id == 0``, so
inactive invocations contribute the maximum's -inf or the sum's identity while
every invocation reaches the barriers. All three backends execute 17 cases of
up to 20,000 elements with MLX's kernel and group-size choice for each row
length, covering masked ``-inf`` entries, values that overflow ``exp`` without
the maximum, running-sum rescaling across passes and the infinite-maximum
branch. Float32 results must stay within 5e-5 plus 2^-21 relative of a float64
reference, and 16-bit results must be a correct rounding of a value in that
interval. The Metal round trip keeps ``fast::exp`` and must match the unchanged
upstream kernel bit for bit. This is a 6/17,832 increment.

``softmax.current-tree.translation.json`` pins the ten ``softmax.metal`` entries
and their 60 artifacts at the same block and looped group sizes. The non-precise
float16 and bfloat16 kernels accumulate in their storage type, so OpenGL and
DirectX predicate binary16 sums and maxima and the bfloat16 overloads' converted
float reductions under ``simd_group_id == 0``. A predicated binary16 sum is only
planned when the guard depends on the subgroup index alone, so each subgroup
takes part completely or not at all and the pairwise rounding order is the
unguarded one; guards that split a subgroup report
``half-masked-participation-unproven``. DirectX also lowers binary16 minima and
maxima, ordering signed zeros and ignoring NaN lanes as the binary32 lowering
does. All three backends execute every element of the logsumexp cases. Results
must be a correct rounding of a value within a per-element bound on the kernel's
own rounding (one rounding per operation in the accumulation type, Metal's
``fast::exp`` allowance and each reduction's depth), and rows containing
``+inf`` must be NaN as in MLX. Translated Metal keeps ``fast::exp`` and must
match the unchanged upstream kernel bit for bit. This is a 10/17,832 increment.

``layer_norm.current-tree.translation.json`` pins the twelve ``layer_norm.metal``
entries and their 108 artifacts: the single-row forward kernel at 32, 288 and
832 threads, the single-row VJP at 32, 288 and 1,024, and both looped kernels
at 1,024, each VJP with and without a weight. DirectX materializes the
``has_w`` function constant; Metal and OpenGL specialize it when the runtime
creates the pipeline. All three backends execute every element of 14 forward
and 14 VJP cases with weight strides 0, 1 and 2 and bias strides 0 and 1.
Results must be a correct rounding of a value within a per-element bound on
the kernel's binary32 rounding and its 16-bit output arithmetic, and a VJP
without a weight must leave ``gw`` untouched. This is a 12/17,832 increment.

``rms_norm.current-tree.translation.json`` pins the twelve ``rms_norm.metal``
entries and their 126 artifacts: the single-row forward kernel at 32, 288 and
1,024 threads, the single-row VJP at 32, 64, 288 and 1,024, and both looped
kernels at 1,024, each VJP with and without a weight. The single-row VJP loops
over the rows of its threadgroup and reduces two partials under one
``simd_group_id == 0`` guard, so the OpenGL and DirectX software subgroups
predicate both sums in source order inside that loop. All three backends
execute every element of 14 forward and 16 VJP cases, including 513 and 1,025
rows whose last threadgroup covers fewer rows. Results must be a correct
rounding of a value within a per-element bound on the kernel's binary32
rounding and its 16-bit output arithmetic, and a VJP without a weight must
leave ``gw`` untouched. This is a 12/17,832 increment.

``ternary.current-tree.translation.json`` pins all 212 ``ternary.metal``
``Select`` entries and their 636 artifacts: 14 base kernels for each of the 13
element types plus the ``vn``, ``vsn`` and ``svn`` work-per-thread kernels for
the ten types narrower than eight bytes. Every entry is compiled by the
platform validator and executed on Metal, Mesa EGL and Direct3D 12 WARP, two
cases per kernel: flat runs that end inside a work-per-thread tile, scalar
operands, 2-D ``v2`` grids, and 1-D to 5-D strided shapes with transposed,
permuted and broadcast operands on both the 32-bit and 64-bit index paths.
Results must match bit for bit, including signaling NaN payloads, signed zeros,
subnormals and integer extremes. Every dispatch checks 16 output guards, four
decoys after each operand and that every input is unchanged, and Metal also
runs the unchanged upstream kernel. OpenGL relies on 11 host index-range
preconditions in signed 32-bit range. This is a 212/17,832 increment; it does
not run the upstream MLX test suite or redirect the MLX host runtime.

``binary_two.current-tree.translation.json`` pins all 225 ``binary_two.metal``
``DivMod`` entries and their 675 artifacts: 15 base kernels for each of the 13
element types plus the ``svn``, ``vsn`` and ``vvn`` work-per-thread kernels
for the ten types narrower than eight bytes. ``DivMod`` returns its floor
quotient and remainder as one ``metal::array<T, 2>``; Metal output keeps it as
``array<T, 2>``, OpenGL as ``T[2]`` and DirectX as a two-component vector, or a
two-member struct for complex64. Every entry is compiled by the platform
validator and executed on Metal, Mesa EGL and Direct3D 12 WARP, two cases per
kernel over the same flat, 2-D and strided layouts as ``ternary.metal``.
Operands cover every sign combination, integer extremes, booleans, signed
zeros, infinities, NaNs, subnormals and complex64 values with exactly
representable parts. Results must match exact integer and floating models word
for word, any NaN matching any NaN, including the upstream float16 floor of a
binary16-rounded quotient. Every dispatch checks 16 output guards, four decoys
after each operand and that every input is unchanged, and Metal also runs the
unchanged upstream kernel. OpenGL relies on seven host index-range
preconditions in signed 32-bit range. This is a 225/17,832 increment; it does
not run the upstream MLX test suite or redirect the MLX host runtime.

``rope.current-tree.translation.json`` pins all 18 ``rope.metal`` entries and
their 360 artifacts: single-step and general kernels, with base-derived or
explicit frequencies and 32-bit or 64-bit indices, in every specialization of
the ``forward``, ``traditional`` and ``hs_transpose`` function constants that
the host can select. DirectX materializes the constants; Metal and OpenGL
specialize them when the runtime creates the pipeline. All three backends
execute 504 dispatches covering decode offsets, ragged head groups, per-batch
offsets, partial rotations, strided and transposed inputs and a 4,000-position
context. Translated Metal keeps ``fast::cos`` and ``fast::sin`` and must match
the unchanged upstream kernel bit for bit. OpenGL and DirectX lower them to
their default functions, so results are checked against a per-element bound on
the kernel's binary32 rounding that allows ``2^-17 + 2^-20 * abs(theta)`` per
trigonometric value, at least twice the worst error measured for Apple's fast
functions, Mesa llvmpipe and WARP; unrotated elements must keep their exact
bits. Metal also runs the unchanged upstream kernel against the same bound.
OpenGL relies on one host index-range precondition for the single kernels and
six for the general kernels. This is an 18/17,832 increment; it does not run
the upstream MLX test suite or redirect the MLX host runtime.

``scaled_dot_product_attention.current-tree.translation.json`` pins all 82
``scaled_dot_product_attention.metal`` entries and their 1,254 artifacts:
the blocked scale-copy and causal-mask kernels, and every one-pass
``sdpa_vector``, ``sdpa_vector_2pass_1`` first-pass,
``sdpa_vector_2pass_1_gqa`` grouped-query first-pass and
``sdpa_vector_2pass_2`` aggregation kernel. Each one-pass kernel is pinned in
eight specializations of
its six function constants that pair every mask mode with both query layouts
and both sink settings; each first-pass kernel takes the same eight with 32,
64 or 128 blocks, its seventh constant. DirectX materializes the constants;
Metal and OpenGL specialize them when the runtime creates the pipeline,
except that OpenGL freezes the first pass's, whose key loop steps by the
block count from the block index. OpenGL and DirectX lower the SIMD
reductions to 32-lane software subgroups. OpenGL and DirectX execute 1,062
dispatches, and Metal every one whose threadgroup the device fits, covering
grouped-query decode, multi-query prefill, transposed and cached keys and
values, per-head, broadcast and transposed masks with fully masked rows,
attention sinks, first passes over 1,024 to 1,061 keys, where MLX takes the
two-pass path on Max and Ultra devices, grouped-query first passes over
8,192 to 8,320 keys in 64 to 256 blocks, and aggregation over 32 to 128
blocks. Attention outputs are checked
against a per-element bound on the kernel's binary32 rounding, with
``fast::exp`` kept on Metal and lowered to the default ``exp`` on OpenGL and
DirectX; rows without keys must
be exactly zero and the blocked kernels bit-exact. First-pass maxima must lie
within the largest score error of the exact maxima, and their sums and
partial outputs are checked relative to the reported maxima. Metal also runs
the unchanged upstream kernel against the same bounds, and the translation
must reproduce it bit for bit. Where MLX's own pipeline cannot run the 1,024
threads it requests (``check_kernel_threadgroup_size`` would throw), the Metal
gate records both pipelines' limits and skips that dispatch; elsewhere the
translation must fit them too. OpenGL relies on 19 host index-range
preconditions, and the first pass on a host precondition bounding ``N``. The
grouped-query first pass gives each SIMD group its own key sub-chunk, so
OpenGL and DirectX run every group for the sub-chunk's full length and mask
the steps past its keys, keeping every software-subgroup reduction uniform.
This is an 82/17,832 increment; it does not run the upstream MLX test suite or
redirect the MLX host runtime.

``gated_delta_update.current-tree.translation.json`` pins all 90 entries of
``gated_delta_update.metal``, the 72 sequential ``gated_delta_seq`` entries
and the 18 eight-step ``gated_delta_fused_chunk`` entries, and their 486
artifacts. Each sequential entry is pinned with function constant 200,
``save_state``, off for inference and on for the backward's save pass; DirectX
materializes the constant, while Metal and OpenGL specialize it when the
runtime creates the pipeline. The fused kernel runs only for inference and
never reads the constant. OpenGL and DirectX lower the sequential kernel's two
32-lane reductions to software subgroups. Each target executes 342 dispatches
with MLX's grids. The sequential kernel runs a one-token decode and an
eight-token, two-batch prefill for inference, and a one-token decode and
2 * Ckpt + 1 steps, three checkpoint slots, for the save pass. Outputs, final
states and checkpoints are checked against a float64 recurrence with a
per-element bound on the kernel's binary32 rounding that carries each state
row's error through ``g (I - beta k k^T)``. The fused kernel holds its 8 x 8
tiles in ``simdgroup_float8x8`` values, moves them with ``simdgroup_load``,
``simdgroup_store`` and ``thread_elements()`` at MLX's ``(fm, fn)`` lane
coordinates, registered as the ``tile_4x4_row_pair`` fragment mapping, and
multiplies them with ``simdgroup_multiply`` and
``simdgroup_multiply_accumulate``. OpenGL and DirectX lower the tiles to
32-lane software subgroups and prove the chunk loop uniform from a host
precondition ``0 <= T <= 2^31 - 9``. Each fused entry runs a one-step tail
(T = 9), a five-step tail over two batches (T = 13) and three chunks
(T = 24) with keys that share a direction, so the WY inverse departs far from
the identity; its bound adds any summation order of every tile product and the
specified error of ``fast::exp`` and ``fast::log``. Metal also runs the
unchanged upstream kernels, and the translation must reproduce them bit for
bit. No backend needs an index-range precondition. This is a 90/17,832
increment; it does not run the upstream MLX test suite or redirect the MLX
host runtime.

``gated_delta_update_vjp.current-tree.translation.json`` pins all 72
``gated_delta_vjp_seq`` entries of ``gated_delta_update_vjp.metal`` and their
216 artifacts. Each entry replays every checkpoint segment's entry states into
registers and walks it backward, adding the query, key, decay and beta
gradients through ``mlx_atomic<float>``: Metal keeps ``atomic_float``,
OpenGL uses a compare-and-swap loop on ``uint`` words and DirectX an
``InterlockedCompareExchangeFloatBitwise`` loop, and both lower the four
32-lane reductions to software subgroups. OpenGL needs one index-range
precondition, bounding the complete offset of every atomic update. Each target
executes 216 dispatches with MLX's grid: one step, three steps of two batches,
and 2 * Ckpt + 1 steps across three checkpoint segments. All six gradients are
checked against a float64 backward recurrence with a per-element bound on the
kernel's binary32 rounding, including any summation order of the atomic
contributions. Metal also runs the unchanged upstream kernel: it must meet the
same bounds and match the translation bit for bit in the value and
initial-state gradients, which do not depend on thread order. This is a
72/17,832 increment; it does not run the upstream MLX test suite or redirect
the MLX host runtime.

``sort.current-tree.translation.json`` pins all 264 entries of ``sort.metal``
and their 792 artifacts: 228 single-block sort and argsort kernels for
contiguous and non-contiguous rows and the 36 ``sort_mbsort``,
``partition_mbsort`` and ``merge_mbsort`` kernels, for 12 element types. MLX
sorts with a stable merge sort under ``LessThan``, with NaN after every
ordered value. Threadgroup declarations carry no runtime initialization, as
Metal never constructs threadgroup storage on the device; this re-pins the 128
complex64 entries of each 846d reduce contract that stage partials there.
Value-only sorts pass ``nullptr`` for the index tile: DirectX
binds it to an inert specialization that fails closed if read, and OpenGL
folds it away. OpenGL needs host preconditions that keep every helper's tile
view within ``[0, N_PER_BLOCK - 1]`` and the 64-bit element offsets of the
non-contiguous kernels within signed int range. Each target executes 690
dispatches with the shapes ``sort.cpp`` routes to each kernel, including a
three-tile multi-block pipeline that checks the output of every sort,
partition and merge stage. Values include ties, NaN and both zero signs, and
every output word is compared exactly. Metal also runs the unchanged upstream
kernels, which must produce the same words. This is a 264/17,832 increment; it
does not run the upstream MLX test suite or redirect the MLX host runtime.

``conv.current-tree.translation.json`` pins the 27 non-Winograd entries of
``conv.metal`` and their 126 artifacts: the one- to three-dimensional
``naive_unfold_nd`` and ``naive_unfold_transpose_nd`` kernels of the explicit
GEMM convolution, ``depthwise_conv_2d`` and ``depthwise_conv_1d`` with its
int64-index forms, for float32, float16 and bfloat16. ``MLXConvParams<N>``
reflects as a record whose bytes match MLX's C struct on every target, and
``depthwise_conv_2d``'s ``constant`` reference to it as a std140 uniform block
on OpenGL and a ``ConstantBuffer`` on DirectX. That kernel's tile load indexes
threads as ``simd_gid * 32 + simd_lid``, so OpenGL and DirectX use 32-lane
software subgroups; its kernel size, strides and flip are function constants,
and DirectX materializes six specializations. Mesa folds the depthwise
accumulators' ``+0.0`` start away even from a ``precise`` addition once a 1x1
kernel unrolls, so OpenGL and DirectX accumulate with the software
``rne-flush`` binary32 addition. OpenGL needs index-range
preconditions for the composed 64-bit offsets. Each target executes 114
dispatches with ``conv.cpp``'s groups on grids that tile exactly, as OpenGL
and DirectX cannot clip partial groups, covering padding, strides, dilation,
flipped kernels, batches, row tiles and ragged edges. Inputs keep every
product and sum exact, so each output word is compared exactly; Metal also
runs the unchanged upstream kernels, which must produce the same words. The
nine Winograd transforms have their own gate. This is a 27/17,832 increment;
it does not run the upstream MLX test suite or redirect the MLX host runtime.

``winograd.current-tree.translation.json`` pins all nine Winograd entries of
``conv.metal``: the weight, input and output transforms for float32, bfloat16
and float16. Each SIMD-group forms ``G g G^T``, ``B^T d B`` or ``A^T m A``
with two ``simdgroup_matrix<float, 8, 8>`` products per channel, which Metal
keeps and OpenGL and DirectX lower with 32-lane software subgroups as the
``fma-rne-gradual`` software multiply-accumulate: fused binary32 steps from
+0.0 in increasing k, ties to even, gradual underflow, as on Apple's matrices.
``G``'s and ``B``'s constants make many steps round, so the reference comes
from an exact fused multiply-add oracle and each output word is compared
exactly; ``A``'s powers of two make the output transform's products exact, so
that transform checks summation order rather than fusion. Metal also runs the
unchanged upstream kernels. The channel loops hold barriers, so OpenGL and
DirectX check the metadata precondition ``C <= INT_MAX - 31`` and the record
member preconditions ``params.C`` and ``params.O <= INT_MAX - 31``; the input
and output transforms' matrix loops start at the SIMD-group index with equal
trip counts in every SIMD-group, which translation proves for the software
subgroup barriers. Two cases over ``MLXConvParams<2>`` records cover one and
two batches, one and three channel blocks, and partial output tiles in both
directions. This is a 9/17,832 increment; it does not run the upstream MLX
test suite or redirect the MLX host runtime.

``scan.current-tree.translation.json`` pins all 456 entries of ``scan.metal``:
``contiguous_scan`` and ``strided_scan`` for its 57 sum, product, max, min and
``logaddexp`` helpers over bool, integer, float16, float32, bfloat16 and
complex64 elements, inclusive and exclusive, forward and reverse. Sums and
products of types narrower than 8 bytes use Apple's ``simd_prefix_*``
intrinsics, the rest the kernels' ``simd_shuffle_and_fill_up`` loops; OpenGL
and DirectX lower both to 32-lane software subgroups, and DirectX selects
``native-precise`` binary32 addition and own-value out-of-range relative
shuffles. DirectX lane moves now pass an exact bfloat16 value's bits rather
than ``uint(float(x))``, which gave the 24 bfloat16 max, min and
``logaddexp`` scans wrong words on Direct3D 12 WARP; their DirectX artifacts
are re-pinned. ``contiguous_scan`` reads all 32 ``simdgroup_sums`` slots after
writing one per SIMD-group, and DXC loads ``groupshared bool`` words with
``!range [0, 2)``, which makes loads of never-written words undefined; with
bool words the four contiguous bool maximum scans failed on WARP in two
consecutive runs. The gate sets DirectX's ``groupshared_bool_storage =
"uint"``, which keeps groupshared bools in ``uint`` words converted at each
use, and re-pins the 24 bool maximum, minimum and product DirectX artifacts.
The software subgroup convergence pass now stages guarded inputs
around helpers that call ``cos``, ``sin``, ``sqrt`` or ``atan2``, end in
``if``/``else`` returns or call the generated precise binary32 square root,
which complex ``logaddexp`` needs. Each target executes 1,824 dispatches with
``scan.cpp``'s groups: contiguous rows of 1, 37, one block and two blocks plus
five elements, and strided ``(outer, axis, stride)`` arrays of ``(3, 1, 7)``,
``(3, 37, 5)``, ``(3, 64, 32)`` and ``(1, 33, 70)``. A step-by-step model of
both kernels, checked against a left fold of each line, gives the expected
words; inputs keep every result exact in any association order, except one
dense ``logaddexp`` case per kernel with a propagated 16 ulp (complex: 64 ulp)
per-operation bound. Checked metadata preconditions bound ``axis_size`` and
``stride`` to [1, 4096] and ``stride_blocks`` to [1, 128], and OpenGL needs
index-range preconditions for the reverse kernels, which the model checks
access by access. Metal also runs the unchanged upstream kernels, which the
translation must reproduce exactly; the development machine's Metal compiler
gives both words that differ from the model for reverse contiguous scans of
one-byte types, and only those 26 entries may. This is a 456/17,832 increment;
it does not run the upstream MLX test suite or redirect the MLX host runtime.

``gemv.current-tree.translation.json`` pins 224 of the 256 entries of
``gemv.metal``: ``gemv`` and ``gemv_t`` for float32, float16, bfloat16 and
complex64 over every tile shape ``matmul.cpp`` selects, with and without
non-contiguous batches and ``axpby``, and ``gemv_gather`` and
``gemv_t_gather``. The 32 ``gemv_wide`` and ``gemv_wide_gather`` entries
have their own gate, below. Software
subgroups need every invocation at their barriers, so programs whose early
returns cannot be proven uniform are rendered again with those returns
converged into predicated statements, and a void function's top-level branch
whose condition may vary and which holds a barrier, such as ``gemv_t``'s row
loop, runs its barriers and counted loops in every invocation and its other
statements under the saved condition. The Metal frontend lowers aggregate
``x += y`` on thread-local values through a value form of the source's thread
``operator+=``, which complex64 accumulators need. Each target executes 672
dispatches with ``matmul.cpp``'s ``(32, bn, bm)`` threadgroups: short, ragged
and aligned cases over contiguous, non-contiguous and gathered batches. A
thread-level model of ``GEMVKernel::run`` and ``GEMVTKernel::run`` must store
each direct dot product; small integer inputs keep every float32 sum, every
result and the ``axpby`` terms exact, so every word is compared exactly. The
model justifies OpenGL's index-range preconditions load by load, and Metal
also runs the unchanged upstream kernels, which the translation must
reproduce exactly. This is a 224/17,832 increment; it does not run the
upstream MLX test suite or redirect the MLX host runtime.

``gemv_wide.current-tree.translation.json`` pins the other 32 entries of
``gemv.metal``, ``gemv_wide`` and ``gemv_wide_gather``, in the 80
function-constant specializations ``matmul.cpp`` selects. Metal and OpenGL
keep the constants, so one artifact serves an entry's specializations, and
DirectX bakes each into its own artifact. The kernels' chunk loop steps a
32-bit counter by the threadgroup count around SIMD-group shuffles; software
subgroups accept such a loop because a modular update visits the same values
in every invocation for any uniform operand. Each target executes 240
dispatches with ``(32, k_lanes / 8, 1)`` threadgroups: short, ragged and
aligned cases with partial vector chunks, clamped row blocks, round-robin
chunks over two grid columns, unrolled and leftover loads, and contiguous,
non-contiguous and gathered batches. A lane-level model of ``GemvWide::run``
must store each direct product; small integer inputs keep every float32 sum,
result and ``axpby`` term exact, so every word is compared exactly, and
Metal also runs the unchanged upstream kernels with the same function
constants. This is a 32/17,832 increment; it does not run the upstream MLX
test suite or redirect the MLX host runtime.

``gemv_masked.current-tree.translation.json`` pins all 528 entries of
``gemv_masked.metal``: ``gemv_masked`` and ``gemv_t_masked`` for float32,
float16 and bfloat16 over the tile shapes ``BlockMaskedMM`` selects, with
every pairing of a boolean, multiplicative or absent output and operand mask
except no masks at all, over one and two batch dimensions. Software subgroup
convergence of early returns now also handles a later
``if (c) { ...; return; }`` whose statements before the return neither
synchronize nor leave, as the masked kernels' exits for masked-out output
blocks are: invocations taking it run those statements and skip every later
statement except barriers and collectives. A branch on a folded template
constant around such an exit is converged in place or dropped,
``uniform<int>`` bounds of a later barrier loop are computed by every
invocation, and predicated ``bool``, half and bfloat16 declarations start
from ``false`` or ``T(0)``. The Metal round trip restores ``_NoMask``'s
literal ``constexpr operator bool()`` members, which Metal needs for
statically dead ``bool(mask[i])`` branches. Metal host reflection reads past
non-virtual member function definitions, which leave a struct's layout
unchanged, so absent masks bind as one-byte ``_NoMask`` records on Metal and
as 32-bit records on OpenGL and Direct3D, which widen ``char`` to ``int``.
Each target executes 1,584
dispatches over short, ragged and aligned cases with contiguous and
``[2, 2]`` batches whose masks have their own batch strides. A thread-level
model of both kernels must reach the masked product computed directly; mask
scales of 1, -1 and 2 and small integer inputs keep every result exact, so
every word is compared exactly. The model justifies OpenGL's batch-offset
index-range preconditions load by load, and Metal also runs the unchanged
upstream kernels, which the translation must reproduce exactly. This is a
528/17,832 increment; it does not run the upstream MLX test suite or redirect
the MLX host runtime.

``copy.current-tree.translation.json`` pins all 2,496 entries of
``copy.metal``: every layout over all 169 pairings of MLX's 13 dtypes and the
same-type general copies with destination strides and dynamic offsets. The
copy sources are byte-identical at the corpus commit and the shared headers
only gained declarations the copy kernels do not use, so each entry must
reproduce the Metal, OpenGL and DirectX artifact its complete-family
translation contract pins there. Each target executes 4,992 dispatches over
tails, short last tiles, strided, transposed, permuted and broadcast layouts
of up to five dimensions, padded destinations and dynamic offsets. Read
elements start at each source type's boundary values, and a model of
``cast_to<U>(T)`` gives every expected element: same-type copies are
bit-exact, floating conversions round to nearest even and overflow to
infinity, floats truncate into integers, integers wrap, and a NaN only has
to stay a NaN when the types differ. Casts C++ leaves undefined are kept out
of the inputs. Padding, trailing guards and inputs must be unchanged, and
Metal also runs the unchanged upstream kernels, which the translation must
match byte for byte. This is a 2,496/17,832 increment; it does not run the
upstream MLX test suite or redirect the MLX host runtime.

``binary.current-tree.translation.json`` pins all 4,122 entries of
``binary.metal``: every scalar, vector, two-dimensional and strided layout for
all 24 operators over the dtypes each instantiates. Only complex ``Power``
changed at the corpus commit, so its 15 entries reproduce artifacts this
contract pins and every other entry reproduces the Metal, OpenGL and DirectX
artifact its complete-family translation contract pins. Each target executes
8,244 dispatches. Operands come from per-operator domains in which every pair
is defined in C++ and exactly modelled: integers wrap after promotion with
undefined pairs kept out, floating results need no rounding and stay normal
next to signed zeros, infinities and NaNs, float16 remainders follow Metal's
binary16 ``fmod`` (``x - trunc(x / y) * y`` in float32), ``LogAddExp``,
``ArcTan2`` and ``Power`` use only exactly defined values, and complex operands
follow ``complex.h`` with exact intermediates. Guards and inputs must be
unchanged, and Metal also runs the unchanged upstream kernels, which the
translation must match byte for byte. Rounding and subnormal behaviour stays
with the per-operator gates. This is a 4,122/17,832 increment; it does not run
the upstream MLX test suite or redirect the MLX host runtime.

``unary.current-tree.translation.json`` pins all 877 entries of
``unary.metal``: every layout for all 37 operators over the dtypes each
instantiates, complex64 and the FP8 conversions. ``Sigmoid``, the FP8
decoding and the complex operators built on ``hypot`` changed at the corpus
commit, so their 75 entries, with the five ``uint32`` ``Sign`` kernels that
carry the complex overload, reproduce artifacts this contract pins; every
other entry reproduces the Metal, OpenGL and DirectX artifact its
complete-family translation contract pins. Each target executes 1,754
dispatches. Inputs come from per-operator domains whose results are exactly
defined: integer and rounding operators take every boundary value C++
defines, FP8 conversions cover all 256 codes with their bit algorithm, and
transcendental operators take only signed zeros, infinities, NaN, poles,
domain errors and the overflow, underflow and saturation thresholds. Complex
operands follow ``complex.h`` and ``cexpf.h`` with exact intermediates.
Guards and inputs must be unchanged, and Metal also runs the unchanged
upstream kernels, which the translation must match byte for byte. Accuracy
away from those points stays with the per-operator gates. This is an
877/17,832 increment; it does not run the upstream MLX test suite or redirect
the MLX host runtime.

``reduce.current-tree.translation.json`` pins all 2,396 entries of
``reduce.metal`` and their 7,188 artifacts: ``init_reduce`` for 40 operator
and type pairs and, for 62 operator, input and output type triples,
``all_reduce``, ``row_reduce_simple``, the small and looped row kernels and
the small, long-column, looped and two-pass column kernels, the last six with
32- and 64-bit indices over one, two and five reduction dimensions.
``reduce.metal`` is unchanged at the corpus
commit; ``reduce_col.h`` now adds ``IdxT(lid.x)`` to the long-column offset,
which OpenGL's index-range precondition follows. Each entry translates with
the threadgroup ``reduce.cpp`` launches for the gate's geometries, and the
six kernels that call SIMD-group operations lower them to 32-lane software
subgroups on OpenGL and DirectX, with ``native-precise`` binary32 addition and
own-value out-of-range relative shuffles on DirectX. Software subgroups now
reduce binary16 products, which OpenGL computes node by node in binary32 and
rounds to binary16, and 16-bit integer minima and maxima, which DirectX
compares in native registers. Each target executes 2,830 dispatches over
single and two-pass ``all_reduce`` rows, rows reduced by one thread and by one
SIMD-group, a ``row_reduce_simple`` grid whose last threadgroup moves back,
and column reductions with partial tiles and the ``[32, out_size]``
intermediates, with decoys between rows and blocks. A step-by-step model of
each kernel, checked against a left fold of each output, gives the expected
words. Inputs of types narrower than eight bytes give the same words in every
association order, as Apple's SIMD-group intrinsics leave the order
unspecified; eight-byte types follow the kernel's explicit shuffle order.
Every word must match, a NaN result accepts any NaN, and guards and inputs
must be unchanged. Metal also runs the unchanged upstream kernels, which the
translation must match byte for byte. This is a 2,396/17,832 increment; it
does not run the upstream MLX test suite or redirect the MLX host runtime.

``quantized.current-tree.translation.json`` pins 108 of the 2,052 entries of
``quantized.metal``: ``affine_quantize`` and ``affine_dequantize`` for float,
float16 and bfloat16, group sizes 128, 64 and 32 and 2, 3, 4, 5, 6 and 8 bits.
Quantize's ``simd_min``, ``simd_max`` and ``simd_shuffle_down`` lower to
32-lane software subgroups on OpenGL and DirectX, Metal's ``round`` keeps its
halfway-away-from-zero rule, and ``uint8_t`` buffers widen to one 32-bit word
per byte there. Each target executes 216 dispatches on 64-thread workgroups,
a row of workgroups and a two-by-two grid that exercises the
``threads_per_grid`` row stride. Quantize groups make every scale, edge and
code exact, except an all-zero group (the ``eps`` scale), an all-negative
group (``w_max`` starts at zero), a positive constant and free dyadic values
that need correctly rounded float32 division. DirectX quantization selects
``binary32_division_profile = "rne-flush"`` to prevent rounded reciprocal
multiplication from changing codes under DXC optimization. A lane-level model
must write
every byte once and reproduce MLX's least-significant-bit-first code stream;
dequantize's power-of-two scales and biases ``-a * scale`` keep every output
exact in the element type. Every word is compared exactly, guards and inputs
must be unchanged, and Metal also runs the unchanged upstream kernels. Its
matrix-vector and matrix-matrix kernels are outside this gate. This is a
108/17,832 increment; it does not run the upstream MLX test suite or redirect
the MLX host runtime.

``quantized_qmv.current-tree.translation.json`` pins a further 162 entries of
``quantized.metal``: ``affine_qmv_fast`` with and without batching and
``affine_gather_qmv_fast`` for the same types, group sizes and bit widths.
``simd_sum`` lowers to 32-lane software subgroups on OpenGL and DirectX, and
OpenGL is given the batch offsets, ``elem_to_loc_broadcast`` offsets and
gather index subscripts as index-range preconditions the model justifies.
Each target executes 378 dispatches on ``quantized.cpp``'s ``(32, 2, 1)``
threadgroups and ``(M, N / 8, B)`` grids: unbatched vectors and matrices,
stacked, broadcast two-dimensional and strided shared-weight batches, and
one- and two-dimensional gather index batches. Quarter inputs, power-of-two
scales and biases ``-a * scale`` keep every float32 intermediate exact in
any order, so only the final ``static_cast<T>`` rounds, to nearest even and
including halfway cases. A lane-level model of ``load_vector`` and ``qdot``
must reach direct dequantized dot products; every word is compared exactly,
guards and inputs must be unchanged, and Metal also runs the unchanged
upstream kernels. This is a 162/17,832 increment; it does not run the
upstream MLX test suite or redirect the MLX host runtime.

``quantized_qmv_unaligned.current-tree.translation.json`` pins a further 162
entries of ``quantized.metal``: ``affine_qmv`` with and without batching and
``affine_gather_qmv``, which ``quantized.cpp`` picks over the ``_fast``
kernels when ``N`` is not a multiple of 8 or ``K`` of
``qmv_fast_k_alignment(bits)``. ``qmv_impl``'s guarded rows, its
simdgroup-uniform return before ``simd_sum`` and its moved-back last tile run
on 32-lane software subgroups on OpenGL and DirectX, and DirectX bounds the
private pointer that ``qdot_safe`` advances by seeding helper parameters
with the intervals of their call arguments. Each target executes 486
dispatches on ``(32, 2, 1)`` threadgroups and ``(M, ceil(N / 8), B)`` grids
that reach every path, including a ``K`` below one block and the safe tail
of each lane. A lane-level model must reach direct dequantized dot products
and store each moved row twice with the same value; every word is compared
exactly, guards and inputs must be unchanged, and Metal also runs the
unchanged upstream kernels. This is a 162/17,832 increment; it does not run
the upstream MLX test suite or redirect the MLX host runtime.

``quantized_qmv_wide.current-tree.translation.json`` pins a further 432
entries of ``quantized.metal``: ``affine_qmv_wide`` with and without batching
for the same types, group sizes and bit widths and tiles of 2 to 5 input
vectors, eight lanes per output row. Its ``simd_shuffle_down`` ladder lowers
to 32-lane software subgroups on OpenGL and DirectX, and OpenGL is given the
batched index-range preconditions of ``qmv_fast``. A by-value template
parameter deduced from a local array, such as ``dequantize``'s output
pointer, now decays to a thread pointer for DirectX as it does for OpenGL, so
its pointer steps move the view's base. Each target executes 1,080 dispatches
on ``quantized.cpp``'s ``(32, 2, 1)`` threadgroups and
``(ceil(M / vecs_per_tg), ceil(N / 8), B)`` grids, with ``M`` values
``qmv_wide`` gives each tile: a single tile with two groups for some lanes,
partial row groups and vector tiles, and stacked, broadcast two-dimensional
and strided shared-weight batches. The inputs keep every float32
intermediate exact, so only the final ``static_cast<T>`` rounds. A lane-level
model of ``dequantize``'s sub-chunks, the shuffle ladder and the partial-tile
guards must reach direct dequantized dot products; every word is compared
exactly, guards and inputs must be unchanged, and Metal also runs the
unchanged upstream kernels. This is a 432/17,832 increment; it does not run
the upstream MLX test suite or redirect the MLX host runtime.

``quantized_qvm.current-tree.translation.json`` pins a further 162 entries of
``quantized.metal``: ``affine_qvm`` with and without batching and
``affine_gather_qvm``, which ``quantized.cpp`` picks for weights that are not
transposed. ``qvm_impl`` gives each lane one input row per block of 32 and
reads that row's 32 codes through a private byte view of its ``vec_w``
record; ``simd_sum`` runs on 32-lane software subgroups on OpenGL and
DirectX. DirectX emits a private pointer helper only for the backings its
calls reach, so the byte view of ``qouter`` is not shadowed by an identical
default overload. Each target executes 432 dispatches on ``(32, 2, 1)`` threadgroups
and ``(M, ceil(N / 64), B)`` grids that reach a ``K`` below one block, whole
blocks, a last pass on some lanes and the early return of a simdgroup past
``N``. A lane-level model must reach direct dequantized dot products and
store each column once; every word is compared exactly, guards and inputs
must be unchanged, and Metal also runs the unchanged upstream kernels. This
is a 162/17,832 increment; it does not run the upstream MLX test suite or
redirect the MLX host runtime.

``quantized_qvm_split_k.current-tree.translation.json`` pins a further 108
entries of ``quantized.metal``: ``affine_qvm_split_k`` with ``split_k`` 8 and
32, which ``quantized.cpp``'s ``qvm`` picks for weights that are not
transposed when ``K`` is 1,024 or more. Each ``tid.z`` runs ``qvm_impl`` on
one part of ``ceil(K / split_k)`` rows, the last of the final block size,
with the whole ``K`` as the activation stride, and writes its partial
products to an intermediate. Each target executes 540 dispatches on
``(32, 2, 1)`` threadgroups and ``(M, ceil(N / 64), B * split_k)`` grids
that bind the host's own views and reach parts of whole blocks, of blocks
and a last pass, of a last pass alone and a final part of another size. The
qvm gate's lane-level model replays each part; every word is compared
exactly, guards and inputs must be unchanged, and Metal also runs the
unchanged upstream kernels. This is a 108/17,832 increment; it does not run
the upstream MLX test suite or redirect the MLX host runtime.

DirectX now gives ``p + n``, ``n + p`` and ``p - n`` the element type of the
pointer or buffer view ``p`` rather than of the integer offset, so values read
through an offset pointer such as ``(v_ + dv_idx)[i]`` keep their float,
half or bfloat16 encoding; this re-pins the 18 fused gated-delta DirectX
artifacts.

All 24 entries use checked metadata rather than trusted index-range assertions.
The runtime validates six conditions against uploaded shape, stride and size
values. Each numerical case checks 32 output guards and observes seven inputs
for modification. Input allocations include inactive-lane pointer formation and
the final pointer update, preserving the original rows and expected indices.
Bindings are matched through source-resource provenance and checked against
their physical ABI. Windows retains the executed DXIL and packed bindings before
dispatch. These checks share the existing OS jobs; the two-entry exploratory
runtime path is no longer executed separately.

The following reviews predate the checked-metadata artifact refresh. Their
Windows execution results are historical, not qualification of the refreshed
artifacts; native Windows execution must pass again.

The earlier OpenGL identities were reviewed against retained sources after
signed 64-bit remainder and narrow-value lowering changed. All 24 entries
compile and validate, and the 16 required float32 execution cases pass on Mesa
EGL. A separate local review covers 80 half/bfloat cases with source-rounded
inputs; it does not expand the required CI runtime subset. A separate Metal
identity review covers native narrow aggregate fields and contextual byte-field
initializers. All 24 generated entries compile with ``-Werror``. The eight changed
bodies retain their source, specialization and execution metadata, while the
16 other bodies are unchanged. The four byte entries also match unchanged
upstream Metal execution and independent integer indices in 64 local cases
covering extrema, ties, constant rows, tails and strides. Required CI runtime
coverage includes the float32 and integer subsets described above.

All 24 DirectX identities have complete-body review and warning-fatal DXC
compilation. Reviewed changes make source-width index conversions and byte
conversion boundaries explicit, preserve the negative-infinity sentinel, and
decode logical half values from integer storage. Source hashes, template
materializations, resource registers, launch geometry, reduction comparisons
and tie-breaking are unchanged. The two float32 entries retain the required
32-lane software subgroup path. The existing Windows numerical gate remains
required and passed all 16 float32 cases with those reviewed shaders. A separate
audit verifies all 32 reduction indices against retained uploaded input words,
including strides, tails, ties and non-finite values, and checks parameter
bindings and executed module identities. This does not expand numerical
coverage to bool, byte, half or bfloat entries or establish upstream-suite parity.

Constrained Metal free-function fallbacks are materialized only when their
recognized constraints select a unique implementation. A visible ordinary or
unconstrained-template overload with compatible argument count causes a
structured specialization diagnostic when precedence cannot be proven;
forward declarations participate even without a definition. General C++
conversion ranking and template partial ordering are not implemented by this
path. Use a distinct helper name or an explicit specialization to remove the
unresolved competition rather than relying on a guessed overload.
Generated helper names reserve existing source identifiers, including local
variables and parameters. A deterministic suffix avoids name collisions while
repeated calls to the same specialization reuse one helper.

Global explicit free-function specializations are selected after the constrained
primary is resolved; they are not competing overloads. Selection compares
canonical template arguments and concrete parameter signatures, including
visible aliases and deduced or defaulted arguments. A matching specialization
keeps its own body, while an unrelated specialization does not displace the
primary. Ambiguous canonical identities and unproven ordinary-overload
precedence still produce diagnostics. Namespaced explicit specializations remain
outside this global selection path. Required Metal, DirectX and OpenGL execution
tests distinguish primary addition from specialized subtraction and verify both
default and non-default template arguments.

Kernel translation omits an out-of-line call-operator definition only when its
unique unqualified owner has no references outside its own declaration and
call-operator definitions. This prevents unused, unmaterialized helper bodies
from blocking arithmetic-profile validation. Library-only sources, aliases,
template references, nested calls, exported definitions and uncertain owner
bindings retain their bodies. Source offsets and line breaks are preserved.
Required native checks distinguish the selected qualified implementation from
its template fallback; no reachable computation is replaced by a placeholder.

The two current-tree DirectX float32 entries explicitly select
``software_subgroup_width = 32``. Their shuffle helpers use shared storage and
workgroup barriers instead of hardware wave instructions. A private
per-invocation index initialized by the entry point preserves lane identity
through helper calls. Early returns are accepted only under proven
workgroup-uniform control flow; a subgroup-ID condition is uniform only when
the workgroup contains one logical subgroup. Writes, shadowing, and mutable
helper arguments invalidate the corresponding uniformity assumptions.
Other DirectX entries retain the native-wave compiler path. Both paths still
require DXC validation, and the numerical entries retain every existing case
and the required Windows WARP execution gate. Compilation alone does not
establish numerical parity or resolve a native-runtime timeout.

Explicit-width software subgroups on DirectX and OpenGL also support scalar
Boolean all/any votes. The native host workflow requires compiler validation and
device readbacks across multiple logical subgroups, repeated calls, single
operand evaluation and untouched output guards. macOS additionally executes
the original Metal source. Divergent exits before a later collective are
rejected rather than dropping the exit or forcing inactive lanes into the vote.
OpenGL resolves unconditional wrapper chains to their exact collective helper
overloads. Entry-point calls may occur in proven uniform loops or branches
controlled by immutable workgroup-uniform inputs. Recursive chains, ambiguous
root overloads, lane-dependent branches and escaped or mutated uniform inputs
remain diagnostic. The native vote gate exercises three wrapper levels and
different predicates in even and odd workgroups; shared scratch accesses use
explicit memory ordering as well as execution barriers. This does not establish
complete MLX reduction support: collective-result uniformity and full host
reduction dispatch remain separate work.

Metal ``simdgroup_barrier(mem_flags::mem_none)`` also participates in the
software subgroup convergence proof on DirectX and OpenGL, including direct
calls, barrier-only helpers and repeated finite loops. Workgroup synchronization
is emitted only when participation is proven; it is not an unconditional
replacement for a native subgroup barrier. Native tests keep explicit atomic
memory fences separate from execution barriers, exchange values within four
logical subgroups, and check repeated scratch reuse, a uniformly inactive
workgroup and untouched output guards. The Metal round trip retains the
execution-only source flag. These controls do not establish support for every
Metal barrier flag or complete GEMM numerical parity.

Scalar float32, int32 and uint32 products use an adjacent-pair reduction tree
with strides 1, 2, 4, 8 and 16 on both software backends. Integer products wrap
at 32 bits. The required native product gate checks repeated helper calls,
single operand evaluation, overflow, signed zeros, infinities, NaNs and
order-sensitive rounding against original Metal and independent references.
Input words, readbacks, output guards and validation modules are retained.
NaN payloads are not required to match. This defines the software reduction
order; it is not a claim of identical floating-point results for every possible
native hardware reduction order or denormal mode.
Narrow products remain diagnostic until their intermediate rounding is
preserved, even where a target normally uses a 32-bit carrier for that type.

Scalar binary16 sums use the same adjacent-pair tree, rounding every combine
to binary16 rather than accumulating in binary32 and narrowing only the result.
DirectX uses native 16-bit scratch and precise arithmetic; OpenGL keeps a
distinct logical-half helper with explicit binary16 rounding in its float
carrier. Partial groups include only existing lanes. The existing native
partial-subgroup gate checks nested helpers, scratch reuse, single operand
evaluation, cancellation, signed zeros, subnormals, overflow, infinities and
NaNs, with original Metal comparison and input/output guards. NaNs are compared
by classification, not payload identity. The software tree's order is explicit;
this does not establish identical rounding on every possible hardware reduction
implementation or support for narrow products and prefix scans.
Lane-masked half sums remain diagnostic: a conditional source call does not
yet establish which lanes the native source compiler keeps active. This is
separate from a final partial subgroup caused by the dispatch size.

The same explicit software mode supports inclusive and exclusive prefix sums
and products for scalar float32, int32 and uint32. Prefixes follow increasing
logical lane order within each 32-lane partition, including a final partial
partition. A five-stage block-prefix tree combines the lower half's prefix
into each upper-half lane at strides 1, 2, 4, 8 and 16. Exclusive results
read the preceding lane's completed prefix; inclusive results combine that
exclusive result with the current operand. Integer arithmetic wraps at 32 bits.
Floating-point combines round separately to binary32, with negative zero as
the exclusive sum identity and one as the product identity. This is an explicit
software accumulation order, not a guarantee that every native device chooses
the same order or denormal mode.

The native product job also checks scans on Windows, Linux and macOS, retaining
input words, all output words, guards, and original Metal comparisons. Cases
exercise repeated helper calls, single operand evaluation, signed zeros,
non-finite values, overflow, cancellation, and order-sensitive products.
Narrow floats, wide integers, vectors and Boolean scans remain diagnostic;
a wider carrier does not justify changing source-width rounding. Unproven
barrier participation remains rejected. These checks do not establish complete
MLX cumulative-operation host routing or unchanged upstream-suite parity.

Pure scalar helpers of the form ``if (subgroup_vote) return fallback; return
subgroup_reduce(value);`` can converge both collectives before selecting each
logical subgroup's result. This retains different subgroup votes within one
workgroup; it does not reclassify them as workgroup-uniform. The lowering requires
safe by-value operands and rejects memory loads, effectful calls, mutation and
unproven arithmetic. Direct votes, negated votes and immutable Boolean locals
are supported, including unused method receivers. Native-mode branches remain
unchanged. Required native tests compare generated outputs with independent
references and original Metal compiled without fast math, retaining exact
finite results, NaN classification, repeated calls and guards. Full MLX host
reduction planning and arbitrary collective control flow remain separate work.

A scalar declaration initialized by a pure collective wrapper may
also move outside a proven subgroup-ID branch. Its input expressions are
evaluated under the original guard, including side effects in unused scalar
arguments. The wrapper executes for every workgroup invocation with private
zero inputs for disabled subgroups, followed by the guarded source stores.
This preserves collective inputs without speculating memory accesses or
treating subgroup uniformity as workgroup uniformity. The proof requires an
immutable, unescaped subgroup ID, private scalar state and no else branch
around the original call. Pure comparison helpers and checked finite loops
are supported. Positive multiplicative and left-shift induction must preserve
the source integer range, including the terminating update. Unchecked runtime
bounds, speculative integer overflow and floating-to-integer conversions
remain diagnostic; representable literal integer identities may be narrowed.
Resource-reading wrappers, scalar reference effects, ambiguous
overloads and nested collective arguments remain diagnostic. Existing
convergence analysis still checks the transformed program. Required native
controls retain original Metal comparisons, partial groups, inactive groups,
repeated calls, input readbacks and output guards.

The wrapper proof also supports flat private structs of scalar fields. It
tracks straight-line constructor field initialization, rejects reads of
uninitialized fields or incomplete whole values, and preserves typed zero
initializers for inactive subgroups. Nested overloads require a unique
compatible signature. Storage-qualified or reference parameters, resource
effects and unsupported aggregate layouts remain diagnostic. Complete pinned
complex scans have separate native demo controls at 64, 128 and 256 threads;
those controls do not establish full cumulative-operation host routing.
HLSL aggregate values used inside expressions are constructed through typed
helpers, since HLSL does not support struct constructor calls or compound
literals. Helper names avoid source declarations and are reset for each
generated program; native controls retain field argument order and single
evaluation, and strict compiler checks cover the complete Metal source path.

Pure wrapper expressions may also call unshadowed scalar binary32 ``min``,
``max``, ``exp`` and ``log`` helpers, and divide floating-point operands. The
proof checks argument types, arity and dependencies; it does not assume that
an owned function with the same name is a builtin. Stateless local receivers
must have no fields, inheritance, generic parameters, attributes or initializer.
Integer division, speculative memory accesses and floating-to-integer
conversions remain unsupported. Disabled subgroups may produce unused NaNs or
infinities, but retain the same barrier schedule and cannot publish their
results. Native controls compare collective results with lane-local evaluation
on the same device, including exceptional inputs and partial groups. These
controls establish scheduling and guarded effects, not cross-device
transcendental accuracy or complete log-add-exp scan parity.

The bounded Windows test also records live Direct3D 12 debug-layer messages
through ``tools/run_directx_diagnostics.py``. Each entry's JSON-lines log is
stored outside pytest's temporary directory and uploaded with the runtime
evidence, including on timeout. Missing debug-layer tooling is reported
explicitly and does not skip or replace the numerical test. The wrapper
preserves the wrapped Python module's arguments and exit status.
After device creation it records the loaded Direct3D and WARP DLL paths and
SHA-256 hashes, independently of the runtime installation log. The Windows
project-porting job pins WARP 1.0.21 with an archive checksum. Its
`release notes <https://www.nuget.org/packages/Microsoft.Direct3D.WARP/1.0.21>`_
describe revised uniform and divergent control-flow handling; numerical tests
remain the acceptance gate for the runtime update.
The current DirectX kernel test additionally saves the actual runtime-loaded
DXIL and the final packed register payloads immediately before each dispatch in
``native-dispatches``. This distinguishes the executed module and bindings from
the standalone compiler check's artifacts, including when dispatch never returns.

Before the full kernels, Windows also executes eight small translated reduction
checks: sparse-register metadata reads, signed 64-bit index division, helper-level
software shuffles, their combination with a bounded input loop, NaN-aware pair
comparisons, non-finite inputs, a second shared-memory reduction stage, and
private-array accumulation. Each runs in a separate process with a two-minute
deadline and exact readback comparisons.
The diagnostic logs, CrossGL input, generated HLSL, DXIL, and completed readbacks
are retained under ``mlx-current-results`` and uploaded before the full kernels
run. All eight checks are attempted and any failure fails the step. These isolate
runtime failures; they neither replace the MLX numerical cases nor establish
whole-kernel parity.

The current-pinned MLX integration exercises entry-scoped translation for all
877 discovered entries from the include-expanded ``unary.metal`` source. The
finite split is 183 each for ``v_``, ``v2_``, ``gn1_``, and ``gn4large_``, plus
145 ``vn_`` entries, spanning 37 operators, 20 concrete input/output type pairs,
and 16 semantic families. ``v_`` uses explicit ``N=1``; ``v2_`` and ``vn_``
retain ``N=WorkPerThread<T>::n`` from the source default; ``gn1_`` uses
``N=1, IdxT=int``; and ``gn4large_`` uses ``N=4`` plus the source-default
``IdxT=int64_t``.

Every independently translated artifact has one selected operator implementation
and one kernel. Vector shapes materialize one specialization and reflect input,
output, and size resources. Gather shapes additionally materialize the reachable
``elem_to_loc<int>`` or ``elem_to_loc<int64_t>`` helper and reflect constant
shape/stride buffers plus a read-only device dimension binding. The resulting
877 artifacts contain 1,243 exact materializations, preserve a host-owned
``[1, 1, 1]`` workgroup contract, and reject residual templates, ``decltype``,
call operators, unsupported placeholders, and non-selected operator bodies.

Required macOS CI compiles all 877 exact artifacts with
``xcrun -sdk macosx metal -Werror -c`` and requires a non-empty AIR output from
each with no warning exemption. The generic path resolves source typedef chains
and bfloat reconstruction, materializes source-compatible constrained free
operators, infers aggregate aliases, recognizes branch-complete returns,
preserves narrow ``as_type`` storage, and admits only a proven read-only
immediate scalar-parameter view as a matching thread-local single-field
aggregate. The non-scalar path also retains
constant-resource provenance, const-device references, postfix update position,
and native Metal union aliasing while ambiguous address spaces and unsupported
reinterpretation continue to fail closed. This complete discovered-unary
selected-entry proof does not claim Metal numerical execution, host-runtime
redirection, or the MLX test suite.

The same entry-scoped pipeline translates all 877 historical unary entries to
standalone OpenGL ``main`` artifacts at revision
``846d176227a0ac13d2667e58d2bb68b322109ab0``. The schema-v2
``unary.opengl-translation.json`` contract preserves the same five-shape,
37-operator, 20-type-pair classification and all 1,243 exact materializations,
while pinning 5,316,552 generated GLSL bytes and all 3,363 target-reflected
resources. Vector artifacts expose read-only input, read-write output, and an
entry-scoped size uniform block. Gather artifacts expose input, output, shape,
stride, and the read-only ``ndimBuffer`` storage resource; scalar uses of the
source ``device const int& ndim`` alias element zero without changing expression
typing.

OpenGL's 32-bit index profile cannot represent every source 64-bit index
implicitly. The complete contract therefore records explicit host/runtime
bounds of ``[0, 2147483647]`` for ``offset + i``, ``out_idx++``, and ``idx``.
These are declared portability preconditions, not inferred facts or generated
runtime checks; absent proof continues to fail closed. GLSL also has no native
``log10`` overload, so reachable Metal base-ten logarithms lower through a
single evaluation of ``log2(value) * 0.3010299956639812`` while user-defined
``log10`` functions remain ordinary calls.

Required CI partitions the family across five disjoint Linux shards. Each shard
retranslates its exact entries, verifies generated identity, materialization,
``main`` workgroup metadata, and reflected ABI, compiles with
``glslangValidator --target-env opengl --target-env spirv1.3 -S comp``, validates
with ``spirv-val --target-env spv1.3``, and requires every SPIR-V module to be
non-empty. This closes whole-family OpenGL translation, reflection, and native
compiler coverage; it does not claim OpenGL numerical execution, MLX host
runtime redirection, or MLX test-suite parity.

The same entry-scoped pipeline translates all 877 unary entries from the legacy
reference revision ``846d176227a0ac13d2667e58d2bb68b322109ab0`` to standalone
DirectX ``CSMain`` artifacts. The schema-v2
``unary.directx-translation.json`` contract retains the exact five-shape,
37-operator, 20-type-pair classification and 1,243 materializations. ``v_`` and
``vn_`` artifacts reflect input, output, and entry-scoped size resources;
``v2_`` artifacts additionally reflect the generated ``CrossGLDispatchInfo``
workgroup-count cbuffer at ``b3``. Gather artifacts reflect input, output,
shape, stride, and read-only ``ndim`` structured buffers plus their generated
``CrossGLDispatchInfo`` cbuffer at ``b0``, for 3,912 reflected HLSL resources
in total. Source scalar uses of
``device const int& ndim`` alias ``ndim[0]`` and source ``out_idx++`` remains a
postfix update.

The reviewed HLSL references total 5,159,617 bytes. The earlier complete review
compared all 877 artifacts against hash-verified originals and compiled them
with warnings fatal, updating 603 identities and retaining 274. A subsequent
review recompiles all 34 Exp and Sigmoid entries and updates 20 identities:
15 scalar Exp entries preserve precise exponential calls and five bfloat
Sigmoid entries preserve source arithmetic boundaries. The remaining 14
selected bodies are unchanged. Source pins, materialization, ABI and launch
contracts are unchanged. Five unsigned 32-bit Sign identities additionally
retain a precise square-root helper in an unused complex overload; their
warning-fatal DXC modules are byte-identical before and after the change.
Current-pin Windows native execution separately covers all five Sign layouts,
3,303 output values and 40 output guards. The separate required Windows Square
and ArcCos numerical tests keep their existing artifact identities and tolerances.
This compiler contract does not establish numerical parity for all 877 entries.
Historical Sigmoid uses the default exponential and no optional division
profile; the current-pin, explicitly profiled numerical proof is separate.

The latest complete source and interface review accounts for every remaining
unary identity change. Of the 875 changed HLSL sources, 756 retain byte-identical
DXIL; 119 contain reviewed math or conversion corrections. Two sources are
unchanged. The GLSL review covers 134 changed math and FP8 artifacts and 743
unchanged sources. Every current artifact compiles with strict validation;
math corrections are not numerical-equivalence claims. Entry coverage,
materialization, resource bindings and dispatch contracts remain unchanged.

DirectX bfloat support covers every unary intrinsic required by the family.
The inverse-hyperbolic ``acosh``, ``asinh``, and ``atanh`` paths decode bfloat
to float, invoke their portable float helpers, and round back to the exact
bfloat payload representation. Explicit contextual HLSL constructors preserve
float-to-native-16 return and initializer narrowing without ``-Wconversion``
diagnostics when warnings are fatal. All artifacts derive
``-enable-16bit-types`` from retained native 16-bit declarations and target
shader model ``cs_6_2``. As with OpenGL, explicit host/runtime bounds of
``[0, 2147483647]`` for ``offset + i``, ``out_idx++``, and ``idx`` remain
portability preconditions rather than inferred or generated checks.

Canonical Metal ``fabs``, ``fmin``, and ``fmax`` calls lower to HLSL
``abs``, ``min``, and ``max``. Boolean scalar/vector ``select`` conditions
use HLSL's ``select(condition, trueValue, falseValue)`` ordering, preserving
evaluation of both value arguments. Source-defined helpers keep their own
semantics. Unresolved or non-Boolean conditions, invalid argument counts, and
shadowed target intrinsics produce source-located diagnostics; integer-mask
selection is not inferred from a Boolean conversion. The HLSL frontend normalizes
condition-first selection before target lowering and rejects three-argument
calls when a same-named source overload makes ownership unresolved. A bounded Windows test
requires DXC compilation and Direct3D readbacks for finite values, NaNs,
infinities, vector selection, and eager evaluation. This isolated arithmetic
check does not establish whole-MLX numerical or host-runtime coverage.

Canonical ``atan2(y, x)`` uses typed HLSL helpers to retain the sign of zero
when selecting a quadrant. The helpers also handle both-infinite operands,
signed axis angles, and NaNs explicitly. Finite angles use exponent-scaled
significands and a range-reduced polynomial instead of the target intrinsic;
``precise`` intermediates preserve the polynomial evaluation order. Float
scalar/vector operands are supported, with
explicit promotion and narrowing for half/minimum-precision forms and the
existing scalar bfloat decode/encode path. Arguments are evaluated once.
Unknown, mismatched or unsupported operand types and shadowed target intrinsics
fail closed. Source-defined overloads retain their behavior, and HLSL-to-HLSL
``atan2`` round trips remain native. This is not a general subnormal or
transcendental-accuracy contract.

A bounded Windows readback test checks 1,121 operand pairs through scalar and
vector calls, including raw-bit and float-upload echoes, signed zeros,
infinities, NaNs, extreme normal exponents, all quadrants and range-reduction
boundaries. Finite results must stay within the unchanged absolute error
bound of ``2e-6``; signed axis and infinite angles retain exact bit checks.
macOS executes the unchanged Metal source with fast math
disabled as a control; fast-math compilation may ignore signed zeros and
non-finite values. The pinned MLX complex-power test separately retains its
existing numerical reference and error bound; passing an isolated angular
test does not replace that end-to-end proof.

The frozen binary and unary HLSL contracts at MLX revision
``846d176227a0ac13d2667e58d2bb68b322109ab0`` retain their source pins,
entry classifications, materialization counts and interfaces. The angular
lowering changes 84 binary and 28 complex-unary artifact identities; each
changed entry is recompiled with DXC and warnings fatal before refreshing its
fingerprint. The other 4,887 identities remain unchanged. This historical
contract refresh is separate from current-revision MLX runtime coverage.

OpenGL also lowers canonical ``fabs``, ``fmin``, ``fmax``, and Boolean
``select`` for desktop GLSL 4.00 and later. Floating min/max helpers explicitly
return the numeric operand when the other operand is NaN, in either argument
order. Selection uses typed helper parameters and per-component conditions,
without numeric interpolation or short-circuiting the source value arguments.
Both values must have matching scalar/vector types; the condition may be a
scalar Boolean or a Boolean vector of the same width. Integer masks, mismatched
types, unresolved operands, and shadowed required target builtins produce
structured diagnostics. Generated helpers avoid source identifiers, and
source-defined functions retain their own behavior. Required Linux CI compiles
the generated GLSL, validates its SPIR-V, and executes the source through the
OpenGL runtime, retaining readbacks for finite values, NaNs, infinities, signed
zeros, mixed vector masks, and argument side effects. This is a focused operation
contract, not evidence that every MLX kernel or its host integration works.

Floating ``fmod`` is distinct from floor-based ``mod``. OpenGL uses integer
significand division for binary32 and binary64 scalar and vector operands,
including scalar broadcasting. This avoids floating quotient overflow and
preserves representable remainders and signed zero. Zero divisors, infinite
dividends and NaN operands produce NaN; payload identity is not promised.
Binary64 requires target 64-bit integer support as well as double precision.
Unsupported operand shapes and profiles produce structured diagnostics.
Unprofiled half and bfloat calls remain unsupported until their source-specific
exceptional-value rules can be represented; they are not silently widened.
Required Linux execution compares raw output words with an independent rational
oracle, including subnormals, extreme ratios, exceptional values and guards.
This does not establish parity with every source compiler's fast-math profile:
the MLX controls expose half signed-zero and infinite-divisor differences and
bfloat subnormal-policy differences tracked in
`issue #2097 <https://github.com/CrossGL/crosstl/issues/2097>`_.

Metal sources can explicitly select
``binary16_remainder_profile = "binary32-quotient"`` in their source options.
This profile widens half operands exactly, rounds division to binary32 with
ties to even, truncates the quotient toward zero, separately rounds its
product and the subtraction to binary32, and narrows the result to half.
It is not exact mathematical remainder: quotient cancellation can change
finite results, negative zero becomes positive zero for a finite nonzero divisor,
and an infinite divisor produces NaN. Integer arithmetic helpers enforce the
selected rounding steps without target contraction or reciprocal approximation.
Scalar and two-, three- and four-component forms support scalar broadcasting
and single argument evaluation. Other floating types and source-owned overloads
retain their own behavior.

Source-defined ``asfloat``, ``asint`` and ``asuint`` overloads receive distinct
internal names before target generation. Their source calls retain namespace
and overload identity while generated arithmetic helpers retain builtin bitcasts.

The option is explicit, not inferred from the target OS or selected for every
Metal source. Project reports retain ``binary16RemainderProfile`` and reject
provenance that disagrees with the resolved options. Native controls compare
5,176 input pairs, ten result forms per pair, side-effect counters and exact
guards against a separately evaluated rounding model. Unchanged-original
Metal is a required macOS control; this does not promise that all Metal devices
or compiler modes implement the same profile. Half-vector component bitcasts
use their logical 16-bit width rather than the widened GLSL representation.

Metal sources can also select
``binary16_widening_profile = "canonical-nan"`` after characterizing their
source execution environment. Numeric half-to-float conversions then map every
binary16 NaN to positive binary32 ``0x7fc00000``. Finite values, subnormals,
infinities and signed zeros widen exactly. The option is not a universal Metal
default: other devices or compiler modes may require different policies.
It covers explicit scalar/vector constructors and implicit conversions in
initializers, assignments, function arguments, returns and conditional expressions.
Helper parameters preserve single evaluation. Reports and runtime artifact
manifests retain ``binary16WideningProfile`` and reject inconsistent provenance.
Unrepresentable global conversions fail with a structured diagnostic.

Bitcasts and same-type half copies are not numeric widening. OpenGL represents
these values with lossless binary32 carriers and uses integer bit manipulations
for scalar binary16 and packed binary16-pair bitcasts; floating pack/unpack
builtins are not used because they may change NaN payloads. Actual float-to-half
rounding remains a separate operation. Native conversion controls exhaust all
65,536 half encodings, check scalar/vector and implicit conversions, count
side effects, and verify exact output and guards with two initializations.
The unchanged Metal control characterizes the selected profile, not every
possible Metal implementation.

Metal normalization retains qualified math builtin ownership when source helpers
share the same name. Source overloads receive deterministic, collision-safe
internal names before namespace qualification is removed. Explicit global calls,
namespace-local helpers, imported names, materialized templates, and source-owned
extensions in ``metal`` remain distinguishable from the builtin overloads.
Existing overload diagnostics reject unresolved bindings instead of guessing.
Required bounded CI on Windows, Linux, and macOS compiles and executes a reduced
ownership fixture, preserving generated source, native modules, and readbacks;
macOS also executes the original Metal source for comparison. These focused
checks cover the regression in
`issue #1947 <https://github.com/CrossGL/crosstl/issues/1947>`_, not general C++
namespace conformance or whole-repository runtime integration.

Required CI partitions the family across five disjoint Ubuntu shards. Each
shard retranslates its exact entries, verifies deterministic identity,
materialization, ``CSMain`` workgroup metadata, and reflected ABI, then compiles
with pinned DXC using ``-enable-16bit-types -WX -T cs_6_2 -E CSMain`` and
requires non-empty DXIL modules. Together, the DirectX and OpenGL contracts
close complete discovered-unary translation, reflection, and native compiler
coverage on both targets; they do not claim numerical execution, MLX host
runtime redirection, or MLX test-suite parity.

**Ordered loop updates.**

Metal and OpenGL emit comma-separated ``for`` updates individually in source
order, matching DirectX. Updates remain in the loop header so ``continue``
executes them and ``break`` skips them. Prefix/postfix increments, dependent
compound assignments, empty and single updates retain their loop-local scopes.
Required three-platform native tests check these effects and guarded outputs.
Metal explicitly discards unused narrow-integer update results in expression
statements and ``for`` initializers and updates, without suppressing compiler
warnings. The stored conversion and update remain unchanged; values used by
assignments, arguments, conditions, returns and non-void tail expressions are
preserved. Strict Metal compilation and original-source numerical comparisons
cover signed and unsigned byte/short boundaries, vector and field updates,
ordered updates, loop exits and output guards. This does not extend support to
updates with unproven or side-effecting lvalues.

The pinned MLX ``gather_front<float, int, int, N>`` proof instantiates unchanged
upstream bodies for ``N`` equal to 1, 4 and 8. It checks negative and repeated
indices, empty slices, partial chunks and exact binary32 storage words, including
NaN payloads and signed zeros. The source wrapper only includes upstream headers
and declares template instantiations. This is kernel execution coverage, not
MLX Gather host integration or complete indexing support.

**Mixed-width integer arithmetic.**

The Metal frontend retains structured-buffer compound assignments as lvalues in
the intermediate representation, including side-effecting indices. It does not
expand them into a load/store pair that evaluates the index twice.

For built-in integer vector/scalar arithmetic and comparisons, the Metal
frontend explicitly converts the scalar to the vector element type before the
operation. Both operand orders, compound assignments and inferred local types
retain this source rule in saved CrossGL. Scalar pairs still use their usual
integer conversions, and shifts still promote their operands independently.
This does not change standalone CrossGL or native HLSL conversion rules.

Metal-to-HLSL lowering explicitly applies source integer conversions before arithmetic,
bitwise operations, comparisons and conditional selection. In particular,
``int64_t`` combined with ``uint`` uses signed 64-bit arithmetic, not HLSL's
implicit unsigned conversion. Vector/scalar pairs retain the vector element
type, and shifts preserve independent operand promotions, including compound
shifts with scalar counts and vector destinations. Compound assignments
convert before narrowing back to the destination. Before an indexed destination
is passed to a typed ``inout`` helper, effectful indices are captured in private
temporaries: DXC may otherwise evaluate a private-array index on both copy-in
and copy-out. Captures remain in the assignment expression, not at function
entry, preserving conditional selection, loop updates and returned values.
Nested arrays and structure-member arrays retain their destination identity.
A right operand that may modify a copied
destination receives a diagnostic rather than an unproven copy-in/copy-out
translation.

Native HLSL input retains its DXC integer conversion rules instead: the frontend
records explicit common operand types in CrossGL, including unsigned preference
across widths and scalar/vector conversions. Nested expressions, conditional
arms, aliases, uniquely resolved function results and compound assignments carry
those types through a saved intermediate file. Integer literal suffixes and
canonical 64-bit vector types are retained. Ambiguous wide return types and
unproven compound-assignment copy-in aliases remain diagnostic.

OpenGL signed remainder uses truncating division and multiplication rather than
relying on ``%`` for negative operands. Typed helpers evaluate operands once and
broadcast scalar operands to the result vector shape. Constant initializers
retain constant expressions. Unsupported side-effecting compound destinations
remain diagnostic. Zero divisors and signed-minimum divided by minus one are
outside the source-defined numerical contract.

OpenGL shifts convert 64-bit counts to matching signed or unsigned 32-bit scalar
or vector operands for native driver compatibility. This does not change the
promoted left operand or the result type. Compound shifts retain native indexed
lvalues and evaluate both the index and count once. Negative counts and counts
at least as large as the promoted left operand's bit width are outside the
source-defined contract; this lowering does not assign them portable semantics.

Required native CI checks scalar mixed-width arithmetic and scalar/vector signed
remainder on Windows, Linux and macOS, including original Metal controls,
unchanged inputs and guarded outputs. Wide shift counts cover both directions,
32-bit and 64-bit left operands, vector widths two through four, scalar
broadcasts and side-effecting binary and compound expressions on every target.
The original Metal shift controls use representable, nonnegative signed left-shift
operands and valid counts throughout. Vector/scalar arithmetic controls include
values outside the vector component range, both operand orders, comparisons,
inferred locals, nested operations and single-evaluation compound assignments.
The original Metal mixed-sign comparison controls disable only ``-Wsign-compare``
and record that flag; generated artifacts retain warning-fatal compilation.
Native HLSL controls additionally compare generated results on each target and
the original HLSL on Windows; optimized DXC checks cover direct and saved-CrossGL
round trips.
Twelve additional original/generated Metal and required Windows cases cover
eight compound operators on private arrays, nested arrays and structure-member
arrays in statement, expression, conditional and loop-update positions. Each
case checks modified elements, unchanged neighbors, evaluation counts, the
assignment result, input preservation and output guards. OpenGL retains an
explicit diagnostic for these unsupported effectful compound destinations.
These checks do not establish full MLX-suite parity.

**Precise scalar promotions.**

Metal's qualified precise sine, cosine, arctangent and inverse hyperbolic
cosine accept scalar ``half`` and ``bfloat`` operands by promotion to
``float``. Translation preserves that binary32 result, including inferred
``auto`` declarations, and evaluates each operand once. Narrow vectors still
require an explicit conversion to the corresponding float vector, matching
the source compiler's overload rules.

Native checks compare implicit and explicit promotions, output guards and
evaluation counts. Generated results retain the existing binary32 accuracy
and signed-zero requirements. The original Metal arctangent control uses its
separate documented zero/subnormal policy; that allowance does not apply to
translated output. This contract does not change unqualified half-math
overloads or establish full application parity.

**Binary32 negation.**

DirectX lowers unary minus on known ``float`` scalars and two- to four-lane
vectors by toggling the payload sign bit. This preserves subnormals, signed
zeros, infinities and NaN payloads without depending on the compiler's
floating-point denormal mode. Operand evaluation occurs once. Literal
constants retain constant-expression syntax; integer, narrow-float, double,
matrix and complex operations retain their separate lowering rules.

The existing native math job also checks scalar and vector negation, aliases,
structure members, array indexing and side-effecting helper calls against an
independent integer oracle. It retains raw input, expected and readback words,
compiler outputs, module hashes, evaluation counts and guards. This adds no
runner matrix and does not change the tolerance of the precise math checks.
Nested signs retain their grouping in Metal and HLSL instead of combining
into increment or decrement tokens. The same native job checks nested signs
on integer and binary32 scalars and vectors, genuine prefix/postfix updates,
numeric casts and side-effecting calls. It compares both results and final
operand values with independent word-level expectations; on macOS it also
executes the unchanged source as a control. OpenGL retains its existing
parenthesized expressions. These checks do not establish full application
parity or change the separate narrow-float and complex lowering rules.

**Source resource identities.**

Metal buffer parameters can retain their original entry-point name, parameter
name and parameter index through saved CrossGL, HLSL, GLSL and Metal output:

.. code-block:: toml

   [project.source_options.metal]
   preserve_resource_origins = true

The single-file API accepts the same boolean in ``source_options``. The option
is disabled by default. It applies to device and constant buffer pointers,
references and one-dimensional array parameters, not textures or samplers.
CrossGL stores ``@source_resource`` attributes; target declarations carry
versioned ``crosstl-resource-origin`` comments identifying the actual emitted
resource, including its entry point for Metal. Reflection rejects malformed,
duplicate, ambiguous and unmatched records. Package inspection also rejects
differences between declared identities and packaged shader metadata.

Reflected resources expose the identity at
``metadata.provenance.sourceResource``. Native loader bindings expose it at
``provenance.sourceResource``, allowing callers to select resources without
guessing generated names. The surrounding artifact supplies source path and
content hashes; parameter identities are not globally unique. These records
do not prove buffer bounds, authorize integer narrowing, validate uploaded
metadata, or establish that a resource remains immutable during execution.
Those checks require separate runtime contracts. Existing layout and native
runtime limitations still apply.

**Checked integer metadata.**

Project translation can attach integer bounds to an original constant-reference
parameter. These bounds are enforced by the Python native loader and native
Metal, DirectX and OpenGL dispatch adapters, unlike caller-trusted
``index_range_assertions``:

.. code-block:: toml

   [[project.metadata_preconditions]]
   source = "kernels/scan.metal"
   entryPoint = "scan_values"
   parameter = "axis_size"
   minimum = 1
   maximum = 65535

Enable ``preserve_resource_origins`` for the Metal source frontend. Each matched
translation must select one compute entry. A missing or ambiguous source
parameter, writable resource, vector or record produces a diagnostic. Without an
element selector, the contract requires a constant integer scalar reference.
An explicit ``element`` selects a zero-based element of a read-only integer
buffer. Multiple requirements may select distinct elements of the same buffer;
scalar and element requirements cannot be mixed for one resource:

.. code-block:: toml

   [[project.metadata_preconditions]]
   source = "kernels/copy.metal"
   entryPoint = "copy_values"
   parameter = "shape"
   element = 0
   minimum = 1
   maximum = 1024

Array element contracts require a reflected, unpacked scalar layout. The native
loader validates the selected element within the supplied allocation view, and
the adapter checks the same element in its owned upload snapshot. A missing
element fails before dispatch. Elements without a requirement remain unknown;
checking one array element does not bound its neighbors or the buffer offset.
OpenGL can bound direct metadata reads only when every possible selected index
has a checked element requirement. Shadowed or rebased resource names do not
inherit these facts. Record fields and unproved helper forwarding remain
unsupported.

Requirements are scoped by both the source pattern and the original entry name,
including template ``host_name`` entries. Explicit and discovered jobs, and
DirectX/OpenGL dispatch-plan jobs, retain only their own requirements;
automatic OpenGL entry splitting uses
the same selection. A wildcard source pattern applies to matching files that
actually contain the named entry, not to every kernel in those files. Discovery
also checks configured define variants when the base scan cannot find an entry.
Unknown entries and selected parameters still fail translation, and unmatched
source patterns produce configuration errors. Requirements for a known but
unselected entry are not applied to another artifact. Their parameter types and
bounds are checked when that entry is translated. Each emitted package still
requires the corresponding checked values at native dispatch.
Serial and parallel translation use the same project-wide entry ownership.

The generated ``crosstl-metadata-preconditions`` header records the original
resource identity, integer width, signedness and inclusive bounds. Reflection
matches it to the actual read-only scalar layout; package inspection rejects
changed or removed requirements. The native loader checks typed uploaded values
and rechecks descriptor coordinates and layouts against the artifact. Immediately
before native dispatch, the adapter checks the immutable upload bytes and records
the value, requirement, allocation view and payload digest in its execution
receipt. Offset views are supported; sharing the metadata allocation with any
other binding is rejected, including disjoint views and implicit allocation
identities. Values outside the range are rejected, never clamped.

DirectX and OpenGL software collective lowering can use these checked bounds to
prove fixed-stride integer loops terminate. The bound must be the original
immutable scalar parameter itself, with a constant initial value and stride.
The proof checks both range endpoints, zero-trip paths and the terminating update
against the counter's source integer width. Shadowing, mutation, volatile values,
divergent exits and unproved helper forwarding do not acquire range facts.
The loop remains runtime-controlled; the bounds are not substituted as constants.
Direct code-generator entry points also validate source identity and retain the
obligation header, and dispatch-region program identities include the contracts.

OpenGL index lowering also uses these immutable scalar bounds in source arithmetic,
local variables, guards and fixed-stride loops. Narrowing requires the computed
range to fit the target index type without source overflow. Wide pointer updates
and subsequent subscripts must also prove the complete signed logical offset;
proving the delta or subscript separately is insufficient. Unproved conversions
remain structured errors. Native controls cover direct and rebased reads at all
configured endpoints, unchanged inputs and guarded outputs.
Signed relative subscripts may be negative after a pointer rebase. They acquire
a narrowing proof only when checked bounds establish that the base and relative
offset each fit the signed target representation and their complete sum is a
non-negative index within the target limit and any known fixed extent. Trusted
assertions on an individual operand cannot establish this composed proof. An
explicit assertion on the complete pointer expression (``pointer + index``, with
``pointer + 0`` for a dereference) can replace a missing composed proof when its
range lies in ``[0, 2147483647]`` and the relative index is still representable
as ``int``. A wide update ``pointer += delta`` with a proved base likewise
accepts an assertion on ``delta`` or on ``pointer + delta``; it does not
describe later subscripts, which keep their own complete-offset obligations.
Without a proved base (a pointer advanced inside a loop), the update requires
both an assertion on ``pointer + delta`` in ``[0, 2147483647]`` and a proved or
asserted ``delta`` that fits signed ``int``. Negative
absolute indices, missing contracts, overflow, shadowing and escaped values remain
rejected; runtime allocation checks are still required.
A pointer base proved to be exactly zero remains an ordinary absolute subscript.
Existing explicit assertions retain their ``asserted`` status in that case; they
are not promoted to checked metadata or native execution evidence.
For rejected wide-index proofs, a range spanning negative and non-negative values
reports ``index-range-may-be-negative`` with ``rangeStatus: unproven``. A wholly
negative range reports ``negative-index`` with ``rangeStatus: out-of-range``.
The report retains the inferred bounds in either case; an over-approximation is
not presented as a demonstrated invalid input.
Pointer-offset proofs retain source identifier ownership when GLSL emission
renames a reserved word. A similarly named local variable cannot supply bounds
for that pointer; composed offsets still undergo overflow and scope checks.
Local-name allocation keeps emitted resource aliases reserved, including when
a local uses the preferred escaped spelling or shadows another local in a block.

Increasing unit-stride loop counters retain their initial value under conjunctive
bounds and affine conditions such as ``offset + i < size``. The offset and limit
must remain stable, and the initial expression, source arithmetic, implicit
conversions and terminating update must preserve the proved integer values.
Validated immutable scalar metadata references can provide these limits;
unchecked references, escaping state, unknown effects and possible wrapping do
not establish bounds. An upper trip bound only describes possible body-entry
values, not an exact sequence for pointer rebasing or a final counter value.
Native tail-copy cases exercise partial two-element tiles and three-dimensional
launches with tight inputs, unchanged metadata and output guards. Source loops
and tail conditions remain intact.

Pure integer-returning helpers can retain checked caller ranges
through nested calls, integer casts and integer arithmetic.
Analysis uses source overloads, parameter conversions and return conversions;
each callee has its own bindings. Fixed-width unsigned wrapping is represented
before signed conversion, while possible signed arithmetic overflow establishes
no proof. Unknown callees, ambiguous overloads, recursion, mutable or volatile
references, captured caller locals and unmodeled effects remain unproved.
Checked immutable integer-array elements can pass through these helpers without
becoming unchecked pointer-offset facts. Scalar- and vector-return helpers support
local scalar/vector assignments, conditional branches and finite ``for`` loops,
with a maximum of 64 iterations per loop and a shared expression budget. By-value
parameters have private state; their updates cannot change the caller's facts.
Ranges include every possible exit. Writes through reference or array parameters,
constant writes, escaping locals, uncovered elements and division by a possibly
zero value cannot establish a proof. Explicit early returns retain the value and
private state at each reachable exit; non-void fallthrough and unmodeled control
flow remain unsupported. Native controls compare expression, local, loop and parameter
forms with exact indexed readbacks, caller-value observations and untouched guards.
The pinned reversed one- and four-dimensional copy demos use checked shape,
stride and offset uploads and a checked OpenGL dispatch region rather than
trusted index assertions.

Bounded increasing loops retain conservative ranges for a single additive scalar
or storage-pointer update, or a single additive update per private integer-vector
component, in the loop body. A counter bound within a side-effect-free conjunction
can provide a conservative maximum iteration count. The bound and stride must
remain invariant, the counter's terminating increment must fit its source type,
and every possible accumulated offset must fit the target domain. Entry and exit
ranges include early exits and zero iterations; the actual loop is not unrolled
or replaced with a fixed result. Checked scalar increments and decrements retain
facts only when they cannot overflow or wrap. Aliased or volatile values, unknown
strides, mutations of bound or stride components, and shadowed bindings do not
establish this proof. These ranges establish index representability, not the
size of the bound allocation; runtime buffer and metadata checks still apply.

Local storage-pointer declarations participate in this analysis, including
positive and negative strides in nested loops. Bounds cover pointer formation
by inactive lanes and the terminal update, not just dereferenced addresses.
An integer helper can provide a loop limit when its return and effects are
proved and its arguments remain invariant across calls. Counter-dependent calls,
even those that appear effect-free on the first iteration, cannot establish this
proof. Native controls retain the source loop and verify exact output values,
untouched inputs and output guards. The pinned arg-reduce demo separately exercises
checked metadata for all 24 compiler entries and 14 numerical entries; it does
not establish full reduction coverage.

Owned private aggregates can carry initialized integer fields through resolved
getters and bounded helper updates. Field writes retain their declared storage
conversions, nested calls retain private frames, and by-value arguments cannot
change the caller's aggregate. A caller loop can summarize one unconditional
additive field update when its increment is independent of aggregate state.
Positive counter strides and bounded varying starts are supported when their
dependencies remain invariant and all terminating updates fit the source types.
Finite caller loops can also replay source-owned aggregate state within a shared
work budget, including state-dependent updates, conditional resets and nested
owned fields. Nested helper calls preserve sibling fields and reject overlapping
mutable references. Lexically distinct loop counters may reuse a name without
borrowing each other's facts. Bounds at a repeated occurrence include every
analyzed iteration; an inner loop is not removed because only its first visit
has zero iterations. The analysis does not replace helper computations or unroll
live loops. OpenGL omits a statement after an early return only when no analyzed
path reaches it; reanalysis discards previous reachability decisions. Aliased
references, volatile objects, unknown effects and source arithmetic overflow
cannot establish a range. Native Metal and OpenGL controls check
per-invocation results, unchanged inputs and output guards; Windows requires
the same controls through the existing native job.

Generated storage-pointer offsets retain interval facts from the current source
occurrence through alias construction. The facts belong to a fresh expression,
not the alias table, so later mutations require a new flow proof. Generated
name collisions cannot borrow unrelated local bounds. Byte-offset views and
reinterpreted pointers do not reuse bounds measured in the original element
units. Native controls cover repeated rebasing, loops, relative offsets and
saved aliases. The pinned column demo uses this analysis for half and float
first-pass partial sums, with upload-checked metadata, unchanged inputs and
output guards. This does not establish general pointer escape support or
complete application runtime coverage.

Addressed storage pointers passed to extent-one pointer-array helpers retain
their offset expressions during OpenGL specialization discovery. Rendering is
deferred until the caller's buffer bindings are available, including when an
offset is computed by another storage-reading helper. Read-only views and
mutable offset writeback preserve single evaluation, forwarding and lexical
shadowing. The existing native pointer controls check these cases on each target
without adding a CI job. Unproven wide offsets remain structured failures; this
does not establish bounds for an entire upstream reduction kernel.

OpenGL fixed storage-pointer arrays carry checked element-offset bounds through
helper specialization. Facts belong to source array elements and each use, not
to generated parameter names. Assignments, branches, lexical scopes and bounded
loop iterations update or invalidate those facts; repeated uses retain the
combined range from every analyzed iteration, including recurrences that read
a different initialized predecessor slot on each visit. An unknown visit
invalidates that occurrence, and reanalysis clears previous read proofs.
Lowered offsets retain typed array subscripts rather than generated expression
text. Arrays without a proven backing buffer produce a structured diagnostic
instead of raw pointer declarations. Every scalar proof transferred into a
resource helper participates in its specialization identity; an unchecked
caller cannot reuse another caller's checked stride. Different caller bounds produce
distinct specializations. Unknown effects, absent metadata, negative offsets,
overflow and byte-reinterpreted views cannot acquire a proof from this analysis.
Loop analysis has a bounded work budget and falls back to conservative analysis
when exact replay cannot be established. Required native controls cover multiple
elements, addressed views, forwarding, saved pointers and repeated calls with
guards and unchanged inputs. DirectX still diagnoses selected pointer-element
aliases and element reassignment outside its canonical initialization plan.
This does not qualify the full upstream row-reduction family or its iterator
variants, and does not establish complete repository runtime integration. The
separate checked row demo exercises unchanged float32 entries with padded
132-element rows and nested cursor resets, and 33 contiguous 65-element rows
with pointer recurrence and a partial final workgroup. Both use three numerical
profiles, output guards and unchanged-input checks. Metadata uploads and launch
geometry are checked before dispatch; target compilation alone is not treated
as numerical validation.

DirectX mutable storage-pointer array helpers write back their logical offsets
on both fallthrough and explicit returns. Value returns are evaluated once
before writeback, including forwarded calls that mutate the array. Conditional
exits, loops, switches, void expressions and structure results use the same
exit handling; read-only helpers do not acquire writeback parameters. The native
pointer tests check returned values, multiple caller-visible offsets, unchanged
inputs and output guards. This does not relax initialization, backing identity,
aliasing, access or offset-range requirements, or establish full runtime coverage.

Ordinary arithmetic uses each operand's lexical source type, including promoted
byte and short operands mixed with 32-bit or 64-bit locals. Stored integer facts
apply the destination conversion before subsequent bounds checks; the range of
an initializer is not reused after a narrowing store. Assignments and supported
compound updates retain these conversions for scalar locals and vector
components. Ranges crossing a sign or wrap boundary include every converted
value. Signed arithmetic overflow, escaped values and missing facts still do
not establish a proof. Native controls retain input observations and output
guards and compare generated Metal against the original source. Metal short
conversions and OpenGL/HLSL narrow component writes use the same source widths
as the analysis; a successful compiler check alone is not numerical evidence.

The historical MLX Metal round-trip references were reconciled for these 16-bit
conversions and for explicit discarded narrow updates. Reversing only those
rewrites recovers all 3,735 accepted bodies across the binary, copy, reduce,
quantized and unary contracts. Of the changed artifacts, 1,612 compile to
byte-identical AIR. The other 2,123 retain conversions that the original sources
declare as 16-bit. Native comparisons with the original source cover 150 of
them; the rest have source and compiler review only. The MLX demo README records
the per-family scope.

For straight-line top-level loops with at most 64 iterations, a conjunction of
invariant affine integer guards can further constrain entry ranges. For example,
``x + i < width`` excludes the unused lane of a partial tile before deriving its
buffer offset. The analysis follows stable local initializers and checked helper
results, preserving source arithmetic and conversions. Wrapping arithmetic is not
treated as an unbounded affine expression. Unknown effects, conditional updates,
early exits, escaped values and ambiguous bindings retain the conservative proof
instead. Exit ranges remain conservative, and generated control flow is unchanged.

Division proofs use exact integer arithmetic with truncation toward zero, including
64-bit endpoints. Checked nonzero divisor intervals can bound division and
non-negative remainder expressions without substituting a constant divisor.
Ranges containing zero establish no proof, and source overflow is checked
separately from target index representability.
Checked unsigned 32-bit and 64-bit shifts retain range facts when every shift
count fits the operand width and left shifts cannot wrap. Signed shifts and
narrow operands requiring integer promotion do not acquire this proof. Source
declaration types remain lexical when propagating shift results through locals.
Exact checked facts for non-volatile signed 32-bit scalars can also prove that
a loop never starts or select a conditional branch. The proof preserves lexical
scope and requires effect-free, non-wrapping condition evaluation. Unknown
values, unsigned coercions, and observable initializer or condition effects
remain reachable; no unsupported live computation is discarded.

These are conditional integer-range proofs, not allocation bounds checks. They do
not establish bounds for arbitrary dispatches or helper state. An exact
``dispatch_region`` contract can additionally bound the selected entry's physical
group and invocation coordinates. These facts remain entry-scoped and must be
enforced together with scalar metadata requirements. Scalar, two-component and
three-component coordinates are rebuilt component by component, preserving
range proofs through partial-group offsets without nested vector swizzles.
Changing a coordinate or removing either contract still invalidates the proof.
A custom caller must retain
the artifact's obligations even when
the emitted source uses a narrower index type.
Record fields, GPU-produced metadata, shared allocations and C execution wrappers
remain unsupported. C wrapper generation rejects descriptors carrying these
requirements rather than silently dropping enforcement. A custom runtime that
bypasses the Python native adapters must implement equivalent checks before using
a constrained artifact; the shader comment is not a device-side guard.

Frozen specializations, resource storage encodings, checked metadata and exact
dispatch regions share a
leading artifact contract block. Existing single-record artifacts remain valid;
readers accept each distinct record in any order. Writers use frozen constants,
storage, metadata, then dispatch-region order. The block starts on the first source line and
contains at most one record of each supported type, with a 64 KiB limit per
record. Blank lines, ordinary comments and shader text end the block. Duplicates,
malformed records and reserved markers later in the source are rejected, not
silently ignored. Individual schema and resource-layout validation still apply.
Formatting, package inspection and native dispatch retain these contracts,
including frozen half-buffer kernels with checked scalar bounds.

The existing three-platform native resource job compiles and executes a
record-buffer gather with a scalar constant, choosing all uploads by source
identity and checking exact integer results and untouched output guards.

**Fused arithmetic profiles.**

``crosstl.translator.fused_math`` provides an internal binary32 fused
multiply-add helper expressed with pairs of 32-bit unsigned words. It rounds
the exact product plus addend once, to nearest with ties to even, without
requiring 64-bit integers or double precision. It supports gradual underflow
or explicit signed-zero flushing of subnormal inputs and results before
rounding. NaNs are canonicalized; payloads and floating-point exception flags
are not represented. Generated artifacts retain the Berkeley SoftFloat license
for the adapted arithmetic.

Required Windows/DirectX, Linux/OpenGL and macOS/Metal checks execute both
policies against an independent integer oracle. Evidence includes the input
bits, expected bits, readbacks, generated source, native modules and runtime
diagnostics. The macOS source control separately compares default and precise
Metal ``fma`` against the flush-before-rounding policy. This is a selected
profile check, not a claim that every Metal device or compilation mode uses
that policy: the Metal language permits different rounding and subnormal
behavior.

Set ``source_options={"binary32_fma_profile": "rne-flush"}`` in the single-file
API, or select a profile in the project configuration:

.. code-block:: toml

   [project.source_options.metal]
   binary32_fma_profile = "rne-flush"

``rne-flush`` selects nearest-even rounding with signed-zero flushing of
subnormal operands and results before rounding. ``rne-gradual`` selects
nearest-even rounding with gradual underflow. The option lowers resolved
binary32 ``fma`` and ``precise::fma`` calls for scalars and two- to four-lane
vectors. Arguments are evaluated once. Helpers have translation-unit-private
Metal linkage so independently translated modules can share a library.
User-defined overloads, explicit ``fast::fma`` and non-binary32 operations
retain their existing lowering; this option does not configure other arithmetic.

The default is unset and preserves existing artifact identities. The selected
profile is an explicit source-environment assumption, not inferred from the
destination backend. Verify it against the original source environment before
enabling it for a repository: the
`Metal numerical contract <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>`_
allows multiple rounding and subnormal behaviors. Round-toward-zero arithmetic
is not implemented by these profiles. Existing corpus identities must be
regenerated and compiler-validated when a profile changes emitted code.

Required native CI tests source-call lowering as well as the integer helper;
macOS additionally links independently translated modules. This does not by
itself establish MLX host integration or complete upstream-suite parity.
The pinned ``d9add9d`` native package gate translates the unchanged
``v_Erffloat32float32`` and ``v_Expm1float32float32`` entries with ``rne-flush``.
It checks the upstream numerical regression inputs and a dense neighborhood
of zero against independent references, including signed zero, normal/subnormal
boundaries and Expm1 overflow. No upstream kernel is edited; this gate is
separate from running the upstream Python tests through an adapted MLX host.
`Issue #1962 <https://github.com/CrossGL/crosstl/issues/1962>`_ remains the
integration tracker for the MLX Expm1/Erf failures.

**Division arithmetic support.**

``crosstl.translator.division_math`` provides an internal binary32 division
helper using only 32-bit integer arithmetic. It rounds the exact quotient
once, to nearest with ties to even. It supports gradual underflow or signed-zero
flushing of subnormal operands and results before rounding, including values
just below the normal boundary that would otherwise round up to a normal value.
NaNs are canonicalized; payloads and exception flags are not represented.

The existing native arithmetic jobs compile and execute both policies on
DirectX, OpenGL and Metal against an independent exact-rational oracle.
Tests cover every exponent, signed zeros, subnormal and overflow boundaries,
infinities, NaNs, randomized operands and eight output guard words. Evidence
includes input and expected words, readbacks, generated source, native modules
and runtime diagnostics. The macOS control also executes unchanged Metal
division with ``-std=metal3.1 -fno-fast-math`` against the flush policy.

Select ``source_options={"binary32_division_profile": "rne-flush"}`` in the
single-file API, or use the project configuration:

.. code-block:: toml

   [project.source_options.metal]
   binary32_division_profile = "rne-flush"

``rne-gradual`` preserves subnormal operands and results; ``rne-flush`` flushes
them before rounding. The default is unset. Resolved binary32 division,
reciprocal expressions such as ``1.0f / x``, and default/precise ``divide`` calls
use the selected helper. Scalar and vector operands are evaluated once; bfloat
results retain their narrowing before later arithmetic. User-defined operators,
explicit fast builtins and other arithmetic types keep their separate rules.
Compound division supports plain local storage and direct buffer elements;
effectful operands, unsupported reference destinations and global constant
division produce a structured diagnostic instead of silently changing the
selected policy. Explicitly sequence effectful compound operands first.

Project reports and runtime manifests retain ``binary32DivisionProfile`` in
artifact provenance. Report validation compares it with the resolved source,
target and path-specific configuration. Saved CrossGL includes the arithmetic
implementation, so later target generation does not need the source option.
Native source-expression checks cover aliases, scalar/vector operations,
members, compound results, evaluation counts and narrow intermediates. Finite
results and guards are bit-exact; NaNs are compared by classification only after
the separate bfloat conversion and in the original Metal control.

The source control establishes behavior only for its tested device and compiler
settings, not a universal Metal contract. The pinned project demo tests Sigmoid
division boundaries on all three native targets. A full current-pin OpenGL
sweep now resolves the three subnormal-division differences but still has one
exponential midpoint difference tracked in
`issue #2085 <https://github.com/CrossGL/crosstl/issues/2085>`_. This is not complete
Sigmoid or upstream-suite parity. The remaining source-profile work is tracked
in `issue #2086 <https://github.com/CrossGL/crosstl/issues/2086>`_.

Metal sources can also select an explicit binary32 comparison policy:

.. code-block:: toml

   [project.source_options.metal]
   binary32_comparison_profile = "flush-subnormals"

``preserve-subnormals`` compares the represented binary32 values;
``flush-subnormals`` treats subnormal operands as signed zero for the comparison.
The default is unset. The six relational and equality operators use integer-word
ordering to preserve the selected behavior on every target, including signed
zero, infinities and unordered NaNs. Scalar and two-, three- and four-lane
operations evaluate each operand once. Existing source conversions are retained,
including bfloat narrowing before its binary32 comparison.

This policy does not modify stored operands, Boolean conversions, arithmetic,
half-only or binary64 comparisons. Source-defined operators retain their own
bodies. Unresolved operands and global constant comparisons that would require
a runtime helper produce ``project.translate.metal-comparison-profile-unsupported``.
Reports and packages retain ``binary32ComparisonProfile``; validation requires
it to match the resolved source, target and path-specific configuration. Saved
CrossGL retains the helper implementation without requiring the option again.
Native source controls establish behavior for their tested compiler and device,
not a universal Metal policy. Remainder and conversion subnormal behavior remain
separate contracts; selecting a comparison profile does not establish full
project numerical parity.

Precise two-argument arctangent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Metal ``precise::atan2`` retains a portable implementation for binary32 scalars
and two- to four-lane vectors, including scalar-to-vector broadcasts. Scalar half
and bfloat arguments promote to binary32, as required by the precise overload.
Each argument is evaluated once;
source-defined functions and default/fast-mode calls keep their separate paths.
Unsupported operand types, shapes and global runtime initializers produce
``project.translate.metal-precise-math-unsupported`` diagnostics.

The implementation handles signed-zero axes, infinities and NaN classification
explicitly, using integer-word division and the existing precise arctangent
range reduction for finite operands. Native checks use an independent decimal
reference and the six-ULP binary32 limit in Table 8.1 of the
`Metal Shading Language Specification
<https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>`_.
Axis results, zero signs, operand copies, evaluation counts and guards are exact.
This accuracy contract is not a claim of bit-identical transcendental results.
At the normal/subnormal midpoint, the arctangent correction breaks a division
rounding tie toward the smaller magnitude. Integer significand checks preserve
that boundary before applying the underflow profile. Native regression inputs
cover all 127 applicable exponent scalings, adjacent operands and both signs;
neighboring normal results must not be flushed.

The default preserves represented subnormals. A characterized source execution
can select an explicit policy:

.. code-block:: toml

   [project.source_options.metal]
   binary32_atan2_profile = "flush-subnormals"

``preserve-subnormals`` preserves operands and gradual result underflow.
``flush-subnormals`` replaces subnormal operands with signed zero and flushes
finite result underflow to positive zero. Exact signed-zero axes are unchanged.
The latter matches the tested original Metal implementation; it is not a
universal device assumption. Neither policy modifies stored inputs, normalizes
readbacks, nor selects a policy for other arithmetic or default/fast intrinsics.

Reports and runtime manifests retain ``binary32Atan2Profile`` when explicitly
selected. Validation checks it against resolved target/path source options, and
saved CrossGL retains the selected implementation. Generated Metal and OpenGL
tests cover both policies; the unchanged Metal control checks the characterized
flush policy with the same numerical assertions. DirectX native execution is a
separate required CI gate.

Generated Metal scopes contraction and reassociation directives to each precise
function body. It does not change the compiler's contraction setting for later
helpers or entry points. Native source/roundtrip checks exercise arithmetic
across function-return boundaries with contraction disabled, expression-local,
and unrestricted, while retaining the precise helper's numerical checks.

Precise logarithms
~~~~~~~~~~~~~~~~~~

Metal ``precise::log`` uses a portable binary32 implementation for scalars and
two- to four-lane vectors. Scalar half and bfloat arguments promote to binary32;
arguments are evaluated once. Default and fast calls, and source-defined
functions, keep their existing paths. Unsupported operand types and global
runtime initializers produce ``project.translate.metal-precise-math-unsupported``.

Integer normalization avoids arithmetic on subnormal operands. Range reduction
around one and an odd-series expansion preserve accuracy near one without
depending on the target's native logarithm. Native regression checks use a
120-digit decimal reference and the four-ULP binary32 limit in Table 8.1 of the
`Metal Shading Language Specification
<https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>`_.
The tests cover every binary32 exponent, dense neighborhoods around one and
the reduction boundaries, signed inputs, infinities, NaNs and narrow promotion.
Classifications, zero signs, input copies, evaluation counts and guards are
checked separately from finite numerical accuracy.

The default preserves represented subnormal operands. A characterized source
can select an explicit policy:

.. code-block:: toml

   [project.source_options.metal]
   binary32_log_profile = "flush-subnormals"

``preserve-subnormals`` computes a finite logarithm for positive subnormals and
NaN for negative subnormals. ``flush-subnormals`` treats either sign of a
subnormal operand as signed zero, yielding negative infinity. Both policies
leave stored inputs unchanged. The unchanged Metal source control checks the
flush policy for its tested compiler and device, not a universal Metal rule.
Neither policy changes other arithmetic, default/fast logarithms or readbacks.

Reports and runtime manifests retain an explicit ``binary32LogProfile``;
validation requires it to match resolved target/path source options. Saved
CrossGL retains the implementation without requiring the option again. Native
DirectX, OpenGL and Metal checks are part of the existing project-demo jobs.
This is a logarithm contract, not proof of complete project numerical parity.

Base-two logarithms
~~~~~~~~~~~~~~~~~~~

Metal ``log2`` and ``precise::log2`` have independent opt-in operand and
finite-accuracy policies. Neither changes ``binary32_log_profile`` or natural
logarithms. Explicit fast calls and source-owned overloads keep their existing
behavior. With both options unset, generated code is unchanged.

.. code-block:: toml

   [project.source_options.metal]
   binary32_log2_operand_profile = "flush-subnormals"
   binary32_log2_accuracy_profile = "portable-finite"

The operand option accepts ``preserve-subnormals`` or ``flush-subnormals``.
Preservation normalizes positive subnormal inputs using integer storage;
flushing treats either sign of a subnormal as signed zero, producing negative
infinity. Stored input bits are unchanged. When the operand option is unset,
subnormal inputs retain the target's native behavior, even if finite accuracy
is selected. Operand-only profiles leave ordinary finite calculations native.

``portable-finite`` uses integer normalization and a reduced odd-series
approximation, avoiding the target logarithm's cancellation near one. Native
regressions compare against independently rounded 120- and 180-digit decimal
references with a four-ULP binary32 bound. This tested bound is not an exhaustive
accuracy proof or a promise of bit identity with every source device. The
original Metal control characterizes the tested compiler/device's flush policy.

Scalar and two- to four-lane calls preserve the selected source overload's
conversions, narrow result storage and single operand evaluation. Unsupported
profiled types and runtime global initializers produce structured diagnostics.
Reports, manifests and packages retain ``binary32Log2OperandProfile`` and
``binary32Log2AccuracyProfile``; validation rejects values inconsistent with
resolved project options. Saved CrossGL carries the implementation directly.
Required numerical tests run in the existing three-platform project-demo jobs.

Precise square roots
~~~~~~~~~~~~~~~~~~~~

Metal ``precise::sqrt`` uses a correctly rounded binary32 helper for scalar and
two- to four-lane calls. Scalar half and bfloat arguments promote to binary32;
default/fast calls and source-owned functions keep their existing behavior.
Unsupported operands and global runtime initializers produce
``project.translate.metal-precise-math-unsupported``.

The helper normalizes the operand using integer bits and computes a 24-bit
square root with a remainder for nearest rounding. It does not depend on native
approximate square roots, subnormal arithmetic, double precision or 64-bit
integers. Signed zeros and infinities are exact; negative nonzero operands and
NaNs produce NaN. Native tests compare against an independent 120-digit decimal
reference without a finite ULP tolerance, including squared rounding midpoints,
all binary32 exponents and composed square roots.

The default preserves subnormal operands. A characterized source can select:

.. code-block:: toml

   [project.source_options.metal]
   binary32_sqrt_profile = "flush-subnormals"

``flush-subnormals`` treats a subnormal operand as a zero with the same sign.
``preserve-subnormals`` returns a nonzero result for positive subnormals and NaN
for negative subnormals. Neither option changes stored inputs, other arithmetic
or readbacks. The unchanged Metal control characterizes the explicit flush
policy for its tested compiler and device, not every Metal implementation.

Reports, runtime manifests and packages retain ``binary32SqrtProfile`` and
validate it against resolved target/path options. Saved CrossGL retains the
implementation. Required native checks share the existing project-demo runners;
the contract does not establish complete project numerical parity.

Precise reciprocal square roots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Metal ``precise::rsqrt`` and ``metal::precise::rsqrt`` use correctly rounded
binary32 helpers for scalar and two- to four-lane calls. Scalar half and bfloat
arguments promote to binary32. Default/fast calls and source-owned functions
are unchanged; unsupported operands and global runtime initializers produce
``project.translate.metal-precise-math-unsupported``.

The helper retains both the division remainder and the integer square-root
remainder until final nearest-even rounding. It uses only bounded 32-bit integer
arithmetic, without target reciprocal-root approximations or an intermediate
rounded square root. Native tests use an independent 120-digit decimal reference,
including reciprocal output midpoints and nested calls. Signed zeros produce
signed infinities; positive infinity produces positive zero; negative nonzero
operands and NaNs produce NaN.

The default preserves subnormal operands. A characterized source may select
``binary32_rsqrt_profile = "flush-subnormals"`` in its Metal source options to
treat subnormal operands as signed zero. ``preserve-subnormals`` remains an
explicit alternative. The policy is independent of ``binary32_sqrt_profile``
and does not alter stored operands or readbacks. Reports and runtime packages
retain ``binary32RsqrtProfile`` and validate it against resolved target/path
options; saved CrossGL retains the implementation. Native source controls
characterize the selected policy on the tested Metal compiler and device.

Power-function domains
~~~~~~~~~~~~~~~~~~~~~~

Metal default and precise ``pow`` calls preserve signed-zero, negative-base,
infinity and NaN domain behavior before evaluating a positive finite base.
Integral-exponent parity is determined from binary32
bits, including the boundary above which all represented integers are even.
This avoids target-dependent negative-base behavior and out-of-range integer
conversions. Scalar and two- to four-lane calls evaluate each argument once;
scalar broadcasts, half return narrowing and materialized standard-library
bfloat wrappers retain their conversion boundaries. An exponent of exactly one
returns the base without arithmetic, preserving finite values, subnormals,
signed zeros and infinities exactly. By default, other calls with a subnormal
operand keep their existing target-native path. Source-defined functions
and explicit fast-mode calls keep their separate implementations.

Characterized sources may select an operand policy:

.. code-block:: toml

   [project.source_options.metal]
   binary32_power_operand_profile = "flush-subnormals"

This treats binary32 subnormal operands as signed zero after the exact
exponent-one shortcut, before applying the domain rules. It does not change
stored operands, flush normal inputs or select an output-underflow policy.
The default remains unset; preservation on arbitrary target-native paths is
not offered as a separate profile. Reports, runtime manifests and packages
retain ``binary32PowerOperandProfile`` and validate it against resolved
target/path options. Native source controls characterize the tested Metal
compiler and device, not a universal Metal policy.

Saved CrossGL retains these helpers. Global initializers requiring the runtime
helper produce ``project.translate.metal-precise-math-unsupported``. Native
regressions compare domain results exactly and selected ordinary finite results
against an independent decimal reference, including unchanged Metal source
execution. Binary32 controls use the specification's 16-ULP power bound; selected
half controls use a one-storage-ULP regression bound. Separate identity controls
require exact results across all binary32 exponent classes, boundary
significands and deterministic sampled inputs, with original Metal execution,
evaluation counters and buffer guards. NaN payload identity is not claimed.
Applications that require portable finite-result accuracy can opt in separately:

.. code-block:: toml

   [project.source_options.metal]
   binary32_power_accuracy_profile = "portable-finite"

The default remains target-native, preserving existing Metal round-trip results.
The profile is retained as ``binary32PowerAccuracyProfile`` in reports, runtime
manifests and packages, and validated against resolved target/path options.
For ordinary finite results, its licensed fdlibm-derived approximation retains
separate high and low logarithm and exponent products. This avoids the large
errors caused by multiplying a rounded target logarithm by a large exponent
when the base is close to one. The same scalar helper serves all vector lanes
and narrowing boundaries. Near-one inputs and seeded general inputs are checked
against a high-precision decimal reference in native execution. It is an
accuracy contract, not bit-exact reproduction of a source device's approximation.
In particular, original Metal execution on the characterized device returns
infinity for ``pow(0x3f7fffff, 0xce7fffff)`` (binary32 operand words), although the
mathematical result is finite. Extended-exponent approximation tests use the
decimal reference; the native source controls are recorded separately.

Range-boundary behavior remains target-native: the approximation is used only
when its logarithmic result is strictly between -125 and 127. The surrounding
binades, overflow and underflow retain the existing intrinsic, rather than
guessing a source-device flushing rule. This is not an output-underflow profile
or a full-domain accuracy guarantee. Cross-target result-underflow parity and
complete project numerical parity remain unresolved.

Binary32 remainder
~~~~~~~~~~~~~~~~~~

Binary32 ``fmod`` has a separate, opt-in operation policy:

.. code-block:: toml

   [project.source_options.metal]
   binary32_remainder_profile = "flush-arithmetic-subnormals"

``preserve-subnormals`` computes exact truncating remainder for the represented
operands, including subnormal results. ``flush-arithmetic-subnormals`` treats a
subnormal divisor as zero and flushes a computed subnormal remainder to signed
zero. Its no-division path still preserves the numerator when its magnitude is
less than a valid divisor. This distinction rules out blanket input or output
flushing. Both policies preserve zero signs and canonicalize arithmetic NaNs.
Stored operands and surrounding comparisons are unchanged.

The default is unset. The integer-significand implementation supports scalar
and two-, three- and four-lane binary32 calls, evaluating each operand once.
Materialized Metal standard-library bfloat wrappers keep their float computation
and narrow return. User-defined overloads, explicit fast calls, half-only and
binary64 calls retain their existing behavior. Global constant calls that would
require a runtime helper produce
``project.translate.metal-remainder-profile-unsupported``. Reports, runtime
manifests and packages retain ``binary32RemainderProfile`` and validate it against
the resolved configuration; saved CrossGL retains the implementation.

Native controls characterize the tested source compiler and device only. This
option does not select a universal Metal policy, change Boolean conversions or
establish whole-project numerical parity. Half remainder and binary32 comparisons
remain independently configured contracts.

Binary32 addition and subtraction can also select an explicit policy:

.. code-block:: toml

   [project.source_options.metal]
   binary32_additive_profile = "rne-flush"

``rne-gradual`` preserves subnormal operands and results; ``rne-flush`` treats
subnormal operands as signed zero and flushes subnormal results to signed zero.
Both round once to nearest, ties to even, and canonicalize arithmetic NaNs. The
default is unset. The implementation shares the integer-word exact-rounding
support with fused arithmetic, but the two policies remain independent. It does
not replace separate source operations with a fused expression.

``native-precise`` instead emits a separate ``precise`` scalar helper for each
addition and subtraction. HLSL and GLSL materialize a precise local result;
Metal disables contraction and reassociation in the helper. Metal execution
also requires a compiler configuration without fast math, matching the native
reference controls; source qualifiers alone do not select a compiler mode.
This policy retains native arithmetic rather than emitting integer-word
emulation. It preserves the selected operation boundaries but does not specify
subnormal handling, NaN payloads or a universal device rounding policy. It is not
equivalent to either ``rne`` profile or to honoring lexical source pragmas.
The helper bodies remain ordinary checked arithmetic for conditional subgroup
lowering; selecting this policy does not bypass convergence or effect analysis.

CrossGL ``@precise`` function returns are materialized through precise local
values in HLSL as well as GLSL. This includes early returns, generic
specializations and return expressions with effects; the expressions are still
evaluated once. The HLSL return qualifier remains present, but is not relied on
alone to preserve arithmetic through optimized helper inlining. Native runtime
correctness remains a separate requirement from compiler qualification.

Scalar and vector ``+`` and ``-`` evaluate operands once, retain source
conversions and preserve bfloat result narrowing. Proven side-effect-free
``+=`` and ``-=`` support writable thread storage, direct buffer elements and
indexed vector components. Wide vectors use per-lane lowering. Unresolved
types, runtime helpers in global constants and unproven compound assignments
produce ``project.translate.metal-additive-profile-unsupported``. Source-defined
operators keep their dispatch; arithmetic inside their bodies follows the
selected policy. Integer, half-only, binary64 and pointer arithmetic, increment,
decrement, unary negation and stored operand words are unchanged.

One-dimensional C-style arrays retain built-in array-to-pointer conversion
when combined with an integral offset, including commuted addition and nested
offsets. The pointer retains the array's element type and address space; only
numeric arithmetic on loaded values uses the selected floating-point policy.
This does not infer pointer-to-array or pointer-to-pointer conversions, decay
``metal::array`` objects, or admit non-integral pointer offsets.

Reports, runtime manifests and packages retain ``binary32AdditiveProfile`` and
validate it against resolved source, target and path-specific options. Saved
CrossGL retains the implementation. Native controls characterize only the
tested source compiler and device; selecting this profile alone does not prove
whole-project numerical parity or select comparison and remainder policies.

Binary32 multiplication has a separate, opt-in policy:

.. code-block:: toml

   [project.source_options.metal]
   binary32_multiplication_profile = "rne-flush"

``rne-gradual`` preserves subnormal operands and results. ``rne-flush`` treats
subnormal operands as signed zero and flushes results whose exact magnitude is
below the smallest normal value before rounding. Both round to nearest, ties
to even, preserve the product's zero sign, and canonicalize arithmetic NaNs.
Integer-word multiplication prevents target constant-folding rules from
discarding the sign or exceptional result of a literal zero product.

Scalar and vector ``*`` preserve source conversions and bfloat result narrowing;
wide vectors lower per lane. Proven side-effect-free ``*=`` supports writable
thread storage, direct buffer elements and indexed vector components. Unknown
types, global constant expressions requiring runtime helpers and unproven
compound assignments produce
``project.translate.metal-multiplication-profile-unsupported``. Source-defined
operators retain their ownership. Integer, half-only and binary64 arithmetic,
stored operand words and the other arithmetic profiles are unchanged.

Reports, manifests and runtime packages retain ``binary32MultiplicationProfile``
and validate it against resolved source, target and path-specific options.
Saved CrossGL retains the implementation. This profile introduces an explicit
rounding boundary at each selected product; it does not reproduce a source
compiler's optional contraction of products with surrounding additions or
subtractions. Qualify the policy against the source build and device before
using it for a project. It does not establish whole-project numerical parity.

Metal compiles default arithmetic with expression-level contraction, including
under ``-fno-fast-math``: an addition or subtraction that directly consumes a
floating product of the same type in the same expression is one fused
operation. Targets that round the product first can differ in the last place.
An opt-in policy reproduces those fused pairs:

.. code-block:: toml

   [project.source_options.metal.target_options.directx]
   floating_contraction_profile = "rne-flush"

The selection follows the source compiler: ``a * b + c``, ``c + a * b``,
``a * b - c``, ``c - a * b`` and proven side-effect-free ``+=``/``-=`` of a
product are fused; for ``a * b + c * d`` only the left product is fused.
Negated products such as ``-(a * b) + c``, products stored in a separate
statement, products widened, narrowed or splatted before the addition,
integer products, products of two literals and source-defined operators remain
separate operations. binary32 pairs use the portable single-rounding
multiply-add; ``rne-flush`` treats subnormal operands as signed zero and
flushes exact results below the smallest normal value, ``rne-gradual`` keeps
them. binary16 pairs form the exact product in binary32 and round the sum once
to binary16. bfloat pairs use the binary32 multiply-add of the selected mode,
then round once to bfloat. Scalars and two- to four-lane vectors are supported;
wider vectors, compound updates that need a destination conversion and
unproven compound destinations produce
``project.translate.metal-contraction-profile-unsupported``.

Native regressions compare the unchanged Metal source, compiled with
``-fno-fast-math``, against an exact oracle for every selected and unselected
pattern, then require the same words from the translated OpenGL and DirectX
programs. On the characterized Apple GPU, binary32 contraction flushed
subnormals, binary16 contraction kept them and bfloat contraction flushed them
in binary32. The profile reproduces only the fused pairs. Combine it with the
binary32 additive and multiplication profiles when the remaining separate
operations must also be protected from target compiler contraction. Reports,
manifests and runtime packages retain ``floatingContractionProfile`` and
validate it against resolved source, target and path-specific options.

Metal source contraction controls are not yet preserved across translation.
Active ``#pragma clang fp contract(...)``, ``#pragma STDC FP_CONTRACT`` and
``#pragma OPENCL FP_CONTRACT`` directives, including their ``_Pragma`` forms,
produce ``project.translate.metal-contraction-unsupported``. The check runs
after conditional and macro expansion but before template materialization,
so moving or pruning a template cannot silently erase its source policy.
Comments, string literals, inactive branches and unused macro definitions do
not select a policy. Keep affected source on its original backend until the
directive's lexical semantics can be represented. Neither ``-fno-fast-math``
nor an individual arithmetic profile is a substitute for contraction control.
This diagnostic does not establish equivalence for directive-free source or
implement source compiler flags, scoped lowering or contraction provenance.

The legacy MLX copy reference at
``846d176227a0ac13d2667e58d2bb68b322109ab0`` proves all 2,496 discovered entries
from ``copy.metal`` through Metal-to-CrossGL-to-Metal translation. The family
covers 30 shapes, 16 concrete templates, 13 input and output types, and all 169
conversion pairs. Its schema-v2 contract pins every artifact and shape ABI,
including source/default parameter provenance. The 2,496 artifacts contain
6,566 exact materializations and 8,684 reflected resources. Every artifact has
one exact ``cast_to`` materialization; only the 14 complex-to-Boolean entries
add the nested ``cast_to<bool, float>`` body, so materialization accounting is
explicitly data-dependent rather than a false fixed per-shape count.

The generic lowering preserves MLX's float and bfloat16 Boolean bit tests,
recovers native ``ushort`` width after frontend ``uint16_t`` normalization, and
projects registered ``complex64_t`` values only after validating the ordered
``real``/``imag`` float representation. Wrong registered shapes fail closed and
unregistered lookalikes remain untouched. Reflected interfaces contain three to
eight exact resources and retain host-owned ``[1, 1, 1]`` workgroup metadata.
Required Ubuntu CI verifies and exports 24 disjoint 104-entry source shards.
One dependent macOS job verifies the complete source bundle and compiles all 2,496 exact
artifacts with ``xcrun -sdk macosx metal -Werror -c``, requiring a non-empty AIR
object for each. Reports, source bundles and compiler results are retained.
This proves translation, reflection, and native compiler
acceptance; it does not claim Metal numerical execution, MLX host-runtime
redirection, or MLX test-suite parity.

The compiler-gated identity refresh retains source coverage, materializations
and resource ABI. Its generated-source changes are limited to helper linkage
qualifiers. ``artifactIdentityRefresh`` records this audit separately from the
historical ``proof`` metadata; it does not establish coverage of a newer MLX
revision or add numerical execution claims.

A subsequent complete copy review accepts byte conversions and typed vector
initialization: 1,760 artifacts are unchanged, and reversing only the reviewed
conversions and ``int2``/``long2`` initializer syntax recovers all 736 changed
bodies. All 2,496 freshly generated artifacts compile with warnings fatal;
materializations, source/default provenance and resource ABI remain unchanged.
Negative controls reject changed indexing, byte widths, address arithmetic and
loop bounds. Historical proof and linkage-refresh metadata are preserved.
This is reference and compiler evidence, not an additional numerical-runtime
claim. Other families remain under
`issue #1966 <https://github.com/CrossGL/crosstl/issues/1966>`_.

The same selected-entry pipeline translates all 2,496 copy entries at historical
revision ``846d176227a0ac13d2667e58d2bb68b322109ab0`` to standalone OpenGL
``main`` artifacts. The schema-v2
``copy.opengl-translation.json`` contract preserves all 30 shapes, 16 concrete
kernel templates, 13 input/output types, all 169 conversion pairs, 6,566 exact
materializations, and 8,684 reflected target resources. Scalar and vector forms
expose source and destination storage buffers plus an entry-scoped size block;
fixed generalized forms add scalar stride blocks or stride storage buffers;
rank-generic forms expose shape and stride buffers plus an entry-scoped rank
block; dynamic forms add exact source and destination offset blocks.

The reference review covers all 2,496 complete sources and target interfaces.
Narrow-conversion helpers account for 308 changed artifacts; 2,188 remain
byte-identical. All indexing and interfaces are preserved. Sixty-one fresh
translation checks cover every changed conversion pair and shape, in addition
to complete compiler validation. This does not extend the historical revision
coverage or establish full numerical parity.

The generic registered-structure contract recognizes both canonical
``complex64_t`` and its emitted ``complex_t_float`` representation. The 150
complex-to-scalar entries project the real field only after validating the exact
ordered ``real``/``imag`` float shape, evaluate the source once, and retain
narrow target conversion semantics. A malformed registered representation
fails closed rather than emitting an invalid GLSL constructor.

OpenGL's 32-bit index profile cannot implicitly preserve every source 64-bit
buffer index. The complete contract therefore declares explicit host/runtime
bounds of ``[0, 2147483647]`` for ``offset + i``, ``src_idx``, ``dst_idx``,
``dst_idx + i``, ``src_idx + src_offset``, ``dst_idx + dst_offset``, ``idx.x``,
and ``idx.y``. These are portability preconditions rather than inferred facts
or generated checks; absent proof continues to fail closed.

Required Linux CI partitions the family into 24 disjoint 104-entry shards. Each
shard retranslates its entries, verifies deterministic artifact identity,
materialization provenance, standalone ``main`` workgroup metadata, and exact
target reflection, compiles with
``glslangValidator --target-env opengl --target-env spirv1.3 -S comp``, validates
with ``spirv-val --target-env spv1.3``, and requires a non-empty SPIR-V module.
Together with the Metal contract this closes complete discovered-copy
translation, reflection, and native compiler coverage for those two targets; it
does not claim numerical execution, MLX host-runtime redirection, or MLX
test-suite parity.

The DirectX path translates the same 2,496 entries to standalone ``CSMain``
artifacts. Its compact schema-v2 ``copy.directx-translation.json`` contract
preserves all 30 shapes, 16 concrete templates, 13 input/output types, all 169
conversion pairs, and 6,566 exact materializations. Exact target reflection
contains 10,036 resources. The ``g2``, ``g2large``, ``g3``, ``g3large``,
``gn2``, ``gn4large``, ``s2``, and ``v2`` shapes additionally expose generated
``CrossGLDispatchInfo`` metadata at their exact shape-specific ``b0``, ``b6``,
or ``b3`` binding.

The shared registered-structure conversion contract admits emitted
``complex_t_float`` values only after validating their ordered ``real`` and
``imag`` float fields. Exactly 150 complex-to-scalar entries project the real
field, malformed registered representations fail closed with structured
project diagnostics, and unregistered lookalikes remain untouched. Unlike the
OpenGL lowering, this native HLSL path introduces no additional source-scoped
32-bit index-range portability promise.

Required Ubuntu CI partitions the DirectX family into 24 disjoint 104-entry
shards. Each shard retranslates its exact entries, verifies deterministic HLSL
identity, materialization provenance, ``CSMain`` workgroup metadata, and exact
reflected ABI, then compiles with checksum-pinned DXC using
``-enable-16bit-types -WX -T cs_6_2 -E CSMain`` and requires a non-empty DXIL
module. Together with the Metal and OpenGL contracts this closes complete
discovered-copy translation, reflection, and native compiler coverage on all
three targets; it does not claim numerical execution, MLX host-runtime
redirection, or MLX test-suite parity.

The reviewed HLSL copy refresh at historical source revision ``846d1762``
compares all 2,496 bodies and compiles each one with warnings fatal. The 1,820
changed bodies preserve byte conversions, explicit integer arithmetic and
half rounding. All 6,566 materializations and 10,036 resource bindings are
checked; half resources use two-byte ``uint16_t`` storage with explicit
binary16 metadata. A subsequent 28-entry correction replaces float32-mediated
wide-integer conversions with direct integer rounding, with complete body and
binding comparison and warning-fatal compilation. Four unchanged controls retain
their identities. The source total is 4,859,340 bytes. Required Windows numerical
execution is tracked separately in
`#2089 <https://github.com/CrossGL/crosstl/issues/2089>`_; original and generated
Metal controls cover 64,276 integer inputs without changing tolerances. Nested bfloat
constructors in Metal bitcasts have a separate type-inference rejection in
`#2090 <https://github.com/CrossGL/crosstl/issues/2090>`_. These remain explicit
limitations while the compiler contracts preserve complete entry coverage.

The current-pinned MLX binary integration independently proves all 4,122
discovered entries from ``binary.metal``. Fifteen 238-entry base shapes and
three 184-entry work-per-thread shapes span 18 shapes, 11 concrete kernel
templates, 24 operators, and 25 input/output type pairs. Each selected artifact
contains one operator implementation and one kernel, rejects residual templates,
``decltype``, call operators, unsupported placeholders, and non-selected
operator bodies, and preserves explicit or source-default template provenance.

Scalar, size-bounded vector, fixed-dimensional generalized, and rank-generic
forms reflect exact three-, four-, five-, or seven-resource interfaces. The
generalized forms additionally materialize only their reachable
``elem_to_loc_1``, ``elem_to_loc_2``, ``elem_to_loc_3``, or
``elem_to_loc_2_nd`` helper for the exact 32- or 64-bit index type. In total,
4,122 artifacts contain 6,026 exact materializations and 19,106 reflected
resources while preserving a host-owned ``[1, 1, 1]`` workgroup contract. The
238-entry scalar contract remains an exact subset of the complete contract.

The generic path maps concrete signed and unsigned 64-bit vectors to native
Metal vector types, rewrites dependent free operators only to already-emitted
exact helpers, and applies known non-explicit contextual aggregate constructors
without admitting explicit-only or ambiguous conversions. Bfloat minimum and
maximum calls promote their arguments to float before typed reconstruction;
discarded type constructors remain evaluated through unambiguous void casts;
and scalar Boolean relational expressions receive explicit C++ integral
promotion. Focused tests retain conservative rejection boundaries for each
contextual operation.

Template declaration scanning distinguishes comparison and shift operator names
from template argument delimiters. Dependent struct references inside those
operator bodies remain in their enclosing template scope until specialization;
they are not emitted as concrete types with unresolved local constants.

A schema-v2 hash-pinned contract records every identity, shape, template,
classification, byte count, materialization, and resource ABI. Required Ubuntu
CI verifies translation, bodies and reflection across 24 disjoint shards and
exports the exact pinned sources. One dependent macOS job rejects missing,
duplicate or changed entries before compilation. It compiles all 4,122 exact artifacts
with ``xcrun -sdk macosx metal -Werror -c`` and requires non-empty AIR without a
warning exemption. Both phases retain failure evidence; the local round-trip
test still translates and compiles together. This closes discovered selected-entry translation,
reflection, and native compiler coverage. It does not claim Metal numerical
execution, host-runtime redirection, or MLX test-suite parity.

The same selected-entry pipeline translates all 4,122 binary entries to
standalone OpenGL ``main`` artifacts. The schema-v2
``binary.opengl-translation.json`` contract preserves all 18 shapes, 11 kernel
templates, 24 operators, 25 input/output type pairs, and 6,026 exact
materializations. It pins 18,732,199 generated GLSL bytes
and 19,106 reflected resources across exact three-, four-, five-, and
seven-resource target interfaces. Scalar forms expose three storage buffers;
size-bounded forms add an entry-scoped size uniform block; fixed generalized
forms expose scalar stride blocks or stride storage buffers; and rank-generic
forms expose shape and stride buffers plus an entry-scoped rank block.

OpenGL's 32-bit index profile cannot implicitly preserve the source's runtime
64-bit buffer indices. The complete contract therefore declares host/runtime
bounds of ``[0, 2147483647]`` for ``offset + i``, ``a_idx``, ``b_idx``,
``out_idx``, ``out_idx++``, ``idx.x``, and ``idx.y``. These are explicit
portability preconditions, not inferred facts or generated runtime checks;
unproven wide-index translation remains fail-closed.

Required Linux CI partitions the family into 24 disjoint shards. Each shard
retranslates its exact entries, checks deterministic identity, materialization
provenance, standalone ``main`` workgroup metadata, and target reflection,
compiles every artifact with
``glslangValidator --target-env opengl --target-env spirv1.3 -S comp``, validates
it with ``spirv-val --target-env spv1.3``, and requires a non-empty SPIR-V
module. This closes complete discovered-binary OpenGL translation, reflection,
and native compiler coverage; it does not claim numerical execution, MLX
host-runtime redirection, or MLX test-suite parity.

The same selected-entry pipeline translates all 4,122 binary entries to
standalone DirectX ``CSMain`` artifacts. The schema-v2
``binary.directx-translation.json`` contract preserves all 18 shapes, 11
kernel templates, 24 operators, 25 input/output type pairs, 6,026 exact
materializations, and the seven explicit index-range portability preconditions.
DirectX resource namespaces preserve source buffer coordinates. The nine shapes
that consume ``threads_per_grid`` add explicit ``CrossGLDispatchInfo``
reflection, producing 21,248 resources across exact three- through
eight-resource target interfaces.

Computed-result bfloat ``ArcTan2``, ``LogAddExp``, and ``Power`` paths expand
both operands to float, compute in float, and reconstruct the result with exact
round-to-nearest-ties-to-even. ``Maximum`` and ``Minimum`` also expand for the
comparison, but preserve and return the selected original low-16-bit bfloat
payload without requantization. Unproven bfloat builtins remain fail-closed,
and native 16-bit storage remains a Shader Model 6.2 requirement.

Required Ubuntu CI partitions the family into 24 disjoint shards. Each shard
retranslates its exact entries, checks deterministic identity, materialization
provenance, standalone ``CSMain`` workgroup metadata, and target reflection,
then compiles every artifact with checksum-pinned DXC using
``-enable-16bit-types -WX -T cs_6_2 -E CSMain`` and requires a non-empty DXIL
module. This closes complete discovered-binary DirectX translation, reflection,
and native compiler coverage; it does not claim numerical execution, MLX
host-runtime redirection, or MLX test-suite parity.

The binary HLSL identity review compares every historical/current body and all
21,248 reflected resources, with 8,244 strict compiler checks across both sets.
It accounts for source-width arithmetic and two-byte integer storage carrying
IEEE binary16 values without changing bindings, dispatch contracts, loop bounds,
source pins or specialization counts. The 2,435 changed artifacts and 1,687
byte-identical artifacts remain compiler evidence, not full-family numerical
execution evidence.

The current-pinned MLX reduction integration independently proves all 2,396
host-named entries from ``reduce.metal`` through Metal-to-CrossGL-to-Metal
translation. Three base forms plus six multidimensional families across two
index widths and three dimension counts produce 39 exact ABI shapes over nine
kernel templates. The family spans six operator families, 44 concrete operator
types, and 13 input/output types. Its compact schema-v2 contract stores only
per-entry classification, artifact identity and size, exact materialization
identity/count, and exact resource identity/count; transient proof rows,
translation reports, and AIR objects are not tracked.

The complete reflected interface contains 25,088 resources across exact one-
through twelve-resource ABIs and retains the host-owned ``[1, 1, 1]`` workgroup
contract. Comparison-aware template parsing preserves non-type expressions such
as ``(1 > 2)``; proven integral non-type arguments are folded for overload
selection without changing source-spelled specialization identity. Visible
aliases and raw pointer base spellings normalize only to proven concrete types,
and native pointer-plus/minus-integral expressions bypass aggregate free-
operator dispatch only when built-in semantics are type-proven. Concrete-helper
deduplication ignores comments and external whitespace while retaining exact
preprocessing-token literal bytes. Compile-time static members used by
references, address-taking, receivers, or unevaluated storage-sensitive
expressions are hoisted into address-space-correct Metal ``constant`` storage
instead of being substituted as rvalues. Ambiguous aliases, unresolved address
provenance, non-integral offsets, and incompatible overloads remain fail-closed.

Required Ubuntu CI uses 24 disjoint source shards: 20 contain 100 entries and four
contain 99. Every shard verifies deterministic identity, materialization,
workgroup metadata, and reflected ABI before exporting its checked sources.
One dependent macOS job verifies complete contract coverage and every source
identity before warning-fatal compilation with
``xcrun -sdk macosx metal -Werror -c``. All 2,396 AIR objects must be non-empty.
Missing, duplicated or altered sources fail before compilation; both phases
retain evidence. The default local test still translates and compiles each entry.
This closes complete discovered-reduce translation, reflection, and native
compiler coverage; it does not claim Metal numerical execution, DirectX or
OpenGL whole-family coverage, MLX host-runtime redirection, or MLX test-suite
parity.

The same selected-entry pipeline translates all 2,396 reduction entries to
standalone OpenGL ``main`` artifacts. The compact schema-v2
``reduce.opengl-translation.json`` contract preserves the 39 shapes, nine
kernel templates, six operator families, 44 concrete operator types, and 13
input/output types while pinning 9,216 exact materializations,
45,283,816 generated GLSL bytes, and 25,088 resources across
scalar-layout-aware one- through twelve-resource target ABIs.
All source bodies and resource interfaces have complete reference comparisons;
an earlier 85-entry pipeline review covers its edit patterns and ABI shapes.
The 156 complex-buffer metadata updates describe existing storage layouts,
not new resource declarations. The subsequent complete review verifies helper
deduplication and typed extrema, including required int64 extension declarations
when a wide initializer is narrowed to a 32-bit result. All 2,396 current
artifacts compile and validate. These checks establish deterministic translation
and strict compilation, not whole-family numerical execution.

OpenGL's signed 32-bit logical buffer offsets require twenty-one explicit
source-expression portability preconditions: the one-, two-, and
five-dimensional ``LoopedElemToLoc`` source pointer expressions,
``inputs[i - 1] + reduction_size``, the long-column output expression,
``out_idx``, and fifteen complete rebased-pointer offsets such as
``in_ + start_idx``, ``inputs[j] + i`` and ``out_ + i`` for the int64-index
shapes. Each is host-bounded to ``[0, 2147483647]``. These bounds are not
inferred and no runtime check is generated; missing, mismatched, negative, or
over-wide assertions remain fail-closed. The 868 identity changes in 14 kinds
are reviewed as two mechanical rewrites (a zero-trip ``gl_WorkGroupSize.y``
loop removed under the ``[1, 1, 1]`` workgroup and a peeled tail-loop iteration
emitted as a helper specialization) followed by a helper-by-helper
bisimulation; negative controls reject semantic mutations. Bounded unit-step loop-carried
pointer-array initialization is evaluated one exact iteration at a time. A
non-singleton per-invocation dynamic target remains a may-write and cannot
establish full-array definite assignment, while a shadowed loop-index binding
disables the serial-loop proof and remains fail-closed. For-in range and
scalar-count expressions are rendered in the outer lexical environment;
Fixed-array iterable expressions and compile-time extents resolve before
same-named pattern bindings enter scope. same-named patterns use independent
controllers and shadow pointer,
stage-builtin, and flattened stage-struct aliases in the loop body. For-in
resource specialization resolves overloads from exact lexical pattern types.
Dynamic for-in resource specialization resolves overloads from exact call-site
lexical types. Null storage-pointer reachability and elision preserve exact
lexical declaration identity. Null workgroup-pointer reachability and
elision preserve exact lexical declaration identity. Nested
resource-specialization discovery preserves deterministic lexical order across Python
hash seeds. Workgroup-pointer bounds analysis visits every control-flow
expression and preserves outer mutations across lexical blocks. loop-local
fixed-array
storage retains flattened
stage-input
struct declarations, and
fixed-array patterns cannot inherit same-named outer scalar or vector-component
bounds. Repeated loop bounds and generated loop controllers lose their proven intervals when
exact scalar or vector-component dependencies can mutate. logical-offset
mutation through resolved nested helpers is written back to callers. Direct
element and addressed one-element forwarding through overload-resolved scalar
or fixed-array helpers preserves that writeback;
private scalar address views remain confined to their declaring lexical scopes;
nonlocal scalar address views remain fail-closed;
dynamic elements and unresolved, ambiguous, or recursive forwarding remain
fail-closed. Fixed-array storage cannot escape through local private pointer or
reference aliases; lexically shadowed arrays and condition-only reads are not
misattributed, while residual private pointer or reference syntax remains
fail-closed.

Required Linux CI partitions the family into 24 disjoint shards: 20 contain
100 entries and four contain 99. Every shard verifies deterministic artifact,
materialization, workgroup, and scalar-layout-aware reflected ABI identity,
compiles with
``glslangValidator --target-env opengl --target-env spirv1.3 -S comp``, validates
with ``spirv-val --target-env spv1.3``, and requires a non-empty SPIR-V module
for every entry. This closes complete discovered-reduce OpenGL translation,
reflection, and native compiler coverage; it does not claim numerical
execution, DirectX whole-family translation, MLX host-runtime redirection, or
MLX test-suite parity.

The DirectX path translates the same 2,396 reduction entries to standalone
``CSMain`` artifacts. Its compact schema-v2
``reduce.directx-translation.json`` contract preserves all 39 shapes, nine
kernel templates, six operator families, 44 concrete operator types, 13
input/output types, and 9,216 exact materializations. Every entry pins both its
HLSL source and non-empty DXIL identity, byte count, materialization digest, and
resource digest. Exact target reflection contains 27,382 resources across
one- through thirteen-resource interfaces. The 37 shapes other than ``init``
and ``all`` additionally expose generated ``CrossGLDispatchInfo`` metadata;
the source data, shape, stride, rank, and reduction constants retain their
shape-specific coordinates and exact HLSL storage types.

The shared DirectX lowering emits a portable quiet binary32 value for
unshadowed ``NAN``, preserves explicit componentwise complex SIMD shuffle
overloads, and converts only proven direct void tail recursion to an iterative
loop. Device and constant pointer arrays lower to bounded resource offsets with
transitive whole-array and element writeback only after exact lexical overload,
alias, and backing-object provenance resolution. Logical bfloat pointee types
remain distinct from their physical ``uint16_t`` storage, and postfix
pointer-dereference typing removes exactly one pointer layer. Unresolved
aggregate pointer members or arrays, unsafe recursion, ambiguous pointer-array
aliasing or reinterpretation, and unsupported HIP record lifecycle or layout
remain fail-closed. Nested and function-local anonymous records are
materialized deterministically when their complete layout is representable.
Unlike the OpenGL lowering, the native HLSL path introduces no additional
source-scoped 32-bit index-range portability promise.

The pre-release native proof preserves one HLSL and one DXIL artifact for each
entry and independently recompiles all 2,396 HLSL artifacts with
checksum-pinned DXC, requiring byte-identical non-empty DXIL before accepting
the compact contract. Required Ubuntu CI partitions the contract into 24
disjoint shards: 20 contain 100 entries and four contain 99. Each shard
retranslates its exact entries, verifies deterministic HLSL identity,
materialization provenance, ``CSMain`` workgroup metadata, and exact reflected
ABI, then compiles with ``-enable-16bit-types -WX -T cs_6_2 -E CSMain`` and
requires a non-empty DXIL module. Together with the Metal and OpenGL contracts
this closes complete discovered-reduce translation, reflection, and native
compiler coverage on all three targets; it does not claim numerical execution,
MLX host-runtime redirection, or MLX test-suite parity.

OpenGL Pointer and Matrix Policies
----------------------------------

DirectX and OpenGL share a lowering for non-escaping scalar views of fixed
private vector arrays. Two- and four-component vectors with matching 32-bit
scalar elements retain their original storage; reads and writes address the
vector component rather than a copied array. This includes direct casts,
single-return reference-receiver accessors, nested fields, local view aliases,
and bounded scalar offsets. Array selections and offsets are captured when a
local view is bound. Const views remain read-only.

Scalar accesses require a proven in-bounds range. Literal indices, bounded
unit-step loops, and bounded masks are supported; side-effecting indices,
unresolved layouts, escaping views, and reference-argument copyout remain
diagnostics. This lowering does not provide general local aggregate references
or establish full-kernel numerical equivalence.

Private pointer helpers lower to fixed array parameters with separate element
offsets. Bounded cumulative pointer updates retain those offsets; proven-zero,
side-effect-free updates may be omitted only when their result is discarded.
Unknown rebasing, escaping pointer values, and unsupported aliasing remain
translation errors.

Two optional target policies are available through project source options:

.. code-block:: toml

   [project.source_options.metal.target_options.opengl]
   cooperative_matrix_software_lowering = true
   private_pointer_out_of_bounds_read = "error"

   [project.source_options.metal.target_options.directx]
   cooperative_matrix_software_lowering = true

``cooperative_matrix_software_lowering`` defaults to ``false`` on both targets.
DirectX supports checked lane-local element access, copying, addition,
subtraction, elementwise multiplication, and negation. Combined with
``software_subgroup_width = 32`` and
``relative_wave_shuffle_out_of_range = "self"``, it also supports float 8-by-8
multiply-accumulate with the ``tile_4x4_row_pair`` mapping and two elements per
lane. This path uses shared memory within complete logical subgroups, independent
of the hardware wave width. Workgroup dimensions must be concrete and all
barriers, including those reached through helpers, must have proven uniform
control flow. Divergent calls and incomplete subgroups are rejected.

The optional ``cooperative_matrix_mma_profile`` on DirectX and OpenGL selects
``"rne-gradual"`` or ``"rne-flush"`` for software multiply-accumulate. Each product and
sum is rounded separately to binary32, in increasing inner-dimension order,
using integer arithmetic with round-to-nearest, ties-to-even. The gradual
profile preserves subnormals; the flush profile flushes subnormal operands and
underflowing results to signed zero. Both use canonical quiet NaNs. This avoids
reliance on a driver's handling of ``precise`` floating-point instructions.
The default, ``None``, retains the existing native arithmetic. The option does
not change lane-local matrix operations or establish arbitrary-input Metal
parity; select it only when its arithmetic matches the required source contract.

The alternatives ``"fma-rne-gradual"`` and ``"fma-rne-flush"`` fuse each inner
product with the accumulator before rounding once. Both retain increasing
inner-dimension order. A rounding-sensitive native control with twelve matrices
and two successive products matches this fused reference on original and
round-tripped Metal; 321 of its 768 results differ from separate rounding.
This bounded evidence does not establish the ordering or arithmetic of every
Metal matrix implementation or device.

Reports and runtime packages retain the selected policy as
``cooperativeMatrixMmaProfile`` and include it in dispatch-region identities.
This records configuration, not proof that an entry uses matrix multiplication.
No rounding helpers are emitted for entries without a software matrix product.

DirectX emits a fixed eight-iteration loop for each fragment component, with
unrolling disabled for the integer arithmetic profiles. This retains the
inner-dimension order and both shared-memory barriers while avoiding repeated
copies of the rounding helper's control flow. Smaller compiled artifacts are
not evidence of native correctness. The required Windows gates separately
execute all four profiles with special values, dense quarter-valued inputs,
rounding-sensitive inputs, two successive products and native project packages.
The repeated-product controls cover both ``[32, 4, 1]`` and ``[16, 8, 1]``
workgroups, with twelve logical subgroups per case.

Native qualification at ``3b48ce23`` passes these 24 controls on WARP 1.0.21
and on the 1.0.13 comparison: 18,432 exact result words and 768 guard words per
runtime. The allocation-view controls also verify unchanged input bytes and
physical padding, the loaded child-process runtime identity, and completion
within the existing deadlines. Package controls retain upload and artifact
identities and output guards; they do not retain device-side input readbacks or
child-process runtime receipts. This evidence qualifies the bounded integer
profiles, not default native arithmetic, every WARP capability, complete MLX
GEMM execution or arbitrary-input Metal parity. See `the native matrix
qualification <https://github.com/CrossGL/crosstl/issues/2203>`_.

OpenGL supports the same float 8-by-8 mapping with
``cooperative_matrix_software_lowering = true`` and
``software_subgroup_width = 32``. Matrix operands are exchanged through bounded
workgroup memory partitioned by logical subgroup, without native subgroup
intrinsics or a hardware-width requirement. The software applicability policy
records matrix products even when no other subgroup operation is present.
By default, products and sums use separately rounded binary32 instructions in
increasing inner-dimension order. The four explicit MMA profiles described above
instead use integer-word arithmetic and retain the selected policy in packages.
They also apply to OpenGL matrix exchange under an explicit hardware-width
contract; the policy does not establish that the device satisfies that width.
This is not a promise of arbitrary-input bitwise Metal
parity. Linux execution controls check both fragment elements, multiple logical
subgroups, repeated products, nonzero accumulators, rounding-sensitive inputs
and output guards. Profiled controls additionally check signed zero, subnormals,
ties, overflow and canonical NaNs against an independent integer reference.
These controls do not establish complete MLX GEMM execution.

DirectX and OpenGL software-subgroup control-flow analysis follows resolved read-only
helper returns through scalar values, value-only aggregate fields, immutable
locals and nested helpers. Straight-line private aggregate construction is
supported when every field is initialized exactly once before the value is used.
Return and branch dependencies are checked against each caller's workgroup-uniform
arguments; one uniform call does not establish a fact for other calls.

Storage reads require both an immutable binding derived from the entry's explicit
constant-storage contract and a workgroup-uniform index, including any generated
handle offset. Typed local aliases and resolved helper arguments retain those
dependencies. A read-only resource view alone does not establish immutable backing
storage. The application must preserve the source address-space and non-aliasing
contracts; the analysis does not establish numeric bounds or authorize index
narrowing. Mutation, reference escapes, volatile fields, recursion, unknown calls
and unproven storage reads remain diagnostics.
Read-only storage parameters are checked against their bodies and resolved
forwarding calls; qualifiers alone do not exempt calls from mutation analysis.
OpenGL resource specializations retain convergence facts for the exact source
nodes that were proved, rather than inferring safety from helper names.

Native controls on all three targets cover scalar constant references, constant
buffer reads, nested collective helpers, early returns, zero-trip loops and
guarded output. DirectX and Metal additionally execute the generated
resource-handle control; OpenGL rejects its unproven wide output index even when
the helper return is uniform. Aggregate constant-reference helpers have compiler
coverage, not a complete constant-buffer upload contract. These proofs do not
establish convergence for every collective loop form or complete MLX translation.

Fixed additive collective loops in DirectX and OpenGL accept ``+=`` and ``-=`` when
the initial value, bound and stride are checked integer constants. The proof
uses the source induction width and signedness and includes the update after
the last executed iteration. Positive and negative strides, reversed comparisons
and zero-trip loops are covered. Overflow, unsigned wraparound, unknown or varying
control values, induction aliases and divergent exits remain diagnostics. Native
controls compare original and generated Metal kernels and require DirectX
execution on Windows and OpenGL execution on Linux. Canonical buffer stores
preserve unconditional collective-helper value evaluation when their index is
free of calls and side effects; source-owned functions do not inherit this rule.

Both DirectX and OpenGL distinguish writable reference destinations from their
value-only array indices and member selectors in resolved helper calls. Passing
``values[index]`` to an ``out`` or ``inout`` parameter invalidates ``values``,
not ``index``. Selector increments, nested writes, aliases and unknown calls
remain conservative. OpenGL entry-exit checks use the resolved effects instead
of discarding uniform arguments merely because they were passed by value.
Required native controls cover nested reference forwarding, private arrays and
record members, repeated collectives, uniform early returns and untouched guards
on each target's existing operating-system job. These controls do not establish
full-kernel or upstream-suite parity.

Resolved pointer arguments follow the same distinction using source types from
the call's lexical scope. A write through ``output + count - lane`` invalidates
the destination storage, not the scalar address values. Nested offsets,
addressed array elements and conditional destinations retain their storage
roots; unknown pointer producers, casts and effectful selectors remain
conservative. This is a write-location proof, not a purity or alias-immutability
proof. Required native tests cover forwarded pointers and array parameters,
zero and varying uniform loop counts, multiple and partial logical subgroups,
uniform early returns and output guards without additional CI runners.

The default DirectX fallback requests separately rounded binary32 products and
sums in increasing inner-dimension order using ``precise``. Compiled instructions
alone do not prove that the runtime preserves this contract: WARP 1.0.21 returns
contracted results for the retained precise scalar and matrix controls
(`native arithmetic conformance <https://github.com/CrossGL/crosstl/issues/2193>`_).
The explicit integer profiles above avoid that dependency for supported matrix
products without changing the default path or its required conformance tests.
Metal does not specify a bitwise accumulation order for its matrix intrinsic.
Native comparison remains required for the application's numerical tolerance;
this policy does not promise bitwise equivalence for arbitrary floating-point
inputs. Plain matrix multiplication, matrix loads and
stores remain structured translation errors. Fragment dimensions, scalar types,
registered mappings and source provenance must still be compatible.

On OpenGL, supported fragment operations lower to explicit scalar storage and
subgroup operations. Multiply-accumulate currently requires the float 8-by-8,
32-lane, two-elements-per-lane ``tile_4x4_row_pair`` contract, matching operand
dimensions, and an exact subgroup-width contract. Unsupported mappings are
rejected. This option does not remove the hardware subgroup requirement or
establish numerical equivalence for an entire kernel.

``private_pointer_out_of_bounds_read`` defaults to ``"error"``. An explicit
``"zero"`` policy guards eligible reads through const private pointers backed
by complete, statically sized local arrays, returning a typed zero outside the
array. It does not permit out-of-bounds writes or unresolved array slices.
The selected policy is retained in project source options; recovered artifacts
also contain ``CROSSTL_PRIVATE_POINTER_OOB_READ_ZERO``. This is an explicit
behavior choice for an otherwise invalid source access, not evidence that the
result matches the source runtime. Numerical validation remains necessary.

Metal sources may also opt into ``promote_derived_pointer_members = true`` under
``project.source_options.metal`` to separate supported derived pointer members
into backing objects and offsets during materialization. Unknown origins and
unsupported pointer mutations remain errors rather than guessed bindings.

The pinned MLX quantized-wide OpenGL gate covers 18 selected entries with strict
pointer bounds and no upstream edits. It binds the pinned host dispatch to its
source-faithful ``[32, 2, 1]`` workgroup for both two- and four-vector kernels:
two logical 32-lane software subgroups and 64 total invocations. The gate
validates source identities, reports, array preservation, logical subgroup
indexing, exact execution metadata, GLSL compilation, and SPIR-V modules on
Linux without claiming KHR hardware-subgroup metadata. It does not claim full
quantized coverage or MLX runtime integration.

OpenGL Software Subgroup Specialization
----------------------------------------

Metal compute sources that require fixed 32-lane SIMD semantics can opt into
a bounded OpenGL shared-memory lowering when the deployment device does not
provide the KHR subgroup extensions:

.. code-block:: toml

   [project.entry_points]
   "kernels/quantized.metal" = "affine_quantize_float_gs_32_b_2"

   [project.entry_workgroup_size_rules."kernels/quantized.metal"]
   affine_quantize_float_gs_32_b_2 = [32, 1, 1]

   [project.source_options.metal.target_options.opengl]
   software_subgroup_width = 32

This option is target-scoped and explicit; it does not change Metal parsing or
other target artifacts. The only accepted width is ``32``. The selected output
must contain exactly one compute entry with concrete positive local dimensions
and no more than 1,024 total invocations.

A project containing both collective and ordinary entries can additionally set
``software_subgroup_applicability = "when-used"`` in the same target option table.
CrossTL evaluates this policy after entry selection and dependency pruning, so
ordinary entries do not acquire subgroup scratch storage, barriers, or hardware
subgroup requirements. The policy works with explicit or discovered entry
selection, source-pattern overrides, and translation variants on DirectX and
OpenGL. DirectX collective entries still require the explicit
``relative_wave_shuffle_out_of_range = "self"`` policy.

The default applicability is ``"required"``: requesting software lowering for an
entry without subgroup operations remains an error. ``"when-used"`` does not
suppress invalid widths, unknown applicability values, unsupported operations,
divergent collective control flow, or hardware-width conflicts. Subgroup system
inputs count as dependencies even without a collective call; unsupported
query-only entries remain errors instead of falling back to physical waves.
An explicit applicability option requires ``software_subgroup_width``.

When applicability is explicit, artifact provenance includes
``softwareSubgroupPolicy`` with the requested width, effective width (``null``
for an ordinary entry), applicability, and selected dependency names. Reports,
runtime packages, and recovered checkpoint artifacts retain this record.
Report validation checks its shape and consistency with the resolved source
options. This provenance describes the lowering decision; it is not a substitute
for native compilation and numerical execution.

CrossTL partitions the linear invocation range into an exact compile-time count
of independent 32-lane software subgroups. Subgroup count, subgroup index,
subgroup width, and lane index lower respectively to the workgroup invocation
count divided by 32 and rounded up, ``gl_LocalInvocationIndex / 32``,
``CROSSTL_SOFTWARE_SUBGROUP_WIDTH``, and
``gl_LocalInvocationIndex % 32``. The one-subgroup case retains the simpler
``1u``, ``0u``, and ``gl_LocalInvocationIndex`` forms.

A fixed workgroup may end in a subgroup with fewer than 32 real invocations.
Reductions combine only those invocations, scratch storage retains the exact
workgroup size, and shuffle-down helpers use the calling value when their source
falls beyond the real lanes. Source shuffle results from inactive lanes are not
portable; numerical controls select only results from active source lanes.
Relative offsets are checked before adding the lane index, so unsigned overflow
cannot wrap an out-of-range shuffle into another lane's value.
Metal shuffle and broadcast lane parameters retain their source ``ushort``
conversion before target lowering, including arguments passed through helpers.
Canonical 32-bit wave arguments retain their own width. Unqualified source wave
calls respect namespace visibility, including local ``using namespace``
directives, before builtin argument conversion is applied.
Writes through an indexed destination do not invalidate the index's uniformity;
actual index mutations and divergent exits still prevent software lowering.
All real invocations must still reach the same workgroup barriers. This does not
provide dynamic nonuniform final dispatch groups: the runtime must preserve the
selected workgroup shape rather than round a source thread grid up silently.

Native loader requests can express an exact source thread grid with
``threadGridSize``, in addition to ``workgroupSize`` and ``workgroupCount``.
The group count must equal the component-wise ceiling of the thread grid divided
by the workgroup size. For example, ``threadGridSize = [1025, 1, 1]`` with
``workgroupSize = [1024, 1, 1]`` requires ``workgroupCount = [2, 1, 1]``.
When supplied, ``globalSize`` and ``gridSize`` must describe the exact thread
grid, not its padded bounds. Omitting ``threadGridSize`` retains full-group
dispatch and its existing dimension checks.

The Metal runtime implements this contract through native ``dispatchThreads``
after checking device support and grid limits. Original and generated Metal
controls check partial groups, source coordinates, subgroup sums and buffer
guards across one-, two- and three-dimensional grids. DirectX and OpenGL reject
this explicit contract with ``exact-thread-grid-unsupported`` until their
lowering can preserve partial-group identities and active lanes. This contract
does not yet enable the MLX small-row host plan on those targets.

The translator's ``plan_dispatch_regions`` and ``specialize_dispatch_region``
APIs provide the lowering needed for portable exact grids. The planner partitions
a grid into at most eight rectangular regions with uniform active workgroup
shapes. Specialization preserves source global IDs, workgroup IDs, grid extents
and group counts while leaving local IDs and subgroup operations tied to each
region's actual group shape. It operates on an independent, single-entry AST;
source kernels are not patched. Per-invocation coordinate captures remain private
to each invocation, including when read by helper functions.

Native tests execute these region artifacts through one shared-allocation
dispatch sequence on DirectX and OpenGL, with original and roundtrip Metal
controls. The caller must compile each specialization, submit its exact physical
counts and share allocations across the complete plan.

Project translation accepts a target-scoped ``dispatch_region`` source option
for DirectX and OpenGL. Its fields match ``DispatchRegion.to_json()``; the
configured ``workgroup_size`` must match the region's physical ``workgroupSize``.
For the final five invocations of a 37-thread grid with nominal width 32:

.. code-block:: toml

   [project]
   workgroup_size = [5, 1, 1]

   [project.source_options.metal.target_options.opengl.dispatch_region]
   threadGridSize = [37, 1, 1]
   sourceWorkgroupSize = [32, 1, 1]
   workgroupOffset = [1, 0, 0]
   workgroupCount = [1, 1, 1]
   workgroupSize = [5, 1, 1]

The region is lowered after entry selection and retained in artifact, package
and native-loader provenance as ``dispatchRegion``. The loader checks its
physical size against the reflected entry and rejects a different launch count
or a second thread-grid mapping. Each region needs its own output directory or
otherwise distinct artifact identity; a grid or offset change changes the
specialized program, even when the physical workgroup shape stays the same.

New specialized artifacts also retain a leading ``crosstl-dispatch-region``
record. Reflection checks its physical size against the compute entry. Native
request construction rejects missing or differing descriptor region metadata;
the execution adapter independently rechecks the actual launch against the
artifact snapshot, including global extents. The execution receipt records the
checked region. C execution wrappers currently reject region-specialized
descriptors because they do not implement these checks. Custom runtimes must
enforce the same launch obligation; the header is not a device-side guard.

Region packages also record ``dispatchRegionProgram``: the selected source
entry, resolved CrossGL program hash (including included definitions), target,
lowering settings and installed translation-code identity. This is a consistency
record, not a cryptographic signature or a proof of numerical equivalence.

``select_native_loader_dispatch_regions`` accepts descriptor/package-root pairs
and orders the complete canonical plan for a requested exact grid and nominal
workgroup size. Missing, duplicate, unrelated or mixed-program regions are
rejected. Older packages without the program identity can still be loaded
individually, but cannot participate in automatic region selection.

``prepare_native_loader_dispatch_regions`` additionally verifies every artifact,
preflights its bindings and compiles through the native adapter before yielding
requests for ``dispatch_sequence``. Source allocations are shared, with uploads
only on the first region; derived launch uniforms remain region-local. Use the
context manager for the whole dispatch so compiled modules remain alive and are
cleaned up on success or failure.

On-demand specialization caching and MLX host dispatch integration are not yet
wired to these APIs. Single-request DirectX/OpenGL ``threadGridSize`` rejection
remains in place; use the complete region preparation API for an exact grid,
not an ordinary rounded dispatch.

The bounded mode supports scalar ``float``, ``int``, or ``uint`` sum, minimum,
maximum, and shuffle-down operations. Shared scratch spans the complete
workgroup, while every helper derives a subgroup-local lane and base so reads
and reductions cannot cross a 32-lane boundary. Subgroup operations normally
must execute in workgroup-uniform control flow. They may appear directly in the
entry or in a uniquely identified helper only when every helper call is direct,
unconditional, top-level, and owned by that sole entry. Compile-time branches,
canonical constant loops, and canonical runtime loops whose integer bounds are
proven workgroup-uniform remain valid.

One narrow lane-dependent reduction form is also supported: a direct
``WaveActiveSum``, ``WaveActiveMin``, or ``WaveActiveMax`` assignment in a
top-level entry-owned ``if`` with no ``else``. The condition must be
side-effect-free, the branch must contain exactly that one subgroup operation,
declarations and escaping control flow cannot precede it, and the payload and
target must be matching 32-bit numeric scalars. CrossTL evaluates the branch
prefix only for active lanes, contributes a typed identity from inactive lanes,
invokes the barriered subgroup helper uniformly across the workgroup, and
exposes the result only to active lanes. Sum uses zero. Minimum uses positive
infinity, ``INT_MAX``, or ``UINT_MAX``; maximum uses negative infinity,
``INT_MIN``, or unsigned zero. Conditional shuffle, nested or multi-operation
branches, and other unproven shapes continue to fail closed.

Direct source references to raw ``gl_Subgroup*`` or ``subgroup*`` builtins,
empty operation sets, unsupported payloads or operations, and unresolved helper
ownership also fail closed. Successful lowering emits barriered ``shared``
memory helpers and the marker ``CROSSTL_SOFTWARE_SUBGROUP_WIDTH``. It
deliberately emits no ``GL_KHR_shader_subgroup*`` extension,
``gl_Subgroup*`` use, ``CROSSTL_REQUIRED_SUBGROUP_WIDTH`` marker, or hardware
``subgroupWidth`` execution metadata. This keeps the software execution
contract distinct from the default hardware-subgroup path and its host
preflight. When the option is absent, KHR subgroup generation and exact-width
enforcement are unchanged.

The lowering establishes shader semantics only for this constrained contract.
It does not infer dispatch counts, prove arbitrary divergent control flow,
rewrite an application's loader, or claim parity for other entries. Invalid
requests fail with
``project.translate.opengl-software-subgroup-invalid`` and include the width,
workgroup size, operation, and reason available at the rejection site.

Project Index-Range Assertions
------------------------------

Some source index types cannot be represented directly by a target's legal
scalar index types. When the application already constrains an index at the
host or runtime boundary, record that precondition in ``crosstl.toml``:

.. code-block:: toml

   [[project.index_range_assertions]]
   source = "kernels/*.metal"
   function = "gather_values"
   expression = "element_index"
   minimum = 0
   maximum = 1023

Each assertion table has these fields:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``source``
     - Repository-relative source glob. The assertion is considered only for
       matching translation units; omitting it defaults to ``*``.
   * - ``function``
     - Optional exact source function name. When omitted, the assertion can
       apply in any function containing the matching expression.
   * - ``expression``
     - Source index expression covered by the assertion. Matching ignores
       whitespace but otherwise preserves the expression identity.
   * - ``minimum``
     - Inclusive integer lower bound for the expression.
   * - ``maximum``
     - Inclusive integer upper bound for the expression. It must not be less
       than ``minimum``.

Index-range assertions are explicit host/runtime portability preconditions.
CrossGL does not infer, emit, or enforce them at runtime. It uses an assertion
only to justify a semantics-preserving target index conversion when the full
asserted range is legal for the target representation and indexed extent. An
assertion does not clamp, wrap, or otherwise redefine out-of-range source
values; the application remains responsible for satisfying the precondition on
every execution.

Project Workgroup-Access Assertions
-----------------------------------

OpenGL cannot represent a source workgroup pointer directly. It specializes
pointer-free helpers against a concrete entry-owned ``shared`` array and must
prove that every composed access remains within that backing array. When the
source runtime already enforces an entry-specific absolute element range,
record that precondition explicitly:

.. code-block:: toml

   [[project.workgroup_access_assertions]]
   source = "kernels/fft.metal"
   entry_point = "fft_mem_256_*"
   function = "ReadWriter_*"
   parameter = "crosstl_ptr_buf"
   minimum = 0
   maximum = 255

Each assertion table has these fields:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``source``
     - Repository-relative source glob. Omitting it defaults to ``*``.
   * - ``entry_point``
     - Required source entry-point pattern. The assertion cannot cross entry
       ownership boundaries.
   * - ``function``
     - Helper-function pattern. Omitting it defaults to ``*``.
   * - ``parameter``
     - Workgroup-pointer parameter pattern. Omitting it defaults to ``*``.
   * - ``minimum``
     - Inclusive absolute element offset into the concrete backing array.
   * - ``maximum``
     - Inclusive absolute element offset. It must not be less than ``minimum``.

The assertion does not provide backing identity, extent, element type, or a
pointer offset. OpenGL must still derive those properties from the source call
graph and emits the original composed runtime offset expression. Matching
assertions are intersected with any statically derived access range; a
contradiction or an asserted range outside the concrete backing array fails
before artifact emission. Entries without a matching assertion continue to
require a complete static proof.

Workgroup-access assertions are host/runtime portability preconditions. CrossGL
records them in the project report but does not emit runtime checks or change
source indexing behavior. The application is responsible for satisfying every
assertion on each dispatch.

The portability report records the configured tables under
``project.workgroupAccessAssertions`` and their count under
``project.workgroupAccessAssertionCount``. Report consumers can therefore audit
the host/runtime assumptions used during translation alongside the generated
artifacts.

Exact DirectX bfloat16 Contract
-------------------------------

Exact DirectX bfloat16 lowering preserves bfloat16 storage and conversion
semantics instead of substituting IEEE half precision. When generated HLSL uses
native 16-bit storage, its artifact and runtime metadata advertise DirectX 12,
a minimum Shader Model of 6.2, and entry profiles such as ``cs_6_2``. DXC
validation and runtime loader commands for that HLSL require
``-enable-16bit-types``; application-specific compiler wrappers and build
commands must preserve the same option.

HLSL resolves retained scalar type aliases before selecting bfloat conversions.
Their logical bfloat type remains distinct from the physical ``uint`` register
payload, including initialization, helper parameters, returns and arithmetic.
The Metal frontend retains dependent primitive aliases and resolves namespace
ownership before emitting the intermediate source. Dependencies bind at their
declarations, so a later local alias cannot change a helper's parameter or return
type. Required native controls cover floating-point, integer, half and bfloat
chains, local shadowing and namespace-owned overloads, with unchanged Metal
sources as controls on macOS. Invalid primitive alias resolution produces
``project.translate.metal-scalar-alias-unresolved`` with a ``scalarAlias`` detail
record containing ``aliasName``, ``reason`` and, for cycles, ``dependencyChain``.
This does not establish arbitrary C++ name lookup or volatile scalar-value
support; unsupported volatile value aliases fail translation.

Successful artifacts that use this path record a ``bfloat16Lowering`` object
with ``status`` set to ``exact``, ``approximationUsed`` set to ``false``, and
the register, storage, and rounding representations used by the generated
HLSL. This keeps exact lowering distinct from a compile-only half-precision
substitution in machine-readable reports.

If an operation cannot be lowered with exact bfloat16 semantics, translation
fails closed with a structured
``project.translate.directx-bfloat16-unsupported`` diagnostic. Its
``details.bfloat16Lowering`` record identifies available context such as the
target profile, operation, source type, and reason instead of silently changing
precision or behavior.

These compiler requirements and diagnostics define an artifact contract only.
They do not imply automatic host runtime or backend integration: CrossGL does
not modify application loader code, configure a DirectX backend, bind
resources, or wire generated artifacts into a framework.

Fail-Closed Pointer Provenance
------------------------------

Targets that cannot represent a source pointer directly must prove its backing
storage and composed offset before emitting an artifact. For workgroup storage,
that proof includes the concrete backing declaration, entry-point ownership,
element extent and type, offset composition through helper calls, and the
affected materialization or specialization. Dynamic backing selection,
unresolved offsets, incompatible declarations, escaped identity, and
cross-entry ownership fail closed instead of producing target code with altered
aliasing or synchronization behavior.

When the target exception provides provenance, an OpenGL workgroup-pointer
diagnostic records the available ``function``, ``parameter``, ``backingName``,
``offsetExpression``, ``materializationName``, and ``reason`` values under
``details.workgroupPointer``. Unavailable values are omitted so consumers can
distinguish retained evidence from assumptions. The surrounding diagnostic
also identifies the source path, intended artifact, target, and missing target
capability.

This report contract localizes translation work and preserves actionable
evidence. It does not establish whole-repository semantic parity, rewrite host
runtime integration, or prove execution correctness for a framework or corpus.

Project scan, report, and translation commands also accept repeatable
``--source-root``, ``--include-dir``, ``--define``, and ``--source-override``
overrides. CLI source roots replace the configured source roots for that
command. CLI defines use ``NAME`` or ``NAME=VALUE`` syntax and override matching
names loaded from ``crosstl.toml``. CLI source overrides use ``PATTERN=BACKEND``
syntax and override matching source patterns loaded from ``crosstl.toml``.
These overrides are recorded in the emitted project report.
Scan, report, and translation commands accept repeatable ``--variant NAME``
selectors. ``crosstl.toml`` can set ``selected_variants = ["debug"]`` as the
default scoped variant list for project runs; explicit ``--variant`` arguments
override that configured default for the command. Scoped scan and report output
evaluates only the selected declared variants for variant-aware include and
define metadata, records the selected variant list, and does not claim omitted
variants as scanned.

Unsupported target backend names are reported as configuration diagnostics in
scan, report, and translation output. Translation still records per-artifact
failures for any artifact attempt that cannot be generated.

Validate artifacts referenced by a report:

.. code-block:: bash

   python -m crosstl validate-project crosstl-out/portability-report.json \
     --format text

Validation exits nonzero when the report metadata is malformed, artifact
records, source-map records, or preserved diagnostics are malformed, source-map
mapping lists are empty, file-granularity source maps do not contain one
file-level mapping, finer-grained mappings are not positive-length or fall
outside artifact-level file anchors, source-map, diagnostic location, or
diagnostic ``originalLocation`` spans are internally inconsistent, diagnostic
location file paths or artifact source paths are not repository-relative, project target
lists are not normalized and deduplicated, diagnostic or artifact targets are
not declared by the report, artifact sources are not declared translation units,
embedded validation records reference artifacts not declared by the report,
embedded toolchain runs reference failed report artifacts,
validation records contain duplicate identities or inconsistent status fields,
full reports with embedded validation artifacts omit validation summaries,
summarized embedded validation omits declared artifacts, embedded toolchain-run
coverage omits OK validation artifacts for available toolchains, failed
embedded toolchain runs omit matching diagnostics, external corpus entry presence,
discovery, or source-backend fields do not match the project root and declared
units, full reports omit units or skipped files that the current
project scan discovers, translated outputs are missing, artifact paths resolve
outside the repository, generated artifact hashes no longer match the files on
disk, source files with recorded hashes are missing or changed, or opt-in
toolchain smoke checks fail. The report file being validated is ignored during
that freshness scan, and files under the configured output directory remain
excluded from discovery.
Toolchain smoke checks only run for translated artifacts that still exist inside
the repository. Each smoke check is bounded by a short subprocess timeout, and
timeouts are reported as failed toolchain runs. Targets that need backend-
specific entry points, profiles, package metadata, or SDK context record bounded
tool availability runs instead of claiming artifact compilation. Validation
reports include
severity, diagnostic-code, and missing-capability rollups for generated and
preserved diagnostics, plus artifact target, artifact source-backend,
artifact variant, hash-status, source-size status, generated-size status,
source-map status, source-remap status, toolchain status, toolchain-run status,
toolchain-run target, toolchain-run source backend, diagnostic check kind,
toolchain-run check kind, toolchain-run tool, and toolchain-run variant rollups
for validation results.
The JSON validation report uses schema version 1 with a fixed top-level field
set so automation can detect contract drift. It includes compact project
context with the project root, output directory, configured targets, source
roots, include/exclude patterns, include directories, selected variants,
define/variant names without exposing raw define values, and the source report
hash used for validation provenance.
The default output is JSON; ``--format text`` prints a concise validation
summary with validation report identity metadata, source report hash, project
context, and the same rollups, and ``--format sarif`` emits validation
diagnostics as SARIF with project context and source report hash metadata in
invocation properties.

Inspect an existing report as a concise JSON, text, or SARIF summary:

.. code-block:: bash

   python -m crosstl inspect-report crosstl-out/portability-report.json \
     --format text \
     --max-diagnostics 20 \
     --max-failed-artifacts 20 \
     --max-source-map-artifacts 20 \
     --max-artifact-matrix-artifacts 20 \
     --max-artifact-provenance-artifacts 20 \
     --max-define-processing-artifacts 20 \
     --max-include-path-processing-artifacts 20 \
     --max-include-dependencies 20 \
     --max-skipped-sources 20 \
     --max-validation-artifacts 20 \
     --max-toolchain-runs 20 \
     --max-migration-actions 20 \
     --max-runtime-references 20 \
     --max-external-corpus-entries 20

Report inspection includes inspection identity, SARIF invocation metadata, and
source report schema/kind metadata, source report hash metadata,
source report generation metadata,
validation status,
invalid/unavailable report status, project counts, project configuration path,
project root, output directory, configuration counts, normalized source-root,
include-pattern, exclude-pattern, and include-directory lists, runtime-reference
rollups and bounded runtime-reference samples, failed artifacts
with variant labels when present, diagnostic code and missing-capability rollups,
validation diagnostic-code, missing-capability, artifact target,
artifact source-backend, artifact variant, hash-status, source-map status,
source-remap status, toolchain status, and toolchain-run target,
source-backend, check-kind, tool, and variant rollups, report
source-backend, source override mappings, file-extension, and artifact
target rollups, source-map count, granularity, target, and source-backend
rollups, source-remap count, mapping-count, granularity, target, and
source-backend rollups,
artifact matrix completion counts, matrix source provenance, target and
variant completion rollups, sampled missing and extra artifact identities,
bounded validation artifact and validation toolchain-run samples with
truncation counts, failed validation metadata on artifact provenance samples,
include-directory status counts, inactive source-root and include-directory
record details, diagnostics, configurable diagnostic and failed-artifact
truncation counts, external corpus rollups, sampled missing and
present-but-undiscovered external corpus entries with retained provenance
metadata and configurable sample limits, and migration actions.
Inspection sample-limit options accept non-negative integer counts and default
to ``20`` for each sampled report section.
The JSON inspection report uses schema version 1 with a fixed top-level field
set so automation can detect contract drift while optional report sections
remain present with ``available: false`` until their source report data exists.
Migration action inspection is bounded and records truncation counts for large
reports.
``--format sarif`` emits the inspection diagnostics as SARIF for
code-scanning workflows. SARIF invocation properties include the source report
path, source report hash, report identity metadata, project root, output
directory, configured targets, source roots, include/exclude patterns, include
directories, and selected variants. SARIF locations include line and column
metadata and positive-length character spans when diagnostics carry source
offsets.

Build a metadata-only runtime integration plan from a portability report:

.. code-block:: bash

   python -m crosstl plan-runtime crosstl-out/portability-report.json \
     --format text \
     --max-runtime-references 20

Runtime planning emits a ``crosstl-runtime-integration-plan`` JSON document
with source report hash metadata, validation diagnostics from the source
report, project target summaries, runtime-reference rollups and bounded
samples, per-target compiler runtime-plan request commands, and manual actions
for runtime references found in host or build files. The compiler request
entries point at the metadata-only ``runtime-loader-plan-v1`` contract request
in the compiler repository. This is planning evidence only: it does not import
compiler internals, execute device code, or rewrite host application code.

Build a runtime artifact manifest for downstream host or package tooling:

.. code-block:: bash

   python -m crosstl runtime-manifest crosstl-out/portability-report.json \
     --format text

Runtime artifact manifests emit a ``crosstl-runtime-artifact-manifest`` JSON
document from a validated portability report. The manifest lists translated
artifacts by target with source/backend/variant identity, generated artifact
hash and byte-size metadata, source-map anchors, optional compiler
``source-remap`` sidecars, and the runtime planning contract summary required
by downstream packaging or host integration tooling. Invalid source reports
produce diagnostic-only failed manifests. The manifest is a handoff contract;
it does not generate runtime framework code, execute device code, or rewrite
host application code.

Validation, runtime planning and manifest construction each read the report once.
Within a call, the unit inventory and configuration-diagnostic checks share one
current project scan, and the manifest reuses that validation for its runtime
plan. The reported hash identifies the exact report bytes that were validated.
No scan or validation result is retained across calls; subsequent calls recheck
sources, includes, configuration and generated artifacts. Keep project files
unchanged during validation and packaging: this is not an atomic filesystem
snapshot.

Within one artifact-manifest build, variants of the same source/backend/target
reuse source-wide reflection, including unavailable results. Generated artifacts
are still validated and reflected individually, and entry-specific execution
and specialization metadata is merged separately. Source results are not cached
across manifest builds.

Build a backend-neutral runtime binding manifest for host integrations:

.. code-block:: bash

   python -m crosstl runtime-binding-manifest crosstl-out/portability-report.json \
     --output crosstl-out/runtime-bindings.json

``translate-project`` can write the same binding manifest beside the
portability report in one run:

.. code-block:: bash

   python -m crosstl translate-project /path/to/repo \
     --target cgl \
     --output-dir crosstl-out \
     --report crosstl-out/portability-report.json \
     --runtime-binding-manifest crosstl-out/runtime-bindings.json

Runtime binding manifests emit a ``crosstl-runtime-binding-manifest`` JSON
document derived from the validated portability report and runtime artifact
metadata. Each entry is backend-neutral and includes ``sourceFile``,
``sourceBackend``, ``targetBackend``, ``artifactPath``, ``entryPoint``,
``resourceBindings``, ``bufferMutability``, ``scalarConstants``,
``specializationConstants``, ``dispatchDimensions``, ``sourceProvenance``, and
``validation``. Resource bindings include set/binding coordinates, access, and
derived mutability.
Dispatch dimensions record reflected workgroup size data when available while
leaving workgroup, global, and grid counts unset for host code to provide.
Reflection, runtime artifact manifests, and runtime binding manifests keep
function and specialization constants in dedicated ``specializationConstants``
records with their own counts. They are not reported as ``resources`` or
``resourceBindings``, nor as ordinary ``constants`` or ``scalarConstants``.

Build a deterministic runtime handoff package from a runtime artifact manifest:

.. code-block:: bash

   python -m crosstl package-runtime crosstl-out/runtime-manifest.json \
     --package-dir crosstl-runtime-package \
     --format text

Runtime packages emit a ``crosstl-runtime-package`` JSON report and write a
package manifest, translated artifacts, source-remap sidecars, and a short
integration guide into the package directory. Packaging revalidates artifact
hash and byte-size metadata before copying files so stale generated outputs are
reported as structured diagnostics instead of hidden. The package is a handoff
artifact for host or build-system tooling; it does not rewrite host application
code, execute device code, generate runtime framework code, or install target
SDKs.

Inspect a runtime handoff package before host binding:

.. code-block:: bash

   python -m crosstl inspect-runtime-package \
     crosstl-runtime-package/runtime-package.json \
     --format text

Runtime package inspections emit a ``crosstl-runtime-package-inspection`` JSON
document with ready and failed host-binding records. The inspection is read-only
and verifies copied packaged artifacts and source-remap sidecars against the
package manifest's recorded paths, hashes, and byte sizes. Missing, stale, or
malformed package contents are reported as structured diagnostics before host
loader or build-system tooling consumes the handoff package. Inspection
preserves the ``runtime-loader-plan-v1`` summary linkage and does not rewrite
host application code, execute device code, generate runtime framework code, or
install target SDKs.

Build a host binding plan from a runtime package manifest:

.. code-block:: bash

   python -m crosstl plan-host-bindings \
     crosstl-runtime-package/runtime-package.json \
     --format text

Host binding plans emit a ``crosstl-runtime-host-binding-plan`` JSON document
with per-target packaged artifact paths, package-inspection readiness metadata,
``bind-runtime-artifact`` actions for host loader or build-system tooling, and
``review-runtime-references`` actions when the source repository contained
runtime API references. The planner reuses runtime package inspection and only
emits bind actions for ready package records; missing or stale package artifacts
remain diagnostics instead of host-integration work items. The plan preserves the
``runtime-loader-plan-v1`` summary linkage from earlier reports. It is an action
plan only; it does not rewrite host application code, execute device code,
generate runtime framework code, or install target SDKs.

Build a target-scoped runtime adapter plan from a runtime package manifest:

.. code-block:: bash

   python -m crosstl plan-runtime-adapters \
     crosstl-runtime-package/runtime-package.json \
     --format text

Runtime adapter plans emit a ``crosstl-runtime-adapter-plan`` JSON document
from the same package handoff metadata used by package inspection. The plan
lists ready package bindings by target with ``adapterKind``, ``artifactFormat``,
``requiredTools``, ``hostResponsibilities``, source-remap handoff paths,
parser-derived ``hostInterface`` entry point and resource summaries where the
packaged artifact frontend is available, and ``wire-runtime-adapter`` actions
for host loader or build-system tooling. When host interface metadata is
unavailable or not ready, the plan emits ``resolve-host-interface-metadata``
actions so host and build tooling can provide reflection or backend-specific
binding metadata before wiring the adapter. Source targets with registered
frontends can contribute parser-derived interface summaries; formats that need
compiled reflection, such as SPIR-V handoff artifacts without reflected entry
point/resource data, remain explicit follow-up actions. The plan also carries
through package inspection diagnostics and
``review-runtime-references`` actions when the source repository contained
runtime API references. The plan is a target-scoped integration contract; it
does not rewrite host application code, execute device code, generate runtime
framework code, or install target SDKs.

Materialize runtime adapter descriptor files from a runtime package manifest:

.. code-block:: bash

   python -m crosstl materialize-runtime-adapters \
     crosstl-runtime-package/runtime-package.json \
     --adapter-dir crosstl-runtime-adapters \
     --format text

Runtime adapter descriptor packages emit a
``crosstl-runtime-adapter-package`` JSON document and write a deterministic
``runtime-adapters.json`` manifest, an ``ADAPTERS.md`` summary, and one
``adapters/<target>/*.adapter.json`` descriptor per ready or blocked runtime
adapter plan record. Each descriptor preserves the packaged artifact path,
target adapter identity, source-remap handoff path, host-interface metadata,
required tools, host responsibilities, and validation readiness for downstream
host loader or build-system tooling. The descriptor package is metadata only:
it does not rewrite host application code, execute device code, generate
runtime framework code, or install target SDKs.

Runtime Adapter Execution Contracts
-----------------------------------

Runtime fixture execution uses a backend-agnostic adapter contract carried on
each ``RuntimeExecutionRequest`` as ``adapter_contract``. The contract can be
loaded from a fixture's ``runtimeAdapter`` object and merged with manifest
metadata already recorded on a translated artifact. This keeps execution
fixtures stable across downstream runtimes while letting package inspection
provide reflected entry points, resource bindings, and dispatch workgroup
sizes when they are available.

The contract fields are intentionally limited to kernel execution metadata:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Purpose
   * - ``entryPoints``
     - Names, stages, execution config, optional parameter records, and
       workgroup-size metadata for callable translated kernels or shaders.
   * - ``resourceBindings``
     - Backend-neutral resource names, kinds, types, set/binding numbers,
       access modes, and optional fixture value names that an adapter maps to
       runtime buffers, textures, samplers, or parameter blocks.
   * - ``specializationConstants`` / ``functionConstants``
     - Specialization or function constant identifiers, dtypes, values,
       defaults, and required flags needed before launching the entry point.
   * - ``dispatch``
     - Entry point, workgroup size, workgroup count, global size, or grid size
       for compute-style launches. Fixture dispatch counts can augment
       artifact-manifest workgroup sizes.
   * - ``validationHooks``
     - Expected pre-run, runtime, post-run, or comparison checks that a
       downstream executor should perform or report as skipped/unavailable.

Runtime planning scopes artifact execution metadata to the selected compiled
entry before merging the fixture's requested adapter contract. When both the
selected entry and requested dispatch provide a concrete workgroup size, the
values must match. A mismatch produces
``project.runtime-verification.workgroup-size-mismatch`` during runtime setup,
records both sizes and the selected-entry provenance, and leaves the test case
unplanned. When only one side provides the size, the planner carries that value
into the merged dispatch, so either missing side of the runtime contract can be
completed from the other. This completion does not hide a disagreement when
both sides are present.

Downstream runtimes implement the ``RuntimeAdapter`` protocol or subclass
``RuntimeExecutor`` and receive the merged contract in ``run(request)``. For
example, an MLX validation adapter can consume translated Metal artifacts and
map neutral fixture buffers and function constants to MLX runtime objects, but
the fixture contract remains expressed in terms of entry points, bindings,
constants, dispatch geometry, and validation hooks rather than MLX APIs.

Saved project test-runner plans can execute deterministic runtime fixtures when
callers supply adapter implementations explicitly:

.. code-block:: bash

   python -m crosstl execute-test-runner \
     crosstl-out/project-test-runner-plan.json \
     --runtime-executor native-vulkan=tools.runtime.vulkan:VulkanRuntimeAdapter \
     --output crosstl-out/project-test-runner-report.json

``--runtime-executor`` is repeatable and uses
``EXECUTOR=MODULE:OBJECT``. ``MODULE`` may be a dotted Python module name or a
``.py`` file path. ``OBJECT`` may be an adapter instance, adapter class, or
factory returning an object with ``run(request)`` or the parity-adapter methods
``prepare_buffers(state)``, ``dispatch(state, buffers)``, and
``collect_outputs(state, result)``.

For the built-in DirectX, OpenGL, and Vulkan native parity adapters, callers can
also pass ``--native-runtime-adapter TARGET`` or
``--native-runtime-adapter TARGET=MODULE:OBJECT``. The optional object is the
backend runtime driver consumed by the native adapter after artifact validation.
Use ``--no-native-runtime-validation`` only when the caller has already handled
toolchain validation or is running a controlled test fixture.

CrossTL includes optional ``DirectXComputeRuntime``, ``OpenGLComputeRuntime``,
and ``VulkanComputeRuntime`` reference drivers for bounded compute fixtures.
They import target dependencies lazily and report structured unavailability or
setup failures when the required API, loader, device, or resource contract is
not available. Their supported resource shapes are intentionally narrower than
the translated shader languages; a successful translation does not imply that
one of these reference drivers can execute the complete host workload.

Native Dispatch Limits
----------------------

The Python native drivers reject invalid launch dimensions before submitting
device work. Counts and local sizes must contain one to three positive integers;
zero, negative, Boolean and fractional values are not clamped or coerced.
Dimension fields must be sequences, not scalar values, mappings, sets, strings
or byte strings. False-valued malformed metadata cannot select a fallback or
bypass local-size checks. The existing empty-sequence/``None`` convention for
omitted dimensions is preserved. Derived group counts use exact integer ceiling
division. Malformed dimensions report ``dispatch-dimensions-invalid``, identifying
the field and, for a sequence, the failing node before any device work.

DirectX checks Direct3D 12 compute limits. OpenGL queries the current context's
per-axis group-count and local-size limits and maximum invocations per group.
Vulkan queries the selected physical device before creating a logical device.
The Metal worker checks device local-size limits and the compiled pipeline's
maximum threads per group. OpenGL and DirectX preflight every request in a
sequence before allocating resources or executing the first node.

An exceeded limit reports ``dispatch-limit-exceeded`` with ``field``,
``requested``, ``maximum``, the requested geometry and the applicable ``limits``.
Per-axis failures include ``axis`` (zero-based); aggregate invocation failures
do not. Sequence diagnostics include ``nodeIndex``. Metal returns the detailed
worker record under ``dispatchValidation`` while preserving its process logs.
Missing or invalid capability information is an error, not an assumed limit.
These checks do not tile an oversized workload, infer missing local sizes from
shader binaries, or replace compiler validation of the actual shader layout.

OpenGL also checks API error state after setup, resource binding, submission,
synchronization and readback. A non-success state reports ``opengl-api-error``
with ``phase`` and ``glError``; inability to read that state is a failure too.
Unchanged output after a rejected submission cannot be reported as successful
execution. These Python-driver checks do not add limit validation to generated
C++ loader adapters or establish complete host-runtime integration.

Byte Field Updates
------------------

Byte-valued variables and fields retain their signed or unsigned eight-bit range
after compound assignment and prefix/postfix updates on Metal, DirectX and
OpenGL. This includes nested fields rooted in a stable local variable. Postfix
expressions return the value before the update, including at overflow boundaries.
Updates whose owner requires evaluating an array index or function call remain
outside this contract; unsupported forms produce a diagnostic instead of
duplicating an observable evaluation.

Lossless Float Buffer Storage
-----------------------------

Typed float32 buffers can opt into ``encoding: "ieee754-binary32"`` in the
Python native dispatch API and runtime-verification fixtures. Each ``values``
element is then an unsigned 32-bit storage word, not a numeric float:

.. code-block:: json

   {
     "dtype": "float32",
     "shape": [3],
     "encoding": "ieee754-binary32",
     "values": [2143363909, 1065353216, 2147483648]
   }

These words represent a noncanonical quiet NaN, 1.0 and negative zero. Native
packing and readback retain their bits without converting through host floats.
The reflected buffer remains float32, so kernels still perform floating-point
arithmetic. Encoding does not bypass physical-layout, size, binding or allocation
checks. Unknown encodings, incompatible dtypes, Boolean words, fractional words
and words outside the unsigned 32-bit range are rejected.

Upload and readback choose their encodings independently, including an
initialized read-write buffer. An output request may omit ``values`` and select
encoded readback using only dtype, shape and encoding. Returned words describe
actual native storage; no original input values are substituted after execution.
Encoded expected outputs require exact word equality and matching encoding,
regardless of numeric tolerance settings. This checks storage preservation, not
a requirement that arithmetic NaN results preserve an operand's payload.

Without ``encoding``, the existing numeric buffer and non-finite token contracts
remain unchanged. Native regression gates exercise generated Metal, DirectX and
OpenGL partial updates, with original Metal controls and shared-allocation
DirectX/OpenGL sequences. Metal's process-isolated runtime has no sequence API.
The representation does not cover double storage or generated C++ adapter
serialization.

Bfloat16 Storage
~~~~~~~~~~~~~~~~

Native Metal ``bfloat`` buffers use ``dtype: "bfloat16"`` and
``encoding: "bfloat16-bits"``. Uploads and readbacks contain unsigned 16-bit
storage words, not host floating-point conversions. An explicit encoding is
required so bfloat16 cannot be mistaken for IEEE binary16 or float32. Exact-word
comparisons retain signed zeros, subnormals and NaN payloads.

DirectX's bfloat lowering exposes ``uint16_t`` buffers. Supply ``dtype: "uint16"``
and unsigned integer values to that physical interface. Explicit HLSL
``int16_t`` and ``uint16_t`` buffers, and Metal ``short`` and ``ushort`` buffers,
retain two-byte integer storage. DXC requires native 16-bit types to be enabled.
OpenGL's widened bfloat lowering exposes float32 storage; exact bfloat carriers
are binary32 words with the original word in the upper 16 bits. Neither target
accepts a logical bfloat16 payload in place of its reflected physical layout.

Required native copy controls cover all 65,536 bfloat storage patterns and
output guards, with original Metal execution as a round-trip control. Storage
support does not establish bfloat arithmetic, conversion rounding or complete
MLX host integration. Padded vectors and unsupported allocation views remain
subject to the native loader's existing restrictions.
Metal and OpenGL controls additionally cover two- and four-lane vectors and
homogeneous structures. DirectX vector and aggregate lowering remain tracked
separately in issue #1488.

OpenGL rounds float32-to-bfloat numeric conversions to nearest, with ties to
even, using integer operations on the binary32 representation. Constructors,
assignments, function arguments and returns, scalar/vector arithmetic, and
vector components retain bfloat precision before widening to float32. Existing
bfloat values are not requantized during identity copies or vector repacking.
The helper quiets NaNs while retaining their sign and upper payload bits;
same-type copies preserve every payload bit instead. Native Metal can
canonicalize NaNs during numeric conversion. Its conversion controls require
exact agreement between original and generated Metal, and separately check
NaN classification, sign and bfloat representability. They do not establish
cross-target NaN payload identity. All finite conversion results and identity
copies retain exact bitwise comparisons. Runtime scalar 32-bit integer conversions use
integer rounding in OpenGL and DirectX to avoid an intermediate float32 rounding
step. OpenGL runtime conversions from double and wider integers that could
introduce double rounding fail with a structured diagnostic. This does not
establish support for every bfloat operation in an MLX host backend.
Scalar integer constant initializers also round directly to bfloat, without
an intermediate float32 value. Constructor, cast, alias and implicit initializer
forms preserve the difference between ``bfloat(16842753u)`` (16908288) and
``bfloat(float(16842753u))`` (16777216). Native controls retain signed zero,
midpoint neighbors, ties of both parities, exponent carry and 32/64-bit extrema.
Unsupported constant expressions, including unproven unsigned negation,
narrowing integer casts and integer literals outside their source type's range,
remain diagnostic rather than silently using a rounded float32 initializer.

Metal integer literals retain their signedness and 32/64-bit width through
canonical parsing and target emission, including small values whose suffix
selects wide shifts or overloads. Decimal, hexadecimal, binary and legacy octal
forms use the first representable source integer type; an unsuffixed decimal
value does not become unsigned merely because it exceeds the signed 32-bit
range. Invalid suffixes and values outside the supported 64-bit types are
rejected. Native controls cover original/generated Metal and generated
DirectX/OpenGL arithmetic; they do not establish complete MLX runtime coverage.
DirectX emits an explicit signed type for 32-bit literal shift operands,
including literal-only arithmetic and conditional expressions. The promoted
left operand determines the shift result type independently of the shift count
and enclosing expression. Runtime controls cover signed right shifts, 32/64-bit
counts, subsequent wide arithmetic and single evaluation of count updates.
In particular, direct ``bfloat(9259400833873739777ul)`` retains payload ``0x5f01``,
while an explicit float32 intermediate retains the distinct payload ``0x5f00``.

Binary16 Storage
~~~~~~~~~~~~~~~~

Metal ``half`` and DirectX ``float16_t`` buffers can use ``dtype: "float16"``
with ``encoding: "ieee754-binary16"``. Values are unsigned 16-bit words; for
example, ``[0, 32768, 1, 32257]`` represents positive zero, negative zero,
the smallest positive subnormal and a quiet NaN with a payload. Uploads and
readbacks use two bytes per scalar, without conversion through host floats.
The same exact-word comparison rules apply as for binary32 storage.

Reflection retains the actual element size, stride and alignment through
packaging and native-loader descriptors. Scalar, naturally aligned vectors,
homogeneous structures and constant-buffer values retain their physical
layouts. Padded layouts that cannot be represented by a tightly packed value
array remain unsupported. Ambiguous HLSL ``half`` declarations are not assumed
to have native 16-bit storage; explicit ``float16_t`` is required.

Without an encoding, finite numeric values are rounded to binary16 on upload;
overflow is rejected. Explicit non-finite input tokens remain available.
Use encoded values when signed-zero or NaN payload identity must be checked.
Boolean words, negative words, values above 65535, inconsistent encodings,
misaligned views and truncated allocations fail before dispatch.
Metal supports aligned offset views. The DirectX Python driver realizes aligned
structured-buffer ranges with native descriptors; this does not change the
shader's storage type or repair values after readback.

Metal device references to supported scalar and naturally aligned two- or
four-component vector values retain ``storageLayout: "metal-buffer"`` with
``runtimeSized: false``. Their ``blockSizeBytes`` and
``minimumBindingSizeBytes`` both describe one physical value. Const references
remain read-only; writable references support initialized readback through the
native loader. Offset views must satisfy the same alignment and minimum-size
checks as other buffers. Pointer arrays and constant-address-space references
retain their existing layouts. This layout contract does not establish support
for unresolved aliases, padded aggregates or every source reference-lowering
case.

Writable entry references remain bound storage rather than value-result entry
parameters. Ordinary helper references retain their existing direction
qualifiers. OpenGL distinguishes value updates through an entry reference from
pointer-offset updates, including when both occur in the same function.
Required native tests cover initialized signed 32-bit scalar writes on all
three targets, including two- and four-component vector writes.
The tests retain buffer guards, aligned offsets and disjoint writable views of
one allocation; Metal also executes the unchanged original source. DirectX
promotes read-only device vector references to structured buffers and constant
references to constant blocks, retaining access modes, component types and
widths in the package interface. Buffer-reference expressions address the
first value, not a mutable pointer offset. Local values and helper parameters
that shadow a global buffer retain their arithmetic updates. Unresolved aliases,
padded source layouts and overlapping writable views are not established by
this coverage.

An HLSL artifact may explicitly distinguish logical binary16 values from native
unsigned 16-bit storage. Its leading contract block contains a versioned resource contract,
serialized by ``crosstl.translator.resource_storage.resource_storage_header``:

.. code-block:: hlsl

   // crosstl-resource-storage: {"schemaVersion":1,"resources":{"values":{"logicalElementType":"float16","encoding":"ieee754-binary16"}}}
   StructuredBuffer<uint16_t> values : register(t0);

Reflection retains ``elementType: "uint16"`` and the actual ``physicalType``;
it adds ``storageEncoding`` containing the logical type and encoding above.
The loader accepts ``float16`` values for that declared representation without
changing their bytes. Shader loads decode with ``asfloat16`` and stores encode
with ``asuint16``; arithmetic remains floating-point arithmetic. This contract
supports tightly packed scalar, vector and homogeneous-struct structured
buffers, and one reflected scalar or vector in a fixed constant buffer.
Textures, padded structures and widened storage are not covered by this codec.
Unknown encodings, duplicate or misplaced headers, undeclared resources and
incompatible physical layouts fail before dispatch. The comment declares an
ABI convention, not proof of the shader's implementation or numerical behavior.

The HLSL generator uses integer storage for native half structured buffers and
promoted entry-point scalar constant references. Local variables, shared memory,
function values and arithmetic retain their logical half types. Structures use
separate physical declarations and member-wise conversion helpers, including
nested structures and fixed arrays; the native loader still requires a
representable reflected layout. Existing user-declared constant buffers and
typed textures are not rewritten. Side-effecting compound-assignment targets
and shadowed bitcast intrinsics produce diagnostics rather than ambiguous code.

This lowering avoids typed ``float16_t`` resource loads, which can quiet
signaling NaNs on the pinned Windows runtime. The required Windows storage gate
passes with the integer representation. An independent readback audit verifies
49 generated cases: 19 copy forms, eight buffer forms, eight exhaustive-test
dispatches covering all 65,536 binary16 words in two storage forms, six
arithmetic/update cases and eight constant-reference payloads. It checks guards,
offset-allocation bytes, freshly regenerated shader
identities and compiled-module identities. Handwritten ABI controls remain
separate from this generated-code evidence. This establishes exact transport for
the tested resource forms, not numerical parity for arbitrary half arithmetic or
complete host-runtime integration.

OpenGL's widened half lowering continues to expose float32 physical storage.
A binary16 payload cannot bind that storage. Conversion tests use the reported
physical representation and verify the half-rounded result; they do not claim
native two-byte OpenGL storage. Required native tests cover copy payloads,
conversion rounding and output guards, but do not by themselves establish
complete MLX half-precision operation coverage.

Metal roundtrip and OpenGL same-type half copies preserve their logical
representation, including NaN payloads, through buffer loads, local storage,
helper arguments and returns,
vector components and homogeneous structures. OpenGL uses exact widened
binary32 words for this contract; callers must supply the representation of
the logical binary16 value. Repacking existing half components does not round
them again. Actual conversions from float32 and half arithmetic still apply
the binary16 rounding contract. Required native copy controls cover every
binary16 word as well as pointer and vector access paths.

Rounding a mathematical result to binary16 does not establish bitwise parity
with every native half-precision intrinsic. The sine control currently matches
the unchanged source in the Metal round trip but differs from the mathematical
reference used by OpenGL. Its evidence retains both comparisons; it is not a
cross-backend parity claim. Compiler/profile accuracy remains tracked in
`issue #2068 <https://github.com/CrossGL/crosstl/issues/2068>`_.

This contract applies to the Python native runtime drivers. The generated C++
DirectX adapter still requires four-byte-multiple structured-buffer strides;
its two-byte view support and encoded-value serialization remain separate work.

Shared Native Allocation Views
------------------------------

A runtime fixture value can include an optional ``allocation`` object to keep
native allocation identity separate from the reflected resource name and
binding coordinate. The allocation object has the following fields:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Purpose
   * - ``id``
     - Required, stable allocation identity. Values attached to separate
       bindings refer to one allocation only when this value is identical.
   * - ``byteOffset``
     - Byte offset of the typed resource view. The default is ``0``.
   * - ``byteLength``
     - Bounded view length in bytes. When omitted, runtime planning derives it
       from the fixture shape and exact scalar layout when possible.
   * - ``allocationByteLength``
     - Total allocation size in bytes. When omitted, runtime planning derives
       it from the greatest known end offset among views with the same ``id``.

The view remains typed by the surrounding fixture value's ``kind``, ``dtype``,
``shape``, and physical layout metadata. Its access mode and binding coordinates
remain part of the corresponding ``resourceBindings`` entry. The allocation
object does not replace those contracts or permit a fixture to reinterpret an
incompatible physical layout.

For example, separate reflected input and output bindings can intentionally
refer to the same full allocation while retaining distinct binding coordinates:

.. code-block:: json

   {
     "inputs": [
       {
         "name": "source",
         "kind": "buffer",
         "dtype": "float32",
         "shape": [2],
         "values": [1.0, 2.0],
         "allocation": {
           "id": "working-set",
           "byteOffset": 0,
           "byteLength": 8,
           "allocationByteLength": 8
         }
       }
     ],
     "expectedOutputs": [
       {
         "name": "destination",
         "kind": "buffer",
         "dtype": "float32",
         "shape": [2],
         "values": [2.0, 4.0],
         "allocation": {
           "id": "working-set",
           "byteOffset": 0,
           "byteLength": 8,
           "allocationByteLength": 8
         }
       }
     ],
     "runtimeAdapter": {
       "resourceBindings": [
         {
           "name": "source",
           "kind": "buffer",
           "binding": 0,
           "access": "read"
         },
         {
           "name": "destination",
           "kind": "buffer",
           "binding": 1,
           "access": "write"
         }
       ]
     }
   }

Runtime planning preserves the explicit allocation ID on each bound resource
and validates the complete group before adapter execution. It rejects malformed
ranges, conflicting total sizes, out-of-bounds or misaligned views, explicit
input/output views that disagree for one binding, incompatible overlapping
physical layouts, and overlapping writable views without a synchronization
plan. Validation diagnostics identify the allocation and affected binding
coordinates or byte ranges. Driver-specific failures also identify the
applicable target constraint. A plan containing these errors is not dispatched.

The DirectX and OpenGL reference drivers group compatible bindings by allocation
ID, combine non-conflicting fixture uploads, create one physical device
allocation for the group, and bind that allocation at each reflected coordinate.
Conflicting upload bytes fail setup instead of causing an implicit conversion or
per-binding allocation. DirectX requires one dtype and stride across the group
and rejects simultaneous aliases of a constant buffer within one dispatch.
An ordered DirectX sequence may reuse an immutable constant allocation across
nodes when every view has the same scalar layout and extent at a 256-byte-aligned offset.
Conflicting uploads, different allocation sizes, and mixing constant buffers with
SRV/UAV resources remain errors. Source constants are uploaded once; derived
region constants retain distinct allocations. OpenGL supports bounded
uniform-block and storage-buffer ranges, subject to the offset-alignment limits
reported by the active context. It rejects mixed uniform/storage groups,
incompatible overlapping scalar layouts, and overlapping writable ranges.

DirectX requests containing allocation subranges or raw byte-address buffers use a process-isolated D3D12
worker. MSVC with the Windows SDK is required to build this worker; unchanged
builds are reused within the Python process. The selected adapter must match
the caller's device. Each allocation ID creates one physical buffer, while
CBV addresses and structured SRV/UAV descriptors preserve the requested offset
and extent. Ordered dispatches keep that allocation alive and synchronize queue
completion before the next dispatch. Output decoding selects the requested
range from a complete allocation readback, retaining its hash and GPU address
in runtime evidence. Execution is bounded to 120 seconds, with a 256 MiB limit
per allocation and 512 MiB per request. No shader rewriting, host evaluation or
independent zero-offset copies substitute for ranged binding.

``ByteAddressBuffer`` and ``RWByteAddressBuffer`` use explicit raw SRV/UAV
descriptors with ``R32_TYPELESS`` format, zero structured stride, and bounds
measured in 32-bit words. Offsets must be multiples of 16 bytes and lengths
multiples of four bytes; backing-allocation padding does not enlarge the view.
The public package loader accepts these resources as ``uint32`` words without
a scalar layout or value encoding. It does not infer a structure from untyped
storage. Runtime evidence identifies raw views separately from structured
views. Typed ``Buffer<T>`` resources remain unsupported by the reference driver.
The existing Windows buffer-view checks include exact raw bounds, output guards
and complete allocation readbacks. Local protocol validation alone does not
establish native execution or resolve a device's structured-buffer or arithmetic
defects. The alignment and descriptor rules follow the
`Direct3D view specification <https://github.com/microsoft/DirectX-Specs/blob/master/d3d/D3D12RevisedCreateViews.md>`_.

The worker records its own process ID and loaded ``d3d12.dll``,
``D3D12Core.dll`` and ``d3d10warp.dll`` paths immediately after device creation.
``directxRuntime.workerRuntime`` retains this observation separately from the
launcher's runtime information. The launcher hashes the files at those observed
paths; these are disk-file identities, not hashes of mapped executable memory.
Unloaded libraries, path-inspection errors and unreadable files remain explicit.
The receipt is retained when a later dispatch fails or exceeds its deadline; failures before
device creation can have no receipt. Successful worker execution requires a
well-formed receipt. Numerical readbacks and guards remain independent checks.

``directxRuntime.workerProgress`` records the last complete worker phase marker,
its zero-based dispatch index when applicable, and elapsed milliseconds since
worker execution began. Markers precede device creation, upload submission,
pipeline creation, dispatch submission and readback; they identify the last
observed phase, not the precise instruction or cause of a failure. Standard
output and error tails are retained with truncation flags, bounded to 8 KiB per
stream. A malformed or partially written runtime receipt is reported separately
and never replaces the original timeout or dispatch error. These diagnostics do
not extend the execution deadline or establish numerical correctness.

For controlled runtime comparisons, ``execute_buffer_views`` accepts an explicit
``sdk_directory`` containing ``D3D12Core.dll`` and an integer ``sdk_version``.
Both are required together. The worker exports the Agility SDK selection symbols
and loads a private snapshot of the supplied core; the version and core digest
are part of its build-cache key. This does not change the Python launcher's SDK
or replace its WARP library. Only use a trusted, architecture-compatible SDK:
this option loads native executable code, not project data. No SDK is downloaded
or selected implicitly.

``directxRuntime.requestedSDK`` records the requested version, source path and
SHA-256. A successful dispatch must report a loaded core with the same digest;
an OS fallback, unreadable core or different loaded library fails with
``sdk-selection-mismatch`` rather than counting as a valid comparison. The
worker still uses the same DXIL, adapter selection, descriptors and numerical
checks. SDK selection alone does not establish arithmetic correctness.

The same executor accepts ``tool_debug=True`` for an explicit WARP pipeline
comparison. The worker checks the selected adapter's LUID against the WARP
adapter before setting ``D3D12_PIPELINE_STATE_FLAG_TOOL_DEBUG``. Hardware
adapters are rejected; the default remains ``D3D12_PIPELINE_STATE_FLAG_NONE``.
Receipts retain ``requestedPipelineStateFlags`` and require the worker's
``workerPipelineStateFlags`` confirmation for successful debug dispatches.
Shader bytes, buffer descriptors and execution limits do not change. Microsoft's
`pipeline flag contract <https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_pipeline_state_flags>`_
describes additional debugging information, not an arithmetic workaround or
a guarantee that optimizations are disabled. A passing diagnostic does not
qualify default execution.

``debug_layer=True`` independently enables the Direct3D validation layer before
device creation. It requires the Windows graphics debug components and never
falls back silently when they are unavailable. The option retains up to 64
``ID3D12InfoQueue`` messages with identifiers, categories, severities and bounded
descriptions under ``directxRuntime.debugLayer``. Description truncation and
discarded-message counts are explicit. Corruption/error messages, discarded
messages or a missing/malformed receipt prevent a successful validation result.
A dispatch failure or timeout remains the primary error; any available validation
messages accompany it rather than replacing it. This option does not change the
shader or enable GPU-based shader instrumentation. Neither debug option is
enabled during ordinary execution.

Ranged execution supports shared read-only views, disjoint UAV views and
cross-dispatch read/write reuse. Simultaneous SRV/CBV reads and UAV writes to one
allocation within a dispatch remain unsupported by the worker's classic
resource-state model and fail with ``allocation-state-incompatible``. This
limitation is not evidence that an equivalent enhanced-barrier implementation
is impossible. Windows CI requires native offset, shared-allocation and
ordered-reuse controls, including complete allocation guards.

The built-in Vulkan driver does not currently realize shared allocation IDs or
bounded allocation views, and no native shared-allocation support is claimed for
Metal, WebGL, WGSL, CUDA, HIP, Mojo, Rust, or Slang targets. Runtime-plan
serialization on those targets is not evidence that one physical allocation was
reused. Target-specific synchronization, resource-state transitions, allocation
lifetime, and framework memory planning remain host-runtime responsibilities.

The ``allocation`` field is optional for backward compatibility. When it is
absent, runtime planning assigns a deterministic per-binding allocation ID, so
existing independent bindings remain independent. A single initialized
``read_write`` binding continues to use one allocation for upload and readback.
Aliasing between separate bindings is never inferred from equal values or
similar names; it requires an explicit shared ``id``.

The translator stops at this contract boundary. Full framework rewrites,
non-kernel host API ports, application command scheduling, target SDK
installation, build-system migration, memory lifetime policy, and production
runtime framework generation remain downstream integration work.

Runtime Execution Graphs
------------------------

The runtime execution graph API represents a bounded multi-operation workload
without embedding target runtime calls in the project report. Use
``parse_runtime_execution_graph`` to load a versioned graph,
``validate_runtime_execution_graph`` to obtain structured diagnostics,
``inspect_runtime_graph_package`` to verify packaged artifact and interface
references, and ``execute_runtime_graph`` to run a supported native graph. The
types and functions are available from ``crosstl.project``.

A graph declares resources separately from operations. Resource records carry
their role, kind, exact physical layout, optional allocation view, and bounded
lifetime. Nodes use stable IDs and explicit ``dependsOn`` edges and have one of
the following operation records:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Node kind
     - Contract
   * - ``dispatch``
     - Selects one translated artifact and entry point, maps named bindings to
       graph resources with explicit access, and records dispatch geometry and
       constants.
   * - ``copy``
     - Describes bounded source and destination byte ranges between compatible
       resources.
   * - ``fill``
     - Describes a bounded byte range and scalar fill value for one resource.
   * - ``barrier``
     - Makes an explicit write-to-read or write-to-write visibility transition
       for named resources between dependency-ordered operations.

Dispatch, copy, and fill nodes may also carry bounded repeat or condition
records. Validation rejects unbounded controls, dependency cycles, missing
resource or artifact references, unsafe access ordering, missing visibility
barriers, incompatible layouts or ranges, and temporary-resource lifetimes that
do not cover their producers and consumers. Failures are returned as structured
diagnostics with graph, node, resource, path, and missing-capability context;
unsupported constructs are not silently removed.

Package inspection validates the graph before reading package metadata. For
each dispatch it resolves the artifact selector deterministically, requires one
ready packaged artifact, checks the requested entry point, and verifies named
binding presence, uniqueness, and access compatibility against the reflected
host interface. Inspection is read-only and records explicitly that device
execution was not performed.

The DirectX and OpenGL reference runtimes currently execute dependency-ordered
``dispatch`` and ``barrier`` paths. All requests are validated before device
work begins, resources are allocated once for the graph sequence, and a
payload-free temporary resource can retain one physical allocation from its
producer through its consumer. Intermediate results remain on the device and
output readback is deferred until the sequence completes. The native proofs
execute a two-stage reduction: an input of shape ``[4]`` containing
``[0, 1, 7, 42]`` produces two partial sums in a temporary of shape ``[2]``,
then an output of shape ``[1]`` containing ``[50]``. Direct3D 12 and OpenGL 4.3
both verify the exact result.

``copy`` and ``fill`` nodes and bounded control records are part of the
serializable and validated graph model, but the native executor currently
returns explicit unsupported-capability diagnostics for them. Vulkan native
graph execution is not supported, and no native graph execution is claimed for
Metal, WebGL, WGSL, CUDA, HIP, Mojo, Rust, or Slang. The backend-neutral graph
contract can still be parsed and validated independently of those targets.

The pinned MLX revision
``4367c73b60541ddd5a266ce4644fd93d20223b6e`` is a corpus and CI reference for
the runtime-porting work, not a special case in the graph schema or executor.
This capability does not rewrite host applications or runtime frameworks and
does not claim that the complete MLX runtime or upstream test suite has been
ported.

Project Runtime Fixture Manifest Generation
-------------------------------------------

Curated repository fixture metadata can be converted into a standard
``crosstl-project-runtime-test-manifest`` document without adding a native
runtime adapter. This lets project ports describe parity cases as deterministic
inputs, expected outputs, tolerances, artifact selectors, runtime adapter
contracts, resource bindings, function or specialization constants, and dispatch
geometry, then reuse the existing runtime test manifest parser, planner, and
report writer.

Build a project runtime test manifest from a translated artifact report or
runtime artifact manifest plus curated fixture metadata:

.. code-block:: bash

   python -m crosstl runtime-test-manifest \
     crosstl-out/runtime-manifest.json \
     fixtures/runtime-fixtures.json \
     --format text

The fixture metadata convention is a small repository-agnostic input document:

.. code-block:: json

   {
     "kind": "crosstl-project-runtime-fixture-metadata",
     "fixtures": [
       {
         "id": "reduced-binary-add-f32",
         "selector": {
           "source": "mlx/backend/metal/kernels/binary.metal",
           "target": "metal",
           "path": "out/metal/reduced_binary_add.metal"
         },
         "inputs": [{"name": "lhs", "values": [1.0, 2.0]}],
         "expectedOutputs": [{"name": "out", "values": [3.0, 4.0]}],
         "runtimeAdapter": {
           "entryPoints": [{"name": "binary_add", "stage": "compute"}],
           "resourceBindings": [
             {"name": "lhs", "kind": "buffer", "binding": 0, "value": "lhs"},
             {"name": "out", "kind": "buffer", "binding": 1, "value": "out"}
           ],
           "functionConstants": [
             {"name": "element_count", "id": 0, "value": 2}
           ],
           "dispatch": {"entryPoint": "binary_add", "globalSize": [2, 1, 1]}
         }
       }
     ]
   }

The generator validates each fixture selector against the translated artifacts.
Incomplete fixture data, duplicate ids, missing expected outputs, unresolved
artifacts, and ambiguous selectors are emitted as structured diagnostics on the
generated manifest. Valid fixture records remain in the manifest so
``plan_runtime_test_manifest`` and ``verify_runtime_test_manifest`` can apply
the same adapter dependency checks and runtime planning used by hand-authored
manifests.

Generated test records also include ``metadata.runtimeMetadata``. When the
selected artifact carries ``runtimeDataStatus`` from a runtime artifact
manifest, that status is preserved; otherwise the generator derives readiness
from the merged runtime adapter contract. The manifest summary includes
``runtimeMetadataStatusCounts`` so downstream tooling can separate incomplete
fixture data from incomplete artifact metadata before attempting native runtime
execution.

Build a runtime loader manifest from a runtime package manifest:

.. code-block:: bash

   python -m crosstl runtime-loader-manifest \
     crosstl-runtime-package/runtime-package.json \
     --format text

Runtime loader manifests emit a ``crosstl-runtime-loader-manifest`` JSON
document derived from runtime adapter planning. The manifest groups per-target
load units with package-relative artifact paths, adapter kind, artifact format,
source-remap handoff paths, parser-derived ``hostInterface`` metadata when
available, required target tools, host responsibilities, ordered loader steps,
and blockers that must be resolved before a host loader or build-system adapter
can consume the artifact safely. Unavailable interface reflection remains an
explicit ``resolve-host-interface-metadata`` blocker instead of being hidden or
treated as generated host code. The manifest carries package inspection
diagnostics and runtime-reference review actions forward, and it remains a
metadata contract only: it does not rewrite host application code, execute
device code, generate runtime framework code, or install target SDKs.

Native Metal Package Execution
------------------------------

``MetalRuntimeParityAdapter`` and ``MetalComputeRuntime`` execute compute
artifacts through the public package, loader descriptor and dispatch-request
APIs on macOS 13 or newer. They require Xcode's Metal and Swift tools and an
available Metal device. No additional Python GPU binding is required. The
runtime compiles an identity-checked source snapshot with warnings fatal and
fast math disabled, links a Metal library, and runs a shipped Swift worker.
``MetalRuntimeParityAdapter(language_version="4.1")`` selects an explicit
Metal language version when the source requires newer features, such as
acquire/release atomic fences. The default leaves version selection to the
installed compiler. The compile command includes the requested version; an
unsupported version fails compilation without a fallback. An
existing ``.metallib`` is not recompiled or relabeled by this option.

Requests require explicit or reflected ``workgroupSize`` and
``workgroupCount``; the runtime does not guess a group size from a maximum-thread
attribute. Buffer arguments share one index namespace, from 0 through 30, in
resource set zero. Scalar buffers, tightly packed two-/four-component 32-bit
vectors and flat homogeneous scalar structs retain their exact physical layout.
``constant T*`` parameters are read-only runtime buffers; supported
``constant T&`` scalar/vector parameters are fixed-size values. Metal ``long``
and ``ulong`` scalar storage uses signed/unsigned 64-bit host values. Scalar
``bool`` buffers and constants use one-byte storage with Boolean host values;
integer values are not implicitly converted. Readback rejects bytes other than
zero and one. Padded vectors, half storage, Boolean vectors, nested or mixed
structs, textures, samplers and dynamic threadgroup arguments are not supported
by this buffer runtime.

Dispatch values describe physical storage, not a target-independent tensor
encoding. DirectX and OpenGL Boolean buffers retain their reflected
``uint32`` representation with four-byte elements. Callers must use that
representation rather than upload Metal's one-byte Boolean payloads. The
loader rejects mismatched widths. Required native checks exercise translated
comparison and mask kernels, guarded odd-sized buffers, read-modify-write
values and offset views; macOS also executes the original Metal source.
This storage support does not by itself implement MLX Boolean host operations.

The worker specializes Boolean, float32, int32 and uint32 function constants
by numeric ID, verifies required compiled buffer arguments and alignment,
checks device and pipeline threadgroup limits, dispatches three-dimensional
threadgroups, synchronizes and reads the requested byte views. Compatible
aliased views share one allocation; contradictory overlapping initial bytes
are rejected. The default combined buffer limit is 256 MiB, configurable with
``MetalComputeRuntime(max_buffer_bytes=...)``. This is not dynamic shader
memory-safety analysis.

Compilation, probing and execution each have a default 120-second deadline,
configurable with ``MetalRuntimeParityAdapter(timeout_seconds=...)``. A timeout
terminates and reaps the worker process group and reports a structured failure;
it never substitutes host computation. The default runtime shares a verified
helper executable within the Python process when the helper source, compiler
options, toolchain, SDK and architecture are unchanged. Every probe and dispatch
still starts a separate bounded process; shader compilation, GPU state and
readbacks are not cached. ``close()`` releases instance-owned resources without
invalidating another instance's shared helper. Shared executables are removed
at process exit; custom command runners or tool resolvers retain private helpers
that ``close()`` removes. Device name,
compiled-library hash, execution width and dispatch geometry accompany readback
evidence. The implementation follows Metal's
`compiled binding reflection
<https://developer.apple.com/documentation/metal/mtlcomputepipelinereflection>`_.

Required macOS CI covers package execution, sparse bindings, multidimensional
indexing, function constants and invalid contracts. The current MLX binary-shape
gate additionally runs all 15 selected complex-power shapes through this public
runtime, alongside original-source and roundtrip Metal comparisons. This adds
a native reference path; it does not provide persistent MLX streams, automatic
host-code redirection, a generated Metal C++ loader, full upstream-suite parity,
or the remaining runtime adapters tracked by issue #1424.

A separate `MLX Metal host integration harness
<https://github.com/CrossGL/crosstl/blob/main/demos/integrations/mlx/METAL_HOST.md>`_
builds the pinned upstream runtime with a documented per-entry library resolver.
It runs the unchanged upstream operations module with selected translated
complex-power entries and records dispatches from MLX's own command encoder.
All 15 independently emitted entries must compile and link into one Metal
library before host execution. Source ``static`` and anonymous-namespace helpers
retain internal linkage, including materialized template helpers. Inline
definitions and implicit template instantiations retain repeatable linkage, and
generated constructors and lowered member helpers remain artifact-private.
Required native tests also link unrelated modules with identical private helper
names, verify their distinct results and exercise exported visible callables.
Declaration-only external functions remain a separate limitation: the Metal
frontend currently drops those declarations, so a translated caller cannot link
against a definition in another source file. This is tracked in
`issue #1978 <https://github.com/CrossGL/crosstl/issues/1978>`_; the selected MLX
library proof does not exercise or establish that capability.
Eight array layouts execute three datasets with 402 retained complex readbacks,
including both signed-zero branch cuts. Per-dataset traces must account for each
selected entry exactly once, and the verifier recomputes references independently
of the workload's success flags. Original MLX JIT signed-zero behavior is documented
separately and is not used as the numerical reference for these additional cases.
Other operations remain on the original backend; this is partial, explicit host
integration, not automatic C++ runtime translation or full upstream-suite parity.

The `portable MLX host adapter
<https://github.com/CrossGL/crosstl/blob/main/demos/integrations/mlx/portable_host/README.md>`_
instead builds MLX with its original Metal and CUDA backends disabled. Its
synchronous callback connects ``Arange`` for five scalar types and 30 float32 unary entries to public
DirectX/OpenGL/Metal runtime packages. Shared-buffer view primitives reuse upstream
shape, stride and ownership logic, including broadcasts, transposes, splits and
non-copying reshapes. Layout-changing contiguous conversions, reshapes, flattening
and unflattening dispatch the unchanged uint32 general-copy specialization for
float32/int32/uint32 storage. Bounds-checked, rebased source spans preserve negative
strides; integer-word transport preserves NaN payloads and subnormals. Six binary
primitives dispatch 16 unchanged entries: addition, subtraction, multiplication,
minimum and maximum for float32/int32/uint32, and float32 division. Translated
copies materialize non-row-contiguous and broadcast inputs before arithmetic;
no CPU elementwise fallback is used. Unsupported casts remain explicit errors. Other
primitives retain upstream unsupported-GPU errors.
Linux/OpenGL, Windows/Direct3D 12 and macOS/generated Metal CI build the adapted
library, require 21 unchanged upstream tests against CPU and translated GPU execution, and check
20 array-creation records plus 130 unary records against independent references,
including 8,193 consecutive binary32 inputs at and above one for inverse
hyperbolic cosine at unchanged upstream tolerances.
The verifier reconstructs the five allowed source adaptations from the pin,
requires exact bytes before and after execution, and rejects unrelated tracked
changes. Forty additional records cover view layouts, chained unary dispatch,
source preservation and exact 64-bit values. The verifier independently checks
complete numerical records, another 33 copying/source-preservation records with
1,716 exact storage words, 134 binary records with 5,274 outputs, the initial 360
nonempty dispatches, all 52 entries, and eleven required
rejection cases before publishing a schema-version-2 summary. Copy checks include
invalid allocation bounds, unsupported storage widths and excessive sizes. Each
copy dispatch retains its geometry and checks a 128-byte destination guard.
Binary checks require preserved operand words, exact integer results, zero signs,
nonfinite classification and ``rtol=2e-6, atol=1e-6`` for finite float32 outputs.
Minimum/maximum ties select the second operand as specified by the pinned source.
Binary dispatches also retain a checked 128-byte destination guard. Required native
conditional-selection controls cover operand bits and lazy branch evaluation.
Unary execution is limited to contiguous float32 inputs and 65,535 stored
elements. Source
checks do not attest to a separately supplied binary; CI retains its build log.
The Metal path compiles the generated package with warnings fatal and fast math
disabled; the original MLX Metal backend is unavailable in this build. All three
targets use explicit one-thread-per-workgroup dispatch. This is selected
host-staging coverage, not a complete translated MLX backend.

Exact Scalar Physical Resource Layouts
--------------------------------------

Source reflection records a ``scalarLayout`` only when the target-language
resource has an exact physical representation covered by the project runtime
contract. HLSL reflection supports ``StructuredBuffer`` and
``RWStructuredBuffer`` resources whose element type is a scalar or up-to-four
component vector of ``float``, ``int``, or ``uint``, plus scalar ``int64_t``
and ``uint64_t`` elements. Single-member ``cbuffer`` declarations support the
same types. GLSL reflection supports explicit ``std430`` buffer blocks
containing one scalar runtime-array member and explicit ``std140`` uniform
blocks containing one scalar member of type ``float``, ``int``, ``uint``,
``int64_t``, or ``uint64_t``.

The reflected layout records ``physicalType``, ``elementType``,
``elementSizeBytes``, ``elementStrideBytes``, ``alignmentBytes``,
``memberOffsetBytes``, ``storageLayout``, and ``runtimeSized``. It also records
``memberName`` when the source block provides one and ``blockSizeBytes`` for a
fixed scalar block. These fields are preserved through runtime packages and
native loader ABI descriptors. Descriptor-to-request validation rejects
missing, incomplete, or mismatched layouts instead of inferring a host ABI.

Native runtime allocation consumes the same physical contract. DirectX
constant-buffer views require a fixed HLSL scalar block and allocate at least
the reflected block size, rounded to the 256-byte API alignment. OpenGL
uniform buffers require a fixed ``std140`` scalar block and zero-pad the upload
to ``blockSizeBytes``; ``std430`` scalar runtime arrays retain their logical
payload size for storage-buffer readback.

Standard GLSL ``vec``, ``ivec``, ``uvec``, and ``bvec`` block members now
receive exact component widths, vector widths, and ``std140``/``std430``
alignment metadata; ``i64vec`` and ``u64vec`` use the supported 64-bit physical
tables. Runtime ``vec2`` and ``vec4`` storage arrays remain tightly packed,
while ``vec3`` records its logical element size and padded array stride
separately. The current native loader rejects padded storage vectors rather
than uploading a falsely tight layout. GLSL ``dvec`` values, HLSL 64-bit
vectors, matrices, fixed arrays, unsupported aggregate shapes, narrow or
floating-point scalar widths, implicit GLSL block layouts, arbitrary member
offsets, and multi-member HLSL blocks do not receive usable loader metadata. Those
shapes remain unresolved or fail closed when a native loader request requires
a physical layout. Native requests range-check signed and unsigned 64-bit
values and preserve them with little-endian 8-byte packing; 64-bit
specialization constants remain intentionally unsupported.

Flat homogeneous structs are supported as HLSL structured-buffer elements and
GLSL ``std430`` storage-array elements. Each may contain 1-64 members of one
supported scalar type: ``float``, ``int``, ``uint``, ``int64_t``, or ``uint64_t``.
The layout retains the actual struct name as ``physicalType``, a
``componentCount``, and ordered ``structMembers`` containing each member's
``name``, scalar ``physicalType``, and ``offsetBytes``. Structs are not relabeled
as native vectors: two float members have an 8-byte element size and 4-byte
alignment on these storage paths. Allocation sizing divides the flattened scalar
count by the member count before applying the element stride.

Dispatch validation requires unique member names, exact homogeneous types and
offsets, tight stride, complete elements, and the matching target storage class.
Nested structs, mixed scalar types, arrays, padded records, explicit member
qualifiers and duplicate declarations remain unsupported on these storage paths.
No MLX-specific type-name mapping is used.

Fixed OpenGL uniform blocks may contain mixed supported scalar/vector members.
Their ``scalarLayout`` retains the block name as ``physicalType`` and ordered
``blockMembers`` with names, actual GLSL types, component types, vector widths,
byte offsets, sizes and alignments. Packing follows the `OpenGL std140 rules
<https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf>`_ (section 7.6.2.2).
For example, ``int, vec2, vec3, float, uint`` members occupy offsets
``0, 8, 16, 28, 32`` in a 48-byte block. This is the generated target layout,
not an assumption about the original language's struct ABI.

``payloadEncoding: uint32-le-words`` identifies byte transport, not homogeneous
shader data. Inputs provide the complete little-endian block as uint32 words,
including internal and trailing padding; no numerical conversion of mixed
fields is performed. Reflection, package descriptors, contract comparison and
dispatch retain the same member records. Both loader and driver independently
validate the supplied layout against standard packing before upload. Missing,
overlapping, misaligned or contradictory records and wrong payload lengths are
rejected. Arrays, matrices, nested types and explicit member qualifiers remain
unsupported and produce incomplete-reflection diagnostics. Required Linux CI
executes translated mixed scalar and padded scalar/vector controls through the
public package and native-loader APIs.

For a single reflected HLSL or GLSL entry, ``minimumBindingSizeBytes`` records
the minimum buffer footprint proven by constant-index accesses in its mandatory
straight-line prefix. Analysis uses the existing target parsers, follows unique
helper definitions, and propagates buffer identity and bounded integer offsets.
Known single-argument HLSL and GLSL bitcasts retain accesses in their operands,
including HLSL's 16-bit storage bitcasts. Their result bits are not treated as
known integer offsets. User-defined overloads still follow the helper rules;
output-parameter overloads such as three-argument HLSL ``asuint`` stop analysis.
It stops at unresolved calls, ambiguous overloads, recursion and dynamic control
flow. Unresolved preprocessing disables this analysis; array parameter extents
alone do not establish required sizes. Missing metadata means unknown, not zero.
This is a lower bound from proven accesses, not a complete dynamic bounds check.

Packaging and native loader descriptors preserve this requirement. Runtime
preflight checks both the bound view and the provided values against the binding's
physical layout. A larger backing allocation or weaker value metadata cannot
make a short view valid. Larger usable buffers remain accepted. Malformed or
overflowing minimum byte counts and undersized views produce structured errors
before native execution. Required native CI covers a general helper-based
buffer access and rejects truncated stride buffers for the current MLX two- and
three-dimensional complex-power entries before executing the unchanged valid
numerical workloads.

The storage rules follow `DXC buffer packing
<https://github.com/microsoft/DirectXShaderCompiler/wiki/Buffer-Packing>`_ and
the `GLSL buffer layout specification
<https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html#uniform-and-shader-storage-block-layout-qualifiers>`_.

Required native CI exercises a reduced two-field transform and the unmodified
``g1_Powercomplex64`` entry from MLX commit
``9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8``. The latter runs 256 finite and
zero-base cases through translated HLSL/GLSL runtime packages; macOS compiles
and executes both original and roundtrip Metal. The numerical bound is
``5e-5 * max(1, abs(reference))`` for the complex absolute error. Reports retain
every input, output, error and bound, and buffers use nonzero sentinels to detect
missed writes. This proves the selected kernel and layout path, not full binary
family coverage, upstream MLX-suite execution, or MLX host-runtime redirection.

The separate binary-shape gate discovers and executes every one of the 15
``Powercomplex64`` entry shapes at that pin. Its 873 complex outputs exercise
scalar/vector broadcasting, two- and three-dimensional dispatch, non-contiguous
storage, zero strides, 32/64-bit index paths and partial final tiles in
four-dimensional gathers. Each shape retains its random workload and adds two
branch-cut workloads: negative-real bases with positive or negative imaginary
zero, raised to the power one half. Each dataset retains its own inputs and
readback evidence, without weakening the complex-error bound above.
Expected input locations are enumerated from logical
coordinates and strides independently of the shader's index helpers. Metal
executes original and translated kernels and separately consumes the public
runtime package, as DirectX and OpenGL do. All three Metal paths execute each
dataset, retaining 873 comparisons per path. Package construction happens once
per entry; each dataset still receives fresh buffers, artifact identity checks,
native compilation and its own device/library/dispatch evidence. DirectX also
compiles with warnings fatal, and OpenGL validates a generated SPIR-V 1.3 module
before native GLSL execution.

The gate rejects a changed source census, missing native tools, missing entry
outputs and numerical mismatches. OpenGL's explicit ``[0, 511]`` index-range
assertions apply only to these bounded workloads; they are not inferred bounds
for arbitrary tensors. A 900-second outer deadline and always-uploaded artifacts
preserve failures without converting them to optional skips. This extends
access-shape coverage for one operator and type, not every binary operation or
the upstream MLX test suite.

At pinned MLX commit ``4367c73b60541ddd5a266ce4644fd93d20223b6e``, the
``arangeuint32`` entry from ``arange.metal`` is translated to DirectX and
OpenGL, reflected, packaged, converted through the public native loader bridge,
and executed in Windows Direct3D and Linux EGL CI. With ``start = 3``,
``step = 2``, and four invocations, both readbacks are exactly
``[3, 5, 7, 9]``. This is an end-to-end proof for one scalar kernel contract;
it is not a claim of vector or aggregate layout support, full MLX runtime
integration, or MLX test-suite parity.

At current pinned MLX commit
``846d176227a0ac13d2667e58d2bb68b322109ab0``, the selected
``fft_mem_256_float2_float2`` entry also passes an entry-scoped OpenGL proof.
Six index assertions and one 256-element workgroup-access assertion
bound its host contract. The resource-helper bound applies only to one
unrebased 256-point transform with batch size one and grid ``[1,1,64]``;
it must not be reused for arbitrary tensors. Translation materializes 37 specializations from 42
reachable records, prunes 2,120 candidates, and preserves 21 reachable
function constants for deferred specialization. The deterministic GLSL is
109,547 bytes with SHA-256
``5c6fefea7315d7d091641d024aea27bbdcecf1f2f4b7a63a2db5af401e036512``;
``glslangValidator`` and ``spirv-val`` accept it, and its SPIR-V has 19 control
barriers with no group-nonuniform instruction.

The reflected ABI has two ``std430`` float32 ``vec2`` arrays at 8-byte size,
stride, and alignment plus two 16-byte ``std140`` integer blocks. The runtime
variant registry has no blocked keys and produces a verified deferred SPIR-V
request for workgroup size ``[1, 1, 64]``. Linux Mesa llvmpipe executes five
inputs: an index-1 unit impulse, seeded float32 complex data, a complex
constant, alternating real values, and a final-element complex impulse.
Across these dispatches, 2,560 scalar outputs must match analytical references
or a double-precision direct DFT at ``2e-4`` absolute and relative tolerance;
40 trailing scalar guards must remain exactly unchanged. Translation and
packaging run once; the first dispatch publishes the verified compilation cache
and the remaining four must hit it. The same controls are required in the
existing Windows native-loader jobs, without adding runners. Changed HLSL
references still require Windows numerical execution before merge.
This is one selected current-pinned plan; it does
not redirect the MLX host runtime, cover other FFT plans or dtypes, prove a
Metal round trip, or establish full backend parity.

At the same current pin, a bounded GEMV proof selects
``gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0`` for one contiguous
float32 vector-matrix product with ``M=1``, ``N=32``, and ``K=32``. The
host-derived contract fixes workgroup ``[32, 2, 1]``, subgroup width 32, and
one dispatched workgroup. Entry-scoped translation materializes the selected
GEMV and ``elem_to_loc_uint`` only. Its 8,188-byte HLSL has SHA-256
``f300bbea75b2ed9e47c29313a56f882ed848cbb93858f1347fbc97a60e167223``
and passes official DXC 1.9.2602.24 under ``cs_6_6``,
``-enable-16bit-types``, and warnings as errors. DirectX explicitly enables a
32-lane target-scoped software subgroup because physical waves need not contain
contiguous flattened ``SV_GroupIndex`` values. Logical subgroup and lane IDs
are ``SV_GroupIndex / 32`` and ``SV_GroupIndex % 32``. A 64-float
``groupshared`` array carries each shuffle between two
``GroupMemoryBarrierWithGroupSync`` calls; the source lane is validated before
addition so an extreme unsigned delta cannot wrap the scratch index, and an
out-of-range source returns the calling invocation's value. The HLSL contains
no ``WaveReadLaneAt``, ``WaveGetLaneIndex``, or physical-wave atomic allocator.
``[WaveSize(32)]`` remains the source/reflection contract without making the
reduction depend on physical lane topology.

The superseded 8,410-byte physical-wave artifact remains recorded as rejected
evidence under SHA-256
``f8f1107d0de251fd300c7a16ce6638796bd08dd2eadd8f7959e37c78d0aa170d``.
Windows workflow run 33268998061, job 99143984804 mismatched every output: the
reduction replaced logical lanes 5 through 8 with physical lanes 21 through
24, reaching maximum absolute error 1.90625. This exact substitution rules out
a tolerance adjustment or merely guarding invalid high-lane reads.

OpenGL uses two logical 32-lane software subgroups in the 64-thread workgroup.
Both target-specific fail-closed analyses admit the source's
``sm >= 1; sm >>= 1`` loop as the integral-equivalent positive-to-zero form of
``sm > 0``. DirectX also requires one bounded compute entry, concrete
width-compatible dimensions, explicit calling-invocation fallback, a supported
scalar shuffle, unambiguous helper identity, logical invocation identity, and
statically uniform control flow before artifact emission. OpenGL retains
rejection for wider bounds, mutation, nontermination, escaping control flow,
and indirect or nested helper calls. The 7,705-byte GLSL has SHA-256
``f5ef8900ee65d63a6df2818ef111f56b4f269c6366c82d82a9d97c967042f562``;
``glslangValidator`` and ``spirv-val`` accept it, and its SPIR-V contains three
control barriers with no group-nonuniform instruction.

The reflected DirectX and OpenGL ABIs each contain 15 resources, including
signed 64-bit batch strides and seven scalar argument blocks. A deterministic
binary-fraction workload compares all 32 output columns at ``1e-5`` absolute
and relative tolerance. Linux Mesa llvmpipe executes the software-subgroup
artifact in required mode, and Windows CI requires the same workload through
Direct3D 12 WARP. This proof covers one host-valid current-corpus entry; gather,
wide, batched, axpby, and remaining GEMV entries, MLX host redirection, selected
Metal compilation, and the full MLX suite remain outside the claim. It does
not change the separate historical 224-entry aggregate compiler gates.

The same current pin has a bounded MXFP4 quantize/dequantize contract for
``mxfp4_quantize_dequantize_float_gs_32_b_4_hgs_false``. Host provenance from
``quantized.cpp`` and ``fp_quantized.h`` fixes float32 input, group size 32,
four payload bits, no global scale, workgroup ``[32, 1, 1]``, and one dispatched
workgroup. Entry-scoped translation materializes only that specialization. The
9,123-byte HLSL has SHA-256
``3fe38e171ba8c8ea1adfc8efad20b242ca02dd05e1a5a53a9b9d1e18459d8c7d`` and
passes DXC under ``cs_6_6``, ``-enable-16bit-types``, and warnings as errors.
Its 4,716-byte DXIL uses the explicit DirectX-only
``project.source_options.metal.target_options.directx.widen_native_float16``
mode. Source ``as_type<float16_t>(uint16_t)`` reconstructs its payload directly
as float32 with integer IEEE-754 masks, while logical ``float16_t`` locals,
parameters, and returns stay widened through arithmetic, sign application, and
the consuming conversion. DXIL contains ``uitofp i32`` and ``fmul float`` but
no ``LegacyF16ToF32``, ``half``, ``fptrunc``, or ``fpext``. The default native
binary16 contract remains exact ``asfloat16``/``asint16``/``asuint16`` and is
unchanged when this target-scoped option is absent.

Four Windows failures bound this contract. The 7,809-byte HLSL under SHA-256
``3591e38d20a612b4061fe3154ef0ea3deb035283294fbd27376ef90627569361``
produced numeric ``uitofp i16 ... to half``; workflow run 33271117475, job
99149649480 collapsed all 28 nonzero values to signed zero on WARP. Exact
``bitcast i16 ... to half`` produced same-size HLSL under SHA-256
``4e8044758d65b6b2c189092ce56fff3c5ba7948221883de490c1a4b9c5563352``,
but run 33272842347, job 99154326814 consumed the constructed subnormal in
``fmul half`` and produced the same result. Moving arithmetic to float32
produced 7,909-byte HLSL under SHA-256
``938ca6fac1c47ea633453836b5d76833c294853bb92d6a410a2c4772dd7fa627``;
``dx.op.legacyF16ToF32`` still failed in run 33274360343, job 99158370210.
Integer decoding produced 9,240-byte HLSL under SHA-256
``936088a24a6b575e50dc97e16a4c0dca63a76200ddd94d5211e4bf312fec1625``,
but the remaining ``fptrunc float to half``, half sign operation, and
``fpext half to float`` returned the identical 28 signed zeros in run
33275550062, job 99161501105. Removing every half instruction produced a
9,118-byte artifact under SHA-256
``7afdc612f9091ae47abca8c4fd9d2171e8ea42c6539e02a40bbad2de7d1a7c6a``,
but run 33277494856, job 99166677942 still returned the same signed zeros.
DXIL then exposed the actual remaining defect: absent Metal/C++ scalar integer
promotion reduced ``uint16_t(bits) << 23`` to a 16-bit shift by 7 plus an
``0xffff`` mask. The corrected HLSL emits ``int(uint16_t(bits)) << 23`` and
DXIL shifts the 32-bit value by 23 before ``asfloat``, with no native-half
operation anywhere in the selected path.

The 9,571-byte explicit-software-subgroup GLSL has SHA-256
``cbbe989c40317c04ffe915f1f314f55db8896edfd38f04ad4b8882be53b2a4da``;
``glslangValidator`` and ``spirv-val`` accept its three-barrier SPIR-V, which
contains no group-nonuniform instruction. GLSL widens binary16 values to
float32, so the same source bitcast preserves the exact low 16-bit payload
through ``unpackHalf2x16`` rather than a float32 bitcast; inverse forms use
``packHalf2x16`` with exact low-bit extraction.

Exact scale semantics require the source ``fp8_e8m0(float)`` constructor
factory before conversion back to float. Qualified ``metal::round`` maps to the
portable math intrinsic, while OpenGL ``metal::isfinite`` and ``signbit`` use
single-evaluation IEEE-754 bit tests. Read-only private scalar-to-struct views
are accepted only for exact one-member layouts. Unsupported predicate types,
receiver mutation, writes, and unresolved constructor branches remain
fail-closed.

The reflected data ABI is float32 input at binding 0, an inert declared
``global_scale`` input at binding 1, and float32 read-write output at binding 2.
DirectX also uses ``b0`` for generated dispatch input, legally distinct from
``t0`` in the HLSL register namespaces. The real host omits ``global_scale``
for this specialization; the generic loader allocates it because reflection
retains the declaration, and the selected code contains no read. A 32-element
workload uses only exactly representable FP4 E2M1 values with maximum magnitude
6. The scale divisor is 6 and the MX scale is exactly 1, so Windows WARP and
Linux Mesa required-mode tests demand bit-exact float32 readback at zero
tolerance. Other quantized entries and parameter families, MLX host redirection,
selected-entry Metal compilation, and full-suite parity remain outside this
bounded proof.

At current pinned MLX commit
``846d176227a0ac13d2667e58d2bb68b322109ab0``, a bounded arg-reduce proof
selects ``argmin_float32`` and ``argmax_float32`` for two axis-32 rows. The
checked host dispatch formula produces workgroups ``[32, 1, 1]`` and dispatch
``[1, 2, 1]`` with subgroup width 32. Signature-aware source instantiation
materializes the scalar ``elem_to_loc<int64_t>`` helper rather than its
``uint3`` overload. DirectX generation explicitly sets
``project.source_options.metal.target_options.directx.relative_wave_shuffle_out_of_range``
to ``"self"`` for these artifacts. Relative source lanes outside the wave then
retain the calling lane's value; generated helpers select a proven in-range
source before an unconditional ``WaveReadLaneAt``. This deterministically
refines source undefined behavior without changing in-range shuffle semantics.
The default policy remains ``"undefined"`` so unrelated project artifacts are
unchanged. The generated HLSL artifacts pass official DXC 1.9.2602.24 under
``cs_6_6`` with ``-enable-16bit-types`` and warnings as errors. OpenGL uses the
explicit software subgroup and admits direct shuffle-helper calls only inside
a proven canonical workgroup-uniform halving loop. Both GLSL modules pass
``glslangValidator`` and ``spirv-val``, contain five control barriers, and
contain no group-nonuniform SPIR-V instruction.

The reflected runtime contract includes exact signed and unsigned 64-bit
scalar resources: float32 input, uint32 output, int32 shape, int64 stride, and
uint64 size buffers or scalar blocks. Linux arm64 Mesa EGL executes rows with
tied extrema and reads back argmin indices ``[5, 7]`` and argmax indices
``[3, 2]``, proving lowest-index tie behavior; Windows CI requires the same
workloads through Direct3D 12 WARP. This bounded proof does not unblock the
24-entry aggregate DirectX/OpenGL artifact, which remains fail-closed with
``project.translate.workgroup-size-entry-ambiguous``. Other axis sizes,
dtypes, and entries, MLX host redirection, and the full MLX test suite remain
outside the claim. Entry-scoped Metal also remains unavailable because the
per-entry workgroup specialization rule fails explicitly with
``project.translate.workgroup-size-rule-unsupported-target``; this proof does
not claim a Metal round trip.

The same current pin now has a bounded one-pass scaled-attention proof for
``sdpa_vector_float_64_64``. The checked host contract selects batch/head/query
counts of one, key length 4, dimensions ``D=64`` and ``V=64``, scale ``0.125``,
and no mask, causal mode, or sinks. It fixes ``[1024, 1, 1]`` with 32 logical
subgroups and one dispatched workgroup. Function constants 20 through 25 are
all false; two-pass-only ID 26 is not part of this artifact. The exact HLSL is
8,721 bytes with SHA-256
``003c8b9e85bad7363bae2e3d80380d979cbe0b8988d0d98751131c3acfbff6b6``
and passes DXC 1.9.2602.24 under ``cs_6_6``, ``-enable-16bit-types``, and
warnings as errors. Its 32 physical waves receive unique, wave-uniform IDs
through the synchronized allocator; no lane-varying flattened-index quotient
remains.

OpenGL uses an explicit 32-lane software artifact across the full 1,024-thread
workgroup. The subgroup-ID-strided runtime loop is synchronized round by round;
inactive subgroups supply typed reduction identities so all barriers remain
uniform. Its 12,089-byte GLSL has SHA-256
``9b7cb7dc9a76b9fb93c30fd93d13ad639f5493f60fd97b965514db0fe6b4840b``.
The validated SPIR-V has nine control barriers, six false specialization
constants, local size ``1024 1 1``, and no group-nonuniform instruction.

The native-loader ABIs contain 19 DirectX and 18 OpenGL resources. Deferred
optional resources receive placeholders, including uint32 physical storage for
``bmask``. DirectX concretizes the six constants and executes through WARP;
OpenGL builds a verified deferred compilation request, specializes the six
constants, and executes through Mesa surfaceless EGL. Both compare all 64
outputs with a stable CPU reference at ``2e-4`` absolute and relative
tolerance. The local Mesa proof measured maximum absolute error
``4.082320426146424e-08`` and maximum relative error
``4.2163276126605175e-06``. Masked, causal, sinks, two-pass and full-attention
paths, other dimensions and dtypes, MLX host redirection, selected Metal
round-trip validation, and the full MLX suite remain outside this bounded
claim. The historical 42-entry aggregate DirectX/OpenGL run remains fail-closed
because it does not consume this entry-scoped contract; it is not evidence that
a bounded attention runtime proof is absent.

At current pinned MLX commit
``846d176227a0ac13d2667e58d2bb68b322109ab0``, a bounded LayerNorm VJP proof
selects ``vjp_layer_normfloat32`` for one axis-32 row with function constant
``has_w=true``, workgroup and subgroup width 32, and eight reflected resources.
DirectX materializes the constant and executes the generated HLSL through
Direct3D 12 WARP. OpenGL retains constant ID 20 in GLSL, lowers the reductions
to an explicit software subgroup, derives a verified deferred-compilation
request, compiles to SPIR-V, specializes ``has_w=true``, and executes through
Mesa EGL. Both paths compare all 32 input-gradient and 32 per-row
weight-gradient values. The one-row boundary means the per-row weight-gradient
temporary is also the final host-reduced result. This proof excludes the
separate bias-gradient dispatch, multi-row weight reduction, other dtypes and
axis sizes, MLX host redirection, and the full MLX test suite.

The same current pin also has a bounded RMSNorm VJP proof for
``vjp_rmsfloat32`` with one axis-32 row, ``has_w=true``, one 32-thread
workgroup, and ten reflected resources. DirectX concretizes the function
constant and executes HLSL through WARP. OpenGL retains constant ID 20, lowers
four reductions through the explicit software subgroup, and admits the
kernel's runtime row loop only after proving its initializer and bound
workgroup-uniform. The deferred path compiles GLSL to SPIR-V, specializes the
constant, verifies the interface, and executes through Mesa EGL. Both targets
compare all 32 input-gradient and 32 one-group weight-gradient values. The
one-row/one-group boundary makes the group-local weight gradient equal to the
final host reduction; multi-row reduction, other dtypes and axes, MLX host
redirection, and the full MLX test suite remain outside this proof.

The current pin additionally has a bounded Softmax proof for
``block_softmax_float32``. Its checked dispatch contract applies the pinned
host formula ``32 * ceilDiv(ceilDiv(axisSize, 4), 32)`` to two rows of axis 32
and one row of axis 2049, producing workgroups ``[32, 1, 1]`` and
``[544, 1, 1]``. Both DirectX artifacts retain ``WaveSize(32)``. OpenGL keeps
the default guarded hardware-subgroup artifacts and separately packages
explicit 32-lane software-subgroup artifacts; the wide artifact partitions 544
invocations into 17 logical subgroups and uses typed inactive-lane identities
for masked maximum and sum reductions. Official DXC accepts both HLSL artifacts
under ``cs_6_6`` with ``-enable-16bit-types`` and warnings as errors. The
32-thread entry is provably one wave and retains the zero-ID quotient fast path;
the 544-thread entry allocates one uniform ID for each physical wave through a
workgroup-synchronized counter. Both software modules pass
``glslangValidator`` and ``spirv-val``, contain 11 control barriers, and contain
no group-nonuniform SPIR-V instruction.

Windows CI requires both workloads to execute through Direct3D 12 WARP, and
Linux CI requires both to execute through surfaceless Mesa EGL. Every output is
compared with a stable float32 CPU Softmax reference at ``5e-5`` absolute and
relative tolerance. This evidence covers only the two finite float32 block
workloads: axis sizes above 4096, half and bfloat16 entries, MLX host
redirection, and the full MLX test suite remain outside the claim. Selected
entry-scoped Metal generation also remains unavailable and fails explicitly
with ``project.translate.entry-point-target-unsupported``; this bounded proof
does not claim a Metal round trip.

Build a versioned native loader ABI descriptor and optional C declarations for
one ready load unit:

.. code-block:: bash

   python -m crosstl native-loader-abi \
     crosstl-runtime-package/runtime-loader-manifest.json \
     --load-unit copy:directx \
     --output copy.directx.abi.json \
     --declarations-output copy.directx.abi.h \
     --execution-output copy.directx.native-loader-execution.h \
     --target-adapter-output copy.directx.native-loader-adapter.hpp

``native-loader-abi`` selects exactly one ready load unit. ``--load-unit`` is
optional only when the input manifest contains one unit. The command emits a
schema-v1 ``crosstl-native-loader-abi-descriptor`` JSON document containing
the target entry point, artifact identity and hash, source identity and remap,
binding namespaces and coordinates, access modes, scalar layout,
specialization constants, and provenance. A blocked unit, incomplete host
interface, malformed artifact identity, ambiguous selection, or invalid
binding coordinate produces a structured diagnostic instead of declarations.

``--declarations-output`` writes deterministic C declarations for the same
descriptor. The header contains a guarded, versioned ABI type contract and
immutable unit and binding descriptor data. The declarations compile as C11
and C++17 and can be included for more than one load unit without redefining
the shared ABI types. They describe what downstream host integration must
load and bind; they do not execute the unit.

``--execution-output`` writes a deterministic, allocation-free C11/C++17
execution wrapper for the selected unit. The header includes the declaration
contract and adds request, adapter-callback, result, and structured-error
types plus a unit-specific ``*_execute`` function. Before consulting the
adapter, that function validates the ABI version, target, exact binding
identity and coordinates, access modes, specialization identities and types,
payload presence, and dispatch geometry.

After validation, the wrapper loads the artifact, applies specialization
values to that artifact, creates the pipeline, binds resources in descriptor
order, dispatches, synchronizes, and reads back writable resources. It then
releases resources in reverse binding order, destroys the pipeline, and
unloads the artifact on both success and failure paths. The wrapper uses
fixed-size stack storage derived from the descriptor and performs no heap
allocation.

The caller owns the execution request, binding and specialization payloads,
readback destinations, and adapter context for the duration of the call. The
adapter owns each native artifact, pipeline, and resource handle that its
callbacks return; the wrapper passes those handles to the corresponding
unload, destroy, and release callbacks. A nonzero callback return value is
preserved as ``adapter_status`` with the failing phase and binding or
specialization index. ``error`` records the primary execution failure, while
``cleanup_error`` records the first cleanup failure without replacing an
earlier primary failure. A cleanup failure becomes the primary error only
when all preceding execution phases succeeded.

``--target-adapter-output`` writes the deterministic C++17 reference adapter
for the selected target. Include the unit execution header before this adapter
header. The adapter fills the shared callback table rather than replacing the
unit-specific validation wrapper, so request validation, execution order,
structured errors, and cleanup behavior remain defined by the common ABI.
Direct3D 12 and OpenGL compute have reference adapters. A target without a
reference adapter produces a structured
``project.native-loader-target-adapter.target-unsupported`` diagnostic.

Build descriptors and declarations for every ready unit in one operation:

.. code-block:: bash

   python -m crosstl native-loader-abi-package \
     crosstl-runtime-package/runtime-loader-manifest.json \
     crosstl-native-loader-abi

``native-loader-abi-package`` validates every unit and generates every
available target adapter before writing output. It emits target-scoped
descriptor, declaration, and execution header files, one adapter header per
supported target, plus ``native-loader-abi-package.json`` and prints that
package manifest as JSON.
Each packaged unit records ``executionABIPath`` and ``executionABIHash``;
the generated-file inventory identifies execution headers as
``native-loader-execution-abi`` and reference adapters as
``native-loader-target-adapter``. The schema-v3 ``targetAdapters`` array records
the target, availability, generated path, and SHA-256 hash. Targets without a
reference implementation remain in that array with
``reason: target-adapter-unavailable`` rather than being silently omitted.

Schema-v3 packages also publish ``runtime/runtime-variant-registry.json`` and
record its file hash, registry identity, and exact variant count in
``runtimeVariantRegistry``. When that registry is ready, the package verifies
and copies every referenced translated artifact, recording each as
``runtime-target-artifact``; descriptor byte sizes and hashes remain available
for independent verification. A metadata-only loader manifest that lacks the
provenance required for exact lookup remains valid, but marks the registry and
native header unavailable instead of claiming dispatch readiness.

When every registry target has a reference adapter, the package also emits
``native-runtime-variant-registry.hpp``. This deterministic C++17 header maps
canonical runtime variant keys to unit execution wrappers and retains the exact
target, entry point, workgroup size, subgroup width, and specialization
payloads selected during packaging. Packages containing targets without a
reference adapter keep the JSON registry but mark the native header unavailable.
A ready OpenGL GLSL-source variant that still carries specialization constants
also keeps its JSON registry ready, copied artifact, and exact lookup key, but
marks the native header unavailable with reason
``specialization-requires-deferred-compilation``. The C++ header cannot express
that compile-then-specialize step; callers instead derive and execute the
verified deferred SPIR-V compilation request. Other registry-generation errors
remain fatal and are not converted into this fallback.

The same operations are available through the public project API:

.. code-block:: python

   from crosstl.project import (
       NATIVE_LOADER_ABI_PACKAGE_KIND,
       NATIVE_LOADER_ABI_PACKAGE_MANIFEST,
       NATIVE_LOADER_ABI_PACKAGE_VERSION,
       NATIVE_RUNTIME_VARIANT_REGISTRY_HEADER_PATH,
       NATIVE_RUNTIME_VARIANT_REGISTRY_PATH,
       NATIVE_LOADER_TARGET_ADAPTER_KIND,
       NATIVE_LOADER_TARGET_ADAPTER_VERSION,
       NativeRuntimeVariantRegistryError,
       build_native_loader_abi_descriptor,
       build_native_loader_abi_package,
       build_runtime_variant_dispatch_request,
       generate_native_loader_declarations,
       generate_native_loader_execution_abi,
       generate_native_loader_target_adapter,
       generate_native_runtime_variant_registry,
       native_loader_target_adapter_targets,
   )

   descriptor = build_native_loader_abi_descriptor(
       loader_manifest,
       load_unit_id="copy:directx",
   )
   declarations = generate_native_loader_declarations(descriptor)
   execution_abi = generate_native_loader_execution_abi(descriptor)
   target_adapter = generate_native_loader_target_adapter(
       descriptor["target"]
   )

   package = build_native_loader_abi_package(
       "crosstl-runtime-package/runtime-loader-manifest.json",
       "crosstl-native-loader-abi",
   )

``build_native_loader_abi_descriptor`` validates and normalizes one loader
unit before returning deterministic JSON-compatible metadata.
``generate_native_loader_declarations`` validates that descriptor before
rendering the C representation. ``generate_native_loader_execution_abi``
validates the same descriptor before rendering the executable callback
wrapper. ``generate_native_loader_target_adapter`` renders the target callback
implementation for a supported canonical target, while
``native_loader_target_adapter_targets`` reports the available target names.
``build_native_loader_abi_package`` validates all load units and adapter output
before writing, then emits one descriptor, declaration header, and execution
header per unit, one reference adapter per supported target, and a
deterministic package manifest. The manifest records content hashes, source
loader-manifest identity, generated paths, target, adapter, and unit counts,
and uses the exported
``NATIVE_LOADER_ABI_PACKAGE_KIND``, ``NATIVE_LOADER_ABI_PACKAGE_VERSION``, and
``NATIVE_LOADER_ABI_PACKAGE_MANIFEST`` constants for its kind, schema version,
and file name. A blocked, incomplete, or duplicate unit prevents package
publication rather than producing a partially described package.

The generated Direct3D adapter owns its device, queue, fence, pipelines, and
resource allocations within an explicit caller-created context. The generated
OpenGL adapter consumes a caller-owned current desktop context and an explicit
function table; it does not create a window-system context or choose EGL, GLX,
WGL, or another loader. Both adapters reject unsupported artifact, resource,
layout, and capability contracts through nonzero adapter statuses. They are
reference implementations for one unit execution lifecycle, not repository
schedulers or host-runtime rewrites. Host application rewriting, primitive
selection, graph policy, full MLX runtime integration, and MLX test-suite
parity are not provided or claimed.

The Direct3D 12 adapter accepts packaged HLSL compute source and DXIL
containers. HLSL source compilation uses the DXC API with shader model 6.2 and
native 16-bit types enabled. Hosts that compile HLSL source need the official
``dxcapi.h`` header; ``dxcompiler.dll`` and ``dxil.dll`` must be discoverable
at execution time. DXIL-only hosts do not need the DXC API, but cannot apply
source specializations. Generated HLSL source specialization requires both the
reflected constant name and numeric ID so the adapter can replace the exact
CrossTL fallback declaration before passing a numeric definition to DXC.
Structured-buffer SRV and UAV bindings and constant-buffer CBV bindings are
supported when the descriptor includes a compatible scalar layout. Other
resource kinds fail closed.

The OpenGL adapter accepts GLSL compute source with entry point ``main`` and
OpenGL SPIR-V compute binaries. GLSL source can run on a current OpenGL 4.3
desktop context but does not support specialization. SPIR-V specialization
requires OpenGL 4.6 or caller-confirmed ``GL_ARB_gl_spirv`` support and the
matching ``glShaderBinary`` and ``glSpecializeShader`` entry points. The
adapter supports set-zero shader-storage and uniform-buffer bindings; nonzero
sets and texture, image, sampler, scalar-uniform, and shared-allocation
contracts fail closed.

Native CI translates reduced source fixtures through CrossTL, verifies
packaged artifact hashes, binds distinct input and output buffers, applies one
specialization, dispatches on Direct3D 12 and surfaceless OpenGL, and compares
deterministic readback with the expected values. These checks prove the
generated adapter lifecycle for the reduced contracts. They do not establish
semantic parity for every translated kernel or execute an upstream
repository's complete test suite.

For a complete DirectX or OpenGL compute descriptor, the public project API can
construct and preflight the backend-neutral runtime request consumed by the
native parity adapters:

.. code-block:: python

   import json
   from pathlib import Path

   from crosstl.project import build_native_loader_dispatch_request

   package_root = Path("crosstl-native-loader-abi")
   descriptor = json.loads(
       (package_root / "descriptors/directx/copy.abi.json").read_text()
   )
   request = build_native_loader_dispatch_request(
       descriptor,
       package_root,
       input_values={
           "input_values": {
               "dtype": "float32",
               "shape": [4],
               "values": [1.0, 2.0, 3.0, 4.0],
           },
           # The descriptor reflects this binding as read_write.
           "output_values": {
               "dtype": "float32",
               "shape": [4],
               "values": [1.0, 2.0, 3.0, 4.0],
           },
       },
       output_values={
           "output_values": {
               "dtype": "float32",
               "shape": [4],
               "values": [2.0, 4.0, 6.0, 8.0],
           }
       },
       dispatch_geometry={"workgroupCount": [1, 1, 1]},
       specialization_values={3: 4},
       expected_target="directx",
   )

``build_native_loader_dispatch_request`` supports compute-stage Metal, DirectX
HLSL and OpenGL GLSL artifacts. It validates the descriptor, requires exact
reflected binding names, verifies the artifact size and SHA-256 digest inside
the package root, validates specialization values and dispatch geometry, and
returns a preflighted ``RuntimeExecutionRequest``. Buffer bindings require a
complete supported scalar, vector or storage-record layout; missing or ambiguous physical
layout metadata is a structured error rather than an inferred ABI.

The returned request also carries a frozen ``RuntimeArtifactIdentity`` copied
from the validated descriptor's byte size, SHA-256 digest, and unit ID. Native
execution uses this pinned record instead of rereading identity fields from the
request's public artifact mapping. Later mutation of that mapping therefore
cannot replace the expected identity used at execution. The record pins
identity metadata only; it does not freeze other artifact metadata or the file
at the artifact path.

Request construction copies input and expected-output buffer payloads before
validating them, including nested values and exact-bit storage encodings.
Subsequent changes to the caller's lists cannot alter an existing request or
its prepared resource bindings. Construct a new request to submit new values.
This ownership guarantee does not make the returned request's nested objects
immutable, enforce caller-trusted index-range assertions, or validate metadata
produced or modified by a preceding GPU dispatch.

Native adapter preparation additionally encodes each initialized buffer once
into an immutable upload snapshot. The built-in Metal, DirectX, OpenGL and
Vulkan drivers consume those exact bytes, not the request's mutable value lists.
Prepared binding reports retain the payload hash, byte size, dtype, shape and
encoding without embedding buffer contents. Later changes to a prepared
binding's shape, dtype or encoding are rejected if they conflict with its
snapshot. Uninitialized outputs and later dispatch-region graph nodes do not
acquire a spurious upload.

Snapshots pin bytes at native binding preparation, not at initial request
construction, and do not freeze GPU memory after upload. This is the data
foundation for checked host metadata, not an authorization to narrow indices:
source-resource identities, value preconditions and mutable allocation aliases
still require a separate checked contract. Custom runtime drivers must consume
the snapshot to provide the same upload guarantee. Construct a new request or
a fresh low-level binding when supplying different data.

Pass ``observe_inputs=("input_values", "axis_size")`` to
``build_native_loader_dispatch_request`` to verify selected read-only inputs
after execution. Names must identify uploaded, reflected read-only bindings;
they cannot also become writable outputs. The built-in Metal, DirectX and
OpenGL drivers read the same bound buffers, preserving their resource classes,
allocation identities, offsets and logical views. Storage buffers and
constant/uniform buffers are supported. Unsupported drivers fail explicitly
before dispatch rather than reporting an unobserved input as unchanged.

Observations compare exact device bytes against the immutable upload snapshot,
including encoded floating-point words and caller-supplied guard elements.
There is no numerical tolerance or NaN canonicalization. Results retain
``inputObservations`` separately from writable outputs, with observed bytes,
upload and readback hashes, binding provenance and allocation metadata.
Missing, malformed, truncated or changed observations fail execution.
Uniform alignment padding outside the validated input payload is not claimed
as observed caller data. This opt-in post-dispatch check does not establish
immutability during execution, authorize index-range assumptions or observe
every allocation automatically.

Flat mixed-scalar storage records retain each field's name, physical type, byte
offset, size and alignment, together with the padded record stride. This
contract applies to Metal pointer buffers, HLSL structured buffers and OpenGL
std430 storage arrays containing 32-bit floats or signed/unsigned 32/64-bit
integers. It does not use uniform-block packing or describe a mixed record as a
homogeneous vector.

``pack_storage_records(layout, records)`` packs named field mappings into
little-endian ``uint32`` transport words and zeroes padding. Pass the returned
words with ``dtype="uint32"`` and shape
``(len(records), layout["elementStrideBytes"] // 4)``. The reflected
``elementType="record"`` and ``payloadEncoding="uint32-le-words"`` distinguish
physical transport from shader field types. Explicit physical words can retain
particular floating-point bit patterns, including NaN payloads. Request
construction rejects incomplete records, contradictory layouts and invalid
allocation views; native preparation revalidates the record contract.
Nested records, member arrays, pointer fields, vectors and mixed narrow-scalar
records remain unsupported rather than receiving guessed layouts.

An exact binding name may appear in both ``input_values`` and ``output_values``
only when the descriptor reflects that resource as ``read_write``. In the
example above, the input payload initializes one native allocation and the
expected output contract marks that same allocation for readback and
comparison. Request construction requires the two roles to agree on dtype and
shape and validates both against the complete reflected physical scalar layout.
An overlap for another access mode, or an incompatible contract, produces a
structured diagnostic.

This API prepares one request and its native resource contract. Repository
scheduling, host application rewriting, and full MLX test-suite parity remain
outside its scope. Generated descriptors for the exact HLSL and GLSL scalar
forms described above carry the complete physical layout into request
construction; unsupported resource shapes remain rejected rather than
inferred. The pinned MLX ``arangeuint32`` proof establishes native DirectX and
OpenGL execution for that one contract, not full MLX runtime integration or
numerical parity across the MLX test suite.

Build a deterministic runtime variant registry from either a ready runtime
package or loader manifest:

.. code-block:: bash

   crosstl runtime-variant-registry \
     crosstl-runtime-package/runtime-loader-manifest.json \
     --output runtime-variant-registry.json

Runtime variant registries emit a schema-v1
``crosstl-runtime-variant-registry`` JSON document whose runtime variant key
schema is version 2. Each ``variants`` entry is indexed by a canonical
``crosstl-rvk2:`` key: URL-safe base64 without padding over canonical JSON
containing the source unit and source entry, target and target profile, the
selected binding-interface entry point's execution identity, type and value
template arguments, specialization constant IDs and values, and defines. The
``execution`` identity contains ``workgroupSize`` and ``subgroupWidth``; each
field remains ``null`` when the selected entry does not provide an exact value.
Unselected entry points and project-level aggregate metadata do not affect the
key. Key fields are sorted before encoding, registry records and target
summaries are ordered by key, and ``registryHash`` covers the key schema and
records. Equivalent input records therefore produce the same registry
regardless of package or loader record order.

Each registry record preserves source and target names separately and maps the
exact key to the target artifact path, format, hash and byte size, target entry
point, binding resources and ordinary constants, pipeline specialization
constants, and translation and source provenance. Inputs use closed package
and loader field sets. Malformed schemas fail before records are emitted;
duplicate keys and keys with conflicting artifacts or metadata are diagnosed
and rejected. Package inspection hash or size failures remain explicit
``stale`` records, and loader blockers remain ``blocked`` records. Both are
listed as available exact keys but have ``lookup.eligible`` set to false.

The public ``build_runtime_variant_registry`` API builds the document,
``encode_runtime_variant_key`` and ``decode_runtime_variant_key`` expose the
key contract, and ``lookup_runtime_variant`` performs exact lookup with the
available keys included in not-found diagnostics. Lookup validates the closed
registry schema, ``registryHash``, canonical key-to-record identity, and record
eligibility before returning a ready artifact. Modified or malformed registry
records fail as invalid rather than participating in selection. When the
non-execution identity matches but the requested execution identity does not,
the diagnostic reports ``requestedExecution`` and the exact
``availableExecutionAlternatives`` with their keys, status, workgroup size,
and subgroup width. There is no fallback to one of those alternatives. Legacy
``crosstl-rvk1:`` keys are rejected with guidance to regenerate both the key
and registry. This remains deterministic selection and packaging metadata;
target compilation, deferred compilation, host runtime dispatch, device
execution, and numerical parity are not established by the registry.

Exact native variant dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A schema-v3 native loader package binds the exact JSON registry to the
descriptors, execution wrappers, translated artifacts, and optional native
registry header by SHA-256 identity. Build a preflighted request from one
canonical key:

.. code-block:: python

   import json
   from pathlib import Path

   from crosstl.project import build_runtime_variant_dispatch_request

   package_root = Path("crosstl-native-loader-abi")
   package = json.loads(
       (package_root / "native-loader-abi-package.json").read_text()
   )
   registry = json.loads(
       (package_root / package["runtimeVariantRegistry"]["path"]).read_text()
   )
   key = registry["lookup"]["readyKeys"][0]

   request = build_runtime_variant_dispatch_request(
       registry,
       key,
       package_root,
       input_values,
       output_values,
       {"workgroupCount": [1, 1, 1]},
   )

``build_runtime_variant_dispatch_request`` performs exact lookup and rejects a
registry that does not belong to the package. It verifies the packaged registry
hash and, when a native header is published, the native header hash. It also
checks descriptor size and hash, source and target provenance, artifact
identity, entry point, workgroup size, specialization identity and value, and
translated artifact bytes before delegating to
``build_native_loader_dispatch_request``. Runtime callers provide resource
values and dispatch counts but cannot replace the selected workgroup size or
specialization values.

Exact variant selection extends the request's frozen artifact identity with the
selected variant name and canonical variant key. Native execution retains the
pinned descriptor size, hash, and artifact ID plus those variant fields even if
the request's public artifact mapping is later mutated.

At native runtime setup, the DirectX and OpenGL adapters read the selected
translated artifact once and compute its byte size and SHA-256 digest. When the
request records both values, a mismatch produces a structured setup diagnostic
with the expected and observed identities before target validation, compilation,
or runtime loading. Partial identity metadata and malformed fields fail closed.
An artifact without either recorded field is still captured, but the result is
reported as ``not-recorded`` rather than verified. Artifact read and snapshot
materialization failures also produce structured setup diagnostics. The
captured bytes are materialized in an adapter-owned temporary directory, and
subsequent source validation and target compilation use that snapshot instead
of reopening the original artifact path. Source-format loading reads the
snapshot, while compiled modules are produced from it. Replacing the original
file after capture therefore cannot change the bytes consumed by that
execution.

This snapshot isolates one execution from later changes to the original path;
it is not a filesystem security boundary. It does not prevent another process
with access to the temporary directory from changing the snapshot, attest
compiler output, or verify artifacts consumed by a caller-supplied runtime
outside these DirectX and OpenGL adapters.

The generated C++17 header exposes
``crosstl_native_runtime_variant_lookup``,
``crosstl_native_runtime_variant_make_request``, and
``crosstl_native_runtime_variant_execute``. Execution accepts only a pointer
returned by the generated registry, checks ABI and target identity, and compares
the exact specialization payloads before invoking the selected unit wrapper.
DirectX HLSL and OpenGL SPIR-V specializations are supported according to the
target adapter contracts; precompiled DXIL specialization and GLSL source
specialization remain fail-closed.

This bridge selects and prepares one native compute dispatch from an existing
ready variant. It does not synthesize a missing variant, perform deferred target
compilation, rewrite a repository's host runtime, schedule multiple kernels,
translate framework control flow, or establish full MLX test-suite parity.

Bounded deferred native compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Repositories that select from a finite set of source variants can compile one
fully resolved variant after packaging without accepting arbitrary source
generation or compiler arguments. This path is defined by the schema-v1
``crosstl-native-deferred-compilation-request`` contract and the public project
API:

.. code-block:: python

   from crosstl.project import (
       build_native_deferred_compilation_dispatch_request,
       build_native_deferred_compilation_request,
       compile_native_deferred_compilation_request,
       execute_native_deferred_compilation_request,
   )

   request = build_native_deferred_compilation_request(
       source,
       includes,
       target,
       variant,
       expected_loader_descriptor,
   )
   compilation = compile_native_deferred_compilation_request(
       request,
       package_root,
       cache_root,
   )
   dispatch = build_native_deferred_compilation_dispatch_request(
       compilation,
       input_values,
       output_values,
       {"workgroupCount": [1, 1, 1]},
   )

An exact ready runtime variant can form the same closed request directly from a
schema-v3 native loader package:

.. code-block:: python

   from crosstl.project import (
       build_runtime_variant_deferred_compilation_request,
       execute_native_deferred_compilation_request,
   )

   request = build_runtime_variant_deferred_compilation_request(
       registry,
       key,
       package_root,
   )
   result = execute_native_deferred_compilation_request(
       request,
       package_root,
       cache_root,
       input_values,
       output_values,
       {"globalSize": [count, 1, 1]},
   )

``build_runtime_variant_deferred_compilation_request`` performs the same exact
lookup and package-to-registry identity checks as native variant dispatch. It
then verifies the selected source artifact bytes, loader descriptor size and
digest, source and target identity, entry point, execution configuration, and
specialization interface before deriving the request. The canonical variant
key, type and value arguments, compile definitions, specialization values, and
execution identity are copied from the selected registry record rather than
accepted from the runtime caller.

This bridge accepts DirectX HLSL source and OpenGL GLSL source records. Binary
DXIL or SPIR-V records remain ahead-of-time artifacts and cannot be treated as
compiler source. Native CI selects the exact record, compiles it with DXC or
``glslangValidator``, and dispatches the resulting binary on Direct3D 12 or a
software OpenGL device. The current package bridge emits an empty include list,
so the selected target source must be self-contained. A source artifact with
literal includes fails during verified materialization until the native loader
package carries a complete include closure.

``source`` and every ``includes`` record carry a portable package path, source
format, byte size, and SHA-256 digest. ``target`` fixes the backend, profile,
compute entry point, and binary output format. ``variant`` carries the canonical
runtime variant key, finite type and value arguments, compile definitions,
specialization values, workgroup size, and optional subgroup width. The request
also pins the expected native loader ABI descriptor by path, size, and digest.
A canonical ``requestHash`` covers the complete closed-schema document.
Unknown fields, arbitrary compiler flags, unresolved or non-finite values,
source/target format mismatches, and inconsistent variant keys fail contract
validation.

Before a compiler is consulted,
``materialize_native_deferred_compilation_inputs`` verifies the source, every
include, and the loader descriptor against their recorded size and digest. It
rejects symbolic links, path escapes, portable case collisions, non-regular
files, undeclared or unreachable includes, ambiguous angle includes, and
dynamic include operands. Only a complete literal include closure is copied
into an isolated source tree; the compiler receives include directories derived
from that verified closure.

Compilation then validates the loader descriptor against the request, including
target, stage, entry point, source identity, execution configuration, and exact
specialization identity and value rules. It separately reflects the source and
compares entry-point execution, bindings, scalar layout, constants, and the
OpenGL specialization interface with the descriptor. Any drift fails before
target compilation or dispatch. Source reflection currently evaluates the
primary translation unit. Include files remain byte-verified compiler inputs,
but declarations visible only after preprocessing an include are not folded
into host-interface reflection in this path and therefore cannot satisfy the
interface check.

DirectX requests compile packaged HLSL compute source to DXIL with ``dxc``.
OpenGL requests compile packaged GLSL compute source to SPIR-V with
``glslangValidator``. Compiler commands are derived from the validated request;
callers cannot append arbitrary flags. The schema-v1 compilation result records
the tool name, reported version, executable hash, probe and compile commands,
source and include provenance, compiler diagnostics, and the compiled output's
format, size, and SHA-256 digest.

Successful outputs use a deterministic cache key derived from the complete
request hash and the toolchain name, version, and executable hash. Cache entries
retain the expected interface identity and output identity, are published
atomically, and are revalidated on lookup. The toolchain executable is also
rechecked before a cached or newly compiled result is returned. Compiler
failures, missing or malformed output, interface drift, and partial cache
entries are never published as successful cache entries.

``build_native_deferred_compilation_dispatch_request`` converts a successful
compiled result into the same preflighted ``RuntimeExecutionRequest`` used by
the native loader path. The result retains the normalized source request;
dispatch revalidates its hash and rejects target, specialization, or descriptor
provenance drift. It preserves exact binding, specialization, execution, and
compiled-artifact identity. ``execute_native_deferred_compilation_request``
performs compilation or exact cache reuse and then dispatches through the
DirectX or OpenGL native runtime adapter. Callers still supply resource values
and dispatch counts. An injected OpenGL context remains caller-owned when
``OpenGLComputeRuntime`` is constructed with ``release_context=False``.

This is a bounded compute compilation and dispatch contract. It does not
generate arbitrary source, rewrite host applications or runtime frameworks,
choose repository scheduling policy, or provide framework-specific runtime
integration. It does not claim complete MLX kernel coverage, MLX runtime
integration, or parity with the MLX test suite. The implemented scope is
tracked in `GitHub issue #1854
<https://github.com/CrossGL/crosstl/issues/1854>`_.

Build deterministic host loader scaffold metadata from a runtime loader
manifest:

.. code-block:: bash

   python -m crosstl scaffold-host-loaders \
     crosstl-runtime-package/runtime-loader-manifest.json \
     --scaffold-dir crosstl-host-loaders \
     --format text

Host loader scaffolds emit a ``crosstl-runtime-host-loader-scaffolds`` JSON
document and write a small metadata bundle with ``host-loader-scaffolds.json``,
``HOST_LOADERS.md``, and one target-scoped ``*.loader.json`` file for each
ready load unit. The scaffold files preserve the loader manifest's artifact
paths, source-remap handoff paths, host interface metadata, required tools,
host responsibilities, and ordered load steps so host loader or build-system
tooling can consume the package contract deterministically. Load units with
unresolved blockers, such as missing host interface reflection metadata, remain
listed in the scaffold report and guide but do not get target loader files.
The bundle is metadata for integration tooling only: it does not rewrite host
application code, execute device code, generate runtime framework code, or
install target SDKs.

Inspect host loader scaffold files before runtime tooling consumes them:

.. code-block:: bash

   python -m crosstl inspect-host-loader-scaffolds \
     crosstl-host-loaders/host-loader-scaffolds.json \
     --format text

Host loader scaffold inspections emit a
``crosstl-runtime-host-loader-scaffolds-inspection`` JSON document that verifies
the scaffold manifest, integration guide, and target-scoped loader metadata
files are present, readable, and consistent with the scaffold manifest. Ready
loader metadata files are parsed and checked for matching scaffold identity,
target, adapter kind, package path, and status. Blocked load units remain
explicitly blocked without requiring target loader files. Missing, malformed,
or mismatched scaffold files are reported as structured diagnostics before host
loader or build-system tooling consumes the metadata. Inspection remains
read-only and does not rewrite host application code, execute device code,
generate runtime framework code, or install target SDKs.

Build a read-only host loader consumption plan from scaffold metadata:

.. code-block:: bash

   python -m crosstl plan-host-loader-consumption \
     crosstl-host-loaders/host-loader-scaffolds.json \
     --format text

Host loader consumption plans emit a
``crosstl-runtime-host-loader-consumption-plan`` JSON document. Planning runs
scaffold inspection first, reads only ready target-scoped host loader unit JSON
files, carries required tools and host responsibilities forward, and promotes
ordered ``loadSteps`` into actionable records for host build or runtime
integration tooling. Blocked scaffold records remain actionable
``resolve-loader-scaffold-blockers`` entries, and failed scaffold inspection
diagnostics are reported without reading unsafe unit files. The plan remains
metadata only: it does not rewrite host application code, execute device code,
generate runtime framework code, or install target SDKs.

Write a deterministic host integration handoff bundle from a consumption plan:

.. code-block:: bash

   python -m crosstl host-integration-handoff \
     crosstl-host-loaders/host-loader-consumption-plan.json \
     --handoff-dir crosstl-host-integration \
     --format text

Host integration handoff bundles emit a
``crosstl-runtime-host-integration-handoff`` JSON report and write
``host-integration.json``, ``HOST_INTEGRATION.md``, and one
``targets/*.integration.json`` file per target. The bundle is designed as a
stable handoff for build-system and runtime integration tools: it preserves the
validated loader units, promoted actions, required tools, host responsibilities,
package paths, scaffold files, and blocked-unit records from the consumption
plan. It remains a metadata bundle only and does not rewrite host application
code, execute device code, generate runtime framework code, or install target
SDKs.

Inspect host integration handoff files before downstream tooling consumes them:

.. code-block:: bash

   python -m crosstl inspect-host-integration-handoff \
     crosstl-host-integration/host-integration.json \
     --format text

Host integration handoff inspections emit a
``crosstl-runtime-host-integration-handoff-inspection`` JSON document that
verifies the handoff manifest, guide, and per-target
``targets/*.integration.json`` files are present, readable, and consistent with
the handoff manifest. Target files are parsed for matching kind, target, status,
loader-unit counts, and action counts. Missing, malformed, wrong-kind, or
mismatched handoff files are reported as structured diagnostics. Inspection is
read-only and remains bundle-local: it does not rewrite host application code,
execute device code, generate runtime framework code, install target SDKs, or
re-run host integration.

Build a read-only host integration execution plan from an inspected handoff:

.. code-block:: bash

   python -m crosstl plan-host-integration-execution \
     crosstl-host-integration/host-integration.json \
     --host-root . \
     --format text

Host integration execution plans emit a
``crosstl-runtime-host-integration-execution-plan`` JSON document. Planning
runs handoff inspection first, records optional host-root readiness, and
normalizes per-target handoff actions into stable phase-ordered execution
steps. The step phases cover tool preparation, loader consumption, artifact
loading, host responsibility handling, blocker resolution, and other host
actions. Plans carry required tools, host responsibilities, package paths,
scaffold files, target status, and structured diagnostics so downstream host
or build-system tooling can decide what to run next. Plans also include a
``deviceExecution`` readiness block that declares the adapter-backed dispatch
inputs a target will need, including the runtime package root, runtime adapter
descriptor root, and an external target runtime runner. The plan remains
metadata only: it does not rewrite host application code, execute device code,
generate runtime framework code, or install target SDKs.

Execute deterministic host integration checks from a saved execution plan:

.. code-block:: bash

   python -m crosstl execute-host-integration \
     crosstl-host-integration-execution-plan.json \
     --host-root . \
     --scaffold-root crosstl-host-loader-scaffolds \
     --package-root crosstl-runtime-package \
     --adapter-root crosstl-runtime-package/runtime-adapters \
     --format text

Host integration execution results emit a
``crosstl-runtime-host-integration-execution-result`` JSON document. Execution
validates the plan, revalidates the planned host root or an explicit
``--host-root`` override, records scaffold-root and package-root readiness,
verifies generated loader scaffold files when ``--scaffold-root`` is provided,
checks package artifact and source-remap paths when ``--package-root`` is
provided, verifies runtime adapter descriptor manifests and descriptor files
when ``--adapter-root`` is provided, checks required host tools on ``PATH``, and
reports project-specific host responsibilities as skipped actionable steps.
Blocked plan steps remain blocked in the result, and missing files, stale
descriptor hashes, or invalid paths are emitted as structured diagnostics.
The result ``deviceExecution`` block reports whether each target has a ready
runtime package and verified runtime adapter descriptor for an external runner.
Pass ``--runner-manifest`` with a
``crosstl-runtime-device-runner-manifest`` JSON file to record target runner
readiness alongside package and adapter readiness. Runner manifests list
target-specific runner ids, statuses, optional capabilities, and optional
commands; they are validated as readiness metadata only. This command still
does not rewrite host application code, dispatch device work, generate runtime
framework code, or install target SDKs.

Diagnostics with ``originalLocation`` keep the generated or validation
location as the primary SARIF location and attach the original source span as a
related location. Remapped diagnostics also expose sanitized
``diagnosticLocation`` and ``originalLocation`` SARIF properties for tools that
filter or group results without walking related locations.
Inspection exits nonzero when validation finds report errors.

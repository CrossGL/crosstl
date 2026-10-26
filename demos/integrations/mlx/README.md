# MLX Integration Demo

MLX is an external test corpus for CrossTL's repository translation and runtime
APIs. Its host adaptations, pinned contracts, execution evidence checks and tests
belong to this demo; the translator does not depend on MLX.

## Directory Guide

| Path | Purpose |
| --- | --- |
| `project.toml`, `run_porting.py` | Repository scan and translation harness |
| `contracts/` | Pinned discovery, artifact and dispatch contracts |
| `portable_host/` | DirectX/OpenGL host adapter and bounded execution checks |
| `run_metal_host.py`, [METAL_HOST.md](METAL_HOST.md) | Generated Metal host integration |
| `run_native_metal.py`, [NATIVE_METAL.md](NATIVE_METAL.md) | Unchanged upstream Metal reference |
| `tests/` | Harness, contract, discovery and CI tests |
| `tests/host/` | Portable host adapter tests |
| `tests/kernels/` | Pinned kernel translation and native execution tests |
| `tests/fixtures/` | MLX-specific reference programs |
| `fixtures/project_porting/` | Reduced source-specialization contracts |
| `fixtures/runtime_verification/` | Runtime inputs, expected values and artifact selectors |

Generic project and runtime tests also consume these reduced fixtures. Keeping
the inputs here preserves their MLX provenance without making the core fixture
tree project-specific. Moving a fixture does not change its shader or expected
numerical results.

Run the demo's local tests from the repository root:

```sh
python -m pytest -q -n auto demos/integrations/mlx/tests
```

Native execution remains opt-in locally and mandatory in its corresponding CI
job. The [Project Demo Testing workflow](../../../.github/workflows/demo-project-testing.yml)
contains the Windows DirectX, Linux OpenGL and macOS Metal jobs, including the
reference baseline and host integration checks. Each job retains its own pinned
source revision, required toolchains, numerical assertions and evidence uploads.
Complete DirectX corpus compilation uses pinned DXC on Ubuntu; Direct3D runtime
execution remains on Windows. Metal compilation and execution remain on macOS.
See the [CI coverage policy](../../../.github/TESTING.md) for the platform split.
Full native, parity, corpus and host jobs run nightly, on manual dispatch, and
for main-branch pushes and merge-queue groups that change
`demos/integrations/mlx/**` or this demo's workflow. Every merge-queue group
runs the smoke job. A pull request runs them only when it carries the `ci:full` label,
because the full tier is several hundred jobs on runners every open pull
request shares. Run `gh workflow run demo-project-testing.yml --ref <branch>`
for a single full run of a branch. Pull requests that change the MLX demo,
core, tests or toolchains run a short Linux smoke job that translates pinned integer
and float32 `arange` entries to all three targets, checks their artifact hashes,
validates HLSL/GLSL and executes their numerical cases with Mesa EGL, including
input observations and output guards. The selector
compares the complete Git diff, including deleted and renamed paths.
Each job and matrix leg has its own queue.
Active project-demo jobs finish even when a newer revision arrives. The pending
job in each concurrency group can be replaced by a newer pending job; these
groups are not first-in, first-out queues. No test matrix or numerical gate is
removed. A previous revision's successful proof does not satisfy the current
revision's required checks.
Core and demo tests also remain part of the complete test suite.

The longest current-tree kernel families run in the `mlx-current-native-parity`
job rather than the MLX Metal project porting job: ternary and `binary_two`
(group `elementwise`), scaled dot-product attention and its two-pass first pass
(`attention`), rope and gated delta (`sequence`), the gated delta backward
(`backward`), sort (`ordering`), and the unfold, depthwise and Winograd
convolutions (`convolution`). The job has one matrix leg per runner and group.
GitHub expands at most 256 legs from one matrix, so the quantized groups run in
`mlx-current-quantized-parity` and the GEMM and steel convolution groups in
`mlx-current-gemm-parity`. Both reuse the first job's steps through a YAML
alias, and a workflow test keeps every matrix within the limit. Each leg
repeats the porting job's toolchain setup and pinned kernel checkout, runs only
its group's bounded steps, and uploads its results as
`mlx-current-parity-<os>-<group>`. A workflow test checks that each group's
step deadlines, plus 30 minutes for setup, fit within the leg's timeout.

Independent native regression batches, the portable host build, package
translation and upstream test execution use explicit prerequisites, so an
earlier numerical failure does not hide unrelated results. General batches and
source preparation require successful host contract or ABI checks.
Windows additionally rechecks the selected WARP library's pinned digest after
any runtime comparison; a missing or changed library prevents the subsequent
batches and source preparation. Pinned kernel batches, the build and translation
require successful source preparation, and native upstream tests require both
build and translation products. Cancellation stops this work. Earlier
failures remain fatal to the job, all numerical checks and deadlines are
unchanged, and no extra runner is added.

At translator revision `f1eda50bcc7242e71576f2f9a4cec0e9167809c5`, the
[Windows host artifact](https://github.com/CrossGL/crosstl/actions/runs/38223366797/artifacts/11736927297)
records 34 selected upstream tests passing on both CPU and translated DirectX
paths, with no skips, failures or errors. Its base trace contains 1,210 native
dispatches across all 93 configured entries, plus 18 checked rejection cases.
Four boundary batches cover 112 cases per path and another 194 native dispatches.
Independent checks verify artifact identities, retained boundary modules,
before/after source preparation, exact result hashes and input preservation.
The same job still fails the separate precise multiply-add controls. This
selected host proof is not a complete upstream-suite or backend qualification.

Complete DirectX scan tests select the translator's `native-precise` binary32
addition policy. The generated helpers materialize a precise result to preserve
each addition and subtraction through optimization. The pinned upstream source,
original-Metal reference words, buffer layouts and guard checks are unchanged.
Reports and packages retain the arithmetic policy. This is a target translation
configuration, not an upstream source patch or a relaxed comparison.
The mode does not select software IEEE emulation or resolve the separate WARP
multiply-add defect in #2193.

At translator revision `6305687879bb6f707c5b07ea002abfcc374fec9a`, native
Windows and macOS evidence covers all 121 configured sum, extrema and complex
scan cases per target: 149,328 computed scalar words and 4,544 guard words.
Independent receipt checks verify the original-Metal reference words, inputs,
launch dimensions, generated source identity and validated/executed module
hashes. The 21 complex cases at widths 64, 128 and 256 resolve the reproduced
cancellation failures in [#2215](https://github.com/CrossGL/crosstl/issues/2215).
[Windows complex-scan evidence](https://github.com/CrossGL/crosstl/actions/runs/38198627802/artifacts/11724483886)
and [macOS original/generated evidence](https://github.com/CrossGL/crosstl/actions/runs/38198627802/artifacts/11724523773)
retain every numerical case and guard. macOS additionally checks unchanged
read-only inputs after original-source execution. These results do not establish
whole-family coverage or a complete MLX backend.

At translator revision `bbb815b4ddf468717291039747ac873ba6864696`, all three
platforms additionally retain generated read-only input observations for every
one of those 121 scan cases: 242 input/launch-metadata observations, 149,328
computed words and 4,544 input and output guard words per target. Independent
artifact checks reconstruct the uploaded bytes and verify source, module,
binding and allocation identities. The [Windows native host artifact](https://github.com/CrossGL/crosstl/actions/runs/38213774655/artifacts/11731281467)
also verifies generic scalar, metadata-record and 256-byte-offset input views.
The [three-platform scan artifacts](https://github.com/CrossGL/crosstl/actions/runs/38213774655)
retain the complete numerical cases; the wider Windows host job still fails
the distinct arithmetic controls below. Observing unchanged inputs after
dispatch does not prove immutability during execution or complete host routing.

The project-porting binary32 step selects every native execution and Metal
linking case without repeating portable parser and compiler checks from the
main suite. It limits test workers and software-OpenGL threads to two while
retaining the 120-second deadline, all numerical inputs and all guard checks.
A workflow regression checks that every device test remains selected.
The two generated arithmetic-ownership profiles run first as separate test
work units on the existing two workers. This avoids queueing both expensive
controls behind one worker; per-case progress and durations are retained.
The selected cases, inputs, compiler options and 120-second bound are unchanged.
The larger collective and resource batches also cap software-OpenGL threads
at two. Their complete module selections are unchanged. Each batch has a
separate 600-second deadline: on macOS the two batches took 257-375 s and
282-358 s, so the earlier 360-second bound stopped complete passing runs.
Per-test progress and durations make timeout diagnosis visible.

If the required Windows collective checks fail, the existing job compares the
same precise scalar and cooperative-matrix controls with default pipeline flags
and WARP's `TOOL_DEBUG` flag. Each pair shares one compiled shader, identical
request bytes, inputs and exact expected words. Both modes execute even if the
first fails. The comparison retains requested and observed flags and fails on
either numerical failure; it is diagnostic, not a replacement for the required
gate. Its process limit is 180 seconds within the existing eight-minute step.
Both diagnostic modes request the Direct3D validation layer and retain its
bounded messages, including messages from failed pipeline creation. At
`afa595de`, all five debug-flag attempts report Direct3D error 922: the flag is
deprecated and rejected. They fail at `CreateComputePipelineState` with
`0x80070057`; none executes a debug-mode shader. Four default-mode cases differ
numerically and the ordinary matrix control passes. Default dispatches report
only initial-buffer-state warnings, not validation errors. The
[retained Windows artifact](https://github.com/CrossGL/crosstl/actions/runs/38219130449/artifacts/11734215973)
contains complete messages for all ten attempts. Identical source, module,
request, input, reference and default-readback identities bind this result to
the preceding comparison. A clean validation queue does not establish correct
arithmetic, and this flag is not a usable workaround on the captured runtime.

The same bounded diagnostic also compares unannotated Shader Model 6.6 controls
with explicit wave sizes 4, 8, 16, 32, 64 and 128. Each variant preserves the
scalar or matrix arithmetic, dispatch dimensions, exact inputs and reference
words, offset buffer views and trailing guards. Separate lane-count witnesses
check that execution honors each requested width. Compiler listings retain the
wave-size annotation and verify separate, non-fast scalar multiply/add operations.
The comparison explicitly selects WARP and retains runtime identities and
failure receipts; it does not select a production wave width or replace the
required unannotated controls. WARP's documented
[width-dependent code generation](https://www.nuget.org/packages/Microsoft.Direct3D.WARP/1.0.16.1)
motivates this comparison, but compilation alone does not establish a numerical
fix. Unsupported widths must retain a pipeline failure, not a passing numerical
result. No additional runner or longer diagnostic deadline is required.
The completed comparison step at `d9623102` reports scalar numerical failures
for all seven settings, including the unannotated control. The
[Windows job log](https://github.com/CrossGL/crosstl/actions/runs/38230653906/job/114749015598)
therefore rules out these annotations as a general workaround on that runner.
An independent audit of all 27 retained controls also confirms that the seven
ordinary matrix variants pass all 768 output words, while each rounding-sensitive
matrix variant differs in the same 321 words. All six lane-count witnesses return
their requested width. Shader and module hashes, original inputs, exact
references, allocation offsets, guards and the loaded WARP 1.0.21 identity were
checked against the [retained artifact](https://github.com/CrossGL/crosstl/actions/runs/38230653906/artifacts/11742207193).
Wave-width selection works on this runner but does not repair precise arithmetic.

For manually dispatched workflow runs, that diagnostic instead compares the
same precise scalar, cooperative-matrix and narrow-buffer controls on the
system WARP library and checksum-pinned stable releases: 1.0.13, 1.0.14.1,
1.0.14.2, 1.0.15, 1.0.15.1, 1.0.16, 1.0.16.1, 1.0.17, 1.0.18 and 1.0.19.
The downloads and comparisons share the existing 360-second deadline, with
the eight-minute step limit leaving time for restoration. They retain library
hashes, file versions, test results
and runtime evidence, and restore the required WARP 1.0.21 library even after
a test failure. This focused diagnostic replaces the duplicate broad rerun;
the complete required and isolated qualification jobs are unchanged. No
comparison qualifies a fallback or replaces the required numerical gate.
The arithmetic and logical-buffer-tail defects remain tracked in #2193 and
#2197 respectively.

The pinned unary arithmetic batch retains its 300-second process deadline,
two workers, per-test progress, duration report and native execution requirements.
Its timed Python stack dump is disabled: an isolated x86-64 Python 3.12.15
translation reproduced a segmentation fault while writing that dump, without
loading a graphics runtime. The same translation completed with the timer off.
This isolates the diagnostic path, not a shader correctness failure, and does
not replace the numerical checks. A related
[CPython watchdog race](https://github.com/python/cpython/issues/158200) is under
investigation upstream; the reproduction does not establish that it is the
same defect.

### Full-Entry GEMM Execution

`tests/kernels/test_current_gemm.py` translates the unchanged
`steel_gemm_fused_nn_float32_float32_bm32_bn32_bk16_wm2_wn2` entry at commit
`9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8` through the project API, package
verification and native Metal, DirectX or OpenGL loader. All three targets
compare generated execution with an exact matrix-product reference. Metal
additionally compares with the independently compiled upstream kernel; Windows
and Linux do not claim to run the original Metal control.
Fifteen cases cover scalar, partial-tile, aligned-tile and multi-tile matrices,
contiguous and two-dimensional broadcast batches, padded rows and batch strides,
and extra workgroups. Output padding and eight trailing guard values must remain
unchanged.

The six Boolean function constants are explicitly frozen for each of six
configurations: plain multiplication, addition of an output source, scaled
fused output, broadcast batches, broadcast batches with scaled fused output,
and fully aligned dimensions. The output-source cases use noncontiguous columns
and padded rows; the scaled cases use alpha 0.5 and beta -0.25. Broadcast inputs
use different batch axes for A and B, and a shared C for the fused variant.
The original Metal kernel receives matching native function constants.
The test uses the source-backed fragment mapping documented in
`contracts/cooperative-matrix-fragment-mapping.json`, a 32-by-2-by-2 workgroup,
and reflected parameter layouts. It does not edit upstream or generated kernels,
or substitute a reference shader. DirectX and OpenGL use two source-backed
workgroup access assertions for the complete vector loads in
`steel/gemm/loader.h`.
For this entry, 128 threads load four adjacent floats each into A's 32-by-20
and B's 16-by-36 shared arrays. The test enumerates every destination, proves
unique coverage of the live elements, and bounds the addresses to 0..635 and
0..571 respectively. This header, `steel/gemm/gemm.h` and `steel/gemm/mma.h`
are hash-checked against the pin before execution. DirectX takes no
index-range assertions. GLSL indexes with 32-bit integers, so OpenGL takes
eleven: the four device reads of each loader vector and the float and
`int64_t` device helpers stay below 2^31 elements, as MLX buffers do, and the
A and B shared-array helpers stay in the same 0..635 and 0..571. A lane-level
model of `mma.h`'s tile reads must reach exactly the slots the loaders write.
OpenGL lowers the simdgroup matrix products to 32-lane software subgroups, as
DirectX does.
The original source is compiled with Metal 3.2 and the C++17/C++20 extension
warning exceptions from upstream's kernel build. Both original and generated
compilation disable fast math and treat remaining warnings as errors.

The macOS, Windows and Linux project-porting jobs run this required check and retain
translation reports, packages, compiler logs, input records, compiled modules,
readbacks and per-case parity results. Windows requires DXC compilation with
warnings as errors and Direct3D 12 execution; Linux requires glslangValidator
and spirv-val and Mesa EGL execution. Quarter-integer inputs make these
products and sums exactly representable; this gate does not replace the separate
rounding-sensitive cooperative-matrix controls tracked by #2193.
Run it locally against a clean checkout
of the pinned kernel tree:

```sh
mkdir -p gemm-results
CROSTL_MLX_CURRENT_ROOT=/path/to/mlx \
CROSTL_REQUIRE_MLX_CURRENT_GEMM=1 \
python -m pytest -q -n auto \
  --basetemp=gemm-results/pytest --junitxml=gemm-results/results.xml \
  demos/integrations/mlx/tests/kernels/test_current_gemm.py
```

This proves one float32 NN entry and six function-constant configurations. The
family gate below extends it to the other real-typed entries; neither covers
the upstream MLX suite or every mixed alignment combination. OpenGL's
device offsets are asserted, not checked against runtime metadata, which #2179
tracks. This check does not establish full host-runtime
redirection for any backend. A configured gate is not proof of success until
its native job passes for the relevant revision.

#### GEMM family

`test_current_gemm_family.py` executes all 72 float32, float16 and bfloat16
entries of `steel_gemm_fused.metal`: six tile shapes and the four NN, NT, TN
and TT transpositions per type. `gemm_fused_model.py` follows `gemm.h`,
`loader.h` and `mma.h` to name every entry's two `BlockLoader` instantiations,
the slots they write, and the slots `BlockMMA` reads; a non-native test
requires the reads to equal the writes for all 96 entries. Each entry's
DirectX and OpenGL assertions come from that model, as the representative's
do, and its workgroup is `(32, WN, WM)`.

Each entry runs three frozen configurations: unaligned (shapes 4x5x7,
33x35x17 with two padded batches and a spare threadgroup per axis, and
70x69x41, which spans several tiles in every dimension), aligned (one tile,
then 64x64x32), and broadcast batches with `alpha * AB + beta * C`. Transposed
operands store the same logical matrices, so all four layouts share one
reference. float32 and float16 use quarter-integer operands; bfloat16 uses
halves so every product, sum and epilogue keeps within its 8 significant bits.
The tests check that every reference value is exact in the output type, so
accumulation order and the final conversion round nothing; they do not test
rounding. Outputs are compared word for word, guards included, and on macOS
with the original source as well. The current-tree native parity job runs the
216 entry and configuration pairs in five shards per operating system:

```sh
CROSTL_MLX_CURRENT_ROOT=/path/to/mlx \
CROSTL_REQUIRE_MLX_CURRENT_GEMM_FAMILY=1 \
CROSTL_MLX_CURRENT_GEMM_FAMILY_SHARD_INDEX=0 \
CROSTL_MLX_CURRENT_GEMM_FAMILY_SHARD_COUNT=5 \
python -m pytest -q -n auto demos/integrations/mlx/tests/kernels/test_current_gemm_family.py
```

The 24 complex64 entries are not covered yet. Their `BlockMMA` partial
specialization now matches through the `complex64_t` alias, but its local tile
types and `reinterpret_cast` float views stop translation with a typed
diagnostic; #2261 tracks the partial-specialization materialization.

#### GEMM split-K

`test_current_gemm_splitk.py` executes all 192 float32, float16 and bfloat16
`gemm_splitk` entries of `steel_gemm_splitk.metal` (four tile shapes, four
transpositions and four `MN_aligned`/`K_aligned` pairs per type) and the six
real-typed `gemm_splitk_accum` and `_axpby` kernels. `gemm_splitk_model.py`
names every entry and reproduces the partition count, iteration count and
partition size that `matmul.cpp` dispatches. Entries unaligned in both run
4x5x7, 33x35x70 with padded leading dimensions and a spare threadgroup in each
tile axis, and 17x40x100, which splits into four partitions with a `K` tail in
the last one. `K`-aligned entries use multiples of 16 for `K`, and
`MN`-aligned entries scale `M` and `N` to whole tiles. Every partition word of
`C_split`, guards included, is compared with an exact reference, and on macOS
with the original source too. The accumulation kernels sum two to four
partitions, with `alpha * sum + beta * C` over a strided addend for `_axpby`.

Two struct specializations whose spelled arguments differed only in
punctuation, such as `!flag` and `flag`, used to receive one name, so the
second one reused the first one's members. The Metal frontend now gives them
distinct names, and `tests/test_backend/test_metal/test_struct_specialization_names.py`
covers the case independently of MLX. The current-tree native parity job runs
the entries in seven shards per operating system; shard 0 also runs the
accumulation kernels:

```sh
CROSTL_MLX_CURRENT_ROOT=/path/to/mlx \
CROSTL_REQUIRE_MLX_CURRENT_GEMM_SPLITK=1 \
CROSTL_MLX_CURRENT_GEMM_SPLITK_SHARD_INDEX=0 \
CROSTL_MLX_CURRENT_GEMM_SPLITK_SHARD_COUNT=7 \
python -m pytest -q -n auto demos/integrations/mlx/tests/kernels/test_current_gemm_splitk.py
```

The 64 complex64 split-K entries and two complex64 accumulation kernels are
not covered; they need the same `BlockMMA` partial specialization as the
fused complex64 entries (#2261).

#### Gathered GEMM

`test_current_gemm_gather.py` executes all 60 float32, float16 and bfloat16
`gather_mm` entries of `steel_gemm_gather.metal` (five tile shapes and four
transpositions per type), each with three frozen function-constant
configurations: unaligned without an output batch, unaligned with
`has_batch`, and fully aligned. `gemm_gather_model.py` names every entry,
`gather_mm_rhs` included, and the test checks that list against entry
discovery. The dispatch binds the buffers `matmul.cpp` binds: `lhs_indices`
and `rhs_indices` select operand batches from a two-dimensional `A` batch and
a one-dimensional `B` batch, with repeated and reordered indices, a strided
`lhs_indices` read and `rhs_indices` broadcast along the first index axis. The
multi-product cases also pad every leading dimension and batch stride and add
a spare threadgroup in each tile axis. Every output
word, guards included, is compared with an exact reference, and on macOS with
the original source too. The current-tree native parity job runs the 180
entry-variant pairs in seven shards per operating system:

```sh
CROSTL_MLX_CURRENT_ROOT=/path/to/mlx \
CROSTL_REQUIRE_MLX_CURRENT_GEMM_GATHER=1 \
CROSTL_MLX_CURRENT_GEMM_GATHER_SHARD_INDEX=0 \
CROSTL_MLX_CURRENT_GEMM_GATHER_SHARD_COUNT=7 \
python -m pytest -q -n auto demos/integrations/mlx/tests/kernels/test_current_gemm_gather.py
```

The two `gather_mm_rhs` entries are not covered. They derive each row tile
from `offsets` with subgroup scans and return early on their reduction.
Immutable `offsets` storage (#2263) is not enough: DirectX and OpenGL can
lower the early return only once reductions of values equal in every subgroup
count as workgroup-uniform (#2546).

#### Block-masked GEMM

`test_current_gemm_masked.py` executes all 768 `block_masked_gemm` entries of
`steel_gemm_masked.metal`: three element types, two tile shapes, four
transpositions, four alignment pairs and the eight pairings of a boolean,
multiplicative or absent output mask with an operand mask.
`gemm_masked_model.py` names every entry and gives the block-masked product:
a false or zero output-mask block is written as zeros, a K block contributes
only when its lhs and rhs mask values are both non-zero, and multiplicative
masks also scale. Each entry runs three dispatches with the buffers
`BlockMaskedMM::eval_gpu` binds: no batch, a one-dimensional batch, and a
two-dimensional batch that takes the `elem_to_loc` path. Operands and masks
broadcast along batch axes, mask matrices are row- or column-major with
padding, and unread mask words hold a value no mask uses. The shapes give
partial tiles and K tails as each entry's alignment allows. The masks are
declared immutable (`immutable_storage`), which lets the DirectX and OpenGL
software-subgroup proofs treat mask reads at uniform indices as uniform.
Every output word, guards included, is compared with an exact reference, and
on macOS with the original source too. The current-tree native parity job
runs the entries in thirteen shards per operating system:

```sh
CROSTL_MLX_CURRENT_ROOT=/path/to/mlx \
CROSTL_REQUIRE_MLX_CURRENT_GEMM_MASKED=1 \
CROSTL_MLX_CURRENT_GEMM_MASKED_SHARD_INDEX=0 \
CROSTL_MLX_CURRENT_GEMM_MASKED_SHARD_COUNT=13 \
python -m pytest -q -n auto demos/integrations/mlx/tests/kernels/test_current_gemm_masked.py
```

#### Segmented GEMM

`test_current_gemm_segmented.py` executes all 60 `segmented_mm` entries of
`steel_gemm_segmented.metal`: three element types, five tile shapes and four
transpositions. Output batch `z` multiplies A and B over the inner range that
`segments` gives for it, as pairs (`segments_contiguous`) or as consecutive
boundaries. Each entry is translated with four frozen function-constant
configurations, which take both segment layouts and every `align_M` and
`align_N` pair, and runs two dispatches with the buffers `segmented_mm`
binds: one segment over all of K, and five segments that are empty, shorter
than a K tile, exactly one tile, several tiles with a tail, unaligned or
overlapping, with padded strides and extra threadgroups.
`gemm_segmented_model.py` names every entry and gives the segment product.
The segments are declared immutable (`immutable_storage`), so the DirectX and
OpenGL software-subgroup proofs read the K range, and with it the K loop's
trip count, as uniform. Every output word, guards included, is compared with
an exact reference, and on macOS with the original source too. The
current-tree native parity job runs the pairs in seven shards per operating
system:

```sh
CROSTL_MLX_CURRENT_ROOT=/path/to/mlx \
CROSTL_REQUIRE_MLX_CURRENT_GEMM_SEGMENTED=1 \
CROSTL_MLX_CURRENT_GEMM_SEGMENTED_SHARD_INDEX=0 \
CROSTL_MLX_CURRENT_GEMM_SEGMENTED_SHARD_COUNT=7 \
python -m pytest -q -n auto demos/integrations/mlx/tests/kernels/test_current_gemm_segmented.py
```

#### Implicit-GEMM convolution

`test_current_steel_conv.py` executes all 108 `implicit_gemm_conv_2d` entries
of `steel_conv.metal` and all 36 `implicit_gemm_conv_3d` entries of
`steel_conv_3d.metal`: three element types and six block shapes, with the
general-channel input loader for small or large filters and, in 2-D, the
loaders for one to four channels per group. `steel_conv_model.py` names every
entry, models the threadgroup slots each loader writes and `BlockMMA` reads,
and rebuilds the `MLXConvParams<N>` and `ImplicitGemmConv{2,3}DParams` that
`implicit_gemm_conv_{2,3}D_gpu` encode; the test checks the entry lists
against entry discovery. Each entry runs three convolutions that meet its
loader preconditions: padded with a full output tile, strided, dilated and
flipped with a partial column tile and several row tiles, and grouped with a
surplus threadgroup in each tile axis. The host never selects the `bm64_bn32`
entries or the 64-row entries for four or fewer channels; the gate executes
them all the same. Every output word, guards included, is compared with an
exact reference, and on macOS with the original source too. The current-tree
native parity job runs the entries in six shards per operating system:

```sh
CROSTL_MLX_CURRENT_ROOT=/path/to/mlx \
CROSTL_REQUIRE_MLX_CURRENT_STEEL_CONV=1 \
CROSTL_MLX_CURRENT_STEEL_CONV_SHARD_INDEX=0 \
CROSTL_MLX_CURRENT_STEEL_CONV_SHARD_COUNT=6 \
python -m pytest -q -n auto demos/integrations/mlx/tests/kernels/test_current_steel_conv.py
```

`test_current_steel_conv_general.py` executes all 18
`implicit_gemm_conv_2d_general` entries of `steel_conv_general.metal`, each
translated twice with `align_C` frozen true and false.
`steel_conv_general_model.py` rebuilds the jump parameters, the
`base_h`/`base_w` residue tables, the grid and the block selection of
`implicit_gemm_conv_2D_general_gpu`. A Python walk of the kernel's tap loop
over those tables must store every output exactly once with the exact
reference. Each specialization runs three one-group convolutions, chosen so
that `align_C` matches the channel count as the host sets it. One is
undilated with a full output tile. One is input-dilated, dilated, strided and
flipped, with partial tiles and a surplus threadgroup in each tile axis. In
the third, filter taps are several taps apart, or residues have no tap at
all. Outputs are compared as in the `steel_conv` gate, in three shards per
operating system:

```sh
CROSTL_MLX_CURRENT_ROOT=/path/to/mlx \
CROSTL_REQUIRE_MLX_CURRENT_STEEL_CONV_GENERAL=1 \
python -m pytest -q -n auto demos/integrations/mlx/tests/kernels/test_current_steel_conv_general.py
```

### Full-Entry Attention Execution

`tests/kernels/test_current_attention.py` translates seventeen configurations
of unchanged entries of `steel_attention.metal` at commit
`9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8`. The float32 float-mask entries cover
head dimensions 64, 72, 80, 96, 128 and 256 and the asymmetric query and value
head dimensions (96, 64)
(`steel_attention_float32_bq32_bk32_bd64_bv64_wm4_wn1_maskfloat32`, its `bd72`,
`bd80`, `bd96_bv64` counterparts,
`steel_attention_float32_bq32_bk16_bd128_bv128_wm4_wn1_maskfloat32` and
`steel_attention_float32_bq32_bk16_bd256_bv256_wm4_wn2_maskfloat32`).
The float16 and bfloat16 float-mask entries cover head dimension 64. The
boolean-mask (`maskbool_`) entries cover float32 at 64 and 256, float16 at 96
and 128 and bfloat16 at 80. Head dimension 256 uses the `wm4_wn2` entries:
two simdgroups per query row each multiply half of the head dimension and add
their partial scores through the K/V tile reinterpreted as float32. Each entry
is dispatched through the native Metal, DirectX or OpenGL loader. These
thirteen freeze the six function constants to the aligned, masked
configuration. Four more freeze other sets the host selects: unaligned
sequences with attention sinks (float32 float mask at 64, float16 boolean mask
at 128), an unaligned causal mask with the logsumexp (float32 at 80) and an
aligned causal mask with sinks and the logsumexp (bfloat16 at 64). The host
returns the logsumexp only without an array mask. Not yet covered: head
dimension 192, whose 16-bit tiles the OpenGL and bfloat16 DirectX outputs
widen past the 32 KiB workgroup limit (#2307), and the float16 and bfloat16
`wn2` entries, which multiply a float32 score tile by a 16-bit value tile
(#2306). `AttnParams` and
`AttnMaskParams` are filled as
`scaled_dot_product_attention.cpp` fills them, and the grid is the upstream
`(NQ, H, B)` grid of `(32, 4, WN)` threadgroups.

Every entry runs two cases. The first has two heads with 64-row queries and keys and
a per-head mask. The second has two batches, four query heads over two
key/value heads, a 96-row key sequence (`qL_off = 64`), the sequence-major Q/O
layout and a mask broadcast over heads. Inputs are rounded to the entry's
element type, and buffers use the storage each backend reflects: the element
type on Metal, 16-bit words on DirectX, float32 on OpenGL, and 32-bit words
for DirectX and OpenGL booleans. A boolean mask keeps three quarters of the
keys; every third query row drops its whole first key block, which the online
softmax must then discard. Unaligned entries run a 50-row query over 77 keys
with a per-head mask, and a single partial 21-row query block over 70 keys in
the second layout, so both the partial query block and the partial key block
are exercised. A causal row keeps the keys up to its own position plus
`qL_off`, and a sink adds one logit per query head without a value.

With `WN == 1` the kernel widens its tiles to float32, and the float32 `wn2`
entries need no widening. Both accumulate in float32, so each case must match a
float64 evaluation of `softmax(scale * Q K^T + mask) V` to within
`1e-5 * (1 + |reference|)`. A float16 or bfloat16 output must also be a value
that its final store can round that interval to. A saved logsumexp is float32
and must meet the same tolerance. Every case must leave four output guards
unchanged, as well as the four `lse` guards or the unused `lse` buffer. On
Metal, the original MLX
kernel, compiled with the same constants, must meet the same reference. The
CI step runs the entries in parallel `loadgroup` groups, one translation each.

The threadgroup Q, K and V views need no access assertions: OpenGL proves
their offsets from the loaders and from `BaseMMAFrag::get_coord`, and the view
identity selects the tile each accessor reads. OpenGL asserts that device
offsets, including the boolean mask's, stay below 2^31 elements, as the GEMM
gate does. DirectX and OpenGL accept the causal bound because their software
subgroup proofs follow the block limit's writes outside loops, including the
locals declared in the causal branch. DirectX and OpenGL lower the `wn2`
score exchange's `reinterpret_cast<threadgroup float*>` of the float32 K/V
tile as the tile itself, because the cast keeps the element type. This does
not cover head dimension 192, the float16 and bfloat16 `wn2` entries or the
upstream suite.

### Resource-Index Controls

Guarded resource-index controls execute on original/generated Metal and on
Mesa 25.0.7 llvmpipe with exact results and output guards; the same 31 HLSL
artifacts compile with strict DXC settings. The local ARM64 Mesa 22.3.6 driver
returned incorrect results for a signed 64-bit conditional comparison. Updating
the validation environment resolves that case without changing generated GLSL.
This driver result is not evidence of complete MLX runtime coverage.

These controls cover scalar guards and integer fields of by-value records,
including nested records, copies, helper calls and preconditioned loops. Field
mutation and escaping references invalidate proofs; an unrelated field's bound
does not justify an access. Chained offsets cover reads, writes and atomic
destinations, including contended atomic updates. Finite nested value calls
preserve their bounds without authorizing recursive definitions or unsafe
intermediate arithmetic. The Metal round trip also preserves 8-bit and 16-bit
aggregate initializer conversions, including truncation and single evaluation.
Runtime-derived GEMM metadata still requires a checked source or host contract;
these controls do not supply arbitrary bounds for it.

The host adaptation documentation records upstream changes explicitly. A passing
kernel or bounded host check is not a claim that the entire upstream MLX suite
passes on a translated backend.

The selected tiled GEMM entry at pin `9c3d3557` now passes explicit `auto*`
deduction through the public project pipeline, without changing upstream
sources or bypassing specialization analysis. It now produces a Metal artifact:
`MMATile::frag_at(i, j)` aliases retain their original storage and binding-time
indices, including writes through `accum[k]`. The selected float32, non-transposed
entry now passes strict Metal compilation with the six Boolean function constants
set to false. Source-used parameter intent survives specialization and explicit
discard lowering ([#2164](https://github.com/CrossGL/crosstl/issues/2164)).
This is a compilation result, not numerical matrix execution or coverage of all
GEMM specializations. The constant
`GEMMParams` and `GEMMAddMMParams` pointers now retain their indirection, address
space and bindings ([#2160](https://github.com/CrossGL/crosstl/issues/2160)).
Conditional scalar-buffer parameters now retain their bindings and conditions
([#2153](https://github.com/CrossGL/crosstl/issues/2153)). The pointer-deduction
controls cover native Metal/OpenGL execution and strict HLSL compilation;
Windows execution is a separate required CI check. Pointer-object qualifiers
and foreign-target private aliases remain explicit diagnostics under
[#2152](https://github.com/CrossGL/crosstl/issues/2152) and
[#2055](https://github.com/CrossGL/crosstl/issues/2055). These results do not
establish matrix host dispatch or complete upstream-suite parity.
Mixed-access HLSL helper calls remain separately blocked by
[#2158](https://github.com/CrossGL/crosstl/issues/2158): a read-only view of a
writable buffer must preserve the physical resource class when passed to a
helper. The passing execution controls do not cover that unresolved combination.

Conditional-resource controls execute all four combinations of two Boolean
constants on Metal and OpenGL, with sparse binding indices, scalar constant
references and output guards. Metal results also match the original source.
OpenGL execution includes the specialized binary load path tracked in
[#2159](https://github.com/CrossGL/crosstl/issues/2159). Eight scalar/aggregate
HLSL variants have separate strict DXC compilation checks; Windows numerical
execution remains a required CI gate. These controls currently supply all
reflected bindings, including inactive inputs. Conditional aggregate references
compile, but their native packing remains under
[#1991](https://github.com/CrossGL/crosstl/issues/1991).

Constant aggregate pointer controls use read-only structured buffers on DirectX
and OpenGL, preserving element indexing instead of treating every pointer as a
single constant-buffer object. Twenty-four cases execute on Metal and OpenGL
through project packages and reflected two-member layouts. They cover offsets,
single evaluation, local aliases, helper calls, constructors, conditional
specializations, wraparound and output guards. Generated Metal matches original
Metal; all twenty-four HLSL variants compile with strict DXC checks. Windows
numerical execution is required separately. These homogeneous-record controls
do not establish heterogeneous aggregate packing, inactive-binding omission,
complete GEMM translation or matrix host dispatch.

Unnamed helper and constructor parameters retain their intent through saved
CrossGL and Metal round trips ([#2161](https://github.com/CrossGL/crosstl/issues/2161)).
Seven native controls on Metal and OpenGL verify argument side effects, defaults,
overload selection and output guards. Strict Metal compilation keeps unrelated
unused-parameter warnings enabled. These controls do not resolve the remaining
matrix helper materialization and host-dispatch requirements.

Parameter-use controls cover synthesized receivers, object-based static calls,
type-only expressions, explicit discards, local type aliases and compute-input
bindings. Thirteen cases execute on Metal and OpenGL with exact readback,
argument side-effect counts and output guards; Metal also executes the original
sources. Genuine unused named parameters remain warning-fatal. The same cases
are required in the Windows native job, separately from local HLSL compilation.

Partial-specialization static-owner controls cover primary forward declarations,
default arguments, scoped aliases, constructor initializers and method calls
([#2157](https://github.com/CrossGL/crosstl/issues/2157)). Thirteen cases execute
on Metal and OpenGL with exact readback, side-effect counts and output guards;
Metal also executes the original sources. All thirteen HLSL variants compile
with strict DXC checks, with Windows execution required separately. These controls
do not establish complete GEMM compilation or numerical matrix execution.

Aggregate-field receiver controls cover implicit fields, explicit `this`,
parameter and local shadowing, sibling templates, call operators, nested fields
and multidimensional arrays
([#2162](https://github.com/CrossGL/crosstl/issues/2162)). Thirteen cases execute on
Metal and OpenGL, including read-only receivers, argument/index side effects,
wraparound and output guards; Metal also executes the unchanged source controls.
All thirteen HLSL variants compile with strict DXC checks. Windows execution
remains a separate required gate in the existing native job.

Reference-capture controls cover mutable accessor aliases over nested value
fields ([#2163](https://github.com/CrossGL/crosstl/issues/2163)). Thirteen cases
execute on Metal and OpenGL, including changed indices, index side effects,
shared aliases, loop-local bindings, narrow parameter conversions, deduced
constness and output guards. Generated Metal matches the original sources, and
all thirteen HLSL variants compile with strict DXC checks; Windows execution
remains required separately. Unsupported escapes and alias uses are diagnosed.
These controls do not establish complete GEMM execution or upstream-suite parity.

Reduction reference review found an alias-specialization error
([#2100](https://github.com/CrossGL/crosstl/issues/2100)): `float16_t` selected
the primary `Limits` template instead of the explicit `half` specialization.
The generic translator correction restores the positive/negative infinity
initializers for `init_reduce_minfloat16` and `init_reduce_maxfloat16` at both
the historical `846d1762` and current `9c3d3557` pins. All twelve regenerated
Metal/HLSL/GLSL artifacts compile; Metal and OpenGL each pass 68 exact output
values and 64 guard values. Windows execution remains separate. This bounded
check does not accept the complete reduction family's artifact references or
establish its numerical coverage.

The same review exposed a remaining half-precision NaN predicate in HLSL
([#2008](https://github.com/CrossGL/crosstl/issues/2008)). Optimized DXC folded
`value != value` to false inside min/max reduction helpers. Binary16 payload
classification now preserves that predicate. Eight historical whole-array and
row artifacts differ only in the predicate; those and four current-pin artifacts
strictly compile with input-dependent votes retained in DXIL. Generic native
controls cover all 65,536 half payloads on Metal and OpenGL; Windows execution
remains a separate required check. No complete reduction reference set is
accepted from these results.

Eight byte-valued small-row artifacts exposed a writable-argument regression
([#2101](https://github.com/CrossGL/crosstl/issues/2101)). The HLSL correction
preserves the accumulator's storage location instead of narrowing it into a
temporary. All eight historical artifacts now pass strict DXC compilation;
their interfaces are unchanged and each body differs only at the writable
`total_val` arguments. The same eight entries at the current `9c3d3557` pin
also pass strict DXC compilation. This does not accept the full reduction
reference set or establish whole-kernel numerical parity. Upstream kernels are
unchanged.
Generic reference controls cover signed/unsigned overflow, aliases, nested
calls, scalar/vector values and indexed calls under conditionals and loops.

The subsequent complete historical HLSL reduction review covers all 2,396
entries and 39 shapes. Exact old sources were reproduced before comparing every
changed function and declaration: 2,302 artifacts change and 94 remain identical.
All 4,792 old/current strict DXC compiles pass, and a second independent compile
of every current artifact reproduces all 2,396 DXIL modules byte for byte.
The 9,216 materializations and 27,382 reflected resources remain covered.
Among them, 384 logical-half resources in 232 entries retain their two-byte
stride while using explicit unsigned-16 storage encoding. Other binding, access,
layout and entry-point fields agree; a non-required subgroup-size constant still
evaluates to 32. The recorded compiler identity is the pinned Linux DXC binary.
These are translation, interface and compiler checks, not whole-family numerical
execution or current-pin upstream-suite parity.

The separate Metal exponential review exposed numeric grouping loss in source
conversion and target generation
([#2104](https://github.com/CrossGL/crosstl/issues/2104)). Both stages now preserve
addition, multiplication and bitwise expression trees. Generic controls compare
cancellation, rounded products, signed zeros and mixed-width integer conversions
on native Metal and OpenGL, including saved CrossGL intermediates and output
guards. The exponential accuracy checks retain their existing tolerances.
The 15 affected Metal exponential references remain unchanged pending corpus
reconciliation; hosted Windows execution is verified separately.
For the current pinned, profiled bfloat Sigmoid case, only the Metal helper's
result parentheses change. Its reviewed artifact still matches original Metal
over all 65,282 non-NaN inputs, with eight output guards. The HLSL and GLSL
artifacts are byte-identical to their previous references; the OpenGL full-domain
execution also remains exact. The later precision-scope correction moves
`contract(off)` into the precise exponential body and removes the file-level
`contract(fast)` reset. Complete shader comparison leaves all other bytes and
the interface unchanged. Before accepting the new Metal fingerprint, the old
artifact, new artifact and unchanged upstream source each returned the same
65,282 results and eight guards, with read-only buffers unchanged. This review
does not accept the remaining exponential or complex-unary family references.
A subsequent unused-receiver annotation changes only the Metal source fingerprint.
The previous and annotated shaders compile to identical library bytes and both
match the unchanged upstream kernel across the same 65,282 inputs and eight
guards. Read-only buffers and the reflected interface remain unchanged; no
other corpus reference is updated by this review.

Cross-backend controls also found a separate Metal narrow-field reference
failure ([#2102](https://github.com/CrossGL/crosstl/issues/2102)): the field
retains byte storage but its writable helper parameter is widened. Unsigned
byte-vector compound assignment remains a distinct arithmetic limitation under
[#2023](https://github.com/CrossGL/crosstl/issues/2023). Passing DirectX/OpenGL
reference tests does not establish Metal round-trip support for those cases.
Shared source references also require alias-aware lowering
([#2103](https://github.com/CrossGL/crosstl/issues/2103)). The byte-argument fix
rejects potentially overlapping reference arguments rather than relying on
HLSL copy-in/copy-out; wider reference aliasing remains unresolved.

## Corpus Revisions

This directory contains the project-level MLX porting checks used by CrossTL.
The checks are pinned to MLX commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e` and exercise the Metal kernel tree
as a source repository, not as isolated parser snippets. This pinned revision is
an active repository-level verification target: configured coverage and expected
baselines are not, by themselves, evidence that every kernel translates or
passes a target validator.

The next corpus increment is separately pinned to
`9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8`. Its authoritative discovery census
is 49 Metal units, 17,832 entries, and zero discovery diagnostics. The compact
`contracts/arg_reduce.current-tree.translation.json` contract pins all 24
entries from `arg_reduce.metal` and all 72 deterministic Metal, OpenGL, and
DirectX artifacts. Required CI compiles every entry with the applicable native
validator. The common numerical subset is float32:
`argmin_float32` and `argmax_float32` run on Metal, Mesa EGL, and Direct3D 12
WARP over axis sizes 32 and 129, strides 1 and 2, ordinary values, and
NaN/Infinity inputs; macOS additionally compares against the upstream metallib.
All three backends also require all twelve signed/unsigned 16/32/64-bit arg-reduce
entries across 96 integer cases: axis sizes 31, 32, 33 and 129, both strides,
extrema, lowest-index ties and partial groups. Metal compares unchanged upstream
and generated kernels. DirectX uses exact-width software shuffles, including
resolved signed and unsigned overloads around two-word 64-bit payloads.
OpenGL widens 16-bit buffer elements to 32-bit storage; DirectX retains native
16-bit storage. Runtime inputs are checked against each reflected ABI.
This is a 24/17,832 deterministic translation and native-compiler increment with
14/17,832 common required numerical entries, not full-tree
coverage, upstream MLX test-suite execution, or MLX host-runtime redirection.

The compiler and runtime tests use the same checked-metadata configuration for
all 24 entries. The 14 numerical entries retain all 112 cases per backend and
use six upload-validated metadata requirements with no trusted index-range
assertions. Each dispatch
observes all seven read-only inputs and checks 32 output guards; Metal also
compares the unchanged upstream kernel. The input allocation includes space for
pointer formation by inactive lanes and the terminal pointer update, with
extremal padding outside the original rows. Row values, strides, tail conditions
and expected indices are unchanged. The pinned artifact identities now cover
this checked package path. DirectX execution remains required by the existing
Windows job, with the actual DXIL and packed resource bindings retained before
dispatch. A passing compiler check does not qualify numerical execution, the
full MLX runtime or the upstream test suite.

`contracts/searchsorted.current-tree.translation.json` pins all 26 entries of
`searchsorted.metal` (13 element types, left and right) and the 78
deterministic Metal, OpenGL and DirectX artifacts at the same pin. Required CI
compiles each entry with the platform validator and executes all 26 on Metal,
Mesa EGL and Direct3D 12 WARP. Each entry runs three sequences: length 1,
length 33, and length 40 with stride 2 and decoy padding. Each sequence is
searched with 48 queries on 1D and 3D grids, which checks MLX's flattened output
offset. The queries cover duplicates, absent extremes, unsigned sign-bit
boundaries and NaN ordering for float16, bfloat16, float32 and complex64.
Every dispatch checks 16 output guards and four unchanged inputs; Metal also
compares the unchanged upstream kernel. OpenGL uses two host index-range
preconditions, `int64(mid) * a_stride` and `offset` in signed 32-bit range.
Metal and DirectX use none. DirectX float16 buffers carry logical binary16
words in their reflected 16-bit storage. This increment required two translator
fixes.
Metal-source type traits now treat the native `bfloat` spelling behind
`bfloat16_t` as floating point, so `LessThan<bfloat16_t>` keeps its NaN
ordering. DirectX helpers now index widened 8-bit root storage one carrier per
element ([#2145](https://github.com/CrossGL/crosstl/issues/2145)). That second
fix also re-pins 104 row-reduction entries of the 846d DirectX reduce contract.
The increment covers 26/17,832 entries. It does not run the upstream MLX test
suite or redirect the MLX host runtime.

Two units are new at `9c3d3557`: `gather_mm_offsets.metal` and
`cross_entropy.metal` do not exist at either earlier pin. The
`contracts/gather_mm_offsets.current-tree.translation.json` contract pins the
single `gather_mm_offsets` entry and its three artifacts. The kernel is not a
template, so the project-level `workgroup_size = [1, 1, 1]` reproduces MLX's
one-thread-per-group grid; Metal records no execution entry because it sets the
threadgroup size at dispatch time. Required CI compiles and executes it on
Metal, Mesa EGL and Direct3D 12 WARP over six cases: no rows, one row, grouped
and empty experts, 1,000 sorted rows, unsigned sign-bit indices that a signed
comparison would misorder, and an unsorted sequence that checks the exact
midpoint and branch order. Each case pads the indices with four decoys that
would change the result if read, checks 16 output guards and two unchanged
inputs, and Metal also runs the unchanged upstream kernel. No backend needs an
index-range assertion.

`contracts/cross_entropy.current-tree.translation.json` pins all six
`cross_entropy.metal` entries (forward and VJP for float32, float16 and
bfloat16) and their 18 artifacts. Each entry uses a 128-thread workgroup, so
the forward kernel's four SIMD groups exercise the threadgroup reduction.
OpenGL and DirectX lower the forward SIMD reductions to software 32-lane groups,
so results do not depend on the device's subgroup width; the VJP has no
subgroup operations and uses no subgroup options. Required CI executes all six
entries on all three backends over rows of 7, 512 and 1,300 classes, which cover
idle threads, one exact pass and three passes with a tail. The cases include
negative targets, masked `-inf` classes, logits near 1,000 that need the
running maximum, and (forward only) a `+inf` logit that selects MLX's
infinite-maximum branch. Float32 results must be within 5e-5 absolute plus 5e-5
relative of a float64 reference, the tolerance of the existing softmax and
logsumexp proofs. 16-bit VJP results must be within one unit in the last place,
and at least 99% must equal the correctly rounded value; a truncating
conversion fails this check. Every dispatch checks 16 output guards and three
or five unchanged inputs, and Metal also runs the unchanged upstream kernel.
OpenGL relies on host preconditions that the row offset and composed element
indices fit in signed 32-bit range: five expressions for the forward kernel and
six for the VJP. The Metal round trip keeps `fast::exp`, so translated Metal
calls the same relaxed exponential as upstream and must reproduce the
unchanged upstream kernel bit for bit; OpenGL and DirectX call their default
`exp`, and all three stay within the tolerance. The VJP's
donated-input aliasing of `out` and `in` is not covered. These two increments
cover 7/17,832 entries; they do not run the upstream MLX test suite or redirect
the MLX host runtime.

`contracts/arange.current-tree.translation.json` pins all 11 `arange.metal`
entries (eight integer types, float16, float32 and bfloat16) and their 33
artifacts. Each entry dispatches one thread per element on Metal, Mesa EGL and
Direct3D 12 WARP and must match exactly. Integer cases wrap modulo the type
width, step downwards through unsigned wraparound, and form 64-bit products
above 2^32; int64 cases stay within the signed range because Metal's signed
overflow is undefined. Metal fuses `start + index * step` even under
`-fno-fast-math`, so OpenGL and DirectX select
`floating_contraction_profile = "rne-flush"`. For 0.1 + index * 0.3,
separate rounding would differ from the pinned library in 1,269, 906 and 227 of
4,096 float32, float16 and bfloat16 results. The floating cases also cover
binary32 subnormal flushing of operands and of an exact result below the
smallest normal value, binary32 and binary16 overflow, binary16 subnormal
results, and integer-to-binary16 and integer-to-bfloat16 rounding of indices up
to 65,534. Every dispatch checks 16 output guards and two unchanged constants,
and Metal also runs the unchanged upstream kernel against the same expected
words. Metal itself keeps the native expression, which its compiler contracts
in the same way. This covers 11/17,832 entries; it does not run the upstream
MLX test suite or redirect the MLX host runtime.

`contracts/random.current-tree.translation.json` pins both `random.metal`
entries (`rbitsc` for row-contiguous keys and `rbits` for strided keys) and
their six artifacts. Both kernels are non-template, so the project-level
`workgroup_size = [1, 1, 1]` makes `threads_per_grid` equal the
(keys, half_size + odd) workgroup count that MLX dispatches; DirectX reads it
from the generated dispatch-count constant buffer. The `rbits` union of a
`uint2` and two `uchar4` views lowers to one packed word pair on OpenGL and
DirectX, and the `char` output widens to sign-extended 32-bit slots. Required CI
executes both entries on Metal, Mesa EGL and Direct3D 12 WARP over 24 cases and
21,403 key bytes, and every byte must match exactly. 15 cases reproduce the
Threefry 2x32 known answers that MLX's C++ suite (`tests/random_tests.cpp`,
generated with JAX) pins for `random::bits` at widths 4, 2 and 1, for odd and
even word counts with edge bytes, and for the vmapped row and column key
layouts. The other cases cover all-ones and sign-bit keys that wrap the key
schedule, every combination of word-count parity and edge bytes, 2,193
workgroups, broadcast and permuted key views, and unread decoy words between
strided keys. Widths 1 and 2 still store the whole dropped-last word, so their
expected stores include the bytes after the key. Each dispatch checks 16 output
guards and three or six unchanged inputs; Metal runs every case under two
complementary byte guards and also runs the unchanged upstream kernel. OpenGL
relies on host preconditions that the per-key byte offset and the composed byte
index fit in signed 32-bit range. CI now also checks out MLX's `tests`
directory so the gate can verify the known-answer file. This covers 2/17,832
entries; it does not run the upstream MLX test suite or redirect the MLX host
runtime.

`contracts/dot.current-tree.translation.json` pins the three `dot.metal`
entries (float32, float16 and bfloat16; 32 items per thread, 512-thread groups
of 16 SIMD groups) and their nine artifacts. MLX reads four consecutive
elements through `*reinterpret_cast<const device metal::vec<T, 4>*>`. Metal
keeps that native vector view. OpenGL widens 16-bit storage to one binary32 array
element per value and DirectX keeps one 16-bit structured-buffer element per
value, so both gather the four lanes element by element instead of
reinterpreting storage bits; DirectX decodes bfloat16 lanes with an exact shift.
OpenGL and DirectX reduce in software 32-lane groups. The second reduction runs
under `tid < SIMD_GROUPS`, so both targets predicate it: every invocation
reaches the barriers and inactive invocations contribute the exact sum identity
-0.0, so an all -0.0 sum keeps its sign. Native DirectX waves remain subject to
[#2120](https://github.com/CrossGL/crosstl/issues/2120). Required CI executes
all three entries on Metal, Mesa EGL and Direct3D 12 WARP over 13 cases from 1
to 40,000 elements, crossing vector tails, SIMD blocks and three groups. Eleven
cases use multiples of 1/8, so every partial sum is exact in any order and must
match bit for bit. A rounding case must stay within 128 * 2^-24 of the sum of
absolute products, and an infinite case must produce +inf. Each dispatch pads
both inputs with 16 NaNs, checks 16 output guards and three unchanged inputs.
Metal also runs the unchanged upstream kernel, whose partial sums must equal the
translated kernel's bit for bit. This covers 3/17,832 entries. The
host-side all-reduce of the partials is not covered, the upstream MLX test suite
is not run, and the MLX host runtime is not redirected.

`contracts/logsumexp.current-tree.translation.json` pins the six
`logsumexp.metal` entries (block and looped kernels for float32, float16 and
bfloat16) and their 36 artifacts. MLX sizes block threadgroups by row length
(32 threads per 128 elements, up to 4,096 elements) and runs longer rows on the
looped kernel with 1,024 threads, so the block kernel is pinned at 32, 288 and
1,024 threads and the looped kernel at 1,024. OpenGL and DirectX reduce in
software 32-lane groups. The block kernel's second maximum and sum run under
`simd_group_id == 0`, so both targets predicate them: every invocation reaches
the barriers and inactive invocations contribute the identity (-inf for the
maximum, -0.0 for the sum). Required CI executes all six entries on
Metal, Mesa EGL and Direct3D 12 WARP over 17 cases of 1 to 20,000 elements, each
dispatched with the kernel and threadgroup size MLX would choose. The cases
include partial reads, idle threads, a last SIMD group with one active thread,
masked `-inf` entries, values near 1,000 that overflow `exp` without the
maximum, a rising row that forces the looped kernel to rescale its running sum
in each of three passes, and a `+inf` entry that selects MLX's
infinite-maximum branch in the second pass. Float32 results must be within
5e-5 plus 2^-21 relative of a float64 reference; float16 and bfloat16 results
must be a correct rounding of a value in that interval, which rejects truncating
conversions and one-ulp slips. Every dispatch pads the input with 16 NaNs and
checks 16 output guards and two unchanged inputs, and Metal also runs the
unchanged upstream kernel against the same bounds. OpenGL relies on host
preconditions that the row offset and the composed element indices fit in
signed 32-bit range. As for cross entropy, the Metal round trip keeps
`fast::exp` and must reproduce the upstream kernel bit for bit. Rows whose every
element is `-inf` are not covered because GLSL
leaves `log(0)` undefined. This covers 6/17,832 entries; it does not run the
upstream MLX test suite or redirect the MLX host runtime.

`contracts/softmax.current-tree.translation.json` pins the ten `softmax.metal`
entries (block and looped kernels for float32, float16 and bfloat16, plus the
precise float16 and bfloat16 variants that accumulate in float) and their 60
artifacts, at the same 32-, 288- and 1,024-thread block groups and 1,024-thread
looped groups as logsumexp. The non-precise float16 and bfloat16 kernels
accumulate in their storage type, so OpenGL and DirectX now predicate binary16
sums and maxima, and the bfloat16 overloads' converted float reductions, under
`simd_group_id == 0`. A predicated binary16 sum requires a guard on the
subgroup index alone, so each subgroup takes part completely or not at all and
the pairwise rounding order is unchanged; guards that split a subgroup are
still rejected. Required CI executes all ten entries on Metal, Mesa EGL and
Direct3D 12 WARP over the logsumexp cases, checking every element of every row.
Each output is checked against a float64 softmax with a per-element bound on the
kernel's own rounding: one rounding per operation in the accumulation type,
Metal's `fast::exp` allowance of 3 + 2|x| binary32 units in the last place, and
the depth of each reduction (31 sequential binary32 additions or the five
rounded levels of the binary16 tree). Results must be a correct rounding of a
value inside that bound, so the binary32 and precise variants stay within about
1e-5 relative while the bound for the worst-case binary16 and bfloat16 chains of
the looped kernel reaches 4% and 18%. Rows that contain `+inf` must be NaN, as
in MLX's kernel. Every dispatch pads the input with 16 NaNs and checks 16 output
guards and two unchanged inputs, and Metal also runs the unchanged upstream
kernel against the same bounds; the Metal round trip keeps `fast::exp` and must
match that kernel bit for bit. OpenGL relies on host preconditions that the
row offsets and composed element indices fit in signed 32-bit range. This covers
10/17,832 entries; it does not run the upstream MLX test suite or redirect the
MLX host runtime.

`contracts/layer_norm.current-tree.translation.json` pins the twelve
`layer_norm.metal` entries (single-row and looped forward and VJP kernels for
float32, float16 and bfloat16) and their 108 artifacts. MLX sizes single-row
groups by row length, so the forward kernel is pinned at 32, 288 and 832
threads (its 6,656-element limit) and the VJP at 32, 288 and 1,024 (8,192
elements); looped kernels run 1,024 threads. The VJP specializes `has_w`
(function constant 20) on whether a weight was given: DirectX materializes
both values, while Metal and OpenGL keep a function or specialization constant
that the runtime sets when it creates the pipeline. Required CI executes every
variant on Metal, Mesa EGL and Direct3D 12 WARP over 14 forward and 14 VJP cases
of 1 to 20,000 elements, with weights of stride 0, 1 and 2 (NaN between strided
elements), biases of stride 0 and 1, constant rows, rows whose mean is 32 times
their spread, and rows whose variance is comparable to eps. Without a weight,
the VJP must leave `gw` untouched. Each output is checked against a float64
reference with a per-element bound on the kernel's binary32 rounding (the
depth of each sum, 2.5 units in the last place per division and 2 per precise
root) and on the 16-bit output arithmetic. Results must be a correct rounding
of a value inside that bound. Every dispatch pads the inputs with 16 NaNs and
checks 16 guards per output and that every input is unchanged, and Metal also
runs the unchanged upstream kernel, with the same function constants, against
the same bounds. Idle threads of MLX's single-row kernels run the padding loop
from a negative start, so they store below `thread_x`; the translation keeps
those stores as written. OpenGL relies on host preconditions that the row
offsets and composed element indices fit in signed 32-bit range. This covers
12/17,832 entries; it does not run the upstream MLX test suite or redirect the
MLX host runtime.

`contracts/rms_norm.current-tree.translation.json` pins the twelve
`rms_norm.metal` entries (single-row and looped forward and VJP kernels for
float32, float16 and bfloat16) and their 126 artifacts. MLX sizes single-row
groups by row length up to 4,096 elements, so the forward kernel is pinned at
32, 288 and 1,024 threads and the VJP at 32, 64, 288 and 1,024; looped kernels
run 1,024 threads. The single-row VJP spreads its rows over at most 512
threadgroups, so a group loops over several rows, clamps its last row to the
row count and accumulates `gw` over its rows into MLX's `gw_temp`; its two
SIMD-group sums share one `simd_group_id == 0` guard. OpenGL and DirectX
predicate both sums in every invocation, in source order, inside the row
loop. The VJP specializes `has_w` (function constant 20): DirectX materializes
both values, while Metal and OpenGL keep a function or specialization constant
that the runtime sets when it creates the pipeline. Required CI executes every
variant on Metal, Mesa EGL and Direct3D 12 WARP over 14 forward and 16 VJP
cases of 1 to 20,000 elements per row and up to 1,025 rows, with weights of
stride 0, 1 and 2 (NaN between strided elements), zero rows, and rows whose
mean square is comparable to eps. Without a weight, the VJP must leave `gw`
untouched. Each output is checked against a float64 reference with a
per-element bound on the kernel's binary32 rounding (the depth of each sum,
2.5 units in the last place per division and 2 per precise reciprocal root)
and on the 16-bit output arithmetic. Results must be a correct rounding of a
value inside that bound. Every dispatch pads the inputs with 16 NaNs and checks
16 guards per output and that every input is unchanged, and Metal also runs
the unchanged upstream kernel, with the same function constants, against the
same bounds. OpenGL relies on host preconditions that the row offsets and
composed element indices fit in signed 32-bit range. The host's `gw_temp`
column reduction is not part of this gate. This covers 12/17,832 entries; it
does not run the upstream MLX test suite or redirect the MLX host runtime.

`contracts/ternary.current-tree.translation.json` pins all 212 `ternary.metal`
`Select` entries and their 636 artifacts: the 14 base kernels for each of the
13 element types, and the `vn`, `vsn` and `svn` work-per-thread kernels for
the ten types narrower than eight bytes. Required CI compiles every entry with
the platform validator and executes it on Metal, Mesa EGL and Direct3D 12 WARP,
two cases per kernel (424 dispatches per target). Flat kernels run 37 elements,
ending inside a work-per-thread tile, and 64 elements, with a scalar `b` or `c`
for the `sv` and `vs` kernels. `v2` grids of 3x2 and 4x3 threads cover the
64-bit offset path, the 3x2 grid with a short last tile. The strided kernels
run 1-D to 5-D shapes with transposed, permuted and stride-0 broadcast
operands, through both the 32-bit and the 64-bit (`large`) index paths.
`Select` must move every payload bit unchanged, so each result is compared
bit for bit. The inputs include integer extremes and sign boundaries, signed
zeros, infinities, quiet and signaling NaNs with payloads, subnormals, the
largest finite values, booleans and complex64 pairs with NaN and infinite
parts. Every operand is followed by four decoys that no valid output may read.
Every dispatch checks 16 output guards and that every input is unchanged.
Metal also runs the unchanged upstream kernel and compares its output bytes.
OpenGL widens bool and 8- and 16-bit integer storage to 32-bit words, and holds
float16 and bfloat16 values as their exact binary32 widening. The readback
therefore checks the widened NaN payloads and signaling bits. DirectX keeps
16-bit integer storage and carries float16 and bfloat16 as 16-bit words.
OpenGL selects float32, float16 and bfloat16 values through their bit
patterns, and DirectX selects float16 words the same way. OpenGL relies on
11 host preconditions that the flat, 64-bit and composed strided indices fit
in signed 32-bit range; Metal and DirectX use none. This covers 212/17,832
entries; it does not run the upstream MLX test suite or redirect the MLX host
runtime.

`contracts/binary_two.current-tree.translation.json` pins all 225
`binary_two.metal` `DivMod` entries and their 675 artifacts: the 15 base
kernels for each of the 13 element types, and the `svn`, `vsn` and `vvn`
work-per-thread kernels for the ten types narrower than eight bytes. Each
kernel writes MLX's `FloorDivide` result to `c` and its `Remainder` result to
`d` from one `metal::array<T, 2>` value. Metal output keeps that value as
`array<T, 2>`, since Metal functions cannot return C arrays, and OpenGL keeps a
`T[2]` array. DirectX returns a two-component vector, or a two-member struct for
complex64. Required CI compiles every entry with the platform validator and
executes it on Metal, Mesa EGL and Direct3D 12 WARP, two cases per kernel (450
dispatches per target). The scalar kernel writes one and six outputs. Flat
kernels run 37 elements, ending inside a work-per-thread tile, and 80
elements. `*2` grids of 3x2 and 4x3 threads cover the 64-bit offset path, the
3x2 grid with a short last tile. The strided kernels run 1-D to 5-D shapes with
transposed, permuted and stride-0 broadcast operands, through both the 32-bit
and the 64-bit (`large`) index paths. The operand pairs cover every sign
combination with integer extremes for each width, booleans, signed zeros,
infinities, NaNs, subnormals, the largest finite values and quotients that
round to an integer. The same-index kernels realize every pair of each type.
Integer results are compared exactly against floor division and a remainder
with the divisor's sign, narrowed after integer promotion. Floating results are
compared bit for bit with an exact model of Metal's binary32 division, its
remainder and its binary16 remainder, which the same per-operator profiles
reproduce on OpenGL and DirectX; any NaN matches any NaN. A float16 quotient
is floored after binary16 rounding, so 39.90625 / 6.65234375 floors to 6 while
its remainder, 6.64453125, keeps the truncated quotient 5, as upstream does.
Complex64 uses exactly representable components, so its quotient and truncated
remainder do not depend on contraction. Its remainder is
`a - b * static_cast<int64_t>(a / b)`; Mesa folds that integer round trip into
`trunc()`, whose -0.0 turns a -0.0 dividend's remainder into +0.0, so OpenGL
converts such integers back to float through a helper that returns +0.0 for a
zero integer, as Metal does. Every dispatch checks 16 guards per output, four
decoys after each operand and that every input is unchanged. Metal also runs
the unchanged upstream kernel against the same expected words. OpenGL relies on
seven host preconditions that the flat, 64-bit and composed strided indices fit
in signed 32-bit range; Metal and DirectX use none. This covers 225/17,832
entries; it does not run the upstream MLX test suite or redirect the MLX host
runtime.

`contracts/rope.current-tree.translation.json` pins all 18 `rope.metal`
entries and their 360 artifacts: the single-step and general kernels, each
with base-derived or explicit frequencies, the general kernels with 32-bit and
64-bit (`large`) indices, for float16, bfloat16 and float32. `rope.cpp`
specializes three function constants, `forward` (1), `traditional` (2) and
`hs_transpose` (3), and the single kernels read only the first two, so each
single kernel is pinned in four specializations and each general kernel in
eight, 120 per target. DirectX materializes the constants; Metal and OpenGL
keep a function or specialization constant that the runtime sets when it
creates the pipeline, so every specialization of an entry shares one
artifact. As `rope.cpp` dispatches them, the kernels run one thread per
rotated pair, and one-thread workgroups give OpenGL and DirectX the same
`threads_per_grid`. Required CI executes every specialization on Metal, Mesa
EGL and Direct3D 12 WARP, 504 dispatches per target. The single kernels run
five decode cases with zero, negative and 4,000-position offsets and a partial
rotation of a copied row. The general kernels run six cases: head counts that
leave one, two or three heads in the last group of four, per-batch offsets, a
partial rotation, a strided 3-D view whose skipped elements are NaN, and a
4,000-position context. With `hs_transpose` set they run two cases on a
`[B, T, N, D]` buffer viewed as `[B, N, T, D]`, the one layout for which
`rope.cpp` sets it. Explicit frequencies use strides 1 and 2, with NaN between
strided frequencies. Each output is checked against a float64 rotation with a
per-element bound on the kernel's binary32 rounding: one unit in the last
place per operation, 2.5 per division, and the larger of four and 3 + 2|y|
units for `exp2`, carried through the angle. Translated Metal keeps
`fast::cos` and `fast::sin` and must match the upstream kernel bit for bit;
OpenGL and DirectX lower them to their default functions, so each may be
`2^-17 + 2^-20 * abs(theta)` from the exact value, at least twice the worst
error measured on these arguments for Apple's fast functions, Mesa llvmpipe
and WARP, whose `sin` of small negative angles is up to 2.4e-6 off. Results
must be a correct rounding of a value inside that bound, and unrotated
elements must keep their exact bits. Every dispatch pads each input with 16
NaNs, or offsets with positions far from any case, and checks 16 output guards
and that every input is unchanged. Metal also runs the unchanged upstream
kernel, with all three function constants, against the same bounds. The 64-bit
index kernels run the same small cases, although MLX selects them only above
`INT32_MAX` elements. OpenGL relies on host preconditions that the frequency,
offset and composed element indices fit in signed 32-bit range: one for the
single kernels and six for the general kernels. This covers 18/17,832 entries;
it does not run the upstream MLX test suite or redirect the MLX host runtime.

`contracts/scaled_dot_product_attention.current-tree.translation.json` pins all
82 `scaled_dot_product_attention.metal` entries and their 1,254 artifacts: the
four `sdpa_blocked` scale-copy and causal-mask kernels for float16 and
bfloat16, and all 24 one-pass `sdpa_vector`, all 24 `sdpa_vector_2pass_1`
first-pass, all 12 `sdpa_vector_2pass_1_gqa` grouped-query first-pass and all
18 `sdpa_vector_2pass_2` aggregation kernels for float32, bfloat16 and
float16. `sdpa_vector()` specializes six
boolean function constants, `has_mask` (20) through `has_sinks` (25), and
`sdpa_vector_2pass()` adds the block count (26). Each one-pass kernel is
pinned in eight specializations that pair every mask mode (none, boolean,
additive float or causal) with both query layouts and both sink settings, and
every query layout with both sink settings; each first-pass kernel takes the
same eight, each with 32, 64 or 128 blocks, so every block count meets both
query layouts, both sink settings and both threadgroup shapes; the
grouped-query first passes declare no function constants: 418
specializations per target. DirectX materializes the constants; Metal and
OpenGL keep function or specialization constants, so every one-pass
specialization of an entry shares one artifact. The one-pass and
aggregation kernels run 32 SIMD groups of 32 threads, which OpenGL and DirectX
lower to 32-lane software subgroups, predicating the masked key loop so that
every invocation reaches the barriers. Required CI executes every
specialization on Mesa EGL and Direct3D 12 WARP, 1,062 dispatches per target
(the first passes in their own step), and on Metal wherever the device fits
MLX's threadgroup (below). The one-pass cases are a single-query decode over grouped-query heads,
fewer keys than SIMD groups, a four-query prefill over transposed keys and
values, three batches of one head over a longer key/value cache, and keys
whose scores rise along the sequence, so every SIMD group rescales its
accumulators at every key. Array masks are per head, broadcast or transposed,
with rows that drop every key, and buffers that a specialization disables
hold decoys that would change every result. The aggregation kernels combine
32, 64 and 128 synthetic first-pass blocks, including empty blocks, a row
without keys and a dominant block. Each attention output is checked against a
float64 reference with a per-element bound on the kernel's binary32 rounding:
one unit in the last place per operation, 2.5 per division, the larger of four
and 3 + 2|x| units per exponential plus its rounded argument, scores summed
from D / 32 products per lane across 32 lanes in any order, and every rescaling
a key's weight passes through. Translated Metal keeps `fast::exp`; OpenGL and
DirectX use their default `exp`. Results must be a correct rounding of a value
inside that
bound, rows without keys must be exactly zero, and the blocked kernels, which
round once, must be bit-exact, signed zeros included. Every dispatch pads the
floating inputs with 16 NaNs and checks 16 output guards and that every input
is unchanged. Metal also runs the unchanged upstream kernel, with all its
function constants, against the same bounds, and the translated kernel must
reproduce it bit for bit. Before dispatching, the Metal gate reads both
pipelines' `maxTotalThreadsPerThreadgroup`, as MLX's
`check_kernel_threadgroup_size` does: where MLX's own pipeline fits the 1,024
threads it requests, the translation must fit them too. Where neither does, as
for the float32, float16 and seven of the eight bfloat16 head-dimension-512
specializations on an M2 Max (640 or 832 threads for both pipelines), MLX
itself would throw, so the gate records both limits and skips that dispatch;
OpenGL and DirectX still execute them. OpenGL relies on host
preconditions that composed element indices and pointer offsets fit in signed
32-bit range: two for scale copy, one for the causal mask, four each for
the one-pass and first-pass kernels and eight for the grouped-query first
pass.

The first pass runs one SIMD group per query of every query head that shares
a key/value head, and OpenGL and DirectX compile that threadgroup in: its
translations group two query heads with one query (32 x 2 x 1 threads) or,
for transposed queries and the causal mask, four (32 x 2 x 4). Its key loop
starts at the block index and steps by the block count, so OpenGL and DirectX
freeze all seven constants, making every first-pass specialization its own
artifact, and a host precondition that `N` is at most 2^31 - 1024 bounds the
loop's last step. Its cases hold 1,024 to 1,061 keys, where MLX takes the
two-pass path on Max and Ultra devices: a two-batch decode over grouped-query
heads with a fully masked row, a one-head decode over a longer cache with
unscaled scores, a four-query prefill over transposed keys, values and masks,
and rising keys. Each block's maximum must lie within the largest score error
of the exact maximum; its sum and unnormalized partial outputs are checked
against float64 references relative to the maximum it reports, which
`sdpa_vector_2pass_2` rescales by, with the same per-operation bounds, and
blocks without keys must keep exactly their initial values. On Metal the
translated first pass must reproduce the unchanged upstream kernel's
partials, sums and maxima bit for bit.

The 12 `sdpa_vector_2pass_1_gqa` first passes are what `sdpa_vector_2pass()`
takes for one query over 8, 12 or 16 query heads per key/value head, without
masks or sinks, from 8,192 keys. Each threadgroup runs G SIMD groups
(32 x G x 1 threads): every group covers HPT of its key/value head's G query
heads (8, 4, 4 or 2 per instantiation) over one of HPT contiguous key
sub-chunks of its block, stores its normalized partials, sums and maxima in
threadgroup memory, and after a barrier combines its own head's HPT
sub-chunks. Because each SIMD group's key loop runs over its own sub-chunk,
OpenGL and DirectX run every group for the sub-chunk's full length and mask
the steps past its keys, so each software-subgroup reduction stays uniform.
Its cases decode 8,192 keys over two key/value heads in 128 blocks of whole
sub-chunks, 8,197 keys of a two-batch cache with unscaled scores in 64
blocks, and 8,320 rising keys in 256 blocks, which leave short and empty
sub-chunks and three blocks without keys; each block count is MLX's default
for its shape on some device. The checks are the first pass's, with each
key's weight also passing through the factor that rescales its sub-chunk to
the block's maximum. OpenGL relies on eight index-range preconditions here,
bounding the advanced key and value pointers and their 64-bit sequence
strides. This covers 82/17,832 entries; it does not run the upstream MLX test
suite or redirect the MLX host runtime.

`contracts/gated_delta_update.current-tree.translation.json` pins all 90
entries of `gated_delta_update.metal`, the 72 sequential `gated_delta_seq`
entries and the 18 eight-step `gated_delta_fused_chunk` entries, and their 486
artifacts: float32, bfloat16 and float16 storage, the six head layouts that
`supported_gated_delta_shape()` accepts, and, for the sequential kernel,
checkpoint intervals 1, 4, 8 and 16. The host names keep the C++ type
spellings, such as `seq_gated_delta_float16_t_128_128_16_64_1`.
`gated_delta_update.cpp` clears function constant 200, `save_state`, for
inference and sets it for the backward's sequential save pass, so each
sequential entry is pinned in both specializations, 144 per target; the fused
kernel runs only for inference and never reads the constant. DirectX
materializes the constant; Metal and OpenGL keep a function or specialization
constant, so both specializations share one artifact. Each SIMD group of the
sequential kernel updates one row of a head's 128 x 128 state, and OpenGL and
DirectX lower its two reductions to 32-lane software subgroups. Required CI
executes every specialization on Metal, Mesa EGL and Direct3D 12 WARP, 342
dispatches per target. The sequential kernel runs with MLX's
`(32, Dv, B * Hv)` grid of `(32, 4, 1)` groups: inference decodes one token
and prefills eight tokens of two batches, the longest sequence MLX's inference
sends to this kernel; the save pass decodes one token and runs 2 * Ckpt + 1
steps, so it stores three checkpoint slots. Queries and keys are unit-norm
rows rounded to the storage type; values, decays in [0.5, 1), betas in [0, 1)
and the initial state are random. Every output, every final-state element and
every stored checkpoint is checked against a float64 recurrence with a
per-element bound on the kernel's binary32 rounding: one rounding of relative
size 2^-24 per operation, an allowance for flushed subnormal results, any
summation order of each 128-term dot product across the 32 lanes, and the
error each state row carries from step to step, which `g (I - beta k k^T)`
does not amplify while `beta |k|^2 <= 2`. Results must be a correct rounding
to their storage type of a value inside that bound, which stays below 2^-12
of their root mean square. Without `save_state` the cache must stay
untouched. A float64 simulation of the sequential kernel's own pointer
arithmetic matches the reference, and 14 injected defects are rejected.

The fused kernel processes eight steps at a time in the WY form. It holds its
8 x 8 tiles in `simdgroup_float8x8` values: it loads and stores the state
transposed with `simdgroup_load` and `simdgroup_store`, reads the keys,
queries and values through `thread_elements()`, the keys both straight and
transposed, multiplies tiles with `simdgroup_multiply` and
`simdgroup_multiply_accumulate`, inverts the chunk's `I + beta K K^T` through
a Neumann product and prefix-sums the log decays across the SIMD group. Its
fragments sit at MLX's own `(fm, fn)` lane coordinates, which the gate
registers as the `tile_4x4_row_pair` fragment mapping. OpenGL and DirectX lower the tiles to
32-lane software subgroups, each lane holding its two fragment elements, and
prove the chunk loop `for (; t + C <= T; t += C)` uniform from a host
precondition `0 <= T <= 2^31 - 9`, which keeps `t + C` from overflowing. MLX
picks this kernel for T > 8 on devices without NAX and dispatches a
`(32, Dv / 8, B * Hv)` grid of `(32, 4, 1)` groups, each SIMD group holding an
eight-column block of every state row of one head. Each entry runs one chunk
and a one-step tail (B = 1, T = 9), a five-step tail in each of two batches
(B = 2, T = 13) and three chunks (B = 1, T = 24), 54 dispatches per target.
A head's keys share a direction, so the WY inverse departs far from the
identity and every Neumann factor counts. The bound gives every tile product
any summation order, `fast::exp` 3 + 2|x| ulp and `fast::log` 2^-21 absolute
error on [0.5, 2], tracks the rounding committed within a chunk element by
element and carries the state's error between chunks as a row 2-norm through
each step's `g (I - beta k k^T)`; it stays below 2^-7 of each output's root
mean square, and the Metal outputs use under 3% of it. A float64 simulation
of the fused kernel's pointers, tail masks and tile transposes matches the
reference, and 14 injected defects are rejected. DirectX used to give an
offset pointer such as `v_ + dv_idx` the type of its integer offset, so
`static_cast<float>((v_ + dv_idx)[i])` passed each value through an unsigned
integer and bfloat16 loads and stores lost their encoding; `p + n`, `n + p`
and `p - n` now keep the pointer's element type.

Every dispatch of either kernel pads each input with 16 NaNs and checks 16
guards per output and that every input is unchanged. Metal also runs the
unchanged upstream kernel with the same function constant, and the
translation must reproduce it bit for bit. No backend needs an index-range
precondition. This covers 90/17,832 entries; it does not run the upstream MLX
test suite or redirect the MLX host runtime.

`contracts/gated_delta_update_vjp.current-tree.translation.json` pins all 72
`gated_delta_vjp_seq` entries of `gated_delta_update_vjp.metal` and their 216
artifacts: float32, bfloat16 and float16 storage, the six head layouts that
`supported_gated_delta_shape()` accepts, and checkpoint intervals 1, 4, 8 and
16, such as `seq_gated_delta_vjp_float16_t_128_128_16_64_4`. MLX's sequential
backward host path always selects interval 1; the other intervals are
instantiated in the metallib and replay their segments from sparser
checkpoints. Each SIMD group walks one state row backward through every
checkpoint segment, first replaying the segment's entry states into a
`float seg[Ckpt][4]` register array. The query, key, decay and beta gradients
are added to zero-filled buffers through `mlx_atomic<float>`: Metal keeps
`atomic_fetch_add_explicit` on `atomic_float`, OpenGL lowers it to a
compare-and-swap loop on `uint` words that hold the float bits, and DirectX to
an `InterlockedCompareExchangeFloatBitwise` loop. OpenGL and DirectX lower the
four 32-lane reductions to software subgroups. OpenGL needs one index-range
precondition: the complete element offset `object + offset` of every atomic
update must fit signed int range. Required CI executes every entry on Metal,
Mesa EGL and Direct3D 12 WARP with MLX's `(32, Dv, B * Hv)` grid of
`(32, 4, 1)` groups, 216 dispatches per target: one step, three steps of two
batches, and 2 * Ckpt + 1 steps, three checkpoint segments with a partial last
one. The checkpoint cache holds the exact forward states of a random initial
state rounded to float, and the cotangents are random. All six gradients are
checked element by element against a float64 backward recurrence with a bound
on the kernel's binary32 rounding: each segment's replay error starts from its
checkpoint, the cotangent state carries its error through `g (I - beta k k^T)`
across all steps, and the atomic gradients allow any summation order of their
128 to 512 contributions. Results must be a correct rounding of a value inside
that bound; a binary32 evaluation in the kernel's own operation order uses
under 10% of it in every case. Every dispatch pads each input with 16 NaNs and
checks 16 guards per output and that every input is unchanged; the value and
initial-state gradients start as guard words, so an unwritten element fails.
Metal also runs the unchanged upstream kernel: it must meet the same bounds,
and its value and initial-state gradients, which do not depend on thread order,
must match the translation bit for bit. A float64 simulation of the kernel's
own pointer arithmetic matches the reference, and 18 injected defects are
rejected. This covers 72/17,832 entries; it does not run the upstream MLX test
suite or redirect the MLX host runtime.

`contracts/sort.current-tree.translation.json` pins all 264 entries of
`sort.metal` and their 792 artifacts: the 228 single-block kernels, contiguous
(`c`) and non-contiguous (`nc`) sort and argsort for 12 element types with
32 to 512 threads (256 for 8-byte types), such as
`ncarg_block_sort_float16_uint32_bn256_tn4`, and the 36 multi-block kernels
`sort_mbsort`, `partition_mbsort` and `merge_mbsort`. MLX sorts with a stable
merge sort under `LessThan`, which puts NaN after every ordered value and
compares complex values by real then imaginary part. Each thread sorts its
four values in a register array passed by reference (`thread T (&vals)[N]`),
and the merge helpers take `const threadgroup T*` views of the block's tile.
Metal never runs the constructor of a threadgroup variable on the device, so
the complex64 tiles start with unspecified values. Translation used to lower
that default construction to stores by every thread over the whole tile, with
no barrier before other threads' first writes; threadgroup declarations now
carry no runtime initialization. This re-pins the 128 complex64 entries of each
846d reduce contract that stage partials in threadgroup memory.
Value-only sorts pass `nullptr` for the index tile: DirectX binds that view to
an inert specialization and fails closed if the helper could read it, and
OpenGL folds the statically null view away. OpenGL needs two kinds of host
precondition: every tile view a helper receives stays in
`[0, N_PER_BLOCK - 1]`, because the merge partitions are data-dependent binary
searches the translator cannot bound, and the 64-bit element offsets of the
non-contiguous and `sort_mbsort` kernels fit signed int range. Required CI
executes every entry on Metal, Mesa EGL and Direct3D 12 WARP, 690 dispatches
per target, with the shapes `sort.cpp` routes to each kernel. Contiguous block
kernels sort a 1-D row whose segment strides are `INT32_MAX`, two contiguous
rows of a full tile and three columns along axis 0; non-contiguous ones sort a
strided 1-D view and a transposed `(2, 3, size)` array into a fresh output.
Each multi-block type sorts the rows and the columns of a two-row array whose
sorted axis spans three tiles: `sort_mbsort` sorts the tiles, then
`partition_mbsort` and `merge_mbsort` merge them with `merge_tiles` 2 and 4,
and every stage's output is checked before the next stage reads it. The values
cover every level of the type, many ties, NaN, both zero signs, complex values
with equal real parts and 64-bit values with equal low words, so stability and
the NaN and signed-zero order are observable. Every output word is compared
exactly, 16 guards per output must survive, every input must be unchanged, and
storage the kernel must not read holds the smallest value. Metal also runs the
unchanged upstream kernels, which must match the expected words exactly. A
model of `merge_partition` and the three multi-block stages reproduces the
stable sort, and an unstable merge, NaN first and IEEE zero order are
rejected. This covers 264/17,832 entries; it does not run the upstream MLX
test suite or redirect the MLX host runtime.

`contracts/conv.current-tree.translation.json` pins the 27 non-Winograd
entries of `conv.metal` and their 126 artifacts: `naive_unfold_nd` and
`naive_unfold_transpose_nd` for one to three spatial dimensions, which
`conv.cpp` runs before the GEMM of its explicit convolution without and with
groups, `depthwise_conv_2d` and `depthwise_conv_1d`, each for float32, float16
and bfloat16, plus the int64-index `_large` forms of the 1-D kernel.
`MLXConvParams<N>` mixes ints, fixed `int64_t` stride arrays and a trailing
`bool`; translation reflects it as a 120-, 176- or 224-byte record on every
target whose bytes match MLX's C struct, with Metal's one-byte `bool` and the
four-byte std430 and structured-buffer `bool` at the same offset.
`depthwise_conv_2d` takes it by `constant` reference, which OpenGL binds as a
448-byte std140 uniform block and DirectX as a 432-byte `ConstantBuffer`; the
gate packs both from their reflected layouts. Its tile load uses
`simd_gid * 32 + simd_lid` as the linear thread index, so OpenGL and DirectX
lower it with 32-lane software subgroups. The depthwise kernels accumulate into
`float o = 0.`; once a 1x1 kernel unrolls, Mesa folds that +0.0 away even from
a `precise` addition, so a negative-zero product would end as -0.0 where Metal
gives +0.0. OpenGL and DirectX therefore accumulate both depthwise kernels with
the software binary32 addition (`binary32_additive_profile = "rne-flush"`),
which works on the operands' bits. The depthwise kernel size, strides
and flip are function constants: Metal and OpenGL keep them, and DirectX
materializes each of six specializations (3x3 at strides 1 and 2, flipped
5x5 and 7x7, 1x1 and a flipped 3x5 at strides (1, 2)). OpenGL needs
index-range preconditions that keep the composed 64-bit element offsets in
signed int range. Required CI executes every entry on Metal, Mesa EGL and
Direct3D 12 WARP, 114 dispatches per target, with `conv.cpp`'s groups:
`(32, 8, 1)` for 32-channel unfolds, `(8, 8, 4)` for depthwise 2-D and
`get_block_dims`' `(16, 16, 4)` for depthwise 1-D. Metal clips the last
groups of `dispatch_threads`, which OpenGL and DirectX cannot, so every
unfold and 1-D case tiles its grid exactly and partial groups are not
exercised. The unfold cases cover padding, strides, kernel and input dilation,
flipped kernels, batches and later row tiles; the depthwise cases cover ragged
output edges, asymmetric padding and batches. The unfold kernels copy distinct
values, and depthwise inputs are quarters and weights halves in `[-1, 1]`, so
every product and sum is exact in the storage type and each output word is
compared exactly with a reference built from the convolution's definition.
Every dispatch pads its inputs with NaN, checks 16 output guards and checks
that every input is unchanged. Metal also runs the unchanged upstream
kernels, which must produce the same words. Simulations of the kernels' own
index arithmetic and tile loads match the references, and injected flip,
padding, stride, dilation, row-offset, layout, batch and channel defects are
rejected. The nine Winograd transforms have their own gate, described next.
This covers 27/17,832 entries; it does not run the upstream MLX test suite or
redirect the MLX host runtime.

`contracts/winograd.current-tree.translation.json` pins all nine Winograd
entries of `conv.metal`, the weight, input and output transforms for float32,
bfloat16 and float16, and their 27 artifacts. `winograd_conv_2D_gpu` computes
a stride-one 3x3 convolution as F(6x6, 3x3) products: it moves each filter
slice into the 8x8 Winograd domain as `G g G^T`, each overlapping 8x8 input
tile as `B^T d B` and, after its batched GEMM, each tile back to a 6x6 output
block as `A^T m A`. Each SIMD-group forms these with two
`simdgroup_matrix<float, 8, 8>` products per channel, the lanes holding the
row pairs of Apple's 4x4-tile layout (`tile_4x4_row_pair`). Metal keeps the
hardware matrices. OpenGL and DirectX lower each product with 32-lane software
subgroups as the software multiply-accumulate with the `fma-rne-gradual`
profile: every element is a chain of fused binary32 multiply-adds from +0.0 in
increasing k with ties to even and gradual underflow, which Apple's matrices
match. `G` holds non-dyadic constants such as -2/9, which every compiler
rounds once to binary32, and `B` constants such as 21/4, so many steps round
and the reference comes from an exact fused multiply-add oracle. `A` holds
powers of two, so the output transform's products are exact and it checks
summation order and rounding rather than fusion. The channel loops hold
barriers, so OpenGL and DirectX carry checked metadata preconditions: the
weight transform's `C <= INT_MAX - 31` and the input and output transforms'
`params.C` and `params.O` record members in the same range, read at their
reflected offsets in the `MLXConvParams<2>` record, std140 block or constant
buffer. The input and output transforms' matrix loops start at the SIMD-group
index and step by the SIMD-group count with equal trip counts in every
SIMD-group, which translation proves for the software subgroup barriers.
OpenGL index-range preconditions keep the composed offsets in signed int
range. Required CI executes every entry on Metal, Mesa EGL and Direct3D 12
WARP with `conv.cpp`'s launch geometry: `(32, 4, 1)` groups over
`(O / 4, 1, 1)` for the weight transform and `(32, 2, 2)` groups over
`(tiles_w, tiles_h, N)` for the others, with `conv.cpp`'s padded input and
`MLXConvParams<2>`. One case has one batch of whole 6x6 output tiles, 32
channels and 64 filters; the other has two batches with partial output tiles
in both directions, 96 channels and 32 filters. Every output word is compared
exactly. All -0.0 inputs must give +0.0 words and a unit impulse the outer
product of a column of the left factor and a row of the right one. Inputs are
padded with NaN, 16 output guards are checked and every input must be
unchanged; Metal also runs the unchanged upstream kernels, which must produce
the same words. Replays of the kernels' index arithmetic, shared-tile loads
and guarded stores match the reference layouts, and unfused products, reversed
or reassociated products, transposed inputs, untransposed factors and
neighbouring channels, filters or tiles are rejected. This covers 9/17,832
entries; it does not run the upstream MLX test suite or redirect the MLX host
runtime.

`contracts/scan.current-tree.translation.json` pins all 456 entries of
`scan.metal` and their 1,368 artifacts: `contiguous_scan` and `strided_scan`
for each of its 57 helpers (sum, product, max and min over bool, the eight
integer types, float16, float32, bfloat16 and complex64, and `logaddexp` over
the three floats and complex64), inclusive and exclusive, forward and reverse,
such as `strided_scan_reverse_exclusive_prod_int64_int64`. Each thread reads
`N_READS` elements, two for 8-byte types and four otherwise. Sums and products
of narrower types scan with Apple's `simd_prefix_*` intrinsics, everything
else with the kernels' own `simd_shuffle_and_fill_up` loops. OpenGL and
DirectX lower both to 32-lane software subgroups; DirectX adds with the
`native-precise` binary32 policy, and its relative wave shuffles return the
lane's own value out of range, as Apple's do. DirectX holds an exact bfloat16
value as its bits in a `uint` and its lane moves now pass those bits on; they
used to move `uint(float(x))`, which gave the 24 bfloat16 max, min and
`logaddexp` scans wrong words on Direct3D 12 WARP, and those DirectX artifacts
are re-pinned. `contiguous_scan` writes one `simdgroup_sums` slot per
SIMD-group and then reads all 32, and DXC loads a `groupshared bool` word with
`!range [0, 2)`, so loads of the never-written words are undefined in DXIL
rather than unspecified values. With bool words, the four contiguous bool
maximum scans gave lanes 1 to 4 of SIMD-group 0 a `true` from outside their
prefix on WARP in two consecutive runs, after identical HLSL had passed once.
The scan gate therefore sets DirectX's `groupshared_bool_storage = "uint"`,
which declares groupshared bool scalars and arrays as `uint` words that HLSL
converts to and from `bool` at each use; the 24 DirectX artifacts of the bool
maximum, minimum and product scans are re-pinned. Complex `logaddexp` needed the software subgroup convergence
pass to stage guarded inputs around helpers that call `cos`, `sin`, `sqrt` or
`atan2`, end in `if`/`else` returns (complex `log1p`) or call the generated
precise binary32 square root.
Required CI executes every entry on Metal, Mesa EGL and Direct3D 12 WARP,
1,824 dispatches and 3,019,068 checked elements per target, on grids of three
groups. Contiguous kernels use `scan.cpp`'s 128 threads on three rows of 1 and
37 elements, one block of `N_READS * 128` and two blocks plus five, which
carries the prefix across the block loop. Strided kernels use
`(32 / N_READS) * 32` threads on axis 1 of `(outer, axis, stride)` arrays of
`(3, 1, 7)`, `(3, 37, 5)`, `(3, 64, 32)` and `(1, 33, 70)`. A model of both
kernels follows their block loops, SIMD-group scans, threadgroup carries and
guarded loads and stores step by step, keeps unspecified threadgroup memory
out of every result, checks that each input is read and each output written
once, and matches an independent left fold of each line. Inputs make results
exact in any association order: quarters with bounded prefix sums, signed
powers of two, wrapping integers, overflow-free products, one zero sign per
line for extrema and `logaddexp` values of 64 times an odd number or -inf.
One case per `logaddexp` kernel takes dense inputs instead, with a bound
propagated at 16 ulp per operation (64 for complex values); every other word
is compared exactly. Outputs start different from their expected values, 16
trailing output guards and 16 padded input elements are checked and every
input must be unchanged. Checked metadata preconditions keep `axis_size` and
`stride` in [1, 4096] and `stride_blocks` in [1, 128]. OpenGL also needs
index-range preconditions for the reverse kernels, whose block pointers may
sit before their line, and the model checks every executed access against
them. Metal also runs the unchanged upstream kernels, and the translation
must reproduce their words exactly. On the development machine (M2 Max,
Metal 32023.918) both give words that differ from the model for reverse
contiguous scans of one-byte types, which address blocks as
`in + axis_size - offset - N_READS`; the same pointers formed with signed
offsets give the model's words. Only those 26 entries may differ from the
model on Metal, and only together with the upstream kernel. This covers
456/17,832 entries; it does not run the upstream MLX test suite or redirect
the MLX host runtime.

`contracts/gemv.current-tree.translation.json` pins 224 of the 256 entries of
`gemv.metal` and their 672 artifacts: `gemv` and `gemv_t` for float32,
float16, bfloat16 and complex64 over the seven and five tile shapes
`matmul.cpp` selects, each with and without non-contiguous batches and
`axpby`, and `gemv_gather` and `gemv_t_gather` over their three and five tile
shapes. The 32 `gemv_wide` and `gemv_wide_gather` entries have their own gate,
described below. OpenGL and DirectX lower the kernels'
SIMD-group shuffles to 32-lane software subgroups, which need every invocation
at their barriers, so three translator changes came with this family. Programs
whose early returns cannot be proven uniform, such as `GEMVKernel::run`'s exit
for rows past the output, are rendered again with those returns converged into
predicated statements. A void function's top-level branch whose condition may
differ between invocations and which holds a barrier, such as `gemv_t`'s row
loop under `if (out_col < out_vec_size)`, runs its barriers and their counted
loops in every invocation and its other statements under the saved condition.
The Metal frontend materializes thread `operator+=`-style templates in value
form and lowers aggregate `x += y` on thread-local values to
`x = helper(x, y)`, which complex64 accumulators need. Required CI executes
every entry on Metal, Mesa EGL and Direct3D 12 WARP, 672 dispatches and
262,440 checked elements per target, with `matmul.cpp`'s `(32, bn, bm)`
threads and grids of `(ceil(outputs / outputs per threadgroup), 1, batches)`.
Each entry runs a short case with fewer inputs than one block iteration, a
ragged one with a partial tail block, leftover inputs and partial loads, and
an aligned one; contiguous batches, `[2, 2]` batches with a broadcast vector
and permuted matrix strides, and gathers over one- and two-dimensional index,
vector and matrix layouts. A model of `GEMVKernel::run` and
`GEMVTKernel::run` replays each thread's block position, early exit, tail
shift, iteration and leftover loads, SIMD-group shuffle-down sums,
threadgroup reduction and guarded stores, and must store each direct dot
product. Inputs are small integers: float32 accumulates their products
exactly in any order, results stay within 96 (bfloat16), 512 (float16) or
2^20 (float32) and are never zero, and `alpha = 2` and `beta = -0.5` over
biases that are 2 mod 4 keep `axpby` exact, so every target must give the
model's words. Outputs start at a sentinel no result can equal, 16 trailing
output guards are checked, unread input elements hold exact poison values and
every input must be unchanged. OpenGL's index-range preconditions for the
batch, gather, row and column offsets are justified by the model, which
checks every load and store against its buffer, and mutated models that drop
the leftover inputs, the tail shift, the batch offsets, `beta * bias` or the
threadgroup reduction are rejected. Metal also runs the unchanged upstream
kernels, and the translation must reproduce their words exactly. This covers
224/17,832 entries; it does not run the upstream MLX test suite or redirect
the MLX host runtime.

`contracts/gemv_wide.current-tree.translation.json` pins the other 32 entries
of `gemv.metal`, `gemv_wide` and `gemv_wide_gather` for float16 and bfloat16
with two to five vectors per threadgroup and 16 or 32 K lanes, in the 80
function-constant specializations `matmul.cpp` selects: every
`(gemv_wide_has_batch, gemv_wide_do_axpby)` pair for `gemv_wide` and
`do_axpby = false` for `gemv_wide_gather`. Metal keeps the function constants
and OpenGL keeps them as specialization constants, so one artifact serves an
entry's specializations there, while DirectX bakes each pair into its own
artifact. The kernels' `for (chunk = tid.x; chunk < n_chunks; chunk +=
tgpg.x)` loop holds SIMD-group shuffles, so the software subgroup translation
now accepts a 32-bit counter stepped by a uniform operand: the update is
modular, so every invocation visits the same values whatever the operand.
Required CI executes 240 dispatches and 48,208 checked elements per target on
Metal, Mesa EGL and Direct3D 12 WARP with `matmul.cpp`'s `(32, k_lanes / 8, 1)`
threadgroups: a short case whose second vector chunk is partial, whose last
row block is clamped and whose K needs only the leftover loop, a ragged case
whose two grid columns walk three chunks round-robin over an unrolled block
and leftover loads, and an aligned case, over contiguous, `[2, 2]`
non-contiguous and gathered batches and broadcast, strided and contiguous
biases. A lane-level model of `GemvWide::run` must store each direct product;
small integer inputs keep every float32 sum, result and `axpby` term exact, so
every word is compared exactly. The model justifies OpenGL's 16 or 18
index-range preconditions load by load, and mutated models that drop the
leftover loads, the chunk stride, half of the K lanes, the batch offsets or
`beta * bias` are rejected. Metal also runs the unchanged upstream kernels with
the same function constants, and the translation must reproduce their words
exactly. This covers 32/17,832 entries; it does not run the upstream MLX test
suite or redirect the MLX host runtime.

`contracts/gemv_masked.current-tree.translation.json` pins all 528 entries of
`gemv_masked.metal` and their 1,584 artifacts: `gemv_masked` and
`gemv_t_masked` for float32, float16 and bfloat16 over the five and six tile
shapes `BlockMaskedMM` selects, with every pairing of a boolean,
multiplicative or absent output mask and operand mask except no masks at all,
each with one and two batch dimensions. Their exits sit inside branches, so
the software subgroup convergence of early returns now also handles a later
`if (c) { ...; return; }` whose statements before the return neither
synchronize nor leave: invocations taking it run those statements and then
skip every later statement except barriers and collectives. A branch on a
folded template constant, such as `if (has_output_mask)` around such an exit,
is converged in place when the constant is true and dropped when it is false,
`uniform<int>` loop bounds that a
later barrier loop reads are computed by every invocation, and predicated
`bool`, half and bfloat16 declarations start from `false` or `T(0)`. The
Metal round trip restores `_NoMask`'s literal `constexpr operator bool()`
members for each receiver address space, because Metal type-checks the
statically dead `!mask` and `bool(mask[i])` branches. Metal host reflection
now reads past non-virtual member function definitions in a struct, which do
not change its layout, so each absent mask binds as a one-byte `_NoMask`
record on Metal; OpenGL and Direct3D widen its `char` member to a 32-bit
`int`. Required CI executes
every entry on Metal, Mesa EGL and Direct3D 12 WARP in two legs, 1,584
dispatches and 371,088 checked elements per target, with `matmul.cpp`'s
`(32, bn, bm)` threads. Each entry runs a short case with fewer inputs than
one block iteration, a ragged one with three output and input mask blocks,
partial tails and leftover inputs, and an aligned one, over contiguous and
`[2, 2]` batches whose output, matrix and vector masks have their own batch
strides. A model of the masked `GEMVKernel::run` and `GEMVTKernel::run`
replays each thread's block position, early exit, tail shift, output mask,
operand masks and scales, loads, SIMD-group shuffle-down sums and guarded
stores, and must reach the masked product computed directly: a false or zero
output block is zero, a multiplicative output mask scales its block, and a
false or zero input block is skipped while multiplicative operand masks scale
it. Mask scales are 1, -1 and 2 and inputs are small integers, so every
result is exact and every word is compared. Outputs start at a sentinel, 16
trailing output guards are checked, unread mask elements hold a value no mask
uses and every input must be unchanged. The model justifies OpenGL's six
index-range preconditions for the batch offsets load by load, and mutated
models that ignore either mask, drop the scales, the leftover inputs or the
tail shift, or address every batch or its masks as the first are rejected.
Metal also runs the unchanged upstream kernels, and the translation must
reproduce their words exactly. This covers 528/17,832 entries; it does not
run the upstream MLX test suite or redirect the MLX host runtime.

`contracts/copy.current-tree.translation.json` pins all 2,496 entries of
`copy.metal`: the scalar, vector, two-dimensional and general strided copies,
with and without work per thread, over all 169 pairings of MLX's 13 dtypes,
and the same-type general copies with destination strides and dynamic
offsets. `copy.metal` and `copy.h` are byte-identical at the corpus commit,
and `utils.h` and `defines.h` only gained declarations the copy kernels do
not use, so each entry must reproduce, byte for byte, the Metal, OpenGL and
DirectX artifact its complete-family translation contract pins there, under
the same configuration, including OpenGL's twelve index-range preconditions.
Required CI executes every entry on Metal, Mesa EGL and Direct3D 12 WARP in
six legs, 4,992 dispatches and 233,220 checked elements per target: flat
copies with a tail inside the last thread's elements and with an exact size,
two-dimensional grids with a short last tile, strided, transposed, permuted
and broadcast layouts of up to five dimensions whose innermost extent ends
inside a thread's tile, destination strides that leave padding the kernel
must not touch, and dynamic source and destination offsets. Each read
element takes the next value of a domain that starts with the source type's
boundary values: integer extremes and wrap points, values around every
narrower range, float32 rounding ties and the overflow thresholds of float16
and bfloat16, signed zeros, infinities, quiet and signaling NaNs, subnormals
and the largest finite values. A model of `cast_to<U>(T)` gives every
expected element: same-type copies are bit-exact, conversions to a floating
type round the exact source value to nearest even and overflow to infinity,
floats truncate toward zero into integers, integers wrap, float32, bfloat16
and complex64 test their bits for `bool`, complex sources convert their real
part, and a NaN only has to stay a NaN when the types differ. Casts that C++
leaves undefined, from non-finite or out-of-range floats to integers and from
integers to an overflowing float16, are kept out of the inputs (427 of 3,458
boundary casts). Outputs start at values that differ from their results,
padding and 16 trailing guards must keep their values, four decoys follow
every source and every input must be unchanged. Metal also runs the
unchanged upstream kernels, which must produce the same bytes. This covers
2,496/17,832 entries; it does not run the upstream MLX test suite or
redirect the MLX host runtime.

`contracts/binary.current-tree.translation.json` pins all 4,122 entries of
`binary.metal`: the scalar and vector kernels with one element per thread,
with work per thread and on two-dimensional grids, and the strided kernels of
one to three dimensions and of any rank, for all 24 operators over every
dtype each one instantiates. `binary.metal` and `binary.h` are byte-identical
at the corpus commit and `binary_ops.h` only changed complex `Power`, which
now takes `log(hypot(re, im))`. Every other entry must reproduce, byte for
byte, the Metal, OpenGL and DirectX artifact its complete-family translation
contract pins there, under the same configuration and arithmetic profiles;
the 15 complex power entries reproduce the current artifacts this contract
pins. Required CI executes every entry on Metal, Mesa EGL and Direct3D 12
WARP in twelve legs, 8,244 dispatches and 390,122 checked elements per target,
over the same tails, short last tiles and strided, transposed, permuted and
broadcast layouts as the copy family, with separate strides for both
operands. Operands come from 254 domains over the 238 operator and dtype
families in which every pair is defined in C++ and has an exactly defined
result: integers wrap after promotion, with signed overflow, division by
zero, `MIN / -1`, negative or oversized shifts and overflowing powers kept
out; floating sums, products, quotients and remainders need no rounding and
stay normal, next to signed zeros, infinities and NaNs, and float16
remainders follow Metal's binary16 `fmod`, `x - trunc(x / y) * y` in float32,
which gives NaN for an infinite divisor and +0 for an exact zero; selections and
comparisons also take the extremes, the smallest normals and values one ulp
from one; `LogAddExp`, `ArcTan2` and `Power` use only their exactly defined
values, and complex operands follow `complex.h` with exact float32
intermediates. An independent model gives every output element and a NaN
result accepts any NaN. Outputs start at values that differ from their
results, 16 trailing guards must keep their values, four decoys follow both
inputs and every input must be unchanged. Metal also runs the unchanged
upstream kernels, which must produce the same bytes. Rounding, subnormal and
accuracy behaviour stays with the per-operator gates. This covers
4,122/17,832 entries; it does not run the upstream MLX test suite or
redirect the MLX host runtime.

`contracts/unary.current-tree.translation.json` pins all 877 entries of
`unary.metal`: the scalar, vector and two-dimensional kernels, with and
without work per thread, and the strided kernels with one and four elements
per thread, for all 37 operators over every dtype each one instantiates,
complex64 and the FP8 conversions. `unary.metal` and `unary.h` are
byte-identical at the corpus commit. `unary_ops.h` now computes `Sigmoid`
with the precise exponential and complex `Abs` and `Sign` with the `hypot`
that `utils.h` added, which complex `Log`, `Log2`, `Log10`, `Sqrt`, `Rsqrt`,
`ArcCos`, `ArcSin` and `ArcTan` reach as well, and `fp8.h` rewrote the E4M3
decoding. The 75 entries those changes reach, including the five `uint32`
`Sign` kernels whose artifacts also carry the complex overload, reproduce the
current artifacts this contract pins; every other entry must reproduce, byte
for byte, the Metal, OpenGL and DirectX artifact its complete-family
translation contract pins there, under the same configuration, including
OpenGL's and DirectX's three index-range preconditions. Required CI executes
every entry on Metal, Mesa EGL and Direct3D 12 WARP in two legs, 1,754
dispatches and 90,945 checked elements per target, guards included: flat
kernels with a tail inside the last thread's elements and with an exact size,
two-dimensional grids with a short last tile, and a transposed and a
broadcast five-dimensional layout whose innermost extents end inside a
four-wide tile. Inputs come from 183 operator and dtype families, 4,428
values whose results are exactly defined, and each case reads its family's
domain where the previous case stopped: integer and rounding operators take
every boundary value C++ defines, with overflowing `Abs`, `Negative` and
`Square` inputs kept out, FP8 conversions follow `fp8.h`'s bit algorithm over
all 256 codes, and transcendental operators take only the points where
IEEE-754, C99 or the MLX helper fixes the result: signed zeros, infinities,
NaN, poles, domain errors and the overflow, underflow and saturation
thresholds valid in every evaluation width. Complex operands follow
`unary_ops.h`, `complex.h`, `cexpf.h` and `utils.h` with exact float32
intermediates. An independent model gives every output element and a NaN
result accepts any NaN. Outputs start at values that differ from their
results, 16 trailing guards must keep their values, four decoys follow every
input and every input must be unchanged. Metal also runs the unchanged
upstream kernels, which must produce the same bytes. Accuracy away from those
points stays with the per-operator unary gates. This covers 877/17,832
entries; it does not run the upstream MLX test suite or redirect the MLX host
runtime.

`contracts/reduce.current-tree.translation.json` pins all 2,396 entries of
`reduce.metal` and their 7,188 artifacts: `init_reduce` for 40 operator and
type pairs and, for 62 operator, input and output type triples (`And` and
`Or` of seven types, `Sum` and `Prod` of twelve with MLX's widened outputs,
`Min` and `Max` of twelve), `all_reduce`, `row_reduce_simple`,
`row_reduce_small`, `row_reduce_looped`, `col_reduce_small`,
`col_reduce_longcolumn`, `col_reduce_looped` and `col_reduce_2pass`, the
last six with 32- and 64-bit indices over one, two and five reduction
dimensions. `reduce.metal` is byte-identical at the corpus commit;
`reduce_col.h` now forms `col_reduce_longcolumn`'s lane offset as
`in_idx + IdxT(lid.x)`, so OpenGL's index-range precondition for it reads
`in_ + in_idx + int64(lid.x)`, and its other 20 are the complete family's.
Each entry translates with the threadgroup `reduce.cpp` launches for the
geometries the gate runs: 32 threads for `init_reduce` and
`row_reduce_small`, 64 for `all_reduce`, 128 for the simple and looped row
kernels, 32 x 8 for `col_reduce_small`, the stride (7) x 32 for
`col_reduce_longcolumn` and 256 for the tiled column kernels. The six kernels
that call SIMD-group operations lower them to 32-lane software subgroups on
OpenGL and DirectX; DirectX adds with the `native-precise` binary32 policy,
and its relative wave shuffles return the lane's own value out of range, as
Apple's `simd_shuffle_down` does. Software subgroups used to reject binary16
products and 16-bit integer minima and maxima, which `Prod<half>`,
`Min<int16_t>` and their neighbours need: OpenGL now computes each node of a
binary16 product tree in binary32, which is exact for two binary16 factors,
and rounds it to binary16, and DirectX multiplies `float16_t` and compares
16-bit integer registers natively. Required CI executes every entry on Metal,
Mesa EGL and Direct3D 12 WARP in twelve legs (`reduce-0` to `reduce-11`),
2,830 dispatches and 1,186,324 checked elements per target, guards included:
three `init_reduce` threadgroups; `all_reduce` over 203 elements in one
threadgroup and the first pass of MLX's two-pass reduction of 16,589
elements, 128 rows of 130 with the last cut to 79; `row_reduce_small` with
each row reduced by one thread and by one SIMD-group; `row_reduce_simple`
over 33 rows of 522, whose last threadgroup moves back and rewrites three
outputs; and the column kernels over 130, 7, 70 and 33 contiguous columns,
including partial column tiles and the `[32, out_size]` intermediates of
`col_reduce_longcolumn` and `col_reduce_2pass`. Rows, reduced dimensions and
outer outputs are separated by decoys that would change every finite result
if read, and four more follow the input. A model of each kernel follows its
thread loops, guarded loads, SIMD-group shuffle-down trees, threadgroup
combines and guarded stores step by step and must match an independent left
fold of every output. Types narrower than eight bytes reduce with Apple's
SIMD-group intrinsics, whose association order is unspecified, so their
inputs make every fold order give the same words: quarters with a bounded
absolute sum, signed powers of two with bounded exponent sums, wrapping
unsigned and overflow-free signed integers, logical inputs whose true values
include NaN and infinities and whose false values take both zero signs, and
one zero sign per output for extrema, with infinities and NaN in their own
patterns. Eight-byte types reduce with the kernel's explicit shuffle loop,
which the model follows in order, so complex products keep the zero signs of
that order. Every word must match and a NaN result accepts any NaN. Outputs
start at values that differ from their results, 16 trailing guards must keep
their values and every input must be unchanged. Metal also runs the unchanged
upstream kernels, which must produce the same bytes. This covers
2,396/17,832 entries; it does not run the upstream MLX test suite or redirect
the MLX host runtime.

`contracts/quantized.current-tree.translation.json` pins 108 of the 2,052
entries of `quantized.metal` and their 324 artifacts: `affine_quantize` and
`affine_dequantize` for float, float16 and bfloat16, group sizes 128, 64 and
32 and 2, 3, 4, 5, 6 and 8 bits. `affine_quantize` reduces each group with
`simd_min` and `simd_max` and packs neighbouring lanes' codes with
`simd_shuffle_down`, which OpenGL and DirectX lower to 32-lane software
subgroups whose out-of-range shuffles keep the lane's own value; Metal's
`round` becomes an explicit halfway-away-from-zero helper on both. Narrow
`uint8_t` buffers widen to one 32-bit word per byte on OpenGL and DirectX, and
OpenGL widens float16 and bfloat16 storage to float32 while rounding each
binary16 or bfloat16 operation. OpenGL is given 5 quantize and 18 dequantize
index-range preconditions, each entry using those its packing reaches, among
them the complete rebased subscripts `out_ + i` and `w + k`; the model shows
each names an element inside its buffer. Required CI executes 216 dispatches
and 131,208 checked elements per target on Metal, Mesa EGL and Direct3D 12
WARP in the `quantized` leg, with 64-thread workgroups: a row of three
(quantize) or two (dequantize) workgroups and a two-by-two grid, whose
`threads_per_grid` row stride the flattened offset uses. `quantize_impl`
launches up to `maxTotalThreadsPerThreadgroup` threads per group; any multiple
of 32 keeps each quantization group inside one SIMD-group, as the kernel
requires.
Quantize groups hold `(k + d - a) * 2^e` for codes `k`, quarter offsets `d`
and an `a` that puts the larger magnitude at either end, so each scale, edge
and code is exact; others are all zero (the `eps` scale), all negative (the
kernel's `w_max` starts at zero), a positive constant and free multiples of
1/64, whose scales and codes need correctly rounded float32 division. A
DirectX target option selects `binary32_division_profile = "rne-flush"`
for quantization so DXC cannot replace a division by a constant with a rounded
reciprocal multiplication. Dequantization and the Metal/OpenGL configurations
retain their existing arithmetic. A lane-level model of the reduction,
rounding, shuffles and byte writes must
write every output byte once and reproduce MLX's least-significant-bit-first
code stream, and mutated models that drop the shuffled codes, truncate codes
or always take the upper edge are rejected. Dequantize unpacks random codes
that include 0 and the top code in every group, with power-of-two scales of
either sign and biases `-a * scale`, so every output is exact in the element
type whatever the evaluation order. Every word is compared exactly, eight
trailing guards per output must keep their values and inputs must be
unchanged; Metal also runs the unchanged upstream kernels, which must produce
the same words. Its matrix-vector and matrix-matrix kernels are outside
this gate. This covers 108/17,832 entries; it does
not run the upstream MLX test suite or redirect the MLX host runtime.

`contracts/quantized_qmv.current-tree.translation.json` pins a further 162
entries of `quantized.metal` and their 486 artifacts: `affine_qmv_fast` with
and without batching and `affine_gather_qmv_fast`, for the same three element
types, three group sizes and six bit widths. Their `simd_sum` lowers to
32-lane software subgroups on OpenGL and DirectX, and OpenGL is given 7
batched and 9 gather index-range preconditions: the one-dimensional batch
offsets of `adjust_matrix_offsets`, the `elem_to_loc_broadcast` offsets of
its general branch and gather's index subscripts, each reached by some case
and shown by the model to name an element inside its buffer. Required CI
executes 378 dispatches and 20,736 checked elements per target on Metal, Mesa
EGL and Direct3D 12 WARP in the `quantized-qmv-0` and `quantized-qmv-1`
legs, following `quantized.cpp`'s `qmv` and `gather_qmv`: `(32, 2, 1)`
threadgroups over a `(M, N / 8, B)` grid. Unbatched entries run one vector
and a three-row matrix; the unchanged upstream kernel gets only the seven
buffers MLX binds when `B <= 1`, and the translations, which reflect every
parameter, also get the unbatched shapes and strides `add_strides_and_shapes`
would bind. Batched entries run stacked batches, a two-dimensional
batch with transposed activations and broadcast weights, and strided
activations sharing one weight matrix (`w_batch_ndims` 0); gather entries
pick activations and weights through a strided one-dimensional index batch
and a two-dimensional one whose right-hand indices are broadcast. Inputs are
quarters in `[-1, 1]`, codes that include 0 and the top code in every group,
power-of-two scales of either sign and biases `-a * scale`, so every float32
product, partial sum and `simd_sum` is a multiple of 2^-5 below 2^19 and
exact in any order; only the final `static_cast<T>` rounds, to nearest even,
which 5,977 float16 and bfloat16 results need, 1,416 of them halfway cases. A
lane-level model of `load_vector`, `qdot`'s masked unpacking and the offsets
must reach direct dequantized dot products and write every output once, and
mutated models that drop the bias term, share one scale per block, reverse
each pack, ignore the simdgroup row or the weight batch offset are rejected.
Every word is compared exactly, eight trailing guards must keep their values
and inputs must be unchanged; Metal also runs the unchanged upstream kernels,
which must produce the same words. The gates below cover the other
matrix-vector and matrix-matrix kernels of `quantized.metal`. This covers
162/17,832 entries; it does not run the upstream MLX test suite or redirect
the MLX host runtime.

`contracts/quantized_qmv_unaligned.current-tree.translation.json` pins a
further 162 entries of `quantized.metal` and their 486 artifacts:
`affine_qmv` with and without batching and `affine_gather_qmv`, for the same
element types, group sizes and bit widths, which `quantized.cpp`'s `qmv` and
`gather_qmv` pick over the `_fast` kernels whenever `N` is not a multiple of 8
or `K` of `qmv_fast_k_alignment(bits)`. `qmv_impl` reads one pack per lane,
runs whole blocks and then a tail that `load_vector_safe` and `qdot_safe`
read only where it reaches the lane; below eight output rows it guards each
row and a simdgroup past the last row returns before `simd_sum`, and
otherwise its last tile moves back to end on row `N - 1`. These
simdgroup-uniform returns now converge on the 32-lane software subgroups of
OpenGL and DirectX, which are given the index-range preconditions of
`qmv_fast`. For 3-, 5- and 6-bit packs `qdot_safe` advances its private
`x_thread` pointer by a count that only the caller's `clamp` bounds, so the
DirectX private pointer analysis now seeds a helper's signed integer
parameters with the hull of their arguments at every call. Required CI
executes 486 dispatches and 21,060 checked elements
per target on Metal, Mesa EGL and Direct3D 12 WARP in the
`quantized-qmv-unaligned-0` through `quantized-qmv-unaligned-3` legs, on
`(32, 2, 1)` threadgroups over a `(M, ceil(N / 8), B)` grid. Unbatched
entries run five guarded rows, three rows whose second simdgroup returns
with `K` below one block for most widths, thirteen rows whose last tile is
moved back, and whole tiles with a misaligned `K`; batched and gather entries
combine guarded, returning and moved tiles with the layouts of the
`qmv_fast` gate, and `K` grows by a group wherever `quantized.cpp` would pick
`qmv_fast` or `qmv_quad` instead. The inputs of `qmv_fast` keep every float32
intermediate exact; only the final `static_cast<T>` rounds, which 5,103
float16 and bfloat16 results need, 1,362 of them halfway cases. A lane-level
model of the blocks, the tail, the guarded rows, the return and the moved
tile must reach direct dequantized dot products, store every row once and
each moved row twice with the same value, and mutated models that drop the
tail, the bias, the simdgroup row or the weight batch offset, share one
scale per block or reverse a pack are rejected. Every word is compared
exactly, eight trailing guards must keep their values and inputs must be
unchanged; Metal also runs the unchanged upstream kernels, which must produce
the same words. This covers 162/17,832 entries; it does not run the upstream
MLX test suite or redirect the MLX host runtime.

`contracts/quantized_qmv_wide.current-tree.translation.json` pins a further
432 entries of `quantized.metal` and their 1,296 artifacts: `affine_qmv_wide`
with and without batching, for the same element types, group sizes and bit
widths and its tiles of 2, 3, 4 and 5 input vectors, each output row reduced
by eight lanes. Its `simd_shuffle_down` ladder lowers to 32-lane software
subgroups on OpenGL and DirectX, DirectX keeping a lane's own value past the
simdgroup's end as Metal does, and OpenGL is given the 7 batched index-range
preconditions of `qmv_fast`, each reached by some case and shown by the model
to name an element inside its buffer. `dequantize` deduces its output pointer
type from the local `w_dq` array; DirectX now decays that array to a thread
pointer as C++ and the OpenGL output do, so the 3, 5 and 6-bit steps
`w_local += 8 * i` move the view's base. Required CI executes 1,080
dispatches and 117,504 checked elements per target on Metal, Mesa EGL and
Direct3D 12 WARP in the `quantized-qmv-wide-0` through `quantized-qmv-wide-8`
legs, following `quantized.cpp`'s `qmv_wide`: `(32, 2, 1)` threadgroups over
a `(ceil(M / vecs_per_tg), ceil(N / 8), B)` grid, where every case's `M` is
one `qmv_wide` gives the entry's tile, below the vector limit of 13, and `K`
avoids `qmv_quad`'s 64 and 128. Unbatched entries run one tile against nine
groups per row, so lane 0 of each row reduces two, and a tiled matrix whose
last row group, and for tiles of 4 and 5 last vector tile, is partial; the
unchanged upstream kernel gets only the eight buffers MLX binds unbatched.
Batched entries run stacked batches, a two-dimensional batch with transposed
activations and broadcast weights, and strided activations sharing one
weight matrix. The inputs of `qmv_fast` make every dequantized weight,
product, partial sum and shuffle a multiple of 2^-5 below 2^19, exact in any
order; only the final `static_cast<T>` rounds, which 26,506 float16 and
bfloat16 results need, 7,567 of them halfway cases. A lane-level model of
`dequantize`'s eight-value sub-chunks, the shuffle ladder and the
partial-tile guards must reach direct dequantized dot products and write
every output once, and mutated models that drop the bias, reverse a
sub-chunk, drop the six-bit steps, give a lane one group, stream one vector,
ignore the lane's row, sum all 32 lanes or ignore the weight batch offset are
rejected. Every word is compared exactly, eight trailing guards must keep
their values and inputs must be unchanged; Metal also runs the unchanged
upstream kernels, which must produce the same words. This covers
432/17,832 entries; it does not run the upstream MLX test suite or redirect
the MLX host runtime.

`contracts/quantized_qvm.current-tree.translation.json` pins a further 162
entries of `quantized.metal` and their 486 artifacts: `affine_qvm` with and
without batching and `affine_gather_qvm`, for the same element types, group
sizes and bit widths, which `quantized.cpp`'s `qvm` and `gather_qvm` pick
for weights that are not transposed, `K` rows of `N` codes, when `M` is
below 4 and, for `qvm`, `K` below 1,024. `qvm_impl` gives each simdgroup 32
output columns and each lane one input row per block of 32: the lane copies
its row's `vec_w` record, reads it through a private byte view and `qouter`
adds `x * (scale * w + bias)` into all 32 results, which `simd_sum` reduces
on 32-lane software subgroups on OpenGL and DirectX. On DirectX, `qouter`'s
pointer becomes an array parameter emitted once per backing; for 3, 5 and
6 bits its own default backing would become the same HLSL overload as the
byte view, so DirectX emits only the backings that calls reach and refuses
any two that remain indistinguishable. After the whole blocks,
a last pass reads only where `K` reaches the lane and zeroes `x`, the scale
and the bias of the other lanes, and a simdgroup whose first column is past
`N` returns before `simd_sum`. Required CI executes 432 dispatches and
293,760 checked elements per target on Metal, Mesa EGL and Direct3D 12 WARP
in the `quantized-qvm-0` through `quantized-qvm-3` legs, on `(32, 2, 1)`
threadgroups over a `(M, ceil(N / 64), B)` grid. Unbatched entries run a `K`
of 13 below one block, whole blocks and two blocks with a nine-lane last
pass, with an odd number of 32-column groups where group size 32 allows it so
that the last simdgroup returns; batched and gather entries combine them with
the layouts of the `qmv_fast` gate. Its inputs keep every float32
intermediate exact; only the final `static_cast<T>` rounds, which 61,872
float16 and bfloat16 results need, 19,574 of them halfway cases. A
lane-level model of the blocks, the last pass and the return must reach
direct dequantized dot products and store every column once, and mutated
models that drop the bias, the last pass, the simdgroup column offset, the
column group or the weight batch offset, keep the first block's scales or
reverse a pack are rejected. Every word is compared exactly, eight trailing
guards must keep their values and inputs must be unchanged; Metal also runs
the unchanged upstream kernels, which must produce the same words. This
covers 162/17,832 entries; it does not run the upstream MLX test suite or
redirect the MLX host runtime.

`contracts/quantized_qvm_split_k.current-tree.translation.json` pins a
further 108 entries of `quantized.metal` and their 324 artifacts:
`affine_qvm_split_k` for the same element types, group sizes and bit widths
and `split_k` 8 and 32, which `quantized.cpp`'s `qvm` picks for weights that
are not transposed when `M` is below 4 and `K` is 1,024 or more, with
`split_k` 32 above `K = 8,192`. The host splits `K` into `split_k` parts of
`split_D = ceil(K / split_k)` rows, the last one of
`final_block_size = K - (split_k - 1) * split_D`, adds a batch dimension of
`split_k` to the activation, weight, scale and bias views, and each `tid.z`
runs `qvm_impl` on its part with the whole `K` as the activation row stride,
writing one partial product per column to an intermediate that a separate
reduction sums. Required CI executes 540 dispatches and 3,229,920 checked
elements per target on Metal, Mesa EGL and Direct3D 12 WARP in the
`quantized-qvm-split-k-0` through `quantized-qvm-split-k-3` legs, on
`(32, 2, 1)` threadgroups over a `(M, ceil(N / 64), B * split_k)` grid.
The cases bind the host's own views, split sizes and final sizes; they reach
a routed `K` of 1,100 with four blocks and a last pass per part, parts of one
whole block, parts of a block and a last pass followed by a whole final
part, parts that are a last pass alone and a final part of another size,
with one or more rows, stacked weights and broadcast weights. The `split_k`
32 entries run the same formulas at smaller `K`, which the kernel cannot
tell apart. The intermediate keeps every float32 partial exact; only the
final `static_cast<T>` rounds, which 273,197 float16 and bfloat16 partials
need, 106,975 of them halfway cases. The qvm gate's lane-level model replays
each part and must reach the dequantized partial dot products; its seven
mutations and two more that give every part the final part's size or step
activation rows by the part's size instead of `K` are rejected. Every word
is compared exactly, guards and inputs must be unchanged, and Metal also
runs the unchanged upstream kernels, which must produce the same words. This
covers 108/17,832 entries; it does not run the upstream MLX test suite or
redirect the MLX host runtime.

`contracts/quantized_qmm_t.current-tree.translation.json` pins a further 216
entries of `quantized.metal` and their 648 artifacts: `affine_qmm_t` for the
same element types, group sizes and bit widths, with aligned and unaligned
`N`, with and without batching. `quantized.cpp`'s `qmm` picks it for
transposed weights when `M` reaches the vector limit (33 at most for `K` and
`N` up to 2,048), for batches, and for single matrices where `qmm_splitk`
finds no split, on devices without NAX kernels. Each `(32, 2, 2)`
threadgroup of a `(ceil(N / 32), ceil(M / 32), B)` grid computes a 32 by 32
tile: per block of 32 codes a `BlockLoader` copies the activations and a
`QuantizedBlockLoader` dequantizes the weights into threadgroup tiles, and
`BlockMMA`'s four simdgroups multiply them as eight by eight fragments,
which DirectX and OpenGL lower to software fragments with
`BaseMMAFrag::get_coord`'s lane mapping. Required CI executes 540 dispatches
and 1,697,328 checked elements per target on Metal, Mesa EGL and Direct3D 12
WARP in the `quantized-qmm-t-0` through `quantized-qmm-t-6` legs. The cases
reach every loop and store path: whole and partial row tiles, whole and
partial column tiles where `N` is unaligned, one to three column tiles, one
or two groups along `K`, stacked, broadcast and shared weights. Every
dequantized weight is exact in the element type and float32 holds every
accumulator exactly; only the final `static_cast<T>` rounds, which 222,077
float16 and bfloat16 outputs need, 82,302 of them halfway cases. A
tile-level model replays both loaders, the group-stepped scales and the
fragment products and must reach direct dequantized dot products; ten
mutations, among them dropped scale steps, steps every block, a transposed
weight tile and whole-tile stores past `M` and `N`, are rejected where they
are defects. Every word is compared exactly, guards and inputs must be
unchanged, and Metal also runs the unchanged upstream kernels, which must
produce the same words. This covers 216/17,832 entries; it does not run the
upstream MLX test suite or redirect the MLX host runtime.

`contracts/quantized_qmm_n.current-tree.translation.json` pins a further 108
entries and their 324 artifacts: `affine_qmm_n` for the same element types,
group sizes and bit widths, with and without batching. `quantized.cpp`'s
`qmm` picks it for weights that are not transposed, `K` rows of `N` codes
quantized along `N`, once `M` reaches the vector limit of 4, on devices
without NAX kernels; there is no split-K fallback. It runs the same
`(32, 2, 2)` threadgroups over a `(N / 32, ceil(M / 32), B)` grid, but its
activation `BlockLoader` copies two rows of four elements per thread, its
`QuantizedBlockLoader` reduces along its 32 weight rows and steps the scales
by a block of rows, and a `K` that is not a multiple of 32 ends with a pass
that loads only the remaining rows of both tiles. Required CI executes 324
dispatches and 1,274,400 checked elements per target on Metal, Mesa EGL and
Direct3D 12 WARP in the `quantized-qmm-n-0` through `quantized-qmm-n-4`
legs. The cases reach every loop and store path: whole and partial row tiles,
each with whole blocks of `K` only and with a last partial block, a `K`
below one block at the vector limit, two to four column tiles, stacked,
broadcast and shared weights. As for `qmm_t`, only the final
`static_cast<T>` rounds, which 261,891 float16 and bfloat16 outputs need,
84,915 of them halfway cases. The tile-level model replays both loaders, the
row-stepped scales and the fragment products and must reach direct
dequantized dot products; twelve mutations, among them dropped or unbounded
last blocks, scales stepped as along `K` and a transposed weight tile, are
rejected where they are defects. Every word is compared exactly, guards and
inputs must be unchanged, and Metal also runs the unchanged upstream kernels,
which must produce the same words. This covers 108/17,832 entries; it does
not run the upstream MLX test suite or redirect the MLX host runtime.

`contracts/quantized_qmm_t_splitk.current-tree.translation.json` pins a
further 108 entries and their 324 artifacts: `affine_qmm_t_splitk` for the
same element types, group sizes and bit widths, with aligned and unaligned
`N`. `quantized.cpp`'s `QuantizedMatmul` sends single matrices with
transposed weights and `M` at the vector limit to `qmm_splitk` on every
device, which splits `K` into `split_k` partitions of whole groups and whole
32-column blocks, aiming at about 512 threadgroups, and falls back to `qmm`
when no split exists. Each `(32, 2, 2)` threadgroup of a `(ceil(N / 32),
ceil(M / 32), split_k)` grid moves the activations, the weight bytes, the
scales and the biases to its partition's first column and its output to the
partition's `M` by `N` slice, then runs `qmm_t_impl` over the partition's
columns while the rows keep the stride `K`; the weights reach it as a
`uint32_t` view of the moved byte pointer, which DirectX and OpenGL now lower
by restoring the viewed word storage. The kernel writes the `(split_k, M, N)`
partial sums that a strided reduction then folds. Required CI executes 216
dispatches and 1,349,136 checked elements per target on Metal, Mesa EGL and
Direct3D 12 WARP in the `quantized-qmm-t-splitk-0` through
`quantized-qmm-t-splitk-4` legs. The cases split `K` into two and three
partitions over whole and partial row tiles and whole and partial column
tiles, reaching every load and store path; only each partition's final
`static_cast<T>` rounds, which 116,872 float16 and bfloat16 partial sums
need, 52,184 of them halfway cases. The tile-level model replays
`qmm_t_impl` per partition and must reach direct dequantized partial dot
products; fifteen mutations, among them partitions that keep the first
partition's activations, weights or scales, run over all of `K` or overlap
their output slices, are rejected where they are defects. Every word is
compared exactly, guards and inputs must be unchanged, and Metal also runs
the unchanged upstream kernels, which must produce the same words. This
covers 108/17,832 entries; it does not run the upstream MLX test suite or
redirect the MLX host runtime.

`contracts/quantized_gather_qmm.current-tree.translation.json` pins a further
162 entries and their 486 artifacts: `affine_gather_qmm_t`, with aligned and
unaligned `N`, and `affine_gather_qmm_n` for the same element types, group
sizes and bit widths. `quantized.cpp`'s `GatherQMM` sends gathered products
with `M` at or above the vector limit to `gather_qmm` on devices without NAX
kernels. Each `(32, 2, 2)` threadgroup of a `(ceil(N / 32), ceil(M / 32), B)`
grid reads its batch's lhs and rhs indices in `adjust_matrix_offsets`, moves
the activations, weights, scales and biases to the matrices they select and
its output to its batch's slice, then runs `qmm_t_impl` or `qmm_n_impl`. The
cases take the gather_qmv and gather_qvm gates' interleaved one-dimensional
indices and their `(2, 2)` batch with broadcast rhs indices, over shapes that
reach every load and store path of both implementations. Required CI executes
324 dispatches and 2,531,808 checked elements per target on Metal, Mesa EGL
and Direct3D 12 WARP in the `quantized-gather-qmm-0` through
`quantized-gather-qmm-6` legs; only the final `static_cast<T>` rounds, which
467,366 float16 and bfloat16 outputs need, 154,413 of them halfway cases. The
qmm_t and qmm_n tile-level models replay the gathered offsets and must reach
direct dequantized dot products; their mutations, and one that uses `tid.z`
in place of the gathered indices, are rejected where they are defects. Every
word is compared exactly, guards and inputs must be unchanged, and Metal also
runs the unchanged upstream kernels, which must produce the same words. This
covers 162/17,832 entries; it does not run the upstream MLX test suite or
redirect the MLX host runtime.

`contracts/quantized_qmv_quad.current-tree.translation.json` pins a further
216 entries and 504 artifacts: `affine_qmv_quad` for the same element
types, group sizes and bit widths, `D` of 64 and 128, with and without
batching. `quantized.cpp`'s `dispatch_qmv` sends every matrix-vector product
with `K` of 64 or 128 and a power-of-two width to `qmv_quad`: one simdgroup
of `(32, 1, 1)` threads per 64 output rows over an `(M, ceil(N / 64), B)`
grid, each of its eight quads computing eight rows eight apart and each quad
lane a quarter of `K`, combined by `quad_sum`. DirectX and OpenGL lower
`quad_sum` to a software reduction over four consecutive invocations, the
grouping of Metal's `quadgroup_index_in_threadgroup`. For 3, 5 and 6 bits
`qmv_quad_impl`'s `32 / bits` pack factor misplaces each lane's codes, and
`qdot` advances `x_thread` cumulatively (`x_thread += 8 * i`), so a lane with
three or more steps reads past its `D / 4` values: 3 and 5 bits at `D` 128 and
6 bits at both widths, 72 entries that no host path dispatches. Out-of-bounds
pointer arithmetic is undefined in the C++14 that the Metal Shading Language
builds on ([expr.add]), so DirectX and OpenGL refuse those 72 with their
`private-pointer-unsupported` bounds diagnostic, whose element counts the
gate derives from the same pointer steps; Metal keeps the upstream
arithmetic. Every other artifact is translated, compiled and validated per
platform. The 90 entries the host can dispatch, power-of-two widths whose
group size divides `D`, also execute: `quantize` needs `K` to be a multiple of
the group size, and `dispatch_qmv` routes only power-of-two widths to
`qmv_quad`. Required CI
executes 270 dispatches and 26,955 checked elements per target on Metal, Mesa
EGL and Direct3D 12 WARP in the `quantized-qmv-quad-0` through
`quantized-qmv-quad-2` legs. The cases reach whole tiles, a partial last tile
and quads with no row below `N`, unbatched and with stacked, broadcast and
shared weights. A row past `N` skips its `qdot` and store but still loads its
scale and bias, so the cases continue both buffers with poison through the
last tile and the model shows those values never reach an output. Only the
final `static_cast<T>` rounds, which 4,040 float16 and bfloat16 outputs need,
1,216 of them halfway cases. The lane-level model replays `load_vector`,
`qdot` and the row guard and must reach direct dequantized dot products; six
mutations, among them a store of lane 0's partial sum in place of the
`quad_sum` and rows that ignore the quad's offset, are rejected where they are
defects. Every word is compared exactly, guards and inputs must be unchanged,
and Metal also runs the unchanged upstream kernels, which must produce the
same words. This covers 216/17,832 entries; it does not run the upstream MLX
test suite or redirect the MLX host runtime.

The quantized gates now cover 1,944 of the 2,052 `quantized.metal` entries.
The 108 `affine_gather_qmm_rhs_nt` and `affine_gather_qmm_rhs_nn` entries are
not covered. Like `gather_mm_rhs`, they derive each row tile from `offsets`
with subgroup scans and return early on their reduction, which DirectX and
OpenGL can lower only once reductions of values equal in every subgroup count
as workgroup-uniform (#2546).

Checked column-reduction translation now retains the promoted value of local
integer arithmetic such as an `int` group count multiplied by a byte-sized
subgroup width. Narrow storage conversions are reflected in subsequent bounds,
rather than treating pre-conversion values as valid offsets. The unchanged
`col_reduce_2pass_large_1_32_32_reduce_sumfloat16` probe now retains private cursor
updates and the `LoopedElemToLoc::location` return range through its strided loop.
Generated pointer offsets now retain the source bounds of each use, without
reusing facts after mutation or treating byte offsets as element offsets.

`tests/kernels/test_checked_columns.py` exercises the float16 and float32
first-pass column kernels on 512 and 1,025 rows with 33 columns. Twelve cases
cover constant, signed and binary-fraction inputs, repeated cursor updates,
1,056 partial sums, 32 output guards and 11 unchanged inputs. All metadata
requirements are checked against uploaded values; no index-range assertions
are used. The existing platform jobs require native execution, with strict
target compilation and retained packages, descriptors and execution records.
Local Metal and software OpenGL runs pass these cases; Windows execution must
pass its separate CI gate.

`tests/kernels/test_checked_rows.py` exercises the unchanged
`row_reduce_looped_large_5_reduce_sumfloat32` entry at the current pin. It checks
two outputs over three padded 132-element rows per output, with constant,
signed and binary-fraction inputs. Nested cursor resets, early returns and
partial row reads retain source-owned bounds through pointer-array helpers.
Eight upload-checked metadata requirements and the checked launch geometry
replace trusted index assertions. Each profile checks 32 output guards and
nine unchanged inputs. Local Metal and software OpenGL execution match the
independent reference; Metal also matches the original upstream registration.
The existing platform jobs require these same numerical cases, including
Windows execution.

The same module also requires `row_reduce_simple_sumfloat32` on 33 contiguous
65-element rows. This layout is captured from the pinned MLX host dispatch:
nine workgroups of 32 threads, with a partial final group and no extra input
padding. Its pointer-array initialization reads each previously initialized
slot. Two checked metadata uploads establish the row length and output count;
three numerical profiles check all 33 results, 32 output guards and three
unchanged inputs. These cases share the existing checked-reduction CI batch.
This covers two float32 entries and geometries, not the complete row-reduction
family, host routing or upstream test suite.

The Metal reference wrapper selects the same registration macro and includes
the unchanged upstream implementation. It does not rewrite the kernel body.
Only the original-header compilation suppresses the C++17-extension warning
with `-Wno-c++17-extensions`; generated compilation remains warning-strict.
This proves the selected first-pass partial sums, not second-pass integration,
complete reduction-family execution or the full upstream suite.

Separate half-sum controls exposed two translator limitations: DirectX rejected
the software operation ([#2240](https://github.com/CrossGL/crosstl/issues/2240)),
and OpenGL lost intermediate half rounding
([#2241](https://github.com/CrossGL/crosstl/issues/2241)). Exact-width software
sums now round each adjacent-pair combine to binary16. The generic native gate
covers cancellation, exceptional values and partial groups beyond the column
demo's exact-value inputs. DirectX's reflected half buffers use encoded
unsigned 16-bit words; the demo packs those words without changing the source
resource layout. Native Windows results remain a required separate check.

Conditional half sums remain unsupported until the source active-lane contract
is established ([#2242](https://github.com/CrossGL/crosstl/issues/2242)). The
translator reports that limitation rather than substituting a lexical mask
that may disagree with the original Metal execution.

The following reviews describe earlier artifact revisions, before the checked
metadata identity refresh. Their Windows numerical results do not qualify the
refreshed artifacts; those require a new native Windows CI run.

The earlier OpenGL identities include explicit signed 64-bit remainder,
source-typed half loads, and bfloat rounding at shuffle and sentinel boundaries.
All 24 bodies were reviewed against the retained sources and compiled with
`glslangValidator` and `spirv-val`. The 16 float32 and 96 integer cases are
required on Mesa EGL. An additional local review executed 80 half/bfloat cases with source-rounded
inputs, including rounding ties, subnormals, signed zeros, NaNs, and infinities;
these do not expand the contract's required CI runtime subset.

The Metal identities have a separate review of narrow aggregate storage and
byte-field initialization. All 24 generated entries compile with `-Werror`.
The eight changed bodies retain native byte/short fields; byte constructors
convert promoted values back to the declared field type. Source identity,
specialization counts, workgroup metadata and the 16 other bodies are unchanged.
A local execution review compares the four byte entries against the unchanged
upstream metallib and independent integer indices across 64 cases, including
extrema, ties, constant rows, tails and strided inputs. These cases supplement,
but do not expand, the required CI runtime subset.

The four 16-bit Metal references additionally retain explicit `short`/`ushort`
conversions at aggregate construction. Removing only the two field conversions
in each entry reproduces its prior accepted source hash. The reviewed 32/64-bit
changes preserve typed extrema; recreating the preceding translator revision
reproduces every prior accepted hash. The 96 required integer cases above
validate these bodies against the unchanged upstream metallib, including
64-bit inputs above the 32-bit range. No upstream source adaptation is applied.

A subsequent unused-parameter annotation update changes all 24 Metal source
hashes without changing their compiled libraries. The review reproduces every
previous source identity and checks unchanged source, specialization and
workgroup contracts before accepting the updated references.

The DirectX identities have a separate complete-body review and strict DXC
compilation for all 24 entries. Changes preserve source-width index arithmetic,
byte conversions at aggregate and call boundaries, the negative-infinity
sentinel, and logical half loads from integer storage. Source hashes, template
materializations, resource registers, launch geometry, reduction comparisons
and tie-breaking remain unchanged. The 32-lane software subgroup path remains
required for the two float32 entries. Those identities allowed the existing
Windows numerical gate to run. Its 16 cases passed with those reviewed shaders;
an independent audit of retained uploads and outputs verifies all 32 reduction
indices, parameter bindings and executed module identities. This confirms the
existing float32 subset, not numerical coverage of all 24 scalar entries.

The four DirectX byte-entry references additionally include logical element
addressing for widened byte buffers. Their complete-body diffs replace packed
word extraction with one indexed resource read, retaining signed-byte extension.
Source hashes, template materializations, resource declarations and workgroup
metadata are unchanged. All four updated artifacts pass strict DXC compilation;
this reference review does not add native byte-entry numerical coverage.

Original Metal reference libraries use the unchanged upstream sources with
compiler warnings retained but not promoted to errors. The pinned headers use
C++17 constructs that some Metal 3.1 toolchains diagnose as extensions.
Generated Metal libraries still compile with `-Werror`; both paths retain
`-fno-fast-math` and the same numerical assertions. No upstream headers are
modified to accommodate the reference compiler.

A separate local FP8 review at `9c3d3557` compares all six scalar conversion
entries against unchanged upstream Metal: all 65,536 half and bfloat input
words, 1,024 float32 boundary inputs, and all 256 encodings for each decoder.
Generated Metal matches the original readback bytes. OpenGL matches the other
five entries, but its half encoder differs for all 1,023 negative NaNs:
the Metal reference returns `0x7e`, while OpenGL returns `0xfe`. Output guards
remain intact. Encoder comparisons are byte-exact; decoder comparisons retain
exact finite values and signed zeros but compare NaNs by classification.
An isolated half-to-float conversion followed by integer bit inspection
reproduces the source-profile gap tracked in
[crosstl#2081](https://github.com/CrossGL/crosstl/issues/2081). This is not a
lossless-copy defect or permission to canonicalize uploads. These local probes
do not establish Windows FP8 execution or expand the required CI runtime set.

The current-pin `v2_ToFP8float16uint8` package regression now selects the
explicit `binary16_widening_profile = "canonical-nan"` source policy.
All 65,536 half encodings match the unchanged original Metal kernel and an
independent FP8 rounding oracle on generated Metal and OpenGL, including the
1,023 negative NaNs above. Two output initializations, exact tail guards,
the two-dimensional launch, and the selected policy are retained in the
package evidence. No upstream code or host input values are changed.
The existing unary CI step requires this case on all three target platforms;
native Windows execution remains a separate required result.

Generic controls separately exhaust half encodings for numeric widening,
same-type copies, and integer reinterpretation. OpenGL bitcasts retain NaN
sign and payload bits through integer-based physical-carrier conversion
([#2117](https://github.com/CrossGL/crosstl/issues/2117)); numeric conversion
uses the selected profile instead. DirectX packed 16-bit pair support is
tracked in [#2201](https://github.com/CrossGL/crosstl/issues/2201). These
checks do not resolve the other source NaN-policy questions in #2081 or
establish complete FP8-family or upstream-suite coverage.

The arithmetic correction for
[crosstl#2082](https://github.com/CrossGL/crosstl/issues/2082) retains bfloat
intermediates instead of inferring half precision from their storage width.
Mixed integer operands are converted before arithmetic; each narrow result
boundary remains observable. Nineteen reduced cases match unchanged original
Metal and generated Metal/OpenGL over 228 values per path, with exact output
bits and guards. Seventeen supported HLSL cases compile with warnings fatal
and pass Windows numerical execution. The two
bfloat-vector cases retain DirectX unsupported-type diagnostics rather than
claiming execution coverage.

The math-wrapper correction for
[crosstl#2084](https://github.com/CrossGL/crosstl/issues/2084) preserves float
computation and the source's bfloat return before subsequent arithmetic.
Nine reduced default, fast and precise wrapper cases match original Metal and
generated Metal/OpenGL over 108 values per path; all nine HLSL cases compile
with warnings fatal and pass Windows numerical execution. Independent review
of the [Windows artifacts](https://github.com/CrossGL/crosstl/actions/runs/37305074470/job/111746672534)
checks all 26 arithmetic/math cases against original Metal readbacks: 312
values, 52 guard words and fresh matching shader identities. They run within
the existing native conversion jobs.
Fresh current and historical `v_Sigmoidbfloat16bfloat16` entries compile to all
three targets. Local Metal execution matches the unchanged original kernel for
all 65,282 non-NaN bfloat inputs at each pin, including infinities and signed
zeros. Inputs and guards are checked; outputs are not preloaded with expected
values. The same OpenGL sweep has four differences with intact guards:
one exponential rounding midpoint
([crosstl#2085](https://github.com/CrossGL/crosstl/issues/2085)) and three
subnormal division results
([crosstl#2086](https://github.com/CrossGL/crosstl/issues/2086)). Independent
stage and division-only probes retain these failures. This does not establish
Windows Sigmoid parity or resolve the separate NaN policy.

The portable precise-exponential lowering and explicit
`binary32_division_profile = "rne-flush"` source option resolve those four
differences at the current pin. Generated Metal and OpenGL now match the
unchanged original Metal kernel for all 65,282 non-NaN bfloat inputs, with
guards intact. An independent Decimal reference, rounded at the source's
binary32 and bfloat boundaries, also matches every original readback. The
division profile is retained in the saved project configuration, report and
runtime-artifact provenance. The numerical test checks the generated source
hash and size against the reviewed artifact for each target before execution.

The existing native project job now checks that full domain and eight trailing
guards on each platform, instead of adding a separate runner. It retains 19
explicit original-Metal boundary results, including the exponential midpoint
and division underflow neighbors. Required primitive exponential tests also
exercise 16,132 binary32 inputs, scalar/vector calls and exact narrowing of
the computed float result against a 100-digit reference. The
[Windows native job at `c7579e41`](https://github.com/CrossGL/crosstl/actions/runs/37384697268/job/112014981317)
passes all 48 unary checks without skips. Independent retained-artifact review
verifies all 65,282 Sigmoid results against original Metal, eight output guards,
the saved division profile, and generated-source/DXIL identities. The repaired
half/bfloat exponential alias cases also pass: 90 result words and 64 guards
match an independent 120-digit reference within the existing two-ULP bound,
with single evaluation preserved. The same job passes the 34 selected upstream
host tests and 1,208 translated dispatches; independent review verifies 40,872
values on each CPU and DirectX path. This is not full MLX-suite parity, a
universal device-equivalence claim or a NaN payload policy. No upstream kernel
is changed.

Mixed integer-lvalue compound assignments retain both the integer-to-bfloat
conversion and the rounded arithmetic result before integer writeback
([crosstl#2083](https://github.com/CrossGL/crosstl/issues/2083)). Forty-eight
reduced cases cover signed and unsigned 32-bit integers, aliases, local and
resource arrays, and struct members. Original/generated Metal, generated
OpenGL and Windows DirectX match over 576 values per path. The Windows artifact
audit verifies all 48 cases, 60 output buffers and 120 guard words against the
unchanged original Metal controls, with fresh generated-source identities
([native Windows run](https://github.com/CrossGL/crosstl/actions/runs/37308563739/job/111758088868)).
The tests use existing native CI jobs. Unproven wide-integer conversions and operands
with observable side effects produce structured diagnostics; these checks do
not claim complete mixed compound-assignment support.

The Metal unary references have a complete review of all 877 artifacts:
538 are unchanged and 339 retain explicit byte conversions, Boolean promotions,
bfloat computation boundaries or the existing precise math implementations.
All 877 compile with warnings fatal. Their 3,363 reflected resources and 1,243
materializations are unchanged. The separate scalar contract and ArcCos loader
identity match the same reviewed artifacts; no source selection or tolerance
was changed.

A subsequent complete review recompiles all 877 retained outputs with warnings
fatal. Fifteen exponential entries now use the already reviewed precise-exp
helper with source expression grouping preserved; the other 862 are byte-for-byte
unchanged. All host interfaces and materialization counts remain unchanged.
The three affected scalar-subset identities match the complete family. This
reference update does not expand numerical tolerances or claim full-suite parity.

A local original/generated Metal comparison covers 69 changed operator/type
selections and 1,863,009 values per path. Fifty-three selections match exactly;
sixteen have floating-point bit differences. Sampled scalar float32 results
satisfy their existing four-ULP bounds. Required native precision checks also
verify exact narrowing of each computed float result near half/bfloat rounding
midpoints. They retain the original Metal controls separately, without claiming
bitwise transcendental parity. Complex differences and wider source/target
accuracy policy remain tracked in
[crosstl#2068](https://github.com/CrossGL/crosstl/issues/2068).
Whole-family DirectX/OpenGL reference reconciliation remains incomplete.
Required Square and ArcCos numerical cases remain separate and unchanged.

A separate local binary-math comparison at `9c3d3557` covers all 65,536 half
or bfloat input payloads against 12 fixed partners: signed zero, positive and
negative 0.5, 1, 2, 3, and infinity. Each selected `vv_` entry executes 786,432
pairs through unchanged original Metal, generated Metal and generated OpenGL.
The Metal round trip matches every comparison; OpenGL results are:

| Entry | Differing results | Output guards checked |
| --- | ---: | ---: |
| `vv_LogAddExpfloat16` | 1,123 | 112 |
| `vv_LogAddExpbfloat16` | 746 | 112 |
| `vv_Remainderfloat16` | 0 | 112 |
| `vv_Remainderbfloat16` | 271,292 | 112 |

Comparisons preserve finite bits, infinities and signed zeros; NaNs are compared
by classification, not payload. The remainder count includes signed-zero and
special-value differences, including 512 zero-to-infinity results. A reduced
float kernel confirms that translating `fmod` to GLSL `mod` loses source
semantics ([crosstl#2097](https://github.com/CrossGL/crosstl/issues/2097)).
LogAddExp also loses some nonzero results; its accuracy investigation remains
under [crosstl#2068](https://github.com/CrossGL/crosstl/issues/2068). A reduced
compensated-logarithm expression isolates lost binary32 rounding dependencies
in GLSL ([crosstl#2098](https://github.com/CrossGL/crosstl/issues/2098)).
The nested half/integer arithmetic gap in
[crosstl#1992](https://github.com/CrossGL/crosstl/issues/1992#issuecomment-6006266688)
now has source-width operand conversions and intermediate rounding in GLSL.
The generic regression executes all 65,536 half input encodings through nested
division, vector arithmetic, comparisons and conditional selection. Original
and generated Metal and Linux OpenGL each match 917,504 reference values,
including single-evaluation and unselected-branch controls; all 96 output
guard words pass. Metal uses `-fno-fast-math` for this rounding contract, and
NaNs are compared by classification, not payload. The checks run within the
existing native jobs without adding runners. A separate execution of the
unchanged historical `v_Sigmoidfloat16float16` entry at `846d1762` matches
original/generated Metal across all 63,490 non-NaN half inputs. Its generated
OpenGL still differs on four inputs; all guards pass. These remaining numerical
differences stay under #2068, and the historical GLSL reference is not accepted.
An independent HLSL conditional narrowing case fails strict DXC compilation
and remains tracked in [crosstl#2099](https://github.com/CrossGL/crosstl/issues/2099);
the GLSL correction does not establish support for that DirectX case.
All four binary-math
HLSL artifacts compile with strict DXC options and all four GLSL artifacts pass
glslangValidator and SPIR-V validation. These compiler results do not resolve
the numerical differences. The four-entry binary-math review adds no required CI case, changes
no reference identity or tolerance, and establishes no Windows numerical result.

The alias-bitcast defect in
[crosstl#2088](https://github.com/CrossGL/crosstl/issues/2088) is corrected by
resolving logical bfloat aliases before OpenGL bitcast lowering and resolving
elided local/namespace aliases in Metal bitcast type arguments. Generic translator
tests exercise every 16-bit payload through six paths, including qualified
aliases, helper returns, nested bitcasts and an operand evaluated exactly once.
Original/generated Metal and OpenGL execution preserve all 65,536 payloads,
including NaNs, without numerical tolerances. Separate exact widening controls
cover namespace aliases that conflict with an outer float alias. The required
[Windows checks](https://github.com/CrossGL/crosstl/actions/runs/37321634266/job/111801894608)
also pass: retained DXIL identities and readbacks verify the same exact alias
payloads and the precise-result narrowing controls above. These tests reuse
the existing native jobs. Precision tests retain explicit
input bitcast types to isolate arithmetic from alias resolution.

The complete unary Metal compile gate generates sources in five Ubuntu shards
and compiles all 877 verified artifacts in one macOS job. It retains every
entry, template-materialization and resource-interface check. Source reports
and JUnit results survive failed reference checks; missing, duplicate or
changed artifacts cannot pass the native consumer. This saves four macOS
runner starts without replacing any numerical execution gate.

The current pin adds cross-entropy, gated-delta forward and backward kernels
(including NAX variants), matrix-multiplication gather offsets, and attention
backward kernels. These files are included in discovery; they are not covered by
the arg-reduce translation and numerical checks above.

A separate native gate exercises all 15 discovered `Powercomplex64` entry
shapes at the same `9c3d3557` pin. Its 873 complex outputs cover scalar/vector
broadcasting, multidimensional grids, non-contiguous inputs, zero strides,
32/64-bit index variants, and partial final tiles in four-dimensional gathers.
Every shape retains its random cases and tests both signed-zero sides of the
negative-real branch cut with a half exponent, using the same error bound.
Inputs and readbacks are retained separately for all three datasets.
DirectX and OpenGL use the public runtime-package and native-loader APIs.
Metal executes the original source, generated kernels and public runtime package
for every dataset, retaining 873 comparisons on each path. Each output is
checked against an independently indexed CPU reference; required CI retains
the compiler logs, packages, bindings and numerical results. The OpenGL
index-range assertions are bounded fixture preconditions, not general runtime
bounds checks. Other binary operators and types, the complete upstream suite,
and MLX host-runtime redirection remain outside this proof.

The DirectX/OpenGL two- and three-dimensional cases also verify that truncated
stride buffers are rejected before dispatch. Their minimum lengths come from
proven constant-index accesses in the generated helpers, not from MLX-specific
buffer names. Valid workloads still run unchanged; this preflight guarantee does
not cover arbitrary dynamic indexing.

The existing native-host jobs also run real-valued power through public runtime
packages. Identity checks retain all 65,536 half and bfloat storage patterns and
5,632 binary32 inputs. A separate batch selects the explicit binary32 power
operand profile for 82,848 float32/bfloat operand pairs. Every non-NaN result,
signed zero and guard is checked exactly; NaNs use classification checks.
Those five cases retain 219,552 outputs and 40 guards per target. A third batch
selects `binary32_power_accuracy_profile = "portable-finite"` for 4,982 float32
pairs, including near-one bases with large exponents and seeded general inputs.
It compares normal finite results against a high-precision decimal reference
within 16 representable steps; guards remain exact. All six cases retain
unchanged original-source readbacks on Metal. Two-thread workgroups keep the complete
input sets within DirectX dispatch limits without adding inactive invocations.
No upstream source is modified. This covers the vector/vector entries' identity,
operand-domain behavior and selected finite accuracy, not result underflow,
all real-power layouts or full MLX host integration. The three-minute step uses
the existing native runners and preserves every earlier check and deadline.

## Scope

The general-gather jobs in [Project Demo Testing](../../../.github/workflows/demo-project-testing.yml)
uses the unchanged JIT template and indexing headers at
`9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8`. It executes 24 original/generated
Metal workloads through public project, package and native-dispatch APIs.
Coverage includes zero through two index buffers, scalar through three-dimensional
indices, negative and repeated indices, and dense, transposed, strided and
broadcast source/index layouts. Binary32 outputs are compared as exact storage
words, including trailing guards. No upstream source or test is modified.

The no-index specialization requires a valid zero-extent standard array.
CrossTL retains that native object and its layout instead of emitting an illegal
C-style array. Separate native controls exercise aliases, nested arrays,
packed/narrow elements, copies and neighboring fields. DirectX/OpenGL empty-array
representation remains unsupported ([#2042](https://github.com/CrossGL/crosstl/issues/2042)).
Private pointer-bearing aggregates use resource identities and signed offsets
on DirectX and OpenGL, with concrete buffer specialization on OpenGL
([#1544](https://github.com/CrossGL/crosstl/issues/1544)). Required Windows/Linux
gates execute the 20 indexed workloads. OpenGL uses explicit allocation-derived
index-range assertions; these bounded cases do not establish unrestricted
64-bit addressing. Returned void helper calls retain their computation and
argument effects ([#2043](https://github.com/CrossGL/crosstl/issues/2043)).
These are direct kernel proofs, not integration of general Gather into the MLX
host adapter or a passing complete upstream suite.

The same workflow requires pointer-offset execution controls: 13 cases on Metal
and 11 on DirectX/OpenGL, with exact outputs and neighboring guards. Metal also
executes the original source. Offset values, including struct members and
conditional scalar loads, do not change a pointer's address space. Both operand
orders preserve resource-backed aggregate pointers without repeating offset
side effects. Incompatible Metal helper arguments fail project translation
instead of replacing a reachable computation with zero
([#2053](https://github.com/CrossGL/crosstl/issues/2053)). DirectX/OpenGL still
reject same-buffer conditional pointers
([#2054](https://github.com/CrossGL/crosstl/issues/2054)) and local struct-pointer
aliases ([#2055](https://github.com/CrossGL/crosstl/issues/2055)); explicit negative
controls retain those diagnostics. These cases do not count as native passes.
Generated resource helpers use explicit 64-bit index arguments; native OpenGL
execution checks this contract independently of glslang acceptance
([#2056](https://github.com/CrossGL/crosstl/issues/2056)).

The [Metal host integration harness](METAL_HOST.md) redirects selected current-pinned
complex-power kernels through MLX's own runtime and runs its unchanged upstream
operations tests. Its dispatch trace distinguishes actual host execution from
compile-only coverage. Eight host layouts retain 402 complex readbacks across
ordinary inputs and both signed-zero branch cuts; unselected operations still
use upstream kernels.

The [portable host adapter](portable_host/README.md) connects MLX's C++ GPU
evaluation to native DirectX/OpenGL/generated Metal packages with the original
Metal and CUDA backends disabled. It covers five typed array-creation entries,
30 float32 unary entries, shared views, bit-exact 32-bit layout copies and 16
binary arithmetic entries, with 21 unchanged upstream tests and explicit
unsupported-operation failures.
This synchronous adapter does not yet
implement a complete MLX backend.

The current harness verifies:

- the [pinned native MLX Metal reference baseline](NATIVE_METAL.md) for exact
  upstream commit `4367c73b60541ddd5a266ce4644fd93d20223b6e`. On the Arm64
  `macos-26` runner, it compiles all 40 native Metal units (33 Metal 3.2 / 7
  Metal 4.0), runs
  776 Python tests with 44 skips, and passes 260/260 C++ cases and 3,490/3,490
  assertions. This native upstream reference does not establish
  translated-target correctness, runtime parity, or numerical parity;
- discovery of the MLX Metal kernel project surface under
  `mlx/backend/metal/kernels`;
- Metal-to-CrossGL-to-Metal translation of pinned `fence.metal`, including
  project artifact hashes, sizes, source maps, provenance, and native Metal
  compilation on macOS CI. The gate requires all three device-memory,
  sequentially consistent, system-scope atomic fences to survive the round
  trip without a weaker barrier fallback. Resource coherence and volatility
  preservation remain blocked by
  [#1660](https://github.com/CrossGL/crosstl/issues/1660), so this is not yet a
  complete semantic-equivalence claim;
- selected-entry Metal-to-CrossGL-to-Metal translation of all 877 discovered
  entries in ``unary.metal`` at the legacy reference revision
  `846d176227a0ac13d2667e58d2bb68b322109ab0`: 183 each for ``v_``, ``v2_``,
  ``gn1_``, and ``gn4large_``, plus 145 ``vn_`` entries. The exact contract
  spans 37 operators and 20 input/output type pairs across bfloat16, Boolean,
  complex64, float16, float32, signed and unsigned integers, FP8 encode/decode,
  and complex projection. Reachability pruning keeps one selected operator and
  kernel, materializes the required gather index helper only for ``gn*``, and
  reflects either three vector resources or five gather resources. Source
  ``constant`` shape/stride provenance, a read-only ``device`` dimension
  reference, and postfix output indexing survive the round trip. A host-owned
  ``[1, 1, 1]`` dispatch contract is preserved, and macOS CI compiles every
  deterministic artifact to non-empty AIR. This is complete discovered unary
  Metal compiler/reflection coverage, not Metal numerical execution. The reviewed
  `v_Squarefloat32float32` reference includes an unused-parameter annotation;
  removing only that annotation reproduces its prior accepted source hash, and
  both versions compile to byte-identical Metal libraries with warnings fatal
  and fast math disabled. This reference update does not approve unrelated
  pending corpus changes or establish current-pin runtime parity.
  A subsequent full inventory checks all 877 current artifacts, recovering 125
  missing CI results through fresh public project translation. For 619 entries,
  accepted sources reconstruct to their recorded hashes and byte counts with
  unchanged reflected interfaces and materializations. All 1,238 old/new sources
  pass warnings-fatal Metal compilation and link to byte-identical library
  pairs. This updates 617 annotation-only identities, including 127 scalar
  entries; two already match their references. Removing only unused-parameter
  annotations makes each changed pair byte-identical. A separate arithmetic
  review accepts eight complex Abs/Sign references across `v`, `v2`, `gn1`,
  and `gn4large`, including two scalar entries. Their accepted sources
  reconstruct exactly, all 16 old/new sources compile with warnings fatal and
  fast math disabled, and fresh historical translations preserve interfaces
  and materializations. Original MLX, accepted output and updated output agree
  on 18,100 complex values per path and initialization, with two finite output
  initializations, 256 guard words per path and unchanged read-only buffers.
  Non-NaN words match exactly; NaN classification is compared without requiring
  payload identity. A further 38 Sqrt/Rsqrt references, including eight scalar
  entries, explicitly select `flush-subnormals` for binary32 square root and
  reciprocal square root. The policy is recorded in the contracts and project
  provenance, and applies only to these Metal demo selections; translator
  defaults and other target references are unchanged. Accepted predecessors
  reconstruct exactly and all 76 old/new sources compile with warnings fatal.
  Original, accepted and freshly translated historical sources agree on
  1,199,068 values per path and initialization across the five layouts, with
  two output initializations, 1,216 guard words per path and unchanged read-only
  buffers. Twelve contiguous narrow cases cover every 16-bit storage pattern.
  Separate current-pin Metal checks cover 402,076 values per path and
  initialization across eight entries, including four exhaustive narrow cases
  and 256 guards per path. Non-NaN words, including signed zero, match exactly.
  Different output initializations detect unwritten results without excluding
  legitimate values from the exhaustive domains.
  A further 15 `ToFP8` references, including three scalar entries, change only
  the sign mask's unsigned literal suffix and an unused-parameter annotation.
  Fresh translation with the original translator reproduces every accepted
  source. Accepted/current historical sources and separate current-pin sources
  pass 45 warnings-fatal compile/link checks and produce byte-identical Metal
  libraries for each entry. Re-reflection with the same implementation preserves
  all resource layouts; historical materializations are unchanged. The current
  pin has 878 additional pruned candidates, not additional selected entries.
  Bounded raw-buffer checks at each pin cover 592,758 outputs per path and
  initialization across all five layouts, two output initializations, six
  exhaustive 16-bit cases, 960 guard bytes per path and unchanged read-only
  buffers. Original Metal and translated output match exactly. A separate
  65,536-pattern source-device control establishes half-to-float NaN
  canonicalization before FP8 saturation; that conversion is accounted for
  explicitly in the independent rounding oracle, not treated as a tolerance.
  These are Metal results, not DirectX/OpenGL numerical qualification or a
  complete upstream-suite result.
  A further 30 Expm1/Erf references, including six scalar entries, retain
  source-owned power semantics in their shared exponential helper. Their exact
  accepted sources are reproduced, all 60 old/new sources pass strict Metal
  compile/link checks, and fresh historical/current translations preserve the
  selected interfaces and materializations. The compiled libraries differ;
  compiler success alone is not the basis for accepting these references.
  Independent readback audits compare original, accepted and updated historical
  output, plus original and generated current-pin output. At each pin they cover
  1,186,152 values per path and initialization across five layouts and three
  storage types, with two output initializations, 64 guard words per entry and
  unchanged read-only buffers. Twelve contiguous cases exhaust all 65,536
  half/bfloat16 storage patterns. Finite values, infinities and signed zeros
  match exactly; NaNs are compared by classification, not payload.
  Historical `vn` evidence used a wildcard workgroup rule; fresh runs select
  the exact entry. Both resolve to `[1, 1, 1]` with identical specialization
  parameters, and their different rule identities are verified separately.
  Existing per-entry corpus CI retains the updated source identities and strict
  compiler checks. These bounded Metal experiments do not establish exhaustive
  float32 accuracy or native DirectX/OpenGL parity for all 30 entries.
  Fifteen ArcCosh references, including three scalar entries, now retain
  function-scoped precision directives and the unused-parameter annotation.
  Every recorded predecessor is reproduced byte-for-byte. Accepted and updated
  sources compile to identical Metal libraries with matching interfaces;
  fresh historical and current-pin translations preserve the selected entries
  and parameters. Native audits at each pin cover 617,565 output values per
  implementation and initialization, all five layouts, two initializations,
  unchanged read-only buffers and 64 guard words per entry. Six contiguous
  cases exhaust the half/bfloat16 storage domains. Accepted and updated output
  agree exactly apart from NaN payloads; narrow results also match original
  Metal under that comparison. Float32 is not bit-identical to original Metal:
  both remain within two ULP of a 90-digit decimal reference on these inputs,
  inside the existing four-ULP bound. That bound and the native gates are
  unchanged; this review does not establish exhaustive float32 or cross-target
  numerical parity.
  Eight complex Cosh/Sinh references, including two scalar entries, have a
  separate source and native review. Reversing only precision-directive scope
  and unused-parameter annotations reproduces every accepted source exactly.
  All old/new sources compile with warnings fatal and fast math disabled;
  their Metal libraries differ, so compilation alone does not establish
  equivalence. Fresh translations at both pins preserve resources and selected
  materializations. The retained scalar Sinh run used a wildcard workgroup
  rule; fresh runs select the exact entry, with the same `[1, 1, 1]` geometry
  and independently verified rule identities.
  Native readback audits at each pin cover 95,460 complex outputs per path and
  initialization across `v`, `v2`, `gn1` and `gn4large`, two output
  initializations, 512 guard words per path and unchanged read-only buffers.
  Inputs include signed zeros, subnormals, infinities, NaNs, overflow neighbors
  and seeded raw and finite-range samples. Accepted and updated outputs match
  exactly apart from NaN payloads. Both differ from original Metal in 23,258
  finite component words per initialization, by at most four ULP on these
  inputs; zero signs and non-finite classifications agree. This is a measured
  difference, not a new tolerance or a claim of bit-identical original output.
  No numerical gate, source pin or whole-family parity status is changed.
  Four complex Log1p references have a separate original-source audit. Each
  previously accepted source is reconstructed by its recorded hash and size;
  old and updated sources compile with warnings fatal and fast math disabled.
  Their libraries differ, so native execution is checked separately. At both
  the historical and current pins, the audit covers 50,610 complex outputs per
  path and initialization across all four layouts, two output initializations,
  256 guard words per path and unchanged input and constant buffers. Inputs
  include signed zeros, subnormals, infinities, NaNs, boundary neighbors and
  seeded raw and finite-range samples. Original, accepted and updated outputs
  match word-for-word apart from NaN payloads. This bounded result does not
  establish exhaustive complex Log1p accuracy or qualify another target.
  Eight complex Log2/Log10 references record the explicit
  `binary32_atan2_profile = "flush-subnormals"` option in
  `sourceOptionsByEntryPoint`. For Log2/Log10 it applies only to complex Metal
  corpus entries; translator defaults, their real-valued entries and other
  target contracts are unchanged. At both pins, audits cover 101,220 complex
  outputs per path and initialization, two initializations, 512 guard words
  per path and unchanged
  input/constants. The profile removes the observed sign and classification
  mismatches. Sampled finite component differences from original Metal reach
  two float32 ULPs at the current pin and three at the historical pin. They
  are not bit-identical results or a new ULP tolerance.
  A current-pin regression translates these entries and compares their
  native readbacks using the pinned upstream logarithm tests' existing
  `numpy.allclose` tolerances (`rtol=1e-5`, `atol=1e-8`). It additionally
  requires exact zero signs, matching non-finite classifications, unchanged
  read-only buffers, intact guards and independence from output initialization.
  The test verifies the upstream tolerance declarations from `python/tests`,
  now included in the current-tree CI checkout. It runs in the existing
  five-minute roots/logarithms step without additional jobs, retaining
  reports, modules and raw readbacks. The precise log and atan2 helpers retain
  their separate independent high-precision oracle checks and unchanged bounds.
  Nineteen natural-log references additionally select
  `binary32_log_profile = "flush-subnormals"` through `sourceOptionsByOperator`.
  Their four complex entries also select the explicit atan2 policy above.
  All 19 accepted predecessors reconstruct to their recorded identities and
  compile with warnings fatal; fresh default/profiled sources compile at both
  pins with unchanged reflected interfaces and source-owned indexing.
  Independent native audits cover 694,296 output words per implementation and
  initialization at each pin, across five layouts, two initializations and
  1,216 guard words per implementation. Contiguous half/bfloat cases cover all
  65,536 storage patterns. Profiled outputs preserve zero signs and non-finite
  classifications; half/bfloat outputs match original Metal exactly apart from
  NaN payloads. Float32 and complex components satisfy the existing upstream
  tolerance, with sampled differences of at most one and two float32 ULPs,
  respectively. Three half inputs previously differed by one half ULP in the
  accepted artifacts; the updated artifacts match original Metal.
  The required current-pin regression adds all four complex Log layouts and
  contiguous real Log checks, including both complete 16-bit input domains.
  It remains in the existing five-minute step, with the original-source
  comparison limited to Metal. Other target contracts are not updated from
  this evidence, and complete unary-family numerical parity remains unproven.
  A separate required test runs current-pin `v_Logfloat32float32` through
  project translation, runtime packaging and the native loader on each target.
  Its 28,024 inputs cover every binary32 exponent, signed pairs, special values
  and 8,193 consecutive values around one. It checks a 120-digit Decimal
  logarithm oracle with the existing four-ULP bound, exact zero/infinity signs
  and NaN classification. The report, artifact manifest and runtime package
  must retain the explicit logarithm policy; the executed source identity must
  match the generated artifact. Oracle words and native execution receipts are
  retained with the test evidence. Local Metal and Linux software-OpenGL runs
  pass; Windows execution is required in CI and is not inferred from these
  results. This value-based package test does not preserve NaN payloads or
  retain input-device readbacks or guard regions. It supplements, rather than
  replaces, the guarded original-Metal comparisons above and does not establish
  complete host routing or upstream-suite parity.
  Nineteen ArcTan references have a separate two-pin native audit. The accepted
  sources reproduce exactly with their historical translator. All 15 real-valued
  entries retain byte-identical compiled libraries and matching non-NaN output
  words; their source changes only precision-pragma scope and unused-parameter
  annotations. The four complex entries instead use the updated square-root,
  logarithm and atan2 helpers, so their changed computations are checked by
  execution rather than compiler identity alone.
  The audit reopens 190 dispatches across five layouts, two initializations,
  694,296 output words per path and initialization at each pin, 1,216 guard words
  per path and unchanged read-only buffers. Complete half/bfloat storage domains
  are covered in the contiguous layouts. Real-valued differences from original
  Metal are unchanged from the accepted artifacts: original Metal can return
  positive zero for negative zero or flush binary32 subnormals, while the
  generated precise helper preserves those bits. Other sampled float32 results
  differ by at most one ULP. Complex components retain non-finite classification
  and zero signs and satisfy the existing upstream `numpy.allclose` tolerance,
  with sampled differences of at most two ULPs at both pins. The independent
  precise-atan oracle and current-pin native-package tests retain their existing
  bounds. No source policy, tolerance or whole-family parity status is changed.
  A further 29 real ArcSin/ArcCos references, including five scalar entries,
  change only precision-pragma scope and unused-parameter annotations.
  All accepted sources reproduce exactly; old and updated sources compile to
  byte-identical Metal libraries at both pins. Independent audits retain
  1,119,742 output words per path and initialization, two initializations,
  1,856 guard words per path and unchanged read-only buffers. Contiguous narrow
  entries exhaust all 65,536 storage patterns. Accepted and updated results
  agree exactly apart from NaN payloads. They are not uniformly bit-identical
  to original Metal: three half ArcCos inputs differ by one half ULP, and
  sampled float32 ArcCos/ArcSin results differ by at most two/three ULPs.
  These pre-existing differences remain within the pinned upstream dtype
  tolerances; no reference update changes the computations or numerical gates.
  Eight complex ArcSin/ArcCos references select the explicit
  `binary32_atan2_profile = "flush-subnormals"` policy. Default helpers preserve
  some tiny results that original Metal flushes; translator defaults and other
  target contracts remain unchanged. Independent two-pin audits reopen 64
  original/generated dispatches, covering 101,220 complex outputs per path and
  initialization, two initializations, 512 guard words per path, and unchanged
  read-only buffers. Zero signs and non-finite classifications match exactly;
  sampled finite component differences reach two ULPs within the existing
  upstream `rtol=1e-5`, `atol=1e-8` comparison. This is not bit-identical output
  or an exhaustive accuracy claim. A required current-pin regression repeats
  all eight layouts with guards and raw readbacks in a separate 180-second
  step in the existing macOS job. It checks the pinned upstream comparison
  declarations and preserves the existing unary step's cases and deadline.
  Roots and logarithms have their own five-minute step after the growing
  combined batch exceeded its limit on a hosted runner. No native case or
  required flag is removed; the original arithmetic step still has its
  five-minute limit, and each step retains its own failure evidence.
  Required current-pin float32 ArcSin and ArcCos package tests now run in the
  existing three-platform unary step. Each executes 4,394 inputs, including a
  dense domain sweep, endpoint neighbors, small normal values, signed zeros,
  infinities and NaNs, against host binary64 functions rounded to binary32.
  They enforce a four-ULP bound, exact zero signs and invalid-domain NaNs, and
  verify source identity across translation, packaging and native execution.
  Local Metal and Linux software-OpenGL pass; Windows remains a required CI
  result. These value-based tests do not retain input-device readbacks, guards
  or NaN payloads and do not establish full backend or upstream-suite parity.
  The final 57 Cos, Sin and Exp references have also been reviewed. Their
  complete source changes are precision-pragma scope, unused receiver
  annotations, and an explicit unsigned sign-bit mask in complex Exp. The
  mask retains the same bit-31 condition for every 32-bit input. All 45 real
  accepted/current libraries are byte-identical. Eight complex Cos/Sin
  libraries differ, so their acceptance relies on retained native execution,
  not binary identity. Across 570 dispatches at both pins, every accepted and
  updated non-NaN output word agrees. Each pin covers 2,082,888 output words per
  path and initialization, two initializations, 3,648 guards per path, and
  unchanged read-only buffers. All 16-bit contiguous input patterns are covered.
  Differences from original Metal remain explicit: some half/bfloat trigonometry
  results differ by one storage ULP, sampled float32/complex trigonometry by up
  to three/four ULPs, and real Exp retains subnormal results original Metal
  flushes. These predate the reference update; no numerical expectation or
  policy is changed to accept them. The unary reference review is complete,
  but whole-family source parity and other Metal corpus families remain open
  under [#1966](https://github.com/CrossGL/crosstl/issues/1966).
  The historical pin and complete entry selection are preserved.
  At current pin `9c3d3557`, Abs and Sign use upstream's scaled `hypot`
  implementation rather than the historical sum-of-squares formula. A separate
  fresh-translation regression checks those same four layouts against the
  current original Metal source over the same bounded inputs and guards.
  It runs in the existing macOS unary CI step without additional runners or a
  longer deadline, and retains compiler records and full input/output readbacks.
  Current-pin float32 Sqrt/Rsqrt also use the public package and native-loader
  path against an independent decimal oracle, with exact non-NaN word checks.
  These cases are required on all three existing platform jobs; the
  original-source raw-buffer comparison is Metal-only.
  These are bounded Metal results, not DirectX/OpenGL qualification, complete
  unary-family numerical parity or full MLX host integration;
- selected-entry translation of all 877 discovered `unary.metal` entries
  at the same legacy reference revision to OpenGL. The schema-v2
  `contracts/unary.opengl-translation.json` contract pins every standalone
  `main` artifact across the same five shapes, 37 operators, 20 type pairs,
  1,243 materializations, and 3,363 reflected resources, totaling 5,316,552
  generated GLSL bytes. The references incorporate reviewed scalar/vector
  conversions and numeric helpers; every complete shader body and host
  interface is compared before accepting an identity change. The
  references now include all 134 remaining unary math and FP8 changes; the
  other 743 sources are unchanged. A later refresh re-pins 113 artifacts:
  Metal's `precise::` `sinh`, `tanh`, `asinh` and `atanh` of float16 and
  float32 now lower through helpers that give the result the operand's sign
  and pass NaN through, `fma` through a `precise` result and `cexpf.h`'s
  `x - x` through a `precise` difference, since Mesa otherwise loses the sign
  of zero, turns `tanh(NaN)` into -1, splits `fma` and folds `x - x` to zero.
  A further refresh re-pins 68 artifacts for Mesa 25.2's IEEE folds: bfloat
  `precise::` `sinh`, `tanh`, `asinh` and `atanh` keep their helpers, `log1p`'s
  `1 + x` locals that feed an equality test are `precise`, so Mesa no longer
  rewrites `(1 + x) == 1` as `x == 0`, complex `log1p` calls a canonical
  `atan2` that keeps the sign of a zero `y`, and MLX's free complex operators
  settle zero operands with integer tests before a `precise` IEEE operation,
  because Mesa folds `0 * x` to +0 and `x + 0` to `x` even under `precise`.
  The five uint32 `Sign` artifacts change too: `Sign` has a non-template
  `uint32_t` overload, so they also carry its `complex64_t` overload and
  those operators.
  Every current artifact compiles and passes
  SPIR-V validation. These math corrections are not predecessor-equivalence
  claims, and whole-family numerical parity remains unproven. Translation
  requires three explicit host/runtime
  index-range preconditions for `offset + i`, `out_idx++`, and `idx`; these are
  portability promises rather than inferred or runtime-enforced bounds. The
  lowering maps Metal `log10` through one-evaluation GLSL `log2`, preserves user
  overloads, and exposes the read-only `device const int& ndim` as a storage
  buffer whose scalar expression aliases element zero. Five required Linux CI
  shards compile every artifact for OpenGL/SPIR-V 1.3 with
  `glslangValidator` and validate every module with `spirv-val`. The gate
  requires 877 non-empty SPIR-V 1.3 modules. This is complete OpenGL
  translation, reflection, and native compiler coverage, not numerical
  execution or MLX host runtime redirection;
- selected-entry translation of all 877 discovered `unary.metal` entries at the
  legacy reference revision `846d176227a0ac13d2667e58d2bb68b322109ab0` to DirectX.
  The schema-v2
  `contracts/unary.directx-translation.json` contract pins every standalone
  `CSMain` artifact across the same five shapes, 37 operators, 20 type pairs,
  and 1,243 materializations, with 3,912 reflected HLSL resources: three per
  `v_` and `vn_` artifact, four per `v2_` artifact, and six per gather artifact.
  Work-per-thread and gather forms include the generated `CrossGLDispatchInfo`
  dispatch cbuffer at bindings `b3` and `b0`, respectively. DirectX bfloat
  lowering covers the
  complete unary intrinsic family, while `acosh`, `asinh`, and `atanh` decode
  through portable float helpers before exact bfloat reconstruction. Explicit
  contextual casts preserve float-to-native-16 return and initializer narrowing
  without DXC `-Wconversion` diagnostics under warnings-fatal compilation. The
  read-only source `device const int& ndim` becomes a `StructuredBuffer<int>`
  scalar view at element zero, and source `out_idx++` remains postfix in HLSL.
  The same three explicit host/runtime index-range preconditions are required.
  Five required Ubuntu CI shards compile every artifact with pinned Linux DXC,
  source-derived `-enable-16bit-types`, warnings fatal, profile `cs_6_2`, and
  entry point `CSMain`; the gate requires 877 non-empty DXIL modules. Together
  with the OpenGL proof this records complete discovered unary translation,
  reflection, and native compiler coverage at the recorded revisions, not
  numerical execution or MLX host runtime redirection. The earlier reference
  update compiled all 877 HLSL outputs and compared their complete
  bodies against hash-verified originals: 603 changed and 274 unchanged. The
  differences are accounted for by source-width arithmetic, byte and half
  conversions, Boolean promotion, sign preservation and precise math helpers.
  Checks with deliberately altered bindings, index steps, conversion widths
  and helper coefficients fail. A subsequent review recompiles all 34 Exp and
  Sigmoid entries: 15 scalar Exp and five bfloat Sigmoid bodies change, while
  the other 14 bodies are unchanged. The Exp changes preserve the source's
  precise exponential call and half/bfloat return boundaries; the Sigmoid
  changes preserve bfloat arithmetic instead of inferring a half temporary.
  Their complete body review rejects altered bindings, index steps, rounding
  constants and temporary types. A further review updates only the five
  unsigned 32-bit Sign layouts: the source changes retain a precise square-root
  helper in an unused complex overload, while warning-fatal DXC compilation
  produces identical binaries before and after the change. Current-pin Windows
  execution separately checks all five layouts, 3,303 output values and 40
  output guards through the native loader. The subsequent complete review
  accounts for 875 changed references and two unchanged sources, totaling
  5,159,617 bytes. Of the changed artifacts, 756 compile to byte-identical DXIL;
  the remaining 119 incorporate reviewed precise math or conversion corrections.
  A follow-up review updates 230 bodies for explicit precise helper returns
  and FP8 record construction, bringing the source total to 5,264,433 bytes.
  Of those bodies, 35 retain identical DXIL and 195 preserve additional
  precision constraints. A further review updates the 46 Sin, Cos and complex
  Sinh/Cosh bodies whose Payne-Hanek reduction converts unsigned 32-bit words;
  they now call one correctly rounded unsigned-to-float helper, bringing the
  source total to 5,303,901 bytes. Removing the helper and restoring the
  `float(...)` constructors reconstructs all 46 accepted sources; old and new
  bodies compile with warnings fatal and their DXIL differs. An exhaustive model
  of the helper matches correctly rounded conversion for all 2^32 inputs.
  A later review updates 91 bodies, bringing the source total to 5,335,152
  bytes: 30 float16, float32 and bfloat `precise::` Sinh and Tanh bodies call
  helpers that keep the operand's sign at zero, pass NaN through and saturate
  Tanh at the binary32 rounding threshold (WARP returns +0 for `sinh(-0)` and
  NaN for `tanh(x)` once `exp(2|x|)` overflows), complex Exp-based bodies
  keep `cexpf.h`'s `y - y` as an exact difference, and complex bodies return
  MLX's free complex operators through `precise` results, so DXC no longer
  folds the zero components of complex constants.
  The latest review updates the five float16 Sinh bodies, bringing the source
  total to 5,340,547 bytes: Metal rounds a binary32 `sinh` beyond the binary16
  range to infinity, while WARP's native `float16_t` conversion clamps it to
  65504, so these bodies narrow through the round-to-nearest-even binary16
  helper. Tanh cannot exceed one, so its narrow bodies are unchanged.
  All 877 current artifacts compile with warnings fatal.
  Source pins, classifications, materializations, resource bindings and dispatch
  contracts remain unchanged. The required Windows Square and ArcCos numerical tests retain
  their existing inputs, expected values and tolerances; the ArcCos source
  identity now includes the explicit precise returns. Windows execution remains
  a required check, separate from this source and compiler review. Other
  historical HLSL families remain under review in
  [crosstl#2071](https://github.com/CrossGL/crosstl/issues/2071). Whole-family
  numerical parity is not claimed; bit-observable floating conversions also
  require the source-policy work tracked in
  [crosstl#2081](https://github.com/CrossGL/crosstl/issues/2081). The historical
  Sigmoid source uses the default exponential, unlike the current pin's
  precise call. This contract retains that distinction and does not enable
  the optional binary32 division profile. The separately configured current-pin
  full-domain Sigmoid test does not establish historical whole-family parity;
- selected-entry Metal-to-CrossGL-to-Metal translation of all 2,496
  discovered copy entries from `copy.metal` at the legacy reference revision
  `846d176227a0ac13d2667e58d2bb68b322109ab0`. The schema-v2
  `contracts/copy.metal-roundtrip.json` contract spans 30 shapes, 16 concrete
  templates, 13 source types, all 169 conversion pairs, 6,566 exact
  materializations, and 8,684 reflected resources. It records the conditional
  nested specialization on exactly 14 complex-to-Boolean entries, preserves
  explicit float and bfloat16 bit-pattern semantics, and validates registered
  complex representation shape before scalar projection. Twenty-four required
  Ubuntu shards each export 104 exact artifacts; one macOS consumer compiles
  all 2,496 sources with warnings fatal and requires a non-empty AIR object for
  every entry. This is complete copy translation,
  reflection, and native compiler coverage, not numerical execution or MLX host
  runtime redirection. The compiler-gated identity refresh retains source
  coverage, materializations and resource ABI. `artifactIdentityRefresh` retains
  the historical linkage-only audit separately from the earlier `proof` metadata.
  The subsequent complete review also accounts for byte conversions and typed
  vector initializers in 736 changed bodies, with 1,760 bodies unchanged. These
  reference updates do not establish coverage of a newer MLX revision;
- selected-entry translation of all 2,496 discovered historical
  `copy.metal` entries to OpenGL at revision
  `846d176227a0ac13d2667e58d2bb68b322109ab0`. The schema-v2
  `contracts/copy.opengl-translation.json` contract pins every standalone
  `main` artifact across all 30 shapes, 16 templates, 13 input/output types,
  169 conversion pairs, 6,566 materializations, and 8,684 reflected target
  resources. Registered `complex_t_float` representations project through the
  real field for the 150 complex-to-scalar entries only after exact ordered
  `real`/`imag` float-shape validation; malformed registered shapes continue to
  fail closed. Twelve explicit host/runtime index-range preconditions bound
  `offset + i`, `src_idx`, `dst_idx`, `dst_idx + i`,
  `src_idx + src_offset`, `dst_idx + dst_offset`, `idx.x`, `idx.y`,
  `src_offset`, `dst_offset`, `src + idx.x`, and `dst + idx.y` to
  signed 32-bit OpenGL index space. These are portability promises, not
  inferred or runtime-enforced checks, and unproven wide indices remain
  rejected. The last four cover the dynamic general shapes, whose int64
  `src += src_offset` rebases and subsequent `src[idx.x]`/`dst[idx.y]` reads
  must keep the complete signed logical offset in range; the 26 affected
  entries reproduce their accepted artifact identities byte-for-byte.
  Twenty-four required Linux CI shards retranslate every exact
  entry, verify deterministic identity, materialization, workgroup metadata,
  and reflected ABI, compile for OpenGL/SPIR-V 1.3 with
  `glslangValidator`, validate with `spirv-val`, and require 2,496 non-empty
  SPIR-V modules. Together with the Metal proof this closes complete copy
  translation, reflection, and native compiler coverage for those two targets.
  A complete source review accounts for narrow-conversion helpers in 308
  changed artifacts; 2,188 artifacts are byte-identical. All interfaces and
  indexing remain unchanged, and 61 fresh translation checks cover every
  changed conversion pair and shape. A subsequent full review preserves 2,482
  artifacts and updates only the 14 bfloat16-to-Boolean entries. Their sole
  change explicitly extracts the unsigned 16-bit word and promotes it to
  `int` before the unchanged sign-bit mask. Exact accepted-source reconstruction
  and 2,510 compiler/validator runs cover the whole family and both versions
  of those 14 entries. The compiled modules differ; this is not a claim of
  byte-identical compiler output. The conversion is checked over every bfloat16
  word, while all resource interfaces, materializations and index preconditions
  remain unchanged. The 28 complex64-to-float16 and complex64-to-bfloat16
  entries now round the projected real part through the same narrow-float
  helpers as float32 sources, because float32 storage kept it unrounded (65520.0
  stayed finite instead of +inf and bfloat16 pi kept its float32 low bits on
  Mesa).
  Separately, `test_current_copy.py` executes current-pin contiguous and strided
  bfloat16-to-Boolean copies over all 65,536 bit patterns per entry, including
  signed zero, subnormals, infinities and NaNs. Generated Metal and OpenGL match
  unmodified original Metal across 131,072 computed values and 128 output guards
  per target. Inputs use lossless bit-pattern transport; output buffers start
  with the opposite expected value. The same module now passes all 52 tests in
  the Windows native job at `29e8eff3`. Independent review of the retained
  [Windows evidence](https://github.com/CrossGL/crosstl/actions/runs/38107583878/artifacts/11691531710)
  verifies all 131,072 bfloat-to-Boolean results and 128 guards against the
  conversion oracle and original Metal readbacks, alongside the HLSL identities
  and strict DXC receipts. These cases do not request generated input observations, so this
  does not establish input immutability. The copy harness uses the
  existing host-runtime entry-name normalization for reflected HLSL constants,
  including exported names ending in an underscore. This changes test bindings,
  not upstream sources, generated shaders or numerical expectations.
  These two entries do not
  establish whole-family numerical parity or complete MLX host redirection;
- selected-entry translation of all 2,496 discovered historical
  `copy.metal` entries to DirectX at revision
  `846d176227a0ac13d2667e58d2bb68b322109ab0`. The compact schema-v2
  `contracts/copy.directx-translation.json` contract pins every standalone
  `CSMain` artifact across all 30 shapes, 16 templates, 13 input/output types,
  169 conversion pairs, and 6,566 exact materializations. Exact HLSL target
  reflection contains 10,036 resources; `g2`, `g2large`, `g3`, `g3large`,
  `gn2`, `gn4large`, `s2`, and `v2` receive generated
  `CrossGLDispatchInfo` metadata at their shape-specific `b0`, `b6`, or `b3`
  binding. Registered `complex_t_float` representations project through the
  real field for the 150 complex-to-scalar entries only after exact ordered
  `real`/`imag` float-shape validation, while malformed registered shapes fail
  closed. Unlike the OpenGL lowering, this native HLSL path introduces no
  additional source-scoped 32-bit index-range portability promise. Twenty-four
  required Ubuntu CI shards retranslate every exact entry, verify identity,
  materialization, workgroup metadata, and reflected ABI, and compile with
  checksum-pinned DXC using `-enable-16bit-types -WX -T cs_6_2 -E CSMain`,
  requiring 2,496 non-empty DXIL modules. Together with the Metal and OpenGL
  proofs this closes complete copy translation, reflection, and native
  compiler coverage on all three targets, not numerical execution or MLX host
  runtime redirection. A subsequent full-family reference review reconstructs
  every previously accepted source and verifies byte-identical strict DXIL
  and reflected interfaces for all 2,496 pairs, preserving 6,566
  materializations and 10,036 resources. The changes are unused structure
  fields and signed/unsigned casts of nonnegative 16-bit values before
  masking in bfloat-to-bool helpers. The review retains the historical source
  pin and all launch contracts. A later conversion review updates the 80
  unsigned 32-bit and 8-bit to float16, float32 and complex64 entries, which now
  call one correctly rounded unsigned-to-float helper. Removing the helper and
  restoring `float(...)` reconstructs every accepted source; all 80 old and
  new sources compile with warnings fatal, and materializations and reflected
  resources are unchanged. The 14 complex64-to-float16 entries now round the
  real part through the same binary16 helper as float32-to-float16 copies,
  because WARP's native `float16_t(...)` turned 65520.0 and 2^31 into 65504
  instead of infinity (all 14 compile with warnings fatal). The 80 int64 and
  uint64 to float16, float32 and complex64 entries now convert through exact
  64-bit helpers that keep 32 leading bits plus a sticky bit and round once,
  because WARP's native `float(uint64_t)` rounds twice and turned
  2^63 + 2^39 + 1 into 2^63 (5,852,576 source bytes in total; all 80 compile
  with warnings fatal). Compiler
  equivalence does not establish current-pin numerical execution or full
  upstream-suite parity;
- selected-entry Metal-to-CrossGL-to-Metal translation of all 4,122
  discovered current-pinned binary entries from ``binary.metal``. The complete
  contract spans 18 shapes and 11 concrete kernel templates, 24 operators, and
  25 input/output type pairs. It emits one kernel and one or two reachable
  materializations per artifact (6,026 exact materializations total), reflects
  each exact three- through seven-resource ABI, and preserves a host-owned
  ``[1, 1, 1]`` dispatch. Required macOS CI compiles every artifact with
  warnings fatal and requires 4,122 non-empty AIR outputs. This proves every
  discovered binary instantiation for Metal translation, reflection, and native
  compilation, not Metal numerical execution;
- selected-entry translation of all 4,122 discovered current-pinned
  `binary.metal` entries to OpenGL. The schema-v2
  `contracts/binary.opengl-translation.json` contract pins every standalone
  `main` artifact across all 18 shapes, 11 templates, 24 operators, 25 type
  pairs, 6,026 materializations, and 19,106 reflected resources, totaling
  18,732,199 generated GLSL bytes. Seven explicit
  host/runtime index-range preconditions bound `offset + i`, `a_idx`, `b_idx`,
  `out_idx`, `out_idx++`, `idx.x`, and `idx.y` to signed 32-bit OpenGL index
  space; they are portability promises, not inferred or runtime-enforced
  checks, and unproven wide indices still fail closed. Twenty-four required
  Linux CI shards retranslate every exact entry, verify deterministic identity,
  materialization, workgroup metadata, and reflected ABI, compile for
  OpenGL/SPIR-V 1.3 with `glslangValidator`, validate with `spirv-val`, and
  require 4,122 non-empty SPIR-V modules. All known binary reference changes
  have complete source, interface and strict compiler reviews. The 123 changed
  math artifacts are not claimed to be numerically equivalent to their predecessors.
  Reviewed half-remainder entries have bounded numerical evidence described
  below, not whole-family parity or MLX host runtime redirection;
- selected-entry translation of all 4,122 discovered historical
  `binary.metal` entries to DirectX at revision
  `846d176227a0ac13d2667e58d2bb68b322109ab0`. The schema-v2
  `contracts/binary.directx-translation.json` contract pins every standalone
  `CSMain` artifact across all 18 shapes, 11 templates, 24 operators, 25 type
  pairs, and 6,026 materializations. Nine source shapes that consume
  `threads_per_grid` receive explicit `CrossGLDispatchInfo` interfaces, raising
  the exact aggregate reflection count to 21,248 resources across three-
  through eight-resource ABIs. Computed-result bfloat `ArcTan2`, `LogAddExp`,
  and `Power` paths expand to float and reconstruct with round-to-nearest-even;
  `Maximum` and `Minimum` expand only for comparison and return the selected
  original bfloat payload without requantization. All other unproven bfloat
  builtins remain fail-closed. The
  same seven explicit host/runtime index-range preconditions remain required.
  Twenty-four required Ubuntu CI shards retranslate every exact entry, verify
  deterministic identity, materialization, `CSMain` workgroup metadata, and
  reflected ABI, then compile with checksum-pinned DXC using native 16-bit
  types and warnings fatal. The gate requires 4,122 non-empty DXIL modules.
  The half minimum/maximum review updates 36 references across all 18 shapes.
  Complete before/after bodies, interfaces and materializations were checked;
  the only changes are a bit-preserving half-selection helper and its calls.
  Both versions compile with strict DXC, and independent rebuilds reproduce
  the current DXIL exactly. Two vector-vector artifacts match the shaders
  executed in Windows CI: 9,344 results and 16 guards agree bit for bit,
  including NaN payloads and signed zero. These samples do not establish
  whole-family numerical execution. The subsequent complete-family review
  accounts for all remaining changes: 4,053 padding-only artifacts retain
  byte-identical DXIL, and 69 reviewed math artifacts compile successfully;
- selected-entry Metal-to-CrossGL-to-Metal translation of all 2,396
  discovered reduction entries from `reduce.metal` at historical revision
  `846d176227a0ac13d2667e58d2bb68b322109ab0`. The compact
  schema-v2 `contracts/reduce.metal-roundtrip.json` contract spans 39 exact ABI
  shapes, nine kernel templates, six operator families, 44 concrete operator
  types, and 13 input/output types. It pins every artifact identity plus exact
  materialization and resource digests while avoiding transient proof rows in
  git. The one- through twelve-resource interfaces contain 25,088 reflected
  resources in aggregate and retain a host-owned `[1, 1, 1]` workgroup
  contract. Twenty-four required Ubuntu shards cover 20 groups of 100 entries
  and four groups of 99. One macOS consumer compiles every exported artifact
  with warnings fatal and requires 2,396 non-empty AIR objects. Complete source
  and interface comparisons account for 558 updated references: source product
  grouping and half-precision infinity initializers. Native checks of the two
  half initializer kernels match unchanged upstream Metal and independent
  expected words, including output guards. This is complete reduce translation, reflection,
  and native compiler coverage, not numerical execution or MLX host runtime
  redirection. Subsequent reviews of all 400 entries in source shards 0 through 3
  update 393 identities for unused-parameter annotations and typed integer boundary
  constants. Every previous source is reconstructed to its accepted hash;
  previous and current sources compile and link to byte-identical Metal libraries
  with warnings fatal. The reviews preserve 1,534 materializations and 4,156
  resources, leaving the other 2,003 records unchanged. This compiler-equivalence
  check adds no numerical execution or current-pin coverage. Reference
  reconciliation for the remaining shards is still tracked by
  [#1966](https://github.com/CrossGL/crosstl/issues/1966);
- selected-entry Metal-to-CrossGL-to-Metal translation of all 2,052 host-named
  entries from `quantized.metal`. The compact schema-v2
  `contracts/quantized.metal-roundtrip.json` contract covers 38 normalized
  variants over 17 source templates for three data types, group sizes 32/64/128,
  and bit widths 2/3/4/5/6/8. It pins 14,904 exact materializations and
  42,976,818 generated Metal bytes without checking transient proof rows into
  git. Thirty deduplicated exact ABI contracts span four- through
  twenty-two-resource interfaces and contain 31,374 reflected resources in
  aggregate, all with host-owned `[1, 1, 1]` workgroups. The original terminal external
  proof compiled every entry twice with Metal 3.1 and warnings fatal, required
  byte-identical non-empty AIR, and is bound by the checked-in terminal,
  independent-audit, and adversarial identities. The `proof` section retains
  that original audit provenance; `entries` and `artifactContract` describe the
  reconciled output at historical revision `846d1762`. The latest review
  reconstructs the complete previous source identities for 1,998 changed
  entries; 54 dequantizers are unchanged. Changes preserve pointer-reference
  updates in batch-offset helpers and division grouping in matrix loaders,
  annotate unused parameters, type integer masks explicitly and lower `round`
  through the existing Metal helper. The pointer and grouping changes correct
  computation and are not treated as compiler-equivalent formatting. Exact
  materialization and resource contracts are unchanged. All 2,052 reviewed
  sources pass strict Metal compilation with empty compiler streams.
  Twenty-four required Ubuntu CI
  shards rediscover the exact source entries, retranslate and reflect each one;
  twelve shards contain 86 entries and twelve contain 85. One dependent macOS
  job verifies every source identity and compiles all entries with Metal 3.1,
  warnings fatal and empty compiler streams. Missing, duplicate or changed
  sources fail before compilation, and the gate requires all 2,052 AIR outputs
  to be non-empty. This
  is complete quantized Metal translation, reflection, and native compiler
  coverage. The same macOS job additionally executes the 108 affine quantization
  and dequantization variants from the verified source bundle: three scalar
  types, three group sizes and six bit widths for both operations. Five groups
  per variant exercise positive and negative scales, exact dyadic values and
  zero ranges. Original upstream Metal and generated Metal must independently
  match expected packed bytes, scales and biases, with readonly inputs and
  output guards preserved. `prove_quantized_metal.py` retains native commands,
  modules, inputs, expected outputs and readbacks. The refreshed-source check
  covers all 108 variants and 40,320 scalar values per original/generated path.
  Nine additional matrix/vector entries cover float32, float16 and bfloat16
  across batched wide vectors and both matrix-loader layouts. Twenty-four cases
  compare 13,536 output values per original/generated path with independently
  calculated dyadic dot products. They include nonzero batch offsets,
  two-dimensional broadcast strides, a partial matrix tile and two output
  initializations. Every dispatch checks all sixteen bindings, including
  readonly contents and sixteen guard bytes per binding. These bounded checks
  add no macOS runner or repeated translation and do not establish numerical
  parity for the complete quantized corpus or MLX host runtime redirection;
- selected-entry translation of all 2,396 discovered `reduce.metal` entries at
  historical revision `846d176227a0ac13d2667e58d2bb68b322109ab0` to OpenGL. The compact schema-v2
  `contracts/reduce.opengl-translation.json` contract spans the same 39 exact
  ABI shapes, nine kernel templates, six operator families, 44 concrete
  operator types, and 13 input/output types. It pins 9,216 exact
  materializations, 45,283,816 generated GLSL bytes, and 25,088
  resources across scalar-layout-aware one- through twelve-resource target ABIs.
  Complete source comparisons and strict compiler checks cover every entry.
  The latest review reconstructs every previous source identity and verifies
  each specialized helper body, including its tail-loop continuation, before
  accepting duplicated helper definitions. All 2,396 current modules compile
  and validate; 1,124 optimized modules are byte-identical to their predecessors.
  Integer initializers retain their source values and explicitly require the
  64-bit integer extension when narrowing a wide literal.
  Resource declarations are unchanged. The 156 complex-buffer reference updates
  add explicit scalar-layout metadata rather than changing buffer storage.
  A later review re-pins the 76 complex64 sum and product sources outside
  `init` for Mesa 25.2's IEEE folds: MLX's free complex operators settle zero
  operands with integer tests before a `precise` operation, because Mesa folds
  `0 * x` to +0 and `x + 0` to `x` even under `precise`. All 76 compile and
  validate, and every other source is unchanged.
  Twenty-one explicit host/runtime index-range preconditions bound the source
  pointer and output expressions required by wide-index shapes to signed 32-bit
  OpenGL index space; they are deployment promises rather than inferred or
  generated checks. Fifteen of them name complete rebased-pointer offsets such
  as `in_ + start_idx`, `inputs[j] + i` and `out_ + i`, because the int64
  `IdxT` shapes rebase storage pointers by wide deltas whose operands alone
  cannot prove the signed sum. With them, the 1,240 entries of the 20 wide-index
  kinds translate again. 868 entries in 14 kinds (62 each) change identity:
  removal of a zero-trip `gl_WorkGroupSize.y` loop under the `[1, 1, 1]`
  workgroup in the long-column kinds, and one peeled tail-loop iteration
  emitted as a separate helper specialization in the five-dimensional kinds.
  The review normalizes exactly those two rewrites, compares every helper by
  greatest bisimulation over overload sets, and accepts all 868 pairs, while
  its negative controls reject every semantic mutation tried.
  Bounded unit-step loop-carried pointer-array initialization is proven
  one exact iteration at a time; a non-singleton per-invocation dynamic target
  remains a may-write and cannot establish full-array definite assignment,
  while shadowed loop-index bindings remain fail-closed. For-in range and
  scalar-count expressions are rendered in the outer lexical environment;
  Fixed-array iterable expressions and compile-time extents resolve before
  same-named pattern bindings enter scope. same-named patterns use
  independent controllers and shadow pointer,
  stage-builtin, and flattened stage-struct aliases in the loop body.
  For-in resource specialization resolves overloads from exact lexical
  pattern types. Dynamic for-in resource specialization resolves overloads
  from exact call-site lexical types. Null storage-pointer reachability and
  elision preserve exact lexical declaration identity. Null workgroup-pointer
  reachability and elision preserve exact lexical declaration identity.
  Nested resource-specialization discovery preserves deterministic lexical
  order across Python hash seeds. Workgroup-pointer bounds analysis visits
  every control-flow expression and preserves outer mutations across lexical
  blocks. loop-local fixed-array
  storage retains
  flattened stage-input struct
  declarations, and fixed-array patterns cannot inherit same-named outer
  scalar or vector-component bounds. Repeated loop bounds and generated loop
  controllers lose their proven
  intervals when exact scalar or vector-component dependencies can mutate.
  logical-offset mutation through resolved nested helpers is written back to
  callers. Direct
  element and addressed one-element forwarding through overload-resolved
  scalar or fixed-array helpers preserves that writeback;
  private scalar address views remain confined to their declaring lexical scopes;
  nonlocal scalar address views remain fail-closed;
  dynamic elements and unresolved, ambiguous, or recursive forwarding remain
  fail-closed. Fixed-array storage cannot escape through local private pointer or
  reference aliases; lexically shadowed arrays and condition-only reads are not
  misattributed, while residual private pointer or reference syntax remains
  fail-closed;
  unsupported wide
  indices, pointers, and recursion still fail closed. Twenty-four required
  Linux CI shards retranslate every exact entry, verify deterministic artifact,
  materialization, workgroup, and reflected ABI
  identity, compile for OpenGL/SPIR-V 1.3 with `glslangValidator`, validate with
  `spirv-val`, and require 2,396 non-empty SPIR-V modules. This is complete
  reduce OpenGL translation, reflection, and native compiler coverage, not
  numerical execution or MLX host runtime redirection;
- selected-entry translation of all 2,396 discovered `reduce.metal` entries to
  DirectX at the historical reference revision
  `846d176227a0ac13d2667e58d2bb68b322109ab0`. The compact schema-v2
  `contracts/reduce.directx-translation.json` contract pins every standalone
  `CSMain` HLSL artifact and its native DXIL identity across the same 39 exact
  ABI shapes, nine kernel templates, six operator families, 44 concrete
  operator types, 13 input/output types, and 9,216 exact materializations.
  Exact HLSL target reflection contains 27,382 resources across one- through
  thirteen-resource interfaces; the 37 shapes other than `init` and `all`
  receive generated `CrossGLDispatchInfo` metadata. Portable unshadowed `NAN`,
  explicit complex SIMD shuffle overloads, proven direct void tail recursion,
  bounded device/constant pointer-array offsets and writeback, logical bfloat
  pointee types over physical storage, and nested anonymous records lower only
  in their natively proven forms. Unsafe recursion, ambiguous pointer-array
  provenance, unresolved aggregate pointer members or arrays, and unsupported
  HIP record lifecycle or layout remain fail-closed. Unlike the OpenGL
  lowering, this native HLSL path introduces no additional source-scoped
  32-bit index-range portability promise. Twenty-four required Ubuntu CI
  shards retranslate every exact entry, verify deterministic identity,
  materialization, workgroup metadata, and reflected ABI, and compile with
  checksum-pinned DXC using `-enable-16bit-types -WX -T cs_6_2 -E CSMain`,
  requiring 2,396 non-empty DXIL modules. The reviewed references total
  37,727,715 HLSL bytes and 24,201,740 DXIL bytes. Source references preserve
  the original integer-product grouping. The subsequent half-selection review
  reproduces all 78 previous half minimum/maximum sources and DXIL identities,
  compares complete bodies and interfaces, and updates 76 changed references;
  the two initializer entries remain byte-identical. Only the bit-preserving
  selection helper and its calls change. Strict independent recompilation
  reproduces all 78 current modules. Shared Windows selection checks cover
  every binary16 encoding in both operand positions, with 853,840 exact values
  and 64 guards; they do not execute these complete reduction kernels.
  The latest review reproduces every previous source and DXIL identity and
  independently recompiles all 2,396 current modules. Of those modules, 2,152
  remain byte-identical. The other 244 include 52 bfloat NaN-predicate corrections
  and 192 signed/unsigned byte-load corrections; these are not presented as
  equivalent to the previous behavior. Byte loads now address one physical
  storage element per logical element instead of unpacking four values from
  it. Existing required native tests cover the shared value-load and floating
  classification rules, without establishing execution of every reduction
  kernel. A subsequent formatting review removes redundant condition
  parentheses from 104 artifacts; all 104 retain their exact DXIL identities.
  The other 2,292 sources are unchanged, and 71 fresh local translations
  independently reproduce the retained sources. All materialization and
  resource-interface identities are unchanged.
  A further review updates the 76 complex64 sum and product sources outside
  `init`: MLX's free complex operators return `precise` results, as in the
  binary family, so DXC no longer folds the zero components of complex
  constants. Checksum-pinned DXC recompiles all 76 with warnings fatal, and
  their DXIL identities are re-pinned.
  This records complete
  translation, reflection and compiler coverage for this historical HLSL corpus,
  not numerical execution or MLX host runtime redirection. Other target reference
  updates remain separate;
- a checked-in reduced Metal fixture that mirrors MLX's reference-returning
  `frag_at` accessor over `val_frags[i * width + j]`. The fixture is translated
  to DirectX and OpenGL through the public `translate-project` CLI and retains
  three separate source contracts. The mutable scalar receiver is declared
  through a function-local `using` alias and assigned through `frag_at`. An
  implicit const scalar call is passed directly to a read-only helper. A second
  outer value owns a `float2`-backed `nestedTile`; its const `store` method binds
  `thread const auto& accum = nestedTile.frag_at(i, j)` and reads `accum[k]`.
  Each generated target must assign the scalar sentinel directly to the original
  `val_frags[...]` lvalue, read back that exact element, lower the implicit const
  call to its backing storage, and replace the nested accessor and `accum` alias
  with a read from `self.nestedTile.val_frags[...][k]`. A value-return helper,
  retained alias, or copied tile does not satisfy the proof. Windows CI requires
  DXC compilation; Linux CI requires `glslangValidator` compilation and
  `spirv-val` validation for OpenGL 4.5. The macOS matrix leg verifies the same
  proof set in both generated artifacts without requiring either target
  compiler. These checks do not execute an MLX runtime or establish numerical
  parity;
- a separate checked-in Metal fixture for the MLX `BaseMMAFrag::load` call
  shape in which a templated tile method passes `&(src[index])` to a templated
  fragment helper. Project translation must emit both DirectX and OpenGL
  artifacts with zero diagnostics. The materialized fragment helper must keep
  a pointer-backed source view and read it at `stride`; a scalar `float src`
  parameter is rejected. The DirectX proof requires a `StructuredBuffer<float>`
  source and preserves the addressed `src[index]` position as a composed
  `src_offset + index` view. The OpenGL proof requires the equivalent global
  storage-buffer plus `src_offset` form, carries `index` into that offset, and
  reads `src[src_offset + stride]`. Source-style unresolved member calls are
  rejected. This gate inspects generated structure;
  it does not require native target compilation, execute a shader, or claim
  runtime parity;
- target-separated DirectX and Vulkan project runs for the same 11-source
  reduced frontier: `arange.metal`, `arg_reduce.metal`, `binary_two.metal`,
  `layer_norm.metal`, `logsumexp.metal`, `random.metal`, `rms_norm.metal`,
  `rope.metal`, `scaled_dot_product_attention.metal`, `softmax.metal`, and
  `ternary.metal`. Vulkan must translate and structurally validate all 11.
  DirectX emits five aggregate artifacts whose entries do not require a
  runtime-selected workgroup size, two entry-scoped `layer_norm.metal`
  artifacts, two entry-scoped `logsumexp.metal` artifacts, and 12 entry-scoped
  `rms_norm.metal` artifacts selected by checked-in host dispatch contracts. It
  records exact expected failures for the other three sources. Each blocked
  report must retain
  the pinned total specialization
  count and exactly match its diagnostic entry names to the materialized host
  names, with no additional diagnostics. Separate configs prevent DirectX
  workgroup contracts from being silently ignored by Vulkan, where project
  workgroup rules are unsupported.
  This establishes target-specific structural and toolchain coverage, not
  semantic readiness or runtime parity;
- a separate project-level expected-failure check for pinned `fence.metal`
  across DirectX, OpenGL, and Vulkan. Each target must report its exact
  `project.translate.*-atomic-fence-unsupported` diagnostic, target-specific
  `*.atomic-thread-fence-contract-lowering` missing capability, and requested
  `mem_device`, `memory_order_seq_cst`, `thread_scope_system` contract without
  emitting a target file. The blocked contract is tracked by
  [#1537](https://github.com/CrossGL/crosstl/issues/1537);
- materialization evidence for all 24 host-named `arg_reduce.metal` compute
  entries within 39 total specializations. Vulkan emits the aggregate artifact.
  DirectX and OpenGL fail before emission with
  `project.translate.workgroup-size-entry-ambiguous` because pinned host
  dispatch uses runtime axis and pipeline-limit operands unavailable to source
  materialization;
- DirectX HLSL compiler checks with official DXC v1.9.2602.24 on Windows CI for
  the 21-artifact frontier representing eight pinned sources: `arange.metal`,
  `binary_two.metal`, two bounded `layer_norm.metal` entries, two bounded
  `logsumexp.metal` entries, `random.metal`, 12 test-derived `rms_norm.metal`
  entries, `rope.metal`, and `ternary.metal`. At the pinned revision the gate
  compiles 11, 225, 2, 2, 2, 12, 18, and 212 entries respectively, for 484
  generated compute entries in total. Each LayerNorm, LogSumExp, and RMSNorm
  artifact is emitted independently with its host-derived workgroup size and
  exact subgroup width; specialization constants are retained where required.
  The 16 normalization artifact references include explicit source-width index
  conversions and sign-preserving floating constants. The float16 RMSNorm
  artifact uses two-byte binary16 storage and explicit conversion at the source
  rounding boundary. Complete old/current body and interface comparisons and
  strict DXC compilation cover all 16 references; these updates do not expand
  the separate numerical execution claims below.
  The
  pinned rope translation supplies required function constant IDs through the
  quoted `"1"`, `"2"`, and `"3"` selectors in
  `[project.specialization_constants]` and materializes the concrete DirectX
  variant before compilation. Aggregate conditional lowering
  completed under [#1695](https://github.com/CrossGL/crosstl/issues/1695) admits
  every pinned `ternary.metal` entry to this compiler gate. Target-ABI overload
  identity [#1694](https://github.com/CrossGL/crosstl/issues/1694) and
  minimum-precision arithmetic widening
  [#1701](https://github.com/CrossGL/crosstl/issues/1701) admit all 225
  `binary_two.metal` entries. Exact-layout DirectX union lowering
  [#1728](https://github.com/CrossGL/crosstl/issues/1728) admits both
  `random.metal` entries; broader union layouts remain tracked by
  [#1696](https://github.com/CrossGL/crosstl/issues/1696), and runtime dispatch
  metadata remains tracked by
  [#1542](https://github.com/CrossGL/crosstl/issues/1542). Host dispatch contract
  import was completed under [#1793](https://github.com/CrossGL/crosstl/issues/1793).
  These are compilation claims only. The current-pin
  [random readiness audit](portable_host/README.md#random-generation-readiness)
  now passes its 20 bounded numerical cases on OpenGL, DirectX and generated
  Metal, including required Windows native execution. `RandomBits` host
  integration is not enabled.
  The three pending aggregate sources cover 76 compute entries. Those historical
  aggregate runs do not consume the later entry-scoped bounded contracts and
  remain asserted as failed artifacts; no placeholder workgroup size is
  restored. Separate bounded GEMV, arg-reduce, Softmax, and scaled-attention
  translation, package, and native-runtime proofs are documented below.
  `fence.metal` is
  excluded because its DirectX translation intentionally fails under
  [#1537](https://github.com/CrossGL/crosstl/issues/1537) before DXC. This gate
  establishes compiler acceptance only; it does not dispatch these kernels or
  establish numerical parity;
- a separate Windows CI Direct3D 12 execution proof for a checked-in,
  MLX-shaped Metal compute fixture with a file-scope immutable two-dimensional
  lookup table. The fixture is translated to HLSL during the test, compiled to
  DXIL with DXC, dispatched through the built-in DirectX runtime adapter and
  `compushady`, and read back as four exact unsigned values:
  `[5, 19, 11, 13]`. Because every result selects a table entry, this is a
  value-sensitive proof that the generated `static const` initializer survives;
  it is independent of the compiler-only frontier gate above;
- a second, independent Windows CI Direct3D 12 execution proof that translates
  the actual pinned `mlx/backend/metal/kernels/arange.metal` source with the MLX
  repository root as an include path. Project configuration selects the
  materialized `arangeuint32` entry and emits a standalone artifact at the
  deterministic `arange/arangeuint32.hlsl` path. The proof verifies the pinned
  source hash, entry-scoped provenance, source mapping, and the generated runtime
  artifact manifest before compiling `CSMain` to DXIL with DXC. It binds only
  the reflected `b0` start, `b1` step, and `u2` output resources and dispatches
  through the built-in Direct3D 12 adapter. Seven invocations use `start = 300`
  and `step = 17`; the required zero-tolerance readback is
  `[300, 317, 334, 351, 368, 385, 402]`;
- Direct3D 12 and OpenGL 4.3 native-loader execution of the actual pinned
  `mlx/backend/metal/kernels/binary.metal` source. The project request selects
  `ss_Addfloat32`, packages one standalone target entry, and verifies the exact
  upstream source hash, entry-scoped provenance, reflected resource bindings,
  and scalar buffer layouts. Four invocations read `1.5` and `2.25` from the
  two source buffers and must return `[3.75, 3.75, 3.75, 3.75]` from the
  translated kernel. Windows runs the HLSL artifact through Direct3D 12; Linux
  runs the GLSL artifact through a surfaceless Mesa OpenGL context. This proves
  one real binary operator entry and does not claim coverage of the other
  materialized binary variants or the upstream MLX host runtime;
- the `arangeuint32` and `ss_Addfloat32` native-loader checks above, together
  with the scalar copy, float32 dot-product, bounded GEMV, bounded Softmax,
  bounded scaled-attention, and unary Square and ArcCos checks described below,
  use the
  current corpus at commit
  `846d176227a0ac13d2667e58d2bb68b322109ab0`. The broader 40-unit frontier and
  its remaining proofs retain their recorded historical revision until each
  source contract is remeasured;
- Vulkan assembly and validator checks for the existing non-fence regression
  frontier when SPIR-V tools are available. Vulkan atomic-fence feature work is
  deferred; the separate `fence.metal` contract check prevents generated
  barriers from being mistaken for semantic support;
- entry-scoped OpenGL packaging for the materialized `arangeuint32` compute
  entry from `arange.metal`. Project configuration selects the source entry,
  emits it at the deterministic `arange/arangeuint32.glsl` path as OpenGL
  `main`, and records the source-to-target entry identity in the portability
  report. The standalone artifact exposes only the `start`, `step`, and `out`
  resources and preserves the source arithmetic without an MLX source rewrite;
- an eight-source OpenGL frontier containing `arg_reduce.metal`,
  `binary_two.metal`, `logsumexp.metal`, `rms_norm.metal`, `rope.metal`,
  `scaled_dot_product_attention.metal`, `softmax.metal`, and `ternary.metal`
  in two project runs. `binary_two.metal`, `rope.metal`, and `ternary.metal`
  must emit with zero diagnostics and compile for OpenGL/SPIR-V 1.3 before
  `spirv-val`. Their project configuration supplies 24 source-qualified
  index-range assertions, and the portability report must reproduce the exact
  assertion count and content. The other five sources must produce the same
  exact fail-closed workgroup-size diagnostic and no target file. This is native
  artifact validation only; the gate does not run the kernels or establish
  runtime parity;
- a separate bounded OpenGL dispatch-contract proof for the actual pinned
  `logsumexp.metal` source. The two float32 workloads emit standalone GLSL
  artifacts with workgroup sizes `[32, 1, 1]` and `[288, 1, 1]`, an exact
  subgroup width of 32, and preserved subgroup max and sum reductions. Linux CI
  compiles both artifacts to OpenGL SPIR-V 1.3 and validates both binaries. The
  runtime contract requires `GL_KHR_shader_subgroup` and rejects a device whose
  `GL_SUBGROUP_SIZE_KHR` value is not 32 before shader compilation or dispatch;
- full project materialization of pinned `gemv.metal` for DirectX. The gate
  requires 226 materialized specializations—224 host-named entries plus both
  reachable `elem_to_loc` helper overloads—no unsupported materializations or
  unresolved residue, no bare pure value-discard statements, one aggregate HLSL
  artifact, and exactly 224 host-named report execution entries joined by
  materialization identity. Every generated target entry and `numthreads`
  declaration must match its report contract. The emitted native 16-bit types
  require DXC to compile `CSMain`, `CSMain_85`, and `CSMain_113` under
  `cs_6_2` with `-enable-16bit-types` and zero diagnostics, then compile all
  224 functions in one `lib_6_6` invocation with the same flag, an exact
  export set, and exact profile-warning classification;
- full project materialization of pinned `gemv.metal` for OpenGL. The gate
  requires all 226 specializations materialized, all 224 entry-scoped GLSL
  artifacts emitted, exact workgroup and subgroup execution records, and the
  fifteen explicit host index-range preconditions used for 64-bit source
  indices. Thirteen of them bound row, helper-load and batch offsets and their
  complete pointer-rebased forms to `[0, 2^31 - 1]`, because OpenGL carries
  each checked pointer offset in a signed `int`; the two gather batch offsets
  `batch_offsets.x` and `batch_offsets.y` keep `[0, 2^32 - 1]`.
  Linux CI compiles every artifact with `glslangValidator` and validates every
  resulting SPIR-V 1.3 module with `spirv-val`. This historical aggregate gate
  does not establish runtime execution or numerical parity; a distinct bounded
  current-pinned native-runtime proof is documented below;
- on Linux CI, full project materialization and translation of `gemv.metal` to
  Vulkan produces 226 specializations and 224 `GLCompute` entry points. The
  generated artifact passes both `spirv-as` and `spirv-val` for `vulkan1.1`
  with zero semantic warnings and no known codegen fallbacks. This is structural
  validation, not numerical runtime parity;
- runtime artifact manifest, runtime-test manifest, and runtime-test plan
  generation for reduced `arange` readiness probes across DirectX, OpenGL, and
  Vulkan;
- reference runtime fixture execution reports for the reduced `arange`
  readiness probes, using supplied project test-runner adapters and
  deterministic expected-output checks;
- native runtime execution-readiness reports for the same reduced probes,
  using the built-in DirectX, OpenGL, and Vulkan native adapter contracts with
  missing runtime drivers reported as structured blockers;
- on Linux CI, native Vulkan execution and readback for the generated MLX
  `arange.metal` unsigned 32-bit, signed 32-bit, and floating-point entry points
  through the optional Vulkan compute runtime and Mesa Vulkan software driver.
- on Linux CI, native OpenGL 4.3 execution and readback for the selected
  `arangeuint32` artifact through ModernGL and Mesa EGL. Four invocations use
  `start = 300` and `step = 17`; the required zero-tolerance comparison is
  `[300, 317, 334, 351]`.
- on Linux CI, OpenGL SPIR-V specialization through PyOpenGL and Mesa EGL for a
  reduced generated compute artifact. The native adapter compiles the GLSL to
  SPIR-V, applies numeric constant ID 7, and requires exact readback for two
  independently selected unsigned values;
- on Windows CI, native Direct3D 12 execution and exact readback for the reduced
  file-scope immutable lookup fixture, one generated uint32 entry from pinned
  `arange.metal`, and one generated float addition entry from pinned
  `binary.metal`. Linux CI executes the same binary entry through OpenGL. None
  of these proofs executes the upstream MLX host runtime.

Pull requests run the 12-source pinned reduced scope: 11 non-fence frontier sources
and the explicitly blocked `fence.metal` contract source. They also run the
separate checked-in reference-accessor and template-member pointer fixtures;
those fixtures do not change the pinned MLX source count. Scheduled and
manually triggered CI also run the full-corpus artifact scout with finite Metal
template materialization budgets.
The generated full-corpus project config caps
`max_template_specializations` at 4096 and
`max_template_materialization_work` at 131072. Each artifact translation also
has a 120-second limit. A timed-out artifact is recorded with a structured
`project.translate.timeout` diagnostic, and the remaining canonical plan
continues. The outer 900-second command limit remains an aggregate CI boundary;
the durable checkpoint preserves every completed result before that boundary.
The scout discovers all 40 pinned MLX Metal kernel units and attempts 120
DirectX, OpenGL, and Vulkan artifacts.
Its fence-aware success condition is 117 translated artifacts and three expected
failed `fence.metal` records, one per target, with no fence target files emitted.
That condition is a gate expectation, not a claim that the pinned full corpus
currently satisfies it. Additional failures remain issue-backed scout results.
CI uploads generated portability reports, validation summaries embedded in
those reports, the durable full-corpus translation checkpoint, generated logs,
available generated artifacts, and a concise JSON summary. A subsequent
full-corpus invocation resumes verified completed jobs from a running or
interrupted checkpoint.
Because `binary_two.metal` was already in the DirectX/Vulkan frontier, its OpenGL
promotion does not change either reduced source count.
The same applies to `ternary.metal`: its OpenGL promotion expands the native
toolchain gate without changing the 11-source non-fence reduced frontier.

The checked-in historical full-corpus scout snapshot against MLX revision
`968d264f2903d578e699c4452a4dbf48633921aa`
scanned 40 Metal kernels and attempted 120 target artifacts across DirectX,
OpenGL, and Vulkan. It translated 24 artifacts and reported 96 structured
artifact failures behind tracked issues. The current materialization pass
rejects template-hostile targets when concrete variants are missing instead of
emitting generic artifacts, so full-corpus counts should not be treated as
runtime-complete coverage or as the current fence-aware expected baseline.

This is shader/kernel artifact coverage. It does not claim that the MLX host
runtime has been ported to Direct3D, OpenGL, or Vulkan. Running the upstream MLX
GPU unit tests against non-Metal targets requires runtime adapters, host-side
dispatch wiring, data layout validation, and backend-specific build integration.
The reduced harness now emits runtime readiness artifacts so those gaps are
visible in CI reports without claiming runtime parity. The current readiness
manifests consume reflected runtime artifact metadata, including entry points,
resource bindings, and dispatch geometry. Runtime-test plans now resolve
source-level fixture names against common generated resource aliases. The
reference-accessor fixture is not included in those runtime manifests; its
scope ends at generated-source inspection and native compiler validation. The
reduced arange fixtures select the translated artifact by source and target,
then select `CSMain`, `main`, or `arangeuint32` independently for DirectX,
OpenGL, or Vulkan dispatch. OpenGL project translation packages source entry
`arangeuint32` as a standalone `main` artifact with an entry-scoped reflected
interface; its unsigned fixture uses values above the 8-bit range to detect
entry drift. The general DirectX readiness probe still describes the aggregate
artifact's first `uint8` entry. The separate required Windows device proof
packages `arangeuint32` as a standalone `CSMain` artifact. Vulkan execution selects
`arangeuint32`, `arangeint32`, and
`arangefloat32` explicitly and uses the same wide unsigned probe.
Plans report remaining non-blocking platform, layout, and entry-point ownership
warnings. The reduced fixture execution
report exercises the project runner and adapter contract with reference
buffers. The native execution report attempts the built-in native adapter
contract separately. On Windows CI the generated DirectX frontier HLSL must
compile with DXC. Separate native tests translate and execute the reduced
immutable lookup fixture and the pinned source's generated uint32 arange entry
through Direct3D 12. The arange test is numerical evidence for that one source,
entry, dtype, dispatch shape, and fixture only; it does not turn the frontier
compiler gate into a general runtime-parity claim. The five aggregate DirectX
frontier artifacts carry exact per-source bfloat16 report evidence. All five
report `status=exact`, `approximationUsed=false`, a `uint-low-16-bits` register
representation, and round-to-nearest, ties-to-even conversion. All five
aggregate sources require native `uint16` storage declarations and report the
`directx.native-16bit-types` capability. The harness compares each artifact's
`bfloat16Lowering` and `requiredCapabilities` fields with this pinned contract
and fails closed if either field is missing or changes. Native-profile bfloat
helpers now use exact `uint16_t` boundaries, and the two selected
`random.metal` entries compile without the promotion warnings tracked by
[#1799](https://github.com/CrossGL/crosstl/issues/1799). DXC reports zero
warnings across all 484 entry-point runs in the 21-artifact emitted frontier.
The harness records this as a warning-clean contract and rejects any newly
observed warning. Contextual destination conversion under
[#1801](https://github.com/CrossGL/crosstl/issues/1801) is resolved for the
pinned frontier. The arange assignment is emitted as
`arangeint16_out[index] = int16_t((uint(arangeint16_start) +
(index * uint(arangeint16_step))));`. The rope assignments are emitted as
`index_1 = uint(((2 * pos.x) + (pos.y * stride)));` and
`index_1 = uint((pos.x + (pos.y * stride)));`. All 18 rope entries compile with
DXC profile `cs_6_2`, `-enable-16bit-types`, and `-WX`. Native 16-bit arithmetic
conversion under [#1802](https://github.com/CrossGL/crosstl/issues/1802) is also
resolved for the pinned frontier. The float16 assignment is emitted as
`arangefloat16_out[index] = (arangefloat16_start + (float16_t(index) *
arangefloat16_step));`. All 11 arange entries compile without the
destination-conversion warning previously tracked by #1801 and with DXC profile
`cs_6_2`, `-enable-16bit-types`, and `-WX`. This preserves the resolved int16
destination-conversion evidence. The five-source aggregate portion therefore
has compiler acceptance and a warning-clean diagnostic contract. This is
storage, conversion, report, and compiler evidence only; it does not execute a
bfloat16 workload or establish runtime or numerical parity. On macOS CI, the
generated `fence.metal` round-trip artifact must compile to AIR with the native
Metal compiler. This checks generated source and project metadata, not numerical
runtime parity or equivalent resource visibility. CrossGL/crosstl#1660 tracks
preservation of
the source `volatile coherent(system)` pointer contract. On
Linux CI the generated Vulkan
`arangeuint32`, `arangeint32`, and `arangefloat32` entry points must assemble,
load, dispatch, read back, and compare on the Vulkan compute runtime; other
unavailable native backends remain structured blockers until backend runtime
drivers are supplied by integration code. This still does not execute
the upstream MLX host runtime or the upstream MLX Python/C++ unit test suite on
non-Metal backends.

## Running Locally

Clone MLX and check out the pinned revision:

```bash
git clone https://github.com/ml-explore/mlx.git /tmp/mlx
git -C /tmp/mlx checkout 4367c73b60541ddd5a266ce4644fd93d20223b6e
```

Run the project-porting harness from the CrossTL repository:

```bash
python demos/integrations/mlx/run_porting.py --mlx-root /tmp/mlx
```

Run the full-corpus artifact scout:

```bash
python demos/integrations/mlx/run_porting.py \
  --mode full-corpus \
  --mlx-root /tmp/mlx \
  --summary /tmp/mlx/.crosstl-mlx-porting/full-corpus-summary.json
```

On Linux, install the OpenGL, SPIR-V, and Vulkan runtime dependencies to require
the clean OpenGL compiler gates, the pinned GEMV toolchain gate, and native
OpenGL and Vulkan execution of the generated MLX `arange` artifacts:

```bash
sudo apt-get update
sudo apt-get install -y glslang-tools libegl1 libgl1-mesa-dri libglx-mesa0 mesa-vulkan-drivers spirv-tools vulkan-tools
python -m pip install moderngl==5.12.0 PyOpenGL==3.1.10 vulkan==1.3.275.1
python demos/integrations/mlx/run_porting.py \
  --mlx-root /tmp/mlx \
  --require-opengl-frontier-toolchain \
  --require-opengl-gemv-toolchain \
  --require-opengl-native-runtime \
  --require-vulkan-gemv-toolchain \
  --require-vulkan-toolchain \
  --require-vulkan-native-runtime

python demos/integrations/mlx/prove_copy_opengl.py \
  --mlx-root /tmp/mlx \
  --work-dir .crosstl-mlx-porting/copy-opengl

python demos/integrations/mlx/prove_rms_norm_specialization.py \
  --mlx-root /tmp/mlx \
  --work-dir .crosstl-mlx-porting/rms-norm-specialization \
  --require-opengl-toolchain

python demos/integrations/mlx/prove_quantized_opengl.py \
  --mlx-root /tmp/mlx \
  --work-dir .crosstl-mlx-porting/quantized-opengl \
  --require-opengl-toolchain

python demos/integrations/mlx/prove_quantized_opengl.py \
  --mlx-root /tmp/mlx \
  --work-dir .crosstl-mlx-porting/quantized-gather-opengl \
  --entry-point affine_gather_qmv_fast_float_gs_32_b_2 \
  --require-opengl-toolchain
```

On Windows, install DXC to require DirectX HLSL validation for the reduced
frontier, the selected 256-point FFT entry, the pinned GEMV compiler frontier,
all three reference-accessor artifact proofs, the selected LayerNorm entries,
and the selected complex-copy entry:

```bash
python demos/integrations/mlx/run_porting.py \
  --mlx-root C:/path/to/mlx \
  --require-directx-toolchain \
  --require-directx-gemv-compiler-frontier

python demos/integrations/mlx/prove_layer_norm_directx.py \
  --mlx-root C:/path/to/mlx \
  --work-dir .crosstl-mlx-porting/layer-norm-directx \
  --require-directx-toolchain

python demos/integrations/mlx/prove_copy_directx.py \
  --mlx-root C:/path/to/mlx \
  --work-dir .crosstl-mlx-porting/copy-directx \
  --require-toolchain

python demos/integrations/mlx/prove_rms_norm_specialization.py \
  --mlx-root C:/path/to/mlx \
  --work-dir .crosstl-mlx-porting/rms-norm-specialization \
  --require-directx-toolchain
```

Install the DirectX runtime extra separately to execute the value-sensitive
lookup fixture through Direct3D 12:

```powershell
python -m pip install -e ".[directx-runtime]" pytest-xdist
$env:CROSTL_RUN_DIRECTX_LOOKUP_DEVICE_TEST = "1"
python -m pytest -q -n auto `
  tests/test_translator/test_native_runtime_drivers.py `
  -k "directx_compute_runtime_executes_mlx_file_scope_lookup_on_device"
```

With the pinned MLX checkout available, run the generated arange proof
separately:

```powershell
$env:CROSTL_MLX_ROOT = "C:/path/to/mlx"
$env:CROSTL_RUN_DIRECTX_MLX_ARANGE_DEVICE_TEST = "1"
python -m pytest -q -n auto `
  tests/test_translator/test_native_runtime_drivers.py `
  -k "directx_compute_runtime_executes_translated_pinned_mlx_arange_on_device"
```

The selected 256-point FFT proof uses the same pinned checkout and executes the
translated artifact through the native loader on Direct3D 12 WARP:

```powershell
$env:CROSTL_MLX_ROOT = "C:/path/to/mlx"
$env:CROSTL_REQUIRE_MLX_FFT_DIRECTX_NATIVE_LOADER = "1"
python -m pytest -q -n auto `
  demos/integrations/mlx/tests/kernels/test_fft_native_loader.py::test_pinned_mlx_fft_executes_through_directx_native_loader
```

On macOS, require native compilation of the generated Metal round-trip artifact:

```bash
python demos/integrations/mlx/run_porting.py \
  --mlx-root /tmp/mlx \
  --require-metal-toolchain
```

The harness writes reports, generated artifacts, and command logs under
`<mlx-root>/.crosstl-mlx-porting`.

## Corpus Baselines

The reduced compiler and runtime proofs remain pinned to MLX commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e`. Their checked-in source hashes,
dispatch contracts, generated artifacts, and numerical results describe that
exact reference revision and are not relabelled when the upstream corpus moves.

The scheduled full-corpus scout is pinned separately to current MLX commit
`846d176227a0ac13d2667e58d2bb68b322109ab0`. Entry discovery for that revision
records 42 Metal source units and 17,319 host-visible entries with no discovery
diagnostics. The scout plans 84 source-target coordinates across DirectX and
OpenGL, writes resumable progress and portability reports, and reports
unsupported constructs as structured failures. This corpus scan measures
translation coverage only; it does not claim MLX runtime integration or
numerical parity on either target.

## Quantized Wide OpenGL Checks

The Linux CI job `mlx-quantized-wide-opengl` translates 18 selected
`affine_qmv_wide` entries from the current pinned corpus. It covers float32,
float16, and bfloat16 inputs; 3-, 5-, and 6-bit weights; and two or four vectors
per threadgroup. Each case uses group size 128, eight reduction lanes, batch
mode 0, and the source-faithful `[32, 2, 1]` workgroup: two logical 32-lane
SIMD groups and 64 total invocations. The vector count controls input-vector
tiling and does not change the fixed subgroup count. These cases exercise the
local eight-element dequantization array, including zero-offset and nonzero
pointer updates.

The gate checks the pinned host-dispatch and kernel source identities, project
report, source and artifact hashes, preserved array parameters, and exact
execution metadata. It explicitly lowers the two logical SIMD groups in shared
memory with software subgroup width 32, without KHR hardware-subgroup metadata,
then requires GLSL compilation and SPIR-V validation. Missing tools or a missing
pinned checkout fail the required CI job. Generated shaders, portability reports,
SPIR-V modules, and JUnit results are uploaded for inspection. No upstream MLX
source changes or out-of-bounds recovery policy are used by these cases.

To run the gate locally with `glslangValidator` and `spirv-val` on `PATH`:

```bash
CROSTL_MLX_ROOT=/path/to/pinned/mlx \
CROSTL_REQUIRE_MLX_QUANTIZED_WIDE_OPENGL=1 \
python -m pytest -q -n auto \
  demos/integrations/mlx/tests/kernels/test_quantized_wide_opengl.py
```

This is selected-entry compiler coverage, not completion of all 2,052 quantized
entries. It does not execute those entries, establish numerical parity, redirect
the MLX host runtime, or run the upstream MLX test suite on OpenGL.

## Current Translator Gaps

The historical five-file DirectX frontier at `4367c73b` translates after
canonical lowering of materialized Metal `trunc` wrappers
([#2091](https://github.com/CrossGL/crosstl/issues/2091)). Reduced controls execute
on original/generated Metal and OpenGL with exact finite bits; the corresponding
Windows execution remains required. This does not establish full-tree parity.
The same whole-file configuration at `9c3d3557` currently rejects bfloat vector
construction in all five units with `project.translate.directx-bfloat16-unsupported`
for `bfloat16vec2`, under [#1488](https://github.com/CrossGL/crosstl/issues/1488).
Selected-entry contracts and this whole-file boundary are separate coverage.

The five-file configuration targeting Metal at `9c3d3557` translates `arange`,
`random`, `rope` and `ternary`; all four generated files compile with warnings
treated as errors. `binary_two` still rejects the standard-array value returned
by `DivMod::operator()` ([#2093](https://github.com/CrossGL/crosstl/issues/2093)).
This is separate from constructor bitcast typing and does not establish native
execution of these complete files or full-tree parity.

OpenGL lowers the current-pin gated-delta backward kernels' `mlx_atomic<float>`
additions to compare-and-swap loops on `uint` words, and the current-tree gate
above executes all 72 sequential entries natively; general floating-point
atomic support remains tracked by
[#1986](https://github.com/CrossGL/crosstl/issues/1986). Struct
selection now preserves anonymous type-parameter defaults and evaluates supported
`enable_if` partial specializations ([#1985](https://github.com/CrossGL/crosstl/issues/1985)).
The selected `seq_gated_delta_vjp_float_128_128_24_24_1` entry retains the floating
field in `mlx_atomic<float>`. Its generated Metal now compiles with warnings fatal;
the macOS host workflow requires compilation of both the original source and the
translated entry and retains their libraries, compiler logs and identities.
DirectX float buffer addition and exchange now have native execution coverage.
Eight gated-delta backward configurations have required Windows DirectX and
original/generated Metal numerical gates, checking all six gradients against an
independent reference. This covers selected head layouts, checkpoint intervals,
partial segments and float16/float32 storage, not complete MLX host integration.
The DirectX gated-delta and attention fixtures validate the generated binary16
storage metadata before binding physical `uint16_t` resources. They upload the
original half bytes without numeric conversion, retaining structured strides,
guards, tail padding and shared-allocation identities. Half-storage metadata on
float32 or bfloat16 resources is rejected. Numerical reference checks and
tolerances are unchanged.

The attention backward row-dot stage has required Windows, Linux and macOS
gates. They translate all 24 declared float32/float16/bfloat16 entries across
dimensions 64, 72, 80, 96, 128, 192, 256 and 512. Exact dyadic cases distinguish
storage interpretation, indexing and reduction behavior; additional fractional
cases compare against decoded-input float64 dot products at `rtol=atol=1e-5`.
Dispatches preserve the upstream `(32, 1, 1)` workgroup shape and add inactive
rows to test whole-workgroup early returns. An empty-length case must preserve
all output guards. Original and translated Metal read back every buffer;
DirectX and OpenGL read back the writable output. These checks cover the row-dot stage,
not the remaining attention-gradient stages or full upstream attention tests.

OpenGL uses software subgroups of width 32 and validates each generated shader
with glslang before native execution. The fixture records source-scoped index
bounds derived from its dimensions, lengths and six flattened heads. These are
bounded host preconditions, not a general guarantee for 64-bit MLX indices.
Generated GLSL exposes float32 input storage for this stage: the host fixture
decodes the already-quantized float16/bfloat16 source values before uploading
float32 buffers. It preserves the original input files separately from the
uploaded bytes and records both storage widths. The `qL` parameter uses a
checked 16-byte std140 block. No source kernels are rewritten, and this fixture
does not establish automatic layout conversion for a full MLX backend.

The attention tile-derivative stage has separate Windows and macOS gates for
all three declared storage types. Its 24 configurations cover causal masking,
fully masked rows, nonzero tile origins, partial workgroups, fractional inputs,
and the shared score/derivative allocation used by the upstream host. The
32-byte mixed integer/float parameter block is packed at checked field offsets.
Word-sized readback transport preserves native 16-bit buffer strides; an odd
element count has explicit tail padding after the output guard. Exact cases
compare bitwise and fractional cases use `rtol=atol=1e-5` against quantized
float64 references. Original/generated Metal also checks input preservation
and allocation identity. These are kernel-stage checks, not end-to-end
attention backward or upstream host redirection.

The DirectX alias cases use offset-zero views. Simultaneous SRV/UAV access to
one ranged allocation remains unsupported by the process-isolated worker
([#2080](https://github.com/CrossGL/crosstl/issues/2080)). It rejects that request
before submission rather than separating the aliased storage. This limitation
also affects float32; it is independent of binary16 storage encoding.

The attention reduction stage has a separate resident-sequence gate on Windows,
Linux and macOS. Each case executes a prefix of `set`, `add`, `add`, `set` with
distinct source allocations and shared accumulator/output allocations. The
accumulator is initialized once and is not read back or reuploaded between
dispatches. Prefix checks distinguish accumulation from reset behavior and
verify untouched rows, partial workgroups and buffer guards. Dyadic and
fractional inputs use the source storage precision; all output comparisons
are bitwise, including the float32 accumulator and rounded low-precision result.

Windows and original/generated Metal cover float32, float16 and bfloat16.
OpenGL covers float32 and float16, with explicit decoded float32 uploads and
source-scoped bounds derived from the fixture dimensions. OpenGL bfloat16
narrowing remains tracked in [#1488](https://github.com/CrossGL/crosstl/issues/1488).
Metal submits the sequence in one command buffer and verifies every allocation
after completion. Portable drivers perform only the final output readback;
Windows retains native two-byte source/output storage. These are maintained
reduction-stage tests, not a complete attention operation or automatic MLX
host-runtime redirection. No upstream kernels are modified.

The Linux gate executes the same eight derivative configurations for each of
float32 and float16. The 32-byte parameter block uses reflected mixed-field
layout metadata, preserved by the native runtime contract. The test uploads
already-quantized inputs into the generated float32 storage and compares actual
float32 readbacks against decoded source-type references; it does not round
outputs on the host to conceal a missing shader conversion. Original inputs,
uploads, generated modules, parameter layouts and guarded outputs are retained.
Source-qualified index bounds cover the dispatched tile. Actual shared buffer
allocation is tested alongside separate allocations. Bfloat16 derivatives remain
blocked by [issue #1488](https://github.com/CrossGL/crosstl/issues/1488); no complete
attention operation or automatic backend-wide repacking is claimed.

DirectX also requires a reduced native binary16 conversion gate. It shares
258,052 input words with the OpenGL and original/generated Metal controls,
including midpoint neighbors, subnormal boundaries, overflow and signed zeros.
Separate cases exercise implicit and explicit conversion, helper boundaries,
structure fields, vector components and single evaluation. Generated HLSL rounds
the representation bits to nearest-even before constructing a native half value;
the attention derivative buffers retain their two-byte element strides. This
addresses the truncating native-cast behavior tracked in
[issue #1993](https://github.com/CrossGL/crosstl/issues/1993), without changing the
24 derivative configurations or their tolerances. Passing the reduced gate is
not a substitute for passing those MLX cases or the full upstream suite.

Reduced Metal roundtrip tests execute float atomic addition and exchange on
scalar buffers and aggregate fields. They check 513 competing updates, returned
values, final storage and single evaluation of address/value operands. Separate
isolated cases preserve signed zeros, infinities, NaN bit patterns and subnormal
inputs against original-source Metal controls. The macOS workflow requires these
readbacks and retains the original/generated sources, compiled libraries and raw
buffer contents. Float min/max and bitwise atomics remain unsupported; local and
read-only storage are rejected. These tests do not establish portable float
atomic support or a complete gated-delta port.

Unknown predicates, unresolved constraint result types and multiple viable
constrained partials produce diagnostics instead of selecting a primary
declaration with a different layout. Ordering multiple viable partials remains
tracked in [#1987](https://github.com/CrossGL/crosstl/issues/1987). Native
regression tests require 194 exact readback words distinguishing fractional and
integer fields; the same tests run on Windows/DirectX, Linux/OpenGL and
macOS/Metal, with original-source controls on macOS. Parameter inference and named
Boolean free-function constraints have separate native checks. These reduced
tests do not establish complete gated-delta execution or full MLX backend support.

The latest full-corpus scout at MLX commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e` discovered 40 Metal units, 841
include dependencies, and 120 planned target artifacts. It emitted `arange.metal`
for DirectX, OpenGL, and Vulkan plus a Vulkan `arg_reduce.metal` artifact before
the 900-second limit expired without a canonical project report. That recorded
attempt predates durable progress files and therefore does not establish an
active full-corpus coordinate. Current runs write an atomic checkpoint containing
the completed, active, and pending job coordinates, accumulated diagnostics, and
partial artifact matrix. A later invocation resumes only artifacts whose source,
generated output, and source-remap identities still match. CrossGL/crosstl#1376
continues to track bounded materialization runtime.
[#1676](https://github.com/CrossGL/crosstl/issues/1676) remains the
repository-level acceptance target.

A bounded checkpoint probe against the same pin planned 80 DirectX and OpenGL
jobs. It preserved four completed records, identified `binary.metal`/DirectX as
active with 75 jobs pending, and validated the checkpoint after a bare process
interrupt. Resuming the probe retained those four records and returned directly
to the active coordinate. This verifies interruption recovery, not completion
of the full corpus or numerical parity.

Materialization charges configured work budgets to unique reachable entries,
helpers, struct specializations, and actual type-environment resolution. Exact
source-analysis snapshots and span indexes are reused with bounded retention;
the generated source remains unchanged. Artifact metadata keeps reachable
specializations, dependency-discovery work, and pruned eager candidates
separate.

A selected DirectX replay of `quantized.metal` now emits one artifact with zero
translation diagnostics for `affine_quantize_float_gs_32_b_2`. It materializes
five reachable specializations and three concrete records while pruning 110,861
unreachable candidates. The kernel reduces each quantization group with
`simd_min` and `simd_max`, so its result depends on the dispatch shape. The
pinned `quantize_dequantize` host dispatch in `mlx/backend/metal/quantized.cpp`
launches `MTL::Size(thread_group_size, 1, 1)` with up to
`maxTotalThreadsPerThreadgroup` threads, and any multiple of 32 keeps each
group inside one SIMD-group. The project rules therefore give the entry
`[numthreads(64, 1, 1)]` and `[WaveSize(32)]`, which requires Shader Model 6.6.
The generated HLSL is 5,721 bytes with SHA-256
`d9e262e4b274066ff3136ee9091d82e70319ebfc5f8fe64313b90b0525fad52e`.
This path verifies the completed template-member and owner-dependent `constexpr`
work tracked by CrossGL/crosstl#1476 and CrossGL/crosstl#1672. After unreachable
materializations are pruned, this selected float specialization contains no live
native-width 16-bit types, and its report records `requiredCapabilities=[]`.
Native 16-bit HLSL support under
[#1799](https://github.com/CrossGL/crosstl/issues/1799) remains resolved and
validated elsewhere, including the pinned bfloat frontier, but is not required
by this selected specialization. Concrete `static_assert` evaluation under
[#1800](https://github.com/CrossGL/crosstl/issues/1800) is resolved for this
selected entry. Contextual narrowing under
[#1801](https://github.com/CrossGL/crosstl/issues/1801) remains recorded in the
broader DirectX toolchain evidence, but this selected `bits = 2` entry resolves
`OutType` to source `uint32_t` and generated HLSL `uint`. The output buffer's
source `uint8_t` element type is narrower than that accumulator. Its final
store retains this conversion before writing the expanded `uint` resource:
`out_[uint((out_index / uint64_t(writes_per_reduce)))] = (uint(output) & 255u);`.
The artifact contains
no remaining `static_assert`. Its ordinary type contract needs no native-16-bit
profile uplift, while the generated wave intrinsics keep the configured project
and compiler target scoped to `directx-12`. Official DXC validation with profile
`cs_6_6`, no `-enable-16bit-types`, and `-WX` passes.
The locally generated DXIL was nonempty; its byte size is not treated as a
cross-version compiler invariant. This evidence covers translation and compiler
acceptance only; it does not claim runtime execution or numerical parity.

The adjacent DirectX entry `affine_gather_qmv_fast_float_gs_32_b_2` also emits
one artifact with zero translation diagnostics. It materializes 10 reachable
specializations and eight concrete records while pruning 110,861 unreachable
candidates. The specialized `load_vector_float_float_16_2` helper retains the
caller's `thread U x_thread[values_per_thread]` storage with
`values_per_thread = 16`; HLSL represents it as an `inout float[16]` plus a
base offset and proves all four writes in each `i += 4` iteration. The
`qdot_float_16_2` helper retains the read-only `uint8_t` view over its
`uint32_t` storage resource. Each byte read selects the backing word and lane,
and the helper call composes the root word offset, byte-view offset, row offset,
and local alias offset without treating bytes as typed 32-bit elements.

The gather path also materializes the overloaded `elem_to_loc` helper as
`elem_to_loc_uint32_t`, preserving the source `uint32_t` index type. Its HLSL
call carries the independent shape and stride resource offsets, and no
unresolved `elem_to_loc(...)` call remains in the artifact.

The source `adjust_matrix_offsets` helper receives five storage pointers by
mutable reference. DirectX keeps each resource handle unchanged and passes its
logical offset as `inout int64_t`. The entry owns the offset variables, the
helper updates them, and the subsequent `qmv_fast_impl_float_32_2` call consumes
all five updated values. This preserves source pointer rebasing without passing
resource handles by reference or discarding writes to by-value offsets.

The pinned `gather_qmv` host dispatch in `mlx/backend/metal/quantized.cpp` sets
`bk = 32` and `MTL::Size group_dims(bk, 2, 1)`. The project rule therefore
emits `[numthreads(32, 2, 1)]`. The kernel's simdgroup indices require a
32-lane subgroup, so the generated HLSL also emits `[WaveSize(32)]` and requires
Shader Model 6.6. The resulting artifact is 17,917 bytes with SHA-256
`4ce3a1b462477d26047500477b7d53c1219a2f57238e1a80199a5c759043914a`.
Windows CI compiles it with DXC profile `cs_6_6` and `-WX`. This is selected-entry
evidence for the fixed-array alias work tracked by
[#1497](https://github.com/CrossGL/crosstl/issues/1497) and the read-only storage
view work tracked by [#1546](https://github.com/CrossGL/crosstl/issues/1546),
the resource pointer-offset contract tracked by
[#1518](https://github.com/CrossGL/crosstl/issues/1518),
with the broader subgroup contract tracked by
[#1894](https://github.com/CrossGL/crosstl/issues/1894). Issue
[#1786](https://github.com/CrossGL/crosstl/issues/1786) established the exact
subgroup-width metadata and DirectX enforcement model. The remaining issues
cover broader cross-target, writable-view, alignment, subgroup fallback, and
runtime acceptance criteria. This proof does not dispatch the kernel through
Direct3D, run MLX tests, or claim numerical parity.

The current-corpus `affine_quantize_float_gs_32_b_2` entry now has a
selected OpenGL native-loader proof at commit
`846d176227a0ac13d2667e58d2bb68b322109ab0`. The project configuration supplies
three source-qualified index-range assertions for `in_index + i`, `gindex`, and
`out_index / writes_per_reduce`, each with inclusive bounds
`[0, 2147483647]`. These are explicit host/runtime portability preconditions
used to justify legal GLSL `uint` subscripts; they are not inferred or enforced
at runtime.

The target-scoped Metal option
`project.source_options.metal.target_options.opengl.software_subgroup_width = 32`
selects a fail-closed software implementation of this entry's scalar
`WaveActiveMin(float)`, `WaveActiveMax(float)`, and
`WaveShuffleDown(uint, int)` operations. This selected proof uses one compute
entry with local size `[32, 1, 1]`. The bounded mode can also partition a
concrete workgroup of at most 1,024 invocations into multiple 32-lane logical
subgroups when local size X is divisible by 32; unsupported operations, payload
types, helper ownership, raw subgroup builtins, or unproven lane-dependent
control flow still reject translation. The default hardware KHR subgroup path
is unchanged.
The software artifact uses collision-safe shared scratch storage and eight
barriers, emits `CROSSTL_SOFTWARE_SUBGROUP_WIDTH`, and emits no KHR subgroup
extension, `gl_Subgroup*` use, `CROSSTL_REQUIRED_SUBGROUP_WIDTH`, or hardware
subgroup-width runtime metadata.

The exact generated GLSL is 7,949 bytes with SHA-256
`61cc1cf6f33ecab9919191db3f68bf57267d549fb3d638d595d464e68d2f494c`.
Materialization records three concrete and eight reachable specializations, zero
dependency-discovery work, and 104,702 pruned candidates. Linux CI compiles it
for OpenGL/SPIR-V 1.3 with `glslangValidator`; `spirv-val` accepts the module,
whose disassembly contains eight control barriers and no group-nonuniform
instruction. The runtime package reflects `wBuffer` at binding 0 and the
`out_Buffer`, `scalesBuffer`, and `biasesBuffer` read-write resources at
bindings 1 through 3. Through surfaceless EGL and Mesa software OpenGL, one
32-thread dispatch transforms `[0, 1, 2, 3]` repeated eight times into eight
packed `uint32` values of `27`, scale `-1`, and bias `3`.
A second dispatch uses `[0, 0.5, 1.5, 3]` repeated eight times and requires
eight packed values of `47`, with the same scale and bias. This midpoint case
checks Metal's halfway-away rounding; the generated helper also preserves the
source's narrow shuffle operand conversion. The shader reference was reviewed
against original Metal endpoint and midpoint readbacks before being updated.

This is exact translation, packaging, native execution, and numerical parity
for these two bounded affine-quantize workloads. It does not redirect the
MLX host runtime, run MLX's test suite, cover other quantized entries, or turn
the constrained software subgroup mode into a general divergent-subgroup
implementation. [#1515](https://github.com/CrossGL/crosstl/issues/1515) records
the completed index-width normalization contract, while
[#1894](https://github.com/CrossGL/crosstl/issues/1894) continues to track
broader subgroup fallback coverage.

The historical `4367c73b60541ddd5a266ce4644fd93d20223b6e` corpus retains a
separate compile-only OpenGL gate for `affine_gather_qmv_fast_float_gs_32_b_2`.
It emits a 16,040-byte GLSL artifact with SHA-256
`402565bb6227de79fdf324193857653ddead7da0d010acea51ebcd60b8f264ea`, zero
project diagnostics, 10 reachable specializations, and eight concrete
materializations. The project report retains the pinned `32 x 2 x 1`
workgroup rule and nine explicit index-range preconditions in
`[0, 2147483647]`: four for the gather resource lookups and five for the
`adjust_matrix_offsets` batch rebases (`x_idx * x_strides[0]`,
`w_idx * w_strides[0]`, `w_idx * s_strides[0]`, `w_idx * b_strides[0]`, and
the broadcast bias offset `idx.z`), which OpenGL keeps in signed `int` logical
offsets. The generated `load_vector` and `qdot` helpers preserve the
16-element private array. Reinterpreted storage pointers pass their root byte
base separately from their logical byte-view offset, so `qdot` composes the
incoming word offset, row offset, and loop index without capturing caller-local
expressions or scaling the word offset twice. Linux CI compiles the artifact for
OpenGL/SPIR-V 1.3 and validates the module with `spirv-val`. This advances the
cross-target contracts tracked by [#1497](https://github.com/CrossGL/crosstl/issues/1497),
[#1518](https://github.com/CrossGL/crosstl/issues/1518), and
[#1546](https://github.com/CrossGL/crosstl/issues/1546); their broader acceptance
criteria remain open. This remains compile-only evidence and does not claim
OpenGL dispatch, MLX test execution, or numerical parity.

CrossGL/crosstl#1659 is complete; resource-register relocation no longer blocks
the selected aggregate DirectX replay. The checked-in evidence also records
full-kernel Vulkan replays of `fp_quantized.metal` and `quantized_nax.metal` as
failed after selected materialization contracts. Only the explicitly documented
reduced quantized fixtures carry validator evidence.
CrossGL/crosstl#1660 tracks resource coherence and volatility qualifiers that
are currently absent from Metal round-trip artifacts.
Native toolchain validation now runs on the target operating systems, completing
the coverage tracked by CrossGL/crosstl#1312. The bounded Direct3D and OpenGL
compute runtime drivers tracked by CrossGL/crosstl#1472 and
CrossGL/crosstl#1516 are also complete. Packaged artifacts can now be dispatched
through the native-loader bridge; remaining source-specific execution gaps are
recorded with the individual proofs below. CrossGL/crosstl#1388 still tracks the
broader artifact execution metadata contract. OpenGL type aliases are inlined
before host-interface reflection and are not exposed as runtime resources.
CrossGL/crosstl#1471
tracks entry-point ownership for reflected constants in runtime reports.
CrossGL/crosstl#1474 is represented by exact per-artifact DirectX
bfloat16 storage and conversion evidence in the pinned project report. The
harness fails closed unless each emitted source retains the exact
`bfloat16Lowering` object and corresponding `requiredCapabilities` list; this
does not extend the bounded runtime proof to bfloat16 or claim numerical parity.
Every custom DXC invocation derives its effective profile and compiler
arguments from the emitted HLSL through `crosstl.project.directx_toolchain`.
Generated `float16_t`, `int16_t`, or `uint16_t` types select at least Shader
Model 6.2 and add `-enable-16bit-types`; ordinary HLSL retains its selected
profile and command. These checks do not prove Direct3D 10 or 11 compatibility;
CrossGL/crosstl#1670 tracks explicit target profiles, feature gates, and
compiler selection. CrossGL/crosstl#1669 tracks
the fixed arrays of resource aliases introduced by the pinned revision's wide
quantized matrix-vector helpers. CrossGL/crosstl#1671 originally tracked
workgroup backing provenance through nested FFT helper parameters. A dedicated
DirectX project replay now selects the forward complex 256-point entry
`fft_mem_256_float2_float2` from the complete pinned `fft.metal` source. The
host planner's `MTL::Size(1, threadgroup_batch_size, threads_per_fft)` dispatch
maps to `[numthreads(1, 1, 64)]`; the report and generated HLSL must retain that
axis order. The replay materializes 24 reachable template specializations and
21 of the 22 configured function constants. Function constant 3 is pruned
because its Rader transform branch is unreachable for this power-of-two plan.
The generated artifact contains no first-class workgroup pointer residue and is
locked by SHA-256 and byte count. Windows CI compiles it with DXC using
`cs_6_2`, `-enable-16bit-types`, and warnings as errors. It then packages and
reflects the artifact, requiring 8-byte `float2` strides for the input and
output resources and a 16-byte constant-buffer allocation for the generated
`uint3` dispatch input. The native-loader request derives that input from the
physical workgroup count before Direct3D 12 WARP dispatch and readback.

The runtime check supplies five inputs for this 256-point plan: an index-1
unit impulse, seeded float32 complex data, a complex constant, alternating real
values, and a complex impulse at the final element. It compares all outputs
with analytical references or a double-precision direct DFT, using `2e-4`
absolute and relative tolerance. Eight trailing scalar guards must remain
exactly unchanged after each dispatch. Each native test translates and packages
the source once, then executes all five inputs in the existing CI job. This
checks 2,560 result values and 40 guards for one bounded transform plan, not
general FFT coverage. It does not redirect the
MLX host runtime, run the MLX test suite, or establish broad FFT or backend
parity. The aggregate DirectX source materialization limit remains tracked by
[#1916](https://github.com/CrossGL/crosstl/issues/1916); broader runtime grid and
resource-layout contracts remain tracked by
[#1542](https://github.com/CrossGL/crosstl/issues/1542) and
[#1543](https://github.com/CrossGL/crosstl/issues/1543).

The current-corpus fixture selects commit
`846d176227a0ac13d2667e58d2bb68b322109ab0`, where MLX generalizes FFT
workgroup storage through `FFTIOTypeTraits` and adds function constant 22 for
the Bluestein twiddle-table path. Replaying the same 256-point entry with that
constant set to `false` materializes 37 specializations, records 42 reachable
specializations before pruning, and concretizes 21 of 23 configured function
constants. The source reaches two compatible `ReadWriter_float2_float2__load`
overload bodies. DirectX workgroup-pointer reachability now analyzes both
bodies, merges their forwarded calls and access ranges, and retains the
concrete `crosstl_workgroup_shared_in[256]` backing identity. Generated
helpers transport workgroup pointers as typed resource handles carrying the
backing identity and an integer offset. The omitted default `twiddles =
nullptr` argument is carried into the CrossGL intermediate before target
lowering. Because the materialized `use_twiddle_table = false` call chain makes
every twiddle dereference unreachable, DirectX removes that unobserved resource
parameter through the forwarding chain rather than inventing a backing buffer.
A null pointer that can be observed or dereferenced still fails closed.

The current source emits a 175,427-byte HLSL artifact with SHA-256
`33c9aa89cee60ba565ea744ad3452ee4a2213d0a991098e64aa093c3b31b6b37`
and zero project diagnostics. All 20 native-16 ``power`` shift counts are
explicitly promoted to ``int`` before HLSL shifting, matching Metal/C++ integer
promotion rather than retaining minimum-precision count semantics. Windows CI
compiles it with DXC using `cs_6_2`,
`-enable-16bit-types`, and warnings as errors, then packages and dispatches it
through Direct3D 12 WARP. The same five-input workload, reflected
resource layouts, physical workgroup count, and `2e-4` output tolerances are
required for this exact current checkout. This lowering does not permit
arbitrary first-class workgroup pointers: roots without a concrete shared-array
identity or extent still fail closed, and
[#1518](https://github.com/CrossGL/crosstl/issues/1518) continues to track the
broader pointer-offset contract. The historical proof remains separately
recorded under its own commit provenance; it is no longer being used as a
substitute for current-corpus evidence. Its reviewed artifact is 144,151 bytes,
SHA-256 `d611594837648c52f0fc0e7847c1ebdd660f158c830eccc0002aa3dd597bbab9`.
Both artifact revisions preserve reflected bindings and dispatch dimensions;
native Windows numerical execution remains required before merging a reference
update. Successful DXC compilation alone does not satisfy that requirement.

[Windows run 37525218285](https://github.com/CrossGL/crosstl/actions/runs/37525218285/job/112481552970)
verifies the preceding 176,506-byte and 145,277-byte artifacts at `a688ad15`.
The `mlx-fft-directx` artifact
(11443897833) retains all ten input fixtures, dispatch plans, compiler commands,
HLSL/DXIL snapshots and returned values. Independent DFT recomputation from the
saved inputs verifies 5,120 values and 80 exact guards across the two source
pins. Maximum absolute error is below `6.63e-5`, within the unchanged tolerances.
Outputs start at `-999.0`, not the expected answers. Source identities match
their contracts, and both compiled modules have SHA-256
`59c265eecbfff3b88b2746f535e055e1cf976fa0d76755d4eb7157a79c87f81b`.
This verifies these five controls for each historical pin; it does not establish
the `9c3d3557` corpus, arbitrary FFT plans or physical driver-upload bytes.

The current FFT source now also emits a 120,398-byte GLSL artifact with
SHA-256
`1b7abf82bf713abc5640208cc0606232b228f683473f603debcb8898954a1f26`.
Resource-aggregate lowering currently retains one file-level source mapping,
not the preceding artifact's 84 finer-grained mappings. Diagnostics preserve
the original source path but do not yet identify individual rewritten lines.
OpenGL keeps the 21 reachable function constants
as 21 deferred specialization constants, each guarded by
`#if defined(GL_SPIRV) || defined(VULKAN)` so that a direct GLSL compile
without specialization stops at `#error`, and uses the same `[1, 1, 64]`
workgroup contract. A fifth current-source index assertion bounds
`batch_idx + index + r`. A sixth assertion bounds the resource helper's
`reference.offset + index` to `[0,255]` for this exact fixture: buffers are
unrebased, `batch_size=1`, and `min(2*lane + 128*e,254)+r` covers `[0,255]`
for `lane` in `[0,63]`, `e` in `[0,1]`, and `r` in `[0,1]`. This assertion
must not be reused for other batch sizes, dispatches, or rebased resources.
Generic specialization lowering now follows a
statically null storage pointer through direct forwarding calls and omits it
only when every reachable use is dead; observable null uses still fail closed.
The GLSL generator also decodes materialized Metal `vec<T,N>` constructor
names and emits every collision-safe overloaded resource-specialization body.
The generated module compiles with `glslangValidator`, passes `spirv-val`, and
has 19 control barriers and no group-nonuniform instructions.

Host reflection now records each runtime `vec2` SSBO element as `float32x2`
with 8-byte size, stride, and alignment under `std430`; the two scalar argument
blocks retain their 16-byte `std140` block identities. The resulting four-
resource native ABI has no blocked variants and publishes a ready deferred
SPIR-V request. Linux CI specializes and dispatches that request through a
surfaceless Mesa llvmpipe OpenGL context. All five inputs must match the
independent references within the required `2e-4` tolerances, with exact guard
preservation. The first execution must publish the verified compilation cache;
the remaining four must use it. This is one exact current-pinned plan and does not establish full
FFT, MLX host-runtime, or backend parity.

The shared-vector lowering accepts fixed-size arrays of two- or four-component
32-bit numeric vectors and preserves typed loads, stores and element offsets.
Three-component layouts, unknown extents and unsupported qualifiers remain
rejected. General regressions exercise rebased arrays, three workgroups and
exact output and shared-storage guards on Metal and OpenGL. The revised
FFT passes the five unchanged OpenGL controls locally. Its DirectX modules
compile with unchanged reflection; Windows numerical execution of the new
source identities remains required. Earlier Windows results are retained under
their original artifact hashes and do not certify these revised artifacts.

A dedicated project replay now selects `fft_mem_256_float2_float2` from the
pinned `fft.metal` source and translates it to one standalone OpenGL compute
shader, as the DirectX replay does. GLSL fixes `local_size` in the shader and
the kernel uses threadgroup memory and barriers, so the translator refuses a
host-sized workgroup; the replay supplies the `(1, 1, 64)` entry workgroup-size
rule and the function constants MLX uses for a power-of-two 256-point
transform. The report records 24 unique reachable template specializations, 21
concrete function constants, no unsupported materializations, and no project
diagnostics. Four signed host-index assertions and one entry-point-scoped
workgroup access assertion make the dispatch preconditions explicit. The
generated workgroup and storage load/store helpers do not yet inherit those
bounds (issue #2178), so four helper-scoped assertions restate the 256-element
`shared_in` extent and the signed device range. The Linux proof compiles the
emitted GLSL to OpenGL SPIR-V 1.3 with `glslangValidator` and validates the
binary with `spirv-val`. This proves artifact construction and native toolchain
acceptance; OpenGL runtime dispatch and numerical parity with MLX's Metal
backend remain outside this check.

The pinned `gemv.metal` DirectX compiler frontier verifies source SHA-256
`c34db77e61c1fea01f7f5d319a0bec1029a253e54d66bbce9009f32fe828ce9f` and
source size 5,383 bytes before translation. The project report must contain one
clean translated artifact, all 226 materializations with no unsupported
records, no unresolved materialization residue, and no standalone pure
value-discard statements such as `lid;`. Its generated SHA-256 and byte count
are checked against the emitted HLSL. Project configuration sets
`[project.workgroup_size_rules]` for `gemv.metal` to `[32, "BN", "BM"]`; the
report must retain the normalized `["32", "BN", "BM"]` rule. Exactly 224 of the
226 materializations must be host-named; the other two are signature-selected
`elem_to_loc<uint>` and `elem_to_loc<uint32_t>` helpers. Exactly 224 report
execution entries must join the host-named records by
`(hostName, materializedName)` identity. The artifact must expose the exact
target set `CSMain`, `CSMain_2`, ..., `CSMain_224`, independently of report or
materialization list order.

The resolved report sizes must be exactly `[32, 1, 1]`, `[32, 1, 4]`,
`[32, 1, 8]`, `[32, 2, 1]`, `[32, 4, 1]`, `[32, 8, 1]`, and `[32, 16, 1]`.
For every target entry, the emitted `numthreads` declaration must equal that
entry's report contract. This establishes exact workgroup-size specialization
for the generated aggregate artifact. DXC compiles representative scalar,
complex/Wave, and gather/constant-pointer paths (`CSMain`, `CSMain_85`, and
`CSMain_113`) with `cs_6_2` and `-enable-16bit-types`; all three invocations
must produce zero diagnostics. A second `lib_6_6` invocation retains
`-enable-16bit-types` while exporting and code-generating all 224 functions in
one DXIL library.

The library compile admits exactly 224 `numthreads ignored without accompanying
shader attribute` warnings caused by using a library profile. The gate derives
the expected warning source-line counts from the seven generated `numthreads`
forms and requires exact severity, message, source expression, and count
matches. Any unused-value warning, error, or other diagnostic fails the gate.
[#1786](https://github.com/CrossGL/crosstl/issues/1786) established the exact
wave-size specialization contract. This existing GEMV library frontier does not
apply that contract, so library compilation proves that DXC accepts and
code-generates every exported function but does not establish wave semantics,
runtime execution, numerical parity, or whole-kernel semantic validity.

The pinned `gemv.metal` OpenGL gate uses the same 4,096-specialization limit and
2,097,152-item materialization work budget. It requires SHA-256
`c34db77e61c1fea01f7f5d319a0bec1029a253e54d66bbce9009f32fe828ce9f`, one
source unit, all 226 specializations materialized without unsupported records,
and all 224 entry-scoped GLSL artifacts emitted without diagnostics. Every
artifact must retain its source entry identity, generated hash, source map,
source remap, resolved workgroup size, and exact subgroup-width contract.

The project configuration records `[32, "BN", "BM"]` as the workgroup-size
rule and 32 as the subgroup width. Generated shaders require
`GL_KHR_shader_subgroup_basic` and return before dispatch work when
`gl_SubgroupSize` differs from 32; the host must also query
`GL_SUBGROUP_SIZE_KHR` and reject a mismatch before dispatch. Five source-scoped
index-range assertions record the host preconditions that permit MLX's 64-bit
batch and gathered-matrix indices to narrow to OpenGL's 32-bit index domain.
These are explicit runtime preconditions, not inferred ranges, and this harness
does not enforce them at runtime.

Linux CI compiles all 224 artifacts for OpenGL SPIR-V 1.3 with
`glslangValidator` and validates every resulting SPIR-V 1.3 module with
`spirv-val`. [#1894](https://github.com/CrossGL/crosstl/issues/1894) tracks a
semantics-preserving fallback for devices without the required native subgroup
width. The gate proves complete translation and compiler acceptance under the
recorded project contracts; it does not establish runtime execution, host
integration, or numerical parity.

The checked
[`contracts/gemv.native-loader.dispatch.json`](contracts/gemv.native-loader.dispatch.json)
contract selects `gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0` at the
historical `846d176227a0ac13d2667e58d2bb68b322109ab0` revision. Its 6,981-byte
`gemv.metal` source has SHA-256
`0bd8bde0c867a17c345a3651f9f0a6c2909e0c74e76ea2a08f373fe4dcafaeda`.
The host-derived `gemv_axbpy` branch covers one contiguous float32 vector-matrix
product with `M=1`, `N=32`, `K=32`, no gathered or non-contiguous batch, and no
axpby bias path, matching
`python/tests/test_blas.py::TestBlas::test_matrix_vector`. It fixes parameters
`BM=1`, `BN=2`, `SM=8`, `SN=4`, `TM=4`, and `TN=4`, workgroup
`[32, 2, 1]`, subgroup width 32, and dispatch `[1, 1, 1]`. The normalized
contract identity is
`sha256:6b3bb18d130159f13874f06668b536fe4b9270ffbb2a1f44b6d9aac257aba7e4`;
its single variant and artifact identities are
`sha256:acaba2ec4813a364b06d95a5136bda80351797591a4d9f0b3d195f85da287fe3`
and
`sha256:34eab189b10cc699f06f4cbed04faae41a2658a2a3665a6866ed987f5946949a`.

Entry-scoped translation materializes only the selected GEMV and
`elem_to_loc_uint`, with no unsupported record or project diagnostic. The
9,395-byte HLSL has SHA-256
`a78e20a5906b0112db43aa7b6a587f412c944f7e16a4fbda8e4dd3ca2a64df75`,
retains `[numthreads(32, 2, 1)]` and `[WaveSize(32)]`, and passes official DXC
1.9.2602.24 under `cs_6_6`, `-enable-16bit-types`, and warnings as errors.
`GEMVTKernel::run` repeats the entry's `[[buffer(n)]]` attributes on ordinary
pointer parameters, so the HLSL helper takes each `mat`, `in_vec`, `bias` and
`out_vec` view as a resource plus an `int64_t` element offset and the entry
forwards its `tid.z` batch rebases; an earlier artifact indexed the unrebased
roots whenever `tid.z` was nonzero.
Direct3D does not guarantee that a multidimensional workgroup's flattened
`SV_GroupIndex` values are contiguous within each physical wave, so this entry
explicitly enables target-scoped 32-lane software subgroups. Logical subgroup
and lane IDs are `SV_GroupIndex / 32` and `SV_GroupIndex % 32`; a 64-float
`groupshared` array carries each shuffle, with two
`GroupMemoryBarrierWithGroupSync` calls. The source lane is validated before
addition, preventing unsigned-delta wrap, and an out-of-range shuffle returns
the calling invocation's value. The artifact contains no `WaveReadLaneAt`,
`WaveGetLaneIndex`, or physical-wave atomic allocator. `[WaveSize(32)]` remains
a source/reflection contract, not a dependency on physical lane topology.
The kernel's main loop starts each iteration with a `threadgroup_barrier` under
`if (out_col < out_vec_size)`; both targets now run that loop, and so its
barrier, in every invocation and keep the column clamp, loads, accumulation and
leftover pass under the guard, so a partial last column block cannot strand a
barrier. The earlier 9,099-byte HLSL
(`e26c5e45a11d9ea861468bbf641bcfe8905d8c023b3de9439c03e9f3095065b7`) and
7,752-byte GLSL
(`7a82ff4b75a71aaebd4a5b5a3f62882d82aca2b24a1ed27b9ec6bece9a87e8ea`) kept the
barrier inside the guard and differ only there.

The replaced 8,410-byte physical-wave artifact is retained as rejected
diagnostic evidence under SHA-256
`f8f1107d0de251fd300c7a16ce6638796bd08dd2eadd8f7959e37c78d0aa170d`.
Windows workflow run 33268998061, job 99143984804 mismatched all 32 outputs:
its reduction substituted logical lanes 5 through 8 with physical lanes 21
through 24, with maximum absolute error 1.90625. That exact signature rules out
a tolerance adjustment or merely guarding invalid high-lane reads.

OpenGL keeps each batch rebase in a signed `int` logical offset, so the
target asserts four expressions in `[0, 2^31 - 1]`: the selected matrix index
`uint64(bm + tm) * marix_ld + out_col + tn`, its complete rebased form
`mat + uint64(bm + tm) * marix_ld + out_col + tn`, and the two batch offsets
`tid.z * buffer_load(vector_batch_stride, 0)` and
`tid.z * buffer_load(matrix_batch_stride, 0)` (the lowered spelling of
`vector_batch_stride[0]` and `matrix_batch_stride[0]`). The batch offsets reach
the helper as `int` logical offsets. Its 8,038-byte GLSL has SHA-256
`5631a0011d909fa5fae3d720dea11738031a18acaeba9b68472fe35925146274`
and partitions the 64-thread workgroup into two logical 32-lane subgroups with
a 64-float shared shuffle scratch array.

The reference update compares complete historical/current bodies and all 15
reflected resources per target. HLSL changes retain source-width matrix/stride
arithmetic and carry the logical invocation index through helper calls. Both
targets preserve the source's 16-bit shuffle offset conversion; OpenGL also
adds explicit shared-memory ordering and checks the source lane before addition.
Bindings, entry geometry, template materializations and upstream source hashes
are unchanged. Both historical references are reproduced exactly and compiled
before comparison; all four historical/current artifacts pass their validators.
The subsequent empty-struct compatibility update adds one padding member to
each of 14 unused HLSL declarations. Its complete executable body and reflected
interface are unchanged, and strict DXC compilation produces byte-identical
DXIL to the accepted 8,382-byte reference. The GLSL source and compiled module
are unchanged. Native execution remains a separate required check.

Both target-specific software-subgroup analyses recognize either `value > 0`
or the integral-equivalent `value >= 1` as a canonical positive-to-zero
halving loop when paired with `/= 2` or `>>= 1`. That admits the source's
`sm >= 1; sm >>= 1` segmented shuffle reduction. DirectX additionally requires
one bounded compute entry, a concrete width-compatible workgroup, explicit
calling-invocation fallback, a supported scalar shuffle, an unambiguous helper
call graph, logical invocation identity, and statically uniform control flow;
violations fail before artifact emission. OpenGL retains rejection for wider
bounds, mutation, nontermination, escaping control flow, indirect calls, and
nested helper calls. `glslangValidator` and `spirv-val` accept the emitted
OpenGL module; SPIR-V contains three control barriers and no group-nonuniform
instruction or hardware-subgroup extension.

Both runtime packages expose the same 15 logical resources: matrix, vector,
bias placeholder, writable output, batch shape, three signed 64-bit batch
strides, and seven scalar argument blocks. A deterministic binary-fraction
workload uses `vector[row] = (row + 1) / 32` and
`matrix[row,column] = (row + column + 2) / 64`; output column `c` must equal
`5.5859375 + 0.2578125 * (c + 1)` for all 32 columns at `1e-5` absolute and
relative tolerance. Linux arm64 Mesa llvmpipe executes and reads back this
software-subgroup workload in required mode, and Windows CI requires the same
request through Direct3D 12 WARP.
The local OpenGL and hosted Windows readbacks match an independent
exact-rational calculation from the uploaded binary32 inputs for all 32 columns.
Windows run 37417907211, job 112120535710 retains the executed 7,108-byte DXIL
under SHA-256
`8df8a620174a9813e8f2a04437710f035b1ac369c3c2de5ae74e6cf051e6aecc`.
That job subsequently failed at the separate MXFP4 reference check; it is not a
passing whole-project run. Existing Windows/Linux
jobs retain the project report before identity checks, generated sources,
runtime package, compiler records, dispatch inputs, results and JUnit output,
including on failure. No extra runner or relaxed tolerance is required.

This bounded proof does not replace the historical 224-entry aggregate gates:
it adds numerical execution for one host-valid entry at `846d1762`, not the
current `9c3d3557` corpus. Gather, wide, batched, axpby, and the remaining host-named GEMV
entries, MLX host redirection, the full MLX test suite, and selected-entry Metal
compiler validation remain outside the claim. The separate native Metal
aggregate baseline continues to cover the existing round-trip boundary.

The checked
[`contracts/fp_quantized.native-loader.dispatch.json`](contracts/fp_quantized.native-loader.dispatch.json)
contract selects the historical `846d1762` entry
`mxfp4_quantize_dequantize_float_gs_32_b_4_hgs_false` from the 9,700-byte
`fp_quantized.metal` source with SHA-256
`ef4ba099710a63a0b5d27d3e5ce69a8528bee8f1757805aa606c8d8e43de18d4`.
It records the exact host branch from `quantized.cpp` and implementation in
`fp_quantized.h`: float32 MXFP4, group size 32, four payload bits, row-contiguous
layout, and no global scale. The host formula assigns one value per thread,
selects workgroup `[32, 1, 1]`, and dispatches one workgroup. The normalized
contract identity is
`sha256:5256e32b364ac303a6873f28b5ac3e9a1a811ac5c38bc41a977bce9191a025ed`;
its single variant and artifact identities are
`sha256:ebd6ab3f40f5839764f592943180ba64f11a66f10a5561b9468019032c04df8a`
and
`sha256:bde1bfa31c116a52a1dc3b6e546dfa2ee43dc968719393ba04f629b4e2d95319`.

Entry-scoped translation materializes only the selected specialization with
`T=float`, `group_size=32`, `bits=4`, and `has_global_scale=false`. The
10,476-byte HLSL has SHA-256
`c05226f8c415d1e7cb6b25e9eccaed3f4c8afda750aac21fed03c82fb6569d33`,
retains `[numthreads(32, 1, 1)]` and `[WaveSize(32)]`, and passes DXC under
`cs_6_6`, `-enable-16bit-types`, and warnings as errors. Its compiled DXIL is
4,836 bytes. This artifact explicitly enables the DirectX-only
`project.source_options.metal.target_options.directx.widen_native_float16`
mode. Source `as_type<float16_t>(uint16_t)` reconstructs its exact payload as
float32 with integer IEEE-754 masks, and logical `float16_t` locals, function
parameters, and returns stay widened through the selected arithmetic path.
Optimized DXIL contains `uitofp i32` and `fmul float`, with no
`LegacyF16ToF32`, `half`, `fptrunc`, or `fpext`. Default DirectX native
binary16 lowering remains exact `asfloat16`/`asint16`/`asuint16` and is
unchanged unless this source-scoped option is enabled.

The previously accepted 10,335-byte source had SHA-256
`8d2c5257a8d90310d9a96051bd4243c694bd6fa32ec2a80b4f221a6ef287ea37`.
The complete source difference introduces an explicit `fp4_e2m1` construction
helper and uses it at the single dequantization call. Strict DXC compilation
of both sources produces byte-identical modules under SHA-256
`acf03bcb4138c46ac5fc68bcb0f7ce84795f6e47a2c0260f99c281bffacd3569`.
The source pin, materialization, bindings, dispatch contract and numerical
workload are unchanged. This source-identity refresh allows the required Windows
native-loader test to reach execution; compilation is not a new numerical pass.

Four rejected Windows artifacts establish why every native-half boundary must
be absent. The 7,809-byte HLSL under SHA-256
`3591e38d20a612b4061fe3154ef0ea3deb035283294fbd27376ef90627569361`
produced numeric `uitofp i16 ... to half`; workflow run 33271117475, job
99149649480 collapsed all 28 nonzero values to signed zero on WARP. Exact
`bitcast i16 ... to half` produced same-size HLSL under SHA-256
`4e8044758d65b6b2c189092ce56fff3c5ba7948221883de490c1a4b9c5563352`,
but run 33272842347, job 99154326814 still consumed the intentionally
constructed binary16 subnormal in `fmul half` and produced the same 28 signed
zeros. Moving the multiply to float32 produced 7,909-byte HLSL under SHA-256
`938ca6fac1c47ea633453836b5d76833c294853bb92d6a410a2c4772dd7fa627`,
but `dx.op.legacyF16ToF32` yielded the identical result in run 33274360343,
job 99158370210. Integer decoding then produced 9,240-byte HLSL under SHA-256
`936088a24a6b575e50dc97e16a4c0dca63a76200ddd94d5211e4bf312fec1625`,
but its remaining `fptrunc float to half`, half sign operation, and
`fpext half to float` still returned the same 28 signed zeros in run
33275550062, job 99161501105. Removing every half instruction produced the
9,118-byte artifact under SHA-256
`7afdc612f9091ae47abca8c4fd9d2171e8ea42c6539e02a40bbad2de7d1a7c6a`,
but run 33277494856, job 99166677942 still returned the same signed zeros.
Its DXIL exposed the actual remaining defect: Metal/C++ scalar integer promotion
was missing, so `(uint16_t(bits) << 23)` became a 16-bit shift reduced to 7 and
masked with `0xffff`. The corrected HLSL emits
`int(uint16_t(bits)) << 23`; DXIL now shifts the 32-bit value by 23 before
`asfloat`, while retaining zero native-half instructions.

The 11,932-byte GLSL has SHA-256
`3aae7fd8aede237351ea9eeaff8b22bcecb49f920c91451ab021cf1f1ff52451`,
uses one explicit 32-lane software subgroup for `WaveActiveMax(float)`, and
passes `glslangValidator` and `spirv-val`. Its 16,720-byte SPIR-V has three
control barriers, no group-nonuniform instruction, and local size
`[32, 1, 1]`. Because GLSL widens source binary16 values to float32, the same
source bitcast decodes the low 16-bit payload with integer operations before
constructing the float32 representation. This preserves binary16 subnormals
without relying on a target half-conversion instruction's denormal behavior.
The decoder replaces `unpackHalf2x16` at the FP4 conversion; no numerical
tolerance, input, dispatch geometry, or resource binding changes.

The reference update compares complete historical/current bodies and all seven
reflected resources across both targets. HLSL preserves eight-bit assignments,
FP4 sign bits, and source-width index arithmetic. GLSL rounds the FP4 decoding
multiply at its source half-precision boundary, adds shared-memory ordering,
and preserves the scale selection through a bitwise select. All 16 FP4 payloads
decode to exact binary fractions at that rounding boundary. Both targets remove
unused FP8 E4M3 conversion functions; the selected four-bit path retains its FP4
conversion. Resource bindings, element widths, dispatch geometry, and template
materializations are unchanged. The non-required HLSL `sign_bit` metadata
retains the same 0/8 values with explicit byte conversion. Both historical
references reproduce exactly, and all four old/new artifacts pass compilation.
The subsequent HLSL update adds empty-struct padding and explicit integer
types to three constant shifts. Strict compilation produces the same DXIL
as the accepted 9,816-byte source. The GLSL decoder update changes SPIR-V and
requires separate numerical execution; compiler validation alone is not a
runtime parity claim.

A subsequent template-scope correction removes only the unused
`integral_constant_res_t_res` structure, reducing the reachable record count
from nine to eight. The prior HLSL identity
`41852207113971342d1acbf07fe4066168601bb4db19479373a2a40c36347724`
and GLSL identity
`aba7ea0ab5256e12d1ce0893c15b9522aa34c2dda075a794896e2f1bef051868`
are retained here for comparison. Complete source comparisons and strict
recompilation confirm unchanged reflected interfaces and byte-identical DXIL
and SPIR-V. The selected specialization, workload, dispatch and numerical
assertions are unchanged; native execution remains required in the existing jobs.

The current references also include Metal-compatible halfway-away-from-zero
rounding. Compared with the preceding HLSL identity
`c25bb1bb9d47cbec9d94c732caf88b8e6ae1e7501744ce87f5371e1e63f29eb7`
and GLSL identity
`dc23d056d38464ba0fa1a25ef712789e47063532dfd78a2be41433fb83218886`,
the complete change adds the integer-bit rounding helper and replaces the scale
constructor's single `round(le)` call. All other generated source, reflected
bindings, dispatch geometry and template materialization are unchanged. Strict
compilation passes for all four old/current artifacts; compiled binaries differ
because this is a rounding correction, not a formatting change. Independent
rounding regressions compare generated Metal and OpenGL with the original Metal
source and a decimal oracle, including halfway neighbors and narrow inputs.
The native-loader workload and its zero-tolerance assertions are unchanged.

The scale conversion invokes the source `fp8_e8m0(float)` constructor factory
before the selected sibling float conversion operator; aggregate field
initialization would encode the scale incorrectly. Qualified `metal::round`
lowers to a helper preserving Metal halfway-away-from-zero rounding instead of
the target's native rounding rule. OpenGL lowers `metal::isfinite` to
a single-evaluation IEEE-754 float32 exponent-mask test and `signbit` to the
exact sign bit. The private `fp8_e4m3` scalar view is admitted only as a
read-only, exact one-member-layout projection. Unresolved constructor branches,
unsupported predicate types, writes, and receiver mutation remain fail-closed.

The reflected data ABI is float32 input at binding 0, the retained but
statically unread `global_scale` input at binding 1, and float32 read-write
output at binding 2. DirectX additionally reflects generated dispatch input at
`b0`; HLSL register namespaces make this legal beside input `t0`, while
`global_scale` uses `t1` and output uses `u2`. The real no-global-scale host
branch omits buffer 1. The generic reflected-resource loader instead supplies
one inert allocation because it cannot omit a declared resource, and the test
checks that the selected specialization contains no read of it.

The numerical request contains 32 exact FP4 E2M1 values from -6 through 6.
Its maximum absolute value and scale divisor are both exactly 6, so the encoded
MX scale is exactly 1 and quantize/dequantize must return every input bit-exactly
at zero absolute and relative tolerance. Windows CI requires that request
through Direct3D 12 WARP; Linux CI requires the software-subgroup artifact
through Mesa headless EGL. Both paths build the runtime artifact manifest,
package, loader manifest, reflected ABI descriptor, and one-workgroup dispatch
request before native execution.
The existing platform jobs retain source and package identities, compiler
records, uploaded inputs, readbacks, and JUnit output, including on failure.
The project report is saved before reference assertions. No additional runner
or tolerance change is introduced. Local Mesa execution of the updated GLSL
returns all 32 outputs bit-exactly; an independent readback audit derives the
FP4 alphabet and scale from the uploaded inputs rather than the saved expected
outputs. Updated Windows execution remains required.

This is one bounded float32 MXFP4/no-global-scale workload at `846d1762`, not
the current `9c3d3557` corpus. It does not cover
the remaining `fp_quantized` entries, global-scale variants, other group sizes,
bit widths or dtypes, MLX host redirection, selected-entry Metal compilation,
or the full MLX test suite. The separate native Metal aggregate baseline remains
the round-trip boundary.

Owner-dependent `constexpr` helper calls in quantized struct static members now
resolve for the selected pinned replay, completing CrossGL/crosstl#1672.
CrossGL/crosstl#1491 tracks remaining qualified-static-constant materialization
outside the compiler-validated DirectX frontier.
Built-in overloads are resolved alongside user-defined wrappers by source
signature.
Before native validation, the harness verifies
the four numeric-to-Boolean SIMD wrapper conversions and the signed 8-, 16-,
32-, and 64-bit `arange` arithmetic conversions in the generated artifact.
Ubuntu CI installs `glslangValidator` and runs it with an OpenGL/SPIR-V 1.3
target. It first compiles the focused scalar-conversion fixtures successfully,
then compiles the translated `arange.metal` artifact. Shuffle-and-fill wrappers
lower through backend-neutral subgroup semantics and preserve their explicit
fill value for lanes below the delta.
The reduced DirectX/Vulkan frontier and the eight-source OpenGL artifact gate
include `scaled_dot_product_attention.metal`. Function-local scalar
and vector aliases now retain lexical scope and resolve across declarations,
constructors, casts, and generic static-member owners. For the pinned attention
source this resolves 531 concrete uses across all 42 entries, including the
float accumulation type used by the half and bfloat input families. The full
source translates to DirectX, OpenGL, and Vulkan; local OpenGL and Vulkan native
validation passes, and official DXC v1.9.2602.24 compiles all 42 generated
DirectX compute entries. This is compiler validation, not Direct3D runtime
execution or numerical parity.
The project Vulkan artifact is warning-free because project preparation removes
unreachable generic declarations. Direct single-file translation still emits
five such warnings under
[#1568](https://github.com/CrossGL/crosstl/issues/1568). Qualified pointer and
array aliases remain tracked in
[#1567](https://github.com/CrossGL/crosstl/issues/1567).
`arg_reduce.metal` still materializes all 24 host-named entries. The historical
aggregate run emits Vulkan but continues to fail closed for DirectX and OpenGL
with `project.translate.workgroup-size-entry-ambiguous`, because one aggregate
artifact cannot infer the runtime-selected axis and pipeline limits. That
aggregate result no longer describes all available arg-reduce coverage.

The checked
[`contracts/arg_reduce.native-loader.dispatch.json`](contracts/arg_reduce.native-loader.dispatch.json)
contract selects historical pin `846d1762`'s `argmin_float32` and `argmax_float32` for
two axis-32 rows. It applies the host formula
`roundUp(min(ceilDiv(axisSize, 4), maxThreadsPerWorkgroup), simdWidth)`, fixes a
wave width of 32, emits `[32, 1, 1]`, and dispatches `[1, 2, 1]` workgroups.
Signature-aware helper materialization selects the scalar
`elem_to_loc<int64_t>` overload rather than the `uint3` overload. The HLSL
artifacts are 7,129 and 7,191 bytes with SHA-256
`9ad32989ed91db2add926046a2f58c8735ca23d459e394e0ae9c890f12edd086`
and `d7343928b87693f63464227baf181759adcda3a349835b11182a99bc8292d999`;
official DXC 1.9.2602.24 accepts both under `cs_6_6`,
`-enable-16bit-types`, and warnings as errors. This proof explicitly sets
`project.source_options.metal.target_options.directx.relative_wave_shuffle_out_of_range`
to `"self"`: a relative shuffle whose source lane is outside the wave retains
the calling lane's value. Generated helpers select either the valid relative
lane or the calling lane before every unconditional `WaveReadLaneAt`, so the
second reduction cannot consume undefined high-lane state. The default DirectX
policy remains `"undefined"`, preserving existing artifacts unless a project
opts in.

The latest HLSL reference review reproduces both accepted predecessors with
the translator revision that recorded them. The complete source delta adds
one placeholder member to each of 14 empty structs; 13 are unreferenced and
the remaining reduction receiver never accesses the added member. Reflected
bindings, workgroup configuration, shuffle policy and materialization are
unchanged. All four accepted/current sources compile with warnings fatal.
Their DXIL containers are not byte-identical: the second reduction loop's
simultaneous PHI values are reordered and renamed. LLVM comparison and a
complete IR comparison after only value naming and PHI ordering confirm
equivalence, without instruction folding or operand reordering. The recorded
dependency-discovery count is corrected to 75 to match both reports.
This review does not substitute for new-head Windows numerical execution.

The explicit OpenGL software-subgroup artifacts are 8,024 and 8,030 bytes with
SHA-256
`587b409127a5cec9711856acbdac69fbc05fe25fc8004f048de15057e5a9f59f`
and `009389698c327dde691ee87a4f34d5f80c7a33384e800d8d640f97a462ae1be6`.
They admit direct shuffle-helper calls only inside a canonical workgroup-uniform
halving loop (`offset > 0` with `/= constant >= 2` or `>>= constant >= 1`),
while lane-varying, nonterminating, escaping, mutated, indirect, and nested
forms remain rejected. Both modules pass `glslangValidator` and `spirv-val`,
contain five control barriers and four shared-memory fences, and contain no group-nonuniform SPIR-V
instruction.

The reference review compares complete generated bodies, reflected resources,
entry points, and materialized functions. HLSL changes make integer promotions
and 16-bit shuffle arguments explicit and preserve the negative-infinity bits.
The compiled differences are integer no-wrap flags and a nonnegative row-index
comparison, valid for these bounded rows and offsets. OpenGL also makes shared
memory ordering explicit, bounds shuffle offsets before addition, and implements
signed remainder with truncating division. Neither review substitutes for native
execution. With `CROSTL_KEEP_CORPUS_EVIDENCE=1`, failures retain the project
configuration, translation report, generated source, package, compiler records,
dispatch inputs, and any returned readbacks under `.crosstl-corpus-evidence`.

The native ABI reflects float32 input, uint32 output, int32 shape, int64 stride,
and uint64 size resources exactly: nine DirectX bindings include the generated
`CrossGLDispatchInfo`, while OpenGL has eight bindings. Linux arm64 llvmpipe
executed deterministic rows with repeated extrema and read back argmin indices
`[5, 7]` and argmax indices `[3, 2]`, proving lowest-index tie behavior.
The retained [Windows execution results](https://github.com/CrossGL/crosstl/actions/runs/37403863131/job/112076922079)
confirm the same indices. Independent inspection verifies the uploaded rows,
lowest-index ties, generated source and executed DXIL identities. The enclosing
job later failed at the separate GEMV reference check; it is not a passing
project-wide result.
Windows CI requires the same two workloads through Direct3D 12 WARP and Linux
CI requires them through surfaceless Mesa EGL. Other axes, dtypes, and the
remaining 22 host-named entries, MLX host redirection, and the full MLX test
suite remain outside this bounded claim. Entry-scoped Metal output still fails
explicitly at workgroup specialization with
`project.translate.workgroup-size-rule-unsupported-target`; the native macOS
aggregate baseline is separate and this proof does not claim an entry-scoped
Metal round trip. Project dispatch import was completed under
[#1793](https://github.com/CrossGL/crosstl/issues/1793), and remaining broad
entry packaging stays tracked in
[#1523](https://github.com/CrossGL/crosstl/issues/1523).

The checked
[`contracts/scaled_dot_product_attention.native-loader.dispatch.json`](contracts/scaled_dot_product_attention.native-loader.dispatch.json)
contract selects `sdpa_vector_float_64_64` at the historical `846d1762` pin for the bounded
one-pass vector path: batch 1, one query and KV head, query length 1, key length
4, query/value dimensions 64, scale 0.125, and no mask, causal mode, or sinks.
The host-derived contract fixes workgroup `[1024, 1, 1]`, subgroup width 32,
and one dispatched workgroup. Function constant IDs 20 through 25
(`has_mask`, `query_transposed`, `do_causal`, `bool_mask`, `float_mask`, and
`has_sinks`) are all false; ID 26 belongs to the two-pass path and is
intentionally absent. The dispatch content identity is
`97e5ebb69af8da3a0082776015787456f23b8bfdb0cff757f5364db2cfef8d2c`,
with variant ID
`sha256:8b2abb9f7179e051530697fb8d1956d0ff03a324e7acaa5fcdf4f4dd9f1befbb`
and artifact ID
`sha256:dd0138695bd82e1f8ea49bd667052b484420ee96cb2849c6eed20ba5eae39a89`.

The 9,116-byte HLSL artifact has SHA-256
`b6c596dfce2e3695f0d5c28eaf7b837ee97bda579169885ff19cba52c0bbc553`.
The guarded mask rebases before the key loop and the guarded mask advances
inside it update one entry-scoped `int64_t` offset per mask pointer.
Official DXC 1.9.2602.24 accepts `CSMain` under `cs_6_6`,
`-enable-16bit-types`, and warnings as errors, producing 9,000 bytes of DXIL.
Its 32 physical waves receive unique workgroup-synchronized subgroup IDs; no
lane-varying `SV_GroupIndex / WaveGetLaneCount()` derivation remains. The
13,385-byte explicit software-subgroup GLSL artifact has SHA-256
`002b7dcfdcbd65057ded11ded56a348d6f067b63fbd29d3933d20420c9dc614b`.
It partitions 1,024 invocations into 32 logical subgroups and synchronizes the
source subgroup-ID-strided runtime loop across every workgroup round. Inactive
subgroups contribute typed collective identities, so all generated barriers
remain uniform. `glslangValidator` and `spirv-val` accept the resulting
OpenGL/SPIR-V 1.3 module; disassembly contains exactly nine
`OpControlBarrier` instructions, six `OpSpecConstantFalse` declarations,
local size `1024 1 1`, and no `OpGroupNonUniform` instruction.

The reference update compares complete old and current shader bodies and every
reflected resource. HLSL changes are explicit source integer promotions in key
and value offsets and sign-bit negation of the finite-limit constant. GLSL changes
add shared-memory ordering at subgroup barriers and preserve signed remainder.
Dispatch geometry, specializations and resource interfaces remain unchanged.
Both old references are reproduced from the earlier translator and strictly
compiled before comparison; fingerprints alone are not numerical evidence.
The subsequent HLSL reference review adds one padding member to each of 13
unused empty structs. None is instantiated or used in a resource interface.
Removing only those members reproduces the complete accepted source, and
strict DXC compilation produces byte-identical DXIL. The GLSL source and
validated SPIR-V are unchanged. This update does not change the required
Windows execution, dispatch, bindings, or numerical comparisons.

Both targets package complete native-loader requests. DirectX has 19 reflected
resources including generated `CrossGLDispatchInfo`; OpenGL has 18. Optional
`bmask`, `fmask`, mask-stride, and sinks bindings receive harmless placeholders.
The stored Boolean mask ABI is physically uint32 in both HLSL and GLSL block
storage. DirectX concretizes the six false constants. OpenGL retains them for a
verified deferred GLSL-to-SPIR-V specialization request and publishes the JSON
variant registry while explicitly omitting the unsupported native registry
header.

Windows CI requires the 64-value output to execute through Direct3D 12 WARP.
Linux CI binds PyOpenGL to surfaceless EGL, forces llvmpipe, performs deferred
SPIR-V specialization, dispatches through the OpenGL native loader, and compares
all values with a stable CPU scaled-attention reference at `2e-4` absolute and
relative tolerance. The current local Mesa run has maximum absolute error
`4.082320426146424e-08` and maximum relative error
`4.2163276126605175e-06`. An independent 100-digit reference using the uploaded
binary32 inputs gives maximum absolute error `4.688126085821141e-08` across
all 64 outputs. The retained [Windows execution results](https://github.com/CrossGL/crosstl/actions/runs/37403863131/job/112076922079)
also pass. Independent calculation from the uploaded binary32 inputs gives
maximum absolute error `4.700266193284516e-08` across all 64 values, with the
generated source, compiler output and executed DXIL identities verified.
This does not establish normalization execution: those later steps were skipped
after the GEMV reference failure in the same job.
Existing native jobs retain translation reports before identity checks,
packages, compiler records, inputs, results and JUnit reports on failure.
This is bounded evidence for one float32 one-pass
workload only. Masked, causal, sinks, two-pass and full-attention paths, other
dimensions and dtypes, the remaining host-named entries, MLX host redirection,
and the full MLX test suite remain outside the claim. The separate native Metal
baseline does not make this selected native-loader proof a Metal round trip.
The historical 42-entry aggregate DirectX/OpenGL result remains fail-closed
because it does not consume this entry-scoped contract; it no longer means that
no bounded attention dispatch, package, or numerical runtime proof exists.

The reduced reference-accessor fixture covers three non-template paths. The
mutable scalar call returns the direct `val_frags[i * width + j]` lvalue and
must retain storage identity through assignment and readback. The implicit const
scalar call must lower directly into its read-only helper argument. The nested
const path matches the pinned `BlockMMA`/`Ctile` shape with a reduced outer value,
a `float2` fragment tile, a `thread const auto&` alias, and an `accum[k]` read.
For DirectX and OpenGL, the last path must contain neither `frag_at` nor `accum`
and must read `self.nestedTile.val_frags[...][k]`. This proof does not cover
template-indexed or nested forwarding overloads, full `quantized.metal`
translation, shader execution, numerical parity, or upstream MLX host/runtime
integration.
The reduced template-member pointer fixture covers the next `BaseMMAFrag::load`
boundary independently. It requires materialization of the generic
`SrcPtrType` helper from `&(src[index])`, a pointer-backed helper parameter or
equivalent OpenGL buffer-offset view, and an indexed `src[stride]` read whose
base offset still contains the outer `index`. It rejects a scalarized `float`
parameter even when no source-style call remains. The proof ends at artifact
structure and does not establish target compiler acceptance, shader execution,
or numerical parity.
`binary_two.metal` now also belongs to the required OpenGL toolchain frontier.
CrossTL commit `db593d19b` specializes fixed-array helper views to their concrete
runtime storage resources while retaining fixed extents and offsets. For the
pinned source, project translation emits zero diagnostics, the generated GLSL
compiles for OpenGL/SPIR-V 1.3, and the resulting SPIR-V passes `spirv-val`. This
resolves [#1661](https://github.com/CrossGL/crosstl/issues/1661) for the pinned
frontier. It is artifact and toolchain evidence only; it does not establish
numerical or runtime parity.
The current clean OpenGL check explicitly selects
`binary16_remainder_profile = "binary32-quotient"` for the source half `fmod`
operations in `binary_two.metal`. This is a configuration adaptation; upstream
files are unchanged. The profile rounds the quotient, product and subtraction
separately in binary32 before converting to half. Native controls retain the
source's finite cancellation, signed-zero and exceptional-value behavior rather
than substituting exact mathematical remainder. The choice is recorded in the
project report and is not a universal Metal-device assumption. Unconfigured
half calls remain diagnostic, and bfloat comparison limits under #2000 remain
separate. These operation controls do not establish complete DivMod execution
or full MLX-suite parity.
The clean OpenGL frontier supplies 24 configured index-range assertions, all with
inclusive bounds `[0, 2147483647]`. The expressions are `offset + i`, `a_idx`,
`b_idx`, `out_idx`, `out_idx++`, `idx.x`, and `idx.y` for `binary_two.metal`;
`batch_idx * offset_stride`, `freq_stride * pos.x`, `in_index_1`, `in_index_2`,
`out_index_1`, and `out_index_2` for `rope.metal`; and `offset + i`, `a_idx`,
`b_idx`, `c_idx`, `bidx`, `cidx`, `out_idx`, `out_idx++`, `idx.x`, `idx.y`, and
`idx.z` for `ternary.metal`. These records are explicit MLX host/runtime
portability preconditions for OpenGL. They are not inferred guarantees, CrossTL
does not enforce them at runtime, and they do not establish runtime integration
or numerical parity.
The eight-source OpenGL/SPIR-V gate includes `rms_norm.metal`, `rope.metal`, and
`scaled_dot_product_attention.metal`. Their Metal function constants retain
their numeric identifiers as native GLSL specialization constants; the gate
compiles each generated module for OpenGL/SPIR-V 1.3 and validates the resulting
binary. Its reduced native specialization check establishes typed dispatch and
deterministic readback only, not numerical parity for those full aggregate
modules. The bounded `sdpa_vector_float_64_64` proof above is separate and does
compare every selected output with a CPU attention reference. For the pinned
DirectX rope check, project
configuration supplies IDs 1 through 3 and CrossTL materializes a concrete HLSL
variant before DXC.

The focused `prove_copy_opengl.py` gate translates the full upstream
`copy.metal` source pinned at commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e` under one selected entry-point
scope, `s_copycomplex64float32`. The source declares 2,496 entries and expands
to 2,497 preprocessed instantiations. Exactly one selected specialization is
materialized, while the current evidence prunes 69,915 candidate pairs. No
generated wrapper fallback is used.

The generated OpenGL artifact must lower the source
`static_cast<float>(src[0])` conversion to one evaluation of `(src[0]).real`.
Linux CI compiles the resulting GLSL to OpenGL/SPIR-V 1.3 and validates the
module with `spirv-val`. This bounded lowering follows the pinned
`complex64_t` conversion body; generalized user-defined conversion operators
remain tracked by [#1744](https://github.com/CrossGL/crosstl/issues/1744). This
proof does not claim shader execution, numerical parity, runtime integration,
or passage of the MLX test suite.

The companion `prove_copy_directx.py` gate selects
`s_copycomplex64bfloat16` from the same full pinned `copy.metal` source. It
requires exactly one reachable specialization, verifies that each generated
store evaluates the complex source once and projects `.real`, and requires the
exact round-to-nearest-ties-to-even bfloat16 helper. Windows CI compiles the
standalone HLSL entry with DXC using `cs_6_2`, `-enable-16bit-types`, and
warnings as errors. This proves selected-entry translation and native compiler
acceptance; it does not execute the shader, establish numerical parity, port
the MLX runtime dispatch path, or run the MLX test suite.

A separate native-loader check selects `v_copyfloat32float32` from the current
corpus at commit `846d176227a0ac13d2667e58d2bb68b322109ab0`. It materializes
`copy_v<float, float, 1>` and the reachable `cast_to<float, float>` helper while
pruning 62,424 unreachable candidate pairs. It packages one HLSL artifact and
one GLSL artifact with exact reflected scalar buffer layouts, dispatches eight
nonuniform float32 values through Direct3D 12 WARP on Windows and a headless
OpenGL software context on Linux, and requires exact readback on both targets.
The generated artifact hashes, byte counts, binding layouts, and 1-by-1-by-1
workgroup sizes are fixed in the test. This is bounded execution evidence for
one scalar copy entry. It does not cover the complex or bfloat16 entries,
redirect the MLX host runtime, or run the MLX test suite. Aggregate input layout
for `s_copycomplex64float32` remains part of the physical-resource contract
tracked by [#1543](https://github.com/CrossGL/crosstl/issues/1543).

The independent current-pinned complete copy Metal gate covers all 2,496
discovered entries from `copy.metal` at commit
`846d176227a0ac13d2667e58d2bb68b322109ab0`. It spans 30 shapes, 16 concrete
kernel templates, all 169 input/output type pairs over the 13 source types, and
six explicit conversion families. Every entry is translated under its own
entry scope through CrossGL, and a schema-v2 hash-pinned contract records the
exact artifact identity, source/default template provenance, and reflected ABI.
The 2,496 artifacts contain 6,566 exact materializations and 8,684 reflected
resources. Each artifact owns one `cast_to` specialization; the 14
complex-to-bool artifacts additionally own the nested `cast_to<bool, float>`
specialization, so the proof does not falsely assume one fixed specialization
count per shape.

The gate preserves MLX's explicit float and bfloat16 Boolean bit-pattern tests,
including the native-width `as_type<ushort>` bfloat path. Registered
`complex64_t` representation conversions validate the exact ordered
`real: float, imag: float` shape before projecting `.real`; malformed registered
representations fail closed and unregistered lookalikes are not projected.
Scalar/vector forms reflect three resources, generalized forms reflect exact
three- through eight-resource interfaces, and every artifact retains the
host-owned `[1, 1, 1]` workgroup contract. Required Ubuntu CI verifies and exports
the family in 24 disjoint 104-entry source shards. One dependent macOS job
verifies every exported identity against the complete contract, compiles every
artifact with `xcrun -sdk macosx metal -Werror -c`, and requires 2,496 non-empty AIR outputs.
Source reports, separate shard bundles and compiler results are retained.
This is complete discovered-copy translation, reflection, and native compiler
evidence, not Metal numerical execution, MLX host-runtime redirection, or MLX
test-suite parity. The older selected OpenGL, DirectX, and native-loader checks
remain separate bounded evidence.

The current copy references include a complete, compiler-gated review of byte
conversions and typed vector initialization. Of 2,496 freshly translated
artifacts, 1,760 are unchanged; reversing only the reviewed conversions and
`int2`/`long2` initializer syntax recovers every previous source body. All
2,496 current artifacts compile with warnings fatal, preserving the 6,566
materializations and 8,684 resources above. Changed indexing, byte widths,
address arithmetic, and loop bounds are rejected by negative review controls.
Historical proof and linkage-refresh metadata remain unchanged. This refresh
does not add numerical execution or newer upstream coverage; other families
remain under [#1966](https://github.com/CrossGL/crosstl/issues/1966).

The HLSL copy reference refresh compares all 2,496 bodies with their recorded
versions and compiles every entry with warnings fatal. The 1,820 changed bodies
retain source-width byte conversions, explicit integer arithmetic and half
rounding; all 6,566 materializations and 10,036 reflected resources are checked.
Half buffers use two-byte `uint16_t` storage with explicit binary16 metadata,
rather than a floating storage type. A subsequent 28-entry correction replaces
float32-mediated wide-integer-to-bfloat conversions with direct integer rounding
([#2089](https://github.com/CrossGL/crosstl/issues/2089)). Each changed body and
binding is reviewed and compiled with warnings fatal; four unchanged conversion
controls retain their previous identities. The generated source total is
4,859,340 bytes. Generic native tests cover 64,276 signed and unsigned inputs,
including every integer midpoint and adjacent value across the representable
exponents, limits, single evaluation and explicit float32-mediated controls.
Original and generated Metal results are verified independently; required
Windows execution remains separate from compiler acceptance. Nested bfloat
constructors now retain their result types before Metal bitcast validation
([#2090](https://github.com/CrossGL/crosstl/issues/2090)). Generic native controls
cover aliases, scalar and vector payload identity, direct wide-integer rounding
and saved intermediate source. Unknown or unequal storage widths remain errors.
These reduced controls do not establish full-corpus or host-runtime parity.

The current-pin native copy gate separately tests 11 float32 layouts and the
contiguous and strided half-copy entries at
`9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8`. Both half entries transport all
65,536 binary16 payloads, including signed zero, subnormals and NaNs, in bounded
dispatches with 32 untouched output guards. OpenGL uses the exact widened
binary32 representation; DirectX and Metal retain two-byte encoded storage.
The existing native jobs run these checks without additional platform runners.
The [Windows copy step](https://github.com/CrossGL/crosstl/actions/runs/37329793271/job/111829705702)
passes all 35 checks. Independent inspection confirms 131,072 half payloads,
128 guards and matching source/DXIL identities. The job later fails an unrelated
LogSumExp reference check, so this is not a passing-job or full-suite claim.
Copy evidence uploads include hidden generated-package directories as well as
the separately retained shaders, compiler results and readbacks.
Metal also executes the unchanged upstream entries and verifies read-only
inputs. Reports retain source identities, packages, compiler results and raw
readbacks. The HLSL corpus gate checks the physical `uint16_t` half-buffer layout
and logical binary16 metadata explicitly; this does not accept unreviewed
artifact identities or claim complete copy-conversion numerical coverage.

The reversed one- and four-dimensional float32 copies supply checked integer metadata
for shapes, strides and offsets. OpenGL additionally uses a checked dispatch region;
the fixture no longer trusts assertions about the final index expressions.
Source-typed scalar helper summaries retain negative strides through unsigned
arithmetic and signed return conversion. Component-wise vector helper summaries
forward immutable array facts through bounded dimension loops and retain caller
loop updates. Native checks retain all 17 and 120 copied values respectively,
32 tail guards per case, unchanged metadata and source buffers, and original
Metal comparison on macOS. The four-dimensional case checks 15 metadata values,
including negative strides and the partial final two-element copy. These fixed
workloads do not establish unrestricted layouts or complete host-runtime copy routing.

A second reversed four-dimensional case starts both allocations at their first
accessed element, with source and destination offsets 67 and 4. It retains the
120 copied values, untouched holes and 32 tail guards without leading padding.
The OpenGL proof follows the partial-tile loop guard into the helper-derived
offsets; no source edits or asserted index ranges are used. This case runs in the
existing native copy jobs alongside the padded layout.

The same tight layout also covers `ggn4large_dynamic_copyfloat32float32`, with
four-element tiles and signed 64-bit indices. Its launch grid is `[2, 4, 6]`;
the final tile contains one valid element. Checked base and relative-offset
bounds preserve negative relative subscripts without accepting a negative final
address. The unchanged source executes on generated OpenGL and original/generated
Metal with the same 120 values, untouched holes and 32 tail guards. This adds a
wide-index entry to the existing OS jobs, not full copy-conversion or host-routing
coverage. No upstream source changes or padding workarounds are used.

The scalar and vector two-dimensional copies also cover the unchanged
`s2_copyfloat32float32` and `v2_copyfloat32float32` entries with 23 output values,
a `[3, 4, 1]` launch and a partial final two-element tile. The source allocations
contain one scalar or 23 vector elements, respectively; 32 output guards remain
outside the logical result. The fixtures check the uploaded 64-bit size and
read-only inputs, including bitwise preservation of negative zero. The OpenGL
analysis retains the counter's initial value under affine tail conditions and
checks source arithmetic through the terminating update. An upper trip bound
is not treated as an exact iteration sequence for pointer rebasing. These cases
extend the existing native jobs without changing upstream kernels or claiming
complete copy-conversion coverage.

The unchanged `gn4large_copyfloat32float32` entry also exercises a scalar-return
index helper with local accumulation and by-value vector-component updates. Its
checked shape `[2, 3, 4, 5]`, strides `[100, 7, 21, 1]` and dimension count feed a
`[2, 4, 6]` dispatch with four-element tiles. The native fixture checks all 120
copied values, 32 guards and unchanged inputs, with original Metal comparison on
macOS. The existing copy jobs now select 12 entry prefixes and 16 float32
layouts. This does not cover every dtype conversion or complete host routing.

The focused `prove_layer_norm_directx.py` gate translates two host-selected
single-row entries from the pinned `layer_norm.metal` source: forward float32
with axis size 4099 and VJP float32 with axis size 8192 and `has_w=true`. It
reads the exact host dispatch formulas from the pinned `normalization.cpp` blob
and checks that both workloads are exercised by the pinned MLX fast-operation
tests. Those inputs derive workgroup sizes `[544, 1, 1]` and `[1024, 1, 1]`.
Looped entries remain outside this proof because their workgroup sizes depend on
the selected Metal pipeline's `maxTotalThreadsPerThreadgroup()` value.

Both selected source templates declare `SIMD_SIZE = 32`, consume the Metal lane
and simdgroup index builtins, and call the shared `simd_sum` reduction path. The
DirectX project variants therefore require an exact subgroup-width rule of 32.
Each standalone HLSL artifact must retain the rule provenance, emit exactly one
`[WaveSize(32)]` beside its host-derived `numthreads` contract, and compile with
DXC using `cs_6_6` and warnings as errors. Windows CI applies that gate to both
entries. This is source identity, host-dispatch, translation, reflection, and
native compiler evidence. It does not execute either kernel, compare numerical
results, port the MLX host runtime, or claim coverage of the MLX test suite.

The checked-in
[`contracts/layer_norm.dispatch.json`](contracts/layer_norm.dispatch.json)
fixture contains exactly two pinned single-row float32 LayerNorm records from
MLX commit `4367c73b60541ddd5a266ce4644fd93d20223b6e`. It captures each
record's entry point, workgroup size, subgroup width, specialization constants,
and dispatch geometry together with the source and host-dispatch provenance.
The finite records cover the forward axis-size-4099 workload and the VJP
axis-size-8192 workload described above. Only the VJP record applies function
constant `20` (`has_w=true`), matching the pinned host dispatch; the forward
record carries no function constant. These records do not prove runtime
execution, numerical parity, looped variants, or the full MLX test suite.

The project-porting harness consumes this contract against the unchanged pinned
`layer_norm.metal` source and requires two deterministic DirectX artifacts. It
checks source and contract identities, entry-scoped execution provenance,
host-derived workgroup sizes, function-constant values, exact subgroup-width
enforcement, generated hashes, and artifact paths. Windows CI compiles both
artifacts with DXC `cs_6_6` and warnings as errors. This extends the bounded
proof into the repository-level project report; it does not add runtime
execution or numerical parity for those historical axis-size-4099 forward and
axis-size-8192 VJP records.

The reviewed forward artifact is 9,308 bytes with SHA-256
`04e8aef85da4a392ed30161537c231f3023802b94e90dafbc3fa02ca822fd3a0`;
the VJP artifact is 11,152 bytes with SHA-256
`4aac072f816ffb0972c380afe429ec4b7c605015e76810665ac2390d1b629c10`.
Earlier complete body changes preserved the source's precise reciprocal-square-root
and square-root operations, respectively. Resource interfaces, host dispatch
contracts and source-map origins are unchanged. Both pass strict DXC compilation;
the shared helpers have independent native arithmetic checks, but those checks
do not establish execution or numerical parity for these larger workloads.
An earlier compatibility review covered these two artifacts, both LogSumExp
variants and all twelve RMSNorm variants. That update added only padding in
unused empty structs. It reproduced every previous source identity and retained
resource interfaces, with byte-identical strict DXC output for all sixteen
artifacts. Dispatch contracts, specialization constants and subgroup requirements
were unchanged.

A later precision update materializes helper return values explicitly in the two
LayerNorm and twelve RMSNorm HLSL dispatch artifacts. Removing only those return
locals reconstructs every accepted source; interfaces, source-map origins and
dispatch contracts remain unchanged. All old and current sources pass strict
DXC compilation. Their fourteen compiled modules differ and still require native
execution; both LogSumExp artifacts remain byte-identical. The four selected
normalization loaders receive the same reviewed return-local update, with all
inputs, reference values, guards and tolerances retained.

A subsequent conversion update divides by `__crossgl_uint_to_float1(axis_size)`
instead of relying on HLSL's implicit unsigned-to-float conversion in the same
two LayerNorm and twelve RMSNorm artifacts. Removing the helper and restoring
only those bare operands reconstructs every accepted source; interfaces,
source-map origins and dispatch contracts remain unchanged. All old and current
sources pass warning-fatal DXC `cs_6_6` compilation, and their fourteen modules
differ. An exhaustive model of the helper matches correctly rounded conversion
for all 2^32 inputs; axis sizes below 2^24 convert exactly either way. Both
LogSumExp artifacts remain byte-identical.

A separate historical proof selects the forward `layer_normfloat32` entry at
commit `846d176227a0ac13d2667e58d2bb68b322109ab0` through
[`contracts/layer_norm.native-loader.dispatch.json`](contracts/layer_norm.native-loader.dispatch.json).
It records the exact host formula
`32 * ceil_div(ceil_div(axis_size, 8), 32)`, the upstream
`python/tests/test_fast.py::test_layer_norm` provenance, axis size 32, two rows,
a `[32, 1, 1]` workgroup, subgroup width 32, two dispatch workgroups, and no
function constants. An explicit source/entry-qualified workgroup-access
precondition bounds specialized helper views to indices 0 through 31; it is a
stated host portability contract, not an inferred or runtime-enforced bound.
Materialization emits the forward entry plus `initialize_buffer<1>` and
`threadgroup_sum<1>` from six reachable specializations while pruning 194
candidates.

The generated HLSL artifact is 8,809 bytes with SHA-256
`668cf734597226873e79e30459435ae49e281bbff380c8fba4f9fb340d2d3940`;
its divisors use the correctly rounded `__crossgl_uint_to_float1(axis_size)`.
It retains `[WaveSize(32)]`, two `WaveActiveSum` calls, and compiles as
`cs_6_6` with `-enable-16bit-types`; Windows CI requires Direct3D 12 WARP
execution. The generated GLSL artifact is 8,403 bytes with SHA-256
`8baa04a8c4125c96f1c8f0ad42f86bfc7b2497ba5a59178d199a1a3fbe3d110e`.
An earlier reference review covered the four selected normalization loaders
(LayerNorm and RMSNorm, forward and VJP) at `846d1762`: source-width index
conversions in HLSL and shared-memory ordering in GLSL preserved the resource
interfaces. All eight artifacts passed strict compiler checks at that revision.
Linux OpenGL execution and an independent 100-digit calculation checked all 256
returned values within unchanged tolerances, with maximum absolute error below
`1.36e-7`. Windows execution remains required separately.

An earlier HLSL compatibility update added thirteen unused placeholder fields
in helper structs. Removing those declarations reproduced its previous accepted
source, and both complete shaders compiled to byte-identical DXIL with warnings
fatal. The numerical test, resource layout and dispatch requirements were
unchanged. The later precise-return update is reviewed separately above.

A subsequent HLSL conversion update makes all four selected normalization
loaders divide by `__crossgl_uint_to_float1(axis_size)` instead of relying on
implicit unsigned-to-float conversion. Removing the helper and restoring only
the bare operands reconstructs each accepted source; the previous and current
HLSL pass warning-fatal `cs_6_6` DXC compilation, while GLSL, resource layouts,
inputs, references and tolerances are unchanged. The selected axis size of 32
converts exactly either way, and Windows execution remains required.

The forward LayerNorm references now retain the source's explicit precise
reciprocal-square-root helper. Complete body review found only that helper, its
declaration and one call substitution, with unchanged resource interfaces and
source-map origins. Previous and current OpenGL readbacks are identical for the
selected 64-value workload and differ from unchanged original Metal by at most
`7.46e-9`; original Metal output guards and read-only inputs remain intact.
The same helper passes independent replay of retained Windows execution, while
the complete updated HLSL kernel still requires its own Windows CI result.
The existing CI steps retain reports, requests, compiler
records, returned values and JUnit results on success or failure; these selected
cases do not establish whole-family or upstream-suite parity.

The LayerNorm GLSL software path admits a subgroup helper only when the
helper has one stable non-overloaded source identity and every call is direct,
unconditional, top-level, and made from the sole compute entry. Conditional,
nested, indirect, ambiguous, or potentially divergent helper use rejects
translation. Resource-specialized helper lookup may reuse a prevalidated
workgroup proof only when its base key is identical and its intervals cover the
requested narrower proof; incompatible or narrower assumptions remain
rejected.

The software GLSL contains the shared-memory sum helper and six barriers, but no
KHR subgroup extension, `gl_Subgroup*` use, `subgroupAdd`, or SPIR-V
`OpGroupNonUniform` instruction. `glslangValidator` and `spirv-val` accept it;
its disassembly contains six `OpControlBarrier` instructions. The native-loader
package reflects eight ready resources on both targets: float32 input, weight,
bias, and output arrays, plus 16-byte float32 epsilon and uint32 axis-size,
weight-stride, and bias-stride blocks. DirectX uses structured/constant-buffer
layouts and OpenGL uses std430/std140 layouts.

The deterministic workload uses two 32-element rows, 32 weights, 32 biases,
epsilon `1e-5`, and unit weight/bias strides. It compares 64 outputs with
`((x - mean) / sqrt(mean((x - mean)^2) + epsilon)) * weight + bias` at `5e-5`
absolute and relative tolerance. A local Linux arm64 llvmpipe EGL dispatch
passed with maximum absolute error below `8.88e-8`; Linux CI requires the same
Mesa numerical readback. This is bounded execution evidence for one float32
forward workload only. It does not redirect MLX host execution, run the MLX
test suite, or cover VJP, float16, bfloat16, looped entries, other axis sizes,
or the historical wider dispatch records above.

A sibling pinned-corpus proof selects `vjp_layer_normfloat32` at axis size 32
through
[`contracts/layer_norm_vjp.native-loader.dispatch.json`](contracts/layer_norm_vjp.native-loader.dispatch.json).
It fixes one float32 row, `has_w=true` through function constant ID `20`, a
`[32, 1, 1]` workgroup and 32-lane subgroup, one dispatch workgroup, and the
explicit helper-access interval 0 through 95 required by three SIMD-sized
threadgroup slices. The pinned
`python/tests/test_fast.py::test_layer_norm_grad` case supplies the upstream
shape and weighted-gradient provenance. Materialization selects
`vjp_layer_norm_single_row<float, 8>`, `initialize_buffer<3>`,
`threadgroup_sum<3>`, and `threadgroup_sum<1>` from seven reachable
specializations while pruning 194 candidates.

The generated HLSL is 10,652 bytes with SHA-256
`0002c8c3dccb067731cfec1300382c85898c0bae627a4c47488b302ff01ac067`;
its four divisors use the correctly rounded `__crossgl_uint_to_float1(axis_size)`.
It concretizes `has_w=true`, retains `[WaveSize(32)]` and four
`WaveActiveSum` calls, compiles as `cs_6_6` with `-enable-16bit-types`, and
must execute numerically through Direct3D 12 WARP on Windows CI. The generated
software-subgroup GLSL is 10,402 bytes with SHA-256
`0437c2fe66173fd5b4b3c0f76d60fd3ed3dd5e45fc3704b971f3fc84e38989d0`.
It retains deferred OpenGL specialization constant `20`, emits eight control
barriers with no hardware-subgroup extension or SPIR-V group-nonuniform
instruction, and passes `glslangValidator` plus `spirv-val`.

Both references retain the source's explicit `metal::precise::sqrt` through the
integer square-root helper. Relative to the previous artifacts, the only kernel
change is that helper, its declaration, and one call substitution. Resource
interfaces, specialization values and source-map origins are unchanged. The
previous and current OpenGL kernels produce identical readbacks for this
workload; an unchanged original Metal kernel, specialized with `has_w=true`,
also passes the same numerical limits. This local comparison does not replace
the required complete-kernel Windows execution.

The OpenGL ABI package keeps its exact JSON runtime variant registry ready and
actionable. Its generated native C++ registry header is deliberately marked
unavailable with reason `specialization-requires-deferred-compilation`, because
the GLSL source still requires specialization. The runtime derives a bounded
deferred-compilation request, compiles GLSL to SPIR-V, applies `has_w=true`
through the OpenGL SPIR-V specialization API, then dispatches the resulting
module. This is not an unavailable or blocked runtime variant.

The eight-resource native-loader ABI contains float32 `x`, `w`, `g`, `gx`, and
per-row `gw` buffers plus float32 epsilon, uint32 axis-size, and uint32
weight-stride scalar blocks. The deterministic request compares all 32 `gx`
and 32 `gw` values at `7e-5` absolute and relative tolerance. A local Linux
arm64 llvmpipe EGL run passed with maximum absolute errors below `4.82e-8` for
`gx` and `5.44e-8` for `gw`; Linux CI requires the same deferred SPIR-V Mesa
execution.

This proof is intentionally one row: the one-row boundary makes the per-row
`gw` temporary equal to the host's final reduced weight gradient. It does not
include the separate bias-gradient reduction, multi-row weight reduction,
`has_w=false`, float16 or bfloat16, looped entries, other axis sizes, MLX host
redirection, or the full MLX test suite.

Validate the fixture schema, provenance, deterministic identities, and bounded
evaluation with:

```bash
.venv/bin/python -m pytest -q -n auto \
  demos/integrations/mlx/tests/test_dispatch_contract_fixture.py
```

The checked-in
[`contracts/logsumexp.dispatch.json`](contracts/logsumexp.dispatch.json)
fixture captures two float32 block-reduction workloads from the pinned MLX
operation and gradient tests. The records use axis sizes 32 and 1025 to exercise
distinct results of the pinned host formula
`32 * ceil_div(ceil_div(axis_size, 4), 32)`: workgroup sizes `[32, 1, 1]` and
`[288, 1, 1]`. Both dispatch one output row, require a 32-lane subgroup, and
select `block_logsumexp_float32`. The looped path for axis sizes above 4096 and
the float16 and bfloat16 entries remain outside this bounded contract.

The companion
[`contracts/logsumexp.native-loader.dispatch.json`](contracts/logsumexp.native-loader.dispatch.json)
pins the same bounded workloads to historical corpus commit
`846d176227a0ac13d2667e58d2bb68b322109ab0`. The upstream kernel, host dispatch
formula, and referenced workload coverage are unchanged between the two
revisions, so the two evaluated variant sets are identical; only revision
provenance differs. The bounded Direct3D execution, OpenGL toolchain, and
bounded OpenGL software-runtime tests use this companion contract. The
historical contract remains attached to the recorded 40-unit frontier.

The repository harness evaluates both records against the unchanged pinned
`logsumexp.metal` source and emits one standalone DirectX artifact per workload.
It verifies source, contract, dispatch, template-materialization, and artifact
identities; checks the generated max and sum wave reductions, group barriers,
exponential, logarithm, output store, workgroup size, and `[WaveSize(32)]`; and
requires official DXC `cs_6_6` compilation with warnings as errors on Windows.
This establishes translation and native compiler acceptance for the two listed
test-derived dispatches.

The refreshed HLSL references retain the exact dispatch and resource interfaces.
Complete body comparison identifies only explicit unsigned 64-bit operand
promotions and sign-bit negation of the infinity and maximum-finite constants.
Both workgroup variants compile to byte-identical DXIL before and after these
source changes with the pinned DXC release and warnings fatal. Windows native
execution remains required; matching compiled modules is not a new execution
result or a full upstream-suite claim.

A subsequent review of these two historical `846d1762` HLSL references adds
thirteen private placeholder fields in otherwise empty helper structs. Removing
only those declarations reproduces both previously accepted source hashes.
The accepted and current sources again compile to byte-identical DXIL with
warnings fatal; entry points, resources and dispatch contracts are unchanged.
This review does not update other corpus references or change the required
Windows numerical test.

Both revision-specific contracts produce the same two dispatch variants as
standalone OpenGL artifacts.
Each generated GLSL file requires `GL_KHR_shader_subgroup_basic`, declares
`CROSSTL_REQUIRED_SUBGROUP_WIDTH` as 32, and checks `gl_SubgroupSize` before any
translated kernel work. The portability report records the matching
`GL_KHR_shader_subgroup` host requirement and `GL_SUBGROUP_SIZE_KHR` query, and
the runtime artifact manifest retains both the local size and exact subgroup
width. Linux CI requires `glslangValidator` and `spirv-val` to accept both
artifacts. The built-in Python runtime and generated C++ adapter reject a
reported width other than 32 before compiling the shader or allocating its
resources.

The default OpenGL path remains the guarded hardware-subgroup path described
above. A separately selected current-corpus axis-size-32 artifact opts into
``project.source_options.metal.target_options.opengl.software_subgroup_width =
32``. Because its complete workgroup is one 32-lane subgroup, CrossTL can prove
that ``simdgroup_index_in_threadgroup`` is zero and that
``thread_index_in_simdgroup`` is ``gl_LocalInvocationIndex``. It lowers the two
scalar ``WaveActiveMax(float)`` and ``WaveActiveSum(float)`` reductions through
shared-memory trees while continuing to reject direct raw subgroup builtins,
helper-contained operations, unsupported payloads, and unproven divergent
control flow. The default hardware artifacts and host width preflight are
unchanged.

The software artifact is 4,846 bytes with SHA-256
``2aeeafbf86fe61d3ddc8b9d4a6e23945ea8a34724f0689cd8a4149f0f1a8c7fc``.
It contains ``CROSSTL_SOFTWARE_SUBGROUP_WIDTH``, no KHR subgroup extension or
``gl_Subgroup*`` use, and no hardware subgroup execution metadata.
``glslangValidator`` and ``spirv-val`` accept its OpenGL/SPIR-V 1.3 module; the
SPIR-V contains ten control-barrier instructions and no group-nonuniform
instruction. Runtime reflection packages ``in_Buffer`` and ``out_Buffer`` as
std430 float32 resources at bindings 0 and 1, plus the 16-byte std140
axis-size block at binding 2, with one ready native load unit.
The reviewed reference adds six shared-memory barriers in the sum/max helpers;
all arithmetic, bindings and existing control barriers are unchanged. Original
Metal and both GLSL revisions were executed for axis sizes 1, 31, 32, 33, 64, 127
and 128 with eight input patterns per size. All 56 results per version match the
stable reference and original Metal within the existing tolerance; old/new GLSL
readbacks match exactly and all 28 suffix guards remain intact. This retained
reference review is separate from the single-workload CI proof below and uses
the historical ``846d1762`` pin, not the current integration pin.

Linux CI requires a surfaceless Mesa EGL dispatch of that software artifact.
The workload binds the 32 values ``(index - 16) / 8``, sets axis size 32,
dispatches one 32-thread workgroup, and compares the single readback with the
stable CPU LogSumExp reference ``3.9978051379373145`` at ``5e-5`` absolute and
relative tolerance. The Windows native-loader test exercises the same bounded
workload through the guarded HLSL artifact and Direct3D 12 WARP. This is exact
translation, packaging, and numerical execution coverage for one finite
current-corpus workload; it does not redirect the MLX runtime or run the MLX
test suite.

The axis-size-1025 record still produces a ``[288, 1, 1]`` artifact containing
nine logical 32-lane subgroups, but it remains outside this LogSumExp software
runtime package. The compiler now admits structurally constrained typed masked
sum, minimum, and maximum reductions, as exercised by the current-corpus
Softmax proof below; this LogSumExp workload has not been packaged or
numerically dispatched under that mode here. Its checked artifact therefore
retains the compatible width-32 hardware contract. Looped reductions, other
dtypes and shapes, and full-runtime parity remain tracked by
[#1894](https://github.com/CrossGL/crosstl/issues/1894).

Validate the LogSumExp contract schema, provenance, deterministic identities,
and bounded workload set with:

```bash
.venv/bin/python -m pytest -q -n auto \
  demos/integrations/mlx/tests/test_logsumexp_dispatch_contract_fixture.py
```

The checked-in
[`contracts/softmax.native-loader.dispatch.json`](contracts/softmax.native-loader.dispatch.json)
fixture selects `block_softmax_float32` at historical commit `846d1762` and records two
host-derived block workloads. Axis size 32 with two rows uses `[32, 1, 1]` and
dispatches two workgroups. Axis size 2049 with one row uses `[544, 1, 1]` and
dispatches one workgroup; 2049 is directly represented in the pinned upstream
`python/tests/test_ops.py::test_softmax` coverage. Both follow
`32 * ceilDiv(ceilDiv(axisSize, 4), 32)`, stay within the host's block-path
limit of 4096, require logical subgroup width 32, and package three scalar
resources: float32 input and output buffers plus the int32 axis-size block.

DirectX emits one guarded `CSMain` artifact for each workgroup size, retaining
`[WaveSize(32)]`. Official DXC 1.9.2602.24 accepts both under `cs_6_6` with
`-enable-16bit-types` and warnings as errors. Their SHA-256 values are
`2da9311be17e763b0c422fe07a677efeb4ae21e3a87f2fe54a08acf933b353f7`
and `f7d69880528cb189ccc81a3a4913230e84aef798b85d59f4aed482e2f442c14c`,
with sizes 4,655 and 5,226 bytes, respectively. Both complete bodies were
compared against hash-verified accepted sources reproduced with translator
revision `1ea183f4`. Each adds one dummy member to 13 unused empty structs;
removing only those declarations recovers the accepted source byte for byte.
No executable statement changes.
The pinned compiler produces byte-identical DXIL for each old/new pair;
reflected resources, materialization and launch contracts are unchanged.
The current MLX pin `9c3d3557` produces the same shaders, with 142
pruned candidates rather than the historical pin's 131. This is compiler and
contract evidence, not a new Windows numerical result. Required native tests
retain their original workloads and tolerances. CI preserves their project
reports, packages, compiler records, dispatch inputs and returned values on
success or failure.
The 32-thread entry is provably one wave and keeps the zero-ID quotient fast
path; the 544-thread entry allocates one uniform ID per physical wave through a
workgroup-synchronized counter. The default OpenGL path remains a separate
guarded hardware-subgroup artifact.
The runtime proof opts into explicit software subgroups instead, producing a
5,755-byte axis-32 artifact with SHA-256
`c77ddf1ad3c364b6e1898232f7d4ce99e2d5f859cc515089368044eb80667573`
and a 7,399-byte axis-2049 artifact with SHA-256
`1ea2c58cd8588b9a61ce3b1414b02cd578ec8b16fe77f9d0a2637c706a087174`.
Complete comparison against the recovered originals accounts for exactly six
shared-memory fences in the subgroup helpers. Computation, resource bindings,
launch geometry and control barriers are unchanged. The tests require all six
fences in both generated GLSL and validated SPIR-V.
Both pass `glslangValidator` and `spirv-val`, contain exactly 11
`OpControlBarrier` instructions, and contain no `OpGroupNonUniform` operation.

The 544-thread artifact partitions the workgroup into 17 independent logical
subgroups. Its subgroup-zero final reductions execute under a lane-dependent
condition, so CrossTL evaluates the branch prefix only for active lanes and
contributes typed identities from inactive lanes before calling the uniform
barriered helper: negative infinity for float maximum and zero for float sum.
The same fail-closed lifting supports 32-bit float, int, and uint sum, minimum,
and maximum identities. Conditional shuffle and structurally ambiguous masked
collectives remain rejected.

Windows CI requires both parameterized workloads to execute through the
Direct3D 12 WARP native loader. Linux CI requires both software artifacts to
execute through surfaceless Mesa EGL; a local Linux arm64 llvmpipe run passed
both workloads. Every output is compared with a stable float32 CPU Softmax
reference at `5e-5` absolute and relative tolerance. This is bounded evidence
for two block float32 workloads only: the looped path above axis 4096, float16
and bfloat16 variants, MLX host redirection, and the full MLX test suite remain
outside the claim. Selected entry-scoped Metal generation currently fails
explicitly with `project.translate.entry-point-target-unsupported`, so this
proof does not claim the Metal round trip.

The historical reduced-frontier aggregate can still list `softmax.metal` under
workgroup-size dispatch blockers because that aggregate run does not consume
this later current-corpus entry-scoped contract. That result describes the
aggregate pipeline coordinate; it does not mean that no bounded Softmax
translation, package, or native runtime integration exists.

The current-corpus dot-product proof selects
`dot_product_float32_it32_tg512_sg16` directly from `dot.metal`. Translation
materializes the concrete `dot_product<float, 32, 512, 16>` entry with a
512-thread workgroup and an exact subgroup width of 32. The generated artifacts
preserve the read-only `float4` storage views as four bit-preserving scalar
loads, both subgroup reductions, the 16-element workgroup reduction buffer, and
the final output store. Their portability report and runtime manifest retain
the source entry, target entry, workgroup size, subgroup requirement, and all
four reflected resource layouts. For DirectX, the 16 physical waves receive
unique, wave-uniform IDs through the synchronized allocator rather than a
lane-varying flattened-index quotient.

The reviewed 5,422-byte HLSL artifact has SHA-256
`f10d13d3dfbed2c5e59f785ec0e9a61c93026d3bead541738e450fe72ae2113e`.
Its source differs from the previous reference only by thirteen unused empty
struct padding members; strict DXC compilation produces identical DXIL. Current
dot artifacts retain file-level source remapping to `dot.metal`, not the older
line mappings. Neither property establishes native DirectX correctness.

Windows CI compiles the HLSL artifact with DXC, packages its native-loader ABI,
and dispatches one workgroup through Direct3D 12 WARP. The bounded workload
computes the dot product of 1,024 float32 values containing `1.0` and `0.25` and
requires a readback of `256.0`. Linux CI independently preserves and compiles
the guarded 5,188-byte GLSL artifact with `glslangValidator`; its SHA-256 remains
`ef69a757339fe09897a38804c27be279a19a7db146e2e02f85f0349c59f3168d`, and it
continues to reject a device subgroup width other than 32 before translated
work.

The native DirectX case is not currently a passing portability claim: the
checkpoint run returned an incorrect value, tracked in
[#2120](https://github.com/CrossGL/crosstl/issues/2120). The original numerical
assertion remains required. Windows CI retains the translation report, package,
loader descriptor, input fixture, dispatch plan, availability details, DXC
command and compiled artifacts, and returned values under `mlx-dot-directx`,
including when validation fails. These records describe the native request and
readback; they do not establish the exact bytes uploaded by the driver.

A separate software artifact is 6,385 bytes with SHA-256
`2b378330ecee1f440abdb7705e1eff91b7ce370fea4c14ea2d0f6ae4c7d6d130`.
It allocates 512 shared float elements and partitions them into sixteen
independent 32-lane logical subgroups. The first sum reduces within each
partition. For the second sum beneath `tid < 16`, inactive lanes contribute the
exact additive identity `-0.0`, so they cannot change any sum, including one of
`-0.0` values, while all 512 invocations reach the helper barriers uniformly;
only the active lanes consume the subgroup-zero result. The artifact emits no
KHR subgroup extension, hardware `gl_Subgroup*` builtin, or group-nonuniform
SPIR-V operation. `glslangValidator` and `spirv-val` accept its SPIR-V 1.3
module, whose disassembly contains four control barriers, three shared-memory
barriers, and no group-nonuniform instruction. Mesa llvmpipe executes the same workload through surfaceless EGL
and requires the same `256.0` readback at `1e-5` absolute and relative
tolerance.
Linux CI retains this software artifact, its compilation outputs, package,
input fixture, dispatch plan, availability and readback, including failed
results. Evidence retention does not alter numerical comparisons.

macOS CI translates the selected Metal round trip to a 2,887-byte artifact with
SHA-256 `1008434054b323e8845546b6c4133793f9341a15786ee08dfcc745d493fc1c90`.
MLX's `metal::vec<float, 4>` views stay native Metal vector pointer views, and
the Metal compiler accepts the artifact. These checks establish OpenGL
numerical execution for one selected kernel workload and do not redirect MLX
host dispatch or run the MLX test suite. The pinned current-tree dot proof
below covers all three dot entries, including float16 and bfloat16, on every
backend.

The current-corpus unary proofs select `v_Squarefloat32float32` and
`v_ArcCosfloat32float32` from the full include-expanded `unary.metal` source.
Each entry-scoped artifact emits exactly one
`unary_v<T=float, U=float, Op=..., N=1>` specialization. Concrete call-argument
typing keeps the out-of-line complex `ArcCos::operator()` body out of the float
artifact while preserving fail-closed handling when that complex overload is
reachable. Square retains `x * x`. ArcCos retains the source
`metal::precise::acos` contract through a portable float32 range-reduction
implementation with no-contraction qualifiers instead of relying on the target
intrinsic's unspecified accuracy. OpenGL materializes each precise helper result
through a collision-safe `precise` local, avoiding non-portable qualified
function return types while preserving SPIR-V `NoContraction`. The generated
artifacts also retain a one-thread workgroup, the source and output buffers, and
the size constant in their reflected runtime interfaces. Deterministic artifact
hashes and the pinned upstream source hash are checked before packaging.

The HLSL Square and ArcCos loader references and current-pin profiled bfloat
Sigmoid reference include private placeholder fields in otherwise empty structs.
Removing only those declarations reproduces each previously accepted checksum.
Strict DXC compilation of the old and new sources produces byte-identical
modules; computations, bindings and dispatch requirements are unchanged.
Only these reviewed references are updated. The required Windows numerical
tests, including Sigmoid's 65,282 non-NaN inputs and output guards, remain
unchanged; this comparison does not establish whole-corpus execution coverage.

The selected OpenGL references include shared union storage and bit-preserving
selection in the precise ArcCos helper. Both complete bodies were compared with
hash-verified original references and passed their existing Linux/Mesa compiler
and numerical tests without changing inputs or tolerances. This bounded loader
proof is separate from the complete historical OpenGL corpus review in
[crosstl#2073](https://github.com/CrossGL/crosstl/issues/2073).

The Square and ArcCos entries now also round-trip through Metal. Square is one
1,015-byte artifact with SHA-256
``244e34b7aa58b7abe7c3ff09f3f51f3aa283a42bf7585bf88200590767032495``;
ArcCos is one 3,131-byte artifact with SHA-256
``b1aef8dc745343835414e8a0fe98a463bc4ca5b6de3212f82a6f1e8f5a9d1351``.
Entry reachability retains only the selected ``struct Square`` or ``struct
ArcCos`` and its call helpers, with no unrelated unary struct, complex ArcCos
body, or illegal ``[[static]]`` member. The ArcCos artifact retains the
portable float32 range-reduction helper with function-local
``#pragma clang fp contract(off)`` and ``reassociate(off)`` directives. Neither
helper changes contraction settings in its callers. Bounded Metal source
reflection records each exact kernel plus read-only buffer 0, read-write buffer 1,
and read-only constant buffer 2.
Their ``[1, 1, 1]`` workgroup sizes are explicitly host-dispatch-owned because
MSL has no fixed source attribute equivalent to HLSL ``numthreads``. Both exact
artifacts compile with ``xcrun -sdk macosx metal -c`` on macOS CI.

The scalar float32 ArcCos reference was reviewed after moving those directives
inside the two helpers and removing file-level contraction resets. All
computations, resource interfaces and fourteen source-map origins are unchanged.
Previous and current sources compile to identical Metal libraries and produce
identical readbacks for 8,211 inputs with eight trailing guards. The unchanged
upstream kernel passes the same inputs within the existing absolute ``1e-6`` and
relative ``1e-5`` tolerances; its finite results differ by at most two ULPs.
This review changes only ``v_ArcCosfloat32float32``. It does not accept other
unary fingerprints or establish complex ArcCos equivalence.

The scalar ArcCos Metal reference also includes an unused-parameter annotation
on the stateless operator's ``self`` argument. Removing only that annotation
reproduces the previously accepted source checksum. Both versions compile with
warnings treated as errors to identical Metal libraries, retain identical
resource interfaces, and produce identical results for the same 8,211 inputs
and eight guards. The legacy unary manifest's byte total is calculated from
its recorded entry sizes; changing that total does not accept additional
artifact fingerprints. These controls use legacy revision ``846d1762`` and do
not replace validation of the active MLX revision or the full runtime port.

The five unsigned-32-bit Sign references were reviewed separately across ``v_``,
``v2_``, ``vn_``, ``gn1_`` and ``gn4large_`` dispatch shapes. Their only historical
source changes are a precise square-root helper and its call in the retained,
unused complex overload. Removing exactly that helper change recovers each
previous complete source hash. Reflected interfaces are unchanged; old and new
sources compile to byte-identical Metal libraries with matching output names.
The current ``9c3d3557`` pin changes the included complex implementation, but
each selected unsigned entry still compiles to the same library.

Original kernels at both pins and their translated counterparts return identical
results for 3,303 values and 40 output guards per path. Cases cover unsigned
boundaries, zeros, odd tails, two-dimensional dispatch and padded three-dimensional
strides; all read-only buffers remain unchanged. Reports retain file-level
provenance, not precise operation mappings. This review updates only the five
historical unsigned Sign identities and the corresponding scalar-subset entry;
complex Sign, other dtypes and DirectX references are not covered by it.

The same five OpenGL references have a separate full-source and native review.
The square-root helper change reconstructs each recorded previous source hash;
interfaces and template specializations are unchanged. Previous sources and
translations from both MLX revisions compile to byte-identical SPIR-V modules
with glslang and pass `spirv-val`. Linux software OpenGL execution returns the
same 3,303 integer results and 40 guards per path as the unchanged Metal kernels
at both revisions. No generated shader is patched before execution. Only these
five historical OpenGL fingerprints are updated; complex Sign and other unary
changes are covered by the subsequent complete-source and compiler review.
That review accounts for all remaining unary HLSL and GLSL identity changes,
including FP8 conversions, roots, logarithms and complex math. It does not
extend this bounded unsigned Sign numerical result to other operators.

The pinned comparison-arithmetic CI step also exercises unsigned Sign across all
five layouts through reflected packages, with exact integer comparisons and
trailing guards. The Metal job additionally runs the unchanged upstream entries
and checks that their read-only buffers remain unchanged. This is bounded kernel
and host-binding coverage, not completion of the upstream MLX test suite.

The required family gate covers all 877 unary entries at the legacy reference
revision `846d176227a0ac13d2667e58d2bb68b322109ab0`, including 694 non-scalar
entries in addition to the 183
scalar ``v_`` entries. The shape split is exact: 183 ``v_`` kernels instantiate
``unary_v`` with explicit ``N=1``; 183 ``v2_`` kernels instantiate ``unary_v2``
with the source default ``N=WorkPerThread<T>::n``; 145 ``vn_`` kernels use that
same default on ``unary_v``; 183 ``gn1_`` kernels instantiate ``unary_g`` with
``N=1`` and ``IdxT=int``; and 183 ``gn4large_`` kernels instantiate
``unary_g`` with ``N=4`` and the source-default ``IdxT=int64_t``. Together they
classify 37 operators, 20 concrete input/output type pairs, and 16 semantic
families.

Every selected run traverses Metal to CrossGL to Metal with zero project
diagnostics, emits one exact operator implementation and kernel, prunes
non-selected operator bodies, and retains only reachable empty helper tags.
The three vector shapes materialize one kernel specialization. Each gather
shape additionally materializes exactly one call-site ``elem_to_loc``
specialization, for 1,243 specializations across 877 artifacts. ``v_``, ``v2_``,
and ``vn_`` reflect input, output, and size resources; ``gn1_`` and
``gn4large_`` reflect those data buffers plus constant shape/stride buffers and
the read-only device ``ndim`` binding.

The generic round-trip support resolves and elides chained source typedefs,
maps native bfloat types and result reconstruction, materializes constrained
free operators with source-compatible overload selection, infers concrete
aggregate aliases, recognizes branch-complete returns, and preserves narrow
``as_type`` storage. FP8 decode admits only the proven read-only immediate
thread-local view of a scalar parameter as a matching single-field aggregate;
local storage, multiple or mismatched fields, escaped pointers, and every other
unimplemented pointer reinterpretation remain fail-closed. The non-scalar
increment additionally preserves ``constant`` pointer provenance after portable
``StructuredBuffer`` lowering, keeps ``const device`` references read-only, and
retains postfix ``out_idx++`` semantics. Retained source union layout metadata
is reconstructed as a native MSL ``union`` rather than an ignored attribute,
and additive shift operands retain explicit grouping for warning-fatal native
compilation.

Every source entry, shape, template, operator, exact ``T``/``U`` pair, semantic
family, SHA-256, byte count, template-default provenance, materialization count,
and host resource contract is pinned by
[`contracts/unary.metal-roundtrip.json`](contracts/unary.metal-roundtrip.json)
and linked by hash from ``expected-gaps.json``. Forty float16 references
are re-pinned because their `precise::` `tan`, `sinh`, `cosh`, `tanh`,
`asinh`, `atanh`, `log2` and `log10` calls now keep the `metal::precise::`
namespace: under `-fno-fast-math` the plain binary32 overloads match it, but
the binary16 overloads are native half implementations (`tan(-0.0h)` is +0).
Native macOS CI invokes
``xcrun -sdk macosx metal -Werror -c`` for every entry and requires 877
non-empty AIR outputs with no warning exemption. This closes selected-entry
translation, reflection, and native compiler coverage for every discovered unary
instantiation. It remains a compiler proof, not Metal numerical execution,
DirectX/OpenGL whole-family coverage, MLX host runtime redirection, or an MLX
test-suite claim.

The complete binary family gate covers all 4,122 binary entries discovered at
the historical `846d1762` pin: fifteen 238-entry base shapes and three 184-entry
work-per-thread shapes. The 18 shapes and 11 concrete kernel templates span 24
operators and 25 concrete input/output type pairs across Boolean, signed and
unsigned integer, float16, float32, bfloat16, and complex64 values. Every
selected run emits one selected operator implementation and one kernel, prunes
non-selected operator bodies, and rejects residual template, ``decltype``,
call-operator, or unsupported-placeholder syntax.

Scalar-scalar artifacts materialize ``binary_ss`` and reflect three buffers.
Scalar/vector and vector/vector artifacts, including 2-D and source-default
work-per-thread forms, reflect those buffers plus ``size``. Fixed one-, two-,
and three-dimensional generalized artifacts add exact ``elem_to_loc_1``,
``elem_to_loc_2``, or ``elem_to_loc_3`` index helpers and two stride constants.
The ``gn2`` and ``gn4large`` forms add ``elem_to_loc_2_nd`` plus shape, two
stride, and rank constants. The resulting 4,122 artifacts contain 6,026 exact
materializations and 19,106 reflected resources across exact three-, four-,
five-, and seven-resource ABIs. Explicit and source-default ``N``/``IdxT``
provenance, call-site helper provenance, and a host-owned ``[1, 1, 1]``
workgroup contract are pinned per shape.

Generic repository-scale repairs behind this family map concrete 64-bit vectors
to native ``longN``/``ulongN`` types, rebind dependent free-operator calls only
to already materialized exact helpers, and select non-explicit constructors for
known contextual aggregate conversions while rejecting explicit-only and
ambiguous cases. Bfloat ``min``/``max`` arguments promote to float before typed
result reconstruction. Discarded type-constructor expressions retain argument
evaluation through an unambiguous ``(void)(...)`` form, and scalar Boolean
relational operands receive their C++ integral promotion explicitly. The
focused regressions retain fail-closed behavior outside those proven forms.

The binary reference artifacts also retain explicit byte narrowing after
arithmetic, Boolean integral promotions, and typed vector list initialization
for index helpers. These conversions preserve the source value types and
evaluation order; they do not change the selected operators or buffer interfaces.

The historical `846d1762` bfloat reference review compares all 4,122 complete
sources against the previous contract: 4,050 are unchanged and 72 change across
`ArcTan2`, `Power`, `Remainder` and `LogAddExp`, covering all 18 shapes. The changes
make float argument conversions and bfloat result boundaries explicit.
`LogAddExp` now selects the source's bfloat `log1p` overload, narrowing its argument
before the call. Kernel bodies outside those conversions, indexing, resource interfaces
and materialization contracts are unchanged. All 72 changed sources pass
strict native Metal compilation and reproduce through the public project API.

Local Metal 3.1 checks compare unchanged upstream kernels with those exact
reviewed sources. Each operator's vector/vector entry covers all 65,536 bfloat
encodings against 12 fixed partners: 786,432 pairs per operator, or 3,145,728
returned values per path. The reviewed output matches every non-NaN word and
NaN classification. The previous `LogAddExp` output differs in 35,900 cases;
the reviewed output has no differences. An additional 216 shape controls
exercise all 72 changed entries, including strided access, broadcast inputs
and partial work-per-thread tails. They verify 4,176 values and 1,728 output
guards per path, with readonly input and constant buffers unchanged. These
checks use `-fno-fast-math`; generated sources also require `-Werror`.
They do not establish every possible operand pair, other binary operators,
another device's floating-point policy or complete host-runtime parity.

Every entry identity, shape, template, operator, exact input/output pair,
semantic family, SHA-256, byte count, materialization contract, and host ABI is
pinned by
[`contracts/binary.metal-roundtrip.json`](contracts/binary.metal-roundtrip.json)
and linked by hash from ``expected-gaps.json``. The prior
[`contracts/binary.scalar-metal-roundtrip.json`](contracts/binary.scalar-metal-roundtrip.json)
remains an exact 238-entry ``ss_`` subset. Ubuntu runs the complete translation,
body, materialization and reflection checks across 24 disjoint shards. Each
shard exports the exact pinned sources and retains its reports and JUnit results.
One dependent macOS job verifies complete entry coverage and every source hash
against the checked-in contract before invoking
``xcrun -sdk macosx metal -Werror -c``. It requires 4,122 non-empty AIR outputs;
no source-warning exemption is used. Missing, duplicate or altered bundle entries
fail before compilation. Compiler commands, diagnostics and output identities are
retained even on failure. The local round-trip test still performs both phases.
This gate requires selected-entry translation, reflection and native compilation
for every discovered binary instantiation. The bfloat review above adds bounded
numerical evidence, not whole-family numerical parity, MLX host-runtime
redirection or an MLX test-suite claim.

The annotation-only reference review at the historical `846d1762` revision
covers 3,999 entries, including 231 scalar entries. Each accepted predecessor
was reconstructed to its exact recorded source hash and byte count. The only
source change is an unused-parameter annotation; both versions compile with
warnings treated as errors and link to byte-identical Metal libraries. Entry
selection, all 6,026 materializations and all 19,106 resource declarations
remain unchanged. This is compiler-equivalence evidence, not an additional
numerical or current-pin MLX test-suite result.

The narrow-integer reference review at the same historical `846d1762` revision
updates 3,735 Metal round-trip identities after the backend began preserving
source 16-bit conversions and discarded narrow updates. They comprise 684
binary (38 scalar), 692 copy, 1,140 reduce, 1,134 quantized and 85 unary
(17 scalar) entries. Every accepted predecessor is recovered from the last
passing CI source export and matches its recorded hash and byte count.
Reversing only the reviewed rewrites recovers every accepted body exactly.
These rewrites are `int(short(x))`/`uint(ushort(x))` conversions and their
vector forms, widened compound updates and increments, and `(void)` casts
around discarded narrow updates. Negative controls on sampled entries show that
the same check rejects mutated loop bounds, widths, signedness, operators,
indices and dropped lines.
All 11,943 current sources compile with each contract's native compiler, with
warnings fatal and a non-empty AIR object per entry. The 3,735 changed sources
and their predecessors also compile with `-fno-fast-math` and empty compiler
streams. Of the changed entries, 1,612 compile to byte-identical AIR and
libraries, including all 692 copy entries. The other 2,123 differ where the
explicit 16-bit conversions survive optimization. The original MLX sources
declare those values as `short`, `ushort` or 16-bit vectors. For the 141 binary
and unary entries in this group, original MLX, accepted and current Metal agree
byte for byte on 32,178,480 output elements per path, with unchanged inputs and
output guards.
Binary operands use the 512 x 512 product of a fixed edge and random value
set; unary inputs are exhaustive. The existing affine proof also passes for
nine changed matrix entries. The remaining reduce and quantized entries have
source and compiler review only. Materializations, resource ABI and shard
coverage are unchanged. This is reference evidence for the historical pin, not
current-pin coverage, whole-family numerical parity or an MLX test-suite claim.

The 15 complex `LogAddExp` references have a separate arithmetic review.
Their precise square-root helper preserves the explicit `metal::precise::sqrt`
call in upstream complex `log1p`; the other square root remains ordinary,
as in the source. Accepted sources reconstruct to their recorded hashes,
and both accepted and current sources compile with `-Werror -fno-fast-math`.
Original MLX, accepted output and current output agree on 12,489 complex
results per path: 291 layout controls across all 15 entries and 12,198
vector/vector pairs covering the magnitude branch boundary, finite random
inputs, subnormals, signed zeros, infinities and NaNs. Every non-NaN word
matches exactly; NaN payloads are not compared. All 512 output guard words
per path and all read-only input and constant buffers remain unchanged.
The reflected interfaces and materialization contracts are unchanged.

[`test_current_complex_logaddexp.py`](tests/kernels/test_current_complex_logaddexp.py)
repeats these native comparisons after fresh project translation at the current
`9c3d3557` integration pin. It runs in the existing macOS binary-math step,
without adding runners or extending its deadline. DirectX and OpenGL skip
this original-Metal comparison; it does not establish their numerical parity,
complete complex input coverage, MLX host redirection or full upstream-suite
execution. The historical artifact contracts remain pinned to `846d1762`.

The 54 floating `ArcTan2` entries have also been reviewed with an explicit
source underflow policy, as described below. The final 54 floating `Power`
references now have a separate numerical review. Their accepted predecessors
reconstruct to the recorded source identities, and fresh project translation
reproduces every reviewed current artifact with unchanged interfaces and
materializations. All 108 old/new sources pass strict Metal compilation and
linking. The historical pin, 4,122 entries and all non-Power identities remain
unchanged by this final reference update.

The Power review executes original MLX, accepted output and updated output
across 162 layout controls and three vector stress domains. Each path checks
171,068 values per initialization, using two different finite output fills to
reject unwritten results, and 2,640 guards in total. Every read-only buffer
remains unchanged. Non-NaN words agree exactly, including zero signs; NaN
payloads are compared by classification. The identity cases include every
binary16 and bfloat16 storage word with exponent one. This is sampled argument
coverage, not exhaustive coverage of all base/exponent pairs.

Current-pin `test_current_real_power.py` separately checks 224,534 values and
48 guards on each generated Metal/OpenGL path, with original Metal source
controls, across its identity, operand-underflow and portable-finite profiles.
The existing oracles and precision bounds remain unchanged; the finite profile
is not claimed bit-identical to native power. Both local target runs pass eight
tests, and independent audits recheck the actual output words and artifact
identities. Original Metal input readbacks are verified; these generated cases
do not request the optional read-only input observations. Those cases already run
in the existing native host matrix on all three targets. This local review
does not establish a Windows execution pass.

All known changed binary Metal references are now reviewed. Whole-family
numerical parity, complete host redirection and full upstream-suite execution
remain unproven. Other Metal corpus work remains tracked in
[#1966](https://github.com/CrossGL/crosstl/issues/1966).

The same selected-entry pipeline translates all 4,122 binary entries to
standalone OpenGL ``main`` artifacts. The schema-v2
[`contracts/binary.opengl-translation.json`](contracts/binary.opengl-translation.json)
contract preserves all 18 shapes, 11 templates, 24 operators, 25 type pairs,
and 6,026 exact materializations while pinning
18,732,199 generated GLSL bytes and 19,106 reflected
resources. Scalar artifacts expose three storage buffers; size-bounded vector
forms add an entry-scoped uniform block. One-dimensional generalized forms add
entry-scoped stride blocks, fixed two- and three-dimensional forms expose stride
storage buffers, and rank-generic forms expose shape and two stride buffers plus
an entry-scoped rank block.

OpenGL cannot implicitly preserve the source's runtime 64-bit buffer indices.
The complete contract therefore records explicit host/runtime bounds of
``[0, 2147483647]`` for ``offset + i``, ``a_idx``, ``b_idx``, ``out_idx``,
``out_idx++``, ``idx.x``, and ``idx.y``. These bounds are declared portability
preconditions, not inferred facts or generated runtime checks; omitting the
relevant proof keeps wide-index translation fail-closed. The generated target
retains exactly the selected operator implementation and one kernel while pruning
all unrelated operator bodies.

Required Linux CI partitions the family into 24 disjoint shards. Every shard
retranslates its exact entries, verifies artifact identity, source/default and
call-site materialization provenance, ``main`` workgroup metadata, and exact
three- through seven-resource host ABI, then compiles with
``glslangValidator --target-env opengl --target-env spirv1.3 -S comp`` and
validates with ``spirv-val --target-env spv1.3``. All 4,122 SPIR-V modules must
be non-empty. All known changed binary references have been reviewed against
their exact accepted predecessors. Of the sources, 3,999 remain unchanged and
123 incorporate power-domain, complex magnitude or source-width rounding
corrections. All 4,122 compile and validate; complete source comparisons and
reflection checks preserve every materialization and resource binding. A
later review re-pins 159 sources for Mesa 25.2's IEEE folds: the 105 complex64
Add, Subtract, Multiply, Divide, Remainder, Power and LogAddExp sources route
MLX's free complex operators through helpers that settle zero operands with
integer tests before a `precise` operation, and the 54 float32, float16 and
bfloat16 LogAddExp sources keep `log1p`'s `1 + x` local `precise` because it
feeds an equality test. All 159 compile and validate. The
math corrections are not numerical-equivalence claims. Whole-family numerical
parity and MLX runtime redirection remain outside this reference review.

The 18 `Remainderfloat16` entries explicitly select
`binary16_remainder_profile = "binary32-quotient"`. This preserves the observed
Metal half-remainder arithmetic, including the source's subsequent sign
adjustment and half rounding. Other binary entries do not enable that profile.
All 18 complete old/new sources and resource interfaces were reviewed, and both
versions passed GLSL compilation and module validation before updating the
references. The generated source retains the arithmetic helpers' license notice.

Native checks against unchanged upstream Metal passed at both the contract's
`846d1762` revision and the goal's `9c3d3557` revision: 1,044 values across all
18 layouts, plus every binary16 input word against 12 fixed partners in the
vector-vector kernel, for 787,476 results and 624 guards per revision.
Inputs retain signed zero and NaN payload bits; result comparisons require exact
words except for NaN payloads, which use classification. This is bounded coverage,
not every possible input pair or whole-family parity.

`tests/kernels/test_current_half_remainder.py` makes 5,176 edge, seeded and
rounding-boundary pairs repeatable through the generated package loader on
each native target. The existing macOS job also runs unchanged upstream Metal;
Windows requires DXC/WARP and Linux requires GLSL/SPIR-V validation and Mesa
execution. Eight guards and bit-preserving input encodings are required. These
checks do not establish MLX host-runtime redirection or upstream test-suite parity.

The [Windows checkpoint run](https://github.com/CrossGL/crosstl/actions/runs/37541475685/job/112535784677)
at `35f5412f` passed all 101 binary-shape checks without skips. Independent replay
of artifact `11449688692` confirms the pinned half kernel's 5,176 results and
eight guards, using the saved encoded inputs and separately rounded reference.
Its HLSL SHA-256 is `f453f410146b187e758888ef83d0822a80d3f870d843ecee6dcaa2f3b9543f9b`;
the executed DXIL SHA-256 is `1ed6f3afeed8f074b3c30079acfa88699828c3db9f35a2b15b6ee693ff568206`.
Saved host inputs are verified; physical driver-upload bytes are not claimed.

Floating remainder has a separate numerical review at `9c3d3557`. Selecting
`binary32_remainder_profile = "flush-arithmetic-subnormals"` together with
`binary32_comparison_profile = "flush-subnormals"` preserves the characterized
source operation and its sign-adjustment comparisons. The bfloat standard-library
wrapper retains its float computation and narrow return. These are explicit
source execution policies, not a universal Metal-device guarantee.

Comparison and remainder policies alone are insufficient. Although the original
bfloat sweep matched all 786,432 pairs against 12 fixed partners, six of eight
additional cancellation pairs differed during the subsequent addition. Three
of 4,472 float32 pairs exposed the same independent operation defect, tracked in
[#2113](https://github.com/CrossGL/crosstl/issues/2113).

Adding `binary32_additive_profile = "rne-flush"` gives matching original Metal,
generated Metal and OpenGL results for 791,416 vector-vector values and 208
guards. This includes the original sweeps and 512 adjacent, opposite-signed
bfloat cancellation pairs. Raw inputs, readbacks and guards are retained without
normalization. Twelve selected project artifacts compile for Metal, OpenGL and
DirectX. These controls do not establish Windows execution or whole-family parity.

The 36 float32/bfloat remainder references select all three policies across the
18 access shapes. Complete historical and updated sources pass strict GLSL
compilation and module validation, with unchanged resource interfaces. At the
historical `846d1762` pin, native replay covers 133 cases, 792,992 values and
1,064 guards without differences. Only these reviewed references are updated;
the other changed binary references remain under review in #2073.

`tests/kernels/test_current_floating_remainder.py` makes 4,475 float32 and 4,984
bfloat pairs repeatable through generated packages at `9c3d3557`. The cases
include the cancellation regressions, signed zeros, subnormals and NaNs. Each
dtype has eight exact guards; finite results and zero signs must match exactly,
while NaNs compare by classification. The existing native jobs require Metal
execution plus unchanged upstream controls, DXC/WARP on Windows, and GLSL
validation plus Mesa execution on Linux. Reports, package source identities,
encoded inputs and readbacks are retained. This does not claim physical driver
upload-byte capture, exhaustive input pairs or upstream MLX test-suite parity.

`tests/kernels/test_current_integer_remainder.py` checks seven pinned
vector-vector kernels through generated packages. Boolean and signed/unsigned
8-bit cases cover every defined operand pair; signed/unsigned 16-bit and signed
32-/64-bit cases use boundaries and deterministic samples. Together they check
168,730 results and 56 output guards per native target. Division by zero and
signed 32-/64-bit minimum divided by minus one are outside the defined domain.
The corresponding 8-/16-bit division uses promoted integer operands and remains
covered. Expected results use an independent integer model with the source's
divisor-sign remainder convention, without tolerances or readback normalization.
macOS also executes unchanged upstream Metal and checks its input buffers.
The package fixture uses reflected physical storage; this is not a claim of
whole-family parity or physical driver-upload capture.
All seven integer entries share one translation and package build, avoiding six
repeated frontend passes per target. Each kernel retains its own compiled module,
descriptor, inputs, readback and original-source comparison. Failed batches retain
the package and per-kernel evidence; batching does not relax the 900-second limit.

The 123 affected integer remainder OpenGL references have complete historical
and updated source review, with unchanged resource interfaces. All 246 versions
pass strict compilation and module validation. At `846d1762`, original Metal
and generated OpenGL agree with an independent integer model across 376 cases,
175,867 results and 3,008 guards covering every affected access shape. The
change replaces signed `%` with explicit truncating division; source sign
adjustment and narrowing remain intact.

Minimum and maximum select `binary32_comparison_profile = "flush-subnormals"`
only for float32 and bfloat operands. This matches the characterized source
comparison while retaining the selected operand's original bits. Half comparisons
retain gradual subnormals. No remainder or additive policy is enabled for these
operators. Signed-zero ties select the second operand; NaNs preserve the selected
operand's sign and payload, including signaling NaNs.

All 108 affected extrema references have complete historical and updated source
review, unchanged resource interfaces and 216 strict compiler/validator checks.
At `846d1762`, original Metal and generated OpenGL match an independent reference
across 354 cases, 820,728 results and 2,832 guards. Every affected access shape is
covered. The vector sweeps include every half and bfloat storage word against
zero in both operand positions and against a permuted word; float32 uses boundary
and deterministic samples. Comparisons require exact words without NaN exceptions
or readback normalization. Saved input encodings, output bytes and unchanged input
buffers are independently replayed. This is not exhaustive pair coverage.

`tests/kernels/test_current_extrema.py` supplies six vector-vector kernels at
`9c3d3557` and checks 28,032 results and 48 guards through generated packages per
native target. Metal and OpenGL pass locally, with unchanged upstream Metal as
an additional control. All six HLSL packages pass strict DXC compilation;
Windows execution remains a required CI check, not an inferred result. The
fixture shares the existing native runners, evidence upload and 900-second
binary-math bound. Physical driver-upload bytes and full MLX runtime parity are
not claimed.

Addition/subtraction selects `binary32_additive_profile = "rne-flush"` for
float32 and bfloat operands only. Half results retain their explicit binary16
rounding. All 108 affected references have complete historical and updated
source review, unchanged resource interfaces and 216 strict compiler/validator
checks. At `846d1762`, 354 layout/domain cases compare 826,888 results and 2,832
guards with untouched Metal and generated OpenGL. The independent audit uses
decoded arithmetic and explicit rounding, checks unchanged input buffers, and
retains raw NaN payload differences.

`tests/kernels/test_current_additive.py` covers six vector-vector kernels at
`9c3d3557`: 34,192 results and 48 guards per target, including cancellation,
rounding midpoints, subnormal boundaries and seeded inputs. Finite results,
infinities, signed zeros and guards require exact words. NaNs are compared by
classification, with differing payload counts in the report; no readback is
normalized. Metal and OpenGL pass locally, and all six HLSL packages compile
with strict DXC. Windows execution is a separate required check. Bit-observable
numeric NaN conversions remain tracked in #2081. This does not establish
bitwise NaN or whole-family parity.

The binary step schedules floating remainder types and compatible translation batches
across its existing two workers. Original Metal library builds are shared per
source root and worker, but source dispatches and their input checks remain per
case. Explicit native-test selection avoids repeating portable tests already
covered on Ubuntu. Neither the native runner count nor the 900-second deadline
increases. The 15 complex-power layouts share setup in three translation/package
batches of five. Every layout retains its own native compiler check, descriptor
and all three datasets. Exact source-entry matching prevents selecting another
artifact from the same package.

The longest binary batches are listed first and subsequent tests are assigned
one at a time with xdist's load scheduler. This avoids leaving large batches
in one worker's queue while the other worker is idle; it does not add workers.

Division selects `binary32_division_profile = "rne-flush"` for float32/bfloat
operands; half division retains gradual subnormals and binary16 rounding. All
54 affected references have complete source-body/declaration review, unchanged
resource interfaces and 108 strict historical/generated compiler-validator
pairs. At `846d1762`, 177 layout/domain cases compare 427,580 values and 1,416
guards with untouched Metal and generated OpenGL. An independent decoded-value
audit checks physical inputs, readonly buffers and readbacks without rewriting
their values. The integer `sizeof` inference correction tracked in #2115 keeps
work-per-thread constants outside the floating arithmetic profile.

`tests/kernels/test_current_division.py` supplies three vector-vector kernels at
`9c3d3557`, covering 31,232 values and 24 guards per target. Metal and OpenGL
pass locally; all three HLSL packages compile with strict DXC. Native Windows
execution remains a separate requirement. Finite values, infinities, signed
zeros and guards use exact words; NaN payload differences are recorded rather
than normalized. This does not establish bitwise NaN or full runtime parity.

Half multiplication retains gradual subnormals and explicit nearest-even
binary16 rounding, without enabling a binary32 source profile. The 18 affected
OpenGL references have complete source review and 36 strict compiler/validator
checks. The only changes are the half-rounding helper and its use at the product
return boundary; all other source bytes and resource interfaces are unchanged.
At both `846d1762` and `9c3d3557`, unchanged Metal matches an independent integer
significand reference across 61 layout/domain cases, 203,284 results and 488
guards. Generated OpenGL matches the same cases. These include all 65,536 input
words against zero in both positions and against a deterministic permutation;
they do not cover every possible operand pair. Arithmetic NaNs are compared by
classification, while finite values, infinities, zero signs and guards are exact.

`tests/kernels/test_current_multiplication.py` supplies the current-pin
vector-vector half, float32 and bfloat kernels through generated packages on
each native target. Half retains its 5,632 input pairs and gradual underflow.
Float32 and bfloat each add 12,800 boundary, exponent and deterministic pairs,
with eight guards per kernel. Their native batch explicitly selects
`binary32_multiplication_profile = "rne-flush"`, flushing subnormal operands and
exact subnormal products before rounding, while retaining zero signs. The
independent integer reference applies binary32 rounding before bfloat narrowing.
Finite values, infinities, zero signs and guards must match exactly; arithmetic
NaNs compare by classification.

The profile is recorded in project reports, packages and per-kernel evidence.
It is selected only for this float32/bfloat multiplication batch, not half,
complex expressions or the complete corpus configuration. Existing artifact
fingerprints are unchanged. All three targets require strict compilation and
native readbacks; macOS also runs unchanged upstream controls. These cases share
the existing binary-math job and its 900-second bound, without adding a runner
or relaxing comparison rules. Selected vector kernels do not establish
whole-family, composed-expression or full runtime parity.

`tests/kernels/test_current_floating_binary.py` owns native execution for these
four fixture modules. The 18 kernels share five translation/package batches:
six half kernels without a binary32 profile, four comparison kernels, four
additive kernels, two division kernels and two multiplication kernels. Compared
with separate per-family batches, this avoids three repeated frontend and package
setup passes per target.
Every kernel retains its compiler check, descriptor, dispatch, readonly input
checks, result evidence and applicable unchanged Metal control. The inventory
test checks all 124,688 values and 144 guards against the original fixtures and
verifies each batch's settings against single-entry configuration. Comparison
policy is selected per operation: selection preserves exact NaN payloads even
when sharing a batch with arithmetic operations that permit payload differences.

`tests/kernels/test_current_atan2.py` executes the current-pin vector-vector
half, bfloat and float arctangent entries in one package batch. Its 31,232 pairs
and 24 guards include exponent boundaries, signed zeros, non-finite values and
deterministic random inputs. The characterized `flush-subnormals` profile is
explicit in the configuration and retained evidence. Results use an independent
high-precision reference with the source's float/half precision bounds; bfloat
checks apply float accuracy before narrowing. Axes, zero signs, output guards
and source-control read-only inputs remain exact. Widened OpenGL narrow results
must still be representable in their original storage type.

The existing 180-second arctangent step requires this test on each native target,
without another runner. Local original/generated Metal and OpenGL checks pass.
The [Windows arctangent step at `f5bd9c68`](https://github.com/CrossGL/crosstl/actions/runs/37659104143/job/112922487897)
passes in 124 seconds under the same limit. The generic source tests also cover
127 scaled copies of the normal/subnormal rounding midpoint and adjacent
operands. This is three-entry numerical evidence, not execution of every binary
layout or the full MLX suite.

Binary reference reconciliation remains open in #2073. The precise arctangent
fix (#2118) is reconciled across all 54 OpenGL and 54 DirectX variants, including
float32. Both corpus configurations explicitly select the characterized
`flush-subnormals` profile for arctangent only. Complete source and interface
review preserves indexing, bindings, dispatch geometry and materialization.
All recorded baseline identities are verified before accepting changed output:
the older OpenGL narrow-type baselines also account for the intervening
source-width rounding helpers. Both old and new artifacts pass strict native
compilation, and an independent build reproduces all 54 current DXIL modules.
Other binary entries retain their previous identities.

Additional local execution covers all 54 reviewed OpenGL arctangent entries
across 162 broadcast, strided and tail cases: 3,132 results and 1,296 guards per
path. Generated OpenGL and unchanged current-pin Metal pass the independent
precision-specific comparisons, with every read-only buffer unchanged.
Dispatch sizing follows the source's `WorkPerThread<T>` rule, including two
float32 values or four narrow values per thread where applicable. These layout
checks complement the larger vector-vector domain checks above; they do not
claim DirectX execution of every layout or full MLX-suite parity.

The historical Metal corpus and its scalar subset now select the same
`flush-subnormals` policy for `ArcTan2` only. This is a demo configuration
choice, not a change to the translator's default policy. Before that explicit
selection, the generated helper satisfied the preserve-subnormals model but
did not match the original Metal device's underflow behavior: the vector
controls had 208 bfloat and 247 float violations of the flush-profile checks.
No precision bound or numerical oracle was relaxed to reconcile those cases.
The selected policy is required in each project report and artifact's
provenance; unrelated operators retain their existing settings.

All 54 accepted Metal sources reconstruct to their exact recorded identities.
The accepted and updated sources pass 108 strict Metal compile/link checks,
with unchanged reflected interfaces, materializations and indexing bodies.
Local original-source, accepted-output and updated-output execution covers
165 workloads: the 162 layout controls above plus the three vector/vector
domains. Each path checks 34,364 results and 1,320 guards against the existing
precision-specific oracle, with every read-only buffer unchanged and finite
sentinels rejecting unwritten outputs. The observed maximum storage errors
are one half ULP, zero bfloat ULPs and two float ULPs for the updated output;
this is precision-bounded agreement, not bit identity for every finite result.

Fresh current-pin package execution separately repeats the existing 31,232-pair
test, including unchanged original Metal controls. CI already requires that
test in its bounded arctangent step; this review adds no runners or deadlines.
The artifact contracts retain their historical `846d1762` pin. These results
do not establish complete MLX host redirection or full upstream-suite parity.

Floating-power domain failures (#2119), narrow
log-add-exp accuracy (#2068), and float32/bfloat multiplication's source subnormal
policy (#2114) remain separate numerical work. Successful compilation alone
does not justify accepting those references.

The Windows run at `6f325314` exposed signaling-NaN differences in half minimum
selection (#2116). Later retained readbacks preserve all 9,344 half minimum/maximum
results and 16 guards exactly. At `90b6f596`, the complete required binary-math
step passes in 594 seconds under the unchanged 900-second limit. That step result
does not establish complete host-runtime redirection or full upstream-suite parity.

The native arithmetic step runs device execution, original-source controls and
Metal linkage checks only. Platform-independent configuration and generation
tests remain in the complete Ubuntu suite. The native selection is checked
against every opt-in device test in its modules; the 120-second bound and all
numerical cases remain unchanged. No additional native runners are introduced.

All 60 signed and unsigned 64-bit shift references also have complete source and
interface review. Their source changes only convert shift counts to the width
required by the GLSL target; operand widths and buffer layouts are unchanged.
Fresh translations reproduce the reviewed artifacts, and both historical and
updated sources pass strict compilation and module validation.

At the historical `846d1762` pin, untouched Metal and generated OpenGL agree with
an independent integer model across all 15 layouts: 184 cases, 7,332 values and
1,472 guards per path. Vector controls cover every shift count from 0 through 63.
Signed left-shift inputs stay within representable bounds; signed right-shift
cases include negative values. Results are compared as exact 64-bit words. This
does not claim behavior for invalid shift counts or full MLX runtime parity.

The DirectX sibling contract translates all 4,122 discovered entries to
standalone ``CSMain`` artifacts and pins their exact HLSL identities in
[`contracts/binary.directx-translation.json`](contracts/binary.directx-translation.json).
It preserves the same 18 shapes, 11 templates, 24 operators, 25 type pairs,
6,026 materializations, and seven explicit index-range preconditions. DirectX
resource namespaces retain the source buffer coordinates, while nine shapes
that consume ``threads_per_grid`` add ``CrossGLDispatchInfo``. The resulting
21,248 reflected DirectX resources span exact three-, four-, five-, six-, and
eight-resource target interfaces; the rank-generic forms expose shape and
stride buffers, an entry-scoped rank constant block, and generated dispatch
metadata.

Computed-result bfloat ``ArcTan2``, ``LogAddExp``, and ``Power`` paths expand
both operands to float, compute in float, and reconstruct the result in the
exact low-16-bit bfloat register representation with round-to-nearest ties-to-
even. ``Maximum`` and ``Minimum`` expand only for comparison and return the
selected original bfloat payload without requantization. The contract remains
explicit and fail-closed: unrelated bfloat builtins are not admitted, and
storage still requires Shader Model 6.2 native 16-bit types.

Required Ubuntu CI partitions the family into 24 disjoint shards. Every shard
retranslates its exact entries, verifies deterministic identity, source/default
and call-site materialization provenance, ``CSMain`` workgroup metadata, and
exact three- through eight-resource host ABI, then compiles with checksum-pinned
Linux DXC using ``-enable-16bit-types -WX -T cs_6_2 -E CSMain``. All 4,122 DXIL
modules must be non-empty. This closes discovered binary DirectX translation,
reflection, and native compiler coverage; it does not claim numerical
execution, MLX host-runtime redirection, or MLX test-suite parity.

These compiler-only shards do not require a Windows runner. Direct3D 12
execution and readback checks remain on Windows; compiling DXIL on Linux is
not a substitute for those runtime tests.

The current HLSL references total 16,599,843 bytes. Complete comparisons cover
all 4,122 predecessor and generated bodies, all 21,248 reflected resources,
and strict compilation of both sets. Adding legal empty-record padding leaves
4,053 DXIL modules byte-identical. The remaining 69 artifacts incorporate
power-domain or precise complex-magnitude corrections and compile successfully;
their changed binaries are retained as such, not treated as equivalent.
A follow-up review preserves explicit precise returns in 69 sources: 54
retain identical DXIL and 15 retain additional precision constraints. All
4,122 sources compile with warnings fatal. Numerical references and tolerances
remain unchanged; these compiler checks do not qualify Windows execution.
Bindings, dispatch contracts, source pins and specialization counts remain
unchanged. The contract preserves earlier review provenance and records this
follow-up separately. A later follow-up changes 69 more sources. The 54
binary16, binary32 and bfloat16 Remainder entries now select the exact
remainder helpers the OpenGL translation already uses: HLSL fmod returns NaN
for an infinite divisor, and for binary16 it keeps -0 where Metal's fmod,
computed as x - trunc(x / y) * y in binary32, gives +0. The 15 complex64
Remainder entries round their 64-bit integer quotient lanes to binary32 once,
because WARP's native conversion rounds twice. All 69 compile with warnings
fatal. A further review updates the 90 complex64 Add, Subtract, Multiply,
Divide, Remainder and LogAddExp sources: MLX's free complex operators now
return `precise` results, so DXC no longer folds the zero components of
complex constants out of their products. All 90 compile with warnings fatal.
These checks do not establish numerical execution of the complete
binary family.

The complete historical reduction gate covers all 2,396 host-named entries
from `reduce.metal` at commit
`846d176227a0ac13d2667e58d2bb68b322109ab0`. Three base forms cover
initialization, all-reduce, and simple row reduction. Six multidimensional
families each combine 32- and 64-bit index types with one, two, and five
logical dimensions, yielding 39 exact ABI shapes over nine source templates.
The family spans `And`, `Or`, `Sum`, `Prod`, `Min`, and `Max`, 44 concrete
operator types, and 13 source input and output types. Its compact schema-v2
contract records only deterministic classification, generated identity and
size, exact materialization identity/count, and exact resource identity/count
per entry; full transient translation reports and native AIR objects stay out
of git.

The one- through twelve-resource interfaces contain 25,088 reflected resources
in aggregate. Each shape pins normalized Metal resource types, exact binding
coordinates, access, and a host-owned `[1, 1, 1]` workgroup contract. Generic
lowering now distinguishes comparison operators inside non-type template
arguments from closing angle brackets, folds proven integral non-type struct
arguments for specialization selection without renaming source-spelled
materializations, normalizes visible aliases and raw scalar pointer spellings,
and admits pointer-plus/minus-integral built-in arithmetic before aggregate
free-operator dispatch. Concrete-helper deduplication ignores comments and
external whitespace but preserves exact preprocessing-token literal bytes.
Compile-time static members used by references, address-taking, receivers, or
unevaluated storage-sensitive expressions are hoisted into address-space-correct
Metal `constant` storage instead of being substituted as rvalues. Unknown
aliases, unresolved address provenance, non-integral pointer offsets, and
incompatible overloads continue to fail closed.

Reduction reference artifacts retain explicit byte-value conversions, internal
helper linkage and the source shuffle offset's 16-bit width. The reference
refresh preserves every entry, operator, materialization and resource interface;
the updated sources compile with warnings treated as errors. Compilation alone
does not establish numerical parity for the full reduction family.
The remaining 1,969 source identities were reconciled after reviewing 2,096
current artifacts against their exact accepted predecessors. Every pair has
identical reflected interfaces and produces a byte-identical Metal library;
the differences are unused-parameter annotations and explicitly typed integer
boundary constants. The other 300 entries retain their previously reviewed
sources and compiler evidence. All 9,216 materializations and 25,088 resources
are preserved. This completes the historical reference review, not numerical
qualification at the current integration pin.

Required Ubuntu CI partitions the sorted family into 24 disjoint shards: 20 with
100 entries and four with 99. Each shard retranslates its exact entries,
verifies deterministic artifact identity, materialization, workgroup metadata,
and reflected ABI, then exports its checked sources. One dependent macOS job
verifies the complete bundle against the pinned contract before invoking
`xcrun -sdk macosx metal -Werror -c` for every source and requiring a non-empty
AIR object. Both phases retain evidence; missing, duplicated or altered sources
fail before compilation. This removes 23 macOS jobs; the gate still requires all
2,396 AIR outputs. The default local test still translates and compiles each
selected entry.
This proves complete discovered-reduce translation, reflection, and
native compiler acceptance; it does not claim Metal numerical execution,
OpenGL or DirectX whole-family translation, MLX host-runtime redirection, or
MLX test-suite parity.

Windows CI compiles the selected HLSL entries with DXC and executes them through
the native loader on Direct3D 12 WARP. Linux CI compiles the GLSL entries with
`glslangValidator` and executes them through the same loader contract on a
surfaceless Mesa OpenGL context. The Square workload maps
`[-3.0, -1.5, 0.0, 2.0, 4.25]` to
`[9.0, 2.25, 0.0, 4.0, 18.0625]`. The ArcCos workload maps
`[-1.0, -0.5, 0.0, 0.5, 1.0]` to the corresponding five float32 arccos values.
Both platforms enforce explicit numerical tolerances. This is numerical
evidence for two selected unary specializations. This numerical evidence
does not cover the other unary operations or dtypes, redirect the MLX host
runtime, or run the MLX test suite.

The checked-in
[`contracts/rms_norm.dispatch.json`](contracts/rms_norm.dispatch.json) fixture
captures 12 distinct dispatch artifacts exercised by the pinned
`python/tests/test_fast.py::test_rms_norm` and `test_rms_norm_grad` workloads.
The forward records cover float32 workgroups of 32, 64, and 128 threads,
float16 and bfloat16 workgroups of 32 threads, and the 1024-thread looped path.
The VJP records cover 32- and 64-thread single-row paths plus the 1024-thread
looped path, each with both concrete values of function constant `20`
(`has_w`). Axis sizes 31, 32, and 33 share the same 32-thread artifact and are
recorded as covered inputs rather than duplicate artifacts.

The repository harness evaluates those finite records against the unchanged
pinned `rms_norm.metal` source and requires 12 deterministic DirectX artifacts.
Every artifact retains its workload inputs, dispatch workgroup count,
host-derived `numthreads` value, concrete function constants, and
`[WaveSize(32)]` enforcement. Windows CI compiles every artifact with official
DXC `cs_6_6`, `-enable-16bit-types`, and warnings as errors. This is complete
translation and compiler coverage for the listed unit-test dispatch variants;
the MLX runtime is not redirected to these artifacts, the kernels are not
executed, and numerical parity is not claimed.

All twelve compiler references preserve the source's precise reciprocal-square-root
operation. Complete old/current body review isolates one shared helper, its
declaration and one call substitution per artifact. Resource interfaces, dispatch
identities, specialization constants and source origins are unchanged. Both sets
pass strict DXC compilation. The helper matches the independently checked native
arithmetic implementation; this does not extend the bounded 32-element forward
execution proof to other dtypes, gradients or larger workloads.

Validate the RMSNorm contract schema, provenance, deterministic identities, and
bounded workload set with:

```bash
.venv/bin/python -m pytest -q -n auto \
  demos/integrations/mlx/tests/test_rms_norm_dispatch_contract_fixture.py
```

The focused `prove_rms_norm_specialization.py` gate fixes the project-level
RMSNorm specialization contract to the same upstream commit and to
`rms_norm.metal` SHA-256
`5d411a2350ba7ddf84eb35f9dcac7cde0d441bd55fa1e9e1ccc61d490d428dee`.
It translates the upstream source through `crosstl.project.translate_project`.
The source check also requires all four kernel templates to retain
`constexpr int SIMD_SIZE = 32`, both simdgroup lane/group builtins, and all 12
`simd_sum` calls. This is a semantic input contract: compiling a target shader
without an exact 32-lane subgroup guarantee is not sufficient evidence for
these reductions.
The pinned MLX host computes single-row workgroup width as
`32 * ceil_div(ceil_div(axis_size, 4), 32)` and uses the selected pipeline's
`maxTotalThreadsPerThreadgroup` for looped kernels. The proof materializes
`[32, 1, 1]` and `[64, 1, 1]` as representative upstream-valid results of
those host formulas. These two sizes deliberately do not claim complete axis,
device-limit, or runtime-selected workgroup coverage.

For DirectX, two named project variants combine those workgroup sizes with the
required `has_w` function constant through both selector forms:
`has_w=false` by name at `[32, 1, 1]` and `"20"=true` by numeric ID at
`[64, 1, 1]`. The gate verifies variant selector and workgroup provenance,
concrete specialization materialization, the pinned source hash, and the
generated `static const bool has_w` value. The DirectX project configuration
sets `subgroup_width_rules["mlx/backend/metal/kernels/rms_norm.metal"] = 32`.
Each HLSL library artifact must retain the exact subgroup-rule provenance and
Shader Model 6.6 enforcement metadata for all 12 pinned host-named entries,
emit one `[WaveSize(32)]` and one matching `numthreads` attribute per entry,
and retain the reflected workgroup contract. Windows CI uses two
warning-as-error DXC runs to compile one reflected representative entry from
each HLSL library with `cs_6_6`; native 16-bit artifacts additionally pass
`-enable-16bit-types`.

For OpenGL, the `workgroup_32` and `workgroup_64` variants leave `has_w`
deferred, retain `layout(constant_id = 20)`, and split each host-named entry
into a standalone `main` artifact. This existing RMSNorm proof deliberately
does not configure an OpenGL subgroup-width rule, so subgroup provenance and
enforcement metadata remain absent. Linux CI compiles all 24 GLSL artifacts to
OpenGL SPIR-V 1.3 and validates all 24 binaries with `spirv-val`; that result
does not establish the source's 32-lane simdgroup or `simd_sum` semantics. The
bounded LogSumExp proof above exercises the exact-width OpenGL contract without
inflating the RMSNorm claim.
OpenGL keeps each rebased row pointer in a signed `int` logical offset, so the
OpenGL project configuration supplies 14 caller-trusted index-range assertions
in `[0, 2147483647]`: the `size_t` row rebase
`gid * uint64(axis_size) + lid * RMS_N_READS`, the in-row loop indices `i + r`
and `r + i`, and the 11 complete row accesses (`x + i`, `out_ + i`, `g + i`,
`gx + i`, `gw + i`, `x + i + r`, `x + r + i`, `out_ + r + i`, `g + i + r`,
`gx + i + r`, and `gw + i + r`). The proof requires the report to record
exactly these assertions.

The 24-artifact specialization proof above remains translation and native
compilation evidence only. A separate bounded proof selects the historical
`rmsfloat32` entry at commit
`846d176227a0ac13d2667e58d2bb68b322109ab0` through
[`contracts/rms_norm.native-loader.dispatch.json`](contracts/rms_norm.native-loader.dispatch.json).
That contract fixes a `[32, 1, 1]` workgroup, subgroup width 32, two dispatch
workgroups, and no function-constant value for the forward entry. Entry-scoped
runtime reflection now omits the unreachable VJP-only `has_w` constant while
preserving reachable constants, resolving
[#1795](https://github.com/CrossGL/crosstl/issues/1795). The selected template
materializes one `rms_single_row<float, RMS_N_READS>` specialization from four
reachable specializations while pruning 168 unrelated candidates.

The generated HLSL artifact is 7,053 bytes with SHA-256
`4ef21ff639d0bdd21b7a5da5c51b13893198d6a2c972c8a96280607af8b45198`;
it retains `[WaveSize(32)]` and compiles as `cs_6_6` with
`-enable-16bit-types`. Its only change from the previous 6,207-byte source is
the correctly rounded `__crossgl_uint_to_float1(axis_size)` divisor described
above. The generated GLSL
artifact is 6,882 bytes with SHA-256
`04b447795637318a75158fac8025e189a6374dc8a97311d49597dc9f04f76fad`.
Its explicit target-scoped 32-lane software subgroup lowers scalar
`WaveActiveSum`, emits six `OpControlBarrier` instructions and no
`OpGroupNonUniform` instruction, and passes `glslangValidator` plus
`spirv-val`. Default OpenGL generation remains on the hardware-subgroup path.

The updated forward references replace the target intrinsic with the source's
explicit precise reciprocal-square-root helper. Complete source comparison
isolates the helper, its declaration and one call substitution; resources and
source-map origins are unchanged. The new OpenGL readback matches unchanged
original Metal for all 64 selected results, with eight source-output guards and
unchanged read-only buffers. HLSL strictly compiles and uses the same helper as
the independently replayed Windows reciprocal-root test. Full updated-kernel
execution remains required on Windows; no tolerance or dispatch setting changes.

The six-buffer native-loader ABI packages float32 input, weights, and output,
plus 16-byte scalar blocks for epsilon, axis size, and weight stride. The
numerical workload runs two deterministic float32 rows of axis size 32 with 32
weights and compares 64 outputs against
`x * w * rsqrt(mean(x * x) + epsilon)` at `3e-5` absolute and relative
tolerance. CI requires the same package and readback on Direct3D 12 WARP and
Mesa software OpenGL. This is bounded execution evidence for one forward
float32 workload; it does not redirect the MLX host runtime, run the MLX test
suite, or cover complete RMSNorm runtime parity. In particular, that forward
proof does not cover VJP, looped, float16, or bfloat16 entries, other axis
sizes, or the remaining host/device dispatch space.

A sibling current-corpus VJP proof selects `vjp_rmsfloat32` through
[`contracts/rms_norm_vjp.native-loader.dispatch.json`](contracts/rms_norm_vjp.native-loader.dispatch.json).
It fixes one float32 row of axis size 32, `has_w=true` through function constant
ID `20`, a `[32, 1, 1]` workgroup and 32-lane subgroup, one dispatch workgroup,
and the two explicit 0-through-31 index-range assertions needed by the bounded
input and cotangent views. OpenGL additionally asserts the four composed row
accesses `x_in + i`, `g_in + i`, `gx_out + i`, and `gw + i` in
`[0, 2147483647]`, because it keeps each rebased row pointer in a signed `int`
logical offset. Materialization selects
`vjp_rms_single_row<float, RMS_N_READS>` from four reachable specializations
while pruning 168 unrelated candidates.

The generated HLSL is 10,398 bytes with SHA-256
`45e296590cddf7685c3d4f6daa4720db44fa664024c6bcef56238e2cdd1f047d`;
its divisors use the correctly rounded `__crossgl_uint_to_float1(axis_size)`.
It concretizes `has_w=true`, retains `[WaveSize(32)]` and four
`WaveActiveSum` calls, and compiles as `cs_6_6` with
`-enable-16bit-types`. The generated software-subgroup GLSL is 10,414 bytes
with SHA-256
`d9cc8f4bf9d1ebaa82d0f68b742cc5ac018d0c871f1e827b2fc9fb19e8cb32c9`.
It retains deferred OpenGL specialization constant `20`, emits six control
barriers with no hardware-subgroup extension or SPIR-V group-nonuniform
instruction, and passes `glslangValidator` plus `spirv-val`.

Complete source comparison isolates the precise reciprocal-square-root helper,
its declaration and one call substitution; resources and source-map origins are
unchanged. Previous and updated OpenGL readbacks are identical for all 64
gradient values. Comparison with unchanged original Metal gives a maximum
absolute difference below `5.97e-8`, with 16 source-output guards and unchanged
read-only buffers. Strict HLSL compilation and independently replayed Windows
helper tests pass; complete updated-kernel execution remains required on
Windows. No numerical tolerance or dispatch setting changes.
The later empty-struct compatibility update changes only 13 unused HLSL
declarations in each RMSNorm artifact. Both revised HLSL sources produce
byte-identical DXIL to their accepted references. OpenGL's gradient loop now
distinguishes a pointer's storage from the value-only calculation of its
offset; its generated source is identical to the accepted reference.

RMSNorm VJP exercises a canonical runtime row loop. The explicit software
subgroup path accepts that loop only after proving its initializer and bound
workgroup-uniform from `gl_WorkGroupID` and read-only scalar blocks; lane-varying
inputs, bound mutation, unresolved calls, and `break`, `continue`, or `return`
remain fail-closed. Default OpenGL generation remains on the hardware-subgroup
path.

The ten-resource native-loader ABI contains float32 `x`, `w`, `g`, `gx`, and
one-group `gw` buffers plus float32 epsilon and uint32 axis-size, weight-stride,
row-count, and rows-per-group scalar blocks. The deterministic request compares
all 32 `gx` and 32 `gw` values at `7e-5` absolute and relative tolerance. A
local Linux arm64 llvmpipe EGL run passed deferred GLSL-to-SPIR-V execution with
maximum absolute errors below `3.54e-8` for `gx` and `3.40e-8` for `gw`; CI
requires the same package and numerical readback through Direct3D 12 WARP and
Mesa software OpenGL.

This proof is intentionally one row and one group, so the kernel's group-local
`gw` result is also the final host-reduced weight gradient. It does not cover
multi-row weight reduction, `has_w=false`, float16 or bfloat16, looped entries,
other axis sizes, MLX host redirection, or the full MLX test suite.

`fence.metal` emits no DirectX, OpenGL, or Vulkan target artifact. The harness
requires the target-specific structured diagnostics and the exact requested
atomic-fence operands under #1537 instead of accepting generated barrier text as
semantic evidence.
Future scouts should add issue-backed blockers only when there are
concrete repros. Host runtime integration gaps should be handled in repository
integration code or downstream runtime adapters, not hidden as shader
translation successes.
The full GEMV Vulkan gate materializes all 226 source specializations and emits
224 `GLCompute` entry points. The generated artifact passes both `spirv-as` and
`spirv-val` for `vulkan1.1`, with zero semantic warnings and no known codegen
fallbacks. This is structural validation only: runtime integration is not
included, and the result does not establish numerical runtime parity.

Read-only scalar storage-pointer reinterpretation now has a shared AST contract
and target lowering for DirectX, OpenGL, and Vulkan. A 32-bit scalar storage
resource can be viewed through aligned 8-, 16-, or 32-bit scalar elements;
source pointer offsets are converted to bytes before target indexing, and the
generated OpenGL and Vulkan artifacts pass native validators. Writable views,
64-bit backing layouts, and incompatible address-space or alignment cases remain
explicit diagnostics under CrossGL/crosstl#1546. Metal `dispatch_bool` callbacks
with one integral-constant parameter now lower to a runtime branch whose two
callback bodies retain distinct compile-time `true` and `false` values. Nested
dispatches expand the full Cartesian specialization and reduced DirectX,
OpenGL, and Vulkan project fixtures pass their native validators. Other callback
helpers remain tracked in CrossGL/crosstl#1554. Concrete `const_for_loop`
callbacks now expand in source order when all three bounds are integral, the
callback has reference capture and one `auto` parameter, and its body has no
callback-local control transfer. Expansion is enabled only when the source
defines the recognized `integral_constant`, `Int`, recursive loop, and arithmetic
operator contracts; unrelated helpers with the same names remain opaque. Nested
loops preserve exact
`integral_constant<int, N>` argument types; unresolved or unsafe callbacks remain
opaque and fail through the existing structured materialization path. A reduced
Vulkan fixture preserves four stores at indices `0`, `1`, `4`, and `5`, then
passes `spirv-as` and `spirv-val`. OpenGL expected-type propagation for the same
aggregate call arguments remains tracked in CrossGL/crosstl#1559.

An isolated high-budget `quantized_nax.metal` Vulkan project run now expands the
concrete NAX tile callbacks, resolves conditional function-local dimensions, and
materializes `NAXTile<T, BR, BC>` as concrete 2-by-2 specializations. Explicit
member-template binding now preserves `float16_t` threadgroup arrays, and the
template-hostile project path initializes the verified compile-time callback
contracts before member lowering. Bounded, non-variadic namespace-scope alias
templates are now canonicalized after callback and member lowering, then their
backing struct templates are materialized once more. The high-budget report no
longer contains any `Int<...>` use or `using Int` declaration and has fallen from
111 unsupported records to zero. Proven function-local integral constants now
feed inferred and explicit member-template arguments with lexical shadowing and
concrete `sizeof` handling, so `BK_padded` and `BN_padded` no longer create
symbolic helper specializations. Free helper deduction now retains unnamed
parameters, recognizes empty braced type values, and applies the same proven
lexical constants, which materializes `tile_matmad_nax` with concrete tile types
and transpose values. A verified `dispatch_bool` helper whose remaining reachable
calls are lambdas is handed to the existing callback lowering; named functors and
altered helper contracts retain ordinary materialization. Verified
`const_for_loop` callbacks now lower bare callback returns to per-iteration
escapes and fold bare integral-constant parameters only when the source defines
the verified implicit value conversion. Materialization completes with 722
specializations and no unsupported records.

Concrete struct-owned `using` and `typedef` aliases are now resolved inside
C-style and named cast targets after owner materialization. The rewrite respects
qualified owners, lexical shadowing, concrete float and integer specializations,
and aliases whose targets already contain pointer qualifiers. Metal cast nodes
retain source qualifiers while exposing a canonical target type to the strict
CrossGL function-body parser. Reduced DirectX, OpenGL, and Vulkan project
fixtures pass their native validators.

Concrete struct-owned alias templates now resolve their declaring owner,
default arguments, dependent owner constants, and alias chains before member
template deduction. Namespace-qualified and nested same-named owners remain
distinct, and generic vector locals retain their concrete type instead of
borrowing a later same-named declaration. Reduced four-component DirectX,
OpenGL, and Vulkan fixtures pass native validation. This is a partial
implementation of CrossGL/crosstl#1490; dependent function-local aliases and
value expressions outside this contract remain tracked there.

The isolated high-budget `quantized_nax.metal` run still completes 722
specializations with no unsupported records and resolves the NAX fragment
aliases to concrete eight-lane float, half, and bfloat vectors. Metal reverse
translation now represents those local values as fixed aggregate wrappers with
explicit lane storage and element-wise helpers. Reduced DirectX, OpenGL, and
Vulkan fixtures preserve lane reads, writes, arithmetic, and mutable helper
parameters; their generated artifacts pass the available native validators.
The lowering rejects unsupported operators, member selections, mixed vector
shapes, and ABI-visible device or constant storage instead of changing the
source contract. Direct generic-vector canonicalization outside the Metal
frontend remains tracked in CrossGL/crosstl#1569.

Generic member calls now retain their receiver, method, and ordered type and
value arguments in the shared AST, including pointer-member calls and nested
generic types. Metal materialization resolves concrete template methods on
direct and nested struct-field receivers before target generation. Reduced
fixtures containing the five-argument `Atile.load` and `Btile.load` forms from
`fp_quantized.metal` pass the available DirectX, OpenGL, and Vulkan validators.
Calls that reach a target without a concrete specialization fail with a
structured diagnostic instead of losing the generic suffix or computation.

At pinned MLX commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e`, exact high-budget project
replays of the complete `fp_quantized.metal` source now advance past
`epilogue_op.apply` for both DirectX and OpenGL. The receiver declaration is
`thread const TransformNone_float_float& epilogue_op`. This frontier combines
helper array-decay deduction, specialized struct constexpr assertion evaluation,
lexical receiver alias resolution, statement-bounded member-template parsing,
concrete constructor preservation, line-wrapped qualified struct receiver
materialization, and contextual Metal method receiver resolution. CrossTL commit
`c7a3c61ad` resolves the contextual receiver on this path. Specialized struct
constexpr assertion evaluation resolves CrossGL/crosstl#1807.

Dependent helper deduction now resolves the function-local `BK_padded` and
`BN_padded` expressions together with the file-scope `SIMD_SIZE` constant before
specializing plain helper templates. Proven non-type arguments are serialized to
canonical values, so equivalent Boolean and integer spellings identify the same
concrete struct at the kernel call site and in the generated helper signature.
The exact DirectX and OpenGL runs each materialize 604 function
specializations with no unsupported template records. This advances the current
frontier through the applicable CrossGL/crosstl#1479 and CrossGL/crosstl#1490
contracts; both issues retain broader project-materialization scope.

Source-scoped project configuration now supplies concrete `true` values for
`align_M` (ID 200), `align_N` (ID 201), and `align_K` (ID 202) only to
`fp_quantized.metal`. Both target records preserve `project-source-pattern`
provenance. This advances the source-scoped configuration contract in
CrossGL/crosstl#1809 and the concrete function-constant contract in
CrossGL/crosstl#1538 without applying these identifiers to unrelated sources.

Both targets also advance through equivalent duplicate definitions of
`BaseMMAFrag_float_8_8::kFragRows` and through construction of
`QuantizedBlockLoader_float_32_32_36_1_64_16_4`. These paths exercise the
qualified static-constant contract in CrossGL/crosstl#1491 and the constructor
address-space provenance contract in CrossGL/crosstl#1810. Equivalent duplicate
owners now resolve `BaseMMAFrag_float_8_8::frag_type` to its concrete
two-component float vector, including component access at `k`; this resolves
CrossGL/crosstl#1811 for the pinned frontier. Constructor factories preserve the
`BlockLoader_float_16_32_36_1_64::src_ld` const-value initialization and lower
the partially initialized `MMATile_float_2_1_BaseMMAFrag_float_8_8::val_frags`
array through ordered element writes. These results advance the broader
constructor contracts in CrossGL/crosstl#1812 and CrossGL/crosstl#1813.

The complete materialized CrossGL intermediate reaches target generation. Strict
function-body parsing accepts the generic pointer reinterpretation in
`fp_qmv_wide_impl_bfloat16_t_16_4_2_16`,
`(vec<bfloat16_t, 4>*)(xv[v] + k0)`, including the generic pointee type. This
resolves CrossGL/crosstl#1814 and removes
`project.translate.crossgl-function-body-parse-failed` from both exact project
runs.

Source whole-fragment reads and writes through `thread_elements()` references
are now canonicalized to ordered `cooperative_matrix_element` operations. The
resulting cooperative-matrix contract records the `metal_thread_elements`
layout, a 32-lane subgroup, two elements per lane, and
`metal_thread_elements_reference_view` provenance. These fields survive into
both target diagnostics instead of being inferred again after source lowering.
Reduced read and write helpers compile with the native Xcode Metal compiler.
This resolves CrossGL/crosstl#1815 and CrossGL/crosstl#1816 for the pinned
frontier.

The checked-in
[`contracts/cooperative-matrix-fragment-mapping.json`](contracts/cooperative-matrix-fragment-mapping.json)
contract records the concrete `tile_4x4_row_pair` mapping used by this pinned
MLX source. In `mlx/backend/metal/kernels/steel/gemm/mma.h`,
`BaseMMAFrag<T, 8, 8>::get_coord` defines `qid = lane / 4`,
`fm = (qid & 4) + ((lane / 2) % 4)`, and
`fn = (qid & 2) * 2 + (lane % 2) * 2`. The two lane elements therefore map to
`(fm, fn)` and `(fm, fn + 1)`. The contract contains the resulting coordinates
for all 32 lanes and records `mlx_steel_BaseMMAFrag_get_coord` provenance. This
is source-specific evidence for the pinned MLX specialization; it is not a
universal layout claim for Metal cooperative matrices.

The materialized CrossGL intermediate contains 16 source
`CooperativeMatrixType` contract nodes. Before the current contract-flow change,
two nodes carried the complete 12-field contract. In the verified replay, all 16
carry the `metal_thread_elements` layout, subgroup size 32, two elements per
lane, `metal_thread_elements_reference_view` provenance, the
`tile_4x4_row_pair` mapping, and `mlx_steel_BaseMMAFrag_get_coord` mapping
provenance.

Parsing creates eight `CooperativeMatrixOpNode` operations: seven `element`
operations and one `multiply_accumulate` operation. Each element operation now
has scalar `expression_type` `float` and intentionally has no matrix
`result_type`. The multiply-accumulate operation has complete cooperative-matrix
`result_type` and `expression_type` contracts that preserve the accumulator and
destination representation. Shared expression result inference therefore
resolves CrossGL/crosstl#1610 for this contract without claiming that scalar
element expressions require matrix result types.

DirectX and OpenGL now provide an explicit opt-in lane-local cooperative-matrix
software-lowering foundation for the exact registered 8-by-8, 32-lane,
two-elements-per-lane mapping. Reduced target tests compile and validate type
representation, element access, negation, and element-wise addition,
subtraction, and multiplication. Cooperative-matrix load, store, multiply, and
multiply-accumulate operations remain fail closed. The default behavior also
remains fail closed, and the option is not wired through project profiles or
configuration. Full software fallback, target policy, runtime execution, and
numerical parity remain unimplemented. CrossGL/crosstl#1602 and
CrossGL/crosstl#1820 remain open for that work.

Exact high-budget `fp_quantized.metal` replays enable lane-local
cooperative-matrix lowering explicitly through both code-generation factory
paths, `crosstl.project.pipeline.get_codegen` and
`crosstl._crosstl.get_codegen`; this option is not yet available through project
configuration.

Three measured intermediate replays document translation progression and are
not current-boundary claims. In the first replay, DirectX ran for 292.953
seconds and OpenGL for 286.175 seconds before both reported
`project.translate.metal-local-type-unresolved` for local type `vec_w`, whose
extent remained `tn * bytes_per_pack`. In the second replay, DirectX ran for
286.540 seconds and OpenGL for 283.718 seconds. Both reached extent
`(2) * bytes_per_pack`, proving that `tn` had resolved to `2`. In the third
replay, DirectX ran for 418.540 seconds and OpenGL for 364.117 seconds; each
materialized 606 function specializations before branch-insensitive
private-pointer analysis reported a false `view-out-of-bounds` result for
`qouter_float_2_8_4.w`.

The implementation sequence establishes reusable contracts for function-local
struct hoisting, concrete `constexpr` local extents, and defaulted zero-argument
helper materialization. These contracts are backend-independent materialization
behavior rather than MLX-specific rewrites. The completed replay demonstrates
progression through all three contracts for this source specialization.
CrossGL/crosstl#1567 remains open globally because this source-specific evidence
does not establish its complete function-local typedef scope.

DirectX and OpenGL branch pruning, together with the project translation
regression, resolve CrossGL/crosstl#1829. The completed exact replay proves
progression past the earlier false `qouter_float_2_8_4.w` range result.

In the completed replay, DirectX ran for 429.507 seconds, materialized 606
function specializations with no unsupported specializations, and reported
`project.translate.directx-workgroup-pointer-unsupported`. The missing
capability is `directx.workgroup-pointer-lowering`; function
`BlockMMA_float_float_16_32_32_1_2_false_true_36_36__mma`, parameter `As`, stops
with reason `dynamic-control-flow-reassignment` and message `DirectX cannot
preserve workgroup pointer reassignment for 'As' across nested control flow`.
CrossGL/crosstl#1518 covers the required HLSL resource and `groupshared` alias
representation, reassignment and nested-alias semantics, and structured
rejection.

OpenGL ran for 417.522 seconds, materialized the same 606 function
specializations with no unsupported specializations, and reported
`project.translate.opengl-workgroup-pointer-unsupported`. The missing capability
is `opengl.workgroup-pointer-lowering`; parameter `dst_` stops with reason
`bare-pointer-expression` and message `OpenGL cannot emit a workgroup pointer as
a first-class value: dst_`. This target boundary spans two existing contracts:
CrossGL/crosstl#1544 covers pointer-bearing aggregate members and constructors,
including `QuantizedBlockLoader`, while CrossGL/crosstl#1671 covers concrete
workgroup backing provenance through helper parameters. It is not classified as
a shared DirectX/OpenGL boundary.

Each target report contains one failed artifact/provenance record, zero
translated artifacts, and one error; no target artifact was emitted. Native
validation was not attempted because there is no artifact. MLX host runtime
integration and execution were not attempted, and numerical parity was not
evaluated. CrossGL/crosstl#1546 remains open for the broader byte-address
provenance contract across pointer reinterpretation, but it is not the current
exact boundary. Earlier replays also established progression past
`qdot_float_16_4.x_thread` and its `unprovable-view-offset` boundary under
CrossGL/crosstl#1826, as well as the transitive local `constexpr`
materialization contract tracked by CrossGL/crosstl#1824.

A shared reduced CrossGL fixture exercises the resolved partition contract with
the required readback `[100, 101, 102, 103, 200, 201, 202, 203]`.
[GitHub Actions run 29641172600](https://github.com/CrossGL/crosstl/actions/runs/29641172600)
produced this exact readback on `windows-latest` through Direct3D and on
`ubuntu-latest` through OpenGL. The passing steps were
`Prove Direct3D private-pointer partition writeback` and
`Prove OpenGL private-pointer partition writeback`, respectively.

An independent Metal fixture, `private_pointer_word_view.metal`, initializes a
local struct with two 32-bit words, reads its eight bytes through a const
thread-local byte view, and computes the order-sensitive checksum
`sum(byte[index] * (index + 1))`. Its required readback is `[204]`. The Metal
source compiles locally as Metal 3.2 with Apple metal `32023.918`.
[GitHub Actions run 29649620337](https://github.com/CrossGL/crosstl/actions/runs/29649620337)
produced the exact `[204]` readback on `windows-latest` through Direct3D and on
`ubuntu-latest` through OpenGL. The passing steps were
`Prove Direct3D local-struct byte-view native readback` and
`Prove OpenGL local-struct byte-view native readback`, respectively. Both tests
required their native runtime and reported zero mismatches with zero absolute
and relative tolerance.

These reduced fixture-level proofs do not establish complete MLX artifact
translation, full MLX host runtime integration, full MLX test-suite execution,
or numerical parity for MLX workloads.

The previously recorded pinned Vulkan replays confirmed that both affected
kernels advanced past this contract without producing a full artifact.
`fp_quantized.metal` then stopped at
type inference for the reference-returning `frag_at(i, j)` argument tracked in
CrossGL/crosstl#1557. `quantized_nax.metal` next stops because the dependent
static owner of `mma` is absent, so its empty tag argument has no selected
parameter type. Dependent static-owner materialization remains tracked in
CrossGL/crosstl#1574. These results establish translation-frontier progress
only; they do not include runtime integration or numerical parity.

The previously recorded full pinned Vulkan run advanced beyond the
generic-vector-width diagnostic. The contextual initializer contract
implemented for CrossGL/crosstl#1573 now rejects the empty
`metal::bool_constant<...>{}` argument instead of inferring a zero-length array.
The selected parameter type is still
unavailable because the captured intermediate drops the dependent static owner
from `CTile::NAXFrag_t::mma`; CrossGL/crosstl#1574 tracks that remaining
materialization contract. The intermediate also retains unresolved
reference-returning `frag_at` calls, whose receiver identity remains tracked in
CrossGL/crosstl#1557. No full-kernel artifact or validator result is claimed.
Complete address-space, const, pointer-provenance, and
unresolved-alias diagnostic transport remains tracked in CrossGL/crosstl#1566.
Pointer-bearing aggregate propagation remains tracked in CrossGL/crosstl#1544,
and lowered receiver/reference semantics must satisfy CrossGL/crosstl#1557
before the kernel can be considered semantically ready.
Lazy logical and conditional evaluation in SPIR-V remains tracked in
CrossGL/crosstl#1560 for full-corpus semantic coverage.
Nested-return lowering in pointer-preserving SPIR-V inlining is covered by the
passing full GEMV Vulkan gate. Side-effectful compatibility arguments remain
rejected explicitly and tracked in CrossGL/crosstl#1562.

## Resolved Frontier Issues

The current reduced frontier no longer depends on the previously tracked issues:
CrossGL/crosstl#1672, CrossGL/crosstl#1659, CrossGL/crosstl#1516,
CrossGL/crosstl#1476, CrossGL/crosstl#1472, CrossGL/crosstl#1312,
CrossGL/crosstl#1661, CrossGL/crosstl#1573, CrossGL/crosstl#1555,
CrossGL/crosstl#1561,
CrossGL/crosstl#1551,
CrossGL/crosstl#1498,
CrossGL/crosstl#1394,
CrossGL/crosstl#1317,
CrossGL/crosstl#939, CrossGL/crosstl#940,
CrossGL/crosstl#941, CrossGL/crosstl#943, CrossGL/crosstl#944,
CrossGL/crosstl#945, and CrossGL/crosstl#946. CrossGL/crosstl#979,
CrossGL/crosstl#980,
CrossGL/crosstl#981, CrossGL/crosstl#982, CrossGL/crosstl#983,
CrossGL/crosstl#984, CrossGL/crosstl#985, CrossGL/crosstl#1001,
CrossGL/crosstl#1002, CrossGL/crosstl#1003, CrossGL/crosstl#1004,
CrossGL/crosstl#1006, CrossGL/crosstl#1007, CrossGL/crosstl#1012, and
CrossGL/crosstl#1013 are also covered by mainline fixes or superseded by the
current follow-up issue set. CrossGL/crosstl#1019, CrossGL/crosstl#1026,
CrossGL/crosstl#1028, CrossGL/crosstl#1029, CrossGL/crosstl#1030,
CrossGL/crosstl#1031, CrossGL/crosstl#1033, CrossGL/crosstl#1034,
CrossGL/crosstl#1035, and CrossGL/crosstl#1036 are closed by mainline fixes or
superseded by the current issue set and are no longer listed as active MLX
blockers. CrossGL/crosstl#1032, CrossGL/crosstl#1037, CrossGL/crosstl#1038,
CrossGL/crosstl#1039, CrossGL/crosstl#1068, CrossGL/crosstl#1104, and
CrossGL/crosstl#1105 are also closed or superseded by the current scout and
issue set. CrossGL/crosstl#1027 is no longer reported by the latest full-corpus
scout because the generated Metal quantization declarator now parses far enough
to reach target codegen. The current full-corpus scout no longer reports
runtime-adapter contracts, boolean SPIR-V interface lowering, or the previous
closed issue set as active missing capabilities. CrossGL/crosstl#1106,
CrossGL/crosstl#1107, CrossGL/crosstl#1110, CrossGL/crosstl#1111,
CrossGL/crosstl#1122, CrossGL/crosstl#1124, CrossGL/crosstl#1126, and
CrossGL/crosstl#1127 are also closed and are no longer tracked as active MLX
blockers. CrossGL/crosstl#852 is covered by the current OpenGL arange smoke
check. CrossGL/crosstl#1146 is resolved by bounded template replacement scans,
and CrossGL/crosstl#1184 is resolved by the latest mainline materialization
work. CrossGL/crosstl#1155 and CrossGL/crosstl#1160 are covered by the current
frontier after the SPIR-V project-artifact and multi-entry binding fixes.
CrossGL/crosstl#1203, CrossGL/crosstl#1204, and CrossGL/crosstl#1206 were
closed by the latest mainline helper-template, softmax parser, and SPIR-V
pointer-overload fixes. CrossGL/crosstl#1205, CrossGL/crosstl#1207,
CrossGL/crosstl#1218, and CrossGL/crosstl#1222 are also closed by the current
mainline OpenGL template, SIMD helper, steel attention diagnostic, and steel GEMM
materialization fixes. CrossGL/crosstl#1238, CrossGL/crosstl#1239, and
CrossGL/crosstl#1240 are closed by the assembled SPIR-V validation, complex
helper call, and fence initializer fixes. CrossGL/crosstl#1246,
CrossGL/crosstl#1248, CrossGL/crosstl#1249, CrossGL/crosstl#1250,
CrossGL/crosstl#1259, CrossGL/crosstl#1260, and CrossGL/crosstl#1261 are closed
by the current mainline access-chain index, materialization scalability,
templated functor, and Vulkan validation fixes. CrossGL/crosstl#1274 and
CrossGL/crosstl#1287 are closed by the current Vulkan complex helper validation
and full-corpus Metal template materialization fixes. CrossGL/crosstl#1329,
CrossGL/crosstl#1338, CrossGL/crosstl#1340, and CrossGL/crosstl#1346 are closed
by the current project-scale template and SPIR-V validation fixes.
CrossGL/crosstl#1355 is closed by the current OpenGL MLX template binding fix.
CrossGL/crosstl#1354 and CrossGL/crosstl#1362 are closed by the current
full-corpus materialization and Vulkan validation work. CrossGL/crosstl#1452,
CrossGL/crosstl#1453, and CrossGL/crosstl#1454 are covered by bounded template
materialization with source-located diagnostics for unsupported MLX reduction,
scan, and Steel specializations. CrossGL/crosstl#1392 is closed by fixture
resource binding through reflected backend aliases. CrossGL/crosstl#1500 is
covered by mapped-signature collision detection with overload-aware GLSL call
rewriting. CrossGL/crosstl#1502 is covered by contextual GLSL aggregate
construction for struct, fixed-array, vector, and matrix values.
CrossGL/crosstl#1503 is covered by explicit expected-type scalar coercion for
numeric-to-Boolean returns and signed mixed-width `arange` arithmetic.
CrossGL/crosstl#1661 is covered for pinned `binary_two.metal` by fixed-array
resource helper specialization in CrossTL commit `db593d19b` and the required
OpenGL/SPIR-V 1.3 compilation and validation gate.
CrossGL/crosstl#1807 is resolved for the pinned `fp_quantized.metal` frontier by
specialized struct constexpr assertion evaluation; contextual receiver
materialization remains tracked in CrossGL/crosstl#1479.
CrossGL/crosstl#1811 is resolved for the same frontier by equivalent duplicate
struct-alias resolution with concrete component typing and fail-closed conflict
diagnostics.

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
outputs and stop-gradient. Integer and array indexing still require the
unimplemented Gather primitive. Contiguous conversion, reshape, flatten and unflatten
dispatch translated copies when sharing storage is insufficient. Copies support
matching float32, int32 and uint32 arrays, including negative and zero strides.
`Full`, including `zeros`, `ones` and `full_like`, uses those copies to materialize
broadcast values in the same three types. Boolean and other storage widths are
not yet supported by this copy path. General boolean resource reflection and
native buffer transport are tracked in
[#1999](https://github.com/CrossGL/crosstl/issues/1999).
Binary addition, subtraction, multiplication, minimum and maximum support those
three types; division supports float32. Broadcasts and non-row-contiguous binary
inputs are materialized by translated copies before the unchanged vector-vector
binary entry executes. Casts between float32, int32 and uint32 use six unchanged
`v_copy` entries, including automatic promotion before mixed-type arithmetic.
Strided cast inputs are materialized through translated copies. Casts involving
other types still fail explicitly.
Other primitives retain MLX's explicit unsupported-GPU errors. Unary inputs
must be float32. Stored-contiguous broadcasts and column-major views retain
their metadata; noncontiguous inputs use translated copies before unary dispatch.
Dispatch is synchronous, uses host staging buffers and supports at most 65,535
stored elements with one thread per workgroup. Empty arrays do not dispatch. This is a
host integration proof, not a complete MLX backend or a performance benchmark.

The pin is `9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8`. CI builds and runs the same
proof on Linux/OpenGL, Windows/Direct3D 12 and macOS/generated Metal. Mesa software
rendering and pinned WARP make Windows/Linux execution reproducible without
dedicated GPU runners. The macOS path uses the same callback adapter with generated
Metal packages, not MLX's original Metal backend. DirectX 10/11, Vulkan,
asynchronous queues, persistent GPU allocations, automatic operation
selection, and the complete MLX suite are not covered here.

## Upstream Adaptations

`prepare.py` requires a clean, exact-pinned checkout. It changes only the selected
no-GPU backend build definition and adds four explicitly named backend files:

- `crosstl_backend.cpp` registers a versioned synchronous dispatch callback,
  implements device/stream hooks, `Arange::eval_gpu` and 27 unary primitive hooks,
  and stages output storage. MLX's Log and Sqrt primitives select their log-base
  and reciprocal variants, yielding 30 unary kernel entries. View hooks reuse
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
  AsType selects a source/destination-specific copy entry, validates array shapes
  and allocation bounds, and allocates a dense destination. Casts perform no CPU
  elementwise conversion.
  Full receives the upstream broadcast/cast value and materializes it with the
  same general-copy entry. Scalar fills, row/column broadcasts and strided values
  use the source strides, with fresh output allocation and no CPU fill loop.
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

Native traces must start with the exact 493 nonempty workload dispatches, cover
all 58 entries, and retain artifact identities and runtime/device details.
Separate negative processes reject an unsupported primitive, oversized arange,
a missing artifact, unary inputs with unsupported dtype or size, contiguous and
strided unary inputs that exceed their allocations, and
copies with unsupported dtype, excessive size or an invalid source allocation.
Binary inputs with unsupported dtype or excessive size are also rejected, as
are casts with unsupported types, excessive size or an invalid source span.
Full also rejects unsupported boolean storage, excessive output size and a source
view extending before its allocation.
Each of the twenty processes has a hard 180-second process-tree deadline; all
are attempted so a failure does not discard the other diagnostic results.

The evidence directory retains before/after adaptation hashes, command status,
stdout/stderr, upstream test logs, dispatch traces and numerical results. The
schema-version-2 summary records both upstream test-file hashes and is written
only after source and result checks pass,
including explicit rejection messages from all eighteen negative cases. Source
identity checks do not attest to a separately supplied binary; CI builds MLX from
the verified sources and retains its build log. `fullUpstreamSuite` and
`fullTranslatedBackend` remain `false`: extending primitive coverage and then
running the entire suite is the next stage, not an implicit property of this proof.

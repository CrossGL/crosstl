# Portable MLX Host Execution

This harness builds pinned MLX with Metal and CUDA disabled and connects its C++
GPU evaluation path to translated native runtime packages. Python calls such as
`mlx.core.arange` reach an MLX primitive, dispatch the translated upstream kernel,
and receive the device result in the original MLX array. Python operations and
upstream tests are not replaced.

## Scope

The current adapter implements `Arange` for `float32`, `int32`, `uint32`, `int64`
and `uint64`, plus 30 float32 unary operations from the pinned `unary.metal`:
absolute value, negation, sign, square, square root, reciprocal square root,
floor, ceil, round, exponential, expm1, logarithms, log1p, sigmoid, erf, inverse
erf, trigonometric functions and their hyperbolic and inverse forms. The exact
entry list is in `packages.py`. Other primitives retain MLX's explicit
unsupported-GPU errors. Unary inputs must be contiguous and float32.
Dispatch is synchronous, uses host staging buffers and supports at most 65,535
stored elements with one thread per workgroup. Empty arrays do not dispatch. This is a
host integration proof, not a complete MLX backend or a performance benchmark.

The pin is `9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8`. CI builds and runs the same
proof on Linux/OpenGL and Windows/Direct3D 12. Mesa software rendering and pinned
WARP make execution reproducible without dedicated GPU runners. DirectX 10/11,
Vulkan, asynchronous queues, persistent GPU allocations, automatic operation
selection, and the complete MLX suite are not covered here.

## Upstream Adaptations

`prepare.py` requires a clean, exact-pinned checkout. It changes only the selected
no-GPU backend build definition and adds four explicitly named backend files:

- `crosstl_backend.cpp` registers a versioned synchronous dispatch callback,
  implements device/stream hooks, `Arange::eval_gpu` and 27 unary primitive hooks,
  and stages output storage. MLX's Log and Sqrt primitives select their log-base
  and reciprocal variants, yielding 30 unary kernel entries.
- `crosstl_dispatch.h` defines the typed C buffer/callback ABI.
- `crosstl_primitives.cpp` copies upstream unsupported primitive definitions,
  removing only the implemented primitive stubs.
- `crosstl_event.cpp` copies upstream events and adds synchronous GPU wait/signal
  handling. CPU scheduling behavior remains unchanged.

No upstream Metal kernel or Python test is edited. Preparation records hashes of
every adapted file. Integer argument conversion explicitly implements truncation
and modular wrapping; it does not depend on undefined out-of-range C++ casts.
The host adapter is repository-specific; package generation, reflection, artifact
verification and native execution use the public CrossTL project interfaces.
Unary output allocation uses MLX's existing `set_unary_output_data` helper,
including its contiguous strides and buffer-donation behavior. It performs no
CPU calculation. Host inputs are copied before output readback, including when
MLX donates the input allocation. Translation explicitly selects the
`rne-flush` binary32 FMA profile required by the pinned Erf and Expm1 bodies.

`runtime.py` retains the callback for the process lifetime and permits one
registration. It verifies argument names, direction, counts and physical layouts
against the selected descriptor. Native failures propagate as MLX errors; there
is no CPU computation fallback. The element cap is deliberate pending general
[native dispatch-limit validation](https://github.com/CrossGL/crosstl/issues/1959).

## Run

Use Python 3.12, a C++20 toolchain, CMake, Ninja and the relevant runtime.
Linux additionally needs OpenBLAS, LAPACK/LAPACKE and Mesa EGL development/runtime
packages. Windows needs Visual Studio C++ build tools and DXC; the workflow pins
DXC and WARP archives with SHA-256 checks. Run from the CrossTL repository root:

Python 3.13 also works for the Linux build, but the Windows graphics dependency
currently requires the published Python 3.12 wheel. CI uses 3.12 on both platforms
and retains dependency-install output even when setup fails.

```sh
python -m pip install -e . moderngl PyOpenGL setuptools wheel cmake ninja nanobind numpy packaging
git clone https://github.com/ml-explore/mlx.git mlx-upstream
git -C mlx-upstream checkout 9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8
python -m demos.integrations.mlx.portable_host.prepare --mlx-root mlx-upstream --output adaptation.json
CMAKE_ARGS='-DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_CROSTL_HOST=ON -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON' python -m pip install -e ./mlx-upstream --no-build-isolation
python -m demos.integrations.mlx.portable_host.packages --mlx-root mlx-upstream --target opengl --output-dir host-packages
LIBGL_ALWAYS_SOFTWARE=1 python -m demos.integrations.mlx.portable_host.verify --mlx-root mlx-upstream --packages host-packages --output-dir host-evidence
```

For Windows use `directx-runtime` and `--target directx`, with the same build
options set in `CMAKE_ARGS`. The complete platform setup is in
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

The CPU reference uses the unchanged CPU backend in the same adapted MLX build;
it is not a separately rebuilt pristine binary or a Metal comparison.

It also checks 20 array-creation cases spanning all five types and lengths 0, 1,
7 and 257. The 129 unary records cover the same lengths for all 30 operations,
six special-value cases, two large-angle cases and an arange/abs/negative/square
chain, totaling 8,009 unary output values. Independent Python math references
check both CPU and native results, not just agreement between them. Readbacks
must have complete values and unchanged case identities; finite comparisons
use `rtol=2e-5, atol=1e-6`, with exact zero values/signs and nonfinite
classification. Upstream tests and their tolerances are unchanged.

The pinned CPU Erf approximation returns negative zero for both zero input
signs. Its reference records this behavior explicitly; generated GPU Erf must
preserve the input sign as the source Metal implementation does. No readback is
corrected to make the two paths agree.

Native traces must start with the exact 117 nonempty workload dispatches, cover
all 35 entries, and retain artifact identities and runtime/device details.
Separate negative processes reject an unsupported primitive, oversized arange,
a missing artifact, and unary inputs with unsupported dtype, layout or size.
Each of the eight processes has a hard 180-second process-tree deadline; all
are attempted so a failure does not discard the other diagnostic results.

The evidence directory retains before/after adaptation hashes, command status,
stdout/stderr, upstream test logs, dispatch traces and numerical results. The
schema-version-2 summary is written only after source and result checks pass,
including explicit rejection messages from all six negative cases. Source
identity checks do not attest to a separately supplied binary; CI builds MLX from
the verified sources and retains its build log. `fullUpstreamSuite` and
`fullTranslatedBackend` remain `false`: extending primitive coverage and then
running the entire suite is the next stage, not an implicit property of this proof.

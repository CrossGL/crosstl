# Portable MLX Host Execution

This harness builds pinned MLX with Metal and CUDA disabled and connects its C++
GPU evaluation path to translated native runtime packages. Python calls such as
`mlx.core.arange` reach an MLX primitive, dispatch the translated upstream kernel,
and receive the device result in the original MLX array. Python operations and
upstream tests are not replaced.

## Scope

The current adapter implements `Arange` for `float32`, `int32`, `uint32`, `int64`
and `uint64`. Other primitives retain MLX's explicit unsupported-GPU errors.
Dispatch is synchronous, uses host staging buffers and supports at most 65,535
elements with one thread per workgroup. Empty arrays do not dispatch. This is a
host integration proof, not a complete MLX backend or a performance benchmark.

The pin is `d9add9d11f3154111a4c85f267ec2fd307ecd18e`. CI builds and runs the same
proof on Linux/OpenGL and Windows/Direct3D 12. Mesa software rendering and pinned
WARP make execution reproducible without dedicated GPU runners. DirectX 10/11,
Vulkan, asynchronous queues, persistent GPU allocations, automatic operation
selection, and the complete MLX suite are not covered here.

## Upstream Adaptations

`prepare.py` requires a clean, exact-pinned checkout. It changes only the selected
no-GPU backend build definition and adds four explicitly named backend files:

- `crosstl_backend.cpp` registers a versioned synchronous dispatch callback,
  implements device/stream hooks and `Arange::eval_gpu`, and stages output storage.
- `crosstl_dispatch.h` defines the typed C buffer/callback ABI.
- `crosstl_primitives.cpp` copies upstream unsupported primitive definitions,
  removing only the `Arange` stub.
- `crosstl_event.cpp` copies upstream events and adds synchronous GPU wait/signal
  handling. CPU scheduling behavior remains unchanged.

No upstream Metal kernel or Python test is edited. Preparation records hashes of
every adapted file. Integer argument conversion explicitly implements truncation
and modular wrapping; it does not depend on undefined out-of-range C++ casts.
The host adapter is repository-specific; package generation, reflection, artifact
verification and native execution use the public CrossTL project interfaces.

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
git -C mlx-upstream checkout d9add9d11f3154111a4c85f267ec2fd307ecd18e
python -m demos.integrations.mlx.portable_host.prepare --mlx-root mlx-upstream --output adaptation.json
CMAKE_ARGS='-DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_CROSTL_HOST=ON -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON' python -m pip install -e ./mlx-upstream --no-build-isolation
python -m demos.integrations.mlx.portable_host.packages --mlx-root mlx-upstream --target opengl --output-dir host-packages
LIBGL_ALWAYS_SOFTWARE=1 python -m demos.integrations.mlx.portable_host.verify --mlx-root mlx-upstream --packages host-packages --output-dir host-evidence
```

For Windows use `directx-runtime` and `--target directx`, with the same build
options set in `CMAKE_ARGS`. The complete platform setup is in
[`mlx-portable-host.yml`](../../../../.github/workflows/mlx-portable-host.yml).
Output directories must be new so evidence from different runs cannot mix.

## Required Evidence

The verifier checks the pin and unchanged upstream test source, then runs isolated
CPU-reference and translated-GPU processes. Both must pass these upstream tests
without skips:

- `test_arange_overload_dispatch`
- `test_arange_inferred_dtype`
- `test_arange_corner_cases_cast`

The CPU reference uses the unchanged CPU backend in the same adapted MLX build;
it is not a separately rebuilt pristine binary or a Metal comparison.

It also checks 20 array cases spanning all five types and lengths 0, 1, 7 and 257
against NumPy and the explicit MLX CPU baseline. Native traces must include every
selected entry, actual artifact identities and runtime/device details. Separate
negative processes must reject an unsupported primitive, an oversized dispatch
and a missing artifact. Each process has a hard process-tree deadline.

The evidence directory retains command status, stdout/stderr, upstream test logs,
dispatch traces and numerical results. `fullUpstreamSuite` and
`fullTranslatedBackend` remain `false`: extending primitive coverage and then
running the entire suite is the next stage, not an implicit property of this proof.

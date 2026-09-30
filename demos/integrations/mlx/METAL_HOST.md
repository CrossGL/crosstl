# MLX Metal Host Integration

This harness builds MLX at `9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8` and
redirects selected kernel lookups to CrossTL-generated Metal libraries. MLX
continues to own its arrays, allocation views, command encoders, synchronization
and dispatch geometry. No Python implementation substitutes for GPU computation.

## Verified Scope

The required macOS workflow translates and compiles all 15 `Powercomplex64`
entry shapes, then links their AIR modules into one Metal library. It runs the
unchanged upstream `python/tests/test_ops.py` module
twice: once using upstream kernels and once with these entries redirected. The
test and skip counts must match, native dispatch records must identify translated
entries, and a missing required library must fail rather than use an original
kernel. Source, compiled-library, runtime and numerical-test evidence is retained.

Additional MLX array workloads require actual dispatch of eight shapes: `ss`,
`sv`, `vs`, `vv`, `g1`, `g2`, `g3` and `gn2`. They cover broadcasting,
non-contiguous views, zero strides and multidimensional host launch geometry,
with independent NumPy references. Each layout runs ordinary complex inputs and
both signed-zero sides of the negative-real branch cut, totaling 402 complex
outputs. Each dataset has its own process and dispatch trace, requiring exactly
one translated dispatch per layout. The evidence retains every input, reference
and readback; the verifier independently recomputes complex powers and checks the
unchanged `5e-5 * max(1, abs(reference))` bound. The large-index variants remain
compile-only in this host integration check.

The original MLX JIT path is not a signed-zero reference. On the local Apple M2
Max build, `(-4-0i) ** (0.5+0i)` returns `+2i` with overrides disabled, whereas the
precisely compiled translated entry and independent references return `-2i`.
This baseline difference is recorded, not treated as translator parity or hidden
by a wider tolerance. Both unchanged upstream `test_ops` runs must still pass;
the additional branch-cut workloads require the translated path to match the
independent reference, not the original JIT result.

Only entries actually present in the dispatch trace have host-execution coverage.
Building all 15 libraries does not prove all 15 were selected by upstream tests.
Other operations still use the upstream backend. This is partial Metal host
redirection, not a fully translated MLX backend or execution of the entire
upstream suite. DirectX and OpenGL host integration remain separate work.

## Full-Suite Baseline

A separate local discovery run at this revision executed 899 upstream Python
tests on an Apple M2 Max, with 53 skips, four errors and one failure. Repeating
the affected tests with both override variables unset reproduced every failure:

- Four cases in `TestFastSDPA.test_sdpa_vector_head_dim_512` request 1,024
  threads from original pipelines whose reported limits are 640 or 832.
- `TestLosses.test_triplet_loss` fails its `losses_p3` numerical assertion.

This comparison uses the adapted build with original libraries and redirection
disabled; it is not evidence from a separately rebuilt, unmodified binary. These
failures are outside the selected complex-power entries, and no upstream test is
edited or skipped to hide them. The required gate covers `test_ops` and the eight
additional layouts above. Full-suite compatibility remains unproven.

## Upstream Adaptation

`metal-library-overrides.patch` modifies only MLX's `device.cpp` and `device.h`.
`metal_library_overrides.h` supplies the small integration adapter. The preparation
step checks the exact upstream revision and both file hashes before applying it.
The verifier checks the adapted hashes and rejects unrelated tracked changes.
No upstream kernel or test source is edited.

- `Device::get_kernel` substitutes a required per-entry library before pipeline
  lookup. Each library is cached for its Metal device.
- `CommandEncoder` records the selected translated entry and the actual dispatch
  dimensions without changing array bindings or launch geometry.
- `CROSTL_METAL_LIBRARY_OVERRIDES` names an absolute directory containing
  `libraries.txt` and `<entry>.metallib` files. The manifest contains one required
  entry per line. Missing listed libraries are errors; unlisted operations remain
  on the upstream backend.
- `CROSTL_METAL_LIBRARY_TRACE` names the required absolute trace path. Trace-write
  failures are errors, not silently missing evidence.

MLX is built with its supported `MLX_METAL_JIT=ON` option. Its original JIT
library can still be constructed before the selected kernel is redirected; that
library is not the pipeline source for a listed entry. The negative missing-library
test verifies this distinction. This adapter is an explicit MLX integration
change, not automatic translation of its C++ runtime.

Both individual and combined libraries are retained. The host uses copies of
the combined library under each required entry name, matching the adapter's
per-entry lookup contract; every copy has the same verified hash. This exercises
cross-artifact linkage rather than relying on independent libraries to hide
duplicate symbols. Execution records retain the explicitly selected device,
override directory and trace path without copying unrelated environment values.
Source-private and generated helper linkage is preserved by
the translator, without rewriting emitted shader text. This covers the selected
15 entries and all 402 retained host readbacks, not every translation unit in MLX.

External free-function declarations whose definitions live in other source files
are not yet preserved by the Metal frontend, tracked in
[#1978](https://github.com/CrossGL/crosstl/issues/1978). A reduced original
provider/consumer pair links and executes correctly, while its translated caller
fails native compilation because the declaration was dropped. The selected
complex-power entries do not depend on this missing capability; their combined
library proof does not establish arbitrary cross-file callable support.

## Run Locally

Use macOS with a Metal device, current Xcode command-line tools and Python 3.13.
Start with a clean, full MLX checkout at the pinned revision and a separate virtual
environment with `setuptools`, `wheel`, `cmake`, `ninja`, `nanobind`, `numpy` and
`packaging` installed. Install CrossTL in the environment running the harness.

```sh
python demos/integrations/mlx/run_mlx_metal_host.py prepare \
  --mlx-root /path/to/mlx --output-dir /path/to/proof/preparation
CMAKE_ARGS='-DMLX_METAL_JIT=ON -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF' \
  CMAKE_BUILD_PARALLEL_LEVEL=4 /path/to/mlx-env/bin/python -m pip install \
  -e /path/to/mlx --no-build-isolation
python demos/integrations/mlx/run_mlx_metal_host.py verify \
  --mlx-root /path/to/mlx --python /path/to/mlx-env/bin/python \
  --output-dir /path/to/proof/execution
```

Use a fresh execution directory for each run. The final `evidence.json` is written
only after both upstream runs, all three host datasets, dispatch verification,
library integrity checks and the missing-library negative test succeed. The
schema-version-2 report includes per-dataset readbacks and dispatch records.
Intermediate logs, partial numerical records and translation reports remain
available when a step fails.

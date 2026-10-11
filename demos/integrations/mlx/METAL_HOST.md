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

## Backward Kernel Checks

Before the MLX host build, a separate required step executes selected unchanged
`seq_gated_delta_vjp` entries as both original and translated Metal. It uses the
upstream `(32, 4, 1)` workgroup shape and checks all six gradient buffers against
an independent NumPy reference. Cases cover all six declared key/value-head
layouts, multiple batches, checkpoint intervals 1, 4, 8 and 16, partial checkpoint
segments, and float32 and float16 inputs. An exact uniform case checks every
output bit; seeded cases retain the upstream sequential gradient tolerances of
`rtol=atol=1e-5`. Forward-loss finite differences check the reference separately.

Every input buffer must remain unchanged, and guard bytes after all allocations
must survive the dispatch. Evidence includes the pinned source identity,
translation report, compiler commands, actual libraries, inputs, expected
gradients and complete binary readbacks. These tests invoke native kernels
directly. They do not redirect gated-delta through MLX's host runtime, execute
the entire upstream suite, cover every specialization or prove portable-backend
gradient execution.

## Full-Suite Baseline

A separately built, unmodified checkout at the current pin executed all 926
discovered upstream Python tests on an Apple M2 Max using Python 3.13 and
`MLX_METAL_JIT=ON`: one failure, four errors and 54 upstream skips. Both override
variables were absent, and tracked source files were unchanged before and after
execution. The errors are the four attention threadgroup-limit cases below;
the failure is `TestLosses.test_triplet_loss`. This independently reproduces
those problems without the host adaptation. It does not qualify any translated
backend or waive the unchanged upstream assertions.

An earlier adapted-build discovery run executed 899 upstream Python tests on an
Apple M2 Max, with 53 skips, four errors and one failure. Repeating
the affected tests with both override variables unset reproduced every failure:

- Four cases in `TestFastSDPA.test_sdpa_vector_head_dim_512` request 1,024
  threads from original pipelines whose reported limits are 640 or 832.
- `TestLosses.test_triplet_loss` fails its `losses_p3` numerical assertion.

This comparison uses the adapted build with original libraries and redirection
disabled; it is not evidence from a separately rebuilt, unmodified binary. These
failures are outside the selected complex-power entries, and no upstream test is
edited or skipped to hide them. The required gate covers `test_ops` and the eight
additional layouts above. Full-suite compatibility remains unproven.

The current pin also has a seed-dependent failure in `TestOps.test_scans` at
`test_ops.py:2707`, comparing bfloat16 cumulative sums with a separately reduced
reference. A local replay of seeds 0 through 63 reproduced the assertion for
seeds 14 and 55 with overrides both disabled and enabled. All 64 paired results
matched, including the saved input, output and reference arrays for both failing
seeds; none of these isolated scan runs dispatched a translated entry. The
comparison uses the adapted build, not a separately rebuilt unmodified MLX.

For a deterministic single-case reproduction, start a fresh process in the
pinned checkout's `python/tests` directory using its Metal-enabled environment:

```python
import random
import unittest

import mlx.core as mx
import numpy as np

random.seed(14)
np.random.seed(14)
mx.random.seed(14)
mx.set_default_device(mx.gpu)
unittest.main(module=None, argv=["unittest", "test_ops.TestOps.test_scans"])
```

Unset both override variables to run the original path. This baseline defect
does not establish a translated scan regression or successful scan coverage.
The required host gate retains the original test and tolerances and still fails
on any unsuccessful run; no retry, skip or known-failure allowance is built into
the verifier. Its ordinary upstream invocations are not seeded, so their random
inputs need not match between processes.

Both upstream invocations use `unittest_evidence.py`, which preserves the full
module, unittest outcomes, skips and existing deadlines. On a failure or error,
it records local scalar values and NumPy/MLX arrays from upstream traceback
frames under `upstream-original-failures/` or `upstream-translated-failures/`.
Each record includes the test identity, source hash, line, array dtype and shape.
Capture is limited to 16,384 elements per array and 65,536 elements per failure;
larger arrays and unavailable values are explicitly marked. Complex values and
non-finite floats have JSON-safe representations, preserving signed zeros.
No seed, test input, assertion, tolerance or expected result is changed. Evidence
capture errors cannot convert an upstream failure into a passing test.

### Captured Scan Replay

The [retained hosted failure](https://github.com/CrossGL/crosstl/actions/runs/38085353689/artifacts/11682384191)
at translator revision `895dde51` was replayed on the separately rebuilt,
unmodified MLX above. All 256 bfloat16 inputs, cumulative-sum outputs and reduced
reference values match the hosted arrays exactly. Element `[5, 23]` is `9.5`
versus `9.3125`; the absolute error is `0.1875`, exceeding the actual float32
comparison bound `0.18725000321865082`. Float16 and float32 controls pass on the
same inputs. The CPU bfloat16 comparison passes but has different outputs, so it
is not a substitute for the Metal reference.

A deterministic reduction of that captured case is:

```python
import mlx.core as mx

mx.set_default_device(mx.gpu)
a = mx.array([[
    0.5625, 0.890625, 0.953125, 0.4921875,
    0.1728515625, 0.267578125, 0.1015625, 0.765625,
    0.310546875, 0.322265625, 0.1708984375, 0.0966796875,
    0.1689453125, 0.22265625, 0.875, 0.59765625,
    0.314453125, 0.029052734375, 0.87890625, 0.09130859375,
    0.09521484375, 0.5234375, 0.2177734375, 0.322265625,
]], dtype=mx.bfloat16)
out = mx.cumsum(a, axis=-1)
expected = (mx.tri(24).astype(mx.bfloat16) * a[:, None, :]).sum(axis=-1)
print(out[0, -1].item(), expected[0, -1].item())
print(mx.allclose(out, expected, rtol=0.02, atol=1e-3).item())
```

On the tested M2 Max this prints `9.5 9.3125` and `False`, without selecting a
random seed. Independent sequential bfloat16 round-to-nearest-even accumulation
reproduces all 256 original reduction reference values. Rounding the exact
prefix sum only once gives `9.4375` at the failing element; that is not a model
of the original subgroup scan's accumulation order. No source kernel, test,
tolerance or required result is changed to accommodate the discrepancy.

At translator revision `4232aa0a`, both complete
`contig_scan_inclusive_sum_float32_float32` and
`contig_scan_inclusive_sum_bfloat16_bfloat16` entries also translate, package,
strictly compile and execute as Metal round trips on the captured 8x32 case.
The 512 computed outputs and 64 output guards agree with separately compiled
original Metal controls and the unmodified MLX host readbacks. These dispatches
do not retain device input readbacks. This is selected kernel parity, not MLX
scan host redirection or a passing scan-versus-reduction assertion. DirectX and
OpenGL exact-width software prefix scans remain unsupported and are tracked in
[#2196](https://github.com/CrossGL/crosstl/issues/2196).

### Half-Precision Divmod

The pinned original Metal backend also has a half-precision `divmod` boundary
error. The macOS run at translator commit `78f86906` failed
`TestOps.test_divmod` for a float16 vector/scalar input. A deterministic local
reproduction with translation disabled is:

```python
import mlx.core as mx

mx.set_default_device(mx.gpu)
left = mx.array([86.375] * 10, dtype=mx.float16)
right = mx.array([28.796875], dtype=mx.float16)
quotient, remainder = mx.divmod(left, right)
print(quotient.tolist(), remainder.tolist())
```

Original Metal returns ten quotients of `3.0`; NumPy and the MLX CPU backend
return `2.0`. The remainder is `28.78125` on all three paths. A fixed-seed local
sample of 100,000 positive float16 pairs found 40 quotient mismatches on original
Metal and none on CPU. This harness redirects only complex-power entries, not
half-precision division. The defect is not evidence of a translation regression,
and the unchanged required test remains failing when it encounters this case.
These observations use the adapted local build with original kernel libraries,
not a separately rebuilt unmodified MLX binary.

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
python demos/integrations/mlx/run_metal_host.py prepare \
  --mlx-root /path/to/mlx --output-dir /path/to/proof/preparation
CMAKE_ARGS='-DMLX_METAL_JIT=ON -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF' \
  CMAKE_BUILD_PARALLEL_LEVEL=4 /path/to/mlx-env/bin/python -m pip install \
  -e /path/to/mlx --no-build-isolation
python demos/integrations/mlx/run_metal_host.py verify \
  --mlx-root /path/to/mlx --python /path/to/mlx-env/bin/python \
  --output-dir /path/to/proof/execution
```

Use a fresh execution directory for each run. The final `evidence.json` is written
only after both upstream runs, all three host datasets, dispatch verification,
library integrity checks and the missing-library negative test succeed. The
schema-version-2 report includes per-dataset readbacks and dispatch records.
Intermediate logs, partial numerical records and translation reports remain
available when a step fails.

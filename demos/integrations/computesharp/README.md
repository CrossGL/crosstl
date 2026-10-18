# ComputeSharp Portability

This integration ports the compute functionality of
[ComputeSharp](https://github.com/Sergio0694/ComputeSharp) at commit
`40dbe40de0b1eafb570e2af43f78bcdbffdcc00a` to Metal on macOS and desktop OpenGL
on Linux. CrossTL translates the HLSL that the unchanged ComputeSharp source
generator emits, and a portable implementation of the ComputeSharp runtime
executes the translated shaders, so the upstream `ComputeSharp.Tests` project
runs with its original assertions on each target. The original Direct3D 12 run
on Windows (WARP) is the independent baseline.

## Results

| Profile | Device | Execution | Required GPU passed | Capability-gated | Windows-specific | Negative controls |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `directx-warp` (original) | Direct3D 12 WARP | software | 1,809 WARP rows (1,964 cases in total) | 11 | run, 61 cases | n/a |
| `metal-apple-gpu` | Apple M2 Max, Metal 3 | hardware | 3,604 (3,713 cases in total) | 18 | excluded | 5 of 5 rejected |
| `opengl-mesa-llvmpipe` | Mesa 25.2.8 llvmpipe, OpenGL 4.5 core through EGL | software | 3,600 (3,709 cases in total) | 22 | excluded | 5 of 5 rejected |

Every one of the 3,792 discovered cases is accounted for on every profile:
required cases pass, capability-gated cases report the upstream capability
check, Windows-specific cases do not run on translated targets, and CPU-only
cases are reported separately and never counted as translated GPU execution.
Original discrete-device rows are optional on Windows because hosted runners
have no GPU; on the translated targets both device rows run on the one device.

Machine-readable results:

- `coverage.json` and `coverage.lock.json`: the frozen inventory and the expected
  outcome counts of every profile.
- `results.json` of each translated run (`run_tests.py`): translation counts,
  device preflight, accounting, evidence summary and negative controls.
- `qualification.json` of the `Project Demo Testing` workflow
  (`computesharp-qualification` artifact): every check recomputed from the raw
  artifacts of all three profiles, with per-case image metrics and difference
  images.

## How the Port Works

1. **Inventory.** `discovery/` loads the compiled test assembly, expands every
   MSTest data source (including ComputeSharp's combinatorial attributes) and
   exports each case with typed arguments, source location and a stable identity,
   plus every generated HLSL shader with its thread-group size, constant-buffer
   size, static-sampler requirement and resource descriptor ranges.
2. **Translation.** `translate.py` translates each unique HLSL source with
   `crosstl.translate_with_metadata` and writes a manifest with the artifact
   hashes, entry points and the binding slot CrossTL assigned to every HLSL
   register. Two source options select documented HLSL semantics: DXC's
   `-Zpr` row-major matrix packing (`matrix_packing=row_major`, all targets) and
   Direct3D-class trigonometric range reduction
   (`trigonometric_reduction=revolutions`, OpenGL). Metal artifacts are compiled
   and linked with `xcrun metal` during translation; OpenGL artifacts are
   compiled by the driver at pipeline creation. No shader is substituted by hand.
3. **Portable runtime.** `runtime/ComputeSharp.Portable.csproj` builds an
   assembly named `ComputeSharp` with the upstream public API. The pure upstream
   sources (attributes, dispatch intrinsics, pixel types, extension methods,
   views and interfaces) are compiled unchanged from the pinned checkout; the
   Direct3D 12 types (device, buffers, textures, transfer resources, compute
   context, pipeline cache) are reimplemented over the C ABI in
   `runtime/native/csbackend.h`, implemented by `csbackend_metal.m` and
   `csbackend_gl.c` (EGL, OpenGL 4.5 core). The runtime finds the translated
   artifact by the SHA-256 of the HLSL the generator produced and binds each
   descriptor range to the slot CrossTL reported. `runtime/dxc/` provides the
   `ReflectionServices` API off Windows (generated HLSL and the double-precision
   requirement of the executed artifact; DXIL statistics are Windows-only).
   Imaging uses the ImageSharp version pinned by the tests instead of WIC, with
   WIC's JPEG defaults (quality 90, 4:2:0 subsampling).
4. **Test adapter.** `runtime/Portable.targets` swaps the test project's
   references to the Direct3D 12 runtime and DXC for the portable assemblies and
   defines `COMPUTESHARP_TRANSLATED_BACKEND` for patches 0005 and 0006.
5. **Runner.** `run_tests.py` builds the native backend and the tests, checks
   the device against the profile before any test runs, translates, runs
   `dotnet test` without the Windows-specific methods, accounts the TRX with
   `baseline.py`, verifies the dispatch evidence with `evidence.py`, collects
   the output images and runs the negative controls.
6. **Qualification.** `qualify.py` recomputes every result from the uploaded
   artifacts of the three profiles (see [Qualification](#qualification)).

## Setup

- .NET SDK `10.0.400` (`upstream.json`) and Python 3.12 or later with this
  repository on the path (`pip install -e .`).
- macOS: an Apple-silicon Metal device and the Xcode Metal toolchain
  (`xcrun metal`; `xcodebuild -downloadComponent MetalToolchain` when missing).
- Linux: a C compiler and Mesa EGL/OpenGL (`libegl-dev libgl-dev libegl1
  libegl-mesa0 libgl1-mesa-dri libgbm1` on Ubuntu 24.04), or the image from
  `docker/opengl.Dockerfile`, which also works on macOS hosts through Docker.
- Image qualification: `numpy==2.5.4` and `pillow==13.0.0`.

## Commands

Prepare an adapted checkout and export its inventory:

```sh
git clone https://github.com/Sergio0694/ComputeSharp.git /path/to/ComputeSharp
git -C /path/to/ComputeSharp checkout 40dbe40de0b1eafb570e2af43f78bcdbffdcc00a
python demos/integrations/computesharp/prepare_upstream.py \
  --source /path/to/ComputeSharp --output /path/to/adapted \
  --receipt /path/to/adapted-source.json
dotnet build /path/to/adapted/tests/ComputeSharp.Tests/ComputeSharp.Tests.csproj \
  -c Release -p:CrossTLTrustedCorpusBaseline=true \
  -p:CustomAfterMicrosoftCommonTargets=$PWD/demos/integrations/computesharp/runtime/Portable.targets
dotnet run --project demos/integrations/computesharp/discovery/Discovery.csproj -c Release -- \
  /path/to/adapted/tests/ComputeSharp.Tests/bin/Release/net10.0/ComputeSharp.Tests.dll \
  /path/to/discovery
python demos/integrations/computesharp/coverage_contract.py check --discovery /path/to/discovery
```

Run every required case on Metal, then the negative controls:

```sh
python demos/integrations/computesharp/run_tests.py \
  --adapted /path/to/adapted --discovery /path/to/discovery \
  --profile metal-apple-gpu --work /path/to/metal-run --negative-controls
```

Run the same on Mesa llvmpipe in Docker (the adapted checkout must be a separate
Linux build tree, because the build output is platform-specific):

```sh
docker build -f demos/integrations/computesharp/docker/opengl.Dockerfile \
  -t crosstl-computesharp-opengl demos/integrations/computesharp/docker
docker run --rm -v "$PWD:$PWD" -v /path/to/adapted-linux:/path/to/adapted-linux \
  -v /path/to/discovery:/path/to/discovery:ro -v /path/to/gl-run:/path/to/gl-run \
  -w "$PWD" crosstl-computesharp-opengl \
  python3 demos/integrations/computesharp/run_tests.py \
  --adapted /path/to/adapted-linux --discovery /path/to/discovery \
  --profile opengl-mesa-llvmpipe --work /path/to/gl-run --negative-controls
```

Each run exits non-zero unless accounting, evidence and every control pass, and
writes `results.json`, the TRX, the evidence (`evidence.jsonl`), the translated
artifacts with their manifest, the collected outputs, content-addressed image
captures and logs under `--work`. `--filter` runs a subset for diagnosis only
and never qualifies.

Qualify downloaded workflow artifacts (or local directories with the same
layout: `discovery/` and `run/` for each target):

```sh
python demos/integrations/computesharp/qualify.py \
  --directx original-compute-directx-baseline --metal translated-metal \
  --opengl translated-opengl --output qualification
```

Run the tooling tests (they build a small MSTest fixture with the .NET SDK):

```sh
CROSSTL_REQUIRE_DOTNET=1 python -m pytest -q -n auto demos/integrations/computesharp/tests
```

## Coverage Contract

The pinned test assembly exports 303 test methods, 3,792 expanded cases (1,832
WARP rows, 1,817 discrete-device rows and 143 rows without a device argument) and
172 generated HLSL descriptors (148 unique sources). Upstream repeats some data
rows (for example `[Data(0, 64)]` three times on
`Texture1DTests.GetData_RangeToArray_Ok`); MSTest runs each repetition under the
same display name, so repetitions are numbered by `occurrence` and accounted as a
multiset of results.

`coverage.json` classifies every method, with a reason:

| Category | Methods | Cases |
| --- | ---: | ---: |
| required-gpu: portable compute and resource tests | 249 | 3,622 |
| cpu-only: host-side tests without a device | 15 | 109 |
| windows-specific: D3D12/DXGI/DXC interop and DXIL reflection | 39 | 61 |

| Profile | Pass | Capability-gated | Optional | Excluded |
| --- | ---: | ---: | ---: | ---: |
| `directx-warp` | 1,964 | 11 | 1,817 | 0 |
| `metal-apple-gpu` | 3,713 | 18 | 0 | 61 |
| `opengl-mesa-llvmpipe` | 3,709 | 22 | 0 | 61 |

Capability gates follow upstream `Assert.Inconclusive` checks and are frozen per
profile rather than discovered at run time: `fp64` (Metal has no double type;
each fp64 test has an upstream counterpart that must pass when it is absent) and
`rw-texture-rgb32` (three-channel 32-bit read-write textures, unsupported by
every profile, including Direct3D 12). Every other read-write texture element
type is required everywhere. A gated case must report the upstream capability
check; a missing device, failure or unexpected pass fails accounting. The
runner's preflight creates the native device before any test runs: a missing
device stops the run as an infrastructure blocker, and a device whose hardware
or software execution or fp64 support differs from the profile stops it as a
capability mismatch. Changing the contract requires a reviewed edit with an
entry in its `history` and a regenerated lock (`coverage_contract.py freeze`).

## Upstream Changes

The original Windows baseline builds the pinned checkout without changes. Other
builds apply the patch series in `patches/` through `prepare_upstream.py`, which
verifies the pristine checkout first and records patch and diff hashes:

1. `0001-classify-projects-with-posix-paths.patch`: upstream classifies projects
   by comparing a relative path with `src\` and `tests\`; MSBuild produces `/`
   on POSIX hosts, so generators were built for the wrong framework. The patch
   adds the equivalent `/` comparisons.
2. `0002-skip-dxc-bytecode-off-windows.patch`: the generator compiles the HLSL
   to DXIL with Windows-only DXC libraries. Off Windows it leaves the bytecode
   empty through its existing "missing bytecode" path. HLSL text, thread-group
   sizes, constant-buffer layout and descriptors come from unchanged code.
3. `0003-import-system-in-resource-loader.patch`: generated resource accessors
   name a `float` pixel type `Single` and compiled only because the bytecode
   helper imported `System` into the same namespace. The patch imports it for the
   resource loader itself.
4. `0004-match-include-path-case.patch`: two shared files are included as
   `extensions\` and `helpers\`, but the directories are `Extensions` and
   `Helpers`, which case-sensitive file systems cannot find.
5. `0005-use-one-translated-device.patch`: with `COMPUTESHARP_TRANSLATED_BACKEND`
   both `Device` values of the tests resolve to the single translated device, so
   WARP and discrete rows both run on it.
6. `0006-publish-translated-case-name.patch`: with the same symbol the
   combinatorial test attribute publishes the executing case name to the runtime,
   which attributes every dispatch and readback to its case. Execution and
   assertions are unchanged.

`compare_discovery.py` requires every adapted build to expose the same cases and
byte-identical generated HLSL and descriptors as the original Windows build.
No upstream assertion, tolerance, shader, input or expected image is changed.

## Evidence and Negative Controls

The runtime writes one JSON record per device, compiled pipeline, dispatch and
readback, each attributed to the executing case: the HLSL and artifact hashes,
thread-group size, dispatch size and thread groups, constant bytes, each
register with the slot and resource bound to it, and readback sizes and hashes.
Every native buffer carries 256-byte guard regions with a fixed pattern that is
verified after each submission. `evidence.py` checks the evidence against the
manifest, the artifact files, the discovery and the TRX: every pipeline uses a
translated artifact, every descriptor range is bound to the slot CrossTL
reported, the thread groups cover the dispatch, dispatches come only from
required GPU cases, and the device matches the profile.

`negative_controls.json` lists five faults injected through
`CROSSTL_COMPUTESHARP_FAULT` and the upstream methods that must fail under each:
omitted dispatches, swapped bindings, perturbed constants, corrupted readbacks
and an out-of-bounds device write into a buffer guard. Every case of every
listed method must execute and fail (guard corruption must report the guard
violation), must have passed in the unfaulted run, and the control evidence must
carry the fault.

## Image Checks

The upstream `TolerantImageComparer` assertions are kept unchanged. The comparer
sums Manhattan distances of 8-bit RGBA channels but normalizes by `4 * 65535`, so
its thresholds are 257 times looser than they read: the `TwoTiledTruchet`
threshold of 0.00032 allows a mean error of about 21 of 255 per channel, and an
all-black output of the 1D sampling test scored only 0.00115. Several shaders
(`ContouredLayers`, `FourColorGradient`, `PyramidPattern`,
`TriangleGridContouring`, `TwoTiledTruchet`) derive patterns from hashes such as
`frac(sin(x) * 43758.5453)` with large arguments, so their images depend on the
transcendental precision of each implementation, and the upstream reference
images are one implementation's rendering.

`image_checks.json` adds differential checks that `qualify.py` applies without
loosening any upstream threshold:

- every ShadersTests case on a translated target has exactly one captured
  readback; the captured PNG must hash to the readback bytes in the evidence,
  match the reference image dimensions and reproduce the upstream assertion;
- the capture is compared with the original WARP output of the same shader.
  The bounds come only from WARP and the reference image: by the triangle
  inequality, an output that is no less accurate than WARP relative to the
  reference satisfies `MAE(target, WARP) <= 2 * MAE(WARP, reference)` and
  `fraction(|target - WARP| > 16) <= 2 * fraction(|WARP - reference| > 8)`,
  with margins of 0.5 (rounding) and 0.1% of pixels (isolated edge pixels);
- the ImageProcessingTests GPU outputs are compared with the WARP GPU output
  using bounds from the WARP output and the CPU reference of the Windows run;
- amplified difference images are written for every compared output.

## Limitations

- Metal has no double-precision type: the 6 double-precision cases are
  capability-gated on Metal and the 10 cases that require the "unsupported"
  behavior pass there instead. WARP and OpenGL provide fp64, so on those
  profiles it is the other way round.
- Read-write textures of `Float3`, `Int3` and `UInt3` (12 cases on each
  translated target) are unsupported on every profile, as on Direct3D 12.
- Windows-specific tests (DXIL reflection, D3D12/DXGI interop, COM) run only in
  the original baseline.
- The Metal backend requires a unified-memory device and compiles with
  `MTLMathModeSafe` and fast fp32 transcendental functions, which matches the
  Direct3D precision class of the hash-based shaders; correctly rounded sine
  differs more from the Direct3D references. The OpenGL translation reduces
  trigonometric arguments in revolutions for the same reason.
- Evidence records dispatches, pipelines and readbacks. Cases that only allocate
  resources or validate arguments (for example `*_Fail` transfer tests) leave no
  records; their results come from the upstream assertions alone.
- The adapted build uses the ImageSharp codec instead of WIC for the imaging
  tests, with WIC's default JPEG settings.

## Pinned Dependency Exceptions

The unmodified upstream pin uses ImageSharp 3.1.11 and treats NuGet advisory
warnings as errors. Five current advisories prevent restoring that historical
baseline: [ICC profile allocation](https://github.com/advisories/GHSA-gwg2-r3hj-4w44),
[histogram equalization](https://github.com/advisories/GHSA-j3p4-wp97-rph4),
[TIFF T4 encoding](https://github.com/advisories/GHSA-j9gm-c75j-xc9q),
[TIFF T6 encoding](https://github.com/advisories/GHSA-jjfr-hcj7-qf5w), and
[BigTIFF decoding](https://github.com/advisories/GHSA-wmxv-xphr-5c9g).
The identified fixed release, 4.1.2, requires a separate license key to build and
is not substituted into the original baseline. `baseline-audit.props` keeps NuGet
auditing enabled and adds advisory-specific exceptions only for
`ComputeSharp.Tests`, enabled by `CrossTLTrustedCorpusBaseline=true`. This is
restricted to the immutable, trusted upstream test code and its tracked assets;
it must not be used to process external images or as a dependency policy for a
shipped application.

## Qualification

`qualify.py` checks, for each profile: the coverage lock, discovery parity with
the original Windows export, TRX accounting with per-category outcome counts,
the dispatch evidence, every negative control, the job report (including the
device preflight) and the image checks. The workflow runs the original Windows
WARP job, the adapted-source parity job, the macOS Metal job, the Linux Mesa
llvmpipe job and then the qualification job, with bounded timeouts and seven
to fourteen days of artifact retention.

## Licenses

ComputeSharp is MIT-licensed; its `LICENSE` and `ThirdPartyNotices.txt` apply to
the upstream sources compiled into the portable runtime and to the patches.
Upstream dependencies and image assets keep their own terms; several shaders are
ports of Shadertoy works under CC BY-NC-SA 3.0, as their sources state. The
integration code in this directory is part of CrossTL (Apache License 2.0).

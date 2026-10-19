# ComputeSharp Portability

This integration ports the compute functionality of
[ComputeSharp](https://github.com/Sergio0694/ComputeSharp) at commit
`40dbe40de0b1eafb570e2af43f78bcdbffdcc00a` to Metal on macOS and desktop OpenGL
on Linux. CrossTL translates the HLSL that the unchanged ComputeSharp source
generator emits, and a portable implementation of the ComputeSharp runtime
executes the translated shaders, so the upstream `ComputeSharp.Tests` project
and its three portable companion test projects run with their original
assertions on each target. The original Direct3D 12 run
on Windows (WARP) is the independent baseline.

## Results

The qualification job of workflow run
[39288736223](https://github.com/CrossGL/crosstl/actions/runs/39288736223)
(commit `26684d48ebe0f3e2d5d405da849e63e9dcce3109`) recomputed every check from
the fresh artifacts of the three profile jobs and passed, for the main test
project, the three ported companion projects and the two retained projects.
`results.json` is the machine-readable summary that job wrote;
`results_summary.py` regenerates it byte for byte from the job's
`qualification.json`.

This run also verified arithmetic exactness on the target compilers: 144 Metal
artifacts with 11,336 floating-point operations and 148 OpenGL artifacts with
32,235 floating-point operations had no contractible operations. Four Metal
artifacts require fp64 and are not compiled on that profile, as declared by the
frozen capability contract.

| Profile | Device | Execution | Required GPU passed | Capability-gated | Windows-specific | Negative controls |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `directx-warp` (original) | Direct3D 12 WARP (Microsoft Basic Render Driver), Windows Server 2025 | software | 1,809 WARP rows (1,964 cases in total) | 11 | run, 61 cases | n/a |
| `metal-apple-gpu` | Apple Paravirtual device (the GPU of the `macos-15` runner), macOS 15.7.9, Metal | hardware | 3,604 (3,713 cases in total) | 18 | excluded | 6 of 6 rejected |
| `opengl-mesa-llvmpipe` | llvmpipe (LLVM 20.1.2, 256 bits), Mesa 25.2.8, OpenGL 4.5 core through EGL, Ubuntu 24.04 | software | 3,600 (3,709 cases in total) | 22 | excluded | 6 of 6 rejected |

Every one of the 3,792 discovered cases is accounted for on every profile:
required cases pass, capability-gated cases report the upstream capability
check, Windows-specific cases do not run on translated targets, and CPU-only
cases are reported separately and never counted as translated GPU execution.
Original discrete-device rows are optional on Windows because hosted runners
have no GPU; on the translated targets both device rows run on the one device.
Local runs on an Apple M2 Max produce bit-identical shader images to the CI
device.

The companion and retained projects of the same run (cases passed per profile;
optional rows are the original discrete-device rows, which hosted Windows
runners cannot run):

| Project | Scope | `directx-warp` | `metal-apple-gpu` | `opengl-mesa-llvmpipe` |
| --- | --- | --- | --- | --- |
| `DisableDynamicCompilation` | ported | 2 passed, 2 optional | 4 passed | 4 passed |
| `GlobalStatements` | ported | 1 passed | 1 passed | 1 passed |
| `Internals` | ported | 86 passed, 51 optional | 137 passed | 137 passed |
| `DeviceLost` | windows-specific | 23 passed, 20 optional | not run | not run |
| `SourceGenerators` | cpu-only | 110 passed | not run | not run |

Every negative control of the ported companion projects rejected its faulted
run on both targets. The retained projects ran only on the original Windows job
and are not counted as translated GPU execution; the three upstream DeviceLost
steps selected no case, as described in [Retained
projects](#retained-projects).

Image differences of the shader outputs against the original WARP output
(mean absolute error of the RGB channels in 0-255 units; the bound is derived
from WARP and the reference image only, see [Image Checks](#image-checks)):

| Shader | WARP vs reference | Metal vs WARP | OpenGL vs WARP | Bound |
| --- | ---: | ---: | ---: | ---: |
| `ColorfulInfinity` | 0.000 | 0.0000 | 0.0000 | 0.500 |
| `ContouredLayers` | 7.437 | 0.0477 | 0.0418 | 15.373 |
| `ExtrudedTruchetPattern` | 0.002 | 0.1216 | 0.1221 | 0.505 |
| `FourColorGradient` | 9.003 | 0.0000 | 0.0000 | 18.507 |
| `FractalTiling` | 0.000 | 0.0000 | 0.0000 | 0.501 |
| `HelloWorld` | 0.000 | 0.0000 | 0.0000 | 0.500 |
| `MengerJourney` | 0.010 | 0.0042 | 0.0042 | 0.520 |
| `Octagrams` | 0.000 | 0.0088 | 0.0026 | 0.500 |
| `ProteanClouds` | 0.000 | 0.0000 | 0.0000 | 0.500 |
| `PyramidPattern` | 0.046 | 0.1294 | 0.1294 | 0.592 |
| `TerracedHills` | 0.020 | 0.0139 | 0.0139 | 0.540 |
| `TriangleGridContouring` | 0.001 | 0.0001 | 0.0001 | 0.501 |
| `TwoTiledTruchet` | 0.000 | 0.0008 | 0.0009 | 0.500 |

Machine-readable results:

- `coverage.json` and `coverage.lock.json`: the frozen inventory and the expected
  outcome counts of every profile.
- `results.json`: the summary of the qualifying workflow run: devices, case
  outcomes per category, evidence counts, negative controls, arithmetic
  exactness of the translated artifacts, intrinsic precision against WARP and
  the image metrics of every output with its bound.
  `results_summary.py` regenerates it from the run's `qualification.json`.
- `results.json` of each translated run (`run_tests.py`): translation counts,
  device preflight, deadlines, arithmetic exactness, accounting, evidence
  summary and negative controls; `exactness.json` lists every artifact with
  its hash and floating-point operation counts.
- `qualification.json` and `results.json` of the `ComputeSharp Port Testing`
  workflow (`computesharp-qualification` artifact): every check recomputed from
  the raw artifacts of all three profiles, with per-case image metrics and
  difference images.

## How the Port Works

1. **Inventory.** `discovery/` loads the compiled test assembly, expands every
   MSTest data source (including ComputeSharp's combinatorial attributes) and
   exports each case with typed arguments, source location and a stable identity,
   plus every generated HLSL shader with its thread-group size, constant-buffer
   size, static-sampler requirement and resource descriptor ranges.
2. **Translation.** `translate.py` translates each unique HLSL source with
   `crosstl.translate_with_metadata` and writes a manifest with the artifact
   hashes, entry points and the binding slot CrossTL assigned to every HLSL
   register. Four HLSL source options select the semantics of the original
   Direct3D execution for both targets (see [Floating-Point
   Evaluation](#floating-point-evaluation)): DXC's `-Zpr` row-major matrix
   packing (`matrix_packing=row_major`), WARP's sine and cosine evaluation
   (`trigonometric_reduction=warp`), contraction-free arithmetic
   (`floating_point_contraction=off`) and Direct3D's float-to-UNORM store
   conversion (`unorm_store_rounding=single_precision`). Metal artifacts are compiled and linked
   with `xcrun metal` during translation; OpenGL artifacts are compiled by the
   driver at pipeline creation. No shader is substituted by hand.
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
   the device against the profile before any test runs, translates, checks
   with `exactness.py` that the target compiler keeps the translated
   arithmetic as written, runs `dotnet test` without the Windows-specific
   methods, accounts the TRX with `baseline.py`, verifies the dispatch evidence
   with `evidence.py`, collects the output images and runs the negative
   controls.
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

Run the companion test projects (every other ported project of
`test_projects.json`) on the same target after the main suite; each one is
built against the portable runtime, exported, checked against its frozen lock
and run with `run_tests.py --suite <project>` and its negative controls under
`/path/to/suites/<project>`:

```sh
python demos/integrations/computesharp/companions.py translated \
  --adapted /path/to/adapted --profile metal-apple-gpu --output /path/to/suites
```

On Windows, `companions.py baseline --source /path/to/ComputeSharp --output
/path/to/suites` runs the original projects on Direct3D 12 as the upstream
workflow does and accounts them with `baseline.py`.

Also on Windows, retain the original Windows-specific and CPU-only test
projects (see [Retained projects](#retained-projects)):

```sh
python demos/integrations/computesharp/retained.py run \
  --source /path/to/ComputeSharp --output /path/to/retained
```

Each run exits non-zero unless accounting, evidence, arithmetic exactness and
every control pass, and writes `results.json`, `exactness.json`, the TRX, the
evidence (`evidence.jsonl`), the translated artifacts with their manifest, the
collected outputs, content-addressed image captures and logs under `--work`.
`--filter` runs a subset for diagnosis only and never qualifies.

Check the arithmetic exactness of translated artifacts on their own (OpenGL
needs the native library of a run):

```sh
python demos/integrations/computesharp/exactness.py --target opengl \
  --artifacts /path/to/gl-run/artifacts \
  --library /path/to/gl-run/native/libcsbackend_gl.so --output exactness.json
```

Evaluate the intrinsic probe on a translated backend with the native library
of a run (the Windows job runs the same probe against the original runtime on
WARP with `dotnet run --project precision/IntrinsicProbe.csproj -- run warp`):

```sh
python demos/integrations/computesharp/precision/run_probe.py \
  --adapted /path/to/adapted --target metal \
  --native-library /path/to/metal-run/native/libcsbackend_metal.dylib \
  --output /path/to/metal-precision
```

Qualify downloaded workflow artifacts (or local directories with the same
layout: `discovery/`, `run/` and `precision/` for each target,
`suites/<project>/` for each companion suite and, in the Windows artifacts,
`retained/<project>/` for each retained project):

```sh
python demos/integrations/computesharp/qualify.py \
  --directx original-compute-directx-baseline --metal translated-metal \
  --opengl translated-opengl --output qualification
```

Regenerate the published summary from the qualification artifact of a run:

```sh
python demos/integrations/computesharp/results_summary.py \
  --qualification computesharp-qualification/qualification.json \
  --workflow-run https://github.com/CrossGL/crosstl/actions/runs/<run>/attempts/<attempt> \
  --source-commit <commit> --output demos/integrations/computesharp/results.json
```

Run the tooling tests (they build a small MSTest fixture with the .NET SDK):

```sh
CROSSTL_REQUIRE_DOTNET=1 python -m pytest -q -n auto demos/integrations/computesharp/tests
```

## Coverage Contract

### Upstream test projects

`test_projects.json` accounts for all 15 projects under the pinned `tests/`
directory, each with a scope, a specific reason and the exact steps of the
upstream `dotnet.yml` workflow that build or run it; `suites.py check` fails if
a project or workflow step is added, removed or recorded differently.

| Scope | Projects |
| --- | --- |
| ported | `ComputeSharp.Tests`, `ComputeSharp.Tests.DisableDynamicCompilation`, `ComputeSharp.Tests.GlobalStatements`, `ComputeSharp.Tests.Internals` |
| windows-specific | `ComputeSharp.Tests.DeviceLost` (D3D12 device removal and COM reference counts), `ComputeSharp.Tests.NativeLibrariesResolver` (win-x64 native library resolution) |
| cpu-only | `ComputeSharp.Tests.SourceGenerators` (Roslyn generator and analyzer tests) |
| out-of-scope | the five Direct2D and WinUI projects and the three NuGet packaging projects |

Each ported companion project has its own frozen contract, lock, evidence
policy and negative controls under `suites/<project>/`, sharing the profiles,
categories and capabilities of the main contract:

| Project | Cases | WARP pass (optional) | Metal pass | OpenGL pass | Controls |
| --- | ---: | ---: | ---: | ---: | ---: |
| `DisableDynamicCompilation` | 4 | 2 (2) | 4 | 4 | 4 |
| `GlobalStatements` (program) | 1 | 1 | 1 | 1 | 4 |
| `Internals` (109 required GPU, 28 CPU-only) | 137 | 86 (51) | 137 | 137 | 1 |

`GlobalStatements` is a top-level program without MSTest methods: discovery
exports its entry point as its one case and `program_runner.py` runs the
discovered assembly, records a pass only for a zero exit status (the program
throws when the GPU result differs from its CPU evaluation) and writes the
outcome as TRX, so the case is accounted, attributed in the dispatch evidence
and checked by negative controls like an MSTest case. `Internals` dispatches
no shader: its device cases create resources, read them back through the
debugger views and load generated dispatch data on the host, so its evidence
policy names the reason for each case without a dispatch, its one control
corrupts readbacks, and the suite declares that its artifacts (whose bodies only
compare thread ids with the dispatch size) do no float arithmetic instead of
requiring float operations in the exactness check.

#### Retained projects

The Windows-specific and CPU-only projects test what the port does not
translate, so they never run on a translated target and never count as
translated GPU execution. `retained.py run` runs them on the original Windows
job and accounts their results separately, on the `directx-warp` profile of
their own frozen contract and lock under `suites/<project>/`, which classify
every case in the project's scope:

| Project | Scope | Cases | Runs |
| --- | --- | ---: | --- |
| `DeviceLost` | windows-specific | 43, of which 20 are optional discrete-device rows | the three upstream steps verbatim, then each test class on its own (`Release`) |
| `SourceGenerators` | cpu-only | 110 | the upstream step (`Debug`, as upstream) |

The upstream DeviceLost steps filter by `TestCategory`, but no DeviceLost test
declares a category at the pinned commit, so those steps select no case; the
retained run records the number each one selects and requires it to succeed.
The accounted runs follow the intent the upstream workflow states, one test
class per process, and together must report every discovered case once; their
TRX files are merged into the `baseline.trx` that `baseline.py` accounts, and
qualification recomputes the merge. The allocator variant of the steps
(`run-tests-d3d12ma`) is not repeated, as for the main suite.
`NativeLibrariesResolver` is not retained: it runs in the upstream
`verify-packages` job against the published NuGet packages, which the port does
not rebuild.

`SourceGenerators` contains no generated shader; the discovery tool exports
it with `--host-only`, which fails for a project with descriptors, and its lock
records the host-only export. MSTest names `DataRow` cases with their string
arguments in quotes, which the exporter reproduces. On a non-Windows host the
adapted generator (patch 0002) leaves the DXIL bytecode empty, so 6 of these
110 cases cannot pass there: the three generated-source comparisons, which
embed the bytecode, and the DXC compilation and double-precision diagnostics
(`RecursiveMembers`, `MissingRequiresDoublePrecisionSupportAttribute`,
`UnnecessaryRequiresDoublePrecisionSupportAttribute`). A local macOS run of
the adapted generator passed the other 104, and the generated-source
differences were confined to the bytecode. The retained results therefore
come from the pristine Windows build.

### Main test project

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

The runtime writes one JSON record per device, capability query, resource
allocation and release, compiled pipeline, dispatch and readback, each
attributed to the executing case (releases are not, because finalizers release
resources while other cases run): the answer of every public capability query
(`IsDoublePrecisionSupportAvailable`, `Is*Texture*SupportedForType<T>`), the
type, memory (device or host), size and shape of every resource, the HLSL and
artifact hashes, thread-group size, dispatch size and thread groups, constant
bytes, each register with the slot and resource bound to it, and readback sizes
and hashes. Every native buffer carries 256-byte guard regions with a fixed
pattern that is verified after each submission. Runs have deadlines: MSTest
stops the test host when one case runs longer than 5 minutes (the slowest case
takes about 10 seconds), and the runner stops a complete run after 25 minutes
and a control run after 4 minutes, so a hung dispatch fails the run instead of
holding the job. `evidence.py` checks the evidence against the manifest, the
artifact files, the discovery and the TRX: every pipeline uses a translated
artifact, every descriptor range is bound to the slot CrossTL reported and to a
device resource that was allocated and not yet released, every UNORM store
scale buffer of the artifact holds the channel scales of the bound view's
format (from an independent table of the DXGI formats), readbacks read live
device resources, the thread groups cover the dispatch, dispatches come only
from required GPU cases, and the device matches the profile. The runtime also
refuses to dispatch an artifact with a binding that no descriptor range
supplies.

Every executed required GPU case must leave evidence. `evidence_policy.json`
lists the reviewed exceptions with their reasons: the `Allocate_Uninitialized_Fail`
methods and two null-argument imaging tests, whose arguments are rejected
before any device work, `InitializationTests.IsSupported`, which only queries
the process-wide default device, and `InitializationTests.QueryDevices`, whose
body runs once per device with dedicated memory and therefore not at all on the
shared-memory Metal and llvmpipe devices. The policy also states which capability
query answers each capability of the contract: every case of a capability-gated
method must record that query, and every answer must equal the capability the
profile declares, so a gated case cannot be skipped by anything but the device
query.

`negative_controls.json` lists six faults injected through
`CROSSTL_COMPUTESHARP_FAULT` and the upstream methods that must fail under each:
omitted dispatches, swapped bindings, perturbed constants, UNORM store scales
of one level per channel, corrupted readbacks and an out-of-bounds device write
into a buffer guard. Every case of every
listed method must execute and fail (guard corruption must report the guard
violation), must have passed in the unfaulted run, and the control evidence must
carry the fault.

## Floating-Point Evaluation

Several shaders hash coordinates with expressions such as
`frac(sin(x) * 43758.5453)`, and the reference images were rendered by WARP,
so outputs depend on the last bits of single-precision results. The intrinsic
probe in `precision/` measures the evaluation of 25 HLSL expressions over
155,660 fixed inputs on WARP (Windows job) and on each translated target, from
the same generated HLSL. Four general HLSL source options of CrossTL select
the original evaluation:

- `trigonometric_reduction=warp`: WARP reduces sine and cosine arguments in
  revolutions (`t = x * (1 / 2pi)`, `f = t - floor(t)`), looks up one of 32
  table entries and evaluates short polynomials, rounding every step and
  flushing subnormal intermediates; `tan` is `sin / cos`. The frontend emits
  the same sequence as precise helper functions with the tables. Arguments
  that DXC folds to constants keep the exact intrinsic.
- `floating_point_contraction=off`: WARP executes the DXIL as written, so
  every product and sum is rounded, `dot` adds the component products from the
  first component on, `lerp(a, b, s)` is `a + s * (b - a)` and matrix products
  accumulate in DXC's order. Metal fuses multiply-adds, including inside
  `dot`, `mix`, `length`, `normalize`, `smoothstep`, `cross`, `reflect` and
  matrix products, and llvmpipe adds three-component dot products in another
  order. The frontend marks every function precise (Metal
  `#pragma clang fp contract(off)`, GLSL `precise`) and replaces these
  intrinsics with precise helpers that evaluate the DXC lowering one operation
  at a time. GLSL compilers apply `precise` only to the operations whose
  result an assignment to a precise variable consumes, so in a precise
  function the GLSL backend declares every local and output parameter
  precise, copies written value parameters into precise locals, and assigns
  values to precise locals where they are evaluated when no precise variable
  consumes them: control conditions, user-function arguments, stores to
  buffers, images and shared memory, and computed values that a swizzle,
  member access or index selects from.
- `matrix_packing=row_major`: ComputeSharp compiles with DXC's `-Zpr`.
- `unorm_store_rounding=single_precision`: Direct3D stores a float to an
  n-bit UNORM channel by clamping it, multiplying it by `2^n - 1` in single
  precision and rounding to even. Apple GPUs round the exact product, so `0.3f`
  is stored as 77 instead of 76 in an 8-bit channel, which
  `DisableDynamicCompilation`'s `PixelShader_Test_Ok` asserts exactly. The
  frontend quantizes every store to a texture declared `unorm` to
  `round_even(saturate(v) * scale) / scale` in a precise helper, which every
  target then converts to the same level. The scale belongs to the view
  format, not to the shader, so each such texture gets a constant buffer
  `crosstl_unorm_store_<register>` with per-channel scales that the runtime
  binds from the format of the bound view (255 or 65535 for the channels the
  format has).

`exactness.py` checks that the target compiler keeps every translated
artifact's arithmetic as written, using the runtime's compile options: the
Metal LLVM IR (`xcrun metal -S -emit-llvm`) may contain no fused multiply-add,
`air.mix` or run-time `air.dot` and no contracting or reassociating
floating-point instruction, and in the NIR that Mesa compiles each OpenGL
artifact to (`MESA_GLSL=dump` through the native backend) every
floating-point add, multiply, multiply-add, interpolation and dot product must
be exact. `run_tests.py` fails a run with an inexact artifact and
`qualify.py` re-verifies the report against the run's manifest. llvmpipe
never fuses, so the earlier GLSL translation, which declared only function
results precise, still matched WARP there, but 6,120 of its NIR
floating-point operations in 57 artifacts were not exact and a driver that
fuses could have contracted them.

`qualify.py` requires the translated `sin`, `cos` and `tan` to be
bit-identical to WARP (except for operands whose revolution is subnormal) and
the contraction-free probes (`dot2`, `dot3`, `multiply-add`, `mad`, `lerp`,
`smoothstep`) to equal WARP and an independent single-precision CPU model in
DXC's operation order, where only subnormal flushing may differ. The Metal
results of the earlier translation with contraction allowed (macOS CI) fail
this check for five of the six probes, on 9,443 to 33,556 of the 155,660
inputs each. The remaining
functions (`sqrt`, `rsqrt`, `exp`, `log`, `pow`, `atan`) use the native
implementation of each target, and the report gives their distance from WARP
in ulps.

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
images are one implementation's rendering (WARP's, except for
`ContouredLayers` and `FourColorGradient`). With correctly rounded or fused
evaluation, these shaders rendered different hash patterns that the upstream
thresholds still accepted; the [floating-point
evaluation](#floating-point-evaluation) above removes those differences.

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

- Metal has no double-precision type: the scalar types of the Metal Shading
  Language Specification (section 2.1) exclude `double`. The 6
  double-precision cases are capability-gated on Metal and the 10 cases that
  require the "unsupported" behavior pass there instead. WARP and OpenGL
  provide fp64, so on those profiles it is the other way round.
- Read-write textures of `Float3`, `Int3` and `UInt3` (12 cases on each
  translated target) are unsupported on every profile: `MTLPixelFormat`
  defines no three-channel 32-bit formats, OpenGL image load/store accepts no
  three-channel 32-bit formats (OpenGL 4.6 core specification, section 8.26),
  and on Direct3D 12 WARP the upstream capability check reports the DXGI
  R32G32B32 formats as unsupported for read-write textures.
- Windows-specific tests (DXIL reflection, D3D12/DXGI interop, COM) run only in
  the original baseline.
- The Metal backend requires a unified-memory device and compiles with
  `MTLMathModeSafe` (IEEE NaN and infinity semantics) and fast fp32 functions.
  Native `sqrt`, `rsqrt`, `exp`, `log`, `pow` and `atan` differ from WARP by a
  few ulps on both targets (OpenGL's `sqrt` and `log` are closer), and these
  account for the remaining image differences against WARP, which are of the
  same size on both targets. Metal's precise functions bring `sqrt` and `log`
  closer to WARP but `exp` and `pow` further away.
- WARP flushes subnormal operands and results to zero; llvmpipe keeps them,
  so results that involve subnormals can differ from WARP.
- The exactness check inspects the compilers of the CI and local devices
  (Apple's Metal compiler and Mesa); other OpenGL drivers must honor
  `precise`, which GLSL 4.00 requires. Translations that need `precise`
  raise desktop GLSL targets below 4.00 to 4.00 and reject GLSL ES targets
  below 3.20, including WebGL 2, instead of silently relaxing the evaluation.
- The cases of the 12 host-only methods in `evidence_policy.json` leave no
  device records; their results come from the upstream assertions alone.
  `InitializationTests.QueryDevices` passes without running its body on the
  translated targets, because no shared-memory device reports dedicated memory.
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
`ComputeSharp.Tests`, its three ported companion test projects, the portable
runtime and the intrinsic probe, enabled by `CrossTLTrustedCorpusBaseline=true`.
The retained `ComputeSharp.Tests.SourceGenerators` project reaches
System.Security.Cryptography.Pkcs 9.0.6 through the Roslyn test packages and
has one exception of its own,
[GHSA-2fmv-rv2c-59x7](https://github.com/advisories/GHSA-2fmv-rv2c-59x7). This is
restricted to the immutable, trusted upstream test code and its tracked assets;
it must not be used to process external images or as a dependency policy for a
shipped application.

## Qualification

`qualify.py` checks, for each profile: the coverage lock, discovery parity with
the original Windows export, TRX accounting with per-category outcome counts,
the dispatch evidence, every negative control, the job report (including the
device preflight), the arithmetic exactness of the translated artifacts, the
intrinsic precision and the image checks. Every companion suite must pass the
same checks, except the image checks and the intrinsic probe, with its own
frozen files; the qualification fails if any suite is missing on any profile.
Every retained project must pass its coverage lock and accounting on the
original profile, and its recorded runs must match the inventory, with the
merged TRX recomputed; its results are published under `retained`, separately
from the translated suites.
It writes
`qualification.json` with the summary `results.json` (`results_summary.py`). The workflow runs the original Windows
WARP job, the adapted-source parity job, the macOS Metal job, the Linux Mesa
llvmpipe job and then the qualification job, with bounded timeouts and seven
to fourteen days of artifact retention.

Every pull request, merge-queue group and push to `main` starts the workflow,
and it ends in one `ComputeSharp validation result` check that fails when any
job fails or is cancelled. Its `ComputeSharp validation plan` job
(`ci_plan.py`) diffs the event from its merge base and skips the port's jobs
unless this directory, the workflow, the packaging files or the CrossTL code
that the HLSL translation imports changed: the top-level `crosstl/*.py`
modules, `crosstl/backend/**` and `crosstl/translator/**`.
`crosstl/project/**` and other demos do not select it; manual dispatches always
do. `tests/test_workflow.py` translates generated ComputeSharp HLSL in a fresh
interpreter and fails if a loaded CrossTL module falls outside these paths. A
new push to a pull request cancels that pull request's superseded run.

## Licenses

ComputeSharp is MIT-licensed; its `LICENSE` and `ThirdPartyNotices.txt` apply to
the upstream sources compiled into the portable runtime and to the patches.
Upstream dependencies and image assets keep their own terms; several shaders are
ports of Shadertoy works under CC BY-NC-SA 3.0, as their sources state. The
integration code in this directory is part of CrossTL (Apache License 2.0).

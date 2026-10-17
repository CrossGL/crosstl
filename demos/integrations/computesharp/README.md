# ComputeSharp Portability

This integration ports the compute tests of
[ComputeSharp](https://github.com/Sergio0694/ComputeSharp) at commit
`40dbe40de0b1eafb570e2af43f78bcdbffdcc00a` from the HLSL its source generator
emits to Metal on macOS and OpenGL on Linux. The original Direct3D 12 run on
Windows is the independent baseline. Translated execution is not yet qualified;
see [Status](#status).

## Layout

- `upstream.json`: immutable source revision, SDK and qualification status.
- `patches/`: the complete list of upstream source changes used by non-Windows
  builds. The original Windows baseline is built without them.
- `prepare_upstream.py`: verifies a pristine checkout of the pin and applies the
  patch series to a separate clone, writing a receipt of patch and diff hashes.
- `discovery/`: compiled-assembly exporter for the actual MSTest cases and the
  generated HLSL descriptors. It invokes upstream `ITestDataSource.GetData`,
  including ComputeSharp's combinatorial attributes, rather than parsing names.
- `source.py`: content hashes for every tracked upstream file, checked before
  and after the original test run.
- `coverage.json`: the reviewed classification of every upstream test method,
  the target profiles and the optional device capabilities.
- `coverage.lock.json` and `coverage_contract.py`: the frozen case inventory
  derived from the contract and a discovery export (`freeze`/`check`).
- `baseline.py`: accounts TRX results of any profile against the frozen contract.
- `compare_discovery.py`: requires the adapted build to expose the same cases and
  byte-identical HLSL with identical descriptor metadata as the original build.
- `tests/`: exporter and accounting tests using real MSTest discovery and execution
  with a small descriptor fixture. These fixtures are not a translated corpus.
- `baseline-audit.props`: narrowly scoped legacy dependency exceptions described
  below. No upstream assertion, shader, expected image or input is changed.

## Inventory

`discovery.json` preserves typed test arguments, including floating-point bit
patterns, and gives each case a stable content identity. Upstream repeats some
data rows (for example `[Data(0, 64)]` three times on
`Texture1DTests.GetData_RangeToArray_Ok`); MSTest executes each repetition with
the same display name, so repetitions are separate cases numbered by
`occurrence` and accounted as a multiset of results per display name.

The pinned test assembly exports 303 test methods, 3,792 expanded cases (1,832
WARP rows, 1,817 discrete-device rows and 143 rows without a device argument) and
172 generated HLSL descriptors. Each method records its source location from the
assembly's portable PDB. The Windows original and the adapted build produce the
same case identities.

## Coverage Contract

`coverage.json` classifies every method, with a reason, into one of three
categories; `coverage.lock.json` freezes the result per method signature (source
location, case count and a digest of the case identities):

| Category | Methods | Cases |
| --- | ---: | ---: |
| required-gpu: portable compute and resource tests | 249 | 3,622 |
| cpu-only: host-side tests without a device | 15 | 109 |
| windows-specific: D3D12/DXGI/DXC interop and DXIL reflection | 39 | 61 |

Profiles and expected outcomes:

| Profile | Device | Pass | Capability-gated | Optional | Excluded |
| --- | --- | ---: | ---: | ---: | ---: |
| `directx-warp` (original) | D3D12 WARP, software | 1,964 | 11 | 1,817 | 0 |
| `metal-apple-gpu` | system Metal device, hardware | 3,713 | 18 | 0 | 61 |
| `opengl-mesa-llvmpipe` | Mesa llvmpipe via EGL, software | 3,709 | 22 | 0 | 61 |

Translated targets expose one device, and the test adapter maps both upstream
`Device` values to it, so every upstream row runs on every target. Only Discrete
rows of the original baseline are optional (hosted runners have no GPU).

Capability gates follow upstream `Assert.Inconclusive` checks and are frozen per
profile rather than discovered at run time:

- `fp64`: double-precision shaders. WARP and OpenGL 4.x provide it; Metal has no
  double type. Each fp64 test has an upstream counterpart that must pass when
  the capability is absent.
- `rw-texture-rgb32`: three-channel 32-bit read-write textures (`Float3`,
  `Int3`, `UInt3`), which no profile supports. Every other read-write texture
  element type is required on every profile.

A gated case must report the upstream capability check; an inconclusive result
caused by a missing device, a failure or an unexpected pass fails accounting.
Windows-specific cases must not run on translated targets. Tests that call
`ReflectionServices` (DXIL reflection) for incidental checks remain required:
off Windows the adapter reports the generated HLSL and whether the executed
artifact requires double precision. `ImagingTests` keep their GPU transfers;
off Windows the image codec is the ImageSharp version pinned by the tests
instead of WIC.

The first complete original run on WARP matched the frozen `directx-warp`
expectations exactly (1,964 passed, 11 capability-gated, 1,817 optional).
Changing the contract requires a reviewed edit with an entry in its `history`
and a regenerated lock:

```sh
python demos/integrations/computesharp/coverage_contract.py freeze --discovery /path/to/discovery
python demos/integrations/computesharp/coverage_contract.py check --discovery /path/to/discovery
```

Shader exports include the source hash, thread-group dimensions, constant-buffer
size, static-sampler requirement and resource descriptor ranges. They do not
contain live per-dispatch constant bytes or resource contents; those come from
the host adapter at dispatch time. Missing dependencies, empty data sources,
unsupported arguments and incomplete descriptor exports fail without publishing
a partial inventory.

## Original Baseline

The `Project Demo Testing` workflow runs the pinned, unpatched core test project
on Windows and verifies that every discovered case has a corresponding result.
The original tests require WARP; they may report unavailable discrete hardware as
inconclusive. Results are accounted against the frozen contract: Discrete rows
are optional, capability-gated rows must report their capability check, all
other rows must pass, and ignored, missing, extra or unknown results fail
accounting. These original Windows results do not establish Metal or OpenGL
parity.

With the .NET SDK from `upstream.json`, export a successfully built test assembly:

```sh
dotnet run --project demos/integrations/computesharp/discovery/Discovery.csproj \
  -c Release -- /path/to/ComputeSharp.Tests.dll /path/to/new-discovery
python demos/integrations/computesharp/baseline.py \
  --discovery /path/to/new-discovery --trx /path/to/baseline.trx \
  --output /path/to/new-baseline.json
```

Run the tooling contract tests:

```sh
CROSSTL_REQUIRE_DOTNET=1 python -m pytest -q -n auto \
  demos/integrations/computesharp/tests
```

## Non-Windows Builds

The upstream build cannot produce the test assembly on macOS or Linux without
four small source changes, kept as reviewable patches:

1. `0001-classify-projects-with-posix-paths.patch`: upstream classifies projects
   by comparing a relative path against `src\` and `tests\`. MSBuild produces `/`
   separators on POSIX hosts, so generators were built for the wrong framework.
   The patch adds the equivalent `/` comparisons.
2. `0002-skip-dxc-bytecode-off-windows.patch`: the source generator compiles the
   generated HLSL to DXIL with embedded Windows DXC libraries, which cannot load
   elsewhere. Off Windows it now leaves the precompiled bytecode empty through
   the generator's existing "missing bytecode" path. HLSL text, thread-group
   sizes, constant-buffer layout and resource descriptors are generated by the
   unchanged code. DXIL and double-precision diagnostics remain covered by the
   Windows baseline.
3. `0003-import-system-in-resource-loader.patch`: generated resource accessors
   name a `float` pixel type as `Single` and previously compiled only because the
   bytecode helper added `using System;` to the same generated namespace. With no
   bytecode helper the generated code failed to compile. The patch imports
   `System` for the resource loader itself.
4. `0004-match-include-path-case.patch`: the shared source-generation project
   includes two files as `extensions\` and `helpers\` while the directories
   are `Extensions` and `Helpers`. Case-sensitive Linux file systems cannot find
   them; the patch uses the actual directory names.

```sh
python demos/integrations/computesharp/prepare_upstream.py \
  --source /path/to/pristine/ComputeSharp --output /path/to/adapted \
  --receipt /path/to/adapted-source.json
dotnet build /path/to/adapted/tests/ComputeSharp.Tests/ComputeSharp.Tests.csproj \
  -c Release -p:CrossTLTrustedCorpusBaseline=true \
  -p:CustomAfterMicrosoftCommonTargets=$PWD/demos/integrations/computesharp/baseline-audit.props
```

The workflow's parity job builds the adapted tree on Linux, exports it and runs
`compare_discovery.py` against the Windows export: the case inventory and every
generated HLSL source and descriptor must be identical. This is what allows the
translated targets to consume the HLSL that the original baseline executes.

## Pinned Dependency Exceptions

The unmodified upstream pin uses ImageSharp 3.1.11 and treats NuGet advisory
warnings as errors. Five current advisories prevent restoring that historical
baseline: [ICC profile allocation](https://github.com/advisories/GHSA-gwg2-r3hj-4w44),
[histogram equalization](https://github.com/advisories/GHSA-j3p4-wp97-rph4),
[TIFF T4 encoding](https://github.com/advisories/GHSA-j9gm-c75j-xc9q),
[TIFF T6 encoding](https://github.com/advisories/GHSA-jjfr-hcj7-qf5w), and
[BigTIFF decoding](https://github.com/advisories/GHSA-wmxv-xphr-5c9g).
The identified fixed release, 4.1.2, requires a separate license key to build and
is not substituted into the original baseline.

The baseline preserves the pinned dependency and uses explicit advisory-specific
exceptions only for `ComputeSharp.Tests`, enabled by
`CrossTLTrustedCorpusBaseline=true`. NuGet auditing remains enabled; other
advisories and build warnings retain upstream treatment. This is restricted to
the immutable, trusted upstream test code and its tracked assets, whose hashes
are recorded. It must not be used to process external images or as a dependency
policy for a shipped application. The exceptions do not repair the dependency;
a dependency update remains separate work.

## Status

- Done: inventory export with source locations, the frozen coverage contract,
  contract-based result accounting, a complete original WARP run matching it,
  the non-Windows patch series and an adapted macOS build of the test assembly.
- Pending: the live Metal and OpenGL host adapter, CrossTL fixes found by
  executing the translated shaders, and CI evidence of native execution.

ComputeSharp source is MIT-licensed. Upstream dependencies and image assets retain
their own terms; preserve its `LICENSE` and `ThirdPartyNotices.txt`. The Windows
UI, COM and packaging suites remain separate from the compute portability scope.

# Conformance toolchains

Recorded on 2026-10-18 with `python -m tests.conformance.toolchains --markdown` on the `darwin-arm64` scoreboard
host: macOS (Darwin 27.2.0) on Apple M2 Max, Python 3.12.15, Vulkan SDK 1.4.363.0. Re-run the command and update
this file in the same change whenever a validator, driver or container pin moves.

## Validators

| Tool | Version | Status |
|---|---|---|
| `metal` | Apple metal version 32023.918 (metalfe-32023.918.1); macOS SDK 27.0 | available |
| `dxc` | libdxcompiler.dylib: 1.9(5512-01b62ad4)(1.9.0.5512) | available |
| `glslangValidator` | Glslang 11:16.6.0; ESSL 3.20; GLSL 4.60 | available |
| `spirv-as` | SPIRV-Tools v2026.4 | available |
| `spirv-val` | SPIRV-Tools v2026.4 | available |
| `docker` | server 29.5.2 | available |
| `vkrun` | built from `vkrun.c` (dfd360e2b4d00a13) | available |
| `mesa-image` | Mesa 25.2.8-0ubuntu0.24.04.4 aarch64 (`docker/mesa/Dockerfile`) | available |

## Runtimes

| Runtime | Target | Mode | Identity | Status |
|---|---|---|---|---|
| `metal-apple` | metal | host | Apple M2 Max | available |
| `vulkan-moltenvk` | spirv | host | MoltenVK 1.4.2 (Vulkan 1.3.357) on Apple M2 Max | available |
| `vulkan-kosmickrisp` | spirv | host | KosmicKrisp vulkan-sdk-1.4.363 (Vulkan 1.4.363) on Apple M2 Max | available |
| `vulkan-lavapipe` | spirv | container | lavapipe, Mesa 25.2.8 (LLVM 20.1.2, Vulkan 1.4.318) | available |
| `opengl-llvmpipe` | glsl | container | llvmpipe, Mesa 25.2.8 (LLVM 20.1.2), OpenGL 4.5 core | available |
| `d3d12-warp` | hlsl | - | - | unavailable: Direct3D 12 WARP requires Windows |

HLSL cells therefore stop at DXC validation (`compile-only@darwin`) on this host. D3D12 execution needs the
Windows CI path.

## Language versions accepted by these validators

Probed with a trivial compute kernel per version.

- **MSL** (`xcrun metal -std=`): `macos-metal1.0`–`macos-metal2.4`, `ios-metal1.0`–`ios-metal2.4`,
  `metal3.0`, `metal3.1`, `metal3.2`, `metal4.0`, `metal4.1`. `metal3.3` and `metal4.2` are rejected.
- **HLSL** (DXC 1.9.0.5512): profiles `cs_6_0`–`cs_6_9`; `cs_6_10` is rejected. `cs_5_0` and `cs_5_1` are
  accepted and compiled to DXIL, so SM 5.x is a source profile, not a DXBC target. `-HV 2018` and `-HV 202x` are
  accepted; the harness validates at `cs_6_6` with HLSL 2021.
- **GLSL** (glslang 16.6.0): desktop up to 4.60, ES up to 3.20.
- **SPIR-V** (`spirv-val --target-env`): `vulkan1.0`–`vulkan1.4`; the harness runs `vulkan1.1`–`vulkan1.3`.

## Running the harness

```bash
# Unit tests of the harness itself (no native tools needed).
python -m pytest -q tests/conformance
# One source fixture per available runtime, executed unchanged against its oracle.
CROSSTL_CONFORMANCE_NATIVE=1 python -m pytest -q tests/conformance/test_native_runtimes.py
# Full translated matrix under both compile configurations, checked against the scoreboard.
python -m tests.conformance.runner --output /tmp/conformance-run --configs strict,relaxed \
  --baseline tests/conformance/scoreboards/darwin-arm64.json
```

## CI tiers

`.github/workflows/conformance.yml` runs when translator, backend or harness files change, on every push to
`main`, and nightly. Each job checks its results against `scoreboards/ci-<system>-<machine>.json`; a missing
scoreboard, a missing cell, an unavailable required runtime or a worse outcome fails the job. Every run uploads
its `results.jsonl`, `run.json`, toolchain table and a candidate scoreboard (the committed one merged with this
run) as the `conformance-<profile>` artifact; commit that candidate to record a new profile or an improvement.

| Job | Runner | Validators | Runtimes |
|---|---|---|---|
| Linux | `ubuntu-24.04` | DXC v1.9.2602.24 (pinned archive), glslang 16.6.0 (release archive), SPIRV-Tools v2026.4 (built at `ef96ed76` with SPIRV-Headers `49654312`) | `vulkan-lavapipe`, `opengl-llvmpipe` in the Mesa container |
| macOS | `xcode-27` | Xcode 27.1 `metal` | `metal-apple` |
| Windows | `windows-2022` | DXC v1.9.2602.24 (pinned archive) | `d3d12-warp` |

Pull requests run the fixture cases under `strict` and `relaxed`; the nightly tier, `workflow_dispatch` and pull
requests labelled `conformance-full` add six seeded held-out kernels per family and language.

## Corpus tier

`tests/conformance/corpus/` measures translation of third-party programs. `manifest.json` pins seven open-source
projects (repository, commit, licence) and 1,667 translation units in ten slices: GLSL, HLSL and MSL sources, plus
SPIR-V derived from the GLSL slice with glslang 16.6.0 and spirv-dis v2026.4. Each unit carries its SHA-256,
stages, include directories and defines, and is kept only when its project's own toolchain accepts it. The runner
fetches each project as a shallow sparse checkout of exactly the listed files and refuses to measure a corpus
whose files or derived modules differ from the manifest.

```bash
python -m tests.conformance.corpus.run --output /tmp/corpus-run --targets metal,hlsl,glsl,essl,spirv
```

Every (unit, target) cell is translated in a persistent worker (only `crosstl.translate` is timed), checked by
the target's native checker, and classified as usable, degraded (accepted but with a fallback marker or no entry
point), refused (typed diagnostic), invalid, translator error, timeout or worker crash. `summary.json` reports the
usable share per (source, target) pair over feasible cells and the p50/p95/max latency per source language.
`scoreboards/<profile>.json` stores one outcome code per cell; a cell that gets worse, cannot be checked, or a p95
above `budgets.json` fails the run. Latency is wall time, and the worker's CPU time is recorded beside it, because
a loaded runner inflates the first far more than the second. Translations are journalled in
`translations.jsonl` as they finish, so `--resume` continues an interrupted run. `.github/workflows/corpus.yml`
runs the tier nightly, on demand and on pull requests labelled `corpus`: Linux checks HLSL, GLSL, GLSL ES and
SPIR-V outputs, macOS checks MSL outputs.

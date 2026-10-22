# ADR 0003: Measure backend coverage in CI and retire the hand-maintained support matrix

- Status: accepted
- Date: 2026-10-19
- Scope: `support/`, the support and CI bookkeeping scripts in `tools/`, their tests and workflows, the new
  `tools/coverage/` package, the README coverage figure and the issues that track backend gaps

## Context

The support matrix is maintained by hand. `support/features.json` (2.4 MB) and `support/backends.json` list what
each backend is said to support, and `tools/support_matrix.py` regenerates `support/generated/support-matrix.json`
and `docs/source/support-matrix.rst` from them. A contributor who adds a feature has to edit the catalog and rerun
the generator, and CI fails until they do. Nothing in that loop runs the backend's own validator, so a row can say
"supported" for output that `glslang`, DXC or `spirv-val` would reject.

Around the matrix sits about 600 KB of bookkeeping: `tools/ci_coverage.py` (145 KB), `tools/sync_support_issues.py`
(127 KB), `tools/support_ci_summary.py` (124 KB), `tools/support_signals.py` (78 KB), `tools/support_matrix.py`
(61 KB) and `tools/sync_pr_issue_links.py` (59 KB), with roughly 690 KB of tests. `support-issue-sync.yml` ran
hourly and opened 39 `[Support Signals]` and `[Support Matrix]` issues that nobody works from. The matrix does not
say how far each backend is from complete, which is the question users and contributors ask.

## Decision

### 1. A backend is a language and a direction

Output is CrossGL to the target through `crosstl.translate(..., backend=...)`. Input is native source to CrossGL
through the source importers. The first languages are GLSL, HLSL, Metal, SPIR-V and WGSL; CUDA, HIP, Slang, Rust,
Mojo and OpenCL C input follow. A direction CrossTL does not offer, such as WGSL input, is reported as "not offered" rather than 0 %.

### 2. Feature lists come from pinned upstream sources

| Language | Source, pinned by commit and SHA-256 in `tools/coverage/upstream.py` |
|---|---|
| SPIR-V | KhronosGroup/SPIRV-Headers: `extinst.glsl.std.450.grammar.json`; the core grammar's shader subset follows |
| HLSL | microsoft/DirectXShaderCompiler: `utils/hct/gen_intrin_main.txt` |
| GLSL | KhronosGroup/GLSL: `chapters/builtinfunctions.adoc` of the GLSL 4.60 specification |
| WGSL | google/dawn: Tint's `src/tint/lang/wgsl/wgsl.def` |
| Metal | The standard-library headers of Apple's Metal toolchain, read once by a script in the repository into `metal.txt`, which records the toolchain and language version |
| CUDA | llvm/llvm-project: clang's `__clang_cuda_math.h` and `__clang_cuda_device_functions.h` |
| HIP | llvm/llvm-project: clang's `__clang_hip_math.h`; ROCm/clr: `amd_hip_atomic.h` and `amd_device_functions.h`, at the versions of the hipcc that validates the probes |
| Slang | shader-slang/slang: the core module sources `hlsl.meta.slang` and `core.meta.slang` of the slangc release that validates the probes; the global functions of their public sections |
| Mojo | modular/modular: the `std.math` and `std.bit` modules of the Mojo release that validates the probes, limited to the functions their `__init__.mojo` files export |
| OpenCL C | llvm/llvm-project: clang's `opencl-c.h`, at the release of the clang that validates the probes; vendor extensions are left out |
| Rust | Rust-GPU/rust-gpu: the `spirv_std::arch` modules of the release that validates the probes; the `num_traits::Float` trait and glam's `Vec3` methods, at the versions its lock file resolves |

Apple publishes the Metal specification only as a PDF, so Metal is the one list kept in the repository; it changes
only when the script is rerun for a newer toolchain. Language-level categories are added on top of built-ins: types,
control flow, resources, compute (shared memory, barriers, atomics, subgroups) and stages. CI fetches the sources at
their pinned commits, checks their digests and caches them. No feature list is written by hand.

### 3. Each feature is checked by the target's reference validator

For every feature the engine generates a minimal probe, in CrossGL for output and in the source language for input.
SPIR-V is compiler output rather than a language people write, so its input probe is the module glslang compiles the
feature's GLSL probe to.
It translates the probe and validates the result with `glslangValidator` (GLSL), DXC (HLSL), `spirv-val`
(SPIR-V), naga or Tint (WGSL), `xcrun metal` on a macOS runner (Metal), nvcc (CUDA), hipcc (HIP), slangc
compiling to SPIR-V (Slang), the Mojo compiler with MAX's GPU package (Mojo), rustc with the rust-gpu
backend (Rust) or clang for its SPIR target (OpenCL C); none of them needs a GPU to compile. Input probes are
validated by emitting the imported CrossGL back to the source language. CrossTL has no OpenCL C generator, so an
OpenCL C probe is emitted as HLSL, the target whose output row covers the most built-ins, and checked by DXC. Where Core's conformance runtimes can execute a probe, it is
executed through a hook Core provides; the runtime harness is not duplicated.

Each probe ends in one outcome: `supported` (translates, validates and, when executed, matches), `refused` (CrossTL
raises a typed diagnostic), `wrong` (invalid or divergent output) or `untested` (no validator available). The score
for a backend and direction is `supported / (supported + refused + wrong)`; `untested` is reported separately.

`python -m tools.coverage run [--backend B] [--direction input|output]` runs the probes locally and writes
`coverage.json`. A language whose reference tool is missing comes out untested; `python -m tools.coverage tools`
shows which tools a local run will find.

### 4. The CrossGL IR is measured as its own row

CrossGL has no compatibility guarantee: its goal is to express everything any backend can, so that every backend can
reach 100 %. The "CrossGL IR" row measures that goal. Its universe is the union of every backend's features,
normalized to semantic features: a feature's id is the set of SPIR-V instructions its native probe lowers to when the
reference compiler builds it (DXC `-spirv`, glslang, naga, slangc, rust-gpu), minus a baseline probe that reads the
same arguments without the call. HLSL `mad`, GLSL `fma` and SPIR-V `Fma` therefore count once. Metal, CUDA, HIP and
Mojo have no SPIR-V compiler here, and glam writes its vector methods in plain Rust arithmetic, so those features take
the id of the GLSL function CrossGL spells the same way. A feature SPIR-V cannot express is listed by hand in
`tools/coverage/semantic.py` only when it is known to equal another; otherwise it stays a semantic feature of its own.

A semantic feature is expressible when a CrossGL probe for it parses, type-checks and translates correctly to at least
one backend that has it. CrossGL probes come from a small template layer, `tools/coverage/crossgl.py`, and the
spelling tables in `tools/coverage/spellings.py`, which Core updates in the same pull request as any IR change; no
CrossGL probe text is pinned. The row's gaps go to one tracking issue labelled `backend:crossgl` and `area:frontend`,
which Core owns.

### 5. Generated data stays out of `main`

A `coverage` workflow runs on every push to `main` and nightly. It publishes `coverage.json`, `coverage.svg`,
`COVERAGE.md`, a bounded `history.csv` and `trend.svg`, a chart of each row's score over the runs in that history,
to an orphan `coverage` branch as a single replaced commit. Only that job has `contents: write`. The README embeds
`coverage.svg` from the `coverage` branch and `COVERAGE.md` embeds the trend. Both charts are plain SVG written by
small scripts with no plotting dependency. Contributors never run a regeneration step.

### 6. Issues are reconciled from `coverage.json`

`tools/coverage/issues.py` keeps one tracking issue per backend and direction, titled for example "HLSL output
coverage: 87 % (23 gaps)", with a checklist of gaps grouped by category. It edits the issue in place, closes it at
100 % and reopens it on regression. Each `wrong` probe gets a bug issue, at most ten new ones per run, closed when the
probe passes. Every managed issue carries `<!-- crosstl-coverage key=... -->`, and the reconciler never touches an
issue without that marker. Pull requests run it with `--dry-run`; `main` and the nightly run apply it. Issues are
closed, never deleted.

## What is removed and what depends on it

| Item | Depends on it today | Plan |
|---|---|---|
| `support-issue-sync.yml` | none; opened the 39 bot issues | Disabled on 2026-10-19; deleted with its tools. Bot issues closed with a pointer here |
| `support-matrix.yml` (`Support Matrix result`, a required check) | merge-queue ruleset | Replaced by the coverage workflow's result check in the same change that updates the ruleset |
| `pr-issue-links.yml`, `tools/sync_pr_issue_links.py` | `tests/test_pr_issue_links.py` | Removed; GitHub's closing keywords cover the need |
| `tools/support_matrix.py`, `support_signals.py`, `support_ci_summary.py`, `sync_support_issues.py` | their own tests, `.pre-commit-config.yaml` hooks, `docs/source/support-matrix.rst` | Removed with their tests and hooks; the docs page is replaced by `COVERAGE.md` |
| `tools/ci_coverage.py` | `tests/test_ci_workflows.py`; 47 MLX test files use its workflow-text helpers (`workflow_step_section`, `workflow_job_step_section`, `workflow_step_after`, `workflow_event_path_filters`, `workflow_job_text` and four more) | The helpers, about 120 lines, move to a small module; MLX imports change mechanically. The rest is removed |
| `tests/test_ci_workflows.py` (170 KB) | MLX `test_ci.py` imports `RUNNER_OSES`, `_load_ci_coverage_module`, `_matrix_values`, `_workflow_event_paths`, `_workflow_job_section`, `_workflow_texts` | Cut to guards for pinned actions, one result job per gating workflow, `merge_group` triggers, timeouts and permissions; the shared helpers stay importable |
| `.github/ci-coverage-migrations.json` | `support-issue-sync.yml`, `tests/test_ci_workflows.py`, `.github/TESTING.md` | Removed |
| `support/backends.json` | `tests/test_backend/test_package_contracts.py`, `tests/test_translator/test_project_translation.py` (Core) | Core is asked to read the backend list from `crosstl.translator.source_registry` and the codegen registry instead |
| `support/features.json` | `tests/test_translator/test_project_translation.py` (Core), MLX `test_ci.py` path list | Removed after Core and MLX drop the references |
| `support/external-corpus.json` | project pipeline tests (Core) and `docs/source/project-porting.rst` | Moved next to the project tests it configures, with Core's agreement |
| `support/demo-ci-metadata.json`, `tools/demo_ci_metadata.py` | `demo.yml`, `tests/test_demo_open_source_porting.py` | Removed; `demo.yml` names the three tests it selected |
| `support/generated/` | used as a scratch directory for JUnit XML by the test workflows and `demo-project-testing.yml` (MLX) | Scratch output moves to an ignored directory; the three committed JSON files are deleted |
| `further-work-backlog.json` | nothing | Deleted |
| `deploy.sh` | nothing; `release.yml` builds and publishes | Deleted; `.github/RELEASING.md` keeps the commands |
| `setup.cfg` | holds the pytest configuration and a flake8 section | pytest configuration moves to `pyproject.toml`; the file is deleted |

`tools/pytest_failure_summary.py` stays: the test workflows use it to write step summaries. Its references to the
removed tools go. The root `conftest.py` stays: it provides toolchain discovery shared by `tests/` and
`demos/integrations/`.

## Order of work

1. Build the coverage engine for GLSL, HLSL, SPIR-V and WGSL output and GLSL, HLSL and Metal input, then Metal
   output on a macOS job.
2. Publish to the `coverage` branch and embed the chart in the README.
3. Reconcile tracking and bug issues; delete the old issue automation.
4. Delete `support/` and the old tools, slim the guard tests and swap `Support Matrix result` for the coverage
   result check in the ruleset.

## Consequences

Coverage numbers become claims a reader can check: each one names the validator, the upstream commit and the probe.
Early scores will be lower than the matrix implies, because the matrix counted features as supported without
validating them. Probes cover one feature at a time, so they do not replace Core's conformance and differential
tests; they show breadth, and those tests show depth. The coverage job needs the reference validators installed on
its runners, which adds a few minutes to the push-to-`main` pipeline but nothing to pull requests beyond a dry run.

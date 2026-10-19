# ADR 0002: Measure backend coverage in CI and retire the hand-maintained support matrix

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
through the source importers. The first languages are GLSL, HLSL, Metal, SPIR-V and WGSL; CUDA, HIP, Slang, Rust and
Mojo follow. A direction CrossTL does not offer, such as WGSL input, is reported as "not offered" rather than 0 %.

### 2. Feature lists come from pinned upstream sources

| Language | Source, pinned by commit |
|---|---|
| SPIR-V | KhronosGroup/SPIRV-Headers: the unified core grammar and `extinst.glsl.std.450.grammar.json`, shader subset |
| HLSL | microsoft/DirectXShaderCompiler: `utils/hct/gen_intrin_main.txt` |
| GLSL | KhronosGroup/glslang: built-in declarations in `glslang/MachineIndependent/Initialize.cpp` |
| WGSL | gpuweb/gpuweb: the built-in function index of the WGSL specification |
| Metal | The MSL specification's standard-library tables, extracted once by a script kept in the repository, with the specification version recorded |

Language-level categories are added on top of built-ins: types, control flow, resources, compute (shared memory,
barriers, atomics, subgroups) and stages. CI fetches the sources at their pinned commits and caches them. No feature
list is committed by hand.

### 3. Each feature is checked by the target's reference validator

For every feature the engine generates a minimal probe, in CrossGL for output and in the source language for input.
It translates the probe and validates the result with `glslangValidator` (GLSL), DXC (HLSL), `spirv-val`
(SPIR-V), naga or Tint (WGSL), or `xcrun metal` on a macOS runner (Metal). Input probes are validated by emitting
the imported CrossGL back to the source language. Where Core's conformance runtimes can execute a probe, it is
executed through a hook Core provides; the runtime harness is not duplicated.

Each probe ends in one outcome: `supported` (translates, validates and, when executed, matches), `refused` (CrossTL
raises a typed diagnostic), `wrong` (invalid or divergent output) or `untested` (no validator available). The score
for a backend and direction is `supported / (supported + refused + wrong)`; `untested` is reported separately.

`python -m tools.coverage run [--backend B] [--direction input|output]` runs the probes locally and writes
`coverage.json`.

### 4. Generated data stays out of `main`

A `coverage` workflow runs on every push to `main` and nightly. It publishes `coverage.json`, `coverage.svg`,
`COVERAGE.md` and a bounded `history.csv` to an orphan `coverage` branch as a single replaced commit. Only that job
has `contents: write`. The README embeds the SVG from the `coverage` branch. The chart is plain SVG written by a
small script with no plotting dependency. Contributors never run a regeneration step.

### 5. Issues are reconciled from `coverage.json`

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
| `support/demo-ci-metadata.json`, `tools/demo_ci_metadata.py` | `demo.yml`, `tests/test_demo_open_source_porting.py` | Moved under `demos/open-source-porting/` |
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

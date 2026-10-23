# Testing and CI

CI separates portable Python tests from checks that need a native toolchain or
GPU runtime. Every pull request gets Linux, macOS and Windows results, so a
contributor without a Mac or a Windows machine can still land a change that
touches Metal or Direct3D.

## Tiers

- **Pull request tier.** Runs on every push.
  - Translator and backend tests for every component on Python 3.11 and
    3.15 on Ubuntu, plus one job each on macOS and Windows that runs every
    component on Python 3.13.
  - The complete core suite.
  - Shader validators and runtime parity on all three platforms.
  - Native Direct3D and Metal smoke checks.
  - The example matrix.
  - The MLX translation and numerical smoke job.
  - Coverage measurement, with a dry run of the issue changes it would make.
  - Path-filtered demo workflows for the files they cover.
- **Merge queue tier.** The `merge_group` event runs each gating workflow once
  more on the exact commit that will land on `main`. That run covers every
  supported Python version (3.11 through 3.15) and the MLX smoke job. The full
  MLX native, parity, corpus and host tier runs in the queue only when the
  group changes `demos/integrations/mlx/**` or the MLX workflow, as for pushes
  to `main`. A pull request merges only when this run passes.
- **Main and nightly.** Pushes to `main` repeat the full Python sweep and
  publish coverage. The MLX tier, conformance, the corpus and coverage also run
  nightly, and fuzzing runs weekly.

A maintainer can request the merge-queue tier on a pull request before
queueing it. The `ci:full` label applies from the next push, and
`gh workflow run demo-project-testing.yml --ref <branch>` starts one full MLX
run at once.

CrossTL supports every released Python version that has not reached end of
life. When a version reaches end of life, raise `requires-python` in
`pyproject.toml` and remove it from the workflow matrices. Add a new version
once it is released.

## Required checks

Each gating workflow ends in a single `result` job that fails when any job it
waits for fails or is cancelled. Jobs the selected tier skips do not fail it.
The merge queue requires:

- `Translator Tests result`
- `Backend Tests result`
- `Complete Test Suite result`
- `Examples Test result`
- `Coverage result`
- `Lint result`
- `MLX validation result`

A gating workflow runs on every pull request and uses a planning job, not a
`paths` filter, to skip work, so its `result` job always reports.

## Workflows

| Workflow | Checks |
|---|---|
| `translator-tests.yml` | Parser, type checker and code generators; the general tests on Linux and every test on Windows run in three shards each |
| `backend-tests.yml` | Importers for each source language |
| `full-tests.yml` | The whole suite, split by test id into six shards (`tools/pytest_shard.py`); shader validators; runtime parity with Mesa OpenGL and Vulkan on Linux, Metal on macOS and Direct3D on Windows; native compiler smoke tests |
| `examples-test.yml` | Every example translated to each backend its category supports (`examples/test.py`); a backend may refuse only the pairs listed in its `EXPECTED_REJECTIONS` |
| `coverage.yml` | Backend coverage, published to the [`coverage` branch](https://github.com/CrossGL/crosstl/blob/coverage/COVERAGE.md) |
| `lint.yml` | The pinned black hook on the Python files a change touches, which pre-commit.ci skips |
| `demo-project-testing.yml` | The MLX port: smoke tier on pull requests, full native, parity, corpus and host tier on `main` and nightly |
| `computesharp.yml` | The ComputeSharp port, on Metal and OpenGL; its plan job skips the port's jobs when a change cannot affect it, and it ends in one `ComputeSharp validation result` check |
| `conformance.yml` | Differential execution of `tests/conformance`; Linux on pull requests, all three platforms on `main` and nightly |
| `corpus.yml` | The pinned third-party corpus in `tests/conformance/corpus` |
| `fuzz.yml` | Weekly crash fuzzing of every importer |
| `demo.yml`, `native-host-loader.yml`, `deferred-native-compilation.yml` | The open-source porting demo and the native host contracts, for changes to their paths |
| `docs.yml` | The Sphinx documentation build |
| `release.yml` | Distribution builds for packaging changes, and publishing from a release tag (see [RELEASING.md](RELEASING.md)) |
| `labels.yml` | Syncs the labels in `.github/labels.yml` |
| `stale-prs.yml` | Daily: marks pull requests stale after 14 days without activity and closes them 7 days later; issues are left alone |

The workflow files and their header comments describe each job's steps,
toolchain pins and timeouts.

## Platforms

| Workload | Ubuntu | Windows | macOS |
| --- | --- | --- | --- |
| Backend, translator and example tests | Python 3.11 through 3.15 (pull requests: 3.11 and 3.15) | Python 3.13 | Python 3.13 |
| Complete core and demo unit suite | Required | Covered by focused platform jobs | Covered by focused platform jobs |
| Complete DirectX demo corpus compilation | Pinned Linux DXC | Native execution | Not required |
| Direct3D execution and host integration | Not applicable | Required WARP and dispatch checks | Not applicable |
| OpenGL execution and host integration | Required Mesa/EGL checks | Platform-specific controls | Platform-specific controls |
| Metal compilation, reference and host integration | Not available | Not available | Required native checks |

Python-version compatibility is tested fully on Ubuntu only; Windows and macOS
run every backend and translator test once. The macOS translator, backend and
portable-host jobs use the `xcode-27` image with Xcode 27.1, because
acquire/release atomic fences need Metal 4.1.

## Runners

Public-repository runners have 4 vCPUs on Linux and Windows and 3 on macOS,
and the test jobs size `pytest-xdist` to match. Jobs that install with pip
cache their downloads. Every job has a timeout, and every workflow declares
least-privilege permissions.

A new push cancels the superseded pull-request runs of the test workflows.
Pushes to `main` queue behind the run already in progress, and a newer push
replaces the one still waiting, so a burst of merges runs the sweep once for
the newest commit; the merge queue has already tested each of them. The
MLX workflow lets each leg finish, so a long Metal run does not hold up the
next revision's DirectX or OpenGL legs; cancel superseded MLX runs by hand.
Suites that take many minutes on macOS or Windows belong in the merge-queue,
nightly or dispatch tier rather than on every push.

## Running checks locally

```bash
pytest tests/test_translator tests/test_backend -n auto -q   # what most changes need
pytest -n auto -q                                            # the full suite
pre-commit run --all-files                                   # formatting, actionlint, workflow guards
python -m tools.coverage run --backend glsl                  # coverage for one backend
```

Tests that need glslang, DXC, SPIRV-Tools or the Metal toolchain are skipped
when the tool is not installed, and coverage reports a backend whose validator
is missing as untested. Metal checks need macOS with Xcode.
DXC and Mesa failures can be reproduced in Docker on Linux.

`tests/test_ci_workflows.py` checks that actions are pinned to release tags,
that each gating workflow ends in one `result` job and runs in the merge
queue, that every job has a timeout and that permissions stay least-privilege.
A change to the tiers updates this file in the same pull request.

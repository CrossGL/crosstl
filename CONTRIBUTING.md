# Contributing to CrossTL

Thank you for helping. This guide covers setting up a checkout, running the
tests, and getting a pull request merged. Please follow the
[code of conduct](.github/CODE_OF_CONDUCT.md).

## Setup

CrossTL needs Python 3.11 or later.

```bash
git clone https://github.com/CrossGL/crosstl.git
cd crosstl
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]" pre-commit
pre-commit install
```

`pre-commit` runs black, isort, ruff and the repository checks on the files
you commit. Run `pre-commit run --all-files` before your first push if you
installed it late. The Lint check runs the same pinned black on every Python
file a pull request changes.

Native validators are optional. Tests that need glslang, DXC, SPIRV-Tools or
the Metal toolchain are skipped when the tool is missing, and CI runs them on
Linux, Windows and macOS.

## Tests

Run the tests for the code you changed first, then the wider suite:

```bash
pytest tests/test_translator/test_parser.py -q                 # one file
pytest tests/test_translator/test_parser.py -q -k struct       # matching tests
pytest tests/test_translator tests/test_backend -n auto -q     # translator and backends
pytest tests -n auto -q                                        # everything
```

Translator tests live in `tests/test_translator`, importer tests for each
source language in `tests/test_backend/test_<language>`, and differential
conformance checks in `tests/conformance`. [TESTING.md](.github/TESTING.md)
describes what CI runs on each platform.

## Adding a feature

Most changes touch one importer (`crosstl/backend/<language>`) or one code
generator (`crosstl/translator/codegen`). Add a test next to the existing ones
for that language that translates a small input and checks the output, or the
diagnostic when the input cannot be expressed. Unsupported constructs must
raise a diagnostic rather than produce partial output.

Changes to the CrossGL language itself update the parser, every code generator
and importer that handles the construct, and the examples under `examples/` in
the same pull request.

Coverage is measured, so there is no support table to edit. The Coverage
workflow writes a probe for every built-in in each language's pinned upstream
feature list and for a set of language constructs (types, control flow,
resources, compute and stages), translates it, and checks the result with the
language's reference tool. It reports the coverage change on your pull
request, as a comment or, for a pull request from a fork, in the job summary.
Once the change is on `main`, it closes the `coverage` issues whose probes now
pass. Each of those issues gives the command that reproduces it, for example:

```bash
python -m tools.coverage run --backend hlsl --direction input --feature WaveActiveSum
python -m tools.coverage tools    # which reference tools are installed
```

A language whose reference tool is not installed comes out untested. With
`--execute`, the Metal, GLSL and SPIR-V compute probes also run on a local GPU
or Mesa runtime through the conformance harness, as CI does; a result the
native probe does not compute is wrong.

## Picking an issue

Issues labelled [`good first issue`](https://github.com/CrossGL/crosstl/labels/good%20first%20issue)
are small and well scoped; [`help wanted`](https://github.com/CrossGL/crosstl/labels/help%20wanted)
marks larger ones maintainers would welcome help with. `backend:<language>`
labels say which language an issue concerns, and `area:` labels say which part
of the code. Comment on an issue before starting so work is not duplicated.

## Pull requests

Open the pull request against `main`. Required checks run on every push:
translator, backend, complete-suite, example and demo tests on Linux, macOS and
Windows, plus Lint and Coverage. Once they pass and a maintainer has reviewed
the change, it joins the merge queue, which tests the exact commit that will
land, and is squash-merged. A pull request with no activity for 14 days is
marked `stale` and closed 7 days later; any push or comment resets the clock.

Keep a pull request to one change. Write commit messages and the pull request
title in the imperative, describing the behavior that changes ("Reject
recursive functions in the Metal generator"), with a body that says why.

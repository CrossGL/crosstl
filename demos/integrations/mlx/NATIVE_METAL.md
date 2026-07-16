# Pinned MLX Native Metal Reference Baseline

The `MLX Native Metal Reference Baseline` workflow validates upstream MLX commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e` on the GitHub `macos-26` runner.
It is a native upstream reference only. It does not compile or execute a
CrossTL-translated target and does not establish translated-target correctness,
runtime parity, or numerical parity.

## Evidence Sections

The deterministic `evidence.json` report keeps three results separate.

### Source Compile And Link

The workflow creates a sparse checkout containing only
`mlx/backend/metal/kernels`. Before compilation, the helper verifies the exact
checkout revision, requires a clean kernel tree, discovers the complete recursive
`*.metal` surface, and compares every path and SHA-256 digest with the pinned
40-source manifest.

The helper compiles the 33 standard units with `-std=metal3.2` and the seven
`_nax.metal` units with `-std=metal4.0`. Every invocation uses the MLX warning
and floating-point flags `-Wall`, `-Wextra`, `-fno-fast-math`,
`-Wno-c++17-extensions`, and `-Wno-c++20-extensions`, the MLX repository root
as an include directory, and `-mmacosx-version-min=26.2`. A zero exit status and
a nonempty AIR file are required for every unit.

All 40 AIR objects are linked into one nonempty `mlx.metallib`. `metal-nm`
must then return a nonempty defined-symbol listing. The report records source,
AIR, AIR-input-manifest, metallib, and symbol-list hashes together with byte
counts, profile accounting, commands, exit statuses, and log paths. CI uploads
the JSON report and command logs, but not the AIR build tree or the large linked
metallib, whose size depends on the selected Xcode toolchain.

### Upstream Python Tests

A separate full checkout of the same commit is built with Python 3.13. The
workflow installs `setuptools`, `wheel`, `cmake`, `ninja`,
`typing_extensions`, `numpy`, and `ml_dtypes`, then the helper runs:

```bash
python -m pip install -e . --no-build-isolation
python -m unittest discover python/tests
```

Torch is intentionally not installed because the pinned suite does not require
it. The gate requires a zero exit status, exactly 776 tests with 44 skipped and
zero failures or errors, and `mlx.core.default_device()` selecting a GPU. The
report records the Python and MLX versions, selected device, parsed counts, exact
commands, exit statuses, and logs.

### Upstream C++ Tests

The helper configures an independent Release build with Ninja and these CMake
settings:

```text
MLX_BUILD_TESTS=ON
MLX_BUILD_PYTHON_BINDINGS=OFF
MLX_BUILD_EXAMPLES=OFF
MLX_BUILD_BENCHMARKS=OFF
CMAKE_OSX_DEPLOYMENT_TARGET=26.2
```

It builds the `tests` target and runs the aggregate `tests/tests` doctest
executable once. The gate requires a zero exit status, 260 of 260 cases and
3,490 of 3,490 assertions passed, and zero skipped cases. Running the aggregate
executable avoids duplicating per-case process teardown while retaining every
logical doctest case; this command choice is not a skipped test or coverage
waiver. The CMake configuration, build, executable hash, aggregate command,
exit status, counts, and logs are recorded independently from the Python and
source compile/link evidence.

## Running Locally

Create the sparse source checkout:

```bash
git init /private/tmp/mlx-metal-source
git -C /private/tmp/mlx-metal-source remote add origin https://github.com/ml-explore/mlx.git
git -C /private/tmp/mlx-metal-source sparse-checkout init --no-cone
git -C /private/tmp/mlx-metal-source sparse-checkout set /mlx/backend/metal/kernels/
git -C /private/tmp/mlx-metal-source fetch --depth=1 --filter=blob:none origin 4367c73b60541ddd5a266ce4644fd93d20223b6e
git -C /private/tmp/mlx-metal-source checkout --detach FETCH_HEAD
```

Create the full test checkout and Python 3.13 environment:

```bash
git init /private/tmp/mlx-full
git -C /private/tmp/mlx-full remote add origin https://github.com/ml-explore/mlx.git
git -C /private/tmp/mlx-full fetch --depth=1 --filter=blob:none origin 4367c73b60541ddd5a266ce4644fd93d20223b6e
git -C /private/tmp/mlx-full checkout --detach FETCH_HEAD
python3.13 -m venv /private/tmp/mlx-native-venv
/private/tmp/mlx-native-venv/bin/python -m pip install --upgrade pip
/private/tmp/mlx-native-venv/bin/python -m pip install setuptools wheel cmake ninja typing_extensions numpy ml_dtypes
```

Run all three native reference gates:

```bash
/private/tmp/mlx-native-venv/bin/python \
  demos/integrations/mlx/run_mlx_native_metal.py \
  --mlx-root /private/tmp/mlx-metal-source \
  --mlx-python-root /private/tmp/mlx-full \
  --output-dir /private/tmp/mlx-native-metal-evidence \
  --jobs 3
```

The output directory contains `evidence.json`, command logs, AIR files, the
linked metallib, and the C++ build tree. Only `evidence.json` and `logs/` are
intended for CI artifact upload.

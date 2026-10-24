"""Run unchanged upstream op tests on the translated host's package families.

Each test runs unmodified from the pinned ``python/tests`` tree, once on the
MLX CPU and once through the translated host. The native run installs the
base host packages with the families the dedicated CI job translates: random
bits, Boolean and integer bitwise operations, selection, absolute values,
the 64-bit integer family, reduction initialization and on-demand
whole-array and row reductions. General gather and scatter kernels are specialized
from the pinned MLX sources at dispatch time, and each of their dispatches is
recomputed from its retained inputs and native modules. DivMod's
``binary_two`` kernels are specialized the same way, and every DivMod dispatch
is recomputed from its retained operands. Casts and copies that no installed
family covers (the narrow integers and the 16-bit float pairs) translate from
``copy.metal`` on first use; each is recomputed from its retained words and
must name the package its cache derives again. Unary and binary vectors of the
other real dtypes (the 16-bit floats, the narrow integers, Boolean arithmetic
and the 64-bit bitwise operations) translate from ``unary.metal`` and
``binary.metal`` the same way; each must name its cached package and native
modules, and integer and Boolean ones are recomputed from their operands.
Scans and small row and column
reductions are also translated on first use; each of their dispatches must
name the package its cache derives again from the event. Arg reductions are
translated per threadgroup width on first use and every index is recomputed
from the retained input; OpenGL does not run the arg reduction tests yet
(``TARGET_EXCLUSIONS``). Softmax and logsumexp rows are translated per
threadgroup width on first use; each dispatch must name its cached package,
upload its retained input and keep its output guard, and the upstream tests
compare its values. Every other dispatch
must come from one of those indexes, so a test that reaches an uninstalled
family fails the trace instead of passing on an unchecked package.
"""

import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host import (
    arange_evidence,
    arange_packages,
    arg_reduce_evidence,
    arg_reduce_layout,
    copy_evidence,
    divmod_evidence,
    elementwise_evidence,
    gather_evidence,
    lazy_package_evidence,
    random_evidence,
    scatter_axis_evidence,
    scatter_evidence,
    softmax_evidence,
    softmax_layout,
    wide_column_dispatch,
    wide_column_evidence,
    wide_reduction_evidence,
    wide_reduction_packages,
    wide_row_dispatch,
    wide_row_evidence,
)
from demos.integrations.mlx.portable_host.arg_reduce_packages import (
    CACHE_DIRECTORY as ARG_REDUCE_CACHE,
)
from demos.integrations.mlx.portable_host.arg_reduce_packages import (
    ArgReducePackageCache,
)
from demos.integrations.mlx.portable_host.column_small_packages import (
    CACHE_DIRECTORY as SMALL_COLUMN_CACHE,
)
from demos.integrations.mlx.portable_host.column_small_packages import (
    SmallColumnPackageCache,
)
from demos.integrations.mlx.portable_host.copy_packages import CACHE as COPY_CACHE
from demos.integrations.mlx.portable_host.copy_packages import (
    ENTRIES as LAZY_COPY_ENTRIES,
)
from demos.integrations.mlx.portable_host.copy_packages import CopyPackageCache
from demos.integrations.mlx.portable_host.divmod_dispatch import (
    ENTRY_PREFIXES as DIVMOD_PREFIXES,
)
from demos.integrations.mlx.portable_host.elementwise_packages import (
    CACHE as ELEMENTWISE_CACHE,
)
from demos.integrations.mlx.portable_host.elementwise_packages import (
    ENTRIES as LAZY_ELEMENTWISE_ENTRIES,
)
from demos.integrations.mlx.portable_host.elementwise_packages import (
    ElementwisePackageCache,
)
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.random_packages import (
    ENTRIES,
)
from demos.integrations.mlx.portable_host.random_packages import (
    load_index as load_random,
)
from demos.integrations.mlx.portable_host.reduction_packages import (
    load_index as load_reductions,
)
from demos.integrations.mlx.portable_host.reduction_packages import variant_keys
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.small_row_packages import SmallRowPackageCache
from demos.integrations.mlx.portable_host.softmax_packages import (
    CACHE_DIRECTORY as SOFTMAX_CACHE,
)
from demos.integrations.mlx.portable_host.softmax_packages import SoftmaxPackageCache
from demos.integrations.mlx.portable_host.verify import upstream_test_sources
from demos.integrations.mlx.portable_host.verify_bitwise import verify_native_identity
from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts

UPSTREAM_TESTS = (
    tuple(
        "test_ops.TestOps." + name
        for name in (
            "test_trunc",
            "test_erfinv",
            "test_logical_and",
            "test_logical_or",
            "test_full_ones_zeros",
            "test_atleast_1d",
            "test_atleast_2d",
            "test_atleast_3d",
            "test_tile",
            "test_repeat",
            "test_roll",
            "test_roll_errors",
            "test_tri",
            "test_tril",
            "test_triu",
            "test_as_strided",
            "test_view",
            "test_depends",
            "test_eye",
            "test_diag",
            "test_diagonal",
            "test_divmod",
            # Base families only.
            "test_all",
            "test_cummax_cummin_nan",
            "test_empty_matmuls",
            "test_irregular_alignments",
            "test_scan_invalid_axis",
            "test_scan_strided_bool",
            "test_slice_with_negative_stride",
            "test_take_along_axis",
            "test_unary_ops_from_non_array",
            # These also reach the 64-bit integer family.
            "test_add",
            "test_any",
            "test_bitwise_grad",
            "test_count_nonzero",
            "test_irregular_binary_ops",
            "test_linspace",
            "test_linspace_endpoint",
            "test_mean",
            "test_min_and_max",
            "test_prod",
            # This also reaches lazily translated casts.
            "test_subnormal_bool_cast",
            # These reach lazily translated unary and binary vectors.
            "test_binary_ops",
            "test_bitwise_ops",
            "test_divide",
            "test_floor_divide",
            "test_integer_power",
            "test_remainder",
            "test_scalar_inputs",
            "test_sigmoid",
            "test_tensordot",
            "test_unary_ops",
            "test_zeros_ones_empty_like_dtype",
            # These also reach lazily translated complex64 kernels.
            "test_array_equal",
            "test_complex_ops",
            "test_complex_power",
            "test_conjugate",
            "test_diag_zero_size",
            "test_isfinite",
            "test_isinf",
            "test_isnan",
            "test_isneginf",
            "test_isposinf",
            "test_log",
            "test_log10",
            "test_log1p",
            "test_log2",
            "test_logaddexp",
            "test_maximum",
            "test_minimum",
            "test_outer",
            "test_real_imag",
            "test_sign",
            "test_trig_ops",
            # These reach lazily translated selections of every GPU dtype.
            "test_clip",
            "test_concatenate",
            "test_meshgrid",
            "test_nan_to_num",
            "test_pad",
            "test_pad_reflect_symmetric",
            "test_slice_update_reversed",
            "test_take",
            # This also reaches lazily translated aranges.
            "test_round",
            # These reach lazily translated wide whole-array, small-row and
            # small-column reductions.
            "test_std",
            "test_trace",
            "test_var",
            "test_vecdot",
            # This also reaches row_reduce_looped through the on-demand row index.
            "test_sum",
        )
    ) + (
        # These reach lazily translated arg reductions of every real dtype, and
        # of the flattened array in a threadgroup of thirteen simdgroups.
        "test_ops.TestOps.test_argmin_argmax",
        "test_reduce.TestReduce.test_arg_reduce",
        "test_bf16.TestBF16.test_arg_reduction_ops",
    ) + (
        # These reach lazily translated block and looped softmax and logsumexp
        # rows, precise 16-bit softmax, and transposed and broadcast inputs.
        "test_ops.TestOps.test_softmax",
        "test_ops.TestOps.test_logsumexp",
        "test_ops.TestOps.test_logsumexp_shape",
    )
)
# Tests a target cannot run yet, with the issue that tracks each. OpenGL's
# software subgroups cannot run an arg reduction wider than one simdgroup.
TARGET_EXCLUSIONS = {
    "opengl": {
        "test_ops.TestOps.test_argmin_argmax": "CrossGL/crosstl#2207",
        "test_reduce.TestReduce.test_arg_reduce": "CrossGL/crosstl#2207",
        "test_bf16.TestBF16.test_arg_reduction_ops": "CrossGL/crosstl#2207",
    }
}
# Source-specialized indexing kernels by entry prefix, longest prefix first.
INDEXING_PREFIXES = ("gather", "scatter_axis", "scatter")
# Entry-keyed package families; reductions are keyed by width and entry.
FAMILIES = (
    "packages",
    "random",
    "bitwise",
    "selection",
    "absolute",
    "integer64",
)
# Families loaded from a translated index; the random family has its own loader.
INDEXED_FAMILIES = FAMILIES[2:]
REQUIRED_REDUCTIONS = frozenset(("w32/all_reduce_andbool_", "w256/all_reduce_andbool_"))
NEGATIVE_CHECKS = {"missing": "No translated package for vv_Equalint32"}
# Dispatches whose trace records keep their long values for the audits.
RETAINED_VALUES = (
    *ENTRIES,
    *arg_reduce_layout.OPERATIONS,
    *softmax_layout.PREFIXES,
    *INDEXING_PREFIXES,
    *DIVMOD_PREFIXES,
    "all_reduce_",
    "init_reduce_",
)
# The CI step bounds the three workers and the audit at 5,000 seconds; a cold
# native run also translates about a hundred lazy vectors.
TIMEOUTS = {"cpu": 300, "native": 4400, "missing": 60}


def upstream_tests(target):
    """The upstream tests ``target`` runs, in inventory order."""
    excluded = TARGET_EXCLUSIONS.get(target, {})
    return tuple(name for name in UPSTREAM_TESTS if name not in excluded)


def package_target(packages):
    return json.loads((Path(packages) / "index.json").read_text())["target"]


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def install(args, output):
    host = HostRuntime(
        args.packages,
        output / "dispatch.jsonl",
        random=args.random,
        bitwise=args.bitwise,
        selection=args.selection,
        absolute=args.absolute,
        integer64=args.integer64,
        reductions=list(args.reductions),
        mlx_root=args.mlx_root,
        # The audits recompute these dispatches from their values; the other
        # families are checked by artifact and native identity only.
        trace_values=RETAINED_VALUES,
    )
    if args.worker == "missing":
        host.descriptors.pop("vv_Equalint32")
        # Without the pinned source the comparison has no package at all.
        host.elementwise = None
    # Lazy copies and integer or Boolean lazy vectors are recomputed from their
    # values; installed families are not.
    host.trace_values = (
        *RETAINED_VALUES,
        *(entry for entry in LAZY_COPY_ENTRIES if host.lazy_copy(entry)),
        *(
            entry
            for entry in LAZY_ELEMENTWISE_ENTRIES
            if host.lazy_elementwise(entry) and elementwise_evidence.recomputable(entry)
        ),
        *(
            entry
            for entry in arange_packages.ENTRIES
            if host.lazy_arange(entry) and arange_evidence.recomputable(entry)
        ),
    )
    host.install()
    return host


def reject(mx, host, output, check):
    try:
        mx.eval(mx.array_equal(mx.array([1, 2, 3]), mx.array([1, 2, 4])))
    except (ValueError, RuntimeError) as error:
        record = {
            "check": check,
            "error": str(error),
            "dispatchCount": host.dispatch_count,
        }
        write_json(output / "rejection.json", record)
        if NEGATIVE_CHECKS[check] not in str(error) or host.dispatch_count:
            raise RuntimeError(
                "The missing package was not refused before dispatch"
            ) from error
        return
    raise RuntimeError("A comparison without its package was accepted")


def worker(args):
    import mlx.core as mx

    output = args.output_dir
    output.mkdir(parents=True)
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
    host = None
    if args.worker == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        host = install(args, output)
    if args.worker in NEGATIVE_CHECKS:
        return reject(mx, host, output, args.worker)
    os.environ["DEVICE"] = "gpu" if host else "cpu"
    os.environ["CI"] = "1"
    sys.path.insert(0, str(args.mlx_root / "python/tests"))
    tests = upstream_tests(package_target(args.packages))
    records = []
    for name in tests:
        start = host.dispatch_count if host else 0
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromName(name)
        )
        records.append(
            {
                "test": name,
                "testsRun": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skips": len(result.skipped),
                "dispatchCount": host.dispatch_count - start if host else 0,
            }
        )
        write_json(output / "upstream.json", {"tests": records})
    validate_upstream({"tests": records}, native=host is not None, inventory=tests)


def validate_upstream(record, *, native, inventory=UPSTREAM_TESTS):
    tests = record.get("tests", [])
    if [item.get("test") for item in tests] != list(inventory):
        raise ValueError("Upstream op test inventory is incomplete")
    for item in tests:
        expected = {"testsRun": 1, "failures": 0, "errors": 0, "skips": 0}
        count = item.get("dispatchCount")
        if (
            any(item.get(key) != value for key, value in expected.items())
            or type(count) is not int
            or (count <= 0 if native else count != 0)
        ):
            raise ValueError(f"Unchanged upstream test failed: {item.get('test')}")


def load_indexes(args):
    base = json.loads((args.packages / "index.json").read_text())
    target = base["target"]
    indexes = {"packages": base}
    indexes["random"] = {
        "target": target,
        "descriptors": load_random(args.random, target),
    }
    for name in INDEXED_FAMILIES:
        index = json.loads((getattr(args, name) / "index.json").read_text())
        if index.get("target") != target:
            raise ValueError(f"The {name} packages target another backend")
        indexes[name] = index
    reductions = [load_reductions(path, target) for path in args.reductions]
    keys = set().union(*(variant_keys(index) for index in reductions))
    if REQUIRED_REDUCTIONS - keys:
        raise ValueError(
            "Upstream op assertions require reduction packages: "
            + ", ".join(sorted(REQUIRED_REDUCTIONS - keys))
        )
    return target, indexes, reductions


def indexing_auditor(entry):
    """Return the evidence check for a source-specialized indexing entry."""
    auditors = {
        "gather": gather_evidence.audit_event,
        "scatter_axis": scatter_axis_evidence.audit_event,
        "scatter": scatter_evidence.audit_event,
    }
    prefix = next((item for item in INDEXING_PREFIXES if entry.startswith(item)), None)
    return auditors.get(prefix)


def audit_trace(np, args, trace, target, indexes, reductions):
    """Tie every dispatch to exactly one translated package."""
    random_events = [event for event in trace if event["entry"] in ENTRIES]
    for event in random_events:
        random_evidence.audit_event(np, event)
    if random_events:
        random_evidence.verify_packages(
            random_events, args.mlx_root, args.random, target
        )
    owners = [0] * len(trace)
    indexing = 0
    for i, event in enumerate(trace):
        auditor = indexing_auditor(event["entry"])
        if auditor is None:
            continue
        if event["target"] != target:
            raise ValueError("Upstream op indexing dispatch targets another backend")
        # Recomputes the result from the retained inputs and native modules.
        auditor(np, event)
        owners[i] += 1
        indexing += 1
    for i, event in enumerate(trace):
        if not event["entry"].startswith(DIVMOD_PREFIXES):
            continue
        if event["target"] != target:
            raise ValueError("Upstream op DivMod dispatch targets another backend")
        # Recomputes both outputs and checks the retained native modules.
        divmod_evidence.audit_event(event)
        owners[i] += 1
    installed = set().union(*(index["descriptors"] for index in indexes.values()))
    copies = CopyPackageCache(args.mlx_root, args.packages / COPY_CACHE, target)
    lazy_checked = set()
    for i, event in enumerate(trace):
        if event["entry"] not in LAZY_COPY_ENTRIES or event["entry"] in installed:
            continue
        if event["target"] != target:
            raise ValueError("Upstream op copy dispatch targets another backend")
        # Recomputes the destination and derives the cached package again.
        copy_evidence.audit_event(event, copies)
        lazy_checked.add(i)
        owners[i] += 1
    elementwise = ElementwisePackageCache(
        args.mlx_root, args.packages / ELEMENTWISE_CACHE, target
    )
    for i, event in enumerate(trace):
        if (
            event["entry"] not in LAZY_ELEMENTWISE_ENTRIES
            or event["entry"] in installed
        ):
            continue
        if event["target"] != target:
            raise ValueError("Upstream op vector dispatch targets another backend")
        # Derives the cached package again and checks the native modules;
        # integer and Boolean results are recomputed from their operands.
        elementwise_evidence.audit_event(event, elementwise)
        lazy_checked.add(i)
        owners[i] += 1
    aranges = arange_packages.cache(
        args.mlx_root, args.packages / arange_packages.CACHE, target
    )
    for i, event in enumerate(trace):
        if event["entry"] not in arange_packages.ENTRIES or event["entry"] in installed:
            continue
        if event["target"] != target:
            raise ValueError("Upstream op arange dispatch targets another backend")
        # Derives the cached package again and checks the native modules;
        # integer results are recomputed from their scalars.
        arange_evidence.audit_event(event, aranges)
        lazy_checked.add(i)
        owners[i] += 1
    wide = wide_reduction_packages.cache(
        args.mlx_root, args.packages / wide_reduction_packages.CACHE, target
    )
    for i, event in enumerate(trace):
        if event["entry"] not in wide_reduction_packages.ENTRIES:
            continue
        if event["target"] != target:
            raise ValueError("Upstream op wide reduction targets another backend")
        # Derives the cached package again and checks the native modules;
        # integer passes are recomputed from their inputs.
        wide_reduction_evidence.audit_event(event, wide)
        lazy_checked.add(i)
        owners[i] += 1
    wide_rows = SmallRowPackageCache(
        args.mlx_root, args.packages / "small-rows", target
    )
    for i, event in enumerate(trace):
        if event["entry"] not in wide_row_dispatch.ENTRIES:
            continue
        if event["target"] != target:
            raise ValueError("Upstream op wide row reduction targets another backend")
        # Derives the cached packages again and checks the native modules or
        # regions; integer rows are recomputed from their inputs.
        wide_row_evidence.audit_event(
            event, wide_rows, Path(args.output_dir) / "native/native-modules"
        )
        lazy_checked.add(i)
        owners[i] += 1
    wide_columns = SmallColumnPackageCache(
        args.mlx_root, args.packages / SMALL_COLUMN_CACHE, target
    )
    for i, event in enumerate(trace):
        if event["entry"] not in wide_column_dispatch.ENTRIES:
            continue
        if event["target"] != target:
            raise ValueError(
                "Upstream op wide column reduction targets another backend"
            )
        # Derives the cached package again and checks the native modules;
        # integer columns are recomputed from their inputs.
        wide_column_evidence.audit_event(event, wide_columns)
        lazy_checked.add(i)
        owners[i] += 1
    arg_reductions = ArgReducePackageCache(
        args.mlx_root, args.packages / ARG_REDUCE_CACHE, target
    )
    for i, event in enumerate(trace):
        if event["entry"] not in arg_reduce_layout.ENTRIES:
            continue
        if event["target"] != target:
            raise ValueError("Upstream op arg reduction targets another backend")
        # Derives the cached package again, checks the native modules and
        # recomputes every index from the retained input.
        arg_reduce_evidence.audit_event(event, arg_reductions)
        lazy_checked.add(i)
        owners[i] += 1
    softmaxes = SoftmaxPackageCache(
        args.mlx_root, args.packages / SOFTMAX_CACHE, target
    )
    for i, event in enumerate(trace):
        if event["entry"] not in softmax_layout.ENTRIES:
            continue
        if event["target"] != target:
            raise ValueError("Upstream op softmax or logsumexp targets another backend")
        # Derives the cached package again and checks the uploads, guards and
        # native modules; the upstream tests compare the values.
        softmax_evidence.audit_event(event, softmaxes)
        lazy_checked.add(i)
        owners[i] += 1
    lazy = lazy_package_evidence.LazyPackages(args.mlx_root, args.packages, target)
    modules = Path(args.output_dir) / "native/native-modules"
    executed = set()
    for i, event in enumerate(trace):
        if lazy_package_evidence.family(event["entry"]) is None:
            continue
        # Derives the package again and checks scans' retained native modules.
        if lazy_package_evidence.audit_event(event, lazy, modules):
            executed.add(i)
        owners[i] += 1
    verify_native_identity(
        [
            event
            for i, event in enumerate(trace)
            if event["entry"] not in ENTRIES
            and indexing_auditor(event["entry"]) is None
            and not event["entry"].startswith(DIVMOD_PREFIXES)
            and i not in executed
            and i not in lazy_checked
        ],
        target,
    )
    for name in FAMILIES:
        if name == "random":
            selected = [i for i, event in enumerate(trace) if event["entry"] in ENTRIES]
        else:
            descriptors = indexes[name]["descriptors"]
            selected = [
                i for i, event in enumerate(trace) if event["entry"] in descriptors
            ]
            verify_artifacts(
                [trace[i] for i in selected],
                getattr(args, name),
                indexes[name],
                variants=False,
            )
        for i in selected:
            owners[i] += 1
    for path, index in zip(args.reductions, reductions):
        keys = variant_keys(index)
        selected = [
            i
            for i, event in enumerate(trace)
            if f'w{event["workgroupSize"][0]}/{event["entry"]}' in keys
        ]
        verify_artifacts(
            [trace[i] for i in selected],
            path,
            index,
            variants=True,
            mlx_root=args.mlx_root,
        )
        for i in selected:
            owners[i] += 1
    if any(count != 1 for count in owners):
        raise ValueError("Upstream op trace contains an unknown or ambiguous package")
    if not indexing:
        raise ValueError("Upstream indexing tests did not reach gather or scatter")
    return indexing


def run_worker(args, output, mode):
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
        "--timeout-seconds",
        str(TIMEOUTS[mode]),
        "--label",
        f"MLX upstream ops {mode}",
        "--",
        sys.executable,
        "-m",
        __package__ + ".verify_upstream_ops",
        "--worker",
        mode,
    ]
    for name in ("mlx_root", "packages", *FAMILIES[1:]):
        command.extend(
            ("--" + name.replace("_", "-"), str(getattr(args, name).resolve()))
        )
    for path in args.reductions:
        command.extend(("--reductions", str(path.resolve())))
    command.extend(("--output-dir", str(output / mode)))
    with (output / f"{mode}.stdout").open("w") as stdout, (
        output / f"{mode}.stderr"
    ).open("w") as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    record = {"command": command, "returncode": process.returncode}
    write_json(output / f"{mode}.command.json", record)
    return record


def verify(args):
    import numpy as np

    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    before = verify_prepared(args.mlx_root)
    target = package_target(args.packages)
    tests = upstream_tests(target)
    sources = upstream_test_sources(args.mlx_root, tests)
    write_json(output / "adaptation-before.json", before)
    target, indexes, reductions = load_indexes(args)
    evidence = {
        "commit": COMMIT,
        "target": target,
        "passed": False,
        "adaptation": before,
        "upstreamTestSources": sources,
        "upstreamTests": list(tests),
        "excludedUpstreamTests": dict(TARGET_EXCLUSIONS.get(target, {})),
        "workers": {},
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
    }
    try:
        for mode in ("cpu", "native", *NEGATIVE_CHECKS):
            evidence["workers"][mode] = run_worker(args, output, mode)
            write_json(output / "evidence.json", evidence)
        if any(value["returncode"] for value in evidence["workers"].values()):
            raise RuntimeError("An upstream op worker failed; inspect retained logs")
        upstream = {}
        for mode in ("cpu", "native"):
            upstream[mode] = json.loads((output / mode / "upstream.json").read_text())
            validate_upstream(upstream[mode], native=mode == "native", inventory=tests)
        trace = [
            json.loads(line)
            for line in (output / "native/dispatch.jsonl").read_text().splitlines()
        ]
        if sum(item["dispatchCount"] for item in upstream["native"]["tests"]) != len(
            trace
        ):
            raise ValueError("Upstream op trace count differs")
        indexing = audit_trace(np, args, trace, target, indexes, reductions)
        rejections = {}
        for mode, error in NEGATIVE_CHECKS.items():
            item = json.loads((output / mode / "rejection.json").read_text())
            if (
                item.get("check") != mode
                or item.get("dispatchCount") != 0
                or error not in item.get("error", "")
            ):
                raise ValueError("Upstream op rejection evidence is incomplete")
            rejections[mode] = item
        if before != verify_prepared(args.mlx_root) or sources != upstream_test_sources(
            args.mlx_root, tests
        ):
            raise ValueError("MLX source changed during upstream op verification")
        evidence.update(
            passed=True,
            dispatchCount=len(trace),
            indexingDispatchCount=indexing,
            upstream=upstream,
            negativeChecks=rejections,
        )
    except Exception as error:
        evidence["error"] = str(error)
        raise
    finally:
        write_json(output / "evidence.json", evidence)
    return evidence


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "mlx-root",
        "packages",
        "random",
        "bitwise",
        "selection",
        "absolute",
        "integer64",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reductions", type=Path, action="append", required=True)
    parser.add_argument("--worker", choices=("cpu", "native", *NEGATIVE_CHECKS))
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse()
    worker(arguments) if arguments.worker else verify(arguments)

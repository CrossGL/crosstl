"""Translate every lazy unary and binary vector entry before a host run.

The host translates a ``v_`` or ``vv_`` entry the first time a test reaches
it, one entry at a time. Translating the complete table ahead of the run in
parallel worker processes bounds the native run's time and shows that all
415 entries translate and package on the runner's platform. The
packages land in the same content-addressed cache the host and the audits
use, so they are never trusted on name alone.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from demos.integrations.mlx.portable_host.elementwise_packages import (
    CACHE,
    ENTRIES,
    ElementwisePackageCache,
)
from demos.integrations.mlx.portable_host.lazy_kernel_cache import TARGETS


def _translate(job):
    root, directory, target, entries = job
    cache = ElementwisePackageCache(root, directory, target)
    return [
        (entry, cache.get(entry)[0]["artifact"]["hash"]["value"]) for entry in entries
    ]


def warm(root, packages, target, jobs):
    """Translate every entry into ``packages``' lazy vector cache."""
    if target not in TARGETS:
        raise ValueError("Unsupported lazy vector target")
    if type(jobs) is not int or not 1 <= jobs <= 16:
        raise ValueError("Lazy vector translation needs 1 to 16 jobs")
    entries = sorted(ENTRIES)
    directory = Path(packages).resolve() / CACHE
    batches = [
        (Path(root).resolve(), directory, target, entries[index::jobs])
        for index in range(jobs)
    ]
    with ProcessPoolExecutor(jobs) as pool:
        hashes = dict(item for batch in pool.map(_translate, batches) for item in batch)
    if sorted(hashes) != entries:
        raise ValueError("Lazy vector translation did not cover every entry")
    return {"target": target, "entries": len(hashes), "artifacts": hashes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    summary = warm(args.mlx_root, args.packages, args.target, args.jobs)
    if args.summary is not None:
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Translated {summary['entries']} lazy {args.target} vector entries")


if __name__ == "__main__":
    main()

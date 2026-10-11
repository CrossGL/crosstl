"""Partition row translation and verify complete, revision-matched package sets."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

from demos.integrations.mlx.portable_host.prepare import COMMIT
from demos.integrations.mlx.portable_host.reduction_packages import (
    ROW_ENTRIES,
    ROW_WIDTHS,
    SOURCE,
    build_packages,
    load_index,
)

SHARD_COUNT = 8
SCHEMA_VERSION = 1


def widths_for_shard(shard):
    if type(shard) is not int or not 0 <= shard < SHARD_COUNT:
        raise ValueError(f"Row shard must be an integer in [0, {SHARD_COUNT})")
    return ROW_WIDTHS[shard::SHARD_COUNT]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def translator_revision():
    return subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[4]), "rev-parse", "HEAD"],
        text=True,
        timeout=30,
    ).strip()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_shard(root, output, target, shard, *, jobs=2):
    widths = widths_for_shard(shard)
    revision = translator_revision()
    build_packages(root, output, target, family="row", widths=widths, jobs=jobs)
    if translator_revision() != revision:
        raise ValueError("Translator revision changed during row translation")
    record = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "mlx-row-package-shard",
        "shard": shard,
        "commit": COMMIT,
        "translatorRevision": revision,
        "target": target,
        "widths": list(widths),
        "indexSha256": digest(Path(output) / "index.json"),
        "sourceSha256": digest(Path(root) / SOURCE),
    }
    write_json(Path(output) / "shard.json", record)
    return record


def verify_artifact_files(directory, index):
    package = (directory / "package").resolve()
    if not package.is_relative_to(directory.resolve()):
        raise ValueError("Row package escapes its shard directory")
    for key, descriptor in index["descriptors"].items():
        artifact = descriptor.get("artifact", {})
        relative = artifact.get("packagePath")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
        ):
            raise ValueError(f"Invalid row artifact path: {key}")
        path = (package / relative).resolve()
        if not path.is_relative_to(package) or not path.is_file():
            raise ValueError(f"Row artifact is missing or escapes its package: {key}")
        if artifact.get("hash") != {"algorithm": "sha256", "value": digest(path)} or (
            type(artifact.get("sizeBytes")) is not int
            or artifact["sizeBytes"] != path.stat().st_size
        ):
            raise ValueError(f"Row artifact identity differs: {key}")


def inspect_shards(directory, target, revision):
    directory = Path(directory).resolve()
    shard_root = directory / "shards"
    if shard_root.is_symlink():
        raise ValueError("Row shards must be contained directories")
    children = sorted(shard_root.iterdir())
    if len(children) != SHARD_COUNT:
        raise ValueError("Row collection must contain every shard exactly once")
    descriptors, directories, records, source_hashes = {}, [], {}, set()
    for artifact_directory in children:
        child = artifact_directory / "packages"
        if artifact_directory.is_symlink() or child.is_symlink() or not child.is_dir():
            raise ValueError("Row shards must be contained directories")
        record_path = child / "shard.json"
        if record_path.is_symlink() or (child / "index.json").is_symlink():
            raise ValueError("Row shard metadata must be contained files")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("Invalid row shard record")
        shard = record.get("shard")
        widths = widths_for_shard(shard)
        if (
            shard in records
            or type(record.get("schemaVersion")) is not int
            or record["schemaVersion"] != SCHEMA_VERSION
            or record.get("kind") != "mlx-row-package-shard"
            or record.get("commit") != COMMIT
            or record.get("translatorRevision") != revision
            or record.get("target") != target
            or record.get("widths") != list(widths)
            or record.get("indexSha256") != digest(child / "index.json")
            or not isinstance(record.get("sourceSha256"), str)
            or len(record["sourceSha256"]) != 64
            or any(c not in "0123456789abcdef" for c in record["sourceSha256"])
        ):
            raise ValueError("Row shard identity, revision or index differs")
        index = load_index(child, target)
        if (
            index.get("family") != "row"
            or index["widths"] != list(widths)
            or set(index["entries"]) != set(ROW_ENTRIES)
            or descriptors.keys() & index["descriptors"].keys()
        ):
            raise ValueError("Row shards do not preserve the complete variant plan")
        verify_artifact_files(child, index)
        for descriptor in index["descriptors"].values():
            source = descriptor.get("source", {})
            if (
                source.get("path") != SOURCE
                or source.get("backend") != "metal"
                or source.get("hash") != {
                    "algorithm": "sha256",
                    "value": record["sourceSha256"],
                }
            ):
                raise ValueError("Row descriptor does not identify the pinned source")
        source_hashes.add(record["sourceSha256"])
        descriptors.update(index["descriptors"])
        directories.append(child)
        records[shard] = {
            "shard": shard,
            "directory": child.relative_to(directory).as_posix(),
            "recordSha256": digest(record_path),
            "indexSha256": record["indexSha256"],
        }
    expected = {f"w{width}/{entry}" for width in ROW_WIDTHS for entry in ROW_ENTRIES}
    if set(descriptors) != expected or len(source_hashes) != 1:
        raise ValueError(
            "Row collection has missing variants or mixed source revisions"
        )
    index = {
        "target": target,
        "family": "row",
        "widths": list(ROW_WIDTHS),
        "entries": list(ROW_ENTRIES),
        "descriptors": descriptors,
    }
    record = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "mlx-row-package-collection",
        "commit": COMMIT,
        "translatorRevision": revision,
        "sourceSha256": source_hashes.pop(),
        "target": target,
        "widths": list(ROW_WIDTHS),
        "entries": list(ROW_ENTRIES),
        "artifactCount": len(descriptors),
        "shards": [records[shard] for shard in range(SHARD_COUNT)],
    }
    return record, index, directories


def collect_shards(directory, target):
    directory = Path(directory)
    if (directory / "index.json").exists() or (directory / "collection.json").exists():
        raise ValueError("Row collection output already exists")
    record, _, _ = inspect_shards(directory, target, translator_revision())
    write_json(directory / "collection.json", record)
    return record


def load_row_packages(directory, target, *, require_all_widths=False):
    directory = Path(directory)
    collection = directory / "collection.json"
    if collection.is_symlink():
        raise ValueError("Row collection metadata must be a contained file")
    if collection.exists():
        if (directory / "index.json").exists():
            raise ValueError("Ambiguous row package and collection indexes")
        record, index, directories = inspect_shards(
            directory, target, translator_revision()
        )
        if json.loads(collection.read_text(encoding="utf-8")) != record:
            raise ValueError("Row collection changed after verification")
    else:
        index = load_index(directory, target)
        directories = [directory]
    if (
        index.get("family") != "row"
        or set(index["entries"]) != set(ROW_ENTRIES)
        or not {32, 128}.issubset(index["widths"])
    ):
        raise ValueError("Row proof requires every row entry and both threshold widths")
    if require_all_widths and set(index["widths"]) != set(ROW_WIDTHS):
        raise ValueError("Row CI requires every supported width")
    return index, directories


def compile_packages(directory, output, target):
    if (
        sys.platform != {"metal": "darwin", "directx": "win32", "opengl": "linux"}[
            target
        ]
    ):
        raise ValueError("Row compilation requires the target's native CI platform")
    index, directories = load_row_packages(directory, target, require_all_widths=True)
    output = Path(output)
    output.mkdir(parents=True)
    records = []
    for package_directory in directories:
        shard = load_index(package_directory, target)
        for key, descriptor in sorted(shard["descriptors"].items()):
            artifact = descriptor["artifact"]
            source = package_directory / "package" / artifact["packagePath"]
            module = output / (
                key.replace("/", "-")
                + {"metal": ".air", "directx": ".dxil", "opengl": ".spv"}[target]
            )
            if target == "metal":
                command = [
                    "xcrun",
                    "--sdk",
                    "macosx",
                    "metal",
                    "-Werror",
                    "-fno-fast-math",
                    "-c",
                    str(source),
                    "-o",
                    str(module),
                ]
            elif target == "directx":
                command = [
                    "dxc",
                    "-T",
                    "cs_6_6",
                    "-enable-16bit-types",
                    "-WX",
                    "-E",
                    descriptor["entryPoint"]["name"],
                    "-Fo",
                    str(module),
                    str(source),
                ]
            else:
                command = [
                    "glslangValidator",
                    "--target-env",
                    "opengl",
                    "--target-env",
                    "spirv1.3",
                    "-S",
                    "comp",
                    str(source),
                    "-o",
                    str(module),
                ]
            record = {"variant": key, "artifact": artifact, "command": command}
            records.append(record)
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=120, check=False
                )
                record.update(
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
                if (
                    result.returncode
                    or not module.is_file()
                    or module.stat().st_size == 0
                ):
                    raise RuntimeError(f"Row native compilation failed: {key}")
                if target == "opengl":
                    validation = subprocess.run(
                        ["spirv-val", "--target-env", "spv1.3", str(module)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    record["validation"] = {
                        "returncode": validation.returncode,
                        "stdout": validation.stdout,
                        "stderr": validation.stderr,
                    }
                    if validation.returncode:
                        raise RuntimeError(f"Row SPIR-V validation failed: {key}")
                if digest(source) != artifact["hash"]["value"]:
                    raise ValueError("Row artifact changed during compilation")
                record.update(
                    module=module.name, moduleSha256=digest(module), status="passed"
                )
            except (
                OSError,
                subprocess.TimeoutExpired,
                RuntimeError,
                ValueError,
            ) as error:
                record.update(status="failed", error=str(error))
                raise
            finally:
                write_json(output / "results.json", records)
    if {record["variant"] for record in records} != set(index["descriptors"]):
        raise ValueError("Row compilation omitted required variants")
    after, after_directories = load_row_packages(
        directory, target, require_all_widths=True
    )
    if index != after or directories != after_directories:
        raise ValueError("Row packages changed during compilation")
    summary = {"target": target, "compiledArtifacts": len(records), "status": "passed"}
    write_json(output / "evidence.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--mlx-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--shard", type=int, required=True)
    build.add_argument("--jobs", type=int, default=2)
    collect = commands.add_parser("collect")
    collect.add_argument("--directory", type=Path, required=True)
    native = commands.add_parser("compile")
    native.add_argument("--directory", type=Path, required=True)
    native.add_argument("--output-dir", type=Path, required=True)
    for command in (build, collect, native):
        command.add_argument(
            "--target", choices=("metal", "opengl", "directx"), required=True
        )
    args = parser.parse_args()
    if args.command == "build":
        build_shard(
            args.mlx_root, args.output_dir, args.target, args.shard, jobs=args.jobs
        )
    elif args.command == "collect":
        collect_shards(args.directory, args.target)
    else:
        compile_packages(args.directory, args.output_dir, args.target)

"""Rebuild the third-party corpus at the commits pinned in ``manifest.json`` and verify every file.

The manifest lists, per project, the repository, the pinned commit and the licence; per slice, the source
language and every translation unit with its SHA-256, stages, entry point, include directories and defines, plus
the support files its include closure reaches. Units are kept only when the project's own toolchain accepts them
in this layout. A project is fetched as a shallow, blob-filtered, sparse checkout of exactly the listed files, so
the include closure is the recorded one and nothing else. The SPIR-V slice is derived from the GLSL slice with
glslang and spirv-dis and checked against its recorded hashes as well.

Rebuilding fails closed: a missing file, a changed hash or a failed derivation is a problem, and the runner does
not measure a corpus with problems.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..process import run
from ..toolchains import tool

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
SCHEMA = 1
LANGUAGES = ("glsl", "hlsl", "metal", "spirv")
STAGES = (
    "compute",
    "vertex",
    "fragment",
    "geometry",
    "tess_control",
    "tess_eval",
    "mesh",
    "task",
    "ray_generation",
    "ray_miss",
    "ray_closest_hit",
    "ray_any_hit",
    "ray_intersection",
    "ray_callable",
)
GLSLANG_STAGE = {
    "compute": "comp",
    "vertex": "vert",
    "fragment": "frag",
    "geometry": "geom",
    "tess_control": "tesc",
    "tess_eval": "tese",
    "mesh": "mesh",
    "task": "task",
    "ray_generation": "rgen",
    "ray_miss": "rmiss",
    "ray_closest_hit": "rchit",
    "ray_any_hit": "rahit",
    "ray_intersection": "rint",
    "ray_callable": "rcall",
}
RAY_TRACING = {stage for stage in STAGES if stage.startswith("ray_")}
GIT_TIMEOUT = 900


@dataclass(frozen=True)
class Unit:
    slice: str
    language: str
    tag: str
    path: str
    sha256: str
    stages: tuple
    file: Path
    entry: str | None = None
    include_dirs: tuple = ()
    defines: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def id(self):
        return f"{self.slice}:{self.path}"


def load(path=MANIFEST):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schemaVersion") != SCHEMA:
        raise ValueError(f"Corpus manifest {path} is not schema version {SCHEMA}")
    for name, entry in manifest["slices"].items():
        if entry["language"] not in LANGUAGES:
            raise ValueError(f"Slice {name} has unknown language {entry['language']!r}")
        if entry["project"] not in manifest["projects"]:
            raise ValueError(f"Slice {name} names unknown project {entry['project']!r}")
        for unit in entry["units"]:
            unknown = set(unit["stages"]) - set(STAGES)
            if not unit["stages"] or unknown:
                raise ValueError(
                    f"{name}:{unit['path']} has invalid stages {unit['stages']}"
                )
    return manifest


def manifest_digest(path=MANIFEST):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_files(manifest, project):
    """Every repository path the project's slices need: licences, units and support files."""
    files = {name: None for name in manifest["projects"][project]["licenceFiles"]}
    for entry in manifest["slices"].values():
        if entry["project"] != project or "derivedFrom" in entry:
            continue
        for unit in entry["units"]:
            files[unit["path"]] = unit["sha256"]
        files.update(entry.get("support", {}))
    return dict(sorted(files.items()))


def _sparse_pattern(path):
    escaped = "".join("\\" + c if c in "\\*?[!# " else c for c in path)
    return "/" + escaped


def _git(repo, *arguments, stdin=None):
    command = ["git", "-C", str(repo), *arguments]
    if stdin is None:
        return run(command, timeout=GIT_TIMEOUT)
    return run([*command, "--stdin"], timeout=GIT_TIMEOUT, input=stdin)


def fetch_project(manifest, project, source_root):
    """Check out exactly the listed files of ``project`` at its pinned commit; returns (root, problems)."""
    spec = manifest["projects"][project]
    repo = Path(source_root) / project
    files = project_files(manifest, project)
    head = _git(repo, "rev-parse", "HEAD") if (repo / ".git").exists() else None
    if head is None or head.status != "accepted" or head.log.strip() != spec["commit"]:
        repo.mkdir(parents=True, exist_ok=True)
        steps = (
            []
            if (repo / ".git").exists()
            else [("init", "-q"), ("remote", "add", "origin", spec["repository"])]
        )
        steps += [
            ("config", "core.sparseCheckout", "true"),
            (
                "fetch",
                "-q",
                "--depth",
                "1",
                "--filter=blob:none",
                "origin",
                spec["commit"],
            ),
        ]
        for step in steps:
            result = _git(repo, *step)
            if result.status != "accepted":
                return repo, [f"{project}: git {step[0]} failed: {result.log[-400:]}"]
    patterns = "\n".join(_sparse_pattern(path) for path in files) + "\n"
    for step, stdin in (
        (("sparse-checkout", "set", "--no-cone"), patterns),
        (("checkout", "-q", "--force", "--detach", spec["commit"]), None),
    ):
        result = _git(repo, *step, stdin=stdin)
        if result.status != "accepted":
            return repo, [f"{project}: git {step[0]} failed: {result.log[-400:]}"]
    problems = []
    for path, digest in files.items():
        local = repo / path
        if not local.is_file():
            problems.append(f"{project}: missing {path}")
        elif digest is not None and sha256(local) != digest:
            problems.append(f"{project}: changed {path}")
    return repo, problems


def derive_spirv(source, stage, include_dirs, destination):
    """Compile one GLSL unit for Vulkan and disassemble it, as the SPIR-V slice was derived."""
    for name in ("glslangValidator", "spirv-dis"):
        if not tool(name).available:
            return f"{name} is unavailable: {tool(name).reason}"
    flag = GLSLANG_STAGE[stage]
    env = (
        ["--target-env", "vulkan1.2"]
        if stage in RAY_TRACING
        else ["--target-env", "spirv1.4"] if stage in ("mesh", "task") else []
    )
    with tempfile.TemporaryDirectory() as scratch:
        binary = Path(scratch) / "unit.spv"
        includes = [f"-I{directory}" for directory in include_dirs]
        compiled = run(
            [
                *tool("glslangValidator").command,
                "-V",
                *env,
                "-S",
                flag,
                *includes,
                source,
                "-o",
                binary,
            ]
        )
        if compiled.status != "accepted":
            return f"glslang rejected {source}: {compiled.log[-300:]}"
        text = run([*tool("spirv-dis").command, binary], separate_stderr=True)
        if text.status != "accepted":
            return f"spirv-dis failed on {source}: {text.log[-300:]}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text.log)
    return None


def _unit(name, entry, row, root):
    return Unit(
        slice=name,
        language=entry["language"],
        tag=entry["tag"],
        path=row["path"],
        sha256=row["sha256"],
        stages=tuple(row["stages"]),
        file=root / row["path"],
        entry=row.get("entry"),
        include_dirs=tuple(str(root / d) for d in row.get("includeDirs", ())),
        defines=dict(row.get("defines", {})),
    )


def rebuild(cache, manifest=None, slices=None):
    """Fetch, derive and verify the selected slices under ``cache``; returns (units, problems)."""
    manifest = manifest or load()
    cache = Path(cache)
    selected = list(slices or manifest["slices"])
    unknown = set(selected) - set(manifest["slices"])
    if unknown:
        raise ValueError(f"Unknown corpus slices: {sorted(unknown)}")
    needed = set(selected) | {
        manifest["slices"][name]["derivedFrom"]
        for name in selected
        if "derivedFrom" in manifest["slices"][name]
    }
    roots, problems = {}, []
    for project in sorted({manifest["slices"][name]["project"] for name in needed}):
        roots[project], found = fetch_project(manifest, project, cache / "sources")
        problems += found
    units = []
    for name in selected:
        entry = manifest["slices"][name]
        if "derivedFrom" not in entry:
            root = roots[entry["project"]]
            units += [_unit(name, entry, row, root) for row in entry["units"]]
            continue
        origin = manifest["slices"][entry["derivedFrom"]]
        origin_root = roots[origin["project"]]
        origin_rows = {row["path"]: row for row in origin["units"]}
        root = cache / "derived" / name
        for row in entry["units"]:
            source_row = origin_rows.get(row["source"])
            target = root / row["path"]
            if source_row is None:
                problems.append(
                    f"{name}: {row['path']} derives from unknown unit {row['source']}"
                )
                continue
            if not (target.is_file() and sha256(target) == row["sha256"]):
                failure = derive_spirv(
                    origin_root / row["source"],
                    row["stages"][0],
                    [origin_root / d for d in source_row.get("includeDirs", ())],
                    target,
                )
                if failure:
                    problems.append(f"{name}: {failure}")
                    continue
                if sha256(target) != row["sha256"]:
                    problems.append(
                        f"{name}: derived {row['path']} differs from the manifest"
                    )
                    continue
            units.append(_unit(name, entry, row, root))
    return units, problems

"""Prepare an explicit synchronous MLX backend backed by CrossTL dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from demos.integrations.mlx.portable_host.packages import (
    BINARY_OPERATIONS,
    COMPARISON_OPERATIONS,
    LOGICAL_OPERATIONS,
    UNARY_OPERATIONS,
)

COMMIT = "9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8"
HERE = Path(__file__).resolve().parent
VIEW_PRIMITIVES = (
    "AsStrided",
    "Broadcast",
    "BroadcastAxes",
    "Copy",
    "CustomTransforms",
    "Depends",
    "ExpandDims",
    "Reshape",
    "Slice",
    "Split",
    "Squeeze",
    "StopGradient",
    "Transpose",
    "Unflatten",
)
MULTI_OUTPUT_VIEWS = {"CustomTransforms", "Depends", "Split"}
COPY_PRIMITIVES = ("Contiguous", "Flatten", "Full")
CAST_PRIMITIVES = ("AsType",)


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Unexpected upstream source: {before!r}")
    return text.replace(before, after, 1)


def adapted_sources(original, primitives, events):
    """Render the complete adaptation without changing the source tree."""
    for primitive in (
        "Arange",
        "Reduce",
        "BitwiseBinary",
        *UNARY_OPERATIONS,
        *VIEW_PRIMITIVES,
        *COPY_PRIMITIVES,
        *BINARY_OPERATIONS,
        *CAST_PRIMITIVES,
        *COMPARISON_OPERATIONS,
        *LOGICAL_OPERATIONS,
        "LogicalNot",
    ):
        if primitive in {"Log2", "Log10", "Rsqrt"}:
            continue
        macro = "NO_GPU_MULTI" if primitive in MULTI_OUTPUT_VIEWS else "NO_GPU"
        primitives = replace_once(
            primitives,
            f"{macro}({primitive})",
            f"// {primitive} is implemented by the registered native dispatch backend.",
        )
    events = replace_once(
        events,
        "void Event::wait(Stream stream) {",
        """void Event::wait(Stream stream) {
  if (stream.device == Device::gpu) {
    wait();
    return;
  }""",
    )
    events = replace_once(
        events,
        "void Event::signal(Stream stream) {",
        """void Event::signal(Stream stream) {
  if (stream.device == Device::gpu) {
    auto& ec = cast<EventCounter>();
    {
      std::lock_guard lock(ec.mtx);
      ec.value = value();
    }
    ec.cv.notify_all();
    return;
  }""",
    )
    cmake = """option(MLX_CROSTL_HOST "Use the CrossTL synchronous host backend" OFF)
if(MLX_CROSTL_HOST)
  target_sources(mlx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/crosstl_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/crosstl_event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/crosstl_primitives.cpp)
else()
""" + original + "\nendif()\n"
    return {
        "CMakeLists.txt": cmake.encode("utf-8"),
        "crosstl_primitives.cpp": primitives.encode("utf-8"),
        "crosstl_event.cpp": events.encode("utf-8"),
        "crosstl_backend.cpp": (HERE / "backend.cpp").read_bytes(),
        "crosstl_dispatch.h": (HERE / "dispatch.h").read_bytes(),
    }


def require_revision(root):
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    if head != COMMIT:
        raise ValueError(f"Expected MLX {COMMIT}, got {head}")


def source_record(sources):
    return {
        "commit": COMMIT,
        "files": {
            f"mlx/backend/no_gpu/{name}": hashlib.sha256(data).hexdigest()
            for name, data in sources.items()
        },
    }


def verify_prepared(root):
    """Require exactly the current adapter and otherwise unchanged tracked sources."""
    root = Path(root).resolve()
    require_revision(root)
    changed = (
        subprocess.check_output(
            ["git", "-C", str(root), "diff", "--name-only", "-z", "HEAD", "--"],
            timeout=30,
        )
        .decode("utf-8")
        .split("\0")
    )
    if set(changed) - {"", "mlx/backend/no_gpu/CMakeLists.txt"}:
        raise ValueError("Unrelated tracked MLX sources were modified")
    originals = [
        subprocess.check_output(
            ["git", "-C", str(root), "show", f"HEAD:mlx/backend/no_gpu/{name}"],
            timeout=30,
        ).decode("utf-8")
        for name in ("CMakeLists.txt", "primitives.cpp", "event.cpp")
    ]
    sources = adapted_sources(*originals)
    for name, expected in sources.items():
        path = root / "mlx/backend/no_gpu" / name
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise ValueError(f"Prepared MLX source does not match: {name}")
    return source_record(sources)


def prepare(root, output):
    root = Path(root).resolve()
    require_revision(root)
    modified = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True, timeout=30
    )
    if modified.strip():
        raise ValueError("Preparation requires a clean MLX checkout")
    backend = root / "mlx/backend/no_gpu"
    sources = adapted_sources(
        *(
            (backend / name).read_text(encoding="utf-8")
            for name in ("CMakeLists.txt", "primitives.cpp", "event.cpp")
        )
    )
    generated = [backend / name for name in sources if name != "CMakeLists.txt"]
    if any(path.exists() or path.is_symlink() for path in generated):
        raise ValueError("Preparation cannot overwrite an existing backend file")
    if Path(output).exists() or Path(output).is_symlink():
        raise ValueError("Preparation evidence already exists")
    for name, data in sources.items():
        (backend / name).write_bytes(data)
    record = source_record(sources)
    Path(output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.mlx_root, args.output)

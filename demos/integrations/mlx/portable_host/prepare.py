"""Prepare an explicit synchronous MLX backend backed by CrossTL dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

COMMIT = "d9add9d11f3154111a4c85f267ec2fd307ecd18e"
HERE = Path(__file__).resolve().parent


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Unexpected upstream source: {before!r}")
    return text.replace(before, after, 1)


def prepare(root, output):
    root = Path(root).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    if head != COMMIT:
        raise ValueError(f"Expected MLX {COMMIT}, got {head}")
    modified = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True, timeout=30
    )
    if modified.strip():
        raise ValueError("Preparation requires a clean MLX checkout")
    backend = root / "mlx/backend/no_gpu"
    generated = [
        backend / name
        for name in (
            "crosstl_primitives.cpp",
            "crosstl_event.cpp",
            "crosstl_backend.cpp",
            "crosstl_dispatch.h",
        )
    ]
    if any(path.exists() or path.is_symlink() for path in generated):
        raise ValueError("Preparation cannot overwrite an existing backend file")
    if Path(output).exists():
        raise ValueError("Preparation evidence already exists")
    cmake = backend / "CMakeLists.txt"
    original = cmake.read_text(encoding="utf-8")
    primitives = replace_once(
        (backend / "primitives.cpp").read_text(encoding="utf-8"),
        "NO_GPU(Arange)",
        "// Arange is implemented by the registered native dispatch backend.",
    )
    events = (backend / "event.cpp").read_text(encoding="utf-8")
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
    cmake.write_text(
        """option(MLX_CROSTL_HOST "Use the CrossTL synchronous host backend" OFF)
if(MLX_CROSTL_HOST)
  target_sources(mlx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/crosstl_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/crosstl_event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/crosstl_primitives.cpp)
else()
""" + original + "\nendif()\n",
        encoding="utf-8",
    )
    (backend / "crosstl_primitives.cpp").write_text(primitives, encoding="utf-8")
    (backend / "crosstl_event.cpp").write_text(events, encoding="utf-8")
    shutil.copyfile(HERE / "backend.cpp", backend / "crosstl_backend.cpp")
    shutil.copyfile(HERE / "dispatch.h", backend / "crosstl_dispatch.h")
    paths = [cmake, *generated]
    record = {
        "commit": head,
        "files": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }
    Path(output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.mlx_root, args.output)

"""The rust-gpu build that compiles Rust probes to SPIR-V.

rust-gpu is a rustc codegen backend. ``CROSSTL_COVERAGE_RUST_GPU`` names a rust-gpu checkout in whose root these two
commands have run::

    cargo build --release -p rustc_codegen_spirv
    cargo run --release -p compiletests -- --target-env vulkan1.1 lang/core/intrinsics/log10

The first builds the backend. The second runs one of rust-gpu's own compile tests, which builds ``core`` and
``spirv-std`` for the SPIR-V target and writes the target's specification. A probe is then compiled the way those
tests compile theirs: one ``rustc`` call against the prebuilt libraries, which takes well under a second.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from pathlib import Path

TARGET = "spirv-unknown-vulkan1.1"
# The flags spirv-builder passes to every shader crate (rust-gpu's tests/compiletests/src/main.rs, ``rust_flags``).
_FLAGS = (
    "-Zbinary-dep-depinfo",
    "-Csymbol-mangling-version=v0",
    "-Zcrate-attr=feature(register_tool)",
    "-Zcrate-attr=register_tool(rust_gpu)",
    "-Coverflow-checks=off",
    "-Cdebug-assertions=off",
    "-Zinline-mir=off",
    "-Zmir-enable-passes=-GVN",
    "-Zshare-generics=off",
    # Warnings would come before the first error in the log; lints that deny by default still fail the probe.
    "-Awarnings",
)
# The SPIR-V capabilities and extensions a probe may need. A shader crate enables them as target features; the other
# languages' probes get the same operations from their compilers' highest profile.
CAPABILITIES = (
    "Int8",
    "Int16",
    "Int64",
    "Float16",
    "Float64",
    "Int64Atomics",
    "DerivativeControl",
    "DemoteToHelperInvocation",
    "GroupNonUniform",
    "GroupNonUniformVote",
    "GroupNonUniformArithmetic",
    "GroupNonUniformBallot",
    "GroupNonUniformShuffle",
    "GroupNonUniformShuffleRelative",
    "GroupNonUniformClustered",
    "GroupNonUniformQuad",
    "AtomicFloat32AddEXT",
    "AtomicFloat32MinMaxEXT",
    "ShaderClockKHR",
    "VulkanMemoryModelDeviceScope",
    "Geometry",
    "Tessellation",
    "RayTracingKHR",
)
EXTENSIONS = (
    "SPV_EXT_demote_to_helper_invocation",
    "SPV_KHR_ray_tracing",
    "SPV_EXT_shader_atomic_float_add",
    "SPV_EXT_shader_atomic_float_min_max",
    "SPV_KHR_shader_clock",
)
# The libraries a probe links, as spirv-builder passes them: ``core`` and ``compiler_builtins`` replace the prelude.
_LIBRARIES = (
    ("compiler_builtins", True),
    ("core", True),
    ("spirv-std", False),
)


def root() -> Path | None:
    configured = os.environ.get("CROSSTL_COVERAGE_RUST_GPU")
    return Path(configured) if configured else None


def _toolchain(checkout: Path) -> str:
    """The nightly toolchain the checkout's ``rust-toolchain.toml`` pins."""
    pinned = checkout / "rust-toolchain.toml"
    text = pinned.read_text() if pinned.is_file() else ""
    match = re.search(r'^channel\s*=\s*"([^"]+)"', text, re.M)
    return match.group(1) if match else ""


def _rustc(checkout: Path) -> str:
    override = os.environ.get("CROSSTL_COVERAGE_RUSTC")
    if override:
        return override
    rustup = shutil.which("rustup")
    toolchain = _toolchain(checkout)
    if not rustup or not toolchain:
        return ""
    result = subprocess.run(
        [rustup, "which", "rustc", "--toolchain", toolchain],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _library(build: Path, crate: str) -> Path | None:
    """The one library ``crate`` built under ``build``; None when there is none, or more than one to choose from."""
    stem = "lib" + crate.replace("-", "_") + "-"
    found = [
        path
        for path in build.glob(f"{crate}/*/out/{stem}*")
        if path.suffix in (".rlib", ".so", ".dylib", ".dll")
    ]
    return found[0] if len(found) == 1 else None


@functools.lru_cache(maxsize=None)
def command() -> tuple[str, ...]:
    """``rustc`` and every flag a probe compiles with but its file name and output directory; ``()`` without a
    usable build."""
    checkout = root()
    if checkout is None:
        return ()
    target = checkout / "target"
    backend = target / "release" / "librustc_codegen_spirv.so"
    spec = target / "compiletest-target-spec" / f"{TARGET}.json"
    deps = target / "compiletest-deps"
    builds = [deps / TARGET / "debug" / "build", deps / "debug" / "build"]
    if not (backend.is_file() and spec.is_file()):
        return ()
    rustc = _rustc(checkout)
    if not rustc:
        return ()
    externs = []
    for crate, replaces_prelude in _LIBRARIES:
        library = _library(builds[0], crate)
        if library is None:
            return ()
        prefix = "noprelude:" if replaces_prelude else ""
        externs += ["--extern", f"{prefix}{crate.replace('-', '_')}={library}"]
    macros = _library(builds[1], "spirv-std-macros")
    if macros is None:
        return ()
    externs += ["--extern", f"spirv_std_macros={macros}"]
    search = []
    for build in builds:
        search += [f"-Ldependency={out}" for out in sorted(build.glob("*/*/out"))]
    features = [f"+{name}" for name in CAPABILITIES]
    features += [f"+ext:{name}" for name in EXTENSIONS]
    return (
        rustc,
        f"-Zcodegen-backend={backend}",
        *_FLAGS,
        "-Ctarget-feature=" + ",".join(features),
        *search,
        *externs,
        "--edition=2021",
        "--crate-type=dylib",
        "-Zunstable-options",
        f"--target={spec}",
    )


@functools.lru_cache(maxsize=None)
def version() -> str:
    """``rust-gpu <version> (<rustc version>)``, or an empty string without a usable build."""
    base = command()
    checkout = root()
    if not base or checkout is None:
        return ""
    manifest = (checkout / "Cargo.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', manifest, re.M)
    result = subprocess.run([base[0], "--version"], capture_output=True, text=True)
    rustc = result.stdout.strip() if result.returncode == 0 else ""
    return f"rust-gpu {match.group(1) if match else '?'} ({rustc})"

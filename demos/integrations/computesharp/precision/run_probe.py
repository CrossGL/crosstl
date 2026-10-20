"""Evaluate the intrinsic probe on a translated backend.

The probe (``IntrinsicProbe.csproj``) is built against the portable runtime of
an adapted upstream checkout, its generated HLSL is exported and translated by
``translate.py`` with the same source options as the test suite, and the probe
then runs on the native backend library built by ``run_tests.py``. The result
directory has the layout the Windows job writes for WARP, so ``qualify.py`` can
compare the two.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
PROJECT = HERE / "IntrinsicProbe.csproj"
EXECUTABLE = "ComputeSharp.IntrinsicProbe"


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--target", choices=("metal", "opengl"), required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.native_library.is_file():
        raise SystemExit(f"Native backend library not found: {args.native_library}")
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    binaries = output / "bin"
    run(
        [
            "dotnet",
            "build",
            str(PROJECT),
            "-c",
            "Release",
            "-p:ProbeRuntime=portable",
            f"-p:ComputeSharpRoot={args.adapted.resolve()}",
            "-p:CrossTLTrustedCorpusBaseline=true",
            f"-p:CustomAfterMicrosoftCommonTargets={DEMO / 'runtime' / 'Portable.targets'}",
            "-o",
            str(binaries),
        ]
    )
    executable = binaries / EXECUTABLE
    shaders = output / "shaders"
    run([str(executable), "export", str(shaders)])
    artifacts = output / "artifacts"
    run(
        [
            sys.executable,
            str(DEMO / "translate.py"),
            "--discovery",
            str(shaders),
            "--output",
            str(artifacts),
            "--target",
            args.target,
        ]
    )
    environment = dict(os.environ)
    environment.update(
        {
            "CROSSTL_COMPUTESHARP_BACKEND": args.target,
            "CROSSTL_COMPUTESHARP_ARTIFACTS": str(artifacts),
            "CROSSTL_COMPUTESHARP_NATIVE_LIBRARY": str(args.native_library.resolve()),
        }
    )
    for name in ("CROSSTL_COMPUTESHARP_EVIDENCE", "CROSSTL_COMPUTESHARP_FAULT"):
        environment.pop(name, None)
    run([str(executable), "run", "default", str(output / args.target)], env=environment)
    shutil.rmtree(binaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit pinned random kernels through translation, native compilation and execution."""

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

from crosstl.project import (
    build_native_loader_dispatch_request,
)
from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    OpenGLComputeRuntime,
)
from crosstl.project.runtime_verification import (
    DirectXRuntimeParityAdapter,
    MetalRuntimeParityAdapter,
    OpenGLRuntimeParityAdapter,
    RuntimeParityExecutor,
    RuntimeTestAdapterSpec,
)
from demos.integrations.mlx.portable_host import random_packages
from demos.integrations.mlx.portable_host.prepare import COMMIT
from demos.integrations.mlx.portable_host.random_layout import RandomOutputLayout
from demos.integrations.mlx.portable_host.random_packages import (
    ENTRIES,
    build_packages,
    verify_source,
)

KEYS = ((0, 0), (0xFFFFFFFF, 0x80000000), (123, 456))
WORD_COUNTS = (1, 2, 3, 8, 17)
SOURCE = random_packages.SOURCE
SOURCE_SHA256 = random_packages.SOURCE_SHA256
BYTE_COUNTS = (1, 2, 3, 5, 6, 7, 9, 10, 11, 15, 17, 33)
GUARD_COUNT = 17


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def threefry(key, count):
    """Reference Threefry2x32 using explicitly modular Python integer arithmetic."""
    if (
        len(key) != 2
        or len(count) != 2
        or any(
            type(value) is not int or not 0 <= value < 2**32 for value in (*key, *count)
        )
    ):
        raise ValueError("Threefry requires two uint32 key and counter words")
    mask = 2**32 - 1
    schedule = (*key, key[0] ^ key[1] ^ 0x1BD11BDA)
    x, y = ((count[i] + schedule[i]) & mask for i in range(2))
    for step in range(5):
        for rotation in ((13, 15, 26, 6), (17, 29, 16, 24))[step % 2]:
            x = (x + y) & mask
            y = (((y << rotation) | (y >> (32 - rotation))) & mask) ^ x
        x = (x + schedule[(step + 1) % 3]) & mask
        y = (y + schedule[(step + 2) % 3] + step + 1) & mask
    return x, y


def workload(entry, key_count, word_count):
    if entry not in ENTRIES or key_count not in (1, 3) or word_count not in WORD_COUNTS:
        raise ValueError("Unknown bounded random workload")
    case = byte_workload(entry, key_count, word_count * 4)
    case["id"] = f"{entry}-{key_count}-{word_count}"
    return case


def byte_workload(entry, key_count, byte_count):
    if (
        entry not in ENTRIES
        or type(key_count) is not int
        or key_count not in (1, 3)
        or type(byte_count) is not int
        or byte_count < 1
    ):
        raise ValueError("Unknown bounded random byte workload")
    layout = RandomOutputLayout(key_count, byte_count)
    if layout.native_byte_count + GUARD_COUNT > 65535:
        raise ValueError("Random workload exceeds its asserted index bounds")
    keys = KEYS[:key_count]
    word_count = (byte_count + 3) // 4
    width = (word_count + 1) // 2
    expected = []
    logical_expected = []
    for key in keys:
        words = [None] * word_count
        for y in range(width):
            drop = bool(word_count % 2 and y == width - 1)
            pair = threefry(key, (y, 0 if drop else y + width))
            words[y] = pair[0]
            if not drop:
                words[y + width] = pair[1]
        raw = struct.pack("<" + "I" * word_count, *words)
        expected.extend(raw[: layout.native_bytes_per_key])
        logical_expected.extend(raw[:byte_count])
    key_values = (
        [value for key in keys for value in key]
        if entry == "rbitsc"
        else [key[i] for i in range(2) for key in keys]
    )
    # These bounds justify every index assertion used during GLSL translation.
    if len(expected) + GUARD_COUNT > 65535 or len(key_values) > 65535:
        raise ValueError("Random workload exceeds its asserted index bounds")
    return {
        "id": f"{entry}-{key_count}-{byte_count}-bytes",
        "entry": entry,
        "keyCount": key_count,
        "wordCount": word_count,
        "keys": key_values,
        "keyShape": [key_count, 2],
        "keyStrides": [1, key_count],
        "odd": word_count % 2,
        "bytesPerKey": layout.native_bytes_per_key,
        "logicalBytesPerKey": byte_count,
        "logicalExpected": logical_expected,
        "expected": [value if value < 128 else value - 256 for value in expected],
        "execution": {
            "workgroupCount": [key_count, width, 1],
            "workgroupSize": [1, 1, 1],
        },
    }


def workloads(*, byte_tails=False):
    if byte_tails:
        return [
            byte_workload(entry, keys, count)
            for entry in ENTRIES
            for keys in (1, 3)
            for count in BYTE_COUNTS
        ]
    return [
        workload(entry, keys, words)
        for entry in ENTRIES
        for keys in (1, 3)
        for words in WORD_COUNTS
    ]


def dispatch_values(descriptor, case):
    values = {
        "keys": ("uint32", case["keys"]),
        "odd": ("bool" if descriptor["target"] == "metal" else "uint32", [case["odd"]]),
        "bytes_per_key": ("uint64", [case["bytesPerKey"]]),
        "ndim": ("int32", [2]),
        "key_shape": ("int32", case["keyShape"]),
        "key_strides": ("int64", case["keyStrides"]),
    }
    inputs, outputs = {}, {}
    seen = set()
    for binding in descriptor["bindings"]:
        if "executionInput" in binding.get("provenance", {}):
            continue
        layout = binding.get("scalarLayout")
        if not layout:
            raise ValueError(f"Missing scalar layout for {binding['name']}")
        member = layout.get("memberName", binding["name"])
        if descriptor["target"] == "directx":
            member = member.removeprefix(case["entry"] + "_")
        member = {"out": "out_"}.get(member, member)
        if member in seen:
            raise ValueError("Duplicate random binding member")
        seen.add(member)
        dtype = layout["elementType"]
        if member == "out_":
            expected_dtype = "int8" if descriptor["target"] == "metal" else "int32"
            if dtype != expected_dtype:
                raise ValueError("Random output has an unexpected physical dtype")
            data = [91] * (len(case["expected"]) + GUARD_COUNT)
        else:
            if member not in values:
                raise ValueError("Unexpected random binding member")
            expected_dtype, data = values[member]
            if dtype != expected_dtype:
                raise ValueError("Random input has an unexpected physical dtype")
            if dtype == "bool":
                data = [bool(value) for value in data]
        value = {"dtype": dtype, "shape": [len(data)], "values": data}
        inputs[binding["name"]] = value
        if member == "out_":
            outputs[binding["name"]] = value
    required = {"keys", "out_", "odd", "bytes_per_key"}
    if case["entry"] == "rbits":
        required |= {"ndim", "key_shape", "key_strides"}
    if seen != required:
        raise ValueError("Random bindings do not cover the source contract")
    return inputs, outputs


def native_identity(details, descriptor, case):
    target = descriptor["target"]
    artifact = descriptor["artifact"]
    expected = {key: artifact[key] for key in ("hash", "sizeBytes")}
    identity = details.get("artifactIdentityVerification", {})
    native = details.get("nativeRuntimeDispatch", {})
    adapter = details.get("runtimeParityAdapter", {})
    steps = details.get("adapterSteps", [])
    required = {"dispatch-translated-artifact", "collect-runtime-outputs"} | {
        "metal": {"compile-metal-for-native-runtime", "link-metal-for-native-runtime"},
        "opengl": {"validate-glsl-for-opengl-runtime"},
        "directx": {"compile-hlsl-for-directx-runtime"},
    }[target]
    return (
        identity.get("verificationStatus") == "verified"
        and identity.get("target") == target
        and identity.get("expectedIdentity") == expected
        and identity.get("observedIdentity") == expected
        and native.get("artifact", {}).get("packagePath") == artifact["packagePath"]
        and native.get("artifact", {}).get("target") == target
        and native.get("dispatch", {}).get("entryPoint")
        == descriptor["entryPoint"]["name"]
        and all(
            native.get("dispatch", {}).get(key) == value
            for key, value in case["execution"].items()
        )
        and adapter.get("target") == target
        and adapter.get("runtimeAdapter") == target + "-native-runtime"
        and all(step.get("status") == "passed" for step in steps)
        and required <= {step.get("action") for step in steps}
    )


def compare_readback(result, output_name, case, descriptor):
    target = descriptor["target"]
    dtype = "int8" if target == "metal" else "int32"
    expected = case["expected"] + [91] * GUARD_COUNT
    value = result.outputs.get(output_name, {})
    actual = value.get("values", [])
    valid = (
        result.status == "ok"
        and set(result.outputs) == {output_name}
        and value.get("dtype") == dtype
        and value.get("shape") == [len(expected)]
        and isinstance(actual, list)
        and len(actual) == len(expected)
        and all(type(item) is int for item in actual)
        and actual == expected
        and native_identity(result.details, descriptor, case)
    )
    if not valid:
        return False
    layout = RandomOutputLayout(case["keyCount"], case["logicalBytesPerKey"])
    return list(layout.unpack(actual[:-GUARD_COUNT])) == case["logicalExpected"]


def retain_native_module(details, target, destination):
    """Keep the dispatched module before the runtime removes temporary files."""
    module_path = details.get("nativeRuntimeDispatch", {}).get("modulePath")
    if not isinstance(module_path, str) or not module_path:
        raise ValueError("Native random module path is missing")
    try:
        content = Path(module_path).read_bytes()
    except OSError as error:
        raise ValueError("Native random module cannot be retained") from error
    digest = hashlib.sha256(content).hexdigest()
    if target == "metal" and (
        not content.startswith(b"MTLB")
        or digest != details.get("metalRuntime", {}).get("librarySHA256")
    ):
        raise ValueError("Retained Metal module differs from the executed library")
    filename = {
        "metal": "kernel.metallib",
        "directx": "kernel.dxil",
        "opengl": "kernel.glsl",
    }[target]
    (destination / filename).write_bytes(content)
    return {"path": filename, "sizeBytes": len(content), "sha256": digest}


class RandomAuditExecutor(RuntimeParityExecutor):
    """Retain evidence while the dispatch state's temporary files still exist."""

    evidence_directory = None

    def collect_outputs(self, state):
        outputs = super().collect_outputs(state)
        if self.evidence_directory is None:
            raise ValueError("Random evidence directory is missing")
        state.details["retainedNativeModule"] = retain_native_module(
            state.details, self.target, self.evidence_directory
        )
        return outputs


def executor(target):
    adapter = {
        "metal": lambda: MetalRuntimeParityAdapter(),
        "opengl": lambda: OpenGLRuntimeParityAdapter(
            runtime=OpenGLComputeRuntime(context_backends=("egl",))
        ),
        "directx": lambda: DirectXRuntimeParityAdapter(runtime=DirectXComputeRuntime()),
    }[target]()
    return RandomAuditExecutor(
        RuntimeTestAdapterSpec(
            adapter_id="mlx-random-" + target,
            target=target,
            executor=target,
            adapter_kind=target + "-native-runtime",
        ),
        runtime_adapter=adapter,
    )


def audit(root, target, output, *, byte_tails=False):
    output.mkdir(parents=True)
    evidence = {
        "commit": COMMIT,
        "target": target,
        "profile": "byte-tails" if byte_tails else "words",
        "passed": False,
        "cases": [],
        "fullHostIntegration": False,
        "fullUpstreamSuite": False,
    }
    native = None
    try:
        descriptors = build_packages(root.resolve(), target, output.resolve())
        native = executor(target)
        cases = workloads(byte_tails=byte_tails)
        for case in cases:
            destination = output / case["id"]
            destination.mkdir()
            record = {"workload": case, "passed": False}
            try:
                inputs, outputs = dispatch_values(descriptors[case["entry"]], case)
                write_json(
                    destination / "inputs.json", {"inputs": inputs, "outputs": outputs}
                )
                request = build_native_loader_dispatch_request(
                    descriptors[case["entry"]],
                    output / "package",
                    inputs,
                    outputs,
                    case["execution"],
                    expected_target=target,
                )
                native.evidence_directory = destination
                result = native.run(request)
                write_json(
                    destination / "result.json",
                    {
                        "status": result.status,
                        "outputs": result.outputs,
                        "details": result.details,
                    },
                )
                (output_name,) = outputs
                passed = compare_readback(
                    result, output_name, case, descriptors[case["entry"]]
                )
                if passed:
                    layout = RandomOutputLayout(
                        case["keyCount"], case["logicalBytesPerKey"]
                    )
                    logical = layout.unpack(
                        result.outputs[output_name]["values"][:-GUARD_COUNT]
                    )
                    write_json(
                        destination / "logical-output.json",
                        {
                            "bytesPerKey": layout.bytes_per_key,
                            "nativeBytesPerKey": layout.native_bytes_per_key,
                            "values": list(logical),
                        },
                    )
                if passed:
                    record["nativeModule"] = result.details.get("retainedNativeModule")
                    if not record["nativeModule"]:
                        raise ValueError("Native random module was not retained")
                record["passed"] = passed
                if not record["passed"]:
                    record["error"] = (
                        "Native random values, layout, guards or execution identity differ"
                    )
            except (ValueError, RuntimeError) as error:
                record["error"] = str(error)
                record["errorType"] = type(error).__name__
                record["details"] = getattr(error, "details", {})
            evidence["cases"].append(record)
            write_json(output / "evidence.json", evidence)
        evidence["passed"] = len(evidence["cases"]) == len(cases) and all(
            record["passed"] for record in evidence["cases"]
        )
        verify_source(root)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        evidence["passed"] = False
        evidence["error"] = str(error)
        evidence["errorType"] = type(error).__name__
    finally:
        if native is not None:
            close = getattr(native.runtime_adapter.runtime, "close", None)
            if close:
                try:
                    close()
                except (RuntimeError, OSError) as error:
                    evidence["passed"] = False
                    evidence["cleanupError"] = str(error)
        write_json(output / "evidence.json", evidence)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument(
        "--target", choices=("metal", "opengl", "directx"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--byte-tails", action="store_true")
    args = parser.parse_args(argv)
    result = audit(
        args.mlx_root, args.target, args.output_dir, byte_tails=args.byte_tails
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "cases": len(result["cases"]),
                "evidence": str(args.output_dir / "evidence.json"),
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

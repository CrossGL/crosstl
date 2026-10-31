"""Tie quantization dispatches to their cached packages and native modules.

``quantization_dispatch`` runs MLX's affine and block quantize and dequantize
kernels, which ``QuantizationPackageCache`` translates from the unchanged
pinned sources on first use. Each dispatch is checked by deriving its package
again from the cache and comparing the recorded artifact with the unchanged
cached source, by checking that the launch covers complete quantization groups
and that every output kept its guard, and by checking the retained native
modules. The upstream tests compare the values with MLX's CPU quantization.
"""

from pathlib import Path

from demos.integrations.mlx.portable_host import quantization_layout
from demos.integrations.mlx.portable_host.gather_evidence import (
    audit_native_execution,
    require,
)
from demos.integrations.mlx.portable_host.lazy_package_evidence import (
    require_unchanged,
)
from demos.integrations.mlx.portable_host.quantization_dispatch import (
    GUARD_COUNT,
    GUARDS,
)
from demos.integrations.mlx.portable_host.quantization_packages import (
    QuantizationPackageCache,
)

CACHE = "quantization"
GEOMETRY = ("workgroupCount", "workgroupSize")


def is_quantization(entry):
    """Whether ``entry`` names a quantization kernel the host dispatches."""
    try:
        quantization_layout.signature(entry)
    except (TypeError, ValueError):
        return False
    return True


def cache(mlx_root, packages, target):
    """The host's quantization cache for one package directory and target."""
    return QuantizationPackageCache(mlx_root, Path(packages) / CACHE, target)


def audit_event(event, packages):
    """Check one quantization dispatch against its cached package."""
    entry = event["entry"]
    _, _, group, _ = quantization_layout.signature(entry)
    require(
        event["target"] == packages.target,
        "Quantization dispatch targets another backend",
    )
    descriptor, directory = packages.get(entry)
    require_unchanged(descriptor, directory)
    require(
        event["artifact"] == descriptor["artifact"]
        and event["dispatchVersion"] == 3
        and Path(event["packageRoot"]).resolve() == Path(directory).resolve(),
        "Quantization dispatch does not name its cached package",
    )
    elements = event["threads"]
    require(
        type(elements) is int and elements > 0 and elements % group == 0,
        "Quantization dispatch does not cover complete groups",
    )
    groups = elements // group
    require(
        event["quantizationMetadata"] == {
            "groupCount": groups,
            "elementCount": elements,
        }
        and event["workgroupCount"] == [groups, 1, 1]
        and event["workgroupSize"] == [quantization_layout.workgroup_size(entry), 1, 1],
        "Quantization launch does not match complete quantization groups",
    )
    request = event["details"]["request"]
    require(
        request["target"] == packages.target
        and request["entryPoint"] == descriptor["entryPoint"]["name"]
        and all(request["dispatch"][key] == event[key] for key in GEOMETRY),
        "Quantization native launch differs",
    )
    contract = quantization_layout.buffer_contract(entry, groups)
    outputs = {
        name: (dtype, count)
        for name, (dtype, count, output) in contract.items()
        if output
    }
    readbacks = event["quantizationReadbacks"]
    require(
        set(readbacks) == set(outputs)
        and all(
            _guarded(readbacks[name], dtype, count)
            for name, (dtype, count) in outputs.items()
        ),
        "Quantization outputs or their guards differ",
    )
    audit_native_execution(event)


def _guarded(words, dtype, count):
    """Whether a readback holds ``count`` words and then its intact guard.

    The host checked every guard before committing the output; a compacted
    readback keeps only its length and digest, so only its length is checked.
    """
    if isinstance(words, dict):
        compacted = words.get("compactedValues")
        return (
            set(words) == {"compactedValues"}
            and isinstance(compacted, dict)
            and compacted.get("length") == count + GUARD_COUNT
        )
    return (
        isinstance(words, list)
        and len(words) == count + GUARD_COUNT
        and words[count:] == [GUARDS[dtype]] * GUARD_COUNT
    )

"""The upstream op audit of translated quantization dispatches."""

import pytest

from demos.integrations.mlx.portable_host import (
    quantization_evidence,
    quantization_layout,
)
from demos.integrations.mlx.portable_host.quantization_dispatch import (
    GUARD_COUNT,
    GUARDS,
)

TARGETS = ("metal", "directx", "opengl")
ENTRY_POINTS = {"directx": "CSMain", "opengl": "main"}
QUANTIZE = "affine_quantize_float_gs_64_b_4"
DEQUANTIZE = "affine_dequantize_float16_t_gs_32_b_8"


class FakeCache:
    def __init__(self, target, root):
        self.target, self.root, self.requests = target, root, []

    def get(self, entry):
        self.requests.append(entry)
        return (
            {
                "artifact": {"packagePath": f"artifacts/{entry}", "sizeBytes": 7},
                "entryPoint": {"name": ENTRY_POINTS.get(self.target, entry)},
            },
            self.root / entry,
        )


def event_for(target, root, entry, groups=2, *, compact=False):
    _, _, group, _ = quantization_layout.signature(entry)
    readbacks = {}
    for name, (dtype, count, output) in quantization_layout.buffer_contract(
        entry, groups
    ).items():
        if not output:
            continue
        words = [index % 7 for index in range(count)] + [GUARDS[dtype]] * GUARD_COUNT
        readbacks[name] = (
            {"compactedValues": {"length": len(words), "sha256": "0" * 64}}
            if compact and len(words) > 64
            else words
        )
    launch = {
        "workgroupCount": [groups, 1, 1],
        "workgroupSize": [quantization_layout.workgroup_size(entry), 1, 1],
    }
    return {
        "entry": entry,
        "target": target,
        "threads": groups * group,
        **launch,
        "dispatchVersion": 3,
        "artifact": {"packagePath": f"artifacts/{entry}", "sizeBytes": 7},
        "packageRoot": str(root / entry),
        "quantizationMetadata": {"groupCount": groups, "elementCount": groups * group},
        "quantizationReadbacks": readbacks,
        "details": {
            "request": {
                "target": target,
                "entryPoint": ENTRY_POINTS.get(target, entry),
                "dispatch": dict(launch),
            }
        },
    }


@pytest.fixture
def checked(monkeypatch):
    calls = []
    monkeypatch.setattr(
        quantization_evidence,
        "require_unchanged",
        lambda descriptor, directory: calls.append(("source", directory.name)),
    )
    monkeypatch.setattr(
        quantization_evidence,
        "audit_native_execution",
        lambda event: calls.append(("native", event["entry"])),
    )
    return calls


def test_quantization_entries_are_recognized():
    assert quantization_evidence.is_quantization(QUANTIZE)
    assert quantization_evidence.is_quantization(DEQUANTIZE)
    assert quantization_evidence.is_quantization(
        "nvfp4_quantize_float_gs_16_b_4_hgs_true"
    )
    for entry in (
        "affine_quantize_float_gs_48_b_4",
        "mxfp4_quantize_float_gs_16_b_4_hgs_false",
        "v_Absfloat32float32",
        None,
    ):
        assert not quantization_evidence.is_quantization(entry)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("entry", (QUANTIZE, DEQUANTIZE))
@pytest.mark.parametrize("compact", (False, True))
def test_audit_derives_the_package_and_checks_groups_and_guards(
    tmp_path, checked, target, entry, compact
):
    cache = FakeCache(target, tmp_path)
    event = event_for(target, tmp_path, entry, compact=compact)
    assert compact == any(
        isinstance(words, dict) for words in event["quantizationReadbacks"].values()
    )
    quantization_evidence.audit_event(event, cache)
    assert cache.requests == [entry]
    assert checked == [("source", entry), ("native", entry)]


def _mutate(event, path, value):
    *keys, last = path
    target = event
    for key in keys:
        target = target[key]
    target[last] = value


@pytest.mark.parametrize(
    "path, value, message",
    [
        (("target",), "opengl", "another backend"),
        (("artifact", "sizeBytes"), 8, "cached package"),
        (("packageRoot",), "/elsewhere", "cached package"),
        (("dispatchVersion",), 2, "cached package"),
        (("threads",), 100, "complete groups"),
        (("workgroupCount",), [3, 1, 1], "complete quantization groups"),
        (("quantizationMetadata", "groupCount"), 1, "complete quantization groups"),
        (("details", "request", "entryPoint"), "other", "native launch differs"),
        (
            ("details", "request", "dispatch", "workgroupCount"),
            [1, 1, 1],
            "native launch differs",
        ),
        (("quantizationReadbacks", "scales", -1), 0, "guards differ"),
        (("quantizationReadbacks", "biases"), [0] * GUARD_COUNT, "guards differ"),
        (
            ("quantizationReadbacks", "out", "compactedValues", "length"),
            64,
            "guards differ",
        ),
    ],
)
def test_audit_refuses_changed_packages_launches_and_guards(
    tmp_path, checked, path, value, message
):
    event = event_for("metal", tmp_path, QUANTIZE, compact=True)
    _mutate(event, path, value)
    with pytest.raises(ValueError, match=message):
        quantization_evidence.audit_event(event, FakeCache("metal", tmp_path))
    assert ("native", QUANTIZE) not in checked


@pytest.mark.parametrize("change", ("missing", "extra"))
def test_audit_requires_exactly_the_output_readbacks(tmp_path, checked, change):
    event = event_for("directx", tmp_path, QUANTIZE)
    readbacks = event["quantizationReadbacks"]
    if change == "missing":
        del readbacks["biases"]
    else:
        readbacks["w"] = [0] * (128 + GUARD_COUNT)
    with pytest.raises(ValueError, match="guards differ"):
        quantization_evidence.audit_event(event, FakeCache("directx", tmp_path))

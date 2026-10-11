import copy
import itertools
import json
import struct

import pytest

from demos.integrations.mlx import prove_quantized_metal as proof


def _entries():
    return [
        {
            "entryPoint": f"{operation}_{dtype}_{group}_{bits}",
            "variant": operation,
            "dataType": dtype,
            "groupSize": group,
            "bitWidth": bits,
        }
        for operation, dtype, group, bits in itertools.product(
            proof.OPERATIONS, proof.DATA_TYPES, proof.GROUP_SIZES, proof.BIT_WIDTHS
        )
    ]


def _decode(data, dtype):
    if dtype == "float":
        return [value[0] for value in struct.iter_unpack("<f", data)]
    if dtype == "float16_t":
        return [value[0] for value in struct.iter_unpack("<e", data)]
    return [
        struct.unpack("<f", struct.pack("<I", word[0] << 16))[0]
        for word in struct.iter_unpack("<H", data)
    ]


def _unpack_codes(data, bits, count):
    bit_stream = "".join(f"{byte:08b}"[::-1] for byte in data)
    return [
        int(bit_stream[start : start + bits][::-1], 2)
        for start in range(0, count * bits, bits)
    ]


@pytest.mark.parametrize("entry", _entries(), ids=lambda entry: entry["entryPoint"])
def test_affine_dataset_matches_independent_packing_and_arithmetic(entry):
    inputs, outputs, geometry = proof._dataset(entry)
    assert len(inputs) == 4
    assert all(data[-len(proof.GUARD) :] == proof.GUARD for data in inputs)
    assert all(data[-len(proof.GUARD) :] == proof.GUARD for data in outputs.values())
    original = [data[: -len(proof.GUARD)] for data in inputs]
    expected = {slot: data[: -len(proof.GUARD)] for slot, data in outputs.items()}
    dtype, group, bits = entry["dataType"], entry["groupSize"], entry["bitWidth"]
    count = group * 5
    bins = (1 << bits) - 1
    if entry["variant"] == "affine_quantize":
        values = _decode(original[0], dtype)
        scales = _decode(expected[2], dtype)
        biases = _decode(expected[3], dtype)
        codes = _unpack_codes(expected[1], bits, count)
        assert len(values) == len(codes) == count
        assert len(scales) == len(biases) == 5
        for index, (value, code) in enumerate(zip(values, codes)):
            chunk = index // group
            wanted = round((value - biases[chunk]) / scales[chunk])
            assert code == wanted and 0 <= code <= bins
        for chunk in range(4):
            section = values[chunk * group : (chunk + 1) * group]
            assert min(section) == min(
                biases[chunk], biases[chunk] + scales[chunk] * bins
            )
            assert max(section) == max(
                biases[chunk], biases[chunk] + scales[chunk] * bins
            )
        epsilon = {"float": "95bfd6b3", "float16_t": "0280", "bfloat16_t": "d7b3"}
        epsilon_bytes = bytes.fromhex(epsilon[dtype])
        assert expected[2][-len(epsilon_bytes) :] == epsilon_bytes
        assert geometry == {"workgroupCount": [1, 5, 1], "workgroupSize": [32, 1, 1]}
    else:
        codes = _unpack_codes(original[0], bits, count)
        scales = _decode(original[1], dtype)
        biases = _decode(original[2], dtype)
        values = _decode(expected[3], dtype)
        assert len(values) == len(codes) == count
        assert values == [
            code * scales[index // group] + biases[index // group]
            for index, code in enumerate(codes)
        ]
        packed_values_per_thread = (
            8 // bits if bits in (2, 4, 8) else 8 if bits in (3, 5) else 4
        )
        assert geometry["workgroupSize"] == [1, 1, 1]
        assert geometry["workgroupCount"] == [group // packed_values_per_thread, 5, 1]
    for slot, data in expected.items():
        assert original[slot] == b"\xa5" * len(data)


def test_affine_contract_selects_every_configuration_once():
    contract = json.loads(proof.DEFAULT_CONTRACT.read_text())
    entries = proof._selected_entries(contract)
    assert len(entries) == 108
    assert contract["commit"] == proof.MLX_COMMIT
    assert contract["source"] == proof.SOURCE
    assert contract["sourceSha256"] == proof.SOURCE_SHA256


@pytest.mark.parametrize(
    "change", ("missing", "duplicate", "wrong-type", "wrong-bits", "wrong-group")
)
def test_affine_contract_rejects_incomplete_or_changed_coverage(change):
    entries = _entries()
    if change == "missing":
        entries.pop()
    elif change == "duplicate":
        entries[-1] = copy.deepcopy(entries[0])
    else:
        entries[0][
            {
                "wrong-type": "dataType",
                "wrong-bits": "bitWidth",
                "wrong-group": "groupSize",
            }[change]
        ] = "unsupported"
    with pytest.raises(ValueError, match="coverage"):
        proof._selected_entries({"entries": entries})


@pytest.mark.parametrize(
    "bits,codes", ((1, [0] * 8), (3, [0]), (2, [-1] * 4), (2, [4] * 4), (2, [True] * 4))
)
def test_affine_packing_rejects_invalid_values(bits, codes):
    with pytest.raises(ValueError):
        proof._pack_codes(codes, bits)


@pytest.mark.parametrize("slot", range(4))
@pytest.mark.parametrize("change", ("payload", "guard", "truncate", "extend"))
def test_readback_checks_detect_input_output_and_guard_changes(tmp_path, slot, change):
    inputs, outputs, _ = proof._dataset(_entries()[0])
    for index, data in enumerate(inputs):
        (tmp_path / f"buffer-{index}.bin").write_bytes(outputs.get(index, data))
    assert proof._check_readbacks(tmp_path, inputs, outputs)["errors"] == []
    path = tmp_path / f"buffer-{slot}.bin"
    data = bytearray(path.read_bytes())
    if change == "payload":
        data[0] ^= 1
    elif change == "guard":
        data[-1] ^= 1
    elif change == "truncate":
        del data[-1:]
    else:
        data.append(0)
    path.write_bytes(data)
    result = proof._check_readbacks(tmp_path, inputs, outputs)
    assert [error["slot"] for error in result["errors"]] == [slot]


def test_checkout_failure_is_retained_without_claiming_execution(tmp_path, monkeypatch):
    def invalid_checkout(_root):
        raise ValueError("MLX revision differs")

    monkeypatch.setattr(proof, "_verify_checkout", invalid_checkout)
    monkeypatch.setattr(
        proof,
        "run",
        lambda *_args, **_kwargs: pytest.fail("Native execution must not start"),
    )
    output = tmp_path / "proof"
    report = proof.prove(
        tmp_path / "mlx", tmp_path / "bundle", proof.DEFAULT_CONTRACT, output
    )
    assert report["status"] == "failed"
    assert report["numericalExecution"] is False
    assert report["records"] == []
    assert report["failures"] == ["MLX revision differs"]
    assert json.loads((output / "report.json").read_text()) == report


def test_proof_refuses_to_write_inside_inputs(tmp_path):
    root = tmp_path / "mlx"
    with pytest.raises(ValueError, match="separate"):
        proof.prove(root, tmp_path / "bundle", proof.DEFAULT_CONTRACT, root / "proof")
    assert not root.exists()


def test_proof_refuses_to_overwrite_previous_evidence(tmp_path):
    output = tmp_path / "proof"
    output.mkdir()
    sentinel = output / "report.json"
    sentinel.write_text("previous evidence")
    with pytest.raises(FileExistsError):
        proof.prove(
            tmp_path / "mlx", tmp_path / "bundle", proof.DEFAULT_CONTRACT, output
        )
    assert sentinel.read_text() == "previous evidence"

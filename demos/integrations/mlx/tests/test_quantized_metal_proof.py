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


def _matrix_cases():
    return [
        (dtype, family, layout, fill)
        for dtype, family, fill in itertools.product(
            proof.DATA_TYPES, proof.MATRIX_VARIANTS, (0xA5, 0x5A)
        )
        for layout in (("flat", "broadcast") if family == "wide" else ("flat",))
    ]


@pytest.mark.parametrize("dtype,family,layout,fill", _matrix_cases())
def test_matrix_dataset_matches_uploaded_storage_and_batch_strides(
    dtype, family, layout, fill
):
    inputs, outputs, geometry, shape = proof._matrix_dataset(
        dtype, family, layout, fill
    )
    assert len(inputs) == 16 and set(outputs) == {4}
    assert all(data[-16:] == proof.GUARD for data in inputs)
    assert outputs[4][-16:] == proof.GUARD
    assert inputs[4][:-16] == bytes([fill]) * (len(outputs[4]) - 16)
    unpack = lambda slot, code: [
        value[0] for value in struct.iter_unpack("<" + code, inputs[slot][:-16])
    ]
    k, n, m = (unpack(slot, "i")[0] for slot in (5, 6, 7))
    assert (k, n, m) == (shape["K"], shape["N"], shape["M"])
    assert geometry == {
        "workgroupCount": [1, 2, shape["batch"]],
        "workgroupSize": [32, 2, 1] if family == "wide" else [32, 2, 2],
    }
    code_count = len(inputs[0][:-16]) * 8 // shape["bits"]
    codes = _unpack_codes(inputs[0][:-16], shape["bits"], code_count)
    scales, biases, vectors = (_decode(inputs[slot][:-16], dtype) for slot in (1, 2, 3))

    def offset(batch, shape_slot, stride_slot):
        dimensions, strides = unpack(shape_slot, "i"), unpack(stride_slot, "q")
        assert len(dimensions) == len(strides)
        result = 0
        for dimension, stride in reversed(list(zip(dimensions, strides))):
            result += (batch % dimension) * stride
            batch //= dimension
        assert batch == 0
        return result

    expected = []
    for batch in range(shape["batch"]):
        x_offset = offset(batch, 9, 10)
        w_offset = offset(batch, 12, 13) * 32 // shape["bits"]
        s_offset, b_offset = offset(batch, 12, 14), offset(batch, 12, 15)
        for row in range(m):
            for column in range(n):
                value = 0.0
                for inner in range(k):
                    index = (
                        inner * n + column
                        if family == "nontransposed"
                        else column * k + inner
                    )
                    group = index // shape["groupSize"]
                    value += vectors[x_offset + row * k + inner] * (
                        codes[w_offset + index] * scales[s_offset + group]
                        + biases[b_offset + group]
                    )
                expected.append(value)
    # These dyadic controls have exactly representable outputs in all three types.
    assert _decode(outputs[4][:-16], dtype) == expected
    assert len(expected) == shape["batch"] * m * n


def test_matrix_contract_selects_both_loader_layouts_and_batched_vectors():
    entries = proof._matrix_entries(json.loads(proof.DEFAULT_CONTRACT.read_text()))
    assert len(entries) == 9
    assert {(family, entry["dataType"]) for family, entry in entries} == set(
        itertools.product(proof.MATRIX_VARIANTS, proof.DATA_TYPES)
    )
    assert len(_matrix_cases()) == 24


@pytest.mark.parametrize("change", ("missing", "duplicate", "type", "group", "bits"))
def test_matrix_contract_rejects_changed_selection(change):
    contract = json.loads(proof.DEFAULT_CONTRACT.read_text())
    _, selected = proof._matrix_entries(contract)[0]
    if change == "missing":
        contract["entries"].remove(selected)
    elif change == "duplicate":
        contract["entries"].append(copy.deepcopy(selected))
    else:
        selected[
            {"type": "dataType", "group": "groupSize", "bits": "bitWidth"}[change]
        ] = "unsupported"
    with pytest.raises(ValueError, match="coverage"):
        proof._matrix_entries(contract)


@pytest.mark.parametrize("slot", range(16))
@pytest.mark.parametrize("change", ("payload", "guard", "truncate"))
def test_matrix_readback_checks_cover_every_binding(tmp_path, slot, change):
    inputs, outputs, _, _ = proof._matrix_dataset("float", "wide", "broadcast", 0xA5)
    for index, data in enumerate(inputs):
        (tmp_path / f"buffer-{index}.bin").write_bytes(outputs.get(index, data))
    assert proof._check_readbacks(tmp_path, inputs, outputs)["errors"] == []
    path = tmp_path / f"buffer-{slot}.bin"
    data = bytearray(path.read_bytes())
    if change == "truncate":
        del data[-1:]
    else:
        data[0 if change == "payload" else -1] ^= 1
    path.write_bytes(data)
    assert [
        error["slot"]
        for error in proof._check_readbacks(tmp_path, inputs, outputs)["errors"]
    ] == [slot]


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


@pytest.mark.parametrize("allow_drift", [False, True])
def test_proof_forwards_the_drift_policy_to_both_bundle_reads(
    tmp_path, monkeypatch, allow_drift
):
    calls = []

    def records(*_args, allow_drift=False):
        calls.append(allow_drift)
        raise ValueError("stop after the bundle read")

    monkeypatch.setattr(proof, "_verify_checkout", lambda _root: None)
    monkeypatch.setattr(
        proof,
        "_contract",
        lambda _path: (
            {
                "commit": proof.MLX_COMMIT,
                "source": proof.SOURCE,
                "sourceSha256": proof.SOURCE_SHA256,
                "target": "metal",
            },
            {},
            "0" * 64,
        ),
    )
    monkeypatch.setattr(proof, "_selected_entries", lambda _contract: [])
    monkeypatch.setattr(proof, "_bundle_records", records)
    report = proof.prove(
        tmp_path / "mlx",
        tmp_path / "bundle",
        proof.DEFAULT_CONTRACT,
        tmp_path / "proof",
        allow_drift=allow_drift,
    )
    assert calls == [allow_drift]
    assert report["status"] == "failed"
    assert report["failures"] == ["stop after the bundle read"]


def test_cli_allows_drift_only_when_asked(tmp_path, monkeypatch):
    seen = []

    def fake_prove(*_args, allow_drift):
        seen.append(allow_drift)
        return {"status": "passed", "records": [], "matrixRecords": []}

    monkeypatch.setattr(proof, "prove", fake_prove)
    argv = [
        "--mlx-root",
        str(tmp_path / "mlx"),
        "--bundle-root",
        str(tmp_path / "bundle"),
        "--output-dir",
        str(tmp_path / "proof"),
    ]
    proof.main(argv)
    proof.main([*argv, "--allow-drift"])
    assert seen == [False, True]


def test_matrix_controls_check_sources_against_their_bundle_records(
    tmp_path, monkeypatch
):
    # In report mode a drifted record keeps the contract's pin apart from its
    # own bytes; the controls run the exported bytes the record names.
    contract = {
        "entries": [
            {
                "entryPoint": f"{family}_{dtype}",
                "variant": variant,
                "dataType": dtype,
                "groupSize": group,
                "bitWidth": bits,
                "sha256": "0" * 64,
            }
            for family, (variant, group, bits) in proof.MATRIX_VARIANTS.items()
            for dtype in proof.DATA_TYPES
        ]
    }
    source = tmp_path / "kernel.metal"
    source.write_text("kernel", encoding="utf-8")
    artifacts = {
        entry["entryPoint"]: {"path": str(source), "sha256": proof._sha256(source)}
        for entry in contract["entries"]
    }

    class Compiled(Exception):
        pass

    def compile_source(*_args, **_kwargs):
        raise Compiled

    monkeypatch.setattr(proof, "_compile_source", compile_source)
    with pytest.raises(Compiled):
        proof._run_matrix_controls(
            contract, artifacts, tmp_path, tmp_path / "out", None, None, {}
        )
    artifacts["wide_float"]["sha256"] = "1" * 64
    with pytest.raises(ValueError, match="Matrix source changed"):
        proof._run_matrix_controls(
            contract, artifacts, tmp_path, tmp_path / "out", None, None, {}
        )

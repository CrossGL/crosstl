"""Translate and execute the pinned MLX ``random`` kernels natively."""

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    load_project_config,
    translate_project,
    validate_project_report,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/random.metal"
SOURCE_SHA256 = "f1a19b3f11b7b10203824890f13debc6d627959b4e7f17c219e2e9da553c1bd7"
# MLX's C++ suite pins Threefry 2x32 outputs generated with JAX. The known-answer
# cases transcribe them and reach the kernels with the dispatches MLX issues.
KNOWN_ANSWERS = "tests/random_tests.cpp"
KNOWN_ANSWERS_SHA256 = (
    "3fc4e7fb0481fff61dbe6b3776b152ef018e142db89bdc7b9bedbcc9c2374e9d"
)
ENTRIES = ("rbitsc", "rbits")
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_RANDOM"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/random.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
# MLX dispatches a (keys, half_size + odd) thread grid; one-thread workgroups
# reproduce it exactly, so ``threads_per_grid`` equals the workgroup count.
WORKGROUP_SIZE = (1, 1, 1)
# OpenGL indexes 32-bit storage arrays. These host preconditions bound the
# per-key byte offset and the byte index inside a key to signed int range,
# which holds whenever the output has fewer than 2^31 bytes.
OPENGL_INDEX_RANGE_ASSERTIONS = {
    "rbitsc": (
        ("index.x * bytes_per_key", 0, 2**31 - 1),
        ("idx + i", 0, 2**31 - 1),
        ("out_ + idx + i", 0, 2**31 - 1),
    ),
    "rbits": (
        ("uint64(index.x) * bytes_per_key", 0, 2**31 - 1),
        ("idx + i", 0, 2**31 - 1),
        ("out_ + idx + i", 0, 2**31 - 1),
    ),
}
GUARD_COUNT = 16
# Every store writes a sign-extended byte. The 32-bit guard used by the widened
# OpenGL and DirectX storage is outside that range, so it reveals any store;
# Metal keeps bytes and runs every case under two complementary guards.
GUARDS = {
    "metal": (-0x5B, 0x5A),
    "opengl": (0x2468ACE0,),
    "directx": (0x2468ACE0,),
}
# Reflected physical element types of ``out`` and ``odd``.
ELEMENT_TYPES = {
    "metal": {"out_": "int8", "odd": "bool"},
    "opengl": {"out_": "int32", "odd": "uint32"},
    "directx": {"out_": "int32", "odd": "uint32"},
}
ROTATIONS = ((13, 15, 26, 6), (17, 29, 16, 24))
MASK32 = 0xFFFFFFFF


def _lcg_words(seed, count):
    state = seed
    words = []
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        words.append(state >> 32)
    return words


# name -> case. ``view`` is the (shape, strides) of a non-contiguous key array
# for ``rbits``; ``known`` is the dtype and values pinned by MLX's own suite.
CASES = {
    "rbitsc": {
        # random::bits({}, key(0)) and key(1).
        "seed0": {"storage": [0, 0], "bytesPerKey": 4, "known": ("I", [1797259609])},
        "seed1": {"storage": [0, 1], "bytesPerKey": 4, "known": ("I", [507451445])},
        # Widths 2 and 1 still store the whole dropped-last word: the bytes after
        # the key land in the guard region and are part of the expected stores.
        "seed0-width2": {"storage": [0, 0], "bytesPerKey": 2, "known": ("H", [345])},
        "seed1-width2": {
            "storage": [0, 1],
            "bytesPerKey": 2,
            "known": ("H", [6197]),
        },
        "seed0-width1": {"storage": [0, 0], "bytesPerKey": 1, "known": ("B", [89])},
        "seed1-width1": {"storage": [0, 1], "bytesPerKey": 1, "known": ("B", [53])},
        # random::bits({3, 1}, key(0)).
        "seed0-shape3x1": {
            "storage": [0, 0],
            "bytesPerKey": 12,
            "known": ("I", [4146024105, 1351547692, 2718843009]),
        },
        # random::bits({5..8}, 2, key(0)): odd word counts and edge bytes.
        "seed0-uint16x5": {
            "storage": [0, 0],
            "bytesPerKey": 10,
            "known": ("H", [20137, 63263, 64300, 20622, 16513]),
        },
        "seed0-uint16x6": {
            "storage": [0, 0],
            "bytesPerKey": 12,
            "known": ("H", [20137, 63263, 64300, 20622, 16513, 41486]),
        },
        "seed0-uint16x7": {
            "storage": [0, 0],
            "bytesPerKey": 14,
            "known": ("H", [20137, 63263, 1497, 14756, 16513, 41486, 44591]),
        },
        "seed0-uint16x8": {
            "storage": [0, 0],
            "bytesPerKey": 16,
            "known": ("H", [20137, 63263, 1497, 14756, 16513, 41486, 44591, 19423]),
        },
        # vmap(fn)(key) over row-contiguous keys {{0, 0}, {1, 1}}.
        "vmap-rows": {
            "storage": [0, 0, 1, 1],
            "bytesPerKey": 12,
            "known": (
                "I",
                [4146024105, 1351547692, 2718843009, 3725146706, 1802982961]
                + [1349634643],
            ),
        },
        # vmap(vmap(fn))(key) over a contiguous {3, 2, 2} key array.
        "vmap-twice": {
            "storage": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "bytesPerKey": 8,
            "known": (
                "I",
                [928981903, 3453687069, 3606183818, 460005496, 2799733733]
                + [856293553, 4081856343, 3445925136, 2775548010, 1430281703]
                + [305173070, 2615843348],
            ),
        },
        # Wrapping key schedules; even word count with one edge byte.
        "wide-keys-even": {
            "storage": [MASK32, MASK32, 0x80000000, 0x7FFFFFFF, 0x1BD11BDA, 1],
            "bytesPerKey": 13,
        },
        # Odd word count with three edge bytes.
        "wide-keys-odd": {
            "storage": (
                [0xDEADBEEF, 0x01234567, 0, MASK32, 7, 0x1BD11BDA]
                + [0xFFFF0000, 0x0000FFFF]
            ),
            "bytesPerKey": 19,
        },
        # Odd word count without edge bytes, and an even count with two.
        "odd-words": {"storage": _lcg_words(3, 10), "bytesPerKey": 28},
        "edge2": {"storage": _lcg_words(5, 10), "bytesPerKey": 22},
        # 17 keys x 129 rows: 2,193 workgroups and 17,527 bytes.
        "large": {"storage": _lcg_words(7, 34), "bytesPerKey": 1031},
    },
    "rbits": {
        # vmap(fn, 1)(key): the transposed keys select {0, 1} twice.
        "vmap-columns": {
            "storage": [0, 0, 1, 1],
            "view": ([2, 2], [1, 2]),
            "bytesPerKey": 12,
            "known": ("I", [2441914641, 1110694964, 3819641963] * 2),
        },
        # vmap(vmap(fn, 1), 0)(key) over {3, 2, 2} keys.
        "vmap-twice-columns": {
            "storage": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "view": ([3, 2, 2], [4, 1, 2]),
            "bytesPerKey": 8,
            "known": (
                "I",
                [1948878966, 4237131848, 1948878966, 4237131848, 2531170506]
                + [1858648356, 2531170506, 1858648356, 740561898, 4234094099]
                + [740561898, 4234094099],
            ),
        },
        # A zero stride broadcasts one key to three rows.
        "broadcast": {
            "storage": [0x01234567, 0x89ABCDEF],
            "view": ([3, 2], [0, 1]),
            "bytesPerKey": 9,
        },
        # Rows three words apart; reading a decoy word would change the bits.
        "row-gaps": {
            "storage": _lcg_words(11, 12),
            "view": ([4, 2], [3, 1]),
            "bytesPerKey": 22,
        },
        # A permuted three-axis view whose pairs are two words apart.
        "permuted": {
            "storage": _lcg_words(13, 12),
            "view": ([2, 3, 2], [1, 4, 2]),
            "bytesPerKey": 15,
        },
        # 12 strided keys x 33 rows with one edge byte.
        "large": {
            "storage": _lcg_words(17, 60),
            "view": ([12, 2], [5, 2]),
            "bytesPerKey": 257,
        },
    },
}
# Words that a strided view never reads stay in the storage as decoys.
DECOY_CASES = {"row-gaps", "large"}


def _threefry(key, count):
    """Mirror ``threefry2x32_hash``, including its 32-bit wraparound."""
    schedule = (key[0], key[1], key[0] ^ key[1] ^ 0x1BD11BDA)
    x = (count[0] + schedule[0]) & MASK32
    y = (count[1] + schedule[1]) & MASK32
    for round_index in range(5):
        for rotation in ROTATIONS[round_index % 2]:
            x = (x + y) & MASK32
            y = ((y << rotation) | (y >> (32 - rotation))) & MASK32
            y ^= x
        x = (x + schedule[(round_index + 1) % 3]) & MASK32
        y = (y + schedule[(round_index + 2) % 3] + round_index + 1) & MASK32
    return x, y


def _elem_to_loc(elem, shape, strides):
    """Mirror ``elem_to_loc<uint>`` with unsigned 32-bit arithmetic."""
    loc = 0
    for size, stride in zip(reversed(shape), reversed(strides)):
        if elem == 0:
            break
        loc = (loc + (elem % size) * (stride & MASK32)) & MASK32
        elem //= size
    return loc


def _keys(case):
    storage = case["storage"]
    if "view" not in case:
        assert len(storage) % 2 == 0
        return [
            tuple(storage[index : index + 2]) for index in range(0, len(storage), 2)
        ]
    shape, strides = case["view"]
    assert shape[-1] == 2
    count = 1
    for size in shape:
        count *= size
    return [
        (
            storage[_elem_to_loc(2 * index, shape, strides)],
            storage[_elem_to_loc(2 * index + 1, shape, strides)],
        )
        for index in range(count // 2)
    ]


def _grid(case):
    words = (case["bytesPerKey"] + 3) // 4
    half_size, odd = words // 2, words % 2
    return [len(_keys(case)), half_size + odd, 1], bool(odd)


def _stores(case):
    """Return ``{byte offset: byte}`` for every store the kernel performs."""
    bytes_per_key = case["bytesPerKey"]
    (key_count, rows, _), odd = _grid(case)
    half_size = rows - odd
    stores = {}
    for column, key in enumerate(_keys(case)):
        for row in range(rows):
            drop_last = odd and row == half_size
            first, second = _threefry(key, (row, 0 if drop_last else row + rows))
            words = [(row * 4, first, 4)]
            if not drop_last:
                edge = bytes_per_key % 4
                width = edge if row + 1 == half_size and edge else 4
                words.append(((row + rows) * 4, second, width))
            for offset, word, width in words:
                for lane, value in enumerate(struct.pack("<I", word)[:width]):
                    position = column * bytes_per_key + offset + lane
                    # Concurrent stores to one byte would make MLX's result racy.
                    assert position not in stores, position
                    stores[position] = value
    assert len(stores) >= key_count * bytes_per_key
    return stores


def _output_size(case):
    return len(_keys(case)) * case["bytesPerKey"] + GUARD_COUNT


def _signed(byte):
    return byte - 256 if byte >= 128 else byte


def _expected(case, guard):
    stores = _stores(case)
    return [
        _signed(stores[position]) if position in stores else guard
        for position in range(_output_size(case))
    ]


def _known_bytes(case):
    code, values = case["known"]
    return struct.pack(f"<{len(values)}{code}", *values)


def _assertions(target, entry):
    return OPENGL_INDEX_RANGE_ASSERTIONS[entry] if target == "opengl" else ()


def _config(output, target, entry):
    assertions = ""
    for expression, low, high in _assertions(target, entry):
        assertions += f"""
[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "{expression}"
minimum = {low}
maximum = {high}
"""
    return f"""
[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
output_dir = "{output}"
workgroup_size = [{", ".join(str(item) for item in WORKGROUP_SIZE)}]
[project.entry_points]
"{SOURCE}" = "{entry}"
{assertions}"""


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _case_record(case):
    record = {
        "keyCount": len(_keys(case)),
        "bytesPerKey": case["bytesPerKey"],
        "storageWords": len(case["storage"]),
        "grid": _grid(case)[0],
        "knownAnswer": "known" in case,
    }
    if "view" in case:
        record["keyShape"], record["keyStrides"] = (list(item) for item in case["view"])
    return record


def _expected_validation_contract():
    return {
        "workgroupSize": list(WORKGROUP_SIZE),
        "indexRangeAssertions": {
            target: {
                entry: [list(item) for item in _assertions(target, entry)]
                for entry in ENTRIES
            }
            for target in TARGETS
        },
        "cases": {
            entry: {name: _case_record(case) for name, case in CASES[entry].items()}
            for entry in ENTRIES
        },
        "knownAnswerSource": {
            "path": KNOWN_ANSWERS,
            "sha256": KNOWN_ANSWERS_SHA256,
            "caseCount": sum(
                "known" in case for entry in ENTRIES for case in CASES[entry].values()
            ),
        },
        "outputGuardCount": GUARD_COUNT,
        "outputGuards": {target: list(GUARDS[target]) for target in TARGETS},
        "elementTypes": {target: dict(ELEMENT_TYPES[target]) for target in TARGETS},
        "observedInputCount": {"rbitsc": 3, "rbits": 6},
        "compilerValidationRequiredInCi": list(TARGETS),
        "runtimeRequiredInCi": list(TARGETS),
        "upstreamMetallibParityTargets": ["metal"],
    }


def _expected_scope():
    return {
        "coveredEntryCount": len(ENTRIES),
        "runtimeCoveredEntryCount": len(ENTRIES),
        "caseCount": sum(len(CASES[entry]) for entry in ENTRIES),
        "byteCountByTarget": {
            target: sum(
                _output_size(case) - GUARD_COUNT
                for entry in ENTRIES
                for case in CASES[entry].values()
            )
            for target in TARGETS
        },
        "completeCorpusEntryCount": 17832,
        "upstreamMlxTestSuiteExecuted": False,
        "mlxHostRuntimeRedirectionImplemented": False,
    }


def test_current_mlx_random_contract_is_exact():
    contract = _contract()
    assert contract["schemaVersion"] == 1
    assert contract["kind"] == "crosstl-mlx-current-tree-random-contract"
    assert contract["provenance"] == {
        "repository": "https://github.com/ml-explore/mlx.git",
        "commit": MLX_COMMIT,
        "source": SOURCE,
        "sourceSha256": SOURCE_SHA256,
        "knownAnswers": KNOWN_ANSWERS,
        "knownAnswersSha256": KNOWN_ANSWERS_SHA256,
    }
    assert contract["corpusCensus"] == {
        "metalUnitCount": 49,
        "entryCount": 17832,
        "discoveryDiagnosticCount": 0,
    }
    assert tuple(contract["entries"]) == ENTRIES
    assert set(contract["artifacts"]) == set(TARGETS)
    for target in TARGETS:
        artifacts = contract["artifacts"][target]
        assert tuple(artifacts) == ENTRIES
        for record in artifacts.values():
            assert set(record) == {"sha256", "sizeBytes"}
            assert len(record["sha256"]) == 64 and record["sizeBytes"] > 0
    assert contract["validationContract"] == _expected_validation_contract()
    assert contract["scope"] == _expected_scope()


def test_random_reference_reproduces_mlx_known_answers():
    known = 0
    for entry in ENTRIES:
        for name, case in CASES[entry].items():
            if "known" not in case:
                continue
            known += 1
            stores = _stores(case)
            size = len(_keys(case)) * case["bytesPerKey"]
            assert bytes(stores[position] for position in range(size)) == (
                _known_bytes(case)
            ), name
    assert known == 15


def test_random_cases_cover_kernel_paths():
    rbitsc, rbits = CASES["rbitsc"], CASES["rbits"]
    for entry in ENTRIES:
        for name, case in CASES[entry].items():
            stores = _stores(case)
            size = len(_keys(case)) * case["bytesPerKey"]
            # Every byte of every key is stored exactly once.
            assert set(range(size)) <= set(stores), name
            overrun = set(stores) - set(range(size))
            # Only a dropped-last first word can store past a key.
            assert overrun == (
                set(range(size, 4)) if case["bytesPerKey"] < 4 else set()
            ), name
            if case["bytesPerKey"] < 4:
                assert len(_keys(case)) == 1, name
    edges = {case["bytesPerKey"] % 4 for case in rbitsc.values()}
    assert edges == {0, 1, 2, 3}
    parities = {
        (_grid(case)[1], case["bytesPerKey"] % 4 > 0) for case in rbitsc.values()
    }
    assert parities == {(False, False), (False, True), (True, False), (True, True)}
    assert _grid(rbitsc["large"])[0] == [17, 129, 1]
    assert _output_size(rbitsc["large"]) - GUARD_COUNT == 17527
    assert _grid(rbits["large"])[0] == [12, 33, 1]
    # The vmapped column views repeat their keys; the broadcast view reads one.
    assert _keys(rbits["vmap-columns"]) == [(0, 1), (0, 1)]
    assert set(_keys(rbits["broadcast"])) == {(0x01234567, 0x89ABCDEF)}
    for name in DECOY_CASES:
        case = rbits[name]
        shape, strides = case["view"]
        read = {
            _elem_to_loc(elem, shape, strides) for elem in range(2 * len(_keys(case)))
        }
        assert read < set(range(len(case["storage"]))), name
    permuted = rbits["permuted"]
    shape, strides = permuted["view"]
    assert sorted(_elem_to_loc(elem, shape, strides) for elem in range(12)) == list(
        range(12)
    )
    # Threefry adds the schedule with wraparound for the all-ones keys.
    assert _threefry((MASK32, MASK32), (MASK32, MASK32)) != _threefry((0, 0), (0, 0))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_random_configs_scope_assertions(tmp_path, target, entry):
    path = tmp_path / "crosstl.toml"
    path.write_text(_config("out", target, entry), encoding="utf-8")
    config = load_project_config(tmp_path, path)
    assert tuple(config.workgroup_size) == WORKGROUP_SIZE
    assert list(config.targets) == [target]
    assert [item.to_json() for item in config.index_range_assertions] == [
        {"source": SOURCE, "expression": expression, "minimum": low, "maximum": high}
        for expression, low, high in _assertions(target, entry)
    ]


@pytest.fixture(scope="module")
def current_mlx():
    root_value = os.environ.get("CROSTL_MLX_CURRENT_ROOT")
    target = os.environ.get("CROSTL_MLX_CURRENT_TARGET")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native random gate")
    assert root_value and target in TARGETS
    for name in {
        "directx": ["dxc"],
        "opengl": ["glslangValidator", "spirv-val"],
        "metal": ["xcrun"],
    }[target]:
        assert shutil.which(name), f"Required tool is missing: {name}"
    root = Path(root_value).resolve()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    assert revision == MLX_COMMIT
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
            KNOWN_ANSWERS,
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert hashlib.sha256((root / SOURCE).read_bytes()).hexdigest() == SOURCE_SHA256
    assert (
        hashlib.sha256((root / KNOWN_ANSWERS).read_bytes()).hexdigest()
        == KNOWN_ANSWERS_SHA256
    )
    return root, target


def test_random_known_answers_are_upstream_literals(current_mlx):
    root, _ = current_mlx
    text = (root / KNOWN_ANSWERS).read_text(encoding="utf-8")
    for entry in ENTRIES:
        for name, case in CASES[entry].items():
            if "known" in case:
                for value in case["known"][1]:
                    assert re.search(rf"\b{value}u?\b", text), (name, value)


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("random-metal-reference")
    runner = work / "readback"
    _run(
        [
            "xcrun",
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            runner,
        ],
        work,
        "build-runner",
    )
    return runner, _metal_library(
        root / SOURCE, work / "upstream.metallib", root, upstream=True
    )


def _upstream_metal(work, reference, entry, case, name, guard):
    runner, library = reference
    grid, odd = _grid(case)
    payloads = [
        struct.pack(f"<{len(case['storage'])}I", *case["storage"]),
        struct.pack(f"<{_output_size(case)}b", *([guard] * _output_size(case))),
        struct.pack("<?", odd),
        struct.pack("<Q", case["bytesPerKey"]),
    ]
    if entry == "rbits":
        shape, strides = case["view"]
        payloads += [
            struct.pack("<i", len(shape)),
            struct.pack(f"<{len(shape)}i", *shape),
            struct.pack(f"<{len(strides)}q", *strides),
        ]
    files = []
    for index, payload in enumerate(payloads):
        path = work / f"{name}-upstream-input-{index}.bin"
        path.write_bytes(payload)
        files.append(str(path))
    request = work / f"{name}-upstream-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": grid,
                "workgroupSize": list(WORKGROUP_SIZE),
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    output = work / f"{name}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, output], work, f"{name}-upstream")
    )
    raw = (output / "buffer-1.bin").read_bytes()
    for index, payload in enumerate(payloads):
        if index != 1:
            assert (output / f"buffer-{index}.bin").read_bytes() == payload
    return device, list(struct.unpack(f"<{len(raw)}b", raw))


def _dispatch(descriptor, package, target, entry, case, guard):
    grid, odd = _grid(case)
    size = _output_size(case)
    data = {
        "keys": {
            "dtype": "uint32",
            "shape": [len(case["storage"])],
            "values": list(case["storage"]),
        },
        "out_": {
            "dtype": ELEMENT_TYPES[target]["out_"],
            "shape": [size],
            "values": [guard] * size,
        },
        "odd": {
            "dtype": ELEMENT_TYPES[target]["odd"],
            "shape": [1],
            "values": [odd if target == "metal" else int(odd)],
        },
        "bytes_per_key": {
            "dtype": "uint64",
            "shape": [1],
            "values": [case["bytesPerKey"]],
        },
    }
    if entry == "rbits":
        shape, strides = case["view"]
        data["ndim"] = {"dtype": "int32", "shape": [1], "values": [len(shape)]}
        data["key_shape"] = {
            "dtype": "int32",
            "shape": [len(shape)],
            "values": list(shape),
        }
        data["key_strides"] = {
            "dtype": "int64",
            "shape": [len(strides)],
            "values": list(strides),
        }
    inputs, observed, output_name = {}, [], None
    for binding in descriptor["bindings"]:
        if binding.get("provenance", {}).get("kind") == "generated-execution-input":
            assert target == "directx"
            continue
        layout = binding["scalarLayout"]
        # DirectX constant members carry the sanitized entry name as a prefix.
        name = layout.get("memberName", binding["name"]).removeprefix(entry + "_")
        value = data[name]
        assert layout["elementType"] == value["dtype"], (name, layout)
        inputs[binding["name"]] = value
        if name == "out_":
            output_name = binding["name"]
        else:
            observed.append(binding["name"])
    assert output_name is not None and len(inputs) == len(data)
    assert len(observed) == _expected_validation_contract()["observedInputCount"][entry]
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        {output_name: dict(data["out_"])},
        {"workgroupCount": grid, "workgroupSize": list(WORKGROUP_SIZE)},
        expected_target=target,
        observe_inputs=observed,
    )
    assert request.execution_plan is not None
    assert request.execution_plan.diagnostics == ()
    return request, output_name, len(observed)


@pytest.mark.parametrize("entry", ENTRIES)
def test_current_mlx_random_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    record = _contract()["artifacts"][target][entry]
    with tempfile.TemporaryDirectory(prefix=".current-random-", dir=root) as directory:
        work = Path(directory)
        try:
            config = work / "crosstl.toml"
            config.write_text(
                _config(f"{work.name}/out", target, entry), encoding="utf-8"
            )
            translation = translate_project(
                load_project_config(root, config),
                format_output=False,
                validate=True,
                run_toolchains=False,
            )
            report = translation.to_json()
            translation.write_json(work / "report.json")
            validation = validate_project_report(work / "report.json")
            assert validation["success"] is True, validation
            assert report["diagnostics"] == []
            assert report["summary"]["translatedCount"] == 1
            assert report["summary"]["failedCount"] == 0
            (artifact,) = report["artifacts"]
            assert artifact["sourceHash"] == {
                "algorithm": "sha256",
                "value": SOURCE_SHA256,
            }
            check_generated_artifact(
                artifact, sha256=record["sha256"], size_bytes=record["sizeBytes"]
            )
            execution = artifact.get("execution")
            if target == "metal":
                # Metal selects the threadgroup size when the kernel is dispatched.
                assert execution is None
            else:
                assert execution["workgroupSize"] == list(WORKGROUP_SIZE)
                assert execution["sourceEntryPoints"] == [entry]
            generated = root / artifact["path"]
            text = generated.read_text(encoding="utf-8")
            assert "unsupported" not in text
            if target == "metal":
                _metal_library(generated, work / "translated.metallib", root)
            else:
                _validate_generated_artifact(generated, work, target)
            descriptor, package = _prepare_native_package(translation, work)
            executor = _executor(target)
            evidence = []
            try:
                for name, case in CASES[entry].items():
                    for guard in GUARDS[target]:
                        label = f"{name}-guard{guard & 0xFF:02x}"
                        expected = _expected(case, guard)
                        request, output_name, observed = _dispatch(
                            descriptor, package, target, entry, case, guard
                        )
                        availability = executor.is_available(request)
                        assert availability.available, availability.reason
                        result = executor.run(request)
                        (work / f"{label}-result.json").write_text(
                            json.dumps(
                                {
                                    "status": result.status,
                                    "outputs": result.outputs,
                                    "details": result.details,
                                },
                                indent=2,
                                default=str,
                            ),
                            encoding="utf-8",
                        )
                        assert result.status == "ok"
                        output = result.outputs[output_name]
                        assert output["dtype"] == ELEMENT_TYPES[target]["out_"]
                        mismatches = [
                            (index, got, want)
                            for index, (got, want) in enumerate(
                                zip(output["values"], expected)
                            )
                            if got != want
                        ]
                        assert len(output["values"]) == len(expected)
                        assert not mismatches, (label, mismatches[:16])
                        observations = result.details["inputObservations"]
                        assert len(observations) == observed
                        assert all(item["unchanged"] for item in observations.values())
                        upstream = None
                        if target == "metal":
                            device, values = _upstream_metal(
                                work, metal_reference, entry, case, label, guard
                            )
                            assert values == expected
                            upstream = {"device": device, "matched": True}
                        evidence.append(
                            {
                                "case": name,
                                "guard": guard,
                                **_case_record(case),
                                "outputGuardCount": GUARD_COUNT,
                                "observedInputCount": observed,
                                "upstream": upstream,
                            }
                        )
            finally:
                close = getattr(executor.runtime_adapter.runtime, "close", None)
                if callable(close):
                    close()
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entry": entry,
                        "target": target,
                        "artifactSha256": record["sha256"],
                        "cases": evidence,
                        "indexRangeAssertions": [
                            list(item) for item in _assertions(target, entry)
                        ],
                        "upstreamMlxTestSuiteExecuted": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)


def test_ci_requires_native_random_parity():
    from tests import workflow_text
    from tests.ci_helpers import (
        assert_native_step_prerequisites,
        assert_workflow_triggers,
    )

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = workflow_text.workflow_job_step_section(
        workflow,
        "mlx-metal-porting",
        "Prove current MLX random native parity",
    )
    assert_native_step_prerequisites(step, job="mlx-metal-porting")
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "Linux) export CROSTL_MLX_CURRENT_TARGET=opengl",
        "Windows) export CROSTL_MLX_CURRENT_TARGET=directx",
        "macOS) export CROSTL_MLX_CURRENT_TARGET=metal",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/random",
        "--junitxml=mlx-current-results/random.xml",
        "demos/integrations/mlx/tests/kernels/test_current_random.py",
    ):
        assert required in step
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_random.py"
    )
    checkout = workflow_text.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Checkout current MLX kernel tree"
    )
    # The known-answer source lives in MLX's C++ test directory.
    assert (
        "sparse-checkout set mlx/backend/metal/kernels python/tests tests" in checkout
    )
    upload = workflow_text.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Upload current MLX runtime diagnostics"
    )
    assert "if: always()" in upload
    assert "path: mlx-current-results" in upload
    assert workflow.index("Prove current MLX arange native parity") < (
        workflow.index("Prove current MLX random native parity")
    )
    assert workflow.index("Prove current MLX random native parity") < (
        workflow.index("Upload current MLX runtime diagnostics")
    )

"""Deterministic public MLX random workloads and independent word references."""

import math

from demos.integrations.mlx.random_audit import threefry

LAYOUTS = ("single", "contiguous", "transposed", "broadcast", "strided", "reversed")
KEYS = ((0, 0), (0xFFFFFFFF, 0x80000000), (0x01020304, 0xFEDCBA98))


def cases():
    for layout in LAYOUTS:
        for count in (0, 1, 3, 17):
            yield {
                "id": f"split-{layout}-{count}",
                "kind": "split",
                "layout": layout,
                "count": count,
            }
    for seed in (0, 42, 0xFFFFFFFF00000001):
        for count in (0, 1, 2, 3, 7, 17):
            yield {
                "id": f"uniform-{seed}-{count}",
                "kind": "uniform",
                "seed": seed,
                "count": count,
            }

    for layout, count in (
        ("single", 8192),
        ("single", 32768),
        ("single", 65535),
        ("contiguous", 32768),
        ("strided", 32768),
        ("broadcast", 32768),
    ):
        yield {
            "id": f"split-{layout}-{count}",
            "kind": "split",
            "layout": layout,
            "count": count,
        }


def keys(xp, layout):
    source = xp.array(KEYS, dtype=xp.uint32)
    if layout == "single":
        return xp.array(KEYS[0], dtype=xp.uint32)
    if layout == "contiguous":
        return source
    if layout == "transposed":
        return xp.array(list(zip(*KEYS)), dtype=xp.uint32).T
    if layout == "broadcast":
        return xp.broadcast_to(source[:1], (3, 2))
    if layout == "strided":
        return xp.array([[a, 91, b, 92] for a, b in KEYS], dtype=xp.uint32)[:, ::2]
    if layout == "reversed":
        return source[:, ::-1]
    raise ValueError("Unknown random key layout")


def words(key, count):
    result = [0] * count
    width = (count + 1) // 2
    for i in range(width):
        drop = count % 2 and i == width - 1
        pair = threefry(
            tuple(int(value) for value in key), (i, 0 if drop else i + width)
        )
        result[i] = pair[0]
        if not drop:
            result[i + width] = pair[1]
    return result


def expected(np, case):
    count = case["count"]
    if case["kind"] == "split":
        key = keys(np, case["layout"])
        result = [words(pair, count * 2) for pair in key.reshape(-1, 2)]
        shape = (count, 2) if case["layout"] == "single" else (3, count, 2)
        return np.array(result, dtype=np.uint32).reshape(shape)
    seed = case["seed"]
    raw = np.array(words((seed >> 32, seed & 0xFFFFFFFF), count), dtype=np.uint32)
    return np.minimum(
        raw.astype(np.float32) / np.float32(2**32),
        np.nextafter(np.float32(1), np.float32(0)),
    )


def run(mx, np, host, observe):
    records = []
    for case in cases():
        start = host.dispatch_count if host else 0
        if case["kind"] == "split":
            key = keys(mx, case["layout"])
            operation = lambda key: mx.random.split(key, case["count"])
            result = (
                operation(key)
                if case["layout"] == "single"
                else mx.vmap(operation)(key)
            )
        else:
            result = mx.random.uniform(
                shape=(case["count"],), key=mx.random.key(case["seed"])
            )
        mx.eval(result)
        raw = np.array(result)
        record = {
            **case,
            "shape": list(raw.shape),
            "dtype": str(raw.dtype),
            "words": raw.view(np.uint32).reshape(-1).tolist(),
            "dispatchStart": start,
            "dispatchCount": host.dispatch_count - start if host else 0,
        }
        records.append(record)
        observe(records)
    return records


def validate(np, records, trace, *, native):
    inventory = list(cases())
    if len(records) != len(inventory):
        raise ValueError("Random workload inventory is incomplete")
    cursor = 0
    for case, record in zip(inventory, records):
        reference = expected(np, case)
        count = record.get("dispatchCount")
        if (
            any(record.get(key) != value for key, value in case.items())
            or record.get("shape") != list(reference.shape)
            or record.get("dtype") != str(reference.dtype)
            or record.get("words") != reference.view(np.uint32).reshape(-1).tolist()
            or any(type(word) is not int for word in record.get("words", []))
            or record.get("dispatchStart") != cursor
            or type(count) is not int
            or count < 0
            or (not native and count != 0)
        ):
            raise ValueError("Random workload output or dispatch record differs")
        if native:
            events = trace[cursor : cursor + count]
            random = [
                event for event in events if event["entry"] in {"rbits", "rbitsc"}
            ]
            if len(events) != count or len(random) != int(reference.size > 0):
                raise ValueError("Random workload did not execute its generated kernel")
            if random:
                event = random[0]
                key_count = 3 if case.get("layout") not in (None, "single") else 1
                expected_bytes = reference.size * 4
                metadata = event["randomMetadata"]
                if (
                    metadata["keyCount"] != key_count
                    or event["threads"] != expected_bytes
                ):
                    raise ValueError("Random workload key or output allocation differs")
                if case["kind"] == "split":
                    logical = bytes(value & 255 for value in event["randomValues"])
                    if logical != reference.tobytes():
                        raise ValueError(
                            "Random native bytes differ from the MLX array"
                        )
                expected_words = math.prod(reference.shape) // key_count
                if metadata["wordCount"] != expected_words:
                    raise ValueError("Random workload counter selection differs")
        cursor += count
    if native and cursor != len(trace):
        raise ValueError("Random workload contains unaccounted dispatches")

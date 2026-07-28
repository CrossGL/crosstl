import copy
import errno
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

import pytest

from crosstl.project import native_deferred_compilation_cache as cache_module
from crosstl.project.native_deferred_compilation import (
    build_native_deferred_compilation_request,
)
from crosstl.project.native_deferred_compilation_cache import (
    NATIVE_DEFERRED_COMPILATION_CACHE_ENTRY_KIND,
    NATIVE_DEFERRED_COMPILATION_CACHE_ENTRY_VERSION,
    NativeDeferredCompilationCacheError,
    lookup_native_deferred_compilation_cache,
    publish_native_deferred_compilation_cache,
)
from crosstl.project.pipeline import encode_runtime_variant_key


def _sha256(content):
    return {"algorithm": "sha256", "value": hashlib.sha256(content).hexdigest()}


def _request(backend="directx", *, element_count=4, descriptor_revision="v1"):
    if backend == "directx":
        source_path = "shaders/copy.hlsl"
        source_format = "HLSL source"
        profile = "cs_6_6"
        entry_point = "copy"
        output_format = "DXIL binary"
    else:
        source_path = "shaders/copy.glsl"
        source_format = "GLSL source"
        profile = None
        entry_point = "main"
        output_format = "SPIR-V binary"
    source_content = f"source:{backend}:{element_count}".encode()
    descriptor_content = (
        f"descriptor:{backend}:{element_count}:{descriptor_revision}".encode()
    )
    execution = {"workgroupSize": [4, 1, 1], "subgroupWidth": None}
    type_arguments = {"T": "float"}
    value_arguments = {"N": element_count}
    specialization_values = [{"id": 7, "name": "mode", "value": 11}]
    compile_defines = {"COPY_MODE": "1"}
    key = encode_runtime_variant_key(
        source_path,
        entry_point,
        backend,
        target_profile=profile,
        execution=execution,
        type_arguments=type_arguments,
        value_arguments=value_arguments,
        specialization_constants=specialization_values,
        defines=compile_defines,
    )
    return build_native_deferred_compilation_request(
        {
            "path": source_path,
            "format": source_format,
            "hash": _sha256(source_content),
            "sizeBytes": len(source_content),
        },
        [],
        {
            "backend": backend,
            "profile": profile,
            "stage": "compute",
            "entryPoint": entry_point,
            "outputFormat": output_format,
        },
        {
            "key": key,
            "typeArguments": type_arguments,
            "valueArguments": value_arguments,
            "compileDefines": compile_defines,
            "specializationValues": specialization_values,
            "execution": execution,
        },
        {
            "path": f"interfaces/{backend}-copy.json",
            "hash": _sha256(descriptor_content),
            "sizeBytes": len(descriptor_content),
        },
    )


def _toolchain(
    *,
    name="reference-compiler",
    version="1.2.3",
    executable=b"compiler-executable",
):
    return {
        "name": name,
        "version": version,
        "executableHash": _sha256(executable),
    }


def _rewrite_entry(hit, mutate):
    entry = copy.deepcopy(hit["entry"])
    mutate(entry)
    Path(hit["entryPath"]).write_text(
        json.dumps(entry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _error_code(exc):
    prefix = "project.native-deferred-compilation-cache."
    assert exc.value.code.startswith(prefix)
    return exc.value.code[len(prefix) :]


def test_cache_error_has_stable_json():
    error = NativeDeferredCompilationCacheError(
        "entry-invalid",
        "Entry is invalid.",
        path="$.cacheEntry",
        details={"reason": "test"},
    )

    assert str(error) == (
        "$.cacheEntry: Entry is invalid. "
        "(project.native-deferred-compilation-cache.entry-invalid)"
    )
    assert error.to_json() == {
        "severity": "error",
        "code": "project.native-deferred-compilation-cache.entry-invalid",
        "message": "Entry is invalid.",
        "path": "$.cacheEntry",
        "details": {"reason": "test"},
    }


def test_lookup_clean_miss_does_not_create_cache_root(tmp_path):
    root = tmp_path / "cache"

    assert (
        lookup_native_deferred_compilation_cache(
            root,
            _request(),
            _toolchain(),
        )
        is None
    )
    assert not root.exists()


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_publish_and_lookup_verified_success_entry(tmp_path, backend):
    root = tmp_path / "cache"
    request = _request(backend)
    toolchain = _toolchain()
    output = f"compiled:{backend}".encode()

    published = publish_native_deferred_compilation_cache(
        root,
        request,
        toolchain,
        output,
    )
    looked_up = lookup_native_deferred_compilation_cache(
        root,
        request,
        toolchain,
    )

    assert looked_up == published
    assert published["outputBytes"] == output
    assert Path(published["entryPath"]).is_file()
    assert Path(published["outputPath"]).read_bytes() == output
    entry = published["entry"]
    assert entry["kind"] == NATIVE_DEFERRED_COMPILATION_CACHE_ENTRY_KIND
    assert entry["schemaVersion"] == NATIVE_DEFERRED_COMPILATION_CACHE_ENTRY_VERSION
    assert entry["success"] is True
    assert entry["requestHash"] == request["requestHash"]
    assert entry["toolchain"] == toolchain
    assert entry["target"] == {"outputFormat": request["target"]["outputFormat"]}
    assert entry["expectedInterfaceIdentity"] == request["expectedLoaderDescriptor"]
    assert entry["output"] == {
        "relativePath": f"outputs/{hashlib.sha256(output).hexdigest()}.bin",
        "format": request["target"]["outputFormat"],
        "sizeBytes": len(output),
        "hash": _sha256(output),
    }
    entry_parts = PurePosixPath(
        Path(published["entryPath"]).relative_to(root).as_posix()
    ).parts
    assert entry_parts[:2] == (
        "entries",
        f"v{NATIVE_DEFERRED_COMPILATION_CACHE_ENTRY_VERSION}",
    )
    assert len(entry_parts[2]) == 2
    assert len(entry_parts[3]) == 2
    assert len(entry_parts[4]) == 64
    assert entry_parts[5] == "entry.json"
    assert all(character in "0123456789abcdef" for character in entry_parts[4])


def test_repeated_identical_publish_reuses_existing_entry(tmp_path, monkeypatch):
    request = _request()
    toolchain = _toolchain()
    output = b"compiled-output"
    first = publish_native_deferred_compilation_cache(
        tmp_path,
        request,
        toolchain,
        output,
    )

    def unexpected_write(path, content):
        raise AssertionError(f"Repeated publish attempted to rewrite {path}")

    monkeypatch.setattr(cache_module, "_write_new_file", unexpected_write)

    second = publish_native_deferred_compilation_cache(
        tmp_path,
        request,
        dict(reversed(list(toolchain.items()))),
        output,
    )

    assert second == first


def test_repeated_publish_rejects_different_output(tmp_path):
    request = _request()
    toolchain = _toolchain()
    publish_native_deferred_compilation_cache(
        tmp_path,
        request,
        toolchain,
        b"first-output",
    )

    with pytest.raises(NativeDeferredCompilationCacheError) as exc:
        publish_native_deferred_compilation_cache(
            tmp_path,
            request,
            toolchain,
            b"other-output",
        )

    assert _error_code(exc) == "output-conflict"
    assert exc.value.path == "$.outputBytes"


def test_distinct_requests_do_not_collide(tmp_path):
    toolchain = _toolchain()
    first = publish_native_deferred_compilation_cache(
        tmp_path,
        _request(element_count=4),
        toolchain,
        b"compiled-four",
    )
    second = publish_native_deferred_compilation_cache(
        tmp_path,
        _request(element_count=8),
        toolchain,
        b"compiled-eight",
    )

    assert first["entry"]["requestHash"] != second["entry"]["requestHash"]
    assert first["entry"]["cacheKey"] != second["entry"]["cacheKey"]
    assert first["entryPath"] != second["entryPath"]


def test_distinct_expected_interfaces_do_not_collide(tmp_path):
    toolchain = _toolchain()
    first = publish_native_deferred_compilation_cache(
        tmp_path,
        _request(descriptor_revision="v1"),
        toolchain,
        b"compiled-v1",
    )
    second = publish_native_deferred_compilation_cache(
        tmp_path,
        _request(descriptor_revision="v2"),
        toolchain,
        b"compiled-v2",
    )

    assert (
        first["entry"]["expectedInterfaceIdentity"]
        != second["entry"]["expectedInterfaceIdentity"]
    )
    assert first["entry"]["requestHash"] != second["entry"]["requestHash"]
    assert first["entry"]["cacheKey"] != second["entry"]["cacheKey"]


@pytest.mark.parametrize(
    "toolchain",
    [
        _toolchain(name="compiler-a"),
        _toolchain(version="2.0"),
        _toolchain(executable=b"other-executable"),
    ],
)
def test_complete_toolchain_identity_separates_cache_entries(tmp_path, toolchain):
    request = _request()
    baseline = publish_native_deferred_compilation_cache(
        tmp_path,
        request,
        _toolchain(),
        b"baseline",
    )
    distinct = publish_native_deferred_compilation_cache(
        tmp_path,
        request,
        toolchain,
        b"distinct",
    )

    assert baseline["entry"]["cacheKey"] != distinct["entry"]["cacheKey"]
    assert baseline["entryPath"] != distinct["entryPath"]


@pytest.mark.parametrize(
    ("mutate", "code", "path"),
    [
        (
            lambda value: value.update({"extra": "unsupported"}),
            "toolchain-identity-invalid",
            "$.toolchainIdentity",
        ),
        (
            lambda value: value.__setitem__("name", ""),
            "toolchain-name-invalid",
            "$.toolchainIdentity.name",
        ),
        (
            lambda value: value.__setitem__("name", " \t"),
            "toolchain-name-invalid",
            "$.toolchainIdentity.name",
        ),
        (
            lambda value: value.__setitem__("name", " compiler "),
            "toolchain-name-invalid",
            "$.toolchainIdentity.name",
        ),
        (
            lambda value: value.__setitem__("version", ""),
            "toolchain-version-invalid",
            "$.toolchainIdentity.version",
        ),
        (
            lambda value: value.__setitem__("version", "1.0\n"),
            "toolchain-version-invalid",
            "$.toolchainIdentity.version",
        ),
        (
            lambda value: value.__setitem__(
                "executableHash",
                {
                    "algorithm": "sha256",
                    "value": _sha256(b"compiler")["value"].upper(),
                },
            ),
            "hash-invalid",
            "$.toolchainIdentity.executableHash",
        ),
        (
            lambda value: value.__setitem__(
                "executableHash",
                {"algorithm": "sha256", "value": "0" * 64, "size": 1},
            ),
            "hash-invalid",
            "$.toolchainIdentity.executableHash",
        ),
    ],
)
def test_rejects_invalid_toolchain_identity(
    tmp_path,
    mutate,
    code,
    path,
):
    toolchain = _toolchain()
    mutate(toolchain)

    with pytest.raises(NativeDeferredCompilationCacheError) as exc:
        lookup_native_deferred_compilation_cache(
            tmp_path,
            _request(),
            toolchain,
        )

    assert _error_code(exc) == code
    assert exc.value.path == path


@pytest.mark.parametrize("output", [b"", bytearray(b"compiled"), "compiled", None])
def test_publish_accepts_only_nonempty_bytes(tmp_path, output):
    with pytest.raises(NativeDeferredCompilationCacheError) as exc:
        publish_native_deferred_compilation_cache(
            tmp_path,
            _request(),
            _toolchain(),
            output,
        )

    assert _error_code(exc) in {"output-empty", "output-invalid"}


def test_request_contract_failure_is_wrapped(tmp_path):
    request = _request()
    request["requestHash"]["value"] = request["requestHash"]["value"].upper()

    with pytest.raises(NativeDeferredCompilationCacheError) as exc:
        lookup_native_deferred_compilation_cache(
            tmp_path,
            request,
            _toolchain(),
        )

    assert _error_code(exc) == "request-invalid"
    assert exc.value.path == "$.request"
    assert "error" in exc.value.details


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda entry: entry.update({"unknownField": True}),
            "entry-schema-invalid",
        ),
        (
            lambda entry: entry.pop("target"),
            "entry-schema-invalid",
        ),
        (
            lambda entry: entry.__setitem__("kind", "other"),
            "entry-kind-invalid",
        ),
        (
            lambda entry: entry.__setitem__("schemaVersion", 2),
            "entry-version-invalid",
        ),
        (
            lambda entry: entry.__setitem__("success", False),
            "entry-not-successful",
        ),
        (
            lambda entry: entry["cacheKey"].__setitem__("value", "0" * 64),
            "entry-cache-key-mismatch",
        ),
        (
            lambda entry: entry["requestHash"].__setitem__("value", "0" * 64),
            "entry-request-mismatch",
        ),
        (
            lambda entry: entry.__setitem__("requestKind", "other"),
            "entry-request-kind-invalid",
        ),
        (
            lambda entry: entry["toolchain"].__setitem__("version", "stale"),
            "entry-toolchain-mismatch",
        ),
        (
            lambda entry: entry["target"].__setitem__("outputFormat", "stale-format"),
            "entry-output-format-mismatch",
        ),
        (
            lambda entry: entry["expectedInterfaceIdentity"]["hash"].__setitem__(
                "value", "0" * 64
            ),
            "entry-interface-mismatch",
        ),
        (
            lambda entry: entry["output"].__setitem__("sizeBytes", 0),
            "entry-output-size-invalid",
        ),
        (
            lambda entry: entry["output"].__setitem__("sizeBytes", 9),
            "entry-output-identity-mismatch",
        ),
        (
            lambda entry: entry["output"]["hash"].__setitem__(
                "value", entry["output"]["hash"]["value"].upper()
            ),
            "hash-invalid",
        ),
        (
            lambda entry: entry["output"].update({"unknownField": True}),
            "entry-output-invalid",
        ),
        (
            lambda entry: entry["output"].__setitem__(
                "relativePath", "../../outside.bin"
            ),
            "entry-output-path-invalid",
        ),
    ],
)
def test_rejects_malformed_or_stale_entry_metadata(
    tmp_path,
    mutate,
    code,
):
    request = _request()
    toolchain = _toolchain()
    hit = publish_native_deferred_compilation_cache(
        tmp_path,
        request,
        toolchain,
        b"compiled",
    )
    _rewrite_entry(hit, mutate)

    with pytest.raises(NativeDeferredCompilationCacheError) as exc:
        lookup_native_deferred_compilation_cache(
            tmp_path,
            request,
            toolchain,
        )

    assert _error_code(exc) == code


@pytest.mark.parametrize(
    "raw_json",
    [
        b"{not-json",
        b'{"kind":"a","kind":"b"}',
        b'{"value":NaN}',
        b"\xff",
        b"[]",
    ],
)
def test_rejects_noncanonical_entry_json(tmp_path, raw_json):
    request = _request()
    toolchain = _toolchain()
    hit = publish_native_deferred_compilation_cache(
        tmp_path,
        request,
        toolchain,
        b"compiled",
    )
    Path(hit["entryPath"]).write_bytes(raw_json)

    with pytest.raises(NativeDeferredCompilationCacheError) as exc:
        lookup_native_deferred_compilation_cache(
            tmp_path,
            request,
            toolchain,
        )

    assert _error_code(exc) in {
        "entry-json-invalid",
        "entry-schema-invalid",
    }


def test_rejects_same_size_output_tampering(tmp_path):
    request = _request()
    toolchain = _toolchain()
    hit = publish_native_deferred_compilation_cache(
        tmp_path,
        request,
        toolchain,
        b"compiled-a",
    )
    Path(hit["outputPath"]).write_bytes(b"compiled-b")

    with pytest.raises(NativeDeferredCompilationCacheError) as exc:
        lookup_native_deferred_compilation_cache(
            tmp_path,
            request,
            toolchain,
        )

    assert _error_code(exc) == "entry-output-identity-mismatch"
    assert exc.value.details["expected"]["sizeBytes"] == len(b"compiled-a")
    assert exc.value.details["observed"]["sizeBytes"] == len(b"compiled-b")
    assert (
        exc.value.details["expected"]["hash"] != exc.value.details["observed"]["hash"]
    )


def test_rejects_missing_manifest_or_output_as_corruption(tmp_path):
    request = _request()
    toolchain = _toolchain()
    hit = publish_native_deferred_compilation_cache(
        tmp_path / "missing-manifest",
        request,
        toolchain,
        b"compiled",
    )
    Path(hit["entryPath"]).unlink()

    with pytest.raises(NativeDeferredCompilationCacheError) as manifest_exc:
        lookup_native_deferred_compilation_cache(
            tmp_path / "missing-manifest",
            request,
            toolchain,
        )
    assert _error_code(manifest_exc) == "entry-incomplete"

    hit = publish_native_deferred_compilation_cache(
        tmp_path / "missing-output",
        request,
        toolchain,
        b"compiled",
    )
    Path(hit["outputPath"]).unlink()

    with pytest.raises(NativeDeferredCompilationCacheError) as output_exc:
        lookup_native_deferred_compilation_cache(
            tmp_path / "missing-output",
            request,
            toolchain,
        )
    assert _error_code(output_exc) == "entry-incomplete"


def test_rejects_cache_root_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "cache"
    try:
        root.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    with pytest.raises(NativeDeferredCompilationCacheError) as error:
        lookup_native_deferred_compilation_cache(
            root,
            _request(),
            _toolchain(),
        )

    assert _error_code(error) == "cache-root-symlink"


def test_rejects_entry_directory_symlink_escape(tmp_path):
    request = _request()
    toolchain = _toolchain()
    hit = publish_native_deferred_compilation_cache(
        tmp_path / "cache",
        request,
        toolchain,
        b"compiled",
    )
    entry_directory = Path(hit["entryPath"]).parent
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.rmtree(entry_directory)
    try:
        entry_directory.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    with pytest.raises(NativeDeferredCompilationCacheError) as error:
        lookup_native_deferred_compilation_cache(
            tmp_path / "cache",
            request,
            toolchain,
        )

    assert _error_code(error) == "path-symlink"


def test_rejects_output_symlink_escape(tmp_path):
    request = _request()
    toolchain = _toolchain()
    hit = publish_native_deferred_compilation_cache(
        tmp_path / "cache",
        request,
        toolchain,
        b"compiled",
    )
    output_path = Path(hit["outputPath"])
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"compiled")
    output_path.unlink()
    try:
        output_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")

    with pytest.raises(NativeDeferredCompilationCacheError) as error:
        lookup_native_deferred_compilation_cache(
            tmp_path / "cache",
            request,
            toolchain,
        )

    assert _error_code(error) == "path-symlink"


def test_failed_atomic_publication_leaves_a_clean_miss(
    tmp_path,
    monkeypatch,
):
    request = _request()
    toolchain = _toolchain()

    def fail_rename(source, destination):
        raise OSError(errno.EACCES, "publication denied")

    monkeypatch.setattr(cache_module, "_rename_new_entry", fail_rename)
    with pytest.raises(NativeDeferredCompilationCacheError) as error:
        publish_native_deferred_compilation_cache(
            tmp_path,
            request,
            toolchain,
            b"compiled",
        )

    assert _error_code(error) == "publish-failed"
    assert (
        lookup_native_deferred_compilation_cache(
            tmp_path,
            request,
            toolchain,
        )
        is None
    )
    assert not [
        path
        for path in tmp_path.rglob("*")
        if path.name.startswith(".") and path.name.endswith(".tmp")
    ]


def test_existing_partial_entry_is_corruption_not_a_miss(tmp_path):
    request = _request()
    toolchain = _toolchain()
    hit = publish_native_deferred_compilation_cache(
        tmp_path,
        request,
        toolchain,
        b"compiled",
    )
    entry_directory = Path(hit["entryPath"]).parent
    shutil.rmtree(entry_directory)
    entry_directory.mkdir()
    (entry_directory / "outputs").mkdir()
    (entry_directory / "outputs" / "partial.bin").write_bytes(b"partial")

    with pytest.raises(NativeDeferredCompilationCacheError) as error:
        lookup_native_deferred_compilation_cache(
            tmp_path,
            request,
            toolchain,
        )

    assert _error_code(error) == "entry-incomplete"


def test_publish_refuses_to_replace_a_corrupt_existing_entry(tmp_path):
    request = _request()
    toolchain = _toolchain()
    hit = publish_native_deferred_compilation_cache(
        tmp_path,
        request,
        toolchain,
        b"compiled",
    )
    Path(hit["outputPath"]).write_bytes(b"corrupt!")

    with pytest.raises(NativeDeferredCompilationCacheError) as error:
        publish_native_deferred_compilation_cache(
            tmp_path,
            request,
            toolchain,
            b"compiled",
        )

    assert _error_code(error) == "entry-output-identity-mismatch"

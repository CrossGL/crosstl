import copy
import hashlib

import pytest

import crosstl.project.native_deferred_compilation_package as package_module
from crosstl.project.native_deferred_compilation import (
    build_native_deferred_compilation_request,
)
from crosstl.project.native_deferred_compilation_package import (
    NativeDeferredCompilationPackageError,
    materialize_native_deferred_compilation_inputs,
)
from crosstl.project.pipeline import encode_runtime_variant_key


def _hash(content):
    return {"algorithm": "sha256", "value": hashlib.sha256(content).hexdigest()}


def _source_record(path, content, source_format):
    return {
        "path": path,
        "format": source_format,
        "hash": _hash(content),
        "sizeBytes": len(content),
    }


def _descriptor_record(path, content):
    return {
        "path": path,
        "hash": _hash(content),
        "sizeBytes": len(content),
    }


def _request(
    files,
    *,
    backend="directx",
    source_path=None,
    include_paths=(),
    descriptor_path="abi/loader.json",
):
    source_path = source_path or (
        "shaders/main.hlsl" if backend == "directx" else "shaders/main.comp"
    )
    source_format = "HLSL source" if backend == "directx" else "GLSL source"
    output_format = "DXIL binary" if backend == "directx" else "SPIR-V binary"
    profile = "cs_6_0" if backend == "directx" else None
    entry_point = "CSMain" if backend == "directx" else "main"
    execution = {"workgroupSize": [8, 1, 1], "subgroupWidth": None}
    specializations = [{"id": 7, "name": "mode", "value": 2}]
    variant = {
        "key": encode_runtime_variant_key(
            "kernels/copy.metal",
            "copy",
            backend,
            target_profile=profile,
            execution=execution,
            type_arguments={"T": "float"},
            value_arguments={"N": 8},
            specialization_constants=specializations,
            defines={"COPY_MODE": "2"},
        ),
        "typeArguments": {"T": "float"},
        "valueArguments": {"N": 8},
        "compileDefines": {"COPY_MODE": "2"},
        "specializationValues": specializations,
        "execution": execution,
    }
    return build_native_deferred_compilation_request(
        _source_record(source_path, files[source_path], source_format),
        [_source_record(path, files[path], source_format) for path in include_paths],
        {
            "backend": backend,
            "profile": profile,
            "stage": "compute",
            "entryPoint": entry_point,
            "outputFormat": output_format,
        },
        variant,
        _descriptor_record(descriptor_path, files[descriptor_path]),
    )


def _write_files(root, files):
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _nested_package(backend):
    extension = "hlsl" if backend == "directx" else "comp"
    source_path = f"shaders/main.{extension}"
    files = {
        source_path: (
            b'/* #include "ignored.h" */\n'
            b'#include "detail/config.h" // local configuration\n'
            b"#include \\\n"
            b"  <common/math.h>\n"
            b"void kernel_body() {}\n"
        ),
        "shaders/detail/config.h": (
            b'#include "../../shared/types.h"\n#define BLOCK_SIZE 8\n'
        ),
        "shared/types.h": b"struct Element { float value; };\n",
        "vendor/common/math.h": b"#include <common/constants.h>\n",
        "vendor/common/constants.h": b"#define SCALE 2\n",
        "abi/loader.json": b'{"entryPoint":"kernel_body"}\n',
    }
    includes = (
        "vendor/common/constants.h",
        "shared/types.h",
        "shaders/detail/config.h",
        "vendor/common/math.h",
    )
    return files, source_path, includes


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_materializes_verified_nested_include_closure(backend, tmp_path):
    files, source_path, includes = _nested_package(backend)
    package_root = tmp_path / "package"
    output_root = tmp_path / "materialized"
    _write_files(package_root, files)
    output_root.mkdir()
    (output_root / "stale.bin").write_bytes(b"stale")
    request = _request(
        files,
        backend=backend,
        source_path=source_path,
        include_paths=includes,
    )

    result = materialize_native_deferred_compilation_inputs(
        request,
        package_root,
        output_root,
    )

    assert result["requestHash"] == request["requestHash"]
    assert result["target"] == request["target"]
    assert result["variant"] == request["variant"]
    assert result["outputRoot"] == str(output_root)
    assert result["sourceRoot"] == str(output_root / "source")
    assert result["source"] == {
        "packagePath": source_path,
        "path": str(output_root / "source" / source_path),
        "hash": _hash(files[source_path]),
        "sizeBytes": len(files[source_path]),
        "format": request["source"]["format"],
    }
    assert [item["packagePath"] for item in result["includes"]] == sorted(includes)
    assert result["includeDirectories"] == [str(output_root / "source" / "vendor")]
    assert result["interface"] == {
        "packagePath": "abi/loader.json",
        "path": str(output_root / "interface" / "abi/loader.json"),
        "hash": _hash(files["abi/loader.json"]),
        "sizeBytes": len(files["abi/loader.json"]),
    }
    for relative_path in (source_path, *includes):
        assert (output_root / "source" / relative_path).read_bytes() == files[
            relative_path
        ]
    assert (output_root / "interface" / "abi/loader.json").read_bytes() == files[
        "abi/loader.json"
    ]
    assert not (output_root / "stale.bin").exists()
    assert not list(tmp_path.glob(".materialized.*.tmp"))

    (output_root / "obsolete.h").write_bytes(b"obsolete")
    repeated = materialize_native_deferred_compilation_inputs(
        request,
        package_root,
        output_root,
    )
    assert repeated == result
    assert not (output_root / "obsolete.h").exists()


def test_error_json_is_stable_and_defensive():
    details = {"paths": ["first.h"]}
    error = NativeDeferredCompilationPackageError(
        "example",
        "Example failure.",
        path="$.source.path",
        details=details,
    )
    details["paths"].append("second.h")

    payload = error.to_json()

    assert payload == {
        "severity": "error",
        "code": "project.native-deferred-compilation-package.example",
        "message": "Example failure.",
        "path": "$.source.path",
        "details": {"paths": ["first.h"]},
    }
    payload["details"]["paths"].append("third.h")
    assert error.details == {"paths": ["first.h"]}


@pytest.mark.parametrize(
    ("record", "relative_path"),
    [
        ("source", "shaders/main.hlsl"),
        ("include", "include/config.h"),
        ("descriptor", "abi/loader.json"),
    ],
)
def test_rejects_same_size_hash_drift(record, relative_path, tmp_path):
    files = {
        "shaders/main.hlsl": b'#include "../include/config.h"\n',
        "include/config.h": b"#define SIZE 8\n",
        "abi/loader.json": b'{"entry":"CSMain"}\n',
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files, include_paths=("include/config.h",))
    replacement = b"x" * len(files[relative_path])
    (package_root / relative_path).write_bytes(replacement)

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            tmp_path / "out",
        )

    assert caught.value.code.endswith(".hash-mismatch")
    expected_path = {
        "source": "$.source.hash.value",
        "include": "$.includes[0].hash.value",
        "descriptor": "$.expectedLoaderDescriptor.hash.value",
    }[record]
    assert caught.value.path == expected_path
    assert (
        caught.value.details["expectedSHA256"] == _hash(files[relative_path])["value"]
    )
    assert caught.value.details["actualSHA256"] == _hash(replacement)["value"]
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("record", "relative_path"),
    [
        ("source", "shaders/main.hlsl"),
        ("include", "include/config.h"),
        ("descriptor", "abi/loader.json"),
    ],
)
def test_rejects_size_drift(record, relative_path, tmp_path):
    files = {
        "shaders/main.hlsl": b'#include "../include/config.h"\n',
        "include/config.h": b"#define SIZE 8\n",
        "abi/loader.json": b'{"entry":"CSMain"}\n',
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files, include_paths=("include/config.h",))
    (package_root / relative_path).write_bytes(files[relative_path] + b"x")

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            tmp_path / "out",
        )

    assert caught.value.code.endswith(".size-mismatch")
    expected_path = {
        "source": "$.source.sizeBytes",
        "include": "$.includes[0].sizeBytes",
        "descriptor": "$.expectedLoaderDescriptor.sizeBytes",
    }[record]
    assert caught.value.path == expected_path
    assert not (tmp_path / "out").exists()


def test_rejects_missing_package_file(tmp_path):
    files = {
        "shaders/main.hlsl": b"void CSMain() {}\n",
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files)
    (package_root / "shaders/main.hlsl").unlink()

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            tmp_path / "out",
        )

    assert caught.value.code.endswith(".package-file-missing")
    assert caught.value.path == "$.source.path"


def test_rejects_nonregular_package_file(tmp_path):
    files = {
        "shaders/main.hlsl": b'#include "../include/config.h"\n',
        "include/config.h": b"#define SIZE 8\n",
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files, include_paths=("include/config.h",))
    include_path = package_root / "include/config.h"
    include_path.unlink()
    include_path.mkdir()

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            tmp_path / "out",
        )

    assert caught.value.code.endswith(".package-file-not-regular")
    assert caught.value.path == "$.includes[0].path"


@pytest.mark.parametrize("symlink_kind", ["file", "directory"])
def test_rejects_symlinked_package_paths(symlink_kind, tmp_path):
    files = {
        "shaders/main.hlsl": b'#include "../include/config.h"\n',
        "include/config.h": b"#define SIZE 8\n",
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files, include_paths=("include/config.h",))
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        if symlink_kind == "file":
            target = outside / "config.h"
            target.write_bytes(files["include/config.h"])
            package_file = package_root / "include/config.h"
            package_file.unlink()
            package_file.symlink_to(target)
        else:
            (outside / "config.h").write_bytes(files["include/config.h"])
            include_directory = package_root / "include"
            (include_directory / "config.h").unlink()
            include_directory.rmdir()
            include_directory.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symbolic links are unavailable: {exc}")

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            tmp_path / "out",
        )

    assert caught.value.code.endswith(".package-path-symlink")
    assert caught.value.path == "$.includes[0].path"
    assert not (tmp_path / "out").exists()


def test_rejects_symlinked_package_root(tmp_path):
    files = {
        "shaders/main.hlsl": b"void CSMain() {}\n",
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    linked_root = tmp_path / "linked-package"
    try:
        linked_root.symlink_to(package_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symbolic links are unavailable: {exc}")
    request = _request(files)

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            linked_root,
            tmp_path / "out",
        )

    assert caught.value.code.endswith(".package-root-symlink")
    assert caught.value.path == "$.packageRoot"


@pytest.mark.parametrize(
    "source",
    [
        b"#include HEADER_NAME\n",
        b'#include "include/config.h" trailing_tokens\n',
        b"#include <include/config.h\n",
    ],
)
def test_rejects_dynamic_or_malformed_include_directives(source, tmp_path):
    files = {
        "shaders/main.hlsl": source,
        "include/config.h": b"#define SIZE 8\n",
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files, include_paths=("include/config.h",))

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            tmp_path / "out",
        )

    assert caught.value.code.endswith(".include-dynamic")
    assert caught.value.details["line"] == 1


def test_rejects_include_next_directive(tmp_path):
    files = {
        "shaders/main.hlsl": b"#include_next <config.h>\n",
        "include/config.h": b"#define SIZE 8\n",
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files, include_paths=("include/config.h",))

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            tmp_path / "out",
        )

    assert caught.value.code.endswith(".include-directive-unsupported")
    assert caught.value.details["directive"] == "include_next"


def test_rejects_undeclared_include(tmp_path):
    files = {
        "shaders/main.hlsl": b'#include "missing.h"\n',
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files)

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            tmp_path / "out",
        )

    assert caught.value.code.endswith(".include-undeclared")
    assert caught.value.details["resolvedPackagePath"] == "shaders/missing.h"


def test_rejects_quoted_include_package_escape(tmp_path):
    files = {
        "main.hlsl": b'#include "../outside.h"\n',
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files, source_path="main.hlsl")

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            tmp_path / "out",
        )

    assert caught.value.code.endswith(".include-path-escape")
    assert caught.value.details["include"] == "../outside.h"


def test_rejects_angle_include_traversal(tmp_path):
    files = {
        "shaders/main.hlsl": b"#include <include/../config.h>\n",
        "config.h": b"#define SIZE 8\n",
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files, include_paths=("config.h",))

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            tmp_path / "out",
        )

    assert caught.value.code.endswith(".include-path-invalid")
    assert caught.value.details["include"] == "include/../config.h"


def test_rejects_unreachable_declared_include(tmp_path):
    files = {
        "shaders/main.hlsl": b"void CSMain() {}\n",
        "include/unused.h": b"#define UNUSED 1\n",
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files, include_paths=("include/unused.h",))

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            tmp_path / "out",
        )

    assert caught.value.code.endswith(".include-unreachable")
    assert caught.value.details["unreachableIncludes"] == ["include/unused.h"]


def test_rejects_ambiguous_angle_include(tmp_path):
    files = {
        "shaders/main.hlsl": b"#include <common.h>\n",
        "first/common.h": b"#define FIRST 1\n",
        "second/common.h": b"#define SECOND 1\n",
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(
        files,
        include_paths=("first/common.h", "second/common.h"),
    )

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            tmp_path / "out",
        )

    assert caught.value.code.endswith(".include-ambiguous")
    assert caught.value.details["matches"] == [
        "first/common.h",
        "second/common.h",
    ]


def test_rejects_non_utf8_source(tmp_path):
    files = {
        "shaders/main.hlsl": b"\xff\xfe",
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files)

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            tmp_path / "out",
        )

    assert caught.value.code.endswith(".source-encoding-invalid")
    assert caught.value.details["byteOffset"] == 0


@pytest.mark.parametrize("collision", ["duplicate", "case"])
def test_rejects_duplicate_and_case_colliding_destinations(collision, tmp_path):
    files = {
        "shaders/main.hlsl": b'#include "../include/config.h"\n',
        "include/config.h": b"#define SIZE 8\n",
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files, include_paths=("include/config.h",))
    duplicate = copy.deepcopy(request["includes"][0])
    if collision == "case":
        duplicate["path"] = "INCLUDE/CONFIG.H"
    request["includes"].append(duplicate)

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            tmp_path / "out",
        )

    assert caught.value.code.endswith(".request-invalid")
    assert caught.value.details["requestCode"].endswith(
        ".include-path-duplicate"
        if collision == "duplicate"
        else ".include-path-case-collision"
    )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("output_position", ["same", "inside", "parent"])
def test_rejects_output_package_overlap(output_position, tmp_path):
    files = {
        "shaders/main.hlsl": b"void CSMain() {}\n",
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "workspace/package"
    _write_files(package_root, files)
    request = _request(files)
    output_root = {
        "same": package_root,
        "inside": package_root / "generated",
        "parent": package_root.parent,
    }[output_position]

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            output_root,
        )

    assert caught.value.code.endswith(".output-package-overlap")
    assert (package_root / "shaders/main.hlsl").read_bytes() == files[
        "shaders/main.hlsl"
    ]


@pytest.mark.parametrize("output_kind", ["file", "symlink"])
def test_rejects_unsafe_output_collisions(output_kind, tmp_path):
    files = {
        "shaders/main.hlsl": b"void CSMain() {}\n",
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files)
    output_root = tmp_path / "out"
    if output_kind == "file":
        output_root.write_bytes(b"existing")
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        try:
            output_root.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"Symbolic links are unavailable: {exc}")

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            output_root,
        )

    expected_code = (
        ".output-root-collision" if output_kind == "file" else ".output-root-symlink"
    )
    assert caught.value.code.endswith(expected_code)


@pytest.mark.parametrize("output_root", ["", ".", ".."])
def test_rejects_unsafe_output_root_names(output_root, tmp_path, monkeypatch):
    files = {
        "shaders/main.hlsl": b"void CSMain() {}\n",
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files)
    monkeypatch.chdir(tmp_path / "package")

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            output_root,
        )

    assert caught.value.code.endswith(
        ".filesystem-path-invalid" if output_root == "" else ".output-root-invalid"
    )
    assert (package_root / "shaders/main.hlsl").is_file()


def test_write_failure_invalidates_stale_output_without_partial_publication(
    tmp_path,
    monkeypatch,
):
    files = {
        "shaders/main.hlsl": b"void CSMain() {}\n",
        "abi/loader.json": b"{}\n",
    }
    package_root = tmp_path / "package"
    _write_files(package_root, files)
    request = _request(files)
    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "stale.bin").write_bytes(b"stale")

    def fail_write(_root, _item):
        raise OSError("simulated write failure")

    monkeypatch.setattr(package_module, "_write_verified_file", fail_write)

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            output_root,
        )

    assert caught.value.code.endswith(".output-write-failed")
    assert not output_root.exists()
    assert not list(tmp_path.glob(".out.*.tmp"))


@pytest.mark.parametrize("root_kind", ["missing", "file"])
def test_rejects_invalid_package_roots(root_kind, tmp_path):
    files = {
        "shaders/main.hlsl": b"void CSMain() {}\n",
        "abi/loader.json": b"{}\n",
    }
    request = _request(files)
    package_root = tmp_path / "package"
    if root_kind == "file":
        package_root.write_bytes(b"not a package")

    with pytest.raises(NativeDeferredCompilationPackageError) as caught:
        materialize_native_deferred_compilation_inputs(
            request,
            package_root,
            tmp_path / "out",
        )

    expected = (
        ".package-root-missing"
        if root_kind == "missing"
        else ".package-root-not-directory"
    )
    assert caught.value.code.endswith(expected)

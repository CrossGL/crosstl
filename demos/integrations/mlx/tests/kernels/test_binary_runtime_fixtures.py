"""Source-reference setup is shared without affecting per-case verification."""

from demos.integrations.mlx.tests.kernels import conftest


def test_binary_reference_builds_once_per_source_root(
    tmp_path, monkeypatch, binary_metal_reference
):
    runs = []
    libraries = []
    monkeypatch.setattr(conftest, "_run", lambda *args: runs.append(args))

    def compile_library(source, destination, root, *, upstream):
        libraries.append((source, destination, root, upstream))
        return destination

    monkeypatch.setattr(conftest, "_metal_library", compile_library)
    first = binary_metal_reference(tmp_path, "metal")
    assert first == binary_metal_reference(tmp_path, "metal")
    assert len(runs) == len(libraries) == 1
    assert libraries[0][2:] == (tmp_path, True)
    assert libraries[0][0] == tmp_path / conftest.MLX_BINARY_SOURCE
    assert first[1] == libraries[0][1]
    second = binary_metal_reference(tmp_path / "other-pin", "metal")
    assert second != first
    assert len(runs) == len(libraries) == 2


def test_non_metal_reference_does_not_compile(
    tmp_path, monkeypatch, binary_metal_reference
):
    def unexpected(*args, **kwargs):
        raise AssertionError("Non-Metal tests must not build a Metal source control")

    monkeypatch.setattr(conftest, "_run", unexpected)
    monkeypatch.setattr(conftest, "_metal_library", unexpected)
    for target in ("directx", "opengl"):
        assert binary_metal_reference(tmp_path, target) == (None, None)

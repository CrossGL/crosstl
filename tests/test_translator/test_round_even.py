"""roundEven, the hub's ties-to-even rounding, maps to each target's ties-to-even function."""

import pytest

import crosstl


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("directx", "round(a)"),
        ("wgsl", "round(a)"),
        ("metal", "rint(a)"),
        ("cuda", "rintf(a)"),
        ("hip", "rintf(a)"),
        ("opengl", "roundEven(a)"),
    ],
)
def test_round_even_maps_to_each_targets_ties_to_even_rounding(
    tmp_path, backend, expected
):
    path = tmp_path / "round.cgl"
    path.write_text(
        "shader S {\n    compute {\n        void main() {\n            float a = 2.5;\n"
        "            float b = roundEven(a);\n        }\n    }\n}\n"
    )
    out = crosstl.translate(str(path), backend=backend, format_output=False)
    assert expected in out
    if backend != "opengl":
        assert "roundEven" not in out

import importlib.util
import io
import json
from pathlib import Path

import pytest

from tools.coverage import docs, report

from .test_publish import _report

EXTENSION = Path(__file__).resolve().parents[2] / "docs/source/_ext/coverage_pages.py"


def _data():
    data = _report()
    data["validators"]["naga"] = "naga 30.0.1"
    data["backends"].insert(
        2,
        {
            "language": "wgsl",
            "direction": "output",
            "offered": True,
            "score": 100.0,
            "validator": "naga",
            "counts": {"supported": 1, "refused": 0, "wrong": 0, "untested": 0},
            "features": [
                {
                    "name": "abs",
                    "category": "function",
                    "group": "g",
                    "outcome": "supported",
                }
            ],
        },
    )
    data["backends"][0]["validator"] = "glslangValidator"
    return data


def test_every_language_gets_a_page_and_the_index_links_each_one():
    files = docs.pages(_data(), {})
    assert sorted(files) == ["crossgl.md", "glsl.md", "index.md", "wgsl.md"]
    index = files["index.md"]
    assert index.startswith(docs.HEADER)
    assert "| [GLSL](glsl.md) | 75.0 % | 100.0 % |" in index
    assert "| [WGSL](wgsl.md) | 100.0 % | not offered |" in index
    # The CrossGL IR row has no direction of its own; its score is its expressiveness.
    assert "| [CrossGL IR](crossgl.md) | 100.0 % |  |" in index
    toctree = index.split("```{toctree}", 1)[1]
    assert toctree.split() == [":hidden:", "glsl", "wgsl", "crossgl", "```"]


def test_a_language_page_has_its_rows_tools_feature_lists_categories_and_gaps():
    text = docs.pages(_data(), {"glsl": "backends/glsl"})["glsl.md"]
    assert "# GLSL\n" in text
    assert (
        "commit [`0123456`](https://github.com/CrossGL/crosstl/commit/0123456789abcdef)"
        in text
    )
    assert "(CrossTL 1.2.3)" in text
    assert "{doc}`backend guide </backends/glsl>`" in text
    assert (
        "| output | 75.0 % | 3 | 0 | 1 | 2 | 2 on vulkan-lavapipe | glslangValidator (16.0) |"
        in text
    )
    assert "| input | 100.0 % | 4 | 0 | 0 | 0 | not run |  |" in text
    # Only GLSL's own feature lists, each linked at its pinned commit.
    assert (
        "[GLSL built-in functions](https://github.com/KhronosGroup/GLSL/blob/0099bf83b028"
        in text
    )
    assert "GLSL atomic float extension" in text
    assert "MSL library" not in text
    assert "| function | 3/4 | 4/4 |" in text
    gaps = text.split("## Gaps", 1)[1]
    assert "### Output" in gaps and "### Input" not in gaps
    assert (
        "Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q="
        in gaps
    )
    assert "Untested, by reason:" in gaps
    assert "`fwrong0`" in gaps


def test_a_direction_crosstl_does_not_offer_is_listed_with_its_reason():
    text = docs.pages(_data(), {})["wgsl.md"]
    assert "| input | not offered | | | | | | |" in text
    assert "WGSL input is not offered: CrossTL has no WGSL source frontend." in text
    assert "| output | 100.0 % | 1 | 0 | 0 | 0 | not run | naga 30.0.1 |" in text
    assert "## Gaps" not in text
    assert "backend guide" not in text


def test_the_crossgl_page_measures_expressiveness_and_links_no_bug_issues():
    text = docs.pages(_data(), {})["crossgl.md"]
    assert "# CrossGL IR" in text
    assert "| expressiveness | 100.0 % | 2 | 0 | 0 | 5 |" in text
    assert "### Expressiveness" in text
    assert "open bug issues" not in text


def test_a_page_comes_out_of_the_published_files(tmp_path):
    for name, text in report.site(_data()).items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    files = docs.pages(docs.load(str(tmp_path)), {})
    assert files == docs.pages(report.load(tmp_path / "coverage.json"), {})
    assert "| output | 75.0 % | 3 | 0 | 1 | 2 |" in files["glsl.md"]


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _opener(files, asked):
    def opener(url, timeout):
        asked.append(url)
        name = url.removeprefix("https://example.test/coverage/")
        if name not in files:
            raise OSError(f"404 {url}")
        return _Response(files[name].encode("utf-8"))

    return opener


def _with_cell():
    """:func:`_data` with a GLSL to WGSL translation matrix cell, which the coverage branch keeps in its own file."""
    feature = {"name": "abs", "category": "function", "outcome": "supported"}
    cell = {
        "source": "glsl",
        "target": "wgsl",
        "validator": "naga",
        "executes": False,
        "score": 100.0,
        "counts": {"supported": 1, "refused": 0, "wrong": 0, "untested": 0},
        "features": [feature],
    }
    return {**_data(), "pairs": [cell]}


def test_fetch_downloads_the_report_and_every_file_it_names(tmp_path):
    files = report.site(_with_cell())
    assert "features/pairs/glsl-to-wgsl.json" in files
    asked = []
    data = docs.load("https://example.test/coverage", _opener(files, asked))
    assert asked[0] == "https://example.test/coverage/coverage.json"
    assert sorted(asked[1:]) == sorted(
        f"https://example.test/coverage/{name}"
        for name in files
        if name != "coverage.json"
    )
    assert data["pairs"] == _with_cell()["pairs"]
    assert docs.pages(data, {}) == docs.pages(_with_cell(), {})


@pytest.mark.parametrize(
    "name",
    [
        "../secret.json",
        "features/x.txt",
        "/etc/passwd",
        3,
        # A row's file is never a matrix cell's.
        "features/pairs/glsl-to-wgsl.json",
    ],
)
def test_fetch_refuses_a_file_outside_the_published_directories(tmp_path, name):
    published = json.loads(report.site(_data())["coverage.json"])
    published["backends"][0]["file"] = name
    asked = []
    opener = _opener({"coverage.json": json.dumps(published)}, asked)
    with pytest.raises(ValueError, match="not a published file"):
        docs.fetch("https://example.test/coverage/", tmp_path, opener)
    assert asked == ["https://example.test/coverage/coverage.json"]
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "name",
    [
        "features/glsl-output.json",
        "pairs/glsl-to-wgsl.json",
        "features/pairs/../x.json",
    ],
)
def test_fetch_refuses_a_cell_file_not_named_as_a_cell(tmp_path, name):
    published = json.loads(report.site(_with_cell())["coverage.json"])
    published["pairs"][0]["file"] = name
    asked = []
    opener = _opener({"coverage.json": json.dumps(published)}, asked)
    with pytest.raises(ValueError, match="not a published file"):
        docs.fetch("https://example.test/coverage/", tmp_path, opener)
    assert asked == ["https://example.test/coverage/coverage.json"]
    assert not any(tmp_path.iterdir())


def test_write_keeps_unchanged_pages_and_removes_only_stale_generated_ones(tmp_path):
    files = docs.pages(_data(), {})
    assert len(docs.write(tmp_path, files)) == len(files)
    (tmp_path / "notes.md").write_text("# Kept\n", encoding="utf-8")
    (tmp_path / "old.md").write_text(docs.HEADER + "\n# Old\n", encoding="utf-8")
    assert docs.write(tmp_path, files) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([*files, "notes.md"])
    with pytest.raises(ValueError, match="not the name of a page"):
        docs.write(tmp_path, {"../index.md": ""})


def _extension():
    pytest.importorskip("sphinx")
    spec = importlib.util.spec_from_file_location("coverage_pages", EXTENSION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_extension_writes_the_pages_and_links_existing_backend_guides(tmp_path):
    extension = _extension()
    site = tmp_path / "site"
    for name, text in report.site(_data()).items():
        (site / name).parent.mkdir(parents=True, exist_ok=True)
        (site / name).write_text(text, encoding="utf-8")
    srcdir = tmp_path / "source"
    (srcdir / "backends").mkdir(parents=True)
    (srcdir / "backends/glsl.rst").write_text("GLSL\n====\n", encoding="utf-8")
    written = extension.generate(srcdir, str(site), required=True)
    assert sorted(p.name for p in written) == [
        "crossgl.md",
        "glsl.md",
        "index.md",
        "wgsl.md",
    ]
    assert "</backends/glsl>" in (srcdir / "languages/glsl.md").read_text(
        encoding="utf-8"
    )
    assert "backend guide" not in (srcdir / "languages/wgsl.md").read_text(
        encoding="utf-8"
    )


def test_the_extension_maps_hlsl_to_the_directx_guide(tmp_path):
    extension = _extension()
    (tmp_path / "backends").mkdir()
    (tmp_path / "backends/directx.rst").write_text("x\n", encoding="utf-8")
    assert extension.guides(tmp_path, ["hlsl", "metal"]) == {"hlsl": "backends/directx"}


def test_without_the_report_the_extension_links_coverage_md_or_fails_when_required(
    tmp_path,
):
    extension = _extension()
    missing = str(tmp_path / "missing")
    extension.generate(tmp_path, missing, required=False)
    assert sorted(p.name for p in (tmp_path / "languages").iterdir()) == ["index.md"]
    text = (tmp_path / "languages/index.md").read_text(encoding="utf-8")
    assert docs.COVERAGE_MD in text and "CROSSTL_COVERAGE" in text
    from sphinx.errors import ExtensionError

    with pytest.raises(ExtensionError, match="cannot read the coverage report"):
        extension.generate(tmp_path, missing, required=True)


def test_the_documentation_build_reads_the_published_report():
    workflow = (
        Path(__file__).resolve().parents[2] / ".github/workflows/docs.yml"
    ).read_text(encoding="utf-8")
    assert "ref: coverage" in workflow
    assert 'CROSSTL_COVERAGE_REQUIRED: "1"' in workflow
    conf = (EXTENSION.parents[1] / "conf.py").read_text(encoding="utf-8")
    assert '"coverage_pages"' in conf
    index = (EXTENSION.parents[1] / "index.rst").read_text(encoding="utf-8")
    assert "   languages/index\n" in index

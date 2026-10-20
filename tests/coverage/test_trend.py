import xml.etree.ElementTree as ElementTree

from tools.coverage import chart, trend

SVG = "{http://www.w3.org/2000/svg}"
HEADER = "date,commit,language,direction,score,supported,refused,wrong,untested\n"


def _history(*lines):
    return HEADER + "".join(line + "\n" for line in lines)


# Three runs, deliberately out of order. GLSL output rises, HLSL output has no score in the middle run, the CrossGL
# IR row starts in the second run, and SPIR-V output is missing from the latest run.
HISTORY = _history(
    "2026-10-21T00:00:00Z,c3,glsl,output,80.0,8,1,1,0",
    "2026-10-21T00:00:00Z,c3,hlsl,output,50.0,1,1,0,0",
    "2026-10-21T00:00:00Z,c3,crossgl,ir,40.0,2,3,0,9",
    "2026-10-19T00:00:00Z,c1,glsl,output,70.0,7,2,1,0",
    "2026-10-19T00:00:00Z,c1,hlsl,output,60.0,3,2,0,0",
    "2026-10-19T00:00:00Z,c1,spirv,output,90.0,9,1,0,0",
    "2026-10-20T00:00:00Z,c2,glsl,output,75.0,6,1,1,0",
    "2026-10-20T00:00:00Z,c2,hlsl,output,,0,0,0,4",
    "2026-10-20T00:00:00Z,c2,spirv,output,90.0,9,1,0,0",
    "2026-10-20T00:00:00Z,c2,crossgl,ir,40.0,2,3,0,9",
)


def _texts(root):
    return [t.text for t in root.iter(f"{SVG}text")]


def test_series_follow_time_order_and_the_latest_run():
    dates, points = trend.series(HISTORY)
    assert dates == [
        "2026-10-19T00:00:00Z",
        "2026-10-20T00:00:00Z",
        "2026-10-21T00:00:00Z",
    ]
    # The latest run decides which rows are drawn and in what order; SPIR-V is gone.
    assert list(points) == [("glsl", "output"), ("hlsl", "output"), ("crossgl", "ir")]
    assert [score for _, score in points[("glsl", "output")]] == [70.0, 75.0, 80.0]
    assert [score for _, score in points[("hlsl", "output")]] == [60.0, None, 50.0]
    assert len(points[("crossgl", "ir")]) == 2


def test_change_compares_the_latest_score_with_the_first_measured_one():
    _, points = trend.series(HISTORY)
    assert trend.change(points[("glsl", "output")]) == "+10.0 since 2026-10-19"
    assert trend.change(points[("hlsl", "output")]) == "-10.0 since 2026-10-19"
    assert trend.change(points[("crossgl", "ir")]) == "no change since 2026-10-20"
    assert trend.change([("2026-10-19T00:00:00Z", 70.0)]) == ""
    assert (
        trend.change([("2026-10-19T00:00:00Z", 70.0), ("2026-10-20T00:00:00Z", None)])
        == ""
    )


def test_trend_is_valid_svg_with_scores_changes_and_range():
    root = ElementTree.fromstring(trend.render(HISTORY))
    texts = _texts(root)
    assert root.find(f"{SVG}title").text.startswith("CrossTL coverage over time")
    assert "3 runs" in texts
    assert "2026-10-19" in texts and "2026-10-21" in texts
    assert "80.0 %" in texts and "50.0 %" in texts
    assert "+10.0 since 2026-10-19" in texts
    assert "CrossGL IR" in texts and "expressive" in texts
    assert "SPIR-V" not in texts
    assert "GLSL output 80.0 % (+10.0 since 2026-10-19)" in root.find(f"{SVG}desc").text


def test_lines_span_the_track_and_break_where_a_run_has_no_score():
    root = ElementTree.fromstring(trend.render(HISTORY))
    groups = {g.get("class"): g for g in root.iter(f"{SVG}g")}
    assert set(groups) == {"output", "ir"}
    paths = [p.get("d") for p in root.iter(f"{SVG}path")]
    left, right = chart.BAR_X, chart.BAR_X + chart.BAR_WIDTH
    # GLSL runs from the first to the last run; HLSL has no path at all, because its gap leaves two lone points.
    glsl = paths[0].split()
    assert glsl[0] == f"M{left:g}" and glsl[-2] == f"L{right:g}"
    assert len(paths) == 2
    # The CrossGL IR row starts at the second run, halfway along the track.
    assert paths[1].startswith(f"M{left + chart.BAR_WIDTH / 2:g} ")
    # A higher score sits higher: GLSL's y falls as its score rises.
    ys = [float(glsl[i]) for i in (1, 3, 5)]
    assert ys == sorted(ys, reverse=True)
    assert len(list(root.iter(f"{SVG}circle"))) == 4


def test_one_run_and_no_runs_still_render():
    single = trend.render(_history("2026-10-19T00:00:00Z,c1,glsl,output,70.0,7,2,1,0"))
    root = ElementTree.fromstring(single)
    assert "1 run" in _texts(root)
    circle = next(root.iter(f"{SVG}circle"))
    assert float(circle.get("cx")) == chart.BAR_X + chart.BAR_WIDTH
    empty = ElementTree.fromstring(trend.render(HEADER))
    assert "No published runs yet." in _texts(empty)


def test_trend_is_deterministic_and_self_contained():
    svg = trend.render(HISTORY)
    assert svg == trend.render(HISTORY)
    for banned in ("Gradient", "<script", "<filter", "<image", "href="):
        assert banned not in svg
    assert "prefers-color-scheme:dark" in svg and 'class="bg"' in svg

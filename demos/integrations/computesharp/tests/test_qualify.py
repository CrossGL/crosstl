"""Check the cross-profile qualification and its image differential checks.

The image tests build small synthetic profiles: a DirectX output with its
reference image, and a translated run whose evidence records one captured
readback per ShadersTests case, as the portable runtime writes them. Each
defect the checks must reject is introduced into otherwise consistent inputs.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from fixture_support import ROOT, load_module  # noqa: E402

QUALIFY = load_module("computesharp_qualify", ROOT / "qualify.py")
SPEC = json.loads((ROOT / "image_checks.json").read_text())
TRX_NAMESPACE = "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"
SHADER_CASES = [
    "HelloWorld (Warp, ComputeSharp.SwapChain.Shaders.HelloWorld)",
    "HelloWorld (Warp, ComputeSharp.SwapChain.Shaders.Compute.HelloWorld)",
]


def write_trx(path, results):
    """Write a minimal MSTest TRX with (class, test name, outcome) results."""
    definitions, rows = [], []
    for index, (owner, name, outcome) in enumerate(results):
        method = name.split(" (", 1)[0]
        definitions.append(
            f'<UnitTest id="id{index}" name="{method}">'
            f'<TestMethod className="{owner}" name="{method}"/></UnitTest>'
        )
        rows.append(
            f'<UnitTestResult testId="id{index}" testName="{name}" outcome="{outcome}"/>'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<TestRun xmlns="{TRX_NAMESPACE}"><ResultSummary outcome="Completed"/>'
        f"<TestDefinitions>{''.join(definitions)}</TestDefinitions>"
        f"<Results>{''.join(rows)}</Results></TestRun>"
    )
    return path


def save(path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(path)
    return path


def gradient(width=64, height=32):
    x = np.linspace(0, 255, width, dtype=np.float64)
    y = np.linspace(0, 255, height, dtype=np.float64)
    rgb = np.stack(
        [
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y[:, None], (height, width)),
            np.full((height, width), 96.0),
        ],
        axis=2,
    )
    return np.round(rgb).astype(np.uint8)


def capture_record(case, pixels, *, order=(0, 1, 2, 3), fmt=28):
    rgba = np.concatenate(
        [pixels, np.full(pixels.shape[:2] + (1,), 255, np.uint8)], axis=2
    )
    raw = np.ascontiguousarray(rgba[..., list(order)]).tobytes()
    digest = hashlib.sha256(raw).hexdigest()
    height, width = pixels.shape[:2]
    return rgba, {
        "record": "readback",
        "backend": "metal",
        "case": "ComputeSharp.Tests.ShadersTests." + case,
        "resource": "ReadWriteTexture2D<Rgba32, float4>",
        "resourceId": 1,
        "bytes": len(raw),
        "sha256": digest,
        "format": fmt,
        "width": width,
        "height": height,
        "depth": 1,
        "rowPitch": width * 4,
        "capture": digest + ".png",
    }


def build(tmp_path, *, warp=None, candidate=None, reference=None, mutate=None):
    """Create DirectX outputs and a translated run for the HelloWorld cases."""
    reference = gradient() if reference is None else reference
    warp = reference if warp is None else warp
    candidate = warp if candidate is None else candidate
    directx = tmp_path / "directx" / "outputs"
    save(directx / "references" / "Assets" / "HelloWorld.png", reference)
    save(directx / "outputs" / "Shaders" / "HelloWorld.png", warp)
    run = tmp_path / "metal" / "run"
    save(run / "outputs" / "references" / "Assets" / "HelloWorld.png", reference)
    records = []
    for case in SHADER_CASES:
        rgba, record = capture_record(case, candidate)
        save(run / "captures" / record["capture"], rgba)
        records.append(record)
    trx_rows = [
        ("ComputeSharp.Tests.ShadersTests", case, "Passed") for case in SHADER_CASES
    ]
    if mutate:
        mutate(run, records, trx_rows)
    evidence = run / "evidence.jsonl"
    evidence.write_text("".join(json.dumps(item) + "\n" for item in records))
    trx = write_trx(run / "trx" / "results.trx", trx_rows)
    return {
        "run": run,
        "directx_outputs": directx,
        "trx": trx,
        "evidence": evidence,
        "images": tmp_path / "images",
    }


def check(arguments):
    return QUALIFY.shader_images(SPEC, **arguments)


def test_identical_outputs_pass_and_write_difference_images(tmp_path):
    arguments = build(tmp_path)
    result = check(arguments)
    assert result["status"] == "passed", result["errors"]
    assert result["casesExecuted"] == result["casesCompared"] == 2
    (output,) = result["outputs"]
    assert output["versusDirectX"]["meanAbsoluteError"] == 0
    assert output["upstreamDelta"] == 0
    assert (arguments["images"] / output["difference"]).is_file()


def test_rounding_differences_stay_within_the_margin(tmp_path):
    reference = gradient()
    candidate = reference.copy()
    candidate[::2, :, 0] = np.minimum(candidate[::2, :, 0].astype(int) + 1, 255)
    spec = json.loads(json.dumps(SPEC))
    spec["shaderImages"]["upstreamThresholds"]["HelloWorld"] = 1.0
    arguments = build(tmp_path, reference=reference, candidate=candidate)
    result = QUALIFY.shader_images(spec, **arguments)
    assert result["status"] == "passed", result["errors"]
    (output,) = result["outputs"]
    assert 0 < output["versusDirectX"]["meanAbsoluteError"] < 0.5


def test_a_wrong_region_fails_although_the_upstream_threshold_passes(tmp_path):
    # A block of wrong pixels passes a loose upstream threshold, but not the
    # bound derived from an exact DirectX output.
    reference = gradient()
    candidate = reference.copy()
    candidate[8:12, 8:12] = 0
    spec = json.loads(json.dumps(SPEC))
    spec["shaderImages"]["upstreamThresholds"]["HelloWorld"] = 1.0
    arguments = build(tmp_path, reference=reference, candidate=candidate)
    result = QUALIFY.shader_images(spec, **arguments)
    assert result["status"] == "failed"
    assert any("largePixelFraction" in error for error in result["errors"])


def test_the_bound_scales_with_the_directx_deviation(tmp_path):
    reference = gradient()
    warp = reference.copy()
    warp[:, :, 1] = np.clip(warp[:, :, 1].astype(int) + 20, 0, 255)
    candidate = reference.copy()
    candidate[:, :, 1] = np.clip(candidate[:, :, 1].astype(int) - 10, 0, 255)
    spec = json.loads(json.dumps(SPEC))
    spec["shaderImages"]["upstreamThresholds"]["HelloWorld"] = 1.0
    arguments = build(tmp_path, reference=reference, warp=warp, candidate=candidate)
    result = QUALIFY.shader_images(spec, **arguments)
    assert result["status"] == "passed", result["errors"]
    reference_metrics = result["references"]["HelloWorld"]["directxVsReference"]
    assert reference_metrics["meanAbsoluteError"] > 5


def test_the_upstream_assertion_is_reproduced(tmp_path):
    reference = gradient()
    candidate = reference.copy()
    candidate[0, 0:2, 0] ^= 0x80
    result = check(build(tmp_path, reference=reference, candidate=candidate))
    assert result["status"] == "failed"
    errors = result["errors"]
    assert any("upstream delta" in error for error in errors)
    assert not any("DirectX bound" in error for error in errors), errors


def _drop_capture(run, records, rows):
    records.pop()


def _tamper_capture(run, records, rows):
    path = run / "captures" / records[0]["capture"]
    with Image.open(path) as image:
        pixels = np.asarray(image).copy()
    pixels[0, 0, 2] ^= 1
    save(path, pixels)


def _unexecuted_capture(run, records, rows):
    rows.pop()


def _wrong_region(run, records, rows):
    for record in records:
        record["width"] += 1


def _unknown_shader(run, records, rows):
    rows.append(
        (
            "ComputeSharp.Tests.ShadersTests",
            "Missing (Warp, ComputeSharp.SwapChain.Shaders.Missing)",
            "Passed",
        )
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (_drop_capture, "captured readback"),
        (_tamper_capture, "do not hash"),
        (_unexecuted_capture, "without a result"),
        (_wrong_region, "readback region"),
        (_unknown_shader, "no frozen upstream threshold"),
    ],
)
def test_inconsistent_captures_are_rejected(tmp_path, mutate, message):
    result = check(build(tmp_path, mutate=mutate))
    assert result["status"] == "failed"
    assert any(message in error for error in result["errors"]), result["errors"]


def test_missing_directx_output_is_rejected(tmp_path):
    arguments = build(tmp_path)
    (arguments["directx_outputs"] / "outputs" / "Shaders" / "HelloWorld.png").unlink()
    result = check(arguments)
    assert result["status"] == "failed"
    assert any("missing" in error for error in result["errors"])


def test_reference_images_must_match_between_profiles(tmp_path):
    arguments = build(tmp_path)
    other = gradient()
    other[0, 0] = 0
    save(
        arguments["run"] / "outputs" / "references" / "Assets" / "HelloWorld.png",
        other,
    )
    result = check(arguments)
    assert result["status"] == "failed"
    assert any("differs between the profiles" in error for error in result["errors"])


def test_bgra_captures_hash_in_readback_order():
    pixels = gradient(8, 4)
    rgba, record = capture_record(
        "HelloWorld (Warp, ComputeSharp.SwapChain.Shaders.HelloWorld)",
        pixels,
        order=(2, 1, 0, 3),
        fmt=87,
    )
    assert QUALIFY.capture_digest(rgba.astype(np.int16), record) == record["sha256"]
    record["rowPitch"] += 4
    assert QUALIFY.capture_digest(rgba.astype(np.int16), record) is None


def test_upstream_delta_matches_the_tolerant_image_comparer_formula():
    expected = np.zeros((2, 2, 4), np.int16)
    expected[..., 3] = 255
    actual = expected.copy()
    actual[0, 0, :3] = (10, 20, 30)
    actual[1, 1, 3] = 0  # Saved as RGB upstream, so alpha does not count.
    assert QUALIFY.upstream_delta(expected, actual) == pytest.approx(
        60 / 4 / (4 * 65535)
    )


def test_shader_names_follow_the_type_argument():
    assert (
        QUALIFY.shader_of(
            "ComputeSharp.Tests.ShadersTests.ContouredLayers "
            "(Discrete, ComputeSharp.SwapChain.Shaders.Compute.ContouredLayers)"
        )
        == "ContouredLayers"
    )
    assert QUALIFY.shader_of("ComputeSharp.Tests.ShadersTests.Plain") is None


def _processing(tmp_path, gpu_shift=0, warp_shift=0):
    # Mid-range values keep the shifted JPEG images away from clipping.
    base = (gradient(48, 24).astype(int) * 3 // 4 + 32).astype(np.uint8)
    directx = tmp_path / "directx" / "outputs"
    run = tmp_path / "metal" / "run"
    for root, shift in ((directx, warp_shift), (run / "outputs", gpu_shift)):
        gpu = np.clip(base.astype(int) + shift, 0, 255)
        for method in ("bokeh", "gaussian"):
            save(root / "outputs" / "Imaging" / f"city_{method}_gpu.jpg", gpu)
            save(root / "references" / "Imaging" / f"city_{method}_cpu.jpg", base)
    trx = write_trx(
        run / "trx" / "results.trx",
        [
            ("ComputeSharp.Tests.ImageProcessingTests", "BokehBlur (Warp)", "Passed"),
            (
                "ComputeSharp.Tests.ImageProcessingTests",
                "GaussianBlur (Warp)",
                "Passed",
            ),
        ],
    )
    return QUALIFY.processing_images(
        SPEC, run=run, directx_outputs=directx, trx=trx, images=tmp_path / "images"
    )


def test_processing_outputs_within_the_directx_bound_pass(tmp_path):
    result = _processing(tmp_path, gpu_shift=-2, warp_shift=4)
    assert result["status"] == "passed", result["errors"]
    assert set(result["methods"]) == set(SPEC["processingImages"]["cases"])


def test_processing_outputs_beyond_the_directx_bound_fail(tmp_path):
    result = _processing(tmp_path, gpu_shift=40, warp_shift=0)
    assert result["status"] == "failed"
    assert any("meanAbsoluteError" in error for error in result["errors"])


def test_qualification_reports_missing_artifacts_without_crashing(tmp_path):
    report = QUALIFY.qualify(
        directx=tmp_path / "directx",
        targets={"metal": tmp_path / "metal"},
        output=tmp_path / "qualification",
    )
    assert report["status"] == "failed"
    profiles = report["profiles"]
    assert profiles["directx-warp"]["status"] == "failed"
    assert profiles["metal-apple-gpu"]["status"] == "failed"
    assert profiles["opengl-mesa-llvmpipe"]["errors"] == [
        "no opengl artifacts were supplied"
    ]
    written = json.loads(
        (tmp_path / "qualification" / "qualification.json").read_text()
    )
    assert written["status"] == "failed"

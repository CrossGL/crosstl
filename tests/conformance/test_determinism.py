import subprocess
import sys

from . import determinism
from .determinism import (
    DEFAULT_SEEDS,
    ROOTS,
    differences,
    discover_inputs,
    results_by_seed,
    translate_result,
)

SHADER = """
shader Tiny {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        void main(RWStructuredBuffer<float> output @binding(0)) {
            output[0] = 1.0;
        }
    }
}
"""


def test_every_root_contributes_inputs():
    inputs = discover_inputs()
    assert inputs == sorted(set(inputs))
    for root in ROOTS:
        assert any(path.startswith(root + "/") for path in inputs), root
    assert all(not path.startswith("demos/integrations/") for path in inputs)


def test_seeds_reach_the_workers():
    # The check has power only if the seeds really reorder string sets in the workers.
    names = ", ".join(repr(f"name{index}") for index in range(32))
    orders = set()
    for seed in DEFAULT_SEEDS:
        completed = subprocess.run(
            [sys.executable, "-c", f"print(list({{{names}}}))"],
            env=determinism._worker_environment(seed),
            capture_output=True,
            text=True,
            check=True,
        )
        orders.add(completed.stdout)
    assert len(orders) == len(DEFAULT_SEEDS)


def test_results_hash_outputs_and_refusals(tmp_path):
    shader = tmp_path / "tiny.cgl"
    shader.write_text(SHADER)
    first = translate_result(str(shader), "metal")
    assert first.startswith("ok ")
    assert translate_result(str(shader), "metal") == first
    refused = translate_result(str(tmp_path / "missing.cgl"), "metal")
    assert refused.startswith("error ")


def test_differences_name_each_disagreement():
    by_seed = {
        "0": {"a.cgl\tmetal": "ok 1", "b.cgl\tmetal": "ok 2", "c.cgl\tmetal": "ok 3"},
        "1": {"a.cgl\tmetal": "ok 1", "b.cgl\tmetal": "ok 9"},
    }
    assert differences(by_seed) == [
        "b.cgl -> metal: seed 0: ok 2, seed 1: ok 9",
        "c.cgl -> metal: seed 0: ok 3, seed 1: missing",
    ]


def test_workers_return_one_result_per_input_and_backend():
    # The full check runs in the conformance workflow's determinism job; this runs its worker protocol end to end.
    inputs = [path for path in discover_inputs() if path.endswith(".cgl")][:2]
    backends = ("metal", "opengl")
    by_seed = results_by_seed(inputs, backends, DEFAULT_SEEDS, jobs=2)
    for results in by_seed.values():
        assert sorted(results) == sorted(
            f"{path}\t{backend}" for path in inputs for backend in backends
        )
    assert differences(by_seed) == []

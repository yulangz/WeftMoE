"""Config-only workload preparation and invalid resource pruning."""

import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import autotune  # noqa: E402
from benchmarks.models import MODELS  # noqa: E402


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mode", ("training", "inference", "decode"))
def test_presets_generate_valid_searches(model, mode):
    args, spec = autotune.parse_args(["--model", model, "--mode", mode])
    config = autotune.load_incumbent(spec, args)
    autotune.validate(config)
    assert args.hidden == MODELS[model].hidden
    assert args.top_k == MODELS[model].top_k
    incumbent = dict(config=config, name=autotune.candidate_digest(config))
    for stage in spec["stages"]:
        candidates, _ = autotune.candidates_for(stage, incumbent, spec, args)
        assert candidates[0]["config"] == config
        assert all(autotune.generated_header(c["config"]) for c in candidates)


def test_custom_shape_topology_and_resource_pruning():
    args, spec = autotune.parse_args([
        "--model", "flash", "--hidden", "2560", "--intermediate", "768",
        "--experts", "384", "--tokens", "262144", "--gpus-per-node", "4",
        "--remotes", "node1", "node2", "--master-addr", "node0", "--jobs", "64",
    ])
    config = autotune.load_incumbent(spec, args)
    autotune.validate(config)
    assert config["backward"]["dab_tile_n"] == 128
    assert args.tokens == [262144] and args.jobs == 64
    spec = copy.deepcopy(spec)
    spec["probe"] = {"variants": [{
        "forward.static_v4_fc1_stages": 6, "forward.static_v4_fc2_stages": 6,
    }]}
    candidates, rejected = autotune.candidates_for("probe", {"config": config}, spec, args)
    assert len(candidates) == 1 and len(rejected) == 1

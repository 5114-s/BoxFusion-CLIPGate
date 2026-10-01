from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    "scannet_cutr_paired_scorefix.yaml",
    "scannet_cutr_replay_scorefix.yaml",
    "scannet_boxer_observer_scorefix.yaml",
    "scannet_boxer_active_scorefix.yaml",
    "scannet_boxer_pre_observer_scorefix.yaml",
    "scannet_boxer_pre_active_scorefix.yaml",
)
SCORE04_CONFIGS = (
    "scannet_cutr_paired_score04.yaml",
    "scannet_cutr_replay_score04.yaml",
    "scannet_boxer_observer_score04.yaml",
    "scannet_boxer_active_score04.yaml",
)


def _load(name):
    return yaml.safe_load((ROOT / "config" / name).read_text())


def test_only_lifting_and_artifact_roots_change_between_profiles():
    baseline = _load(CONFIGS[0])
    for name in CONFIGS:
        candidate = _load(name)
        candidate["data"]["output_dir"] = baseline["data"]["output_dir"]
        candidate["lifting"] = deepcopy(baseline["lifting"])
        assert candidate == baseline, name


def test_controlled_and_full_replacement_contracts():
    observer = _load("scannet_boxer_observer_scorefix.yaml")["lifting"]["boxer"]
    active = _load("scannet_boxer_active_scorefix.yaml")["lifting"]["boxer"]
    pre_observer = _load("scannet_boxer_pre_observer_scorefix.yaml")["lifting"][
        "boxer"
    ]
    pre_active = _load("scannet_boxer_pre_active_scorefix.yaml")["lifting"][
        "boxer"
    ]

    assert observer["mode"] == "observer"
    assert active["mode"] == "active"
    assert observer["apply_stage"] == active["apply_stage"] == "post_filter"
    assert pre_observer["mode"] == "observer"
    assert pre_active["mode"] == "active"
    assert (
        pre_observer["apply_stage"]
        == pre_active["apply_stage"]
        == "pre_filter"
    )


def test_x0_replay_and_x1_execute_the_same_observer_workload():
    x0 = _load("scannet_cutr_replay_scorefix.yaml")
    x1 = _load("scannet_boxer_observer_scorefix.yaml")
    x0["data"]["output_dir"] = x1["data"]["output_dir"]
    x0["lifting"]["boxer"]["diagnostics_dir"] = x1["lifting"]["boxer"][
        "diagnostics_dir"
    ]
    assert x0 == x1


def test_score04_profiles_are_isolated_and_use_one_cache_namespace():
    configs = [_load(name) for name in SCORE04_CONFIGS]
    outputs = {cfg["data"]["output_dir"] for cfg in configs}

    assert len(outputs) == len(configs)
    for cfg in configs:
        assert cfg["detection"]["score_thresh"] == 0.4
        assert "/score04/" in cfg["data"]["output_dir"]
        cache = cfg["lifting"]["proposal_cache"]
        assert cache["namespace"] == "scannet-score04-gap25-postfilter-v1"
        if cache["mode"] == "replay":
            assert cache["baseline_prediction_root"].endswith(
                "/results/boxer_lifting/score04/x0_cutr"
            )
            assert "/score04/" in cfg["lifting"]["boxer"][
                "diagnostics_dir"
            ]


def test_score04_changes_only_threshold_and_isolated_artifact_roots():
    score05_names = CONFIGS[:4]
    for score05_name, score04_name in zip(score05_names, SCORE04_CONFIGS):
        score05 = _load(score05_name)
        score04 = _load(score04_name)

        score04["detection"]["score_thresh"] = score05["detection"][
            "score_thresh"
        ]
        score04["data"]["output_dir"] = score05["data"]["output_dir"]
        cache04 = score04["lifting"]["proposal_cache"]
        cache05 = score05["lifting"]["proposal_cache"]
        cache04["namespace"] = cache05["namespace"]
        if cache04["mode"] == "replay":
            cache04["baseline_prediction_root"] = cache05[
                "baseline_prediction_root"
            ]
            score04["lifting"]["boxer"]["diagnostics_dir"] = score05[
                "lifting"
            ]["boxer"]["diagnostics_dir"]

        assert score04 == score05, score04_name

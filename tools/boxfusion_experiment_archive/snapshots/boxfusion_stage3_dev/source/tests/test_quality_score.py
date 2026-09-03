import numpy as np
import pytest

from boxfusion.quality_score import (
    QUALITY_FEATURE_DIM,
    QUALITY_FEATURE_NAMES,
    HeuristicQualityScorer,
    LinearQualityScorer,
    MLPQualityScorer,
    aabb_iou_3d,
    load_quality_scorer,
    make_quality_scorer,
    pairwise_aabb_iou_3d,
    quality_feature_matrix,
    quality_feature_vector,
    soft_nms_aabb_3d,
)


def feature_mapping(value=0.5):
    return {name: value for name in QUALITY_FEATURE_NAMES}


def test_fixed_quality_schema_order_and_read_only_vector():
    mapping = {
        name: index / (QUALITY_FEATURE_DIM - 1)
        for index, name in enumerate(reversed(QUALITY_FEATURE_NAMES))
    }
    vector = quality_feature_vector(mapping)
    expected = np.asarray(
        [mapping[name] for name in QUALITY_FEATURE_NAMES],
        dtype=np.float32,
    )
    np.testing.assert_allclose(vector, expected)
    assert vector.flags.writeable is False


def test_quality_schema_rejects_missing_extra_nonfinite_and_out_of_range():
    missing = feature_mapping()
    missing.pop(QUALITY_FEATURE_NAMES[0])
    with pytest.raises(ValueError, match="missing"):
        quality_feature_vector(missing)

    extra = feature_mapping()
    extra["unknown"] = 0.5
    with pytest.raises(ValueError, match="extra"):
        quality_feature_vector(extra)

    nonfinite = feature_mapping()
    nonfinite[QUALITY_FEATURE_NAMES[1]] = np.nan
    with pytest.raises(ValueError, match="finite"):
        quality_feature_vector(nonfinite)

    out_of_range = feature_mapping()
    out_of_range[QUALITY_FEATURE_NAMES[2]] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        quality_feature_vector(out_of_range)


def test_quality_feature_matrix_supports_records_and_empty_batch():
    matrix = quality_feature_matrix(
        [feature_mapping(0.25), feature_mapping(0.75)]
    )
    assert matrix.shape == (2, QUALITY_FEATURE_DIM)
    np.testing.assert_allclose(matrix[0], 0.25)
    np.testing.assert_allclose(matrix[1], 0.75)
    empty = quality_feature_matrix([])
    assert empty.shape == (0, QUALITY_FEATURE_DIM)


def test_default_heuristic_is_bounded_and_monotonic():
    scorer = HeuristicQualityScorer()
    low = np.full(QUALITY_FEATURE_DIM, 0.2, dtype=np.float32)
    high = low.copy()
    high[0] = 0.9
    low_score = scorer(low)
    high_score = scorer(high)
    assert 0.0 <= low_score < high_score <= 1.0
    assert scorer(feature_mapping(1.0)) == pytest.approx(1.0)
    assert scorer(feature_mapping(0.0)) == pytest.approx(0.0)


def test_linear_quality_scorer_matches_known_sigmoid():
    weights = np.zeros(QUALITY_FEATURE_DIM)
    weights[0] = 2.0
    scorer = LinearQualityScorer(weights, bias=-1.0)
    features = np.zeros((2, QUALITY_FEATURE_DIM))
    features[1, 0] = 1.0
    scores = scorer(features)
    np.testing.assert_allclose(
        scores,
        [1.0 / (1.0 + np.e), 1.0 / (1.0 + np.exp(-1.0))],
        rtol=1e-6,
    )


def test_mlp_quality_scorer_matches_known_forward_pass():
    first_weight = np.zeros((QUALITY_FEATURE_DIM, 2))
    first_weight[0] = [1.0, -1.0]
    first_bias = np.asarray([0.0, 0.5])
    second_weight = np.asarray([[2.0], [3.0]])
    second_bias = np.asarray([-1.0])
    scorer = MLPQualityScorer(
        [first_weight, second_weight],
        [first_bias, second_bias],
    )
    features = np.zeros(QUALITY_FEATURE_DIM)
    features[0] = 1.0
    # Hidden ReLU is [1, 0], final logit is 1.
    assert scorer(features) == pytest.approx(
        1.0 / (1.0 + np.exp(-1.0)), rel=1e-6
    )


@pytest.mark.parametrize(
    "call, message",
    [
        (
            lambda: HeuristicQualityScorer(np.zeros(QUALITY_FEATURE_DIM)),
            "positive",
        ),
        (
            lambda: LinearQualityScorer(np.zeros(3)),
            "shape",
        ),
        (
            lambda: MLPQualityScorer(
                [np.zeros((QUALITY_FEATURE_DIM, 2))],
                [np.zeros(3)],
            ),
            "bias_0",
        ),
        (
            lambda: make_quality_scorer("unknown"),
            "method",
        ),
        (
            lambda: make_quality_scorer("linear"),
            "requires weights",
        ),
    ],
)
def test_quality_scorers_fail_fast_on_invalid_configuration(call, message):
    with pytest.raises((TypeError, ValueError), match=message):
        call()


def test_linear_quality_checkpoint_loads_with_exact_schema(tmp_path):
    path = tmp_path / "linear_quality.npz"
    weights = np.arange(QUALITY_FEATURE_DIM, dtype=np.float32) / 10.0
    np.savez(
        path,
        feature_names=np.asarray(QUALITY_FEATURE_NAMES),
        weight=weights,
        bias=np.asarray(0.25),
    )
    loaded = load_quality_scorer(path, method="linear")
    assert isinstance(loaded, LinearQualityScorer)
    np.testing.assert_allclose(loaded.weights, weights)
    assert loaded.bias == pytest.approx(0.25)


def test_quality_checkpoint_rejects_wrong_schema_and_extra_keys(tmp_path):
    wrong_schema = tmp_path / "wrong_schema.npz"
    names = list(QUALITY_FEATURE_NAMES)
    names[0], names[1] = names[1], names[0]
    np.savez(
        wrong_schema,
        feature_names=np.asarray(names),
        weight=np.zeros(QUALITY_FEATURE_DIM),
        bias=np.asarray(0.0),
    )
    with pytest.raises(ValueError, match="schema/order"):
        load_quality_scorer(wrong_schema, method="linear")

    extra_key = tmp_path / "extra_key.npz"
    np.savez(
        extra_key,
        feature_names=np.asarray(QUALITY_FEATURE_NAMES),
        weight=np.zeros(QUALITY_FEATURE_DIM),
        bias=np.asarray(0.0),
        epoch=np.asarray(1),
    )
    with pytest.raises(ValueError, match="keys must be exactly"):
        load_quality_scorer(extra_key, method="linear")


def test_mlp_quality_checkpoint_loads_and_validates_layers(tmp_path):
    path = tmp_path / "mlp_quality.npz"
    np.savez(
        path,
        feature_names=np.asarray(QUALITY_FEATURE_NAMES),
        num_layers=np.asarray(2),
        weight_0=np.zeros((QUALITY_FEATURE_DIM, 4)),
        bias_0=np.zeros(4),
        weight_1=np.zeros((4, 1)),
        bias_1=np.zeros(1),
    )
    loaded = load_quality_scorer(path, method="mlp")
    assert isinstance(loaded, MLPQualityScorer)
    assert len(loaded.weights) == 2


def test_pairwise_aabb_iou_is_exact_and_symmetric():
    first = np.asarray([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
    boxes = np.asarray(
        [
            first,
            [1.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]
    )
    np.testing.assert_allclose(
        aabb_iou_3d(first, boxes), [1.0, 1.0 / 3.0, 0.0]
    )
    pairwise = pairwise_aabb_iou_3d(boxes, boxes)
    np.testing.assert_allclose(pairwise, pairwise.T)
    np.testing.assert_allclose(np.diag(pairwise), 1.0)


def test_soft_nms_linear_decay_and_non_mutation():
    boxes = np.asarray(
        [
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [10.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.8], dtype=np.float32)
    original_boxes = boxes.copy()
    original_scores = scores.copy()
    indices, decayed = soft_nms_aabb_3d(
        boxes,
        scores,
        method="linear",
        iou_threshold=0.25,
        score_threshold=0.01,
    )
    np.testing.assert_array_equal(indices, [0, 2])
    np.testing.assert_allclose(decayed, [0.9, 0.8])
    np.testing.assert_array_equal(boxes, original_boxes)
    np.testing.assert_array_equal(scores, original_scores)


def test_soft_nms_equal_scores_use_original_index_order():
    boxes = np.asarray(
        [
            [float(index * 10), 0.0, 0.0, 1.0, 1.0, 1.0]
            for index in range(4)
        ]
    )
    scores = np.full(4, 0.5)
    indices, result_scores = soft_nms_aabb_3d(boxes, scores)
    np.testing.assert_array_equal(indices, [0, 1, 2, 3])
    np.testing.assert_allclose(result_scores, 0.5)


def test_soft_nms_gaussian_decay_and_max_detections():
    boxes = np.asarray(
        [
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [1.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [10.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]
    )
    scores = np.asarray([0.9, 0.85, 0.8])
    indices, decayed = soft_nms_aabb_3d(
        boxes,
        scores,
        method="gaussian",
        sigma=0.5,
        max_detections=2,
    )
    np.testing.assert_array_equal(indices, [0, 2])
    np.testing.assert_allclose(decayed, [0.9, 0.8])


def test_soft_nms_empty_and_invalid_inputs():
    indices, scores = soft_nms_aabb_3d(
        np.empty((0, 6)), np.empty(0)
    )
    assert indices.shape == (0,)
    assert scores.shape == (0,)

    with pytest.raises(ValueError, match="scores"):
        soft_nms_aabb_3d(np.ones((2, 6)), np.ones(1))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        soft_nms_aabb_3d(np.ones((1, 6)), np.asarray([1.1]))
    with pytest.raises(ValueError, match="method"):
        soft_nms_aabb_3d(
            np.ones((1, 6)), np.asarray([0.5]), method="invalid"
        )
    with pytest.raises(ValueError, match="positive"):
        soft_nms_aabb_3d(
            np.ones((1, 6)),
            np.asarray([0.5]),
            method="gaussian",
            sigma=0.0,
        )

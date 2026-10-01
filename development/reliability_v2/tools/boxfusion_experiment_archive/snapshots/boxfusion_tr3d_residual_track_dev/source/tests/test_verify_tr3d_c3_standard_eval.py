from tools.verify_tr3d_c3_standard_eval import _parse_eval_log


def test_parse_official_three_threshold_log():
    text = "\n".join(
        (
            "eval mAP: 0.440000", "eval APrec: 0.500000", "eval ARecall: 0.600000",
            "eval mAP: 0.390000", "eval APrec: 0.450000", "eval ARecall: 0.550000",
            "eval mAP: 0.240000", "eval APrec: 0.300000", "eval ARecall: 0.400000",
        )
    )
    result = _parse_eval_log(text)
    assert result["mAP"] == [0.44, 0.39, 0.24]
    assert result["APrec"] == [0.5, 0.45, 0.3]
    assert result["ARecall"] == [0.6, 0.55, 0.4]

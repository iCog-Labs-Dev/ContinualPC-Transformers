"""Known-answer tests for the domain-forgetting metrics."""

import math

import numpy as np
import pytest

from fabricpc.experiments.domain_forgetting.metrics import (
    accuracy_forgetting_matrix,
    bwt_accuracy,
    bwt_ce,
    ce_final_retention,
    ce_forgetting_matrix,
    forgetting_matrices,
    forward_transfer,
    learning_gain,
    max_forgetting,
    is_attention_key_bias,
    pc_bp_agreement,
    perplexity_forgetting_matrix,
    relative_param_distance,
    summarize_matrices,
    task_stage_matrix,
)
from fabricpc.experiments.forgetting_benchmark.forgetting_metrics import (
    compute_backward_transfer,
    compute_forgetting_scores,
)

CE = [
    [5.0, 6.0, 7.0],
    [3.0, 5.5, 6.8],
    [3.6, 4.0, 6.5],
    [4.1, 4.5, 3.5],
]
ACC = [
    [0.30, 0.20, 0.25],
    [0.50, 0.22, 0.26],
    [0.45, 0.40, 0.27],
    [0.35, 0.38, 0.55],
]
NAN = math.nan


def _assert_matrix(actual, expected):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-12)


def test_task_stage_matrix_is_transposed_trained_rows():
    tasks = task_stage_matrix(CE)
    _assert_matrix(tasks, [[3.0, 3.6, 4.1], [5.5, 4.0, 4.5], [6.8, 6.5, 3.5]])


def test_ce_forgetting_matrix_positive_is_degradation():
    _assert_matrix(
        ce_forgetting_matrix(CE),
        [
            [NAN, NAN, NAN],
            [NAN, NAN, NAN],
            [0.6, NAN, NAN],
            [1.1, 0.5, NAN],
        ],
    )


def test_perplexity_forgetting_uses_perplexity_differences():
    ppl = np.exp(np.asarray(CE))
    expected = np.full((4, 3), NAN)
    expected[2, 0] = math.exp(3.6) - math.exp(3.0)
    expected[3, 0] = math.exp(4.1) - math.exp(3.0)
    expected[3, 1] = math.exp(4.5) - math.exp(4.0)
    np.testing.assert_allclose(perplexity_forgetting_matrix(ppl), expected)


def test_accuracy_forgetting_matrix_positive_is_degradation():
    _assert_matrix(
        accuracy_forgetting_matrix(ACC),
        [
            [NAN, NAN, NAN],
            [NAN, NAN, NAN],
            [0.05, NAN, NAN],
            [0.15, 0.02, NAN],
        ],
    )


def test_accuracy_gain_after_learning_is_negative_forgetting():
    acc = [[0.1, 0.1], [0.4, 0.1], [0.6, 0.5]]
    assert accuracy_forgetting_matrix(acc)[2, 0] == pytest.approx(-0.2)
    assert bwt_accuracy(acc) == pytest.approx(0.2)


def test_backward_transfer_signs():
    assert bwt_ce(CE) == pytest.approx(np.mean([3.0 - 4.1, 4.0 - 4.5]))
    assert bwt_ce(CE) == pytest.approx(-0.8)
    assert bwt_accuracy(ACC) == pytest.approx(np.mean([0.35 - 0.50, 0.38 - 0.40]))
    assert bwt_accuracy(ACC) == pytest.approx(-0.085)


def test_ce_wrappers_agree_with_forgetting_metrics():
    tasks = task_stage_matrix(CE)
    assert bwt_ce(CE) == compute_backward_transfer(tasks)
    scores = compute_forgetting_scores(tasks)
    assert max_forgetting(CE, higher_is_better=False) == [scores[0], scores[1]]
    acc_scores = compute_forgetting_scores(-task_stage_matrix(ACC))
    assert max_forgetting(ACC, higher_is_better=True) == [acc_scores[0], acc_scores[1]]


def test_max_forgetting_is_worst_later_degradation():
    assert max_forgetting(CE, higher_is_better=False) == pytest.approx([1.1, 0.5])
    assert max_forgetting(ACC, higher_is_better=True) == pytest.approx([0.15, 0.02])


def test_learning_gain_and_forward_transfer():
    assert learning_gain(CE) == pytest.approx([2.0, 1.5, 3.0])
    assert forward_transfer(CE) == pytest.approx([0.0, 0.5, 0.5])


def test_final_retention_is_learned_over_final_ce():
    assert ce_final_retention(CE) == pytest.approx(np.mean([3.0 / 4.1, 4.0 / 4.5]))


def test_single_domain_edge_case():
    ce = [[5.0], [3.0]]
    acc = [[0.1], [0.4]]
    assert bwt_ce(ce) == 0.0
    assert bwt_accuracy(acc) == 0.0
    assert max_forgetting(ce, higher_is_better=False) == []
    assert ce_final_retention(ce) == 1.0
    assert learning_gain(ce) == pytest.approx([2.0])
    assert forward_transfer(ce) == pytest.approx([0.0])
    assert np.isnan(ce_forgetting_matrix(ce)).all()
    summary = summarize_matrices(["news"], ce, np.exp(ce), acc)
    assert summary["max_ce_forgetting"] == {}
    assert summary["final_ce_forgetting"] == {}


def test_rejects_non_stage_matrices():
    with pytest.raises(ValueError, match="stage matrix"):
        bwt_ce([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="domains"):
        summarize_matrices(["a", "b"], CE, CE, ACC)


def test_forgetting_matrices_use_null_for_undefined_cells():
    result = forgetting_matrices(CE, np.exp(CE), ACC)
    assert result["ce_forgetting"][0] == [None, None, None]
    assert result["ce_forgetting"][3][0] == pytest.approx(1.1)
    assert result["ce_forgetting"][3][2] is None
    assert result["accuracy_forgetting"][3][1] == pytest.approx(0.02)


def test_summary_known_answers():
    summary = summarize_matrices(["news", "reviews", "wiki"], CE, np.exp(CE), ACC)
    assert summary["bwt_ce"] == pytest.approx(-0.8)
    assert summary["bwt_accuracy"] == pytest.approx(-0.085)
    assert summary["max_ce_forgetting"] == pytest.approx({"news": 1.1, "reviews": 0.5})
    assert summary["final_ce_forgetting"] == pytest.approx(
        {"news": 1.1, "reviews": 0.5}
    )
    assert summary["learning_gain_ce"] == pytest.approx(
        {"news": 2.0, "reviews": 1.5, "wiki": 3.0}
    )
    assert summary["final_ce"] == pytest.approx(
        {"news": 4.1, "reviews": 4.5, "wiki": 3.5}
    )


def test_relative_param_distance_known_answer():
    reference = {"a": np.array([3.0, 0.0]), "b": np.array([[4.0]])}
    params = {"a": np.array([3.0, 1.0]), "b": np.array([[4.0]])}
    assert relative_param_distance(params, reference) == pytest.approx(0.2)
    assert relative_param_distance(reference, reference) == 0.0
    with pytest.raises(ValueError, match="different leaves"):
        relative_param_distance({"a": params["a"]}, reference)
    with pytest.raises(ValueError, match="shape"):
        relative_param_distance({"a": np.zeros(3), "b": params["b"]}, reference)


def test_pc_bp_agreement_known_answer():
    ce_pc = np.asarray(CE) + np.array([[0.0, 0.0, 0.0]] * 3 + [[0.0, 0.009, 0.0]])
    stage = [{"w": np.array([1.0, 0.0])}] * 4
    shifted = [{"w": np.array([1.0, 0.5])}] + stage[1:]
    result = pc_bp_agreement(ce_pc, CE, shifted, stage)
    assert result["max_abs_ce_diff"] == pytest.approx(0.009)
    assert result["max_rel_ce_diff"] == pytest.approx(0.009 / 4.5)
    assert result["stage_param_rel_distance"] == pytest.approx([0.5, 0.0, 0.0, 0.0])
    assert result["max_stage_param_rel_distance"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="shapes"):
        pc_bp_agreement([[1.0], [2.0]], CE, stage, stage)


def test_key_bias_is_excluded_from_the_second_distance():
    assert is_attention_key_bias("L0_mha/biases/b_k")
    assert is_attention_key_bias("L3_mha/biases/b_k")
    assert not is_attention_key_bias("L0_mha/biases/b_q")
    assert not is_attention_key_bias("L0_mha/weights/W_k")
    assert not is_attention_key_bias("L0_mlp1/biases/b_k")
    reference = {"L0_mha/biases/b_k": np.array([0.0]), "w": np.array([3.0, 4.0])}
    params = {"L0_mha/biases/b_k": np.array([5.0]), "w": np.array([3.0, 4.5])}
    assert relative_param_distance(params, reference) == pytest.approx(
        np.sqrt(25.25) / 5.0
    )
    assert relative_param_distance(
        params, reference, exclude=is_attention_key_bias
    ) == pytest.approx(0.1)
    result = pc_bp_agreement(
        [[1.0], [2.0]], [[1.0], [2.0]], [reference, params], [reference, reference]
    )
    assert result["stage_param_rel_distance"] == pytest.approx(
        [0.0, np.sqrt(25.25) / 5.0]
    )
    assert result["stage_param_rel_distance_no_key_bias"] == pytest.approx([0.0, 0.1])
    assert result["max_stage_param_rel_distance_no_key_bias"] == pytest.approx(0.1)

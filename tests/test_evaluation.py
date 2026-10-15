import pytest

from gow_rag.evaluation import evaluate


def test_metrics_match_hand_calculation():
    # correct papers B and D, ranked 2nd and 4th
    scores = evaluate({"q1": ["A", "B", "C", "D"]}, {"q1": {"B", "D"}})
    assert scores["MAP"] == pytest.approx(0.5)        # (1/2 + 2/4) / 2
    assert scores["P@10"] == pytest.approx(0.2)
    assert scores["Recall@10"] == pytest.approx(1.0)
    assert scores["MRR@10"] == pytest.approx(0.5)
    assert scores["nDCG@10"] == pytest.approx(0.6509, abs=1e-4)


def test_missed_claim_scores_zero():
    scores = evaluate({"q1": ["A"]}, {"q1": {"Z"}})
    assert scores["MAP"] == 0 and scores["Recall@10"] == 0

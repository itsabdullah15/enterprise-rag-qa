from src.evaluation.metrics import recall_at_k, mrr, exact_match, token_f1


def test_recall_at_k_basic():
    retrieved = ["a", "b", "c", "d"]
    relevant = ["b", "d"]
    assert recall_at_k(retrieved, relevant, k=2) == 0.5


def test_mrr_basic():
    retrieved = ["a", "b", "c"]
    relevant = ["c"]
    assert mrr(retrieved, relevant) == 1/3


def test_exact_match_true():
    assert exact_match("Hello World", "hello world") == 1.0


def test_token_f1_partial_overlap():
    pred = "machine learning models"
    gold = "learning models are useful"
    score = token_f1(pred, gold)
    assert 0 < score < 1
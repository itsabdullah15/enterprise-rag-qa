from typing import List
from collections import defaultdict
import math


# =========================
# 🔍 Retrieval Metrics
# =========================

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = len(set(retrieved_k) & set(relevant))
    return hits / len(relevant) if relevant else 0.0


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    for idx, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (idx + 1)
    return 0.0


# =========================
# ✍️ Generation Metrics
# =========================

def exact_match(pred: str, gold: str) -> float:
    return float(pred.strip().lower() == gold.strip().lower())


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# =========================
# 📊 Aggregate Helper
# =========================

def aggregate(scores: List[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0

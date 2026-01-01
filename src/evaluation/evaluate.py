import json
from statistics import mean

from src.generation.rag_chain import RAGChain
from src.retrieval.retrieval import Retriever
from src.evaluation.metrics import (
    recall_at_k,
    mrr,
    exact_match,
    token_f1,
)

TOP_K = 5


def load_queries():
    with open("src/evaluation/test_queries.json") as f:
        return json.load(f)


def extract_source_ids(docs):
    return [
        f"{d.metadata.get('source_file')}#p{d.metadata.get('page')}"
        for d in docs
    ]


def main():

    rag = RAGChain(mode="synthesis", top_k=TOP_K)
    retriever = rag.retriever

    queries = load_queries()

    recall_scores = []
    mrr_scores = []
    em_scores = []
    f1_scores = []

    for item in queries:

        question = item["question"]
        gold_answer = item["answer"]
        gold_source = item["source_file"]

        print(f"\nQ: {question}")

        # ---------- Retrieval ----------
        docs = retriever.retrieve(question)
        retrieved_ids = extract_source_ids(docs)

        relevant = [doc_id for doc_id in retrieved_ids if gold_source in doc_id]

        recall_scores.append(
            recall_at_k(retrieved_ids, relevant, k=TOP_K)
        )

        mrr_scores.append(
            mrr(retrieved_ids, relevant)
        )

        # ---------- Generation ----------
        pred_answer = rag.run(question)

        em_scores.append(
            exact_match(pred_answer, gold_answer)
        )

        f1_scores.append(
            token_f1(pred_answer, gold_answer)
        )

        print(f"Predicted: {pred_answer[:120]}...")

    print("\n===== FINAL RESULTS =====")
    print(f"Recall@{TOP_K}: {mean(recall_scores):.3f}")
    print(f"MRR: {mean(mrr_scores):.3f}")
    print(f"Exact Match: {mean(em_scores):.3f}")
    print(f"Token-F1: {mean(f1_scores):.3f}")

    return {
    "recall": mean(recall_scores),
    "mrr": mean(mrr_scores),
    "exact_match": mean(em_scores),
    "token_f1": mean(f1_scores),
}


import json

if __name__ == "__main__":
    results = main()

    print("\n===== FINAL RESULTS =====")
    print(json.dumps(results))  # 👈 machine-readable

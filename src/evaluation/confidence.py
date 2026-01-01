from typing import List, Tuple
from langchain_core.documents import Document

UNCERTAINTY_KEYWORDS = [
    "may",
    "might",
    "possible",
    "potential",
    "could",
    "limited",
    "inconsistent",
    "bias",
    "struggle",
    "uncertain",
]


def compute_confidence(
    docs: List[Document],
    answer: str,
    verifier_passed: bool | None = None,
) -> Tuple[str, List[str]]:
    """
    Compute confidence score and uncertainty indicators
    based on retrieval + answer characteristics.
    """

    pages = {d.metadata.get("page") for d in docs}
    num_chunks = len(docs)

    uncertainty_reasons = []

    # --- Evidence strength ---
    if num_chunks >= 3 and len(pages) >= 2:
        confidence = "High"
    elif num_chunks >= 2:
        confidence = "Medium"
        uncertainty_reasons.append(
            "Answer is supported by a limited number of document sections."
        )
    else:
        confidence = "Low"
        uncertainty_reasons.append(
            "Answer is based on a single document section."
        )

    # --- Verifier signal ---
    if verifier_passed is False:
        confidence = "Low"
        uncertainty_reasons.append(
            "Answer could not be fully verified against the source documents."
        )

    # --- Language uncertainty ---
    answer_lower = answer.lower()
    for kw in UNCERTAINTY_KEYWORDS:
        if kw in answer_lower:
            uncertainty_reasons.append(
                "Source text explicitly expresses uncertainty or limitations."
            )
            if confidence == "High":
                confidence = "Medium"
            break

    return confidence, list(set(uncertainty_reasons))

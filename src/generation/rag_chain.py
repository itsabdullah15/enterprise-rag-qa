from typing import List
import os

from langchain_core.documents import Document

from src.generation.llm import VertexGeminiLLM
from src.retrieval.retrieval import Retriever
from src.evaluation.confidence import compute_confidence
# from src.generation.verifier_llm import VerifierLLM
# from src.generation.verifier import verify_faithfulness

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =========================
# 🔒 STRICT MODE PROMPT
# =========================
STRICT_SYSTEM_PROMPT = """
You are an enterprise assistant.

STRICT RULES:
- Provide a complete, well-formed answer.
- Do NOT stop mid-sentence.
- Do NOT paraphrase.
- Use only exact phrases from the context.
- Answer the question ONLY using the provided context.
- Do NOT use prior knowledge or assumptions.
- Every paragraph MUST end with a citation in the format [source_file, page].
- If you cannot cite a paragraph, do NOT include it.
- If the answer is not present in the context, say "I don't know".
"""


# =========================
# 🧠 SYNTHESIS MODE PROMPT
# =========================
SYNTHESIS_SYSTEM_PROMPT = """
You are a document-grounded assistant.

Rules:
- Answer using ONLY the provided context.
- Paraphrase in your own words.
- Combine information across multiple context chunks if needed.
- Do NOT introduce external knowledge.
- If information is missing, say "I don't know".
- Limit the answer to the most important points only.

Answer format:

Answer:
<concise synthesized answer>

Sources:
- <source_file>, page <page>
"""


# =========================
# Context Formatter
# =========================
def format_context(docs: List[Document]) -> str:
    context_blocks = []

    for d in docs:
        source = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page", "unknown")
        context_blocks.append(
            f"[Source: {source}, Page: {page}]\n{d.page_content}"
        )

    return "\n\n".join(context_blocks)


# =========================
# RAG Chain
# =========================
class RAGChain:
    """
    Enterprise-grade RAG Chain with:
    - strict mode (evaluation)
    - synthesis mode (user-facing)
    - confidence scoring
    - uncertainty indicators
    """

    def __init__(self, mode: str = "synthesis", top_k: int = 5):
        assert mode in {"strict", "synthesis"}, "mode must be 'strict' or 'synthesis'"

        self.mode = mode
        self.retriever = Retriever(top_k=top_k)
        self.llm = VertexGeminiLLM()

        # Optional verifier (recommended in strict mode)
        # self.verifier_llm = VerifierLLM()

    def run(self, question: str) -> str:
        # 1️⃣ Retrieve
        docs = self.retriever.retrieve(question)

        if not docs:
            return "I don't know. The answer is not present in the provided documents."

        # 2️⃣ Format context
        context = format_context(docs)

        # 3️⃣ Select prompt
        system_prompt = (
            STRICT_SYSTEM_PROMPT
            if self.mode == "strict"
            else SYNTHESIS_SYSTEM_PROMPT
        )

        prompt = f"""
{system_prompt}

Context:
{context}

Question:
{question}

Answer:
"""

        # 4️⃣ Generate answer
        answer = self.llm.generate(prompt)

        # 5️⃣ Optional verification (STRICT MODE)
        verifier_passed = None

        # if self.mode == "strict":
        #     verifier_passed = verify_faithfulness(
        #         self.verifier_llm,
        #         answer,
        #         context
        #     )
        #     if not verifier_passed:
        #         return "I don't know. The answer could not be verified from the provided documents."

        # 6️⃣ Confidence + Uncertainty
        confidence, uncertainty_reasons = compute_confidence(
            docs=docs,
            answer=answer,
            verifier_passed=verifier_passed,
        )

        # 7️⃣ Append trust signals ONLY for synthesis mode
        if self.mode == "synthesis":
            uncertainty_block = (
                "\n".join(f"- {r}" for r in uncertainty_reasons)
                if uncertainty_reasons
                else "None identified."
            )

            answer = f"""{answer}

Confidence: {confidence}

Uncertainty Indicators:
{uncertainty_block}
"""

        return answer

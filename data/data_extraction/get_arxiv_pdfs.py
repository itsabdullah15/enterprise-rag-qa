import arxiv
from pathlib import Path

def download_arxiv_papers(
    query: str,
    max_results: int = 50,
    save_dir: str = "data/raw_pdfs/arxiv"
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    client = arxiv.Client()

    for result in client.results(search):
        result.download_pdf(dirpath=save_dir)
        print(f"Downloaded: {result.title}")

# Example usage
download_arxiv_papers(
    query="rag",
    max_results=15
)



from pathlib import Path
from datetime import datetime, timezone
import json

def build_doc_registry(pdf_dir="data/raw_pdfs/arxiv"):
    registry = []

    for pdf in Path(pdf_dir).glob("*.pdf"):
        registry.append({
            "filename": pdf.name,
            "source": "arxiv",
            "domain": "AI",
            "doc_type": "research",
            "ingested_at": datetime.now(timezone.utc).isoformat()
        })

    with open(f"{pdf_dir}/doc_registry.json", "w") as f:
        json.dump(registry, f, indent=2)

build_doc_registry()



# This metadata will be extremely useful for:
# 	•	Filtering answers by source
# 	•	Debugging hallucinations
# 	•	Auditing responses

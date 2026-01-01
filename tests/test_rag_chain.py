from src.generation.rag_chain import RAGChain


def test_rag_chain_run_is_called(monkeypatch):
    rag = RAGChain(mode="synthesis")

    def fake_run(question: str):
        return "fake answer"

    monkeypatch.setattr(rag, "run", fake_run)

    result = rag.run("hello?")
    
    assert result == "fake answer"
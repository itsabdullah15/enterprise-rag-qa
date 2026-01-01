from fastapi import FastAPI
from pydantic import BaseModel

from src.generation.rag_chain import RAGChain

# Initialize once at startup (recommended)
rag = RAGChain(mode="synthesis")

app = FastAPI()


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask(payload: Question):
    answer = rag.run(payload.question)

    return {
        "question": payload.question,
        "answer": answer
    }



# from src.generation.rag_chain import RAGChain

# rag = RAGChain(mode="synthesis")
# # rag = RAGChain(mode="strict")

# # answer = rag.run("What is Sickle Cell Anemia Management?")
# # answer = rag.run("What is Stance Classification?")
# answer = rag.run("What are the limitations of stance classification models?")
# # answer = rag.run("What dataset size was used in this paper?")


# print("---- RAW ANSWER ----")
# print(answer)
# print("--------------------")
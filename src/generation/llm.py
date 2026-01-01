# import requests
# import json


# class OllamaLLM:
#     def __init__(
#         self,
#         model: str = "qwen3:8b",
#         base_url: str = "http://127.0.0.1:11434",
#     ):
#         self.model = model
#         self.base_url = base_url

#     def generate(self, prompt: str, temperature: float = 0.2) -> str:
#         payload = {
#             "model": self.model,
#             "messages": [
#                 {"role": "user", "content": prompt}
#             ],
#             "options": {
#                 "temperature": temperature,
#                 "num_predict": 500  # 🔒 limit answer length
#             },
#             "stream": False
#         }

#         response = requests.post(
#             f"{self.base_url}/api/chat",
#             headers={"Content-Type": "application/json"},
#             data=json.dumps(payload),
#             timeout=180,
#         )
#         # print(response)

#         if response.status_code != 200:
#             raise RuntimeError(
#                 f"Ollama error {response.status_code}: {response.text}"
#             )

#         return response.json()["message"]["content"]

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig


class VertexGeminiLLM:
    def __init__(
        self,
        project_id: str = "ai-trip-planner-12345",
        location: str = "us-central1",
        model_name: str = "gemini-2.0-flash-001"
    ):
        vertexai.init(
            project=project_id,
            location=location
        )

        print("✅ Using model:", model_name)

        self.model = GenerativeModel(model_name)

        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.8,
            max_output_tokens=500
        )

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config=self.generation_config
        )
        return response.text.strip()

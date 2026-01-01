from src.generation.llm import VertexGeminiLLM
from vertexai.generative_models import GenerationConfig


class VerifierLLM(VertexGeminiLLM):
    def __init__(self, project_id: str):
        super().__init__(
            project_id=project_id,
            model_name="gemini-1.5-pro"  # stricter reasoning
        )

        # Override generation config for verification
        self.generation_config = GenerationConfig(
            temperature=0.0,   # VERY IMPORTANT for verification
            top_p=0.5,
            max_output_tokens=300
        )

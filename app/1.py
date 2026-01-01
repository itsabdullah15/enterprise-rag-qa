from google import genai

PROJECT_ID = "ai-trip-planner-12345"
LOCATION = "global"
MODEL = "gemini-2.5-flash-lite"


def main():
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )

    response = client.models.generate_content(
        model=MODEL,
        contents="Explain LLMs in one para.",
        config={
            "temperature": 0.3,
            "top_p": 0.9,
            "max_output_tokens": 128,
        },
    )

    print(response.text)


if __name__ == "__main__":
    main()



# # Example usage
# client = VertexAIGeminiClient("ai-trip-planner-12345")
# output = client.generate("Explain RAG in simple terms for a student.")
# print(output)


# # from google import genai
# # from google.genai import types

# # # Initialize client
# # client = genai.Client(
# #     vertexai=True,
# #     project="ai-trip-planner-12345",   # replace with your project ID
# #     location="global"
# # )

# # # Google Cloud Storage URI of an image (example image)
# # IMAGE_URI = "gs://generativeai-downloads/images/scones.jpg"

# # # Model to use
# # model = "gemini-2.5-flash-lite"

# # # Send prompt + image
# # response = client.models.generate_content(
# #     model=model,
# #     contents=[
# #         "What is shown in this image?",          # Your text prompt
# #         types.Part.from_uri(                      # Image part from GCS URI
# #             file_uri=IMAGE_URI,
# #             mime_type="image/png"
# #         ),
# #     ],
# # )

# # print(response.text, end="")

from langchain_openai import OpenAIEmbeddings
import os

embeddings = OpenAIEmbeddings(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_API_KEY"],
    model="text-embedding-3-large"
)
import os
from langchain_pinecone import PineconeVectorStore
from app.chat.embeddings.openai import embeddings

# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)



os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

index_name = "docs"

vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

def build_retriever(chat_args, k):
    # logger.info(f">>>>>>> Building retriever for pdf_id: {chat_args.pdf_id}")

    search_kwargs = {
        "filter": {"pdf_id": chat_args.pdf_id},
        "k": k
    }

    return vector_store.as_retriever(
        search_kwargs=search_kwargs
    )
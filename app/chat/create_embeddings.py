from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.chat.vector_stores.pinecone import vector_store

import sys

def create_embeddings_for_pdf(pdf_id: str, pdf_path: str):
    """
    Generate and store embeddings for the given pdf

    1. Extract text from the specified PDF.
    2. Divide the extracted text into manageable chunks.
    3. Generate an embedding for each chunk.
    4. Persist the generated embeddings.

    :param pdf_id: The unique identifier for the PDF.
    :param pdf_path: The file path to the PDF.

    Example Usage:

    create_embeddings_for_pdf('123456', '/path/to/pdf')
    """

    textsplitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    ) 

    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split(textsplitter)

    for doc in docs:
        doc.metadata = {
            "page": doc.metadata["page"],
            "text": doc.page_content,
            "pdf_id": pdf_id,
        }

    # # Set default encoding to utf-8
    # sys.stdout.reconfigure(encoding='utf-8')

    # print(docs)

    vector_store.add_documents(docs)
    

import os
import pinecone

from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from consts import (
    PINECONE_INDEX_NAME,
    PINECONE_ENVIRONMENT,
    DRF_API_GUIDE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from utils.get_docs_url import get_docs_url

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=PINECONE_ENVIRONMENT,
)


def ingest_docs() -> None:
    """
    This function is run to embed our documents and insert the embeddings
    into Pinecone as vectors.
    This function should only be run once.
    """

    loader = ReadTheDocsLoader(path=DRF_API_GUIDE_PATH)
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = get_docs_url(old_path)
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings, index_name=PINECONE_INDEX_NAME)
    print("****** Added to Pinecone vectorstore vectors")


def load_drf_documents():
    loader = UnstructuredHTMLLoader(file_path=DRF_API_GUIDE_PATH)
    raw_documents = loader.load()
    return raw_documents


if __name__ == "__main__":
    # ingest_docs()
    print(load_drf_documents())

    # file_path = DRF_API_GUIDE_PATH
    # for p in Path(file_path).rglob("*"):
    #     if p.is_dir():
    #         continue
    #     with open(p) as f:
    #         text = _clean_data(f.read())
    #     metadata = {"source": str(p)}

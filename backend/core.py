import os
import pinecone

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from consts import (
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
)


pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=PINECONE_ENVIRONMENT,
)


def run_llm(query: str, chat_history: list[dict[str, any]] = []) -> any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))

import os
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHROMA_DIR = 'vector_db'
COLLECTION_NAME = "meeting_transcript"

def get_embedding_model():
    return MistralAIEmbeddings(
    model="mistral-embed",
    api_key = os.getenv('MISTRAL_API_KEY')
)


def vector_store(transcript : str) ->   Chroma:
    print('building vector storage...')

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )

    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index':i})
        for i,chunk in enumerate(chunks)
    ]

    model = get_embedding_model()
    vectorStore = Chroma.from_documents(
        documents=docs,
        embedding=model,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR    

    )


    return vectorStore



def load_storage() ->Chroma:
    model = get_embedding_model()

    vector_store = Chroma(
        embedding_function=model,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )


    return vector_store


def get_retriever(vector_store : Chroma , k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {'k':k}
    )






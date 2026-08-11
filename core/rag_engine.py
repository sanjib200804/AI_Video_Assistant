import os 
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough , RunnableLambda
from core.vector_store import vector_store , get_retriever ,load_storage

def get_llm():
    return ChatMistralAI(
        model="mistral-small-2506",
        temperature=0.4,
        mistral_api_key = os.getenv('MISTRAL_API_KEY')
    )


def docs_formate(docs:list) :
    return '\n\n'.join([doc.page_content for doc in docs])


def build_rag_chain(transcript:str):
    vectorStore = vector_store(transcript)
    retriever = get_retriever(vectorStore)
    llm = get_llm()
    prompt= ChatPromptTemplate(
[
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ]
    )

    rag_chain = (
        {
            'context':retriever | RunnableLambda(docs_formate),
            'question':RunnablePassthrough()
        } 
        | prompt | llm | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    vector_storage = vector_store()
    retriever = get_retriever(vector_storage)
    llm = get_llm()
    prompt= ChatPromptTemplate(
[
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ]
    )

    rag_chain = (
        {
            'context':retriever | RunnableLambda(docs_formate),
            'question':RunnablePassthrough()
        } 
        | prompt | llm | StrOutputParser()
    )

    return rag_chain

def question(rag_chain , question:str)->str:
    print(f'Question:{question}')
    answer = rag_chain.invoke(question)
    print(f'answer:{answer}')

    return answer
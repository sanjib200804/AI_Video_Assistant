from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda , RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
import os


def get_llm():
    return ChatMistralAI(model="mistral-small-2506")



def build_chain(sys_prompt:str) :
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x:{'text':x}) | ChatPromptTemplate.from_messages([
            ('system',sys_prompt),
            ('human','{text}')
        ]) | llm | StrOutputParser()
    )


def extract_action_items(transcript:str) ->str:
    chain = build_chain(
        'you are an expert meeting analyst.From the meeting transcript,'
        'extract all action items. for each provider:\n'
        '- task description\n'
        '-owner (who is responsible)\n'
        'Deadline (if mentions ,else return No specified.)\n\n'
        'Formate as a number list.If none found say "No action item found."'
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)
    
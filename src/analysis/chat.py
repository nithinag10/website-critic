from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_groq import ChatGroq
from ..config.setting import GROQ_API_KEY

def create_chat_chain(retriever):
    """Creates a chat chain for querying the vector store."""
    
    template = """You are an UX export, analyze the following image segment of a website thoroughly and answer the following questions:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    model = ChatGroq(
        model_name='deepseek-r1-distill-qwen-32b',
        temperature=0.7,
        api_key=GROQ_API_KEY
    )
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return chain
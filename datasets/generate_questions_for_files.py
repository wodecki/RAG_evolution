import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def generate_questions(directory: str,n_questions_per_file):
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = ChatPromptTemplate.from_template("""
    You are an assistant for question-generation tasks.
    Use the following pieces of context to generate a list of {n} questions.
    Return only the questions as a list. Not number questions. Not add any other symbol before the question.

    Context: {context}

    Questions:
    """)
    
    all_questions = []
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        loader = TextLoader(file_path)
        documents = loader.load()
        
        for document in documents:
            response = llm.invoke(prompt.invoke({"context": document, "n": n_questions_per_file})).content
            questions_for_file = response.strip().split("\n")
            all_questions.extend(questions_for_file)
    
    return all_questions
directory = "./scientists_bios"
n = 5
questions = generate_questions(directory, n)
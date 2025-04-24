import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def generate_multi_hop_questions(directory, n_questions):
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Load all documents and combine them with visible borders
    all_documents_text = ""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        loader = TextLoader(file_path)
        documents = loader.load()
        
        for document in documents:
            all_documents_text += f"\n\n===== FILE: {filename} =====\n\n"
            all_documents_text += document.page_content
    
    # Create prompt for multi-hop questions
    prompt = ChatPromptTemplate.from_template("""
    You are an assistant for generating multi-hop questions.
    
    Multi-hop questions: 
    1. require information from multiple documents to answer correctly.
    2. can't be answered by looking at a single document.
    3. should be clear and concise.
    4. should be relevant to the content of the documents.
    
    Examples:
    1. List all the scientists born in the same century?
    2. What are the common themes in the works of scientists from different fields?
    3. List all the scientists who were born in the same country?
    4. List all the scientists who have worked in the same field?
    
    Analyze the following collection of documents and generate {n} multi-hop questions.
    Each question should require connecting information from at least two different files.
    
    Return only the questions as a list without numbering or additional symbols, like "- ".
    
    Documents:
    {context}
    
    Multi-hop Questions:
    """)
    
    # Generate multi-hop questions
    response = llm.invoke(prompt.invoke({"context": all_documents_text, "n": n_questions})).content
    questions = response.strip().split("\n")
    
    return questions

directory = "./scientists_bios"
n = 5
questions = generate_multi_hop_questions(directory, n)
print(questions)

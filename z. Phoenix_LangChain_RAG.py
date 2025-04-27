import os
import csv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


questions = []
references = []
csv_file = "./input/reference.csv"
with open(csv_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        if len(row) >= 2:
            questions.append(row[0])
            references.append(row[1])
#limit to 10
questions = questions[:3]
references = references[:3]

directory = "./input/scientists_bios"
loader = DirectoryLoader(directory)

# Create a custom text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)

# Load and split documents
docs = loader.load_and_split(text_splitter=splitter)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()

# Set up vector store
vector_store = Chroma(
    collection_name="scientists_bios",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# Add documents to vector store
vector_store.add_documents(documents=docs)

# Define the retriever
retriever = vector_store.as_retriever()

# Define the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Define the prompt
prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
""")

# Define the grounding chain
grounding_chain = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

# Define the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
    

for i, (question, reference) in enumerate(zip(questions, references)):
    print(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
    
    # Get response and grounding context
    response = rag_chain.invoke(question)
    grounding_docs = grounding_chain.invoke(question)["context"]
            
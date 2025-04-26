import os
import csv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def load_questions_from_csv(csv_file):
    """Load questions and reference answers from the CSV file."""
    questions = []
    references = []
    
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                questions.append(row[0])
                references.append(row[1])
    
    return questions, references

def setup_rag_pipeline(directory):
    """Set up the RAG pipeline using the same approach as in main.py."""
    # Load documents
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
    
    return rag_chain, grounding_chain

def save_rag_results_to_csv(questions, references, rag_chain, grounding_chain, filename="results/rag_results.csv"):
    """Save RAG evaluation results to CSV."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create CSV file with headers
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['question', 'context', 'response', 'reference'])
        
        # Process each question and save results
        for i, (question, reference) in enumerate(zip(questions, references)):
            print(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            # Get response and grounding context
            response = rag_chain.invoke(question)
            grounding_docs = grounding_chain.invoke(question)["context"]
            
            # Extract only the page_content from each document
            cleaned_context = ""
            for doc in grounding_docs:
                cleaned_context += doc.page_content + "\n\n"
            
            # Write to CSV
            csv_writer.writerow([question, cleaned_context.strip(), response, reference])
            
    print(f"\nRAG evaluation saved to {filename}")
    return filename

def main():
    # Define input and output paths
    input_file = "./input/reference.csv"
    output_file = "./results/rag_results.csv"
    input_dir = "./input/scientists_bios"
    
    # Load questions and references
    print(f"Loading questions and references from {input_file}...")
    questions, references = load_questions_from_csv(input_file)
    
    # Set up RAG pipeline
    print("Setting up RAG pipeline...")
    rag_chain, grounding_chain = setup_rag_pipeline(input_dir)
    
    # Process questions and save results
    print(f"Processing {len(questions)} questions...")
    csv_file = save_rag_results_to_csv(questions, references, rag_chain, grounding_chain, output_file)
    
    print(f"Created CSV file for RAG evaluation: {os.path.abspath(csv_file)}")

if __name__ == "__main__":
    main()
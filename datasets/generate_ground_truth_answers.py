import os
import csv
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def load_questions_from_csv(csv_file):
    """Load questions from the CSV file."""
    questions = []
    sources = []
    types = []
    
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 3:
                questions.append(row[0])
                sources.append(row[1])
                types.append(row[2])
    
    return questions, sources, types

def load_document_content(directory, filename=None):
    """Load content from a specific file or all files in the directory."""
    if filename and filename != "multiple":
        # Load a specific file
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            loader = TextLoader(file_path)
            documents = loader.load()
            return documents[0].page_content if documents else ""
    else:
        # Load all files for multi-hop questions
        all_documents_text = ""
        for fname in os.listdir(directory):
            if fname.endswith('.txt'):
                file_path = os.path.join(directory, fname)
                loader = TextLoader(file_path)
                documents = loader.load()
                
                for document in documents:
                    all_documents_text += f"\n\n===== FILE: {fname} =====\n\n"
                    all_documents_text += document.page_content
        
        return all_documents_text

def generate_ground_truth_answer(question, context, question_type, llm):
    """Generate a ground truth answer for a question using the provided context."""
    if question_type == "single-hop":
        prompt = ChatPromptTemplate.from_template("""
        You are an expert assistant tasked with providing accurate answers to questions based on the given context.
        
        Context:
        {context}
        
        Question: {question}
        
        Provide a comprehensive and accurate answer based solely on the information in the context.
        Your answer should be detailed and directly address the question.
        """)
    else:  # multi-hop
        prompt = ChatPromptTemplate.from_template("""
        You are an expert assistant tasked with providing accurate answers to multi-hop questions.
        Multi-hop questions require connecting information from multiple documents to answer correctly.
        
        Context (from multiple documents):
        {context}
        
        Question: {question}
        
        Provide a comprehensive and accurate answer based solely on the information in the context.
        Your answer should connect relevant information from different documents and directly address the question.
        """)
    
    response = llm.invoke(prompt.invoke({"context": context, "question": question})).content
    return response.strip()

def main():
    # File paths
    questions_csv = "./datasets/questions_dataset.csv"
    output_csv = "./datasets/questions_with_answers.csv"
    bios_directory = "./datasets/scientists_bios"
    
    # Load questions from CSV
    print("Loading questions from CSV...")
    questions, sources, types = load_questions_from_csv(questions_csv)
    
    # Initialize GPT-4.1 model
    print("Initializing GPT-4.1 model...")
    llm = ChatOpenAI(model="gpt-4o")  # Using GPT-4o as a proxy for GPT-4.1
    
    # Generate ground truth answers
    print(f"Generating ground truth answers for {len(questions)} questions...")
    answers = []
    
    for i, (question, source, q_type) in enumerate(zip(questions, sources, types)):
        print(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
        
        # Load appropriate context
        context = load_document_content(bios_directory, source)
        
        # Generate answer
        answer = generate_ground_truth_answer(question, context, q_type, llm)
        answers.append(answer)
    
    # Save questions and answers to CSV
    print(f"Saving questions and answers to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Question', 'Source', 'Type', 'Ground Truth Answer'])
        
        for question, source, q_type, answer in zip(questions, sources, types, answers):
            writer.writerow([question, source, q_type, answer])
    
    print("Done!")

if __name__ == "__main__":
    main()
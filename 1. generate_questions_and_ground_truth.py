import os
import csv
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def generate_questions_for_files(directory, n_questions_per_file=5):
    """Generate questions for each file in the directory."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = ChatPromptTemplate.from_template("""
    You are an assistant for question-generation tasks.
    Use the following pieces of context to generate a list of {n} questions.
    Return only the questions as a list. Do not number questions. Do not add any other symbol before the question.

    Context: {context}

    Questions:
    """)
    
    all_questions = []
    file_sources = []
    
    for filename in os.listdir(directory):
        if not filename.endswith('.txt'):
            continue
            
        file_path = os.path.join(directory, filename)
        loader = TextLoader(file_path)
        documents = loader.load()
        
        for document in documents:
            response = llm.invoke(prompt.invoke({"context": document, "n": n_questions_per_file})).content
            questions_for_file = response.strip().split("\n")
            
            # Clean up the questions
            for question in questions_for_file:
                clean_question = question.strip()
                if clean_question.startswith('- '):
                    clean_question = clean_question[2:].strip()
                if clean_question.startswith('* '):
                    clean_question = clean_question[2:].strip()
                if clean_question:
                    all_questions.append(clean_question)
                    file_sources.append(filename)
    
    return all_questions, file_sources

def generate_multi_hop_questions(directory, n_questions=10):
    """Generate multi-hop questions that require information from multiple files."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Load all documents and combine them with visible borders
    all_documents_text = ""
    for filename in os.listdir(directory):
        if not filename.endswith('.txt'):
            continue
            
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
    1. Which scientists were born in the 19th century?
    2. What are the common themes in the works of scientists from different fields?
    3. Which scientists were born in the same country?
    4. Which scientists worked in the same field?
    
    Analyze the following collection of documents and generate {n} multi-hop questions.
    Each question should require connecting information from at least two different files.
    
    Return only the questions as a list without numbering or additional symbols, like "- ".
    
    Documents:
    {context}
    
    Multi-hop Questions:
    """)
    
    # Generate multi-hop questions
    response = llm.invoke(prompt.invoke({"context": all_documents_text, "n": n_questions})).content
    raw_questions = [q.strip() for q in response.strip().split("\n") if q.strip()]
    
    # Clean up questions
    questions = []
    for q in raw_questions:
        clean_q = q.strip()
        if clean_q.startswith('- '):
            clean_q = clean_q[2:].strip()
        if clean_q.startswith('* '):
            clean_q = clean_q[2:].strip()
        if clean_q:
            questions.append(clean_q)
    
    # For multi-hop questions, the source is "multiple"
    sources = ["multiple"] * len(questions)
    
    return questions, sources

def generate_ground_truth_answer(question, context, llm):
    """Generate a ground truth answer for a question using the provided context."""
    prompt = ChatPromptTemplate.from_template("""
    You are an expert assistant tasked with providing accurate answers to questions based on the given context.
    
    Context:
    {context}
    
    Question: {question}
    
    Provide a comprehensive and accurate answer based solely on the information in the context.
    Your answer should be detailed and directly address the question.
    """)
    
    response = llm.invoke(prompt.invoke({"context": context, "question": question})).content
    return response.strip()

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

def main():
    # Define input and output paths
    input_dir = "./input/scientists_bios"
    output_file = "./input/reference.csv"
    
    # Generate questions for individual files (5 questions per file, 5 files = 25 questions)
    print("Generating questions for individual files...")
    single_hop_questions, single_hop_sources = generate_questions_for_files(input_dir, 5)
    
    # Generate multi-hop questions (10 questions)
    print("Generating multi-hop questions...")
    multi_hop_questions, multi_hop_sources = generate_multi_hop_questions(input_dir, 10)
    
    # Combine all questions
    all_questions = single_hop_questions + multi_hop_questions
    all_sources = single_hop_sources + multi_hop_sources
    
    # Initialize LLM for generating ground truth answers
    llm = ChatOpenAI(model="gpt-4.1")
    
    # Generate ground truth answers
    print(f"Generating ground truth answers for {len(all_questions)} questions...")
    answers = []
    
    for i, (question, source) in enumerate(zip(all_questions, all_sources)):
        print(f"Processing question {i+1}/{len(all_questions)}: {question[:50]}...")
        
        # Load appropriate context
        context = load_document_content(input_dir, source)
        
        # Generate answer
        answer = generate_ground_truth_answer(question, context, llm)
        answers.append(answer)
    
    # Save questions and answers to CSV
    print(f"Saving questions and answers to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['question', 'reference'])
        
        for question, answer in zip(all_questions, answers):
            writer.writerow([question, answer])
    
    print("Done!")

if __name__ == "__main__":
    main()
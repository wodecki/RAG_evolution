import os
import csv
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def generate_questions_for_files(directory, n_questions_per_file):
    """Generate questions for each file in the directory."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = ChatPromptTemplate.from_template("""
    You are an assistant for question-generation tasks.
    Use the following pieces of context to generate a list of {n} questions.
    Return only the questions as a list. Not number questions. Not add any other symbol before the question.

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
            
            # Add questions and their sources
            for question in questions_for_file:
                # Clean up the question - remove dashes, bullet points, etc.
                clean_question = question.strip()
                if clean_question.startswith('- '):
                    clean_question = clean_question[2:].strip()
                if clean_question.startswith('* '):
                    clean_question = clean_question[2:].strip()
                if clean_question:
                    all_questions.append(clean_question)
                    file_sources.append(filename)
    
    return all_questions, file_sources

def generate_multi_hop_questions(directory, n_questions):
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

def save_to_csv(questions, sources, question_types, output_file):
    """Save questions, sources, and question types to a CSV file."""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Question', 'Source', 'Type'])
        
        for question, source, q_type in zip(questions, sources, question_types):
            writer.writerow([question, source, q_type])

def main():
    directory = "./datasets/scientists_bios"
    output_file = "./datasets/questions_dataset.csv"
    
    # Generate 5 questions per file (total 25 questions)
    print("Generating questions for individual files...")
    single_hop_questions, single_hop_sources = generate_questions_for_files(directory, 5)
    single_hop_types = ["single-hop"] * len(single_hop_questions)
    
    # Generate 10 multi-hop questions
    print("Generating multi-hop questions...")
    multi_hop_questions, multi_hop_sources = generate_multi_hop_questions(directory, 10)
    multi_hop_types = ["multi-hop"] * len(multi_hop_questions)
    
    # Combine all questions
    all_questions = single_hop_questions + multi_hop_questions
    all_sources = single_hop_sources + multi_hop_sources
    all_types = single_hop_types + multi_hop_types
    
    # Save to CSV
    print(f"Saving {len(all_questions)} questions to {output_file}...")
    save_to_csv(all_questions, all_sources, all_types, output_file)
    print("Done!")

if __name__ == "__main__":
    main()
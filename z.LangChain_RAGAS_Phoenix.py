import pandas as pd

# read the ./results/rag_results.csv file
ragas_evals_df = pd.read_csv("./results/rag_results.csv")

import phoenix as px
px.close_app()
session = px.launch_app()

from phoenix.trace.dsl.helpers import SpanQuery

client = px.Client()
corpus_df = px.Client().query_spans(
    SpanQuery().explode(
        "embedding.embeddings",
        text="embedding.text",
        vector="embedding.vector",
    )
)
corpus_df.head()

px.close_app()
session = px.launch_app()

from datasets import Dataset

#### LangChain example start
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datasets import Dataset
from tqdm.auto import tqdm
import pandas as pd
import csv
import os

# Set up OpenInference instrumentation for Phoenix
from openinference.instrumentation.langchain import LangChainInstrumentor
import nest_asyncio
import phoenix as px
from phoenix.trace import DocumentEvaluations, SpanEvaluations

nest_asyncio.apply()  # needed for concurrent evals in notebook environments
from phoenix.otel import register
tracer_provider = register()

LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)

# Load documents from the prompt-engineering-papers directory
dir_path = "./prompt-engineering-papers"

# Create a custom loader for PDF files
class CustomDirectoryLoader(DirectoryLoader):
    def __init__(self, path, glob="**/*.pdf", loader_cls=PyPDFLoader, **loader_kwargs):
        super().__init__(path, glob=glob, loader_cls=loader_cls, **loader_kwargs)

# Load documents
loader = CustomDirectoryLoader(dir_path, glob="*.pdf", loader_cls=PyPDFLoader)

# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)

# Load and split documents
print("Loading and splitting documents...")
docs = loader.load_and_split(text_splitter=text_splitter)
print(f"Loaded {len(docs)} document chunks")

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()

# Set up vector store
vector_store = Chroma(
    collection_name="prompt_engineering_papers",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# Add documents to vector store
vector_store.add_documents(documents=docs)

# Define the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

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

# Define the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Function to generate questions and ground truth answers
def generate_questions_and_ground_truth(docs, num_questions=25):
    """Generate questions and ground truth answers from documents."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Combine all document chunks into a single context
    all_text = "\n\n".join([doc.page_content for doc in docs[:20]])  # Limit to first 20 chunks to avoid token limits
    
    # Create prompt for question generation
    question_prompt = ChatPromptTemplate.from_template("""
    You are an assistant for question-generation tasks.
    Use the following pieces of context to generate a list of {n} questions.
    The questions should be diverse and cover different aspects of the content.
    For each question, also provide a ground truth answer based solely on the context.
    
    Return your response in the following format:
    Question 1: [question text]
    Answer 1: [answer text]
    
    Question 2: [question text]
    Answer 2: [answer text]
    
    And so on.
    
    Context: {context}
    """)
    
    # Generate questions and answers
    print("Generating questions and ground truth answers...")
    response = llm.invoke(question_prompt.invoke({"context": all_text, "n": num_questions})).content
    
    # Parse the response
    lines = response.strip().split("\n")
    questions = []
    ground_truths = []
    
    current_question = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("Question"):
            if ":" in line:
                current_question = line.split(":", 1)[1].strip()
                questions.append(current_question)
        elif line.startswith("Answer") and current_question is not None:
            if ":" in line:
                answer = line.split(":", 1)[1].strip()
                ground_truths.append(answer)
                current_question = None
    
    # Ensure we have the same number of questions and answers
    min_length = min(len(questions), len(ground_truths))
    questions = questions[:min_length]
    ground_truths = ground_truths[:min_length]
    
    return questions, ground_truths

# Generate questions and ground truth answers
questions, ground_truths = generate_questions_and_ground_truth(docs, num_questions=5)

# Function to generate responses for evaluation
def generate_responses(rag_chain, questions):
    """Generate responses using the RAG chain."""
    print("Generating responses...")
    responses = []
    contexts = []
    
    for question in tqdm(questions):
        # Get response and context
        grounding_chain = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )
        grounding_result = grounding_chain.invoke(question)
        response = rag_chain.invoke(question)
        
        # Extract context
        context_docs = grounding_result["context"]
        context_texts = [doc.page_content for doc in context_docs]
        
        responses.append(response)
        contexts.append(context_texts)
    
    return responses, contexts

# Generate responses
responses, contexts = generate_responses(rag_chain, questions)

# Create dataset for evaluation
def create_evaluation_dataset(questions, responses, contexts, ground_truths):
    """Create a dataset for evaluation."""
    # Create DataFrame first for easier manipulation
    ragas_evals_df = pd.DataFrame({
        "question": questions,
        "response": responses,
        "context": ["\n\n".join(ctx) for ctx in contexts],
        "reference": ground_truths
    })
    
    # Convert to the format expected by RAGAS
    dataset = []
    for _, row in ragas_evals_df.iterrows():
        dataset.append({
            "user_input": row["question"],
            "retrieved_contexts": [row["context"]],  # RAGAS expects a list of contexts
            "response": row["response"],
            "reference": row["reference"]
        })
    
    # Create RAGAS evaluation dataset
    from ragas import EvaluationDataset
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    
    return evaluation_dataset, ragas_evals_df

# Create evaluation dataset
ragas_eval_dataset, ragas_evals_df = create_evaluation_dataset(questions, responses, contexts, ground_truths)

# Save to CSV for Phoenix monitoring
os.makedirs("results", exist_ok=True)
ragas_evals_df.to_csv("results/rag_results.csv", index=False)
print("Saved evaluation results to results/rag_results.csv")

# Display the first few rows
ragas_evals_df.head()

print(session.url)

# # dataset containing embeddings for visualization
# query_embeddings_df = px.Client().query_spans(
#     SpanQuery().explode(
#         "embedding.embeddings", text="embedding.text", vector="embedding.vector"
#     )
# )
# query_embeddings_df.head()

from phoenix.session.evaluation import get_qa_with_reference

# dataset containing span data for evaluation with Ragas
spans_dataframe = get_qa_with_reference(client)
spans_dataframe.head()


from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# Create LLM wrapper for evaluation
evaluator_llm = LangchainLLMWrapper(llm)

# Evaluate using RAGAS metrics
evaluation_result = evaluate(
    dataset=ragas_eval_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm
)

print(evaluation_result)
eval_scores_df = pd.DataFrame(evaluation_result.scores)

from phoenix.trace import SpanEvaluations

# Assign span ids to your ragas evaluation scores (needed so Phoenix knows where to attach the spans).
eval_data_df = pd.DataFrame(evaluation_result.dataset)

# assert eval_data_df.question.to_list() == list(
#     reversed(spans_dataframe.input.to_list())  # The spans are in reverse order.
# ), "Phoenix spans are in an unexpected order. Re-start the notebook and try again."
eval_scores_df.index = pd.Index(
    list(reversed(spans_dataframe.index.to_list())), name=spans_dataframe.index.name
)

# Log the evaluations to Phoenix.
for eval_name in eval_scores_df.columns:
    evals_df = eval_scores_df[[eval_name]].rename(columns={eval_name: "score"})
    evals = SpanEvaluations(eval_name, evals_df)
    px.Client().log_evaluations(evals)
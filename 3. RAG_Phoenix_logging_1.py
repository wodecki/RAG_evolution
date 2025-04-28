#https://github.com/Arize-ai/phoenix/blob/main/tutorials/tracing/langchain_tracing_tutorial.ipynb
import os
import csv
import pandas as pd
import nest_asyncio
from tqdm import tqdm
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openinference.instrumentation.langchain import LangChainInstrumentor

import phoenix as px
from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)
from phoenix.otel import register
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
from phoenix.trace import DocumentEvaluations, SpanEvaluations

nest_asyncio.apply()  # needed for concurrent evals in notebook environments

# Check if the vector store already exists
if os.path.exists("./chroma_langchain_db") and os.path.isdir("./chroma_langchain_db"):
    print("Loading existing vector store from ./chroma_langchain_db")
    # Load existing vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        collection_name="scientists_bios",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
else:
    print("Creating new vector store...")
    # Load and process documents
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
    print("Vector store created and documents added")

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


# Define the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Load the queries from the CSV file
queries = []
queries_csv = "./input/reference.csv"
with open(queries_csv, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        if len(row) >= 2:
            queries.append(row[0])


tracer_provider = register()
LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)

for query in tqdm(queries[:5]):
     rag_chain.invoke(query)

queries_df = get_qa_with_reference(px.Client())
retrieved_documents_df = get_retrieved_documents(px.Client())

eval_model = OpenAIModel(
    model="gpt-4o-mini",
)
hallucination_evaluator = HallucinationEvaluator(eval_model)
qa_correctness_evaluator = QAEvaluator(eval_model)
relevance_evaluator = RelevanceEvaluator(eval_model)

hallucination_eval_df, qa_correctness_eval_df = run_evals(
    dataframe=queries_df,
    evaluators=[hallucination_evaluator, qa_correctness_evaluator],
    provide_explanation=True,
)
relevance_eval_df = run_evals(
    dataframe=retrieved_documents_df,
    evaluators=[relevance_evaluator],
    provide_explanation=True,
)[0]
# Ensure the evaluation dataframes have the correct index structure for Phoenix
# For SpanEvaluations: index must be named 'context.span_id'
# For DocumentEvaluations: index must match the structure of retrieved_documents_df

# Set the proper index for span evaluations (hallucination and QA correctness)
hallucination_eval_df.index = queries_df.index
hallucination_eval_df.index.name = 'context.span_id'

qa_correctness_eval_df.index = queries_df.index
qa_correctness_eval_df.index.name = 'context.span_id'

# For document evaluations, use the original index from retrieved_documents_df
relevance_eval_df.index = retrieved_documents_df.index

# Log the evaluations to Phoenix
# Log each evaluation separately to ensure they all succeed
px.Client().log_evaluations(
    SpanEvaluations(eval_name="Hallucination", dataframe=hallucination_eval_df)
)

px.Client().log_evaluations(
    SpanEvaluations(eval_name="QA Correctness", dataframe=qa_correctness_eval_df)
)

px.Client().log_evaluations(
    DocumentEvaluations(eval_name="Relevance", dataframe=relevance_eval_df)
)

print("All evaluations successfully logged to Phoenix")
  
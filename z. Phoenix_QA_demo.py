#https://github.com/Arize-ai/phoenix/blob/main/tutorials/tracing/langchain_tracing_tutorial.ipynb
import json
import os
from getpass import getpass
from urllib.request import urlopen

import nest_asyncio
import numpy as np
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.retrievers import KNNRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openinference.instrumentation.langchain import LangChainInstrumentor
from tqdm import tqdm

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

df = pd.read_parquet(
    "http://storage.googleapis.com/arize-phoenix-assets/datasets/"
    "unstructured/llm/context-retrieval/langchain/database.parquet"
)
knn_retriever = KNNRetriever(
    index=np.stack(df["text_vector"]),
    texts=df["text"].tolist(),
    embeddings=OpenAIEmbeddings(),
)
chain_type = "stuff"  # stuff, refine, map_reduce, and map_rerank
chat_model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=chat_model_name)
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type=chain_type,
    retriever=knn_retriever,
    metadata={"application_type": "question_answering"},
)

tracer_provider = register()
LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)

url = "http://storage.googleapis.com/arize-phoenix-assets/datasets/unstructured/llm/context-retrieval/arize_docs_queries.jsonl"
queries = []
with urlopen(url) as response:
    for line in response:
        line = line.decode("utf-8").strip()
        data = json.loads(line)
        queries.append(data["query"])
queries[:10]
     

for query in tqdm(queries[:10]):
    chain.invoke(query)
    
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

px.Client().log_evaluations(
    SpanEvaluations(eval_name="Hallucination", dataframe=hallucination_eval_df),
    SpanEvaluations(eval_name="QA Correctness", dataframe=qa_correctness_eval_df),
    DocumentEvaluations(eval_name="Relevance", dataframe=relevance_eval_df),
)
  
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Load documents from data directory
from langchain_community.document_loaders import DirectoryLoader
directory = "./datasets/scientists_bios"
loader = DirectoryLoader(
    directory
)
docs = loader.load()

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = InMemoryVectorStore(embeddings)

vector_store.add_documents(documents=docs)

# Define the retriever
retriever = vector_store.as_retriever()

# Define the LLM and RAG chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
""")
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run the RAG chain
questions = ['What contributions did Ada Lovelace make to the field of computer science?  ',
 "How did Ada Lovelace's education differ from that of other women in the 19th century?  ",
 'What was the significance of the algorithm that Lovelace published for the Analytical Engine?  ',
 'What challenges did Ada Lovelace face in her personal life?  ',
 'Why is Ada Lovelace celebrated today, and how is her legacy honored?']


response = rag_chain.invoke(questions[0])
print(response)
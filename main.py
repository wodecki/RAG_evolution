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

# Chunk the documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Create a custom text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20
)

docs = loader.load_and_split(text_splitter=splitter)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()

from langchain_chroma import Chroma
#vector_store = InMemoryVectorStore(embeddings)
vector_store = Chroma(
    collection_name="scientists_bios",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

vector_store.add_documents(documents=docs)

# Define the retriever
retriever = vector_store.as_retriever()

# Define the LLM and RAG chain
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
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

grounding_chain = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run the RAG chain
questions = ["What was the significance of Ada Lovelace's contributions to computer programming?  ",
 "How did Ada Lovelace's upbringing influence her career in mathematics and science?  ",
 'What collaboration did Ada Lovelace have with Charles Babbage regarding the Analytical Engine?  ',
 'What were some visionary ideas expressed by Ada Lovelace about the potential of computing machines?  ',
 'How is Ada Lovelace recognized and celebrated today in relation to women in technology?',
 'What significant contributions did Isaac Newton make to the field of mathematics?  ',
 "How did Newton's early life and education shape his scientific career?  ",
 "What was the impact of the Great Plague on Newton's development of his theories?  ",
 "In what ways did Newton's personality influence his relationships with contemporaries like Leibniz?  ",
 "How did Newton's work lay the foundations for modern physics and astronomy?",
 'What significant theories did Albert Einstein develop that changed our understanding of physics?  ',
 "How did Einstein's work on the photoelectric effect contribute to the development of quantum theory?  ",
 "In what ways did Einstein's personal life influence his scientific career?  ",
 'Why did Einstein decide not to return to Germany in 1933, and what position did he accept in the United States?  ',
 "What is the significance of Einstein's equation E=mc² in relation to mass and energy?",
 "What were the major influences on Charles Darwin's early life and education?  ",
 "How did Darwin's voyage on the HMS Beagle impact his scientific thinking?  ",
 'What is the theory of evolution by natural selection, as proposed by Darwin?  ',
 "What were some of Darwin's contributions to the field of geology?  ",
 "How has Darwin's legacy influenced fields outside of biology?",
 'What significant contributions did Marie Curie make to the field of radioactivity?  ',
 "How did Marie Curie's early life and education shape her career as a scientist?  ",
 'What are the notable awards and recognitions that Marie Curie achieved during her lifetime?  ',
 'In what ways did Marie Curie impact medical technology during World War I?  ',
 'What challenges did Marie Curie face as a woman in the scientific community, and how did she overcome them?']


question = questions[0]
response = rag_chain.invoke(question)
grounding = grounding_chain.invoke(question)["context"] 
print("\n\n\n")
print("Question:")
print(question)
print("\n")
print("Response:")
print(response)
print("\n")
grounding = grounding[3].page_content
print("Grounding:")
print(grounding)
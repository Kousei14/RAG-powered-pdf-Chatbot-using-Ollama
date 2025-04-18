from chunk_vector_store import ChunkVectorStore as cvs
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

class Rag:
  def __init__(self) -> None:
    self.csv_obj = cvs()
    self.prompt = PromptTemplate.from_template(
      """
      <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context
      to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
        maximum and keep the answer concise. [/INST] </s>
      [INST] Question: {question}
      Context: {context}
      Answer: [/INST]
      """
    )
    self.model = ChatOllama(model = "mistral")

    self.vector_store = None
    self.retriever = None
    self.chain = None

  def set_retriever(self):
    self.retriever = self.vector_store.as_retriever(
      search_type = "similarity_score_threshold",
      search_kwargs = {"k": 3, "score_threshold": 0.5},
    )

  # Augment the context to original prompt
  def augment(self):
    self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
    | self.prompt
    | self.model
    | StrOutputParser())

  # Generate the response
  def ask(self, query: str):
    if not self.chain:
      return "Please upload a PDF file for context"

    return self.chain.invoke(query)

  # Main feed
  def feed(self, file_path: str):

    # 1. Splits PDF into chunks
    chunks = self.csv_obj.split_into_chunks(file_path = file_path)

    # 2. Store chunks into a vector DB
    self.vector_store = self.csv_obj.store_to_vector_database(chunks)

    # 3. Define retrieval parameters
    self.set_retriever()

    # 4. Define chain
    self.augment()


  def clear(self):
    self.vector_store = None
    self.chain = None
    self.retriever = None
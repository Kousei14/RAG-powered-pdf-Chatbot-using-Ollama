from chunk_vector_store import ChunkVectorStore as cvs
from openai import OpenAI

class Rag:
  def __init__(self) -> None:
    self.csv_obj = cvs()

    self.prompt = """
      <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context
      to answer the question. If you don't know the answer, just say that you don't know. Keep it your reponse concise [/INST] </s>
      [INST] Question: {question}
      Context: {context}
      Answer: [/INST] """

    self.vector_store = None
    self.retriever = None

  def ask(self, query: str):

    if not self.retriever:
      return "Please upload a PDF file for context"

    retrieved_documents = self.retriever.invoke(query)

    formatted_prompt = self.prompt.format(question = query,
                                     context = ". ".join([doc.page_content for doc in retrieved_documents]))
    
    GOOGLE_API_KEY = "AIzaSyC2gWOGsO3ANV_KeCEpcAo8YFDPrGuyJA8"
    client = OpenAI(
        api_key = GOOGLE_API_KEY,
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = client.chat.completions.create(
        model = "gemini-2.5-flash-preview-04-17",
        reasoning_effort = "medium",
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": formatted_prompt
            }
        ]
    )

    return response.choices[0].message.content

  def feed(self, file_path: str):

    chunks = self.csv_obj.split_into_chunks(file_path = file_path)
    self.vector_store = self.csv_obj.store_to_vector_database(chunks)

    self.retriever = self.vector_store.as_retriever()

  def clear(self):
    self.vector_store = None
    self.retriever = None

if __name__ == "__main__":
  chat_history = []

  rag = Rag()
  rag.feed("Gonzales_Kenaniah_CV.pdf")
  
  query = input("Query: ")
  while query != "quit":

    chat_history.append({'role' : 'user',
                         'content' : query})
    print("Query:", query)
    response = rag.ask(query)

    chat_history.append({'role' : 'assistant',
                         'content' : response})
    
    print(response)

    query = input("Query: ")

    

  
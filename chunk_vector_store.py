from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.embeddings import fastembed
from langchain_community.vectorstores import FAISS

class ChunkVectorStore:

  def __init__(self) -> None:
    pass

  def split_into_chunks(self, file_path: str):

    doc = PyPDFLoader(file_path).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, 
                                                   chunk_overlap = 50)

    chunks = text_splitter.split_documents(doc)
    # chunks = filter_complex_metadata(chunks)

    return chunks

  def store_to_vector_database(self, chunks):
    vector_store_FAISS = FAISS.from_documents(
      documents = chunks, 
      embedding = fastembed.FastEmbedEmbeddings())
    
    return vector_store_FAISS

if __name__ == "__main__":
  cvs = ChunkVectorStore()
  chunks = cvs.split_into_chunks("Gonzales_Kenaniah_CV.pdf")
  vector_store_FAISS = cvs.store_to_vector_database(chunks)
  
  retriever_FAISS = vector_store_FAISS.as_retriever()
  retrieved_docs_FAISS = retriever_FAISS.invoke("Did Kenaniah Gonzales worked at DXC Technology?")

  print(f"Retrieved Documents:")
  for doc in retrieved_docs_FAISS:
    print(f"\t{doc.id}\t{doc.page_content}")
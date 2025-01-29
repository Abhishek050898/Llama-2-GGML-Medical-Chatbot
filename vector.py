from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

pdf_path = "data/"
vector_store_path = "vectorstores/db_faiss"

def build_vector_store():
    loader = DirectoryLoader(pdf_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2", model_kwargs = {'device':'cpu'})
    vector_store = FAISS.from_documents(texts, embeddings)

    vector_store.save_local(vector_store_path)
    print(f"Vector store saved at {vector_store_path}")

if __name__ == "__main__":
    build_vector_store()
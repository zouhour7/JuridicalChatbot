from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch

cuda = torch.cuda.is_available()
if cuda:
    print("YESSS")


def load_data():
    loader = DirectoryLoader('C:\\Users\\amine\\chatbot\\cases', glob="*case.txt", show_progress=True, loader_cls=lambda x: TextLoader(x, encoding="utf-8"))
    data = loader.load()
    directory_path = "C:\\Users\\amine\\chatbot\\lawsfinal"
    loaderL = DirectoryLoader(directory_path, glob="*.txt", show_progress=True, loader_cls=lambda x: TextLoader(x, encoding="utf-8"))
    Law = loaderL.load()
    return data, Law

def create_and_persist_vector_store():
    data, Law = load_data()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    text_chunks = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={"device":"cuda"})

    if cuda:
        Demb = FAISS.from_documents(text_chunks, embedding=embeddings)
        Lemb = FAISS.from_documents(Law, embedding=embeddings)

    # Save the vector stores
    Demb.save_local("C:\\Users\\amine\\chatbot\\vectors\\Demb")
    Lemb.save_local("C:\\Users\\amine\\chatbot\\vectors\\Lemb")

create_and_persist_vector_store()


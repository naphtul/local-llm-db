import os.path
import pickle

from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


class MedicalDB:
    def __init__(self):
        self.db_name = 'medical-db.pkl'
        if os.path.exists(self.db_name):
            with open(self.db_name, 'rb') as f:
                self.vector_store = pickle.load(f)

    def load_pdfs(self, folder: str) -> None:
        loader = DirectoryLoader(folder, glob="**/*.pdf", show_progress=True,
                                 use_multithreading=False, loader_cls=UnstructuredPDFLoader)
        docs = loader.load()

        chunks = []
        for doc in docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks += text_splitter.split_text(text=doc.dict().get('page_content', ''))
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        with open(self.db_name, 'wb') as f:
            pickle.dump(self.vector_store, f)

    def interact_with_db(self, question: str) -> str:
        docs = self.vector_store.similarity_search(query=question, k=3)
        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type='stuff')
        response = chain.run(input_documents=docs, question=question)
        return response


if __name__ == '__main__':
    mdb = MedicalDB()
    mdb.load_pdfs('C:\\Users\\Public\\Desktop\\test')
    # print(mdb.interact_with_db('When did Michael get his first Tdap shot?'))
    # print(mdb.interact_with_db('on what date did michael get his first measles shot?'))
    # print(mdb.interact_with_db('on what date did michael get his most recent Hepatitis A shot?'))
    # print(mdb.interact_with_db('on what date did michael get his Prevnar 13 shot?'))

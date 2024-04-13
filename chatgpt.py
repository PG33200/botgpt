import os
import sys
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Assurez-vous que c'est importé de langchain-openai
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, RetrievalQA

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

PERSIST = False
query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

class OpenAIEmbeddingWrapper:
    def __init__(self):
        self.embedder = OpenAIEmbeddings()

    def invoke(self, input):
        return self.embedder([input])[0]

if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddingWrapper())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = TextLoader("data/data.txt")
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    result = chain({"question": query, "chat_history": chat_history})
    print(result['answer'])
    chat_history.append((query, result['answer']))
    query = None

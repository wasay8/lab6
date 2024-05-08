from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOllama
from langchain.document_loaders import PDFPlumberLoader

# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFaceTextGenInference

# from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


# TODO needs streaming, needs better idea of document, probably needs sourcery to take a look
# BUG Callbacks creating duplicates


def get_vectorstore(documents_chunks):
    # embeddings = OpenAIEmbeddings()

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={
            "normalize_embeddings": True
        },  # set True to compute cosine similarity
    )

    return FAISS.from_documents(documents=documents_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    # llm = HuggingFacePipeline.from_model_id(
    #     model_id="lmsys/vicuna-7b-v1.3",
    #     task="text-generation",
    #     model_kwargs={"temperature": 0.01},
    # )
    # llm = LlamaCpp(
    #     model_path="models/llama-2-7b-chat.ggmlv3.q4_1.bin",  n_ctx=1024, n_batch=512)

    # llm = HuggingFaceTextGenInference(
    #     # model_id = 'TheBloke/LlamaChat-2-13B-chat-AWQ',
    #     inference_server_url="http://0.0.0.0:8080",
    #     max_new_tokens=512,
    #     top_k=10,
    #     top_p=0.95,
    #     typical_p=0.95,
    #     temperature=0.01,
    #     repetition_penalty=1.03,
    #     streaming=True,
    #     callbacks=[StreamingStdOutCallbackHandler()],
    # )

    llm = ChatOllama(
        model="llama2:13b-chat",
        device="cuda",
        max_tokens=1000,
        # callbacks=[StreamingStdOutCallbackHandler()],
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        ),
        memory=memory,
    )


def main():
    PDF_PATH = "Ads cookbook .pdf"

    #  PDF Extraction And Text Chunk
    pdf_document_chunks = PDFPlumberLoader(PDF_PATH).load_and_split(
        RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    )
    print(pdf_document_chunks[:100])
    # Vector _Datastore
    vector_datastore = get_vectorstore(pdf_document_chunks)
    print("Vectorizing Completed")

    # Conversation Chain
    conversation_chain = get_conversation_chain(vector_datastore)

    while True:
        user_question = input("Question: ")
        if user_question.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break

        response = conversation_chain({"question": user_question})
        print("AI: ", response["answer"])


if __name__ == "__main__":
    main()

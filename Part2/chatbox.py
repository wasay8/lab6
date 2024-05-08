import streamlit as st
from PyPDF2 import PdfReader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOllama
from langchain.document_loaders import PDFPlumberLoader

# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFaceTextGenInference
from htmlTemplates import css, bot_template, user_template
# from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.vectorstores import FAISS
import torch


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            t = page.extract_text()
            t = t.replace("..","")
            text+=t
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100,length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(texts_chunks):
    #embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceEmbeddings(
        #model_name="sentence-transformers/all-MiniLM-L6-v2",
        #)


    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        },  # set True to compute cosine similarity
    )

    return FAISS.from_texts(texts=texts_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI()
    # llm = HuggingFacePipeline.from_model_id(
    #     model_id="lmsys/vicuna-7b-v1.3",
    #     task="text-generation",
    #     model_kwargs={"temperature": 0.01},
    # )
    llm = ChatOllama(
        model="llama2",
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


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Chat with PDFs",
                       page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs :robot_face:")
    user_question = st.text_input("Ask questions about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                print("Splited Completed")

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                print("Vectorizing Completed")

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()

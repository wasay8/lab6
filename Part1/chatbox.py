# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader

"""# PDF Extraction"""

def extract_text_from_pdf_folder(pdf_path):
    text_data = []
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()
        text_data.append(text)

    return text_data


"""# Text Chunk"""

from langchain.text_splitter import CharacterTextSplitter

def split_text_into_chunks(text_data):
    if isinstance(text_data, str):
        splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100,length_function=len)
        chunks = splitter.split_text(text_data)
        return chunks
    else:
        raise ValueError("text_data should be a string")

"""# Vector DataStore"""
from langchain.vectorstores import FAISS
# from transformers import AutoModel, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
# import os
# import getpass
import torch

def create_vector_datastore(text_chunks):
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Add a new padding token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModel.from_pretrained(model_name)

    embeddings = []

    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the embeddings from the model's output
        pooled_output = outputs['last_hidden_state'].mean(dim=1).squeeze().numpy()
        embeddings.append(pooled_output)

    """
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

    #return embeddings

from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from langchain.llms import LlamaCpp
import numpy as np
def get_user_question():
    user_input = input("Question: ")
    if user_input.lower() != 'exit':
      user_input_2 = 100
      return user_input, user_input_2
    return user_input, 0


def create_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key="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",
     max_tokens=1000)
    #llm = LlamaCpp(
    #    model_path="models/llama-7b.ggmlv3.q4_1.bin",  n_ctx=1024, n_batch=512,verbose=True)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}),
        memory=memory,
        max_tokens_limit=4000
    )
    return conversation_chain

def main():

    # PDF Extraction
    extracted_text = extract_text_from_pdf_folder("Ads cookbook .pdf")
    print("PDF Extraction Completed")

    # Text Chunk
    text_chunks = split_text_into_chunks(str(extracted_text))
    print("Text Chunk Completed")


    # Vector _Datastore
    vector_datastore = create_vector_datastore(text_chunks)  # Assuming you have the text chunks
    print("Vectorizing Completed")


    # Conversation Chain
    conversation_chain = create_conversation_chain(vector_datastore)

    while True:
        user_question, max_length = get_user_question()
        if user_question.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break

        response = conversation_chain({'question': user_question})
        print("AI: ",response['answer'])

if __name__ == "__main__":
    main()

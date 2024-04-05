import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI 

app = FastAPI() 

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
processed = False  


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
   Bạn là trợ lý ảo hữu dụng, trả lời dài và đầy đủ câu hỏi người dùng một cách nhiệt tình, đầy đủ về các loại bệnh. 
   Bạn phải trả lời dạng một đoạn văn, không liệt kê.
   Sau đó bạn đưa ra thêm lời khuyên sau cùng nên làm gì tiếp theo để giảm các triệu chứng bệnh cho người bệnh.n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.6)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    global processed

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    st.write("Reply: ", response["output_text"])
    processed = True
    return response


def main():
    global processed
    st.set_page_config("Chat AI")
    st.header("Chat with MedCare💁")

  
    # if user_question:
    #     user_input(user_question)
    vector_store_exists = os.path.exists("faiss_index")

    with st.sidebar:
        st.title("Menu:")
        folder_path = "docs"
        if folder_path and not processed and vector_store_exists == False:  # Check if processing has been done
            with st.spinner("Processing..."):
                pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pdf')]
                st.warning(pdf_files)
                raw_text = get_pdf_text(pdf_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
        elif processed:
            st.info("Processing already done.")
    df = pd.read_excel('test_question.xlsx')
    data = []
    for i in range(df.shape[0]):
        text = df[i:i+1].to_string(index=False)
        sentences = text.split('?')
        second_question = sentences[1].strip() + '?'
        st.write(second_question);
        user_response = user_input(second_question)
        # st.write(user_response)
        data += [second_question, user_response["output_text"]]
    print(data)
    df = pd.DataFrame(data)
    df.to_excel('output.xlsx', index=False, header=False)


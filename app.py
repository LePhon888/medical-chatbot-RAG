import streamlit as st
from PyPDF2 import PdfReader
import os
import uvicorn
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI, Request, Query

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
from transformers import AutoModel, AutoTokenizer
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from pyvi.ViTokenizer import tokenize
from huggingface_hub import login
login(token="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")

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
   Bạn chỉ trả lời các câu hỏi liên quan đến bệnh học, y tế, sức khỏe, chăm sóc sức khỏe. Nếu ngoài chủ đề trên bạn trả lời: Xin lỗi, tôi không thể trả lời câu hỏi này.
   Bạn chỉ trả lời các câu hỏi có trong cơ sở dữ liệu của bạn, không trả lời các câu hỏi ngoài cơ sở dữ liệu của bạn.
   Bạn nên trả lời một cách thân thiện như một bác sĩ đưa ra lời khuyên
   Bạn phải trả lời dạng một đoạn văn, không liệt kê.
   Bạn nên trả lời một cách chi tiết, đầy đủ, không để người dùng phải hỏi lại.
   Bạn nên trả lời có chủ ngữ, vị ngữ.
   Sau đó bạn đưa ra thêm lời khuyên sau cùng nên làm gì tiếp theo để giảm các triệu chứng bệnh cho người bệnh. \n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # model = ChatGoogleGenerativeAI(model="gemini-pro",
    #                          temperature=0.6)
    PhobertTokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    hf = HuggingFacePipeline.from_model_id(
        model_id="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 100}  # Adjust the number of tokens based on your needs
    )

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    # chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    chain = load_qa_chain(llm=hf, chain_type="stuff", prompt=prompt)
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


    
@app.get("/api/chat/") 
async def chat_response(msg: str = Query(...)): 
    user_response = user_input(msg)
    return {"message": user_response["output_text"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
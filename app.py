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
from fastapi import FastAPI, Request, Query
from langchain_cohere import CohereEmbeddings
import uvicorn

app = FastAPI() 

load_dotenv()
os.getenv("GOOGLE_API_KEY")
os.getenv("COHERE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
genai.configure(api_key=os.getenv("COHERE_API_KEY"))
processed = False  


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=5000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
#         Bạn được phép đưa ra lời khuyên y tế.\n
#     Bắt buộc trả lời một đoạn văn dài, cung cấp thông tin thật chi tiết, cụ thể.\n
#     Nếu thông tin bị ngắn bạn được phép đoán dựa trên kiến thức chuyên môn của mình.\n
#     Nếu có thể bạn nên diễn giải chi tiết không được liệt kê\n
#    Bạn nhận được câu hỏi có từ điều trị thì bạn trả lời là cách điều trị bệnh, cách chăm sóc sức khỏe, cách phòng tránh bệnh.\n
#    Bạn phải trả lời có chủ ngữ, vị ngữ.\n
#     Bạn chỉ trả lời các câu hỏi liên quan đến bệnh học, y tế, sức khỏe, chăm sóc sức khỏe, các loại bệnh \n
#    Sau đó bạn đưa ra thêm lời khuyên sau cùng nên làm gì tiếp theo để giảm triệu chứng bệnh.
    prompt_template = """
    Bạn là một chuyên gia về lĩnh vực y học, sức khỏe, chăm sóc sức khỏe, bệnh học.\n
    Nhiệm vụ của bạn là trả lời câu hỏi của người dùng về y học, sức khỏe, chăm sóc sức khỏe, bệnh học.\n
    Bạn hãy trả lời ưu tiên theo ngữ cảnh, nếu ngữ cảnh không phù hợp thì hãy trả lời theo cách của bạn.\n
    Bạn hãy trả lời theo format đoạn văn dài, cung cấp thông tin thật chi tiết, cụ thể.\n
   \n
    Đây là ngữ cảnh:\n {context}?\n
    Đây là câu hỏi của người dùng: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.5)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    global processed

    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question) #topk=5
    print("===============================================================================================================")
    print(docs)
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    processed = True
    return response


def main():
    global processed
  
    with st.sidebar:
        folder_path = "docs"
        if folder_path : 
            pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pdf')]
            print(pdf_files)
            raw_text = get_pdf_text(pdf_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
        # df = pd.read_excel('test_question.xlsx')
        # data = []
        # for i in range(df.shape[0]):
        #     text = df[i:i+1].to_string(index=False)
        #     sentences = text.split('?')
        #     second_question = sentences[1].strip() + '?'
        #     st.write(second_question);
        #     user_response = user_input(second_question)
        #     # st.write(user_response)
        #     data += [second_question, user_response["output_text"]]
        # print(data)


    
@app.get("/api/chat/") 
async def chat_response(msg: str = Query(...)): 
    # main()
    # return {"message": "Hello World"}
    user_response = user_input(msg)
    return {"message": user_response["output_text"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
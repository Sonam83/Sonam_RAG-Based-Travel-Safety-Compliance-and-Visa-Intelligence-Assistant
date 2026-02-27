import streamlit as st
import dotenv
import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["GOOGLE_API_KEY"]=os.getenv("gemini")
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough,RunnableSequence

#page setting
st.set_page_config(
    page_title="Travel & Tourism",
    page_icon="🌍",
    layout="wide")

st.title("🌍 RAG Based Travel Safety, Compliance & Visa Intelligence Assistant")
st.write("Put your Query related to US visa, immigration, travel safety and compliance.")

# Document Loader

@st.cache_resource
def load_documents():
    pdf_files=[r"D:\RAG_UI\visa\rag_folder\US Health and Safety rules.pdf",
    r"D:\RAG_UI\visa\rag_folder\US immigration rules.pdf",
    r"D:\RAG_UI\visa\rag_folder\US Visas for Indians.pdf"]
    docs=[]
    for file in pdf_files:
        loader=PyPDFLoader(file)
        docs.extend(loader.load())
    return docs  #all pages from 3files


documents=load_documents()

#Text Splitting
@st.cache_resource
def split_docs(docs):
    splitter=RecursiveCharacterTextSplitter(separators=["\n\n","\n"," ",""],chunk_size=800,chunk_overlap=150)
    return splitter.split_documents(docs)

chunks=split_docs(documents)

#Embeddings
@st.cache_resource

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

embedding=load_embeddings()

#Vector Database
@st.cache_resource

def create_vectorstore(chunks):
    return Chroma.from_documents(documents=chunks,embedding=embedding)

db=create_vectorstore(chunks)
retriever=db.as_retriever(search_type="similarity",search_kwargs={"k":5})

#prompt template
prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template('''

You are an expert and experienced Travel Safety, Compliance, and Visa Intelligence Assistant.

STRICT RULES:
1. Use ONLY the provided context.
2. Do NOT assume, fabricate, or generate information outside the context.
3. If the answer is not found in the context, respond:
   "Sorry - I cannot answer based on the provided information. Please be specific and relevant."

Provide a professional, structured, and detailed response under the following headings:

1) Visa Rules
2) Visa Types
3) Immigration Rules
4) Immigration Types
5) Safety Guidelines
6) Compliance Requirements
7) Official Contact Information  
   - Provide phone numbers with city names.
   - Mention business hours if available.
   - Advise users to keep Application ID/Case Number, Last Name, and Date of Birth ready if mentioned in context.

8) Address for Visa Application  
   - Provide official Embassy or Consulate addresses where the applicant must apply.
   - Mention city and full address if available in the context.

9) Official Source Website Links  
   - Provide the EXACT official clickable website URLs from the context.
   - Include links for visa application, appointment booking, DS-160 form, status tracking, and emergency services.
   - Do NOT mention only website names; provide full URLs.
   - If links are not available in the context, clearly state:
     "Exact official website link not available in provided context."
   - Do NOT fabricate or assume URLs.

10) Important Warnings  
   - Include compliance risks, visa rejection risks, fraud alerts, and legal warnings if present in the context.

Important:
- Always maintain a professional and authoritative tone.
- Present addresses and links in bullet format for clarity.
- If multiple consulates or application centers exist, list them city-wise.

'''),

HumanMessagePromptTemplate.from_template('''
Context: {context}
Question: {question}
''')
])

 #LLM
 
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

llm=load_llm()

#rag chain
def format_doc(docs):
  return "\n\n".join(doc.page_content for doc in docs)
rag_chain=(RunnableParallel({"context":retriever | RunnableLambda(format_doc),"question":RunnablePassthrough()})|prompt|llm)

#session memory
if "messages" not in st.session_state:
    st.session_state.messages=[]
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


#Query
query=st.chat_input("Ask your Query.....")
if query:
    st.session_state.messages.append({"role":"user","content":query})
    with st.chat_message("user"):
        st.markdown(query)
    response=rag_chain.invoke(query)
    answer=response.content

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role":"assistant","content":answer})

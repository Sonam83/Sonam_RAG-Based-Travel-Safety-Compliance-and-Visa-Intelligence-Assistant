# Title : RAG-Based-Travel-Safety-Compliance-and-Visa-Intelligence-Assistant

**Domain **: Travel and Tourism

**Problem Statement and Motivation** : Individuals and Organizations often struggle with 

Frequent changes in visa and immigrations rules

Lack of reliable, centralized travel safety information

Compliance risks due to outdated or incorrect guidance

Difficulty accessing verified health and regulatory guidelines

Traditional chatbots generating responses on training data, may lead to hallucinations. This project addresses these challenges by combining retrieval from verified sources with generative AI to provide accurate and explainable travel intelligence.

**Why Retrieval Augmented Generation(RAG) ?**

Dynamic and up-to-date, Data privacy, High accuracy with updated information, Trust, Cost efficiency

**Datasets**

Visa and immigration data from Kaggle

U.S. immigration policy documentation from American Immigration Council

Travel safety and health guidelines from Occupational Safety and Health Administration

Regulatory and compliance PDFs and structured datasets

**System Architecture and pipeline**

Document Ingestion

Extracts information from PDFs and structured files using LangChain document loaders.

Text Preprocessing and Cleaning

Removes noise, duplicates, and formatting issues.

Chunking

Uses Recursive Character Text Splitter to create meaningful context-aware chunks.

Embedding Generation

Converts text into vector representations using models from Hugging Face.

Vector Storage

Stores embeddings in Chroma vector database for efficient semantic search.

Retrieval Layer

Retrieves relevant context based on user queries using similarity search.

Prompt Engineering

Combines user query with retrieved context to guide the LLM.

LLM Response Generation

Uses advanced generative models such as Google Gemini to generate grounded, accurate responses.

Answer Formatting and Explainability

Provides structured, transparent, and context-based answers.

**Tech Stack**

Programming: Python

Framework: LangChain

Document Loader: PyPDFLoader, UnstructuredPDFLoader

Text Splitter: RecursiveCharacterTextSplitter

Vector Database: Chroma

LLM: Gemini

Embeddings: HuggingFace Sentence Transformers

Frontend: Streamlit

User Interface and Deployment

The system includes a conversational chatbot interface that allows users to:

Ask travel-related queries in natural language

Get instant visa and compliance guidance

Access safety and health recommendations

Retrieve policy-based answers with sources


**Performance Optimization**

The project uses caching techniques to improve efficiency:

@st.cache_resource ensures heavy resources such as embeddings, vector databases, and LLM models load only once, reducing latency and cost.



import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import requests

# Load environment variables
load_dotenv()

# Document Loading and Splitting
def load_and_split_documents(urls):
    all_splits = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    for url in urls:
        loader = WebBaseLoader(url)
        data = loader.load()
        all_splits += text_splitter.split_documents(data)
    return all_splits

# Set up embeddings and vector store
def setup_vector_store(documents):
    local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(documents=documents, embedding=local_embeddings)

# Function to query the LLaMA 3 endpoint
def query_llama3(inputs, context):
    API_URL = os.getenv(API_URL)
    headers = {
        "Authorization": "Bearer {API_ENDPOINT}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": f"{context}\n\nQuestion: {inputs}",
        # "parameters": {
        #     "top_k": 5,
        #     "top_p": 0.5,
        #     "temperature": 0.4,
        #     "max_new_tokens": 1500,
        #     "return_text": True,
        #     "return_full_text": True
        # }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    # Assuming the response is a list of dictionaries
    response_data = response.json()
    
    if isinstance(response_data, list) and len(response_data) > 0:
        return response_data[0].get('generated_text', '')
    else:
        return "No valid response from the model."

# Create RAG chain
RAG_TEMPLATE = """
You are an assistant specialized in answering questions related to MeraEvents. If the question is outside the scope of MeraEvents, simply respond by saying that you are only able to answer questions related to MeraEvents.

You should answer in the same language as the question, or in the user's preferred language if specified.

Use the following pieces of retrieved context to answer the question. If you don't know the answer based on the context, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# Chain setup with custom LLaMA 3 query function
def generate_answer(question, context):
    formatted_context = "\n\n".join(doc.page_content for doc in context)
    prompt = rag_prompt.format(context=formatted_context, question=question)
    return query_llama3(prompt, formatted_context)

# Load and split documents
urls = ["url1", "url2", "url3"]
documents = load_and_split_documents(urls)
vectorstore = setup_vector_store(documents)

# Create Flask app
app = Flask(__name__)

@app.route('/events', methods=['POST'])
def ask():
    data = request.json
    question = data.get('query')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Perform similarity search and get response
    docs = vectorstore.similarity_search(question)
    response = generate_answer(question, docs)
    
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)

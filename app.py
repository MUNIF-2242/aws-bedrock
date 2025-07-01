import boto3
import streamlit as st
import json
from typing import Iterator

# LangChain Imports
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_community.llms import Bedrock
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Client
bedrock = boto3.client(service_name="bedrock-runtime")

# Embeddings Model
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock
)

# Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# Save FAISS Vector Store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# Claude LLM
def get_claude_llm():
    return BedrockChat(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        client=bedrock,
        model_kwargs={'max_tokens': 512}
    )

# LLaMA2 LLM
def get_llama2_llm():
    return Bedrock(
        model_id="meta.llama3-70b-instruct-v1:0",
        client=bedrock,
        model_kwargs={'max_gen_len': 512}
    )

# Prompt Template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least 50 words with 
detailed explanations. If you don't know the answer, just say that you don't know.
<context>
{context}
</context>

Question: {question}

A:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# RetrievalQA
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# Get context
def get_relevant_context(vectorstore_faiss, query):
    retriever = vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])

# Claude Streaming
def stream_claude_response(context, question, container):
    messages = [
        {
            "role": "user",
            "content": f"""Use the following pieces of context to provide a 
concise answer to the question at the end but use at least 50 words with 
detailed explanations. If you don't know the answer, just say that you don't know.

Context:
{context}

Question: {question}"""
        }
    ]
    
    body = {
        "messages": messages,
        "max_tokens": 512,
        "anthropic_version": "bedrock-2023-05-31"
    }

    try:
        response = bedrock.invoke_model_with_response_stream(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps(body)
        )

        full_response = ""
        for event in response['body']:
            chunk = event.get('chunk')
            if chunk:
                chunk_data = json.loads(chunk['bytes'].decode())
                if chunk_data.get("type") == "content_block_delta":
                    if "delta" in chunk_data and "text" in chunk_data["delta"]:
                        token = chunk_data["delta"]["text"]
                        full_response += token
                        container.markdown(full_response)

    except Exception as e:
        container.error(f"Streaming error: {str(e)}")

# LLaMA2 Streaming
def stream_llama2_response(context, question, container):
    formatted_prompt = f"""<s>[INST] Use the following context to answer the question at the end.
Answer in at least 50 words with detailed explanation.
If you don’t know the answer, say “I don’t know”.

Context:
{context}

Question: {question} [/INST]
"""

    body = {
        "prompt": formatted_prompt,
        "max_gen_len": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    try:
        response = bedrock.invoke_model_with_response_stream(
            modelId="meta.llama3-70b-instruct-v1:0",
            body=json.dumps(body)
        )

        full_response = ""
        for event in response["body"]:
            chunk = event.get("chunk")
            if chunk:
                decoded_chunk = chunk["bytes"].decode("utf-8").strip()
                # Each chunk can have one or more JSON lines, so split and parse each
                for line in decoded_chunk.splitlines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("generation")
                        if token:
                            full_response += token
                            container.markdown(full_response)
                        if data.get("stop_reason") == "stop":
                            return  # End streaming on stop
                    except json.JSONDecodeError:
                        # skip invalid lines
                        continue

    except Exception as e:
        container.error(f"Streaming error: {str(e)}")






# Streamlit UI
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    col1, col2 = st.columns(2)
    

    with col1:
        if st.button("Claude Output"):
            with st.spinner("Processing..."):
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_claude_llm()
                response = get_response_llm(llm, faiss_index, user_question)
                st.write(response)
                st.success("Done")

    with col2:
        if st.button("Llama2 Output"):
            with st.spinner("Processing..."):
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_llama2_llm()
                response = get_response_llm(llm, faiss_index, user_question)
                st.write(response)
                st.success("Done")

    st.subheader("Streaming Responses")
    col3, col4 = st.columns(2)

    with col3:
        if st.button("Claude Streaming"):
            if user_question:
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                context = get_relevant_context(faiss_index, user_question)
                response_container = st.empty()
                stream_claude_response(context, user_question, response_container)
                st.success("Streaming Complete!")
            else:
                st.warning("Please enter a question first!")

    with col4:
        if st.button("Llama2 Streaming"):
            if user_question:
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                context = get_relevant_context(faiss_index, user_question)
                response_container = st.empty()
                stream_llama2_response(context, user_question, response_container)
                st.success("Streaming Complete!")
            else:
                st.warning("Please enter a question first!")

if __name__ == "__main__":
    main()

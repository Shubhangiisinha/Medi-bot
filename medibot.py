import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )


def main():
    st.set_page_config(page_title="MediBot - Your AI Medical Assistant", page_icon="🤖", layout="wide")
    
    st.markdown(
        """
        <style>
            .chat-container { max-width: 800px; margin: auto; }
            .title { text-align: center; font-size: 36px; font-weight: bold; color: #4CAF50; }
            .subtext { text-align: center; font-size: 18px; color: #555; }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<div class='title'>🤖 MediBot - Your AI Medical Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtext'>Ask medical-related questions and get instant AI-powered responses.</div>", unsafe_allow_html=True)
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    with st.container():
        for message in st.session_state.messages:
            role = "user" if message['role'] == 'user' else "assistant"
            with st.chat_message(role):
                st.markdown(message['content'])
    
    prompt = st.chat_input("Type your medical question here...")
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Provide responses based strictly on the given context.

        Context: {context}
        Question: {question}
        """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response.get("result", "I'm not sure how to answer that.")
            source_documents = response.get("source_documents", [])

            result_to_show = f"**Answer:** {result}\n\n**Source Docs:** {source_documents}"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
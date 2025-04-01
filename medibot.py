import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    """Loads the FAISS vector store with embeddings."""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    """Creates a custom prompt template."""
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

@st.cache_resource(show_spinner=False)
def load_llm(huggingface_repo_id):
    """Loads the HuggingFace LLM with a secure token."""
    HF_TOKEN = st.secrets.get("HF_TOKEN")
    if not HF_TOKEN:
        st.error("Hugging Face API token is missing! Add it in Streamlit secrets.")
        return None
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"max_length": 512},
        huggingfacehub_api_token=HF_TOKEN
    )

def main():
    """Main function for the chatbot UI."""
    st.title("Ask Chatbot!")
    st.sidebar.title("MediBot")
    st.sidebar.info("Amans AI-powered medical assistant")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    prompt = st.chat_input("Pass your prompt here")
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer user's question.
            If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
            Don't provide anything out of the given context.

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
        """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        
        try:
            vectorstore = get_vectorstore()
            if not vectorstore:
                st.error("Failed to load the vector store.")
                return
            
            llm = load_llm(HUGGINGFACE_REPO_ID)
            if not llm:
                return
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            response = qa_chain.invoke({'query': prompt})
            
            result = response.get("result", "I'm sorry, but I couldn't find a clear answer to your question.").strip()
            source_documents = response.get("source_documents", [])
            
            sources_text = "\n\n📚 **Source Documents:**\n" if source_documents else ""
            for doc in source_documents:
                page = doc.metadata.get('page', 'N/A')
                source = doc.metadata.get('source', 'Unknown Source').split("\\")[-1]
                text_content = doc.page_content.strip()[:500]
                sources_text += f"📄 **{source} (Page {page})**\n{text_content}\n\n"
            
            final_response = f"{result}{sources_text}"
            st.chat_message('assistant').markdown(final_response)
            st.session_state.messages.append({'role': 'assistant', 'content': final_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

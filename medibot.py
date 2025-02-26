import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm

def main():

    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    st.sidebar.title("MediBot")
    st.sidebar.info("Aman AI-powered medical assistant")


    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})
        
        #st.session_state.messages.append({'role':'assistant', 'content': response})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")
        
        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            response = qa_chain.invoke({'query': prompt})

            # Extract main response
            result = response.get("result", "").strip()

            # Extract relevant source documents
            source_documents = response.get("source_documents", [])

            if not result:
                result = "I'm sorry, but I couldn't find a clear answer to your question."

            # Format sources properly
            sources_text = "\n\n📚 **Source Documents:**\n"
            for doc in source_documents:
                page = doc.metadata.get('page', 'N/A')
                source = doc.metadata.get('source', 'Unknown Source').split("\\")[-1]  # Show only filename
                text_content = doc.page_content.strip()[:500]  # Extract first 500 characters

                sources_text += f"📄 **{source} (Page {page})**\n{text_content}\n\n"

            # Remove source section if no useful documents
            if not source_documents:
                sources_text = ""

            # Combine response  
            final_response = f"{result}{sources_text}"

            # Display output
            st.chat_message('assistant').markdown(final_response)
            st.session_state.messages.append({'role': 'assistant', 'content': final_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
    
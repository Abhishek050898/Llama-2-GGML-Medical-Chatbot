from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from llama_cpp import Llama
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import streamlit as st

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""
DB_FAISS_PATH = "vectorstores/db_faiss"

def load_vector_store(vector_store_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store


def custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input=['context','question'])
    return prompt

    
def load_llm():
    llm = CTransformers(
        #model = "TheBloke/llama-2-7b-chat.ggmlv3.q4_K.bin",
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

def build_qa_chain(llm,vector_store,prompt):
    retriever = vector_store.as_retriever(search_kwargs={"k":2})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents = True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def qa_bot(query):
    llm=load_llm()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    prompt = custom_prompt()
    qa_chain = build_qa_chain(llm,db,prompt)
    response = qa_chain({"query":query})
    return response['result']




if "conversation" not in st.session_state:
    st.session_state.conversation = []

def main():
    st.set_page_config(page_title="Llama-2-GGML Medical Chatbot")

    # Sidebar with metadata
    with st.sidebar:
        st.title('Llama-2-GGML Medical Chatbot! 🚀🤖')
        st.markdown('''
        ## About
        The Llama-2-GGML Medical Chatbot uses the **Llama-2-7B-Chat-GGML** model and was trained on medical data from **"The GALE ENCYCLOPEDIA of MEDICINE"**.
        
        ### 🔄 Bot evolving, stay tuned!
        ## Useful Links 🔗
        - **Model:** [Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML) 📚
        ''')
        if st.button("Clear Chat"):
            st.session_state.conversation = []  # Clear chat history

    st.title("Llama-2-GGML Medical Chatbot")

    # Custom CSS for styling chat bubbles
    st.markdown(
        """
        <style>
            .chat-container {
                display: flex;
                flex-direction: column;
                height: 400px;
                overflow-y: auto;
                padding: 10px;
            }
            .user-bubble {
                background-color: #007bff;
                color: white;
                align-self: flex-end;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
                max-width: 70%;
                word-wrap: break-word;
            }
            .bot-bubble {
                background-color: #f1f1f1;
                color: black;
                align-self: flex-start;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
                max-width: 70%;
                word-wrap: break-word;
            }
        </style>
        """, unsafe_allow_html=True
    )

    # Chat input
    query = st.text_input(
    "Ask your question here:", 
    placeholder="Type your medical question here...",
    key="input_box"  # Use a different key to avoid conflicts
)

    if st.button("Get Answer"):
        if query:
            with st.spinner("Processing your question..."):
                # Append user question
                st.session_state.conversation.append({"role": "user", "message": query})

                # Get chatbot response
                answer = qa_bot(query)

                # Append bot response
                st.session_state.conversation.append({"role": "bot", "message": answer})

            # Clear input field after submission
            st.session_state["user_input"] = ""
            #st.session_state.input_box = ""

        else:
            st.warning("Please enter a question before submitting.")

    # Display chat history
    chat_container = st.empty()
    chat_bubbles = ''.join(
        [f'<div class="{c["role"]}-bubble">{c["message"]}</div>' for c in st.session_state.conversation]
    )
    chat_container.markdown(f'<div class="chat-container">{chat_bubbles}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()












# def main():
#     st.set_page_config(page_title="Medical Chatbot")
#     st.title("Llama-2 Medical Chatbot")

#     query = st.text_input("Ask your medical question:")
#     if st.button("Get Answer"):
#         if query:
#             with st.spinner("Fetching your answer..."):
#                 #vector_store = load_vector_store("vectorstores/db_faiss")
#                 #llm = load_llm()
#                 answer = qa_bot(query)
#                 st.write(f"**Answer:** {answer}")
#         else:
#             st.warning("Please enter a question.")

# if __name__ == "__main__":
#     main()
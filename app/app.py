import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
import os

# Set up Streamlit page configuration
st.set_page_config(page_title="Chatbot Psikologis", page_icon="🤖")
st.title("Chatbot Psikologis untuk dukungan kesehatan mental mahasiswa akhir")

# Function to install required libraries
def install_libraries():
    import subprocess
    subprocess.run(["pip", "install", "langchain-huggingface", "huggingface_hub", 
                    "transformers", "accelerate", "bitsandbytes", 
                    "langchain", "langchain_community"], check=True)

# Install required libraries
install_libraries()

# Set up environment secret keys
sec_key = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = sec_key

# Function to get response from the model
def get_response(user_query, chat_history, repo_id):
    template = """
    Anda adalah sebuah chatbot psikologis yang dirancang untuk dukungan kesehatan mental mahasiswa akhir. Jawab pertanyaan berikut dengan mempertimbangkan riwayat percakapan. Jangan tambahkan pertanyaan atau komentar tambahan, hanya berikan jawaban singkat yang langsung sesuai dengan pertanyaan pengguna:

    Riwayat percakapan: {chat_history}

    Pertanyaan pengguna: {user_question}
    """

    # Prepare chat history as a string
    chat_history_str = "\n".join([f"User: {msg['content']}" if msg['role'] == "user" else f"AI: {msg['content']}" for msg in chat_history])

    prompt = template.format(chat_history=chat_history_str, user_question=user_query)
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, token=sec_key)
    response = llm.invoke(prompt)
    
    return response

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "AI", "content": "Halo, saya adalah bot psikologis yang dirancang untuk membantu Anda. Bagaimana saya bisa membantu Anda hari ini?"},
    ]

# Select model repo ID
repo_id_options = ["HuggingFaceH4/zephyr-7b-beta", "meta-llama/Meta-Llama-3-8B-Instruct"]
selected_repo_id = st.selectbox("Pilih Model", repo_id_options)

# Display conversation history
for message in st.session_state.chat_history:
    if message["role"] == "AI":
        with st.chat_message("AI"):
            st.write(message["content"])
    elif message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])

# User input
user_query = st.chat_input("Ketik pesan Anda di sini...")
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history, selected_repo_id)
        st.write(response)

    st.session_state.chat_history.append({"role": "AI", "content": response})

import streamlit as st
import json
import os
from chat import ChatBotManager
from vectors import VectorManager
from langchain.memory import ConversationBufferMemory

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Constants
MODEL_NAME = "gemini-1.5-pro"
HISTORY_FILE = "chat_history.json"

# Initialize chatbot and vector manager
chatbot = ChatBotManager(model_name=MODEL_NAME)
vector_manager = VectorManager(model_name="models/embedding-001")

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load chat history
def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)
    return {}

# Save chat history
def save_chat_history(history):
    with open(HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

# Streamlit UI
st.title("📄 AI-Powered Chatbot with PDF Support")

# User input for name and phone number
if "username" not in st.session_state:
    st.session_state.username = ""
    st.session_state.phone = ""

st.session_state.username = st.text_input("Enter your name:", value=st.session_state.username)
st.session_state.phone = st.text_input("Enter your phone number:", value=st.session_state.phone)

if st.session_state.username and st.session_state.phone:
    user_id = f"{st.session_state.username}_{st.session_state.phone}"
    chat_history = load_chat_history()
    if user_id not in chat_history:
        chat_history[user_id] = []
    
    # Load past messages into Langchain memory
    for entry in chat_history[user_id]:
        memory.chat_memory.add_user_message(entry["user"])
        memory.chat_memory.add_ai_message(entry["response"])
    
    # Display chat history
    st.subheader("📝 Chat History")
    for entry in chat_history[user_id]:
        with st.expander(f"You: {entry['user']}"):
            st.write(f"**AI:** {entry['response']}")
    
    # PDF Upload
    uploaded_files = st.file_uploader("Upload PDFs for context", accept_multiple_files=True, type="pdf")
    retriever = None
    
    if uploaded_files:
        retriever = vector_manager.create_embeddings(uploaded_files)
        st.success("PDF processed and indexed successfully!")
    
    # Chat interface
    st.subheader("💬 Chat with AI")
    user_input = st.text_input("Ask a question:")
    
    if user_input:
        # Retrieve chat memory context
        context = memory.load_memory_variables({})["chat_history"]
        print("History ----------------- ", context)
        print("---------------------------------")
        response = chatbot.get_response(user_input, context, retriever)
        
        st.text_area("AI Response:", response, height=150)
        
        # Save conversation
        chat_history[user_id].append({"user": user_input, "response": response})
        save_chat_history(chat_history)
        
        # Add to memory
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response)
    
    # Option to download chat history
    st.download_button("📥 Download Chat History", data=json.dumps(chat_history[user_id], indent=4), file_name=f"chat_history_{user_id}.json", mime="application/json")

else:
    st.warning("Please enter your name and phone number to continue.")

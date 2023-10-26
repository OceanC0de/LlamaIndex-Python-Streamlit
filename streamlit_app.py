import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

# Update page title and icon
st.set_page_config(page_title="Chat with the Baden Restaurant Guide, powered by LlamaIndex", page_icon="🍽️", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
# Update the title of the page
st.title("Chat with the Baden Restaurant Guide, powered by LlamaIndex 💬🍽️")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about restaurants in Baden!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    # Update text and input directory path
    with st.spinner(text="Loading and indexing the Baden Restaurant data – hang tight!"):
        reader = SimpleDirectoryReader(input_dir="./data/kb/baden_restaurants", recursive=True)
        docs = reader.load_data()
        # Update the system prompt to match the new domain
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(
                model="gpt-3.5-turbo", 
                temperature=0.5, 
                system_prompt="You are an expert on the restaurants in Baden, Switzerland. Your job is to provide information based on the restaurant guide database. Keep your answers factual and based on the provided data – do not hallucinate features."
            )
        )
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If the last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history
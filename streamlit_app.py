import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

# Update page title and icon
st.set_page_config(page_title="Chat with the Baden Restaurant Guide, powered by LlamaIndex", page_icon="🍽️", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = ""
# Update the title of the page
st.title("Chat with the Power Tower Chef 💬🍽️")


if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

if not st.session_state.chat_started:
    distance = st.slider(
        'Select maximum distance (km):',
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )
    rating = st.slider(
        'Select minimum rating:',
        min_value=3.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )

    def start_chat():
        st.session_state.distance = distance
        st.session_state.rating = rating
        st.session_state.chat_started = True
    
    st.button('Start Chat', on_click=start_chat)

else:
    # Display the selected slider values using st.metric
    col1, col2 = st.columns(2)
    col1.metric(label="Max. Distance to Power Tower(km)", value=st.session_state.distance)
    col2.metric(label="Min. Star Rating", value=st.session_state.rating)
    # Initialize st.session_state.messages if not present
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Based on your preferences above, ask me what restaurants would make a good fit today"}
        ]

    @st.cache_resource(show_spinner=False)
    def load_data():
        # Use the stored values from session state here
        distance = st.session_state.distance
        rating = st.session_state.rating
        with st.spinner(text="Loading and indexing the Baden Restaurant data – hang tight!"):
            reader = SimpleDirectoryReader(input_dir="./data/kb/baden_restaurants", recursive=True)
            docs = reader.load_data()
            service_context = ServiceContext.from_defaults(
                llm=OpenAI(
                    model="gpt-4", 
                    temperature=0.5, 
                    system_prompt=(
                        f"You are an expert on the restaurants in Baden, Switzerland. Your job is to provide "
                        f"information based on the restaurant database. "
                        f"based on their criteria. The restaurants have a different distance to the Power Tower, the user wants max {distance} km), "
                        f"and rating (at least {rating}). Pick the restaurant that is below the max distance, has at least the minimum rating "
                        f"Always pick one restaurant and say that it is below the max {distance} and has at least {rating} or more. Also, provide as much info about the restaurant as possible. "
                    )
                )
            )
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            return index
            
    index = load_data()
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    if prompt := st.chat_input("Your question"):  
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:  
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)

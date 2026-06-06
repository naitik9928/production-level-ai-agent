import streamlit as st
import uuid 


def generate_thread():
    return str (uuid.uuid4())












chat_input=st.chat_input("type here....")

st.sidebar.title("Recent order conversation")

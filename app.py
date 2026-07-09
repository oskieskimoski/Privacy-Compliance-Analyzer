import os

import streamlit as st
from src.pipeline.privacy_compliance_rag import PrivacyComplianceRAG

st.set_page_config(page_title="Privacy Chat", layout="centered")
st.title("GDPR AI Assistant")
st.write("Ask me anything about GDPR compliance!")

@st.cache_resource
def load_rag():
    return PrivacyComplianceRAG()

rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your GDPR compliance assistant. How can I help you today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_query := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.spinner("Analyzing GDPR constraints..."):
        bot_response, _ = rag.ask(user_query)

    with st.chat_message("assistant"):
        st.write(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
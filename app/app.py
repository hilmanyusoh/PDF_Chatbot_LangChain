import streamlit as st
import sys
import os
from qa_system import HadithQA 

# โหลด QA system
@st.cache_resource
def load_qa():
    return HadithQA()

qa = load_qa()

# สร้าง UI
st.title("📖 Hadith Q&A Chatbot")
st.write("ถามคำถามเกี่ยวกับ Hadith แล้วรับคำตอบจากฐานข้อมูล")

# รับคำถามจากผู้ใช้
user_question = st.text_input("❓ คำถามของคุณ")

if user_question:
    with st.spinner("🧠 กำลังค้นหาคำตอบ..."):
        answer = qa.ask_question(user_question)
    st.success("💬 คำตอบ:")
    st.write(answer)

import streamlit as st
from simple_rag import index_pdf_in_memory, answer_question_in_memory

st.set_page_config(page_title="Gemini PDF RAG (No Docker)", page_icon="📚")

st.title("Gemini PDF RAG (In-Memory, No Docker)")

st.sidebar.header("Upload PDF")
pdf_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

if pdf_file is not None:
    pdf_path = f"uploaded_{pdf_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())

    if st.sidebar.button("Index PDF"):
        with st.spinner("Indexing PDF..."):
            n_chunks = index_pdf_in_memory(pdf_path, source_name=pdf_file.name)
        st.sidebar.success(f"Indexed {n_chunks} chunks from {pdf_file.name}")

st.header("Ask a question about your PDFs")

question = st.text_input("Your question")
if st.button("Get answer") and question.strip():
    with st.spinner("Thinking..."):
        answer, sources = answer_question_in_memory(question)
    st.subheader("Answer")
    st.write(answer)
    if sources:
        st.subheader("Sources")
        for s in sources:
            st.write(f"- {s}")

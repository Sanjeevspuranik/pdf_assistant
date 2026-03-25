import streamlit as st
import os
from pdf_extractor import PDFExtractor
from rag_pipeline import RAGPipeline
from graph import PDFGraph
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="PDF AI Agent", layout="wide")
st.title("📄 PDF Assistant")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
    st.session_state.pil_images = []
    st.session_state.messages = []

with st.sidebar:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Process PDF"):
        with st.status("Analyzing PDF Content...", expanded=True) as status:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            extractor = PDFExtractor("temp.pdf")
            pipeline = RAGPipeline(extractor)

            st.write("Extracting Text, Images, and Tables...")
            db, images = pipeline.ingest()

            st.session_state.vector_db = db
            st.session_state.pil_images = images
            status.update(label="RAG Ready!", state="complete")

if st.session_state.vector_db:
    bot = PDFGraph(st.session_state.vector_db, st.session_state.pil_images)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            inputs = {"messages": [HumanMessage(content=prompt)]}
            config = {"configurable": {"thread_id": "1"}}

            # Run Graph
            final_state = bot.app.invoke(inputs)
            response = final_state["messages"][-1].content
            st.markdown(response)

            # Show relevant visuals
            if final_state["relevant_images"]:
                with st.expander("Related Images & Figures"):
                    cols = st.columns(
                        min(len(final_state["relevant_images"]), 3))
                    for idx, img_obj in enumerate(final_state["relevant_images"][:3]):
                        cols[idx % 3].image(
                            img_obj['image'], caption=f"Page {img_obj['page']}")

            st.session_state.messages.append(
                {"role": "assistant", "content": response})
else:
    st.info("Please upload a PDF to begin.")

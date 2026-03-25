# 📄 Multimodal Agentic PDF RAG Assistant

A high-performance, modular RAG (Retrieval-Augmented Generation) pipeline designed to transform static PDFs into interactive, visually-aware AI agents. This system goes beyond text by extracting tables and images, captioning visuals using **BLIP**, and using **LangGraph** to orchestrate an intelligent chatbot interface.

## 🚀 Key Features

* **Multithreaded Extraction:** Fast, parallel processing of text, tables (via `pdfplumber`), and images (via `PyMuPDF`).
* **Visual Intelligence:** Uses the **BLIP (Bootstrapping Language-Image Pre-training)** model to generate searchable text captions for PDF images.
* **Agentic Orchestration:** Built with **LangGraph** to manage state, retrieve context, and provide a unified chat experience.
* **Logic-Based Visual Retrieval:** Automatically maps retrieved text chunks to their original images by page number, providing visual context alongside AI answers.
* **Modular Architecture:** Clean separation between extraction, pipelining, graph logic, and the UI.
* **Local-Ready:** Optimized to run on local hardware (like RTX 3050/32GB RAM) using HuggingFace embeddings.

## 🏗️ Project Structure

The project is organized into four core modules:

1.  **`pdf_extractor.py`**: Handles multithreaded parsing of PDF assets (Images, Tables, Text).
2.  **`rag_pipeline.py`**: Manages BLIP image captioning, text chunking, and ChromaDB vector storage.
3.  **`graph.py`**: Defines the LangGraph state machine and retrieval/generation nodes.
4.  **`app.py`**: A polished Streamlit interface for uploading files and chatting with the agent.

## 🛠️ Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Sanjeevspuranik/pdf_assistant.git
   cd pdf_assistant
   ```

2. **Install Dependencies:**
   ```bash
   pip install streamlit langchain-openai langchain-huggingface langgraph \
               pymupdf pdfplumber torch transformers pillow chromadb
   ```

3. **Set Up Environment Variables:**
   Create a `.env` file or export your API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## 💻 Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

1.  **Upload** a PDF (Technical manual, research paper, or study guide).
2.  **Wait** for the system to extract assets and build the local VectorDB.
3.  **Chat** with your document. When the AI explains a concept found near a diagram, the diagram will automatically appear in the "Related Images" expander.

## 🤖 Technology Stack

* **Orchestration:** [LangGraph](https://github.com/langchain-ai/langgraph)
* **Framework:** [LangChain](https://github.com/langchain-ai/langchain)
* **Vector Store:** [ChromaDB](https://www.trychroma.com/)
* **Vision Model:** [Salesforce BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)
* **Embeddings:** [HuggingFace (all-MiniLM-L6-v2)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
* **UI:** [Streamlit](https://streamlit.io/)

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

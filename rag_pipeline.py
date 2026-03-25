import fitz  # PyMuPDF
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class RAGPipeline:
    def __init__(self, extractor, db_path="./chroma_db"):
        self.extractor = extractor
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _generate_captions(self, image_list):
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base").to(self.device)
        captioned_docs = []
        for item in image_list:
            inputs = processor(
                images=item['image'], return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=40)
                caption = processor.decode(out[0], skip_special_tokens=True)
            captioned_docs.append(Document(
                page_content=f"Visual: {caption}",
                metadata={"page": item['page'], "type": "image"}
            ))
        return captioned_docs

    def ingest(self):
        images, tables = self.extractor.extract_all()
        text_docs = []
        with fitz.open(self.extractor.pdf_path) as doc:
            for i, page in enumerate(doc):
                text_docs.append(Document(page_content=page.get_text(), metadata={
                                 "page": i+1, "type": "text"}))

        table_docs = [Document(page_content=f"Table: {str(t)}", metadata={"page": e['page'], "type": "table"})
                      for e in tables for t in e['tables'] if t]

        image_docs = self._generate_captions(images)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        final_chunks = splitter.split_documents(
            text_docs) + image_docs + table_docs

        vector_db = Chroma.from_documents(
            documents=final_chunks, embedding=self.embeddings, persist_directory=self.db_path)
        return vector_db, images

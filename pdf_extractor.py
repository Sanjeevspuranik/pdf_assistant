import fitz  # PyMuPDF
import pdfplumber
import io
from concurrent.futures import ThreadPoolExecutor
from PIL import Image


class PDFExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        with fitz.open(self.pdf_path) as doc:
            self.pages_count = len(doc)

    def _process_page_images(self, page_index):
        extracted_images = []
        with fitz.open(self.pdf_path) as doc:
            page = doc[page_index]
            for img_idx, img in enumerate(page.get_images(full=True)):
                try:
                    pix = fitz.Pixmap(doc, img[0])
                    if pix.colorspace.n > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
                    if pil_img.size[0] > 100 and pil_img.size[1] > 100:
                        extracted_images.append({
                            "page": page_index + 1,
                            "index": img_idx,
                            "image": pil_img
                        })
                    pix = None
                except Exception:
                    continue
        return extracted_images

    def _process_page_tables(self, page_index):
        with pdfplumber.open(self.pdf_path) as pdf:
            page = pdf.pages[page_index]
            return {"page": page_index + 1, "tables": page.extract_tables()}

    def extract_all(self):
        with ThreadPoolExecutor() as executor:
            images = []
            img_results = executor.map(
                self._process_page_images, range(self.pages_count))
            for res in img_results:
                images.extend(res)
            table_results = list(executor.map(
                self._process_page_tables, range(self.pages_count)))
        return images, table_results

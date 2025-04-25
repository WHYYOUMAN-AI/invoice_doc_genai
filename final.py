import os
import fitz
import streamlit as st
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import torch
import json
from transformers import (
    LayoutLMv3ImageProcessor,
    AutoTokenizer,
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
)
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. Set OpenAI API key
# Replace 'YOUR_OPENAI_API_KEY_HERE' or export OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = "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"

# 2. Initialize LayoutLMv3 processor without internal OCR
timeout = None
image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
tokenizer      = AutoTokenizer.from_pretrained('microsoft/layoutlmv3-base')
processor      = LayoutLMv3Processor(image_processor=image_processor, tokenizer=tokenizer)
# Load model
model = LayoutLMv3ForTokenClassification.from_pretrained('microsoft/layoutlmv3-base')
model.eval()

# 3. PDF → Image conversion for OCR
def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    return convert_from_path(pdf_path, dpi=dpi, fmt='jpeg', thread_count=4, grayscale=True)

# 4. Preprocess image (grayscale, contrast, denoise)
def preprocess_image(img: Image.Image) -> Image.Image:
    gray = img.convert('L')
    enh  = ImageEnhance.Contrast(gray).enhance(2.0)
    return enh.filter(ImageFilter.MedianFilter(size=3))

# 5. Tesseract OCR (word‑level with boxes)
def ocr_image(img: Image.Image, lang: str = 'eng') -> Dict:
    config = r'--oem 1 --psm 3'
    return pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DICT)

# 6. LayoutLMv3 inference on single page
def run_layoutlmv3(page_img: Image.Image, words: List[str], boxes: List[List[int]]) -> List[str]:
    encoding = processor(images=page_img, text=words, boxes=boxes,
                         return_tensors='pt', padding='max_length', truncation=True)
    with torch.no_grad(): outputs = model(**encoding)
    logits = outputs.logits.squeeze(0)
    preds  = logits.argmax(-1).tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(encoding.input_ids.squeeze(0).tolist())
    # aggregate labels per word
    agg = {}
    for label_id, word_idx in zip(preds, encoding.word_ids(batch_index=0)):
        if word_idx is not None:
            agg.setdefault(word_idx, []).append(model.config.id2label[label_id])
    return [max(labels, key=labels.count) for labels in agg.values()]

# 7. Full extract pipeline → words, labels, combined text
def extract_and_label(pdf_path: str) -> (List[Dict], str):
    pages    = pdf_to_images(pdf_path)
    results  = []
    
    full_txt = []
    for i, page in enumerate(pages, start=1):
        img      = preprocess_image(page)
        ocr_data = ocr_image(img)
        words, boxes, page_words = [], [], []
        for text, l, t, w, h in zip(ocr_data['text'], ocr_data['left'], ocr_data['top'],
                                    ocr_data['width'], ocr_data['height']):
            if not text.strip(): continue
            words.append(text); page_words.append(text)
            boxes.append([int(1000*l/page.width), int(1000*t/page.height),
                          int(1000*(l+w)/page.width), int(1000*(t+h)/page.height)])
        labels = run_layoutlmv3(page, words, boxes)
        results.append({'page': i, 'words': words, 'labels': labels})
        full_txt.append(' '.join(page_words))
    return results, '\n'.join(full_txt)

# 8. Streamlit UI and LangChain zero‑shot prompting
st.set_page_config(page_title='📄 Smart PDF Extractor', layout='centered')
st.title('📊 Financial Document Extractor')

uploaded_file = st.file_uploader('Upload PDF (invoice, receipt, etc.):', type='pdf')
if uploaded_file:
    st.success('File uploaded.')
    if st.button('🔍 Analyze Document'):
        with st.spinner('Processing...'):
            # save temp PDF
            tmp = '/tmp/uploaded.pdf'
            with open(tmp, 'wb') as f: f.write(uploaded_file.read())
            # run layout + OCR
            pages, doc_text = extract_and_label(tmp)
            # display LayoutLMv3 labels
            st.subheader('📑 Token Labels')
            for pg in pages:
                st.write(f"Page {pg['page']}")
                st.table({w: lbl for w, lbl in zip(pg['words'], pg['labels'])})
            # prepare and run ChatPromptTemplate + ChatOpenAI
            prompt = ChatPromptTemplate.from_template("""
You are an expert AI trained to extract meaningful information from various types of business and financial documents. These include:

- Invoices
- Receipts
- Bills
- Proforma Invoices
- Credit Notes
- Quotes
- Delivery Notes
- Purchase Orders
- Statements of Account
- Purchase Agreement

YOUR TASK:
1. **Document Classification:**  
   - First, determine the document type (e.g., Invoice, Receipt, etc.).  
   - If the document does not match any known category, label it as “UNKNOWN_TYPE” and proceed using your generic extraction strategy.

2. **Dynamic Field Extraction:**  
   - For **known** types, apply the corresponding schema (e.g., Invoice → invoice_number, date, vendor, line_items, tax, total).  
   - For **unknown** types, automatically identify key–value pairs, tables, and free‑form text blocks by detecting headers, labels, and layout cues.  
   - Normalize field names into a consistent JSON schema (e.g., `field_name`, `value`, `confidence_score`).

3. **Structured JSON Output:**  
   - Always output a single JSON object with:  
     - `document_type`: string  
     - `tables`: array of tables, each as an array of rows with header mapping  
     - `notes`: any warnings or clarifications (e.g., “Merged cells detected,” “Tax calculation mismatched subtotal”)

4. **Validation & Error Handling:**  
   - Perform logical checks (e.g., sum of line_items quantities × unit_price equals subtotal; tax = rate × taxable_amount).  
   - If validation fails or confidence < 0.7, flag the field with `"confidence": 0.xx` and include a note about the uncertainty.

5. **Future‑Ready Behavior:**  
   - If completely unfamiliar structures are encountered (e.g., new document type), infer the most probable extraction schema based on visual cues (tables vs. free text) and common business document patterns.  
   - Pose a clarifying question if essential information is missing or ambiguous (e.g., “I detected an unknown header ‘Shipment Ref.’ — could you specify its meaning?”).

Your task is to analyze the given document and extract relevant structured data.
Return strictly in JSON format.

DOCUMENT:
{text}
""")
            llm   = ChatOpenAI(model='gpt-4', temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])
            chain = prompt | llm
            result = chain.invoke({'text': doc_text})
            st.subheader('✅ Extracted Structured Data')
            st.code(result.content, language='json')

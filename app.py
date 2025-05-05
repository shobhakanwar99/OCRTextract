import streamlit as st
import tempfile
import requests
import os
import re
from pdf_pipeline import (
    pdf_to_images,
    remove_horizontal_lines,
    extract_text_from_images,
    extract_data
)
# Optional: clean up local images after run
def cleanup_dir(path):
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))

def download_pdf_from_drive(drive_link):
    file_id = None
    if "drive.google.com" in drive_link:
        match = re.search(r'/d/([a-zA-Z0-9_-]+)', drive_link)
        if match:
            file_id = match.group(1)
        else:
            match = re.search(r'id=([a-zA-Z0-9_-]+)', drive_link)
            if match:
                file_id = match.group(1)

    if not file_id:
        st.error("Invalid Drive link.")
        return None

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    
    if response.status_code != 200:
        st.error("Failed to download file.")
        return None

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(response.content)
    temp_pdf.close()
    return temp_pdf.name

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🧾 PDF Invoice Data Extractor")

drive_link = st.text_input("🔗 Enter Google Drive PDF Link")
if st.button("Extract Info") and drive_link:
    with st.spinner("Processing..."):
        pdf_path = download_pdf_from_drive(drive_link)
        if pdf_path:
            try:
                image_paths = pdf_to_images(pdf_path, image_dir="temp_images_folder")
                cleaned_images = remove_horizontal_lines(image_paths)
                texts = extract_text_from_images(cleaned_images)

                full_text = '\n\n--- PAGE BREAK ---\n\n'.join(texts)
                total_value, invoice_date, account_name, company_name = extract_data(texts)
                if company_name:
                    company_name = company_name.split(':')[-1].strip()
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("📜 Raw OCR Text")
                    st.text_area("Extracted Text", value=full_text, height=600)

                with col2:
                    st.subheader("🔍 Extracted Information")
                    st.write(f"**Total Value:** {total_value or '❌ Not found'}")
                    st.write(f"**Invoice Date:** {invoice_date or '❌ Not found'}")
                    st.write(f"**Account Name:** {account_name or '❌ Not found'}")
                    st.write(f"**Company Name:** {company_name or '❌ Not found'}")

            finally:
                cleanup_dir("temp_images_folder")
                os.remove(pdf_path)

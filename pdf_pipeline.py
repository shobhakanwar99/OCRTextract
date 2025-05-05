import fitz
import cv2
import os
import re
import pytesseract
import pandas as pd
from pytesseract import Output
from dateutil.parser import parse

# ------------ 1. Convert PDF pages to images -------------
def pdf_to_images(pdf_path, image_dir='images', zoom=6):
    os.makedirs(image_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(zoom, zoom)
    image_paths = []

    for page_no in range(len(doc)):
        page = doc.load_page(page_no)
        pix = page.get_pixmap(matrix=mat)
        image_path = os.path.join(image_dir, f"{page_no}.jpg")
        pix.save(image_path)
        image_paths.append(image_path)

    return image_paths


# ------------ 2. Remove horizontal lines -----------------
def remove_horizontal_lines(image_paths):
    cleaned_paths = []

    for img_path in image_paths:
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for c in cnts:
            cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 6))
        repaired = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

        cv2.imwrite(img_path, repaired)
        cleaned_paths.append(img_path)

    return cleaned_paths


# ------------ 3. Extract text from cleaned images --------
def extract_text_from_images(image_paths):
    extracted_texts = []

    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gauss = cv2.GaussianBlur(img, (3, 3), 0)

        custom_config = r'-l eng --oem 1 --psm 4 -c preserve_interword_spaces=1'
        #r'-l eng --oem 1 --psm 6 -c preserve_interword_spaces=1 -c tessedit_char_whitelist="0123456789- "'
        #custom_config = r'-l eng --oem 1 --psm 6'11
        d = pytesseract.image_to_data(gauss, config=custom_config, output_type=Output.DICT)
        df = pd.DataFrame(d)

        df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]
        sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()

        text = ''
        for block in sorted_blocks:
            curr = df1[df1['block_num'] == block]
            sel = curr[curr.text.str.len() > 3]
            char_w = (sel.width / sel.text.str.len()).mean() if not sel.empty else 10
            prev_par, prev_line, prev_left = 0, 0, 0

            for ix, ln in curr.iterrows():
                if prev_par != ln['par_num']:
                    text += '\n'
                    prev_par = ln['par_num']
                    prev_line = ln['line_num']
                    prev_left = 0
                elif prev_line != ln['line_num']:
                    text += '\n'
                    prev_line = ln['line_num']
                    prev_left = 0

                added = 0
                if ln['left'] / char_w > prev_left + 1:
                    added = int((ln['left']) / char_w) - prev_left
                    text += ' ' * added

                text += ln['text'] + ' '
                prev_left += len(ln['text']) + added + 1
            text += '\n'

        cleaned_text = re.sub(" +", " ", text)
        extracted_texts.append(cleaned_text.strip())

    return extracted_texts

# ------------ 4. Extract amount data -------------------

def extract_final_amount(full_text):
    sub_total_value = None
    total_value = None
    sub_total_match = re.search(r'sub\s*total[^\d\n\r]*([-+]?\d[\d,]*\.?\d*)', full_text, re.IGNORECASE)
    if sub_total_match:
        sub_total_value = sub_total_match.group(1).replace(",", "")
        return sub_total_value 
    total_matches = re.findall(r'\btotal[^\d\n\r]*([-+]?\d[\d,]*\.?\d*)', full_text, re.IGNORECASE)
    if total_matches:
        total_value = total_matches[-1].replace(",", "")
        print(f"Fallback: Found Total → {total_value}")
        return total_value
    if not sub_total_value and not total_value:
        match = re.search(r'\bMRP[:\s]*([\d,]+)', full_text, re.IGNORECASE)
        if not match:
            match = re.search(r'\bMRP[^\d]*([\d,]+)', full_text, re.IGNORECASE)
        return match.group(1).replace(',', '') if match else None

    print("No Sub Total or Total found.")
    return None


def extract_invoice_date_multiline(text):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if 'date' in line.lower():
            # Check current and next line for date
            candidate_lines = [line]
            if i + 1 < len(lines):
                candidate_lines.append(lines[i + 1])

            for cl in candidate_lines:
                match = re.search(
                    r'('
                    r'\d{2}[/-]\d{2}[/-]\d{2,4}|'                      # 12/05/2023 or 12-05-23
                    r'\d{4}[/-]\d{2}[/-]\d{2}|'                        # 2023-05-12
                    r'\d{2}\.\d{2}\.\d{4}|'                            # 12.05.2023
                    r'\d{1,2}[/-][A-Za-z]{3}[/-]?\d{2,4}|'             # 12-May-2023
                    r'\d{1,2}\s+[A-Za-z]{3,9}[,]?\s+\d{2,4}|'          # 12 May 2023
                    r'[A-Za-z]{3,9}\s+\d{1,2}[,]?\s+\d{2,4}'           # May 12, 2023
                    r')',
                    cl, re.IGNORECASE
                )
                if match:
                    date_str = match.group(1).strip()
                    try:
                        date_obj = parse(date_str, dayfirst=True).date()
                        return str(date_obj)
                    except Exception:
                        continue
    return None



# ------------ 5. Extract data from text -----------------

def extract_data(text):
    # If input is a list of lines, join into a single string for searching
    if isinstance(text, list):
        full_text = '\n'.join(text)
    else:
        full_text = text
    full_text = full_text.replace('involce', 'invoice')
    lines = [line.strip() for line in full_text.split('\n') if line.strip() and not all(c in ',_' for c in line.strip())]

    full_text = full_text.replace('Involce', 'invoice')

    account_match = re.search(r'(?:a/c\s*name|acct\s*name|account\s*name|trade\s*name)\s*[:\-]\s*(.+)', full_text, re.IGNORECASE)
    account_name = account_match.group(1).strip() if account_match else None

    total_value = extract_final_amount(full_text)
    invoice_date = extract_invoice_date_multiline(full_text)

    cleaned_text = re.sub(" +", " ", full_text)
    lines = cleaned_text.split('\n')
    company_name = None
    for line in lines:
        if re.search(r"(Pvt\.\s*Ltd\.|Private\s*Limited)", line, re.IGNORECASE):
            if 'ashoka builders' not in line.lower():
                company_name = line.strip()
                name_ = re.sub(r'[^a-zA-Z0-9]', '', company_name).lower()
                if name_.endswith('pvtltd') or name_.endswith('privatelimited'):
                    break
    if not account_name:
        just_name = re.search(r'\bname\b\s*[:\-]\s*(.+)', full_text, re.IGNORECASE)
        account_name = just_name.group(1).strip() if just_name else None

    return total_value, invoice_date, account_name, company_name


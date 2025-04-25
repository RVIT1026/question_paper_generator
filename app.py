import streamlit as st
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
from PIL import Image
import re
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
import mysql.connector
import os
import secrets
import string
from reportlab.lib.pdfencrypt import StandardEncryption
import random
import zipfile
import base64
import hashlib
import tempfile
import shutil

# UI Configuration
st.set_page_config(
    page_title="RVIT - Question Paper Generator",
    page_icon="img/rvitlogo_f.jpg",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
        }
        .stSelectbox {
            border-radius: 5px;
        }
        .main-header {
            text-align: center;
            padding: 1rem;
            background-color: #f0f2f6;
            border-radius: 5px;
            margin-bottom: 1rem;
            background: black;
        }
        .section-header {
            background-color: #e7f3fe;
            padding: 0.5rem;
            border-radius: 5px;
            margin: 1rem 0;
            background: black;
        }
        .info-box {
            background-color: #e7f3fe;
            border-left: 6px solid #2196F3;
            padding: 1rem;
            margin: 1rem 0;
        }
        .copy-button {
            padding: 5px 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        .copy-button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Constants
ACADEMIC_LEVELS = [
    "I B.Tech", "II B.Tech", "III B.Tech", "IV B.Tech",
    "I M.Tech", "II M.Tech", "I MCA", "II MCA", "I BCA", "II BCA", "III BCA"
]

BRANCHES = [
    "CSE", "CSE(AIML)", "CSE(DS)", "ECE", "EVT"
]

MONTHS = [
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"
]

class ExamType(Enum):
    MID1_2 = "mid1 (Unit- 1, 2)"
    MID1_2_5 = "mid1 (Unit- 1, 2 & 3.1)"
    MID2_3_4_5 = "mid2 (Unit- 3, 4 & 5)"
    MID2_5_5 = "mid2 (Unit- 3.2, 4 & 5)"
    REGULAR = "regular"
    SUPPLY = "supply"

SEMESTER_SUBJECTS = {
    ("I B.Tech", "I"): [
        "Linear Algebra & Calculus (RV23991T03)",
        "Engineering Chemistry (RV23991T02)",
        "Basic Civil & Mechanical Engineering (RV23991T04)",
        "Introduction to Programming (RV23991T05)",
        "Communicative English (RV23991T01)",
        "Engineering Physics (RV23991T06)",
        "Basic Electrical & Electronics Engineering (RV23991T07)",
        "Engineering Graphics (RV23991T08)"
    ],
    ("I B.Tech", "II"): [
        "Linear Algebra & Calculus (RV23991T03)",
        "Engineering Chemistry (RV23991T02)",
        "Basic Civil & Mechanical Engineering (RV23991T04)",
        "Introduction to Programming (RV23991T05)",
        "Communicative English (RV23991T01)",
        "Engineering Physics (RV23991T06)",
        "Basic Electrical & Electronics Engineering (RV23991T07)",
        "Engineering Graphics (RV23991T08)"
    ]
}

@dataclass
class QuestionPattern:
    part_a: Dict[float, Dict[str, int]]
    part_b: Dict[float, Dict[str, int]]
    marks_a: Tuple[int, int]
    marks_b: Tuple[int, int]

PATTERNS = {}

def add_default_pattern(subject: str, exam: ExamType):
    if subject in ["Basic Civil & Mechanical Engineering (RV23991T04)", 
                   "Basic Electrical & Electronics Engineering (RV23991T07)"]:
        units_mid1_2 = {1: {'short': 3, 'long': 3}, 2: {'short': 2, 'long': 3}}
        units_mid1_2_5 = {1: {'short': 2, 'long': 2}, 2: {'short': 2, 'long': 2}, 3: {'short': 1, 'long': 2}}
        units_mid2_3_4_5 = {4: {'short': 2, 'long': 2}, 5: {'short': 2, 'long': 2}, 6: {'short': 1, 'long': 2}}
        units_regular = {1: {'short': 2, 'long': 2}, 2: {'short': 2, 'long': 2}, 3: {'short': 1, 'long': 2},
                         4: {'short': 1, 'long': 2}, 5: {'short': 2, 'long': 2}, 6: {'short': 2, 'long': 2}}
    elif subject == "Engineering Graphics (RV23991T08)":
        PATTERNS[(subject, ExamType.MID1_2)] = QuestionPattern(part_a={}, part_b={1: {'long': 3}, 2: {'long': 3}}, marks_a=(0, 0), marks_b=(3, 10))
        PATTERNS[(subject, ExamType.MID1_2_5)] = QuestionPattern(part_a={}, part_b={1: {'long': 2}, 2: {'long': 2}, 3: {'long': 2}}, marks_a=(0, 0), marks_b=(3, 10))
        PATTERNS[(subject, ExamType.MID2_3_4_5)] = QuestionPattern(part_a={}, part_b={4: {'long': 2}, 5: {'long': 2}, 6: {'long': 2}}, marks_a=(0, 0), marks_b=(3, 10))
        PATTERNS[(subject, ExamType.MID2_5_5)] = QuestionPattern(part_a={}, part_b={4: {'long': 2}, 5: {'long': 2}, 6: {'long': 2}}, marks_a=(0, 0), marks_b=(3, 10))
        PATTERNS[(subject, ExamType.REGULAR)] = QuestionPattern(part_a={}, part_b={1: {'long': 2}, 2: {'long': 2}, 3: {'long': 2}, 4: {'long': 2}, 5: {'long': 2}}, marks_a=(0, 0), marks_b=(5, 14))
        PATTERNS[(subject, ExamType.SUPPLY)] = PATTERNS[(subject, ExamType.REGULAR)]
        return
    else:
        units_mid1_2 = {1: {'short': 3, 'long': 3}, 2: {'short': 2, 'long': 3}}
        units_mid1_2_5 = {1: {'short': 2, 'long': 2}, 2: {'short': 2, 'long': 2}, 3.1: {'short': 1, 'long': 2}}
        units_mid2_3_4_5 = {3: {'short': 2, 'long': 2}, 4: {'short': 2, 'long': 2}, 5: {'short': 1, 'long': 2}}
        units_regular = {1: {'short': 2, 'long': 2}, 2: {'short': 2, 'long': 2}, 3.1: {'short': 1, 'long': 1},
                         3.2: {'short': 1, 'long': 1}, 4: {'short': 2, 'long': 2}, 5: {'short': 2, 'long': 2}}

    if exam == ExamType.MID1_2:
        PATTERNS[(subject, exam)] = QuestionPattern(
            part_a={unit: {'short': count['short']} for unit, count in units_mid1_2.items()},
            part_b={unit: {'long': count['long']} for unit, count in units_mid1_2.items()},
            marks_a=(5, 2),
            marks_b=(3, 5)
        )
    elif exam == ExamType.MID1_2_5:
        PATTERNS[(subject, exam)] = QuestionPattern(
            part_a={unit: {'short': count['short']} for unit, count in units_mid1_2_5.items()},
            part_b={unit: {'long': count['long']} for unit, count in units_mid1_2_5.items()},
            marks_a=(5, 2),
            marks_b=(3, 5)
        )
    elif exam == ExamType.MID2_3_4_5:
        PATTERNS[(subject, exam)] = QuestionPattern(
            part_a={unit: {'short': count['short']} for unit, count in units_mid2_3_4_5.items()},
            part_b={unit: {'long': count['long']} for unit, count in units_mid2_3_4_5.items()},
            marks_a=(5, 2),
            marks_b=(3, 5)
        )
    elif exam == ExamType.MID2_5_5:
        PATTERNS[(subject, exam)] = QuestionPattern(
            part_a={unit: {'short': count['short']} for unit, count in units_mid2_3_4_5.items()},
            part_b={unit: {'long': count['long']} for unit, count in units_mid2_3_4_5.items()},
            marks_a=(5, 2),
            marks_b=(3, 5)
        )
    elif exam == ExamType.REGULAR:
        PATTERNS[(subject, exam)] = QuestionPattern(
            part_a={unit: {'short': count['short']} for unit, count in units_regular.items()},
            part_b={unit: {'long': count['long']} for unit, count in units_regular.items()},
            marks_a=(10, 2),
            marks_b=(5, 10)
        )
    elif exam == ExamType.SUPPLY:
        PATTERNS[(subject, exam)] = PATTERNS.get((subject, ExamType.REGULAR), 
            QuestionPattern(
                part_a={unit: {'short': count['short']} for unit, count in units_regular.items()},
                part_b={unit: {'long': count['long']} for unit, count in units_regular.items()},
                marks_a=(10, 2),
                marks_b=(5, 10)
            ))

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="qb_db"
        )
        return connection
    except mysql.connector.Error as e:
        st.error(f"Database connection error: {str(e)}")
        return None

def sanitize_table_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '', name.replace(" ", "_").replace("(", "").replace(")", "").replace("&", "and"))

def configure_question_image(img: Image, max_width: int = 250, max_height: int = 400) -> tuple:
    width, height = img.size
    aspect_ratio = width / height
    if width > max_width or height > max_height:
        if width / max_width > height / max_height:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
    else:
        new_width, new_height = width, height
    return new_width, new_height

def get_image_paths(subject: str) -> Dict[int, str]:
    subject_folder = subject.split('(')[0].strip()
    img_folder = os.path.join("Img", subject_folder)
    image_paths = {}
    if os.path.exists(img_folder):
        for file in os.listdir(img_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    sno = int(file.split('.')[0])
                    image_paths[sno] = os.path.join(img_folder, file)
                except ValueError:
                    continue
    return image_paths

class CustomDocTemplate(SimpleDocTemplate):
    def __init__(self, *args, **kwargs):
        self.top_margin_first = kwargs.pop('top_margin_first', 0)
        self.top_margin_rest = kwargs.pop('top_margin_rest', 30)
        self.encryption = kwargs.pop('encryption', None)
        self.timestamp = kwargs.pop('timestamp', None)
        kwargs['topMargin'] = self.top_margin_first
        if self.encryption:
            kwargs['encrypt'] = self.encryption
        super().__init__(*args, **kwargs)
        
    def handle_pageBegin(self):
        self._calc = self.canv.getPageNumber() > 1
        self.topMargin = self.top_margin_rest if self._calc else self.top_margin_first
        if self.canv.getPageNumber() == 1 and self.timestamp:
            self.canv.saveState()
            self.canv.setFont('Helvetica-Bold', 8)
            timestamp_text = f"Generated on: {self.timestamp}"
            text_width = self.canv.stringWidth(timestamp_text, 'Helvetica-Bold', 8)
            self.canv.drawString(self.width + self.leftMargin - text_width, 20, timestamp_text)
            self.canv.line(self.width + self.leftMargin - text_width, 30, self.width + self.leftMargin, 30)
            self.canv.restoreState()
        return super().handle_pageBegin()

def generate_secure_password(length=10):
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

def create_header_config():
    st.markdown('<div class="section-header"><h2>📝 Paper Configuration</h2></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        academic_level = st.selectbox("Academic Level", ACADEMIC_LEVELS)
        semester = st.selectbox("Semester", ["I", "II"])
    with col2:
        exam_type = st.selectbox(
            "Exam Type",
            ["I MID", "II MID", "REGULAR", "SUPPLY"]
        )
        branches = st.multiselect(
            "Branch",
            BRANCHES,
            default=["CSE", "CSE(AIML)"]
        )
    with col3:
        month = st.selectbox("Month", MONTHS)
        year = st.selectbox("Year", range(2024, 2030))
    return {
        'academic_level': academic_level,
        'semester': semester,
        'exam_type': exam_type,
        'month_year': f"{month}-{year}",
        'branches': branches,
        'date': datetime.now().strftime("%d %B %Y"),
    }

def check_table_exists(table_name: str) -> bool:
    connection = get_db_connection()
    if not connection:
        return False
    try:
        cursor = connection.cursor()
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        result = cursor.fetchone()
        return result is not None
    except mysql.connector.Error as e:
        st.error(f"Error checking table existence: {str(e)}")
        return False
    finally:
        cursor.close()
        connection.close()

def create_table(subject: str):
    connection = get_db_connection()
    if not connection:
        return None
    table_name = sanitize_table_name(subject)
    try:
        cursor = connection.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                sno INT PRIMARY KEY,
                unit VARCHAR(10),
                question TEXT,
                type VARCHAR(10),
                CO VARCHAR(50),
                PO VARCHAR(50),
                BTL VARCHAR(50),
                counter INT DEFAULT 0
            )
        """)
        
        units = [
            (1, 12, "1"), (13, 24, "2"), (25, 30, "3.1"),
            (31, 36, "3.2"), (37, 48, "4"), (49, 60, "5")
        ]
        type_assignments = []
        for start, end, unit in units:
            short_count = 3 if unit in ["3.1", "3.2"] else 6
            for i in range(start, start + short_count):
                type_assignments.append("short")
            for i in range(start + short_count, end + 1):
                type_assignments.append("long")
        
        image_paths = get_image_paths(subject)
        
        for i in range(1, 61):
            unit = next((u for s, e, u in units if s <= i <= e), "")
            question = image_paths.get(i, "")
            q_type = type_assignments[i-1] if i-1 < len(type_assignments) else "long"
            cursor.execute(f"""
                INSERT INTO `{table_name}` (sno, unit, question, type, CO, PO, BTL, counter)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE question=VALUES(question)
            """, (i, unit, question, q_type, "", "", "", 0))
        
        connection.commit()
        st.success(f"Table '{table_name}' created or updated successfully!")
        df = pd.read_sql(f"SELECT * FROM `{table_name}`", connection)
        return df
    except mysql.connector.Error as e:
        st.error(f"Error creating table: {str(e)}")
        return None
    finally:
        cursor.close()
        connection.close()

def save_table_changes(table_name: str, edited_df: pd.DataFrame):
    connection = get_db_connection()
    if not connection:
        return
    try:
        cursor = connection.cursor()
        for index, row in edited_df.iterrows():
            cursor.execute(f"""
                INSERT INTO `{table_name}` (sno, unit, question, type, CO, PO, BTL, counter)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                unit=VALUES(unit),
                question=VALUES(question),
                type=VALUES(type),
                CO=VALUES(CO),
                PO=VALUES(PO),
                BTL=VALUES(BTL),
                counter=VALUES(counter)
            """, (
                int(row['sno']),
                str(row['unit']) if pd.notna(row['unit']) else "",
                str(row['question']) if pd.notna(row['question']) else "",
                str(row['type']) if pd.notna(row['type']) else "",
                str(row['CO']) if pd.notna(row['CO']) else "",
                str(row['PO']) if pd.notna(row['PO']) else "",
                str(row['BTL']) if pd.notna(row['BTL']) else "",
                int(row['counter']) if pd.notna(row['counter']) else 0
            ))
        connection.commit()
        st.success("Changes saved successfully!")
        if f"saved_{table_name}" not in st.session_state:
            st.session_state[f"saved_{table_name}"] = True
            st.rerun()
    except mysql.connector.Error as e:
        st.error(f"Error saving changes: {str(e)}")
    finally:
        cursor.close()
        connection.close()

def upload_zip_and_extract(subject: str, table_name: str):
    st.write("Upload a ZIP file containing images (named as sno, e.g., 1.jpg, 2.png):")
    uploaded_zip = st.file_uploader("Choose a ZIP file", type=['zip'], key=f"zip_{table_name}")
    if uploaded_zip:
        subject_folder = subject.split('(')[0].strip()
        img_folder = os.path.join("Img", subject_folder)
        os.makedirs(img_folder, exist_ok=True)
        
        # Compute hash to avoid processing the same file multiple times
        file_hash = hashlib.md5(uploaded_zip.getvalue()).hexdigest()
        if st.session_state.get(f"processed_zip_{table_name}_{file_hash}", False):
            st.info("This ZIP file has already been processed in this session.")
            return
        
        # Create a temporary directory to extract ZIP contents
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Save the uploaded ZIP to a temporary file
                temp_zip_path = os.path.join(temp_dir, "uploaded.zip")
                with open(temp_zip_path, "wb") as f:
                    f.write(uploaded_zip.getbuffer())
                
                # Extract ZIP contents to the temporary directory
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Move image files to the subject folder
                image_paths = {}
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            try:
                                sno = int(file.split('.')[0])
                                src_path = os.path.join(root, file)
                                dest_path = os.path.join(img_folder, file)
                                shutil.move(src_path, dest_path)
                                image_paths[sno] = dest_path
                            except (ValueError, OSError) as e:
                                st.warning(f"Skipping file '{file}': {str(e)}")
                                continue
                
                if not image_paths:
                    st.warning("No valid image files found in the ZIP.")
                    return
                
                # Update the database with image paths
                connection = get_db_connection()
                if not connection:
                    st.error("Failed to connect to the database.")
                    return
                
                try:
                    cursor = connection.cursor()
                    for sno, path in image_paths.items():
                        cursor.execute(f"""
                            UPDATE `{table_name}`
                            SET question = %s
                            WHERE sno = %s
                        """, (path, sno))
                    connection.commit()
                    st.success(f"Extracted {len(image_paths)} images from ZIP and updated the database!")
                    
                    # Mark this ZIP as processed
                    st.session_state[f"processed_zip_{table_name}_{file_hash}"] = True
                    if f"processed_zip_{table_name}" not in st.session_state:
                        st.session_state[f"processed_zip_{table_name}"] = True
                        st.rerun()
                except mysql.connector.Error as e:
                    st.error(f"Error updating database with ZIP images: {str(e)}")
                finally:
                    cursor.close()
                    connection.close()
            except Exception as e:
                st.error(f"Error processing ZIP file: {str(e)}")

def upload_single_image_for_sno(subject: str, table_name: str, df: pd.DataFrame):
    st.write("Upload a single image for a specific row:")
    if df is None or df.empty:
        st.warning("No rows available to select. Please create or load a table first.")
        return
    
    sno_options = df['sno'].astype(str).tolist()
    selected_sno = st.selectbox("Select Serial Number (sno)", sno_options, key=f"sno_select_{table_name}")
    
    uploaded_image = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'], key=f"image_{table_name}")
    if uploaded_image and selected_sno:
        try:
            sno = int(selected_sno)
            subject_folder = subject.split('(')[0].strip()
            img_folder = os.path.join("Img", subject_folder)
            os.makedirs(img_folder, exist_ok=True)
            
            # Compute hash from binary data to avoid duplicate processing
            file_hash = hashlib.md5(uploaded_image.getvalue()).hexdigest()
            if st.session_state.get(f"processed_image_{table_name}_{file_hash}", False):
                st.info("This image has already been processed in this session.")
                return
            
            img_path = os.path.join(img_folder, f"{sno}.{uploaded_image.name.split('.')[-1]}")
            with open(img_path, "wb") as f:
                f.write(uploaded_image.getbuffer())  # Write binary data directly
            
            connection = get_db_connection()
            if connection:
                try:
                    cursor = connection.cursor()
                    cursor.execute(f"""
                        UPDATE `{table_name}`
                        SET question = %s
                        WHERE sno = %s
                    """, (img_path, sno))
                    connection.commit()
                    st.success(f"Image '{uploaded_image.name}' uploaded and linked to sno {sno} at '{img_path}'!")
                    
                    st.session_state[f"processed_image_{table_name}_{file_hash}"] = True
                    if f"processed_image_{table_name}" not in st.session_state:
                        st.session_state[f"processed_image_{table_name}"] = True
                        st.rerun()
                except mysql.connector.Error as e:
                    st.error(f"Error updating database with image: {str(e)}")
                finally:
                    cursor.close()
                    connection.close()
            else:
                st.error("Failed to establish database connection.")
        except ValueError:
            st.error("Selected sno must be a valid number.")
        except Exception as e:
            st.error(f"Error uploading image: {str(e)}")

def download_table_to_excel(df: pd.DataFrame, subject: str):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Questions')
    buffer.seek(0)
    st.download_button(
        label="📥 Download Table as Excel",
        data=buffer,
        file_name=f"{sanitize_table_name(subject)}_question_bank.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download_excel_{sanitize_table_name(subject)}"
    )

def upload_excel_to_db(subject: str, table_name: str):
    st.write("Upload an Excel file to update the database:")
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'], key=f"excel_{table_name}")
    if uploaded_file:
        # Compute hash from binary data to avoid duplicate processing
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        if st.session_state.get(f"processed_excel_{table_name}_{file_hash}", False):
            st.info("This file has already been processed in this session.")
            return
        
        connection = get_db_connection()
        if not connection:
            st.error("Failed to connect to the database.")
            return
        
        try:
            if not check_table_exists(table_name):
                st.warning(f"Table '{table_name}' does not exist. Creating it now...")
                if create_table(subject) is None:
                    return
                connection = get_db_connection()
            
            # Read Excel file as binary and handle encoding explicitly
            excel_df = pd.read_excel(
                uploaded_file,
                dtype={
                    'sno': str,
                    'unit': str,
                    'question': str,
                    'type': str,
                    'CO': str,
                    'PO': str,
                    'BTL': str,
                    'counter': str
                },
                engine='openpyxl'  # Explicitly use openpyxl to avoid encoding issues
            )
            
            required_columns = ['sno', 'unit', 'question', 'type', 'CO', 'PO', 'BTL', 'counter']
            if not all(col in excel_df.columns for col in required_columns):
                st.error(f"Excel file must contain all required columns: {', '.join(required_columns)}")
                return
            
            excel_df = excel_df[required_columns].copy()
            excel_df['unit'] = excel_df['unit'].fillna('').astype(str)
            excel_df['question'] = excel_df['question'].fillna('').astype(str)
            excel_df['type'] = excel_df['type'].fillna('').astype(str).str.lower()
            excel_df['CO'] = excel_df['CO'].fillna('').astype(str)
            excel_df['PO'] = excel_df['PO'].fillna('').astype(str)
            excel_df['BTL'] = excel_df['BTL'].fillna('').astype(str)
            excel_df['sno'] = pd.to_numeric(excel_df['sno'], errors='coerce').fillna(0).astype(int)
            excel_df['counter'] = pd.to_numeric(excel_df['counter'], errors='coerce').fillna(0).astype(int)
            
            # Validate question paths
            for idx, row in excel_df.iterrows():
                if row['question'] and not os.path.exists(row['question']):
                    st.warning(f"Invalid image path in row {idx + 2} (sno {row['sno']}): {row['question']}")
                    excel_df.at[idx, 'question'] = ""
            
            excel_df = excel_df[excel_df['sno'] > 0]
            if excel_df.empty:
                st.error("No valid rows found in the Excel file.")
                return
            
            cursor = connection.cursor()
            rows_inserted = 0
            rows_updated = 0
            
            for _, row in excel_df.iterrows():
                sql = f"""
                    INSERT INTO `{table_name}` (sno, unit, question, type, CO, PO, BTL, counter)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    unit=VALUES(unit),
                    question=VALUES(question),
                    type=VALUES(type),
                    CO=VALUES(CO),
                    PO=VALUES(PO),
                    BTL=VALUES(BTL),
                    counter=VALUES(counter)
                """
                values = (
                    int(row['sno']),
                    str(row['unit']),
                    str(row['question']),
                    str(row['type']),
                    str(row['CO']),
                    str(row['PO']),
                    str(row['BTL']),
                    int(row['counter'])
                )
                cursor.execute(sql, values)
                if cursor.rowcount == 1:
                    rows_inserted += 1
                elif cursor.rowcount == 2:
                    rows_updated += 1
            
            connection.commit()
            st.success(f"Database updated! {rows_inserted} rows inserted, {rows_updated} rows updated.")
            
            st.session_state[f"processed_excel_{table_name}_{file_hash}"] = True
            if f"processed_excel_{table_name}" not in st.session_state:
                st.session_state[f"processed_excel_{table_name}"] = True
                st.rerun()
        
        except Exception as e:
            st.error(f"Error uploading Excel: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            connection.close()

def display_table(subject: str):
    table_name = sanitize_table_name(subject)
    connection = get_db_connection()
    if not connection:
        st.error("Database connection failed in display_table.")
        return None
    try:
        df = pd.read_sql(f"SELECT * FROM `{table_name}`", connection)
        connection.close()
        
        df['counter'] = df['counter'].astype(int)
        
        st.subheader(f"Question Bank for {subject}")
        
        # Convert image paths to base64 for display in data_editor
        def path_to_image_html(path):
            if pd.notna(path) and os.path.exists(path):
                try:
                    with open(path, "rb") as img_file:
                        encoded = base64.b64encode(img_file.read()).decode('utf-8')
                        return f'data:image/jpeg;base64,{encoded}'
                except Exception as e:
                    st.warning(f"Error loading image at {path}: {str(e)}")
                    return ""
            return ""

        # Create a copy of df for display with image rendering
        display_df = df.copy()
        display_df['question'] = display_df['question'].apply(path_to_image_html)

        edited_df = st.data_editor(
            display_df,
            num_rows="dynamic",
            column_config={
                "sno": st.column_config.NumberColumn("S.No", disabled=True),
                "unit": st.column_config.TextColumn("Unit"),
                "question": st.column_config.ImageColumn(
                    "Question",
                    help="Displays the image if path is valid",
                    width="medium"
                ),
                "type": st.column_config.SelectboxColumn("Type", options=["short", "long"]),
                "CO": st.column_config.TextColumn("CO"),
                "PO": st.column_config.TextColumn("PO"),
                "BTL": st.column_config.TextColumn("BTL"),
                "counter": st.column_config.NumberColumn("Counter", disabled=True)
            },
            use_container_width=True,
            key=f"editor_{table_name}"
        )
        
        # Revert edited_df['question'] back to paths for saving
        edited_df['question'] = df['question']  # Preserve original paths
        
        if len(edited_df) > len(df):
            max_sno = edited_df['sno'].max() if not edited_df['sno'].empty else 0
            new_rows = edited_df.tail(len(edited_df) - len(df))
            for idx in new_rows.index:
                edited_df.at[idx, 'sno'] = max_sno + 1
                max_sno += 1
                edited_df.at[idx, 'question'] = ""  # Default to empty path for new rows
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Save Table Changes", key=f"save_{table_name}"):
                save_table_changes(table_name, edited_df)
        with col2:
            download_table_to_excel(edited_df, subject)
        with col3:
            upload_excel_to_db(subject, table_name)
        
        return edited_df
    except mysql.connector.Error as e:
        st.error(f"Error loading table '{table_name}': {str(e)}")
        if connection:
            connection.close()
        return None
    except Exception as e:
        st.error(f"Unexpected error loading table: {str(e)}")
        if connection:
            connection.close()
        return None

def create_subjects_table():
    connection = get_db_connection()
    if not connection:
        return
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS `subjects` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL
            )
        """)
        cursor.execute("SELECT COUNT(*) FROM `subjects`")
        count = cursor.fetchone()[0]
        if count == 0:
            for subject in SEMESTER_SUBJECTS.get(("I B.Tech", "I"), []):
                cursor.execute("INSERT IGNORE INTO `subjects` (name) VALUES (%s)", (subject,))
        connection.commit()
    except mysql.connector.Error as e:
        st.error(f"Error creating subjects table: {str(e)}")
    finally:
        cursor.close()
        connection.close()

def fetch_subjects():
    connection = get_db_connection()
    if not connection:
        return []
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM `subjects` ORDER BY name")
        subjects = [row[0] for row in cursor.fetchall()]
        return subjects
    except mysql.connector.Error as e:
        st.error(f"Error fetching subjects: {str(e)}")
        return []
    finally:
        cursor.close()
        connection.close()

def add_subject(subject_name):
    connection = get_db_connection()
    if not connection:
        return False
    try:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO `subjects` (name) VALUES (%s) ON DUPLICATE KEY UPDATE name=name", (subject_name,))
        connection.commit()
        return True
    except mysql.connector.Error as e:
        st.error(f"Error adding subject: {str(e)}")
        return False
    finally:
        cursor.close()
        connection.close()

def update_subject(old_name, new_name):
    connection = get_db_connection()
    if not connection:
        return False
    try:
        cursor = connection.cursor()
        cursor.execute("UPDATE `subjects` SET name = %s WHERE name = %s", (new_name, old_name))
        connection.commit()
        return cursor.rowcount > 0
    except mysql.connector.Error as e:
        st.error(f"Error updating subject: {str(e)}")
        return False
    finally:
        cursor.close()
        connection.close()

def delete_subject(subject_name):
    connection = get_db_connection()
    if not connection:
        return False
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM `subjects` WHERE name = %s", (subject_name,))
        connection.commit()
        return cursor.rowcount > 0
    except mysql.connector.Error as e:
        st.error(f"Error deleting subject: {str(e)}")
        return False
    finally:
        cursor.close()
        connection.close()

def create_users_table():
    connection = get_db_connection()
    if not connection:
        return
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS `users` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                role ENUM('faculty', 'admin') NOT NULL
            )
        """)
        cursor.execute("SELECT COUNT(*) FROM `users`")
        count = cursor.fetchone()[0]
        if count == 0:
            cursor.execute("INSERT IGNORE INTO `users` (username, password, role) VALUES (%s, %s, %s)", 
                          ("admin", "admin", "admin"))
            cursor.execute("INSERT IGNORE INTO `users` (username, password, role) VALUES (%s, %s, %s)", 
                          ("faculty", "faculty", "faculty"))
        connection.commit()
    except mysql.connector.Error as e:
        st.error(f"Error creating users table: {str(e)}")
    finally:
        cursor.close()
        connection.close()

def fetch_users():
    connection = get_db_connection()
    if not connection:
        return []
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT username, password, role FROM `users`")
        users = cursor.fetchall()
        return users
    except mysql.connector.Error as e:
        st.error(f"Error fetching users: {str(e)}")
        return []
    finally:
        cursor.close()
        connection.close()

def add_user(username, password, role):
    connection = get_db_connection()
    if not connection:
        return False
    try:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO `users` (username, password, role) VALUES (%s, %s, %s)", 
                      (username, password, role))
        connection.commit()
        return True
    except mysql.connector.Error as e:
        st.error(f"Error adding user: {str(e)}")
        return False
    finally:
        cursor.close()
        connection.close()

def update_user(old_username, new_username, new_password, new_role):
    connection = get_db_connection()
    if not connection:
        return False
    try:
        cursor = connection.cursor()
        cursor.execute("UPDATE `users` SET username = %s, password = %s, role = %s WHERE username = %s", 
                      (new_username, new_password, new_role, old_username))
        connection.commit()
        return cursor.rowcount > 0
    except mysql.connector.Error as e:
        st.error(f"Error updating user: {str(e)}")
        return False
    finally:
        cursor.close()
        connection.close()

def delete_user(username):
    connection = get_db_connection()
    if not connection:
        return False
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM `users` WHERE username = %s", (username,))
        connection.commit()
        return cursor.rowcount > 0
    except mysql.connector.Error as e:
        st.error(f"Error deleting user: {str(e)}")
        return False
    finally:
        cursor.close()
        connection.close()

def check_password():
    def password_entered():
        username = st.session_state.get("username", "").strip()
        password = st.session_state.get("password", "").strip()
        if not username or not password:
            st.session_state["password_correct"] = False
            return
        connection = get_db_connection()
        if not connection:
            st.error("Database connection failed.")
            return
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT password, role FROM `users` WHERE username = %s", (username,))
            result = cursor.fetchone()
            if result:
                stored_password, role = result
                st.session_state["password_correct"] = stored_password == password
                if st.session_state["password_correct"]:
                    st.session_state["role"] = role
            else:
                st.session_state["password_correct"] = False
        except mysql.connector.Error as e:
            st.error(f"Error validating credentials: {str(e)}")
        finally:
            cursor.close()
            connection.close()
            if "password" in st.session_state:
                del st.session_state["password"]
            if "username" in st.session_state:
                del st.session_state["username"]

    logo_path = "img/rvitlogo_f.jpg"
    logo_html = '<p>RV Institute of Technology</p>'
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                encoded_logo = base64.b64encode(f.read()).decode('utf-8')
            logo_html = f'<img src="data:image/jpeg;base64,{encoded_logo}" width="150" style="border-radius: 8px; margin-bottom: 15px;">'
        except Exception as e:
            st.error(f"Failed to load logo '{logo_path}' for login page: {str(e)}")

    if "password_correct" not in st.session_state:
        st.markdown(f'''
    <div style="
        text-align: center;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    ">
        {logo_html}
        <h1 style="margin: 0; font-size: 2.5em; color: #333;">
            RV Institute of Technology
        </h1>
        <h2 style="margin: 5px 0 0; font-size: 1.5em; color: #555;">
            Random Question Paper Generator
        </h2>
    </div>
''', unsafe_allow_html=True)
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Log In", on_click=password_entered)
        return False
    elif not st.session_state["password_correct"]:
        st.markdown('<div class="main-header"><h1>🔐 Login Required</h1></div>', unsafe_allow_html=True)
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.error("😕 Invalid credentials")
        st.button("Log In", on_click=password_entered)
        return False
    else:
        return True

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Logged out successfully!")
    st.rerun()

def manage_subjects():
    st.subheader("🛠 Manage Subjects")
    create_subjects_table()
    
    subjects = fetch_subjects()
    
    if not subjects:
        st.warning("No subjects found in the database.")
    
    with st.expander("Add New Subject", expanded=False):
        new_subject = st.text_input("Enter new subject name (e.g., 'Subject Name (Code)')")
        if st.button("Add Subject"):
            if new_subject and new_subject.strip():
                if add_subject(new_subject.strip()):
                    st.success(f"Subject '{new_subject.strip()}' added successfully!")
                    st.rerun()
                else:
                    st.error("Failed to add subject.")
            else:
                st.error("Subject name cannot be empty.")

    with st.expander("Edit/Delete Subjects", expanded=False):
        if subjects:
            subject_to_edit = st.selectbox("Select subject to edit/delete", subjects)
            col1, col2 = st.columns(2)
            with col1:
                updated_name = st.text_input("New subject name", value=subject_to_edit)
                if st.button("Update Subject"):
                    if updated_name and updated_name.strip() and updated_name.strip() != subject_to_edit:
                        if update_subject(subject_to_edit, updated_name.strip()):
                            st.success(f"Subject updated to '{updated_name.strip()}'!")
                            st.rerun()
                        else:
                            st.error("Failed to update subject.")
                    else:
                        st.error("New name cannot be empty or same as old name.")
            with col2:
                if st.button("Delete Subject"):
                    if delete_subject(subject_to_edit):
                        st.success(f"Subject '{subject_to_edit}' deleted successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to delete subject.")
        else:
            st.info("No subjects available to edit or delete.")

def manage_users():
    st.subheader("👤 Manage Users")
    create_users_table()
    
    users = fetch_users()
    
    if not users:
        st.warning("No users found in the database.")
    
    with st.expander("Add New User", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_username = st.text_input("Username", key="new_username")
        with col2:
            new_password = st.text_input("Password", type="password", key="new_password")
        with col3:
            new_role = st.selectbox("Role", ["faculty", "admin"], key="new_user_role")
        if st.button("Add User", key="add_user_button"):
            if new_username and new_password and new_role:
                if add_user(new_username, new_password, new_role):
                    st.success(f"User '{new_username}' added successfully!")
                    st.rerun()
                else:
                    st.error("Failed to add user.")
            else:
                st.error("All fields are required.")

    with st.expander("Edit/Delete Users", expanded=False):
        if users:
            user_to_edit = st.selectbox("Select user to edit/delete", [user[0] for user in users], key="select_user_to_edit")
            selected_user = next((u for u in users if u[0] == user_to_edit), None)
            if selected_user:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    updated_username = st.text_input("Username", value=selected_user[0], key="edit_username")
                with col2:
                    updated_password = st.text_input("Password", type="password", value=selected_user[1], key="edit_password")
                with col3:
                    updated_role = st.selectbox("Role", ["faculty", "admin"], index=0 if selected_user[2] == "faculty" else 1, key="edit_user_role")
                with col4:
                    if st.button("Update User", key="update_user_button"):
                        if updated_username and updated_password and updated_role:
                            if update_user(selected_user[0], updated_username, updated_password, updated_role):
                                st.success(f"User '{updated_username}' updated successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to update user.")
                        else:
                            st.error("All fields are required.")
                if st.button("Delete User", key="delete_user_button"):
                    if delete_user(selected_user[0]):
                        st.success(f"User '{selected_user[0]}' deleted successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to delete user.")
        else:
            st.info("No users available to edit or delete.")
            
def validate_question_counts(df: pd.DataFrame, subject: str, exam: ExamType) -> list:
    errors = []
    if (subject, exam) not in PATTERNS:
        add_default_pattern(subject, exam)
    pattern = PATTERNS[(subject, exam)]
    df_valid = df[df['question'].notna() & (df['question'] != '') & df['question'].apply(lambda x: os.path.exists(x) if x else False)]
    for unit, type_counts in pattern.part_a.items():
        for qtype, count in type_counts.items():
            available = len(df_valid[(df_valid['unit'] == str(unit)) & (df_valid['type'] == qtype)])
            if available < count:
                errors.append(f"Unit {unit} needs {count} {qtype} questions for Part A, but has only {available}")
    for unit, type_counts in pattern.part_b.items():
        for qtype, count in type_counts.items():
            available = len(df_valid[(df_valid['unit'] == str(unit)) & (df_valid['type'] == qtype)])
            if available < count:
                errors.append(f"Unit {unit} needs {count} {qtype} questions for Part B, but has only {available}")
    return errors

def select_questions(df: pd.DataFrame, subject: str, exam: ExamType) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if (subject, exam) not in PATTERNS:
        add_default_pattern(subject, exam)
    pattern = PATTERNS[(subject, exam)]
    part_a = pd.DataFrame()
    part_b = pd.DataFrame()
    
    connection = get_db_connection()
    if not connection:
        return part_a, part_b
    table_name = sanitize_table_name(subject)
    try:
        df_fresh = pd.read_sql(f"SELECT * FROM `{table_name}`", connection)
        df_valid = df_fresh[df_fresh['question'].notna() & 
                           (df_fresh['question'] != '') & 
                           df_fresh['question'].apply(lambda x: os.path.exists(x) if x else False)].copy()
        
        df_valid = df_valid.sort_values('counter', kind='stable')
        df_valid = df_valid.groupby(['unit', 'type', 'counter']).apply(lambda x: x.sample(frac=1, random_state=random.randint(1, 10000))).reset_index(drop=True)

        for unit, type_counts in pattern.part_a.items():
            for qtype, count in type_counts.items():
                available = df_valid[(df_valid['unit'] == str(unit)) & (df_valid['type'] == qtype)]
                if len(available) >= count:
                    questions = available.head(count)
                    part_a = pd.concat([part_a, questions])
                    df_valid = df_valid[~df_valid['sno'].isin(questions['sno'])]

        for unit, type_counts in pattern.part_b.items():
            for qtype, count in type_counts.items():
                available = df_valid[(df_valid['unit'] == str(unit)) & (df_valid['type'] == qtype)]
                if len(available) >= count:
                    questions = available.head(count)
                    part_b = pd.concat([part_b, questions])
                    df_valid = df_valid[~df_valid['sno'].isin(questions['sno'])]
    except Exception as e:
        st.error(f"Error selecting questions: {str(e)}")
    finally:
        connection.close()
    
    return part_a, part_b

def update_counters(table_name: str, selected_snos: List[int]):
    connection = get_db_connection()
    if not connection:
        return
    try:
        cursor = connection.cursor()
        for sno in selected_snos:
            cursor.execute(f"""
                UPDATE `{table_name}`
                SET counter = counter + 1
                WHERE sno = %s
            """, (sno,))
        connection.commit()
    except mysql.connector.Error as e:
        st.error(f"Error updating counters: {str(e)}")
    finally:
        cursor.close()
        connection.close()

def generate_pdf_with_header(part_a: pd.DataFrame, part_b: pd.DataFrame, 
                           subject: str, exam: ExamType, 
                           header_info: dict,
                           pdf_password: str = None,
                           max_image_width: int = 410) -> bytes:
    buffer = BytesIO()
    user_password = pdf_password.encode('utf-8') if pdf_password else generate_secure_password(10).encode('utf-8')
    encryption = StandardEncryption(
        userPassword=user_password,
        ownerPassword='admin123'.encode('utf-8'),
        strength=128,
        canPrint=1,
        canModify=0,
        canCopy=0,
        canAnnotate=0
    )
    ist_timezone = timezone(timedelta(hours=5, minutes=30))
    current_timestamp = datetime.now(ist_timezone).strftime("%d-%b-%Y %H:%M:%S IST")
    doc = CustomDocTemplate(
        buffer,
        pagesize=A4,
        bottomMargin=30,
        leftMargin=30,
        rightMargin=30,
        top_margin_first=0,
        top_margin_rest=30,
        encryption=encryption,
        timestamp=current_timestamp
    )
    styles = getSampleStyleSheet()
    story = []
    styles.add(ParagraphStyle(
        'QuestionStyle',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        spaceBefore=6,
        spaceAfter=6,
        wordWrap='CJK',
        allowWidows=1,
        allowOrphans=1
    ))
    styles.add(ParagraphStyle(
        'CollegeHeader',
        parent=styles['Title'],
        fontSize=16,
        alignment=1,
        spaceAfter=2
    ))
    styles.add(ParagraphStyle(
        'ExamHeader',
        parent=styles['Title'],
        fontSize=14,
        alignment=1,
        spaceAfter=2
    ))
    styles.add(ParagraphStyle(
        'DetailsLeft',
        parent=styles['Normal'],
        fontSize=11,
        alignment=0,
        leftIndent=20
    ))
    styles.add(ParagraphStyle(
        'DetailsRight',
        parent=styles['Normal'],
        fontSize=11,
        alignment=2,
        rightIndent=20
    ))
    styles.add(ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontSize=12,
        fontName='Helvetica-Bold',
        alignment=1,
        spaceBefore=6,
        spaceAfter=6
    ))

    def create_question_cell(image_path: str) -> List:
        elements = []
        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, 'rb') as f:
                    img_data = BytesIO(f.read())
                img = Image.open(img_data)
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                width, height = configure_question_image(img, max_image_width)
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
                img_byte_arr = img_byte_arr.getvalue()
                img_reader = BytesIO(img_byte_arr)
                elements.append(RLImage(img_reader, width=width, height=height))
            except Exception as e:
                elements.append(Paragraph(f"[Error loading image: {str(e)}]", styles['Normal']))
        return elements

    # Skip rvitlogo_f.jpg and use only rvit_exam_header_f.png
    exam_header_path = "img/rvit_exam_header_f.png"
    if os.path.exists(exam_header_path):
        try:
            with open(exam_header_path, 'rb') as f:
                img_data = BytesIO(f.read())
            img = Image.open(img_data)
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')
            aspect = img.width / img.height
            desired_width = 540
            height = int(desired_width / aspect)
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG', quality=85, optimize=True)
            img_byte_arr = img_byte_arr.getvalue()
            img_reader = BytesIO(img_byte_arr)
            exam_header = RLImage(img_reader, width=desired_width, height=height)
            story.append(exam_header)
        except Exception as e:
            exam_header = (
                f"{header_info['academic_level']} {header_info['semester']} SEMESTER "
                f"{header_info['exam_type']} EXAMINATION {header_info['month_year']}"
            )
            story.append(Paragraph(exam_header, styles['ExamHeader']))
    else:
        exam_header = (
            f"{header_info['academic_level']} {header_info['semester']} SEMESTER "
            f"{header_info['exam_type']} EXAMINATION {header_info['month_year']}"
        )
        story.append(Paragraph(exam_header, styles['ExamHeader']))

    story.append(Spacer(1, 2))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
    story.append(Spacer(1, 2))

    exam_time = "2 Hours" if header_info['exam_type'] in ["I MID", "II MID"] else "3 Hours"
    max_marks = "25M" if header_info['exam_type'] in ["I MID", "II MID"] else "70M"
    branch_text = f"Branch: {', '.join(header_info['branches'])}"
    date_text = f"Date: {header_info['date']}"
    subject_text = f"Subject: {subject}"
    data = [
        [Paragraph(branch_text, styles['DetailsLeft']), 
         Paragraph("Regulation: RV23", styles['DetailsRight'])],
        [Paragraph(date_text, styles['DetailsLeft']), 
         Paragraph(f"Max.Marks: {max_marks}", styles['DetailsRight'])],
        [Paragraph(subject_text, styles['DetailsLeft']), 
         Paragraph(f"Time: {exam_time}", styles['DetailsRight'])]
    ]
    total_width = doc.width
    col_widths = [total_width * 0.7, total_width * 0.3]
    details_table = Table(data, colWidths=col_widths)
    details_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (-1, 0), (-1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(details_table)
    story.append(Spacer(1, 2))
    story.append(Spacer(1, 2))
    total_width = doc.width
    col_widths = [
        total_width * 0.06,
        total_width * 0.74,
        total_width * 0.06,
        total_width * 0.06,
        total_width * 0.06
    ]
    headers = ['Q.No', 'Question', 'CO', 'PO', 'BTL']
    table_data = [headers]
    if (subject, exam) not in PATTERNS:
        add_default_pattern(subject, exam)
    pattern = PATTERNS[(subject, exam)]

    def add_question_to_table(q_num, question_row):
        image_path = question_row['question']
        co = str(question_row['CO']) if pd.notna(question_row['CO']) else ""
        po = str(question_row['PO']) if pd.notna(question_row['PO']) else ""
        btl = str(question_row['BTL']) if pd.notna(question_row['BTL']) else ""
        cell_content = create_question_cell(image_path)
        return [str(q_num), cell_content, co, po, btl]

    if subject == "Engineering Graphics (RV23991T08)":
        q_num = 1
        questions = pd.concat([part_a, part_b]) if not part_a.empty else part_b
        for i in range(0, len(questions), 2):
            if i + 1 < len(questions):
                table_data.append(add_question_to_table(q_num, questions.iloc[i]))
                table_data.append(['', Paragraph('<b>(OR)</b>', styles['QuestionStyle']), '', '', ''])
                table_data.append(add_question_to_table(q_num + 1, questions.iloc[i + 1]))
                q_num += 2
    else:
        if not part_a.empty:
            marks_text = f"<b>PART-A : ANSWER ALL QUESTIONS : ({pattern.marks_a[0]}×{pattern.marks_a[1]}={pattern.marks_a[0]*pattern.marks_a[1]}M)</b>"
            table_data.append(['', Paragraph(marks_text, styles['SectionHeader']), '', '', ''])
            part_a_header_row = len(table_data) - 1
            sorted_part_a = part_a.sort_values(['unit'])
            for idx, (_, row) in enumerate(sorted_part_a.iterrows()):
                question_letter = f"1.{chr(97 + idx)})"
                table_data.append(add_question_to_table(question_letter, row))
        if not part_b.empty:
            marks_text = f"<b>PART-B : ANSWER ONE QUESTION FROM EACH UNIT : ({pattern.marks_b[0]}×{pattern.marks_b[1]}={pattern.marks_b[0]*pattern.marks_b[1]}M)</b>"
            table_data.append(['', Paragraph(marks_text, styles['SectionHeader']), '', '', ''])
            part_b_header_row = len(table_data) - 1
            sorted_part_b = part_b.sort_values('unit')
            q_num = 2
            for i in range(0, len(sorted_part_b), 2):
                if i + 1 < len(sorted_part_b):
                    table_data.append(add_question_to_table(q_num, sorted_part_b.iloc[i]))
                    table_data.append(['', Paragraph('<b>(OR)</b>', styles['QuestionStyle']), '', '', ''])
                    table_data.append(add_question_to_table(q_num + 1, sorted_part_b.iloc[i + 1]))
                    q_num += 2
    question_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table_style = [
        ('BOX', (0, 0), (-1, -1), 1.0, colors.black),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'CENTER'),
        ('ALIGN', (2, 1), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('LINEBELOW', (0, 0), (-1, 0), 1.5, colors.black),
    ]
    if subject != "Engineering Graphics (RV23991T08)":
        if not part_a.empty:
            table_style.append(('BACKGROUND', (0, part_a_header_row), (-1, part_a_header_row), colors.lightgrey))
        if not part_b.empty:
            table_style.append(('BACKGROUND', (0, part_b_header_row), (-1, part_b_header_row), colors.lightgrey))
    question_table.setStyle(TableStyle(table_style))
    story.append(question_table)
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def main():
    if st.session_state.get('logout', False):
        logout()

    if not check_password():
        st.stop()

    role = st.session_state.get("role", "faculty")
    logo_path = "img/rvitlogo_f.jpg"
    logo_html = '<p>RV Institute of Technology</p>'
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                encoded_logo = base64.b64encode(f.read()).decode('utf-8')
            logo_html = f'<img src="data:image/jpeg;base64,{encoded_logo}" width="80" style="margin-bottom: 10px;">'
        except Exception as e:
            st.error(f"Failed to load logo '{logo_path}' for main page: {str(e)}")

    st.markdown(f'''
    <div style="text-align: center;">
        {logo_html}
        <h1 style="margin: 0;">Question Paper Generator</h1>
    </div>
''', unsafe_allow_html=True)

    st.sidebar.button("Logout", on_click=lambda: st.session_state.update({'logout': True}))

    with st.expander("ℹ️ Instructions", expanded=False):
        if role == "faculty":
            st.markdown("""
            ### How to use (Faculty):
            1. Manage subjects (add, edit, delete) in the 'Manage Subjects' section
            2. Select a subject to view or create its question bank
            3. Upload a ZIP file with images, an Excel file, or a single image for a specific sno
            4. Edit the question bank if needed or download as Excel to edit offline
            """)
        elif role == "admin":
            st.markdown("""
            ### How to use (Admin):
            1. Configure the paper header details
            2. Select subject and pattern to generate the question paper
            3. Download the question paper as PDF
            4. Manage users (add, edit, delete) in the 'Manage Users' section
            """)

    if role == "faculty":
        manage_subjects()
        
        st.subheader("📚 Manage Question Bank")
        subjects = fetch_subjects()
        subject_for_bank = st.selectbox(
            "Select Subject for Question Bank",
            subjects if subjects else ["No subjects available"],
            key="subject_bank"
        )
        
        if not subjects or subject_for_bank == "No subjects available":
            st.warning("Please add subjects in the 'Manage Subjects' section to proceed.")
            return
        
        table_name = sanitize_table_name(subject_for_bank)
        
        df = None
        if check_table_exists(table_name):
            connection = get_db_connection()
            if connection:
                df = pd.read_sql(f"SELECT * FROM `{table_name}`", connection)
                connection.close()
            df = display_table(subject_for_bank)
            upload_zip_and_extract(subject_for_bank, table_name)
            upload_single_image_for_sno(subject_for_bank, table_name, df)
        else:
            st.warning(f"Table for '{subject_for_bank}' does not exist.")
            if st.button("Create Table"):
                df = create_table(subject_for_bank)
                if df is not None:
                    st.rerun()
        
        if df is not None and not df.empty:
            st.subheader("📊 Question Bank Overview")
            overview = df.pivot_table(
                index='unit',
                columns='type',
                values='sno',
                aggfunc='count',
                fill_value=0
            )
            st.dataframe(overview, use_container_width=True)

    elif role == "admin":
        manage_users()
        
        header_info = create_header_config()
        
        if header_info:
            col1, col2 = st.columns(2)
            with col1:
                available_subjects = fetch_subjects()
                subject = st.selectbox(
                    "Select Subject",
                    available_subjects if available_subjects else ["No subjects available"],
                    key="subject_paper"
                )
            
            with col2:
                if header_info['exam_type'] == "I MID":
                    exam_options = [ExamType.MID1_2.value, ExamType.MID1_2_5.value]
                elif header_info['exam_type'] == "II MID":
                    exam_options = [ExamType.MID2_3_4_5.value, ExamType.MID2_5_5.value]
                else:
                    exam_options = [ExamType.REGULAR.value, ExamType.SUPPLY.value]
                exam_value = st.selectbox("Select Question Pattern", exam_options)
                exam = next((e for e in ExamType if e.value == exam_value), ExamType.REGULAR)
            
            if st.button("🎯 Generate Question Paper", use_container_width=True):
                if not available_subjects or subject == "No subjects available":
                    st.error("No subjects available. Please add subjects in the 'Manage Subjects' section.")
                    return
                
                connection = get_db_connection()
                if connection:
                    table_name = sanitize_table_name(subject)
                    df = pd.read_sql(f"SELECT * FROM `{table_name}`", connection)
                    connection.close()
                else:
                    st.error("Database connection failed. Cannot generate question paper.")
                    return
                
                try:
                    errors = validate_question_counts(df, subject, exam)
                    if errors:
                        st.error("Insufficient questions:")
                        for error in errors:
                            st.write(f"- {error}")
                        return
                    part_a, part_b = select_questions(df, subject, exam)
                    if part_a.empty and part_b.empty:
                        st.error("No valid questions selected. Please check the question bank.")
                        return
                    
                    if 'current_pdf_password' not in st.session_state:
                        unique_password = generate_secure_password(10)
                        st.session_state['current_pdf_password'] = unique_password
                    else:
                        unique_password = st.session_state['current_pdf_password']
                    
                    st.session_state['current_part_a'] = part_a
                    st.session_state['current_part_b'] = part_b
                    st.session_state['current_subject'] = subject
                    st.session_state['current_exam'] = exam
                    st.session_state['current_header_info'] = header_info
                    
                    tab1, tab2 = st.tabs(["Selected Questions", "Download Options"])
                    with tab1:
                        st.subheader("Selected Questions Overview")
                        selected_questions = pd.concat([
                            part_a.assign(part='A'),
                            part_b.assign(part='B')
                        ]).reset_index(drop=True)
                        if selected_questions.empty:
                            st.warning("No questions selected for display.")
                        else:
                            display_df = selected_questions[['part', 'unit', 'type', 'question', 'CO', 'PO', 'BTL', 'counter']]
                            display_df.columns = [col.title() for col in display_df.columns]

                            def image_to_base64(image_path):
                                if pd.notna(image_path) and os.path.exists(image_path):
                                    try:
                                        with open(image_path, "rb") as img_file:
                                            encoded = base64.b64encode(img_file.read()).decode('utf-8')
                                            return f"data:image/jpeg;base64,{encoded}"
                                    except Exception:
                                        return ""
                                return ""

                            display_df['Question'] = display_df['Question'].apply(image_to_base64)

                            st.dataframe(
                                display_df,
                                column_config={
                                    "Part": st.column_config.TextColumn("Part"),
                                    "Unit": st.column_config.TextColumn("Unit"),
                                    "Type": st.column_config.TextColumn("Type"),
                                    "Question": st.column_config.ImageColumn("Question", width="medium"),
                                    "Co": st.column_config.TextColumn("CO"),
                                    "Po": st.column_config.TextColumn("PO"),
                                    "Btl": st.column_config.TextColumn("BTL"),
                                    "Counter": st.column_config.NumberColumn("Counter")
                                },
                                use_container_width=True,
                                hide_index=True
                            )
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Questions", len(display_df))
                            with col2:
                                st.metric("Part A Questions", len(display_df[display_df['Part'] == 'A']))
                            with col3:
                                st.metric("Part B Questions", len(display_df[display_df['Part'] == 'B']))
                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            pdf_bytes = generate_pdf_with_header(
                                part_a, part_b, subject, exam,
                                header_info, pdf_password=unique_password
                            )
                            st.session_state['current_pdf_bytes'] = pdf_bytes
                            def on_download():
                                selected_snos = list(part_a['sno']) + list(part_b['sno'])
                                update_counters(sanitize_table_name(subject), selected_snos)
                            
                            st.download_button(
                                "📄 Download Question Paper (PDF)",
                                pdf_bytes,
                                f"{subject}_{exam.value}_{header_info['month_year']}.pdf",
                                "application/pdf",
                                on_click=on_download,
                                key="download_pdf"
                            )
                        with col2:
                            current_password = st.session_state['current_pdf_password']
                            st.info(f"PDF Password: `{current_password}`")
                            # Use st.components.v1.html for reliable JavaScript execution
                            copy_script = f"""
                                <button class="copy-button" onclick="navigator.clipboard.writeText('{current_password}').then(() => {{
                                    const tempMsg = document.createElement('div');
                                    tempMsg.innerText = 'Password copied to clipboard!';
                                    tempMsg.style.color = 'green';
                                    tempMsg.style.marginTop = '5px';
                                    document.body.appendChild(tempMsg); // Fallback to body
                                    setTimeout(() => tempMsg.remove(), 2000);
                                }}, () => {{ console.log('Clipboard access failed'); }});">Copy Password</button>
                            """
                            st.components.v1.html(copy_script, height=50)
                            st.warning("Save this password securely. It is required to open the PDF.")
                            if st.button("Generate New Password"):
                                new_password = generate_secure_password(10)
                                st.session_state['current_pdf_password'] = new_password
                                st.rerun()
                except Exception as e:
                    st.error(f"Error generating question paper: {str(e)}")

if __name__ == "__main__":
    main()
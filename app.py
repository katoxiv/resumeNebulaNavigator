import os
from io import BytesIO
import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
import docx2txt
import nltk
from nltk.corpus import stopwords
from rake_nltk import Rake
import pytesseract
from dotenv import load_dotenv
import re
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import matplotlib.pyplot as plt
from wordcloud import WordCloud

load_dotenv()

llm = OpenAI()

llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
             temperature=.7,
             max_tokens=1000,
    )

class CVAnalyzer:
    def __init__(self):
        # Download stopwords if not already downloaded
        nltk.download('stopwords')

    def extract_cv_information(self, uploaded_file):
        try:
            # Convert UploadedFile to bytes
            file_bytes = uploaded_file.read()

            # Extract text from PDF, Word, or plain text file
            text = self.extract_text_from_file(BytesIO(file_bytes))

            # Check if OCR is needed for image-based or scanned PDFs
            if not text and uploaded_file.type.lower() in (".png", ".jpg", ".jpeg", ".tiff"):
                text = self.extract_text_from_image(BytesIO(file_bytes))

            return text
        except Exception as e:
            st.error(f"Error extracting information from {uploaded_file.name}: {e}")
            return None

    def extract_text_from_file(self, file):
        # Extract text from PDF, Word, or plain text file
        if isinstance(file, str) or isinstance(file, os.PathLike):
            file_extension = os.path.splitext(file)[1].lower()

            if file_extension == ".pdf":
                return self.extract_text_from_pdf(file)
            elif file_extension == ".docx":
                return self.extract_text_from_docx(file)
            elif file_extension == ".txt":
                return self.extract_text_from_txt(file)
            else:
                raise ValueError("Unsupported file format")
        elif isinstance(file, BytesIO):  # Handle BytesIO object
            # Determine file type based on content
            file_type = self.get_file_type(file)

            if file_type == "pdf":
                return self.extract_text_from_pdf_bytesio(file)
            elif file_type == "docx":
                return self.extract_text_from_docx_bytesio(file)
            elif file_type == "txt":
                return self.extract_text_from_txt_bytesio(file)
            else:
                raise ValueError("Unsupported file format")
        else:
            raise ValueError("Invalid file object")

    def get_file_type(self, file):
        # Determine file type based on content
        file.seek(0)
        signature = file.read(4)
        file.seek(0)
        if signature == b"%PDF":
            return "pdf"
        elif signature == b"\x50\x4B\x03\x04":  # Check for ZIP file signature (docx)
            return "docx"
        elif signature == b"\xFF\xD8\xFF\xE0" or signature == b"\xFF\xD8\xFF\xE1":  # Check for JPEG signature
            return "jpg"
        elif signature == b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A":  # Check for PNG signature
            return "png"
        else:
            return "txt"

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        try:
            with fitz.open(pdf_path) as pdf_document:
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    text += page.get_text()
        except Exception as e:
            st.warning(f"Error extracting text from PDF: {e}")
        return text

    def extract_text_from_docx(self, docx_path):
        text = ""
        try:
            text = docx2txt.process(docx_path)
        except Exception as e:
            st.warning(f"Error extracting text from DOCX: {e}")
        return text

    def extract_text_from_txt(self, txt_path):
        text = ""
        try:
            with open(txt_path, 'r', encoding='utf-8') as txt_file:
                text = txt_file.read()
        except Exception as e:
            st.warning(f"Error reading text file: {e}")
        return text

    def extract_text_from_image(self, image_path):
        # Use OCR for image-based or scanned resumes
        text = ""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
        except Exception as e:
            st.warning(f"Error extracting text from image: {e}")
        return text

    def extract_text_from_pdf_bytesio(self, pdf_bytesio):
        text = ""
        try:
            with fitz.open("pdf", pdf_bytesio) as pdf_document:
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    text += page.get_text()
        except Exception as e:
            st.warning(f"Error extracting text from PDF: {e}")
        return text

    def extract_text_from_docx_bytesio(self, docx_bytesio):
        text = ""
        try:
            text = docx2txt.process(docx_bytesio)
        except Exception as e:
            st.warning(f"Error extracting text from DOCX: {e}")
        return text

    def extract_text_from_txt_bytesio(self, txt_bytesio):
        text = ""
        try:
            with txt_bytesio:
                text = txt_bytesio.read().decode('utf-8')
        except Exception as e:
            st.warning(f"Error reading text file: {e}")
        return text

    def extract_text_from_image_bytesio(self, image_bytesio):
        # Use OCR for image-based or scanned resumes
        text = ""
        try:
            image = Image.open(image_bytesio)
            text = pytesseract.image_to_string(image)
        except Exception as e:
            st.warning(f"Error extracting text from image: {e}")
        return text

    def summarize_cv_information(self, cv_information):
        template = """
        Question: {question}
        Answer: Let's think step by step.
        """
        prompt = PromptTemplate.from_template(template)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        question = f"Given this CV information:\n {cv_information}\n\n, thoroughly analyse it and give a summary?"
        return llm_chain.run(question)
        
    def extract_keywords(self, text):
        # Initialize Rake for keyword extraction
        rake_object = Rake()

        # Extract keywords from the text
        rake_object.extract_keywords_from_text(text)

        # Get the ranked phrases
        ranked_phrases = rake_object.get_ranked_phrases()

        # Filter out stopwords
        stop_words = set(stopwords.words('english'))
        filtered_phrases = [phrase for phrase in ranked_phrases if phrase.lower() not in stop_words]

        # Remove symbols, numbers, and non-alphabetic characters
        clean_keywords = []
        for phrase in filtered_phrases:
            clean_phrase = re.sub(r'[^a-zA-Z\s]', '', phrase)  # Keep only alphabetic characters and spaces
            clean_phrase = clean_phrase.strip()  # Remove leading and trailing spaces
            if clean_phrase:
                clean_keywords.append(clean_phrase.lower())  # Convert to lowercase
         
        # Remove duplicates while preserving order
        unique_words = ''
        seen = set()
        for word in clean_keywords:
            if word not in seen:
                unique_words += word + ' '
                seen.add(word)

        return unique_words
   
    
    def generate_recommendations(self, cv_information, job_description):
        template = """
        Question: {question}
        Answer: Let's think step by step.
        """
        prompt = PromptTemplate.from_template(template)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        question = f"Given this CV information:\n {cv_information}\n\n, thoroughly analyse it and suggest recommendations on how to tailor it to match this job listing: {job_description} ?"
        return llm_chain.run(question)
    
    # Function to generate word cloud
    def generate_wordcloud(self, text):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig)

def main():
    st.set_page_config(page_title="Resume Nebula Navigator", page_icon=":bar_chart:", layout="wide")
    st.title("Resume Nebula Navigator")
    cv_analyzer = CVAnalyzer()

    # File upload widgets
    st.sidebar.title("Upload Files")
    st.sidebar.write("Upload your CV (PDF, DOCX, TXT, PNG, JPG, JPEG)")
    uploaded_cv = st.sidebar.file_uploader(label="CV", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"], key="uploaded_cv")

    st.sidebar.write("Upload your Job Description (TXT)")
    uploaded_job_description = st.sidebar.file_uploader(label="Job Description", type=["txt"], key="uploaded_job_description")

    # Perform actions based on uploaded files
    if uploaded_cv is not None and uploaded_job_description is not None:
        st.sidebar.write("Files uploaded successfully!")

        # Extract CV information
        cv_information = cv_analyzer.extract_cv_information(uploaded_cv)
        if cv_information:
            st.title("CV Information:")
            st.write(f"="*90)
            st.write(cv_information)
            st.write("Summarized CV Information:")
            st.write(f"="*90)
            summarized_cv_info = cv_analyzer.summarize_cv_information(cv_information)
            st.write(summarized_cv_info)

        else:
            st.error("Error extracting CV information. Please upload a valid file.")

        # Extract keywords from job CV
        cv_keywords = cv_analyzer.extract_keywords(cv_information)
        st.title("Curriculum Vitae WordCloud")
        cv_analyzer.generate_wordcloud(cv_keywords)

        # Extract keywords from job description
        job_description_text = uploaded_job_description.read().decode("utf-8")  # Decode bytes-like object to string
        keywords = cv_analyzer.extract_keywords(job_description_text)

        if keywords:
            st.title("Top Keywords from Job Description:")
            st.write(f"="*90)
            st.write(keywords)
        else:
            st.error("Error extracting keywords. Please upload a valid job description file.")
        
        st.title("Job Description WordCloud")
        cv_analyzer.generate_wordcloud(keywords)

        # Generate recommendations based on CV and Job Description
        st.title("Recommendations")
        st.write("Generating Recommendations:")
        st.write(f"="*90)
        recommendations = cv_analyzer.generate_recommendations(cv_information, job_description_text)
        st.write(recommendations)

if __name__ == "__main__":
    main()
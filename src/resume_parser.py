import fitz  # PyMuPDF
import docx
import os
from typing import List, Dict


class ResumeParser:
    """
    A class to parse resumes from PDF and DOCX formats and extract text content.
    """

    def __init__(self, resume_dir: str = "data/resumes/"):
        """
        Initialize the ResumeParser with the directory containing resumes.

        Args:
            resume_dir (str): Path to the directory containing resume files.
        """
        self.resume_dir = resume_dir

    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: Extracted text from the PDF.
        """
        text = ""
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {e}")
        return text

    def extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text from a DOCX file.

        Args:
            file_path (str): Path to the DOCX file.

        Returns:
            str: Extracted text from the DOCX file.
        """
        text = ""
        try:
            doc = docx.Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error extracting text from DOCX {file_path}: {e}")
        return text

    def parse_resume(self, file_path: str) -> str:
        """
        Parse a resume file (PDF or DOCX) and extract text.

        Args:
            file_path (str): Path to the resume file.

        Returns:
            str: Extracted text from the resume.
        """
        if file_path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            return self.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def get_all_resumes(self) -> Dict[str, str]:
        """
        Get all resumes from the resume directory and extract their text.

        Returns:
            Dict[str, str]: Dictionary with filename as key and extracted text as value.
        """
        resumes = {}
        if not os.path.exists(self.resume_dir):
            print(f"Resume directory {self.resume_dir} does not exist.")
            return resumes

        for filename in os.listdir(self.resume_dir):
            if filename.lower().endswith(('.pdf', '.docx')):
                file_path = os.path.join(self.resume_dir, filename)
                try:
                    text = self.parse_resume(file_path)
                    resumes[filename] = text
                except Exception as e:
                    print(f"Error parsing resume {filename}: {e}")

        return resumes


if __name__ == "__main__":
    # Example usage
    parser = ResumeParser()
    resumes = parser.get_all_resumes()
    print(f"Parsed {len(resumes)} resumes")
    for filename, text in resumes.items():
        print(f"{filename}: {len(text)} characters")
# pdf_tools.py

from pathlib import Path
from typing import Union

from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError


class PDFConverter:
    @staticmethod
    def validate_pdf(file_path: Union[str, Path]) -> bool:
        """
        Validates if the given file is a valid PDF.

        :param file_path: Path to the PDF file
        :return: True if valid PDF, False otherwise
        :raises FileNotFoundError: if the file doesn't exist
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                if b"\x00" in content:
                    return False  # PDF files should not contain null bytes
            return True
        except PDFSyntaxError:
            return False
        except Exception as e:
            raise RuntimeError(f"Failed to validate PDF: {e}")

    @staticmethod
    def pdf_to_text(file_path: Union[str, Path]) -> str:
        """
        Extracts plain text from a PDF file.

        :param file_path: Path to the PDF file
        :return: Extracted text
        :raises FileNotFoundError: if the file doesn't exist
        :raises ValueError: if the PDF is malformed
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            text = extract_text(str(file_path))
            return text.strip()
        except PDFSyntaxError as e:
            raise ValueError(f"Error parsing PDF file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {e}")

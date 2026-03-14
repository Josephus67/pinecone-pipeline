import os
from pypdf import PdfReader

def convert_pdfs_to_text(directory):
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(directory, txt_filename)
            
            print(f"Converting '{filename}' to text...")

            try:
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                
                print(f"Saved text to '{txt_filename}'")

            except Exception as e:
                print(f"Failed to convert '{filename}': {e}")

if __name__ == "__main__":
    # convert PDFs in the 'static' folder
    convert_pdfs_to_text("static")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

# file_name = "job.txt"
from docx import Document


def load_data(file_name):
    doc = Document(file_name)
    data = []
    for table in doc.tables:
        print("=== TABLE ===")
        for row in table.rows:
            data.append([cell.text for cell in row.cells])
    final = []
    for dt in data:
        dt = [d.replace("\n", "") for d in dt]
        final.append(dt)
    return final

# def load_data(file_name):
#     with open(file_name, "r", encoding="utf-8") as f:
#         text_data = f.read()

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=80,       
#             chunk_overlap=30,     
#             length_function=len,  
#         )

#         chunks = text_splitter.split_text(text_data)

#         return chunks

# if __name__ == "__main__":
    
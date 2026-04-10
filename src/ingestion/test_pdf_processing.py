from pdf_processor import PDFProcessor

file_processor = PDFProcessor()
output = file_processor.extract_markdown("/Users/mrtgaliakberov/Projects/Altumatim/data/FAIA241255.pdf")
with open("output.md", "w") as f:
    for i, page in enumerate(output):
        f.write(f"Page {i+1}:\n")
        f.write(page["text"])

        f.write("============================ /n/n")

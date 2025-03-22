from pathlib import Path
from text_summary import *

# Define the folder path
folder_path = Path("ukraine_filtered_reports")

# Get a list of all .txt files
txt_files = list(folder_path.glob("*.txt"))

documents = [Path(file).read_text(encoding="utf-8") for file in folder_path.glob("*.txt")]
for i in documents:
    print(len(i))

summarizer = DocumentSummarizer(documents)
# print(summarizer.tfidf_summary("war"))
# print(summarizer.bart_summary())
print(summarizer.distill_summary())
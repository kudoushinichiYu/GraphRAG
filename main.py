from Data.HAPI.hapi_class import HapiClass
import Data.Visualization.visualization as visual
import Card_Generation.card_generation as cg
from Data.RELIEFWEB.reliefweb_class import ReliefWebClass
from Card_Generation.HAPI_recommender import HAPIRecommender
from Text_Summary.text_summary import DocumentSummarizer
from Text_Summary.helper import TextRetriever
from Data.HAPI_visual import HAPIVisualizer
from huggingface_hub import login
import matplotlib.pyplot as plt
import tempfile
import os
import re

# Example

# Initialize country
country = "UKR"
date_range = ("2025-02-21", "2025-03-21")

# country_data = HapiClass(country)
# plot1 = visual.plot_humanitarian_needs(country_data)
# plot2 = visual.plot_conflict_events(country_data)
# plot3 = visual.plot_funding(country_data)
# plots = [plot3]

# Find documents
# Find documents
reliefweb_articles = ReliefWebClass(country, date_range).get_articles()
print("Get documents complete")
output_folder = "reliefweb_articles" + country
os.makedirs(output_folder, exist_ok=True)

def sanitize_filename(filename, max_length=100):
    # Remove any characters that are not valid in file names
    sanitized = re.sub(r'[\\/*?:"<>|.]', "_", filename)
    # Truncate the filename if it's too long
    return sanitized[:max_length]

for i, article in enumerate(reliefweb_articles, start=1):
    title = article.get("title", f"Article_{i}")  
    content = article.get("body", "No Content")  

    safe_title = sanitize_filename(title)
    file_name = f"{safe_title}.txt" if safe_title else f"Article_{i}.txt"
    file_path = os.path.join(output_folder, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")  
        f.write(content) 
    print(i)

# print(f"All articles saved in folder: {os.path.abspath(output_folder)}")
path = './'+output_folder
idx_path = 'faiss_index_cosine_'+country+'.bin'
files_path = country+'.npy'
retriever = TextRetriever(folder_path=path,index_path=idx_path,filenames_path=files_path)
summarizer = DocumentSummarizer(retriver_li=retriever)
query = "Sudanese Civil War"
summary_result = summarizer.llama_summary(query, top_k=30)


print(summary_result)

# Recommend Graphs
recommended = HAPIRecommender(summary_result).generate_recommendation()

print(recommended)

graphs = HAPIVisualizer(country, recommended, 1).generate_plots()
for fig in graphs:
    plt.show()

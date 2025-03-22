from Data.HAPI.hapi_class import HapiClass
import Data.Visualization.visualization as visual
import Card_Generation.card_generation as cg
from Data.RELIEFWEB.reliefweb_class import ReliefWebClass
from Card_Generation.HAPI_recommender import HAPIRecommender
from Text_Summary.text_summary import DocumentSummarizer
from Data.HAPI_visual import HAPIVisualizer
from huggingface_hub import login
import matplotlib.pyplot as plt

# api_token = ""
# login(api_token)


# Example

# Initialize country
country = "UKR"
date_range = ("2022-02-24", "2022-03-24")

# country_data = HapiClass(country)
# plot1 = visual.plot_humanitarian_needs(country_data)
# plot2 = visual.plot_conflict_events(country_data)
# plot3 = visual.plot_funding(country_data)
# plots = [plot3]

# Find documents
# reliefweb_articles = ReliefWebClass(country, date_range).get_articles()[:10]
# print("Get documents complete")

# # Summarize
# summary = DocumentSummarizer(reliefweb_articles).llama_summary()

# print(summary)

# # Recommend Graphs
# recommended = HAPIRecommender(summary).generate_recommendation()
recommended = ['Humanitarian Need', 'Refugee', 'Poverty Rate', 'Funding', 'National Risk', 'Food Security',
                'Conflict Event', 'Population', 'Operational Presence', 'Food Price', 'Returnee']
print(recommended)

graphs = HAPIVisualizer(country, recommended, 3).generate_plots()
for fig in graphs:
    plt.show()
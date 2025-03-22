from Data.HAPI.hapi_class import HapiClass
import Data.Visualization.visualization as visual
import Card_Generation.card_generation as cg
from Data.RELIEFWEB.reliefweb_class import ReliefWebClass
from Card_Generation.HAPI_recommender import HAPIRecommender
from Text_Summary.text_summary import DocumentSummarizer


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
reliefweb_articles = ReliefWebClass(country, date_range).get_articles()[:10]
print("Get documents complete")

# Summarize
summary = DocumentSummarizer(reliefweb_articles).llama_summary()

print(summary)

# Recommend Graphs
recommended = HAPIRecommender(summary).generate_recommendation()

print(recommended)


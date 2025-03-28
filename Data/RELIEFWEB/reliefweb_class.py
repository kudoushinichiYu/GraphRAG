from .helper import *
from langdetect import detect


class ReliefWebClass:
    APP_IDENTIFIER = "DESKTOP-06R1BSH"
    LIMIT = 500

    def __init__(self, country=None, date_range=None):
        self.event_data = None
        self.country = country
        self.report_data = None
        
        if not date_range:
            self.date_range = None
        elif not isinstance(date_range, dict):
            if len(date_range) != 2:
                raise ValueError('date_range not valid')
            self.date_range = {'from': date_range[0], 'to': date_range[1]}
        else:
            self.date_range = date_range
            
        self.get_report_data()
        self.report_text = []

    def get_report_data(self):
        """
        Retrieves the reports
        """
        theme = "reports"
        base_url = construct_url(self.APP_IDENTIFIER, theme, self.LIMIT, self.country, self.date_range)
        self.report_data = fetch_data(base_url)
        def is_english(text):
            try:
                return detect(text) == "en"
            except:
                return False

        self.report_data = self.report_data[self.report_data["fields_title"].apply(is_english)]

    def get_passage(self, num):
        url = self.report_data['href'].iloc[num]
        text = read_report(url)
        return text
    
    def get_articles(self):
        articles = []
        for index, row in self.report_data.iterrows():
            try:
                url = row['href']
                text = read_report(url)
                title = row['fields_title']
                articles.append({'title': title, 'body': text})
            except Exception as e:
                print(f"Failed to save report {row['fields_title']}: {e}")
        return articles

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from selenium.webdriver.firefox.options import Options
import time
import helper



class GdacsClass:
    def __init__(self, country):
        self.data = None
        self.country = country

    def get_data(self, start_date=None, event_type=None, get_text=False):
        """
        This function will get pass data for given class country
        :param start_date: should be a string of the form "yyyy-mm-dd"
        :param event_type: should be one of the event types in ['drought', 'earthquake', 'tropical cyclone', 'flood',
        'volcano', 'forest fire']
        :return: None
        """

        # Hide Webpage
        options = Options()
        options.headless = True

        # Web Scraping
        driver = webdriver.Firefox(options=options)
        url = 'https://www.gdacs.org/Alerts/default.aspx'
        driver.get(url)

        # Search for specific country
        input_field = driver.find_element(By.ID, "inputCountry")
        input_field.send_keys(self.country)

        # Add a start date
        if start_date:
            input_field = driver.find_element(By.ID, "inputDateFrom")
            input_field.clear()
            input_field.send_keys(start_date)

        # Add event filter
        if event_type:
            if event_type == 'tropical cyclone':
                input_field = driver.find_element(By.ID, "inputChTc")
            if event_type == 'flood':
                input_field = driver.find_element(By.ID, "inputChFl")
            if event_type == 'volcano':
                input_field = driver.find_element(By.ID, "inputChVo")
            if event_type == 'forest fire':
                input_field = driver.find_element(By.ID, "inputChFf")
            if event_type == 'drought':
                input_field = driver.find_element(By.ID, "inputChDr")
            if event_type == 'earthquake':
                input_field = driver.find_element(By.ID, "inputChEq")
            input_field.click()

        # Apply search
        button = driver.find_element(By.ID, "btnsearch")
        button.click()


        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'clickable-row')))

        # Find links
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        driver.quit()

        # Find all links for specific information for each event
        rows = soup.find_all('tr', class_='clickable-row')
        link_li = set([])
        for row in rows:
            # Find all <a> tags within the row
            links = row.find_all('a')
            for link in links:
                # Extract the href attribute
                href = link.get('href')
                link_li.add(href)

        self.data = []
        for link in link_li:
            # event_type is a string, statistics is a dictionary, and description is a list of strings
            event_type, statistics, description = helper.scrap(link, get_text)
            self.data.append((event_type, statistics, description))

a = GdacsClass('United States')
a.get_data('2023-01-01', 'tropical cyclone')
print(a.data)
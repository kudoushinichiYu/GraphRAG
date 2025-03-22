import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
import re


def scrap(link, get_text=False):
    """
    The link provided should be a particular event page
    :param link: a string of url
    :return: event description and summary
    """
    # Web Scraping
    # Hide web page
    option = Options()
    option.headless = True
    driver = webdriver.Firefox(options=option)

    # Get html file
    driver.get(link)
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'slick-track')))
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    driver.quit()

    # Get Event Type
    type_summary = soup.find('p', class_='p_summary').get_text()
    event_type = get_event_type(type_summary)

    # Get Detail Statistics as a dictionary (name: value)
    summary = soup.find('table', class_="summary")
    summary_table_name = summary.find_all('td')
    dict_value = {}
    i = 0
    last = None
    for cell in summary_table_name:
        val = cell.get_text()
        if i % 2 == 1:
            dict_value[last] = val
        i += 1
        last = val

    # Get description
    description_text = []
    if get_text:
        description_table = soup.find('div', class_='slick-track')
        description = description_table.find_all('span', class_='news_text')
        for cell in description:
            description_text.append(cell.get_text())

    return event_type, dict_value, description_text



def get_event_type(text):
    """
    Given summary, get event type
    :param text: a string representing the event summary
    :return: event type as a string
    """
    pattern_drought = 'The drought'
    pattern_earthquake = 'This earthquake'
    pattern_tropical_cyclone = 'Tropical Cyclone'
    pattern_flood = 'This flood'
    pattern_volcano = 'The(.*?)volcano'
    pattern_fire = 'This forest fire'
    if re.match(pattern_drought, text):
        return 'drought'
    if re.match(pattern_earthquake, text):
        return 'earthquake'
    if re.match(pattern_tropical_cyclone, text):
        return 'tropical cyclone'
    if re.match(pattern_flood, text):
        return 'flood'
    if re.search(pattern_volcano, text):
        return 'volcano'
    if re.match(pattern_fire, text):
        return 'forest fire'
    raise RuntimeError('Event type not recognized')


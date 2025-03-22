from urllib import request
import pandas as pd


def fetch_data(base_url,limit=1000):
    """
    Fetch data from the provided base_url with pagination support.

    Args:
    - base_url (str): The base URL endpoint to fetch data from.
    - limit (int): The number of records to fetch per request.

    Returns:
    - list: A list of fetched results.
    """

    idx = 0

    while True:
        offset = idx * limit
        url = f"{base_url}&offset={offset}&limit={limit}"

        with request.urlopen(url) as response:
            print(f"Getting results {offset} to {offset+limit-1}")
            csv_response = pd.read_csv(response)

            if idx == 0:
                results = csv_response
            else:
                results = pd.concat([results,csv_response])

            # If the returned results are less than the limit,
            # it's the last page
            if len(csv_response) < limit:
                break

            if idx > 10:
                break

        idx += 1

    return results


def construct_url(APP_IDENTIFIER, THEME, LOCATION):
    '''
    Construct an url for data requests
    :param APP_IDENTIFIER: string
    :param THEME: string
    :param LOCATION: string
    :return: constructed url as string
    '''
    BASE_URL = (
        f"https://hapi.humdata.org/api/v2/{THEME}?"
        f"output_format=csv"
        f"&location_code={LOCATION}"
        f"&app_identifier={APP_IDENTIFIER}"
    )
    return BASE_URL


def construct_url_all(APP_IDENTIFIER, THEME):
    '''
    Construct an url for data requests
    :param APP_IDENTIFIER: string
    :param THEME: string
    :param LOCATION: string
    :return: constructed url as string
    '''
    BASE_URL = (
        f"https://hapi.humdata.org/api/v2/{THEME}?"
        f"output_format=csv"
        f"&app_identifier={APP_IDENTIFIER}"
    )
    return BASE_URL



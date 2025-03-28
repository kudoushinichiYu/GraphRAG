from helper import *


APP_IDENTIFIER = {
  "U2ltb24gV2FuZzpzaHVyZW4wNDE5QDE2My5jb20="
}
THEME = "coordination-context/conflict-event"
LOCATION = "AFG"
BASE_URL = construct_url(APP_IDENTIFIER, THEME, LOCATION)
LIMIT = 1000

results = fetch_data(BASE_URL, LIMIT)
print(results.columns)
print(results.info())
print(results)
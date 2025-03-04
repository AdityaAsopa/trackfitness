import os
import requests
from pathlib import Path
import json
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webbot import Browser

# Define the login URL and the URL of the page with the table
login_url = 'https://runalyze.com/login'
data_url =  'https://runalyze.com/dashboard'

# Load login credentials from .json
with open('env.json') as f:
    config = json.load(f)

USERID = config['RUNALYZE_USER']
PSWD = config['RUNALYZE_PASSWORD']
record_file_path = config['runalyze_record_file_path']

### -------------------------------------------------- ###
# Specify the path to the ChromeDriver executable
service = Service(r"C:\drivers\chromedriver-win64\chromedriver.exe")

# Initialize the WebDriver with the Service object
driver = webdriver.Chrome(service=service)

# Open the login page
driver.get(login_url)

# Find the username and password fields
username = driver.find_element("name", "_username")
password = driver.find_element("name", "_password")

# Send the login information
username.send_keys(USERID)
password.send_keys(PSWD)

# Submit the form
driver.find_element("name", "submit").click()

# Open the data page
driver.get(data_url)

# Get the page source
page_source = driver.page_source

# Close the browser
driver.quit()

### -------------------------------------------------- ###
# parse pagesource using beautifulsoup
soup = BeautifulSoup(page_source, 'html.parser')

# find object of the class 'panel' with id = 'panel-1753156'
fitness_table = soup.find('div', {'class': 'panel', 'id': 'panel-1753156'})
# get the container "panel-content" from the fitness_table object
panel_content = fitness_table.find('div', {'class': 'panel-content'})
# get the table from the panel_content object, <table class="fullwidth nomargin">
table = panel_content.find('table', {'class': 'fullwidth nomargin'})

# extract rows into a list, seaprating by <td> and <tr>
rows = table.find_all('tr')
data = []
for row in rows:
    cols = row.find_all('td')
    # strip() is used to remove starting and trailing, to remove characters like '%' use replace('%','')
    cols = [ele.text.strip().replace('%', '') for ele in cols]
    data.append([ele for ele in cols if ele])

### -------------------------------------------------- ###
df = pd.DataFrame(data, columns=['label', 'value'], )
# values in value column are all numeric but either have a unit or % attached or use comma as decimal separator
# clean the values
df['value'] = df['value'].str.replace(',', '.')
df['value'] = df['value'].str.replace(' %', '')
# remove trailing spaces
df['value'] = df['value'].str.strip()
# remove '±' prefix
df['value'] = df['value'].str.replace('±', '')

print(df)
# convert the values to numeric
df['value'] = pd.to_numeric(df['value'])

# convert the df into a single row with today's data as the first element\
# and the rest of the data as the next elements
df = df.T
df.columns = df.iloc[0]
df = df.drop('label')
df = df.reset_index(drop=True)
# add today's date
df.insert(0, 'date', pd.to_datetime('today').date())

### -------------------------------------------------- ###


# read the existing data
dfsave = pd.read_csv(record_file_path, delimiter='\t') if Path(record_file_path).exists() else pd.DataFrame()
print(dfsave)
print(df)
# append the new data
dfsave = pd.concat([dfsave, df], ignore_index=True)
# save the file
dfsave.to_csv(record_file_path, sep='\t', index=False)

print("Today's Runalyze data saved successfully")
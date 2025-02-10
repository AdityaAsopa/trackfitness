import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup


class RunalyzeScraper:
    """A scraper to log in to Runalyze, retrieve a fitness table, and return its page source."""

    def __init__(self, login_url: str, data_url: str, username: str, password: str, driver_path: str):
        self.login_url = login_url
        self.data_url = data_url
        self.username = username
        self.password = password
        self.driver_path = driver_path

    def _init_driver(self) -> webdriver.Chrome:
        """Initialize and return a Chrome WebDriver."""
        service = Service(self.driver_path)
        return webdriver.Chrome(service=service)

    def get_page_source(self) -> str:
        """
        Log in to Runalyze and return the HTML source of the data page.
        The method uses Selenium to perform the login and navigation.
        """
        driver = self._init_driver()
        try:
            driver.get(self.login_url)
            driver.find_element("name", "_username").send_keys(self.username)
            driver.find_element("name", "_password").send_keys(self.password)
            driver.find_element("name", "submit").click()
            driver.get(self.data_url)
            return driver.page_source
        finally:
            driver.quit()

    @staticmethod
    def parse_fitness_table(page_source: str) -> pd.DataFrame:
        """
        Parse the Runalyze fitness table from the page source and return it as a DataFrame.
        
        The table is expected to be within a <div> of class "panel" with a specific id.
        """
        soup = BeautifulSoup(page_source, 'html.parser')
        panel = soup.find('div', {'class': 'panel', 'id': 'panel-1753156'})
        if panel is None:
            raise ValueError("Could not find the fitness panel with id 'panel-1753156'.")
        panel_content = panel.find('div', {'class': 'panel-content'})
        table = panel_content.find('table', {'class': 'fullwidth nomargin'})
        rows = table.find_all('tr')

        # Extract and clean cell text from each row
        data = []
        for row in rows:
            cols = row.find_all('td')
            if cols:
                cleaned_cols = [col.get_text(strip=True).replace('%', '') for col in cols]
                data.append(cleaned_cols)

        if not data:
            raise ValueError("No table data found.")
        return pd.DataFrame(data, columns=['label', 'value'])

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and reshape the DataFrame:
          - Replace commas with dots for decimals.
          - Remove percentage signs and extra whitespace.
          - Convert values to numeric.
          - Transpose the DataFrame so that each label becomes a column.
          - Insert today's date.
        """
        df['value'] = (
            df['value']
            .str.replace(',', '.', regex=False)
            .str.replace(' %', '', regex=False)
            .str.strip()
            .str.replace('±', '', regex=False)
        )
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Transpose to convert rows to columns (first row becomes column headers)
        df_transposed = df.T
        df_transposed.columns = df_transposed.iloc[0]
        df_transposed = df_transposed.drop('label').reset_index(drop=True)
        df_transposed.insert(0, 'date', pd.to_datetime('today').date())
        return df_transposed


def load_config(config_file: str) -> dict:
    """Load configuration data (e.g., login credentials and file paths) from a JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def append_data(new_data: pd.DataFrame, record_file_path: str) -> None:
    """
    Append new data to the existing records file.
    
    If the file does not exist, it is created.
    """
    record_path = Path(record_file_path)
    if record_path.exists():
        df_existing = pd.read_csv(record_file_path, delimiter='\t')
    else:
        df_existing = pd.DataFrame()

    df_combined = pd.concat([df_existing, new_data], ignore_index=True)
    df_combined.to_csv(record_file_path, sep=',', index=False)


def main():
    # Load configuration
    config = load_config('runalyze_login.json')
    login_url = 'https://runalyze.com/login'
    data_url = 'https://runalyze.com/dashboard'
    driver_path = r"C:\drivers\chromedriver.exe"  # Use raw string for Windows paths

    # Initialize scraper and fetch page source
    scraper = RunalyzeScraper(
        login_url=login_url,
        data_url=data_url,
        username=config['RUNALYZE_USER'],
        password=config['RUNALYZE_PASSWORD'],
        driver_path=driver_path
    )
    page_source = scraper.get_page_source()

    # Parse and clean the fitness table data
    raw_df = scraper.parse_fitness_table(page_source)
    clean_df = scraper.clean_dataframe(raw_df)

    # Append the new data to the record file
    append_data(clean_df, config['runalyze_record_file_path'])
    print("Today's Runalyze data saved successfully.")


if __name__ == '__main__':
    main()

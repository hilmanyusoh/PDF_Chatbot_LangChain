import requests
from bs4 import BeautifulSoup

url = 'https://sunnah.com/bukhari/8'
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Extract a specific part of the page, such as the titles or hadith texts
# Example: Extract all hadith titles
hadith_titles = soup.find_all('h3')  # You may need to adjust the tag based on the actual HTML structure

for title in hadith_titles:
    print(title.get_text())

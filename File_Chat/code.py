import requests
from bs4 import BeautifulSoup
import pandas as pd

# Target URL
url = "https://sunnah.com/bukhari/8"

# Send a GET request to the webpage
response = requests.get(url)
response.raise_for_status()  # Check for any request errors

# Parse HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find all Hadith sections on the page
hadiths_data = []
hadith_sections = soup.find_all("div", class_="actualHadithContainer")

for section in hadith_sections:
    # Hadith number
    hadith_number = section.find("span", class_="hadith_number").text.strip() if section.find("span", class_="hadith_number") else "N/A"
    
    # Hadith topic
    hadith_topic = section.find("div", class_="hadith_reference").text.strip() if section.find("div", class_="hadith_reference") else "N/A"
    
    # Hadith description
    hadith_description = section.find("div", class_="englishcontainer").text.strip() if section.find("div", class_="englishcontainer") else "N/A"
    
    # Hadith text (only in "Mu" or "prayer" sections)
    hadith_text = section.find("div", class_="text_details").text.strip() if section.find("div", class_="text_details") else "N/A"
    
    # Add extracted data to list
    hadiths_data.append({
        "Hadith Number": hadith_number,
        "Hadith Topic": hadith_topic,
        "Hadith Description": hadith_description,
        "Hadith Text": hadith_text
    })

# Save the data to a CSV file
df = pd.DataFrame(hadiths_data)
df.to_csv("hadith_data.csv", index=False, encoding="utf-8")
print("Data saved to hadith_data.csv")

import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_hadith(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    hadiths_data = []
    hadith_sections = soup.find_all("div", class_="actualHadithContainer")

    for section in hadith_sections:
        hadith_number = section.find("span", class_="hadith_number").text.strip() if section.find("span", class_="hadith_number") else "N/A"
        hadith_topic = section.find("div", class_="hadith_reference").text.strip() if section.find("div", class_="hadith_reference") else "N/A"
        hadith_description = section.find("div", class_="englishcontainer").text.strip() if section.find("div", class_="englishcontainer") else "N/A"
        hadith_text = section.find("div", class_="text_details").text.strip() if section.find("div", class_="text_details") else "N/A"

        hadiths_data.append({
            "Hadith Number": hadith_number,
            "Hadith Topic": hadith_topic,
            "Hadith Description": hadith_description,
            "Hadith Text": hadith_text
        })

    df = pd.DataFrame(hadiths_data)
    df.to_csv("hadith_data.csv", index=False, encoding="utf-8")
    print("Data saved to hadith_data.csv")

if __name__ == "__main__":
    url = "https://sunnah.com/bukhari/8"
    scrape_hadith(url)

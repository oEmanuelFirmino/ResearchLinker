import requests
from bs4 import BeautifulSoup
import json


class WebScraper:
    def __init__(self, start_url, max_pages=25):
        self.start_url = start_url
        self.max_pages = max_pages
        self.data = []

    def scrape_page(self, url):
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Erro ao acessar {url}: {response.status_code}")
            return None
        return BeautifulSoup(response.text, "html.parser")

    def extract_data(self, soup):
        for item in soup.select("a.record_title"):
            titulo = (
                item.select_one("span.title").text.strip()
                if item.select_one("span.title")
                else ""
            )
            link = item["href"] if item.has_attr("href") else ""

            container = item.find_next("div", class_="recordContents")
            autores = (
                container.select_one("span.author").text.strip()
                if container.select_one("span.author")
                else ""
            )
            evento = (
                container.select_one("div.archive_serie").text.strip()
                if container.select_one("div.archive_serie")
                else ""
            )
            data_publicacao = (
                container.select_one("span.list_record_date").text.strip()
                if container.select_one("span.list_record_date")
                else ""
            )

            self.data.append(
                {
                    "titulo": titulo,
                    "autores": autores,
                    "evento": evento,
                    "data_publicacao": data_publicacao,
                    "link": link,
                }
            )

    def get_next_page(self, url, current_page):
        next_page = current_page + 1
        return (
            url.replace(f"searchPage={current_page}", f"searchPage={next_page}")
            if next_page <= self.max_pages
            else None
        )

    def scrape_website(self):
        url = f"{self.start_url}&searchPage=1"
        current_page = 1

        while url:
            print(f"Raspando: {url}")
            soup = self.scrape_page(url)
            if not soup:
                break

            self.extract_data(soup)
            url = self.get_next_page(url, current_page)
            current_page += 1

        return self.data

    def save_to_file(self, filename="data.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        print(f"Dados salvos em {filename}")

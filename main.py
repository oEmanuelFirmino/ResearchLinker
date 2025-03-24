from dotenv import load_dotenv
from src.article_graph_analyzer import ArticleGraphAnalyzer
from src.article_vetorizer import ArticleVectorizer
from src.web_scraper import WebScraper
import os

load_dotenv()


def main():
    scraper = WebScraper(os.getenv("scrapping_url"))
    scraped_data = scraper.scrape_website()
    scraper.save_to_file(scraped_data)

    vectorizer = ArticleVectorizer("data.json", "vectorized_data.json")
    vectorizer.vectorize_data()
    vectorizer.save_vectorized_data()

    analyzer = ArticleGraphAnalyzer("vectorized_data.json")
    analyzer.build_graph()
    analyzer.analyze_graph()
    analyzer.save_results()


if __name__ == "__main__":
    main()

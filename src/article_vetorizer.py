import json
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import csv
import uuid
from sentence_transformers import SentenceTransformer


class ArticleVectorizer:
    def __init__(self, input_path, output_path, model_name="all-MiniLM-L6-v2"):
        self.input_path = input_path
        self.output_path = output_path
        self.model = SentenceTransformer(model_name)
        self.data = self.load_data()
        self.processed_data = []

    def load_data(self):
        with open(self.input_path, encoding="utf-8") as f:
            return json.load(f)

    def vectorize_data(self):
        for i, entry in enumerate(self.data):
            title_embedding = self.model.encode(entry["titulo"]).tolist()
            authors_embedding = self.model.encode(entry["autores"]).tolist()
            event_embedding = self.model.encode(entry["evento"]).tolist()

            self.processed_data.append(
                {
                    "id": i,
                    "title": {"text": entry["titulo"], "embedding": title_embedding},
                    "authors": {
                        "text": entry["autores"],
                        "embedding": authors_embedding,
                    },
                    "event": {"text": entry["evento"], "embedding": event_embedding},
                    "metadata": {
                        "publication_date": entry["data_publicacao"],
                        "link": entry["link"],
                    },
                }
            )

    def save_vectorized_data(self):
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.processed_data, f, indent=4, ensure_ascii=False)
        print(f"Dados vetorizados salvos em {self.output_path}")

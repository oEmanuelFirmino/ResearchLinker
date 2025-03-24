import json
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import csv
import uuid


class ArticleSimilarityAnalyzer:
    def __init__(self, json_path, threshold=0.65):
        self.json_path = json_path
        self.threshold = threshold
        self.data = self.load_data()
        self.titles, self.title_embeddings = self.extract_embeddings()
        self.similarity_matrix = cosine_similarity(self.title_embeddings)
        self.G = nx.Graph()
        self.similarity_data = []

    def load_data(self):
        with open(self.json_path, encoding="utf-8") as f:
            return json.load(f)

    def extract_embeddings(self):
        titles = [entry["title"]["text"] for entry in self.data]
        embeddings = np.array([entry["title"]["embedding"] for entry in self.data])
        return titles, embeddings

    def build_graph(self):
        for entry in self.data:
            title = entry["title"]["text"]
            authors = entry["authors"]["text"]
            event = entry["event"]["text"]
            publication_date = entry["metadata"]["publication_date"]
            link = entry["metadata"]["link"]

            self.G.add_node(
                title,
                text=title,
                authors=authors,
                event=event,
                publication_date=publication_date,
                link=link,
                theme_similarity=False,
                unique_name=str(uuid.uuid4()),
            )

        for i in range(len(self.titles)):
            for j in range(i + 1, len(self.titles)):
                similarity = self.similarity_matrix[i, j]
                if similarity >= self.threshold:
                    self.G.add_edge(
                        self.titles[i], self.titles[j], weight=float(similarity)
                    )
                    self.G.nodes[self.titles[i]]["theme_similarity"] = True
                    self.G.nodes[self.titles[j]]["theme_similarity"] = True
                    self.similarity_data.append(
                        {
                            "title_1": self.titles[i],
                            "title_2": self.titles[j],
                            "similarity": float(similarity),
                        }
                    )
                else:
                    self.G.nodes[self.titles[i]]["unique_name"] = str(uuid.uuid4())
                    self.G.nodes[self.titles[j]]["unique_name"] = str(uuid.uuid4())
                    self.similarity_data.append(
                        {
                            "title_1": self.titles[i],
                            "title_2": self.titles[j],
                            "similarity": float(similarity),
                        }
                    )

    def save_results(
        self,
        json_path="title_similarity.json",
        graphml_path="title_similarity.graphml",
        csv_path="title_similarity.csv",
    ):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.similarity_data, f, indent=4, ensure_ascii=False)

        nx.write_graphml(self.G, graphml_path)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Title 1", "Title 2", "Similarity"])
            for entry in self.similarity_data:
                writer.writerow(
                    [entry["title_1"], entry["title_2"], entry["similarity"]]
                )

        print(f"Similaridade salva em '{json_path}'")
        print(f"Grafo salvo em '{graphml_path}'")
        print(f"Similaridade salva em '{csv_path}'")

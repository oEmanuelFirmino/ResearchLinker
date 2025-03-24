import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfVectorizer
import pyLDAvis
from pyvis.network import Network
import os


class ArticleAnalyzer:
    def __init__(self, json_path, csv_path, output_folder="output"):
        self.json_path = json_path
        self.csv_path = csv_path
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self.data = self.load_data_json()
        self.similarity_data = self.load_data_csv()
        self.graph = self.create_graph()

    def load_data_json(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_data_csv(self):
        return pd.read_csv(self.csv_path)

    def create_graph(self):
        G = nx.Graph()
        for entry in self.data:
            G.add_node(entry["title"]["text"])
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)):
                if i % 2 == 0 and j == i + 1:
                    G.add_edge(
                        self.data[i]["title"]["text"], self.data[j]["title"]["text"]
                    )
        return G

    def cluster_articles(self, n_clusters=5, filename="topic_clusters.png"):
        titles = [article["title"]["text"] for article in self.data]
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(titles)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        plt.figure(figsize=(10, 6))
        plt.scatter(
            range(len(titles)), [0] * len(titles), c=kmeans.labels_, cmap="viridis"
        )
        plt.title("Agrupamento de Artigos por Similaridade")
        plt.xlabel("Artigos")
        plt.ylabel("Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, filename))
        plt.close()

    def plot_advanced_centrality(self, filename="advanced_centrality.png"):
        betweenness = nx.betweenness_centrality(self.graph)
        nodes, centralities = zip(
            *sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        plt.figure(figsize=(10, 6))
        plt.bar(nodes, centralities, color="#1f77b4", alpha=0.7)
        plt.title("Top 10 Artigos - Centralidade de Betweenness")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, filename))
        plt.close()

    def plot_interactive_graph(self, filename="interconnected_graph.html"):
        net = Network(height="800px", width="100%", notebook=True)
        for node in self.graph.nodes:
            net.add_node(node)
        for edge in self.graph.edges:
            net.add_edge(edge[0], edge[1])
        net.show(os.path.join(self.output_folder, filename))

    def topic_trends(self, filename="topic_trends.html"):
        titles = [article["title"]["text"] for article in self.data]
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(titles)
        lda = LDA(n_components=5, random_state=42)
        lda.fit(X)
        vis = pyLDAvis.prepare(lda, X, vectorizer)
        vis.save(os.path.join(self.output_folder, filename))

    def publication_insights(self, filename="publication_insights.png"):
        years = [
            article["metadata"]["publication_date"].split("-")[0]
            for article in self.data
        ]
        year_counts = Counter(years)
        plt.figure(figsize=(10, 6))
        plt.bar(year_counts.keys(), year_counts.values(), color="#1f77b4", alpha=0.7)
        plt.xlabel("Ano de Publicação")
        plt.ylabel("Número de Publicações")
        plt.title("Distribuição de Publicações por Ano")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, filename))
        plt.close()
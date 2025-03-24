import json
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import csv
import uuid


class ArticleGraphAnalyzer:
    def __init__(self, json_path, threshold=0.7):
        self.json_path = json_path
        self.threshold = threshold
        self.data = self.load_data()
        self.titles, self.title_embeddings = self.extract_embeddings()
        self.similarity_matrix = cosine_similarity(self.title_embeddings)
        self.G = nx.Graph()
        self.similarity_data = []
        self.degree_centrality = None
        self.shortest_paths = None
        self.components = None
        self.component_similarity = []
        self.jaccard_similarities = []

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
            self.G.add_node(
                title,
                title=title,
                authors=entry["authors"]["text"],
                event=entry["event"]["text"],
                publication_date=entry["metadata"]["publication_date"],
                link=entry["metadata"]["link"],
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

    def analyze_graph(self):
        self.degree_centrality = nx.degree_centrality(self.G)
        self.shortest_paths = dict(nx.all_pairs_shortest_path_length(self.G))
        self.components = list(nx.connected_components(self.G))

        for component in self.components:
            for node_1 in component:
                for node_2 in component:
                    if node_1 != node_2:
                        self.component_similarity.append(
                            {
                                "title_1": node_1,
                                "title_2": node_2,
                                "component_similarity": 1.0,
                            }
                        )

        for i, node_1 in enumerate(self.titles):
            for j, node_2 in enumerate(self.titles):
                if i < j:
                    neighbors_1 = set(self.G.neighbors(node_1))
                    neighbors_2 = set(self.G.neighbors(node_2))
                    jaccard_similarity = (
                        (
                            len(neighbors_1 & neighbors_2)
                            / len(neighbors_1 | neighbors_2)
                        )
                        if len(neighbors_1 | neighbors_2) > 0
                        else 0.0
                    )
                    self.jaccard_similarities.append(
                        {
                            "title_1": node_1,
                            "title_2": node_2,
                            "jaccard_similarity": jaccard_similarity,
                        }
                    )

    def save_results(
        self, report_path="graph_analysis_report.txt", csv_path="title_similarity.csv"
    ):
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Top 5 Nodes by Degree Centrality:\n")
            for node, centrality in sorted(
                self.degree_centrality.items(), key=lambda x: x[1], reverse=True
            )[:5]:
                f.write(f"Node: {node}, Centrality: {centrality}\n")
            f.write("\n")

            f.write("Shortest Paths between Some Nodes:\n")
            for source_node in self.titles[:5]:
                for target_node in self.titles[:5]:
                    if source_node != target_node:
                        path_length = self.shortest_paths.get(source_node, {}).get(
                            target_node, None
                        )
                        if path_length is not None:
                            f.write(
                                f"Shortest path from {source_node} to {target_node}: {path_length} steps\n"
                            )
            f.write("\n")

            f.write(f"The graph has {len(self.components)} connected components.\n")
            for i, component in enumerate(self.components):
                f.write(f"Component {i+1}: {component}\n")
            f.write("\n")

            f.write("Similarity Based on Connected Components:\n")
            for entry in self.component_similarity:
                f.write(
                    f"{entry['title_1']} and {entry['title_2']} are in the same component with similarity {entry['component_similarity']}\n"
                )
            f.write("\n")

            f.write("Jaccard Similarity between Some Nodes:\n")
            for entry in self.jaccard_similarities[:5]:
                f.write(
                    f"{entry['title_1']} and {entry['title_2']} have Jaccard similarity of {entry['jaccard_similarity']}\n"
                )
            f.write("\n")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Title 1", "Title 2", "Similarity"])
            for entry in self.similarity_data:
                writer.writerow(
                    [entry["title_1"], entry["title_2"], entry["similarity"]]
                )

        print(f"Análise salva em '{report_path}' e '{csv_path}'")
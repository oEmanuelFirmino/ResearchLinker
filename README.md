# Web Scraping and Article Analysis

This project implements a system for collecting academic articles, vectorizing their content, and analyzing relationships between articles using graph theory and natural language processing (NLP) techniques. Additionally, it incorporates various rigorous mathematical approaches for semantic analysis, clustering, and trend evaluation in publications. Below, we detail the mathematical foundations of the project, including formulas and the application of graph algorithms and NLP techniques.

## Project Structure

```
ResearchLinker/
├── src/
│   ├── web_scraper.py
│   ├── article_vectorizer.py
│   ├── article_graph_analyzer.py
│   ├── article_analyzer.py
│   └── article_similarity_analyzer.py
├── main.py
├── .env.example
└── .gitignore
```

## Mathematical and Theoretical Concepts

### 1. Data Collection and Initial Representation

Data is collected through web scraping using the BeautifulSoup library. The extracted data (titles, authors, etc.) is structured and stored in JSON format. While this process does not involve complex mathematics, it forms the foundation for graph construction and subsequent vectorization.

### 2. Text Vectorization with BERT and Embeddings

Text vectorization utilizes the BERT model to transform textual content into high-dimensional vector embeddings. The `all-MiniLM-L6-v2` model is used to generate semantic vectors representing words, phrases, and documents. These embeddings serve as input for constructing a semantic graph and analyzing similarity.

The fundamental equation here is the cosine similarity between two vectors \($ v_1 $\) and \($ v_2 $\), given by:

$$
\text{similarity}(v_1, v_2) = \frac{v_1 \cdot v_2}{||v_1|| \cdot ||v_2||}
$$

where:

- \($ v_1 $\) and \($ v_2 $\) are the BERT-generated vectors for two words, phrases, or articles.
- \($ v_1 \cdot v_2 $\) is the dot product, capturing the directional relationship between vectors.
- The denominator normalizes the vectors, ensuring the similarity score is between -1 and 1, where 1 indicates identical vectors.

### 3. Semantic Graph Construction of Articles

Article similarity analysis is performed by creating an undirected graph \($ G = (V, E) $\), where:

- \($ V $\) is the set of nodes (articles).
- \($ E $\) is the set of edges between articles, representing their similarity.

The adjacency matrix \($ A $\) of the graph is constructed based on article similarity. If the similarity between two articles \($ A_i $\) and \($ A_j $\) exceeds a threshold \($ \theta $\), an edge is created:

$$
A\_{ij} = \begin{cases}
\text{similarity}(A_i, A_j) & \text{if } \text{similarity}(A_i, A_j) \geq \theta \\
0 & \text{otherwise}
\end{cases}
$$

This matrix forms the basis for analyzing semantic relationships and constructing the article graph.

### 4. Centrality and Connectivity Analysis

Once the graph is built, various centrality algorithms are applied to evaluate the importance of articles in the network.

#### 4.1. Degree Centrality

The degree centrality \($ C\_{\text{degree}}(v) $\) of a node \($ v $\) is given by the number of connections that the node has with other nodes:

$$
C*{\text{degree}}(v) = \sum*{u \in V} A\_{vu}
$$

where \($ A\_{vu} $\) is an element of the adjacency matrix indicating an edge between nodes \($ v $\) and \($ u $\).

#### 4.2. Betweenness Centrality

Betweenness centrality \($ C\_{\text{betweenness}}(v) $\) measures how often a node appears in the shortest paths between two other nodes \($ u $\) and \($ w $\):

$$
C*{\text{betweenness}}(v) = \sum*{u, w \in V} \frac{\sigma(u, v, w)}{\sigma(u, w)}
$$

where \($ \sigma(u, w) $\) is the number of shortest paths between \($ u $\) and \($ w $\), and \($ \sigma(u, v, w) $\) is the number of shortest paths passing through \($ v $\).

#### 4.3. PageRank Centrality

PageRank measures the importance of a node in the graph based on its ability to receive "votes" from other nodes. The iterative PageRank equation is:

$$
PR(v) = (1 - d) + d \sum\_{u \in N(v)} \frac{PR(u)}{|N(u)|}
$$

where:

- \($ PR(v) $\) is the PageRank score of node \($ v $\).
- \($ d $\) is the damping factor $(typically $ $0.85)$.
- \($ N(v) $\) is the set of neighbors of \($ v $\).
- \($ |N(u)| $\) is the number of neighbors of node \($ u $\).

These metrics help identify key articles within an academic field and track publication trends.

### 5. Article Clustering (Topic Modeling)

Article clustering is performed using clustering techniques such as K-means, which aims to minimize intra-cluster distance and maximize inter-cluster separation. The K-means formula for centroid \($ C_k $\) of a cluster \($ k $\) is:

$$
C*k = \frac{1}{|S_k|} \sum*{x_i \in S_k} x_i
$$

where:

- \($ S_k $\) is the set of articles in cluster \($ k $\).
- \($ x_i $\) is the feature vector of article \($ i $\).

Additionally, techniques like Latent Dirichlet Allocation $(LDA)$ can be used to model topic distributions across a document collection.

### 6. Interactive Visualization and Trend Analysis

Publication trend analysis over time is performed using co-occurrence matrices or time-series graphs. Tools like Pyvis and Matplotlib enable interactive visualizations to understand the evolution of academic fields.

The trend of publications in different areas can be represented as a function of centrality and co-occurrence frequency between articles, helping detect research peaks in specific topics.

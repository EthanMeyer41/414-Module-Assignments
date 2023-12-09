import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt


df = pd.read_csv('imdb_movies2.csv')

# Select relevant columns
selected_columns = ['names', 'genre', 'score', 'budget_x', 'revenue']

# Fill NaN values with empty string
selected_data = df[selected_columns].fillna('').apply(lambda x: ' '.join(map(str, x)), axis=1)

#CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(selected_data)

# Calculate cosine similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Create a mapping between movie titles and indices
indices = pd.Series(df.index, index=df['names']).drop_duplicates()


def get_top_similar_movies(movie_title):
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11] 
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]


def visualize_network_graph(movie_title):
    G = nx.Graph()

   
    movie_info = df.loc[df['names'] == movie_title, selected_columns]
    G.add_node(movie_title, label=f"{movie_title}\n{movie_info.iloc[0]['genre']}\nScore: {movie_info.iloc[0]['score']}\nBudget: {movie_info.iloc[0]['budget_x']}\nRevenue: {movie_info.iloc[0]['revenue']}")
    
    similar_movies = get_top_similar_movies(movie_title)
    for _, movie in similar_movies.iterrows():
        G.add_node(movie['names'], label=f"{movie['names']}\n{movie['genre']}\nScore: {movie['score']}\nBudget: {movie['budget_x']}\nRevenue: {movie['revenue']}")
        G.add_edge(movie_title, movie['names'])

    # Draw and display the graph
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(15, 15))
    nx.draw(G, pos, with_labels=False, font_size=8, font_color="black", node_size=500, node_color="skyblue", font_weight="bold", edge_color="gray", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="black")
    plt.title(f"Similar Movies Network for '{movie_title}'", fontsize=14)
    plt.show()

# Choose a movie title to visualize its similar movies
movie_to_visualize = 'Herbie Fully Loaded'
visualize_network_graph(movie_to_visualize)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv('imdb_movies2.csv')


query_entities = ['The Batman', 'Shrek', 'Dog']

results = {}


for query_entity in query_entities:
    
    query_index = df.index[df['names'] == query_entity].tolist()[0]

    selected_columns = ['genre', 'score', 'budget_x', 'revenue']
    selected_data = df[selected_columns].fillna('').apply(lambda x: ' '.join(map(str, x)), axis=1)

    
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(selected_data).toarray()

   
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    
    query_similarity = cosine_sim[query_index]

    
    top_similar_items = sorted(enumerate(query_similarity), key=lambda x: x[1], reverse=True)[1:11]

    
    results[query_entity] = [df['names'][i] for i, _ in top_similar_items]


for query_entity, similar_items in results.items():
    print(f"\nTop 10 most similar movies to '{query_entity}' based on selected columns:")
    for item in similar_items:
        print(item)

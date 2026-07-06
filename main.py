import pandas as pd
import numpy as np
import ast
import pickle

# 🔹 Load datasets (FIXED for ParserError)
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv', engine='python', on_bad_lines='skip')

# 🔹 Merge datasets
movies = movies.merge(credits, on='title')

# 🔹 Select important columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# 🔹 Remove null values
movies.dropna(inplace=True)

# 🔹 Convert JSON to list
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])  # top 3 cast

# 🔹 Extract director
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# 🔹 Convert overview to list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# 🔹 Remove spaces
def remove_space(L):
    return [i.replace(" ", "") for i in L]

movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)
movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)

# 🔹 Create tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# 🔹 New dataframe
new_df = movies[['movie_id','title','tags']]

# 🔹 Convert list to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# 🔹 Vectorization
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# 🔹 Similarity
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

# 🔹 Save files
pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("✅ Model built successfully!")
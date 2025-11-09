import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

final_dataset = ratings.pivot_table(index='movieId', columns='userId', values='rating')
final_dataset.fillna(0, inplace=True)

# Filter movies/users with minimum votes
no_user_voted = ratings.groupby('movieId')['rating'].count()
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]
no_movies_voted = ratings.groupby('userId')['rating'].count()
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

csr_data = csr_matrix(final_dataset.values)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

def get_movie_recommendation(movie_name, n_movies_to_reccomend=10, plot=False):
    movie_list = movies[movies['title'].str.contains(movie_name, case=False, regex=False)]
    if len(movie_list):
        movie_id = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset.index == movie_id].index[0]
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_reccomend+1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[1:]
        recommend_frame = []
        for idx, dist in rec_movie_indices:
            movie_id = final_dataset.iloc[idx].name
            title = movies[movies['movieId'] == movie_id]['title'].values[0]
            recommend_frame.append({'Title': title, 'Distance': dist})
        df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_reccomend+1))
        if plot:
            plt.figure(figsize=(10,5))
            plt.barh(df['Title'], df['Distance'], color="teal")
            plt.xlabel("Cosine Distance (Lower = More Similar)")
            plt.title(f"Top {n_movies_to_reccomend} Recommendations for '{movie_name}'")
            plt.gca().invert_yaxis()
            plt.show()
        return df
    else:
        return "No movies found. Please check your input"

print(get_movie_recommendation("Toy Story", plot=True))

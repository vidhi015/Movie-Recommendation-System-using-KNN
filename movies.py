import pandas as pd

movies = pd.read_csv("/content/movies.csv")

movies.head()

movies = pd.read_csv("/content/movies.csv" , usecols = ['movieId' , 'title'])

movies.head()

ratings = pd.read_csv("/content/ratings.csv")

ratings.head()

ratings = pd.read_csv("/content/ratings.csv" , usecols = ['userId' , 'movieId' , 'rating'])

ratings.head()

ratings.shape()

movies.shape()

ratings.pivot(index='movieId' , columns = 'userId' , values = 'rating')

movies_users = ratings.pivot(index='movieId' , columns = 'userId' , values = 'rating').fillna(0)
movies_users.head()

from scipy.sparse import csr_matrix

mat_movies = csr_matrix(movies_users.values)
mat_movies

from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(metric = 'cosine' , algorithm = 'brute' , n_neighbors = 20)
model.fit(mat_movies)

pip install fuzzywuzzy

from fuzzywuzzy import process

recommender('toy story' , mat_movies , 10)

recommender('iron man' , mat_movies , 10)
import numpy as np
import pandas as pd
# Reading ratings file
ratings = pd.read_csv("ratings.csv", sep=',', encoding='latin-1', usecols=["userId", "movieId", "rating", "timestamp"])
# Reading movies file
movies = pd.read_csv('movies.csv', sep=',', encoding='latin-1', usecols=['movieId', 'title', 'genres'])
n_users = ratings.userId.unique().shape[0]
n_movies = ratings.movieId.unique().shape[0]
print ('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_movies))
Ratings = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
R = Ratings.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)
from scipy.sparse.linalg import svds 
U, sigma, Vt = svds(Ratings_demeaned, k = 50)
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds = pd.DataFrame(all_user_predicted_ratings, columns = Ratings.columns)

def recommend_movies(predictions, userID, movies, original_ratings, num_recommendations):# Getting and sorting the user's predictions    
    user_row_number = userID - 1 
    sorted_user_predictions = preds.iloc[user_row_number].sort_values(ascending=False) # User id starts at 1    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings[original_ratings.userId == (userID)]
    user_full = (user_data.merge(movies, how = 'left', left_on = 'movieId', right_on = 'movieId').sort_values(['rating'], ascending=False))
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies[movies['movieId'].isin(user_full['movieId'])].merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',left_on = 'movieId',right_on = 'movieId').rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :-1])
    return user_full, recommendations

from surprise import Reader, Dataset, SVD, evaluate
reader = Reader()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)# Spliting the dataset for 5-fold evaluation
svd = SVD()# Use the SVD algorithm.
evaluate(svd, data, measures=['RMSE'])# Compute the RMSE of the SVD algorithm.
predictions = recommend_movies(preds,111,movies,ratings,5)
print(predictions)

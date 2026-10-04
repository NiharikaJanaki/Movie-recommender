import os
import numpy as np
import scipy.io


def load_dataset(filepath="data/users_movies.mat"):
    """
    Loads the MATLAB dataset file and extracts all essential arrays.
    Handles relative path resolution cleanly whether run from root or src/.
    """
    if not os.path.exists(filepath):
        # Fallback check if script is executed from inside src/
        alt_path = os.path.join("..", filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Could not locate dataset at '{filepath}'.")

    data = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)

    movies = data["movies"]
    users_movies = data["users_movies"]
    users_movies_sort = data["users_movies_sort"]
    index_small = np.asarray(data["index_small"]).flatten()
    trial_user = np.asarray(data["trial_user"]).flatten()

    return movies, users_movies, users_movies_sort, index_small, trial_user


def get_movie_title(movie):
    """Safely extracts a clean movie title string from varying MATLAB object structures."""
    if isinstance(movie, str):
        return movie
    if isinstance(movie, np.ndarray):
        if movie.size == 1:
            return str(movie.item())
        return str(movie.flat[0])
    return str(movie)


def filter_complete_raters(ratings_20):
    """
    Filters rows where users rated all 20 selected movies (no zeros).
    Returns filtered ratings matrix and the boolean mask.
    """
    mask = np.all(ratings_20 != 0, axis=1)
    return ratings_20[mask], mask


def euclidean_similarity(ratings, query):
    """Computes Euclidean distances between matrix rows and a 1D query vector."""
    return np.linalg.norm(ratings - query, axis=1)


def center_rows(X):
    """Subtracts the mean of each row from the row elements (row-centering)."""
    row_means = X.mean(axis=1, keepdims=True)
    return X - row_means


def pearson_similarity(ratings_cent, query_cent):
    """Computes Pearson correlation coefficients using centered rating vectors."""
    numerator = ratings_cent @ query_cent
    denominator = (
        np.linalg.norm(ratings_cent, axis=1) * np.linalg.norm(query_cent)
    )
    # Guard against division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return numerator / denominator


def recommend(
    filtered_user_index,
    users_movies,
    query_ratings,
    index_small,
    movies,
    original_user_indices,
    top_n=10,
):
    """
    Finds movies rated 5 by the most similar user that the current user has not
    already rated 5 in the popular 20 subset.
    """
    original_user = original_user_indices[filtered_user_index]
    liked_by_similar = np.where(users_movies[original_user] == 5)[0]

    # Movies among the 20 subset that the current user rated 5
    user_liked = index_small[query_ratings == 5]

    rec_indices = np.setdiff1d(liked_by_similar, user_liked)

    recommendations = []
    for movie_index in rec_indices:
        title = get_movie_title(movies[movie_index])
        recommendations.append(title)
        if len(recommendations) == top_n:
            break

    return recommendations


def print_movie_list(title_header, movie_list):
    """Helper to display formatted movie recommendation outputs."""
    print(f"\n================ {title_header} ================")
    if not movie_list:
        print("No recommendations found.")
        return
    for i, movie in enumerate(movie_list, start=1):
        print(f"{i:2d}. {movie}")
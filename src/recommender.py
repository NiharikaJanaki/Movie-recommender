import numpy as np
from utils import (
    load_dataset,
    get_movie_title,
    filter_complete_raters,
    euclidean_similarity,
    center_rows,
    pearson_similarity,
    recommend,
)


def run_recommendation_pipeline(query_ratings, top_k=5, top_n_recs=10):
    """
    Executes the linear algebra recommendation pipeline for a given 20-element vector.
    Returns a dictionary with all analysis details, rankings, and recommendations.
    """
    movies, users_movies, users_movies_sort, index_small, _ = load_dataset()
    ratings, complete_mask = filter_complete_raters(users_movies_sort)
    original_user_indices = np.where(complete_mask)[0]

    # 1. Euclidean Distance
    eucl_distances = euclidean_similarity(ratings, query_ratings)
    eucl_sorted_indices = np.argsort(eucl_distances)

    # 2. Pearson Correlation
    ratings_cent = center_rows(ratings)
    query_cent = center_rows(query_ratings.reshape(1, -1)).flatten()
    pearson_coeffs = pearson_similarity(ratings_cent, query_cent)
    pearson_sorted_indices = np.argsort(-pearson_coeffs)

    # 3. Top Matches Metadata
    top_euclidean = [
        {
            "rank": r + 1,
            "user_id": int(original_user_indices[idx]),
            "filtered_idx": int(idx),
            "score": float(eucl_distances[idx]),
        }
        for r, idx in enumerate(eucl_sorted_indices[:top_k])
    ]

    top_pearson = [
        {
            "rank": r + 1,
            "user_id": int(original_user_indices[idx]),
            "filtered_idx": int(idx),
            "score": float(pearson_coeffs[idx]),
        }
        for r, idx in enumerate(pearson_sorted_indices[:top_k])
    ]

    # 4. Generate Recommendations
    rec_eucl = recommend(
        eucl_sorted_indices[0],
        users_movies,
        query_ratings,
        index_small,
        movies,
        original_user_indices,
        top_n=top_n_recs,
    )

    rec_pearson = recommend(
        pearson_sorted_indices[0],
        users_movies,
        query_ratings,
        index_small,
        movies,
        original_user_indices,
        top_n=top_n_recs,
    )

    common_recs = list(set(rec_eucl).intersection(set(rec_pearson)))

    return {
        "movies": movies,
        "index_small": index_small,
        "ratings": ratings,
        "original_user_indices": original_user_indices,
        "eucl_distances": eucl_distances,
        "pearson_coeffs": pearson_coeffs,
        "best_eucl_filtered_idx": eucl_sorted_indices[0],
        "best_pearson_filtered_idx": pearson_sorted_indices[0],
        "top_euclidean": top_euclidean,
        "top_pearson": top_pearson,
        "rec_eucl": rec_eucl,
        "rec_pearson": rec_pearson,
        "common_recs": common_recs,
    }


if __name__ == "__main__":
    # Standard terminal CLI test with the textbook trial user
    _, _, _, _, trial_user = load_dataset()
    print("Running recommender pipeline on Trial User...")
    results = run_recommendation_pipeline(trial_user)

    print("\nTop Euclidean Match:", results["top_euclidean"][0])
    print("Top Pearson Match  :", results["top_pearson"][0])
    print("\nCommon Recommendations:")
    for title in results["common_recs"]:
        print(f" - {title}")
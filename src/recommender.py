import numpy as np
import matplotlib.pyplot as plt
from utils import (
    load_dataset,
    get_movie_title,
    filter_complete_raters,
    euclidean_similarity,
    center_rows,
    pearson_similarity,
    recommend,
    print_movie_list,
)


def prompt_user_ratings(index_small, movies):
    """
    Prompts the user via stdin to rate each of the 20 popular movies from 1 to 5.
    Allows entering 'r' to automatically fill remaining ratings randomly (1-5).
    """
    print("\n================ PERSONAL RATING INPUT ================")
    print("Please rate the following 20 movies on a scale of 1 to 5.")
    print("  1 = Strongly Dislike, 5 = Strongly Like")
    print("  (Type 'r' at any prompt to fill remaining movies with random ratings)\n")

    user_ratings = []
    use_random = False

    for i, idx in enumerate(index_small):
        title = get_movie_title(movies[idx])

        if use_random:
            val = int(np.random.randint(1, 6))
            user_ratings.append(val)
            print(f"[{i+1}/20] {title}: {val} (auto-assigned)")
            continue

        while True:
            raw_input = input(f"[{i+1}/20] Rate '{title}' (1-5 or 'r'): ").strip()
            if raw_input.lower() == 'r':
                use_random = True
                val = int(np.random.randint(1, 6))
                user_ratings.append(val)
                print(f"  -> Filling remaining movies randomly. Assigned: {val}")
                break
            try:
                val = int(raw_input)
                if 1 <= val <= 5:
                    user_ratings.append(val)
                    break
                print("  Invalid range. Enter an integer between 1 and 5.")
            except ValueError:
                print("  Invalid format. Enter a valid number 1-5 or 'r'.")

    return np.array(user_ratings, dtype=int)


def main():
    # 1. Load Data
    movies, users_movies, users_movies_sort, index_small, trial_user = load_dataset()

    print("\n================ DATA INFORMATION ================")
    print(f"movies shape            : {movies.shape}")
    print(f"users_movies shape      : {users_movies.shape}")
    print(f"users_movies_sort shape : {users_movies_sort.shape}")
    print(f"index_small shape       : {index_small.shape}")
    print(f"trial_user shape        : {trial_user.shape}")

    # 2. Display 20 Selected Movies
    print("\n================ 20 SELECTED MOVIES ================")
    for i, index in enumerate(index_small):
        print(f"{i + 1:2d} : {get_movie_title(movies[index])}")

    # 3. Filter Complete Raters
    ratings, complete_mask = filter_complete_raters(users_movies_sort)
    original_user_indices = np.where(complete_mask)[0]

    print("\n================ FILTERING ================")
    print(f"Original users         : {users_movies_sort.shape[0]}")
    print(f"Users after filtering  : {ratings.shape[0]}")
    print(f"Number of movies       : {ratings.shape[1]}")

    # 4. Trial User Analysis - Euclidean Distance
    eucl_dist = euclidean_similarity(ratings, trial_user)
    closest_user_dist = np.argsort(eucl_dist)

    # 5. Trial User Analysis - Pearson Correlation
    ratings_cent = center_rows(ratings)
    trial_cent = center_rows(trial_user.reshape(1, -1)).flatten()
    pearson_coeff = pearson_similarity(ratings_cent, trial_cent)
    closest_user_pearson = np.argsort(-pearson_coeff)

    # 6. Compare Baseline Trial User Results
    print("\n================ TRIAL USER TOP 5 COMPARISON ================")
    print("\nEUCLIDEAN:")
    for rank, idx in enumerate(closest_user_dist[:5], start=1):
        print(f"  {rank} | User: {original_user_indices[idx]:4d} | Distance: {eucl_dist[idx]:.4f}")

    print("\nPEARSON:")
    for rank, idx in enumerate(closest_user_pearson[:5], start=1):
        print(f"  {rank} | User: {original_user_indices[idx]:4d} | Pearson : {pearson_coeff[idx]:.4f}")

    trial_rec_eucl = recommend(
        closest_user_dist[0], users_movies, trial_user, index_small, movies, original_user_indices
    )
    trial_rec_pearson = recommend(
        closest_user_pearson[0], users_movies, trial_user, index_small, movies, original_user_indices
    )

    print_movie_list("TRIAL USER - EUCLIDEAN RECOMMENDATIONS", trial_rec_eucl)
    print_movie_list("TRIAL USER - PEARSON RECOMMENDATIONS", trial_rec_pearson)

    # 7. Interactive Personalization
    myratings = prompt_user_ratings(index_small, movies)
    print("\nRecorded User Ratings Vector:")
    print(myratings)

    # 8. Personalized Computations
    my_eucl_dist = euclidean_similarity(ratings, myratings)
    my_closest_euclidean = np.argsort(my_eucl_dist)

    my_ratings_cent = center_rows(myratings.reshape(1, -1)).flatten()
    my_pearson = pearson_similarity(ratings_cent, my_ratings_cent)
    my_closest_pearson = np.argsort(-my_pearson)

    print("\n================ YOUR TOP 5 MATCHES ================")
    print("\nEUCLIDEAN MATCHES:")
    for rank, idx in enumerate(my_closest_euclidean[:5], start=1):
        print(f"  {rank} | User: {original_user_indices[idx]:4d} | Distance: {my_eucl_dist[idx]:.4f}")

    print("\nPEARSON MATCHES:")
    for rank, idx in enumerate(my_closest_pearson[:5], start=1):
        print(f"  {rank} | User: {original_user_indices[idx]:4d} | Pearson : {my_pearson[idx]:.4f}")

    # 9. Personalized Recommendations
    my_rec_eucl = recommend(
        my_closest_euclidean[0], users_movies, myratings, index_small, movies, original_user_indices
    )
    my_rec_pearson = recommend(
        my_closest_pearson[0], users_movies, myratings, index_small, movies, original_user_indices
    )

    print_movie_list("YOUR PERSONAL EUCLIDEAN RECOMMENDATIONS", my_rec_eucl)
    print_movie_list("YOUR PERSONAL PEARSON RECOMMENDATIONS", my_rec_pearson)

    common_recs = set(my_rec_eucl).intersection(set(my_rec_pearson))
    print("\n================ COMMON PERSONAL RECOMMENDATIONS ================")
    if common_recs:
        for movie in common_recs:
            print(f"- {movie}")
    else:
        print("No overlapping recommendations between both metrics.")

    print("\n" + "=" * 65)
    print("RECOMMENDER EXECUTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
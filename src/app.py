import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils import load_dataset, get_movie_title
from recommender import run_recommendation_pipeline

# Page setup
st.set_page_config(page_title="Movie Recommender Dashboard", layout="wide")
st.title("🎬 Linear Algebra Movie Recommender System")
st.markdown("Project 7: Recommender system using **Euclidean Norms** ($L_2$) and **Pearson Correlation** ($\cos\\theta$).")

# Load base dataset for movie metadata
movies, users_movies, users_movies_sort, index_small, trial_user = load_dataset()

# ----------------- SESSION STATE FOR INPUTS -----------------
if "ratings" not in st.session_state:
    st.session_state.ratings = trial_user.copy()

def set_ratings(new_ratings):
    st.session_state.ratings = np.array(new_ratings, dtype=int)

# ----------------- SIDEBAR & PRESETS -----------------
st.sidebar.header("🕹️ Quick Presets")
if st.sidebar.button("Load Textbook Trial User", use_container_width=True):
    set_ratings(trial_user)
    st.rerun()

if st.sidebar.button("Randomize Ratings (1 to 5)", use_container_width=True):
    set_ratings(np.random.randint(1, 6, size=20))
    st.rerun()

if st.sidebar.button("Reset to Neutral (All 3s)", use_container_width=True):
    set_ratings(np.full(20, 3))
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Adjust the 20 ratings below or click a preset, then click **Run Analysis**.")

# ----------------- INPUT SECTION -----------------
st.subheader("1. Rate the 20 Popular Movies")
cols = st.columns(4)

current_ratings = []
for i, idx in enumerate(index_small):
    title = get_movie_title(movies[idx])
    col = cols[i % 4]
    val = col.slider(
        f"{i+1}. {title}",
        min_value=1,
        max_value=5,
        value=int(st.session_state.ratings[i]),
        key=f"movie_{i}",
    )
    current_ratings.append(val)

query_vector = np.array(current_ratings, dtype=int)

# ----------------- RUN ANALYSIS -----------------
if st.button("🚀 Run Recommender Analysis", type="primary", use_container_width=True):
    results = run_recommendation_pipeline(query_vector)

    st.markdown("---")
    st.header("2. Similarity Analysis Dashboard")

    col_e, col_p = st.columns(2)

    with col_e:
        st.subheader("📏 Euclidean Distance ($L_2$ Norm)")
        st.caption("Measures geometric distance. Lower is closer.")
        best_e = results["top_euclidean"][0]
        st.metric(label=f"Closest User ID: {best_e['user_id']}", value=f"Distance: {best_e['score']:.4f}")

        st.markdown("**Top 5 Euclidean Neighbors:**")
        e_table = [{"Rank": item["rank"], "User ID": item["user_id"], "Distance": f"{item['score']:.4f}"} for item in results["top_euclidean"]]
        st.table(e_table)

    with col_p:
        st.subheader("📐 Pearson Correlation ($\cos\\theta$)")
        st.caption("Measures angle between centered ratings. Closer to +1.0 is better.")
        best_p = results["top_pearson"][0]
        st.metric(label=f"Closest User ID: {best_p['user_id']}", value=f"Correlation: {best_p['score']:.4f}")

        st.markdown("**Top 5 Pearson Neighbors:**")
        p_table = [{"Rank": item["rank"], "User ID": item["user_id"], "Correlation": f"{item['score']:.4f}"} for item in results["top_pearson"]]
        st.table(p_table)

    # ----------------- VISUALIZATION -----------------
    st.markdown("---")
    st.subheader("3. Rating Profile Comparison")
    
    fig, ax = plt.subplots(figsize=(12, 4))
    x_indices = np.arange(1, 21)
    
    best_e_idx = results["best_eucl_filtered_idx"]
    best_p_idx = results["best_pearson_filtered_idx"]
    db_ratings = results["ratings"]

    ax.plot(x_indices, query_vector, 'b-o', label="Your Ratings", linewidth=2.5)
    ax.plot(x_indices, db_ratings[best_e_idx], 'r--s', label=f"Best Euclidean User (#{results['top_euclidean'][0]['user_id']})", alpha=0.7)
    ax.plot(x_indices, db_ratings[best_p_idx], 'g-.^', label=f"Best Pearson User (#{results['top_pearson'][0]['user_id']})", alpha=0.7)
    
    ax.set_xticks(x_indices)
    ax.set_xlabel("Movie Index (1 to 20)")
    ax.set_ylabel("Rating (1 to 5)")
    ax.set_ylim(0.5, 5.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    st.pyplot(fig)

    # ----------------- RECOMMENDATIONS -----------------
    st.markdown("---")
    st.header("4. Movie Recommendations")

    r_col1, r_col2, r_col3 = st.columns(3)

    with r_col1:
        st.markdown("### 🔹 Euclidean Recommendations")
        if results["rec_eucl"]:
            for i, title in enumerate(results["rec_eucl"], 1):
                st.write(f"**{i}.** {title}")
        else:
            st.info("No recommendations found.")

    with r_col2:
        st.markdown("### 🔸 Pearson Recommendations")
        if results["rec_pearson"]:
            for i, title in enumerate(results["rec_pearson"], 1):
                st.write(f"**{i}.** {title}")
        else:
            st.info("No recommendations found.")

    with r_col3:
        st.markdown("### ⭐ Common Recommendations")
        st.caption("Overlap between Euclidean & Pearson lists.")
        if results["common_recs"]:
            for title in results["common_recs"]:
                st.success(f"✔ **{title}**")
        else:
            st.warning("No overlapping titles between the two metrics.")
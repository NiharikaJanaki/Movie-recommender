# Movie Recommender System

A collaborative filtering movie recommender system that predicts films you will like based on your ratings of 20 popular movies[cite: 82, 83]. It matches your preferences against user profiles in the MovieLens dataset using Euclidean distance and Pearson correlation, then provides recommendations along with their common intersection[cite: 82, 83, 189].

---

## What It Does

1. **User Ratings Input:** You rate 20 popular movies on a scale from 1 (strongly dislike) to 5 (strongly like)[cite: 83, 90].
2. **Nearest-Neighbor Matching:** Matches your rating vector against complete raters in the dataset[cite: 84, 189]:
   - **Euclidean Distance:** Identifies users with the closest exact rating scores[cite: 85, 189].
   - **Pearson Correlation:** Identifies users whose relative preference trends align with yours, adjusting for overall rating bias[cite: 86, 189].
3. **Movie Suggestions:** Recommends movies rated 5 by your closest matches that you have not yet rated[cite: 89, 189].
4. **Common Recommendations:** Highlights the overlapping movies found by both metrics for higher-confidence picks[cite: 189].

---

## Tech Stack

- **Python 3.8+**[cite: 1]
- **NumPy**
- **SciPy**[cite: 188]
- **Matplotlib**[cite: 188]
- **Streamlit**

---
``
# Setup & Installation Guide

Complete step-by-step instructions to set up the project environment, install dependencies, and run the application.

---

## Prerequisites

- **Python 3.8+** installed on your system[cite: 1]
- **Git** installed on your system

---

## 1. Clone the Repository

Clone this repository and navigate into the project directory:

```bash
git clone [https://github.com/](https://github.com/)<your-username>/movie-recommender.git
cd movie-recommender

```

---

## 2. Dataset Setup

Make sure your MATLAB dataset file is placed inside the `data/` folder:

```text
movie-recommender/
└── data/
    └── users_movies.mat

```

(If the `data` directory does not exist, create it and copy `users_movies.mat` into it).

---

## 3. Set Up a Virtual Environment

Creating an isolated virtual environment is strongly recommended to avoid system permission issues and package dependency conflicts.

### Windows (PowerShell)

1. Create the environment:
```powershell
python -m venv .venv

```


2. Activate the environment:


```powershell
.\.venv\Scripts\Activate.ps1

```


*If PowerShell shows an `Execution_Policies` restriction error, run:*
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

```



### Windows (Command Prompt / cmd.exe)

1. Create the environment:
```cmd
python -m venv .venv

```


2. Activate the environment:
```cmd
.venv\Scripts\activate.bat

```



### macOS / Linux (Terminal)

1. Create the environment:
```bash
python3 -m venv .venv

```


2. Activate the environment:
```bash
source .venv/bin/activate

```



---

## 4. Install Dependencies

Once your virtual environment is activated, upgrade `pip` and install all required libraries:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt

```

`requirements.txt` includes:

* `numpy`

* `scipy`

* `matplotlib`

* `streamlit`

---

## 5. Verify the Setup

Test that the core recommendation engine runs without errors:

```bash
python src/recommender.py

```

---

## 6. Run the Application

### Option A: Web Dashboard (Streamlit)

Launch the interactive web browser interface:

```bash
streamlit run src/app.py

```

This will automatically open the dashboard in your default browser at `http://localhost:8501`.

### Option B: Terminal CLI

Run the recommender directly in your terminal console:

```bash
python src/recommender.py

```

Follow the prompts to rate movies from 1 to 5, or type `r` to auto-fill the remaining movies randomly.

---

## 7. Exit the Virtual Environment

When you are done working, exit the virtual environment:

```bash
deactivate
```

## Screenshots

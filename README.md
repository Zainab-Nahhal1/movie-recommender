# Movie Recommendation System Using KNN

## Description
This project builds a movie recommendation system using KNN on the MovieLens dataset. Users can get recommendations based on similarity to a given movie.

## Features
- Filter movies/users with minimum ratings
- Build KNN model
- Recommend top N similar movies
- Optional visualization of distances

## How to Run
1. Install dependencies:
   pip install pandas numpy scipy scikit-learn matplotlib seaborn
2. Place movies.csv and ratings.csv in data/
3. Run the main script: python source/main.py
4. Use get_movie_recommendation(movie_name, n_movies_to_reccomend=10, plot=True) to get recommendations

## Structure
- source/: main code
- data/: dataset files
- tests/: test scripts
- sample/: small sample data
- .gitignore: git ignore rules
- Makefile: automate tasks

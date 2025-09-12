Note: This is for learning purpose and goal is to learn how to train and evaluate a model. Instead of training a model from scratch, 
I can use a pre-trained, state-of-the-art model designed specifically for this task. Models from the Hugging Face library like BERT, RoBERTa,
or DistilBERT are already trained on massive amounts of text and are highly effective for sentiment analysis.

This project also uses UV package manager from Astral


# 🎬 Movie Reviews Analysis & Summarization

This project combines **sentiment analysis** and **summarization** of movie reviews.  
It uses:

- **TensorFlow/Keras** → train a sentiment analysis model on the IMDB dataset  
- **Hugging Face Transformers (BART)** → summarize reviews from [The Movie Database (TMDb)](https://www.themoviedb.org/)  
- **TMDb API** → fetch real user reviews per movie  

You can either:
1. Run **sentiment analysis** on any review text.  
2. Fetch reviews of a movie (via TMDb ID) and get a **summarized overview**.  

---

## 📦 Features
- Train or load a **sentiment analysis model** (`SentimentAnalyzer`)  
- Fetch reviews from **TMDb API** (with pagination) or scrape from websites (`ReviewFetcher`)  
- Summarize reviews in **batches** to avoid token limits (`MovieSummarizer`)  
- Combine both: get **positive-only** or **negative-only** summaries (sentiment-aware summarization)  

---

## ⚙️ Installation

Clone this repo and install dependencies:


git clone https://github.com/your-username/movie-reviews-analysis.git
cd movie-reviews-analysis


# To use glove embeddings (for better sentiment analysis)
Pick a size depending on tradeoff between accuracy and speed (e.g., glove.6B.100d.txt).

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

This gives you files like:
glove.6B.50d.txt
glove.6B.100d.txt
glove.6B.200d.txt
glove.6B.300d.txt

copy the file you want to use to an ./embeddings directory. 
Note: Make sure to update the code to use the proper file and make sure to update the embedding_dim to match size used.


#  To run locally using uv

# Movie reviews summarization
uv run python main.py --task summarize --movie_id 550

# Sentiment Analysis
uv run python main.py --task sentiment --text "I didn't like the movie, it was terrible"

# to run inside docker
docker build -t sentiment-analysis-app .
docker run sentiment-analysis-app




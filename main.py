from dotenv import load_dotenv
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from transformers import pipeline
import argparse
from movie_summarizer import MovieSummarizer
from sentiment_analyzer import SentimentAnalyzer

def main():
    
    parser = argparse.ArgumentParser(description="Movie Review Utility")
    parser.add_argument("--task", choices=["sentiment", "summarize"], required=True,
                        help="Choose 'sentiment' for sentiment analysis or 'summarize' for movie review summarization.")
    parser.add_argument("--text", type=str, help="Input text for sentiment analysis")
    parser.add_argument("--movie_id", type=int, help="TMDb movie ID for summarization")
    args = parser.parse_args()

    # loads variables from .env file
    load_dotenv()  
            
    if args.task == "sentiment":
        EMBEDDINGS_GLOVE_PATH = os.getenv("EMBEDDINGS_GLOVE_PATH")
        # just pass glove_path switch to pre-trained embeddings.
        analyzer = SentimentAnalyzer(glove_path=EMBEDDINGS_GLOVE_PATH, use_raw=True)
        if not args.text:
            raise ValueError("Please provide --text for sentiment analysis")

        result = analyzer.predict(args.text)
        print(result)
    elif args.task == "summarize":
        API_TOKEN = os.getenv("TMDB_API_TOKEN")

        summarizer = MovieSummarizer(API_TOKEN)
        if not args.movie_id:
            raise ValueError("Please provide --movie_id for summarization")
        summary = summarizer.summarize(args.movie_id)
        print(summary)

# -------------------------------
# MAIN ENTRY
# -------------------------------
if __name__ == "__main__":
    main()





from transformers import pipeline, BartTokenizer
from review_fetcher import ReviewFetcher

# -------------------------------
# MOVIE SUMMARIZER
# -------------------------------
class MovieSummarizer:


    def __init__(self, api_token, model_name="facebook/bart-large-cnn"):
        self.api_token = api_token
        self.summarizer = pipeline("summarization", model=model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.fetcher = ReviewFetcher(api_token=api_token)
    
    # From TMDb API
    def get_reviews(self, movie_id):

        reviews_api = self.fetcher.retrieve_movie_reviews_from_api(movie_id)
        return reviews_api
    
    
    def _summarize_in_batches(self, reviews, batch_size=3, max_length=150, min_length=50):
        batch_summaries = []

        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            if not batch:
                continue

            batch_text = " ".join(batch)

            # Tokenize and truncate to 1024 tokens (BART limit)
            tokens = self.tokenizer.encode(batch_text, truncation=True, max_length=1024, return_tensors="pt")

            try:
                summary_result = self.summarizer(
                    self.tokenizer.decode(tokens[0], skip_special_tokens=True),
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )

                if summary_result and "summary_text" in summary_result[0]:
                    batch_summaries.append(summary_result[0]["summary_text"])

            except Exception as e:
                print(f"⚠️ Skipping batch due to error: {e}")
                continue

        if not batch_summaries:
            return "No summary could be generated from the provided reviews."

        # Final summary across all batch summaries
        final_input = " ".join(batch_summaries)
        tokens = self.tokenizer.encode(final_input, truncation=True, max_length=1024, return_tensors="pt")

        try:
            final_summary_result = self.summarizer(
                self.tokenizer.decode(tokens[0], skip_special_tokens=True),
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            return final_summary_result[0]["summary_text"]

        except Exception as e:
            print(f"⚠️ Final summarization failed: {e}")
            return " ".join(batch_summaries)
    

    def summarize(self, movie_id, batch_size=10, max_length=150, min_length=50):
        reviews = self.get_reviews(movie_id)
        if not reviews:
            return "No reviews found."
        return self._summarize_in_batches(reviews,
                                    batch_size=batch_size,
                                    max_length=max_length,
                                    min_length=min_length)
# review_fetcher.py
import requests
#from bs4 import BeautifulSoup

BASE_URL = "https://api.themoviedb.org/3/movie/{movie_id}/reviews"

class ReviewFetcher:

    """
    Utility class to fetch reviews from webpages (scraping)
    or from APIs (like TMDb).
    """
    def __init__(self, api_token=None):
        self.api_token = api_token

    def scrape_reviews(self, url, selector="p.review-text"):
        """
        Scrape reviews from a webpage.
        Args:
            url (str): The webpage URL.
            selector (str): CSS selector to extract reviews.
        Returns:
            list[str]: List of extracted review texts.
        """
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch {url}, status {response.status_code}")
        soup = BeautifulSoup(response.text, "html.parser")
        return [r.get_text(strip=True) for r in soup.select(selector)]

    def retrieve_movie_reviews_from_api(self, movie_id):
        """
        Fetch reviews from an API (e.g., TMDb).
        Args:
            api_url (str): API endpoint URL.
        Returns:
            list[str]: List of review contents.
        """
        api_url = BASE_URL.format(movie_id=movie_id)
        headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
        print(f"api_url: {api_url}  token: {self.api_token}")
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        print(f"Fetched {len(data.get('results', []))} reviews from API")
        return [item["content"] for item in data.get("results", [])]

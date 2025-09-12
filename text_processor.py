import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



# Download necessary NLTK data files.
# These commands will check if the data is already present before downloading.
nltk.download("punkt", quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download("stopwords", quiet=True)


class TextPreprocessor:
    """
    A utility class for performing common text preprocessing tasks.
    """

    @staticmethod
    def preprocess_with_nltk(text):
        """
        Preprocesses text using NLTK to lowercase, tokenize, remove punctuation,
        and remove stopwords.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text as a single string with tokens separated by spaces.
        """
        # Lowercase the text
        text = text.lower()
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove punctuation and numbers by keeping only alphabetic tokens
        tokens = [t for t in tokens if t.isalpha()]
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in stopwords.words("english")]
        
        # Join the tokens back into a single string
        return " ".join(tokens)
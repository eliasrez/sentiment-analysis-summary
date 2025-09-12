import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, GlobalAveragePooling1D


class SentimentAnalyzer:
    def __init__(self, 
                 vocab_size=30000, 
                 embedding_dim=100, 
                 input_length=200,
                 glove_path=None,
                 use_raw=False,
                 trainable=False,
                 model_file="sentiment_model.keras"):
        """
        Sentiment Analyzer with optional GloVe embeddings and raw text preprocessing.

        Args:
            vocab_size (int): Maximum vocabulary size.
            embedding_dim (int): Dimension of embeddings (must match GloVe file if provided).
            input_length (int): Max sequence length.
            glove_path (str): Path to GloVe embeddings file (.txt).
            use_raw (bool): If True, load raw IMDB reviews instead of pre-tokenized dataset.
            trainable (bool): If True, fine-tune embeddings.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.glove_path = glove_path
        self.use_raw = use_raw
        self.trainable = trainable

        self.tokenizer = None
        self.word_index = None
        self.embedding_matrix = None

        # Load data
        if self.use_raw:
            print("Loading raw IMDB reviews (tensorflow_datasets)...")
            (self.train_data, self.train_labels), (self.test_data, self.test_labels) = self._load_raw_imdb()
        else:
            print("Loading pre-tokenized IMDB dataset (keras.datasets)...")
            (self.train_data, self.train_labels), (self.test_data, self.test_labels) = self._load_integer_imdb()

        # Check if the 'models' directory exists, if not, create it.
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model_path = os.path.join(model_dir, model_file) 

        if os.path.exists(self.model_path):
            print(f"Loading trained model from {self.model_path}...")
            self.model = load_model(self.model_path)
        else:
            print("No trained model found. Building a new one...")
            self.model = self._build_model()
            self.train()

    # --------------------------
    # Dataset Loaders
    # --------------------------
    def _load_integer_imdb(self):
        (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=self.vocab_size)
        train_data = pad_sequences(train_data, maxlen=self.input_length, padding="post", truncating="post")
        test_data = pad_sequences(test_data, maxlen=self.input_length, padding="post", truncating="post")
        return (train_data, np.array(train_labels)), (test_data, np.array(test_labels))

    def _load_raw_imdb(self):
        train_data, test_data = tfds.load("imdb_reviews", split=["train", "test"], as_supervised=True)

        train_texts, train_labels = [], []
        for text, label in tfds.as_numpy(train_data):
            train_texts.append(text.decode("utf-8"))
            train_labels.append(label)

        test_texts, test_labels = [], []
        for text, label in tfds.as_numpy(test_data):
            test_texts.append(text.decode("utf-8"))
            test_labels.append(label)

        # Tokenize
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(train_texts)
        self.word_index = self.tokenizer.word_index

        # Sequences
        train_sequences = self.tokenizer.texts_to_sequences(train_texts)
        test_sequences = self.tokenizer.texts_to_sequences(test_texts)

        train_padded = pad_sequences(train_sequences, maxlen=self.input_length, padding="post", truncating="post")
        test_padded = pad_sequences(test_sequences, maxlen=self.input_length, padding="post", truncating="post")

        # If GloVe path provided, build embedding matrix
        if self.glove_path:
            self.embedding_matrix = self._build_embedding_matrix()

        return (train_padded, np.array(train_labels)), (test_padded, np.array(test_labels))

    # --------------------------
    # GloVe Support
    # --------------------------
    def _load_glove_embeddings(self):
        embeddings_index = {}
        print(f"Loading GloVe embeddings from {self.glove_path}...")
        with open(self.glove_path, encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                embeddings_index[word] = vector
        print(f"Loaded {len(embeddings_index)} word vectors from GloVe.")
        return embeddings_index

    def _build_embedding_matrix(self):
        embeddings_index = self._load_glove_embeddings()
        embedding_matrix = np.random.normal(size=(self.vocab_size, self.embedding_dim))
        for word, i in self.word_index.items():
            if i < self.vocab_size:
                vec = embeddings_index.get(word)
                if vec is not None:
                    embedding_matrix[i] = vec
        return embedding_matrix

    # --------------------------
    # Model Definition
    # --------------------------
    def _build_model(self):
        if self.embedding_matrix is not None:
            embedding_layer = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                weights=[self.embedding_matrix],
                trainable=self.trainable
            )
        else:
            embedding_layer = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim
            )

        model = Sequential([
            embedding_layer,
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.5),
            GlobalAveragePooling1D(),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    # --------------------------
    # Training & Evaluation
    # --------------------------
    def train(self, epochs=10, batch_size=128):
        history = self.model.fit(
            self.train_data,
            self.train_labels,
            validation_data=(self.test_data, self.test_labels),
            epochs=epochs,
            batch_size=batch_size
        )
        loss, acc = self.model.evaluate(self.test_data, self.test_labels, verbose=0)
        print(f"Accuracy: {acc*100:.2f}%")

        # Save trained model
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")        
        return history

    # --------------------------
    # Prediction
    # --------------------------
    def predict(self, text):
        if self.use_raw and self.tokenizer:
            seq = self.tokenizer.texts_to_sequences([text])
            seq = pad_sequences(seq, maxlen=self.input_length, padding="post", truncating="post")
        else:
            raise ValueError("Prediction only supported in raw text mode right now.")

        prob = self.model.predict(seq, verbose=0)[0][0]
        sentiment = "Positive" if prob > 0.5 else "Negative"
        return {"text": text, "sentiment": sentiment, "probability": float(prob)}

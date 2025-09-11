# sentiment_analyzer.py
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import load_model


class SentimentAnalyzer:
    def __init__(self, vocab_size=30000, embedding_dim=128, input_length=200, model_file="sentiment_model.keras"):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.word_index = tf.keras.datasets.imdb.get_word_index()

        
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

    def _build_model(self):
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim,
                    input_length=self.input_length),
            
            # Use a Bidirectional LSTM to capture context from both directions
            Bidirectional(LSTM(64, return_sequences=True)),
            
            # Apply Dropout to prevent overfitting
            Dropout(0.5),
            
            # Use Global Average Pooling to summarize the sequence for classification
            GlobalAveragePooling1D(),
            
            # Add another Dense layer with ReLU for non-linearity
            Dense(32, activation='relu'),
            
            # Final Dense layer for binary classification
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
            metrics=['accuracy']
        )
        return model

    def encode_text(self, text):
        tokens = [self.word_index.get(word.lower(), 2) for word in text.split()]  # 2 = OOV
        return pad_sequences([tokens], maxlen=self.input_length, padding='post', truncating='post')

    def predict(self, text):
        seq = self.encode_text(text)
        pred_prob = self.model.predict(seq, verbose=0)[0][0]
        sentiment = "Positive" if pred_prob > 0.5 else "Negative"
        return {"text": text, "sentiment": sentiment, "probability": float(pred_prob)}

    def train(self, num_words=20000, max_length=200, epochs=5, batch_size=128):
        #(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(None)
        (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=num_words)

        padded_train = pad_sequences(train_data, maxlen=max_length, padding='post', truncating='post')
        padded_test = pad_sequences(test_data, maxlen=max_length, padding='post', truncating='post')

        history = self.model.fit(
            padded_train, np.array(train_labels),
            validation_data=(padded_test, np.array(test_labels)),
            epochs=epochs,
            batch_size=batch_size
        )

        # Save trained model
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

        return history

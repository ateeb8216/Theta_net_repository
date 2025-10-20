import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.metrics import compute_metrics
import spacy
import nltk
from sklearn.preprocessing import LabelBinarizer

# Load spaCy and NLTK resources
nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Dataset placeholders
DATASET_PATHS = {
    'bbc': 'path_to_bbc.csv',
    'uci': 'path_to_uci.csv',
    'hybrid': 'path_to_hybrid.csv'
}

MAX_SEQUENCE_LENGTH = 300
EMBED_DIM = 50
NUM_CLASSES = 9

# ------------------ Preprocessing ------------------
def preprocess_text(texts):
    cleaned = []
    for doc in nlp.pipe(texts, disable=['parser', 'tagger']):
        tokens = [token.lemma_.lower() for token in doc 
                  if token.is_alpha and token.text.lower() not in stop_words]
        cleaned.append(' '.join(tokens))
    return cleaned

# Load dataset
def load_dataset(path):
    df = pd.read_csv(path)
    texts = preprocess_text(df['text'].tolist())
    labels = LabelBinarizer().fit_transform(df['label'].tolist())
    return texts, labels

# ------------------ HLNN Model ------------------
def build_hlnn_model(max_seq_len=MAX_SEQUENCE_LENGTH, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES):
    # Input Layer
    inputs = layers.Input(shape=(max_seq_len,), dtype='int32')

    # Embedding Layer (trainable, can be extended with pre-trained embeddings)
    x = layers.Embedding(input_dim=50000, output_dim=embed_dim, input_length=max_seq_len)(inputs)

    # CNN path
    cnn_out = layers.Conv1D(filters=64, kernel_size=5, activation='relu')(x)
    cnn_out = layers.MaxPooling1D(pool_size=2)(cnn_out)
    cnn_out = layers.Flatten()(cnn_out)

    # LSTM path
    lstm_out = layers.LSTM(80, return_sequences=False)(x)

    # Concatenate
    merged = layers.concatenate([cnn_out, lstm_out])

    # Dense and Dropout
    dense = layers.Dense(64, activation='relu')(merged)
    dropout = layers.Dropout(0.4)(dense)

    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(dropout)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# ------------------ Attention Visualizations ------------------
def plot_attention(weights, tokens, title='Token Attention'):
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(tokens)), weights)
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title(title)
    plt.show()

# ------------------ Compile Model ------------------
model = build_hlnn_model()
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

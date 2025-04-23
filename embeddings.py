import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# This function prepares tokenized text data and an embedding matrix for a CNN model
def get_embeddings(
    cleaned_filepath="cleaned_suicide_dataset.csv",  # Path to the preprocessed dataset
    max_vocab_size=10000,  # Max number of unique words to keep in tokenizer
    embedding_dim=100,     # Dimensionality of GloVe word vectors
    max_sequence_length=200,  # Max length of sequences (will be padded if shorter)
    glove_path="glove.6B.100d.txt"  # Path to the GloVe embedding file
):
    # Step 1: Load the cleaned dataset
    df = pd.read_csv(cleaned_filepath)
    
    # Remove rows where 'clean_text' is NaN or not a string
    df = df.dropna(subset=['clean_text'])  # Drop rows where clean_text is NaN
    df['clean_text'] = df['clean_text'].astype(str)  # Ensure everything is string


    # Step 2: Initialize tokenizer for text data
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    # Fit the tokenizer on the cleaned text column to build word index
    tokenizer.fit_on_texts(df['clean_text'])

    # Convert each text into a sequence of integers based on the tokenizer
    sequences = tokenizer.texts_to_sequences(df['clean_text'])

    # Pad all sequences to the same length for neural network input
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    # Retrieve the word index mapping (word -> integer)
    word_index = tokenizer.word_index
    print(f"✅ Tokenizer created. Number of unique tokens: {len(word_index)}")

    # Step 3: Load GloVe word embeddings into a dictionary
    embeddings_index = {}  # Dictionary: word -> GloVe vector
    if not os.path.exists(glove_path):
        raise FileNotFoundError(f"GloVe file not found at path: {glove_path}")

    # Read the GloVe file line by line
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]  # First item is the word
            vector = np.asarray(values[1:], dtype='float32')  # Remaining items are the embedding
            embeddings_index[word] = vector

    print(f"✅ Loaded {len(embeddings_index)} word vectors from GloVe.")

    # Step 4: Create the embedding matrix
    # Initialize matrix of zeros: rows = vocab size, cols = embedding dimension
    embedding_matrix = np.zeros((max_vocab_size, embedding_dim))

    # Fill the embedding matrix with GloVe vectors for known words
    for word, i in tokenizer.word_index.items():
        if i < max_vocab_size:
            embedding_vector = embeddings_index.get(word)  # Get GloVe vector
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector  # Insert vector into matrix
            # Else: if word not found in GloVe, the row remains zeros

    print(f"✅ Embedding matrix created. Shape: {embedding_matrix.shape}")

    # Return all prepared components
    return tokenizer, padded_sequences, embedding_matrix, df


# Quick test
if __name__ == "__main__":
    # Run this file directly to test the embedding process
    tokenizer, padded_sequences, embedding_matrix = get_embeddings()

    # Print out the shape of padded sequences and an example
    print("📏 Padded sequences shape:", padded_sequences.shape)
    print("🔁 Example padded sequence:", padded_sequences[0])

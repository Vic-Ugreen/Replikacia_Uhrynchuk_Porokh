import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# This function builds and trains a CNN model for suicide risk detection
def build_and_train_cnn(padded_sequences, labels, embedding_matrix,
                        max_vocab_size=10000,
                        embedding_dim=100,
                        max_sequence_length=200,
                        batch_size=64,
                        epochs=5,
                        test_size=0.2,
                        random_state=42):
    
    # Step 1: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels, test_size=test_size, random_state=random_state)

    # Step 2: Define the CNN model architecture
    model = Sequential()
    
    # Embedding layer initialized with the GloVe matrix (non-trainable)
    model.add(Embedding(input_dim=max_vocab_size,
                        output_dim=embedding_dim,
                        input_length=max_sequence_length,
                        weights=[embedding_matrix],
                        trainable=False))  # We freeze the pre-trained embeddings

    # 1D Convolutional layer
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))

    # Global max pooling
    model.add(GlobalMaxPooling1D())

    # Final dense layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Step 3: Compile the model
    model.compile(optimizer=Adam(),
              loss=BinaryCrossentropy(),
              metrics=[BinaryAccuracy(), Precision(), Recall()])
    
    # Added early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',        # Monitored value
        patience=2,                # How many epochs "to wait" before stopping
        restore_best_weights=True  # Return to the best version
    )

    # Step 4: Train the model
    history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop])

    # Step 5: Evaluate the model on test data
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=2)
    print(f"✅ Evaluation completed:\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Plot training & validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot training & validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['binary_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='Val Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

if __name__ == "__main__":
    from embeddings import get_embeddings
    import numpy as np

    # Load data from previous step
    tokenizer, padded_sequences, embedding_matrix, df = get_embeddings()

    # Use the same df to get labels (synchronized with padded_sequences)
    df['label'] = df['class'].apply(lambda x: 1 if str(x).strip().lower() == 'suicide' else 0)
    labels = np.array(df['label'])

    # Train and evaluate the CNN model
    build_and_train_cnn(padded_sequences, labels, embedding_matrix)


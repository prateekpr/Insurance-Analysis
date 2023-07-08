import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def preprocess_data(data, training=False, training_columns=None):
    # Perform encoding or any other necessary preprocessing steps
    # Assuming you have a column named "files_modified"
    
    # Example encoding: One-Hot Encoding
    if training:
        encoded_data = pd.get_dummies(data["files_modified"])
        training_columns = encoded_data.columns
    else:
        encoded_data = pd.get_dummies(data["files_modified"])
        encoded_data = encoded_data.reindex(columns=training_columns, fill_value=0)

    return encoded_data

def train_autoencoder(training_data, save_path):
    # Prepare the training dataset
    X_train = training_data  # Assuming the data is already preprocessed and encoded
    # Normalize the training data
    X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))

    # Define the architecture of the autoencoder
    input_dim = X_train.shape[1]
    encoding_dim = 64

    input_layer = tf.keras.Input(shape=(input_dim,))
    encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoder)

    # Create the autoencoder model
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder on the training set
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32)

    # Save the trained model
    autoencoder.save(save_path)

def test_autoencoder(test_data, saved_model_path, training_columns):
    # Load the saved model
    autoencoder = tf.keras.models.load_model(saved_model_path)

    # Prepare the test data
    X_test = test_data  # Assuming the data is already preprocessed and encoded
    # Normalize the test data
    X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))

    # Use the autoencoder to reconstruct the test data
    reconstructed_data = autoencoder.predict(X_test)

    # Calculate the reconstruction error as the mean squared error between the original and reconstructed data
    reconstruction_error = np.mean(np.square(X_test - reconstructed_data), axis=1)

    # Calculate the flakiness percentage based on the reconstruction error
    flakiness_percentage = reconstruction_error * 100

    # Add the flakiness percentage to the test data
    test_data['flakiness_percentage'] = flakiness_percentage

    return test_data

# Example usage
# Preprocessing
training_data = pd.read_csv("training_data.csv")  # Load your training data
encoded_training_data = preprocess_data(training_data, training=True)

test_data = pd.read_csv("test_data.csv")  # Load your test data
encoded_test_data = preprocess_data(test_data, training_columns=encoded_training_data.columns)

# Training
train_autoencoder(encoded_training_data, save_path="saved_model.h5")

# Testing
flaky_test_cases = test_autoencoder(encoded_test_data, saved_model_path="saved_model.h5", training_columns=encoded_training_data.columns)
print("Flakiness percentages:")
print(flaky_test_cases)

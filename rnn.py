import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import mediapipe as mp
import tensorflow as tf
import numpy as np
import os
from objdetect import process_video
from sklearn.model_selection import train_test_split

# Define RNN parameters
input_size = 8 
hidden_size = 64
num_layers = 2
output_size = 4  # Four pull-up phases


# RNN Architecture

class PullUpDataset(tf.keras.utils.Sequence):
    def __init__(self, sequences, labels, batch_size):
        self.sequences = np.array(sequences)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.ceil(len(self.sequences) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_sequences = self.sequences[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_sequences, batch_labels

class PullUpRNN(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PullUpRNN, self).__init__()
        self.lstm_layers = [tf.keras.layers.LSTM(hidden_size, return_sequences=True) for _ in range(num_layers)]
        self.fc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size))

    def call(self, x):
        for lstm in self.lstm_layers:
            x = lstm(x)
        x = self.fc(x) 
        return x


def train_model(model, train_loader, val_loader, num_epochs, device):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    history = model.fit(train_loader, validation_data=val_loader, epochs=num_epochs, verbose=1)
    
    # Save the best model
    model.save('best_pullup_model.h5')

    return history


def collect_training_data(video_dir):
    all_sequences = []
    all_labels = []
    
    for video_file in os.listdir(video_dir):
        if video_file.endswith(('.mp4', '.mov', '.avi')):
            video_path = os.path.join(video_dir, video_file)
            sequences, labels = process_video(video_path, 10)
            all_sequences.extend(sequences)
            all_labels.extend(labels)
    
    return np.array(all_sequences), np.array(all_labels)

def main():
    # Hyperparameters
    input_size = 8  # 4 landmarks with x,y coordinates
    hidden_size = 64
    num_layers = 2
    output_size = 4  # Four pull-up phases
    batch_size = 32
    num_epochs = 25
    
    # Collect data from multiple videos
    video_dir = '/Users/ethandao/Desktop/opencvproj/data'  # Directory containing our sample videos
    sequences, labels = collect_training_data(video_dir)
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = PullUpDataset(X_train, y_train, batch_size)
    val_dataset = PullUpDataset(X_val, y_val, batch_size)
    
    # Initialize model
    model = PullUpRNN(input_size, hidden_size, num_layers, output_size)

    # Train the model
    train_model(model, train_dataset, val_dataset, num_epochs, None)
    
    print("Training completed!")

if __name__ == "__main__":
    main()




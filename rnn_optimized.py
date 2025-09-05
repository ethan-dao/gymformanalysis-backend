import os
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from sklearn.model_selection import train_test_split
import asyncio
import concurrent.futures

logger = logging.getLogger(__name__)

class OptimizedPullUpRNN:
    """
    Optimized RNN model for pull-up phase detection with TensorFlow optimizations
    """
    
    def __init__(self, input_size: int = 8, hidden_size: int = 64, 
                 num_layers: int = 2, output_size: int = 4, dropout_rate: float = 0.3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.is_compiled = False
        
        # Thread pool for async operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    
    def build_model(self) -> keras.Model:
        """Build optimized LSTM model with regularization and normalization"""
        
        # Input layer
        inputs = keras.Input(shape=(None, self.input_size), name='pose_sequences')
        
        # Normalization layer
        x = layers.LayerNormalization()(inputs)
        
        # LSTM layers with dropout and residual connections
        for i in range(self.num_layers):
            lstm_out = layers.LSTM(
                self.hidden_size,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                kernel_regularizer=keras.regularizers.l2(0.001),
                name=f'lstm_{i}'
            )(x)
            
            # Add residual connection if dimensions match
            if i > 0 and x.shape[-1] == lstm_out.shape[-1]:
                x = layers.Add()([x, lstm_out])
            else:
                x = lstm_out
            
            # Layer normalization after LSTM
            x = layers.LayerNormalization()(x)
        
        # Attention mechanism for better sequence modeling
        attention = layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=self.hidden_size // 4,
            name='attention'
        )(x, x)
        
        # Combine LSTM output with attention
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization()(x)
        
        # Dense layers with dropout
        x = layers.TimeDistributed(
            layers.Dense(self.hidden_size // 2, activation='relu')
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = layers.TimeDistributed(
            layers.Dense(self.output_size, activation='softmax', name='phase_predictions')
        )(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='PullUpRNN')
        
        return model
    
    def compile_model(self, learning_rate: float = 0.001, 
                     label_smoothing: float = 0.1) -> None:
        """Compile model with optimized settings"""
        
        if self.model is None:
            self.model = self.build_model()
        
        # Use mixed precision for better performance
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        # Custom loss with label smoothing
        loss = keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing,
            from_logits=False
        )
        
        # Adam optimizer with learning rate scheduling
        optimizer = optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        # Compile with mixed precision
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', 'sparse_categorical_accuracy'],
            run_eagerly=False  # Use graph mode for better performance
        )
        
        self.is_compiled = True
        logger.info("Model compiled with optimizations")
    
    def get_callbacks(self, model_save_path: str) -> list:
        """Get training callbacks for optimization"""
        
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpointing
            callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),
            
            # Custom metrics callback
            MetricsCallback()
        ]
        
        return callbacks_list
    
    def train_optimized(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       batch_size: int = 32, epochs: int = 50,
                       model_save_path: str = 'best_pullup_model.h5') -> keras.callbacks.History:
        """Train model with optimizations"""
        
        if not self.is_compiled:
            self.compile_model()
        
        # Convert labels to categorical if needed
        if len(y_train.shape) == 2:  # If not one-hot encoded
            y_train_cat = keras.utils.to_categorical(y_train, self.output_size)
            y_val_cat = keras.utils.to_categorical(y_val, self.output_size)
        else:
            y_train_cat = y_train
            y_val_cat = y_val
        
        # Data augmentation for sequences
        train_dataset = self._create_dataset(X_train, y_train_cat, batch_size, shuffle=True)
        val_dataset = self._create_dataset(X_val, y_val_cat, batch_size, shuffle=False)
        
        # Get callbacks
        callback_list = self.get_callbacks(model_save_path)
        
        logger.info(f"Starting training with {len(X_train)} training samples")
        
        # Train model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1,
            use_multiprocessing=True,
            workers=4
        )
        
        # Convert to TensorFlow Lite for mobile deployment
        self._convert_to_tflite(model_save_path.replace('.h5', '.tflite'))
        
        return history
    
    def _create_dataset(self, X: np.ndarray, y: np.ndarray, 
                       batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        """Create optimized TensorFlow dataset"""
        
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        # Batch and prefetch for performance
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Add data augmentation for training
        if shuffle:
            dataset = dataset.map(
                self._augment_sequence,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        return dataset
    
    def _augment_sequence(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply data augmentation to sequences"""
        
        # Add small random noise
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.01)
        x_aug = x + noise
        
        # Random time shift (small)
        shift = tf.random.uniform([], -2, 3, dtype=tf.int32)
        x_aug = tf.roll(x_aug, shift, axis=1)
        y_aug = tf.roll(y, shift, axis=1)
        
        return x_aug, y_aug
    
    def predict_optimized(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Optimized prediction with batching"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_optimized_model() first.")
        
        # Create dataset for efficient batching
        dataset = tf.data.Dataset.from_tensor_slices(X)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Predict in batches
        predictions = self.model.predict(dataset, verbose=0)
        
        return predictions
    
    async def predict_async(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Asynchronous prediction"""
        
        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(
            self.executor,
            self.predict_optimized,
            X, batch_size
        )
        
        return predictions
    
    def _convert_to_tflite(self, tflite_path: str) -> None:
        """Convert model to TensorFlow Lite for mobile deployment"""
        
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            
            # Optimization settings
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            # Convert model
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"TensorFlow Lite model saved to {tflite_path}")
            
        except Exception as e:
            logger.warning(f"Failed to convert to TFLite: {e}")
    
    @classmethod
    def load_optimized_model(cls, model_path: str) -> 'OptimizedPullUpRNN':
        """Load pre-trained model with optimizations"""
        
        instance = cls()
        
        try:
            # Load model
            instance.model = keras.models.load_model(model_path)
            instance.is_compiled = True
            
            # Apply optimizations
            instance._apply_runtime_optimizations()
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Create and compile new model as fallback
            instance.compile_model()
        
        return instance
    
    def _apply_runtime_optimizations(self) -> None:
        """Apply runtime optimizations to loaded model"""
        
        # Enable XLA compilation
        self.model.compile(
            optimizer=self.model.optimizer,
            loss=self.model.loss,
            metrics=self.model.metrics,
            run_eagerly=False,
            jit_compile=True  # Enable XLA
        )
        
        logger.info("Runtime optimizations applied")
    
    def save_model(self, filepath: str) -> None:
        """Save model with metadata"""
        
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save in SavedModel format for better optimization
        self.model.save(filepath, save_format='tf')
        
        # Also save in H5 format for compatibility
        h5_path = filepath.replace('.tf', '.h5') if filepath.endswith('.tf') else filepath + '.h5'
        self.model.save(h5_path, save_format='h5')
        
        logger.info(f"Model saved to {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            "model_name": self.model.name,
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "total_params": self.model.count_params(),
            "trainable_params": sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
            "layers": len(self.model.layers),
            "optimizer": type(self.model.optimizer).__name__ if self.model.optimizer else None,
            "compiled": self.is_compiled
        }


class MetricsCallback(keras.callbacks.Callback):
    """Custom callback for additional metrics"""
    
    def on_epoch_end(self, epoch, logs=None):
        """Calculate and log additional metrics"""
        logs = logs or {}
        
        # Log phase-wise accuracy if available
        if 'val_sparse_categorical_accuracy' in logs:
            phase_acc = logs['val_sparse_categorical_accuracy']
            logger.info(f"Epoch {epoch + 1} - Phase Accuracy: {phase_acc:.4f}")


class PullUpDataGenerator(keras.utils.Sequence):
    """Optimized data generator for large datasets"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, 
                 batch_size: int = 32, shuffle: bool = True, augment: bool = False):
        self.sequences = sequences
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(sequences))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self) -> int:
        return int(np.ceil(len(self.sequences) / self.batch_size))
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.indices))
        
        batch_indices = self.indices[start_idx:end_idx]
        
        X = self.sequences[batch_indices]
        y = self.labels[batch_indices]
        
        if self.augment:
            X, y = self._apply_augmentation(X, y)
        
        return X, y
    
    def _apply_augmentation(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation"""
        
        # Add noise
        noise_factor = 0.01
        X_aug = X + np.random.normal(0, noise_factor, X.shape)
        
        # Time shifting
        shift_range = 2
        for i in range(len(X_aug)):
            shift = np.random.randint(-shift_range, shift_range + 1)
            if shift != 0:
                X_aug[i] = np.roll(X_aug[i], shift, axis=0)
                y[i] = np.roll(y[i], shift, axis=0)
        
        return X_aug, y
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


# Utility functions for training pipeline
def prepare_training_data(video_dir: str, sequence_length: int = 10, 
                         test_size: float = 0.2) -> Tuple[np.ndarray, ...]:
    """Prepare training data from video directory"""
    
    from objdetect_async import AsyncPullUpDetector
    
    detector = AsyncPullUpDetector()
    all_sequences = []
    all_labels = []
    
    # Process all videos in directory
    for video_file in os.listdir(video_dir):
        if video_file.lower().endswith(('.mp4', '.mov', '.avi')):
            video_path = os.path.join(video_dir, video_file)
            logger.info(f"Processing {video_file}")
            
            try:
                sequences, labels = detector._process_video_sync(video_path, sequence_length)
                all_sequences.extend(sequences)
                all_labels.extend(labels)
            except Exception as e:
                logger.error(f"Error processing {video_file}: {e}")
    
    # Convert to numpy arrays
    X = np.array(all_sequences)
    y = np.array(all_labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=None
    )
    
    logger.info(f"Training data prepared: {len(X_train)} train, {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test


def train_pullup_model(video_dir: str, model_save_path: str = 'pullup_model_optimized') -> OptimizedPullUpRNN:
    """Complete training pipeline"""
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_training_data(video_dir)
    
    # Create and train model
    model = OptimizedPullUpRNN()
    model.compile_model()
    
    # Train
    history = model.train_optimized(
        X_train, y_train, X_test, y_test,
        batch_size=32,
        epochs=50,
        model_save_path=model_save_path + '.h5'
    )
    
    # Save final model
    model.save_model(model_save_path)
    
    # Print model info
    info = model.get_model_info()
    logger.info(f"Model training completed: {info}")
    
    return model


if __name__ == "__main__":
    # Example usage
    video_directory = "/path/to/your/video/data"
    model = train_pullup_model(video_directory)
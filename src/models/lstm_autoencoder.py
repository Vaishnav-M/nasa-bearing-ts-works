"""
LSTM Autoencoder for Time Series Anomaly Detection

This module implements a deep learning approach using LSTM (Long Short-Term Memory)
autoencoders for anomaly detection in time series data.

The autoencoder learns to reconstruct normal patterns. Anomalies produce
higher reconstruction errors, which are used for detection.

Key Features:
- GPU acceleration with TensorFlow
- Sequence-based learning (captures temporal dependencies)
- Early stopping and model checkpointing
- Reconstruction error-based anomaly scoring

Author: Vaishnav M
Date: November 2025
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
import logging
import time
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class LSTMAutoencoder:
    """
    LSTM Autoencoder for time series anomaly detection.
    
    The model consists of:
    1. Encoder: LSTM layers that compress input sequences
    2. Decoder: LSTM layers that reconstruct the original sequences
    
    Anomalies are detected by comparing reconstruction error to a threshold.
    """
    
    def __init__(self, sequence_length=50, n_features=4, 
                 encoding_dim=16, lstm_units=[32, 16], 
                 learning_rate=0.001, dropout_rate=0.2):
        """
        Initialize LSTM Autoencoder.
        
        Simplified architecture to reduce overfitting:
        - Reduced LSTM units: [32, 16] instead of [64, 32]
        - Reduced encoding dimension: 16 instead of 32
        - This prevents the model from memorizing patterns too well
        
        Args:
            sequence_length (int): Length of input sequences (time steps)
            n_features (int): Number of features per time step
            encoding_dim (int): Dimension of the encoded representation
            lstm_units (list): Number of units in each LSTM layer [layer1, layer2, ...]
            learning_rate (float): Learning rate for Adam optimizer
            dropout_rate (float): Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.encoding_dim = encoding_dim
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.threshold = None
        self.training_time = None
        self.history = None
        
        logger.info(f"Initialized LSTM Autoencoder:")
        logger.info(f"  - Sequence Length: {sequence_length}")
        logger.info(f"  - Features: {n_features}")
        logger.info(f"  - Encoding Dimension: {encoding_dim}")
        logger.info(f"  - LSTM Units: {lstm_units}")
        logger.info(f"  - Learning Rate: {learning_rate}")
        logger.info(f"  - Dropout Rate: {dropout_rate}")
        
        # Check GPU availability
        self._check_gpu()
        
        # Build the model
        self._build_model()
    
    def _check_gpu(self):
        """Check if GPU is available and log device information."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU Available: {len(gpus)} device(s)")
            for gpu in gpus:
                logger.info(f"  - {gpu}")
                # Enable memory growth to avoid OOM errors
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logger.warning(f"Could not set memory growth: {e}")
        else:
            logger.warning("No GPU found. Training will use CPU (slower)")
    
    def _build_model(self):
        """Build the LSTM Autoencoder architecture."""
        logger.info("Building LSTM Autoencoder model...")
        
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # ============= ENCODER =============
        encoded = inputs
        
        # First LSTM layer (return sequences for stacking)
        encoded = layers.LSTM(
            self.lstm_units[0],
            activation='tanh',
            return_sequences=True if len(self.lstm_units) > 1 else False,
            name='encoder_lstm_1'
        )(encoded)
        encoded = layers.Dropout(self.dropout_rate)(encoded)
        
        # Additional LSTM layers if specified
        for i, units in enumerate(self.lstm_units[1:], start=2):
            return_seq = i < len(self.lstm_units)  # Return sequences except for last layer
            encoded = layers.LSTM(
                units,
                activation='tanh',
                return_sequences=return_seq,
                name=f'encoder_lstm_{i}'
            )(encoded)
            encoded = layers.Dropout(self.dropout_rate)(encoded)
        
        # Bottleneck (compressed representation)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='bottleneck')(encoded)
        
        # ============= DECODER =============
        # Repeat the encoded vector to match sequence length
        decoded = layers.RepeatVector(self.sequence_length)(encoded)
        
        # Reverse LSTM layers for decoder
        for i, units in enumerate(reversed(self.lstm_units), start=1):
            decoded = layers.LSTM(
                units,
                activation='tanh',
                return_sequences=True,
                name=f'decoder_lstm_{i}'
            )(decoded)
            decoded = layers.Dropout(self.dropout_rate)(decoded)
        
        # Output layer (reconstruct original input)
        outputs = layers.TimeDistributed(
            layers.Dense(self.n_features),
            name='output'
        )(decoded)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='LSTM_Autoencoder')
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',  # Mean Squared Error for reconstruction
            metrics=['mae']  # Mean Absolute Error
        )
        
        logger.info("Model architecture built successfully")
        logger.info(f"Total parameters: {self.model.count_params():,}")
    
    def summary(self):
        """Print model summary."""
        self.model.summary()
    
    def create_sequences(self, data, labels=None):
        """
        Create sequences for LSTM input.
        
        Args:
            data (np.ndarray): Input data of shape (samples, features)
            labels (np.ndarray): Labels of shape (samples,) - optional
        
        Returns:
            np.ndarray: Sequences of shape (n_sequences, sequence_length, features)
            np.ndarray: Labels for sequences (if provided)
        """
        sequences = []
        sequence_labels = []
        
        for i in range(len(data) - self.sequence_length + 1):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
            
            if labels is not None:
                # Label is 1 if ANY point in the sequence is anomalous
                seq_label = int(np.any(labels[i:i + self.sequence_length]))
                sequence_labels.append(seq_label)
        
        sequences = np.array(sequences)
        
        if labels is not None:
            sequence_labels = np.array(sequence_labels)
            return sequences, sequence_labels
        
        return sequences
    
    def fit(self, X_train, epochs=50, batch_size=32, validation_split=0.2,
            early_stopping_patience=10, verbose=1):
        """
        Train the autoencoder on normal data.
        
        Args:
            X_train (np.ndarray): Training sequences of shape (samples, seq_len, features)
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            early_stopping_patience (int): Epochs to wait before early stopping
            verbose (int): Verbosity mode (0, 1, or 2)
        
        Returns:
            self: Fitted model
        """
        logger.info(f"Training LSTM Autoencoder on {X_train.shape[0]} sequences")
        logger.info(f"Sequence shape: ({self.sequence_length}, {self.n_features})")
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model (autoencoder tries to reconstruct its input)
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, X_train,  # Input = Output for autoencoder
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callback_list,
            verbose=verbose
        )
        
        self.training_time = time.time() - start_time
        
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        logger.info(f"Final training loss: {self.history.history['loss'][-1]:.6f}")
        logger.info(f"Final validation loss: {self.history.history['val_loss'][-1]:.6f}")
        
        return self
    
    def calculate_reconstruction_error(self, X):
        """
        Calculate reconstruction error for sequences.
        
        Args:
            X (np.ndarray): Input sequences
        
        Returns:
            np.ndarray: Reconstruction errors (MSE per sequence)
        """
        # Reconstruct sequences
        reconstructed = self.model.predict(X, verbose=0)
        
        # Calculate MSE for each sequence
        mse = np.mean(np.square(X - reconstructed), axis=(1, 2))
        
        return mse
    
    def set_threshold(self, X_normal, percentile=95):
        """
        Set anomaly detection threshold based on normal data.
        
        Args:
            X_normal (np.ndarray): Normal sequences for threshold calculation
            percentile (float): Percentile of reconstruction errors to use as threshold
        
        Returns:
            float: Threshold value
        """
        logger.info(f"Calculating threshold using {percentile}th percentile of normal data...")
        
        # Calculate reconstruction errors on normal data
        errors = self.calculate_reconstruction_error(X_normal)
        
        # Set threshold at specified percentile
        self.threshold = np.percentile(errors, percentile)
        
        logger.info(f"Threshold set to: {self.threshold:.6f}")
        logger.info(f"Min error: {errors.min():.6f}, Max error: {errors.max():.6f}")
        logger.info(f"Mean error: {errors.mean():.6f}, Std: {errors.std():.6f}")
        
        return self.threshold
    
    def predict(self, X, threshold=None):
        """
        Predict anomalies based on reconstruction error.
        
        Args:
            X (np.ndarray): Input sequences
            threshold (float): Threshold for anomaly detection (uses self.threshold if None)
        
        Returns:
            np.ndarray: Binary predictions (1 = anomaly, 0 = normal)
        """
        if threshold is None:
            if self.threshold is None:
                raise ValueError("Threshold not set. Call set_threshold() first or provide threshold.")
            threshold = self.threshold
        
        # Calculate reconstruction errors
        errors = self.calculate_reconstruction_error(X)
        
        # Classify as anomaly if error > threshold
        predictions = (errors > threshold).astype(int)
        
        logger.info(f"Detected {np.sum(predictions)} anomalies out of {len(predictions)} sequences "
                   f"({np.mean(predictions):.2%})")
        
        return predictions
    
    def evaluate(self, X_test, y_test, threshold=None):
        """
        Evaluate model performance.
        
        Args:
            X_test (np.ndarray): Test sequences
            y_test (np.ndarray): True labels
            threshold (float): Threshold for anomaly detection
        
        Returns:
            dict: Performance metrics
        """
        # Get predictions
        y_pred = self.predict(X_test, threshold)
        
        # Get reconstruction errors for ROC-AUC
        scores = self.calculate_reconstruction_error(X_test)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        
        try:
            roc_auc = roc_auc_score(y_test, scores)
        except:
            roc_auc = 0.0
        
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'training_time': self.training_time,
            'threshold': self.threshold if threshold is None else threshold,
            'predictions': y_pred,
            'scores': scores
        }
        
        logger.info(f"Evaluation: Precision={precision:.4f}, Recall={recall:.4f}, "
                   f"F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """Save model to disk."""
        self.model.save(filepath)
        
        # Save threshold separately (handle both Path and str)
        filepath_str = str(filepath)
        threshold_file = filepath_str.replace('.h5', '_threshold.npy')
        np.save(threshold_file, self.threshold)
        
        logger.info(f"Model saved to {filepath}")
        logger.info(f"Threshold saved to {threshold_file}")
    
    def load_model(self, filepath):
        """Load model from disk."""
        self.model = keras.models.load_model(filepath)
        
        # Load threshold (handle both Path and str)
        filepath_str = str(filepath)
        threshold_file = filepath_str.replace('.h5', '_threshold.npy')
        if os.path.exists(threshold_file):
            self.threshold = np.load(threshold_file)
            logger.info(f"Threshold loaded: {self.threshold:.6f}")
        
        logger.info(f"Model loaded from {filepath}")


def main():
    """
    Example usage of LSTM Autoencoder.
    """
    logger.info("Testing LSTM Autoencoder for Anomaly Detection")
    
    # Load raw sensor data (not scaled features - we want raw sensors for LSTM)
    data_path = 'data/raw/synthetic_sensor_data_with_labels.csv'
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Use only raw sensor data
    sensor_cols = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4']
    X = df[sensor_cols].values
    y = df['anomaly'].values
    
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Anomaly ratio: {np.mean(y):.2%}")
    
    # Normalize data (important for neural networks)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize LSTM Autoencoder
    lstm_ae = LSTMAutoencoder(
        sequence_length=50,
        n_features=len(sensor_cols),
        encoding_dim=16,
        lstm_units=[64, 32],
        learning_rate=0.001,
        dropout_rate=0.2
    )
    
    # Print model summary
    lstm_ae.summary()
    
    # Create sequences
    logger.info("\nCreating sequences...")
    X_seq, y_seq = lstm_ae.create_sequences(X_scaled, y)
    logger.info(f"Created {len(X_seq)} sequences of shape {X_seq.shape}")
    
    # Split data (80% train, 20% test)
    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    # Filter training data to only normal samples (unsupervised learning)
    X_train_normal = X_train[y_train == 0]
    logger.info(f"\nTraining on {len(X_train_normal)} normal sequences")
    logger.info(f"Test set: {len(X_test)} sequences ({np.mean(y_test):.2%} anomalies)")
    
    # Train model
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING LSTM AUTOENCODER")
    logger.info("=" * 60)
    
    lstm_ae.fit(
        X_train_normal,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        early_stopping_patience=10,
        verbose=1
    )
    
    # Set threshold using normal validation data
    logger.info("\n" + "=" * 60)
    logger.info("SETTING THRESHOLD")
    logger.info("=" * 60)
    
    lstm_ae.set_threshold(X_train_normal, percentile=95)
    
    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    
    metrics = lstm_ae.evaluate(X_test, y_test)
    
    # Save model
    lstm_ae.save_model('outputs/models/lstm_autoencoder.h5')
    
    logger.info("\n✓ Testing complete!")


if __name__ == '__main__':
    main()

"""
Statistical Anomaly Detection Models

This module implements classical machine learning approaches for anomaly detection:
- Isolation Forest: Tree-based anomaly detection
- Local Outlier Factor (LOF): Density-based anomaly detection

Both models are unsupervised and work by identifying points that deviate
from the normal data distribution.

Author: Vaishnav M
Date: November 2025
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
import joblib
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Isolation Forest for anomaly detection.
    
    Isolation Forest works by randomly selecting features and split values,
    isolating anomalies faster than normal points (anomalies are "easier to isolate").
    
    Best for: High-dimensional data, fast training, good with outliers
    """
    
    def __init__(self, contamination=0.04, n_estimators=100, max_samples='auto', 
                 random_state=42, n_jobs=-1):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination (float): Expected proportion of anomalies (0.04 = 4%)
            n_estimators (int): Number of trees in the forest
            max_samples (int or str): Number of samples to draw for each tree
            random_state (int): Random seed for reproducibility
            n_jobs (int): Number of CPU cores to use (-1 = all cores)
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        self.is_fitted = False
        self.training_time = None
        self.prediction_time = None
        
        logger.info(f"Initialized IsolationForest with contamination={contamination}, "
                   f"n_estimators={n_estimators}")
    
    def fit(self, X):
        """
        Fit the Isolation Forest model.
        
        Args:
            X (np.ndarray or pd.DataFrame): Training data (features only)
        
        Returns:
            self: Fitted model
        """
        logger.info(f"Training Isolation Forest on {X.shape[0]} samples with {X.shape[1]} features")
        
        start_time = time.time()
        self.model.fit(X)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        return self
    
    def predict(self, X):
        """
        Predict anomalies.
        
        Args:
            X (np.ndarray or pd.DataFrame): Data to predict
        
        Returns:
            np.ndarray: Binary predictions (1 = anomaly, 0 = normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        start_time = time.time()
        # Isolation Forest returns -1 for anomalies, 1 for normal
        # Convert to 1 for anomalies, 0 for normal
        predictions = self.model.predict(X)
        predictions = np.where(predictions == -1, 1, 0)
        self.prediction_time = time.time() - start_time
        
        logger.info(f"Prediction completed in {self.prediction_time:.4f} seconds")
        logger.info(f"Detected {np.sum(predictions)} anomalies out of {len(predictions)} samples "
                   f"({np.mean(predictions):.2%})")
        
        return predictions
    
    def score_samples(self, X):
        """
        Get anomaly scores for each sample.
        
        Lower scores indicate more anomalous samples.
        
        Args:
            X (np.ndarray or pd.DataFrame): Data to score
        
        Returns:
            np.ndarray: Anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        # Isolation Forest returns negative scores (more negative = more anomalous)
        scores = self.model.score_samples(X)
        # Convert to positive scores where higher = more anomalous
        scores = -scores
        
        return scores
    
    def evaluate(self, X, y_true):
        """
        Evaluate model performance.
        
        Args:
            X (np.ndarray or pd.DataFrame): Test data
            y_true (np.ndarray): True labels (1 = anomaly, 0 = normal)
        
        Returns:
            dict: Performance metrics
        """
        y_pred = self.predict(X)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        # Get anomaly scores for ROC-AUC
        scores = self.score_samples(X)
        try:
            roc_auc = roc_auc_score(y_true, scores)
        except:
            roc_auc = 0.0
        
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'predictions': y_pred,
            'scores': scores
        }
        
        logger.info(f"Evaluation: Precision={precision:.4f}, Recall={recall:.4f}, "
                   f"F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk."""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


class LOFDetector:
    """
    Local Outlier Factor (LOF) for anomaly detection.
    
    LOF works by comparing the local density of a point with the densities
    of its neighbors. Points in sparser regions are considered anomalies.
    
    Best for: Density-based anomalies, local patterns, smaller datasets
    """
    
    def __init__(self, contamination=0.04, n_neighbors=20, novelty=True, 
                 metric='minkowski', n_jobs=-1):
        """
        Initialize LOF detector.
        
        Args:
            contamination (float): Expected proportion of anomalies (0.04 = 4%)
            n_neighbors (int): Number of neighbors to consider
            novelty (bool): If True, can predict on new data (required for our use case)
            metric (str): Distance metric ('minkowski', 'euclidean', etc.)
            n_jobs (int): Number of CPU cores to use (-1 = all cores)
        """
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.novelty = novelty
        self.metric = metric
        self.n_jobs = n_jobs
        
        self.model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors,
            novelty=novelty,
            metric=metric,
            n_jobs=n_jobs
        )
        
        self.is_fitted = False
        self.training_time = None
        self.prediction_time = None
        
        logger.info(f"Initialized LOF with contamination={contamination}, "
                   f"n_neighbors={n_neighbors}")
    
    def fit(self, X):
        """
        Fit the LOF model.
        
        Args:
            X (np.ndarray or pd.DataFrame): Training data (features only)
        
        Returns:
            self: Fitted model
        """
        logger.info(f"Training LOF on {X.shape[0]} samples with {X.shape[1]} features")
        
        start_time = time.time()
        self.model.fit(X)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        return self
    
    def predict(self, X):
        """
        Predict anomalies.
        
        Args:
            X (np.ndarray or pd.DataFrame): Data to predict
        
        Returns:
            np.ndarray: Binary predictions (1 = anomaly, 0 = normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        start_time = time.time()
        # LOF returns -1 for anomalies, 1 for normal
        # Convert to 1 for anomalies, 0 for normal
        predictions = self.model.predict(X)
        predictions = np.where(predictions == -1, 1, 0)
        self.prediction_time = time.time() - start_time
        
        logger.info(f"Prediction completed in {self.prediction_time:.4f} seconds")
        logger.info(f"Detected {np.sum(predictions)} anomalies out of {len(predictions)} samples "
                   f"({np.mean(predictions):.2%})")
        
        return predictions
    
    def score_samples(self, X):
        """
        Get anomaly scores for each sample.
        
        Higher scores indicate more anomalous samples.
        
        Args:
            X (np.ndarray or pd.DataFrame): Data to score
        
        Returns:
            np.ndarray: Anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        # LOF returns negative scores (more negative = more anomalous)
        scores = self.model.score_samples(X)
        # Convert to positive scores where higher = more anomalous
        scores = -scores
        
        return scores
    
    def evaluate(self, X, y_true):
        """
        Evaluate model performance.
        
        Args:
            X (np.ndarray or pd.DataFrame): Test data
            y_true (np.ndarray): True labels (1 = anomaly, 0 = normal)
        
        Returns:
            dict: Performance metrics
        """
        y_pred = self.predict(X)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        # Get anomaly scores for ROC-AUC
        scores = self.score_samples(X)
        try:
            roc_auc = roc_auc_score(y_true, scores)
        except:
            roc_auc = 0.0
        
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'predictions': y_pred,
            'scores': scores
        }
        
        logger.info(f"Evaluation: Precision={precision:.4f}, Recall={recall:.4f}, "
                   f"F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk."""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple models and return summary.
    
    Args:
        models_dict (dict): Dictionary of {model_name: model_object}
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
    
    Returns:
        pd.DataFrame: Comparison table with metrics
    """
    logger.info("=" * 60)
    logger.info("COMPARING MODELS")
    logger.info("=" * 60)
    
    results = []
    
    for name, model in models_dict.items():
        logger.info(f"\nEvaluating {name}...")
        metrics = model.evaluate(X_test, y_test)
        
        results.append({
            'Model': name,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
            'Training Time (s)': metrics['training_time'],
            'Prediction Time (s)': metrics['prediction_time']
        })
    
    comparison_df = pd.DataFrame(results)
    
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info("\n" + comparison_df.to_string(index=False))
    
    return comparison_df


def main():
    """
    Example usage of statistical models.
    """
    logger.info("Testing Statistical Anomaly Detection Models")
    
    # Load scaled featured data
    data_path = 'data/processed/scaled_featured_data.csv'
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'anomaly']]
    X = df[feature_cols].values
    y = df['anomaly'].values
    
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Anomaly ratio: {np.mean(y):.2%}")
    
    # Split data (80% train, 20% test)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Train Isolation Forest
    logger.info("\n" + "=" * 60)
    logger.info("ISOLATION FOREST")
    logger.info("=" * 60)
    
    iso_forest = IsolationForestDetector(contamination=0.04, n_estimators=100)
    iso_forest.fit(X_train)
    iso_metrics = iso_forest.evaluate(X_test, y_test)
    
    # Train LOF
    logger.info("\n" + "=" * 60)
    logger.info("LOCAL OUTLIER FACTOR")
    logger.info("=" * 60)
    
    lof = LOFDetector(contamination=0.04, n_neighbors=20)
    lof.fit(X_train)
    lof_metrics = lof.evaluate(X_test, y_test)
    
    # Compare models
    models = {
        'Isolation Forest': iso_forest,
        'Local Outlier Factor': lof
    }
    
    comparison = compare_models(models, X_test, y_test)
    
    # Save models
    iso_forest.save_model('outputs/models/isolation_forest.pkl')
    lof.save_model('outputs/models/lof.pkl')
    
    logger.info("\n✓ Testing complete!")


if __name__ == '__main__':
    main()

"""
NASA IMS Bearing Dataset Loader and Preprocessor

This module loads and preprocesses the NASA IMS bearing run-to-failure dataset.
The dataset contains vibration data from bearings running until failure.

Dataset Information:
- Source: NASA IMS (Intelligent Maintenance Systems) Center
- Kaggle: https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset
- 3 test sets with multiple bearings running until failure
- Vibration data collected at 20 kHz sampling rate
- Each file contains 20,480 data points (1 second of data)
- Collected every 10 minutes

Labeling Strategy:
- Last 10% of each bearing's run is labeled as "failure" (anomaly)
- First 90% is labeled as "normal"
- This simulates detecting bearing degradation before catastrophic failure

Author: Vaishnav M
Date: November 2025
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NASABearingDataLoader:
    """
    Loader for NASA IMS Bearing Dataset.
    
    Handles loading raw vibration data, computing statistical features,
    and labeling normal vs failure periods.
    """
    
    def __init__(self, data_path, failure_threshold=0.10):
        """
        Initialize the NASA Bearing Data Loader.
        
        Args:
            data_path (str): Path to the dataset directory
            failure_threshold (float): Proportion of data at end to label as failure (default 0.10 = 10%)
        """
        self.data_path = Path(data_path)
        self.failure_threshold = failure_threshold
        self.test_sets = {}
        
        logger.info(f"Initialized NASA Bearing Data Loader")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Failure threshold: {failure_threshold*100}%")
    
    def load_bearing_file(self, file_path):
        """
        Load a single bearing data file.
        
        Each file contains vibration measurements from multiple bearings (channels).
        Files are typically tab-separated with 4-8 columns (one per bearing).
        
        Args:
            file_path (Path): Path to the bearing data file
        
        Returns:
            pd.DataFrame: DataFrame with bearing measurements
        """
        try:
            # Try different delimiters (files may be space or tab separated)
            df = pd.read_csv(file_path, sep='\t', header=None)
            if df.shape[1] == 1:
                df = pd.read_csv(file_path, sep=' ', header=None)
            if df.shape[1] == 1:
                df = pd.read_csv(file_path, delim_whitespace=True, header=None)
            
            return df
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None
    
    def compute_statistical_features(self, vibration_data):
        """
        Compute statistical features from raw vibration data.
        
        Instead of using all 20,480 raw data points per file, we compute
        summary statistics that capture the vibration characteristics.
        
        Enhanced with advanced bearing diagnostics features from Kaggle research:
        - Clearance Factor: Indicates bearing clearance issues
        - Shape Factor: Measures waveform shape
        - Impulse Factor: Detects impulsive events (impacts, defects)
        
        Args:
            vibration_data (np.ndarray or pd.Series): Raw vibration measurements
        
        Returns:
            dict: Dictionary of computed features
        """
        data = np.array(vibration_data)
        
        # Basic statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
        
        # RMS (Root Mean Square) - Most important for bearing health
        rms = np.sqrt(np.mean(data**2))
        
        # Peak value
        peak = np.max(np.abs(data))
        
        # Crest Factor = Peak / RMS (high values indicate impulsive behavior)
        crest_factor = peak / (rms + 1e-10)
        
        # Clearance Factor = Peak / (Mean of Square Roots)^2
        # Sensitive to early bearing defects
        mean_sqrt = np.mean(np.sqrt(np.abs(data))) ** 2
        clearance_factor = peak / (mean_sqrt + 1e-10)
        
        # Shape Factor = RMS / Mean of Absolute Values
        # Indicates waveform shape changes
        mean_abs = np.mean(np.abs(data))
        shape_factor = rms / (mean_abs + 1e-10)
        
        # Impulse Factor = Peak / Mean of Absolute Values
        # Detects sharp impulses from bearing defects
        impulse_factor = peak / (mean_abs + 1e-10)
        
        features = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'rms': rms,
            'peak_to_peak': max_val - min_val,
            'kurtosis': pd.Series(data).kurtosis(),  # Measure of tailedness
            'skewness': pd.Series(data).skew(),  # Measure of asymmetry
            'crest_factor': crest_factor,
            'clearance_factor': clearance_factor,  # NEW: Bearing clearance indicator
            'shape_factor': shape_factor,          # NEW: Waveform shape indicator
            'impulse_factor': impulse_factor,      # NEW: Impact detection indicator
        }
        
        return features
    
    def load_test_set(self, test_set_name='1st_test', bearing_names=None):
        """
        Load a complete test set (all files for all bearings).
        
        Args:
            test_set_name (str): Name of test set folder ('1st_test', '2nd_test', '3rd_test')
            bearing_names (list): List of bearing names to process (e.g., ['Bearing1', 'Bearing2'])
                                 If None, will auto-detect from files
        
        Returns:
            pd.DataFrame: DataFrame with all data and labels
        """
        test_path = self.data_path / test_set_name
        
        if not test_path.exists():
            logger.error(f"Test set path does not exist: {test_path}")
            return None
        
        logger.info(f"Loading test set: {test_set_name}")
        
        # Get all data files (typically named like '2003.10.22.12.06.24')
        data_files = sorted([f for f in test_path.glob('*.*.*') if f.is_file()])
        
        if len(data_files) == 0:
            logger.warning(f"No data files found in {test_path}")
            return None
        
        logger.info(f"Found {len(data_files)} data files")
        
        # Load first file to determine number of bearings
        first_df = self.load_bearing_file(data_files[0])
        if first_df is None:
            return None
        
        n_bearings = first_df.shape[1]
        logger.info(f"Detected {n_bearings} bearings in dataset")
        
        # If bearing names not provided, create default names
        if bearing_names is None:
            bearing_names = [f'Bearing{i+1}' for i in range(n_bearings)]
        
        # Process each bearing separately
        all_bearing_data = []
        
        for bearing_idx in range(n_bearings):
            bearing_name = bearing_names[bearing_idx] if bearing_idx < len(bearing_names) else f'Bearing{bearing_idx+1}'
            logger.info(f"Processing {bearing_name}...")
            
            bearing_records = []
            
            for file_idx, file_path in enumerate(data_files):
                # Load file
                df = self.load_bearing_file(file_path)
                if df is None or bearing_idx >= df.shape[1]:
                    continue
                
                # Get vibration data for this bearing
                vibration_data = df.iloc[:, bearing_idx]
                
                # Compute statistical features
                features = self.compute_statistical_features(vibration_data)
                
                # Add metadata
                features['file_index'] = file_idx
                features['file_name'] = file_path.name
                features['bearing_name'] = bearing_name
                features['test_set'] = test_set_name
                
                bearing_records.append(features)
            
            # Convert to DataFrame
            bearing_df = pd.DataFrame(bearing_records)
            
            # Add labels: last 10% is failure, rest is normal
            n_samples = len(bearing_df)
            failure_start_idx = int(n_samples * (1 - self.failure_threshold))
            bearing_df['label'] = 0  # Normal
            bearing_df.loc[failure_start_idx:, 'label'] = 1  # Failure/Anomaly
            
            logger.info(f"  - Total samples: {n_samples}")
            logger.info(f"  - Normal samples: {(bearing_df['label'] == 0).sum()}")
            logger.info(f"  - Failure samples: {(bearing_df['label'] == 1).sum()}")
            logger.info(f"  - Anomaly rate: {(bearing_df['label'] == 1).sum() / n_samples * 100:.2f}%")
            
            all_bearing_data.append(bearing_df)
        
        # Combine all bearings
        final_df = pd.concat(all_bearing_data, ignore_index=True)
        
        logger.info(f"Loaded {test_set_name}: {len(final_df)} total samples, "
                   f"{(final_df['label'] == 1).sum()} anomalies ({(final_df['label'] == 1).sum() / len(final_df) * 100:.2f}%)")
        
        return final_df
    
    def prepare_sensor_data(self, df):
        """
        Extract sensor columns for feature engineering.
        
        Args:
            df (pd.DataFrame): DataFrame with all features
        
        Returns:
            tuple: (sensor_df, metadata_df, labels)
                - sensor_df: DataFrame with only sensor columns
                - metadata_df: DataFrame with metadata columns
                - labels: Series with anomaly labels
        """
        # Sensor feature columns (statistical features we computed)
        sensor_cols = ['mean', 'std', 'min', 'max', 'rms', 'peak_to_peak', 
                       'kurtosis', 'skewness', 'crest_factor', 'clearance_factor',
                       'shape_factor', 'impulse_factor']
        
        # Metadata columns
        metadata_cols = ['file_index', 'file_name', 'bearing_name', 'test_set']
        
        sensor_df = df[sensor_cols].copy()
        metadata_df = df[metadata_cols].copy()
        labels = df['label'].copy()
        
        return sensor_df, metadata_df, labels
    
    def load_multiple_test_sets(self, test_sets=['1st_test', '2nd_test', '3rd_test']):
        """
        Load multiple test sets and combine them.
        
        Args:
            test_sets (list): List of test set names to load
        
        Returns:
            pd.DataFrame: Combined DataFrame with all test sets
        """
        all_data = []
        
        for test_set in test_sets:
            df = self.load_test_set(test_set)
            if df is not None:
                all_data.append(df)
        
        if len(all_data) == 0:
            logger.error("No data loaded from any test set")
            return None
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined all test sets: {len(combined_df)} total samples")
        
        return combined_df
    
    def load_bearing_data(self, test_name='1st_test', bearing_number=3):
        """
        Convenience method to load a specific bearing's data.
        
        Args:
            test_name (str): Test set name ('1st_test', '2nd_test', '3rd_test')
            bearing_number (int): Bearing number (1-4 for Set 1&2, 3 only for Set 3)
        
        Returns:
            tuple: (X, y) where X is features DataFrame and y is labels Series
        """
        # Load full test set
        df = self.load_test_set(test_name)
        
        if df is None:
            raise ValueError(f"Failed to load test set: {test_name}")
        
        # Filter for specific bearing
        bearing_name = f'Bearing{bearing_number}'
        bearing_df = df[df['bearing_name'] == bearing_name].copy()
        
        if len(bearing_df) == 0:
            available_bearings = df['bearing_name'].unique().tolist()
            raise ValueError(
                f"Bearing {bearing_number} not found in {test_name}. "
                f"Available bearings: {available_bearings}"
            )
        
        # Prepare features and labels
        X, _, y = self.prepare_sensor_data(bearing_df)
        
        return X, y


def load_nasa_bearing_data(data_path, test_set='1st_test', failure_threshold=0.10):
    """
    Quick function to load NASA bearing data for immediate use.
    
    Args:
        data_path (str): Path to dataset
        test_set (str): Which test set to load ('1st_test', '2nd_test', '3rd_test')
        failure_threshold (float): Proportion of data to label as failure
    
    Returns:
        tuple: (X, y, metadata) where X is features, y is labels, metadata is info
    """
    loader = NASABearingDataLoader(data_path, failure_threshold)
    df = loader.load_test_set(test_set)
    
    if df is None:
        return None, None, None
    
    X, metadata, y = loader.prepare_sensor_data(df)
    
    return X, y, metadata


if __name__ == "__main__":
    # Example usage
    print("NASA IMS Bearing Data Loader")
    print("=" * 50)
    print("\nUsage Example:")
    print("""
    from nasa_data_loader import NASABearingDataLoader
    
    # Initialize loader
    loader = NASABearingDataLoader(
        data_path='path/to/bearing_dataset',
        failure_threshold=0.10  # Last 10% labeled as failure
    )
    
    # Load a test set
    df = loader.load_test_set('1st_test')
    
    # Prepare data for modeling
    X, metadata, y = loader.prepare_sensor_data(df)
    
    # X: sensor features (9 statistical features from vibration data)
    # y: labels (0=normal, 1=failure)
    # metadata: file info and bearing names
    """)

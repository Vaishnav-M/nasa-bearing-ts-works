"""
Feature Engineering Module for Time Series Anomaly Detection

This module provides functions to create features from raw time series data:
- Rolling statistics (mean, std, min, max)
- Lag features (previous values)
- Time-based features (hour, day of week, weekend indicator)
- Rate of change features
- Normalization utilities

Author: Vaishnav M
Date: November 2025
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeSeriesFeatureEngine:
    """
    Feature engineering pipeline for time series data.
    
    This class provides methods to create various features from raw sensor data
    that can improve anomaly detection model performance.
    """
    
    def __init__(self, sensor_columns):
        """
        Initialize the feature engine.
        
        Args:
            sensor_columns (list): List of sensor column names to process
        """
        self.sensor_columns = sensor_columns
        self.scaler = None
        logger.info(f"Initialized TimeSeriesFeatureEngine with {len(sensor_columns)} sensors")
    
    def apply_ema_smoothing(self, df, span=40):
        """
        Apply Exponential Moving Average (EMA) smoothing to reduce noise.
        
        EMA gives more weight to recent observations while smoothing out noise.
        This is particularly effective for vibration data from bearings.
        
        Formula: EMA_t = α * X_t + (1-α) * EMA_(t-1)
        where α = 2 / (span + 1)
        
        Args:
            df (pd.DataFrame): Input dataframe with sensor columns
            span (int): Span for EMA (default 40, from Kaggle research)
        
        Returns:
            pd.DataFrame: DataFrame with EMA-smoothed sensor columns added
        """
        logger.info(f"Applying EMA smoothing with span={span}")
        df_ema = df.copy()
        
        for sensor in self.sensor_columns:
            # Apply exponential moving average
            df_ema[f'{sensor}_ema'] = df[sensor].ewm(span=span, adjust=False).mean()
        
        logger.info(f"Created {len(self.sensor_columns)} EMA-smoothed features")
        return df_ema
    
    def create_rolling_features(self, df, windows=[5, 10, 30]):
        """
        Create rolling window statistics for each sensor.
        
        Rolling features capture temporal patterns and smooth out noise.
        
        Args:
            df (pd.DataFrame): Input dataframe with sensor columns
            windows (list): List of window sizes (in time steps)
        
        Returns:
            pd.DataFrame: DataFrame with added rolling feature columns
        """
        logger.info(f"Creating rolling features with windows: {windows}")
        df_rolled = df.copy()
        
        for sensor in self.sensor_columns:
            for window in windows:
                # Rolling mean - captures local average
                df_rolled[f'{sensor}_rolling_mean_{window}'] = (
                    df[sensor].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling standard deviation - captures local volatility
                df_rolled[f'{sensor}_rolling_std_{window}'] = (
                    df[sensor].rolling(window=window, min_periods=1).std()
                )
                
                # Rolling min - captures lower bound
                df_rolled[f'{sensor}_rolling_min_{window}'] = (
                    df[sensor].rolling(window=window, min_periods=1).min()
                )
                
                # Rolling max - captures upper bound
                df_rolled[f'{sensor}_rolling_max_{window}'] = (
                    df[sensor].rolling(window=window, min_periods=1).max()
                )
                
                # Rolling range (max - min) - captures volatility
                df_rolled[f'{sensor}_rolling_range_{window}'] = (
                    df_rolled[f'{sensor}_rolling_max_{window}'] - 
                    df_rolled[f'{sensor}_rolling_min_{window}']
                )
        
        # Fill any NaN values created by rolling (at the beginning)
        df_rolled = df_rolled.bfill()
        
        logger.info(f"Created {len(df_rolled.columns) - len(df.columns)} rolling features")
        return df_rolled
    
    def create_lag_features(self, df, lags=[1, 2, 3, 5]):
        """
        Create lag features (previous values) for each sensor.
        
        Lag features help models understand temporal dependencies.
        
        Args:
            df (pd.DataFrame): Input dataframe with sensor columns
            lags (list): List of lag periods (in time steps)
        
        Returns:
            pd.DataFrame: DataFrame with added lag feature columns
        """
        logger.info(f"Creating lag features with lags: {lags}")
        df_lagged = df.copy()
        
        for sensor in self.sensor_columns:
            for lag in lags:
                # Previous value at lag time steps ago
                df_lagged[f'{sensor}_lag_{lag}'] = df[sensor].shift(lag)
        
        # Fill NaN values at the beginning with forward fill
        df_lagged = df_lagged.ffill()
        
        logger.info(f"Created {len(df_lagged.columns) - len(df.columns)} lag features")
        return df_lagged
    
    def create_time_features(self, df, timestamp_col='timestamp'):
        """
        Create time-based features from timestamp column.
        
        Time features capture daily, weekly, and seasonal patterns.
        
        Args:
            df (pd.DataFrame): Input dataframe with timestamp column
            timestamp_col (str): Name of the timestamp column
        
        Returns:
            pd.DataFrame: DataFrame with added time feature columns
        """
        logger.info("Creating time-based features")
        df_time = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_time[timestamp_col]):
            df_time[timestamp_col] = pd.to_datetime(df_time[timestamp_col])
        
        # Extract time components
        df_time['hour'] = df_time[timestamp_col].dt.hour
        df_time['day_of_week'] = df_time[timestamp_col].dt.dayofweek  # 0=Monday, 6=Sunday
        df_time['day_of_month'] = df_time[timestamp_col].dt.day
        df_time['month'] = df_time[timestamp_col].dt.month
        
        # Cyclical encoding for hour (24-hour cycle)
        # This preserves the cyclical nature: 23:00 is close to 00:00
        df_time['hour_sin'] = np.sin(2 * np.pi * df_time['hour'] / 24)
        df_time['hour_cos'] = np.cos(2 * np.pi * df_time['hour'] / 24)
        
        # Cyclical encoding for day of week (7-day cycle)
        df_time['day_sin'] = np.sin(2 * np.pi * df_time['day_of_week'] / 7)
        df_time['day_cos'] = np.cos(2 * np.pi * df_time['day_of_week'] / 7)
        
        # Binary indicator for weekend
        df_time['is_weekend'] = (df_time['day_of_week'] >= 5).astype(int)
        
        # Time of day categories
        df_time['is_night'] = ((df_time['hour'] >= 22) | (df_time['hour'] < 6)).astype(int)
        df_time['is_morning'] = ((df_time['hour'] >= 6) & (df_time['hour'] < 12)).astype(int)
        df_time['is_afternoon'] = ((df_time['hour'] >= 12) & (df_time['hour'] < 18)).astype(int)
        df_time['is_evening'] = ((df_time['hour'] >= 18) & (df_time['hour'] < 22)).astype(int)
        
        logger.info(f"Created {len(df_time.columns) - len(df.columns)} time features")
        return df_time
    
    def create_rate_of_change_features(self, df):
        """
        Create rate of change (difference) features for each sensor.
        
        Rate of change helps detect sudden jumps or drops in sensor values.
        
        Args:
            df (pd.DataFrame): Input dataframe with sensor columns
        
        Returns:
            pd.DataFrame: DataFrame with added rate of change columns
        """
        logger.info("Creating rate of change features")
        df_roc = df.copy()
        
        for sensor in self.sensor_columns:
            # First-order difference (change from previous time step)
            df_roc[f'{sensor}_diff_1'] = df[sensor].diff(1)
            
            # Second-order difference (change in the rate of change)
            df_roc[f'{sensor}_diff_2'] = df[sensor].diff(2)
            
            # Percentage change
            df_roc[f'{sensor}_pct_change'] = df[sensor].pct_change()
            
            # Absolute change magnitude
            df_roc[f'{sensor}_abs_diff'] = df_roc[f'{sensor}_diff_1'].abs()
        
        # Fill NaN values
        df_roc = df_roc.fillna(0)
        
        logger.info(f"Created {len(df_roc.columns) - len(df.columns)} rate of change features")
        return df_roc
    
    def create_interaction_features(self, df):
        """
        Create interaction features between sensors.
        
        Interactions capture relationships between different sensors.
        
        Args:
            df (pd.DataFrame): Input dataframe with sensor columns
        
        Returns:
            pd.DataFrame: DataFrame with added interaction feature columns
        """
        logger.info("Creating interaction features between sensors")
        df_interact = df.copy()
        
        # Pairwise ratios and differences
        for i, sensor1 in enumerate(self.sensor_columns):
            for sensor2 in self.sensor_columns[i+1:]:
                # Ratio between sensors
                df_interact[f'{sensor1}_{sensor2}_ratio'] = (
                    df[sensor1] / (df[sensor2] + 1e-8)  # Add small value to avoid division by zero
                )
                
                # Difference between sensors
                df_interact[f'{sensor1}_{sensor2}_diff'] = df[sensor1] - df[sensor2]
        
        # Mean across all sensors
        df_interact['sensors_mean'] = df[self.sensor_columns].mean(axis=1)
        
        # Standard deviation across all sensors
        df_interact['sensors_std'] = df[self.sensor_columns].std(axis=1)
        
        # Min and max across all sensors
        df_interact['sensors_min'] = df[self.sensor_columns].min(axis=1)
        df_interact['sensors_max'] = df[self.sensor_columns].max(axis=1)
        df_interact['sensors_range'] = df_interact['sensors_max'] - df_interact['sensors_min']
        
        logger.info(f"Created {len(df_interact.columns) - len(df.columns)} interaction features")
        return df_interact
    
    def normalize_features(self, df, method='standard', columns_to_normalize=None):
        """
        Normalize/scale features for model training.
        
        Normalization ensures all features are on similar scales.
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): 'standard' for StandardScaler, 'minmax' for MinMaxScaler
            columns_to_normalize (list): Specific columns to normalize. If None, normalizes all numeric columns
        
        Returns:
            pd.DataFrame: DataFrame with normalized features
            sklearn.preprocessing scaler: Fitted scaler object (for inverse transform)
        """
        logger.info(f"Normalizing features using {method} scaling")
        df_normalized = df.copy()
        
        # Determine columns to normalize
        if columns_to_normalize is None:
            # Get all numeric columns except timestamp and labels
            columns_to_normalize = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove label column if present
            if 'anomaly' in columns_to_normalize:
                columns_to_normalize.remove('anomaly')
        
        # Choose scaler
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit and transform
        df_normalized[columns_to_normalize] = self.scaler.fit_transform(df[columns_to_normalize])
        
        logger.info(f"Normalized {len(columns_to_normalize)} features")
        return df_normalized, self.scaler
    
    def create_all_features(self, df, timestamp_col='timestamp', normalize=False, 
                           rolling_windows=[5, 10, 30], lags=[1, 2, 3, 5], 
                           apply_ema=True, ema_span=40):
        """
        Create all features in a single pipeline.
        
        This is a convenience method that applies all feature engineering steps.
        Enhanced with EMA smoothing for noise reduction (proven in Kaggle research).
        
        Args:
            df (pd.DataFrame): Input dataframe
            timestamp_col (str): Name of timestamp column
            normalize (bool): Whether to normalize features at the end
            rolling_windows (list): Window sizes for rolling features
            lags (list): Lag periods for lag features
            apply_ema (bool): Whether to apply EMA smoothing (default True)
            ema_span (int): Span for EMA smoothing (default 40)
        
        Returns:
            pd.DataFrame: Fully engineered feature dataframe
            sklearn.preprocessing scaler (optional): Scaler object if normalize=True
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 60)
        
        # Create features step by step
        df_features = df.copy()
        
        # 0. EMA Smoothing (FIRST, to reduce noise before other features)
        if apply_ema:
            df_features = self.apply_ema_smoothing(df_features, span=ema_span)
        
        # 1. Time features
        df_features = self.create_time_features(df_features, timestamp_col)
        
        # 2. Rolling features
        df_features = self.create_rolling_features(df_features, windows=rolling_windows)
        
        # 3. Lag features
        df_features = self.create_lag_features(df_features, lags=lags)
        
        # 4. Rate of change features
        df_features = self.create_rate_of_change_features(df_features)
        
        # 5. Interaction features
        df_features = self.create_interaction_features(df_features)
        
        logger.info("=" * 60)
        logger.info(f"FEATURE ENGINEERING COMPLETE")
        logger.info(f"Original features: {len(df.columns)}")
        logger.info(f"Total features: {len(df_features.columns)}")
        logger.info(f"New features added: {len(df_features.columns) - len(df.columns)}")
        logger.info("=" * 60)
        
        # 6. Optional normalization
        if normalize:
            df_features, scaler = self.normalize_features(df_features)
            return df_features, scaler
        else:
            return df_features


def main():
    """
    Example usage of the feature engineering module.
    """
    logger.info("Testing Feature Engineering Module")
    
    # Load synthetic data
    data_path = 'data/raw/synthetic_sensor_data_with_labels.csv'
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Initialize feature engine
    sensor_cols = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4']
    feature_engine = TimeSeriesFeatureEngine(sensor_cols)
    
    # Create all features
    df_features = feature_engine.create_all_features(
        df, 
        timestamp_col='timestamp',
        normalize=False,
        rolling_windows=[5, 10, 30],
        lags=[1, 2, 3, 5]
    )
    
    # Save to processed data
    output_path = 'data/processed/featured_sensor_data.csv'
    df_features.to_csv(output_path, index=False)
    logger.info(f"Saved featured data to {output_path}")
    
    # Display sample
    print("\n" + "=" * 80)
    print("SAMPLE OF ENGINEERED FEATURES")
    print("=" * 80)
    print(df_features.head())
    print("\n" + "=" * 80)
    print(f"Feature columns ({len(df_features.columns)}):")
    print("=" * 80)
    for i, col in enumerate(df_features.columns, 1):
        print(f"{i:3d}. {col}")


if __name__ == '__main__':
    main()

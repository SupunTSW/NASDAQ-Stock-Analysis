"""
Data preprocessing module for NASDAQ prediction project

This module handles loading, cleaning, and preparing the NASDAQ data for 
future analysis and modeling, including handling missing values, outliers,
and data normalization.
"""

#import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

def load_data(file_path):
    """
    Load the dataset
    
    Parameters:
    file_path: path to the raw data csv file

    Returns:
    pd.DataFrame: DataFrame with basic preprocessing
    """
    df = pd.read_csv(file_path)
    #convert Date to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    return df

def handle_missing_values(df):
    """
    Detect and handle the missing values in the dataset

    Parameters:
    df: Input DataFrame

    Returns:
    pd.DataFrame: DataFrame after handled the missing values
    """

    #for price columns, forward fill then backward fill
    price_colmn = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    df[price_colmn] = df[price_colmn].fillna(method='ffill').fillna(method='bfill')

    #for volume, use median of the same weekday
    if df['Volume'].isnull().sum() > 0:
        df['Weekday'] = df.index.weekday
        for day in range(5): #0=Monday, 4=Friday
            day_median = df.loc[df['Weekday'] == day, 'Volume'].median()
            day_idx = df[(df['Weekday'] == day) & (df['Volume'].isnull())].index
            df.loc[day_idx, 'Volume'] = day_median
        df.drop('Weekday', axis=1, inplace = True)

    return df

def detect_outliers(df, z_threshold=3):
    """
    Detect outliers using z-score method

    Parameters:
    df: Input Data Frame
    z_threshold: Threshold value for z-score outlier detection

    Returns:
    dict: Dictionary with column names as keys and number of outliers as values
    """

    outliers = {}
    for colmn in df.columns:
        if df[colmn].dtype in ['int64', 'float64']:
            z_scores = np.abs((df[colmn] - df[colmn].mean() / df[colmn].std()))
            outliers[colmn] = z_scores[z_scores > z_threshold].count()
    return outliers

def handle_outliers(df, method='clip'):
    """
    Handle outliers in the Daily_Return column

    Parameters: 
    df: Input DataFrame with Daily_Return column
    method: Method to handle outliers, either 'clip' or 'winsorize'

    Return:
    pd.DataFrame: DataFrame with outliers handled
    """
    df_treated = df.copy()

    #Only treat the Daily_Return column for outliers as price outliers might be legimate market movements
    if 'Daily_Return' in df.columns:
        if method == 'clip':
            #Clip outliers at 3 standard deviations
            mean = df['Daily_Return'].mean()
            std = df['Daily_Return'].std()
            df_treated['Daily_Return'] = df['Daily_Return'].clip(lower=mean-3*std, upper = mean+3*std)

        elif method == 'winsorize':
            #winsorize at 1% and 99%
            df_treated['Daily_Return'] = stats.mstats.winsorize(df['Daily_Return'], limits = [0.01, 0.01])

    return df_treated

def scale_data(df, columns, method = 'minmax'):
    """
    Scale the data using either Min-Max or Standard scaling

    Parameters:
    df: Input DataFrame
    columns: List  of column names to scale
    method: Scaling method, either 'minmax' or 'standard'

    Returns:
    tuple: (scaled DataFrame, fitted scaler object)
    """
    df_scaled = df.copy()

    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()

    #Fit and transform the data
    df_scaled[columns] = scaler.fit_transform(df[columns])

    return df_scaled, scaler

def split_time_series(df, train_ratio = 0.8):
    """
    Split the time series data into training and testing sets.

    Parameters: 
    df: Input Dataframe
    train_ratio: Ratio of data to use for training (0.0-1.0)

    Returns:
    tuple: (training DataFrame, testing DataFrame)
    """
    #Calculate split point
    split_idx = int(len(df) * train_ratio)

    #Split the data
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:]

    print(f"Training set: {len(train_data)} samples ({train_ratio*100:.0f}%)")
    print(f"Testing set: {len(test_data)} samples ({(1-train_ratio)*100:.0f}%)")

    return train_data, test_data

def preprocess_data(raw_file_path, output_dir='data/processed/'):
    """
    Complete preprocessing pipeline for NASDAQ data.

    Parameters:
    raw_file_path: Path to raw file data
    output_dir: Directory to save processed data

    Returns:
    tuple: (training DataFrame, testing DataFrame, scaler object)
    """
    #Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok = True)

    #Load the data
    df = load_data(raw_file_path)
    print(f"Loaded data with shape: {df.shape}")

    #Add daily returns
    df['Daily_Returns'] = df['Close'].pct_change() * 100

    #add log volume
    df['Log_Volume'] = np.log(df['Volume'])

    #handle missing values
    df = handle_missing_values(df)
    print("Missing values handled.")

    #Handle outliers
    outliers = detect_outliers(df)
    print("Outliers detected:", outliers)
    df = handle_outliers(df, method='clip')
    print("Outliers handled using clipping method.")

    #Scale the data
    scale_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Log_Volume']
    df_scaled, scaler = scale_data(df, scale_columns, method='minmax')
    print("Data scaled using Min-Max scaling.")

    #Split into training and testing sets
    train_data, test_data = split_time_series(df_scaled, train_ratio=0.8)

    #Save processed data to CSV
    train_data.to_csv(f"{output_dir}nasdaq_train_processed.csv")
    test_data.to_csv(f"{output_dir}nasdaq_test_processed.csv")
    print(f"Processed data saved to {output_dir}")

    #Also save the original data with added features but without scaling
    df.to_csv(f"{output_dir}nasdaq_preprocessed_unscaled.csv")

    return train_data, test_data, scaler


#Testing
#train_data, test_data, scaler = preprocess_data("data/raw/nasdaq_historical_prices.csv")



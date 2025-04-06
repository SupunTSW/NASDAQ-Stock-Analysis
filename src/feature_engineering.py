"""
Feature engineering module for NASDAQ prediction project

This module creates additional features from the preprocessed NASDAQ data,
including techniacal indicators, time-based features, and lag features and 
implement feature selection methods.
"""

import pandas as pd
import numpy as np
import os
import talib #For technical indicators (install with: pip install ta-lib)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

def add_time_features(df):
    """
    Add calendar-based features to the dataset

    Parameters:
    df: Input DataFrame with DateTimeIndex

    Return:
    pd.DataFrame: DataFrame with added time-based features
    """

    df_with_features = df.copy()

    #Extract date components
    df_with_features['Day'] = df.index.day
    df_with_features['Month'] = df.index.month
    df_with_features['Year'] = df.index.year
    df_with_features['Weekday'] = df.index.weekday #Monday=0, Sunday=6
    df_with_features['Quarter'] = df.index.quarter
    df_with_features['WeekOfYear'] = df.index.isocalendar().week

    #Create day indicators (1 for trading day, 0 for non-trading day)
    df_with_features['Is_Trading_Day'] = 1

    #Add cyclic encoding for month and weekday
    df_with_features['Month_Sin'] = np.sin(2 * np.pi * df_with_features['Month']/12)
    df_with_features['Month_Cos'] = np.cos(2 * np.pi * df_with_features['Month']/12)
    df_with_features['Weekday_Sin'] = np.sin(2 * np.pi * df_with_features['Weekday']/5)
    df_with_features['Weekday_Sin'] = np.cos(2 * np.pi * df_with_features['Weekday']/5)

    return df_with_features

def add_rolling_features(df, windows=[5,10,20,50]):
    """
    Add rolling eindow features

    Parameters: 
    df: Input DataFrame
    windows: List of window sizes for rolling calculations

    Returns:
    pd.DataFrame: DataFrame with added rolling features
    """
    df_with_features = df.copy()

    price_colmn = 'Close'
    volume_colmn = 'Volume'

    for window in windows:
        #Simple Moving Average (SMA)
        df_with_features[f'SMA_{window}'] = df[price_colmn].rolling(window=window).mean()

        #Exponential Moving Average (EMA)
        df_with_features[f'EMA_{window}'] = df[price_colmn].ewm(span=window, adjust=False).mean()

        #Rolling standard deviation (volatility)
        df_with_features[f'volatility_{window}'] = df[price_colmn].rolling(window=window).std()

        #Relative price position: (current price - min) / (max - min)
        rolling_min = df[price_colmn].rolling(window=window).min()
        rolling_max = df[price_colmn].rolling(window=window).max()
        df_with_features[f'Price_Position_{window}'] = (df[price_colmn] - rolling_min) / (rolling_max - rolling_min)

        #Volume features
        df_with_features[f'Volume_SMA_{window}'] = df[volume_colmn].rolling(window=window).mean()
        df_with_features[f'Volume_Ratio_{window}'] = df[volume_colmn] / df[volume_colmn].rolling(window=window).mean()
    
    return df_with_features

def add_technical_indicators(df):
    """
    Add technical indicators to the dataset.

    Parameter:
    df: Input DataFrame

    Returns:
    pd.DataFrame: DataFrame with added technical indicators
    """
    df_with_features = df.copy()

    #Relative Strength Index (RSI)
    df_with_features['RSI_14'] = talib.RSI(df['Close'].astype(np.float64).values, timeperiod=14)

    #Moving Average Convergence Divergence (MACD)
    macd, macd_signal, macd_hist = talib.MACD(df['Close'].astype(np.float64).values, fastperiod=12, slowperiod=26, signalperiod=9)
    df_with_features['MACD'] = macd
    df_with_features['MACD_Signal'] = macd_signal
    df_with_features['MACD_Hist'] = macd_hist

    #Bollinger Bands
    upperband, middleband, lowerband = talib.BBANDS(df['Close'].astype(np.float64).values, timeperiod=20)
    df_with_features['BB_Upper'] = upperband
    df_with_features['BB_Middle'] = middleband
    df_with_features['BB_Lower'] = lowerband

    #Rate of Change (ROC)
    df_with_features['ROC_10'] = talib.ROC(df['Close'].astype(np.float64).values, timeperiod=10)

    #Average Directional Index (ADX)
    df_with_features['ADX_14'] = talib.ADX(df['High'].astype(np.float64).values, df['Low'].astype(np.float64).values, df['Close'].astype(np.float64).values, timeperiod=14)

    #Commodity Channel Index (CCI)
    df_with_features['CCI_14'] = talib.CCI(df['High'].astype(np.float64).values, df['Low'].astype(np.float64).values, df['Close'].astype(np.float64).values, timeperiod=14)

    # On-Balance Volume (OBV)
    df_with_features['OBV'] = talib.OBV(df['Close'].astype(np.float64).values, df['Volume'].astype(np.float64).values)

    return df_with_features

def add_lag_features(df, lag_periods=[1,2,3,5,10]):
    """
    Add lagged features to the dataset.

    Parameters:
    df: Input DataFrame
    lag_periods: List of lag periods to include

    Returns:
    pd.DataFrame: DataFrame with added lag features
    """
    df_with_features = df.copy()

    #Key features to lag
    features_to_lag = ['Close', 'Volume', 'Daily_Return']

    for feature in features_to_lag:
        if feature in df.columns:
            for lag in lag_periods:
                df_with_features[f'{feature}_Lag_{lag}'] = df[feature].shift(lag)
    
    return df_with_features

def add_target_variable(df, forecast_horizon=1):
    """
    Add target variable for prediction.

    Parameters:
    df: Input DataFrame
    forecast_horizon: Number of days ahead to predict

    Returns:
    pd.DataFrame: DataFrame with target variable
    """
    df_with_target = df.copy()

    #Add future close price as target
    df_with_target[f'Target_Close_{forecast_horizon}d'] = df['Close'].shift(-forecast_horizon)

    #Add future return as alternative target
    df_with_target[f'Target_Return_{forecast_horizon}d'] = df['Close'].pct_change(-forecast_horizon) * 100

    return df_with_target

def create_interaction_features(df):
    """
    Create interaction features that combine existing features.

    Parameters:
    df: Input DataFrame

    Returns:
    pd.DataFrame: DataFrame with interaction features
    """
    df_with_features = df.copy()

    #Volume-price interaction
    if 'Volume' in df.columns and 'Close' in df.columns:
        df_with_features['Volume_Price_Ratio'] = df['Volume'] / df['Close']

    #Volatility-volume interaction
    if 'Volatility_20' in df.columns and 'Volume' in df.columns:
        df_with_features['Volatility_Volume_Ratio'] = df['Volatility_20'] / df['Volume']

    return df_with_features

def analyze_feature_correlations(df, target_column, threshold=0.1):
    """
    Analyze correlation between features and target variable

    Parameters:
    df: Input DataFrame with features
    target_column: Name of the target column
    threshold: Minimum absolute correlation to consider feature as important

    Returns:
    pd.Series: Stored correlations with target
    List: Selected feature names based on correlation threshold
    """

    #Calculate correlation with target
    correlations = df.corr()[target_column].sort_values(ascending=False)

    #Select features with correlations above threshold
    selected_features = correlations[
        (abs(correlations) > threshold) & 
        (correlations.index != target_column)
    ].index.tolist()

    return correlations, selected_features

def calculate_tree_importance(df, features, target_column, n_estimators=100):
    """
    Calculate feature importance using a Random Forest model.

    Parameters:
    df: Input DataFrame
    features: List of feature columns
    target_column: Names of the target column
    n_estimators: Number of trees in the forest

    Returns:
    pd.DataFrame: DataFrame with feature importance scores
    lists: Selected feature names based on importance (top 50%)
    """

    #Prepare data
    x = df[features].copy()
    y = df[target_column]

    #Train a Random Forest model
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(x,y)

    #Get feature importances
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    #Select top 50% features
    cumulative_importance = importances['Importance'].cumsum() / importances['Importance'].sum()
    importance_threshold = importances.loc[cumulative_importance <= 0.5, 'Importance'].min()
    selected_features = importances[importances['Importance'] >= importance_threshold]['Feature'].tolist()

    return importances, selected_features


def perform_rfe(df, features, target_column, n_features_to_select=10):
    """
    Perform Recursive Feature Elimimation

    Parametrs:
    df: Input DataFrame
    features: List of feature columns
    target_column: Name of the target column
    n_features_to_select: Number of features to select

    Returns:
    list: Selected feature names
    """

    #Prepare data
    x = df[features].copy()
    y = df[target_column]

    #Create the RFE model
    estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    selector = selector.fit(x,y)

    #Get selected features
    selected_features = [features[i] for i in range(len(features)) if selector.support_[i]]

    return selected_features

def select_features(df, target_column, correlation_threshold=0.1, n_top_features = 20):
    """
    Select the most important features using multiple methods.

    Parameters:
    df: Input DataFrame with all features
    target_column: Name of the target column
    correlation_threshold: Minimum correlation threshold
    n_top_features: Number of top features to select using RFE

    Returns:
    dict: Dictionary with lists of features selected by each method
    list: Final selected features (intersection of methods)
    pd.DataFrame: DataFrame with only the selected features and target
    """
    #Get all feature columns (exclude target and date-based columns)
    all_features = [col for col in df.columns if col != target_column]

    #Method 1: Correlation-based selection
    correlation_results, corr_features = analyze_feature_correlations(df, target_column)
    print(f"Selected {len(corr_features)} features based on correlation")

    #Method 2: Tree-based importance
    importance_results, imp_features = calculate_tree_importance(df, all_features, target_column)
    print(f"Selected {len(imp_features)} features based on tree importance")

    #Method 3: Recursive Feature Elimination
    rfe_features = perform_rfe(df, all_features, target_column, n_features_to_select=n_top_features)
    print(f"Selected {len(rfe_features)} feature using RFE")

    #Combine results (features that appear in at least 2 methods)
    feature_counts = {}
    for feature in all_features:
        feature_counts[feature] = sum([
            feature in corr_features,
            feature in imp_features,
            feature in rfe_features
        ])

    #Select features that appear in at least 2 methods
    final_features = [f for f, count in feature_counts.items() if count >= 2]

    #If fewer than 10 features are selected, add more from RFE
    if len(final_features) < 10:
        additional_features = [f for f in rfe_features if f not in final_features]
        final_features.extend(additional_features[:10 - len(final_features)])
    print(f"Final Selection: {len(final_features)} featurs")

    #Create a DataFrame with selected features and target
    selected_df = df[final_features + [target_column]].copy()

    #Return all results
    all_selections = {
        'correlation' : corr_features,
        'importance' : imp_features,
        'rfe' : rfe_features,
        'final' : final_features
    }

    return all_selections, final_features, selected_df

def scale_engineered_features(df, method='minmax'):
    """
    Scale all features while preserving the target variable.

    Parameters:
    df: Input DataFrame with engineered features
    target_column: Name of the target column
    method: Scaling method ('minmax' or 'standard')

    Returns:
    tuple: (scaled DataFrame, Scaler object)
    """
    #Create copies
    df_scaled = df.copy()

    #Choose Scaler
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard' :
        scaler = StandardScaler()
    else:
        raise ValueError("Method must be 'minmax' or 'standard' ")

    #Scale features
    df_scaled[:] = scaler.fit_transform(df)

    return df_scaled, scaler


def engineer_features(input_file, output_dir='data/processed/', forecast_horizon=1):
    """
    Complete feature engineering pipeline.

    Parameters:
    input_file: Path to preprocessed data file
    output_dir: Directory to save feature engineered data
    forecast_horizon: Number of days ahead to predict

    Returns:
    tuple: (DataFrame with selected features, scaler object, list of selected features)
    """
    #Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    #Load preprocessed data
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    print(f"Loaded preprocessed data with shape: {df.shape}")

    #Apply feature engineering functions
    df = add_time_features(df)
    print("Added time-based features.")

    df = add_rolling_features(df)
    print("Added rolling window features.")

    try: 
        df = add_technical_indicators(df)
        print("Added technical indicators.")
    except:
        print("Could not add technical indicators. Make sure TA-Lib is installed")
    
    df = add_lag_features(df)
    print("Added lagged features.")

    df = create_interaction_features(df)
    print("Added interaction features.")

    #Add target variable last
    df = add_target_variable(df, forecast_horizon=forecast_horizon)
    print(f"Added target variable for {forecast_horizon}-day forecast")

    #Remove rows wit Nan values that result from lagging and rolling calculations
    df_clean = df.dropna()
    print(f"Removed rows with missing values. Final shape: {df_clean.shape}")

    #Define target column
    target_column = f'Target_Close_{forecast_horizon}d'

    #Perform feature selection
    print("\nPerforming feature selection...")
    selection_results, selected_features, selected_df = select_features(
        df_clean, target_column, correlation_threshold=0.1, n_top_features=20
    )
    
    #Save feature selection results
    selection_path = f'{output_dir}feature_selection_results.csv'
    pd.DataFrame({
        'Feature': df_clean.columns,
        'Selected' : [col in selected_features for col in df_clean.columns]
    }).to_csv(selection_path)

    #Save unscaled selected features
    unscaled_path = f'{output_dir}nasdaq_selected_features_unscaled.csv'
    selected_df.to_csv(unscaled_path)
    print(f"Unscaled selected features saved to {unscaled_path}")

    #Scale the selected features
    scaled_df, scaler = scale_engineered_features(selected_df, method='minmax')

    #Save the scaled selected features
    scaled_path = f"{output_dir}nasdaq_selected_features_scaled.csv"
    scaled_df.to_csv(scaled_path)
    print(f"Scaled selected features saved to {scaled_path}")
    
    #Also Save all engineered features 
    all_features_path = f"{output_dir}nasdaq_all_features.csv"
    df_clean.to_csv(all_features_path)
    print(f"All engineered features saved to {all_features_path}")

    return scaled_df, scaler, selected_features

#Testing
engineered_data, scaler, selected_features = engineer_features("data/processed/nasdaq_preprocessed_unscaled.csv")
print(f"Selected {len(selected_features)} features for modeling")

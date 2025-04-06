"""
Training script for NASDAQ prediction project

This script loads the preprocessed data, builds models, trains them,
and evaluates their performance.
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import argparse
import json
import h5py
from datetime import datetime

# Import model module
from model import (BaselineModels, LSTMModel, GRUModel, CNNLSTMModel,
                  prepare_sequence_data, create_data_loaders, train_model,
                  evaluate_model, plot_predictions, plot_training_history,
                  save_model, load_model)

def load_processed_data(data_path):
    """
    Load processed data from CSV file
    
    Parameters:
    data_path: Path to the processed data file
    
    Returns:
    tuple: (features, targets, feature_names, scaler)
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Convert date column to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    
    # Identify target column (the one with 'Target_Close_1d' in its name)
    target_col = [col for col in df.columns if 'Target_Close_1d' in col][0]
    feature_cols = [col for col in df.columns if col != target_col]
    
    print(f"Target column: {target_col}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values
    
    return X, y, feature_cols, None  # None for scaler as data is already scaled

def run_baseline_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate baseline models
    
    Parameters:
    X_train, y_train, X_test, y_test: Training and testing data
    
    Returns:
    dict: Results from baseline models
    """
    print("\n--- Running Baseline Models ---")
    baselines = BaselineModels()
    
    # Train models
    print("Training Linear Regression model...")
    baselines.fit_linear_model(X_train, y_train)
    
    print("Training Random Forest model...")
    baselines.fit_random_forest(X_train, y_train)
    
    # Evaluate models
    results = baselines.evaluate_models(X_test, y_test)
    
    # Print results
    for model_name, metrics in results.items():
        print(f"\n{model_name} Results:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.6f}")
    
    return results

def run_deep_learning_models(X_train, y_train, X_test, y_test, seq_length=10, 
                            epochs=100, batch_size=32, save_dir='models', results_dir='results'):
    """
    Train and evaluate deep learning models
    
    Parameters:
    X_train, y_train, X_test, y_test: Training and testing data
    seq_length: Sequence length for time series models
    epochs: Number of training epochs
    batch_size: Batch size for training
    save_dir: Directory to save the final H5 model
    results_dir: Directory to save all results, plots and .pth models
    
    Returns:
    dict: Results from deep learning models
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare sequence data
    train_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test, seq_length, batch_size
    )
    
    # Model parameters
    input_dim = X_train.shape[1]
    hidden_dim = 64
    num_layers = 2
    dropout = 0.2
    
    results = {}
    
    # Define models to train
    models = {
        'LSTM': LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, 
                          num_layers=num_layers, dropout=dropout),
        'GRU': GRUModel(input_dim=input_dim, hidden_dim=hidden_dim, 
                        num_layers=num_layers, dropout=dropout),
        'CNN-LSTM': CNNLSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, 
                                num_layers=num_layers, dropout=dropout)
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n--- Training {name} Model ---")
        trained_model, history, eval_metrics = train_model(
            model, train_loader, test_loader, epochs=epochs, 
            learning_rate=0.001, weight_decay=1e-5, verbose=True
        )
        
        # Save model .pth file to results directory
        save_path = os.path.join(results_dir, f"{name.lower()}_model.pth")
        save_model(trained_model, save_path)
        
        # Save training history to results directory
        history_path = os.path.join(results_dir, f"{name.lower()}_history.json")
        with open(history_path, 'w') as f:
            json.dump({k: [float(val) for val in v] for k, v in history.items()}, f)
        
        # Plot training history to results directory
        plot_training_history(history, model_name=name, output_dir=results_dir)
        
        # Plot predictions to results directory
        plot_predictions(trained_model, test_loader, model_name=name, output_dir=results_dir, n_samples=100)
        
        # Store results
        results[name] = eval_metrics
        
        # Print evaluation metrics
        print(f"\n{name} Model Evaluation:")
        for metric_name, value in eval_metrics.items():
            print(f"  {metric_name}: {value:.6f}")
    
    return results

def save_results(baseline_results, dl_results, output_dir='results'):
    """
    Save evaluation results to files
    
    Parameters:
    baseline_results: Results from baseline models
    dl_results: Results from deep learning models
    output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine results
    all_results = {**baseline_results, **dl_results}
    
    # Save as JSON
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Create comparison table
    model_names = list(all_results.keys())
    metrics = ['MSE', 'RMSE', 'MAE']
    
    comparison_df = pd.DataFrame(index=model_names, columns=metrics)
    
    for model in model_names:
        for metric in metrics:
            comparison_df.loc[model, metric] = all_results[model][metric]
    
    # Save comparison table
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
    
    # Create comparison plots
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, comparison_df[metric])
        plt.title(f'Model Comparison - {metric}')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'))
        plt.close()
    
    return comparison_df

def find_best_model(comparison_df, metric='RMSE', lower_is_better=True):
    """
    Find the best performing model based on a specified metric
    
    Parameters:
    comparison_df: DataFrame with model comparison metrics
    metric: Metric to use for comparison (default: RMSE)
    lower_is_better: Whether lower values of the metric are better (default: True)
    
    Returns:
    str: Name of the best model
    """
    if lower_is_better:
        best_model = comparison_df[metric].idxmin()
    else:
        best_model = comparison_df[metric].idxmax()
    
    return best_model

def save_as_h5(model_name, results_dir, output_path):
    """
    Save PyTorch model as H5 format
    
    Parameters:
    model_name: Name of the model
    results_dir: Directory containing saved .pth models
    output_path: Path to save the H5 file
    """
    model_path = os.path.join(results_dir, f"{model_name.lower()}_model.pth")
    
    # Load model state dict
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Create H5 file
    with h5py.File(output_path, 'w') as f:
        # Create a group for the model
        model_group = f.create_group('model')
        
        # Save each parameter
        for key, value in state_dict.items():
            model_group.create_dataset(key, data=value.cpu().numpy())
        
        # Save model architecture info as attributes
        model_group.attrs['model_type'] = model_name
        model_group.attrs['creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"Model saved in H5 format at: {output_path}")

def main():
    """Main function to run the training pipeline"""
    parser = argparse.ArgumentParser(description='Train NASDAQ prediction models')
    parser.add_argument('--data', type=str, default='data/processed/nasdaq_selected_features_scaled.csv',
                       help='Path to processed data file')
    parser.add_argument('--seq_length', type=int, default=10,
                       help='Sequence length for time series models')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Print start time
    start_time = datetime.now()
    print(f"Training started at: {start_time}")
    
    # Load data
    print(f"Loading data from {args.data}")
    X, y, feature_names, _ = load_processed_data(args.data)
    
    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, y_train, X_test, y_test = prepare_sequence_data(X, y, args.seq_length)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Run baseline models for benchmarking
    baseline_results = run_baseline_models(X_train, y_train, X_test, y_test)
    
    # Run deep learning models 
    dl_results = run_deep_learning_models(
        X_train, y_train, X_test, y_test,
        seq_length=args.seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        results_dir=args.results_dir
    )
    
    # Save and compare results
    comparison_df = save_results(baseline_results, dl_results, args.results_dir)
    dl_models = ['LSTM', 'GRU', 'CNN-LSTM']
    dl_comparison_df = comparison_df.loc[comparison_df.index.isin(dl_models)]
    
    # Find best DL model based on RMSE (lower is better)
    best_model = find_best_model(dl_comparison_df, metric='RMSE', lower_is_better=True)
    print(f"\nBest performing Deep Learning model based on RMSE: {best_model}")
    
    # Save best model in H5 format
    h5_output_path = os.path.join(args.save_dir, "trained_model.h5")
    save_as_h5(best_model, args.results_dir, h5_output_path)
    
    # Print end time and duration
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nTraining completed at: {end_time}")
    print(f"Total duration: {duration}")
    print(f"Best model saved to: {h5_output_path}")

if __name__ == "__main__":
    main()

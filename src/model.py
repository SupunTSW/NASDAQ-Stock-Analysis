"""
This module defines baseline and deep learning models for predicting NASDAQ stock prices.
It includes model architectures, training functions, and evaluation metrics.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Union

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series financial data
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_length: int = 10):
        """
        Initialize the dataset with features and targets

        Parameters:
        features: Feature array of shape [n_samples, n_features]
        targets: Target array of shape [n_samples]
        seq_length: Number of time steps to use for sequence models
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_length = seq_length
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.features) - self.seq_length + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset"""
        # Get sequence of features and corresponding target
        x = self.features[idx:idx+self.seq_length]
        y = self.targets[idx+self.seq_length-1]  # Target corresponding to the last timestep
        return x, y

def create_data_loaders(X_train: np.ndarray, 
                       y_train: np.ndarray, 
                       X_test: np.ndarray, 
                       y_test: np.ndarray, 
                       seq_length: int = 10, 
                       batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and testing

    Parameters:
    X_train: Training features
    y_train: Training targets
    X_test: Testing features
    y_test: Testing targets
    seq_length: Number of time steps for sequence models
    batch_size: Batch size for training

    Returns:
    tuple: (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_length)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class BaselineModels:
    """
    Baseline models for comparison with deep learning models
    """
    def __init__(self):
        """Initialize baseline models"""
        self.linear_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def fit_linear_model(self, X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
        """
        Fit a linear regression model

        Parameters:
        X_train: Training features
        y_train: Training targets

        Returns:
        LinearRegression: Fitted linear regression model
        """
        self.linear_model.fit(X_train, y_train)
        return self.linear_model
    
    def fit_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        """
        Fit a random forest regression model

        Parameters:
        X_train: Training features
        y_train: Training targets

        Returns:
        RandomForestRegressor: Fitted random forest model
        """
        self.rf_model.fit(X_train, y_train)
        return self.rf_model
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate baseline models on test data

        Parameters:
        X_test: Testing features
        y_test: Testing targets

        Returns:
        dict: Dictionary with evaluation metrics for each model
        """
        results = {}
        
        # Evaluate linear regression
        lr_preds = self.linear_model.predict(X_test)
        results['Linear Regression'] = {
            'MSE': mean_squared_error(y_test, lr_preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, lr_preds)),
            'MAE': mean_absolute_error(y_test, lr_preds),
        }
        
        # Evaluate random forest
        rf_preds = self.rf_model.predict(X_test)
        results['Random Forest'] = {
            'MSE': mean_squared_error(y_test, rf_preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, rf_preds)),
            'MAE': mean_absolute_error(y_test, rf_preds),
        }
        
        return results

class LSTMModel(nn.Module):
    """
    LSTM model for time series prediction
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 output_dim: int = 1, dropout: float = 0.2):
        """
        Initialize the LSTM model

        Parameters:
        input_dim: Number of input features
        hidden_dim: Number of hidden units in LSTM
        num_layers: Number of LSTM layers
        output_dim: Number of output dimensions
        dropout: Dropout rate for regularization
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Parameters:
        x: Input tensor of shape [batch_size, seq_length, input_dim]

        Returns:
        torch.Tensor: Output predictions
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape [batch_size, seq_length, hidden_dim]
        
        # Get output from the last time step
        out = self.dropout(out[:, -1, :])
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        
        return out

class GRUModel(nn.Module):
    """
    GRU model for time series prediction
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 output_dim: int = 1, dropout: float = 0.2):
        """
        Initialize the GRU model

        Parameters:
        input_dim: Number of input features
        hidden_dim: Number of hidden units in GRU
        num_layers: Number of GRU layers
        output_dim: Number of output dimensions
        dropout: Dropout rate for regularization
        """
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Parameters:
        x: Input tensor of shape [batch_size, seq_length, input_dim]

        Returns:
        torch.Tensor: Output predictions
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)  # out: tensor of shape [batch_size, seq_length, hidden_dim]
        
        # Get output from the last time step
        out = self.dropout(out[:, -1, :])
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        
        return out

class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM model for time series prediction
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 output_dim: int = 1, dropout: float = 0.2, kernel_size: int = 3):
        """
        Initialize the CNN-LSTM model

        Parameters:
        input_dim: Number of input features
        hidden_dim: Number of hidden units in LSTM
        num_layers: Number of LSTM layers
        output_dim: Number of output dimensions
        dropout: Dropout rate for regularization
        kernel_size: Kernel size for CNN layers
        """
        super(CNNLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 1D CNN layers
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Parameters:
        x: Input tensor of shape [batch_size, seq_length, input_dim]

        Returns:
        torch.Tensor: Output predictions
        """
        batch_size, seq_len, input_dim = x.size()
        
        # Reshape for CNN [batch_size, input_dim, seq_length]
        x = x.permute(0, 2, 1)
        
        # Apply 1D CNN
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        
        # Reshape back for LSTM [batch_size, seq_length, hidden_dim]
        x = x.permute(0, 2, 1)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from the last time step
        out = self.dropout(out[:, -1, :])
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        
        return out

def train_model(model: nn.Module, 
               train_loader: DataLoader, 
               test_loader: DataLoader, 
               epochs: int = 100, 
               learning_rate: float = 0.001, 
               weight_decay: float = 1e-5,
               patience: int = 10,
               verbose: bool = True) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, float]]:
    """
    Train a PyTorch model

    Parameters:
    model: PyTorch model to train
    train_loader: DataLoader for training data
    test_loader: DataLoader for testing data
    epochs: Number of training epochs
    learning_rate: Learning rate for optimizer
    weight_decay: Weight decay for regularization
    patience: Early stopping patience
    verbose: Whether to print progress

    Returns:
    tuple: (trained model, training history, evaluation metrics)
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        history['val_loss'].append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    
    # Load best model
    model.load_state_dict(best_model)
    
    # Final evaluation
    evaluation = evaluate_model(model, test_loader)
    
    return model, history, evaluation

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
    """
    Evaluate a PyTorch model

    Parameters:
    model: PyTorch model to evaluate
    test_loader: DataLoader for testing data

    Returns:
    dict: Dictionary with evaluation metrics
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze().cpu().numpy()
            predictions.extend(outputs.tolist() if isinstance(outputs, np.ndarray) else [outputs])
            actuals.extend(y_batch.numpy().tolist())
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
    }

def plot_predictions(model: nn.Module, 
                    test_loader: DataLoader, 
                    model_name: str = "",
                    output_dir: str = "",
                    scaler: Optional[Union[dict, object]] = None, 
                    n_samples: int = 100) -> None:
    """
    Plot model predictions against actual values

    Parameters:
    model: Trained PyTorch model
    test_loader: DataLoader for testing data
    model_name: Name of the model for saving unique files
    output_dir: Directory to save plots
    scaler: Scaler object used for normalization (if applicable)
    n_samples: Number of samples to plot
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze().cpu().numpy()
            predictions.extend(outputs.tolist() if isinstance(outputs, np.ndarray) else [outputs])
            actuals.extend(y_batch.numpy().tolist())
            
            if len(predictions) >= n_samples:
                break
    
    # Limit to n_samples
    predictions = predictions[:n_samples]
    actuals = actuals[:n_samples]
    
    # Inverse transform if scaler is provided
    if scaler is not None:
        try:
            # If it's a scikit-learn scaler
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
        except:
            # If it's a custom scaler or dictionary
            pass
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(f'{model_name} - Actual vs Predicted Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save to specified directory with unique name
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"{model_name.lower()}_predictions.png")
        plt.savefig(plot_path)
    else:
        plt.savefig(f'{model_name.lower()}_predictions_plot.png')
    plt.close()
    
    # Plot error
    plt.figure(figsize=(12, 6))
    plt.plot(np.array(actuals) - np.array(predictions))
    plt.title(f'{model_name} - Prediction Error')
    plt.xlabel('Time Steps')
    plt.ylabel('Error')
    plt.grid(True)
    plt.tight_layout()
    
    # Save to specified directory with unique name
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        error_path = os.path.join(output_dir, f"{model_name.lower()}_error.png")
        plt.savefig(error_path)
    else:
        plt.savefig(f'{model_name.lower()}_error_plot.png')
    plt.close()

def plot_training_history(history: Dict[str, List[float]], model_name: str = "", output_dir: str = "") -> None:
    """
    Plot training history

    Parameters:
    history: Dictionary with training and validation loss
    model_name: Name of the model for saving unique files
    output_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save to specified directory with unique name
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        history_path = os.path.join(output_dir, f"{model_name.lower()}_training_history.png")
        plt.savefig(history_path)
    else:
        plt.savefig(f'{model_name.lower()}_training_history.png')
    plt.close()

def save_model(model: nn.Module, filepath: str) -> None:
    """
    Save the trained model

    Parameters:
    model: Trained PyTorch model
    filepath: Path to save the model
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model_class: nn.Module, filepath: str, **model_params) -> nn.Module:
    """
    Load a trained model

    Parameters:
    model_class: PyTorch model class
    filepath: Path to the saved model
    model_params: Parameters for model initialization

    Returns:
    nn.Module: Loaded model
    """
    model = model_class(**model_params)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model

def prepare_sequence_data(X: np.ndarray, 
                         y: np.ndarray, 
                         seq_length: int, 
                         train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for sequence models

    Parameters:
    X: Feature array
    y: Target array
    seq_length: Sequence length
    train_ratio: Ratio for train/test split

    Returns:
    tuple: (X_train, y_train, X_test, y_test)
    """
    # Calculate split point
    split_idx = int(len(X) * train_ratio)
    
    # Split data
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, y_train, X_test, y_test
